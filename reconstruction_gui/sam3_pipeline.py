"""
SAM 3.1 Unified Video Pipeline
================================

End-to-end detection + segmentation + tracking using SAM 3.1's multiplex
video predictor (Object Multiplex).

Replaces the traditional YOLO → SAM2 → tracker chain with a single model
that handles text-prompted detection, instance segmentation, and temporal
tracking across video frames in one unified system.

SAM 3.1's Object Multiplex groups tracked objects into shared-memory buckets,
giving ~7x speedup over SAM 3's per-object inference when tracking many
objects simultaneously.

Frames are loaded as PIL images and passed directly to start_session as a
list, which SAM 3.1 treats as a video sequence for temporal tracking. No
temp directories or sequential JPEG renaming needed.

Pipeline: frames directory → PIL list → SAM3.1 video session → text prompts → propagate → masks

Requires:
    git clone https://github.com/facebookresearch/sam3 && pip install -e .
    HuggingFace access approval for model weights (facebook/sam3.1).
    Python 3.12+, PyTorch 2.7+

Usage:
    from sam3_pipeline import SAM3VideoPipeline, SAM3VideoConfig
    cfg = SAM3VideoConfig(prompts=["person", "tripod"])
    pipeline = SAM3VideoPipeline(cfg)
    pipeline.initialize()
    stats = pipeline.process_frames(frames_dir="/path/to/frames", output_dir="/path/to/output")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Optional imports
# ══════════════════════════════════════════════════════════════════════════════

HAS_SAM3_VIDEO = False
HAS_TORCH = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    HAS_SAM3_VIDEO = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SAM3VideoConfig:
    """Configuration for SAM 3.1 unified video pipeline."""

    # Text prompts for detection
    prompts: List[str] = field(default_factory=lambda: [
        "person", "tripod", "camera equipment"
    ])

    # Device
    device: str = "cpu"

    # Prompt frame
    prompt_frame_index: int = 0  # Frame on which to apply text prompts

    # Output
    output_format: str = "png"
    confidence_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prompts': self.prompts,
            'device': self.device,
            'prompt_frame_index': self.prompt_frame_index,
            'output_format': self.output_format,
            'confidence_threshold': self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SAM3VideoConfig':
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ══════════════════════════════════════════════════════════════════════════════
# SAM 3.1 Video Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class SAM3VideoPipeline:
    """Unified video detection + segmentation + tracking via SAM 3.1.

    Processes a directory of frames (or any ordered image set) through SAM 3.1's
    multiplex video predictor, which handles detection, segmentation, and
    temporal tracking in a single pass with shared-memory object buckets.

    Frames are loaded as PIL images and passed directly to SAM 3.1 as a list,
    which it treats as a video sequence. No temp directories or sequential
    naming required.

    This replaces the traditional per-frame pipeline:
        YOLO detect → SAM2 segment → BoT-SORT track → temporal average

    With a single unified model:
        SAM3.1 video: text prompt → detect + segment + track all frames

    Usage:
        pipeline = SAM3VideoPipeline(config)
        pipeline.initialize()
        result = pipeline.process_frames(frames_dir, output_dir)
    """

    def __init__(self, config: Optional[SAM3VideoConfig] = None):
        self.config = config or SAM3VideoConfig()
        self.predictor = None
        self._session_id = None

    def initialize(self):
        """Load SAM 3.1 multiplex video predictor."""
        if not HAS_SAM3_VIDEO:
            raise ImportError(
                "SAM 3.1 video predictor not available. Install from: "
                "git clone https://github.com/facebookresearch/sam3 && "
                "cd sam3 && pip install -e ."
            )
        if not HAS_TORCH:
            raise ImportError("torch is required. Install with: pip install torch")

        # Detect Flash Attention 3 availability
        fa3_available = False
        try:
            from flash_attn_interface import flash_attn_func  # noqa: F401
            fa3_available = True
        except ImportError:
            pass

        if not fa3_available:
            # SAM 3.1's decoder.py hardcodes sdpa_kernel(FLASH_ATTENTION) when
            # use_fa3=False, which crashes if Flash Attention isn't compiled into
            # PyTorch. Monkey-patch to allow all backends as fallback.
            try:
                from sam3.model import decoder as _dec
                from torch.nn.attention import sdpa_kernel, SDPBackend
                _orig_sdpa_kernel = sdpa_kernel
                _dec.sdpa_kernel = lambda *a, **kw: _orig_sdpa_kernel(
                    [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
                )
                logger.info("Patched SAM 3.1 decoder to allow MATH attention fallback")
            except Exception as e:
                logger.warning(f"Could not patch decoder attention: {e}")

        logger.info("Loading SAM 3.1 multiplex video predictor (FA3=%s)", fa3_available)
        self.predictor = build_sam3_multiplex_video_predictor(
            use_fa3=fa3_available,
        )
        logger.info("SAM 3.1 multiplex video predictor ready")

    def process_frames(
        self,
        frames_dir: str,
        output_dir: str,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Process a directory of frames through SAM 3.1 video pipeline.

        Loads all images from frames_dir as PIL images, passes them to SAM 3.1
        as a video sequence, runs text-prompted detection on the first frame,
        then propagates tracking across all frames.

        Args:
            frames_dir: Directory containing frame images.
            output_dir: Directory to write output masks.
            progress_callback: Optional callable(current, total, message).

        Returns:
            Statistics dict.
        """
        if self.predictor is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        frames_path = Path(frames_dir)
        output_path = Path(output_dir)
        masks_dir = output_path / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Find frame files
        frame_files = sorted(
            f for f in frames_path.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        )
        if not frame_files:
            logger.warning(f"No frame images found in {frames_dir}")
            return {'total_frames': 0, 'processed_frames': 0}

        n_frames = len(frame_files)
        logger.info(f"Processing {n_frames} frames with SAM 3.1 video pipeline")

        # Load as PIL list — SAM 3.1 accepts this directly as resource_path
        pil_images = []
        for f in frame_files:
            pil_images.append(PILImage.open(f).convert("RGB"))

        # Frame dimensions from first image (PIL .size returns (W, H))
        img_w, img_h = pil_images[0].size

        # Start video session — pass PIL list directly (NOT str())
        # offload_video_to_cpu keeps frame tensors in system RAM, reducing VRAM pressure
        response = self.predictor.handle_request({
            "type": "start_session",
            "resource_path": pil_images,
            "offload_video_to_cpu": True,
        })
        self._session_id = response["session_id"]
        logger.info(f"SAM 3.1 session started: {self._session_id}")

        # Add text prompt on the designated frame
        # SAM 3.1 multiplex resets state on each add_prompt call, so we must
        # combine all prompts into a single string. The model's text encoder
        # handles multi-concept detection from a single prompt.
        prompt_idx = min(self.config.prompt_frame_index, n_frames - 1)
        combined_prompt = " . ".join(self.config.prompts)
        try:
            self.predictor.handle_request({
                "type": "add_prompt",
                "session_id": self._session_id,
                "frame_index": prompt_idx,
                "text": combined_prompt,
            })
            logger.info(f"Added prompt '{combined_prompt}' on frame {prompt_idx}")
        except Exception as e:
            logger.warning(f"Prompt '{combined_prompt}' failed: {e}")

        if progress_callback:
            progress_callback(0, n_frames, "Prompts added, propagating...")

        # Propagate through video — streaming API yields per-frame results
        stats = {
            'total_frames': n_frames,
            'processed_frames': 0,
            'total_objects': 0,
            'frames_with_detections': 0,
        }

        outputs_per_frame = {}
        for response in self.predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": self._session_id,
            "propagation_direction": "forward",
        }):
            outputs_per_frame[response["frame_index"]] = response["outputs"]

        # Write masks for each frame
        for frame_idx in range(n_frames):
            frame_outputs = outputs_per_frame.get(frame_idx)

            if frame_outputs is None:
                combined = np.zeros((img_h, img_w), dtype=np.uint8)
                n_objects = 0
            else:
                binary_masks = frame_outputs.get("out_binary_masks")
                if binary_masks is not None and len(binary_masks) > 0:
                    # binary_masks shape: (N, H, W), dtype bool
                    # Merge all objects: logical OR across object dimension
                    combined_bool = np.any(binary_masks, axis=0)
                    combined = combined_bool.astype(np.uint8) * 255
                    n_objects = len(binary_masks)
                else:
                    combined = np.zeros((img_h, img_w), dtype=np.uint8)
                    n_objects = 0

            if n_objects > 0:
                stats['frames_with_detections'] += 1
                stats['total_objects'] += n_objects

            # Resize if SAM output resolution differs from original
            if combined.shape[:2] != (img_h, img_w):
                combined = cv2.resize(combined, (img_w, img_h),
                                      interpolation=cv2.INTER_NEAREST)

            # Save mask
            stem = frame_files[frame_idx].stem
            mask_path = masks_dir / f"{stem}.{self.config.output_format}"
            cv2.imwrite(str(mask_path), combined)
            stats['processed_frames'] += 1

            if progress_callback:
                progress_callback(
                    stats['processed_frames'], n_frames, "Writing masks..."
                )

        # Close session
        self.predictor.handle_request({
            "type": "close_session",
            "session_id": self._session_id,
        })
        self._session_id = None

        # Release PIL images
        del pil_images

        det_pct = stats['frames_with_detections'] / max(n_frames, 1) * 100
        logger.info(
            f"SAM 3.1 pipeline complete: {stats['processed_frames']}/{stats['total_frames']} frames, "
            f"{stats['frames_with_detections']} with detections ({det_pct:.0f}%), "
            f"{stats['total_objects']} total objects"
        )
        return stats

    def cleanup(self):
        """Release model resources."""
        if self._session_id is not None and self.predictor is not None:
            try:
                self.predictor.handle_request({
                    "type": "close_session",
                    "session_id": self._session_id,
                })
            except Exception:
                pass
            self._session_id = None
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SAM3VideoPipeline cleaned up")
