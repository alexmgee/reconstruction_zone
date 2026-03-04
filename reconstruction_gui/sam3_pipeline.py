"""
SAM 3 Unified Video Pipeline
=============================

End-to-end detection + segmentation + tracking using SAM 3's video predictor.

Replaces the traditional YOLO → SAM2 → tracker chain with a single model
that handles text-prompted detection, instance segmentation, and temporal
tracking across video frames in one unified system.

SAM 3's Promptable Concept Segmentation (PCS) takes text prompts (e.g.
"person", "tripod") and automatically detects, segments, and tracks all
matching instances across all frames with built-in temporal consistency.

Pipeline: frames directory → SAM3 video session → text prompts → propagate → masks

Requires:
    git clone https://github.com/facebookresearch/sam3 && pip install -e .
    HuggingFace access approval for model weights.
    Python 3.12+, PyTorch 2.7+

Usage:
    from sam3_pipeline import SAM3VideoPipeline, SAM3VideoConfig
    cfg = SAM3VideoConfig(prompts=["person", "tripod"])
    pipeline = SAM3VideoPipeline(cfg)
    pipeline.initialize()
    masks = pipeline.process_video(frames_dir="/path/to/frames")
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

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
    from sam3.model_builder import build_sam3_video_predictor
    HAS_SAM3_VIDEO = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SAM3VideoConfig:
    """Configuration for SAM 3 unified video pipeline."""

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

    # Batch retrieval
    batch_size: int = 50  # Retrieve masks N frames at a time

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prompts': self.prompts,
            'device': self.device,
            'prompt_frame_index': self.prompt_frame_index,
            'output_format': self.output_format,
            'confidence_threshold': self.confidence_threshold,
            'batch_size': self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SAM3VideoConfig':
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ══════════════════════════════════════════════════════════════════════════════
# RLE Decode
# ══════════════════════════════════════════════════════════════════════════════

def _decode_rle(rle_data: Dict, shape: Tuple[int, int]) -> np.ndarray:
    """Decode run-length encoded mask to binary numpy array.

    Args:
        rle_data: RLE dict with 'counts' and 'size' keys.
        shape: (H, W) output shape.

    Returns:
        Binary mask (H, W), uint8 0/255.
    """
    if isinstance(rle_data, np.ndarray):
        # Already decoded
        return (rle_data > 0).astype(np.uint8) * 255

    if isinstance(rle_data, dict):
        counts = rle_data.get('counts', [])
        size = rle_data.get('size', list(shape))

        if isinstance(counts, str):
            # COCO-style RLE string
            try:
                from pycocotools import mask as mask_utils
                mask = mask_utils.decode(rle_data)
                return (mask > 0).astype(np.uint8) * 255
            except ImportError:
                logger.warning("pycocotools not available for RLE decoding")
                return np.zeros(shape, dtype=np.uint8)

        # Integer run-length counts
        h, w = size[0], size[1]
        flat = np.zeros(h * w, dtype=np.uint8)
        pos = 0
        val = 0
        for count in counts:
            flat[pos:pos + count] = val
            pos += count
            val = 1 - val
        mask = flat.reshape((h, w), order='F')
        return mask.astype(np.uint8) * 255

    # Tensor
    if hasattr(rle_data, 'cpu'):
        arr = rle_data.cpu().numpy()
        return (arr > 0.5).astype(np.uint8) * 255

    return np.zeros(shape, dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# SAM 3 Video Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class SAM3VideoPipeline:
    """Unified video detection + segmentation + tracking via SAM 3.

    Processes an entire video (or directory of frames) through SAM 3's
    video predictor, which handles detection, segmentation, and temporal
    tracking in a single pass.

    This replaces the traditional per-frame pipeline:
        YOLO detect → SAM2 segment → BoT-SORT track → temporal average

    With a single unified model:
        SAM3 video: text prompt → detect + segment + track all frames

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
        """Load SAM 3 video predictor model."""
        if not HAS_SAM3_VIDEO:
            raise ImportError(
                "SAM3 video predictor not available. Install from: "
                "git clone https://github.com/facebookresearch/sam3 && "
                "cd sam3 && pip install -e ."
            )
        if not HAS_TORCH:
            raise ImportError("torch is required. Install with: pip install torch")

        logger.info("Loading SAM3 video predictor (848M params)")
        self.predictor = build_sam3_video_predictor()
        logger.info("SAM3 video predictor ready")

    def process_frames(
        self,
        frames_dir: str,
        output_dir: str,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Process a directory of frames through SAM 3 video pipeline.

        Frames must be sequential JPEG images (0000.jpg, 0001.jpg, ...).
        If frames have different naming, they'll be copied to a temp dir
        with sequential names.

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
        logger.info(f"Processing {n_frames} frames with SAM3 video pipeline")

        # SAM3 video requires sequential JPEG naming (0000.jpg, 0001.jpg, ...)
        # Prepare frames if needed
        prepared_dir, name_map = self._prepare_frames(frame_files)

        try:
            # Start video session
            response = self.predictor.handle_request({
                "type": "start_session",
                "resource_path": str(prepared_dir)
            })
            self._session_id = response["session_id"]
            logger.info(f"SAM3 session started: {self._session_id}")

            # Add text prompts on the designated frame
            prompt_idx = min(self.config.prompt_frame_index, n_frames - 1)
            for prompt_text in self.config.prompts:
                try:
                    self.predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": self._session_id,
                        "frame_index": prompt_idx,
                        "text": prompt_text,
                    })
                    logger.info(f"Added prompt '{prompt_text}' on frame {prompt_idx}")
                except Exception as e:
                    logger.warning(f"Prompt '{prompt_text}' failed: {e}")

            if progress_callback:
                progress_callback(0, n_frames, "Prompts added, propagating...")

            # Retrieve masks in batches
            stats = {
                'total_frames': n_frames,
                'processed_frames': 0,
                'total_objects': 0,
                'frames_with_detections': 0,
            }

            batch_size = self.config.batch_size
            for batch_start in range(0, n_frames, batch_size):
                batch_end = min(batch_start + batch_size, n_frames)
                frame_indices = list(range(batch_start, batch_end))

                response = self.predictor.handle_request({
                    "type": "get_outputs",
                    "session_id": self._session_id,
                    "frame_indices": frame_indices,
                })

                outputs = response.get("outputs", {})

                for frame_idx in frame_indices:
                    frame_data = outputs.get(frame_idx, {})
                    original_file = name_map.get(frame_idx)
                    if original_file is None:
                        continue

                    # Read frame dimensions
                    orig_img = cv2.imread(str(original_file))
                    if orig_img is None:
                        continue
                    h, w = orig_img.shape[:2]

                    # Merge all object masks into one combined mask
                    combined = np.zeros((h, w), dtype=np.uint8)
                    n_objects = 0

                    for obj_id, mask_data in frame_data.items():
                        mask = _decode_rle(mask_data, (h, w))
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask, (w, h),
                                              interpolation=cv2.INTER_NEAREST)
                        combined = np.maximum(combined, mask)
                        n_objects += 1

                    if n_objects > 0:
                        stats['frames_with_detections'] += 1
                        stats['total_objects'] += n_objects

                    # Save mask
                    stem = original_file.stem
                    mask_path = masks_dir / f"{stem}.{self.config.output_format}"
                    cv2.imwrite(str(mask_path), combined)
                    stats['processed_frames'] += 1

                if progress_callback:
                    progress_callback(batch_end, n_frames, "Retrieving masks...")

            # End session
            self.predictor.handle_request({
                "type": "end_session",
                "session_id": self._session_id,
            })
            self._session_id = None

        finally:
            # Clean up prepared frames if we made a copy
            if prepared_dir != frames_path and prepared_dir.exists():
                shutil.rmtree(prepared_dir, ignore_errors=True)

        logger.info(
            f"SAM3 pipeline complete: {stats['processed_frames']}/{stats['total_frames']} frames, "
            f"{stats['total_objects']} total detections"
        )
        return stats

    def _prepare_frames(
        self, frame_files: List[Path]
    ) -> Tuple[Path, Dict[int, Path]]:
        """Ensure frames are sequential JPEGs for SAM3 video predictor.

        If frames already follow 0000.jpg naming, returns the parent dir.
        Otherwise creates a temp directory with symlinks/copies.

        Returns:
            (prepared_dir, {frame_index: original_path})
        """
        name_map = {}

        # Check if already sequential
        parent = frame_files[0].parent
        is_sequential = True
        for i, f in enumerate(frame_files):
            expected = f"{i:04d}.jpg"
            if f.name != expected:
                is_sequential = False
                break
            name_map[i] = f

        if is_sequential and all(f.suffix.lower() in ('.jpg', '.jpeg') for f in frame_files):
            return parent, name_map

        # Need to prepare: create temp dir with sequential naming
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="sam3_frames_"))
        name_map = {}

        for i, src in enumerate(frame_files):
            dst = temp_dir / f"{i:04d}.jpg"
            name_map[i] = src

            # Convert to JPEG if needed
            if src.suffix.lower() in ('.jpg', '.jpeg'):
                # Symlink or copy
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(str(src), str(dst))
            else:
                # Read and save as JPEG
                img = cv2.imread(str(src))
                if img is not None:
                    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        logger.info(f"Prepared {len(name_map)} frames in {temp_dir}")
        return temp_dir, name_map

    def cleanup(self):
        """Release model resources."""
        if self._session_id is not None and self.predictor is not None:
            try:
                self.predictor.handle_request({
                    "type": "end_session",
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
