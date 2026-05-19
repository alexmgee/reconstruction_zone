"""
SAM3 video click session for interactive multi-frame segmentation.

Wraps the SAM 3.1 multiplex video predictor so the user can click on any
frame, accumulate point prompts, and propagate the resulting masks across
an entire frame directory.

Replaces the old single-image SAM3ClickSession (predict_inst) with a full
video session: start_session → add_prompt (points) → propagate_in_video.

Requires:
    git clone https://github.com/facebookresearch/sam3 && pip install -e .
    HuggingFace access approval for model weights (facebook/sam3.1).
    Python 3.12+, PyTorch 2.7+
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Extensions SAM3 io_utils.py recognises (must match exactly)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Object ID for all click prompts (single merged object scope)
_OBJ_ID = 0


def _sort_frame_names(names: List[str]) -> List[str]:
    """Sort frame filenames identically to SAM3's io_utils.py.

    Tries numeric sort by stem first; falls back to lexicographic.
    This MUST match the order SAM3 uses internally so that frame_index
    values correspond to the correct files.
    """
    try:
        return sorted(names, key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        return sorted(names)


def _enumerate_frames(directory: Path) -> List[str]:
    """List and sort image filenames in a directory (SAM3-compatible)."""
    names = [
        p for p in os.listdir(str(directory))
        if os.path.splitext(p)[-1].lower() in _IMAGE_EXTS
    ]
    if not names:
        raise FileNotFoundError(f"No image files found in {directory}")
    return _sort_frame_names(names)


def _detect_fa3() -> bool:
    """Check whether Flash Attention 3 is available."""
    try:
        from flash_attn_interface import flash_attn_func  # noqa: F401
        return True
    except ImportError:
        return False


def _patch_sdpa_fallback() -> None:
    """Monkey-patch SAM 3.1 decoder to allow MATH attention when FA3 is absent.

    SAM 3.1's decoder.py hardcodes sdpa_kernel(FLASH_ATTENTION) when
    use_fa3=False, which crashes if Flash Attention isn't compiled into
    PyTorch.  Reused from sam3_pipeline.py to avoid triplication.
    """
    try:
        from sam3.model import decoder as _dec
        from torch.nn.attention import sdpa_kernel, SDPBackend
        _orig = sdpa_kernel
        _dec.sdpa_kernel = lambda *a, **kw: _orig(
            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION,
             SDPBackend.FLASH_ATTENTION]
        )
        logger.info("Patched SAM 3.1 decoder for MATH attention fallback")
    except Exception as e:
        logger.warning(f"Could not patch decoder attention: {e}")


class SAM3VideoClickSession:
    """Interactive click session over a video (frame directory).

    Usage::

        session = SAM3VideoClickSession()
        session.start(Path("/data/frames"), device="cuda")

        # User clicks on frame 0
        mask, _ = session.add_click(frame_idx=0, x=520, y=375, positive=True)
        mask, _ = session.add_click(frame_idx=0, x=610, y=400, positive=False)

        # Navigate to frame 500, add a corrective click
        mask, _ = session.add_click(frame_idx=500, x=300, y=200, positive=True)

        # Propagate across all frames
        session.propagate(
            output_dir=Path("/data/frames_masks"),
            progress_callback=lambda cur, total, msg: print(f"{cur}/{total}"),
        )

        session.close()

    Points are passed to SAM3 as normalised [0, 1] coordinates.  The caller
    passes pixel coordinates; this class normalises them using the original
    frame dimensions reported by the session.
    """

    MODEL_TAG = "sam3-video-pvs"

    def __init__(self):
        self.predictor = None
        self._device = "cpu"
        self._session_id: Optional[str] = None
        self._frame_dir: Optional[Path] = None
        self._frame_names: List[str] = []
        self._orig_hw: Optional[Tuple[int, int]] = None  # (H, W)
        # Per-frame click accumulator: frame_idx → [(x_px, y_px, label), ...]
        self._clicks: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        # Cached mask from the last add_prompt on each prompted frame
        self._frame_masks: Dict[int, np.ndarray] = {}

    # ── Properties ────────────────────────────────────────────────────

    @property
    def num_frames(self) -> int:
        return len(self._frame_names)

    @property
    def frame_names(self) -> List[str]:
        """Sorted filename list matching SAM3's internal frame ordering."""
        return list(self._frame_names)

    @property
    def prompted_frames(self) -> Dict[int, List[dict]]:
        """Click history per frame: {frame_idx: [{x, y, label}, ...]}."""
        return {
            idx: [{"x": x, "y": y, "label": lab} for x, y, lab in clicks]
            for idx, clicks in self._clicks.items()
            if clicks
        }

    @property
    def is_ready(self) -> bool:
        return self._session_id is not None

    def _autocast(self):
        """Return bfloat16 autocast context for CUDA, nullcontext for CPU.

        SAM 3.1's point-prompt path (add_sam2_new_points → tracker.add_new_points)
        lacks the @torch.autocast decorator that the text-prompt path has.
        The model weights are bfloat16 from the checkpoint; without autocast,
        float32 point inputs cause a dtype mismatch in matmul.
        """
        import torch
        from contextlib import nullcontext
        if self._device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self, input_dir: Path, device: str = "cuda") -> None:
        """Build the video predictor and start a session on *input_dir*.

        Loads all frames via SAM3's async loader (lazy, CPU-offloaded).
        Blocks until the session is ready for prompts.
        """
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        if self.predictor is not None:
            self.close()

        # Enumerate frames BEFORE starting the session so we can cross-check
        self._frame_dir = Path(input_dir)
        self._frame_names = _enumerate_frames(self._frame_dir)

        # FA3 detection + SDPA fallback
        fa3 = _detect_fa3()
        if not fa3:
            _patch_sdpa_fallback()

        logger.info(
            "Building SAM 3.1 multiplex video predictor "
            "(FA3=%s, frames=%d, dir=%s)",
            fa3, len(self._frame_names), input_dir,
        )
        self.predictor = build_sam3_multiplex_video_predictor(
            use_fa3=fa3,
            async_loading_frames=True,
        )
        self._device = device

        # Start the video session with the directory path (SAM3 loads lazily)
        resp = self.predictor.handle_request({
            "type": "start_session",
            "resource_path": str(input_dir),
            "offload_video_to_cpu": True,
        })
        self._session_id = resp["session_id"]

        # Cross-check frame count
        # (inference_state is internal but accessible for validation)
        session_state = self.predictor._all_inference_states[self._session_id]["state"]
        sam3_num = session_state["num_frames"]
        if sam3_num != len(self._frame_names):
            logger.warning(
                "Frame count mismatch: SAM3 loaded %d vs enumerated %d. "
                "Truncating to min.",
                sam3_num, len(self._frame_names),
            )
            self._frame_names = self._frame_names[:sam3_num]

        self._orig_hw = (session_state["orig_height"], session_state["orig_width"])
        self._clicks.clear()
        self._frame_masks.clear()

        logger.info(
            "Video click session ready: %d frames, %dx%d, session=%s",
            len(self._frame_names), self._orig_hw[1], self._orig_hw[0],
            self._session_id,
        )

    def close(self) -> None:
        """Close the session and release GPU memory."""
        if self._session_id is not None and self.predictor is not None:
            try:
                self.predictor.handle_request({
                    "type": "close_session",
                    "session_id": self._session_id,
                    "run_gc_collect": True,
                })
            except Exception:
                pass
        self._session_id = None
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        self._frame_dir = None
        self._frame_names = []
        self._orig_hw = None
        self._clicks.clear()
        self._frame_masks.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Video click session closed")

    # ── Click prompts ─────────────────────────────────────────────────

    def add_click(
        self, frame_idx: int, x: int, y: int, positive: bool,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Add a point prompt on *frame_idx* and return the frame's mask.

        *x*, *y* are pixel coordinates in the original image space.
        They are normalised to [0, 1] before being sent to SAM3.

        Returns ``(mask, score)``.  Mask is uint8 0/1 at original resolution,
        or None if the prompt produced no output.  Score is always 1.0
        for user-clicked objects (SAM3 upstream behaviour).
        """
        if self._session_id is None:
            raise RuntimeError("start() must be called before add_click")
        if not (0 <= frame_idx < len(self._frame_names)):
            raise IndexError(
                f"frame_idx {frame_idx} out of range [0, {len(self._frame_names)})"
            )
        h, w = self._orig_hw
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))

        label = 1 if positive else 0
        self._clicks[frame_idx].append((x, y, label))

        return self._send_prompt(frame_idx)

    def remove_last_click(self, frame_idx: int) -> Tuple[Optional[np.ndarray], float]:
        """Remove the most recent click on *frame_idx* and re-predict.

        Returns the updated (mask, score), or (None, 0.0) if no clicks remain.
        """
        if frame_idx in self._clicks and self._clicks[frame_idx]:
            self._clicks[frame_idx].pop()

        if not self._clicks[frame_idx]:
            # No clicks left — remove the object from this frame
            self._remove_object_from_frame(frame_idx)
            self._frame_masks.pop(frame_idx, None)
            return None, 0.0

        return self._send_prompt(frame_idx)

    def reset_frame(self, frame_idx: int) -> None:
        """Clear all clicks on one frame."""
        self._clicks.pop(frame_idx, None)
        self._frame_masks.pop(frame_idx, None)
        self._remove_object_from_frame(frame_idx)

    def reset_all(self) -> None:
        """Clear all clicks on all frames (keeps session alive)."""
        self._clicks.clear()
        self._frame_masks.clear()
        if self._session_id is not None and self.predictor is not None:
            try:
                self.predictor.handle_request({
                    "type": "reset_session",
                    "session_id": self._session_id,
                })
            except Exception as e:
                logger.warning(f"reset_session failed: {e}")

    def get_frame_clicks(self, frame_idx: int) -> List[dict]:
        """Return click list for one frame: [{x, y, label}, ...]."""
        return [
            {"x": x, "y": y, "label": lab}
            for x, y, lab in self._clicks.get(frame_idx, [])
        ]

    def get_frame_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return the cached mask for a frame, or None."""
        return self._frame_masks.get(frame_idx)

    def num_clicks_on_frame(self, frame_idx: int) -> int:
        return len(self._clicks.get(frame_idx, []))

    # ── Propagation ───────────────────────────────────────────────────

    def propagate(
        self,
        output_dir: Path,
        fisheye_circle_fn: Optional[Callable[[int, int], Optional[np.ndarray]]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event=None,
    ) -> int:
        """Propagate prompted masks across all frames and write PNGs.

        Masks are written as ``(1 - mask) * 255`` (batch pipeline convention:
        masked-out = black, keep = white).

        Returns the number of masks written.  On cancel, returns the count
        of frames written so far (session stays open for re-propagation).
        """
        if self._session_id is None:
            raise RuntimeError("start() must be called before propagate")
        if not self._clicks:
            raise RuntimeError("at least one click is required before propagate")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        h, w = self._orig_hw
        total = len(self._frame_names)
        written = 0

        import torch
        with torch.inference_mode(), self._autocast():
            for resp in self.predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": self._session_id,
                "propagation_direction": "both",
            }):
                if cancel_event is not None and cancel_event.is_set():
                    logger.info("Propagation cancelled at frame %d/%d",
                                written, total)
                    break

                fidx = resp["frame_index"]
                outputs = resp["outputs"]
                binary_masks = outputs.get("out_binary_masks")

                if binary_masks is not None and len(binary_masks) > 0:
                    combined = np.any(binary_masks, axis=0).astype(np.uint8)
                else:
                    combined = np.zeros((h, w), dtype=np.uint8)

                # Resize if SAM output differs from original
                if combined.shape[:2] != (h, w):
                    combined = cv2.resize(
                        combined, (w, h), interpolation=cv2.INTER_NEAREST)

                # Composite with fisheye circle mask if applicable
                if fisheye_circle_fn is not None:
                    circle = fisheye_circle_fn(w, h)
                    if circle is not None:
                        combined = np.maximum(combined, circle)

                # Write with batch pipeline polarity: (1 - mask) * 255
                stem = os.path.splitext(self._frame_names[fidx])[0]
                mask_path = out / f"{stem}.png"
                cv2.imwrite(str(mask_path), (1 - combined) * 255)
                written += 1

                if progress_callback is not None:
                    progress_callback(written, total,
                                      f"Writing masks... {written}/{total}")

        logger.info("Propagation complete: %d/%d masks written to %s",
                     written, total, out)
        return written

    # ── Internal helpers ──────────────────────────────────────────────

    def _send_prompt(self, frame_idx: int) -> Tuple[Optional[np.ndarray], float]:
        """Send accumulated clicks for *frame_idx* to add_prompt.

        Points are normalised to [0, 1] (rel_coordinates=True upstream).
        Returns (mask, score).
        """
        clicks = self._clicks.get(frame_idx, [])
        if not clicks:
            return None, 0.0

        import torch

        h, w = self._orig_hw
        points = [[x / w, y / h] for x, y, _ in clicks]
        labels = [lab for _, _, lab in clicks]

        with torch.inference_mode(), self._autocast():
            resp = self.predictor.handle_request({
                "type": "add_prompt",
                "session_id": self._session_id,
                "frame_index": frame_idx,
                "points": points,
                "point_labels": labels,
                "clear_old_points": True,
                "obj_id": _OBJ_ID,
            })

        # Point prompts go through add_sam2_new_points which does NOT mark
        # previous_stages_out (only text prompts do via _run_single_frame_inference).
        # Without this marker, propagate_in_video's _get_processing_order raises
        # "No prompts are received on any frames" because it only checks
        # previous_stages_out.  Set the same sentinel SAM3 uses internally.
        session_state = self.predictor._all_inference_states[self._session_id]["state"]
        session_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"

        outputs = resp.get("outputs")
        if outputs is None:
            self._frame_masks.pop(frame_idx, None)
            return None, 0.0

        binary_masks = outputs.get("out_binary_masks")
        if binary_masks is None or len(binary_masks) == 0:
            self._frame_masks.pop(frame_idx, None)
            return None, 0.0

        # Merge all object masks and convert to uint8 0/1
        combined = np.any(binary_masks, axis=0).astype(np.uint8)
        if combined.shape[:2] != (h, w):
            combined = cv2.resize(combined, (w, h),
                                  interpolation=cv2.INTER_NEAREST)

        self._frame_masks[frame_idx] = combined

        # Score is always 1.0 for user-clicked objects (SAM3 upstream)
        probs = outputs.get("out_probs")
        score = float(probs[0]) if probs is not None and len(probs) > 0 else 1.0

        return combined, score

    def _remove_object_from_frame(self, frame_idx: int) -> None:
        """Remove the tracked object from a specific frame."""
        if self._session_id is None or self.predictor is None:
            return
        try:
            self.predictor.handle_request({
                "type": "remove_object",
                "session_id": self._session_id,
                "frame_index": frame_idx,
                "obj_id": _OBJ_ID,
            })
        except Exception as e:
            logger.debug(f"remove_object on frame {frame_idx}: {e}")
