"""
VOS (Video Object Segmentation) Propagation for Temporal Consistency
====================================================================

Replaces naive sliding-window mask averaging with real object-level
temporal propagation using LiVOS (CVPR 2025) or Cutie (CVPR 2024).

Instead of averaging masks across a window (which smears moving objects),
VOS propagation tracks objects across frames and produces temporally
consistent masks with identity preservation.

Two operating modes:
  1. Keyframe detect + propagate: Run full detection every N frames,
     propagate to intermediate frames via VOS (faster, default).
  2. Detect-every-frame smoothing: Run detection on all frames,
     use VOS to refine temporal consistency (higher quality).

Pipeline: detector masks (keyframes) → VOS memory → propagated masks

Requires: pip install -e . from cloned LiVOS repo (or Cutie repo)
  LiVOS: git clone https://github.com/uncbiag/LiVOS
  Cutie: git clone https://github.com/hkchengrex/Cutie
Weights auto-download on first use (~135MB for LiVOS base).

Integration:
    from vos_propagation import VOSPropagator, VOSConfig
    cfg = VOSConfig()
    prop = VOSPropagator(cfg)
    prop.initialize()
    mask = prop.step(image_bgr, detected_mask)  # keyframe
    mask = prop.step(image_bgr, None)            # propagation
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Optional imports — try LiVOS first, then Cutie
# ══════════════════════════════════════════════════════════════════════════════

HAS_LIVOS = False
HAS_CUTIE = False
HAS_TORCH = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    from livos.model.livos_wrapper import LIVOS
    HAS_LIVOS = True
except ImportError:
    pass

if not HAS_LIVOS:
    try:
        from cutie.model.cutie import CUTIE
        from cutie.inference.inference_core import InferenceCore as CutieInferenceCore
        HAS_CUTIE = True
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VOSConfig:
    """Configuration for VOS temporal propagation."""

    # Model
    backend: str = 'auto'  # 'livos', 'cutie', or 'auto' (try livos → cutie)
    model_type: str = 'base'  # 'base' or 'small' (LiVOS); 'base-mega' or 'base' (Cutie)
    weights_path: Optional[str] = None  # None = auto-download default weights

    # Device
    device: str = 'cpu'

    # Propagation settings
    keyframe_interval: int = 5  # Run full detection every N frames
    propagation_threshold: float = 0.5  # Confidence threshold for propagated masks
    max_internal_size: int = 480  # Resize long edge for VOS processing (memory vs quality)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'backend': self.backend,
            'model_type': self.model_type,
            'weights_path': self.weights_path,
            'device': self.device,
            'keyframe_interval': self.keyframe_interval,
            'propagation_threshold': self.propagation_threshold,
            'max_internal_size': self.max_internal_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VOSConfig':
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ══════════════════════════════════════════════════════════════════════════════
# VOS Propagator
# ══════════════════════════════════════════════════════════════════════════════

class VOSPropagator:
    """Temporal mask propagation using VOS (Video Object Segmentation).

    Replaces TemporalConsistency's sliding-window averaging with real
    object tracking. Maintains internal VOS memory state across frames.

    Usage:
        propagator.initialize()
        propagator.reset()  # Start new video/sequence

        # Keyframe (with detected mask):
        mask = propagator.step(image_bgr, detected_mask)

        # Propagation frame (no detection):
        mask = propagator.step(image_bgr, None)
    """

    def __init__(self, config: Optional[VOSConfig] = None):
        self.config = config or VOSConfig()
        self.network = None
        self.processor = None
        self._frame_counter = 0
        self._backend = None  # 'livos' or 'cutie'
        self._original_size = None  # (H, W) of input images

    def initialize(self):
        """Load VOS model and weights."""
        if not HAS_TORCH:
            raise ImportError(
                "torch is required for VOS propagation. "
                "Install with: pip install torch"
            )

        backend = self.config.backend
        if backend == 'auto':
            if HAS_LIVOS:
                backend = 'livos'
            elif HAS_CUTIE:
                backend = 'cutie'
            else:
                raise ImportError(
                    "No VOS backend available. Install LiVOS or Cutie:\n"
                    "  LiVOS: git clone https://github.com/uncbiag/LiVOS && "
                    "cd LiVOS && pip install -e .\n"
                    "  Cutie: git clone https://github.com/hkchengrex/Cutie && "
                    "cd Cutie && pip install -e ."
                )

        if backend == 'livos':
            self._init_livos()
        elif backend == 'cutie':
            self._init_cutie()
        else:
            raise ValueError(f"Unknown VOS backend: {backend}")

        self._backend = backend
        logger.info(f"VOS propagator initialized: {backend} ({self.config.model_type})")

    def _init_livos(self):
        """Initialize LiVOS backend."""
        if not HAS_LIVOS:
            raise ImportError("LiVOS not installed")

        model_type = self.config.model_type
        self.network = LIVOS(model_type=model_type)

        if self.config.weights_path:
            weights = torch.load(self.config.weights_path, weights_only=True)
            self.network.load_weights(weights)

        device = self.config.device
        if device == 'cuda' and torch.cuda.is_available():
            self.network = self.network.cuda()
        self.network.eval()

    def _init_cutie(self):
        """Initialize Cutie backend."""
        if not HAS_CUTIE:
            raise ImportError("Cutie not installed")

        self.network = CUTIE()

        if self.config.weights_path:
            weights = torch.load(self.config.weights_path, weights_only=True)
            self.network.load_state_dict(weights)

        device = self.config.device
        if device == 'cuda' and torch.cuda.is_available():
            self.network = self.network.cuda()
        self.network.eval()

    def reset(self):
        """Reset state for new video/sequence. Call before processing a new sequence."""
        self.processor = None
        self._frame_counter = 0
        self._original_size = None

    def is_keyframe(self) -> bool:
        """Check if the current frame (next to be processed) is a keyframe."""
        return self._frame_counter % self.config.keyframe_interval == 0

    @property
    def frame_count(self) -> int:
        """Number of frames processed so far in this sequence."""
        return self._frame_counter

    def step(
        self,
        image_bgr: np.ndarray,
        detected_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Process one frame through VOS.

        Args:
            image_bgr: Input image in BGR format (H, W, 3), uint8
            detected_mask: Binary mask from detector (H, W), uint8, 0/255.
                           Provide on keyframes; None for propagation frames.

        Returns:
            Propagated/refined mask (H, W), uint8, 0 or 255
        """
        if self.network is None:
            raise RuntimeError("VOSPropagator not initialized. Call initialize() first.")

        h, w = image_bgr.shape[:2]
        self._original_size = (h, w)

        # Convert BGR to RGB float tensor [C, H, W]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Resize for VOS processing if needed
        max_size = self.config.max_internal_size
        scale = min(max_size / max(h, w), 1.0)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            image_resized = cv2.resize(image_rgb, (new_w, new_h))
        else:
            image_resized = image_rgb
            new_h, new_w = h, w

        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        device = self.config.device
        if device == 'cuda' and torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        # Create processor on first frame
        if self.processor is None:
            self.processor = self._create_processor()

        # Step through VOS
        if detected_mask is not None and np.any(detected_mask):
            # Keyframe: encode mask into VOS memory
            mask_bin = (detected_mask > 127).astype(np.uint8)
            if scale < 1.0:
                mask_resized = cv2.resize(mask_bin, (new_w, new_h),
                                          interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask_bin

            # Create label mask: 0=background, 1=foreground object
            mask_tensor = torch.from_numpy(mask_resized.astype(np.int64))
            if device == 'cuda' and torch.cuda.is_available():
                mask_tensor = mask_tensor.cuda()

            with torch.no_grad():
                prob = self.processor.step(
                    image_tensor, mask_tensor, [1],
                    is_last_frame=False
                )
        else:
            # Propagation frame
            with torch.no_grad():
                prob = self.processor.step(
                    image_tensor, None, None,
                    is_last_frame=False
                )

        # Extract mask from probability output
        # prob shape: [N+1, H, W] where channel 0 is background
        if prob is not None and prob.dim() >= 2:
            if prob.dim() == 3 and prob.shape[0] > 1:
                # Multi-channel: take foreground (channel 1)
                mask_prob = prob[1].cpu().numpy()
            else:
                # Single channel
                mask_prob = prob.squeeze().cpu().numpy()

            mask_out = (mask_prob > self.config.propagation_threshold).astype(np.uint8)
        else:
            mask_out = np.zeros((new_h, new_w), dtype=np.uint8)

        # Resize back to original if needed
        if scale < 1.0:
            mask_out = cv2.resize(mask_out, (w, h),
                                  interpolation=cv2.INTER_NEAREST)

        # Scale to 0/255
        mask_out = mask_out * 255

        self._frame_counter += 1
        return mask_out

    def _create_processor(self):
        """Create the appropriate inference processor for the backend."""
        if self._backend == 'livos':
            # LiVOS uses ObjectManager or InferenceCore
            try:
                from livos.inference.object_manager import ObjectManager
                return ObjectManager(self.network)
            except ImportError:
                # Try alternative import path
                from livos.eval import InferenceCore
                return InferenceCore(self.network)
        elif self._backend == 'cutie':
            return CutieInferenceCore(
                self.network, cfg=self.network.cfg
            )
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def cleanup(self):
        """Release model resources."""
        if self.network is not None:
            del self.network
            self.network = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VOSPropagator cleaned up")
