"""
Alpha Matting for Photogrammetry Masking
=========================================

Converts binary segmentation masks into alpha mattes with soft edges,
hair detail, and transparency using ViTMatte via HuggingFace Transformers.

Binary masks produce hard edges that cause artifacts in COLMAP feature
matching and 3DGS training. Alpha mattes provide gradient boundaries
that dramatically improve reconstruction quality.

Pipeline: binary mask → trimap (erode/dilate) → ViTMatte → alpha matte

Requires: pip install transformers torch
Weights auto-download from HuggingFace on first use (~100MB for small model).

Integration pattern mirrors shadow_detection.py and sam_refinement.py:
    from matting import MattingRefiner, MattingConfig
    cfg = MattingConfig()
    refiner = MattingRefiner(cfg)
    refiner.initialize()
    alpha = refiner.refine(image_bgr, binary_mask)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Optional imports
# ══════════════════════════════════════════════════════════════════════════════

HAS_TRANSFORMERS = False
HAS_TORCH = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Available models on HuggingFace
VITMATTE_MODELS = {
    'small': 'hustvl/vitmatte-small-composition-1k',
    'base': 'hustvl/vitmatte-base-composition-1k',
}


@dataclass
class MattingConfig:
    """Configuration for alpha matting refinement."""

    # Model selection
    model_size: str = 'small'           # 'small' (~25M params) or 'base' (~45M params)
    model_name: Optional[str] = None    # Override with custom HuggingFace model ID

    # Device
    device: str = 'cpu'

    # Trimap generation from binary mask
    erode_kernel: int = 10              # Kernel size for foreground erosion
    dilate_kernel: int = 10             # Kernel size for background dilation
    erode_iterations: int = 3           # Erosion iterations (more = narrower definite-fg)
    dilate_iterations: int = 3          # Dilation iterations (more = wider unknown region)

    # Quality thresholds
    min_mask_area: int = 100            # Skip matting for masks smaller than this
    unknown_ratio_max: float = 0.8      # Skip if unknown region exceeds this fraction of mask bbox

    def get_model_name(self) -> str:
        """Resolve the HuggingFace model ID."""
        if self.model_name:
            return self.model_name
        return VITMATTE_MODELS.get(self.model_size, VITMATTE_MODELS['small'])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_size': self.model_size,
            'model_name': self.model_name,
            'device': self.device,
            'erode_kernel': self.erode_kernel,
            'dilate_kernel': self.dilate_kernel,
            'erode_iterations': self.erode_iterations,
            'dilate_iterations': self.dilate_iterations,
            'min_mask_area': self.min_mask_area,
            'unknown_ratio_max': self.unknown_ratio_max,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MattingConfig':
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ══════════════════════════════════════════════════════════════════════════════
# Alpha Matting Refiner
# ══════════════════════════════════════════════════════════════════════════════

class MattingRefiner:
    """Converts binary masks to alpha mattes using ViTMatte.

    Takes an image and a binary segmentation mask (0/255 uint8),
    generates a trimap via erosion/dilation, then runs ViTMatte
    to produce a continuous alpha matte (0-255 uint8).
    """

    def __init__(self, config: Optional[MattingConfig] = None):
        self.config = config or MattingConfig()
        self.model = None
        self.processor = None

    def initialize(self):
        """Load ViTMatte model and processor from HuggingFace.

        Weights auto-download on first call (~100MB for small model).
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for alpha matting. "
                "Install with: pip install transformers"
            )
        if not HAS_TORCH:
            raise ImportError(
                "torch is required for alpha matting. "
                "Install with: pip install torch"
            )

        model_name = self.config.get_model_name()
        logger.info(f"Loading ViTMatte: {model_name}")

        self.processor = VitMatteImageProcessor.from_pretrained(model_name)
        self.model = VitMatteForImageMatting.from_pretrained(model_name)
        self.model.eval()

        if self.config.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("ViTMatte running on CUDA")
        else:
            logger.info("ViTMatte running on CPU")

    def refine(self, image_bgr: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to alpha matte.

        Args:
            image_bgr: Input image in BGR format (H, W, 3), uint8
            binary_mask: Binary mask (H, W), uint8, values 0 or 255

        Returns:
            Alpha matte (H, W), uint8, values 0-255 (continuous)
        """
        if self.model is None:
            raise RuntimeError("MattingRefiner not initialized. Call initialize() first.")

        # Skip tiny masks
        mask_area = np.count_nonzero(binary_mask)
        if mask_area < self.config.min_mask_area:
            logger.debug(f"Mask too small for matting ({mask_area}px), returning binary")
            return binary_mask

        # Generate trimap from binary mask
        trimap = self._generate_trimap(binary_mask)

        # Check that the unknown region is reasonable
        unknown_pixels = np.count_nonzero(trimap == 128)
        mask_bbox_area = self._mask_bbox_area(binary_mask)
        if mask_bbox_area > 0 and unknown_pixels / mask_bbox_area > self.config.unknown_ratio_max:
            logger.debug("Unknown region too large, returning binary mask")
            return binary_mask

        # Convert BGR to RGB for ViTMatte
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Run ViTMatte
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image_rgb)
        pil_trimap = PILImage.fromarray(trimap)

        inputs = self.processor(images=pil_image, trimaps=pil_trimap, return_tensors="pt")
        if self.config.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model(**inputs)

        # Extract alpha matte
        alpha = output.alphas.squeeze().cpu().numpy()

        # The output is padded to be divisible by 32; crop back to original size
        h, w = binary_mask.shape[:2]
        alpha = alpha[:h, :w]

        # Scale to uint8
        alpha = (alpha * 255).clip(0, 255).astype(np.uint8)

        # Preserve definite background: where binary mask is 0 AND trimap is 0,
        # force alpha to 0 (prevents bleeding into clearly background areas)
        alpha[trimap == 0] = 0

        return alpha

    def _generate_trimap(self, binary_mask: np.ndarray) -> np.ndarray:
        """Generate trimap from binary mask via erosion/dilation.

        Returns:
            Trimap (H, W), uint8: 0=definite background, 128=unknown, 255=definite foreground
        """
        # Ensure binary
        mask = (binary_mask > 127).astype(np.uint8) * 255

        # Erode to get definite foreground
        e_kernel = np.ones(
            (self.config.erode_kernel, self.config.erode_kernel), np.uint8
        )
        eroded = cv2.erode(mask, e_kernel, iterations=self.config.erode_iterations)

        # Dilate to get expanded region (definite bg is outside this)
        d_kernel = np.ones(
            (self.config.dilate_kernel, self.config.dilate_kernel), np.uint8
        )
        dilated = cv2.dilate(mask, d_kernel, iterations=self.config.dilate_iterations)

        # Build trimap
        trimap = np.full_like(mask, 128, dtype=np.uint8)  # Start with unknown
        trimap[eroded == 255] = 255   # Definite foreground (interior)
        trimap[dilated == 0] = 0      # Definite background (exterior)

        return trimap

    def _mask_bbox_area(self, mask: np.ndarray) -> int:
        """Compute bounding box area of non-zero mask region."""
        coords = cv2.findNonZero(mask)
        if coords is None:
            return 0
        x, y, w, h = cv2.boundingRect(coords)
        return w * h

    def cleanup(self):
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("MattingRefiner cleaned up")
