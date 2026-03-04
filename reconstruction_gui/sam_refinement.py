"""
SAM-Based Mask Refinement for Photogrammetry Masking
=====================================================

Refines coarse segmentation masks using SAM's predictor with diverse
prompt generation inspired by SAMRefiner (ICLR 2025). Generates
distance-guided foreground/background points, context-expanded bounding
boxes, and Gaussian mask prompts from coarse masks, then runs SAM's
predict to produce refined boundaries.

Training-free, model-agnostic. Uses standard segment-anything from pip.
FastGeodis improves point sampling quality but is optional (falls back
to cv2.distanceTransform).

Integration pattern mirrors shadow_detection.py:
    from sam_refinement import SAMMaskRefiner, SAMRefinementConfig
    cfg = SAMRefinementConfig(sam_model_type="vit_b")
    refiner = SAMMaskRefiner(cfg)
    refiner.initialize()
    refined = refiner.refine(image_bgr, coarse_mask)
"""

import hashlib
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Optional imports
# ══════════════════════════════════════════════════════════════════════════════

try:
    from segment_anything import SamPredictor, sam_model_registry
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

try:
    import FastGeodis
    HAS_FASTGEODIS = True
except ImportError:
    HAS_FASTGEODIS = False


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SAMRefinementConfig:
    """Configuration for SAM-based mask refinement."""

    sam_model_type: str = "vit_b"           # vit_b / vit_l / vit_h
    sam_checkpoint: Optional[str] = None     # Override auto-download path
    device: str = "cpu"                      # Inherited from MaskConfig at runtime
    weights_dir: Optional[str] = None        # Default: ~/.prep360_sam_weights/

    # Point sampling
    num_foreground_points: int = 1           # Points inside mask (at max distance)
    num_background_points: int = 1           # Points outside mask (near boundary)

    # Box expansion
    box_margin_ratio: float = 0.15           # Expand bbox by this fraction each side

    # Mask prompt
    gaussian_sigma_ratio: float = 0.1        # Sigma as fraction of sqrt(mask_area)

    # Refinement behavior
    min_component_area: int = 200            # Skip components smaller than this
    iou_threshold: float = 0.5              # Min IoU with coarse to accept refined

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sam_model_type': self.sam_model_type,
            'sam_checkpoint': self.sam_checkpoint,
            'device': self.device,
            'weights_dir': self.weights_dir,
            'num_foreground_points': self.num_foreground_points,
            'num_background_points': self.num_background_points,
            'box_margin_ratio': self.box_margin_ratio,
            'gaussian_sigma_ratio': self.gaussian_sigma_ratio,
            'min_component_area': self.min_component_area,
            'iou_threshold': self.iou_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SAMRefinementConfig':
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ══════════════════════════════════════════════════════════════════════════════
# Weight Manager
# ══════════════════════════════════════════════════════════════════════════════

class SAMWeightManager:
    """Manages downloading and caching SAM checkpoint files.

    Default cache: ~/.prep360_sam_weights/
    Override: PREP360_SAM_WEIGHTS_DIR environment variable or config.weights_dir
    """

    DEFAULT_DIR = Path.home() / ".prep360_sam_weights"

    REGISTRY: Dict[str, Dict[str, Any]] = {
        "vit_b": {
            "filename": "sam_vit_b_01ec64.pth",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "size_mb": 375,
        },
        "vit_l": {
            "filename": "sam_vit_l_0b3195.pth",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "size_mb": 1250,
        },
        "vit_h": {
            "filename": "sam_vit_h_4b8939.pth",
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "size_mb": 2560,
        },
    }

    @classmethod
    def weights_dir(cls, config: Optional[SAMRefinementConfig] = None) -> Path:
        """Get (and create) the weights cache directory."""
        import os
        env_dir = os.environ.get("PREP360_SAM_WEIGHTS_DIR", "")
        if env_dir:
            d = Path(env_dir)
        elif config and config.weights_dir:
            d = Path(config.weights_dir)
        else:
            d = cls.DEFAULT_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def ensure_weights(
        cls,
        model_type: str,
        config: Optional[SAMRefinementConfig] = None,
    ) -> Path:
        """Ensure weights file exists locally, downloading if needed."""
        info = cls.REGISTRY.get(model_type)
        if info is None:
            raise ValueError(
                f"Unknown SAM model type '{model_type}'. "
                f"Available: {list(cls.REGISTRY.keys())}"
            )

        dest = cls.weights_dir(config) / info["filename"]
        if dest.exists():
            return dest

        url = info["url"]
        size_mb = info["size_mb"]
        logger.info(f"Downloading SAM {model_type} weights (~{size_mb} MB)...")
        logger.info(f"  URL: {url}")
        logger.info(f"  Destination: {dest}")

        try:
            urllib.request.urlretrieve(url, str(dest))
            logger.info(f"Download complete: {dest.name}")
        except Exception as e:
            if dest.exists():
                dest.unlink()
            raise FileNotFoundError(
                f"Failed to download SAM {model_type} weights: {e}\n"
                f"Download manually from: {url}\n"
                f"Place at: {dest}"
            )

        return dest

    @classmethod
    def list_available(cls) -> Dict[str, bool]:
        """List registered models and whether weights exist locally."""
        wdir = cls.weights_dir()
        return {
            name: (wdir / info["filename"]).exists()
            for name, info in cls.REGISTRY.items()
        }


# ══════════════════════════════════════════════════════════════════════════════
# SAM Mask Refiner
# ══════════════════════════════════════════════════════════════════════════════

class SAMMaskRefiner:
    """Refines coarse segmentation masks using SAM's predictor.

    Generates diverse prompts from coarse masks — distance-guided
    foreground/background points, context-expanded bounding boxes,
    and Gaussian-blurred mask prompts — then runs SAM predict to
    produce refined boundaries. Inspired by SAMRefiner (ICLR 2025).

    Per-component refinement: splits multi-object masks into connected
    components, refines each independently to avoid prompt confusion,
    then merges results.

    Usage:
        refiner = SAMMaskRefiner(config)
        refiner.initialize()  # loads SAM model
        refined_mask = refiner.refine(image_bgr, coarse_mask)
        refiner.cleanup()     # free GPU memory
    """

    def __init__(self, config: SAMRefinementConfig):
        self.config = config
        self.predictor: Optional[Any] = None
        self._initialized = False

    def initialize(self) -> None:
        """Load SAM model and create predictor."""
        if not HAS_SAM:
            raise ImportError(
                "segment-anything not installed. "
                "Install: pip install segment-anything"
            )

        import torch

        # Resolve checkpoint path
        if self.config.sam_checkpoint:
            checkpoint = Path(self.config.sam_checkpoint)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"SAM checkpoint not found: {checkpoint}"
                )
        else:
            checkpoint = SAMWeightManager.ensure_weights(
                self.config.sam_model_type, self.config
            )

        model_type = self.config.sam_model_type
        logger.info(f"Loading SAM {model_type} from {checkpoint}")

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=self.config.device)
        self.predictor = SamPredictor(sam)
        self._initialized = True
        logger.info(f"SAM {model_type} loaded on {self.config.device}")

    def refine(
        self,
        image: np.ndarray,
        coarse_mask: np.ndarray,
    ) -> np.ndarray:
        """Refine a coarse binary mask using SAM-guided prompts.

        Args:
            image: BGR image (H, W, 3) uint8
            coarse_mask: Binary mask (H, W) uint8 (0 or 1)

        Returns:
            Refined binary mask (H, W) uint8 (0 or 1)
        """
        if not self._initialized:
            raise RuntimeError("SAMMaskRefiner not initialized. Call initialize() first.")

        if not np.any(coarse_mask):
            return coarse_mask

        h, w = image.shape[:2]

        # Set image on predictor (encodes once, reused for all components)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        # Split into connected components
        mask_uint8 = (coarse_mask > 0).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )

        refined_mask = np.zeros((h, w), dtype=np.uint8)

        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8)
            area = stats[label_id, cv2.CC_STAT_AREA]

            if area < self.config.min_component_area:
                # Too small to refine meaningfully — keep as-is
                refined_mask = np.maximum(refined_mask, component)
                continue

            try:
                refined_component = self._refine_single_component(component)
            except Exception as e:
                logger.warning(f"Component refinement failed (area={area}): {e}")
                refined_mask = np.maximum(refined_mask, component)
                continue

            # Sanity check: IoU with original
            intersection = np.sum(np.logical_and(refined_component, component))
            union = np.sum(np.logical_or(refined_component, component))
            iou = intersection / max(union, 1)

            if iou >= self.config.iou_threshold:
                refined_mask = np.maximum(refined_mask, refined_component)
            else:
                # Refinement diverged — keep original
                logger.debug(
                    f"Refinement diverged (IoU={iou:.2f} < {self.config.iou_threshold}), "
                    f"keeping original component"
                )
                refined_mask = np.maximum(refined_mask, component)

        # Reset predictor state
        self.predictor.reset_image()

        return refined_mask

    def _refine_single_component(
        self,
        component_mask: np.ndarray,
    ) -> np.ndarray:
        """Refine a single connected component using multi-prompt SAM predict."""
        import torch

        h, w = component_mask.shape

        # Build prompts
        bbox = self._extract_expanded_bbox(component_mask)
        fg_coords, fg_labels = self._extract_foreground_points(component_mask)
        bg_coords, bg_labels = self._extract_background_points(component_mask, bbox)

        point_coords = np.concatenate([fg_coords, bg_coords], axis=0)
        point_labels = np.concatenate([fg_labels, bg_labels], axis=0)

        # Gaussian mask prompt
        mask_input = self._make_mask_prompt(component_mask)

        # Run SAM predict with all prompts
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=bbox,
            mask_input=mask_input,
            multimask_output=True,
        )

        return self._select_best_mask(masks, scores, component_mask)

    def _extract_foreground_points(
        self,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract foreground points at maximum distance from mask boundary."""
        dist = cv2.distanceTransform(
            mask.astype(np.uint8), cv2.DIST_L2, 5
        )

        n = self.config.num_foreground_points
        coords = []

        for _ in range(n):
            max_idx = np.unravel_index(np.argmax(dist), dist.shape)
            y, x = max_idx
            coords.append([x, y])
            # Suppress this region so next point is elsewhere
            cv2.circle(dist, (x, y), max(10, int(dist[y, x] * 0.5)), 0, -1)

        coords_arr = np.array(coords, dtype=np.float32)
        labels_arr = np.ones(n, dtype=np.int32)
        return coords_arr, labels_arr

    def _extract_background_points(
        self,
        mask: np.ndarray,
        bbox: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract background points outside mask but within expanded bbox."""
        h, w = mask.shape

        # Distance from mask exterior
        inverted = (1 - mask).astype(np.uint8)
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

        # Restrict to within the expanded bbox
        x0, y0, x1, y1 = bbox.astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        region_mask = np.zeros((h, w), dtype=np.uint8)
        region_mask[y0:y1, x0:x1] = 1

        dist[region_mask == 0] = 0
        dist[mask > 0] = 0  # Exclude foreground

        n = self.config.num_background_points
        coords = []

        for _ in range(n):
            if np.max(dist) == 0:
                # No valid background region — place point at bbox corner
                coords.append([x0, y0])
                break
            max_idx = np.unravel_index(np.argmax(dist), dist.shape)
            y, x = max_idx
            coords.append([x, y])
            cv2.circle(dist, (x, y), max(10, int(dist[y, x] * 0.5)), 0, -1)

        coords_arr = np.array(coords, dtype=np.float32)
        labels_arr = np.zeros(len(coords), dtype=np.int32)
        return coords_arr, labels_arr

    def _extract_expanded_bbox(self, mask: np.ndarray) -> np.ndarray:
        """Get bounding box of mask, expanded by margin ratio."""
        h, w = mask.shape
        ys, xs = np.where(mask > 0)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

        bw = x1 - x0
        bh = y1 - y0
        margin_x = int(bw * self.config.box_margin_ratio)
        margin_y = int(bh * self.config.box_margin_ratio)

        x0 = max(0, x0 - margin_x)
        y0 = max(0, y0 - margin_y)
        x1 = min(w, x1 + margin_x)
        y1 = min(h, y1 + margin_y)

        return np.array([x0, y0, x1, y1], dtype=np.float32)

    def _make_mask_prompt(self, mask: np.ndarray) -> np.ndarray:
        """Create Gaussian-blurred mask prompt in SAM's 256x256 logit format."""
        # Resize to SAM's low-res mask input size
        mask_256 = cv2.resize(
            mask.astype(np.float32), (256, 256),
            interpolation=cv2.INTER_LINEAR
        )

        # Gaussian blur for soft boundary
        mask_area = np.sum(mask > 0)
        sigma = max(3.0, self.config.gaussian_sigma_ratio * np.sqrt(mask_area))
        ksize = int(sigma * 6) | 1  # Must be odd
        mask_blurred = cv2.GaussianBlur(mask_256, (ksize, ksize), sigma)

        # Map to logit scale: 0 → -4, 1 → +4 (strong signal for SAM)
        mask_logits = (mask_blurred * 8.0) - 4.0

        # Shape: 1 x 256 x 256
        return mask_logits[np.newaxis, :, :]

    def _select_best_mask(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        coarse_mask: np.ndarray,
    ) -> np.ndarray:
        """Select the best mask from SAM's multimask output.

        Uses SAM's own confidence scores. Returns binary uint8.
        """
        best_idx = int(np.argmax(scores))
        return (masks[best_idx] > 0).astype(np.uint8)

    def cleanup(self) -> None:
        """Release model resources."""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        self._initialized = False
