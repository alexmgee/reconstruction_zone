#!/usr/bin/env python3
"""
Advanced Multi-Geometry Masking System with SAM3
=================================================
Author: Reconstruction Zone
License: GPL-3.0

A production-ready masking system that handles fisheye, pinhole, and equirectangular
images using SAM3's text-based prompting capabilities. Designed for removing capture
equipment, operators, and unwanted objects from 360° reconstruction pipelines.

Key Features:
- SAM3 text-based prompting ("remove tripod operator equipment")
- Multi-geometry support (fisheye, pinhole, equirectangular)
- Temporal consistency for video sequences
- Multiple fallback models (FastSAM, EfficientSAM, MobileSAM)
- Interactive refinement interface
- Batch processing with quality control
- GPU acceleration with automatic CPU fallback
"""

import json
import logging

# Setup logging — use NullHandler when stderr is unavailable (pythonw.exe)
import sys as _sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

if _sys.stderr is None:
    logging.basicConfig(level=logging.INFO, handlers=[logging.NullHandler()])
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Distribution gate — Gumroad builds exclude certain models
try:
    from prep360.distribution import is_gumroad as _is_gumroad
except ImportError:
    def _is_gumroad(): return False

# Try importing segmentation models in order of preference
try:
    # SAM3 / SAM 3.1 - Primary model (Meta)
    # Real API: git clone https://github.com/facebookresearch/sam3 && pip install -e .
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False
    warnings.warn("SAM3 not found. Install from: https://github.com/facebookresearch/sam3")

try:
    # FastSAM - Fast fallback
    if _is_gumroad():
        raise ImportError("FastSAM excluded from Gumroad build")
    from ultralytics import FastSAM
    HAS_FASTSAM = True
except Exception as e:
    HAS_FASTSAM = False
    warnings.warn(f"FastSAM not available: {e}")

try:
    # YOLO26 - Production recommendation for class-based detection
    if _is_gumroad():
        raise ImportError("YOLO excluded from Gumroad build")
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception as e:
    HAS_YOLO = False
    warnings.warn(f"YOLO not available: {e}")

try:
    # RF-DETR-Seg - Transformer-based instance segmentation (Roboflow, ICLR 2026)
    import rfdetr as _rfdetr_module
    HAS_RFDETR = True
except ImportError:
    HAS_RFDETR = False

# OpenCV is required
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    raise ImportError("OpenCV is required. Install with: pip install opencv-python")


# COCO class names (80 classes)
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

COCO_NAME_TO_ID = {name: id for id, name in COCO_CLASSES.items()}

# Common presets for photogrammetry masking
CLASS_PRESETS = {
    "person": [0],
    "photographer": [0, 24, 25, 26, 28],  # Person + bags/umbrella/suitcase
    "equipment": [0, 24, 25, 26, 27, 28, 67],  # Person + accessories + phone
    "vehicles": [1, 2, 3, 5, 6, 7, 8],
    "animals": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "all_dynamic": [0, 1, 2, 3, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19],
}


class ImageGeometry(Enum):
    """Supported image geometries."""
    PINHOLE = "pinhole"           # Standard perspective camera
    FISHEYE = "fisheye"           # Fisheye lens (single)
    DUAL_FISHEYE = "dual_fisheye" # Dual fisheye (360 cameras)
    EQUIRECTANGULAR = "equirect"  # 360° equirectangular
    CUBEMAP = "cubemap"           # Cube map faces


class SegmentationModel(Enum):
    """Available segmentation models."""
    SAM3 = "sam3"                 # Meta SAM3 (text prompts)
    YOLO26 = "yolo26"            # YOLO26 instance segmentation (production)
    RFDETR = "rfdetr"             # RF-DETR-Seg transformer segmentation (Roboflow)
    FASTSAM = "fastsam"          # YOLO-based (fast)


class MaskQuality(Enum):
    """Mask quality levels for review."""
    EXCELLENT = "excellent"  # >0.95 confidence
    GOOD = "good"           # 0.85-0.95
    REVIEW = "review"       # 0.70-0.85
    POOR = "poor"          # 0.50-0.70
    REJECT = "reject"      # <0.50


@dataclass
class MaskConfig:
    """Configuration for masking operations."""
    # Model settings
    model: SegmentationModel = SegmentationModel.SAM3
    model_checkpoint: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Text prompts for SAM3 — use simple nouns (compound phrases score poorly)
    remove_prompts: List[str] = field(default_factory=lambda: [
        "person",
        "tripod",
        "backpack",
        "selfie stick",
    ])
    keep_prompts: List[str] = field(default_factory=list)  # Objects to keep
    keep_classes: List[int] = field(default_factory=list)  # COCO class IDs to exclude (derived from keep_prompts)

    # Multi-pass SAM3 prompting — list of (prompt_text, confidence_threshold) tuples
    # When non-empty, overrides remove_prompts and runs SAM3 once per tuple
    multi_pass_prompts: List[tuple] = field(default_factory=list)

    # YOLO26 settings
    yolo_model_size: str = "n"  # n/s/m/l/x
    yolo_classes: List[int] = field(default_factory=lambda: [0, 24, 25, 26, 28])  # photographer preset

    # Quality control
    confidence_threshold: float = 0.70
    review_threshold: float = 0.85
    min_mask_area: int = 100  # Minimum pixels
    max_mask_area_ratio: float = 0.5  # Maximum ratio of image

    # Processing options
    use_temporal_consistency: bool = True
    temporal_window: int = 5  # Frames for temporal smoothing
    batch_size: int = 4
    num_workers: int = 4

    # Geometry-specific settings
    geometry_aware: bool = True
    handle_distortion: bool = True
    pole_mask_expand: float = 1.2  # Expansion factor for pole regions
    nadir_mask_percent: float = 0.0  # Auto-mask bottom N% of equirect (0=off, 5-15 typical)
    fisheye_circle_mask: bool = True   # Auto-mask corners + periphery of fisheye images
    fisheye_margin_percent: float = 0.0  # % of circle radius to trim inward (0=corners only)

    # Cubemap seam fix
    cubemap_overlap: float = 0.0  # Overlap degrees for cubemap faces (0=off, 10=recommended)

    # Static mask overlays (user-authored persistent masks applied to every frame)
    static_mask_paths: List[str] = field(default_factory=list)

    # RF-DETR-Seg settings
    rfdetr_model_size: str = "small"  # nano/small/medium/large


    # SAM3 temporal tracking (new integrated mode — tracking + full post-processing)
    temporal_tracking: bool = False  # Use SAM3 video predictor for detection, then run full pipeline
    temporal_prompt_frame: int = 0   # Frame index on which to apply text prompts

    multi_label: bool = False    # Export per-class segmentation maps (class IDs as pixel values)

    inpaint_masked: bool = False     # Fill masked regions with plausible background
    inpaint_method: str = "telea"    # "telea" (OpenCV) or "ns" (Navier-Stokes) — no extra deps
    inpaint_radius: int = 5          # Inpainting neighborhood radius

    # Post-processing
    mask_dilate_px: int = 0      # Dilate final mask by N pixels (0=off, grows mask edges)
    fill_holes: bool = False     # Fill interior holes in mask (camera mount, equipment gaps)

    # Performance
    torch_compile: bool = False  # Apply torch.compile() for ~20-50% faster GPU inference

    # Output settings
    save_confidence_maps: bool = False
    save_review_images: bool = True
    save_reject_review_images: bool = False
    output_format: str = "png"  # png, jpg, npy

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model': self.model.value,
            'model_checkpoint': self.model_checkpoint,
            'device': self.device,
            'remove_prompts': self.remove_prompts,
            'keep_prompts': self.keep_prompts,
            'yolo_model_size': self.yolo_model_size,
            'yolo_classes': self.yolo_classes,
            'confidence_threshold': self.confidence_threshold,
            'review_threshold': self.review_threshold,
            'min_mask_area': self.min_mask_area,
            'max_mask_area_ratio': self.max_mask_area_ratio,
            'use_temporal_consistency': self.use_temporal_consistency,
            'temporal_window': self.temporal_window,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'geometry_aware': self.geometry_aware,
            'handle_distortion': self.handle_distortion,
            'pole_mask_expand': self.pole_mask_expand,
            'nadir_mask_percent': self.nadir_mask_percent,
            'fisheye_circle_mask': self.fisheye_circle_mask,
            'fisheye_margin_percent': self.fisheye_margin_percent,
            'cubemap_overlap': self.cubemap_overlap,
            'static_mask_paths': self.static_mask_paths,
            'rfdetr_model_size': self.rfdetr_model_size,
            'multi_label': self.multi_label,
            'inpaint_masked': self.inpaint_masked,
            'inpaint_method': self.inpaint_method,
            'inpaint_radius': self.inpaint_radius,
            'mask_dilate_px': self.mask_dilate_px,
            'fill_holes': self.fill_holes,
            'torch_compile': self.torch_compile,
            'save_confidence_maps': self.save_confidence_maps,
            'save_review_images': self.save_review_images,
            'save_reject_review_images': self.save_reject_review_images,
            'output_format': self.output_format
        }

    def save(self, path: Path):
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()

        if path.suffix == '.yaml':
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'MaskConfig':
        """Load configuration from file."""
        path = Path(path)

        if path.suffix == '.yaml':
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)

        # Convert string back to enum
        data['model'] = SegmentationModel(data['model'])

        return cls(**data)


@dataclass
class MaskResult:
    """Result of a masking operation."""
    mask: np.ndarray
    confidence: float
    quality: MaskQuality
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        """Check if mask needs human review."""
        return self.quality in [MaskQuality.REVIEW, MaskQuality.POOR]

    def should_save_review_image(self, include_rejects: bool = False) -> bool:
        """Check whether this result should get a review overlay image."""
        return self.needs_review or (include_rejects and self.quality == MaskQuality.REJECT)

    @property
    def is_valid(self) -> bool:
        """Check if mask is usable."""
        return self.quality != MaskQuality.REJECT


class BaseSegmenter(ABC):
    """Abstract base class for segmentation models."""

    def __init__(self, config: MaskConfig):
        self.config = config
        self.device = config.device
        self.model = None
        self._fisheye_circle_cache: Optional[Tuple[Tuple[int, int, float], np.ndarray]] = None

        # Pre-composite static mask overlays (applied to every frame).
        # Stored as a hash of the current source paths so the pipeline cache
        # can rebuild it when the user adds/edits/removes a layer.
        self._static_composite = None
        self._static_composite_key = None
        self._rebuild_static_composite()

    def _rebuild_static_composite(self):
        """Recompute self._static_composite from self.config.static_mask_paths.

        Pipeline caching reuses a segmenter across runs and patches
        config.static_mask_paths in place, so this must be callable after
        construction to refresh the precomputed overlay. Idempotent: keyed
        on the path tuple so repeated calls with the same paths skip work.
        """
        paths = tuple(self.config.static_mask_paths or ())
        if paths == self._static_composite_key:
            return
        self._static_composite = None
        self._static_composite_key = paths
        if not paths:
            return
        for path in paths:
            raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                logger.warning(f"Could not load static mask: {path}")
                continue
            # Disk convention: black (0) = masked. Internal: 1 = masked.
            binary = (raw < 128).astype(np.uint8)
            if self._static_composite is None:
                self._static_composite = binary
            else:
                if binary.shape != self._static_composite.shape:
                    binary = cv2.resize(binary,
                        (self._static_composite.shape[1], self._static_composite.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
                self._static_composite = np.maximum(self._static_composite, binary)
        if self._static_composite is not None:
            n = len(paths)
            pct = float(np.sum(self._static_composite > 0) / self._static_composite.size * 100)
            logger.info(f"Static mask: {n} layer(s) loaded, {pct:.1f}% coverage")

    def _get_fisheye_circle_mask(
        self, width: int, height: int, margin_percent: float
    ) -> np.ndarray:
        """Return cached fisheye circle mask (0=valid, 1=masked)."""
        key = (width, height, margin_percent)
        if self._fisheye_circle_cache is not None and self._fisheye_circle_cache[0] == key:
            return self._fisheye_circle_cache[1]
        from prep360.core.fisheye_reframer import generate_fisheye_circle_mask
        circle = generate_fisheye_circle_mask(width, height, margin_percent=margin_percent)
        self._fisheye_circle_cache = (key, circle)
        return circle

    @abstractmethod
    def initialize(self):
        """Initialize the model."""
        pass

    @abstractmethod
    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None
    ) -> List[MaskResult]:
        """Segment a single image."""
        pass

    def preprocess_image(
        self,
        image: np.ndarray,
        geometry: ImageGeometry
    ) -> np.ndarray:
        """Preprocess image based on geometry."""

        if geometry == ImageGeometry.EQUIRECTANGULAR and self.config.handle_distortion:
            # Apply adaptive histogram equalization to handle pole distortion
            image = self._enhance_poles(image)

        elif geometry == ImageGeometry.FISHEYE and self.config.handle_distortion:
            # Apply radial enhancement for fisheye
            image = self._enhance_fisheye(image)

        return image

    def _enhance_poles(self, image: np.ndarray) -> np.ndarray:
        """Enhance pole regions in equirectangular images."""
        h, w = image.shape[:2]

        # Create weight map (higher weight at poles)
        weights = np.ones((h, 1))
        pole_region = int(h * 0.15)  # Top/bottom 15%

        # Gradual weight increase towards poles
        for i in range(pole_region):
            weight = 1.0 + (pole_region - i) / pole_region
            weights[i] = weight
            weights[h - 1 - i] = weight

        # Apply weighted CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Weight blend
        l_blended = (l * (2 - weights) + l_enhanced * weights) / 2
        l_blended = np.clip(l_blended, 0, 255).astype(np.uint8)

        enhanced = cv2.merge([l_blended, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _enhance_fisheye(self, image: np.ndarray) -> np.ndarray:
        """Enhance fisheye images with radial correction."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Create radial gradient mask
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        radial_weights = 1.0 + (dist_from_center / max_dist) * 0.5

        # Apply radial enhancement
        enhanced = image.astype(np.float32)
        for c in range(3):
            enhanced[:, :, c] *= radial_weights

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def postprocess_mask(
        self,
        mask: np.ndarray,
        geometry: ImageGeometry,
        final: bool = False,
    ) -> np.ndarray:
        """Postprocess a binary mask (0/1 uint8).

        Called at two stages:
        - ``final=False`` (per-detection): geometry-specific pole expansion
          + morphological cleanup only.  Runs on each individual detection
          mask at face resolution (e.g. 1024x1024 cubemap face).
        - ``final=True`` (combined mask): also applies user-configured
          dilation and fill-holes.  Runs once on the full-resolution
          combined mask (e.g. 7680x3840 equirect) where gaps are large
          enough to detect and bridge.

        Fill-holes algorithm (when ``final=True`` and ``fill_holes=True``):
        1. Morphological close with ~0.4% image-width kernel (15-51px)
           to bridge narrow channels (e.g. gap between arm and camera body).
        2. Flood-fill from padded border to mark all reachable exterior
           background.  Any interior region the flood can't reach is a
           hole — OR it back into the mask.
        """

        if geometry == ImageGeometry.EQUIRECTANGULAR:
            # Expand masks at poles to account for distortion
            mask = self._expand_pole_masks(mask)

        if final:
            # For fisheye: pre-clip detection mask to valid circle area before dilation.
            # This prevents stray 1s in the corner area from being dilated inward into
            # the valid image region. The circle is applied again after dilation.
            if self.config.fisheye_circle_mask and geometry == ImageGeometry.FISHEYE:
                circle = self._get_fisheye_circle_mask(
                    mask.shape[1], mask.shape[0],
                    self.config.fisheye_margin_percent,
                )
                mask = mask * (circle == 0).astype(np.uint8)

            # General mask dilation (before fill-holes so edges close around gaps)
            if self.config.mask_dilate_px > 0:
                k = self.config.mask_dilate_px * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask = cv2.dilate(mask, kernel, iterations=1)

            # Fill interior holes (camera mount, equipment gaps inside person mask)
            if self.config.fill_holes:
                mask = self._fill_mask_holes(mask)

            # Auto-mask nadir region (photographer body at bottom of 360° images)
            if self.config.nadir_mask_percent > 0 and geometry == ImageGeometry.EQUIRECTANGULAR:
                nadir_rows = int(mask.shape[0] * self.config.nadir_mask_percent / 100.0)
                if nadir_rows > 0:
                    before = int(np.sum(mask[-nadir_rows:] > 0))
                    mask[-nadir_rows:] = 1
                    total = mask.shape[1] * nadir_rows
                    logger.info(f"  Nadir mask: bottom {self.config.nadir_mask_percent:.0f}% "
                                f"= {nadir_rows} rows of {mask.shape[0]} "
                                f"({before}/{total} px were already masked)")
            elif geometry == ImageGeometry.EQUIRECTANGULAR:
                logger.info(f"  Nadir mask: OFF (nadir_mask_percent={self.config.nadir_mask_percent})")

            # Auto-mask fisheye corners + periphery (circle already computed above)
            if self.config.fisheye_circle_mask and geometry == ImageGeometry.FISHEYE:
                mask = np.maximum(mask, circle)

            # Apply user-authored static mask overlays
            if self._static_composite is not None:
                static = self._static_composite
                if static.shape != mask.shape:
                    static = cv2.resize(static, (mask.shape[1], mask.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                before = int(np.sum(mask > 0))
                mask = np.maximum(mask, static)
                added = int(np.sum(mask > 0)) - before
                if added > 0:
                    logger.info(f"  Static mask: added {added} px")

        # Clean up small artifacts
        mask = self._morphological_cleanup(mask)

        return mask

    def _evaluate_mask_quality(self, mask: np.ndarray, confidence: float) -> MaskQuality:
        """Multi-metric mask quality evaluation.

        Combines model confidence (50%), edge coherence (20%), and
        compactness (30%) into a weighted composite score. Hard reject
        gates for masks that are too large or too small.
        """
        h, w = mask.shape[:2]
        mask_area = np.sum(mask > 0)
        total_area = h * w

        # Hard reject gates
        if mask_area < self.config.min_mask_area:
            return MaskQuality.REJECT
        if total_area > 0 and (mask_area / total_area) > self.config.max_mask_area_ratio:
            return MaskQuality.REJECT

        # 1. Confidence score (0-1)
        conf_score = min(1.0, max(0.0, confidence))

        # 2. Edge coherence — smooth boundaries score higher
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask
        edges = cv2.Canny(mask_uint8, 50, 150)
        edge_pixels = np.sum(edges > 0)
        # Normalize by mask perimeter estimate (sqrt of area * 4)
        expected_perimeter = 4 * np.sqrt(mask_area) if mask_area > 0 else 1
        edge_ratio = edge_pixels / max(expected_perimeter, 1)
        # Lower ratio = smoother edges = better (clamp to 0-1)
        edge_score = max(0.0, min(1.0, 1.0 - (edge_ratio - 1.0) / 3.0))

        # 3. Compactness (isoperimetric ratio) — regular shapes score higher
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
            else:
                compactness = 0.0
        else:
            compactness = 0.0
        compact_score = min(1.0, compactness)  # Perfect circle = 1.0

        # Weighted composite
        score = conf_score * 0.5 + edge_score * 0.2 + compact_score * 0.3

        if score >= 0.85:
            return MaskQuality.EXCELLENT
        elif score >= 0.70:
            return MaskQuality.GOOD
        elif score >= 0.55:
            return MaskQuality.REVIEW
        elif score >= 0.35:
            return MaskQuality.POOR
        else:
            return MaskQuality.REJECT

    def _expand_pole_masks(self, mask: np.ndarray) -> np.ndarray:
        """Expand masks in pole regions of equirectangular images.

        Kernel size scales with image width so expansion is proportional
        to the actual polar distortion.  At the equator ``pole_mask_expand``
        has no effect; near the poles a small multiplier produces a
        meaningfully larger dilation because the image is wider.
        """
        h, w = mask.shape[:2]
        pole_region = int(h * 0.1)  # Top/bottom 10%

        # Scale kernel with image width: ~0.5% of width per unit of expand
        # e.g. expand=1.2 on 7680px → kernel=46px; on 1024px → kernel=6px
        base = max(3, int(w * 0.005 * self.config.pole_mask_expand))
        kernel_size = base | 1  # ensure odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Expand top pole region
        if np.any(mask[:pole_region]):
            mask[:pole_region] = cv2.dilate(mask[:pole_region], kernel, iterations=1)

        # Expand bottom pole region
        if np.any(mask[-pole_region:]):
            mask[-pole_region:] = cv2.dilate(mask[-pole_region:], kernel, iterations=1)

        return mask

    def _fill_mask_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill interior holes in the mask.

        Two-step approach:
        1. Morphological close with a moderate kernel to bridge narrow
           channels (e.g. gap between arm and camera body).
        2. Flood-fill from the image border to identify exterior background;
           anything the flood can't reach is an interior hole and gets filled.

        Handles both 0/1 and 0/255 mask ranges.
        """
        max_val = mask.max()
        if max_val == 0:
            return mask
        binary = ((mask > 0).astype(np.uint8)) * 255
        h, w = binary.shape[:2]

        # Step 1: morphological close to bridge narrow channels
        # Kernel ~0.4% of image width, clamped to 15-51px
        close_k = max(15, min(51, int(w * 0.004) | 1))  # ensure odd
        close_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kern)

        # Step 2: flood-fill from border to find exterior background
        # Pad with 1px background border so flood always starts outside mask
        padded = np.zeros((h + 2, w + 2), np.uint8)
        padded[1:-1, 1:-1] = closed
        inv = cv2.bitwise_not(padded)
        flood_mask = np.zeros((h + 4, w + 4), np.uint8)
        cv2.floodFill(inv, flood_mask, (0, 0), 0)
        # inv now: exterior bg=0, interior holes=255, foreground=0
        holes = inv[1:-1, 1:-1]
        filled = closed | holes

        # Convert back to original range
        if max_val <= 1:
            return (filled > 0).astype(mask.dtype)
        return filled

    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Clean up mask with morphological operations."""
        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

        # Close small gaps
        kernel_medium = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)

        # Remove tiny components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.config.min_mask_area:
                mask[labels == i] = 0

        return mask


class SAM3Segmenter(BaseSegmenter):
    """SAM3-based per-image segmentation with text prompts (Meta).

    Uses SAM 3's Promptable Concept Segmentation: text prompts find and
    segment all instances of a concept. 848M params, unified DETR detector
    + SAM2 tracker. Image model API unchanged between SAM 3 and 3.1.

    Real API: build_sam3_image_model() → Sam3Processor → set_image → set_text_prompt
    Requires: git clone https://github.com/facebookresearch/sam3 && pip install -e .
    Also requires HuggingFace access approval for model weights.
    """

    def initialize(self):
        """Initialize SAM3 model and processor."""
        if not HAS_SAM3:
            raise ImportError(
                "SAM3 not available. Install from: "
                "git clone https://github.com/facebookresearch/sam3 && "
                "cd sam3 && pip install -e ."
            )

        logger.info("Loading SAM3 image model (848M params)")
        self.model = build_sam3_image_model(enable_inst_interactivity=True)

        if self.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.processor = Sam3Processor(self.model)

    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None,
        geometry: ImageGeometry = ImageGeometry.PINHOLE
    ) -> List[MaskResult]:
        """Segment image using SAM3 text prompts."""
        from contextlib import nullcontext

        from PIL import Image as PILImage

        # Preprocess based on geometry
        image_processed = self.preprocess_image(image, geometry)

        # SAM3 expects PIL RGB image
        image_rgb = cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        # SAM 3.1's fused MLP (perflib/fused.py) casts fc1 to bfloat16;
        # autocast is required so fc2 matches — same as Meta's eval scripts.
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device == 'cuda' else nullcontext()

        with autocast:
            # Set image (encodes once, reused for all prompts)
            inference_state = self.processor.set_image(pil_image)

            # Get prompts
            if prompts is None:
                prompts = {
                    'remove': self.config.remove_prompts,
                    'keep': self.config.keep_prompts
                }

            results = []

            # Process each text prompt
            for prompt_text in prompts.get('remove', []):
                try:
                    # Reset prompts between calls to avoid stale state
                    self.processor.reset_all_prompts(inference_state)

                    output = self.processor.set_text_prompt(
                        state=inference_state, prompt=prompt_text
                    )

                    masks = output.get('masks')
                    scores = output.get('scores')
                    boxes = output.get('boxes')

                    if masks is None or len(masks) == 0:
                        continue

                    # Process each detected instance
                    for i in range(len(masks)):
                        mask = masks[i]

                        # Convert to numpy uint8 if tensor
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu().numpy()

                        # SAM3 returns (1, H, W) per mask — squeeze to (H, W)
                        if mask.ndim == 3 and mask.shape[0] == 1:
                            mask = mask[0]

                        mask = (mask > 0.5).astype(np.uint8)

                        # Resize if needed
                        if mask.shape[:2] != image.shape[:2]:
                            mask = cv2.resize(
                                mask, (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )

                        # Postprocess based on geometry
                        mask_processed = self.postprocess_mask(mask, geometry)

                        score = float(scores[i]) if scores is not None and i < len(scores) else 0.8
                        box = boxes[i].tolist() if boxes is not None and i < len(boxes) else None

                        results.append(MaskResult(
                            mask=mask_processed,
                            confidence=score,
                            quality=self._evaluate_mask_quality(mask_processed, score),
                            metadata={
                                'prompt': prompt_text,
                                'geometry': geometry.value,
                                'model': 'sam3',
                                'box': box
                            }
                        ))
                except Exception as e:
                    logger.warning(f"SAM3 prompt '{prompt_text}' failed: {e}")

            # Keep prompts: detect protected objects and subtract from results
            keep_mask = None
            for keep_text in prompts.get('keep', []):
                try:
                    self.processor.reset_all_prompts(inference_state)
                    output = self.processor.set_text_prompt(
                        state=inference_state, prompt=keep_text
                    )
                    k_masks = output.get('masks')
                    if k_masks is None or len(k_masks) == 0:
                        continue
                    for km in k_masks:
                        if hasattr(km, 'cpu'):
                            km = km.cpu().numpy()
                        if km.ndim == 3 and km.shape[0] == 1:
                            km = km[0]
                        km = (km > 0.5).astype(np.uint8)
                        if km.shape[:2] != image.shape[:2]:
                            km = cv2.resize(km, (image.shape[1], image.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                        if keep_mask is None:
                            keep_mask = km
                        else:
                            keep_mask = np.maximum(keep_mask, km)
                    logger.info(f"  Keep prompt '{keep_text}': protected region found")
                except Exception as e:
                    logger.warning(f"SAM3 keep prompt '{keep_text}' failed: {e}")

            # Subtract keep regions from all remove results
            if keep_mask is not None:
                for r in results:
                    r.mask = (r.mask & ~keep_mask).astype(np.uint8)

        # Merge overlapping masks from same prompt
        return self._merge_similar_masks(results)

    def _run_multi_pass(self, image, multi_pass_prompts, geometry=None):
        """Run SAM3 once per (prompt, threshold) tuple, union all masks."""
        if geometry is None:
            geometry = ImageGeometry.PINHOLE

        combined_mask = None
        all_results = []

        for prompt_text, threshold in multi_pass_prompts:
            results = self.segment_image(
                image,
                prompts={'remove': [prompt_text]},
                geometry=geometry
            )
            results = [r for r in results if r.confidence >= threshold]

            for r in results:
                if combined_mask is None:
                    combined_mask = r.mask.copy()
                else:
                    combined_mask = np.maximum(combined_mask, r.mask)
                all_results.append(r)

        if combined_mask is not None:
            max_conf = max(r.confidence for r in all_results)
            return [MaskResult(
                mask=combined_mask,
                confidence=max_conf,
                quality=self._evaluate_mask_quality(combined_mask, max_conf),
                metadata={
                    'class_name': 'multi_pass_fused',
                    'passes': [(p, t) for p, t in multi_pass_prompts],
                    'individual_results': len(all_results)
                }
            )]
        return []

    def _merge_similar_masks(self, results: List[MaskResult]) -> List[MaskResult]:
        """Merge overlapping masks from the same prompt."""
        if not results:
            return results

        # Group by prompt
        prompt_groups: Dict[str, List[MaskResult]] = {}
        for result in results:
            prompt = result.metadata.get('prompt', 'unknown')
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append(result)

        # Merge within each group
        merged_results = []
        for prompt, group in prompt_groups.items():
            if len(group) == 1:
                merged_results.append(group[0])
            else:
                merged_mask = np.zeros_like(group[0].mask)
                confidences = []

                for result in group:
                    merged_mask = np.logical_or(merged_mask, result.mask).astype(np.uint8)
                    confidences.append(result.confidence)

                avg_confidence = float(np.mean(confidences))

                merged_results.append(MaskResult(
                    mask=merged_mask,
                    confidence=avg_confidence,
                    quality=self._evaluate_mask_quality(merged_mask, avg_confidence),
                    metadata={
                        'prompt': prompt,
                        'merged_count': len(group),
                        'model': 'sam3'
                    }
                ))

        return merged_results


class FastSAMSegmenter(BaseSegmenter):
    """FastSAM-based segmentation (10-100x faster than SAM)."""

    def initialize(self):
        """Initialize FastSAM model."""
        if not HAS_FASTSAM:
            raise ImportError("FastSAM not available")

        model_path = self.config.model_checkpoint or "FastSAM-x.pt"
        self.model = FastSAM(model_path)
        logger.info("Initialized FastSAM")

    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None,
        geometry: ImageGeometry = ImageGeometry.PINHOLE
    ) -> List[MaskResult]:
        """Segment using FastSAM."""

        # Preprocess
        image_processed = self.preprocess_image(image, geometry)

        # Run inference
        results = self.model(
            image_processed,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9
        )

        mask_results = []

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes

            # Process each detected mask
            for i, mask in enumerate(masks):
                mask = (mask > 0.5).astype(np.uint8)

                # Postprocess
                mask_processed = self.postprocess_mask(mask, geometry)

                # Get confidence
                confidence = float(boxes.conf[i]) if boxes is not None else 0.75

                mask_results.append(MaskResult(
                    mask=mask_processed,
                    confidence=confidence,
                    quality=self._evaluate_mask_quality(mask_processed, confidence),
                    metadata={
                        'geometry': geometry.value,
                        'model': 'fastsam'
                    }
                ))

        return mask_results



class YOLO26Segmenter(BaseSegmenter):
    """YOLO26-based instance segmentation (production recommendation).

    Uses ultralytics YOLO26-seg models for class-based detection with
    instance segmentation masks. NMS-free end-to-end inference.
    """

    def initialize(self):
        """Initialize YOLO26 model."""
        if not HAS_YOLO:
            raise ImportError("YOLO not available. Install with: pip install ultralytics")

        size = self.config.yolo_model_size
        model_name = f"yolo26{size}-seg.pt"

        if self.config.model_checkpoint:
            model_path = self.config.model_checkpoint
        else:
            # Check shared model directory first, fall back to ultralytics auto-download
            try:
                try:
                    from model_paths import resolve_yolo26_weights
                except ImportError:
                    from reconstruction_gui.model_paths import resolve_yolo26_weights
                resolved = resolve_yolo26_weights(size)
                model_path = str(resolved) if resolved else model_name
            except Exception:
                model_path = model_name

        logger.info(f"Loading YOLO26 from {model_path}")
        self.model = YOLO(model_path)

    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None,
        geometry: ImageGeometry = ImageGeometry.PINHOLE
    ) -> List[MaskResult]:
        """Segment using YOLO26 instance segmentation."""

        # Preprocess for geometry
        image_processed = self.preprocess_image(image, geometry)

        # Use configured class filter
        classes = self.config.yolo_classes if self.config.yolo_classes else None

        results = self.model.predict(
            image_processed,
            device=self.device,
            classes=classes,
            retina_masks=True,
            conf=self.config.confidence_threshold,
            verbose=False
        )

        mask_results = []

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes

            for i, mask in enumerate(masks):
                class_id = int(boxes.cls[i])

                # Skip detections that match keep_classes
                if self.config.keep_classes and class_id in self.config.keep_classes:
                    continue

                # Ensure mask matches original image size
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

                mask = (mask > 0.5).astype(np.uint8)

                # Postprocess for geometry
                mask_processed = self.postprocess_mask(mask, geometry)

                confidence = float(boxes.conf[i]) if boxes is not None else 0.8
                class_name = results[0].names[class_id]
                box = boxes.xyxy[i].cpu().numpy().astype(int).tolist()

                mask_results.append(MaskResult(
                    mask=mask_processed,
                    confidence=confidence,
                    quality=self._evaluate_mask_quality(mask_processed, confidence),
                    metadata={
                        'geometry': geometry.value,
                        'model': 'yolo26',
                        'class': class_name,
                        'class_id': class_id,
                        'box': box
                    }
                ))

        return mask_results


class RFDETRSegmenter(BaseSegmenter):
    """RF-DETR-Seg instance segmentation (Roboflow, ICLR 2026).

    Transformer-based detector with DINOv2 backbone. Architecturally
    different from YOLO (CNN), making it ideal for ensemble with YOLO26.
    Uses the rfdetr pip package; returns supervision-compatible Detections.
    """

    # Map config size strings to rfdetr class names
    _MODEL_CLASSES = {
        'nano': 'RFDETRSegNano',
        'small': 'RFDETRSegSmall',
        'medium': 'RFDETRSegMedium',
        'large': 'RFDETRSegLarge',
    }

    def initialize(self):
        """Initialize RF-DETR-Seg model."""
        if not HAS_RFDETR:
            raise ImportError(
                "rfdetr is required for RF-DETR-Seg. "
                "Install with: pip install rfdetr"
            )

        size = self.config.rfdetr_model_size
        cls_name = self._MODEL_CLASSES.get(size, 'RFDETRSegSmall')

        if not hasattr(_rfdetr_module, cls_name):
            raise ImportError(
                f"rfdetr module has no {cls_name}. "
                f"Available sizes: {list(self._MODEL_CLASSES.keys())}. "
                f"Update with: pip install -U rfdetr"
            )

        model_cls = getattr(_rfdetr_module, cls_name)

        # Check shared model directory for pre-downloaded weights
        model_kwargs = {}
        try:
            try:
                from model_paths import resolve_rfdetr_seg_weights
            except ImportError:
                from reconstruction_gui.model_paths import resolve_rfdetr_seg_weights
            resolved = resolve_rfdetr_seg_weights(size)
            if resolved:
                model_kwargs["pretrain_weights"] = str(resolved)
                logger.info(f"Loading RF-DETR-Seg: {cls_name} from {resolved}")
            else:
                logger.info(f"Loading RF-DETR-Seg: {cls_name} (rfdetr will download)")
        except Exception:
            logger.info(f"Loading RF-DETR-Seg: {cls_name} (rfdetr will download)")

        self.model = model_cls(**model_kwargs)

    def segment_image(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]] = None,
        geometry: ImageGeometry = ImageGeometry.PINHOLE
    ) -> List[MaskResult]:
        """Segment using RF-DETR-Seg instance segmentation."""
        from PIL import Image as PILImage

        # Preprocess for geometry
        image_processed = self.preprocess_image(image, geometry)

        # RF-DETR expects PIL RGB image
        image_rgb = cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        # Run inference
        detections = self.model.predict(
            pil_image, threshold=self.config.confidence_threshold
        )

        mask_results = []

        if detections.mask is not None and len(detections) > 0:
            masks = detections.mask  # (N, H, W) boolean numpy
            class_ids = detections.class_id  # (N,)
            confidences = detections.confidence  # (N,)
            boxes = detections.xyxy  # (N, 4)

            for i in range(len(detections)):
                class_id = int(class_ids[i])

                # Apply class filter (same COCO classes as YOLO)
                if self.config.yolo_classes and class_id not in self.config.yolo_classes:
                    continue

                # Skip detections that match keep_classes
                if self.config.keep_classes and class_id in self.config.keep_classes:
                    continue

                mask = masks[i].astype(np.uint8)

                # Resize to match input image if needed
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(
                        mask, (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                # Postprocess for geometry
                mask_processed = self.postprocess_mask(mask, geometry)

                confidence = float(confidences[i])
                class_name = COCO_CLASSES.get(class_id, f'class_{class_id}')
                box = boxes[i].astype(int).tolist()

                mask_results.append(MaskResult(
                    mask=mask_processed,
                    confidence=confidence,
                    quality=self._evaluate_mask_quality(mask_processed, confidence),
                    metadata={
                        'geometry': geometry.value,
                        'model': 'rfdetr',
                        'class': class_name,
                        'class_id': class_id,
                        'box': box
                    }
                ))

        return mask_results


class TemporalConsistency:
    """Handle temporal consistency for video sequences."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.mask_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)

    def add_frame(self, mask: np.ndarray, confidence: float):
        """Add frame to history."""
        if self.mask_history and mask.shape != self.mask_history[0].shape:
            self.mask_history.clear()
            self.confidence_history.clear()
        self.mask_history.append(mask)
        self.confidence_history.append(confidence)

    def get_smoothed_mask(self) -> np.ndarray:
        """Get temporally smoothed mask."""
        if not self.mask_history:
            return None

        if len(self.mask_history) == 1:
            return self.mask_history[0]

        # Weighted average based on confidence
        weights = np.array(self.confidence_history)
        weights = weights / weights.sum()

        # Weighted voting
        accumulated = np.zeros_like(self.mask_history[0], dtype=np.float32)
        for mask, weight in zip(self.mask_history, weights):
            accumulated += mask.astype(np.float32) * weight

        # Threshold
        smoothed = (accumulated > 0.5).astype(np.uint8)

        return smoothed

    def detect_inconsistency(self, threshold: float = 0.3) -> bool:
        """Detect temporal inconsistency."""
        if len(self.mask_history) < 2:
            return False

        # Compare recent masks
        recent = self.mask_history[-1]
        previous = self.mask_history[-2]

        # Calculate IoU
        intersection = np.logical_and(recent, previous).sum()
        union = np.logical_or(recent, previous).sum()

        if union == 0:
            return False

        iou = intersection / union

        return iou < threshold


class CubemapProjection:
    """Equirectangular ↔ cubemap conversion for 360 segmentation.

    Standard convention:
    - Equirect: width = 2*height, longitude -pi..pi left-to-right,
      latitude pi/2 (top=north) to -pi/2 (bottom=south)
    - Six faces: front(-Z), back(+Z), left(-X), right(+X), up(+Y), down(-Y)
    """

    FACE_DIRS = {
        'front':  {'axis': 'z', 'sign': -1},
        'back':   {'axis': 'z', 'sign':  1},
        'left':   {'axis': 'x', 'sign': -1},
        'right':  {'axis': 'x', 'sign':  1},
        'up':     {'axis': 'y', 'sign':  1},
        'down':   {'axis': 'y', 'sign': -1},
    }

    def __init__(self, face_size: int = 1024, overlap_degrees: float = 0.0):
        self.face_size = face_size
        self.overlap_degrees = overlap_degrees
        # Standard cubemap: 90° FOV → half-angle 45° → tan(45°) = 1.0
        # With overlap: (90 + overlap)° FOV → tan((90 + overlap) / 2)
        half_fov = (90.0 + overlap_degrees) / 2.0
        self._grid_extent = np.tan(np.radians(half_fov))

    def equirect2cubemap(self, equirect: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert equirectangular image to 6 cubemap faces."""
        h, w = equirect.shape[:2]
        faces = {}
        fs = self.face_size

        # Face grid: standard (-1..1) or expanded for overlap
        extent = self._grid_extent
        grid = np.linspace(-extent, extent, fs)
        u, v = np.meshgrid(grid, grid)

        for name in self.FACE_DIRS:
            x, y, z = self._face_to_xyz(name, u, v)
            # Spherical coords
            lon = np.arctan2(x, -z)  # -pi..pi, front is lon=0
            lat = np.arctan2(y, np.sqrt(x**2 + z**2))  # -pi/2..pi/2

            # Map to equirect pixel coords
            map_x = ((lon / np.pi + 1) / 2 * w).astype(np.float32)
            map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)

            faces[name] = cv2.remap(
                equirect, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP
            )
        return faces

    def cubemap2equirect(
        self,
        face_masks: Dict[str, np.ndarray],
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """Merge 6 face masks back into a single equirectangular mask.

        With overlap_degrees > 0, faces have expanded FOV and overlap at edges.
        In overlap regions, takes the union (np.maximum) of all contributing faces
        so objects at face boundaries are captured if detected by any overlapping face.
        """
        w, h = output_size
        fs = self.face_size
        extent = self._grid_extent

        # Equirect pixel grid → spherical
        u_eq = np.linspace(0, 1, w)
        v_eq = np.linspace(0, 1, h)
        uu, vv = np.meshgrid(u_eq, v_eq)
        lon = (uu - 0.5) * 2 * np.pi  # -pi..pi
        lat = (0.5 - vv) * np.pi       # pi/2..-pi/2

        x = np.cos(lat) * np.sin(lon)
        y = np.sin(lat)
        z = -np.cos(lat) * np.cos(lon)

        output = np.zeros((h, w), dtype=np.uint8)

        if self.overlap_degrees > 0:
            # Overlap mode: each face contributes to an expanded region,
            # union-merge where faces overlap
            for name in self.FACE_DIRS:
                face_mask = face_masks.get(name)
                if face_mask is None or not np.any(face_mask):
                    continue

                # Forward-facing pixels for this face
                facing = self._face_facing(name, x, y, z)
                if not np.any(facing):
                    continue

                # Project facing pixels onto face UV plane
                fu, fv = self._xyz_to_face(
                    name, x[facing], y[facing], z[facing]
                )

                # Pixels within expanded FOV
                in_range = (np.abs(fu) <= extent) & (np.abs(fv) <= extent)
                if not np.any(in_range):
                    continue

                # Map UV to pixel coords (normalized by extent, not 1.0)
                fu_v = fu[in_range]
                fv_v = fv[in_range]
                px = np.clip(
                    ((fu_v / extent + 1) / 2 * (fs - 1)), 0, fs - 1
                ).astype(int)
                py = np.clip(
                    ((fv_v / extent + 1) / 2 * (fs - 1)), 0, fs - 1
                ).astype(int)

                # Build valid mask in output space
                valid = np.zeros((h, w), dtype=bool)
                valid[facing] = in_range

                # Union merge: object detected by ANY overlapping face is kept
                output[valid] = np.maximum(output[valid], face_mask[py, px])
        else:
            # Standard mode: hard face assignment (no overlap)
            abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
            for name in self.FACE_DIRS:
                face_mask = face_masks.get(name)
                if face_mask is None:
                    continue
                region = self._get_face_region(
                    name, x, y, z, abs_x, abs_y, abs_z
                )
                if not np.any(region):
                    continue
                fu, fv = self._xyz_to_face(
                    name, x[region], y[region], z[region]
                )
                px = ((fu + 1) / 2 * (fs - 1)).astype(np.float32)
                py = ((fv + 1) / 2 * (fs - 1)).astype(np.float32)
                px = np.clip(px, 0, fs - 1).astype(int)
                py = np.clip(py, 0, fs - 1).astype(int)
                output[region] = face_mask[py, px]

        return output

    @staticmethod
    def _face_to_xyz(name, u, v):
        """Map face (u,v) in -1..1 to 3D direction.

        Convention: v=-1 is top of face image, v=+1 is bottom.
        Side faces use y=-v so top of image = +Y (up in world).
        """
        ones = np.ones_like(u)
        if name == 'front':
            return u, -v, -ones
        elif name == 'back':
            return -u, -v, ones
        elif name == 'left':
            return -ones, -v, -u
        elif name == 'right':
            return ones, -v, u
        elif name == 'up':
            return u, ones, -v
        elif name == 'down':
            return u, -ones, v

    @staticmethod
    def _xyz_to_face(name, x, y, z):
        """Project 3D direction onto face plane → (u,v) in -1..1.

        Inverse of _face_to_xyz. Side faces negate y for v.
        """
        if name == 'front':
            return x / np.abs(z), -y / np.abs(z)
        elif name == 'back':
            return -x / np.abs(z), -y / np.abs(z)
        elif name == 'left':
            return -z / np.abs(x), -y / np.abs(x)
        elif name == 'right':
            return z / np.abs(x), -y / np.abs(x)
        elif name == 'up':
            return x / np.abs(y), -z / np.abs(y)
        elif name == 'down':
            return x / np.abs(y), z / np.abs(y)

    @staticmethod
    def _face_facing(name, x, y, z):
        """Check if pixels are in the forward hemisphere of this face."""
        if name == 'front':
            return z < 0
        elif name == 'back':
            return z > 0
        elif name == 'left':
            return x < 0
        elif name == 'right':
            return x > 0
        elif name == 'up':
            return y > 0
        elif name == 'down':
            return y < 0

    @staticmethod
    def _get_face_region(name, x, y, z, ax, ay, az):
        """Boolean mask of equirect pixels belonging to this face."""
        if name == 'front':
            return (z < 0) & (az >= ax) & (az >= ay)
        elif name == 'back':
            return (z > 0) & (az >= ax) & (az >= ay)
        elif name == 'left':
            return (x < 0) & (ax >= ay) & (ax >= az)
        elif name == 'right':
            return (x > 0) & (ax >= ay) & (ax >= az)
        elif name == 'up':
            return (y > 0) & (ay >= ax) & (ay >= az)
        elif name == 'down':
            return (y < 0) & (ay >= ax) & (ay >= az)


def resolve_segmentation_model() -> SegmentationModel:
    """Pick the best available segmentation model from module-level HAS_* flags.

    Pure (no instance state). Used by the GUI cache to normalize an
    ``auto_select_model=True`` request into a concrete enum *before* keying
    the cache, so ``("auto", True)`` and ``("SAM3", False)`` resolve to the
    same cached pipeline when SAM3 is auto-resolvable.
    """
    if HAS_SAM3:
        return SegmentationModel.SAM3
    elif HAS_RFDETR:
        logger.info("SAM3 not available, using RF-DETR-Seg")
        return SegmentationModel.RFDETR
    elif HAS_YOLO:
        logger.info("SAM3/RF-DETR not available, using YOLO26")
        return SegmentationModel.YOLO26
    elif HAS_FASTSAM:
        logger.warning("SAM3/RF-DETR/YOLO26 not available, using FastSAM")
        return SegmentationModel.FASTSAM
    else:
        raise RuntimeError("No segmentation models available")


class MaskingPipeline:
    """Main masking pipeline orchestrator."""

    def __init__(
        self,
        config: Optional[MaskConfig] = None,
        auto_select_model: bool = True
    ):
        """
        Initialize masking pipeline.
        
        Args:
            config: Masking configuration
            auto_select_model: Automatically select best available model
        """
        self.config = config or MaskConfig()

        if auto_select_model:
            self.config.model = self._auto_select_model()

        self.segmenter = self._create_segmenter()
        self.segmenter.initialize()

        self.temporal_consistency = None
        if self.config.use_temporal_consistency:
            self.temporal_consistency = TemporalConsistency(
                self.config.temporal_window
            )
        self._frame_counter = 0

        # torch.compile acceleration (optional, ~20-50% faster GPU inference)
        if self.config.torch_compile:
            self._apply_torch_compile()

        logger.info(f"Initialized masking pipeline with {self.config.model.value}")

    def get_sam3_click_handles(self):
        """Return (model, processor) for click-PVS sessions.

        Raises RuntimeError if the active segmenter is not SAM3.
        """
        seg = self.segmenter
        if not isinstance(seg, SAM3Segmenter):
            raise RuntimeError(
                f"Click-PVS requires SAM3 segmenter, got {type(seg).__name__}"
            )
        return seg.model, seg.processor

    def _apply_torch_compile(self):
        """Apply torch.compile() to all loaded models for faster GPU inference.

        Compiles segmenter model encoder.
        Requires PyTorch 2.0+ and CUDA. Skips gracefully if unavailable.
        First inference after compilation is slow (warmup); subsequent calls
        are 20-50% faster.
        """
        try:
            import torch
            if not hasattr(torch, 'compile'):
                logger.warning("torch.compile requires PyTorch 2.0+, skipping")
                return
            if self.config.device != 'cuda' or not torch.cuda.is_available():
                logger.info("torch.compile skipped (CPU mode, minimal benefit)")
                return
        except ImportError:
            return

        compile_kwargs = dict(mode="reduce-overhead", dynamic=True)

        # Compile segmenter model
        if hasattr(self.segmenter, 'model') and self.segmenter.model is not None:
            try:
                seg_model = self.segmenter.model
                if hasattr(seg_model, 'model'):
                    # YOLO/FastSAM: inner PyTorch model
                    seg_model.model = torch.compile(seg_model.model, **compile_kwargs)
                elif hasattr(seg_model, 'image_encoder'):
                    # SAM-style: compile the heavy encoder
                    seg_model.image_encoder = torch.compile(
                        seg_model.image_encoder, **compile_kwargs
                    )
                logger.info("torch.compile applied to segmenter")
            except Exception as e:
                logger.warning(f"torch.compile failed for segmenter: {e}")

    def _auto_select_model(self) -> SegmentationModel:
        """Automatically select best available model.

        Delegates to module-level ``resolve_segmentation_model()`` so the GUI
        cache can call the same logic without needing an instance.
        """
        return resolve_segmentation_model()

    def _create_segmenter(self) -> BaseSegmenter:
        """Create appropriate segmenter based on config."""
        if self.config.model == SegmentationModel.SAM3:
            return SAM3Segmenter(self.config)
        elif self.config.model == SegmentationModel.RFDETR:
            return RFDETRSegmenter(self.config)
        elif self.config.model == SegmentationModel.YOLO26:
            return YOLO26Segmenter(self.config)
        elif self.config.model == SegmentationModel.FASTSAM:
            return FastSAMSegmenter(self.config)
        else:
            raise NotImplementedError(f"Model {self.config.model} not implemented")

    def _segment(
        self,
        image: np.ndarray,
        prompts: Optional[Dict[str, Any]],
        geometry: ImageGeometry
    ) -> List[MaskResult]:
        """Segment image using the configured model."""
        if self.config.multi_pass_prompts and isinstance(self.segmenter, SAM3Segmenter):
            return self.segmenter._run_multi_pass(
                image, self.config.multi_pass_prompts, geometry
            )
        return self.segmenter.segment_image(image, prompts, geometry)

    # ── SAM3 temporal tracking ────────────────────────────────────────

    def _start_tracking_session(
        self,
        frame_paths: List[Path],
    ) -> Optional[Dict[int, np.ndarray]]:
        """Run SAM3 video predictor over frames and return per-frame masks.

        Loads all frames, starts a SAM 3.1 multiplex video session, applies
        text prompts on the configured keyframe, propagates forward (and
        backward if keyframe > 0), and returns a dict mapping frame index
        to a combined binary mask (0/1 uint8, H x W).

        Returns None if SAM3 video predictor is unavailable or the session
        fails. Callers should fall back to per-frame detection.
        """
        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor
        except ImportError:
            logger.warning("SAM3 video predictor not available — temporal tracking disabled")
            return None

        from PIL import Image as PILImage

        n_frames = len(frame_paths)
        if n_frames == 0:
            return {}

        logger.info(f"Temporal tracking: loading {n_frames} frames for SAM3 video session")

        # ── Flash Attention 3 detection + MATH fallback ──
        fa3_available = False
        try:
            from flash_attn_interface import flash_attn_func  # noqa: F401
            fa3_available = True
        except ImportError:
            pass

        if not fa3_available:
            try:
                from sam3.model import decoder as _dec
                from torch.nn.attention import SDPBackend, sdpa_kernel
                _orig_sdpa_kernel = sdpa_kernel
                _dec.sdpa_kernel = lambda *a, **kw: _orig_sdpa_kernel(
                    [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
                )
                logger.info("Patched SAM 3.1 decoder for MATH attention fallback")
            except Exception as e:
                logger.warning(f"Could not patch decoder attention: {e}")

        # ── Build predictor ──
        try:
            predictor = build_sam3_multiplex_video_predictor(use_fa3=fa3_available)
        except Exception as e:
            logger.error(f"SAM3 video predictor build failed: {e}")
            return None

        # ── Load frames as PIL ──
        pil_images = []
        for f in frame_paths:
            pil_images.append(PILImage.open(f).convert("RGB"))

        img_w, img_h = pil_images[0].size

        # ── Start session ──
        try:
            response = predictor.handle_request({
                "type": "start_session",
                "resource_path": pil_images,
                "offload_video_to_cpu": True,
            })
            session_id = response["session_id"]
        except Exception as e:
            logger.error(f"SAM3 video session start failed: {e}")
            del pil_images, predictor
            return None

        # ── Add text prompts ──
        prompt_idx = min(self.config.temporal_prompt_frame, n_frames - 1)
        prompts = self.config.remove_prompts or ["person"]
        combined_prompt = " . ".join(prompts)
        try:
            predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_idx,
                "text": combined_prompt,
            })
            logger.info(f"Temporal tracking: prompt '{combined_prompt}' on frame {prompt_idx}")
        except Exception as e:
            logger.warning(f"Temporal tracking: prompt failed: {e}")

        # ── Propagate forward ──
        outputs_per_frame: Dict[int, Any] = {}
        try:
            for resp in predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
            }):
                outputs_per_frame[resp["frame_index"]] = resp["outputs"]
        except Exception as e:
            logger.error(f"Temporal tracking: forward propagation failed: {e}")

        # ── Propagate backward (if keyframe is not frame 0) ──
        if prompt_idx > 0:
            try:
                for resp in predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "backward",
                }):
                    # Only fill frames not already covered by forward pass
                    fidx = resp["frame_index"]
                    if fidx not in outputs_per_frame:
                        outputs_per_frame[fidx] = resp["outputs"]
            except Exception as e:
                logger.warning(f"Temporal tracking: backward propagation failed: {e}")

        # ── Close session and release resources ──
        try:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })
        except Exception:
            pass
        del pil_images, predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Convert outputs to 0/1 uint8 masks ──
        tracked_masks: Dict[int, np.ndarray] = {}
        for frame_idx in range(n_frames):
            frame_out = outputs_per_frame.get(frame_idx)
            if frame_out is None:
                tracked_masks[frame_idx] = np.zeros((img_h, img_w), dtype=np.uint8)
                continue

            binary_masks = frame_out.get("out_binary_masks")
            if binary_masks is not None and len(binary_masks) > 0:
                # (N, H, W) bool → combined 0/1 uint8
                combined = np.any(binary_masks, axis=0).astype(np.uint8)
            else:
                combined = np.zeros((img_h, img_w), dtype=np.uint8)

            # Resize if SAM3 output resolution differs from original
            if combined.shape[:2] != (img_h, img_w):
                combined = cv2.resize(combined, (img_w, img_h),
                                      interpolation=cv2.INTER_NEAREST)
            tracked_masks[frame_idx] = combined

        det_count = sum(1 for m in tracked_masks.values() if np.any(m))
        logger.info(f"Temporal tracking: {det_count}/{n_frames} frames with detections")

        # ── Keep prompts: subtract protected objects per frame ──
        if self.config.keep_prompts and isinstance(self.segmenter, SAM3Segmenter):
            logger.info("Temporal tracking: applying keep prompts frame-by-frame")
            for frame_idx, tracked in tracked_masks.items():
                if not np.any(tracked):
                    continue
                image = cv2.imread(str(frame_paths[frame_idx]))
                if image is None:
                    continue
                keep_results = self.segmenter.segment_image(
                    image,
                    prompts={'remove': [], 'keep': self.config.keep_prompts},
                    geometry=ImageGeometry.PINHOLE,
                )
                # segment_image returns masks for keep prompts via the
                # keep subtraction logic — but here we need the raw keep
                # masks themselves. Use remove=keep_prompts to detect them.
                keep_mask = None
                for r in self.segmenter.segment_image(
                    image,
                    prompts={'remove': self.config.keep_prompts},
                    geometry=ImageGeometry.PINHOLE,
                ):
                    if keep_mask is None:
                        keep_mask = r.mask.copy()
                    else:
                        keep_mask = np.maximum(keep_mask, r.mask)
                if keep_mask is not None:
                    tracked_masks[frame_idx] = (tracked & ~keep_mask).astype(np.uint8)

        return tracked_masks

    def process_image(
        self,
        image: np.ndarray,
        geometry: ImageGeometry = ImageGeometry.PINHOLE,
        custom_prompts: Optional[Dict[str, List[str]]] = None,
        precomputed_mask: Optional[np.ndarray] = None,
    ) -> MaskResult:
        """Process a single image and return a combined mask.

        For EQUIRECTANGULAR geometry with ``geometry_aware=True``, delegates
        to ``_process_equirectangular`` (cubemap decomposition path) — unless
        a precomputed_mask is provided (e.g. from SAM3 temporal tracking),
        in which case the tracked mask is used directly at equirect resolution.

        For other geometries (PINHOLE, FISHEYE, etc.):
        1. Segment — individual detections, each with lightweight postprocess
           (or use precomputed_mask if provided, skipping detection)
        2. Combine — ``_combine_masks`` merges detections via weighted union
        3. Temporal smoothing — sliding-window averaging across frames
        4. Final postprocess (``final=True``) — dilation + fill-holes on
           the combined mask at full resolution

        Args:
            precomputed_mask: Optional 0/1 uint8 mask from SAM3 temporal
                tracking. When provided, skips _segment() and uses this mask
                as the detection result. Full post-processing still applies.

        Mask values are 0/1 uint8 throughout.
        """

        # For equirectangular with geometry_aware: use cubemap strategy
        # SKIP cubemap when a precomputed tracked mask is provided — the
        # tracker already operated on the original equirect frames and
        # provides equirect-resolution masks directly.
        if (geometry == ImageGeometry.EQUIRECTANGULAR
                and self.config.geometry_aware
                and precomputed_mask is None):
            result = self._process_equirectangular(image, custom_prompts)
            self._frame_counter += 1
            return result

        # Detection step
        if precomputed_mask is not None:
            # Use tracked mask directly — skip _segment()
            results = [MaskResult(
                mask=precomputed_mask,
                confidence=0.85,
                quality=MaskQuality.GOOD,
                metadata={'method': 'sam3_tracking', 'frame': self._frame_counter}
            )]
        else:
            # Normal detection (per-frame, current behavior)
            results = self._segment(image, custom_prompts, geometry)

        if not results:
            # No model masks found. Still run the final geometry-aware
            # postprocess so automatic masks like fisheye circle/corner
            # masking or equirect nadir masking are applied.
            empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            geometry_mask = self.segmenter.postprocess_mask(
                empty_mask, geometry, final=True
            )
            self._frame_counter += 1

            if np.any(geometry_mask):
                return MaskResult(
                    mask=geometry_mask,
                    confidence=1.0,
                    quality=MaskQuality.GOOD,
                    metadata={
                        'message': 'No model masks found; geometry-only mask applied',
                        'geometry_only_mask': True,
                    }
                )

            return MaskResult(
                mask=geometry_mask,
                confidence=0.0,
                quality=MaskQuality.REJECT,
                metadata={'message': 'No masks found'}
            )

        # Combine masks
        combined_mask = self._combine_masks(results)

        coverage = float(np.sum(combined_mask.mask > 0) / combined_mask.mask.size * 100)
        logger.info(f"  Detections: {len(results)}, coverage: {coverage:.1f}%, "
                     f"quality: {combined_mask.quality.value} (conf={combined_mask.confidence:.2f})")

        # Temporal consistency (sliding-window averaging)
        if self.temporal_consistency:
            self.temporal_consistency.add_frame(
                combined_mask.mask,
                combined_mask.confidence
            )
            smoothed_mask = self.temporal_consistency.get_smoothed_mask()
            if smoothed_mask is not None:
                combined_mask.mask = smoothed_mask

        # Final postprocess on combined mask (dilation + fill holes)
        before_pct = float(np.sum(combined_mask.mask > 0) / combined_mask.mask.size * 100)
        combined_mask.mask = self.segmenter.postprocess_mask(
            combined_mask.mask, geometry, final=True
        )
        after_pct = float(np.sum(combined_mask.mask > 0) / combined_mask.mask.size * 100)
        if abs(after_pct - before_pct) > 0.01:
            logger.info(f"  Postprocess: {before_pct:.1f}% → {after_pct:.1f}% "
                        f"(dilation={self.config.mask_dilate_px}px, fill_holes={self.config.fill_holes})")

        self._frame_counter += 1
        return combined_mask

    def reset_sequence(self):
        """Reset state for a new video/image sequence.

        Call this before processing a new batch of sequential frames.
        Resets temporal consistency state.
        """
        self._frame_counter = 0
        if self.temporal_consistency is not None:
            self.temporal_consistency = TemporalConsistency(
                self.config.temporal_window
            )

    def _process_equirectangular(
        self,
        image: np.ndarray,
        custom_prompts: Optional[Dict[str, List[str]]] = None
    ) -> MaskResult:
        """Process equirectangular image via cubemap projection.

        Pipeline stages (in order):
        1. **Cubemap decomposition** — equirect (e.g. 7680x3840) split into
           6 perspective faces (front/back/left/right/up/down) at
           min(1024, w//4) resolution.  Optional overlap_degrees for seam
           blending.
        2. **Per-face segmentation** — each face is segmented independently
           using PINHOLE geometry (SAM3 text prompts or YOLO classes).
           Individual detection masks get lightweight postprocessing
           (morphological cleanup only, ``final=False``).
        3. **Per-face combination** — all detections on a face are merged via
           ``np.maximum`` into a single face mask.
        4. **Cubemap → equirect merge** — ``cubemap2equirect()`` reprojects
           the 6 face masks back to equirectangular space.
        5. **Final postprocessing** (``final=True``) — dilation, fill-holes
           (morphological close + flood-fill), and pole expansion.  Runs
           at full equirect resolution where gaps are large enough to
           detect and bridge.

        Important: Dilation and fill-holes only run during step 7, NOT on
        per-detection masks in step 2.  Per-detection postprocessing at
        1024px face level cannot reliably close gaps that are only a few
        pixels wide at face scale.  The final combined mask at full
        resolution (e.g. 7680x3840) makes these gaps large enough for
        morphological close to bridge and flood-fill to fill.

        Mask values throughout the pipeline are 0/1 uint8 (NOT 0/255).
        Any thresholding must use ``mask > 0``, not ``mask > 127``.
        """
        h, w = image.shape[:2]
        face_size = min(1024, w // 4)  # reasonable face resolution
        overlap = self.config.cubemap_overlap
        cubemap = CubemapProjection(face_size, overlap_degrees=overlap)

        overlap_str = f", overlap={overlap}°" if overlap > 0 else ""
        logger.info(f"Cubemap strategy: {w}x{h} equirect → {face_size}px faces{overlap_str}")
        faces = cubemap.equirect2cubemap(image)

        # Segment each face independently
        face_masks = {}
        all_results = []
        total_face_det = 0
        for face_name, face_img in faces.items():
            results = self._segment(face_img, custom_prompts, ImageGeometry.PINHOLE)
            total_face_det += len(results)
            if results:
                # Combine per-face results into single face mask
                combined = np.zeros(face_img.shape[:2], dtype=np.uint8)
                for r in results:
                    combined = np.maximum(combined, r.mask)
                    all_results.append(r)
                face_masks[face_name] = combined
            else:
                face_masks[face_name] = np.zeros(face_img.shape[:2], dtype=np.uint8)
            logger.info(f"  {face_name}: {len(results)} detections")

        # Merge face masks back to equirectangular
        equirect_mask = cubemap.cubemap2equirect(face_masks, (w, h))

        coverage = float(np.sum(equirect_mask > 0) / equirect_mask.size * 100)
        logger.info(f"  Cubemap merge: {total_face_det} detections → {coverage:.1f}% coverage")

        # Compute combined confidence
        if all_results:
            avg_confidence = np.mean([r.confidence for r in all_results])
        else:
            avg_confidence = 0.0

        result = MaskResult(
            mask=equirect_mask,
            confidence=avg_confidence,
            quality=self.segmenter._evaluate_mask_quality(equirect_mask, avg_confidence),
            metadata={
                'method': 'cubemap',
                'face_size': face_size,
                'face_detections': {name: int(np.any(m)) for name, m in face_masks.items()}
            }
        )

        # Final postprocess on combined equirect mask (dilation + fill holes)
        # This is needed because per-detection postprocess runs at face level
        # (1024px) where the hole may not be enclosed; at full equirect
        # resolution the morphological close + flood-fill can bridge and fill it.
        before_pct = float(np.sum(result.mask > 0) / result.mask.size * 100)
        result.mask = self.segmenter.postprocess_mask(
            result.mask, ImageGeometry.EQUIRECTANGULAR, final=True
        )
        after_pct = float(np.sum(result.mask > 0) / result.mask.size * 100)
        if abs(after_pct - before_pct) > 0.01:
            logger.info(f"  Postprocess: {before_pct:.1f}% → {after_pct:.1f}% "
                        f"(dilation={self.config.mask_dilate_px}px, fill_holes={self.config.fill_holes})")

        return result

    def inpaint_masked_region(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Fill masked regions with plausible background using OpenCV inpainting.

        When masks leave black holes in training images, 3DGS may create dark
        artifacts. Inpainting fills those regions with plausible texture so the
        optimizer gets gradient signal from approximate background instead of void.

        Args:
            image: Input image BGR (H, W, 3), uint8.
            mask: Binary mask (H, W), uint8, 0/255. Non-zero = region to inpaint.

        Returns:
            Inpainted image (H, W, 3), uint8.
        """
        if not np.any(mask):
            return image

        # Ensure mask is binary uint8
        inpaint_mask = (mask > 127).astype(np.uint8) * 255

        method = self.config.inpaint_method
        if method == "ns":
            flag = cv2.INPAINT_NS
        else:
            flag = cv2.INPAINT_TELEA

        inpainted = cv2.inpaint(image, inpaint_mask, self.config.inpaint_radius, flag)
        return inpainted

    def _combine_masks(self, results: List[MaskResult]) -> MaskResult:
        """Combine multiple mask results."""
        if len(results) == 1:
            return results[0]

        # Initialize combined mask
        h, w = results[0].mask.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)

        # Weight by confidence
        weights = []
        valid_results = []

        for result in results:
            if result.is_valid:
                valid_results.append(result)
                weights.append(result.confidence)

        if not valid_results:
            return MaskResult(
                mask=combined,
                confidence=0.0,
                quality=MaskQuality.REJECT,
                metadata={'message': 'All masks rejected'}
            )

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Weighted accumulation — higher-confidence masks contribute more
        accumulated = np.zeros((h, w), dtype=np.float32)
        for result, weight in zip(valid_results, weights):
            accumulated += result.mask.astype(np.float32) * weight
        combined = (accumulated > 0.3).astype(np.uint8)

        # Average confidence
        avg_confidence = np.mean([r.confidence for r in valid_results])

        return MaskResult(
            mask=combined,
            confidence=avg_confidence,
            quality=self.segmenter._evaluate_mask_quality(combined, avg_confidence),
            metadata={
                'combined_count': len(valid_results),
                'method': 'weighted_union'
            }
        )

    def process_batch(
        self,
        images: List[np.ndarray],
        geometry: ImageGeometry = ImageGeometry.PINHOLE,
        custom_prompts: Optional[Dict[str, List[str]]] = None,
        parallel: bool = True
    ) -> List[MaskResult]:
        """
        Process batch of images.
        
        Args:
            images: List of images
            geometry: Image geometry type
            custom_prompts: Optional custom prompts
            parallel: Use parallel processing
        
        Returns:
            List of mask results
        """

        if parallel and self.config.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = []
                for image in images:
                    future = executor.submit(
                        self.process_image,
                        image, geometry, custom_prompts
                    )
                    futures.append(future)

                results = [f.result() for f in futures]
        else:
            results = []
            for image in tqdm(images, desc="Processing images"):
                result = self.process_image(image, geometry, custom_prompts)
                results.append(result)

        return results

    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        geometry: ImageGeometry = ImageGeometry.PINHOLE,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        skip_frames: int = 0,
        save_review: bool = True,
        mask_dir: Path = None,
        review_dir: Path = None,
    ) -> Dict[str, Any]:
        """
        Process video file.
        
        Args:
            video_path: Path to video
            output_dir: Output directory
            geometry: Video geometry type
            start_frame: Starting frame
            end_frame: Ending frame
            skip_frames: Frames to skip
            save_review: Save frames needing review
            mask_dir: Override mask output directory (default: output_dir/masks)
            review_dir: Override review output directory (default: output_dir/review)
        
        Returns:
            Processing statistics
        """

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output structure — use overrides if provided, otherwise
        # default to subdirs inside output_dir.
        if mask_dir is None:
            mask_dir = output_dir / "masks"
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(exist_ok=True)

        if save_review:
            if review_dir is None:
                review_dir = output_dir / "review"
            review_dir = Path(review_dir)
            review_dir.mkdir(exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Processing video: {total_frames} frames @ {fps:.2f} FPS")

        # Set frame range
        end_frame = end_frame or total_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Pre-compute tracked masks if temporal tracking enabled
        tracked_masks = None
        _tracking_temp_dir = None
        if self.config.temporal_tracking:
            import tempfile
            _tracking_temp_dir = Path(tempfile.mkdtemp(prefix="sam3_track_"))
            logger.info(f"Temporal tracking: extracting frames to {_tracking_temp_dir}")
            _track_cap = cv2.VideoCapture(str(video_path))
            _track_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            _track_paths = []
            _tidx = start_frame
            while _tidx < end_frame:
                ret, _tframe = _track_cap.read()
                if not ret:
                    break
                if skip_frames > 0 and _tidx % (skip_frames + 1) != 0:
                    _tidx += 1
                    continue
                _tpath = _tracking_temp_dir / f"{_tidx:06d}.jpg"
                cv2.imwrite(str(_tpath), _tframe)
                _track_paths.append(_tpath)
                _tidx += 1
            _track_cap.release()

            if _track_paths:
                tracked_masks_raw = self._start_tracking_session(_track_paths)
                if tracked_masks_raw is not None:
                    # Re-key from sequential index to actual frame_idx
                    tracked_masks = {}
                    _actual_indices = []
                    _aidx = start_frame
                    while _aidx < end_frame:
                        if skip_frames == 0 or _aidx % (skip_frames + 1) == 0:
                            _actual_indices.append(_aidx)
                        _aidx += 1
                    for seq_idx, actual_idx in enumerate(_actual_indices):
                        if seq_idx in tracked_masks_raw:
                            tracked_masks[actual_idx] = tracked_masks_raw[seq_idx]
                else:
                    logger.warning("Tracking session failed, falling back to per-frame detection")

            # Clean up temp frames
            import shutil
            shutil.rmtree(_tracking_temp_dir, ignore_errors=True)

        # Statistics
        stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'review_frames': 0,
            'rejected_frames': 0,
            'processing_time': 0
        }

        frame_idx = start_frame

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if needed
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            start_time = time.time()

            # Process frame — pass precomputed tracked mask if available
            precomputed = tracked_masks.get(frame_idx) if tracked_masks is not None else None
            result = self.process_image(frame, geometry, precomputed_mask=precomputed)

            stats['processing_time'] += time.time() - start_time
            stats['total_frames'] += 1

            # Save mask
            mask_path = mask_dir / f"{frame_idx:06d}.{self.config.output_format}"

            if self.config.output_format == 'npy':
                np.save(mask_path, result.mask)
            else:
                cv2.imwrite(str(mask_path), (1 - result.mask) * 255)

            # Handle review
            if result.should_save_review_image(self.config.save_reject_review_images) and save_review:
                review_path = review_dir / f"review_{frame_idx:06d}.jpg"

                # Create review image with mask overlay
                review_img = self._create_review_image(frame, result.mask)
                cv2.imwrite(str(review_path), review_img)

                stats['review_frames'] += 1

            if result.quality == MaskQuality.REJECT:
                stats['rejected_frames'] += 1
            else:
                stats['processed_frames'] += 1

            frame_idx += 1

            # Log progress
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{end_frame} frames")

        cap.release()

        # Save statistics
        stats['average_time'] = stats['processing_time'] / max(stats['total_frames'], 1)

        with open(output_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Processing complete: {stats}")

        return stats

    def _create_review_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create review image with mask overlay."""

        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 2] = mask * 255  # Red channel

        # Blend
        review = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)

        return review

    @staticmethod
    def _create_multi_label_map(result: MaskResult) -> np.ndarray:
        """Create a multi-label segmentation map from a MaskResult.

        Pixel values: 0=background, 1-N=different object classes.
        Class assignment comes from metadata['class_id'] or the CLASS_PRESETS
        category mapping (person=1, vehicle=2, equipment=3, animal=4, other=5).

        For Gaussian Grouping and semantic scene decomposition workflows.
        """
        mask = result.mask
        segmap = np.zeros(mask.shape[:2], dtype=np.uint8)

        # Default class mapping from COCO class IDs
        # person=0 → label 1, vehicles=2,3,5,7 → label 2, etc.
        CLASS_LABEL_MAP = {
            0: 1,   # person
            24: 3, 25: 3, 26: 3, 28: 3,  # equipment (backpack, umbrella, handbag, suitcase)
            2: 2, 3: 2, 5: 2, 7: 2,      # vehicles (car, motorcycle, bus, truck)
            14: 4, 15: 4, 16: 4, 17: 4,   # animals (bird, cat, dog, horse)
        }

        class_id = result.metadata.get('class_id')
        if class_id is not None:
            label = CLASS_LABEL_MAP.get(class_id, 5)
            segmap[mask > 127] = label
        else:
            # No class info: assign label 1 (foreground)
            segmap[mask > 127] = 1

        return segmap

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        geometry: ImageGeometry = ImageGeometry.PINHOLE,
        pattern: str = "*.jpg",
        recursive: bool = False,
        result_callback=None,
        skip_existing: bool = False,
        merge_existing: bool = False,
        cancel_event=None,
        mask_dir: Path = None,
        review_dir: Path = None,
    ) -> Dict[str, Any]:
        """
        Process directory of images.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            geometry: Image geometry type
            pattern: File pattern
            recursive: Process recursively
            result_callback: Optional callable(stem, result) called after each image
            skip_existing: Skip images that already have a mask file
            merge_existing: OR new detections into existing mask files instead of
                overwriting. Mutually exclusive with skip_existing — if both are
                True, merge wins.
            mask_dir: Override mask output directory (default: output_dir/masks)
            review_dir: Override review output directory (default: output_dir/review)

        Returns:
            Processing statistics
        """

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find images (supports space-separated multi-patterns, e.g. "*.jpg *.png")
        files = []
        for pat in pattern.split():
            if recursive:
                files.extend(input_dir.rglob(pat))
            else:
                files.extend(input_dir.glob(pat))
        image_files = sorted(set(files))
        # Auto-detect if explicit patterns found nothing
        if not image_files:
            for ext in ("*.jpg", "*.png", "*.jpeg", "*.tif", "*.tiff"):
                image_files = sorted(input_dir.rglob(ext) if recursive else input_dir.glob(ext))
                if image_files:
                    break

        logger.info(f"Found {len(image_files)} images")

        # Create output structure — use overrides if provided, otherwise
        # default to subdirs inside output_dir
        if mask_dir is None:
            if output_dir.name.lower() == "masks":
                mask_dir = output_dir
            else:
                mask_dir = output_dir / "masks"
        mask_dir = Path(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)

        if review_dir is None:
            review_dir = output_dir / "review"
        review_dir = Path(review_dir)
        if self.config.save_review_images:
            review_dir.mkdir(parents=True, exist_ok=True)

        # Process images
        stats = {
            'total_images': len(image_files),
            'processed_images': 0,
            'review_images': 0,
            'rejected_images': 0,
            'processing_time': 0
        }

        # Pre-compute tracked masks if temporal tracking enabled
        tracked_masks = None
        if self.config.temporal_tracking:
            tracked_masks = self._start_tracking_session(image_files)
            if tracked_masks is None:
                logger.warning("Tracking session failed, falling back to per-frame detection")

        skipped = 0
        for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
            # Reset temporal state — directory images are independent, not
            # sequential video frames, so temporal smoothing must not bleed
            # masks from one viewpoint into another.
            # Skip reset when temporal tracking is active (tracked masks
            # provide continuity instead).
            if tracked_masks is None:
                self.reset_sequence()

            # Check for cancellation
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Processing stopped by user")
                break

            # Skip if mask already exists (merge takes precedence if both are set)
            if skip_existing and not merge_existing:
                mask_name = f"{img_path.stem}.{self.config.output_format}"
                if (mask_dir / mask_name).exists():
                    skipped += 1
                    continue

            start_time = time.time()

            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.error(f"Failed to load: {img_path}")
                continue

            # Process — pass precomputed tracked mask if available
            precomputed = tracked_masks.get(idx) if tracked_masks is not None else None
            result = self.process_image(image, geometry, precomputed_mask=precomputed)

            stats['processing_time'] += time.time() - start_time

            # Save mask
            mask_name = f"{img_path.stem}.{self.config.output_format}"
            mask_path = mask_dir / mask_name

            # Merge with existing mask on disk (additive pass) — mutate result.mask
            # so all downstream consumers (multi-label map, inpaint, review image)
            # automatically see the merged value without threading a separate var.
            if merge_existing and mask_path.exists():
                if self.config.output_format == 'npy':
                    existing_internal = np.load(mask_path)
                else:
                    existing = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    existing_internal = (existing < 128).astype(np.uint8) if existing is not None else None
                if existing_internal is not None:
                    if existing_internal.shape == result.mask.shape:
                        result.mask = np.maximum(result.mask, existing_internal)
                    else:
                        logger.warning(f"Merge skipped (size mismatch): {mask_name}")

            if self.config.output_format == 'npy':
                np.save(mask_path, result.mask)
            else:
                cv2.imwrite(str(mask_path), (1 - result.mask) * 255)

            # Save confidence map if requested
            if self.config.save_confidence_maps:
                conf_path = mask_dir / f"conf_{img_path.stem}.npy"
                np.save(conf_path, result.confidence)

            # Save multi-label segmentation map if enabled
            if self.config.multi_label:
                segmap_dir = output_dir / "segmaps"
                segmap_dir.mkdir(exist_ok=True)
                segmap = self._create_multi_label_map(result)
                cv2.imwrite(str(segmap_dir / f"{img_path.stem}.png"), segmap)

            # Save inpainted image if enabled
            if self.config.inpaint_masked and np.any(result.mask):
                inpaint_dir = output_dir / "inpainted"
                inpaint_dir.mkdir(exist_ok=True)
                inpainted = self.inpaint_masked_region(image, result.mask)
                cv2.imwrite(str(inpaint_dir / f"{img_path.stem}.jpg"), inpainted)

            # Handle review
            if result.should_save_review_image(self.config.save_reject_review_images) and self.config.save_review_images:
                review_path = review_dir / f"review_{img_path.stem}.jpg"
                review_img = self._create_review_image(image, result.mask)
                cv2.imwrite(str(review_path), review_img)
                stats['review_images'] += 1

            if result.quality == MaskQuality.REJECT:
                stats['rejected_images'] += 1
            else:
                stats['processed_images'] += 1

            if result_callback:
                result_callback(img_path.stem, result)

        # Save statistics
        stats['skipped_existing'] = skipped
        if skipped:
            logger.info(f"Skipped {skipped} images with existing masks")
        stats['average_time'] = stats['processing_time'] / max(stats['processed_images'], 1)

        with open(output_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Masking complete: {stats['processed_images']}/{stats['total_images']} images "
                     f"in {stats['processing_time']:.1f}s ({stats['average_time']:.1f}s/image)")
        logger.info(f"  Review: {stats.get('review_images', 0)}, Rejected: {stats['rejected_images']}")
        if skipped:
            logger.info(f"  Skipped (existing): {skipped}")

        return stats

class InteractiveMaskRefiner:
    """Interactive tool for mask refinement."""

    def __init__(self, pipeline: MaskingPipeline):
        self.pipeline = pipeline
        self.current_image = None
        self.current_mask = None
        self.history = []

    def refine_mask(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        window_name: str = "Mask Refinement"
    ) -> np.ndarray:
        """
        Interactive mask refinement interface.
        
        Controls:
        - Left click: Add to mask
        - Right click: Remove from mask
        - 'r': Reset to initial
        - 's': Save and exit
        - 'q': Cancel
        - 'u': Undo last change
        - '+/-': Adjust brush size
        """

        self.current_image = image.copy()
        self.current_mask = initial_mask.copy()
        self.history = [initial_mask.copy()]

        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        # Drawing parameters
        self.drawing = False
        self.brush_size = 10
        self.mode = 1  # 1: add, 0: remove

        logger.info("Interactive refinement started. Press 's' to save, 'q' to cancel")

        while True:
            # Create display
            display = self._create_display()
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # Save
                logger.info("Mask saved")
                break
            elif key == ord('q'):
                # Cancel
                logger.info("Refinement cancelled")
                self.current_mask = initial_mask
                break
            elif key == ord('r'):
                # Reset
                self.current_mask = initial_mask.copy()
                self.history = [initial_mask.copy()]
            elif key == ord('u') and len(self.history) > 1:
                # Undo
                self.history.pop()
                self.current_mask = self.history[-1].copy()
            elif key == ord('+'):
                self.brush_size = min(50, self.brush_size + 5)
                logger.info(f"Brush size: {self.brush_size}")
            elif key == ord('-'):
                self.brush_size = max(1, self.brush_size - 5)
                logger.info(f"Brush size: {self.brush_size}")

        cv2.destroyWindow(window_name)
        return self.current_mask

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.mode = 1  # Add
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.mode = 0  # Remove
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Draw on mask
            cv2.circle(
                self.current_mask,
                (x, y),
                self.brush_size,
                self.mode,
                -1
            )
        elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
            if self.drawing:
                self.drawing = False
                # Save to history
                self.history.append(self.current_mask.copy())
                if len(self.history) > 20:
                    self.history.pop(0)

    def _create_display(self) -> np.ndarray:
        """Create display image with mask overlay."""

        # Create colored mask
        colored_mask = np.zeros_like(self.current_image)
        colored_mask[:, :, 1] = self.current_mask * 200  # Green
        colored_mask[:, :, 2] = self.current_mask * 100  # Red

        # Blend
        display = cv2.addWeighted(self.current_image, 0.7, colored_mask, 0.3, 0)

        # Add text
        cv2.putText(
            display, f"Brush: {self.brush_size}px",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            display, "LMB: Add | RMB: Remove | S: Save | Q: Cancel",
            (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        return display


def main():
    """Command-line interface."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced masking system for 360° reconstruction pipelines"
    )

    parser.add_argument(
        "input", type=Path,
        help="Input image, video, or directory"
    )
    parser.add_argument(
        "output", type=Path,
        help="Output directory for masks"
    )

    # Model selection
    parser.add_argument(
        "--model", choices=["sam3", "yolo26", "rfdetr", "fastsam", "auto"],
        default="auto",
        help="Segmentation model to use"
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Model checkpoint path"
    )

    # Geometry
    parser.add_argument(
        "--geometry", choices=["pinhole", "fisheye", "dual_fisheye", "equirect", "cubemap"],
        default="pinhole",
        help="Image geometry type"
    )

    # Prompts
    parser.add_argument(
        "--remove", nargs="+",
        default=["tripod", "operator", "equipment"],
        help="Objects to remove"
    )
    parser.add_argument(
        "--keep", nargs="+", default=[],
        help="Objects to keep"
    )

    # Quality control
    parser.add_argument(
        "--confidence", type=float, default=0.7,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--review-threshold", type=float, default=0.85,
        help="Review threshold"
    )

    # Processing options
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of workers"
    )
    parser.add_argument(
        "--temporal", action="store_true",
        help="Use temporal consistency"
    )

    # Video options
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Starting frame"
    )
    parser.add_argument(
        "--end-frame", type=int,
        help="Ending frame"
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="Frames to skip"
    )

    # Output options
    parser.add_argument(
        "--format", choices=["png", "jpg", "npy"],
        default="png",
        help="Output format"
    )
    parser.add_argument(
        "--save-review", action="store_true",
        help="Save review images"
    )
    parser.add_argument(
        "--save-confidence", action="store_true",
        help="Save confidence maps"
    )

    # Other
    parser.add_argument(
        "--config", type=Path,
        help="Configuration file"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enable interactive refinement"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use"
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = MaskConfig.load(args.config)
    else:
        # Map model choice
        model_map = {
            'sam3': SegmentationModel.SAM3,
            'yolo26': SegmentationModel.YOLO26,
            'rfdetr': SegmentationModel.RFDETR,
            'fastsam': SegmentationModel.FASTSAM,
            'auto': None
        }

        # Set device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device

        config = MaskConfig(
            model=model_map[args.model] if args.model != 'auto' else None,
            model_checkpoint=str(args.checkpoint) if args.checkpoint else None,
            device=device,
            remove_prompts=args.remove,
            keep_prompts=args.keep,
            confidence_threshold=args.confidence,
            review_threshold=args.review_threshold,
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_temporal_consistency=args.temporal,
            save_confidence_maps=args.save_confidence,
            save_review_images=args.save_review,
            output_format=args.format
        )

    # Save config
    config.save(args.output / "mask_config.yaml")

    # Map geometry
    geometry_map = {
        'pinhole': ImageGeometry.PINHOLE,
        'fisheye': ImageGeometry.FISHEYE,
        'dual_fisheye': ImageGeometry.DUAL_FISHEYE,
        'equirect': ImageGeometry.EQUIRECTANGULAR,
        'cubemap': ImageGeometry.CUBEMAP
    }
    geometry = geometry_map[args.geometry]

    # Create pipeline
    pipeline = MaskingPipeline(
        config=config,
        auto_select_model=(args.model == 'auto')
    )

    # Process input
    if args.input.is_file():
        # Check if video or image
        if args.input.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video
            stats = pipeline.process_video(
                video_path=args.input,
                output_dir=args.output,
                geometry=geometry,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                skip_frames=args.skip_frames,
                save_review=args.save_review
            )
        else:
            # Single image
            image = cv2.imread(str(args.input))
            if image is None:
                logger.error(f"Failed to load: {args.input}")
                return

            # Process
            result = pipeline.process_image(image, geometry)

            # Interactive refinement if requested
            if args.interactive and result.needs_review:
                refiner = InteractiveMaskRefiner(pipeline)
                result.mask = refiner.refine_mask(image, result.mask)

            # Save
            args.output.mkdir(parents=True, exist_ok=True)
            mask_path = args.output / f"mask.{args.format}"

            if args.format == 'npy':
                np.save(mask_path, result.mask)
            else:
                cv2.imwrite(str(mask_path), (1 - result.mask) * 255)

            logger.info(f"Saved mask to: {mask_path}")
            logger.info(f"Quality: {result.quality.value}, Confidence: {result.confidence:.3f}")

    else:
        # Directory
        stats = pipeline.process_directory(
            input_dir=args.input,
            output_dir=args.output,
            geometry=geometry,
            pattern="*.jpg",
            recursive=False
        )

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
