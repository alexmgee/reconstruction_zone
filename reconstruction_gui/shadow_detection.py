"""
Shadow Detection Framework for Photogrammetry Masking
=====================================================

Extensible shadow detection with pluggable detectors, verification,
and spatial filtering. Mirrors the BaseSegmenter ABC pattern from reconstruction_pipeline.py.

Adding a new detector:
    1. Add enum value to ShadowDetectorType
    2. Subclass BaseShadowDetector with initialize() and detect()
    3. Add to _DETECTOR_MAP in create_shadow_detector()
    4. (Optional) Register weights in ShadowWeightManager.REGISTRY
    5. Add string to GUI dropdown in reconstruction_zone.py

Pipeline stages:
    Primary Detection -> Verification -> Spatial Filtering -> Cleanup
    Each stage is optional/configurable via ShadowConfig.
"""

import logging
import hashlib
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class ShadowDetectorType(Enum):
    """Available shadow detection methods."""
    # Tier 1 — immediate (no ML dependencies)
    BRIGHTNESS_HEURISTIC = "brightness"
    CHROMATICITY_C1C2C3 = "c1c2c3"
    HYBRID_INTENSITY = "hybrid"
    # Tier 1 — targeted (per-person search, no ML dependencies)
    TARGETED_PERSON = "targeted_person"
    # Tier 1 — ML (PyTorch required, pretrained weights available)
    SDDNET = "sddnet"
    SILT = "silt"
    CAREAGA_INTRINSIC = "careaga"
    # Tier 2 — Instance shadow detection (SAM 2.1 + Grounding DINO)
    OPSEG_INSTANCE = "opseg"


class ShadowSpatialMode(Enum):
    """How to spatially filter detected shadows relative to object masks."""
    ALL = "all"                    # Keep all detected shadows
    NEAR_OBJECTS = "near_objects"  # Only shadows within dilation_radius of objects
    CONNECTED = "connected"        # Only shadows contiguous with object mask edges


class ShadowMaskQuality(Enum):
    """Shadow mask quality levels — mirrors reconstruction_pipeline.MaskQuality."""
    EXCELLENT = "excellent"
    GOOD = "good"
    REVIEW = "review"
    POOR = "poor"
    REJECT = "reject"


@dataclass
class ShadowMaskResult:
    """Result of shadow detection — duck-type compatible with reconstruction_pipeline.MaskResult.

    reconstruction_pipeline.process_image only accesses .mask and .metadata from shadow results,
    so this lightweight class avoids importing reconstruction_pipeline (which pulls in torch).
    """
    mask: np.ndarray
    confidence: float
    quality: ShadowMaskQuality
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return self.quality in (ShadowMaskQuality.REVIEW, ShadowMaskQuality.POOR)

    @property
    def is_valid(self) -> bool:
        return self.quality != ShadowMaskQuality.REJECT


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShadowConfig:
    """Configuration for the shadow detection pipeline."""

    # Which detectors to run
    primary_detector: ShadowDetectorType = ShadowDetectorType.OPSEG_INSTANCE
    verification_detector: Optional[ShadowDetectorType] = None

    # Spatial filtering
    spatial_mode: ShadowSpatialMode = ShadowSpatialMode.NEAR_OBJECTS
    dilation_radius: int = 50           # Pixels to dilate object mask for proximity
    min_shadow_area: int = 100          # Minimum shadow region size in pixels

    # Detection thresholds
    confidence_threshold: float = 0.5   # For ML detector probability maps
    verification_agreement: float = 0.3 # Min IoU per region to survive verification

    # Classical detector parameters
    darkness_ratio_threshold: float = 0.7   # Pixel/local_mean ratio below = dark
    chromaticity_threshold: float = 0.15    # Max chromaticity deviation for c1c2c3

    # ML detector parameters
    model_checkpoint: Optional[str] = None  # Override auto-download path
    device: str = "cpu"                     # Inherited from MaskConfig at runtime
    weights_dir: Optional[str] = None       # Default: ~/.prep360_shadow_weights/

    # OpSeg-specific
    grounding_dino_prompt: Optional[str] = None  # Auto-derived from remove prompts if None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_detector': self.primary_detector.value,
            'verification_detector': (
                self.verification_detector.value
                if self.verification_detector else None
            ),
            'spatial_mode': self.spatial_mode.value,
            'dilation_radius': self.dilation_radius,
            'min_shadow_area': self.min_shadow_area,
            'confidence_threshold': self.confidence_threshold,
            'verification_agreement': self.verification_agreement,
            'darkness_ratio_threshold': self.darkness_ratio_threshold,
            'chromaticity_threshold': self.chromaticity_threshold,
            'model_checkpoint': self.model_checkpoint,
            'device': self.device,
            'weights_dir': self.weights_dir,
            'grounding_dino_prompt': self.grounding_dino_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShadowConfig':
        data = dict(data)
        if 'primary_detector' in data and isinstance(data['primary_detector'], str):
            data['primary_detector'] = ShadowDetectorType(data['primary_detector'])
        if 'verification_detector' in data:
            v = data['verification_detector']
            data['verification_detector'] = (
                ShadowDetectorType(v) if v else None
            )
        if 'spatial_mode' in data and isinstance(data['spatial_mode'], str):
            data['spatial_mode'] = ShadowSpatialMode(data['spatial_mode'])
        return cls(**data)


# ══════════════════════════════════════════════════════════════════════════════
# Base class
# ══════════════════════════════════════════════════════════════════════════════

class BaseShadowDetector(ABC):
    """Abstract base class for shadow detectors.

    Mirrors BaseSegmenter from reconstruction_pipeline.py but with shadow-specific interface:
    - detect() receives image + optional object_mask context
    - Returns (shadow_mask, confidence) tuple
    """

    def __init__(self, config: ShadowConfig):
        self.config = config
        self.device = config.device
        self.model = None
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Load model weights / prepare detector. Called once before detect().
        Must set self._initialized = True on success.
        Raise ImportError if required packages are missing.
        Raise FileNotFoundError if weights cannot be found/downloaded.
        """
        pass

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Detect shadows in an image.

        Args:
            image: BGR image (H, W, 3) uint8
            object_mask: Optional binary mask (H, W) uint8 of detected objects.
                         Used by some detectors for spatial context.

        Returns:
            shadow_mask: Binary mask (H, W) uint8, 255 = shadow pixel
            confidence: Float 0.0-1.0 representing detection confidence
        """
        pass

    @property
    def requires_gpu(self) -> bool:
        """Whether this detector needs a GPU. Override in ML subclasses."""
        return False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def cleanup(self) -> None:
        """Release model resources."""
        self.model = None
        self._initialized = False


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1 Classical Detectors
# ══════════════════════════════════════════════════════════════════════════════

class BrightnessHeuristicDetector(BaseShadowDetector):
    """Brightness-ratio shadow detection — port of reconstruction_pipeline._detect_shadows().

    Finds pixels significantly darker than their local neighborhood,
    optionally restricted to regions adjacent to object masks via
    downward dilation. Pure NumPy/OpenCV, no ML dependencies.
    """

    def initialize(self) -> None:
        self._initialized = True

    def detect(
        self,
        image: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Relative darkness against local mean
        local_mean = cv2.blur(gray, (51, 51))
        darkness_ratio = gray.astype(np.float32) / (local_mean.astype(np.float32) + 1)
        shadow_candidates = (
            (darkness_ratio < self.config.darkness_ratio_threshold) * 255
        ).astype(np.uint8)

        # Bridge gaps
        kernel = np.ones((7, 7), np.uint8)
        shadow_candidates = cv2.morphologyEx(
            shadow_candidates, cv2.MORPH_CLOSE, kernel
        )

        # If object mask provided, restrict to downward attachment zone
        if object_mask is not None and np.any(object_mask):
            attach_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 40))
            person_extended = cv2.dilate(
                (object_mask > 0).astype(np.uint8) * 255,
                attach_kernel, iterations=1
            )
            shadow_mask = cv2.bitwise_and(shadow_candidates, person_extended)
        else:
            shadow_mask = shadow_candidates

        # Remove small fragments
        obj_area = int(np.sum(object_mask > 0)) if object_mask is not None else 0
        min_area = max(self.config.min_shadow_area, int(obj_area * 0.05))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            shadow_mask, connectivity=8
        )
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                shadow_mask[labels == i] = 0

        shadow_pixels = int(np.sum(shadow_mask > 0))
        if shadow_pixels < min_area:
            return np.zeros((h, w), dtype=np.uint8), 0.0

        return shadow_mask, 0.7


class ChromaticityC1C2C3Detector(BaseShadowDetector):
    """Shadow detection via c1c2c3 chromaticity invariants.

    Exploits the physics of shadow formation: shadows change illumination
    intensity but largely preserve chromaticity (color ratios) on Lambertian
    surfaces. Finds regions where intensity drops but chromaticity is stable.

    Excellent on uniform surfaces (sand, concrete, asphalt). Fails on
    textured or non-uniform surfaces. Can serve as standalone detector
    OR as a verification filter for ML detector output.

    Pure NumPy/OpenCV. No model weights.
    """

    def initialize(self) -> None:
        self._initialized = True

    def detect(
        self,
        image: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        img_f = image.astype(np.float64) + 1e-6
        B, G, R = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

        # c1c2c3 chromaticity features (illumination-invariant)
        c1 = np.arctan2(R, np.maximum(G, B))
        c2 = np.arctan2(G, np.maximum(R, B))
        c3 = np.arctan2(B, np.maximum(R, G))
        chroma = np.stack([c1, c2, c3], axis=-1)

        # Intensity (grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Local statistics
        ksize = 51
        local_mean_intensity = cv2.blur(gray, (ksize, ksize))
        local_mean_chroma = np.stack([
            cv2.blur(chroma[:, :, i], (ksize, ksize)) for i in range(3)
        ], axis=-1)

        # Criterion 1: significantly darker than local mean
        intensity_ratio = gray / (local_mean_intensity + 1e-6)
        is_dark = intensity_ratio < self.config.darkness_ratio_threshold

        # Criterion 2: chromaticity stable (same material, just darker)
        chroma_diff = np.sqrt(np.sum((chroma - local_mean_chroma) ** 2, axis=-1))
        is_same_chroma = chroma_diff < self.config.chromaticity_threshold

        # Shadow = dark AND chromatically stable
        shadow_mask = (is_dark & is_same_chroma).astype(np.uint8) * 255

        # Cleanup
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

        shadow_pixels = int(np.sum(shadow_mask > 0))
        if shadow_pixels < self.config.min_shadow_area:
            return np.zeros((h, w), dtype=np.uint8), 0.0

        return shadow_mask, 0.65


class HybridIntensityChromaticityDetector(BaseShadowDetector):
    """Combined HSV intensity + hue stability shadow detection.

    Uses V channel for intensity drop and H channel for chromaticity
    verification in a single pass. More robust than c1c2c3 alone on
    varied outdoor surfaces because HSV hue handles the circular
    nature of color better.

    Pure NumPy/OpenCV. No model weights.
    """

    def initialize(self) -> None:
        self._initialized = True

    def detect(
        self,
        image: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Intensity drop: V channel relative to local mean
        local_mean_V = cv2.blur(V, (51, 51))
        intensity_ratio = V / (local_mean_V + 1e-6)
        is_dark = intensity_ratio < self.config.darkness_ratio_threshold

        # Hue stability (circular, 0-180 in OpenCV HSV)
        local_mean_H = cv2.blur(H, (51, 51))
        hue_diff = np.abs(H - local_mean_H)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        hue_stable = hue_diff < 15  # Within 15 degrees

        # Saturation should not spike dramatically in shadow
        local_mean_S = cv2.blur(S, (51, 51))
        sat_ratio = S / (local_mean_S + 1e-6)
        sat_reasonable = sat_ratio < 1.5

        # Shadow = dark + hue stable + saturation reasonable
        shadow_mask = (is_dark & hue_stable & sat_reasonable).astype(np.uint8) * 255

        # Cleanup
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

        shadow_pixels = int(np.sum(shadow_mask > 0))
        if shadow_pixels < self.config.min_shadow_area:
            return np.zeros((h, w), dtype=np.uint8), 0.0

        return shadow_mask, 0.70


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1 Targeted Detector
# ══════════════════════════════════════════════════════════════════════════════

class TargetedPersonShadowDetector(BaseShadowDetector):
    """Per-person targeted shadow search.

    Instead of detecting ALL shadows globally and filtering, this detector
    starts from each individual person detection and searches outward from
    their feet for attached cast shadows. This inverts the usual approach:

        Classical: detect everywhere → filter to near person  (misses weak shadows)
        Targeted:  start at person → search outward            (finds weak shadows)

    Works because constraining WHERE to look allows much more relaxed
    thresholds. A 5-10% brightness drop that's invisible globally becomes
    a strong signal when you know you're searching a shadow corridor
    extending from a person's feet.

    Pure NumPy/OpenCV. No model weights. Designed for outdoor fisheye/360
    imagery with strong directional sunlight.
    """

    # Configurable parameters with sensible defaults for outdoor fisheye
    DARKNESS_RATIO = 0.93       # Much more relaxed than classical 0.70
    KERNEL_SCALE = 0.05         # Local mean kernel = 5% of image width
    SEARCH_RADIUS_SCALE = 2.5   # Search corridor = 2.5x person height
    MIN_SHADOW_FRACTION = 0.03  # Shadow must be >= 3% of person area
    HUE_TOLERANCE = 20          # Degrees of hue stability
    FOOT_BAND_FRACTION = 0.15   # Bottom 15% of person bbox = feet region

    def initialize(self) -> None:
        self._initialized = True

    def detect(
        self,
        image: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Fallback when individual masks aren't available.

        Runs connected-component analysis on the combined object mask to
        extract individual person regions, then searches each one.
        """
        h, w = image.shape[:2]
        if object_mask is None or not np.any(object_mask):
            return np.zeros((h, w), dtype=np.uint8), 0.0

        # Split combined mask into individual components
        obj_binary = (object_mask > 0).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            obj_binary, connectivity=8
        )

        individual_masks = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 500:  # Skip tiny fragments
                continue
            component = (labels == i).astype(np.uint8)
            individual_masks.append(component)

        if not individual_masks:
            return np.zeros((h, w), dtype=np.uint8), 0.0

        return self.detect_per_person(image, individual_masks, object_mask)

    def detect_per_person(
        self,
        image: np.ndarray,
        individual_masks: List[np.ndarray],
        combined_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Search for cast shadows attached to each person detection.

        Args:
            image: BGR image (H, W, 3) uint8
            individual_masks: List of binary masks (H, W) uint8, one per person
            combined_mask: Optional combined object mask for exclusion

        Returns:
            shadow_mask: Binary mask (H, W) uint8, 255 = shadow pixel
            confidence: Float 0.0-1.0
        """
        h, w = image.shape[:2]

        # Precompute image features once (shared across all persons)
        # Use GRAYSCALE for intensity (not HSV V, which is max(R,G,B) and
        # doesn't represent perceptual brightness on colored surfaces).
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
        H_chan = hsv[:, :, 0]
        S = hsv[:, :, 1]

        # Exclude fisheye black border from all searches
        gray_u8 = gray.astype(np.uint8)
        _, valid_roi = cv2.threshold(gray_u8, 10, 255, cv2.THRESH_BINARY)

        # Build combined exclusion mask (don't search inside any person)
        if combined_mask is not None:
            exclusion = (combined_mask > 0).astype(np.uint8)
        else:
            exclusion = np.zeros((h, w), dtype=np.uint8)
            for pm in individual_masks:
                exclusion = np.maximum(exclusion, pm)

        # ── Masked local mean: exclude objects from reference ──
        # Standard cv2.blur includes person bodies and dark objects in the
        # local mean, which drags it down and makes actual shadows appear
        # BRIGHTER than the mean. Using a masked blur gives a clean "lit
        # background surface" reference.
        ksize = max(51, int(w * self.KERNEL_SCALE) | 1)  # Ensure odd
        bg_valid = ((exclusion == 0) & (valid_roi > 0)).astype(np.float64)
        gray_bg = gray * bg_valid
        H_bg = H_chan * bg_valid
        S_bg = S * bg_valid

        gray_sum = cv2.blur(gray_bg, (ksize, ksize))
        H_sum = cv2.blur(H_bg, (ksize, ksize))
        S_sum = cv2.blur(S_bg, (ksize, ksize))
        bg_count = cv2.blur(bg_valid, (ksize, ksize))

        # Avoid division by zero in areas dominated by objects
        safe_count = np.maximum(bg_count, 0.01)
        local_mean_gray = gray_sum / safe_count
        local_mean_H = H_sum / safe_count
        local_mean_S = S_sum / safe_count

        # Intensity ratio: grayscale vs background-only reference
        ratio = gray / (local_mean_gray + 1e-6)

        # Hue stability map (circular distance)
        hue_diff = np.abs(H_chan - local_mean_H)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)

        # Use configurable darkness threshold
        darkness_threshold = getattr(
            self.config, 'targeted_darkness_ratio', self.DARKNESS_RATIO
        )

        # Shadow candidates: dark + hue-stable + within valid ROI
        is_dark = ratio < darkness_threshold
        hue_stable = hue_diff < self.HUE_TOLERANCE
        candidates_bool = is_dark & hue_stable & (valid_roi > 0)

        # Saturation check: shadows shouldn't spike saturation
        sat_ratio = S / (local_mean_S + 1e-6)
        candidates_bool &= (sat_ratio < 1.5)

        candidates = candidates_bool.astype(np.uint8) * 255

        # Light morphological cleanup on candidates
        morph_k = np.ones((3, 3), np.uint8)
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, morph_k)
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, morph_k)

        # Search each person
        all_shadows = np.zeros((h, w), dtype=np.uint8)
        confidences = []

        for person_mask in individual_masks:
            shadow, conf = self._find_person_shadow(
                person_mask, candidates, exclusion, h, w
            )
            if shadow is not None:
                all_shadows = np.maximum(all_shadows, shadow)
                confidences.append(conf)

        if np.sum(all_shadows) == 0:
            return np.zeros((h, w), dtype=np.uint8), 0.0

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return all_shadows, avg_conf

    def _find_person_shadow(
        self,
        person_mask: np.ndarray,
        candidates: np.ndarray,
        exclusion: np.ndarray,
        h: int,
        w: int,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Search for a cast shadow attached to one person.

        Strategy:
        1. Find the person's feet (bottom edge of mask)
        2. Create a search fan extending outward from feet
        3. Find shadow candidates within the fan
        4. Grow connected components from seeds touching the feet
        """
        person_ys, person_xs = np.where(person_mask > 0)
        if len(person_ys) == 0:
            return None, 0.0

        # Person bounding box
        y_min, y_max = int(person_ys.min()), int(person_ys.max())
        x_min, x_max = int(person_xs.min()), int(person_xs.max())
        person_height = y_max - y_min
        person_width = x_max - x_min
        person_area = int(np.sum(person_mask > 0))

        if person_height < 20 or person_area < 200:
            return None, 0.0  # Too small to reliably find shadow

        # ── Step 1: Locate feet ──
        # Feet = bottom portion of person mask
        foot_band = max(10, int(person_height * self.FOOT_BAND_FRACTION))
        foot_threshold_y = y_max - foot_band
        foot_pixels_y = person_ys[person_ys >= foot_threshold_y]
        foot_pixels_x = person_xs[person_ys >= foot_threshold_y]

        if len(foot_pixels_y) == 0:
            return None, 0.0

        foot_center_x = int(np.median(foot_pixels_x))
        foot_center_y = int(np.max(foot_pixels_y))

        # ── Step 2: Create search fan ──
        search_radius = int(person_height * self.SEARCH_RADIUS_SCALE)
        search_mask = np.zeros((h, w), dtype=np.uint8)

        # Wide fan: 240° arc centered below the person's feet
        # In image space, shadows typically extend outward from feet
        # Use ellipse for the fan shape — wider than tall to cover ground plane
        cv2.ellipse(
            search_mask,
            (foot_center_x, foot_center_y),
            (search_radius, search_radius),
            0,        # rotation angle
            -120,     # start angle (left of downward)
            120,      # end angle (right of downward)
            255,
            -1,       # filled
        )

        # Also add a smaller upward fan for shadows cast behind (sun behind person)
        upward_radius = int(search_radius * 0.6)
        cv2.ellipse(
            search_mask,
            (foot_center_x, foot_center_y),
            (upward_radius, upward_radius),
            0,
            -240,     # upward arc
            -120,
            255,
            -1,
        )

        # Exclude all detected objects from search
        search_mask[exclusion > 0] = 0

        # ── Step 3: Shadow candidates in search region ──
        search_candidates = cv2.bitwise_and(candidates, search_mask)

        if np.sum(search_candidates) == 0:
            return None, 0.0

        # ── Step 4: Proximity-based seed finding ──
        # The shadow may not start right at the person's feet — there can
        # be a gap (e.g., person stands on grass, shadow falls on concrete
        # beyond). Use proximity to the ENTIRE person mask, not just feet.
        person_binary = person_mask * 255

        # Stage A: tight proximity (0.15x person height) — high confidence
        prox_tight = max(20, int(person_height * 0.15))
        prox_kernel_t = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (prox_tight * 2 + 1, prox_tight * 2 + 1)
        )
        person_expanded_t = cv2.dilate(person_binary, prox_kernel_t)
        person_expanded_t[exclusion > 0] = 0
        seeds = cv2.bitwise_and(search_candidates, person_expanded_t)

        if np.sum(seeds) == 0:
            # Stage B: wider proximity (0.5x person height) — bridges gaps
            prox_wide = max(50, int(person_height * 0.5))
            prox_kernel_w = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (prox_wide * 2 + 1, prox_wide * 2 + 1)
            )
            person_expanded_w = cv2.dilate(person_binary, prox_kernel_w)
            person_expanded_w[exclusion > 0] = 0
            seeds = cv2.bitwise_and(search_candidates, person_expanded_w)
            if np.sum(seeds) == 0:
                return None, 0.0

        # Grow: find all connected shadow components containing seeds
        combined = cv2.bitwise_or(seeds, search_candidates)
        num_labels, labels = cv2.connectedComponents(combined)
        seed_labels = set(np.unique(labels[seeds > 0]))

        shadow = np.zeros((h, w), dtype=np.uint8)
        for label_id in seed_labels:
            if label_id == 0:
                continue
            region = (labels == label_id).astype(np.uint8) * 255
            shadow = np.maximum(
                shadow, cv2.bitwise_and(region, search_candidates)
            )

        # ── Step 5: Validate ──
        shadow_area = int(np.sum(shadow > 0))
        min_area = max(
            self.config.min_shadow_area,
            int(person_area * self.MIN_SHADOW_FRACTION),
        )

        if shadow_area < min_area:
            return None, 0.0

        # Confidence based on shadow-to-person ratio and area
        # Reasonable shadows are 0.2-3x person area
        ratio = shadow_area / max(person_area, 1)
        if ratio > 5.0:
            # Suspiciously large — probably grabbing non-shadow dark areas
            return None, 0.0
        elif ratio > 3.0:
            conf = 0.5
        elif ratio > 0.1:
            conf = 0.8
        else:
            conf = 0.6

        return shadow, conf


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1 ML Detector Stubs
# ══════════════════════════════════════════════════════════════════════════════

class SDDNetDetector(BaseShadowDetector):
    """SDDNet shadow detection (ACM MM 2023).

    Style-guided dual-layer disentanglement network. ~32 FPS on RTX 4090.
    Pretrained weights auto-downloaded from Unveiling Deep Shadows benchmark.

    Setup:
        pip install torch torchvision efficientnet-pytorch
        git clone https://github.com/rmcong/SDDNet_ACMMM23 ~/.prep360_shadow_weights/SDDNet_ACMMM23

    Weights are auto-downloaded (273MB) from GitHub Releases on first use.
    The model architecture is loaded dynamically from the cloned repo.
    """

    # Default location for the cloned SDDNet repo
    _DEFAULT_REPO_DIR = Path.home() / ".prep360_shadow_weights" / "SDDNet_ACMMM23"

    @property
    def requires_gpu(self) -> bool:
        return True

    def initialize(self) -> None:
        import sys

        try:
            import torch
        except ImportError:
            raise ImportError(
                "SDDNet requires PyTorch. Install: pip install torch torchvision"
            )

        try:
            import efficientnet_pytorch  # noqa: F401
        except ImportError:
            raise ImportError(
                "SDDNet requires efficientnet-pytorch. "
                "Install: pip install efficientnet-pytorch"
            )

        # Locate the SDDNet repo (for model architecture)
        repo_dir = self._find_repo_dir()

        # Add repo to sys.path so `from networks.sddnet import SDDNet` works
        repo_str = str(repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            from networks.sddnet import SDDNet
        except ImportError as e:
            raise ImportError(
                f"Cannot import SDDNet architecture from {repo_dir}. "
                f"Clone the repo: git clone https://github.com/rmcong/SDDNet_ACMMM23 "
                f"{self._DEFAULT_REPO_DIR}\nOriginal error: {e}"
            )

        # Resolve weights (auto-download if needed)
        weights_path = self._resolve_weights()

        logger.info(f"Loading SDDNet from {weights_path}")

        # Instantiate model with verified constructor args from test.py
        self.model = SDDNet(
            backbone='efficientnet-b3',
            proj_planes=16,
            pred_planes=32,
            use_pretrained=False,  # we have the full checkpoint
            fix_backbone=False,
            has_se=False,
            dropout_2d=0,
            normalize=True,  # model handles ImageNet normalization internally
            mu_init=0.4,
            reweight_mode='manual',
        )

        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        # Checkpoint may use 'model' key or be a raw state_dict
        state_dict = ckpt.get('model', ckpt)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._initialized = True

    def _find_repo_dir(self) -> Path:
        """Locate the SDDNet repo directory."""
        # Check config override first
        if self.config.weights_dir:
            candidate = Path(self.config.weights_dir) / "SDDNet_ACMMM23"
            if (candidate / "networks" / "sddnet.py").exists():
                return candidate

        # Check default location
        if (self._DEFAULT_REPO_DIR / "networks" / "sddnet.py").exists():
            return self._DEFAULT_REPO_DIR

        # Check alongside this file
        local = Path(__file__).parent / "SDDNet_ACMMM23"
        if (local / "networks" / "sddnet.py").exists():
            return local

        raise FileNotFoundError(
            f"SDDNet repository not found. Clone it:\n"
            f"  git clone https://github.com/rmcong/SDDNet_ACMMM23 "
            f"{self._DEFAULT_REPO_DIR}"
        )

    def _resolve_weights(self) -> Path:
        if self.config.model_checkpoint:
            p = Path(self.config.model_checkpoint)
            if p.exists():
                return p
            raise FileNotFoundError(f"SDDNet checkpoint not found: {p}")

        return ShadowWeightManager.ensure_weights(
            model_name="sddnet",
            filename="SDDNet_512.zip",
        )

    def detect(
        self,
        image: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        import torch

        h, w = image.shape[:2]

        # Preprocess: resize to 512x512, convert BGR→RGB, to tensor
        # NOTE: normalize=True in constructor means the model normalizes internally
        img_resized = cv2.resize(image, (512, 512))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_t = (
            torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        )
        img_t = img_t.to(self.device)

        with torch.no_grad():
            output = self.model(img_t)
            # Eval mode returns {'logit': tensor}
            if isinstance(output, dict):
                logit = output['logit']
            elif isinstance(output, (list, tuple)):
                logit = output[0]
            else:
                logit = output
            prob_map = torch.sigmoid(logit).squeeze().cpu().numpy()

        prob_map = cv2.resize(prob_map, (w, h))

        shadow_mask = (
            (prob_map > self.config.confidence_threshold) * 255
        ).astype(np.uint8)

        confidence = (
            float(np.mean(prob_map[shadow_mask > 0]))
            if np.sum(shadow_mask) > 0 else 0.0
        )
        return shadow_mask, confidence

    def cleanup(self):
        if self.model is not None:
            import torch
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._initialized = False


class SILTDetector(BaseShadowDetector):
    """SILT shadow detection (ICCV 2023).

    Shadow-aware Iterative Label Tuning with PVTv2-B3 backbone.
    Best single-model accuracy among easy-to-deploy detectors.

    STATUS: No trained shadow detection checkpoints are publicly available.
    The SILT repo only provides backbone pretrained weights (e.g., pvt_v2_b3.pth),
    NOT fully trained shadow detection models. Training from scratch requires
    the SBU dataset + GPU resources.

    If/when trained weights become available, update _resolve_weights() and
    the architecture loading in initialize().

    Requires: PyTorch (+ timm if backbone weights only)
    """

    @property
    def requires_gpu(self) -> bool:
        return True

    def initialize(self) -> None:
        raise FileNotFoundError(
            "SILT trained shadow detection weights are not publicly available.\n"
            "The SILT repo (github.com/hanyangclarence/SILT) only provides\n"
            "backbone pretrained weights, not trained detection checkpoints.\n"
            "Options:\n"
            "  1. Train from scratch using their training script + SBU dataset\n"
            "  2. Use SDDNet instead (trained weights available, similar accuracy)\n"
            "  3. Check back — authors may release weights in the future"
        )

    def _resolve_weights(self) -> Path:
        if self.config.model_checkpoint:
            p = Path(self.config.model_checkpoint)
            if p.exists():
                return p
            raise FileNotFoundError(f"SILT checkpoint not found: {p}")
        raise FileNotFoundError("No SILT trained weights available for download")

    def detect(
        self,
        image: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        raise RuntimeError("SILT not initialized — no trained weights available")

    def cleanup(self):
        if self.model is not None:
            import torch
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._initialized = False


class CareagaIntrinsicDetector(BaseShadowDetector):
    """Shadow detection via intrinsic image decomposition (Careaga & Aksoy).

    Decomposes the image into albedo + shading layers. The shading layer
    encodes illumination — dark shading = shadow. Threshold the diffuse
    shading to produce a shadow mask.

    Physics-based approach: no shadow training data needed, zero domain gap.
    Excellent on uniform surfaces (sand, concrete) where shadows produce
    strong shading contrast.

    Setup:
        pip install https://github.com/compphoto/Intrinsic/archive/main.zip

    Weights auto-download (~200MB total) from GitHub Releases on first use.
    """

    @property
    def requires_gpu(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "careaga_intrinsic"

    def initialize(self) -> None:
        try:
            from intrinsic.pipeline import load_models
        except ImportError:
            raise ImportError(
                "Careaga intrinsic decomposition not installed.\n"
                "Install: pip install "
                "https://github.com/compphoto/Intrinsic/archive/main.zip"
            )

        logger.info("Loading Careaga intrinsic decomposition models...")
        self.model = load_models('v2')
        self._initialized = True
        logger.info("Careaga intrinsic models loaded")

    def detect(
        self,
        image: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        from intrinsic.pipeline import run_pipeline

        h, w = image.shape[:2]

        # Convert BGR→RGB, uint8→float32 [0,1]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Run intrinsic decomposition
        result = run_pipeline(
            self.model,
            img_rgb,
            resize_conf=0.0,
            base_size=384,
            linear=False,
            device=self.device,
        )

        # Use diffuse shading — lower values = darker = more likely shadow
        shading = result.get('dif_shd')
        if shading is None:
            # Fall back to grayscale shading if full pipeline didn't run
            shading = result.get('gry_shd')
        if shading is None:
            return np.zeros((h, w), dtype=np.uint8), 0.0

        # Convert multi-channel shading to grayscale if needed
        if shading.ndim == 3:
            shading_gray = np.mean(shading, axis=-1)
        else:
            shading_gray = shading

        # Resize to original resolution if needed
        if shading_gray.shape[:2] != (h, w):
            shading_gray = cv2.resize(shading_gray, (w, h))

        # Threshold: low shading = shadow
        # Use darkness_ratio_threshold as the shading cutoff (default 0.7)
        threshold = self.config.darkness_ratio_threshold
        shadow_mask = ((shading_gray < threshold) * 255).astype(np.uint8)

        # Confidence based on how distinct the shading contrast is
        if np.sum(shadow_mask > 0) > 0:
            shadow_mean = float(np.mean(shading_gray[shadow_mask > 0]))
            lit_mean = float(np.mean(shading_gray[shadow_mask == 0])) if np.sum(shadow_mask == 0) > 0 else 1.0
            # Higher contrast between shadow and lit → higher confidence
            contrast = max(0.0, lit_mean - shadow_mean)
            confidence = min(1.0, contrast / 0.5)  # normalize: 0.5 contrast → 1.0
        else:
            confidence = 0.0

        return shadow_mask, confidence

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        self._initialized = False


# ══════════════════════════════════════════════════════════════════════════════
# Weight Manager
# ══════════════════════════════════════════════════════════════════════════════

class ShadowWeightManager:
    """Manages downloading and caching shadow detection model weights.

    Default cache: ~/.prep360_shadow_weights/
    Override: PREP360_SHADOW_WEIGHTS_DIR environment variable or config.weights_dir
    """

    DEFAULT_DIR = Path.home() / ".prep360_shadow_weights"

    # Registry of known models with verified download URLs
    REGISTRY: Dict[str, Dict[str, Any]] = {
        "sddnet": {
            "filename": "SDDNet_512.zip",
            "url": "https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Weights_SD/SDDNet_512.zip",
            "sha256": None,
            "size_mb": 273,
            "is_zip": True,
            "source": "Unveiling Deep Shadows benchmark (GitHub Releases, direct HTTP)",
        },
        "sddnet_256": {
            "filename": "SDDNet_256.zip",
            "url": "https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Weights_SD/SDDNet_256.zip",
            "sha256": None,
            "size_mb": 272,
            "is_zip": True,
            "source": "Unveiling Deep Shadows benchmark (GitHub Releases, direct HTTP)",
        },
        "fdrnet": {
            "filename": "FDRNet_512.zip",
            "url": "https://github.com/xw-hu/Unveiling-Deep-Shadows/releases/download/Weights_SD/FDRNet_512.zip",
            "sha256": None,
            "size_mb": 189,
            "is_zip": True,
            "source": "Unveiling Deep Shadows benchmark (GitHub Releases, direct HTTP)",
        },
        "silt": {
            "filename": "silt_pvtv2_b3.pth",
            "url": None,  # No trained weights publicly available
            "sha256": None,
            "size_mb": 200,
            "source": "NOT AVAILABLE — only backbone weights at github.com/hanyangclarence/SILT",
        },
    }

    @classmethod
    def weights_dir(cls, config: Optional[ShadowConfig] = None) -> Path:
        """Get (and create) the weights cache directory."""
        import os
        env_dir = os.environ.get("PREP360_SHADOW_WEIGHTS_DIR", "")
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
        model_name: str,
        filename: str,
        url: Optional[str] = None,
        sha256: Optional[str] = None,
    ) -> Path:
        """Ensure weights file exists locally, downloading if needed."""
        dest = cls.weights_dir() / filename
        is_zip = cls.REGISTRY.get(model_name, {}).get("is_zip", False)

        # If this is a zip model, check for already-extracted directory
        if is_zip and dest.suffix.lower() == ".zip":
            extract_dir = dest.parent / dest.stem
            if extract_dir.is_dir():
                pth_files = list(extract_dir.rglob("*.pth"))
                if pth_files:
                    return pth_files[0]

        if dest.exists():
            if sha256 and cls._file_hash(dest) != sha256:
                logger.warning(f"Hash mismatch for {filename}, re-downloading")
            else:
                return dest

        # Look up URL from registry
        if url is None:
            info = cls.REGISTRY.get(model_name, {})
            url = info.get("url")

        if url is None:
            source = cls.REGISTRY.get(model_name, {}).get("source", "unknown")
            raise FileNotFoundError(
                f"Weights for '{model_name}' not found at {dest}.\n"
                f"No auto-download URL configured yet.\n"
                f"Download manually from: {source}\n"
                f"Place the file at: {dest}"
            )

        size_mb = cls.REGISTRY.get(model_name, {}).get("size_mb", "?")
        is_zip = cls.REGISTRY.get(model_name, {}).get("is_zip", False)
        logger.info(f"Downloading {model_name} weights (~{size_mb} MB)...")
        try:
            urllib.request.urlretrieve(url, str(dest))
            logger.info(f"Saved to {dest}")
        except Exception as e:
            if dest.exists():
                dest.unlink()
            raise FileNotFoundError(
                f"Failed to download {model_name} weights: {e}"
            )

        # Extract zip archives and return path to the .pth inside
        if is_zip and dest.suffix.lower() == ".zip":
            import zipfile
            extract_dir = dest.parent / dest.stem
            logger.info(f"Extracting {dest.name} to {extract_dir}...")
            with zipfile.ZipFile(str(dest), "r") as zf:
                zf.extractall(str(extract_dir))
            dest.unlink()  # Remove the zip after extraction
            # Find the .pth file inside
            pth_files = list(extract_dir.rglob("*.pth"))
            if pth_files:
                logger.info(f"Extracted weights: {pth_files[0].name}")
                return pth_files[0]
            logger.warning(f"No .pth file found in {extract_dir}")
            return extract_dir

        return dest

    @classmethod
    def list_available(cls) -> Dict[str, bool]:
        """List registered models and whether weights exist locally."""
        wdir = cls.weights_dir()
        return {
            name: (wdir / info["filename"]).exists()
            for name, info in cls.REGISTRY.items()
        }

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

_DETECTOR_MAP: Dict[ShadowDetectorType, type] = {
    ShadowDetectorType.BRIGHTNESS_HEURISTIC: BrightnessHeuristicDetector,
    ShadowDetectorType.CHROMATICITY_C1C2C3: ChromaticityC1C2C3Detector,
    ShadowDetectorType.HYBRID_INTENSITY: HybridIntensityChromaticityDetector,
    ShadowDetectorType.TARGETED_PERSON: TargetedPersonShadowDetector,
    ShadowDetectorType.SDDNET: SDDNetDetector,
    ShadowDetectorType.SILT: SILTDetector,
    ShadowDetectorType.CAREAGA_INTRINSIC: CareagaIntrinsicDetector,
}


def create_shadow_detector(
    detector_type: ShadowDetectorType,
    config: ShadowConfig,
) -> BaseShadowDetector:
    """Create a shadow detector instance.

    Mirrors MaskingPipeline._create_segmenter() pattern.
    """
    cls = _DETECTOR_MAP.get(detector_type)
    if cls is None:
        available = [t.value for t in _DETECTOR_MAP]
        raise NotImplementedError(
            f"Shadow detector '{detector_type.value}' not yet implemented. "
            f"Available: {available}"
        )
    return cls(config)


# ══════════════════════════════════════════════════════════════════════════════
# Shadow Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class ShadowPipeline:
    """Orchestrates the multi-stage shadow detection pipeline.

    Stages:
        1. Primary Detection  — ML or classical detector → raw shadow mask
        2. Verification        — Cross-check with a second method (optional)
        3. Spatial Filtering   — Keep only shadows near detected objects (optional)
        4. Cleanup             — Morphological cleanup + fragment removal

    Returns a MaskResult compatible with reconstruction_pipeline.py's existing pipeline.

    Usage:
        pipeline = ShadowPipeline(config)
        pipeline.initialize()
        result = pipeline.run(image, object_mask)
    """

    def __init__(self, config: ShadowConfig):
        self.config = config
        self.primary: Optional[BaseShadowDetector] = None
        self.verifier: Optional[BaseShadowDetector] = None

    def initialize(self) -> None:
        """Initialize detector(s). Called once before first run()."""
        self.primary = create_shadow_detector(
            self.config.primary_detector, self.config
        )
        self.primary.initialize()
        logger.info(f"Shadow primary detector: {self.primary.name}")

        if self.config.verification_detector is not None:
            self.verifier = create_shadow_detector(
                self.config.verification_detector, self.config
            )
            self.verifier.initialize()
            logger.info(f"Shadow verifier: {self.verifier.name}")

    def run(
        self,
        image: np.ndarray,
        object_mask: Optional[np.ndarray] = None,
        individual_masks: Optional[List[np.ndarray]] = None,
    ) -> Optional[Any]:
        """Run the full shadow detection pipeline.

        Args:
            image: BGR input image (H, W, 3) uint8
            object_mask: Binary mask (H, W) of detected objects
            individual_masks: Optional list of per-person binary masks (H, W)
                              uint8. Used by TargetedPersonShadowDetector for
                              per-person shadow search. If None, the detector
                              falls back to splitting the combined object_mask
                              via connected components.

        Returns:
            MaskResult with shadow mask, or None if no shadows detected.
            Import of MaskResult is deferred to avoid circular imports.
        """
        if self.primary is None:
            raise RuntimeError(
                "ShadowPipeline not initialized. Call initialize() first."
            )

        h, w = image.shape[:2]

        # Stage 1: Primary detection
        # If the detector supports per-person search, use it
        if (
            individual_masks
            and isinstance(self.primary, TargetedPersonShadowDetector)
        ):
            shadow_mask, confidence = self.primary.detect_per_person(
                image, individual_masks, object_mask
            )
        else:
            shadow_mask, confidence = self.primary.detect(image, object_mask)
        if shadow_mask is None or np.sum(shadow_mask) == 0:
            return None

        # Stage 2: Verification
        if self.verifier is not None:
            verifier_mask, verifier_conf = self.verifier.detect(image, object_mask)
            if verifier_mask is not None and np.sum(verifier_mask) > 0:
                shadow_mask = self._apply_verification(shadow_mask, verifier_mask)
                confidence = confidence * 0.6 + verifier_conf * 0.4
            if np.sum(shadow_mask) == 0:
                return None

        # Stage 3: Spatial filtering
        if (
            object_mask is not None
            and self.config.spatial_mode != ShadowSpatialMode.ALL
        ):
            shadow_mask = self._apply_spatial_filter(shadow_mask, object_mask)
            if np.sum(shadow_mask) == 0:
                return None

        # Stage 4: Cleanup
        shadow_mask = self._cleanup_mask(shadow_mask)
        if np.sum(shadow_mask) < self.config.min_shadow_area:
            return None

        quality = self._assess_quality(shadow_mask, confidence, object_mask)

        return ShadowMaskResult(
            mask=(shadow_mask > 0).astype(np.uint8),
            confidence=confidence,
            quality=quality,
            metadata={
                'type': 'shadow',
                'detector': self.config.primary_detector.value,
                'verifier': (
                    self.config.verification_detector.value
                    if self.config.verification_detector else None
                ),
                'spatial_mode': self.config.spatial_mode.value,
                'shadow_pixels': int(np.sum(shadow_mask > 0)),
            }
        )

    # ── Stage implementations ──

    def _apply_verification(
        self, primary_mask: np.ndarray, verifier_mask: np.ndarray
    ) -> np.ndarray:
        """Keep primary shadow regions that have sufficient overlap with verifier."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            primary_mask, connectivity=8
        )
        verified = np.zeros_like(primary_mask)
        thresh = self.config.verification_agreement

        for i in range(1, num_labels):
            region = (labels == i)
            region_area = stats[i, cv2.CC_STAT_AREA]
            intersection = np.sum(region & (verifier_mask > 0))
            iou = intersection / max(region_area, 1)
            if iou >= thresh:
                verified[region] = 255

        return verified

    def _apply_spatial_filter(
        self, shadow_mask: np.ndarray, object_mask: np.ndarray
    ) -> np.ndarray:
        """Keep only shadows near detected objects."""
        obj_binary = (object_mask > 0).astype(np.uint8) * 255

        if self.config.spatial_mode == ShadowSpatialMode.NEAR_OBJECTS:
            r = self.config.dilation_radius
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)
            )
            proximity_zone = cv2.dilate(obj_binary, kernel, iterations=1)
            return cv2.bitwise_and(shadow_mask, proximity_zone)

        elif self.config.spatial_mode == ShadowSpatialMode.CONNECTED:
            # Slightly expand object mask to create seed overlap
            small_kernel = np.ones((5, 5), np.uint8)
            expanded = cv2.dilate(obj_binary, small_kernel, iterations=2)

            # Find where shadow touches expanded object
            seeds = cv2.bitwise_and(shadow_mask, expanded)
            if np.sum(seeds) == 0:
                return np.zeros_like(shadow_mask)

            # Keep only shadow components that contain seed pixels
            combined = cv2.bitwise_or(seeds, shadow_mask)
            num_labels, labels = cv2.connectedComponents(combined)
            seed_labels = set(np.unique(labels[seeds > 0]))

            connected = np.zeros_like(shadow_mask)
            for label_id in seed_labels:
                if label_id == 0:
                    continue
                region = (labels == label_id).astype(np.uint8) * 255
                connected = np.maximum(
                    connected, cv2.bitwise_and(region, shadow_mask)
                )
            return connected

        return shadow_mask

    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        """Morphological cleanup and fragment removal."""
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

        kernel_medium = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)

        # Remove tiny components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.config.min_shadow_area:
                mask[labels == i] = 0

        return mask

    def _assess_quality(
        self,
        shadow_mask: np.ndarray,
        confidence: float,
        object_mask: Optional[np.ndarray],
    ) -> ShadowMaskQuality:
        """Assess shadow mask quality."""
        shadow_area = int(np.sum(shadow_mask > 0))

        # Sanity: shadow should not dwarf the object
        if object_mask is not None:
            obj_area = int(np.sum(object_mask > 0))
            if obj_area > 0 and shadow_area > obj_area * 5:
                return ShadowMaskQuality.REVIEW

        if confidence >= 0.8:
            return ShadowMaskQuality.EXCELLENT if shadow_area > 200 else ShadowMaskQuality.GOOD
        elif confidence >= 0.6:
            return ShadowMaskQuality.GOOD
        elif confidence >= 0.4:
            return ShadowMaskQuality.REVIEW
        else:
            return ShadowMaskQuality.POOR

    def cleanup(self) -> None:
        """Release all detector resources."""
        if self.primary:
            self.primary.cleanup()
        if self.verifier:
            self.verifier.cleanup()
