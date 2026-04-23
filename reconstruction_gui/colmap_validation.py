"""
COLMAP Geometric Validation for Mask Consistency
=================================================

Post-mask validation using COLMAP sparse reconstruction data.
Projects 3D points into camera views and checks whether each point
falls inside or outside the mask consistently across all views.

Inconsistencies indicate masks that are too tight (missed object) or
too loose (masked background) in specific frames.

Pipeline: COLMAP sparse → parse → project 3D→2D → check masks → report

Usage:
    from colmap_validation import GeometricValidator, ValidationConfig
    cfg = ValidationConfig(colmap_dir="path/to/sparse/0")
    validator = GeometricValidator(cfg)
    validator.load_reconstruction()
    report = validator.validate_masks(masks_dir="path/to/masks",
                                      images_dir="path/to/images")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# COLMAP Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class COLMAPCamera:
    """A COLMAP camera model."""
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]

    def get_intrinsics(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix K."""
        if self.model == 'SIMPLE_PINHOLE':
            f, cx, cy = self.params[:3]
            return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        elif self.model == 'PINHOLE':
            fx, fy, cx, cy = self.params[:4]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        elif self.model in ('SIMPLE_RADIAL', 'RADIAL'):
            f, cx, cy = self.params[:3]
            return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        elif self.model == 'OPENCV':
            fx, fy, cx, cy = self.params[:4]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        else:
            # Fallback: assume first param is focal length
            f = self.params[0]
            cx, cy = self.width / 2, self.height / 2
            return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    def get_distortion(self) -> Optional[np.ndarray]:
        """Return distortion coefficients for OpenCV undistort, or None."""
        if self.model == 'SIMPLE_RADIAL':
            k = self.params[3] if len(self.params) > 3 else 0.0
            return np.array([k, 0, 0, 0], dtype=np.float64)
        elif self.model == 'RADIAL':
            k1 = self.params[3] if len(self.params) > 3 else 0.0
            k2 = self.params[4] if len(self.params) > 4 else 0.0
            return np.array([k1, k2, 0, 0], dtype=np.float64)
        elif self.model == 'OPENCV':
            k1, k2, p1, p2 = self.params[4:8] if len(self.params) >= 8 else [0]*4
            return np.array([k1, k2, p1, p2], dtype=np.float64)
        return None


@dataclass
class COLMAPImage:
    """A COLMAP registered image."""
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    points2d: List[Tuple[float, float, int]] = field(default_factory=list)

    def get_rotation(self) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        q = np.array([self.qw, self.qx, self.qy, self.qz])
        q = q / np.linalg.norm(q)
        w, x, y, z = q

        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        return R

    def get_translation(self) -> np.ndarray:
        """Return translation vector t."""
        return np.array([self.tx, self.ty, self.tz], dtype=np.float64)

    def get_camera_center(self) -> np.ndarray:
        """Compute camera center in world coordinates: C = -R^T @ t."""
        R = self.get_rotation()
        t = self.get_translation()
        return -R.T @ t


@dataclass
class COLMAPPoint3D:
    """A COLMAP 3D point."""
    point3d_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: List[Tuple[int, int]]  # (image_id, point2d_idx) pairs


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationConfig:
    """Configuration for COLMAP geometric validation."""

    # COLMAP reconstruction path (containing cameras.txt, images.txt, points3D.txt)
    colmap_dir: str = ""

    # Validation thresholds
    agreement_threshold: float = 0.7  # Min fraction of views that must agree
    min_track_length: int = 3         # Ignore points seen in fewer views
    max_reprojection_error: float = 4.0  # Ignore points with high COLMAP error

    # Mask loading
    mask_suffix: str = ".png"  # Expected mask file extension

    # Point sampling (for large reconstructions)
    max_points: int = 50000   # Subsample 3D points for speed (0 = all)

    # Per-frame flagging
    inconsistency_threshold: float = 0.15  # Flag frame if >15% of its points disagree

    def to_dict(self) -> Dict[str, Any]:
        return {
            'colmap_dir': self.colmap_dir,
            'agreement_threshold': self.agreement_threshold,
            'min_track_length': self.min_track_length,
            'max_reprojection_error': self.max_reprojection_error,
            'mask_suffix': self.mask_suffix,
            'max_points': self.max_points,
            'inconsistency_threshold': self.inconsistency_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationConfig':
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ══════════════════════════════════════════════════════════════════════════════
# Frame Validation Result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameValidationResult:
    """Validation result for a single frame."""
    image_name: str
    total_points: int = 0         # 3D points projecting into this frame
    consistent_points: int = 0    # Points where mask agrees with majority
    inconsistent_points: int = 0  # Points where mask disagrees with majority
    missing_mask: bool = False    # No mask file found

    # Points where mask says background but 3D consensus says foreground (mask too tight)
    false_negative_points: int = 0
    # Points where mask says foreground but 3D consensus says background (mask too loose)
    false_positive_points: int = 0

    @property
    def consistency_ratio(self) -> float:
        """Fraction of points that are consistent (0-1)."""
        if self.total_points == 0:
            return 1.0
        return self.consistent_points / self.total_points

    @property
    def is_flagged(self) -> bool:
        """Whether this frame should be flagged for review."""
        return False  # Set externally based on threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_name': self.image_name,
            'total_points': self.total_points,
            'consistent_points': self.consistent_points,
            'inconsistent_points': self.inconsistent_points,
            'consistency_ratio': round(self.consistency_ratio, 4),
            'false_negative_points': self.false_negative_points,
            'false_positive_points': self.false_positive_points,
            'missing_mask': self.missing_mask,
        }


@dataclass
class ValidationReport:
    """Full validation report across all frames."""
    frames: Dict[str, FrameValidationResult] = field(default_factory=dict)
    total_points_checked: int = 0
    flagged_frames: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        n_frames = len(self.frames)
        n_flagged = len(self.flagged_frames)
        ratios = [f.consistency_ratio for f in self.frames.values()
                  if f.total_points > 0]
        return {
            'total_frames': n_frames,
            'flagged_frames': n_flagged,
            'flagged_pct': round(100 * n_flagged / n_frames, 1) if n_frames else 0,
            'total_points_checked': self.total_points_checked,
            'avg_consistency': round(np.mean(ratios), 4) if ratios else 1.0,
            'min_consistency': round(np.min(ratios), 4) if ratios else 1.0,
            'flagged_names': self.flagged_frames[:20],  # First 20 for display
        }


# ══════════════════════════════════════════════════════════════════════════════
# COLMAP Parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_cameras_txt(path: Path) -> Dict[int, COLMAPCamera]:
    """Parse COLMAP cameras.txt file."""
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = COLMAPCamera(
                camera_id=camera_id, model=model,
                width=width, height=height, params=params
            )
    return cameras


def parse_images_txt(path: Path) -> Dict[int, COLMAPImage]:
    """Parse COLMAP images.txt file.

    Two lines per image:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      POINTS2D[] as (X Y POINT3D_ID)
    """
    images = {}
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]

        img = COLMAPImage(
            image_id=image_id, qw=qw, qx=qx, qy=qy, qz=qz,
            tx=tx, ty=ty, tz=tz, camera_id=camera_id, name=name
        )

        # Parse points2D line (next line)
        points2d = []
        if i + 1 < len(lines):
            p2d_parts = lines[i + 1].split()
            # Every 3 values: X Y POINT3D_ID
            for j in range(0, len(p2d_parts) - 2, 3):
                x = float(p2d_parts[j])
                y = float(p2d_parts[j + 1])
                pid = int(p2d_parts[j + 2])
                points2d.append((x, y, pid))
            i += 2
        else:
            i += 1

        img.points2d = points2d
        images[image_id] = img

    return images


def parse_points3d_txt(path: Path) -> Dict[int, COLMAPPoint3D]:
    """Parse COLMAP points3D.txt file.

    Each line: POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID POINT2D_IDX)
    """
    points = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue

            point_id = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])])
            error = float(parts[7])

            # Track: pairs of (IMAGE_ID, POINT2D_IDX)
            track = []
            for j in range(8, len(parts) - 1, 2):
                img_id = int(parts[j])
                pt_idx = int(parts[j + 1])
                track.append((img_id, pt_idx))

            points[point_id] = COLMAPPoint3D(
                point3d_id=point_id, xyz=xyz, rgb=rgb,
                error=error, track=track
            )

    return points


# ══════════════════════════════════════════════════════════════════════════════
# Geometric Validator
# ══════════════════════════════════════════════════════════════════════════════

class GeometricValidator:
    """Validates mask consistency using COLMAP sparse reconstruction.

    Projects 3D points into each camera view and checks whether the
    projected 2D location falls inside or outside the mask. Points that
    are masked in most views but unmasked in a few (or vice versa) indicate
    mask inconsistencies in those frames.

    Usage:
        validator = GeometricValidator(config)
        validator.load_reconstruction()
        report = validator.validate_masks(masks_dir, images_dir)
        for name in report.flagged_frames:
            print(f"Review: {name}")
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.cameras: Dict[int, COLMAPCamera] = {}
        self.images: Dict[int, COLMAPImage] = {}
        self.points3d: Dict[int, COLMAPPoint3D] = {}
        self._name_to_id: Dict[str, int] = {}  # image name → image_id

    def load_reconstruction(self, colmap_dir: Optional[str] = None):
        """Load COLMAP sparse reconstruction from text files.

        Args:
            colmap_dir: Path to COLMAP sparse dir. If None, uses config.colmap_dir.
        """
        cdir = Path(colmap_dir or self.config.colmap_dir)
        if not cdir.is_dir():
            raise FileNotFoundError(f"COLMAP directory not found: {cdir}")

        cameras_path = cdir / "cameras.txt"
        images_path = cdir / "images.txt"
        points_path = cdir / "points3D.txt"

        for p in [cameras_path, images_path, points_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing COLMAP file: {p}")

        self.cameras = parse_cameras_txt(cameras_path)
        self.images = parse_images_txt(images_path)
        self.points3d = parse_points3d_txt(points_path)

        # Build name lookup
        self._name_to_id = {img.name: img.image_id for img in self.images.values()}

        logger.info(
            f"Loaded COLMAP reconstruction: {len(self.cameras)} cameras, "
            f"{len(self.images)} images, {len(self.points3d)} points"
        )

    def project_point(
        self, point_xyz: np.ndarray, image: COLMAPImage, camera: COLMAPCamera
    ) -> Optional[Tuple[float, float]]:
        """Project a 3D world point into a camera view.

        Returns:
            (x, y) pixel coordinates, or None if behind camera.
        """
        R = image.get_rotation()
        t = image.get_translation()

        # Transform to camera coordinates: P_cam = R @ P_world + t
        p_cam = R @ point_xyz + t

        # Check if point is in front of camera
        if p_cam[2] <= 0:
            return None

        # Apply intrinsics
        K = camera.get_intrinsics()
        p_proj = K @ p_cam
        x = p_proj[0] / p_proj[2]
        y = p_proj[1] / p_proj[2]

        # Check bounds
        if x < 0 or x >= camera.width or y < 0 or y >= camera.height:
            return None

        return (x, y)

    def validate_masks(
        self,
        masks_dir: str,
        images_dir: Optional[str] = None,
        progress_callback=None
    ) -> ValidationReport:
        """Validate masks against 3D reconstruction.

        For each 3D point visible in multiple views, checks whether
        the projected location is consistently inside or outside the
        mask across all views. Flags frames where the mask disagrees
        with the multi-view consensus.

        Args:
            masks_dir: Directory containing mask images.
            images_dir: Directory containing source images (for name matching).
            progress_callback: Optional callable(current, total) for progress.

        Returns:
            ValidationReport with per-frame results and flagged frames.
        """
        if not self.images:
            raise RuntimeError("No reconstruction loaded. Call load_reconstruction() first.")

        masks_path = Path(masks_dir)
        report = ValidationReport()

        # Initialize per-frame results
        for img in self.images.values():
            report.frames[img.name] = FrameValidationResult(image_name=img.name)

        # Load masks into memory (cached by image name)
        mask_cache = self._load_masks(masks_path)

        # Subsample points if needed
        point_ids = list(self.points3d.keys())
        if self.config.max_points > 0 and len(point_ids) > self.config.max_points:
            rng = np.random.RandomState(42)
            point_ids = rng.choice(point_ids, self.config.max_points, replace=False).tolist()

        total = len(point_ids)
        logger.info(f"Validating {total} 3D points against {len(mask_cache)} masks")

        for idx, pid in enumerate(point_ids):
            pt = self.points3d[pid]

            # Skip points with high reprojection error
            if pt.error > self.config.max_reprojection_error:
                continue

            # Skip points seen in too few views
            if len(pt.track) < self.config.min_track_length:
                continue

            # Project into each observing view and check mask
            mask_votes = []  # List of (image_name, is_masked)

            for img_id, pt2d_idx in pt.track:
                if img_id not in self.images:
                    continue
                img = self.images[img_id]
                if img.camera_id not in self.cameras:
                    continue
                cam = self.cameras[img.camera_id]

                # Project 3D point to 2D
                proj = self.project_point(pt.xyz, img, cam)
                if proj is None:
                    continue

                px, py = int(round(proj[0])), int(round(proj[1]))

                # Check mask
                mask = mask_cache.get(img.name)
                if mask is None:
                    report.frames[img.name].missing_mask = True
                    continue

                is_masked = mask[py, px] > 127 if (
                    0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]
                ) else False

                mask_votes.append((img.name, is_masked))

            if len(mask_votes) < self.config.min_track_length:
                continue

            # Determine consensus: is this point masked in the majority of views?
            n_masked = sum(1 for _, m in mask_votes if m)
            n_total = len(mask_votes)
            consensus_masked = n_masked / n_total >= self.config.agreement_threshold

            report.total_points_checked += 1

            # Score each frame's agreement with consensus
            for img_name, is_masked in mask_votes:
                frame = report.frames[img_name]
                frame.total_points += 1

                if is_masked == consensus_masked:
                    frame.consistent_points += 1
                else:
                    frame.inconsistent_points += 1
                    if consensus_masked and not is_masked:
                        # Majority says masked, this frame doesn't → mask too tight
                        frame.false_negative_points += 1
                    elif not consensus_masked and is_masked:
                        # Majority says not masked, this frame masks it → mask too loose
                        frame.false_positive_points += 1

            if progress_callback and idx % 1000 == 0:
                progress_callback(idx, total)

        # Flag inconsistent frames
        for name, frame in report.frames.items():
            if frame.total_points == 0:
                continue
            inconsistency = frame.inconsistent_points / frame.total_points
            if inconsistency > self.config.inconsistency_threshold:
                report.flagged_frames.append(name)

        report.flagged_frames.sort()

        summary = report.summary()
        logger.info(
            f"Validation complete: {summary['flagged_frames']}/{summary['total_frames']} "
            f"frames flagged, avg consistency={summary['avg_consistency']:.3f}"
        )

        return report

    def _load_masks(self, masks_dir: Path) -> Dict[str, np.ndarray]:
        """Load mask images, matching by stem name.

        Handles both prefixed (mask_XXXX.png) and same-stem (XXXX.png) naming.
        """
        masks = {}
        suffix = self.config.mask_suffix

        for img in self.images.values():
            stem = Path(img.name).stem

            # Strategy 1: same-stem in masks dir
            candidates = [
                masks_dir / f"{stem}{suffix}",
                masks_dir / f"{stem}.png",
                masks_dir / f"{stem}.jpg",
                # Strategy 2: mask_* prefix
                masks_dir / f"mask_{stem}{suffix}",
                masks_dir / f"mask_{stem}.png",
                # Strategy 3: *_mask suffix
                masks_dir / f"{stem}_mask{suffix}",
                masks_dir / f"{stem}_mask.png",
            ]

            for candidate in candidates:
                if candidate.exists():
                    mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        masks[img.name] = mask
                    break

        logger.info(f"Loaded {len(masks)}/{len(self.images)} masks from {masks_dir}")
        return masks

    def get_projection_overlay(
        self,
        image_name: str,
        mask: np.ndarray,
        max_points: int = 2000
    ) -> np.ndarray:
        """Generate a colored overlay showing 3D point projections on a frame.

        Green dots: point projects into mask area (consistent if consensus=masked)
        Red dots: point projects outside mask (inconsistent if consensus=masked)
        Orange dots: point has no 3D support but mask covers that area

        Args:
            image_name: COLMAP image name.
            mask: Binary mask (H, W), uint8.
            max_points: Max overlay points for performance.

        Returns:
            BGR overlay image (H, W, 3), uint8.
        """
        if image_name not in self._name_to_id:
            return np.zeros((*mask.shape[:2], 3), dtype=np.uint8)

        img_id = self._name_to_id[image_name]
        img = self.images[img_id]
        cam = self.cameras[img.camera_id]

        h, w = mask.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Collect points visible in this frame
        projected = []
        for pt in self.points3d.values():
            for tid, _ in pt.track:
                if tid == img_id:
                    proj = self.project_point(pt.xyz, img, cam)
                    if proj is not None:
                        projected.append(proj)
                    break

            if len(projected) >= max_points:
                break

        # Draw points
        for px_f, py_f in projected:
            px, py = int(round(px_f)), int(round(py_f))
            if 0 <= py < h and 0 <= px < w:
                is_masked = mask[py, px] > 127
                # Green = inside mask, Red = outside mask
                color = (0, 200, 0) if is_masked else (0, 0, 200)
                cv2.circle(overlay, (px, py), 2, color, -1)

        return overlay

    def cleanup(self):
        """Release loaded data."""
        self.cameras.clear()
        self.images.clear()
        self.points3d.clear()
        self._name_to_id.clear()
        logger.info("GeometricValidator cleaned up")
