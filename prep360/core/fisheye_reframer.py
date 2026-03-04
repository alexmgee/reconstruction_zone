"""
Fisheye → Perspective Reframing Engine

Extract pinhole perspective crops from calibrated fisheye images.
Parallel path to reframer.py (ERP → perspective). Both produce
identical output format: JPEG perspective crops + PINHOLE intrinsics.

    Input is ERP image    → reframer.py    (py360convert-based)
    Input is fisheye pair → fisheye_reframer.py (cv2.fisheye-based)

Usage:
    from prep360.core.fisheye_calibration import DualFisheyeCalibration
    from prep360.core.fisheye_reframer import FisheyeReframer, FISHEYE_PRESETS

    calib = DualFisheyeCalibration.load("osmo360.json")
    reframer = FisheyeReframer(calib)
    config = FISHEYE_PRESETS["osv-wide-f110-dual-12"]

    # Single frame pair
    results = reframer.extract_all_views(front_img, back_img, config)

    # Batch from directories
    reframer.batch_extract(frame_pairs, config, output_dir)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from .fisheye_calibration import FisheyeCalibration, DualFisheyeCalibration


# --- Default calibration for DJI Osmo 360 ---
# Approximate intrinsics when no ChArUco calibration is available.
# Based on 3840x3840 sensor, ~190° FOV equidistant fisheye.
# These are reasonable starting values — a proper calibration will be better.

def default_osmo360_calibration() -> DualFisheyeCalibration:
    """Approximate calibration for DJI Osmo 360 (3840x3840 per lens).

    Uses equidistant fisheye model with estimated focal length.
    For production quality, run ChArUco calibration.
    """
    # For equidistant: r = f * theta
    # At image edge (r=1920px), theta ~= 95° (half of ~190° FOV)
    # f = r / theta = 1920 / radians(95) ≈ 1158
    f_est = 1920.0 / np.radians(95.0)
    cx, cy = 1920.0, 1920.0

    K = np.array([
        [f_est, 0, cx],
        [0, f_est, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    # Small distortion — real lens will deviate, but this is serviceable
    D = np.zeros((4, 1), dtype=np.float64)

    single = FisheyeCalibration(
        camera_matrix=K,
        dist_coeffs=D,
        image_size=(3840, 3840),
        rms_error=-1.0,  # indicates approximate
        num_images_used=0,
        fov_degrees=190.0,
    )

    return DualFisheyeCalibration(
        front=single,
        back=FisheyeCalibration(
            camera_matrix=K.copy(),
            dist_coeffs=D.copy(),
            image_size=(3840, 3840),
            rms_error=-1.0,
            num_images_used=0,
            fov_degrees=190.0,
        ),
        front_rotation_deg=0.0,
        back_rotation_deg=180.0,
        camera_model="DJI Osmo 360 (approximate)",
    )


# --- View definitions ---

@dataclass
class FisheyeView:
    """A single perspective view extracted from a fisheye image."""
    name: str               # e.g. "front_h+0_000" or "back_h+0_180"
    yaw_deg: float          # yaw relative to fisheye optical axis
    pitch_deg: float        # pitch relative to fisheye optical axis
    fov_deg: float          # horizontal FOV of the perspective crop
    source_lens: str        # "front" or "back"


@dataclass
class FisheyeViewConfig:
    """Configuration for all perspective views from a dual-fisheye pair."""
    views: List[FisheyeView] = field(default_factory=list)
    crop_size: int = 1600       # output resolution (square)
    quality: int = 95           # JPEG quality

    def total_views(self) -> int:
        return len(self.views)

    def views_for_lens(self, lens: str) -> List[FisheyeView]:
        return [v for v in self.views if v.source_lens == lens]

    @classmethod
    def from_preset(cls, preset_name: str) -> "FisheyeViewConfig":
        if preset_name not in FISHEYE_PRESETS:
            available = ", ".join(FISHEYE_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        return FISHEYE_PRESETS[preset_name]

    def summary(self) -> str:
        front = len(self.views_for_lens("front"))
        back = len(self.views_for_lens("back"))
        fovs = sorted(set(v.fov_deg for v in self.views))
        return (
            f"Views: {self.total_views()} total "
            f"({front} front, {back} back)\n"
            f"FOV: {', '.join(f'{f:.0f}°' for f in fovs)}\n"
            f"Crop: {self.crop_size}x{self.crop_size} @ Q{self.quality}"
        )


def _make_views(
    lens: str,
    positions: List[Tuple[float, float]],
    fov: float,
    name_prefix: str = "v",
) -> List[FisheyeView]:
    """Create views from explicit (yaw, pitch) positions relative to lens axis.

    All yaw/pitch values are relative to the lens's own optical axis.
    Keep angles within the usable range: |angle_from_center| < (lens_fov/2 - crop_fov/2).
    For ~190° fisheye with 90° crops, max center offset ≈ 50°.
    For ~190° fisheye with 110° crops, max center offset ≈ 40°.
    """
    views = []
    for i, (yaw, pitch) in enumerate(positions):
        views.append(FisheyeView(
            name=f"{lens}_{name_prefix}{i:02d}_y{yaw:+.0f}_p{pitch:+.0f}",
            yaw_deg=yaw,
            pitch_deg=pitch,
            fov_deg=fov,
            source_lens=lens,
        ))
    return views


# --- Preset definitions ---
# Naming: osv-{coverage}-f{fov}-dual-{total_views}
#
# View positions are in lens-local coordinates:
#   yaw=0, pitch=0 = looking straight through the optical center
#   yaw>0 = rotate right, yaw<0 = rotate left
#   pitch>0 = tilt up, pitch<0 = tilt down
#
# For DJI Osmo 360 (~190° per fisheye lens):
#   110° FOV crops: safe center offset up to ~40°
#    90° FOV crops: safe center offset up to ~50°

FISHEYE_PRESETS: Dict[str, FisheyeViewConfig] = {
    # ── Full hemisphere tiling ──────────────────────────────────────
    # These presets tile the usable hemisphere of each lens as
    # completely as possible within the safe center-offset limit.
    #
    # 90° FOV  → safe offset ≤ 50°  → 5 pitch tiers, 13 views/lens
    # 110° FOV → safe offset ≤ 40°  → 3 pitch tiers, 11 views/lens

    # 26 views at 90° FOV — full hemisphere tiling (DEFAULT)
    # 13 per lens: cap + 3 up + 5 horizon + 3 down + bottom cap
    "osv-full-f90-dual-26": FisheyeViewConfig(
        views=(
            _make_views("front", [
                (0, 45),                                               # top cap
                (-30, 30), (0, 30), (30, 30),                          # up ring
                (-45, 0), (-22, 0), (0, 0), (22, 0), (45, 0),         # horizon
                (-30, -30), (0, -30), (30, -30),                       # down ring
                (0, -45),                                              # bottom cap
            ], fov=90.0)
            + _make_views("back", [
                (0, 45),
                (-30, 30), (0, 30), (30, 30),
                (-45, 0), (-22, 0), (0, 0), (22, 0), (45, 0),
                (-30, -30), (0, -30), (30, -30),
                (0, -45),
            ], fov=90.0)
        ),
        crop_size=1600,
    ),

    # 22 views at 110° FOV — full hemisphere tiling, wider crops
    # 11 per lens: 3 up + 5 horizon + 3 down
    "osv-full-f110-dual-22": FisheyeViewConfig(
        views=(
            _make_views("front", [
                (-25, 30), (0, 30), (25, 30),                         # up ring
                (-40, 0), (-20, 0), (0, 0), (20, 0), (40, 0),         # horizon
                (-25, -30), (0, -30), (25, -30),                       # down ring
            ], fov=110.0)
            + _make_views("back", [
                (-25, 30), (0, 30), (25, 30),
                (-40, 0), (-20, 0), (0, 0), (20, 0), (40, 0),
                (-25, -30), (0, -30), (25, -30),
            ], fov=110.0)
        ),
        crop_size=1600,
    ),

    # ── Lighter presets ─────────────────────────────────────────────

    # 14 views at 90° FOV — balanced coverage without extreme tilt
    # 7 per lens: 5 horizon + 1 up + 1 down
    "osv-balanced-f90-dual-14": FisheyeViewConfig(
        views=(
            _make_views("front", [
                (0, 35),                                               # up
                (-45, 0), (-22, 0), (0, 0), (22, 0), (45, 0),         # horizon
                (0, -35),                                              # down
            ], fov=90.0)
            + _make_views("back", [
                (0, 35),
                (-45, 0), (-22, 0), (0, 0), (22, 0), (45, 0),
                (0, -35),
            ], fov=90.0)
        ),
        crop_size=1600,
    ),

    # 10 views at 90° FOV — fast testing, still covers up+down
    # 5 per lens: 3 horizon + 1 up + 1 down
    "osv-light-f90-dual-10": FisheyeViewConfig(
        views=(
            _make_views("front", [
                (0, 30),                                               # up
                (-35, 0), (0, 0), (35, 0),                             # horizon
                (0, -30),                                              # down
            ], fov=90.0)
            + _make_views("back", [
                (0, 30),
                (-35, 0), (0, 0), (35, 0),
                (0, -30),
            ], fov=90.0)
        ),
        crop_size=1600,
    ),
}


# --- Rotation math ---

def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation matrix for virtual camera orientation.

    Yaw rotates around Y axis, pitch around X axis.
    Convention matches reframer.py's _create_rotation_matrix.
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ], dtype=np.float64)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ], dtype=np.float64)

    return Rx @ Ry


# --- Reframer ---

class FisheyeReframer:
    """Extract perspective crops from calibrated fisheye images.

    Caches remap tables for views that share the same parameters,
    so repeated calls (e.g., processing a video) are fast after
    the first frame.
    """

    def __init__(self, calibration: DualFisheyeCalibration):
        self.calib = calibration
        self._map_cache: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def with_defaults(cls) -> "FisheyeReframer":
        """Create reframer with approximate DJI Osmo 360 calibration."""
        return cls(default_osmo360_calibration())

    def _get_lens_calib(self, lens: str) -> FisheyeCalibration:
        if lens == "front":
            return self.calib.front
        elif lens == "back":
            return self.calib.back
        raise ValueError(f"Unknown lens: {lens}. Use 'front' or 'back'.")

    def _get_lens_rotation(self, lens: str) -> float:
        if lens == "front":
            return self.calib.front_rotation_deg
        return self.calib.back_rotation_deg

    def _build_remap_tables(
        self,
        lens_calib: FisheyeCalibration,
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float,
        crop_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build remap tables for a specific perspective view.

        Uses cv2.fisheye.initUndistortRectifyMap with a rotation
        matrix to orient the virtual camera.
        """
        # Virtual pinhole camera for the output crop
        f_virtual = crop_size / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
        new_K = np.array([
            [f_virtual, 0, crop_size / 2.0],
            [0, f_virtual, crop_size / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)

        # Rotation: orient virtual camera to (yaw, pitch) direction
        R = _rotation_matrix(yaw_deg, pitch_deg)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K=lens_calib.camera_matrix,
            D=lens_calib.dist_coeffs,
            R=R,
            P=new_K,
            size=(crop_size, crop_size),
            m1type=cv2.CV_32FC1,
        )
        return map1, map2

    def extract_view(
        self,
        fisheye_image: np.ndarray,
        view: FisheyeView,
        crop_size: int = 1600,
    ) -> Optional[np.ndarray]:
        """Extract a single perspective view from a fisheye image.

        Returns the perspective crop, or None if the view extends
        beyond the lens's usable FOV.

        Note: view.yaw_deg and view.pitch_deg are relative to the
        lens's own optical axis. The lens rotation (front=0°, back=180°)
        is NOT applied here — it only matters when computing world-space
        poses for COLMAP. Each lens is an independent optical system.
        """
        lens_calib = self._get_lens_calib(view.source_lens)

        cache_key = (
            view.source_lens, view.yaw_deg, view.pitch_deg,
            view.fov_deg, crop_size,
        )
        if cache_key not in self._map_cache:
            self._map_cache[cache_key] = self._build_remap_tables(
                lens_calib, view.yaw_deg, view.pitch_deg,
                view.fov_deg, crop_size,
            )

        map1, map2 = self._map_cache[cache_key]
        crop = cv2.remap(
            fisheye_image, map1, map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return crop

    def extract_all_views(
        self,
        front_image: np.ndarray,
        back_image: np.ndarray,
        config: FisheyeViewConfig,
    ) -> List[Tuple[FisheyeView, np.ndarray]]:
        """Extract all views from a dual-fisheye frame pair.

        Args:
            front_image: Front fisheye image (stream 1 from OSV).
            back_image: Back fisheye image (stream 0 from OSV).
            config: View configuration with preset views.

        Returns:
            List of (view, crop_image) tuples.
        """
        results = []
        for view in config.views:
            img = front_image if view.source_lens == "front" else back_image
            crop = self.extract_view(img, view, config.crop_size)
            if crop is not None:
                results.append((view, crop))
        return results

    def extract_and_save(
        self,
        front_image: np.ndarray,
        back_image: np.ndarray,
        config: FisheyeViewConfig,
        output_dir: str,
        frame_name: str = "frame",
    ) -> List[str]:
        """Extract all views and save as JPEGs.

        Returns list of saved file paths.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved = []

        results = self.extract_all_views(front_image, back_image, config)
        for view, crop in results:
            filename = f"{frame_name}_{view.name}.jpg"
            filepath = out_path / filename
            cv2.imwrite(
                str(filepath), crop,
                [cv2.IMWRITE_JPEG_QUALITY, config.quality],
            )
            saved.append(str(filepath))

        return saved

    def get_virtual_pinhole_intrinsics(
        self, fov_deg: float, crop_size: int
    ) -> dict:
        """PINHOLE intrinsics for the perspective crops.

        All crops with the same FOV and size share identical intrinsics.
        This is what goes into COLMAP cameras.txt.

        Returns dict with model, width, height, params (fx, fy, cx, cy).
        """
        f = crop_size / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
        cx = crop_size / 2.0
        cy = crop_size / 2.0
        return {
            "model": "PINHOLE",
            "width": crop_size,
            "height": crop_size,
            "params": [f, f, cx, cy],
        }


# --- Batch processing ---

def _process_frame_pair(args):
    """Worker function for multiprocessing batch extraction."""
    (
        front_path, back_path,
        calib_dict, config_dict,
        output_dir, frame_name,
    ) = args

    # Reconstruct objects in worker process
    calib = DualFisheyeCalibration.load(calib_dict["_path"]) \
        if "_path" in calib_dict else _calib_from_dict(calib_dict)
    config = _config_from_dict(config_dict)

    reframer = FisheyeReframer(calib)

    front = cv2.imread(str(front_path))
    back = cv2.imread(str(back_path))

    if front is None:
        return [], f"Failed to load front image: {front_path}"
    if back is None:
        return [], f"Failed to load back image: {back_path}"

    saved = reframer.extract_and_save(
        front, back, config, output_dir, frame_name,
    )
    return saved, None


def _calib_from_dict(d: dict) -> DualFisheyeCalibration:
    """Reconstruct calibration from serialized dict (for multiprocessing)."""
    def _single(s):
        return FisheyeCalibration(
            camera_matrix=np.array(s["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(s["dist_coeffs"], dtype=np.float64).reshape(4, 1),
            image_size=tuple(s["image_size"]),
            rms_error=s["rms_error"],
            num_images_used=s["num_images_used"],
            fov_degrees=s["fov_degrees"],
        )
    return DualFisheyeCalibration(
        front=_single(d["front"]),
        back=_single(d["back"]),
        front_rotation_deg=d.get("front_rotation_deg", 0.0),
        back_rotation_deg=d.get("back_rotation_deg", 180.0),
        camera_model=d.get("camera_model", "Unknown"),
    )


def _config_from_dict(d: dict) -> FisheyeViewConfig:
    """Reconstruct view config from serialized dict."""
    views = [
        FisheyeView(
            name=v["name"],
            yaw_deg=v["yaw_deg"],
            pitch_deg=v["pitch_deg"],
            fov_deg=v["fov_deg"],
            source_lens=v["source_lens"],
        )
        for v in d["views"]
    ]
    return FisheyeViewConfig(
        views=views,
        crop_size=d["crop_size"],
        quality=d["quality"],
    )


def _config_to_dict(config: FisheyeViewConfig) -> dict:
    """Serialize view config for multiprocessing."""
    return {
        "views": [
            {
                "name": v.name,
                "yaw_deg": v.yaw_deg,
                "pitch_deg": v.pitch_deg,
                "fov_deg": v.fov_deg,
                "source_lens": v.source_lens,
            }
            for v in config.views
        ],
        "crop_size": config.crop_size,
        "quality": config.quality,
    }


def _calib_to_dict(calib: DualFisheyeCalibration) -> dict:
    """Serialize calibration for multiprocessing."""
    def _single(c):
        return {
            "camera_matrix": c.camera_matrix.tolist(),
            "dist_coeffs": c.dist_coeffs.flatten().tolist(),
            "image_size": list(c.image_size),
            "rms_error": c.rms_error,
            "num_images_used": c.num_images_used,
            "fov_degrees": c.fov_degrees,
        }
    return {
        "front": _single(calib.front),
        "back": _single(calib.back),
        "front_rotation_deg": calib.front_rotation_deg,
        "back_rotation_deg": calib.back_rotation_deg,
        "camera_model": calib.camera_model,
    }


def batch_extract(
    frame_pairs: List[Tuple[str, str]],
    config: FisheyeViewConfig,
    calibration: DualFisheyeCalibration,
    output_dir: str,
    num_workers: int = 1,
    progress_callback=None,
) -> Tuple[int, List[str]]:
    """Extract perspective views from multiple fisheye frame pairs.

    Args:
        frame_pairs: List of (front_path, back_path) tuples.
        config: View configuration.
        calibration: Dual fisheye calibration.
        output_dir: Output directory for crops.
        num_workers: Parallel workers (1 = sequential, recommended
                     since remap tables are large).
        progress_callback: Called with (current, total, message).

    Returns:
        (total_crops_saved, list_of_errors)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    calib_dict = _calib_to_dict(calibration)
    config_dict = _config_to_dict(config)

    total_saved = 0
    errors = []

    if num_workers > 1:
        args_list = [
            (front, back, calib_dict, config_dict, output_dir, f"frame_{i:05d}")
            for i, (front, back) in enumerate(frame_pairs)
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, (saved, error) in enumerate(
                executor.map(_process_frame_pair, args_list)
            ):
                if error:
                    errors.append(error)
                else:
                    total_saved += len(saved)
                if progress_callback:
                    progress_callback(i + 1, len(frame_pairs), f"pair {i+1}")
    else:
        # Sequential — reuse single reframer instance (cached remap tables)
        reframer = FisheyeReframer(calibration)
        for i, (front_path, back_path) in enumerate(frame_pairs):
            front = cv2.imread(str(front_path))
            back = cv2.imread(str(back_path))

            if front is None:
                errors.append(f"Failed to load: {front_path}")
                continue
            if back is None:
                errors.append(f"Failed to load: {back_path}")
                continue

            saved = reframer.extract_and_save(
                front, back, config, output_dir, f"frame_{i:05d}",
            )
            total_saved += len(saved)

            if progress_callback:
                progress_callback(i + 1, len(frame_pairs), f"pair {i+1}")

    return total_saved, errors


# --- CLI ---

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract perspective crops from fisheye images"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # List presets
    list_p = sub.add_parser("presets", help="List available presets")

    # Extract from a single pair
    single_p = sub.add_parser("extract", help="Extract views from one frame pair")
    single_p.add_argument("front", help="Front fisheye image")
    single_p.add_argument("back", help="Back fisheye image")
    single_p.add_argument("-o", "--output", required=True, help="Output directory")
    single_p.add_argument("-p", "--preset", default="osv-wide-f110-dual-12",
                           help="View preset name")
    single_p.add_argument("-c", "--calib", help="Calibration JSON (omit for defaults)")
    single_p.add_argument("--size", type=int, help="Override crop size")
    single_p.add_argument("--quality", type=int, help="Override JPEG quality")
    single_p.add_argument("--name", default="frame",
                           help="Output filename prefix")

    # Batch extract from directories
    batch_p = sub.add_parser("batch", help="Batch extract from frame directories")
    batch_p.add_argument("front_dir", help="Directory of front fisheye images")
    batch_p.add_argument("back_dir", help="Directory of back fisheye images")
    batch_p.add_argument("-o", "--output", required=True, help="Output directory")
    batch_p.add_argument("-p", "--preset", default="osv-wide-f110-dual-12")
    batch_p.add_argument("-c", "--calib", help="Calibration JSON")
    batch_p.add_argument("--size", type=int, help="Override crop size")
    batch_p.add_argument("--workers", type=int, default=1)

    # Show intrinsics
    intr_p = sub.add_parser("intrinsics", help="Show virtual pinhole intrinsics")
    intr_p.add_argument("-p", "--preset", default="osv-wide-f110-dual-12")
    intr_p.add_argument("--size", type=int, help="Override crop size")

    args = parser.parse_args()

    try:
        if args.command == "presets":
            for name, config in FISHEYE_PRESETS.items():
                print(f"\n--- {name} ---")
                print(config.summary())
            return 0

        elif args.command == "extract":
            config = FisheyeViewConfig.from_preset(args.preset)
            if args.size:
                config.crop_size = args.size
            if args.quality:
                config.quality = args.quality

            if args.calib:
                calib = DualFisheyeCalibration.load(args.calib)
            else:
                print("No calibration file — using approximate defaults")
                calib = default_osmo360_calibration()

            reframer = FisheyeReframer(calib)

            front = cv2.imread(args.front)
            back = cv2.imread(args.back)
            if front is None:
                print(f"Error: cannot load {args.front}")
                return 1
            if back is None:
                print(f"Error: cannot load {args.back}")
                return 1

            print(f"Extracting {config.total_views()} views...")
            saved = reframer.extract_and_save(
                front, back, config, args.output, args.name,
            )
            print(f"Saved {len(saved)} crops to {args.output}")

            # Show intrinsics
            fovs = set(v.fov_deg for v in config.views)
            for fov in sorted(fovs):
                intr = reframer.get_virtual_pinhole_intrinsics(fov, config.crop_size)
                print(f"  FOV {fov:.0f}°: f={intr['params'][0]:.1f} "
                      f"cx={intr['params'][2]:.1f} cy={intr['params'][3]:.1f}")
            return 0

        elif args.command == "batch":
            config = FisheyeViewConfig.from_preset(args.preset)
            if args.size:
                config.crop_size = args.size

            if args.calib:
                calib = DualFisheyeCalibration.load(args.calib)
            else:
                print("No calibration file — using approximate defaults")
                calib = default_osmo360_calibration()

            # Pair front/back by sorted filename
            front_dir = Path(args.front_dir)
            back_dir = Path(args.back_dir)
            exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

            front_imgs = sorted(
                f for ext in exts for f in front_dir.glob(ext)
            )
            back_imgs = sorted(
                f for ext in exts for f in back_dir.glob(ext)
            )

            if len(front_imgs) != len(back_imgs):
                print(f"Warning: {len(front_imgs)} front vs {len(back_imgs)} back images")
            pairs = list(zip(
                [str(f) for f in front_imgs],
                [str(b) for b in back_imgs],
            ))

            if not pairs:
                print("No image pairs found")
                return 1

            def progress(cur, total, msg):
                print(f"  [{cur}/{total}] {msg}")

            print(f"Processing {len(pairs)} frame pairs × "
                  f"{config.total_views()} views...")
            total, errors = batch_extract(
                pairs, config, calib, args.output,
                num_workers=args.workers,
                progress_callback=progress,
            )

            print(f"Saved {total} crops to {args.output}")
            if errors:
                print(f"{len(errors)} errors:")
                for e in errors[:5]:
                    print(f"  {e}")
            return 0

        elif args.command == "intrinsics":
            config = FisheyeViewConfig.from_preset(args.preset)
            if args.size:
                config.crop_size = args.size
            reframer = FisheyeReframer.with_defaults()
            fovs = sorted(set(v.fov_deg for v in config.views))
            print(f"Preset: {args.preset}")
            print(f"Crop size: {config.crop_size}x{config.crop_size}")
            for fov in fovs:
                intr = reframer.get_virtual_pinhole_intrinsics(fov, config.crop_size)
                p = intr["params"]
                print(f"\n  FOV {fov:.0f}°:")
                print(f"    Model: PINHOLE")
                print(f"    fx={p[0]:.2f}  fy={p[1]:.2f}")
                print(f"    cx={p[2]:.2f}  cy={p[3]:.2f}")
                print(f"    COLMAP line: PINHOLE {config.crop_size} {config.crop_size} "
                      f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f}")
            return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
