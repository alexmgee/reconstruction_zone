"""
Reframe Engine Module

Convert equirectangular images to multiple pinhole perspective views
using a ring-based configuration system.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

# Try to import py360convert for better quality
try:
    import py360convert
    HAS_PY360 = True
except ImportError:
    HAS_PY360 = False


@dataclass
class Ring:
    """Configuration for a ring of views at a specific pitch."""
    pitch: float            # Degrees from horizon (-90 to +90)
    count: int              # Number of views in this ring
    fov: float = 65.0       # Field of view in degrees
    start_yaw: float = 0.0  # Starting yaw offset in degrees

    def get_yaw_positions(self) -> List[float]:
        """Get yaw angles for all views in this ring."""
        if self.count == 0:
            return []
        step = 360.0 / self.count
        return [self.start_yaw + i * step for i in range(self.count)]


@dataclass
class ViewConfig:
    """Configuration for all perspective views."""
    rings: List[Ring] = field(default_factory=list)
    include_zenith: bool = True     # Top view (pitch=90)
    include_nadir: bool = False     # Bottom view (pitch=-90)
    zenith_fov: float = 65.0        # FOV for zenith/nadir
    output_size: int = 1920         # Output image size (square)
    jpeg_quality: int = 95          # JPEG quality (1-100)

    def total_views(self) -> int:
        """Calculate total number of views per frame."""
        count = sum(ring.count for ring in self.rings)
        if self.include_zenith:
            count += 1
        if self.include_nadir:
            count += 1
        return count

    def get_all_views(self) -> List[Tuple[float, float, float, str]]:
        """
        Get all view parameters as (yaw, pitch, fov, name) tuples.
        """
        views = []

        # Ring views
        for ring_idx, ring in enumerate(self.rings):
            for view_idx, yaw in enumerate(ring.get_yaw_positions()):
                name = f"{ring_idx:02d}_{view_idx:02d}"
                views.append((yaw, ring.pitch, ring.fov, name))

        # Zenith
        if self.include_zenith:
            views.append((0, 90, self.zenith_fov, "ZN_00"))

        # Nadir
        if self.include_nadir:
            views.append((0, -90, self.zenith_fov, "ND_00"))

        return views

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "rings": [
                {"pitch": r.pitch, "count": r.count, "fov": r.fov, "start_yaw": r.start_yaw}
                for r in self.rings
            ],
            "include_zenith": self.include_zenith,
            "include_nadir": self.include_nadir,
            "zenith_fov": self.zenith_fov,
            "output_size": self.output_size,
            "jpeg_quality": self.jpeg_quality,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ViewConfig":
        """Deserialize from dictionary."""
        rings = [
            Ring(
                pitch=r["pitch"],
                count=r["count"],
                fov=r.get("fov", 65.0),
                start_yaw=r.get("start_yaw", 0.0)
            )
            for r in data.get("rings", [])
        ]
        return cls(
            rings=rings,
            include_zenith=data.get("include_zenith", True),
            include_nadir=data.get("include_nadir", False),
            zenith_fov=data.get("zenith_fov", 65.0),
            output_size=data.get("output_size", 1920),
            jpeg_quality=data.get("jpeg_quality", 95),
        )


# Built-in presets
VIEW_PRESETS = {
    "prep360_default": ViewConfig(
        rings=[
            Ring(pitch=0, count=8, fov=65),      # Horizon: 8 views @ 45° spacing
            Ring(pitch=-20, count=4, fov=65),   # Below: 4 views @ 90° spacing
        ],
        include_zenith=True,
        include_nadir=False,
    ),
    "dense_coverage": ViewConfig(
        rings=[
            Ring(pitch=0, count=8, fov=60),
            Ring(pitch=30, count=4, fov=60),
            Ring(pitch=-30, count=4, fov=60),
        ],
        include_zenith=True,
        include_nadir=True,
    ),
    "lightweight": ViewConfig(
        rings=[
            Ring(pitch=0, count=6, fov=75),
        ],
        include_zenith=True,
        include_nadir=False,
    ),
    "octa_horizon": ViewConfig(
        rings=[
            Ring(pitch=0, count=8, fov=65),
        ],
        include_zenith=True,
        include_nadir=False,
    ),
    "full_sphere": ViewConfig(
        rings=[
            Ring(pitch=0, count=8, fov=60),
            Ring(pitch=45, count=4, fov=60),
            Ring(pitch=-45, count=4, fov=60),
        ],
        include_zenith=True,
        include_nadir=True,
    ),
}


def _create_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0) -> np.ndarray:
    """Create rotation matrix from Euler angles (degrees)."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    # Rotation matrices
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    return Rz @ Rx @ Ry


def _equirect_to_perspective_custom(
    equirect: np.ndarray,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    out_size: int,
    mode: str = "bilinear"
) -> np.ndarray:
    """
    Extract perspective view from equirectangular image.
    Custom implementation for when py360convert is unavailable.
    """
    h_eq, w_eq = equirect.shape[:2]

    # Focal length from FoV
    fov_rad = np.radians(fov_deg)
    f = (out_size / 2) / np.tan(fov_rad / 2)

    # Create pixel grid for output image
    u = np.arange(out_size) - out_size / 2
    v = np.arange(out_size) - out_size / 2
    u, v = np.meshgrid(u, v)

    # Convert to 3D rays (camera coordinates)
    x = u
    y = -v  # Flip y for image coordinates
    z = np.full_like(u, f, dtype=np.float64)

    # Stack and rotate to world coordinates
    xyz = np.stack([x, y, z], axis=-1)
    R = _create_rotation_matrix(yaw_deg, pitch_deg, 0)
    xyz_rot = xyz @ R.T

    # Convert to spherical coordinates
    x_r, y_r, z_r = xyz_rot[..., 0], xyz_rot[..., 1], xyz_rot[..., 2]

    # Longitude (theta) and latitude (phi)
    theta = np.arctan2(x_r, z_r)
    norm = np.linalg.norm(xyz_rot, axis=-1)
    phi = np.arcsin(np.clip(y_r / norm, -1, 1))

    # Map to equirectangular coordinates
    u_eq = (theta / np.pi + 1) / 2 * w_eq
    v_eq = (0.5 - phi / np.pi) * h_eq

    # Interpolation
    if mode == "nearest":
        u_eq = np.round(u_eq).astype(int) % w_eq
        v_eq = np.clip(np.round(v_eq).astype(int), 0, h_eq - 1)
        return equirect[v_eq, u_eq]
    else:
        # Bilinear interpolation using cv2.remap
        map_x = u_eq.astype(np.float32) % w_eq
        map_y = np.clip(v_eq.astype(np.float32), 0, h_eq - 1)

        return cv2.remap(
            equirect,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )


def reframe_view(
    equirect: np.ndarray,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    out_size: int,
    mode: str = "bilinear"
) -> np.ndarray:
    """
    Extract a single perspective view from equirectangular image.

    Args:
        equirect: Input equirectangular image (H, W, C)
        fov_deg: Field of view in degrees
        yaw_deg: Yaw angle in degrees (0 = front)
        pitch_deg: Pitch angle in degrees (0 = horizon, 90 = up)
        out_size: Output size (square)
        mode: Interpolation mode ("bilinear" or "nearest")

    Returns:
        Perspective view as numpy array
    """
    if HAS_PY360:
        return py360convert.e2p(
            equirect,
            fov_deg=fov_deg,
            u_deg=yaw_deg,
            v_deg=pitch_deg,
            out_hw=(out_size, out_size),
            mode=mode
        )
    else:
        return _equirect_to_perspective_custom(
            equirect, fov_deg, yaw_deg, pitch_deg, out_size, mode
        )


def _process_single_image(args):
    """Worker function for multiprocessing."""
    (
        image_path,
        output_dir,
        config,
        frame_idx,
        mask_path,
    ) = args

    # Load image
    equirect = cv2.imread(str(image_path))
    if equirect is None:
        return None, f"Failed to load {image_path}"

    # Load mask if provided
    mask = None
    if mask_path:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, f"Failed to load mask {mask_path}"
        if mask.shape[:2] != equirect.shape[:2]:
            return None, f"Mask dimensions {mask.shape[:2]} don't match image {equirect.shape[:2]} for {image_path}"

    stem = Path(image_path).stem
    output_dir = Path(output_dir)
    output_files = []

    # Mask output directory (sibling of image output dir)
    mask_dir = None
    if mask is not None:
        mask_dir = output_dir.parent / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

    # Process all views
    views = config.get_all_views()
    for yaw, pitch, fov, view_name in views:
        persp = reframe_view(
            equirect,
            fov_deg=fov,
            yaw_deg=yaw,
            pitch_deg=pitch,
            out_size=config.output_size
        )

        # Output filename
        out_name = f"{stem}_{view_name}.jpg"
        out_path = output_dir / out_name

        # Save with quality setting
        cv2.imwrite(
            str(out_path),
            persp,
            [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
        )
        output_files.append(out_name)

        # Reframe and save mask with nearest-neighbor interpolation
        if mask is not None:
            mask_persp = reframe_view(
                mask,
                fov_deg=fov,
                yaw_deg=yaw,
                pitch_deg=pitch,
                out_size=config.output_size,
                mode="nearest"
            )
            mask_persp = (mask_persp > 0).astype(np.uint8) * 255
            mask_out = f"{stem}_{view_name}.png"
            cv2.imwrite(str(mask_dir / mask_out), mask_persp)

    return output_files, None


@dataclass
class ReframeResult:
    """Result of reframing operation."""
    success: bool
    input_count: int
    output_count: int
    output_dir: str
    errors: List[str] = field(default_factory=list)


class Reframer:
    """Reframe equirectangular images to perspective views."""

    def __init__(self, config: Optional[ViewConfig] = None):
        self.config = config or VIEW_PRESETS["prep360_default"]

    def reframe_single(
        self,
        image_path: str,
        output_dir: str,
        mask_path: Optional[str] = None,
    ) -> Tuple[List[str], Optional[str]]:
        """
        Reframe a single equirectangular image.

        Returns:
            (list of output filenames, error message or None)
        """
        args = (image_path, output_dir, self.config, 0, mask_path)
        return _process_single_image(args)

    def reframe_batch(
        self,
        input_dir: str,
        output_dir: str,
        mask_dir: Optional[str] = None,
        num_workers: int = 4,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> ReframeResult:
        """
        Reframe all equirectangular images in a directory.

        Args:
            input_dir: Directory containing equirectangular images
            output_dir: Output directory for perspective views
            mask_dir: Optional directory containing masks (matched by stem name)
            num_workers: Number of parallel workers
            progress_callback: Called with (current, total, message)

        Returns:
            ReframeResult with operation details
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(input_path.glob(ext))
        images = sorted(images)

        if not images:
            return ReframeResult(
                success=False,
                input_count=0,
                output_count=0,
                output_dir=str(output_path),
                errors=["No images found in input directory"]
            )

        # Build mask lookup by stem name
        mask_map = {}
        if mask_dir:
            mask_path = Path(mask_dir)
            for ext in extensions:
                for m in mask_path.glob(ext):
                    mask_map[m.stem] = str(m)
            matched = sum(1 for img in images if img.stem in mask_map)
            unmatched = len(images) - matched
            if unmatched > 0:
                print(f"Warning: {unmatched} of {len(images)} frames have no matching mask")

        # Prepare arguments for parallel processing
        args_list = [
            (str(img), str(output_path), self.config, i,
             mask_map.get(img.stem) if mask_dir else None)
            for i, img in enumerate(images)
        ]

        # Process
        errors = []
        total_outputs = 0

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for i, (outputs, error) in enumerate(executor.map(_process_single_image, args_list)):
                    if error:
                        errors.append(error)
                    elif outputs:
                        total_outputs += len(outputs)

                    if progress_callback:
                        progress_callback(i + 1, len(images), images[i].name)
        else:
            for i, args in enumerate(args_list):
                outputs, error = _process_single_image(args)
                if error:
                    errors.append(error)
                elif outputs:
                    total_outputs += len(outputs)

                if progress_callback:
                    progress_callback(i + 1, len(images), images[i].name)

        return ReframeResult(
            success=len(errors) == 0,
            input_count=len(images),
            output_count=total_outputs,
            output_dir=str(output_path),
            errors=errors
        )

    def preview_view_positions(self) -> str:
        """Generate text description of view positions."""
        lines = [f"Total views: {self.config.total_views()}"]

        for i, ring in enumerate(self.config.rings):
            spacing = 360.0 / ring.count if ring.count > 0 else 0
            lines.append(f"Ring {i}: pitch={ring.pitch:+.0f}° count={ring.count} spacing={spacing:.0f}° fov={ring.fov}°")

        if self.config.include_zenith:
            lines.append(f"Zenith: pitch=+90° fov={self.config.zenith_fov}°")
        if self.config.include_nadir:
            lines.append(f"Nadir: pitch=-90° fov={self.config.zenith_fov}°")

        return "\n".join(lines)


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reframe equirectangular images to perspective views")
    parser.add_argument("input", help="Input directory with equirectangular images")
    parser.add_argument("output", help="Output directory for perspective views")
    parser.add_argument("--preset", "-p", choices=list(VIEW_PRESETS.keys()),
                        default="prep360_default", help="View configuration preset")
    parser.add_argument("--size", "-s", type=int, default=1920,
                        help="Output image size (default: 1920)")
    parser.add_argument("--quality", "-q", type=int, default=95,
                        help="JPEG quality (default: 95)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--info", action="store_true",
                        help="Show preset info and exit")

    args = parser.parse_args()

    config = VIEW_PRESETS[args.preset]
    config.output_size = args.size
    config.jpeg_quality = args.quality

    reframer = Reframer(config)

    if args.info:
        print(f"Preset: {args.preset}")
        print(reframer.preview_view_positions())
        return 0

    def progress(current, total, name):
        print(f"[{current}/{total}] {name}")

    result = reframer.reframe_batch(
        args.input,
        args.output,
        num_workers=args.workers,
        progress_callback=progress
    )

    if result.success:
        print(f"Reframed {result.input_count} images -> {result.output_count} views")
        return 0
    else:
        print(f"Completed with {len(result.errors)} errors:")
        for err in result.errors[:5]:
            print(f"  {err}")
        return 1


if __name__ == "__main__":
    exit(main())
