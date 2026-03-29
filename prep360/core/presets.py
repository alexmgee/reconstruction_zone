"""
Preset Manager Module

Save and load camera/extraction presets for different configurations.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

from .reframer import ViewConfig, Ring


@dataclass
class Preset:
    """Complete preset configuration."""
    name: str
    description: str = ""

    # Camera info
    camera_model: str = ""
    lens_mm: float = 0.0

    # Extraction settings
    extraction_interval: float = 2.0
    extraction_max_gap: float = 3.0
    extraction_mode: str = "fixed"  # fixed, scene, sharpest
    extraction_quality: int = 95

    # Reframe settings
    reframe_rings: List[Dict[str, Any]] = field(default_factory=list)
    reframe_zenith: bool = True
    reframe_nadir: bool = False
    reframe_zenith_fov: float = 65.0
    reframe_output_size: int = 1920
    reframe_jpeg_quality: int = 95

    # Color settings
    lut_path: Optional[str] = None
    color_shadow: int = 50
    color_highlight: int = 50

    # Sky filter settings
    sky_filter_enabled: bool = True
    sky_pitch_threshold: float = 30.0
    sky_brightness_threshold: float = 0.85
    sky_keypoint_threshold: int = 50

    # Masking settings
    mask_model: str = "yolo11s-seg"
    mask_classes: List[int] = field(default_factory=lambda: [0, 24, 25, 26, 27, 28])
    mask_confidence: float = 0.5

    # COLMAP export settings
    colmap_export_xmp: bool = True
    colmap_export_rig: bool = False
    colmap_xmp_pose_prior: str = "locked"
    colmap_undistort_frame: bool = True

    def get_view_config(self) -> ViewConfig:
        """Convert preset to ViewConfig."""
        rings = [
            Ring(
                pitch=r.get("pitch", 0),
                count=r.get("count", 8),
                fov=r.get("fov", 65.0),
                start_yaw=r.get("start_yaw", 0.0)
            )
            for r in self.reframe_rings
        ]
        return ViewConfig(
            rings=rings,
            include_zenith=self.reframe_zenith,
            include_nadir=self.reframe_nadir,
            zenith_fov=self.reframe_zenith_fov,
            output_size=self.reframe_output_size,
            jpeg_quality=self.reframe_jpeg_quality,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Preset":
        """Deserialize from dictionary."""
        # Handle missing fields with defaults
        return cls(
            name=data.get("name", "Unnamed"),
            description=data.get("description", ""),
            camera_model=data.get("camera_model", ""),
            lens_mm=data.get("lens_mm", 0.0),
            extraction_interval=data.get("extraction_interval", 2.0),
            extraction_max_gap=data.get("extraction_max_gap", 3.0),
            extraction_mode=data.get("extraction_mode", "fixed"),
            extraction_quality=data.get("extraction_quality", 95),
            reframe_rings=data.get("reframe_rings", []),
            reframe_zenith=data.get("reframe_zenith", True),
            reframe_nadir=data.get("reframe_nadir", False),
            reframe_zenith_fov=data.get("reframe_zenith_fov", 65.0),
            reframe_output_size=data.get("reframe_output_size", 1920),
            reframe_jpeg_quality=data.get("reframe_jpeg_quality", 95),
            lut_path=data.get("lut_path"),
            color_shadow=data.get("color_shadow", 50),
            color_highlight=data.get("color_highlight", 50),
            sky_filter_enabled=data.get("sky_filter_enabled", True),
            sky_pitch_threshold=data.get("sky_pitch_threshold", 30.0),
            sky_brightness_threshold=data.get("sky_brightness_threshold", 0.85),
            sky_keypoint_threshold=data.get("sky_keypoint_threshold", 50),
            mask_model=data.get("mask_model", "yolo11s-seg"),
            mask_classes=data.get("mask_classes", [0, 24, 25, 26, 27, 28]),
            mask_confidence=data.get("mask_confidence", 0.5),
            colmap_export_xmp=data.get("colmap_export_xmp", True),
            colmap_export_rig=data.get("colmap_export_rig", False),
            colmap_xmp_pose_prior=data.get("colmap_xmp_pose_prior", "locked"),
            colmap_undistort_frame=data.get("colmap_undistort_frame", True),
        )


# Built-in presets
BUILTIN_PRESETS = {
    "prep360_default": Preset(
        name="prep360_default",
        description="Default: 8 horizon + 4 below + zenith",
        extraction_interval=2.0,
        reframe_rings=[
            {"pitch": 0, "count": 8, "fov": 65},
            {"pitch": -20, "count": 4, "fov": 65},
        ],
        reframe_zenith=True,
        reframe_nadir=False,
    ),
    "dji_osmo_360": Preset(
        name="dji_osmo_360",
        description="DJI Osmo Action 360 with D-Log",
        camera_model="DJI Osmo Action 360",
        lens_mm=16.0,
        extraction_interval=2.0,
        reframe_rings=[
            {"pitch": 0, "count": 8, "fov": 65},
            {"pitch": -20, "count": 4, "fov": 65},
        ],
        reframe_zenith=True,
        reframe_nadir=False,
        lut_path="DJI_DLog_M_to_Rec709.cube",
    ),
    "insta360_x3": Preset(
        name="insta360_x3",
        description="Insta360 X3 with I-Log",
        camera_model="Insta360 X3",
        lens_mm=15.0,
        extraction_interval=2.0,
        reframe_rings=[
            {"pitch": 0, "count": 8, "fov": 65},
            {"pitch": -20, "count": 4, "fov": 65},
        ],
        reframe_zenith=True,
        reframe_nadir=False,
        lut_path="Insta360_ILog_to_Rec709.cube",
    ),
    "gopro_max": Preset(
        name="gopro_max",
        description="GoPro MAX with Protune",
        camera_model="GoPro MAX",
        lens_mm=16.0,
        extraction_interval=2.0,
        reframe_rings=[
            {"pitch": 0, "count": 8, "fov": 65},
            {"pitch": -20, "count": 4, "fov": 65},
        ],
        reframe_zenith=True,
        reframe_nadir=False,
        lut_path="GoPro_Protune_to_Rec709.cube",
    ),
    "lightweight": Preset(
        name="lightweight",
        description="Lightweight: 6 horizon views only",
        extraction_interval=3.0,
        reframe_rings=[
            {"pitch": 0, "count": 6, "fov": 75},
        ],
        reframe_zenith=True,
        reframe_nadir=False,
    ),
    "dense": Preset(
        name="dense",
        description="Dense coverage: full sphere with many views",
        extraction_interval=1.5,
        reframe_rings=[
            {"pitch": 0, "count": 8, "fov": 60},
            {"pitch": 30, "count": 4, "fov": 60},
            {"pitch": -30, "count": 4, "fov": 60},
        ],
        reframe_zenith=True,
        reframe_nadir=True,
    ),
}


class PresetManager:
    """Manage presets: load, save, list."""

    def __init__(self, presets_dir: Optional[str] = None):
        if presets_dir:
            self.presets_dir = Path(presets_dir)
        else:
            # Default to presets folder next to this file
            self.presets_dir = Path(__file__).parent.parent / "presets"

        self.presets_dir.mkdir(parents=True, exist_ok=True)

        # Cache of loaded presets
        self._cache: Dict[str, Preset] = {}

    def save(self, preset: Preset) -> Path:
        """
        Save preset to JSON file.

        Returns:
            Path to saved file
        """
        filename = f"{preset.name}.json"
        filepath = self.presets_dir / filename

        with open(filepath, 'w') as f:
            json.dump(preset.to_dict(), f, indent=2)

        self._cache[preset.name] = preset
        return filepath

    def load(self, name: str) -> Preset:
        """
        Load preset by name.

        Checks user presets first, then built-in presets.
        """
        # Check cache
        if name in self._cache:
            return self._cache[name]

        # Check user presets
        filepath = self.presets_dir / f"{name}.json"
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            preset = Preset.from_dict(data)
            self._cache[name] = preset
            return preset

        # Check built-in presets
        if name in BUILTIN_PRESETS:
            return BUILTIN_PRESETS[name]

        raise ValueError(f"Preset not found: {name}")

    def list_presets(self) -> List[str]:
        """List all available preset names."""
        # User presets
        user_presets = [p.stem for p in self.presets_dir.glob("*.json")]

        # Built-in presets
        builtin = list(BUILTIN_PRESETS.keys())

        # Combine and deduplicate
        all_presets = sorted(set(user_presets + builtin))
        return all_presets

    def list_user_presets(self) -> List[str]:
        """List only user-created presets."""
        return sorted([p.stem for p in self.presets_dir.glob("*.json")])

    def list_builtin_presets(self) -> List[str]:
        """List only built-in presets."""
        return sorted(BUILTIN_PRESETS.keys())

    def delete(self, name: str) -> bool:
        """
        Delete a user preset.

        Returns:
            True if deleted, False if not found or is built-in
        """
        if name in BUILTIN_PRESETS:
            return False  # Can't delete built-in

        filepath = self.presets_dir / f"{name}.json"
        if filepath.exists():
            filepath.unlink()
            if name in self._cache:
                del self._cache[name]
            return True

        return False

    def get_preset_info(self, name: str) -> Dict[str, Any]:
        """Get preset info without full loading."""
        try:
            preset = self.load(name)
            view_config = preset.get_view_config()
            return {
                "name": preset.name,
                "description": preset.description,
                "camera_model": preset.camera_model,
                "total_views": view_config.total_views(),
                "is_builtin": name in BUILTIN_PRESETS,
            }
        except Exception:
            return {"name": name, "error": "Failed to load"}


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Manage prep360 presets")
    parser.add_argument("command", choices=["list", "show", "export"],
                        help="Command to execute")
    parser.add_argument("name", nargs="?", help="Preset name (for show/export)")
    parser.add_argument("--dir", help="Custom presets directory")

    args = parser.parse_args()

    manager = PresetManager(args.dir)

    if args.command == "list":
        print("Available presets:")
        for name in manager.list_presets():
            info = manager.get_preset_info(name)
            builtin = "[builtin]" if info.get("is_builtin") else "[user]"
            views = info.get("total_views", "?")
            desc = info.get("description", "")[:50]
            print(f"  {name:20} {builtin:10} {views:2} views  {desc}")

    elif args.command == "show":
        if not args.name:
            print("Error: preset name required")
            return 1

        try:
            preset = manager.load(args.name)
            print(json.dumps(preset.to_dict(), indent=2))
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif args.command == "export":
        if not args.name:
            print("Error: preset name required")
            return 1

        try:
            preset = manager.load(args.name)
            path = manager.save(preset)
            print(f"Exported to: {path}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
