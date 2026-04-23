"""
Rig preset definitions for COLMAP's rig_configurator.

Each preset produces a COLMAP-format rig config JSON and a human-readable summary.
Presets are defined as Python dicts — COLMAP consumes the JSON at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RigPreset:
    """A named rig configuration preset."""
    name: str
    display_name: str
    camera_model: str
    description: str
    config: List[Dict[str, Any]]  # COLMAP rig config JSON structure
    intrinsic_priors: Dict[str, str] = field(default_factory=dict)
    # Map of image_prefix -> camera_params string for ImageReader


# ── DJI Osmo 360 Dual Fisheye ──────────────────────────────────────
# Calibration: 248 frame pairs, scale bar verified to 0.0% error.
# Rotation: 180° around Y (179.996° measured, sub-degree tolerance).
# Baseline: 26mm along +Z in front camera frame.
# Source: osmo360_rig_calibration_report.md

DJI_OSMO_360 = RigPreset(
    name="dji_osmo_360",
    display_name="DJI Osmo 360 Dual Fisheye",
    camera_model="OPENCV_FISHEYE",
    description="Dual 190° fisheye, 26mm baseline, 180° Y rotation",
    config=[{
        "cameras": [
            {
                "image_prefix": "front/",
                "ref_sensor": True,
            },
            {
                "image_prefix": "back/",
                "cam_from_rig_rotation": [0, 0, 1, 0],
                "cam_from_rig_translation": [0, 0, 0.026],
            },
        ]
    }],
    intrinsic_priors={
        "front/": "1046,1046,1916,1919,0,0,0,0",
        "back/": "1045,1045,1912,1918,0,0,0,0",
    },
)

# Registry of all built-in presets
PRESETS: Dict[str, RigPreset] = {
    DJI_OSMO_360.name: DJI_OSMO_360,
}

PRESET_DISPLAY_NAMES: List[str] = (
    ["None"] + [p.display_name for p in PRESETS.values()] + ["Custom..."]
)


def get_preset_by_display_name(display_name: str) -> Optional[RigPreset]:
    """Look up a preset by its display name. Returns None for 'None' or 'Custom...'."""
    for preset in PRESETS.values():
        if preset.display_name == display_name:
            return preset
    return None


def write_rig_config(preset_or_config: RigPreset | List[Dict], output_path: Path) -> Path:
    """Write a COLMAP rig config JSON file. Returns the written path."""
    config = (
        preset_or_config.config
        if isinstance(preset_or_config, RigPreset)
        else preset_or_config
    )
    output_path = Path(output_path)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_path


def load_rig_config(path: Path) -> List[Dict[str, Any]]:
    """Load a rig config JSON file. Returns the parsed config list."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_rig_summary(
    config: List[Dict[str, Any]], preset: Optional[RigPreset] = None
) -> str:
    """Format a human-readable summary of a rig config for the read-only display."""
    if not config:
        return "(empty rig config)"

    lines = []
    for rig in config:
        cameras = rig.get("cameras", [])
        lines.append(f"Cameras: {len(cameras)}")

        for cam in cameras:
            prefix = cam.get("image_prefix", "?")
            if cam.get("ref_sensor"):
                lines.append(f"  {prefix} — ref sensor (identity)")
            else:
                rot = cam.get("cam_from_rig_rotation", [])
                trans = cam.get("cam_from_rig_translation", [])
                rot_str = f"quat {rot}" if rot else "identity"
                if trans and any(t != 0 for t in trans):
                    trans_mm = [round(t * 1000, 1) for t in trans]
                    trans_str = f"{trans_mm} mm"
                else:
                    trans_str = "zero"
                lines.append(f"  {prefix} — rot: {rot_str}, trans: {trans_str}")

    if preset:
        lines.append(f"Camera model: {preset.camera_model}")
        for prefix, params in preset.intrinsic_priors.items():
            lines.append(f"  {prefix} params: {params}")

    return "\n".join(lines)
