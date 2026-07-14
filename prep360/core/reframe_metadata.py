"""
Metashape-oriented reframe metadata writer.

Writes reframe_metadata.json for rig, station, and flat ERP pinhole output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

from .reframer import (
    OutputLayout,
    ViewConfig,
    compute_pinhole_intrinsics,
    create_rotation_matrix,
)
from .rig_config import rotation_matrix_to_quaternion


def _intrinsics_payload(config: ViewConfig) -> dict:
    views = config.get_all_views()
    fovs = sorted({v.fov for v in views})
    by_fov = {f"{fov}": compute_pinhole_intrinsics(fov, config.output_size) for fov in fovs}
    default_fov = views[0].fov if views else 90.0
    return {
        "default_pinhole_intrinsics": by_fov[f"{default_fov}"],
        "pinhole_intrinsics_by_fov": by_fov,
    }


def _metashape_sensors(config: ViewConfig) -> List[dict]:
    views = config.get_all_views()
    if not views:
        return []

    ref = views[0]
    R_ref = create_rotation_matrix(ref.yaw, ref.pitch)
    sensors: List[dict] = []

    for i, view in enumerate(views):
        entry = {
            "folder": view.name,
            "label": view.name,
            "ref_sensor": i == 0,
        }
        if i == 0:
            entry["cam_from_ref_rotation_deg"] = [0.0, 0.0, 0.0]
        else:
            R_i = create_rotation_matrix(view.yaw, view.pitch)
            R_rel = R_i @ R_ref.T
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_rel)
            entry["cam_from_ref_rotation_quat_wxyz"] = [qw, qx, qy, qz]
        sensors.append(entry)
    return sensors


def _stations_payload(images: List[Path], config: ViewConfig) -> List[dict]:
    views = config.get_all_views()
    stations = []
    for img in images:
        stem = img.stem
        stations.append({
            "source_image": img.name,
            "image_directory": f"images/{stem}/",
            "mask_directory": f"masks/{stem}/",
            "views": [f"{stem}_{v.name}.jpg" for v in views],
        })
    return stations


def _frames_payload(images: List[Path], config: ViewConfig, output_layout: OutputLayout) -> List[dict]:
    views = config.get_all_views()
    frames = []
    for img in images:
        stem = img.stem
        if output_layout == OutputLayout.RIG:
            view_paths = {v.name: f"images/{v.name}/{stem}.jpg" for v in views}
        elif output_layout == OutputLayout.FLAT:
            view_paths = {v.name: f"images/{stem}_{v.name}.jpg" for v in views}
        else:
            view_paths = {v.name: f"images/{stem}/{stem}_{v.name}.jpg" for v in views}
        frames.append({
            "source_image": img.name,
            "views": view_paths,
        })
    return frames


def build_reframe_metadata(
    config: ViewConfig,
    output_layout: OutputLayout,
    preset_name: str,
    images: Optional[List[Path]] = None,
) -> dict:
    views = config.get_all_views()
    intr = _intrinsics_payload(config)

    metadata = {
        "version": 2,
        "output_layout": output_layout.value,
        "reframe_config": {
            "preset": preset_name,
            "output_size": config.output_size,
            "jpeg_quality": config.jpeg_quality,
            "views": [
                {
                    "name": v.name,
                    "yaw": v.yaw,
                    "pitch": v.pitch,
                    "fov": v.fov,
                    "flip_vertical": v.flip_vertical,
                }
                for v in views
            ],
        },
        "pinhole_intrinsics": intr["default_pinhole_intrinsics"],
        "default_pinhole_intrinsics": intr["default_pinhole_intrinsics"],
        "pinhole_intrinsics_by_fov": intr["pinhole_intrinsics_by_fov"],
        "mask_layout": "mirrors_output_layout",
        "frames": [],
        "stations": [],
        "metashape": {
            "workflow": output_layout.value,
            "import_images_from": "images/",
            "pinhole_setup": {
                "projection": "Perspective",
                "fix_calibration": True,
                "note": "Set fx, fy, cx, cy from default_pinhole_intrinsics or pinhole_intrinsics_by_fov and lock before alignment",
            },
            "sensors": [],
        },
    }

    if output_layout == OutputLayout.RIG:
        metadata["metashape"]["sensors"] = _metashape_sensors(config)
        if images:
            metadata["frames"] = _frames_payload(images, config, output_layout)
    elif output_layout == OutputLayout.STATION and images:
        metadata["stations"] = _stations_payload(images, config)
        metadata["metashape"]["workflow"] = "station"
    elif output_layout == OutputLayout.FLAT:
        metadata["metashape"]["workflow"] = "flat"
        if images:
            metadata["frames"] = _frames_payload(images, config, output_layout)

    return metadata


def write_reframe_metadata(
    output_root: Path,
    config: ViewConfig,
    output_layout: OutputLayout,
    preset_name: str,
    images: Optional[List[Path]] = None,
) -> Path:
    payload = build_reframe_metadata(config, output_layout, preset_name, images)
    path = output_root / "reframe_metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def log_pinhole_intrinsics(
    config: ViewConfig,
    output_layout: OutputLayout,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    if log is None:
        return
    views = config.get_all_views()
    if not views:
        return
    fov = views[0].fov
    intr = compute_pinhole_intrinsics(fov, config.output_size)
    log("Output pinhole camera parameters (set and lock in Metashape):")
    log("  projection: Perspective (pinhole)")
    log(f"  width: {intr['width']}  height: {intr['height']}")
    log(f"  fx: {intr['fx']}  fy: {intr['fy']}  cx: {intr['cx']}  cy: {intr['cy']}")
    log("  distortion: k1=k2=k3=p1=p2=0")

    if output_layout == OutputLayout.RIG:
        sensors = _metashape_sensors(config)
        log(f"Rig sensors: {len(sensors)}")
        log(f"  Reference: {sensors[0]['folder']}/")
        for sensor in sensors[1:4]:
            q = sensor.get("cam_from_ref_rotation_quat_wxyz", [])
            if q:
                log(
                    f"  Sensor {sensor['folder']}/: "
                    f"rotation (w,x,y,z) = ({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f})"
                )
        if len(sensors) > 4:
            log(f"  ... and {len(sensors) - 4} more sensors (see reframe_metadata.json)")
