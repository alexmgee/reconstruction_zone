"""
COLMAP-compatible rig configuration from ERP ViewConfig geometry.

Relative rotations use R_i @ R_ref.T where R = create_rotation_matrix(yaw, pitch).
Provenance: reimplemented from documented ERP pinhole spec (LichtFeld plugin report);
no GPL source pasted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from .reframer import ViewConfig, create_rotation_matrix


def rotation_matrix_to_quaternion(R: np.ndarray) -> list[float]:
    """Convert 3x3 rotation matrix to unit quaternion [w, x, y, z], w >= 0."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = [float(w), float(x), float(y), float(z)]
    if q[0] < 0:
        q = [-c for c in q]
    norm = np.sqrt(sum(c * c for c in q))
    return [c / norm for c in q]


def generate_rig_config(view_config: ViewConfig) -> list[dict]:
    """Build COLMAP rig JSON for camera-first (rig) folder layout."""
    views = view_config.get_all_views()
    if not views:
        return [{"cameras": []}]

    ref = views[0]
    R_ref = create_rotation_matrix(ref.yaw, ref.pitch)

    cameras: List[dict] = []
    for i, view in enumerate(views):
        entry: dict = {"image_prefix": f"{view.name}/"}
        if i == 0:
            entry["ref_sensor"] = True
        else:
            R_i = create_rotation_matrix(view.yaw, view.pitch)
            R_rel = R_i @ R_ref.T
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_rel)
            entry["cam_from_rig_rotation"] = [qw, qx, qy, qz]
            entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
        cameras.append(entry)

    return [{"cameras": cameras}]


def write_rig_config(view_config: ViewConfig, output_path: str) -> str:
    """Write rig config JSON; returns absolute path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = generate_rig_config(view_config)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path.resolve())
