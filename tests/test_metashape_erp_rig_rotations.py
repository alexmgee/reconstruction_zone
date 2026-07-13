"""Tests for Metashape ERP rig slave-offset rotation math."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from prep360.core.reframer import VIEW_PRESETS
from prep360.core.rig_config import generate_rig_config

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "metashape_apply_erp_rig.py"


def _load_rig_module():
    spec = importlib.util.spec_from_file_location("metashape_erp_rig", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _quat_to_rotation_matrix(q: list[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@pytest.fixture(scope="module")
def rig():
    return _load_rig_module()


def _ref_view():
    return {"yaw": 0.0, "pitch": 0, "flip_vertical": False}


class TestMetashapeSlaveToMaster:
    def test_right_face_sees_master_forward_on_left(self, rig):
        """+90° cubemap face: master forward (+Z) projects on slave -X (left)."""
        R_sm = rig._metashape_slave_to_master(
            _ref_view(),
            {"yaw": 90.0, "pitch": 0, "flip_vertical": False},
        )
        master_forward = np.array([0.0, 0.0, 1.0])
        slave_dir = R_sm.T @ master_forward
        np.testing.assert_allclose(slave_dir, np.array([-1.0, 0.0, 0.0]), atol=1e-10)

    def test_inverse_of_colmap_cam_from_rig(self, rig):
        ref = _ref_view()
        slave = {"yaw": 90.0, "pitch": 0, "flip_vertical": False}
        R_colmap = rig._colmap_cam_from_rig(ref, slave)
        R_metashape = rig._metashape_slave_to_master(ref, slave)
        np.testing.assert_allclose(R_metashape @ R_colmap, np.eye(3), atol=1e-10)

    def test_cubemap_matches_prep360_rig_config(self, rig):
        """Every Metashape slave matrix is inverse of prep360 rig_config quaternion."""
        config = VIEW_PRESETS["cubemap"]
        rig_json = generate_rig_config(config)
        cams = {c["image_prefix"].rstrip("/"): c for c in rig_json[0]["cameras"]}
        views = {v.name: v for v in config.get_all_views()}
        ref = views["00_00"]

        ref_view = {"yaw": ref.yaw, "pitch": ref.pitch, "flip_vertical": ref.flip_vertical}
        for name, cam in cams.items():
            if cam.get("ref_sensor"):
                continue
            view = views[name]
            slave_view = {
                "yaw": view.yaw,
                "pitch": view.pitch,
                "flip_vertical": view.flip_vertical,
            }
            R_colmap = _quat_to_rotation_matrix(cam["cam_from_rig_rotation"])
            R_metashape = rig._metashape_slave_to_master(ref_view, slave_view)
            np.testing.assert_allclose(R_metashape @ R_colmap, np.eye(3), atol=1e-10)

    def test_flip_vertical_does_not_change_rig_rotation(self, rig):
        """flip_vertical is for JPEG flipud only — not rig relative rotation."""
        base = {"yaw": 0.0, "pitch": -90, "flip_vertical": False}
        flipped = {"yaw": 0.0, "pitch": -90, "flip_vertical": True}
        R_base = rig._metashape_slave_to_master(_ref_view(), base)
        R_flip = rig._metashape_slave_to_master(_ref_view(), flipped)
        np.testing.assert_allclose(R_base, R_flip, atol=1e-10)
