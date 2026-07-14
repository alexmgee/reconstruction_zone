"""Tests for ERP rig_config generation."""

import math

import numpy as np
import pytest

from prep360.core.reframer import FreeView, ViewConfig
from prep360.core.rig_config import generate_rig_config, rotation_matrix_to_quaternion


def _quat_norm(q):
    return math.sqrt(sum(c * c for c in q))


def _reference_rotation_matrix(yaw_deg, pitch_deg):
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    fwd = np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw),
    ])
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right_len = np.linalg.norm(right)
    if right_len < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_len
    up = np.cross(right, fwd)
    return np.array([right, up, -fwd])


def _assert_same_quaternion(got, expected):
    same = all(abs(a - b) < 1e-5 for a, b in zip(got, expected))
    negated = all(abs(a + b) < 1e-5 for a, b in zip(got, expected))
    assert same or negated


class TestRigConfig:
    def test_ref_sensor_first(self):
        cfg = ViewConfig(
            views=[FreeView("00_00", 0, 0, 90), FreeView("00_01", 90, 0, 90)],
            include_zenith=False,
            include_nadir=False,
        )
        rig = generate_rig_config(cfg)
        cams = rig[0]["cameras"]
        assert cams[0]["ref_sensor"] is True
        assert cams[0]["image_prefix"] == "00_00/"
        assert "cam_from_rig_rotation" not in cams[0]

    def test_quaternion_unit(self):
        cfg = ViewConfig(
            views=[FreeView("00_00", 0, 0, 90), FreeView("00_01", 45, 0, 90)],
            include_zenith=False,
            include_nadir=False,
        )
        rig = generate_rig_config(cfg)
        q = rig[0]["cameras"][1]["cam_from_rig_rotation"]
        assert abs(_quat_norm(q) - 1.0) < 1e-6

    def test_relative_yaw_90(self):
        cfg = ViewConfig(
            views=[FreeView("00_00", 0, 0, 90), FreeView("00_01", 90, 0, 90)],
            include_zenith=False,
            include_nadir=False,
        )
        rig = generate_rig_config(cfg)
        R_ref = _reference_rotation_matrix(0, 0)
        R_i = _reference_rotation_matrix(90, 0)
        R_rel = R_i @ R_ref.T
        q = rig[0]["cameras"][1]["cam_from_rig_rotation"]
        q_back = rotation_matrix_to_quaternion(R_rel)
        _assert_same_quaternion(q, q_back)

    def test_relative_rotation_with_nonzero_reference(self):
        cfg = ViewConfig(
            views=[FreeView("ref", 45, -20, 90), FreeView("next", 120, 10, 90)],
            include_zenith=False,
            include_nadir=False,
        )
        rig = generate_rig_config(cfg)
        R_ref = _reference_rotation_matrix(45, -20)
        R_i = _reference_rotation_matrix(120, 10)
        expected = rotation_matrix_to_quaternion(R_i @ R_ref.T)
        got = rig[0]["cameras"][1]["cam_from_rig_rotation"]
        _assert_same_quaternion(got, expected)

    def test_image_prefix_matches_view_names(self):
        cfg = ViewConfig(
            views=[FreeView("00_00", 0, -35, 90), FreeView("01_00", 22.5, 35, 90)],
            include_zenith=False,
            include_nadir=False,
        )
        prefixes = [c["image_prefix"] for c in generate_rig_config(cfg)[0]["cameras"]]
        assert prefixes == ["00_00/", "01_00/"]
