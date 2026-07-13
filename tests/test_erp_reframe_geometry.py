"""Geometry tests for ERP reframe_view (independent reference math in tests)."""

import numpy as np
import pytest

from prep360.core.reframer import create_rotation_matrix, reframe_view


def _reference_rotation_matrix(yaw_deg, pitch_deg):
    """Independent implementation of the audited ERP convention."""
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


def _reference_reframe_no_fliplr(equirect, fov_deg, yaw_deg, pitch_deg, out_size):
    """Independent reference without fliplr — documents convention delta."""
    import cv2
    h_eq, w_eq = equirect.shape[:2]
    fov_rad = np.radians(fov_deg)
    f = (out_size / 2) / np.tan(fov_rad / 2)
    cxx = cy = out_size / 2
    R = _reference_rotation_matrix(yaw_deg, pitch_deg)
    px = np.arange(out_size, dtype=np.float64)
    px_grid, py_grid = np.meshgrid(px, px)
    crx = (px_grid - cxx) / f
    cry = -(py_grid - cy) / f
    wx = R[0, 0] * crx + R[1, 0] * cry - R[2, 0]
    wy = R[0, 1] * crx + R[1, 1] * cry - R[2, 1]
    wz = R[0, 2] * crx + R[1, 2] * cry - R[2, 2]
    L = np.sqrt(wx * wx + wy * wy + wz * wz)
    wx, wy, wz = wx / L, wy / L, wz / L
    theta = np.arctan2(wx, wz)
    phi = np.arcsin(np.clip(wy, -1, 1))
    u_eq = ((theta / np.pi + 1) / 2) * w_eq
    v_eq = (0.5 - phi / np.pi) * h_eq
    map_x = u_eq.astype(np.float32) % w_eq
    map_y = np.clip(v_eq.astype(np.float32), 0, h_eq - 1)
    return cv2.remap(equirect, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _checkerboard_erp(w=256, h=128, tile=16):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if ((x // tile) + (y // tile)) % 2 == 0:
                img[y, x] = (200, 200, 200)
            else:
                img[y, x] = (40, 40, 40)
    return img


# Audited fixture: mean B-channel of center 8x8 after reframe yaw=0 pitch=0 FOV=90 out=64
GOLDEN_CENTER_MEAN_B = 118.6
GOLDEN_TOLERANCE = 8.0


class TestErpReframeGeometry:
    def test_center_mean_matches_golden(self):
        erp = _checkerboard_erp()
        out = reframe_view(erp, 90, 0, 0, 64)
        cy, cx = 32, 32
        patch = out[cy - 4:cy + 4, cx - 4:cx + 4, 0]
        mean_b = float(patch.mean())
        assert abs(mean_b - GOLDEN_CENTER_MEAN_B) < GOLDEN_TOLERANCE

    def test_fliplr_changes_horizontal_mean(self):
        erp = _checkerboard_erp()
        with_flip = reframe_view(erp, 90, 0, 0, 64)
        no_flip = _reference_reframe_no_fliplr(erp, 90, 0, 0, 64)
        left_flip = float(with_flip[:, :8, 0].mean())
        left_noflip = float(no_flip[:, :8, 0].mean())
        right_flip = float(with_flip[:, -8:, 0].mean())
        right_noflip = float(no_flip[:, -8:, 0].mean())
        assert abs(left_flip - right_noflip) < 5
        assert abs(right_flip - left_noflip) < 5

    def test_grayscale_mask_shape(self):
        erp = _checkerboard_erp()
        mask = np.ones((128, 256), dtype=np.uint8) * 255
        out = reframe_view(mask, 90, 0, 0, 32, mode="nearest")
        assert out.ndim == 2
        assert out.shape == (32, 32)

    def test_create_rotation_matrix_forward_points_plus_z(self):
        R = create_rotation_matrix(0, 0)
        fwd = np.array([0.0, 0.0, 1.0])
        cam_z = R @ fwd
        assert cam_z[2] < 0  # -forward row
