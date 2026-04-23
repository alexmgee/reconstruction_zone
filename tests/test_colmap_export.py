"""Tests for prep360 COLMAP export — quaternion math, writers, and focal length."""

import math
import numpy as np

from prep360.core.colmap_export import (
    rotation_to_quaternion,
    focal_length_to_35mm,
    write_colmap_cameras,
    write_colmap_images,
    write_colmap_points3d_empty,
    COLMAPCameraEntry,
    COLMAPImage,
)


class TestRotationToQuaternion:
    def test_identity_matrix(self):
        R = np.eye(3)
        qw, qx, qy, qz = rotation_to_quaternion(R)
        assert abs(qw - 1.0) < 1e-10
        assert abs(qx) < 1e-10
        assert abs(qy) < 1e-10
        assert abs(qz) < 1e-10

    def test_90deg_around_z(self):
        """90-degree rotation around Z axis."""
        R = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)
        qw, qx, qy, qz = rotation_to_quaternion(R)
        # Expected: qw=cos(45deg), qz=sin(45deg)
        expected_qw = math.cos(math.radians(45))
        expected_qz = math.sin(math.radians(45))
        assert abs(qw - expected_qw) < 1e-6
        assert abs(qx) < 1e-6
        assert abs(qy) < 1e-6
        assert abs(qz - expected_qz) < 1e-6

    def test_180deg_around_x(self):
        """180-degree rotation around X axis."""
        R = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1],
        ], dtype=float)
        qw, qx, qy, qz = rotation_to_quaternion(R)
        # qw should be ~0, qx should be ~1
        assert abs(qw) < 1e-6
        assert abs(abs(qx) - 1.0) < 1e-6

    def test_qw_nonnegative(self):
        """COLMAP convention: qw >= 0."""
        # A rotation that would naturally produce negative qw
        R = np.array([
            [-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0,  1],
        ], dtype=float)
        qw, _, _, _ = rotation_to_quaternion(R)
        assert qw >= 0

    def test_unit_quaternion(self):
        """Output should be a unit quaternion."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            # Random rotation via SVD
            A = rng.standard_normal((3, 3))
            U, _, Vt = np.linalg.svd(A)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R = -R
            qw, qx, qy, qz = rotation_to_quaternion(R)
            norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
            assert abs(norm - 1.0) < 1e-6


class TestFocalLengthTo35mm:
    def test_standard_conversion(self):
        # 1000px focal on a 2000px wide sensor → 36mm * 1000/2000 = 18mm
        assert abs(focal_length_to_35mm(1000.0, 2000) - 18.0) < 1e-6

    def test_full_frame_equiv(self):
        # If sensor is 36mm wide (full frame), f_px == f_35mm when
        # image_width matches the sensor's pixel pitch. For a 7680px wide image:
        # f_35mm = 2880 * 36 / 7680 = 13.5
        result = focal_length_to_35mm(2880.0, 7680)
        assert abs(result - 13.5) < 1e-6


class TestWriteCOLMAPCameras:
    def test_writes_valid_format(self, tmp_dir):
        cameras = [
            COLMAPCameraEntry(
                camera_id=1, model="PINHOLE", width=1920, height=1080,
                params=[960.0, 960.0, 960.0, 540.0],
            ),
        ]
        path = str(tmp_dir / "cameras.txt")
        write_colmap_cameras(cameras, path)

        with open(path) as f:
            lines = f.readlines()

        # 3 comment lines + 1 data line
        assert len(lines) == 4
        assert lines[0].startswith("#")
        data = lines[3].strip()
        parts = data.split()
        assert parts[0] == "1"         # camera_id
        assert parts[1] == "PINHOLE"   # model
        assert parts[2] == "1920"      # width
        assert parts[3] == "1080"      # height
        assert len(parts) == 8         # id, model, w, h, fx, fy, cx, cy

    def test_multiple_cameras_sorted(self, tmp_dir):
        cameras = [
            COLMAPCameraEntry(camera_id=3, model="PINHOLE", width=100, height=100, params=[50, 50, 50, 50]),
            COLMAPCameraEntry(camera_id=1, model="PINHOLE", width=200, height=200, params=[100, 100, 100, 100]),
        ]
        path = str(tmp_dir / "cameras.txt")
        write_colmap_cameras(cameras, path)

        with open(path) as f:
            lines = [l for l in f.readlines() if not l.startswith("#")]
        assert lines[0].startswith("1 ")
        assert lines[1].startswith("3 ")


class TestWriteCOLMAPImages:
    def test_writes_valid_format(self, tmp_dir):
        images = [
            COLMAPImage(
                image_id=1, qw=1.0, qx=0.0, qy=0.0, qz=0.0,
                tx=1.5, ty=2.5, tz=3.5, camera_id=1, name="frame_0001.jpg",
            ),
        ]
        path = str(tmp_dir / "images.txt")
        write_colmap_images(images, path)

        with open(path) as f:
            lines = f.readlines()

        # 3 comment lines + 1 data line + 1 empty line
        assert len(lines) == 5
        data = lines[3].strip()
        assert data.endswith("frame_0001.jpg")
        assert data.startswith("1 ")

    def test_empty_line_after_each_image(self, tmp_dir):
        images = [
            COLMAPImage(1, 1, 0, 0, 0, 0, 0, 0, 1, "a.jpg"),
            COLMAPImage(2, 1, 0, 0, 0, 0, 0, 0, 1, "b.jpg"),
        ]
        path = str(tmp_dir / "images.txt")
        write_colmap_images(images, path)

        with open(path) as f:
            lines = f.readlines()
        # 3 comments + (data + empty) * 2 = 7
        assert len(lines) == 7
        assert lines[4].strip() == ""  # empty after first image
        assert lines[6].strip() == ""  # empty after second image


class TestWritePoints3DEmpty:
    def test_creates_file_with_header(self, tmp_dir):
        path = str(tmp_dir / "points3D.txt")
        write_colmap_points3d_empty(path)

        with open(path) as f:
            content = f.read()
        assert "Number of points: 0" in content
