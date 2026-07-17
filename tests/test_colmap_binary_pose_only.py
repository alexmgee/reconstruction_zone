"""Tests for pose-only COLMAP binary reading."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from prep360.core.colmap_binary import (
    COLMAP_CAMERA_MODELS,
    read_colmap_pose_model_binary,
)
from prep360.core.colmap_types import ColmapModelCompleteness, ColmapReadMode


def _write_cameras_bin(
    path: Path,
    *,
    camera_id: int = 1,
    model_id: int = 1,
    width: int = 1920,
    height: int = 1080,
    params: tuple[float, ...] | None = None,
) -> None:
    model_name, num_params = COLMAP_CAMERA_MODELS[model_id]
    if params is None:
        params = tuple(float(i + 100.0) for i in range(num_params))

    payload = struct.pack("<Q", 1)
    payload += struct.pack("<iiQQ", camera_id, model_id, width, height)
    payload += struct.pack(f"<{num_params}d", *params)
    path.write_bytes(payload)


def _write_images_bin(
    path: Path,
    records: list[dict],
) -> None:
    payload = struct.pack("<Q", len(records))
    for record in records:
        qvec = record["qvec"]
        tvec = record["tvec"]
        payload += struct.pack("<i", record["image_id"])
        payload += struct.pack("<4d", *qvec)
        payload += struct.pack("<3d", *tvec)
        payload += struct.pack("<i", record["camera_id"])
        payload += record["name"].encode("utf-8") + b"\x00"

        observations = record.get("observations", [])
        payload += struct.pack("<Q", len(observations))
        for x, y, point3d_id in observations:
            payload += struct.pack("<ddq", x, y, point3d_id)

    path.write_bytes(payload)


def _sample_pose_records() -> list[dict]:
    return [
        {
            "image_id": 101,
            "camera_id": 1,
            "name": "frame_00001_t12.5s.jpg",
            "qvec": (0.9238795325112867, 0.0, 0.3826834323650898, 0.0),
            "tvec": (1.0, 2.0, 3.0),
            "observations": [],
        },
        {
            "image_id": 205,
            "camera_id": 1,
            "name": "frame_00002.jpg",
            "qvec": (0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
            "tvec": (-0.5, 1.25, 4.0),
            "observations": [(10.0, 20.0, 9001), (30.0, 40.0, -1)],
        },
        {
            "image_id": 307,
            "camera_id": 1,
            "name": "frame_00003.jpg",
            "qvec": (0.8660254037844386, 0.1, 0.2, 0.3),
            "tvec": (0.25, -1.5, 2.75),
            "observations": [(5.5, 6.5, 42)],
        },
    ]


def test_read_cameras_bin_pose_only_fields(tmp_path: Path) -> None:
    params = (800.0, 900.0, 960.0, 540.0)
    _write_cameras_bin(
        tmp_path / "cameras.bin",
        camera_id=7,
        model_id=1,
        width=1920,
        height=1080,
        params=params,
    )
    _write_images_bin(tmp_path / "images.bin", [])

    model = read_colmap_pose_model_binary(tmp_path, variant="colmap")
    camera = model.cameras[7]

    assert camera.camera_id == 7
    assert camera.model_name == "PINHOLE"
    assert camera.width == 1920
    assert camera.height == 1080
    assert camera.params == params


def test_read_images_bin_pose_only_centers(tmp_path: Path) -> None:
    _write_cameras_bin(tmp_path / "cameras.bin")
    records = [
        {
            "image_id": 1,
            "camera_id": 1,
            "name": "a.jpg",
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (0.0, 0.0, 5.0),
            "observations": [],
        },
    ]
    _write_images_bin(tmp_path / "images.bin", records)

    model = read_colmap_pose_model_binary(tmp_path, variant="colmap")
    image = model.images[1]

    np.testing.assert_allclose(image.qvec, np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(image.tvec, np.array([0.0, 0.0, 5.0]))
    np.testing.assert_allclose(image.center, np.array([0.0, 0.0, -5.0]))


def test_pose_only_traverses_mixed_observation_counts(tmp_path: Path) -> None:
    records = _sample_pose_records()
    _write_cameras_bin(tmp_path / "cameras.bin")
    _write_images_bin(tmp_path / "images.bin", records)

    model = read_colmap_pose_model_binary(tmp_path, variant="colmap")

    assert set(model.images) == {101, 205, 307}
    assert model.images[101].name == "frame_00001_t12.5s.jpg"
    assert model.images[205].name == "frame_00002.jpg"
    assert model.images[307].name == "frame_00003.jpg"
    assert model.images[101].camera_id == 1
    assert model.images[205].camera_id == 1
    assert model.images[307].camera_id == 1

    np.testing.assert_allclose(
        model.images[205].qvec,
        np.array([0.7071067811865476, 0.0, 0.0, 0.7071067811865476]),
    )
    np.testing.assert_allclose(
        model.images[307].tvec,
        np.array([0.25, -1.5, 2.75]),
    )


def test_pose_only_does_not_retain_observations(tmp_path: Path) -> None:
    records = _sample_pose_records()
    _write_cameras_bin(tmp_path / "cameras.bin")
    _write_images_bin(tmp_path / "images.bin", records)

    model = read_colmap_pose_model_binary(tmp_path, variant="colmap")

    assert model.read_mode is ColmapReadMode.POSE_ONLY
    assert model.completeness is ColmapModelCompleteness.POSE_ONLY
    assert not hasattr(model.images[101], "points2d")
    assert all(
        not hasattr(image, "observations")
        for image in model.images.values()
    )


def test_missing_cameras_or_images_bin_has_clear_error(tmp_path: Path) -> None:
    _write_cameras_bin(tmp_path / "cameras.bin")

    with pytest.raises(FileNotFoundError, match="images.bin"):
        read_colmap_pose_model_binary(tmp_path, variant="colmap")

    images_only = tmp_path / "images_only"
    images_only.mkdir()
    _write_images_bin(images_only / "images.bin", _sample_pose_records())

    with pytest.raises(FileNotFoundError, match="cameras.bin"):
        read_colmap_pose_model_binary(images_only, variant="colmap")


def test_unknown_camera_model_id_has_clear_error(tmp_path: Path) -> None:
    payload = struct.pack("<Q", 1)
    payload += struct.pack("<iiQQ", 1, 99, 640, 480)
    (tmp_path / "cameras.bin").write_bytes(payload)
    _write_images_bin(tmp_path / "images.bin", [])

    with pytest.raises(ValueError, match="Unknown COLMAP camera model id: 99"):
        read_colmap_pose_model_binary(tmp_path, variant="colmap")


def test_truncated_images_bin_has_clear_error(tmp_path: Path) -> None:
    _write_cameras_bin(tmp_path / "cameras.bin")

    payload = struct.pack("<Q", 1)
    payload += struct.pack("<i", 1)
    payload += struct.pack("<4d", 1.0, 0.0, 0.0, 0.0)
    payload += struct.pack("<3d", 0.0, 0.0, 1.0)
    payload += struct.pack("<i", 1)
    payload += b"truncated.jpg\x00"
    payload += struct.pack("<Q", 1)
    (tmp_path / "images.bin").write_bytes(payload)

    with pytest.raises(ValueError, match="Truncated images.bin"):
        read_colmap_pose_model_binary(tmp_path, variant="colmap")


def test_pycolmap_absent_not_required() -> None:
    import importlib
    import sys

    module_names = [
        "prep360.core.colmap_binary",
        "prep360.core.colmap_types",
    ]
    for name in module_names:
        if name in sys.modules:
            del sys.modules[name]

    colmap_binary = importlib.import_module("prep360.core.colmap_binary")
    assert "pycolmap" not in colmap_binary.__dict__
