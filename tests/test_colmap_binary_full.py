"""Tests for FULL-mode COLMAP binary reading."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from prep360.core.colmap_binary import (
    COLMAP_CAMERA_MODELS,
    COLMAP_INVALID_POINT3D_ID_BINARY,
    normalize_point3d_id,
    read_colmap_full_model_binary,
    read_colmap_pose_model_binary,
)
from prep360.core.colmap_types import ColmapReadMode
from reconstruction_gui.colmap_validation import (
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
)


def _encode_point3d_id_for_binary(point3d_id: int) -> int:
    if point3d_id < 0:
        return COLMAP_INVALID_POINT3D_ID_BINARY
    return point3d_id


def _write_cameras_bin(path: Path, cameras: list[dict]) -> None:
    payload = struct.pack("<Q", len(cameras))
    model_name_to_id = {name: model_id for model_id, (name, _n) in COLMAP_CAMERA_MODELS.items()}
    for camera in cameras:
        model_id = model_name_to_id[camera["model_name"]]
        params = camera["params"]
        payload += struct.pack(
            "<iiQQ",
            camera["camera_id"],
            model_id,
            camera["width"],
            camera["height"],
        )
        payload += struct.pack(f"<{len(params)}d", *params)
    path.write_bytes(payload)


def _write_images_bin(path: Path, images: list[dict]) -> None:
    payload = struct.pack("<Q", len(images))
    for image in images:
        qvec = image["qvec"]
        tvec = image["tvec"]
        payload += struct.pack("<i", image["image_id"])
        payload += struct.pack("<4d", *qvec)
        payload += struct.pack("<3d", *tvec)
        payload += struct.pack("<i", image["camera_id"])
        payload += image["name"].encode("utf-8") + b"\x00"
        observations = image.get("observations", [])
        payload += struct.pack("<Q", len(observations))
        for x, y, point3d_id in observations:
            payload += struct.pack(
                "<ddQ",
                x,
                y,
                _encode_point3d_id_for_binary(point3d_id),
            )
    path.write_bytes(payload)


def _write_points3d_bin(path: Path, points: list[dict]) -> None:
    payload = struct.pack("<Q", len(points))
    for point in points:
        x, y, z = point["xyz"]
        r, g, b = point["rgb"]
        payload += struct.pack("<Q", point["point3d_id"])
        payload += struct.pack("<3d", x, y, z)
        payload += struct.pack("<BBB", r, g, b)
        payload += struct.pack("<d", point["error"])
        track = point["track"]
        payload += struct.pack("<Q", len(track))
        for image_id, point2d_idx in track:
            payload += struct.pack("<ii", image_id, point2d_idx)
    path.write_bytes(payload)


def _write_cameras_txt(path: Path, cameras: list[dict]) -> None:
    lines = ["# Camera list with one line of data per camera"]
    for camera in cameras:
        params = " ".join(str(value) for value in camera["params"])
        lines.append(
            f"{camera['camera_id']} {camera['model_name']} "
            f"{camera['width']} {camera['height']} {params}",
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_images_txt(path: Path, images: list[dict]) -> None:
    lines = ["# Image list with two lines of data per image"]
    for image in images:
        qvec = image["qvec"]
        tvec = image["tvec"]
        lines.append(
            f"{image['image_id']} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
            f"{tvec[0]} {tvec[1]} {tvec[2]} {image['camera_id']} {image['name']}",
        )
        obs_parts = []
        for x, y, point3d_id in image.get("observations", []):
            obs_parts.extend([str(x), str(y), str(point3d_id)])
        lines.append(" ".join(obs_parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_points3d_txt(path: Path, points: list[dict]) -> None:
    lines = ["# 3D point list with one line of data per point"]
    for point in points:
        x, y, z = point["xyz"]
        r, g, b = point["rgb"]
        track_parts = []
        for image_id, point2d_idx in point["track"]:
            track_parts.extend([str(image_id), str(point2d_idx)])
        track_text = " ".join(track_parts)
        lines.append(
            f"{point['point3d_id']} {x} {y} {z} {r} {g} {b} {point['error']} {track_text}",
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _full_fixture_data() -> dict:
    return {
        "cameras": [
            {
                "camera_id": 10,
                "model_name": "PINHOLE",
                "width": 1920,
                "height": 1080,
                "params": (800.0, 801.0, 960.0, 540.0),
            },
            {
                "camera_id": 42,
                "model_name": "PINHOLE",
                "width": 1280,
                "height": 720,
                "params": (600.0, 601.0, 640.0, 360.0),
            },
        ],
        "images": [
            {
                "image_id": 205,
                "qvec": (0.9238795325112867, 0.0, 0.3826834323650898, 0.0),
                "tvec": (1.0, 2.0, 3.0),
                "camera_id": 42,
                "name": "cafe_frame_205.jpg",
                "observations": [
                    (111.0, 222.0, 9007),
                    (333.0, 444.0, -1),
                    (555.0, 666.0, 9001),
                ],
            },
            {
                "image_id": 307,
                "qvec": (0.7071067811865476, 0.0, 0.7071067811865476, 0.0),
                "tvec": (4.0, 5.0, 6.0),
                "camera_id": 42,
                "name": "frame_307.jpg",
                "observations": [
                    (777.0, 888.0, 9001),
                ],
            },
            {
                "image_id": 101,
                "qvec": (1.0, 0.0, 0.0, 0.0),
                "tvec": (0.0, 0.0, 0.0),
                "camera_id": 10,
                "name": "frame_zero_obs.jpg",
                "observations": [],
            },
        ],
        "points": [
            {
                "point3d_id": 9001,
                "xyz": (1.5, 2.5, 3.5),
                "rgb": (12, 34, 56),
                "error": 0.75,
                "track": [(205, 2), (307, 0)],
            },
            {
                "point3d_id": 9007,
                "xyz": (4.5, 5.5, 6.5),
                "rgb": (78, 90, 12),
                "error": 1.125,
                "track": [(205, 0)],
            },
        ],
    }


def _write_full_binary_fixture(model_dir: Path) -> None:
    data = _full_fixture_data()
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_cameras_bin(model_dir / "cameras.bin", data["cameras"])
    _write_images_bin(model_dir / "images.bin", data["images"])
    _write_points3d_bin(model_dir / "points3D.bin", data["points"])


def _write_full_text_fixture(model_dir: Path) -> None:
    data = _full_fixture_data()
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_cameras_txt(model_dir / "cameras.txt", data["cameras"])
    _write_images_txt(model_dir / "images.txt", data["images"])
    _write_points3d_txt(model_dir / "points3D.txt", data["points"])


def _assert_fixture_track_observation_consistency(data: dict) -> None:
    images_by_id = {image["image_id"]: image for image in data["images"]}
    for point in data["points"]:
        point3d_id = point["point3d_id"]
        for image_id, point2d_idx in point["track"]:
            image = images_by_id[image_id]
            observations = image.get("observations", [])
            assert point2d_idx < len(observations), (
                f"track ({image_id}, {point2d_idx}) for point {point3d_id} "
                f"references missing observation index"
            )
            observed_point3d_id = observations[point2d_idx][2]
            assert observed_point3d_id == point3d_id, (
                f"track ({image_id}, {point2d_idx}) for point {point3d_id} "
                f"does not match observation point3D_id {observed_point3d_id}"
            )


def test_full_fixture_track_observation_consistency() -> None:
    _assert_fixture_track_observation_consistency(_full_fixture_data())


def test_read_full_model_requires_points3d(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse"
    data = _full_fixture_data()
    model_dir.mkdir()
    _write_cameras_bin(model_dir / "cameras.bin", data["cameras"])
    _write_images_bin(model_dir / "images.bin", data["images"])

    with pytest.raises(FileNotFoundError, match="points3D.bin"):
        read_colmap_full_model_binary(model_dir)


def test_read_full_images_preserves_points2d(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse"
    _write_full_binary_fixture(model_dir)

    model = read_colmap_full_model_binary(model_dir)

    assert len(model.images[101].points2d) == 0
    assert len(model.images[205].points2d) == 3
    assert model.images[205].points2d[0].point3d_id == 9007
    assert model.images[205].points2d[1].point3d_id == -1
    assert model.images[205].points2d[2].point3d_id == 9001


def test_pose_only_still_does_not_retain_points2d(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse"
    _write_full_binary_fixture(model_dir)

    model = read_colmap_pose_model_binary(model_dir)

    assert model.read_mode is ColmapReadMode.POSE_ONLY
    assert not hasattr(model.images[205], "points2d")


def test_points3d_preserves_error_color_and_track(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse"
    _write_full_binary_fixture(model_dir)

    model = read_colmap_full_model_binary(model_dir)
    point = model.points3d[9001]

    assert point.rgb == (12, 34, 56)
    assert point.error == pytest.approx(0.75)
    assert [(element.image_id, element.point2d_idx) for element in point.track] == [
        (205, 2),
        (307, 0),
    ]


def test_point3d_id_sentinel_normalized_to_minus_one() -> None:
    assert normalize_point3d_id(COLMAP_INVALID_POINT3D_ID_BINARY) == -1


def test_non_contiguous_ids_are_dict_keys(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse"
    _write_full_binary_fixture(model_dir)

    model = read_colmap_full_model_binary(model_dir)

    assert set(model.cameras) == {10, 42}
    assert set(model.images) == {101, 205, 307}
    assert set(model.points3d) == {9001, 9007}
    assert model.points3d[9001].track[0].image_id == 205
    assert model.images[205].points2d[0].point3d_id == 9007


def test_text_and_binary_full_semantics_match(tmp_path: Path) -> None:
    binary_dir = tmp_path / "binary"
    text_dir = tmp_path / "text"
    _write_full_binary_fixture(binary_dir)
    _write_full_text_fixture(text_dir)

    binary_model = read_colmap_full_model_binary(binary_dir)
    text_cameras = parse_cameras_txt(text_dir / "cameras.txt")
    text_images = parse_images_txt(text_dir / "images.txt")
    text_points = parse_points3d_txt(text_dir / "points3D.txt")

    assert set(binary_model.cameras) == set(text_cameras)
    for camera_id, text_camera in text_cameras.items():
        binary_camera = binary_model.cameras[camera_id]
        assert binary_camera.model_name == text_camera.model
        assert binary_camera.width == text_camera.width
        assert binary_camera.height == text_camera.height
        assert list(binary_camera.params) == text_camera.params

    assert set(binary_model.images) == set(text_images)
    for image_id, text_image in text_images.items():
        binary_image = binary_model.images[image_id]
        assert binary_image.name == text_image.name
        assert binary_image.camera_id == text_image.camera_id
        np.testing.assert_allclose(
            binary_image.qvec,
            np.array([text_image.qw, text_image.qx, text_image.qy, text_image.qz]),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            binary_image.tvec,
            np.array([text_image.tx, text_image.ty, text_image.tz]),
            atol=1e-9,
        )
        assert len(binary_image.points2d) == len(text_image.points2d)
        for binary_obs, text_obs in zip(binary_image.points2d, text_image.points2d):
            assert binary_obs.x == pytest.approx(text_obs[0])
            assert binary_obs.y == pytest.approx(text_obs[1])
            assert binary_obs.point3d_id == text_obs[2]

    assert set(binary_model.points3d) == set(text_points)
    for point_id, text_point in text_points.items():
        binary_point = binary_model.points3d[point_id]
        np.testing.assert_allclose(binary_point.xyz, text_point.xyz, atol=1e-9)
        assert binary_point.rgb == tuple(int(v) for v in text_point.rgb)
        assert binary_point.error == pytest.approx(text_point.error)
        assert [
            (element.image_id, element.point2d_idx)
            for element in binary_point.track
        ] == text_point.track


def test_pycolmap_semantic_parity_if_available(tmp_path: Path) -> None:
    pycolmap = pytest.importorskip("pycolmap")

    model_dir = tmp_path / "sparse"
    _write_full_binary_fixture(model_dir)

    native = read_colmap_full_model_binary(model_dir)
    reconstruction = pycolmap.Reconstruction(str(model_dir))

    assert set(native.cameras) == set(reconstruction.cameras)
    assert set(native.images) == set(reconstruction.images)
    assert set(native.points3d) == set(reconstruction.points3D)

    for camera_id, native_camera in native.cameras.items():
        cam = reconstruction.cameras[camera_id]
        model_name = getattr(cam, "model_name", None)
        if model_name is None:
            model_name = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
        assert native_camera.model_name == model_name
        assert native_camera.width == cam.width
        assert native_camera.height == cam.height
        np.testing.assert_allclose(
            list(native_camera.params),
            [float(p) for p in cam.params],
            atol=1e-6,
        )

    for image_id, native_image in native.images.items():
        img = reconstruction.images[image_id]
        cfw = img.cam_from_world()
        quat = cfw.rotation.quat
        tvec = cfw.translation
        np.testing.assert_allclose(
            native_image.qvec,
            np.array([quat[3], quat[0], quat[1], quat[2]]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            native_image.tvec,
            np.array([tvec[0], tvec[1], tvec[2]]),
            atol=1e-6,
        )
        assert native_image.name == img.name
        assert len(native_image.points2d) == len(img.points2D)
        for native_obs, pt2d in zip(native_image.points2d, img.points2D):
            assert native_obs.point3d_id == normalize_point3d_id(int(pt2d.point3D_id))
            assert native_obs.x == pytest.approx(float(pt2d.xy[0]))
            assert native_obs.y == pytest.approx(float(pt2d.xy[1]))

    for point_id, native_point in native.points3d.items():
        pt = reconstruction.points3D[point_id]
        np.testing.assert_allclose(native_point.xyz, np.array(pt.xyz), atol=1e-6)
        assert native_point.rgb == tuple(int(v) for v in pt.color)
        assert native_point.error == pytest.approx(float(pt.error))
        assert [
            (element.image_id, element.point2d_idx)
            for element in native_point.track
        ] == [(elem.image_id, elem.point2D_idx) for elem in pt.track.elements]


def test_read_model_dir_pose_only_compatibility(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse"
    _write_full_binary_fixture(model_dir)

    pose_model = read_colmap_pose_model_binary(model_dir)

    assert pose_model.read_mode is ColmapReadMode.POSE_ONLY
    assert set(pose_model.images) == {101, 205, 307}
    assert pose_model.images[205].name == "cafe_frame_205.jpg"


def test_truncated_points3d_before_track_length_has_clear_error(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse"
    data = _full_fixture_data()
    model_dir.mkdir()
    _write_cameras_bin(model_dir / "cameras.bin", data["cameras"])
    _write_images_bin(model_dir / "images.bin", data["images"])

    payload = struct.pack("<Q", 1)
    payload += struct.pack("<Q", 9001)
    payload += struct.pack("<3d", 1.0, 2.0, 3.0)
    payload += struct.pack("<BBB", 1, 2, 3)
    payload += struct.pack("<d", 0.5)
    (model_dir / "points3D.bin").write_bytes(payload)

    with pytest.raises(ValueError, match="Truncated points3D.bin: incomplete point header"):
        read_colmap_full_model_binary(model_dir)


def test_truncated_points3d_has_clear_error(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse"
    data = _full_fixture_data()
    model_dir.mkdir()
    _write_cameras_bin(model_dir / "cameras.bin", data["cameras"])
    _write_images_bin(model_dir / "images.bin", data["images"])

    payload = struct.pack("<Q", 1)
    payload += struct.pack("<Q", 9001)
    payload += struct.pack("<3d", 1.0, 2.0, 3.0)
    payload += struct.pack("<BBB", 1, 2, 3)
    payload += struct.pack("<d", 0.5)
    payload += struct.pack("<Q", 1)
    (model_dir / "points3D.bin").write_bytes(payload)

    with pytest.raises(ValueError, match="Truncated points3D.bin"):
        read_colmap_full_model_binary(model_dir)
