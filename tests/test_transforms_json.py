"""Tests for transforms.json export from COLMAP binary pose models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prep360.cli import main as cli_main
from prep360.core.colmap_binary import qvec_to_rotmat
from prep360.core.colmap_types import (
    ColmapCamera,
    ColmapImagePose,
    ColmapModelCompleteness,
    ColmapPoseModel,
    ColmapReadMode,
)
from prep360.core.transforms_json import (
    colmap_pose_model_to_transforms,
    colmap_w2c_to_nerf_c2w,
    export_colmap_binary_to_transforms_json,
    write_transforms_json,
)
from tests.test_colmap_binary_pose_only import (
    _sample_pose_records,
    _write_cameras_bin,
    _write_images_bin,
)


def _camera_center(qvec: tuple[float, float, float, float], tvec: tuple[float, float, float]) -> np.ndarray:
    q = np.array(qvec, dtype=np.float64)
    t = np.array(tvec, dtype=np.float64)
    return -qvec_to_rotmat(q).T @ t


def _make_image_pose(
    *,
    image_id: int,
    name: str,
    qvec: tuple[float, float, float, float],
    tvec: tuple[float, float, float],
    camera_id: int = 1,
) -> ColmapImagePose:
    q = np.array(qvec, dtype=np.float64)
    t = np.array(tvec, dtype=np.float64)
    return ColmapImagePose(
        image_id=image_id,
        qvec=q,
        tvec=t,
        camera_id=camera_id,
        name=name,
        center=_camera_center(qvec, tvec),
    )


def _make_camera(
    *,
    camera_id: int = 1,
    model_name: str = "PINHOLE",
    width: int = 1920,
    height: int = 1080,
    params: tuple[float, ...] | None = None,
) -> ColmapCamera:
    if params is None:
        if model_name == "SIMPLE_PINHOLE":
            params = (1000.0, 960.0, 540.0)
        elif model_name == "PINHOLE":
            params = (1000.0, 1001.0, 960.0, 540.0)
        elif model_name == "SIMPLE_RADIAL":
            params = (1000.0, 960.0, 540.0, 0.01)
        elif model_name == "RADIAL":
            params = (1000.0, 960.0, 540.0, 0.01, -0.02)
        elif model_name == "OPENCV":
            params = (1000.0, 1001.0, 960.0, 540.0, 0.01, -0.02, 0.001, 0.002)
        elif model_name == "SPHERE":
            params = (1000.0, 960.0, 540.0)
        else:
            raise ValueError(f"No default params for {model_name}")

    return ColmapCamera(
        camera_id=camera_id,
        model_name=model_name,
        width=width,
        height=height,
        params=params,
    )


def _make_pose_model(
    *,
    cameras: dict[int, ColmapCamera],
    images: dict[int, ColmapImagePose],
) -> ColmapPoseModel:
    return ColmapPoseModel(
        cameras=cameras,
        images=images,
        source_dir=Path("."),
        read_mode=ColmapReadMode.POSE_ONLY,
        completeness=ColmapModelCompleteness.POSE_ONLY,
    )

def test_colmap_identity_to_nerf_axes() -> None:
    c2w = colmap_w2c_to_nerf_c2w(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    )
    expected = np.diag([1.0, -1.0, -1.0, 1.0])
    np.testing.assert_allclose(c2w, expected, atol=1e-9)


def test_colmap_translation_to_camera_center() -> None:
    c2w = colmap_w2c_to_nerf_c2w(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 5.0]),
    )
    np.testing.assert_allclose(c2w[:3, 3], np.array([0.0, 0.0, -5.0]), atol=1e-9)


def test_colmap_rotated_pose_expected_axes() -> None:
    c2w = colmap_w2c_to_nerf_c2w(
        np.array([0.70710678, 0.0, 0.0, 0.70710678]),
        np.array([0.0, 0.0, 0.0]),
    )
    expected_rotation = np.array([
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    np.testing.assert_allclose(c2w[:3, :3], expected_rotation, atol=1e-6)


def test_transform_translation_matches_a1_center() -> None:
    image = _make_image_pose(
        image_id=1,
        name="frame.jpg",
        qvec=(0.9238795325112867, 0.0, 0.3826834323650898, 0.0),
        tvec=(1.0, 2.0, 3.0),
    )
    c2w = colmap_w2c_to_nerf_c2w(image.qvec, image.tvec)
    np.testing.assert_allclose(c2w[:3, 3], image.center, atol=1e-9)


def test_rotation_block_is_orthonormal() -> None:
    image = _make_image_pose(
        image_id=1,
        name="frame.jpg",
        qvec=(0.70710678, 0.0, 0.0, 0.70710678),
        tvec=(1.0, 2.0, 3.0),
    )
    rotation = colmap_w2c_to_nerf_c2w(image.qvec, image.tvec)[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)


def test_write_transforms_json_root_intrinsics(tmp_path: Path) -> None:
    camera = _make_camera(model_name="PINHOLE")
    image = _make_image_pose(
        image_id=1,
        name="frame.jpg",
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
    )
    model = _make_pose_model(cameras={1: camera}, images={1: image})
    dataset = colmap_pose_model_to_transforms(model)

    output = tmp_path / "transforms.json"
    write_transforms_json(dataset, output)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["camera_model"] == "OPENCV"
    assert data["w"] == 1920
    assert data["h"] == 1080
    assert data["fl_x"] == 1000.0
    assert data["fl_y"] == 1001.0
    assert data["cx"] == 960.0
    assert data["cy"] == 540.0
    assert "fl_x" not in data["frames"][0]


def test_per_frame_intrinsics_for_multiple_cameras() -> None:
    cameras = {
        1: _make_camera(
            camera_id=1,
            model_name="PINHOLE",
            params=(1000.0, 1001.0, 960.0, 540.0),
        ),
        2: _make_camera(
            camera_id=2,
            model_name="PINHOLE",
            params=(1100.0, 1101.0, 970.0, 550.0),
        ),
    }
    images = {
        1: _make_image_pose(
            image_id=1,
            name="a.jpg",
            qvec=(1.0, 0.0, 0.0, 0.0),
            tvec=(0.0, 0.0, 0.0),
            camera_id=1,
        ),
        2: _make_image_pose(
            image_id=2,
            name="b.jpg",
            qvec=(1.0, 0.0, 0.0, 0.0),
            tvec=(1.0, 0.0, 0.0),
            camera_id=2,
        ),
    }
    dataset = colmap_pose_model_to_transforms(_make_pose_model(cameras=cameras, images=images))

    assert dataset.camera_model == "OPENCV"
    assert dataset.w is None
    assert dataset.fl_x is None
    assert dataset.frames[0].fl_x == 1000.0
    assert dataset.frames[1].fl_x == 1100.0
    assert dataset.frames[1].fl_y == 1101.0


@pytest.mark.parametrize(
    "model_name",
    ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"],
)
def test_all_supported_models_emit_opencv_camera_model(model_name: str, tmp_path: Path) -> None:
    camera = _make_camera(model_name=model_name)
    image = _make_image_pose(
        image_id=1,
        name="frame.jpg",
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
    )
    dataset = colmap_pose_model_to_transforms(
        _make_pose_model(cameras={1: camera}, images={1: image}),
    )
    output = tmp_path / f"{model_name}.json"
    write_transforms_json(dataset, output)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["camera_model"] == "OPENCV"


def test_colmap_parameter_ordering() -> None:
    camera = _make_camera(
        model_name="OPENCV",
        params=(1111.0, 2222.0, 3333.0, 4444.0, 0.11, -0.22, 0.33, -0.44),
    )
    image = _make_image_pose(
        image_id=1,
        name="frame.jpg",
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
    )
    dataset = colmap_pose_model_to_transforms(
        _make_pose_model(cameras={1: camera}, images={1: image}),
    )

    assert dataset.fl_x == 1111.0
    assert dataset.fl_y == 2222.0
    assert dataset.cx == 3333.0
    assert dataset.cy == 4444.0
    assert dataset.k1 == 0.11
    assert dataset.k2 == -0.22
    assert dataset.p1 == 0.33
    assert dataset.p2 == -0.44


def test_opencv_distortion_fields_export(tmp_path: Path) -> None:
    camera = _make_camera(
        model_name="OPENCV",
        params=(1000.0, 1001.0, 960.0, 540.0, 0.01, -0.02, 0.001, 0.002),
    )
    image = _make_image_pose(
        image_id=1,
        name="frame.jpg",
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
    )
    dataset = colmap_pose_model_to_transforms(
        _make_pose_model(cameras={1: camera}, images={1: image}),
    )
    write_transforms_json(dataset, tmp_path / "transforms.json")
    data = json.loads((tmp_path / "transforms.json").read_text(encoding="utf-8"))

    assert data["k1"] == 0.01
    assert data["k2"] == -0.02
    assert data["p1"] == 0.001
    assert data["p2"] == 0.002


def test_unsupported_sphere_camera_refuses() -> None:
    camera = _make_camera(model_name="SPHERE")
    image = _make_image_pose(
        image_id=1,
        name="frame.jpg",
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="Camera model SPHERE is not supported"):
        colmap_pose_model_to_transforms(
            _make_pose_model(cameras={1: camera}, images={1: image}),
        )


def test_missing_camera_reference_refuses() -> None:
    image = _make_image_pose(
        image_id=1,
        name="frame.jpg",
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
        camera_id=99,
    )
    with pytest.raises(ValueError, match="references missing camera_id 99"):
        colmap_pose_model_to_transforms(
            _make_pose_model(cameras={}, images={1: image}),
        )


def test_file_paths_are_relative_posix() -> None:
    camera = _make_camera()
    image = _make_image_pose(
        image_id=1,
        name=r"subdir\frame.jpg",
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
    )
    dataset = colmap_pose_model_to_transforms(
        _make_pose_model(cameras={1: camera}, images={1: image}),
        image_prefix=r"train\images",
    )
    assert dataset.frames[0].file_path == "train/images/subdir/frame.jpg"


def test_absolute_image_prefix_refuses() -> None:
    camera = _make_camera()
    image = _make_image_pose(
        image_id=1,
        name="frame.jpg",
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="image_prefix must be a relative path"):
        colmap_pose_model_to_transforms(
            _make_pose_model(cameras={1: camera}, images={1: image}),
            image_prefix=r"C:\images",
        )


def test_absolute_colmap_image_name_refuses() -> None:
    camera = _make_camera()
    image = _make_image_pose(
        image_id=1,
        name="/abs/frame.jpg",
        qvec=(1.0, 0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="COLMAP image name must be a relative path"):
        colmap_pose_model_to_transforms(
            _make_pose_model(cameras={1: camera}, images={1: image}),
        )


def test_transform_translation_matches_a1_binary_fixture(tmp_path: Path) -> None:
    from prep360.core.colmap_binary import read_colmap_pose_model_binary

    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    _write_cameras_bin(sparse_dir / "cameras.bin")
    _write_images_bin(sparse_dir / "images.bin", _sample_pose_records())

    model = read_colmap_pose_model_binary(sparse_dir, variant="colmap")
    dataset = colmap_pose_model_to_transforms(model)

    centers_by_id = {image_id: image.center for image_id, image in model.images.items()}
    sorted_ids = sorted(model.images)
    for frame, image_id in zip(dataset.frames, sorted_ids):
        translation = np.array(frame.transform_matrix)[:3, 3]
        np.testing.assert_allclose(translation, centers_by_id[image_id], atol=1e-9)


def test_export_colmap_binary_to_transforms_json(tmp_path: Path) -> None:
    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    _write_cameras_bin(sparse_dir / "cameras.bin")
    _write_images_bin(sparse_dir / "images.bin", _sample_pose_records())

    output = tmp_path / "transforms.json"
    frame_count = export_colmap_binary_to_transforms_json(
        sparse_dir,
        output,
        image_prefix="images",
    )

    assert frame_count == 3
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["camera_model"] == "OPENCV"
    assert len(data["frames"]) == 3
    assert data["frames"][0]["file_path"] == "images/frame_00001_t12.5s.jpg"


def test_cli_transforms_json_command(tmp_path: Path, capsys, monkeypatch) -> None:
    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    _write_cameras_bin(sparse_dir / "cameras.bin")
    _write_images_bin(sparse_dir / "images.bin", _sample_pose_records())
    output = tmp_path / "transforms.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prep360",
            "transforms-json",
            "--colmap-sparse",
            str(sparse_dir),
            "--output",
            str(output),
            "--force",
        ],
    )
    exit_code = cli_main()

    assert exit_code == 0
    assert output.is_file()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data["frames"]) == 3
    captured = capsys.readouterr()
    assert "Exported 3 frames" in captured.out


def test_cli_refuses_existing_output_without_force(tmp_path: Path, capsys, monkeypatch) -> None:
    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    _write_cameras_bin(sparse_dir / "cameras.bin")
    _write_images_bin(sparse_dir / "images.bin", _sample_pose_records())
    output = tmp_path / "transforms.json"
    output.write_text("existing", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "prep360",
            "transforms-json",
            "--colmap-sparse",
            str(sparse_dir),
            "--output",
            str(output),
        ],
    )
    exit_code = cli_main()

    assert exit_code == 1
    assert output.read_text(encoding="utf-8") == "existing"
    captured = capsys.readouterr()
    assert "Output already exists" in captured.err


def test_cli_text_colmap_directory_has_clear_error(tmp_path: Path, capsys, monkeypatch) -> None:
    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    (sparse_dir / "cameras.txt").write_text("# cameras\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("# images\n", encoding="utf-8")
    output = tmp_path / "transforms.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prep360",
            "transforms-json",
            "--colmap-sparse",
            str(sparse_dir),
            "--output",
            str(output),
        ],
    )
    exit_code = cli_main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "requires binary COLMAP" in captured.err


def test_json_output_is_deterministic(tmp_path: Path) -> None:
    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    _write_cameras_bin(sparse_dir / "cameras.bin")
    records = list(reversed(_sample_pose_records()))
    _write_images_bin(sparse_dir / "images.bin", records)
    output_a = tmp_path / "a.json"
    output_b = tmp_path / "b.json"

    export_colmap_binary_to_transforms_json(sparse_dir, output_a, overwrite=True)
    export_colmap_binary_to_transforms_json(sparse_dir, output_b, overwrite=True)

    assert output_a.read_text(encoding="utf-8") == output_b.read_text(encoding="utf-8")
    data = json.loads(output_a.read_text(encoding="utf-8"))
    assert [frame["file_path"] for frame in data["frames"]] == [
        "images/frame_00001_t12.5s.jpg",
        "images/frame_00002.jpg",
        "images/frame_00003.jpg",
    ]
