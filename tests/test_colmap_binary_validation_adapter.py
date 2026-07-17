"""Tests for the COLMAP FULL binary validation adapter (Packet A2b)."""

from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from prep360.core.colmap_binary import (
    COLMAP_INVALID_POINT3D_ID_BINARY,
    read_colmap_full_model_binary,
    read_colmap_pose_model_binary,
)
from prep360.core.colmap_types import (
    ColmapCamera,
    ColmapFullModel,
    ColmapImageRecord,
    ColmapModelCompleteness,
    ColmapPoint2D,
    ColmapPoint3DRecord,
    ColmapReadMode,
    ColmapTrackElement,
)
from reconstruction_gui.colmap_binary_validation_adapter import (
    FULL_MODEL_REQUIRED_MSG,
    colmap_full_model_to_validation_dict,
    read_colmap_binary_full_for_validation,
)
from reconstruction_gui.colmap_validation import (
    COLMAPImage,
    COLMAPPoint3D,
    GeometricValidator,
    ValidationConfig,
    from_pycolmap_reconstruction,
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
)
from tests.test_colmap_binary_full import (
    _full_fixture_data,
    _write_cameras_bin,
    _write_cameras_txt,
    _write_full_binary_fixture,
    _write_full_text_fixture,
    _write_images_bin,
    _write_images_txt,
    _write_points3d_bin,
    _write_points3d_txt,
)


def _validation_consumer_fixture_data() -> dict:
    """Tiny deterministic model: one point observed in three views at image center."""
    return {
        "cameras": [
            {
                "camera_id": 1,
                "model_name": "SIMPLE_PINHOLE",
                "width": 640,
                "height": 480,
                "params": (500.0, 320.0, 240.0),
            },
        ],
        "images": [
            {
                "image_id": 1,
                "qvec": (1.0, 0.0, 0.0, 0.0),
                "tvec": (0.0, 0.0, 0.0),
                "camera_id": 1,
                "name": "view_001.jpg",
                "observations": [(320.0, 240.0, 100)],
            },
            {
                "image_id": 2,
                "qvec": (1.0, 0.0, 0.0, 0.0),
                "tvec": (0.0, 0.0, 0.0),
                "camera_id": 1,
                "name": "view_002.jpg",
                "observations": [(320.0, 240.0, 100)],
            },
            {
                "image_id": 3,
                "qvec": (1.0, 0.0, 0.0, 0.0),
                "tvec": (0.0, 0.0, 0.0),
                "camera_id": 1,
                "name": "view_003.jpg",
                "observations": [(320.0, 240.0, 100)],
            },
        ],
        "points": [
            {
                "point3d_id": 100,
                "xyz": (0.0, 0.0, 10.0),
                "rgb": (255, 128, 64),
                "error": 1.0,
                "track": [(1, 0), (2, 0), (3, 0)],
            },
        ],
    }


def _write_consumer_binary_fixture(model_dir: Path) -> None:
    data = _validation_consumer_fixture_data()
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_cameras_bin(model_dir / "cameras.bin", data["cameras"])
    _write_images_bin(model_dir / "images.bin", data["images"])
    _write_points3d_bin(model_dir / "points3D.bin", data["points"])


def _write_consumer_text_fixture(model_dir: Path) -> None:
    data = _validation_consumer_fixture_data()
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_cameras_txt(model_dir / "cameras.txt", data["cameras"])
    _write_images_txt(model_dir / "images.txt", data["images"])
    _write_points3d_txt(model_dir / "points3D.txt", data["points"])


def _write_consumer_masks(masks_dir: Path) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask = np.full((480, 640), 255, dtype=np.uint8)
    for name in ("view_001.jpg", "view_002.jpg", "view_003.jpg"):
        cv2.imwrite(str(masks_dir / f"{Path(name).stem}.png"), mask)


def _explicit_validation_config(colmap_dir: str) -> ValidationConfig:
    return ValidationConfig(
        colmap_dir=colmap_dir,
        agreement_threshold=0.7,
        min_track_length=3,
        max_reprojection_error=4.0,
        max_points=0,
        inconsistency_threshold=0.15,
    )


def _assert_reports_match(report_a, report_b) -> None:
    assert report_a.total_points_checked == report_b.total_points_checked
    assert report_a.summary() == report_b.summary()
    assert set(report_a.frames) == set(report_b.frames)
    for name, frame_a in report_a.frames.items():
        frame_b = report_b.frames[name]
        assert frame_a.total_points == frame_b.total_points
        assert frame_a.consistent_points == frame_b.consistent_points
        assert frame_a.inconsistent_points == frame_b.inconsistent_points
        assert frame_a.false_negative_points == frame_b.false_negative_points
        assert frame_a.false_positive_points == frame_b.false_positive_points
        assert frame_a.missing_mask == frame_b.missing_mask


def _make_inconsistent_full_model() -> ColmapFullModel:
    camera = ColmapCamera(
        camera_id=1,
        model_name="SIMPLE_PINHOLE",
        width=640,
        height=480,
        params=(500.0, 320.0, 240.0),
    )
    image = ColmapImageRecord(
        image_id=205,
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        camera_id=1,
        name="test.jpg",
        center=np.zeros(3),
        points2d=(ColmapPoint2D(320.0, 240.0, 9007),),
    )
    point = ColmapPoint3DRecord(
        point3d_id=9001,
        xyz=np.array([0.0, 0.0, 10.0]),
        rgb=(1, 2, 3),
        error=1.0,
        track=(ColmapTrackElement(205, 0),),
    )
    return ColmapFullModel(
        cameras={1: camera},
        images={205: image},
        points3d={9001: point},
        source_dir=Path("."),
        read_mode=ColmapReadMode.FULL,
        completeness=ColmapModelCompleteness.FULL,
    )


@pytest.fixture
def a2_binary_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "a2_binary"
    _write_full_binary_fixture(model_dir)
    return model_dir


def test_full_model_to_validation_dict_populates_validation_fields(
    a2_binary_dir: Path,
) -> None:
    model = read_colmap_full_model_binary(a2_binary_dir, variant="colmap")
    data = colmap_full_model_to_validation_dict(model)

    image = data["images"][205]
    point = data["points3D"][9001]

    assert image.points2d
    assert point.track
    assert point.error == pytest.approx(0.75)


def test_adapter_preserves_non_contiguous_ids_as_dict_keys(
    a2_binary_dir: Path,
) -> None:
    data = read_colmap_binary_full_for_validation(a2_binary_dir, variant="colmap")

    assert set(data["cameras"]) == {10, 42}
    assert set(data["images"]) == {101, 205, 307}
    assert set(data["points3D"]) == {9001, 9007}


def test_adapter_preserves_zero_observation_image(a2_binary_dir: Path) -> None:
    data = read_colmap_binary_full_for_validation(a2_binary_dir, variant="colmap")

    assert 101 in data["images"]
    assert data["images"][101].points2d == []


def test_adapter_normalizes_invalid_observation_sentinel(a2_binary_dir: Path) -> None:
    data = read_colmap_binary_full_for_validation(a2_binary_dir, variant="colmap")

    assert data["images"][205].points2d[1][2] == -1


def test_adapter_refuses_pose_only_model(a2_binary_dir: Path) -> None:
    pose_model = read_colmap_pose_model_binary(a2_binary_dir, variant="colmap")

    with pytest.raises(ValueError, match=FULL_MODEL_REQUIRED_MSG):
        colmap_full_model_to_validation_dict(pose_model)


def test_adapter_refuses_inconsistent_track_reference() -> None:
    with pytest.raises(ValueError, match="Inconsistent COLMAP track: point 9001"):
        colmap_full_model_to_validation_dict(_make_inconsistent_full_model())


def test_binary_adapter_matches_text_parser_semantics(
    tmp_path: Path,
) -> None:
    binary_dir = tmp_path / "binary"
    text_dir = tmp_path / "text"
    _write_full_binary_fixture(binary_dir)
    _write_full_text_fixture(text_dir)

    adapted = read_colmap_binary_full_for_validation(binary_dir, variant="colmap")
    text_cameras = parse_cameras_txt(text_dir / "cameras.txt")
    text_images = parse_images_txt(text_dir / "images.txt")
    text_points = parse_points3d_txt(text_dir / "points3D.txt")

    assert set(adapted["cameras"]) == set(text_cameras)
    for camera_id, text_camera in text_cameras.items():
        adapted_camera = adapted["cameras"][camera_id]
        assert adapted_camera.model == text_camera.model
        assert adapted_camera.width == text_camera.width
        assert adapted_camera.height == text_camera.height
        assert adapted_camera.params == text_camera.params

    assert set(adapted["images"]) == set(text_images)
    for image_id, text_image in text_images.items():
        adapted_image = adapted["images"][image_id]
        assert adapted_image.name == text_image.name
        assert adapted_image.camera_id == text_image.camera_id
        np.testing.assert_allclose(
            [adapted_image.qw, adapted_image.qx, adapted_image.qy, adapted_image.qz],
            [text_image.qw, text_image.qx, text_image.qy, text_image.qz],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            [adapted_image.tx, adapted_image.ty, adapted_image.tz],
            [text_image.tx, text_image.ty, text_image.tz],
            atol=1e-9,
        )
        assert adapted_image.points2d == text_image.points2d

    assert set(adapted["points3D"]) == set(text_points)
    for point_id, text_point in text_points.items():
        adapted_point = adapted["points3D"][point_id]
        np.testing.assert_allclose(adapted_point.xyz, text_point.xyz, atol=1e-9)
        assert tuple(int(v) for v in adapted_point.rgb) == tuple(int(v) for v in text_point.rgb)
        assert adapted_point.error == pytest.approx(text_point.error)
        assert adapted_point.track == text_point.track


def test_binary_adapter_matches_pycolmap_adapter_if_available(
    a2_binary_dir: Path,
) -> None:
    pycolmap = pytest.importorskip("pycolmap")

    adapted = read_colmap_binary_full_for_validation(a2_binary_dir, variant="colmap")
    reconstruction = pycolmap.Reconstruction(str(a2_binary_dir))
    pycolmap_data = from_pycolmap_reconstruction(reconstruction)

    assert set(adapted["cameras"]) == set(pycolmap_data["cameras"])
    assert set(adapted["images"]) == set(pycolmap_data["images"])
    assert set(adapted["points3D"]) == set(pycolmap_data["points3D"])

    for camera_id in adapted["cameras"]:
        adapted_camera = adapted["cameras"][camera_id]
        pycolmap_camera = pycolmap_data["cameras"][camera_id]
        assert adapted_camera.model == pycolmap_camera.model
        assert adapted_camera.params == pytest.approx(pycolmap_camera.params, rel=1e-6)

    for image_id in adapted["images"]:
        adapted_image = adapted["images"][image_id]
        pycolmap_image = pycolmap_data["images"][image_id]
        assert adapted_image.name == pycolmap_image.name
        assert adapted_image.points2d == pycolmap_image.points2d

    for point_id in adapted["points3D"]:
        adapted_point = adapted["points3D"][point_id]
        pycolmap_point = pycolmap_data["points3D"][point_id]
        np.testing.assert_allclose(adapted_point.xyz, pycolmap_point.xyz, atol=1e-6)
        assert adapted_point.track == pycolmap_point.track
        assert adapted_point.error == pytest.approx(pycolmap_point.error)


def test_pycolmap_adapter_normalizes_invalid_sentinel_if_available(
    a2_binary_dir: Path,
) -> None:
    pycolmap = pytest.importorskip("pycolmap")

    reconstruction = pycolmap.Reconstruction(str(a2_binary_dir))
    data = from_pycolmap_reconstruction(reconstruction)

    invalid_ids = {
        observation[2]
        for image in data["images"].values()
        for observation in image.points2d
        if observation[2] < 0
    }
    assert invalid_ids == {-1}
    assert COLMAP_INVALID_POINT3D_ID_BINARY not in {
        observation[2]
        for image in data["images"].values()
        for observation in image.points2d
    }


def test_load_reconstruction_data_warns_on_empty_points3d(caplog) -> None:
    validator = GeometricValidator()
    empty_data = {
        "cameras": {},
        "images": {},
        "points3D": {},
    }

    with caplog.at_level("WARNING"):
        validator.load_reconstruction_data(empty_data, source_label="empty fixture")

    assert any("zero points3D" in record.message for record in caplog.records)


def test_validate_masks_warns_when_zero_points_checked(
    tmp_path: Path,
    caplog,
) -> None:
    binary_dir = tmp_path / "binary"
    masks_dir = tmp_path / "masks"
    _write_consumer_binary_fixture(binary_dir)
    _write_consumer_masks(masks_dir)

    config = _explicit_validation_config(str(binary_dir))
    config.min_track_length = 99
    validator = GeometricValidator(config)
    validator.load_reconstruction_data(
        read_colmap_binary_full_for_validation(binary_dir, variant="colmap"),
        source_label=str(binary_dir),
    )

    with caplog.at_level("WARNING"):
        report = validator.validate_masks(str(masks_dir))

    assert report.total_points_checked == 0
    assert any(
        "checked zero 3D points" in record.message for record in caplog.records
    )


def test_geometric_validator_loads_adapted_binary_data(a2_binary_dir: Path) -> None:
    validator = GeometricValidator()
    data = read_colmap_binary_full_for_validation(a2_binary_dir, variant="colmap")

    validator.load_reconstruction_data(data, source_label=str(a2_binary_dir))

    assert validator.cameras
    assert validator.images
    assert validator.points3d
    assert validator._name_to_id["cafe_frame_205.jpg"] == 205


def test_geometric_validator_binary_and_text_results_match_non_vacuously(
    tmp_path: Path,
) -> None:
    binary_dir = tmp_path / "binary"
    text_dir = tmp_path / "text"
    masks_dir = tmp_path / "masks"
    _write_consumer_binary_fixture(binary_dir)
    _write_consumer_text_fixture(text_dir)
    _write_consumer_masks(masks_dir)

    config = _explicit_validation_config(str(text_dir))
    text_validator = GeometricValidator(config)
    text_validator.load_reconstruction(str(text_dir))
    text_report = text_validator.validate_masks(str(masks_dir))

    binary_validator = GeometricValidator(config)
    binary_validator.load_reconstruction_data(
        read_colmap_binary_full_for_validation(binary_dir, variant="colmap"),
        source_label=str(binary_dir),
    )
    binary_report = binary_validator.validate_masks(str(masks_dir))

    assert text_report.total_points_checked >= 1
    assert binary_report.total_points_checked >= 1
    assert any(frame.total_points > 0 for frame in text_report.frames.values())
    assert any(frame.total_points > 0 for frame in binary_report.frames.values())
    for name in ("view_001.jpg", "view_002.jpg", "view_003.jpg"):
        assert text_report.frames[name].total_points == 1
        assert text_report.frames[name].consistent_points == 1
        assert text_report.frames[name].inconsistent_points == 0

    _assert_reports_match(text_report, binary_report)


def test_geometric_validator_consumer_test_has_negative_control(
    tmp_path: Path,
) -> None:
    binary_dir = tmp_path / "binary"
    masks_dir = tmp_path / "masks"
    _write_consumer_binary_fixture(binary_dir)
    _write_consumer_masks(masks_dir)

    config = _explicit_validation_config(str(binary_dir))
    validator = GeometricValidator(config)
    adapted_data = read_colmap_binary_full_for_validation(binary_dir, variant="colmap")
    validator.load_reconstruction_data(adapted_data, source_label=str(binary_dir))
    valid_report = validator.validate_masks(str(masks_dir))
    assert valid_report.total_points_checked >= 1

    broken_data = copy.deepcopy(adapted_data)
    broken_point = broken_data["points3D"][100]
    broken_data["points3D"][100] = COLMAPPoint3D(
        point3d_id=broken_point.point3d_id,
        xyz=broken_point.xyz + np.array([100.0, 100.0, 0.0]),
        rgb=broken_point.rgb,
        error=broken_point.error,
        track=broken_point.track,
    )

    broken_validator = GeometricValidator(config)
    broken_validator.load_reconstruction_data(broken_data, source_label="broken")
    broken_report = broken_validator.validate_masks(str(masks_dir))

    assert broken_report.total_points_checked != valid_report.total_points_checked
