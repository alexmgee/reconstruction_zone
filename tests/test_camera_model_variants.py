"""Phase B contract tests: camera-model ID namespaces (upstream vs SphereSfM fork).

Upstream COLMAP >= 4.1.0 uses IDs 0-17 (11=RAD_TAN_THIN_PRISM_FISHEYE/16 params,
16=EUCM/6, 17=EQUIRECTANGULAR/2). The legacy SphereSfM fork assigned 11=SPHERE/3.
The variant is trusted provenance from the engine that produced the model;
wrong-variant reads fail fast on these fixtures (unknown ID / size mismatch),
which is defense in depth — not an absolute guarantee for arbitrary files.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from prep360.core.colmap_binary import (
    COLMAP_CAMERA_MODELS,
    SPHERESFM_CAMERA_MODELS,
    read_colmap_full_model_binary,
    read_colmap_pose_model_binary,
)
from reconstruction_gui.colmap_binary_validation_adapter import (
    read_colmap_binary_full_for_validation,
)
from reconstruction_gui.colmap_validation import (
    COLMAPCamera,
    supports_geometric_validation,
)


def _write_cameras_bin_raw(path: Path, cameras: list[tuple[int, int, int, int, list[float]]]) -> None:
    """Write cameras.bin from raw (camera_id, model_id, width, height, params)."""
    payload = struct.pack("<Q", len(cameras))
    for camera_id, model_id, width, height, params in cameras:
        payload += struct.pack("<iiQQ", camera_id, model_id, width, height)
        payload += struct.pack(f"<{len(params)}d", *params)
    path.write_bytes(payload)


def _write_minimal_images_bin(path: Path, camera_ids: list[int]) -> None:
    payload = struct.pack("<Q", len(camera_ids))
    for i, camera_id in enumerate(camera_ids, start=1):
        payload += struct.pack("<i", i)
        payload += struct.pack("<4d", 1.0, 0.0, 0.0, 0.0)
        payload += struct.pack("<3d", 0.0, 0.0, 0.0)
        payload += struct.pack("<i", camera_id)
        payload += f"img{i}.png".encode() + b"\x00"
        payload += struct.pack("<Q", 0)
    path.write_bytes(payload)


def _write_empty_points3d_bin(path: Path) -> None:
    path.write_bytes(struct.pack("<Q", 0))


def _upstream_model_dir(tmp_path: Path) -> Path:
    """Model with EUCM (16) and EQUIRECTANGULAR (17) cameras."""
    d = tmp_path / "upstream"
    d.mkdir(exist_ok=True)
    _write_cameras_bin_raw(d / "cameras.bin", [
        (1, 16, 1920, 1080, [900.0, 900.0, 960.0, 540.0, 0.6, 1.1]),   # EUCM, 6
        (2, 17, 7680, 3840, [3840.0, 1920.0]),                          # EQUIRECTANGULAR, 2
    ])
    _write_minimal_images_bin(d / "images.bin", [1, 2])
    _write_empty_points3d_bin(d / "points3D.bin")
    return d


def _spheresfm_model_dir(tmp_path: Path) -> Path:
    """Model with the fork's SPHERE camera (id 11, 3 params)."""
    d = tmp_path / "spheresfm"
    d.mkdir(exist_ok=True)
    _write_cameras_bin_raw(d / "cameras.bin", [
        (1, 11, 7680, 3840, [1.0, 3840.0, 1920.0]),                     # SPHERE, 3
    ])
    _write_minimal_images_bin(d / "images.bin", [1])
    _write_empty_points3d_bin(d / "points3D.bin")
    return d


def test_tables_diverge_only_at_11_and_above():
    for model_id in range(11):
        assert COLMAP_CAMERA_MODELS[model_id] == SPHERESFM_CAMERA_MODELS[model_id]
    assert COLMAP_CAMERA_MODELS[11] == ("RAD_TAN_THIN_PRISM_FISHEYE", 16)
    assert SPHERESFM_CAMERA_MODELS[11] == ("SPHERE", 3)
    assert COLMAP_CAMERA_MODELS[16] == ("EUCM", 6)
    assert COLMAP_CAMERA_MODELS[17] == ("EQUIRECTANGULAR", 2)
    assert 16 not in SPHERESFM_CAMERA_MODELS
    assert 17 not in SPHERESFM_CAMERA_MODELS


def test_upstream_model_parses_under_colmap_variant(tmp_path):
    model = read_colmap_full_model_binary(_upstream_model_dir(tmp_path), variant="colmap")
    assert model.cameras[1].model_name == "EUCM"
    assert len(model.cameras[1].params) == 6
    assert model.cameras[2].model_name == "EQUIRECTANGULAR"
    assert len(model.cameras[2].params) == 2


def test_spheresfm_model_parses_under_spheresfm_variant(tmp_path):
    model = read_colmap_full_model_binary(_spheresfm_model_dir(tmp_path), variant="spheresfm")
    assert model.cameras[1].model_name == "SPHERE"
    assert len(model.cameras[1].params) == 3


def test_upstream_model_under_spheresfm_variant_raises(tmp_path):
    with pytest.raises(ValueError):
        read_colmap_full_model_binary(_upstream_model_dir(tmp_path), variant="spheresfm")


def test_spheresfm_model_under_colmap_variant_raises(tmp_path):
    # SPHERE has 3 params; upstream id 11 expects 16 → param-size / trailing
    # mismatch must raise, never misread.
    with pytest.raises(ValueError):
        read_colmap_full_model_binary(_spheresfm_model_dir(tmp_path), variant="colmap")


def test_variant_is_required_keyword(tmp_path):
    d = _upstream_model_dir(tmp_path)
    with pytest.raises(TypeError):
        read_colmap_full_model_binary(d)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        read_colmap_pose_model_binary(d)  # type: ignore[call-arg]


def test_unknown_variant_rejected(tmp_path):
    with pytest.raises(ValueError):
        read_colmap_full_model_binary(_upstream_model_dir(tmp_path), variant="metashape")


def test_pose_reader_respects_variant(tmp_path):
    model = read_colmap_pose_model_binary(_spheresfm_model_dir(tmp_path), variant="spheresfm")
    assert model.cameras[1].model_name == "SPHERE"
    with pytest.raises(ValueError):
        read_colmap_pose_model_binary(_spheresfm_model_dir(tmp_path), variant="colmap")


def test_adapter_threads_variant(tmp_path):
    data = read_colmap_binary_full_for_validation(
        _upstream_model_dir(tmp_path), variant="colmap",
    )
    assert data["cameras"][1].model == "EUCM"
    assert data["cameras"][2].model == "EQUIRECTANGULAR"
    with pytest.raises(ValueError):
        read_colmap_binary_full_for_validation(
            _upstream_model_dir(tmp_path), variant="spheresfm",
        )


def test_validation_refuses_unsupported_models():
    for model, params in [
        ("EUCM", [900.0, 900.0, 960.0, 540.0, 0.6, 1.1]),
        ("EQUIRECTANGULAR", [3840.0, 1920.0]),
        ("SPHERE", [1.0, 3840.0, 1920.0]),
        ("OPENCV_FISHEYE", [900.0, 900.0, 960.0, 540.0, 0.1, 0.1, 0.1, 0.1]),
    ]:
        cam = COLMAPCamera(camera_id=1, model=model, width=100, height=100, params=params)
        assert not supports_geometric_validation(model)
        with pytest.raises(ValueError):
            cam.get_intrinsics()


def test_validator_refuses_unsupported_models_at_load(tmp_path):
    # Codex P2: a zero-point EQUIRECTANGULAR model must not "pass" validation;
    # refusal must happen at load, not lazily per projected point.
    from reconstruction_gui.colmap_validation import GeometricValidator

    data = read_colmap_binary_full_for_validation(
        _upstream_model_dir(tmp_path), variant="colmap",
    )
    validator = GeometricValidator()
    with pytest.raises(ValueError, match="does not support camera model"):
        validator.load_reconstruction_data(data)


def test_runner_engine_detection_before_binary_validation(tmp_path):
    # Codex P1: _is_spheresfm_like()/_use_pycolmap() must reflect the explicit
    # engine config even when no binary validation has run (e.g. Load Model).
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reconstruction_gui"))
    from reconstruction_gui.colmap_runner import ColmapRunner

    fake_binary = tmp_path / "colmap.exe"
    fake_binary.write_bytes(b"")
    sphere_runner = ColmapRunner(
        str(fake_binary), camera_model="SPHERE",
        workspace_root=str(tmp_path), engine_name="spheresfm",
    )
    assert sphere_runner._binary_validation is None
    assert sphere_runner._is_spheresfm_like() is True
    assert sphere_runner._use_pycolmap() is False

    stock_runner = ColmapRunner(
        str(fake_binary), camera_model="PINHOLE",
        workspace_root=str(tmp_path), engine_name="colmap",
    )
    assert stock_runner._is_spheresfm_like() is False


def test_validation_still_supports_pinhole_family():
    cam = COLMAPCamera(
        camera_id=1, model="PINHOLE", width=640, height=480,
        params=[600.0, 600.0, 320.0, 240.0],
    )
    assert supports_geometric_validation("PINHOLE")
    K = cam.get_intrinsics()
    assert np.allclose(K, [[600, 0, 320], [0, 600, 240], [0, 0, 1]])
