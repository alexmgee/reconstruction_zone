"""Phase C qualification contract for the active pycolmap installation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from reconstruction_gui.colmap_runner import ColmapRunner

# The entire module is a contract against a live pycolmap install; skip where the
# optional dependency is absent (e.g. CI), matching the convention used in
# tests/test_backend_qualification.py and tests/test_colmap_binary_full.py.
pycolmap = pytest.importorskip("pycolmap")


def _runner(tmp_path: Path) -> ColmapRunner:
    binary = tmp_path / "colmap.exe"
    binary.write_bytes(b"")
    images = tmp_path / "images"
    images.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner = ColmapRunner(
        str(binary),
        camera_model="PINHOLE",
        workspace_root=str(tmp_path),
        engine_name="colmap",
    )
    runner.current_run_dir = run_dir
    runner.current_images_dir = images
    return runner


def test_feature_and_matcher_enum_contract():
    assert not hasattr(pycolmap, "FeatureMatchingType")
    for name in ("SIFT", "ALIKED_N16ROT", "ALIKED_N32"):
        assert hasattr(pycolmap.FeatureExtractorType, name)
    for name in (
        "SIFT_BRUTEFORCE",
        "SIFT_LIGHTGLUE",
        "ALIKED_BRUTEFORCE",
        "ALIKED_LIGHTGLUE",
    ):
        assert hasattr(pycolmap.FeatureMatcherType, name)


def test_incremental_pipeline_unconditional_contract():
    opts = pycolmap.IncrementalPipelineOptions()
    for name in (
        "ba_use_gpu",
        "ba_gpu_index",
        "min_num_matches",
        "ba_refine_focal_length",
        "ba_refine_principal_point",
        "ba_refine_extra_params",
    ):
        assert hasattr(opts, name)
    assert hasattr(pycolmap.BundleAdjustmentBackend, "CERES")


def test_camera_model_410_contract():
    if not all(
        hasattr(pycolmap.CameraModelId, name)
        for name in ("EUCM", "EQUIRECTANGULAR", "RAD_TAN_THIN_PRISM_FISHEYE")
    ):
        pytest.skip("pycolmap lacks the COLMAP 4.1 camera-model namespace")
    assert pycolmap.CameraModelId.EUCM.value == 16
    assert pycolmap.CameraModelId.EQUIRECTANGULAR.value == 17
    assert pycolmap.CameraModelId(11) == pycolmap.CameraModelId.RAD_TAN_THIN_PRISM_FISHEYE


def test_caspar_backend_contract():
    if not hasattr(pycolmap.BundleAdjustmentBackend, "CASPAR"):
        pytest.skip("pycolmap lacks the CASPAR backend")
    opts = pycolmap.IncrementalPipelineOptions()
    if not hasattr(opts, "ba_local_backend") or not hasattr(opts, "ba_global_backend"):
        pytest.skip("pycolmap lacks incremental CASPAR backend controls")
    opts.ba_local_backend = pycolmap.BundleAdjustmentBackend.CASPAR
    opts.ba_global_backend = pycolmap.BundleAdjustmentBackend.CASPAR
    assert opts.ba_local_backend == pycolmap.BundleAdjustmentBackend.CASPAR
    assert opts.ba_global_backend == pycolmap.BundleAdjustmentBackend.CASPAR


def test_feature_extraction_limit_contract():
    opts = pycolmap.FeatureExtractionOptions()
    if not hasattr(opts, "max_image_size") or not hasattr(opts.sift, "max_num_features"):
        pytest.skip("pycolmap lacks feature-extraction limit controls")
    opts.max_image_size = 1234
    opts.sift.max_num_features = 4321
    assert opts.max_image_size == 1234
    assert opts.sift.max_num_features == 4321


def test_extract_features_option_plumbing(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    captured = {}

    def fake_extract_features(database_path, image_path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pycolmap, "extract_features", fake_extract_features)
    runner._extract_features_pycolmap(
        feature_type="ALIKED_N32",
        max_features=3456,
        max_image_size=2345,
        effective_masks_dir=None,
        extra_args=None,
        progress_callback=None,
    )

    opts = captured["extraction_options"]
    assert opts.type == pycolmap.FeatureExtractorType.ALIKED_N32
    assert opts.max_image_size == 2345
    assert opts.sift.max_num_features == 3456
    if hasattr(opts, "aliked") and hasattr(opts.aliked, "max_num_features"):
        assert opts.aliked.max_num_features == 3456


def test_match_features_option_plumbing(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    captured = {}

    def fake_match_exhaustive(database_path, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pycolmap, "match_exhaustive", fake_match_exhaustive)
    runner._match_features_pycolmap(
        strategy="exhaustive",
        guided=True,
        max_num_matches=9876,
        vocab_tree_path=None,
        extra_args={"FeatureMatching.type": "ALIKED_LIGHTGLUE"},
        progress_callback=None,
    )

    opts = captured["matching_options"]
    assert opts.guided_matching is True
    assert opts.max_num_matches == 9876
    assert opts.type == pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE


def test_incremental_mapping_option_plumbing(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    captured = {}

    def fake_incremental_mapping(database_path, image_path, output_path, options):
        captured["options"] = options
        return {}

    monkeypatch.setattr(pycolmap, "incremental_mapping", fake_incremental_mapping)
    runner._reconstruct_pycolmap(
        mapper_key="incremental",
        sparse_root=runner.current_run_dir / "sparse",
        min_num_inliers=23,
        extra_args={"Mapper.ba_refine_principal_point": True},
        progress_callback=None,
        ba_backend="AUTO",
        ba_use_gpu=True,
        ba_gpu_index=2,
    )

    opts = captured["options"]
    assert opts.min_num_matches == 23
    assert opts.ba_use_gpu is True
    assert opts.ba_gpu_index == "2"
    assert opts.ba_refine_principal_point is True
    if hasattr(pycolmap.BundleAdjustmentBackend, "CASPAR"):
        # Hybrid: global BA on Caspar, local BA stays Ceres (measured faster
        # with identical quality — see PROGRESS.md 2026-07-17).
        assert opts.ba_local_backend == pycolmap.BundleAdjustmentBackend.CERES
        assert opts.ba_global_backend == pycolmap.BundleAdjustmentBackend.CASPAR


def test_explicit_caspar_rejects_ineligible_camera(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    database_path = runner.current_run_dir / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE cameras (camera_id INTEGER, model INTEGER)")
        connection.execute("INSERT INTO cameras VALUES (1, 5)")

    called = False

    def fake_incremental_mapping(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(pycolmap, "incremental_mapping", fake_incremental_mapping)
    with pytest.raises(RuntimeError, match=r"camera model IDs \[5\]"):
        runner._reconstruct_pycolmap(
            mapper_key="incremental",
            sparse_root=runner.current_run_dir / "sparse",
            min_num_inliers=15,
            extra_args=None,
            progress_callback=None,
            ba_backend="CASPAR",
        )
    assert called is False
