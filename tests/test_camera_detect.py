"""Tests for read-only camera/pose candidate detection."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from prep360.cli import main as cli_main
from prep360.core.camera_detect import (
    CameraFormat,
    scan_camera_candidates,
)
from tests.test_colmap_binary_pose_only import (
    _sample_pose_records,
    _write_cameras_bin,
    _write_images_bin,
    _write_images_txt,
)


def _write_cameras_txt(path: Path, count: int = 1) -> None:
    lines = ["# Camera list"]
    for camera_id in range(1, count + 1):
        lines.append(f"{camera_id} PINHOLE 1920 1080 1000 1000 960 540")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_points_txt(path: Path) -> None:
    path.write_text(
        "# 3D point list\n1 0 0 0 255 255 255 0.0 0\n",
        encoding="utf-8",
    )


def _write_points_bin(path: Path) -> None:
    path.write_bytes(struct.pack("<Q", 0))


def _write_transforms_json(path: Path, *, frame_count: int = 1) -> None:
    frames = [
        {
            "file_path": f"images/frame_{index:04d}.jpg",
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        for index in range(frame_count)
    ]
    path.write_text(
        json.dumps(
            {
                "camera_model": "OPENCV",
                "w": 1920,
                "h": 1080,
                "fl_x": 1000.0,
                "fl_y": 1000.0,
                "cx": 960.0,
                "cy": 540.0,
                "frames": frames,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_metashape_xml(path: Path, *, camera_count: int = 1, sensor_count: int = 1) -> None:
    sensors = "\n".join(
        f'    <sensor id="{sensor_id}" type="frame"/>'
        for sensor_id in range(1, sensor_count + 1)
    )
    cameras = "\n".join(
        f'      <camera id="{camera_id}" sensor_id="1" label="frame_{camera_id:04d}"/>'
        for camera_id in range(1, camera_count + 1)
    )
    path.write_text(
        "\n".join([
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<document>",
            "  <chunk>",
            f"{sensors}",
            "    <cameras>",
            f"{cameras}",
            "    </cameras>",
            "  </chunk>",
            "</document>",
            "",
        ]),
        encoding="utf-8",
    )


def _find_kind(candidates, kind: CameraFormat):
    return [candidate for candidate in candidates if candidate.kind == kind]


def test_detect_colmap_bin_full(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    _write_cameras_bin(sparse / "cameras.bin")
    _write_images_bin(sparse / "images.bin", _sample_pose_records())
    _write_points_bin(sparse / "points3D.bin")

    candidates = scan_camera_candidates(tmp_path)
    matches = _find_kind(candidates, CameraFormat.COLMAP_BIN_FULL)

    assert len(matches) == 1
    assert matches[0].path == str(sparse.resolve())
    assert matches[0].frame_count == 3
    assert matches[0].camera_count == 1
    assert matches[0].points_path.endswith("points3D.bin")


def test_detect_colmap_bin_pose_only(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    _write_cameras_bin(sparse / "cameras.bin")
    _write_images_bin(sparse / "images.bin", _sample_pose_records())

    candidates = scan_camera_candidates(tmp_path)
    matches = _find_kind(candidates, CameraFormat.COLMAP_BIN_POSE_ONLY)

    assert len(matches) == 1
    assert matches[0].path == str(sparse.resolve())
    assert matches[0].points_path == ""


def test_detect_colmap_txt_full(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    _write_cameras_txt(sparse / "cameras.txt")
    _write_images_txt(sparse / "images.txt", _sample_pose_records())
    _write_points_txt(sparse / "points3D.txt")

    candidates = scan_camera_candidates(tmp_path)
    matches = _find_kind(candidates, CameraFormat.COLMAP_TXT_FULL)

    assert len(matches) == 1
    assert matches[0].frame_count == 3
    assert matches[0].camera_count == 1


def test_detect_colmap_txt_pose_only(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    _write_cameras_txt(sparse / "cameras.txt")
    _write_images_txt(sparse / "images.txt", _sample_pose_records())

    candidates = scan_camera_candidates(tmp_path)
    matches = _find_kind(candidates, CameraFormat.COLMAP_TXT_POSE_ONLY)

    assert len(matches) == 1
    assert matches[0].points_path == ""


def test_text_and_binary_variants_are_separate_candidates(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    _write_cameras_bin(sparse / "cameras.bin")
    _write_images_bin(sparse / "images.bin", _sample_pose_records())
    _write_cameras_txt(sparse / "cameras.txt")
    _write_images_txt(sparse / "images.txt", _sample_pose_records())

    candidates = scan_camera_candidates(tmp_path)
    kinds = {candidate.kind for candidate in candidates}

    assert CameraFormat.COLMAP_BIN_POSE_ONLY in kinds
    assert CameraFormat.COLMAP_TXT_POSE_ONLY in kinds
    assert len(_find_kind(candidates, CameraFormat.COLMAP_BIN_POSE_ONLY)) == 1
    assert len(_find_kind(candidates, CameraFormat.COLMAP_TXT_POSE_ONLY)) == 1


def test_multiple_sparse_models_are_preserved(tmp_path: Path) -> None:
    for index in (0, 1):
        sparse = tmp_path / "sparse" / str(index)
        sparse.mkdir(parents=True)
        _write_cameras_bin(sparse / "cameras.bin")
        _write_images_bin(sparse / "images.bin", _sample_pose_records()[:1])

    candidates = scan_camera_candidates(tmp_path)
    matches = _find_kind(candidates, CameraFormat.COLMAP_BIN_POSE_ONLY)

    assert len(matches) == 2
    paths = {candidate.path for candidate in matches}
    assert str((tmp_path / "sparse" / "0").resolve()) in paths
    assert str((tmp_path / "sparse" / "1").resolve()) in paths


def test_partial_colmap_pair_is_not_candidate(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    _write_images_bin(sparse / "images.bin", _sample_pose_records())

    candidates = scan_camera_candidates(tmp_path)
    colmap_kinds = {
        CameraFormat.COLMAP_BIN_FULL,
        CameraFormat.COLMAP_BIN_POSE_ONLY,
        CameraFormat.COLMAP_TXT_FULL,
        CameraFormat.COLMAP_TXT_POSE_ONLY,
    }
    assert not any(candidate.kind in colmap_kinds for candidate in candidates)


def test_transforms_json_detected(tmp_path: Path) -> None:
    _write_transforms_json(tmp_path / "transforms.json", frame_count=2)

    candidates = scan_camera_candidates(tmp_path)
    matches = _find_kind(candidates, CameraFormat.TRANSFORMS_JSON)

    assert len(matches) == 1
    assert matches[0].frame_count == 2
    assert matches[0].path.endswith("transforms.json")


def test_invalid_transforms_json_rejected(tmp_path: Path) -> None:
    (tmp_path / "transforms.json").write_text('{"frames": "not-a-list"}', encoding="utf-8")
    (tmp_path / "other.json").write_text('{"frames": []}', encoding="utf-8")
    (tmp_path / "transforms_train.json").write_text(
        json.dumps({"frames": [{"transform_matrix": [[1]]}]}),
        encoding="utf-8",
    )

    candidates = scan_camera_candidates(tmp_path)
    assert _find_kind(candidates, CameraFormat.TRANSFORMS_JSON) == []


def test_metashape_camera_xml_detected(tmp_path: Path) -> None:
    _write_metashape_xml(tmp_path / "cameras.xml", camera_count=2, sensor_count=1)

    candidates = scan_camera_candidates(tmp_path)
    matches = _find_kind(candidates, CameraFormat.METASHAPE_XML)

    assert len(matches) == 1
    assert matches[0].frame_count == 2
    assert matches[0].camera_count == 1


def test_unrelated_xml_rejected(tmp_path: Path) -> None:
    (tmp_path / "config.xml").write_text(
        "<root><item>value</item></root>",
        encoding="utf-8",
    )

    candidates = scan_camera_candidates(tmp_path)
    assert _find_kind(candidates, CameraFormat.METASHAPE_XML) == []


def test_scan_depth_limit(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c" / "sparse"
    deep.mkdir(parents=True)
    _write_cameras_bin(deep / "cameras.bin")
    _write_images_bin(deep / "images.bin", _sample_pose_records()[:1])

    shallow = scan_camera_candidates(tmp_path, max_depth=2)
    deep_scan = scan_camera_candidates(tmp_path, max_depth=8)

    assert _find_kind(shallow, CameraFormat.COLMAP_BIN_POSE_ONLY) == []
    assert len(_find_kind(deep_scan, CameraFormat.COLMAP_BIN_POSE_ONLY)) == 1


def test_ignored_directories_are_pruned(tmp_path: Path) -> None:
    for ignored in (".git", "node_modules"):
        sparse = tmp_path / ignored / "sparse"
        sparse.mkdir(parents=True)
        _write_cameras_bin(sparse / "cameras.bin")
        _write_images_bin(sparse / "images.bin", _sample_pose_records()[:1])

    visible = tmp_path / "visible" / "sparse"
    visible.mkdir(parents=True)
    _write_cameras_bin(visible / "cameras.bin")
    _write_images_bin(visible / "images.bin", _sample_pose_records()[:1])

    candidates = scan_camera_candidates(tmp_path)
    matches = _find_kind(candidates, CameraFormat.COLMAP_BIN_POSE_ONLY)

    assert len(matches) == 1
    assert matches[0].path == str(visible.resolve())


def test_hidden_dot_directory_is_pruned(tmp_path: Path) -> None:
    hidden = tmp_path / ".hidden" / "sparse"
    hidden.mkdir(parents=True)
    _write_cameras_bin(hidden / "cameras.bin")
    _write_images_bin(hidden / "images.bin", _sample_pose_records()[:1])

    visible = tmp_path / "visible" / "sparse"
    visible.mkdir(parents=True)
    _write_cameras_bin(visible / "cameras.bin")
    _write_images_bin(visible / "images.bin", _sample_pose_records()[:1])

    candidates = scan_camera_candidates(tmp_path)
    matches = _find_kind(candidates, CameraFormat.COLMAP_BIN_POSE_ONLY)

    assert len(matches) == 1
    assert matches[0].path == str(visible.resolve())


def test_malformed_text_colmap_scan_continues(tmp_path: Path) -> None:
    bad_sparse = tmp_path / "bad_sparse"
    bad_sparse.mkdir()
    _write_cameras_txt(bad_sparse / "cameras.txt")
    (bad_sparse / "images.txt").write_bytes(b"\xff\xfe not-valid-utf8")

    good_sparse = tmp_path / "good_sparse"
    good_sparse.mkdir()
    _write_cameras_bin(good_sparse / "cameras.bin")
    _write_images_bin(good_sparse / "images.bin", _sample_pose_records()[:1])

    candidates = scan_camera_candidates(tmp_path)

    bad_matches = [
        candidate for candidate in candidates
        if candidate.path == str(bad_sparse.resolve())
    ]
    good_matches = _find_kind(candidates, CameraFormat.COLMAP_BIN_POSE_ONLY)

    assert len(bad_matches) == 1
    assert bad_matches[0].kind == CameraFormat.COLMAP_TXT_POSE_ONLY
    assert bad_matches[0].frame_count is None
    assert bad_matches[0].camera_count == 1
    assert "Could not read COLMAP text counts" in bad_matches[0].notes
    assert len(good_matches) == 1
    assert good_matches[0].path == str(good_sparse.resolve())


def test_candidates_are_deterministically_sorted(tmp_path: Path) -> None:
    second = tmp_path / "b_sparse"
    first = tmp_path / "a_sparse"
    for sparse in (first, second):
        sparse.mkdir()
        _write_cameras_bin(sparse / "cameras.bin")
        _write_images_bin(sparse / "images.bin", _sample_pose_records()[:1])

    first_scan = scan_camera_candidates(tmp_path)
    second_scan = scan_camera_candidates(tmp_path)

    assert [candidate.path for candidate in first_scan] == [
        candidate.path for candidate in second_scan
    ]
    assert first_scan[0].path == str(first.resolve())


def test_cli_detect_cameras_json(tmp_path: Path, capsys, monkeypatch) -> None:
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    _write_cameras_bin(sparse / "cameras.bin")
    _write_images_bin(sparse / "images.bin", _sample_pose_records()[:1])

    monkeypatch.setattr(
        "sys.argv",
        ["prep360", "detect-cameras", str(tmp_path), "--json"],
    )
    exit_code = cli_main()

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert len(payload) == 1
    assert payload[0]["kind"] == CameraFormat.COLMAP_BIN_POSE_ONLY.value


def test_cli_detect_cameras_text(tmp_path: Path, capsys, monkeypatch) -> None:
    _write_transforms_json(tmp_path / "transforms.json", frame_count=1)

    monkeypatch.setattr(
        "sys.argv",
        ["prep360", "detect-cameras", str(tmp_path)],
    )
    exit_code = cli_main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert CameraFormat.TRANSFORMS_JSON.value in captured.out
    assert str((tmp_path / "transforms.json").resolve()) in captured.out
