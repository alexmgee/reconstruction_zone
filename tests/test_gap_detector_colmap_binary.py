"""Coverage integration tests for COLMAP binary pose sources."""

from __future__ import annotations

from pathlib import Path

import pytest

from prep360.core.gap_detector import GapDetector
from tests.test_colmap_binary_pose_only import (
    _sample_pose_records,
    _write_cameras_bin,
    _write_images_bin,
    _write_images_txt,
)


def test_gap_detector_accepts_binary_only_sparse_root(tmp_path: Path) -> None:
    records = _sample_pose_records()
    _write_cameras_bin(tmp_path / "cameras.bin")
    _write_images_bin(tmp_path / "images.bin", records)

    report = GapDetector(cluster_eps=100.0, min_cluster_size=1).analyze(
        colmap_dir=str(tmp_path),
    )

    assert report.num_cameras_aligned == 3
    names = {cam.name for cam in report.camera_positions}
    assert names == {
        "frame_00001_t12.5s.jpg",
        "frame_00002.jpg",
        "frame_00003.jpg",
    }
    camera_ids = {cam.camera_id for cam in report.camera_positions}
    assert camera_ids == {1}
    assert report.camera_positions[0].timestamp == 12.5


def test_gap_detector_accepts_binary_only_sparse_zero_subdir(tmp_path: Path) -> None:
    sparse_dir = tmp_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    records = _sample_pose_records()
    _write_cameras_bin(sparse_dir / "cameras.bin")
    _write_images_bin(sparse_dir / "images.bin", records)

    report = GapDetector(cluster_eps=100.0, min_cluster_size=1).analyze(
        colmap_dir=str(tmp_path),
    )

    assert report.num_cameras_aligned == 3
    assert {cam.name for cam in report.camera_positions} == {
        "frame_00001_t12.5s.jpg",
        "frame_00002.jpg",
        "frame_00003.jpg",
    }


def test_gap_detector_prefers_existing_text_behavior(tmp_path: Path) -> None:
    records = _sample_pose_records()
    _write_cameras_bin(tmp_path / "cameras.bin")
    _write_images_bin(tmp_path / "images.bin", records)

    text_records = [
        {
            **records[0],
            "name": "text_only_frame.jpg",
        },
        *records[1:],
    ]
    _write_images_txt(tmp_path / "images.txt", text_records)

    report = GapDetector(cluster_eps=100.0, min_cluster_size=1).analyze(
        colmap_dir=str(tmp_path),
    )

    names = {cam.name for cam in report.camera_positions}
    assert "text_only_frame.jpg" in names
    assert "frame_00001_t12.5s.jpg" not in names


def test_gap_detector_missing_pose_source_mentions_text_and_binary(
    tmp_path: Path,
) -> None:
    detector = GapDetector()

    with pytest.raises(FileNotFoundError) as excinfo:
        detector.analyze(colmap_dir=str(tmp_path))

    message = str(excinfo.value)
    assert "images.txt" in message
    assert "cameras.bin + images.bin" in message


def test_gap_detector_binary_timestamp_parity(tmp_path: Path) -> None:
    records = [
        {
            "image_id": 1,
            "camera_id": 1,
            "name": "frame_00001_t12.5s.jpg",
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (0.0, 0.0, 0.0),
            "observations": [],
        },
    ]

    binary_dir = tmp_path / "binary"
    binary_dir.mkdir()
    _write_cameras_bin(binary_dir / "cameras.bin")
    _write_images_bin(binary_dir / "images.bin", records)

    text_path = tmp_path / "images.txt"
    _write_images_txt(text_path, records)

    detector = GapDetector()
    text_cam = detector.parse_colmap_images(str(text_path))[0]
    binary_cam = detector.parse_colmap_binary_poses(str(binary_dir))[0]

    assert text_cam.timestamp == binary_cam.timestamp == 12.5
