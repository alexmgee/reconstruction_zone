"""Tests for pose-aware camera thinning."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from prep360.cli import main as cli_main
from prep360.core.camera_thinner import (
    THINNING_MODE_POSE_AWARE,
    CameraThinningEntry,
    angle_between_vectors_degrees,
    colmap_camera_center,
    colmap_forward_world,
    find_colmap_images_txt,
    load_colmap_poses,
    match_source_images_to_poses,
    select_pose_aware_entries,
    thin_camera_folder,
)
from prep360.core.colmap_binary import COLMAP_CAMERA_MODELS, read_colmap_pose_model_binary
from prep360.core.extractor import MANIFEST_FILENAME
from prep360.core.frame_thinner import (
    THINNING_MANIFEST_FILENAME,
    FrameOrderingPolicy,
    discover_frame_files,
)


def _write_frames(input_dir: Path, names: list[str], *, content_prefix: str = "img") -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (input_dir / name).write_bytes(
            f"{content_prefix}-{name}".encode("utf-8"),
        )


def _write_cameras_bin(path: Path) -> None:
    model_id = 0
    _, num_params = COLMAP_CAMERA_MODELS[model_id]
    params = tuple(500.0 + float(i) for i in range(num_params))
    payload = struct.pack("<Q", 1)
    payload += struct.pack("<iiQQ", 1, model_id, 640, 480)
    payload += struct.pack(f"<{num_params}d", *params)
    path.write_bytes(payload)


def _write_images_bin(path: Path, records: list[dict]) -> None:
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


def _write_images_txt(path: Path, records: list[dict]) -> None:
    lines = ["# Image list with two lines of data per image"]
    for record in records:
        qvec = record["qvec"]
        tvec = record["tvec"]
        lines.append(
            f"{record['image_id']} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
            f"{tvec[0]} {tvec[1]} {tvec[2]} {record['camera_id']} {record['name']}",
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _entry(
    index: int,
    position: tuple[float, float, float],
    forward: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> CameraThinningEntry:
    name = f"frame_{index:03d}.jpg"
    return CameraThinningEntry(
        row_id=f"{index:05d}:{name}",
        image_name=name,
        source_path=Path(name),
        relative_path=name,
        order_index=index,
        image_id=index + 1,
        position=np.array(position, dtype=np.float64),
        forward=np.array(forward, dtype=np.float64),
    )


def _cumulative_drift_entries() -> list[CameraThinningEntry]:
    positions = [
        (0.0, 0.0, 0.0),
        (0.6, 0.0, 0.0),
        (1.2, 0.0, 0.0),
        (1.8, 0.0, 0.0),
        (2.4, 0.0, 0.0),
    ]
    return [_entry(index, position) for index, position in enumerate(positions)]


def _pose_records_for_chain(names: list[str], positions: list[tuple[float, float, float]]) -> list[dict]:
    records = []
    for index, (name, position) in enumerate(zip(names, positions)):
        records.append({
            "image_id": index + 1,
            "camera_id": 1,
            "name": name,
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (-position[0], -position[1], -position[2]),
        })
    return records


def test_simulate_threshold_basic_chain() -> None:
    plan = select_pose_aware_entries(
        _cumulative_drift_entries(),
        max_distance=1.0,
        max_angle_degrees=5.0,
    )

    kept_indices = [decision.order_index for decision in plan.decisions if decision.keep]
    removed_indices = [decision.order_index for decision in plan.decisions if not decision.keep]

    assert kept_indices == [0, 2, 4]
    assert removed_indices == [1, 3]


def test_removed_entries_do_not_reset_anchor() -> None:
    # A broken implementation that compares each entry to the immediately previous
    # entry (resetting the anchor every step) would keep only index 0 and remove
    # indices [1, 2, 3, 4].
    plan = select_pose_aware_entries(
        _cumulative_drift_entries(),
        max_distance=1.0,
        max_angle_degrees=5.0,
    )

    kept_indices = [decision.order_index for decision in plan.decisions if decision.keep]
    removed_indices = [decision.order_index for decision in plan.decisions if not decision.keep]

    assert kept_indices == [0, 2, 4]
    assert removed_indices == [1, 3]


def test_angle_above_threshold_keeps_entry() -> None:
    close_position = (0.5, 0.0, 0.0)
    yaw_qvec = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)
    rotated_forward = colmap_forward_world(np.array(yaw_qvec))

    entries = [
        _entry(0, (0.0, 0.0, 0.0)),
        CameraThinningEntry(
            row_id="00001:rotated.jpg",
            image_name="rotated.jpg",
            source_path=Path("rotated.jpg"),
            relative_path="rotated.jpg",
            order_index=1,
            image_id=2,
            position=np.array(close_position),
            forward=rotated_forward,
        ),
    ]

    plan = select_pose_aware_entries(
        entries,
        max_distance=1.0,
        max_angle_degrees=5.0,
    )

    assert plan.kept_count == 2
    assert plan.decisions[1].keep is True
    assert plan.decisions[1].reason == "above_distance_or_angle"


def test_distance_above_threshold_keeps_entry() -> None:
    entries = [
        _entry(0, (0.0, 0.0, 0.0)),
        _entry(1, (2.0, 0.0, 0.0)),
    ]

    plan = select_pose_aware_entries(
        entries,
        max_distance=1.0,
        max_angle_degrees=5.0,
    )

    assert plan.kept_count == 2
    assert plan.decisions[1].keep is True


def test_angle_clamps_float_noise() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 1e-15, 0.0])
    angle = angle_between_vectors_degrees(a, b)
    assert angle == pytest.approx(0.0, abs=1e-6)
    assert not math.isnan(angle)


def test_invalid_thresholds_refuse() -> None:
    entries = _cumulative_drift_entries()

    with pytest.raises(ValueError, match="max_distance"):
        select_pose_aware_entries(entries, max_distance=-0.1, max_angle_degrees=5.0)

    with pytest.raises(ValueError, match="max_angle_degrees"):
        select_pose_aware_entries(entries, max_distance=1.0, max_angle_degrees=-1.0)


def test_empty_entries_refuse() -> None:
    with pytest.raises(ValueError, match="No thinning entries"):
        select_pose_aware_entries([], max_distance=1.0, max_angle_degrees=5.0)


def test_colmap_text_and_binary_positions_match(tmp_path: Path) -> None:
    names = [f"frame_{index:03d}.jpg" for index in range(3)]
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    records = _pose_records_for_chain(names, positions)

    text_dir = tmp_path / "text_sparse"
    text_dir.mkdir()
    _write_images_txt(text_dir / "images.txt", records)

    binary_dir = tmp_path / "binary_sparse"
    binary_dir.mkdir()
    _write_cameras_bin(binary_dir / "cameras.bin")
    _write_images_bin(binary_dir / "images.bin", records)

    text_poses = load_colmap_poses(text_dir)
    binary_poses = load_colmap_poses(binary_dir)

    for name in names:
        np.testing.assert_allclose(
            text_poses[name].position,
            binary_poses[name].position,
            atol=1e-9,
        )


def test_colmap_forward_vector_known_identity() -> None:
    qvec = np.array([1.0, 0.0, 0.0, 0.0])
    forward = colmap_forward_world(qvec)
    np.testing.assert_allclose(forward, np.array([0.0, 0.0, 1.0]), atol=1e-9)


def test_colmap_forward_vector_known_yaw() -> None:
    qvec = np.array([math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0])
    forward = colmap_forward_world(qvec)
    np.testing.assert_allclose(forward, np.array([1.0, 0.0, 0.0]), atol=1e-9)


def test_colmap_text_and_binary_forward_match(tmp_path: Path) -> None:
    names = ["a.jpg", "b.jpg"]
    records = [
        {
            "image_id": 1,
            "camera_id": 1,
            "name": "a.jpg",
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (0.0, 0.0, 0.0),
        },
        {
            "image_id": 2,
            "camera_id": 1,
            "name": "b.jpg",
            "qvec": (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0),
            "tvec": (0.0, 0.0, 0.0),
        },
    ]

    text_dir = tmp_path / "text_sparse"
    text_dir.mkdir()
    _write_images_txt(text_dir / "images.txt", records)

    binary_dir = tmp_path / "binary_sparse"
    binary_dir.mkdir()
    _write_cameras_bin(binary_dir / "cameras.bin")
    _write_images_bin(binary_dir / "images.bin", records)

    text_poses = load_colmap_poses(text_dir)
    binary_poses = load_colmap_poses(binary_dir)

    for name in names:
        np.testing.assert_allclose(
            text_poses[name].forward,
            binary_poses[name].forward,
            atol=1e-9,
        )


def test_binary_pose_loading_uses_core_reader(tmp_path: Path) -> None:
    binary_dir = tmp_path / "binary_sparse"
    binary_dir.mkdir()
    records = _pose_records_for_chain(["frame_000.jpg"], [(0.0, 0.0, 0.0)])
    _write_cameras_bin(binary_dir / "cameras.bin")
    _write_images_bin(binary_dir / "images.bin", records)

    with patch(
        "prep360.core.camera_thinner.read_colmap_pose_model_binary",
        wraps=read_colmap_pose_model_binary,
    ) as reader:
        poses = load_colmap_poses(binary_dir)

    reader.assert_called_once_with(binary_dir)
    assert "frame_000.jpg" in poses


def test_matches_source_images_to_pose_names_exact(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["frame_001.jpg"])
    frames = discover_frame_files(frames_dir)

    poses = load_colmap_poses_from_records([
        {"image_id": 1, "camera_id": 1, "name": "frame_001.jpg", "qvec": (1, 0, 0, 0), "tvec": (0, 0, 0)},
    ])

    matched, unmatched_images, unmatched_poses, _warnings = match_source_images_to_poses(
        frames,
        poses,
    )

    assert len(matched) == 1
    assert matched[0].image_name == "frame_001.jpg"
    assert unmatched_images == 0
    assert unmatched_poses == 0


def load_colmap_poses_from_records(records: list[dict]) -> dict:
    poses = {}
    for record in records:
        qvec = np.array(record["qvec"], dtype=np.float64)
        tvec = np.array(record["tvec"], dtype=np.float64)
        name = record["name"]
        from prep360.core.camera_thinner import ColmapPoseRecord

        poses[name] = ColmapPoseRecord(
            image_id=record["image_id"],
            image_name=name,
            position=colmap_camera_center(qvec, tvec),
            forward=colmap_forward_world(qvec),
            qvec=qvec,
            tvec=tvec,
        )
    return poses


def test_matches_source_images_to_pose_names_by_basename(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["frame_001.jpg"])
    frames = discover_frame_files(frames_dir)

    poses = load_colmap_poses_from_records([
        {
            "image_id": 1,
            "camera_id": 1,
            "name": "images/frame_001.jpg",
            "qvec": (1, 0, 0, 0),
            "tvec": (0, 0, 0),
        },
    ])

    matched, *_ = match_source_images_to_poses(frames, poses)
    assert len(matched) == 1
    assert matched[0].image_name == "images/frame_001.jpg"


def test_ambiguous_basename_refuses(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["frame_001.jpg"])
    frames = discover_frame_files(frames_dir)

    poses = load_colmap_poses_from_records([
        {"image_id": 1, "camera_id": 1, "name": "left/frame_001.jpg", "qvec": (1, 0, 0, 0), "tvec": (0, 0, 0)},
        {"image_id": 2, "camera_id": 1, "name": "right/frame_001.jpg", "qvec": (1, 0, 0, 0), "tvec": (0, 0, 0)},
    ])

    with pytest.raises(ValueError, match="Ambiguous COLMAP image basename"):
        match_source_images_to_poses(frames, poses)


def test_unmatched_images_are_counted_and_not_copied(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    sparse_dir = tmp_path / "sparse"
    output_dir = tmp_path / "thinned"

    names = ["frame_000.jpg", "frame_001.jpg", "frame_002.jpg", "extra.jpg"]
    _write_frames(frames_dir, names)
    records = _pose_records_for_chain(names[:3], [(0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (2.4, 0.0, 0.0)])
    sparse_dir.mkdir()
    _write_images_txt(sparse_dir / "images.txt", records)

    result = thin_camera_folder(
        frames_dir,
        output_dir,
        colmap_sparse=sparse_dir,
        max_distance=1.0,
        max_angle_degrees=5.0,
    )

    assert result.unmatched_image_count == 1
    assert not (output_dir / "extra.jpg").exists()
    manifest = json.loads((output_dir / THINNING_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["unmatched_image_count"] == 1


def test_unmatched_poses_are_counted(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    sparse_dir = tmp_path / "sparse"
    output_dir = tmp_path / "thinned"

    _write_frames(frames_dir, ["frame_000.jpg", "frame_001.jpg"])
    records = _pose_records_for_chain(
        ["frame_000.jpg", "frame_001.jpg", "frame_only_in_colmap.jpg"],
        [(0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (3.0, 0.0, 0.0)],
    )
    sparse_dir.mkdir()
    _write_images_txt(sparse_dir / "images.txt", records)

    result = thin_camera_folder(
        frames_dir,
        output_dir,
        colmap_sparse=sparse_dir,
        max_distance=1.0,
        max_angle_degrees=5.0,
    )

    assert result.unmatched_pose_count == 1
    manifest = json.loads((output_dir / THINNING_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["unmatched_pose_count"] == 1


def test_no_matching_poses_refuses(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    sparse_dir = tmp_path / "sparse"
    output_dir = tmp_path / "thinned"

    _write_frames(frames_dir, ["on_disk.jpg"])
    sparse_dir.mkdir()
    _write_images_txt(sparse_dir / "images.txt", [
        {
            "image_id": 1,
            "camera_id": 1,
            "name": "different.jpg",
            "qvec": (1, 0, 0, 0),
            "tvec": (0, 0, 0),
        },
    ])

    with pytest.raises(ValueError, match="No source images matched COLMAP poses"):
        thin_camera_folder(
            frames_dir,
            output_dir,
            colmap_sparse=sparse_dir,
            max_distance=1.0,
            max_angle_degrees=5.0,
        )


def test_manifest_order_drives_sequential_order(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    sparse_dir = tmp_path / "sparse"
    output_dir = tmp_path / "thinned"

    _write_frames(frames_dir, ["c.jpg", "a.jpg", "b.jpg"])
    (frames_dir / MANIFEST_FILENAME).write_text(
        json.dumps({
            "video": "sample.mp4",
            "frames": [
                {"filename": "b.jpg", "index": 1},
                {"filename": "a.jpg", "index": 2},
                {"filename": "c.jpg", "index": 3},
            ],
        }) + "\n",
        encoding="utf-8",
    )

    records = _pose_records_for_chain(
        ["b.jpg", "a.jpg", "c.jpg"],
        [(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0)],
    )
    sparse_dir.mkdir()
    _write_images_txt(sparse_dir / "images.txt", records)

    result = thin_camera_folder(
        frames_dir,
        output_dir,
        colmap_sparse=sparse_dir,
        max_distance=1.0,
        max_angle_degrees=5.0,
        ordering=FrameOrderingPolicy.EXTRACTION_MANIFEST,
    )

    assert result.kept_count == 2
    kept_names = sorted(path.name for path in output_dir.glob("*.jpg"))
    assert kept_names == ["b.jpg", "c.jpg"]


def test_colmap_text_images_txt_resolves_sparse_subdir(tmp_path: Path) -> None:
    records = _pose_records_for_chain(["frame_000.jpg"], [(0.0, 0.0, 0.0)])

    root_txt = tmp_path / "root_txt"
    root_txt.mkdir()
    _write_images_txt(root_txt / "images.txt", records)
    assert find_colmap_images_txt(root_txt) == root_txt / "images.txt"

    sparse_txt = tmp_path / "sparse_txt"
    (sparse_txt / "sparse").mkdir(parents=True)
    _write_images_txt(sparse_txt / "sparse" / "images.txt", records)
    assert find_colmap_images_txt(sparse_txt) == sparse_txt / "sparse" / "images.txt"

    nested_txt = tmp_path / "nested_txt"
    (nested_txt / "sparse" / "0").mkdir(parents=True)
    _write_images_txt(nested_txt / "sparse" / "0" / "images.txt", records)
    assert find_colmap_images_txt(nested_txt) == nested_txt / "sparse" / "0" / "images.txt"


def test_thin_camera_folder_copies_kept_images_only(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    sparse_dir = tmp_path / "sparse"
    output_dir = tmp_path / "thinned"

    names = [f"frame_{index:03d}.jpg" for index in range(5)]
    positions = [(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0), (1.8, 0.0, 0.0), (2.4, 0.0, 0.0)]
    _write_frames(frames_dir, names)
    sparse_dir.mkdir()
    _write_images_txt(sparse_dir / "images.txt", _pose_records_for_chain(names, positions))

    result = thin_camera_folder(
        frames_dir,
        output_dir,
        colmap_sparse=sparse_dir,
        max_distance=1.0,
        max_angle_degrees=5.0,
    )

    copied = sorted(path.name for path in output_dir.glob("*.jpg"))
    assert copied == ["frame_000.jpg", "frame_002.jpg", "frame_004.jpg"]
    assert result.kept_count == 3
    assert result.removed_count == 2


def test_copy_output_is_independent(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    sparse_dir = tmp_path / "sparse"
    output_dir = tmp_path / "thinned"

    _write_frames(frames_dir, ["frame_000.jpg"], content_prefix="original")
    sparse_dir.mkdir()
    _write_images_txt(
        sparse_dir / "images.txt",
        _pose_records_for_chain(["frame_000.jpg"], [(0.0, 0.0, 0.0)]),
    )

    thin_camera_folder(
        frames_dir,
        output_dir,
        colmap_sparse=sparse_dir,
        max_distance=1.0,
        max_angle_degrees=5.0,
    )

    source_path = frames_dir / "frame_000.jpg"
    output_path = output_dir / "frame_000.jpg"
    assert output_path.read_bytes() == b"original-frame_000.jpg"

    source_path.write_bytes(b"mutated-source")
    assert output_path.read_bytes() == b"original-frame_000.jpg"


def test_refuses_non_empty_output_dir(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    sparse_dir = tmp_path / "sparse"
    output_dir = tmp_path / "thinned"

    _write_frames(frames_dir, ["frame_000.jpg"])
    sparse_dir.mkdir()
    _write_images_txt(
        sparse_dir / "images.txt",
        _pose_records_for_chain(["frame_000.jpg"], [(0.0, 0.0, 0.0)]),
    )
    output_dir.mkdir()
    (output_dir / "existing.jpg").write_bytes(b"x")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        thin_camera_folder(
            frames_dir,
            output_dir,
            colmap_sparse=sparse_dir,
            max_distance=1.0,
            max_angle_degrees=5.0,
        )


def test_refuses_existing_manifest(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    sparse_dir = tmp_path / "sparse"
    output_dir = tmp_path / "thinned"

    _write_frames(frames_dir, ["frame_000.jpg"])
    sparse_dir.mkdir()
    _write_images_txt(
        sparse_dir / "images.txt",
        _pose_records_for_chain(["frame_000.jpg"], [(0.0, 0.0, 0.0)]),
    )
    output_dir.mkdir()
    (output_dir / THINNING_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="existing manifest"):
        thin_camera_folder(
            frames_dir,
            output_dir,
            colmap_sparse=sparse_dir,
            max_distance=1.0,
            max_angle_degrees=5.0,
        )


def test_manifest_records_pose_aware_decisions(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    sparse_dir = tmp_path / "sparse"
    output_dir = tmp_path / "thinned"

    names = [f"frame_{index:03d}.jpg" for index in range(3)]
    positions = [(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0)]
    _write_frames(frames_dir, names)
    sparse_dir.mkdir()
    _write_images_txt(sparse_dir / "images.txt", _pose_records_for_chain(names, positions))

    thin_camera_folder(
        frames_dir,
        output_dir,
        colmap_sparse=sparse_dir,
        max_distance=1.0,
        max_angle_degrees=5.0,
    )

    manifest = json.loads((output_dir / THINNING_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["mode"] == THINNING_MODE_POSE_AWARE
    assert manifest["selection"]["max_distance"] == 1.0
    assert manifest["selection"]["max_angle_degrees"] == 5.0
    assert manifest["pose_source"]["kind"] == "colmap"
    assert manifest["kept_count"] == 2
    assert manifest["removed_count"] == 1
    assert manifest["kept"][0]["distance"] is None
    assert manifest["removed"][0]["distance"] == pytest.approx(0.6)
    assert manifest["removed"][0]["angle_degrees"] == pytest.approx(0.0)


def test_cli_thin_cameras(tmp_path: Path, capsys, monkeypatch) -> None:
    frames_dir = tmp_path / "frames"
    sparse_dir = tmp_path / "sparse"
    output_dir = tmp_path / "thinned"

    names = [f"frame_{index:03d}.jpg" for index in range(3)]
    positions = [(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0)]
    _write_frames(frames_dir, names)
    sparse_dir.mkdir()
    _write_images_txt(sparse_dir / "images.txt", _pose_records_for_chain(names, positions))

    monkeypatch.setattr(
        "sys.argv",
        [
            "prep360",
            "thin-cameras",
            "--input", str(frames_dir),
            "--output", str(output_dir),
            "--colmap-sparse", str(sparse_dir),
            "--max-distance", "1.0",
            "--max-angle", "5.0",
        ],
    )
    exit_code = cli_main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Pose-thinned 3 images (3 posed) -> 2 kept" in captured.out
    assert (output_dir / THINNING_MANIFEST_FILENAME).is_file()
    assert len(list(output_dir.glob("*.jpg"))) == 2


def test_cli_refuses_invalid_threshold(tmp_path: Path, capsys, monkeypatch) -> None:
    frames_dir = tmp_path / "frames"
    sparse_dir = tmp_path / "sparse"
    output_dir = tmp_path / "thinned"

    _write_frames(frames_dir, ["frame_000.jpg"])
    sparse_dir.mkdir()
    _write_images_txt(
        sparse_dir / "images.txt",
        _pose_records_for_chain(["frame_000.jpg"], [(0.0, 0.0, 0.0)]),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "prep360",
            "thin-cameras",
            "--input", str(frames_dir),
            "--output", str(output_dir),
            "--colmap-sparse", str(sparse_dir),
            "--max-distance", "-1.0",
            "--max-angle", "5.0",
        ],
    )
    exit_code = cli_main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "max_distance" in captured.err
