"""Tests for temporal frame thinning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prep360.cli import main as cli_main
from prep360.core.extractor import MANIFEST_FILENAME
from prep360.core.frame_thinner import (
    THINNING_MANIFEST_FILENAME,
    FrameOrderingPolicy,
    discover_frame_files,
    select_every_nth,
    thin_frame_folder,
)


def _write_frames(input_dir: Path, names: list[str]) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (input_dir / name).write_bytes(b"image-bytes-" + name.encode("utf-8"))


def _write_extraction_manifest(
    input_dir: Path,
    frames: list[dict],
) -> None:
    payload = {
        "video": "sample.mp4",
        "frames": frames,
    }
    (input_dir / MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def test_temporal_every_nth(tmp_path: Path) -> None:
    names = [f"frame_{index:05d}.jpg" for index in range(100)]
    _write_frames(tmp_path / "frames", names)

    result = thin_frame_folder(
        tmp_path / "frames",
        tmp_path / "thinned",
        every_nth=5,
    )

    assert result.input_count == 100
    assert result.kept_count == 20
    assert result.removed_count == 80
    assert len(list((tmp_path / "thinned").glob("*.jpg"))) == 20


def test_every_nth_one_keeps_all(tmp_path: Path) -> None:
    names = [f"frame_{index:03d}.jpg" for index in range(10)]
    _write_frames(tmp_path / "frames", names)

    result = thin_frame_folder(
        tmp_path / "frames",
        tmp_path / "thinned",
        every_nth=1,
    )

    assert result.kept_count == 10
    assert result.removed_count == 0


def test_every_nth_larger_than_count_keeps_first(tmp_path: Path) -> None:
    names = ["a.jpg", "b.jpg", "c.jpg"]
    _write_frames(tmp_path / "frames", names)

    result = thin_frame_folder(
        tmp_path / "frames",
        tmp_path / "thinned",
        every_nth=10,
    )

    assert result.kept_count == 1
    assert (tmp_path / "thinned" / "a.jpg").is_file()


def test_invalid_every_nth_refuses(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames", ["a.jpg"])

    with pytest.raises(ValueError, match="every_nth must be"):
        select_every_nth(discover_frame_files(tmp_path / "frames"), 0)

    with pytest.raises(ValueError, match="every_nth must be"):
        thin_frame_folder(
            tmp_path / "frames",
            tmp_path / "thinned",
            every_nth=0,
        )


def test_natural_filename_order(tmp_path: Path) -> None:
    names = ["frame_10.jpg", "frame_2.jpg", "frame_1.jpg"]
    _write_frames(tmp_path / "frames", names)

    frames = discover_frame_files(
        tmp_path / "frames",
        ordering=FrameOrderingPolicy.FILENAME_NATURAL,
    )

    assert [frame.relative_path for frame in frames] == [
        "frame_1.jpg",
        "frame_2.jpg",
        "frame_10.jpg",
    ]


def test_extraction_manifest_order_used(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["c.jpg", "a.jpg", "b.jpg"])
    _write_extraction_manifest(frames_dir, [
        {"filename": "b.jpg", "index": 1, "time_sec": 2.0},
        {"filename": "a.jpg", "index": 2, "time_sec": 1.0},
        {"filename": "c.jpg", "index": 3, "time_sec": 3.0},
    ])

    discovered = discover_frame_files(frames_dir, ordering=FrameOrderingPolicy.AUTO)

    assert [frame.relative_path for frame in discovered] == [
        "b.jpg",
        "a.jpg",
        "c.jpg",
    ]


def test_manifest_order_index_is_zero_based(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["first.jpg", "second.jpg", "third.jpg"])
    _write_extraction_manifest(frames_dir, [
        {"filename": "third.jpg", "index": 3, "time_sec": 2.0},
        {"filename": "first.jpg", "index": 1, "time_sec": 0.0},
        {"filename": "second.jpg", "index": 2, "time_sec": 1.0},
    ])

    frames = discover_frame_files(
        frames_dir,
        ordering=FrameOrderingPolicy.EXTRACTION_MANIFEST,
    )
    kept, removed = select_every_nth(frames, every_nth=2)

    assert [frame.relative_path for frame in kept] == ["third.jpg", "second.jpg"]
    assert [frame.order_index for frame in kept] == [0, 2]
    assert [frame.relative_path for frame in removed] == ["first.jpg"]
    assert removed[0].order_index == 1


def test_manifest_required_when_explicit(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["a.jpg"])

    with pytest.raises(FileNotFoundError, match="Extraction manifest not found"):
        discover_frame_files(
            frames_dir,
            ordering=FrameOrderingPolicy.EXTRACTION_MANIFEST,
        )


def test_auto_malformed_manifest_falls_back_to_natural(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["frame_2.jpg", "frame_10.jpg", "frame_1.jpg"])
    (frames_dir / MANIFEST_FILENAME).write_text("{not valid json", encoding="utf-8")

    discovery = discover_frame_files(frames_dir, ordering=FrameOrderingPolicy.AUTO)
    result = thin_frame_folder(
        frames_dir,
        tmp_path / "thinned",
        every_nth=1,
        ordering=FrameOrderingPolicy.AUTO,
    )

    assert [frame.relative_path for frame in discovery] == [
        "frame_1.jpg",
        "frame_2.jpg",
        "frame_10.jpg",
    ]
    assert result.input_count == 3
    manifest = json.loads((tmp_path / "thinned" / THINNING_MANIFEST_FILENAME).read_text())
    assert manifest["selection"]["ordering"] == "filename_natural"


def test_auto_empty_manifest_falls_back_to_natural(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["frame_2.jpg", "frame_1.jpg"])
    _write_extraction_manifest(frames_dir, [
        {"filename": "missing_a.jpg", "index": 1, "time_sec": 0.0},
        {"filename": "missing_b.jpg", "index": 2, "time_sec": 1.0},
    ])

    discovery = discover_frame_files(frames_dir, ordering=FrameOrderingPolicy.AUTO)
    thin_frame_folder(
        frames_dir,
        tmp_path / "thinned",
        every_nth=1,
        ordering=FrameOrderingPolicy.AUTO,
    )

    assert [frame.relative_path for frame in discovery] == [
        "frame_1.jpg",
        "frame_2.jpg",
    ]
    manifest = json.loads((tmp_path / "thinned" / THINNING_MANIFEST_FILENAME).read_text())
    assert manifest["selection"]["ordering"] == "filename_natural"


def test_explicit_extraction_manifest_fails_on_malformed(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["a.jpg"])
    (frames_dir / MANIFEST_FILENAME).write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid extraction manifest"):
        discover_frame_files(
            frames_dir,
            ordering=FrameOrderingPolicy.EXTRACTION_MANIFEST,
        )


def test_explicit_extraction_manifest_fails_when_no_existing_images(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["a.jpg"])
    _write_extraction_manifest(frames_dir, [
        {"filename": "missing.jpg", "index": 1, "time_sec": 0.0},
    ])

    with pytest.raises(ValueError, match="references no existing images"):
        discover_frame_files(
            frames_dir,
            ordering=FrameOrderingPolicy.EXTRACTION_MANIFEST,
        )


def test_manifest_missing_entries_are_not_fatal(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["a.jpg", "b.jpg"])
    _write_extraction_manifest(frames_dir, [
        {"filename": "a.jpg", "index": 1, "time_sec": 0.0},
        {"filename": "missing.jpg", "index": 2, "time_sec": 1.0},
        {"filename": "b.jpg", "index": 3, "time_sec": 2.0},
    ])

    result = thin_frame_folder(
        frames_dir,
        tmp_path / "thinned",
        every_nth=1,
        ordering=FrameOrderingPolicy.EXTRACTION_MANIFEST,
    )

    manifest = json.loads((tmp_path / "thinned" / THINNING_MANIFEST_FILENAME).read_text())
    assert result.input_count == 2
    assert manifest["missing_manifest_entries"] == 1


def test_manifest_unlisted_on_disk_files_warned(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["listed.jpg", "extra.jpg"])
    _write_extraction_manifest(frames_dir, [
        {"filename": "listed.jpg", "index": 1, "time_sec": 0.0},
    ])

    result = thin_frame_folder(
        frames_dir,
        tmp_path / "thinned",
        every_nth=1,
        ordering=FrameOrderingPolicy.EXTRACTION_MANIFEST,
    )

    manifest = json.loads((tmp_path / "thinned" / THINNING_MANIFEST_FILENAME).read_text())
    assert result.unlisted_on_disk_count == 1
    assert manifest["warnings"]
    assert "absent from extraction_manifest.json" in manifest["warnings"][0]


def test_manifest_filename_rejects_path_separator(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["good.jpg", "nested.jpg"])
    (frames_dir / "nested.jpg").write_bytes(b"nested")
    _write_extraction_manifest(frames_dir, [
        {"filename": "subdir/nested.jpg", "index": 1, "time_sec": 0.0},
        {"filename": "good.jpg", "index": 2, "time_sec": 1.0},
    ])

    result = thin_frame_folder(
        frames_dir,
        tmp_path / "thinned",
        every_nth=1,
        ordering=FrameOrderingPolicy.EXTRACTION_MANIFEST,
    )

    manifest = json.loads((tmp_path / "thinned" / THINNING_MANIFEST_FILENAME).read_text())
    assert result.input_count == 1
    assert manifest["missing_manifest_entries"] == 1
    assert (tmp_path / "thinned" / "good.jpg").is_file()


def test_output_copy_mode(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    source = frames_dir / "a.jpg"
    _write_frames(frames_dir, ["a.jpg"])
    source.write_bytes(b"original")

    thin_frame_folder(frames_dir, tmp_path / "thinned", every_nth=1)

    output = tmp_path / "thinned" / "a.jpg"
    assert output.is_file()
    source.write_bytes(b"changed")
    assert output.read_bytes() == b"original"


def test_refuses_non_empty_output_dir(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    output_dir = tmp_path / "thinned"
    _write_frames(frames_dir, ["a.jpg"])
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        thin_frame_folder(frames_dir, output_dir, every_nth=2)


def test_refuses_existing_manifest(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    output_dir = tmp_path / "thinned"
    _write_frames(frames_dir, ["a.jpg"])
    output_dir.mkdir()
    (output_dir / THINNING_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="existing manifest"):
        thin_frame_folder(frames_dir, output_dir, every_nth=2)


def test_manifest_records_kept_and_removed(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["a.jpg", "b.jpg", "c.jpg"])

    thin_frame_folder(frames_dir, tmp_path / "thinned", every_nth=2)

    manifest = json.loads((tmp_path / "thinned" / THINNING_MANIFEST_FILENAME).read_text())
    assert manifest["input_count"] == 3
    assert manifest["kept_count"] == 2
    assert manifest["removed_count"] == 1
    assert manifest["kept"][0]["source"] == "a.jpg"
    assert manifest["removed"][0]["source"] == "b.jpg"
    assert manifest["output"]["mode"] == "copy"


def test_non_image_files_ignored(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames(frames_dir, ["a.jpg", "b.jpg"])
    (frames_dir / MANIFEST_FILENAME).write_text('{"frames": []}', encoding="utf-8")
    (frames_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    result = thin_frame_folder(
        frames_dir,
        tmp_path / "thinned",
        every_nth=1,
        ordering=FrameOrderingPolicy.FILENAME_NATURAL,
    )

    assert result.input_count == 2


def test_cli_thin_frames(tmp_path: Path, capsys, monkeypatch) -> None:
    frames_dir = tmp_path / "frames"
    output_dir = tmp_path / "thinned"
    _write_frames(frames_dir, ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"])

    monkeypatch.setattr(
        "sys.argv",
        [
            "prep360",
            "thin-frames",
            "--input",
            str(frames_dir),
            "--output",
            str(output_dir),
            "--every-nth",
            "2",
        ],
    )
    exit_code = cli_main()

    assert exit_code == 0
    assert (output_dir / THINNING_MANIFEST_FILENAME).is_file()
    assert len(list(output_dir.glob("*.jpg"))) == 3
    captured = capsys.readouterr()
    assert "Thinned 5 frames -> 3 kept" in captured.out


def test_cli_refuses_invalid_every_nth(tmp_path: Path, capsys, monkeypatch) -> None:
    frames_dir = tmp_path / "frames"
    output_dir = tmp_path / "thinned"
    _write_frames(frames_dir, ["a.jpg"])

    monkeypatch.setattr(
        "sys.argv",
        [
            "prep360",
            "thin-frames",
            "--input",
            str(frames_dir),
            "--output",
            str(output_dir),
            "--every-nth",
            "0",
        ],
    )
    exit_code = cli_main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "every_nth must be" in captured.err
