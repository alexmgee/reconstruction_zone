"""Temporal frame thinning for ordered image folders."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from .extractor import MANIFEST_FILENAME

THINNING_MANIFEST_FILENAME = "thinning_manifest.json"

IMAGE_EXTENSIONS = frozenset({
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
})


class FrameOrderingPolicy(Enum):
    AUTO = "auto"
    EXTRACTION_MANIFEST = "extraction_manifest"
    FILENAME_NATURAL = "filename_natural"


@dataclass(frozen=True)
class FrameRef:
    source_path: Path
    relative_path: str
    order_index: int
    manifest_index: int | None = None
    time_sec: float | None = None


@dataclass(frozen=True)
class ThinFramesResult:
    input_count: int
    kept_count: int
    removed_count: int
    unlisted_on_disk_count: int
    output_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class _DiscoveryResult:
    frames: list[FrameRef]
    ordering_used: FrameOrderingPolicy
    source_manifest: str
    missing_manifest_entries: int
    unlisted_on_disk_count: int
    warnings: list[str]


def discover_frame_files(
    input_dir: str | Path,
    *,
    ordering: FrameOrderingPolicy = FrameOrderingPolicy.AUTO,
) -> list[FrameRef]:
    return _discover_frames(input_dir, ordering=ordering).frames


def select_every_nth(
    frames: Sequence[FrameRef],
    every_nth: int,
) -> tuple[list[FrameRef], list[FrameRef]]:
    if every_nth < 1:
        raise ValueError("every_nth must be an integer greater than or equal to 1")

    kept = [frame for frame in frames if frame.order_index % every_nth == 0]
    removed = [frame for frame in frames if frame.order_index % every_nth != 0]
    return kept, removed


def thin_frame_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    every_nth: int,
    ordering: FrameOrderingPolicy = FrameOrderingPolicy.AUTO,
) -> ThinFramesResult:
    if every_nth < 1:
        raise ValueError("every_nth must be an integer greater than or equal to 1")

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    manifest_output_path = output_path / THINNING_MANIFEST_FILENAME
    if manifest_output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {manifest_output_path}",
        )
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty output directory: {output_path}",
        )

    discovery = _discover_frames(input_path, ordering=ordering)
    if not discovery.frames:
        raise ValueError(f"No image files found in {input_path}")

    kept, removed = select_every_nth(discovery.frames, every_nth)

    output_path.mkdir(parents=True, exist_ok=True)
    for frame in kept:
        destination = output_path / frame.relative_path
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output file: {destination}",
            )
        shutil.copy2(frame.source_path, destination)

    manifest_path = _write_thinning_manifest(
        input_path=input_path,
        output_path=output_path,
        discovery=discovery,
        every_nth=every_nth,
        kept=kept,
        removed=removed,
    )

    return ThinFramesResult(
        input_count=len(discovery.frames),
        kept_count=len(kept),
        removed_count=len(removed),
        unlisted_on_disk_count=discovery.unlisted_on_disk_count,
        output_dir=output_path,
        manifest_path=manifest_path,
    )


def _discover_frames(
    input_dir: str | Path,
    *,
    ordering: FrameOrderingPolicy,
) -> _DiscoveryResult:
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    disk_images = _list_top_level_images(input_path)

    if ordering == FrameOrderingPolicy.FILENAME_NATURAL:
        return _discover_natural_order(input_path, disk_images)

    manifest_path = input_path / MANIFEST_FILENAME
    if ordering == FrameOrderingPolicy.EXTRACTION_MANIFEST:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Extraction manifest not found: {manifest_path}",
            )
        return _discover_manifest_order(
            input_path,
            disk_images,
            manifest_path,
            allow_empty=False,
        )

    if manifest_path.is_file():
        manifest_result = _discover_manifest_order(
            input_path,
            disk_images,
            manifest_path,
            allow_empty=True,
        )
        if manifest_result.frames:
            return manifest_result

    return _discover_natural_order(input_path, disk_images)


def _discover_natural_order(
    input_path: Path,
    disk_images: list[Path],
) -> _DiscoveryResult:
    sorted_images = sorted(disk_images, key=lambda path: _natural_sort_key(path.name))
    frames = [
        FrameRef(
            source_path=path,
            relative_path=path.name,
            order_index=index,
        )
        for index, path in enumerate(sorted_images)
    ]
    return _DiscoveryResult(
        frames=frames,
        ordering_used=FrameOrderingPolicy.FILENAME_NATURAL,
        source_manifest="",
        missing_manifest_entries=0,
        unlisted_on_disk_count=0,
        warnings=[],
    )


def _discover_manifest_order(
    input_path: Path,
    disk_images: list[Path],
    manifest_path: Path,
    *,
    allow_empty: bool = True,
) -> _DiscoveryResult:
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        if allow_empty:
            return _DiscoveryResult([], FrameOrderingPolicy.FILENAME_NATURAL, "", 0, 0, [])
        raise ValueError(f"Invalid extraction manifest: {manifest_path}") from exc

    frames_data = manifest_data.get("frames")
    if not isinstance(frames_data, list):
        if allow_empty:
            return _DiscoveryResult([], FrameOrderingPolicy.FILENAME_NATURAL, "", 0, 0, [])
        raise ValueError(f"Invalid extraction manifest: {manifest_path}")

    disk_by_name = {path.name: path for path in disk_images}
    manifest_filenames: set[str] = set()
    frames: list[FrameRef] = []
    missing_manifest_entries = 0
    order_index = 0

    for entry in frames_data:
        if not isinstance(entry, dict):
            missing_manifest_entries += 1
            continue

        filename = entry.get("filename")
        if not isinstance(filename, str) or not _is_valid_manifest_basename(filename):
            missing_manifest_entries += 1
            continue

        manifest_filenames.add(filename)
        source_path = disk_by_name.get(filename)
        if source_path is None:
            missing_manifest_entries += 1
            continue

        manifest_index = entry.get("index")
        if manifest_index is not None and not isinstance(manifest_index, int):
            missing_manifest_entries += 1
            continue

        time_sec = entry.get("time_sec")
        if time_sec is not None and not isinstance(time_sec, (int, float)):
            missing_manifest_entries += 1
            continue

        frames.append(FrameRef(
            source_path=source_path,
            relative_path=filename,
            order_index=order_index,
            manifest_index=manifest_index,
            time_sec=float(time_sec) if time_sec is not None else None,
        ))
        order_index += 1

    if not frames and not allow_empty:
        raise ValueError(
            f"Extraction manifest references no existing images: {manifest_path}",
        )

    unlisted_on_disk_count = sum(
        1 for path in disk_images if path.name not in manifest_filenames
    )
    warnings: list[str] = []
    if unlisted_on_disk_count:
        warnings.append(
            f"{unlisted_on_disk_count} image file(s) were present on disk but absent "
            f"from {MANIFEST_FILENAME} and were not considered.",
        )

    return _DiscoveryResult(
        frames=frames,
        ordering_used=FrameOrderingPolicy.EXTRACTION_MANIFEST,
        source_manifest=str(manifest_path.resolve()),
        missing_manifest_entries=missing_manifest_entries,
        unlisted_on_disk_count=unlisted_on_disk_count,
        warnings=warnings,
    )


def _list_top_level_images(input_dir: Path) -> list[Path]:
    images: list[Path] = []
    for entry in input_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry)
    return images


def _is_valid_manifest_basename(filename: str) -> bool:
    if not filename:
        return False
    if "/" in filename or "\\" in filename:
        return False
    if re.match(r"^[A-Za-z]:", filename):
        return False
    return filename == Path(filename).name


def _natural_sort_key(name: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", name.lower())
    ]


def _write_thinning_manifest(
    *,
    input_path: Path,
    output_path: Path,
    discovery: _DiscoveryResult,
    every_nth: int,
    kept: Sequence[FrameRef],
    removed: Sequence[FrameRef],
) -> Path:
    manifest_path = output_path / THINNING_MANIFEST_FILENAME
    payload = {
        "schema_version": 1,
        "mode": "temporal",
        "selection": {
            "every_nth": every_nth,
            "ordering": discovery.ordering_used.value,
        },
        "output": {
            "mode": "copy",
        },
        "source_dir": str(input_path),
        "output_dir": str(output_path),
        "source_manifest": discovery.source_manifest,
        "input_count": len(discovery.frames),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "missing_manifest_entries": discovery.missing_manifest_entries,
        "unlisted_on_disk_count": discovery.unlisted_on_disk_count,
        "kept": [
            _frame_manifest_entry(frame, include_output=True)
            for frame in kept
        ],
        "removed": [
            _frame_manifest_entry(frame, include_output=False)
            for frame in removed
        ],
        "warnings": list(discovery.warnings),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _frame_manifest_entry(frame: FrameRef, *, include_output: bool) -> dict:
    entry = {
        "order_index": frame.order_index,
        "source": _posix_relative(frame.relative_path),
    }
    if include_output:
        entry["output"] = _posix_relative(frame.relative_path)
    if frame.time_sec is not None:
        entry["time_sec"] = frame.time_sec
    return entry


def _posix_relative(path: str) -> str:
    return path.replace("\\", "/")
