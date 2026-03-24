"""
Helpers for preparing raw dual-fisheye datasets for Metashape.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DualFisheyePairRecord:
    """Canonical description of one front/back capture pair."""

    pair_index: int
    frame_id: str
    front_image: Path
    back_image: Path
    time_sec: Optional[float] = None
    source_front_frame: Optional[int] = None
    source_back_frame: Optional[int] = None
    front_mask: Optional[Path] = None
    back_mask: Optional[Path] = None
    source_key: Optional[str] = None


@dataclass(frozen=True)
class DualFisheyeClipDataset:
    """One validated clip working folder for a dual-fisheye session."""

    clip_id: str
    clip_root: Path
    frames_root: Path
    masks_root: Optional[Path]
    pair_records: list[DualFisheyePairRecord]


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES


def _iter_image_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if _is_image_file(path))


def _lens_dirs(root: Path, kind: str) -> tuple[Path, Path]:
    if kind == "images":
        candidates = [
            (root / "front" / "images", root / "back" / "images"),
            (root / "front" / "frames", root / "back" / "frames"),
            (root / "images" / "front", root / "images" / "back"),
            (root / "frames" / "front", root / "frames" / "back"),
            (root / "front", root / "back"),
        ]
        expected = (
            f"{root / 'front'} + {root / 'back'} or "
            f"{root / 'images' / 'front'} + {root / 'images' / 'back'} or "
            f"{root / 'frames' / 'front'} + {root / 'frames' / 'back'} or "
            f"{root / 'front' / 'frames'} + {root / 'back' / 'frames'}"
        )
    elif kind == "masks":
        candidates = [
            (root / "masks" / "front", root / "masks" / "back"),
            (root / "front" / "masks", root / "back" / "masks"),
            (root / "front", root / "back"),
            (root / "front_masks", root / "back_masks"),
        ]
        expected = (
            f"{root / 'masks' / 'front'} + {root / 'masks' / 'back'} or "
            f"{root / 'front' / 'masks'} + {root / 'back' / 'masks'} or "
            f"{root / 'front'} + {root / 'back'} or "
            f"{root / 'front_masks'} + {root / 'back_masks'}"
        )
    else:
        candidates = [
            (root / kind / "front", root / kind / "back"),
            (root / "front" / kind, root / "back" / kind),
            (root / "front", root / "back"),
        ]
        expected = (
            f"{root / kind / 'front'} + {root / kind / 'back'} or "
            f"{root / 'front' / kind} + {root / 'back' / kind} or "
            f"{root / 'front'} + {root / 'back'}"
        )

    for front_dir, back_dir in candidates:
        if front_dir.is_dir() and back_dir.is_dir():
            return front_dir, back_dir

    raise FileNotFoundError(
        f"Could not find paired {kind} directories under {root}. Expected {expected}."
    )


def _normalize_match_key(path: Path, *, is_mask: bool) -> str:
    stem = path.stem
    if is_mask:
        stem = re.sub(r"(?i)^mask[_-]", "", stem)
        stem = re.sub(r"(?i)[_-]mask$", "", stem)
    stem = re.sub(r"(?i)^(front|back)[_-]", "", stem)
    stem = re.sub(r"(?i)[_-](front|back)$", "", stem)
    digits = re.findall(r"\d+", stem)
    if digits:
        return f"{int(digits[-1]):012d}"
    return stem.lower()


def _normalize_source_key(value: str | None, *, fallback_path: Path) -> str:
    if value:
        digits = re.findall(r"\d+", value)
        if digits:
            return f"{int(digits[-1]):012d}"
        return value.strip().lower()
    return _normalize_match_key(fallback_path, is_mask=False)


def _build_path_map(paths: Iterable[Path], *, is_mask: bool) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in paths:
        key = _normalize_match_key(path, is_mask=is_mask)
        if key in mapping:
            raise ValueError(
                f"Duplicate source key '{key}' for {mapping[key].name} and {path.name}"
            )
        mapping[key] = path
    return mapping


def _format_id_mismatches(front_keys: set[str], back_keys: set[str]) -> str:
    front_only = sorted(front_keys - back_keys)[:5]
    back_only = sorted(back_keys - front_keys)[:5]
    parts = []
    if front_only:
        parts.append(f"front-only IDs: {', '.join(front_only)}")
    if back_only:
        parts.append(f"back-only IDs: {', '.join(back_only)}")
    return "; ".join(parts)


def discover_image_pairs(root: str | Path) -> tuple[list[Path], list[Path], list[str]]:
    """Discover paired front/back images from nested or legacy flat folders."""

    root_path = Path(root)

    try:
        front_dir, back_dir = _lens_dirs(root_path, "images")
        front_map = _build_path_map(_iter_image_files(front_dir), is_mask=False)
        back_map = _build_path_map(_iter_image_files(back_dir), is_mask=False)
    except FileNotFoundError:
        flat_front = sorted(
            path for path in root_path.iterdir()
            if _is_image_file(path) and re.match(r"(?i)^front[_-]", path.stem)
        )
        flat_back = sorted(
            path for path in root_path.iterdir()
            if _is_image_file(path) and re.match(r"(?i)^back[_-]", path.stem)
        )
        if not flat_front and not flat_back:
            raise
        front_map = _build_path_map(flat_front, is_mask=False)
        back_map = _build_path_map(flat_back, is_mask=False)

    if not front_map or not back_map:
        raise ValueError("Need both front and back images to build a dual-fisheye dataset.")

    if set(front_map) != set(back_map):
        mismatch = _format_id_mismatches(set(front_map), set(back_map))
        raise ValueError(f"Front/back frame IDs do not match: {mismatch}")

    source_keys = sorted(front_map.keys())
    front_paths = [front_map[key] for key in source_keys]
    back_paths = [back_map[key] for key in source_keys]
    return front_paths, back_paths, source_keys


def build_pair_records(
    front_paths: Sequence[str | Path],
    back_paths: Sequence[str | Path],
    *,
    time_secs: Optional[Sequence[Optional[float]]] = None,
    source_front_frames: Optional[Sequence[Optional[int]]] = None,
    source_back_frames: Optional[Sequence[Optional[int]]] = None,
    source_keys: Optional[Sequence[Optional[str]]] = None,
) -> list[DualFisheyePairRecord]:
    """Build pair records from ordered front/back paths."""

    if len(front_paths) != len(back_paths):
        raise ValueError(
            f"Front/back image count mismatch: {len(front_paths)} != {len(back_paths)}"
        )

    pair_records: list[DualFisheyePairRecord] = []
    for index, (front_path, back_path) in enumerate(zip(front_paths, back_paths), start=1):
        pair_records.append(
            DualFisheyePairRecord(
                pair_index=index,
                frame_id=f"{index:06d}",
                front_image=Path(front_path),
                back_image=Path(back_path),
                time_sec=(time_secs[index - 1] if time_secs is not None else None),
                source_front_frame=(
                    source_front_frames[index - 1]
                    if source_front_frames is not None else index - 1
                ),
                source_back_frame=(
                    source_back_frames[index - 1]
                    if source_back_frames is not None else index - 1
                ),
                source_key=(source_keys[index - 1] if source_keys is not None else None),
            )
        )
    return pair_records


def load_pair_records_from_manifest(
    frames_root: str | Path,
) -> Optional[list[DualFisheyePairRecord]]:
    """Load pair records from a paired split extraction manifest, if present."""

    root_path = Path(frames_root)
    manifest_path = root_path / "paired_extraction_manifest.json"
    if not manifest_path.exists():
        return None

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = data.get("pairs", [])
    pair_records: list[DualFisheyePairRecord] = []
    for index, entry in enumerate(pairs, start=1):
        front_image = root_path / entry["front_image"]
        source_key = _normalize_source_key(
            entry.get("frame_id", f"{index:06d}"),
            fallback_path=front_image,
        )
        pair_records.append(
            DualFisheyePairRecord(
                pair_index=index,
                frame_id=entry.get("frame_id", f"{index:06d}"),
                front_image=front_image,
                back_image=root_path / entry["back_image"],
                time_sec=entry.get("time_sec"),
                source_front_frame=entry.get("source_front_frame"),
                source_back_frame=entry.get("source_back_frame"),
                source_key=source_key,
            )
        )
    return pair_records


def filter_pair_records(
    pair_records: Sequence[DualFisheyePairRecord],
    kept_indices: Sequence[int],
) -> list[DualFisheyePairRecord]:
    """Keep only selected records and renumber them sequentially."""

    seen = set()
    selected: list[DualFisheyePairRecord] = []
    for idx in kept_indices:
        if idx < 0 or idx >= len(pair_records):
            raise IndexError(f"Selected pair index out of range: {idx}")
        if idx in seen:
            continue
        selected.append(pair_records[idx])
        seen.add(idx)

    renumbered: list[DualFisheyePairRecord] = []
    for new_index, record in enumerate(selected, start=1):
        renumbered.append(
            replace(record, pair_index=new_index, frame_id=f"{new_index:06d}")
        )
    return renumbered


def attach_masks_to_pair_records(
    pair_records: Sequence[DualFisheyePairRecord],
    masks_root: str | Path,
) -> list[DualFisheyePairRecord]:
    """Attach discovered masks to existing pair records."""

    root_path = Path(masks_root)
    front_dir, back_dir = _lens_dirs(root_path, "masks")
    front_map = _build_path_map(_iter_image_files(front_dir), is_mask=True)
    back_map = _build_path_map(_iter_image_files(back_dir), is_mask=True)

    updated: list[DualFisheyePairRecord] = []
    for record in pair_records:
        source_key = _normalize_source_key(record.source_key, fallback_path=record.front_image)
        updated.append(
            replace(
                record,
                front_mask=front_map.get(source_key),
                back_mask=back_map.get(source_key),
            )
        )
    return updated


def validate_pair_records(
    pair_records: Sequence[DualFisheyePairRecord],
    *,
    require_masks: bool = False,
) -> None:
    """Validate image/mask integrity for the final dataset."""

    seen_ids: set[str] = set()
    any_masks = any(record.front_mask or record.back_mask for record in pair_records)

    for record in pair_records:
        if record.frame_id in seen_ids:
            raise ValueError(f"Duplicate frame_id: {record.frame_id}")
        seen_ids.add(record.frame_id)

        if not record.front_image.exists():
            raise FileNotFoundError(f"Missing front image: {record.front_image}")
        if not record.back_image.exists():
            raise FileNotFoundError(f"Missing back image: {record.back_image}")

        if record.front_mask is None and record.back_mask is None:
            if require_masks:
                raise FileNotFoundError(f"Missing masks for frame_id {record.frame_id}")
            continue

        if record.front_mask is None or record.back_mask is None:
            raise FileNotFoundError(
                f"One-sided mask pair for frame_id {record.frame_id}"
            )
        if not record.front_mask.exists():
            raise FileNotFoundError(f"Missing front mask: {record.front_mask}")
        if not record.back_mask.exists():
            raise FileNotFoundError(f"Missing back mask: {record.back_mask}")

    if require_masks and not any_masks:
        raise FileNotFoundError("Masks were required, but no mask files were attached.")


def _looks_like_clip_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (
        ((path / "frames" / "front").is_dir() and (path / "frames" / "back").is_dir())
        or ((path / "front" / "frames").is_dir() and (path / "back" / "frames").is_dir())
    )


def _resolve_clip_frames_root(clip_path: Path) -> Path:
    if (clip_path / "front" / "frames").is_dir() and (clip_path / "back" / "frames").is_dir():
        return clip_path
    if (clip_path / "frames" / "front").is_dir() and (clip_path / "frames" / "back").is_dir():
        return clip_path / "frames"
    raise FileNotFoundError(
        f"Could not find paired frames for clip {clip_path}. Expected either "
        f"{clip_path / 'front' / 'frames'} + {clip_path / 'back' / 'frames'} or "
        f"{clip_path / 'frames' / 'front'} + {clip_path / 'frames' / 'back'}."
    )


def _resolve_clip_masks_root(clip_path: Path) -> Optional[Path]:
    candidates = [
        clip_path,
        clip_path / "masks",
    ]
    for candidate in candidates:
        try:
            _lens_dirs(candidate, "masks")
            return candidate
        except FileNotFoundError:
            continue
    return None


def discover_clip_roots(root: str | Path) -> list[Path]:
    """Discover one or more clip working folders under *root*."""

    root_path = Path(root)
    if _looks_like_clip_root(root_path):
        return [root_path]

    clip_roots = sorted(
        child for child in root_path.iterdir()
        if _looks_like_clip_root(child)
    )
    if not clip_roots:
        raise FileNotFoundError(
            f"Could not find any clip folders under {root_path}. "
            "Expected either a clip root containing front/frames + back/frames "
            "or frames/front + frames/back, "
            "or a parent folder whose child folders do."
        )
    return clip_roots


def load_clip_dataset(
    clip_root: str | Path,
    *,
    require_masks: bool = False,
) -> DualFisheyeClipDataset:
    """Load one clip working folder without copying any files."""

    clip_path = Path(clip_root)
    frames_root = _resolve_clip_frames_root(clip_path)
    masks_root = _resolve_clip_masks_root(clip_path)
    has_masks = masks_root is not None

    manifest_records = load_pair_records_from_manifest(frames_root)
    if manifest_records is not None:
        pair_records = manifest_records
    else:
        front_paths, back_paths, source_keys = discover_image_pairs(frames_root)
        pair_records = build_pair_records(
            front_paths,
            back_paths,
            source_front_frames=list(range(len(front_paths))),
            source_back_frames=list(range(len(back_paths))),
            source_keys=source_keys,
        )

    if has_masks:
        pair_records = attach_masks_to_pair_records(pair_records, masks_root)

    validate_pair_records(pair_records, require_masks=require_masks)
    return DualFisheyeClipDataset(
        clip_id=clip_path.name,
        clip_root=clip_path,
        frames_root=frames_root,
        masks_root=masks_root if has_masks else None,
        pair_records=list(pair_records),
    )


def _check_clear_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing dataset content in {path}. "
            f"Choose an empty output folder."
        )


def ensure_dataset_root_is_empty(dataset_root: str | Path, *, include_masks: bool) -> None:
    """Ensure the final dataset root does not already contain files."""

    root_path = Path(dataset_root)
    _check_clear_directory(root_path / "images" / "front")
    _check_clear_directory(root_path / "images" / "back")
    _check_clear_directory(root_path / "masks" / "front")
    _check_clear_directory(root_path / "masks" / "back")
    manifest_path = root_path / "dual_fisheye_manifest.json"
    info_path = root_path / "dataset_info.json"
    if manifest_path.exists() or info_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset metadata in {root_path}. "
            f"Choose an empty output folder."
        )


def finalize_pair_records(
    pair_records: Sequence[DualFisheyePairRecord],
    dataset_root: str | Path,
) -> list[DualFisheyePairRecord]:
    """Copy images and masks into the final dataset layout."""

    root_path = Path(dataset_root)
    has_masks = any(record.front_mask or record.back_mask for record in pair_records)
    ensure_dataset_root_is_empty(root_path, include_masks=has_masks)

    front_images_dir = root_path / "images" / "front"
    back_images_dir = root_path / "images" / "back"
    front_images_dir.mkdir(parents=True, exist_ok=True)
    back_images_dir.mkdir(parents=True, exist_ok=True)

    front_masks_dir = root_path / "masks" / "front"
    back_masks_dir = root_path / "masks" / "back"
    if has_masks:
        front_masks_dir.mkdir(parents=True, exist_ok=True)
        back_masks_dir.mkdir(parents=True, exist_ok=True)

    finalized: list[DualFisheyePairRecord] = []
    for record in pair_records:
        front_dest = front_images_dir / f"{record.frame_id}{record.front_image.suffix.lower()}"
        back_dest = back_images_dir / f"{record.frame_id}{record.back_image.suffix.lower()}"
        shutil.copy2(record.front_image, front_dest)
        shutil.copy2(record.back_image, back_dest)

        front_mask_dest = None
        back_mask_dest = None
        if record.front_mask and record.back_mask:
            front_mask_dest = front_masks_dir / f"{record.frame_id}_mask{record.front_mask.suffix.lower()}"
            back_mask_dest = back_masks_dir / f"{record.frame_id}_mask{record.back_mask.suffix.lower()}"
            shutil.copy2(record.front_mask, front_mask_dest)
            shutil.copy2(record.back_mask, back_mask_dest)

        finalized.append(
            replace(
                record,
                front_image=front_dest,
                back_image=back_dest,
                front_mask=front_mask_dest,
                back_mask=back_mask_dest,
            )
        )
    return finalized


def _relative_path(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    return path.relative_to(root).as_posix()


def _manifest_path_value(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _slugify_session_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    safe = safe.strip("._-")
    return safe or "dual_fisheye_session"


def write_dual_fisheye_manifest(
    dataset_root: str | Path,
    pair_records: Sequence[DualFisheyePairRecord],
    *,
    source_video: Optional[str] = None,
    camera_model: Optional[str] = None,
    container_type: Optional[str] = None,
    front_stream_index: Optional[int] = None,
    back_stream_index: Optional[int] = None,
    frame_interval_sec: Optional[float] = None,
    selection_mode: str = "none",
) -> Path:
    """Write the machine-readable dual-fisheye dataset manifest."""

    root_path = Path(dataset_root)
    has_masks = any(record.front_mask or record.back_mask for record in pair_records)

    manifest = {
        "schema_version": 1,
        "dataset_type": "dual_fisheye_metashape",
        "source_video": source_video,
        "source_video_stem": Path(source_video).stem if source_video else None,
        "camera_model": camera_model,
        "container_type": container_type,
        "front_stream_index": front_stream_index,
        "back_stream_index": back_stream_index,
        "frame_interval_sec": frame_interval_sec,
        "selection_mode": selection_mode,
        "images_dir": "images",
        "masks_dir": "masks" if has_masks else None,
        "pairs": [],
    }

    for record in pair_records:
        manifest["pairs"].append(
            {
                "pair_index": record.pair_index,
                "frame_id": record.frame_id,
                "front_image": _relative_path(record.front_image, root_path),
                "back_image": _relative_path(record.back_image, root_path),
                "front_mask": _relative_path(record.front_mask, root_path),
                "back_mask": _relative_path(record.back_mask, root_path),
                "time_sec": record.time_sec,
                "source_front_frame": record.source_front_frame,
                "source_back_frame": record.source_back_frame,
            }
        )

    manifest_path = root_path / "dual_fisheye_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_dataset_info(
    dataset_root: str | Path,
    pair_records: Sequence[DualFisheyePairRecord],
    *,
    source_video: Optional[str] = None,
    source_frames_dir: Optional[str] = None,
    source_masks_dir: Optional[str] = None,
    selection_mode: str = "none",
    frame_interval_sec: Optional[float] = None,
) -> Path:
    """Write a human-readable dataset sidecar with summary metadata."""

    root_path = Path(dataset_root)
    has_masks = any(record.front_mask or record.back_mask for record in pair_records)

    info = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": source_video,
        "source_frames_dir": source_frames_dir,
        "source_masks_dir": source_masks_dir,
        "selection_mode": selection_mode,
        "frame_interval_sec": frame_interval_sec,
        "pair_count": len(pair_records),
        "mask_pair_count": sum(
            1 for record in pair_records
            if record.front_mask is not None and record.back_mask is not None
        ),
        "has_masks": has_masks,
        "frame_ids": [record.frame_id for record in pair_records],
    }

    info_path = root_path / "dataset_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info_path


def write_dual_fisheye_session_manifest(
    session_root: str | Path,
    clip_datasets: Sequence[DualFisheyeClipDataset],
    *,
    session_name: Optional[str] = None,
) -> Path:
    """Write a no-copy session manifest that references existing clip folders."""

    if not clip_datasets:
        raise ValueError("Need at least one clip dataset to assemble a session manifest.")

    root_path = Path(session_root)
    root_path.mkdir(parents=True, exist_ok=True)

    session_name = session_name or f"{root_path.name}_metashape_session"
    manifest_path = root_path / f"{_slugify_session_name(session_name)}.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing session manifest: {manifest_path}"
        )

    manifest = {
        "schema_version": 1,
        "dataset_type": "dual_fisheye_metashape_session",
        "session_name": session_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clip_count": len(clip_datasets),
        "pair_count": 0,
        "clips": [],
        "pairs": [],
    }

    global_pair_index = 1
    for clip in clip_datasets:
        has_masks = any(
            record.front_mask is not None and record.back_mask is not None
            for record in clip.pair_records
        )
        manifest["clips"].append(
            {
                "clip_id": clip.clip_id,
                "clip_root": _manifest_path_value(clip.clip_root, root_path),
                "frames_root": _manifest_path_value(clip.frames_root, root_path),
                "masks_root": _manifest_path_value(clip.masks_root, root_path),
                "pair_count": len(clip.pair_records),
                "has_masks": has_masks,
            }
        )

        for record in clip.pair_records:
            pair_id = f"{clip.clip_id}__{record.frame_id}"
            manifest["pairs"].append(
                {
                    "global_pair_index": global_pair_index,
                    "pair_id": pair_id,
                    "clip_id": clip.clip_id,
                    "clip_pair_index": record.pair_index,
                    "clip_frame_id": record.frame_id,
                    "front_image": _manifest_path_value(record.front_image, root_path),
                    "back_image": _manifest_path_value(record.back_image, root_path),
                    "front_mask": _manifest_path_value(record.front_mask, root_path),
                    "back_mask": _manifest_path_value(record.back_mask, root_path),
                    "time_sec": record.time_sec,
                    "source_front_frame": record.source_front_frame,
                    "source_back_frame": record.source_back_frame,
                }
            )
            global_pair_index += 1

    manifest["pair_count"] = len(manifest["pairs"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


__all__ = [
    "DualFisheyeClipDataset",
    "DualFisheyePairRecord",
    "attach_masks_to_pair_records",
    "build_pair_records",
    "discover_clip_roots",
    "discover_image_pairs",
    "ensure_dataset_root_is_empty",
    "filter_pair_records",
    "finalize_pair_records",
    "load_clip_dataset",
    "load_pair_records_from_manifest",
    "validate_pair_records",
    "write_dataset_info",
    "write_dual_fisheye_manifest",
    "write_dual_fisheye_session_manifest",
]
