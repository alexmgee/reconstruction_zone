"""Read-only detection of camera/pose artifact candidates in folder trees."""

from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

IGNORED_DIR_NAMES = frozenset({
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    "dist",
    "build",
})
IGNORED_DIR_NAMES_LOWER = frozenset(name.lower() for name in IGNORED_DIR_NAMES)


class CameraFormat(Enum):
    COLMAP_BIN_FULL = "colmap_bin_full"
    COLMAP_BIN_POSE_ONLY = "colmap_bin_pose_only"
    COLMAP_TXT_FULL = "colmap_txt_full"
    COLMAP_TXT_POSE_ONLY = "colmap_txt_pose_only"
    METASHAPE_XML = "metashape_xml"
    TRANSFORMS_JSON = "transforms_json"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CameraCandidate:
    kind: CameraFormat
    path: str
    label: str
    confidence: float
    root_dir: str = ""
    cameras_path: str = ""
    images_path: str = ""
    points_path: str = ""
    frame_count: int | None = None
    camera_count: int | None = None
    notes: str = ""


def scan_camera_candidates(
    root: str | Path,
    *,
    max_depth: int = 8,
) -> list[CameraCandidate]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {root_path}")

    candidates: list[CameraCandidate] = []
    _scan_directory(
        root_path.resolve(),
        root_path.resolve(),
        depth=0,
        max_depth=max_depth,
        candidates=candidates,
    )
    return _sort_candidates(candidates)


def candidate_to_dict(candidate: CameraCandidate) -> dict:
    payload = {
        "kind": candidate.kind.value,
        "path": candidate.path,
        "label": candidate.label,
        "confidence": candidate.confidence,
    }
    if candidate.root_dir:
        payload["root_dir"] = candidate.root_dir
    if candidate.cameras_path:
        payload["cameras_path"] = candidate.cameras_path
    if candidate.images_path:
        payload["images_path"] = candidate.images_path
    if candidate.points_path:
        payload["points_path"] = candidate.points_path
    if candidate.frame_count is not None:
        payload["frame_count"] = candidate.frame_count
    if candidate.camera_count is not None:
        payload["camera_count"] = candidate.camera_count
    if candidate.notes:
        payload["notes"] = candidate.notes
    return payload


def candidates_to_dicts(candidates: Sequence[CameraCandidate]) -> list[dict]:
    return [candidate_to_dict(candidate) for candidate in candidates]


def _scan_directory(
    scan_root: Path,
    current_dir: Path,
    *,
    depth: int,
    max_depth: int,
    candidates: list[CameraCandidate],
) -> None:
    if depth > max_depth:
        return

    try:
        candidates.extend(_detect_colmap_in_dir(current_dir, scan_root))
        transforms_candidate = _detect_transforms_json(current_dir / "transforms.json", scan_root)
        if transforms_candidate is not None:
            candidates.append(transforms_candidate)

        for entry in sorted(current_dir.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_symlink():
                continue
            if entry.is_file() and entry.suffix.lower() == ".xml":
                xml_candidate = _detect_metashape_xml(entry, scan_root)
                if xml_candidate is not None:
                    candidates.append(xml_candidate)
            elif entry.is_dir() and not _should_skip_directory(entry.name):
                _scan_directory(
                    scan_root,
                    entry,
                    depth=depth + 1,
                    max_depth=max_depth,
                    candidates=candidates,
                )
    except OSError:
        return


def _detect_colmap_in_dir(model_dir: Path, scan_root: Path) -> list[CameraCandidate]:
    candidates: list[CameraCandidate] = []
    resolved_dir = model_dir.resolve()

    bin_cameras = model_dir / "cameras.bin"
    bin_images = model_dir / "images.bin"
    bin_points = model_dir / "points3D.bin"
    if bin_cameras.is_file() and bin_images.is_file():
        kind = (
            CameraFormat.COLMAP_BIN_FULL
            if bin_points.is_file()
            else CameraFormat.COLMAP_BIN_POSE_ONLY
        )
        notes = ""
        camera_count = _read_binary_count(bin_cameras)
        frame_count = _read_binary_count(bin_images)
        if camera_count is None or frame_count is None:
            notes = "Could not read COLMAP binary counts"
        candidates.append(CameraCandidate(
            kind=kind,
            path=str(resolved_dir),
            label=_colmap_label(kind),
            confidence=1.0,
            root_dir=str(scan_root.resolve()),
            cameras_path=str(bin_cameras.resolve()),
            images_path=str(bin_images.resolve()),
            points_path=str(bin_points.resolve()) if bin_points.is_file() else "",
            frame_count=frame_count,
            camera_count=camera_count,
            notes=notes,
        ))

    txt_cameras = model_dir / "cameras.txt"
    txt_images = model_dir / "images.txt"
    txt_points = model_dir / "points3D.txt"
    if txt_cameras.is_file() and txt_images.is_file():
        kind = (
            CameraFormat.COLMAP_TXT_FULL
            if txt_points.is_file()
            else CameraFormat.COLMAP_TXT_POSE_ONLY
        )
        notes = ""
        camera_count = _count_colmap_cameras_txt(txt_cameras)
        frame_count = _count_colmap_images_txt(txt_images)
        if camera_count is None or frame_count is None:
            notes = "Could not read COLMAP text counts"
        candidates.append(CameraCandidate(
            kind=kind,
            path=str(resolved_dir),
            label=_colmap_label(kind),
            confidence=1.0,
            root_dir=str(scan_root.resolve()),
            cameras_path=str(txt_cameras.resolve()),
            images_path=str(txt_images.resolve()),
            points_path=str(txt_points.resolve()) if txt_points.is_file() else "",
            frame_count=frame_count,
            camera_count=camera_count,
            notes=notes,
        ))

    return candidates


def _detect_transforms_json(path: Path, scan_root: Path) -> CameraCandidate | None:
    if path.name != "transforms.json" or not path.is_file():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    frames = data.get("frames")
    if not isinstance(frames, list):
        return None

    if frames:
        if not any(
            isinstance(frame, dict) and "transform_matrix" in frame
            for frame in frames
        ):
            return None
    elif not any(key in data for key in ("camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy")):
        return None

    return CameraCandidate(
        kind=CameraFormat.TRANSFORMS_JSON,
        path=str(path.resolve()),
        label="transforms.json",
        confidence=1.0,
        root_dir=str(scan_root.resolve()),
        frame_count=len(frames),
        notes="",
    )


def _detect_metashape_xml(path: Path, scan_root: Path) -> CameraCandidate | None:
    if path.suffix.lower() != ".xml" or not path.is_file():
        return None

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except (ET.ParseError, OSError):
        return None

    total_cameras = 0
    found_chunk_with_cameras = False

    for chunk in root.iter():
        if _local_tag(chunk) != "chunk":
            continue

        for cameras_elem in chunk:
            if _local_tag(cameras_elem) != "cameras":
                continue
            camera_elems = [
                child for child in cameras_elem
                if _local_tag(child) == "camera"
            ]
            if camera_elems:
                found_chunk_with_cameras = True
                total_cameras += len(camera_elems)

    if not found_chunk_with_cameras:
        return None

    sensor_count = sum(1 for elem in root.iter() if _local_tag(elem) == "sensor")
    notes = ""
    if sensor_count:
        notes = f"{sensor_count} sensor element(s) found"

    return CameraCandidate(
        kind=CameraFormat.METASHAPE_XML,
        path=str(path.resolve()),
        label=path.name,
        confidence=1.0,
        root_dir=str(scan_root.resolve()),
        frame_count=total_cameras,
        camera_count=sensor_count or None,
        notes=notes,
    )


def _read_binary_count(path: Path) -> int | None:
    try:
        data = path.read_bytes()
        if len(data) < 8:
            return None
        count, = struct.unpack_from("<Q", data, 0)
        return int(count)
    except OSError:
        return None


def _count_colmap_cameras_txt(path: Path) -> int | None:
    try:
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            try:
                int(parts[0])
                int(parts[2])
                int(parts[3])
            except ValueError:
                continue
            count += 1
        return count
    except (OSError, UnicodeDecodeError):
        return None


def _count_colmap_images_txt(path: Path) -> int | None:
    try:
        count = 0
        lines = path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            index += 1
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            try:
                int(parts[0])
                float(parts[1])
                float(parts[2])
                float(parts[3])
                float(parts[4])
                float(parts[5])
                float(parts[6])
                float(parts[7])
                int(parts[8])
            except ValueError:
                continue

            count += 1
            if index < len(lines):
                index += 1

        return count
    except (OSError, UnicodeDecodeError):
        return None


def _should_skip_directory(dir_name: str) -> bool:
    if dir_name.startswith("."):
        return True
    return dir_name.lower() in IGNORED_DIR_NAMES_LOWER


def _colmap_label(kind: CameraFormat) -> str:
    labels = {
        CameraFormat.COLMAP_BIN_FULL: "COLMAP binary (full)",
        CameraFormat.COLMAP_BIN_POSE_ONLY: "COLMAP binary (pose-only)",
        CameraFormat.COLMAP_TXT_FULL: "COLMAP text (full)",
        CameraFormat.COLMAP_TXT_POSE_ONLY: "COLMAP text (pose-only)",
    }
    return labels.get(kind, kind.value)


def _local_tag(element: ET.Element) -> str:
    tag = element.tag
    return tag.split("}")[-1] if "}" in tag else tag


def _sort_candidates(candidates: list[CameraCandidate]) -> list[CameraCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            Path(candidate.path).as_posix().lower(),
            candidate.kind.value,
        ),
    )
