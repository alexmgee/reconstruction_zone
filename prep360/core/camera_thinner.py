"""Pose-aware camera thinning for ordered image folders with COLMAP poses."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .colmap_binary import qvec_to_rotmat, read_colmap_pose_model_binary
from .frame_thinner import (
    THINNING_MANIFEST_FILENAME,
    FrameOrderingPolicy,
    discover_frame_files,
)

POSE_SOURCE_KIND_COLMAP = "colmap"
THINNING_MODE_POSE_AWARE = "pose_aware"


@dataclass(frozen=True)
class ColmapPoseRecord:
    image_id: int
    image_name: str
    position: np.ndarray
    forward: np.ndarray
    qvec: np.ndarray
    tvec: np.ndarray


@dataclass(frozen=True)
class CameraThinningEntry:
    row_id: str
    image_name: str
    source_path: Path
    relative_path: str
    order_index: int
    image_id: int | None
    position: np.ndarray
    forward: np.ndarray


@dataclass(frozen=True)
class CameraThinningDecision:
    row_id: str
    image_name: str
    source: str
    output: str | None
    order_index: int
    keep: bool
    reference_row_id: str | None
    distance: float | None
    angle_degrees: float | None
    reason: str
    image_id: int | None = None
    position: tuple[float, float, float] | None = None
    forward: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class CameraThinningPlan:
    input_count: int
    matched_count: int
    kept_count: int
    removed_count: int
    unmatched_image_count: int
    unmatched_pose_count: int
    decisions: tuple[CameraThinningDecision, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ThinCamerasResult:
    input_count: int
    matched_count: int
    kept_count: int
    removed_count: int
    unmatched_image_count: int
    unmatched_pose_count: int
    output_dir: Path
    manifest_path: Path


def colmap_camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Return COLMAP camera center in world coordinates."""
    rot_w2c = qvec_to_rotmat(qvec)
    t = np.asarray(tvec, dtype=np.float64)
    return -rot_w2c.T @ t


def colmap_forward_world(qvec: np.ndarray) -> np.ndarray:
    """Return unit camera forward vector in world coordinates."""
    rot_w2c = qvec_to_rotmat(qvec)
    rot_c2w = rot_w2c.T
    forward = rot_c2w @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-12:
        raise ValueError("Camera forward vector has near-zero norm")
    return forward / norm


def angle_between_vectors_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angle between two vectors in degrees."""
    a_vec = np.asarray(a, dtype=np.float64)
    b_vec = np.asarray(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(a_vec))
    norm_b = float(np.linalg.norm(b_vec))
    if norm_a < 1e-12 or norm_b < 1e-12:
        raise ValueError("Cannot compute angle for near-zero vector")
    cos_theta = float(np.dot(a_vec, b_vec) / (norm_a * norm_b))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def find_colmap_images_txt(colmap_sparse: Path) -> Path | None:
    candidates = [
        colmap_sparse / "images.txt",
        colmap_sparse / "sparse" / "images.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    sparse_dirs = sorted(colmap_sparse.glob("sparse/*/images.txt"))
    if sparse_dirs:
        return sparse_dirs[0]
    return None


def find_colmap_binary_model_dir(colmap_sparse: Path) -> Path | None:
    candidates = [colmap_sparse, colmap_sparse / "sparse"]
    for candidate in candidates:
        if (candidate / "cameras.bin").is_file() and (candidate / "images.bin").is_file():
            return candidate
    sparse_dirs = sorted(colmap_sparse.glob("sparse/*"))
    for candidate in sparse_dirs:
        if (candidate / "cameras.bin").is_file() and (candidate / "images.bin").is_file():
            return candidate
    return None


def load_colmap_poses(colmap_sparse: str | Path) -> dict[str, ColmapPoseRecord]:
    """Load COLMAP poses keyed by COLMAP image name."""
    sparse_path = Path(colmap_sparse).resolve()
    if not sparse_path.exists():
        raise FileNotFoundError(f"COLMAP sparse path not found: {sparse_path}")

    images_txt = find_colmap_images_txt(sparse_path)
    if images_txt is not None:
        return _load_colmap_poses_from_text(images_txt)

    binary_dir = find_colmap_binary_model_dir(sparse_path)
    if binary_dir is not None:
        return _load_colmap_poses_from_binary(binary_dir)

    raise FileNotFoundError(
        f"No COLMAP pose files found under {sparse_path}. "
        "Expected images.txt or cameras.bin + images.bin.",
    )


def _load_colmap_poses_from_text(images_txt: Path) -> dict[str, ColmapPoseRecord]:
    poses: dict[str, ColmapPoseRecord] = {}
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        image_id = int(parts[0])
        qvec = np.array(
            [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
            dtype=np.float64,
        )
        tvec = np.array(
            [float(parts[5]), float(parts[6]), float(parts[7])],
            dtype=np.float64,
        )
        name = parts[9]

        if index < len(lines):
            index += 1

        poses[name] = ColmapPoseRecord(
            image_id=image_id,
            image_name=name,
            position=colmap_camera_center(qvec, tvec),
            forward=colmap_forward_world(qvec),
            qvec=qvec,
            tvec=tvec,
        )
    return poses


def _load_colmap_poses_from_binary(model_dir: Path) -> dict[str, ColmapPoseRecord]:
    model = read_colmap_pose_model_binary(model_dir)
    poses: dict[str, ColmapPoseRecord] = {}
    for image in model.images.values():
        poses[image.name] = ColmapPoseRecord(
            image_id=image.image_id,
            image_name=image.name,
            position=np.asarray(image.center, dtype=np.float64),
            forward=colmap_forward_world(image.qvec),
            qvec=np.asarray(image.qvec, dtype=np.float64),
            tvec=np.asarray(image.tvec, dtype=np.float64),
        )
    return poses


def _posix_relative(path: str) -> str:
    return path.replace("\\", "/")


def _row_id_for_frame(relative_path: str, order_index: int) -> str:
    return f"{order_index:05d}:{_posix_relative(relative_path)}"


def match_source_images_to_poses(
    frames: Sequence,
    poses: dict[str, ColmapPoseRecord],
) -> tuple[list[CameraThinningEntry], int, int, list[str]]:
    """Match discovered source frames to COLMAP poses in source order."""
    exact_index = {_posix_relative(name): pose for name, pose in poses.items()}
    basename_map: dict[str, list[str]] = {}
    for name in poses:
        basename = Path(name).name
        basename_map.setdefault(basename, []).append(name)

    ambiguous_basenames = {
        basename
        for basename, names in basename_map.items()
        if len(names) > 1
    }
    if ambiguous_basenames:
        joined = ", ".join(sorted(ambiguous_basenames))
        raise ValueError(
            f"Ambiguous COLMAP image basename match for: {joined}",
        )

    matched_entries: list[CameraThinningEntry] = []
    matched_pose_names: set[str] = set()
    unmatched_image_count = 0
    warnings: list[str] = []

    for frame in frames:
        relative = _posix_relative(frame.relative_path)
        pose = exact_index.get(relative)
        if pose is None:
            basename = Path(relative).name
            pose_name = basename_map.get(basename, [None])[0]
            pose = poses.get(pose_name) if pose_name is not None else None

        if pose is None:
            unmatched_image_count += 1
            warnings.append(
                f"No COLMAP pose matched source image {relative}; image excluded.",
            )
            continue

        matched_pose_names.add(pose.image_name)
        matched_entries.append(CameraThinningEntry(
            row_id=_row_id_for_frame(frame.relative_path, frame.order_index),
            image_name=pose.image_name,
            source_path=frame.source_path,
            relative_path=frame.relative_path,
            order_index=frame.order_index,
            image_id=pose.image_id,
            position=np.asarray(pose.position, dtype=np.float64),
            forward=np.asarray(pose.forward, dtype=np.float64),
        ))

    unmatched_pose_count = len(poses) - len(matched_pose_names)
    if unmatched_pose_count:
        warnings.append(
            f"{unmatched_pose_count} COLMAP pose(s) had no matching source image.",
        )

    return matched_entries, unmatched_image_count, unmatched_pose_count, warnings


def select_pose_aware_entries(
    entries: Sequence[CameraThinningEntry],
    *,
    max_distance: float,
    max_angle_degrees: float,
) -> CameraThinningPlan:
    """Select entries using sequential distance/angle threshold thinning."""
    if max_distance < 0:
        raise ValueError("max_distance must be greater than or equal to 0")
    if max_angle_degrees < 0:
        raise ValueError("max_angle_degrees must be greater than or equal to 0")
    if not entries:
        raise ValueError("No thinning entries provided")

    decisions: list[CameraThinningDecision] = []
    anchor = entries[0]
    decisions.append(_decision_for_entry(
        anchor,
        keep=True,
        reference_row_id=None,
        distance=None,
        angle_degrees=None,
        reason="first_entry",
    ))

    for entry in entries[1:]:
        distance = float(np.linalg.norm(entry.position - anchor.position))
        angle_degrees = angle_between_vectors_degrees(entry.forward, anchor.forward)
        remove = distance <= max_distance and angle_degrees <= max_angle_degrees
        if remove:
            decisions.append(_decision_for_entry(
                entry,
                keep=False,
                reference_row_id=anchor.row_id,
                distance=distance,
                angle_degrees=angle_degrees,
                reason="below_thresholds",
            ))
        else:
            decisions.append(_decision_for_entry(
                entry,
                keep=True,
                reference_row_id=anchor.row_id,
                distance=distance,
                angle_degrees=angle_degrees,
                reason="above_distance_or_angle",
            ))
            anchor = entry

    kept_count = sum(1 for decision in decisions if decision.keep)
    return CameraThinningPlan(
        input_count=len(entries),
        matched_count=len(entries),
        kept_count=kept_count,
        removed_count=len(entries) - kept_count,
        unmatched_image_count=0,
        unmatched_pose_count=0,
        decisions=tuple(decisions),
        warnings=(),
    )


def thin_camera_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    colmap_sparse: str | Path,
    max_distance: float,
    max_angle_degrees: float,
    ordering: FrameOrderingPolicy = FrameOrderingPolicy.AUTO,
) -> ThinCamerasResult:
    """Copy pose-thinned images from input_dir to output_dir."""
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    sparse_path = Path(colmap_sparse).resolve()

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

    frames = discover_frame_files(input_path, ordering=ordering)
    if not frames:
        raise ValueError(f"No image files found in {input_path}")

    poses = load_colmap_poses(sparse_path)
    if not poses:
        raise ValueError(f"No COLMAP poses found under {sparse_path}")

    matched_entries, unmatched_image_count, unmatched_pose_count, warnings = (
        match_source_images_to_poses(frames, poses)
    )
    if not matched_entries:
        raise ValueError(
            "No source images matched COLMAP poses; cannot thin camera folder.",
        )

    plan = select_pose_aware_entries(
        matched_entries,
        max_distance=max_distance,
        max_angle_degrees=max_angle_degrees,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    entry_by_row_id = {entry.row_id: entry for entry in matched_entries}
    for decision in plan.decisions:
        if not decision.keep:
            continue
        entry = entry_by_row_id[decision.row_id]
        destination = output_path / entry.relative_path
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output file: {destination}",
            )
        shutil.copy2(entry.source_path, destination)

    manifest_path = _write_pose_aware_manifest(
        input_path=input_path,
        output_path=output_path,
        sparse_path=sparse_path,
        ordering=ordering,
        max_distance=max_distance,
        max_angle_degrees=max_angle_degrees,
        input_count=len(frames),
        matched_count=len(matched_entries),
        unmatched_image_count=unmatched_image_count,
        unmatched_pose_count=unmatched_pose_count,
        plan=plan,
        warnings=warnings,
    )

    return ThinCamerasResult(
        input_count=len(frames),
        matched_count=len(matched_entries),
        kept_count=plan.kept_count,
        removed_count=plan.removed_count,
        unmatched_image_count=unmatched_image_count,
        unmatched_pose_count=unmatched_pose_count,
        output_dir=output_path,
        manifest_path=manifest_path,
    )


def _decision_for_entry(
    entry: CameraThinningEntry,
    *,
    keep: bool,
    reference_row_id: str | None,
    distance: float | None,
    angle_degrees: float | None,
    reason: str,
) -> CameraThinningDecision:
    return CameraThinningDecision(
        row_id=entry.row_id,
        image_name=entry.image_name,
        source=_posix_relative(entry.relative_path),
        output=_posix_relative(entry.relative_path) if keep else None,
        order_index=entry.order_index,
        keep=keep,
        reference_row_id=reference_row_id,
        distance=distance,
        angle_degrees=angle_degrees,
        reason=reason,
        image_id=entry.image_id,
        position=tuple(float(v) for v in entry.position),
        forward=tuple(float(v) for v in entry.forward),
    )


def _write_pose_aware_manifest(
    *,
    input_path: Path,
    output_path: Path,
    sparse_path: Path,
    ordering: FrameOrderingPolicy,
    max_distance: float,
    max_angle_degrees: float,
    input_count: int,
    matched_count: int,
    unmatched_image_count: int,
    unmatched_pose_count: int,
    plan: CameraThinningPlan,
    warnings: list[str],
) -> Path:
    manifest_path = output_path / THINNING_MANIFEST_FILENAME
    kept = []
    removed = []
    for decision in plan.decisions:
        entry = {
            "order_index": decision.order_index,
            "source": decision.source,
            "image_name": decision.image_name,
            "image_id": decision.image_id,
            "position": list(decision.position) if decision.position is not None else None,
            "forward": list(decision.forward) if decision.forward is not None else None,
            "reference": decision.reference_row_id,
            "distance": decision.distance,
            "angle_degrees": decision.angle_degrees,
            "reason": decision.reason,
        }
        if decision.keep:
            entry["output"] = decision.output
            kept.append(entry)
        else:
            removed.append(entry)

    payload = {
        "schema_version": 1,
        "mode": THINNING_MODE_POSE_AWARE,
        "selection": {
            "max_distance": max_distance,
            "max_angle_degrees": max_angle_degrees,
            "ordering": ordering.value,
        },
        "pose_source": {
            "kind": POSE_SOURCE_KIND_COLMAP,
            "path": str(sparse_path),
        },
        "output": {
            "mode": "copy",
        },
        "source_dir": str(input_path),
        "output_dir": str(output_path),
        "input_count": input_count,
        "matched_count": matched_count,
        "kept_count": plan.kept_count,
        "removed_count": plan.removed_count,
        "unmatched_image_count": unmatched_image_count,
        "unmatched_pose_count": unmatched_pose_count,
        "kept": kept,
        "removed": removed,
        "warnings": list(warnings),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path
