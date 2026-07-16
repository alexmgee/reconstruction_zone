"""COLMAP core data types for binary pose and full model reading."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class ColmapReadMode(Enum):
    POSE_ONLY = "pose_only"
    FULL = "full"


class ColmapModelCompleteness(Enum):
    FULL = "full"
    POSE_ONLY = "pose_only"
    INVALID = "invalid"


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model_name: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    center: np.ndarray


@dataclass(frozen=True)
class ColmapPoseModel:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImagePose]
    source_dir: Path
    read_mode: ColmapReadMode
    completeness: ColmapModelCompleteness


@dataclass(frozen=True)
class ColmapPoint2D:
    x: float
    y: float
    point3d_id: int


@dataclass(frozen=True)
class ColmapTrackElement:
    image_id: int
    point2d_idx: int


@dataclass(frozen=True)
class ColmapImageRecord:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    center: np.ndarray
    points2d: tuple[ColmapPoint2D, ...]


@dataclass(frozen=True)
class ColmapPoint3DRecord:
    point3d_id: int
    xyz: np.ndarray
    rgb: tuple[int, int, int]
    error: float
    track: tuple[ColmapTrackElement, ...]


@dataclass(frozen=True)
class ColmapFullModel:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImageRecord]
    points3d: dict[int, ColmapPoint3DRecord]
    source_dir: Path
    read_mode: ColmapReadMode
    completeness: ColmapModelCompleteness
