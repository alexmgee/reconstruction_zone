"""Native COLMAP binary reader for pose-only and full sparse models."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .colmap_types import (
    ColmapCamera,
    ColmapFullModel,
    ColmapImagePose,
    ColmapImageRecord,
    ColmapModelCompleteness,
    ColmapPoint2D,
    ColmapPoint3DRecord,
    ColmapPoseModel,
    ColmapReadMode,
    ColmapTrackElement,
)

# Upstream COLMAP camera model IDs 0-17 (name, num_params).
# Verified against pycolmap 4.1.0 (CameraModelId + Camera.create param counts).
COLMAP_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
    12: ("SIMPLE_DIVISION", 4),
    13: ("DIVISION", 5),
    14: ("SIMPLE_FISHEYE", 3),
    15: ("FISHEYE", 4),
    16: ("EUCM", 6),
    17: ("EQUIRECTANGULAR", 2),
}

# Legacy SphereSfM-fork table: IDs 0-10 match upstream, but the fork assigned
# 11 to its SPHERE model (3 params) — upstream 11 is RAD_TAN_THIN_PRISM_FISHEYE
# (16 params). The variant is trusted out-of-band provenance (the caller knows
# which engine produced the model); reading with the wrong table typically
# fails fast on an unknown ID or a size mismatch, but that is defense in
# depth, not a guarantee — crafted multi-camera files can realign by chance.
SPHERESFM_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    **{k: v for k, v in COLMAP_CAMERA_MODELS.items() if k <= 10},
    11: ("SPHERE", 3),
}

CAMERA_MODEL_TABLES: dict[str, dict[int, tuple[str, int]]] = {
    "colmap": COLMAP_CAMERA_MODELS,
    "spheresfm": SPHERESFM_CAMERA_MODELS,
}

COLMAP_INVALID_POINT3D_ID_BINARY = 2**64 - 1
_OBSERVATION_RECORD_SIZE = 24  # float64 x, float64 y, uint64 point3D_id


@dataclass(frozen=True)
class _ParsedImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    center: np.ndarray
    points2d: tuple[ColmapPoint2D, ...]


def _camera_model_table(variant: str) -> dict[int, tuple[str, int]]:
    if variant not in CAMERA_MODEL_TABLES:
        raise ValueError(
            f"Unknown camera-model variant: {variant!r} "
            f"(expected one of {sorted(CAMERA_MODEL_TABLES)})",
        )
    return CAMERA_MODEL_TABLES[variant]


def read_colmap_pose_model_binary(
    model_dir: str | Path,
    *,
    variant: str,
) -> ColmapPoseModel:
    """Read cameras.bin and images.bin as pose-only COLMAP data.

    variant selects the camera-model ID namespace: "colmap" (upstream 0-17)
    or "spheresfm" (legacy fork, 11=SPHERE). No default — callers must state
    which engine produced the model.
    Does not require points3D.bin.
    Does not retain observations.
    Raises FileNotFoundError/ValueError on missing or malformed input.
    """
    camera_models = _camera_model_table(variant)
    model_path = Path(model_dir)
    cameras_path = model_path / "cameras.bin"
    images_path = model_path / "images.bin"

    if not cameras_path.is_file():
        raise FileNotFoundError(f"Missing COLMAP binary file: {cameras_path}")
    if not images_path.is_file():
        raise FileNotFoundError(f"Missing COLMAP binary file: {images_path}")

    cameras = _parse_cameras_bin(cameras_path.read_bytes(), camera_models, variant)
    parsed_images = _parse_images_bin(images_path.read_bytes(), retain_observations=False)

    points_path = model_path / "points3D.bin"
    completeness = (
        ColmapModelCompleteness.FULL
        if points_path.is_file()
        else ColmapModelCompleteness.POSE_ONLY
    )

    images = {
        image_id: ColmapImagePose(
            image_id=image.image_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            center=image.center,
        )
        for image_id, image in parsed_images.items()
    }

    return ColmapPoseModel(
        cameras=cameras,
        images=images,
        source_dir=model_path,
        read_mode=ColmapReadMode.POSE_ONLY,
        completeness=completeness,
    )


def read_colmap_full_model_binary(
    model_dir: str | Path,
    *,
    variant: str,
) -> ColmapFullModel:
    """Read cameras.bin, images.bin, and points3D.bin as a full COLMAP model.

    variant selects the camera-model ID namespace: "colmap" (upstream 0-17)
    or "spheresfm" (legacy fork, 11=SPHERE). No default — callers must state
    which engine produced the model.
    """
    camera_models = _camera_model_table(variant)
    model_path = Path(model_dir)
    cameras_path = model_path / "cameras.bin"
    images_path = model_path / "images.bin"
    points_path = model_path / "points3D.bin"

    if not cameras_path.is_file():
        raise FileNotFoundError(f"Missing COLMAP binary file: {cameras_path}")
    if not images_path.is_file():
        raise FileNotFoundError(f"Missing COLMAP binary file: {images_path}")
    if not points_path.is_file():
        raise FileNotFoundError(f"Missing COLMAP binary file: {points_path}")

    cameras = _parse_cameras_bin(cameras_path.read_bytes(), camera_models, variant)
    parsed_images = _parse_images_bin(images_path.read_bytes(), retain_observations=True)
    points3d = _parse_points3d_bin(points_path.read_bytes())

    images = {
        image_id: ColmapImageRecord(
            image_id=image.image_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            center=image.center,
            points2d=image.points2d,
        )
        for image_id, image in parsed_images.items()
    }

    return ColmapFullModel(
        cameras=cameras,
        images=images,
        points3d=points3d,
        source_dir=model_path,
        read_mode=ColmapReadMode.FULL,
        completeness=ColmapModelCompleteness.FULL,
    )


def normalize_point3d_id(raw_point3d_id: int) -> int:
    """Map COLMAP binary/pycolmap invalid sentinel to canonical core -1."""
    if raw_point3d_id == COLMAP_INVALID_POINT3D_ID_BINARY:
        return -1
    if raw_point3d_id < 0:
        return -1
    return int(raw_point3d_id)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(qvec))
    if norm == 0.0:
        raise ValueError("Invalid zero-length COLMAP quaternion")
    qw, qx, qy, qz = qvec / norm
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
    ])


def _camera_center_from_pose(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rot = qvec_to_rotmat(qvec)
    return -rot.T @ tvec


def _parse_cameras_bin(
    data: bytes,
    camera_models: dict[int, tuple[str, int]],
    variant: str,
) -> dict[int, ColmapCamera]:
    if len(data) < 8:
        raise ValueError("Truncated cameras.bin: missing camera count")

    offset = 0
    camera_count, = struct.unpack_from("<Q", data, offset)
    offset += 8

    cameras: dict[int, ColmapCamera] = {}
    for _ in range(camera_count):
        if offset + 24 > len(data):
            raise ValueError("Truncated cameras.bin: incomplete camera header")

        camera_id, model_id, width, height = struct.unpack_from(
            "<iiQQ", data, offset,
        )
        offset += 24

        if model_id not in camera_models:
            raise ValueError(
                f"Unknown COLMAP camera model id: {model_id} "
                f"under variant '{variant}' — if this model came from the "
                f"other engine, re-read it with the matching variant",
            )

        model_name, num_params = camera_models[model_id]
        params_size = num_params * 8
        if offset + params_size > len(data):
            raise ValueError(
                f"Truncated cameras.bin: incomplete params for camera {camera_id}",
            )

        params = struct.unpack_from(f"<{num_params}d", data, offset)
        offset += params_size

        cameras[camera_id] = ColmapCamera(
            camera_id=camera_id,
            model_name=model_name,
            width=int(width),
            height=int(height),
            params=tuple(float(p) for p in params),
        )

    if offset != len(data):
        raise ValueError("Truncated cameras.bin: unexpected trailing bytes")

    return cameras


def _parse_images_bin(
    data: bytes,
    *,
    retain_observations: bool,
) -> dict[int, _ParsedImage]:
    if len(data) < 8:
        raise ValueError("Truncated images.bin: missing image count")

    offset = 0
    image_count, = struct.unpack_from("<Q", data, offset)
    offset += 8

    images: dict[int, _ParsedImage] = {}
    for _ in range(image_count):
        if offset + 68 > len(data):
            raise ValueError("Truncated images.bin: incomplete image header")

        image_id = struct.unpack_from("<i", data, offset)[0]
        offset += 4

        qvec = np.array(struct.unpack_from("<4d", data, offset), dtype=np.float64)
        offset += 32

        tvec = np.array(struct.unpack_from("<3d", data, offset), dtype=np.float64)
        offset += 24

        camera_id = struct.unpack_from("<i", data, offset)[0]
        offset += 4

        name, offset = _read_null_terminated_string(data, offset)

        if offset + 8 > len(data):
            raise ValueError(
                f"Truncated images.bin: missing num_points2D for image {image_id}",
            )

        num_points2d, = struct.unpack_from("<Q", data, offset)
        offset += 8

        observations_size = num_points2d * _OBSERVATION_RECORD_SIZE
        if offset + observations_size > len(data):
            raise ValueError(
                f"Truncated images.bin: incomplete observations for image {image_id}",
            )

        points2d: tuple[ColmapPoint2D, ...] = ()
        if retain_observations and num_points2d:
            observations: list[ColmapPoint2D] = []
            for _obs_index in range(num_points2d):
                x, y, raw_point3d_id = struct.unpack_from("<ddQ", data, offset)
                offset += _OBSERVATION_RECORD_SIZE
                observations.append(ColmapPoint2D(
                    x=float(x),
                    y=float(y),
                    point3d_id=normalize_point3d_id(int(raw_point3d_id)),
                ))
            points2d = tuple(observations)
        else:
            offset += observations_size

        center = _camera_center_from_pose(qvec, tvec)
        images[image_id] = _ParsedImage(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            center=center,
            points2d=points2d,
        )

    if offset != len(data):
        raise ValueError("Truncated images.bin: unexpected trailing bytes")

    return images


def _parse_points3d_bin(data: bytes) -> dict[int, ColmapPoint3DRecord]:
    if len(data) < 8:
        raise ValueError("Truncated points3D.bin: missing point count")

    offset = 0
    point_count, = struct.unpack_from("<Q", data, offset)
    offset += 8

    points: dict[int, ColmapPoint3DRecord] = {}
    for _ in range(point_count):
        if offset + 51 > len(data):
            raise ValueError("Truncated points3D.bin: incomplete point header")

        point3d_id, = struct.unpack_from("<Q", data, offset)
        offset += 8

        x, y, z = struct.unpack_from("<3d", data, offset)
        offset += 24

        r, g, b = struct.unpack_from("<BBB", data, offset)
        offset += 3

        error, = struct.unpack_from("<d", data, offset)
        offset += 8

        track_length, = struct.unpack_from("<Q", data, offset)
        offset += 8

        track_size = track_length * 8
        if offset + track_size > len(data):
            raise ValueError(
                f"Truncated points3D.bin: incomplete track for point {point3d_id}",
            )

        track: list[ColmapTrackElement] = []
        for _ in range(track_length):
            image_id, point2d_idx = struct.unpack_from("<ii", data, offset)
            offset += 8
            track.append(ColmapTrackElement(
                image_id=int(image_id),
                point2d_idx=int(point2d_idx),
            ))

        points[int(point3d_id)] = ColmapPoint3DRecord(
            point3d_id=int(point3d_id),
            xyz=np.array([x, y, z], dtype=np.float64),
            rgb=(int(r), int(g), int(b)),
            error=float(error),
            track=tuple(track),
        )

    if offset != len(data):
        raise ValueError("Truncated points3D.bin: unexpected trailing bytes")

    return points


def _read_null_terminated_string(data: bytes, offset: int) -> tuple[str, int]:
    end = data.find(b"\x00", offset)
    if end == -1:
        raise ValueError("Truncated images.bin: unterminated image name")
    name = data[offset:end].decode("utf-8")
    return name, end + 1
