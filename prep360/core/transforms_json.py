"""Export nerfstudio-first transforms.json from COLMAP pose models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .colmap_binary import qvec_to_rotmat, read_colmap_pose_model_binary
from .colmap_types import ColmapCamera, ColmapImagePose, ColmapPoseModel

SUPPORTED_PERSPECTIVE_CAMERA_MODELS = frozenset({
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
})

UNSUPPORTED_CAMERA_MODELS = frozenset({
    "SPHERE",
    "OPENCV_FISHEYE",
    "FULL_OPENCV",
    "FOV",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
    "THIN_PRISM_FISHEYE",
})


@dataclass(frozen=True)
class TransformsFrame:
    file_path: str
    transform_matrix: list[list[float]]
    w: int | None = None
    h: int | None = None
    fl_x: float | None = None
    fl_y: float | None = None
    cx: float | None = None
    cy: float | None = None
    k1: float | None = None
    k2: float | None = None
    k3: float | None = None
    p1: float | None = None
    p2: float | None = None


@dataclass(frozen=True)
class TransformsDataset:
    frames: list[TransformsFrame]
    camera_model: str | None = None
    w: int | None = None
    h: int | None = None
    fl_x: float | None = None
    fl_y: float | None = None
    cx: float | None = None
    cy: float | None = None
    k1: float | None = None
    k2: float | None = None
    k3: float | None = None
    p1: float | None = None
    p2: float | None = None


@dataclass(frozen=True)
class _CameraIntrinsics:
    w: int
    h: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    k1: float | None = None
    k2: float | None = None
    k3: float | None = None
    p1: float | None = None
    p2: float | None = None


def normalize_qvec(qvec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(qvec))
    if norm == 0.0:
        raise ValueError("Invalid zero-length COLMAP quaternion")
    return qvec / norm


def colmap_w2c_to_nerf_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP world-to-camera pose to OpenGL/nerfstudio camera-to-world."""
    qvec = normalize_qvec(np.asarray(qvec, dtype=np.float64))
    tvec = np.asarray(tvec, dtype=np.float64)
    rot_w2c = qvec_to_rotmat(qvec)
    rot_c2w = rot_w2c.T
    center = -rot_c2w @ tvec

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rot_c2w @ np.diag([1.0, -1.0, -1.0])
    c2w[:3, 3] = center
    return c2w


def colmap_pose_model_to_transforms(
    model: ColmapPoseModel,
    *,
    image_prefix: str = "images",
) -> TransformsDataset:
    if not model.images:
        raise ValueError("COLMAP model contains no images")

    use_root_intrinsics = len(model.cameras) == 1
    root_intrinsics: _CameraIntrinsics | None = None
    if use_root_intrinsics:
        root_camera = next(iter(model.cameras.values()))
        root_intrinsics = _camera_intrinsics_from_colmap(root_camera)

    frames: list[TransformsFrame] = []
    for image_id in sorted(model.images):
        image = model.images[image_id]
        camera = model.cameras.get(image.camera_id)
        if camera is None:
            raise ValueError(
                f"Image {image_id} ({image.name!r}) references missing camera_id "
                f"{image.camera_id}",
            )

        intrinsics = _camera_intrinsics_from_colmap(camera)
        c2w = colmap_w2c_to_nerf_c2w(image.qvec, image.tvec)
        frame_kwargs: dict = {
            "file_path": _frame_file_path(image.name, image_prefix),
            "transform_matrix": _matrix_to_nested_list(c2w),
        }
        if not use_root_intrinsics:
            frame_kwargs.update(_intrinsics_to_frame_fields(intrinsics))
        frames.append(TransformsFrame(**frame_kwargs))

    dataset_kwargs: dict = {
        "frames": frames,
        "camera_model": "OPENCV",
    }
    if root_intrinsics is not None:
        dataset_kwargs.update(_intrinsics_to_root_fields(root_intrinsics))
    return TransformsDataset(**dataset_kwargs)


def write_transforms_json(dataset: TransformsDataset, output_path: str | Path) -> None:
    payload = _dataset_to_json_dict(dataset)
    path = Path(output_path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def export_colmap_binary_to_transforms_json(
    sparse_dir: str | Path,
    output_path: str | Path,
    *,
    image_prefix: str = "images",
    overwrite: bool = False,
) -> int:
    sparse_path = Path(sparse_dir)
    output = Path(output_path)

    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    _ensure_binary_colmap_sparse(sparse_path)
    model = read_colmap_pose_model_binary(sparse_path, variant="colmap")
    dataset = colmap_pose_model_to_transforms(model, image_prefix=image_prefix)
    write_transforms_json(dataset, output)
    return len(dataset.frames)


def _ensure_binary_colmap_sparse(sparse_dir: Path) -> None:
    has_binary = (
        (sparse_dir / "cameras.bin").is_file()
        and (sparse_dir / "images.bin").is_file()
    )
    if has_binary:
        return

    has_text = (
        (sparse_dir / "cameras.txt").is_file()
        or (sparse_dir / "images.txt").is_file()
    )
    if has_text:
        raise ValueError(
            "B1 transforms-json export requires binary COLMAP (cameras.bin + images.bin); "
            "text-model export is deferred.",
        )


def _camera_intrinsics_from_colmap(camera: ColmapCamera) -> _CameraIntrinsics:
    model_name = camera.model_name
    if model_name in UNSUPPORTED_CAMERA_MODELS:
        raise ValueError(
            f"Camera model {model_name} is not supported by B1 transforms.json export. "
            "Use a perspective/cubemap export packet instead.",
        )
    if model_name not in SUPPORTED_PERSPECTIVE_CAMERA_MODELS:
        raise ValueError(
            f"Camera model {model_name} is not supported by B1 transforms.json export.",
        )

    params = camera.params

    if model_name == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return _CameraIntrinsics(
            w=camera.width,
            h=camera.height,
            fl_x=f,
            fl_y=f,
            cx=cx,
            cy=cy,
        )
    if model_name == "PINHOLE":
        fx, fy, cx, cy = params
        return _CameraIntrinsics(
            w=camera.width,
            h=camera.height,
            fl_x=fx,
            fl_y=fy,
            cx=cx,
            cy=cy,
        )
    if model_name == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        return _CameraIntrinsics(
            w=camera.width,
            h=camera.height,
            fl_x=f,
            fl_y=f,
            cx=cx,
            cy=cy,
            k1=_optional_distortion(k1),
        )
    if model_name == "RADIAL":
        f, cx, cy, k1, k2 = params
        return _CameraIntrinsics(
            w=camera.width,
            h=camera.height,
            fl_x=f,
            fl_y=f,
            cx=cx,
            cy=cy,
            k1=_optional_distortion(k1),
            k2=_optional_distortion(k2),
        )
    if model_name == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        return _CameraIntrinsics(
            w=camera.width,
            h=camera.height,
            fl_x=fx,
            fl_y=fy,
            cx=cx,
            cy=cy,
            k1=_optional_distortion(k1),
            k2=_optional_distortion(k2),
            p1=_optional_distortion(p1),
            p2=_optional_distortion(p2),
        )

    raise ValueError(
        f"Camera model {model_name} is not supported by B1 transforms.json export.",
    )


def _optional_distortion(value: float) -> float | None:
    if abs(value) < 1e-12:
        return None
    return float(value)


def _is_absolute_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    if normalized.startswith("//"):
        return True
    if re.match(r"^[A-Za-z]:/", normalized):
        return True
    return False


def _frame_file_path(name: str, image_prefix: str) -> str:
    if _is_absolute_path(name):
        raise ValueError(
            f"COLMAP image name must be a relative path, got: {name!r}",
        )

    normalized_name = name.replace("\\", "/")
    if not image_prefix:
        return normalized_name

    if _is_absolute_path(image_prefix):
        raise ValueError(
            f"image_prefix must be a relative path, got: {image_prefix!r}",
        )

    prefix = image_prefix.replace("\\", "/").rstrip("/")
    return f"{prefix}/{normalized_name}"


def _matrix_to_nested_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.tolist()]


def _intrinsics_to_root_fields(intrinsics: _CameraIntrinsics) -> dict:
    fields = {
        "w": intrinsics.w,
        "h": intrinsics.h,
        "fl_x": intrinsics.fl_x,
        "fl_y": intrinsics.fl_y,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
    }
    for key in ("k1", "k2", "k3", "p1", "p2"):
        value = getattr(intrinsics, key)
        if value is not None:
            fields[key] = value
    return fields


def _intrinsics_to_frame_fields(intrinsics: _CameraIntrinsics) -> dict:
    return _intrinsics_to_root_fields(intrinsics)


def _dataset_to_json_dict(dataset: TransformsDataset) -> dict:
    payload: dict = {}
    if dataset.camera_model is not None:
        payload["camera_model"] = dataset.camera_model

    for key in ("w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "p1", "p2"):
        value = getattr(dataset, key)
        if value is not None:
            payload[key] = value

    payload["frames"] = []
    for frame in dataset.frames:
        frame_dict: dict = {
            "file_path": frame.file_path,
            "transform_matrix": frame.transform_matrix,
        }
        for key in ("w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "p1", "p2"):
            value = getattr(frame, key)
            if value is not None:
                frame_dict[key] = value
        payload["frames"].append(frame_dict)
    return payload


def make_transforms_frame_from_pose(
    image: ColmapImagePose,
    *,
    image_prefix: str = "images",
    intrinsics: _CameraIntrinsics | None = None,
) -> TransformsFrame:
    """Build one frame record; exposed for focused coordinate tests."""
    c2w = colmap_w2c_to_nerf_c2w(image.qvec, image.tvec)
    frame_kwargs: dict = {
        "file_path": _frame_file_path(image.name, image_prefix),
        "transform_matrix": _matrix_to_nested_list(c2w),
    }
    if intrinsics is not None:
        frame_kwargs.update(_intrinsics_to_frame_fields(intrinsics))
    return TransformsFrame(**frame_kwargs)
