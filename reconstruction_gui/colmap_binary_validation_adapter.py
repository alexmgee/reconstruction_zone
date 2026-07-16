"""Bridge prep360 FULL COLMAP binary models into reconstruction_gui validation types."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from prep360.core.colmap_binary import (
    normalize_point3d_id,
    read_colmap_full_model_binary,
)
from prep360.core.colmap_types import (
    ColmapFullModel,
    ColmapPoseModel,
    ColmapReadMode,
)
from reconstruction_gui.colmap_validation import (
    COLMAPCamera,
    COLMAPImage,
    COLMAPPoint3D,
)

FULL_MODEL_REQUIRED_MSG = (
    "Full COLMAP model required for geometric validation: "
    "points3D and image observations are missing."
)


def _assert_track_observation_consistency(
    images: Dict[int, COLMAPImage],
    points3d: Dict[int, COLMAPPoint3D],
) -> None:
    for point_id, point in points3d.items():
        for image_id, point2d_idx in point.track:
            if image_id not in images:
                raise ValueError(
                    f"Inconsistent COLMAP track: point {point_id} track references "
                    f"missing image {image_id}.",
                )
            observations = images[image_id].points2d
            if point2d_idx < 0 or point2d_idx >= len(observations):
                raise ValueError(
                    f"Inconsistent COLMAP track: point {point_id} track references "
                    f"image {image_id} observation {point2d_idx}, but that image has "
                    f"{len(observations)} observations.",
                )
            observed_point3d_id = observations[point2d_idx][2]
            if observed_point3d_id != point_id:
                raise ValueError(
                    f"Inconsistent COLMAP track: point {point_id} track references "
                    f"image {image_id} observation {point2d_idx}, but that observation "
                    f"points to {observed_point3d_id}.",
                )


def colmap_full_model_to_validation_dict(model: ColmapFullModel) -> Dict[str, Any]:
    """Convert a prep360 FULL COLMAP model to the GeometricValidator data shape."""
    if isinstance(model, ColmapPoseModel) or not isinstance(model, ColmapFullModel):
        raise ValueError(FULL_MODEL_REQUIRED_MSG)
    if model.read_mode != ColmapReadMode.FULL:
        raise ValueError(FULL_MODEL_REQUIRED_MSG)

    cameras: Dict[int, COLMAPCamera] = {}
    for camera_id, camera in model.cameras.items():
        cameras[camera_id] = COLMAPCamera(
            camera_id=camera_id,
            model=camera.model_name,
            width=camera.width,
            height=camera.height,
            params=[float(value) for value in camera.params],
        )

    images: Dict[int, COLMAPImage] = {}
    for image_id, image in model.images.items():
        qvec = image.qvec
        tvec = image.tvec
        images[image_id] = COLMAPImage(
            image_id=image_id,
            qw=float(qvec[0]),
            qx=float(qvec[1]),
            qy=float(qvec[2]),
            qz=float(qvec[3]),
            tx=float(tvec[0]),
            ty=float(tvec[1]),
            tz=float(tvec[2]),
            camera_id=image.camera_id,
            name=image.name,
            points2d=[
                (
                    float(observation.x),
                    float(observation.y),
                    normalize_point3d_id(observation.point3d_id),
                )
                for observation in image.points2d
            ],
        )

    points3d: Dict[int, COLMAPPoint3D] = {}
    for point_id, point in model.points3d.items():
        points3d[point_id] = COLMAPPoint3D(
            point3d_id=point_id,
            xyz=np.asarray(point.xyz, dtype=np.float64),
            rgb=np.asarray(point.rgb, dtype=np.uint8),
            error=float(point.error),
            track=[
                (element.image_id, element.point2d_idx)
                for element in point.track
            ],
        )

    _assert_track_observation_consistency(images, points3d)

    return {
        "cameras": cameras,
        "images": images,
        "points3D": points3d,
    }


def read_colmap_binary_full_for_validation(model_dir: str | Path) -> Dict[str, Any]:
    """Read a binary sparse COLMAP directory and return validation-shaped data."""
    model = read_colmap_full_model_binary(model_dir)
    return colmap_full_model_to_validation_dict(model)
