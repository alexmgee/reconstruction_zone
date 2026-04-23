"""
Reusable backend alignment profiles for stock COLMAP and SphereSfM.

These helpers keep Phase 1 focused on backend semantics without forcing the
GUI to hard-code command-line flags in multiple places.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AlignmentProfile:
    """Serializable runner profile for one alignment workflow."""

    name: str
    engine_name: str
    camera_model: str
    feature_type: str = "SIFT"
    matching_strategy: str = "exhaustive"
    guided_matching: bool = False
    mapper: str = "incremental"
    vocab_tree_path: str = ""
    extract_extra_args: Dict[str, Any] = field(default_factory=dict)
    match_extra_args: Dict[str, Any] = field(default_factory=dict)
    reconstruct_extra_args: Dict[str, Any] = field(default_factory=dict)


def build_colmap_pinhole_profile(
    camera_model: str = "PINHOLE",
    single_camera: bool = False,
    camera_params: Optional[str] = None,
    matching_strategy: str = "exhaustive",
    guided_matching: bool = False,
    mapper: str = "incremental",
    extract_extra_args: Optional[Dict[str, Any]] = None,
    match_extra_args: Optional[Dict[str, Any]] = None,
    reconstruct_extra_args: Optional[Dict[str, Any]] = None,
) -> AlignmentProfile:
    """Profile for standard perspective-image alignment with stock COLMAP."""
    extract_args: Dict[str, Any] = {}
    if single_camera:
        extract_args["ImageReader.single_camera"] = True
    if camera_params:
        extract_args["ImageReader.camera_params"] = camera_params
    if extract_extra_args:
        extract_args.update({k: v for k, v in extract_extra_args.items() if v is not None})

    match_args = {
        k: v for k, v in (match_extra_args or {}).items() if v is not None
    }
    reconstruct_args = {
        k: v for k, v in (reconstruct_extra_args or {}).items() if v is not None
    }

    return AlignmentProfile(
        name="colmap-pinhole",
        engine_name="colmap",
        camera_model=camera_model,
        matching_strategy=matching_strategy,
        guided_matching=guided_matching,
        mapper=mapper,
        extract_extra_args=extract_args,
        match_extra_args=match_args,
        reconstruct_extra_args=reconstruct_args,
    )


def build_colmap_rig_profile(
    camera_model: str = "OPENCV_FISHEYE",
    rig_config_path: str = "",
    matching_strategy: str = "exhaustive",
    guided_matching: bool = False,
    mapper: str = "incremental",
    extract_extra_args: Optional[Dict[str, Any]] = None,
    match_extra_args: Optional[Dict[str, Any]] = None,
    reconstruct_extra_args: Optional[Dict[str, Any]] = None,
) -> AlignmentProfile:
    """Profile for COLMAP alignment with a multi-camera rig constraint."""
    extract_args: Dict[str, Any] = {
        "ImageReader.single_camera_per_folder": True,
    }
    if extract_extra_args:
        extract_args.update({k: v for k, v in extract_extra_args.items() if v is not None})

    match_args = {
        k: v for k, v in (match_extra_args or {}).items() if v is not None
    }

    reconstruct_args: Dict[str, Any] = {
        "Mapper.ba_refine_sensor_from_rig": False,
        "Mapper.ba_refine_focal_length": True,
    }
    if reconstruct_extra_args:
        reconstruct_args.update(
            {k: v for k, v in reconstruct_extra_args.items() if v is not None}
        )

    return AlignmentProfile(
        name="colmap-rig",
        engine_name="colmap",
        camera_model=camera_model,
        matching_strategy=matching_strategy,
        guided_matching=guided_matching,
        mapper=mapper,
        extract_extra_args=extract_args,
        match_extra_args=match_args,
        reconstruct_extra_args=reconstruct_args,
    )


def build_spheresfm_erp_profile(
    image_width: int,
    image_height: int,
    pose_path: Optional[str] = None,
    camera_mask_path: Optional[str] = None,
    matching_strategy: Optional[str] = None,
    spatial_is_gps: int = 0,
    spatial_max_distance: Optional[float] = None,
    guided_matching: bool = False,
    vocab_tree_path: str = "",
    extract_extra_args: Optional[Dict[str, Any]] = None,
    match_extra_args: Optional[Dict[str, Any]] = None,
    reconstruct_extra_args: Optional[Dict[str, Any]] = None,
) -> AlignmentProfile:
    """Profile for ERP alignment with a SphereSfM-style binary."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")

    strategy = matching_strategy or ("spatial" if pose_path else "exhaustive")
    cx = image_width / 2.0
    cy = image_height / 2.0

    extract_args: Dict[str, Any] = {
        "ImageReader.single_camera": True,
        "ImageReader.camera_params": f"1,{cx:g},{cy:g}",
    }
    if pose_path:
        extract_args["ImageReader.pose_path"] = pose_path
    if camera_mask_path:
        extract_args["ImageReader.camera_mask_path"] = camera_mask_path
    if extract_extra_args:
        extract_args.update({k: v for k, v in extract_extra_args.items() if v is not None})

    match_args: Dict[str, Any] = {}
    if strategy == "spatial":
        match_args["SpatialMatching.is_gps"] = int(bool(spatial_is_gps))
        if spatial_max_distance is not None:
            match_args["SpatialMatching.max_distance"] = spatial_max_distance
    if match_extra_args:
        match_args.update({k: v for k, v in match_extra_args.items() if v is not None})

    reconstruct_args: Dict[str, Any] = {
        "Mapper.sphere_camera": True,
        "Mapper.ba_refine_focal_length": False,
        "Mapper.ba_refine_principal_point": False,
        "Mapper.ba_refine_extra_params": False,
    }
    if reconstruct_extra_args:
        reconstruct_args.update(
            {k: v for k, v in reconstruct_extra_args.items() if v is not None}
        )

    return AlignmentProfile(
        name="spheresfm-erp",
        engine_name="spheresfm",
        camera_model="SPHERE",
        matching_strategy=strategy,
        guided_matching=guided_matching,
        mapper="incremental",
        vocab_tree_path=vocab_tree_path,
        extract_extra_args=extract_args,
        match_extra_args=match_args,
        reconstruct_extra_args=reconstruct_args,
    )
