"""
COLMAP/XMP Export Module

Reads exported Metashape data (cameras.xml + PLY + images) and produces
perspective crops with COLMAP text files and RealityScan XMP sidecars.

This decouples the conversion step from Metashape, enabling rapid iteration
on crop settings, presets, and filtering without Metashape running.

Ported from:
  - multicam-extract.py (duckbill-workflow) — pose math, COLMAP writers, undistortion
  - realityscan-extract.py (duckbill-workflow) — XMP sidecar format
  - equirect_to_perspectives.py (legacy) — rig config, XML parsing base
"""

import json
import math
import os
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .reframer import ViewConfig, Ring, _create_rotation_matrix, reframe_view
from .presets import PresetManager


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SensorCalibration:
    """Parsed Metashape sensor calibration."""
    sensor_id: int
    label: str
    sensor_type: str        # "frame" or "spherical"
    width: int
    height: int
    f: Optional[float] = None   # focal length in pixels
    cx: float = 0.0             # principal point offset X (from center)
    cy: float = 0.0             # principal point offset Y (from center)
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


@dataclass
class MetashapeCamera:
    """Parsed per-camera data from Metashape XML."""
    camera_id: int
    label: str              # image filename
    sensor_id: int
    enabled: bool
    transform: Optional[np.ndarray] = None  # 4x4 camera-to-chunk, None if unaligned


@dataclass
class ChunkTransform:
    """Parsed chunk transform from Metashape XML."""
    rotation: np.ndarray    # 3x3 (may include scale)
    translation: np.ndarray # 3-vec
    scale: float


@dataclass
class MetashapeProject:
    """Full parsed Metashape XML project."""
    sensors: Dict[int, SensorCalibration]
    cameras: List[MetashapeCamera]
    chunk_transform: Optional[ChunkTransform]

    def spherical_cameras(self) -> List[MetashapeCamera]:
        return [c for c in self.cameras
                if c.transform is not None and c.enabled
                and self.sensors[c.sensor_id].sensor_type == "spherical"]

    def frame_cameras(self) -> List[MetashapeCamera]:
        return [c for c in self.cameras
                if c.transform is not None and c.enabled
                and self.sensors[c.sensor_id].sensor_type == "frame"]

    def aligned_cameras(self) -> List[MetashapeCamera]:
        return [c for c in self.cameras
                if c.transform is not None and c.enabled]


@dataclass
class COLMAPCameraEntry:
    """One row in cameras.txt."""
    camera_id: int
    model: str          # "PINHOLE" or "OPENCV"
    width: int
    height: int
    params: List[float] # [fx, fy, cx, cy] or [fx, fy, cx, cy, k1, k2, p1, p2]


@dataclass
class COLMAPImage:
    """One row in images.txt."""
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str


@dataclass
class ExportResult:
    """Result of the full export operation."""
    success: bool
    images_exported: int = 0
    cameras_written: int = 0
    points_written: int = 0
    colmap_dir: Optional[str] = None
    xmp_count: int = 0
    errors: List[str] = field(default_factory=list)
    skipped_existing: int = 0
    skipped_unaligned: int = 0


# =============================================================================
# XML PARSING
# =============================================================================

def _classify_sensor_type(width: int, height: int, xml_type: Optional[str] = None) -> str:
    """Classify sensor as 'spherical' or 'frame'.
    Uses XML type attribute if available, otherwise aspect ratio heuristic.
    """
    if xml_type is not None:
        lower = xml_type.lower()
        if "spherical" in lower or "equirectangular" in lower:
            return "spherical"
        if "frame" in lower:
            return "frame"
    # Heuristic: width:height >= 1.9 is equirectangular
    if height > 0 and width / height >= 1.9:
        return "spherical"
    return "frame"


def _parse_sensors(chunk_elem) -> Dict[int, SensorCalibration]:
    """Parse <sensors> from Metashape XML."""
    sensors = {}
    sensors_elem = chunk_elem.find("sensors")
    if sensors_elem is None:
        return sensors

    for sensor_elem in sensors_elem.findall("sensor"):
        sid = int(sensor_elem.get("id", "0"))
        label = sensor_elem.get("label", f"sensor_{sid}")
        xml_type = sensor_elem.get("type")

        # Parse calibration
        f_val = None
        cx_val = cy_val = 0.0
        k1 = k2 = k3 = p1 = p2 = 0.0
        width = height = 0

        calib = sensor_elem.find("calibration")
        if calib is not None:
            res = calib.find("resolution")
            if res is not None:
                width = int(res.get("width", "0"))
                height = int(res.get("height", "0"))

            f_elem = calib.find("f")
            if f_elem is not None and f_elem.text:
                f_val = float(f_elem.text)
            cx_elem = calib.find("cx")
            if cx_elem is not None and cx_elem.text:
                cx_val = float(cx_elem.text)
            cy_elem = calib.find("cy")
            if cy_elem is not None and cy_elem.text:
                cy_val = float(cy_elem.text)

            for coeff_name in ["k1", "k2", "k3", "p1", "p2"]:
                elem = calib.find(coeff_name)
                if elem is not None and elem.text:
                    val = float(elem.text)
                    if coeff_name == "k1": k1 = val
                    elif coeff_name == "k2": k2 = val
                    elif coeff_name == "k3": k3 = val
                    elif coeff_name == "p1": p1 = val
                    elif coeff_name == "p2": p2 = val

        # Fallback: check resolution directly on sensor element
        if width == 0:
            res = sensor_elem.find("resolution")
            if res is not None:
                width = int(res.get("width", "0"))
                height = int(res.get("height", "0"))

        sensor_type = _classify_sensor_type(width, height, xml_type)

        sensors[sid] = SensorCalibration(
            sensor_id=sid,
            label=label,
            sensor_type=sensor_type,
            width=width,
            height=height,
            f=f_val,
            cx=cx_val,
            cy=cy_val,
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
        )

    return sensors


def _parse_cameras(chunk_elem) -> List[MetashapeCamera]:
    """Parse <cameras> from Metashape XML."""
    cameras = []
    cameras_elem = chunk_elem.find("cameras")
    if cameras_elem is None:
        return cameras

    for cam_elem in cameras_elem.findall("camera"):
        cid = int(cam_elem.get("id", "0"))
        label = cam_elem.get("label", "")
        sensor_id = int(cam_elem.get("sensor_id", "0"))
        enabled = cam_elem.get("enabled", "true").lower() != "false"

        transform = None
        t_elem = cam_elem.find("transform")
        if t_elem is not None and t_elem.text:
            values = [float(x) for x in t_elem.text.split()]
            if len(values) == 16:
                transform = np.array(values, dtype=np.float64).reshape(4, 4)

        cameras.append(MetashapeCamera(
            camera_id=cid,
            label=label,
            sensor_id=sensor_id,
            enabled=enabled,
            transform=transform,
        ))

    return cameras


def _parse_chunk_transform(chunk_elem) -> Optional[ChunkTransform]:
    """Parse <transform> from Metashape XML chunk."""
    t_elem = chunk_elem.find("transform")
    if t_elem is None:
        return None

    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    scale_val = 1.0

    rot_elem = t_elem.find("rotation")
    if rot_elem is not None and rot_elem.text:
        rotation = np.array(
            [float(x) for x in rot_elem.text.split()], dtype=np.float64
        ).reshape(3, 3)

    trans_elem = t_elem.find("translation")
    if trans_elem is not None and trans_elem.text:
        translation = np.array(
            [float(x) for x in trans_elem.text.split()], dtype=np.float64
        )

    scale_elem = t_elem.find("scale")
    if scale_elem is not None and scale_elem.text:
        scale_val = float(scale_elem.text)

    return ChunkTransform(
        rotation=rotation,
        translation=translation,
        scale=scale_val,
    )


def parse_metashape_xml(xml_path: str) -> MetashapeProject:
    """Parse Metashape cameras.xml export.

    Args:
        xml_path: Path to cameras.xml

    Returns:
        MetashapeProject with all parsed data
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find first chunk
    chunk = root.find(".//chunk")
    if chunk is None:
        raise ValueError(f"No <chunk> element found in {xml_path}")

    sensors = _parse_sensors(chunk)
    cameras = _parse_cameras(chunk)
    chunk_transform = _parse_chunk_transform(chunk)

    return MetashapeProject(
        sensors=sensors,
        cameras=cameras,
        chunk_transform=chunk_transform,
    )


# =============================================================================
# POSE MATH
# =============================================================================

def get_chunk_rotation_and_scale(
    chunk_transform: Optional[ChunkTransform],
    flip_yz: bool = True,
) -> Tuple[np.ndarray, float]:
    """Extract chunk-to-world rotation (with Y/Z flip) and scale.

    The rotation matrix from Metashape XML <transform><rotation> may include
    scale. We extract the pure rotation by dividing out the determinant-derived
    scale factor.

    The Y/Z flip (diag([1,-1,-1])) converts from Metashape's coordinate
    convention to COLMAP/3DGS convention. This is baked into R_cw so all
    downstream transforms (cameras + points) are consistent.

    Port of: multicam-extract.py:258-268, 600-602
    """
    if chunk_transform is None:
        R_cw = np.eye(3, dtype=np.float64)
        cw_scale = 1.0
    else:
        RS = chunk_transform.rotation.copy()
        # The XML rotation may have scale baked in via the <scale> element,
        # but we also extract scale from the determinant for robustness
        det_scale = abs(np.linalg.det(RS)) ** (1.0 / 3.0)
        if det_scale < 1e-12:
            R_cw = np.eye(3, dtype=np.float64)
            cw_scale = 1.0
        else:
            R_cw = RS / det_scale
            # Use the XML scale element for the world scale factor
            cw_scale = chunk_transform.scale

    if flip_yz:
        flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        R_cw = flip @ R_cw

    return R_cw, cw_scale


def rotation_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz).

    Uses Shepperd's method for numerical stability.
    COLMAP convention: qw >= 0.

    Port of: multicam-extract.py:271-285
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        qw, qx, qy, qz = (
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw, qx, qy, qz = (
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        )
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw, qx, qy, qz = (
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        )
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw, qx, qy, qz = (
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        )

    # COLMAP convention: ensure qw >= 0
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz

    return qw, qx, qy, qz


def compute_perspective_pose(
    R_parent: np.ndarray,
    t_parent: np.ndarray,
    R_cw: np.ndarray,
    cw_scale: float,
    yaw_deg: float,
    pitch_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute world-to-camera pose for one perspective crop from an ERP camera.

    Uses prep360's _create_rotation_matrix() for face rotation to ensure
    consistency with the Reframer that generates the actual crop images.

    Math (adapted from multicam-extract.py:733-744):
        R_face = _create_rotation_matrix(yaw_deg, pitch_deg)
        R_c2w = R_cw @ (R_parent @ R_face)
        center = cw_scale * (R_cw @ t_parent)
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ center

    Returns:
        (R_w2c, t_w2c, center_world)
    """
    R_face = _create_rotation_matrix(yaw_deg, pitch_deg, 0)
    R_c2w = R_cw @ (R_parent @ R_face)
    center = cw_scale * (R_cw @ t_parent)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ center
    return R_w2c, t_w2c, center


def compute_frame_camera_pose(
    R_parent: np.ndarray,
    t_parent: np.ndarray,
    R_cw: np.ndarray,
    cw_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute world-to-camera pose for a frame (drone/pinhole) camera.

    Same as compute_perspective_pose but without face rotation.

    Returns:
        (R_w2c, t_w2c, center_world)
    """
    R_c2w = R_cw @ R_parent
    center = cw_scale * (R_cw @ t_parent)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ center
    return R_w2c, t_w2c, center


# =============================================================================
# COLMAP TEXT WRITERS
# =============================================================================

def write_colmap_cameras(cameras: List[COLMAPCameraEntry], output_path: str) -> None:
    """Write cameras.txt in COLMAP text format.

    Port of: multicam-extract.py:438-447
    """
    with open(output_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam in sorted(cameras, key=lambda c: c.camera_id):
            params_str = " ".join(f"{p:.6f}" for p in cam.params)
            f.write(f"{cam.camera_id} {cam.model} {cam.width} {cam.height} {params_str}\n")


def write_colmap_images(images: List[COLMAPImage], output_path: str) -> None:
    """Write images.txt in COLMAP text format.

    Each image is 2 lines: pose data + empty line (no 2D points).
    Port of: multicam-extract.py:450-462
    """
    with open(output_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write(f"# Number of images: {len(images)}\n")
        for img in images:
            f.write(
                f"{img.image_id} "
                f"{img.qw:.12f} {img.qx:.12f} "
                f"{img.qy:.12f} {img.qz:.12f} "
                f"{img.tx:.12f} {img.ty:.12f} {img.tz:.12f} "
                f"{img.camera_id} {img.name}\n"
            )
            f.write("\n")


def write_colmap_points3d_empty(output_path: str) -> None:
    """Write an empty points3D.txt (header only)."""
    with open(output_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")


# =============================================================================
# PLY READER + POINTS3D WRITER
# =============================================================================

def _read_ply_ascii(f, vertex_count: int, prop_names: List[str]) -> List[Tuple]:
    """Read ASCII PLY vertices."""
    points = []
    x_i = prop_names.index("x")
    y_i = prop_names.index("y")
    z_i = prop_names.index("z")
    r_i = prop_names.index("red") if "red" in prop_names else None
    g_i = prop_names.index("green") if "green" in prop_names else None
    b_i = prop_names.index("blue") if "blue" in prop_names else None

    for _ in range(vertex_count):
        line = f.readline()
        if not line:
            break
        vals = line.split()
        x, y, z = float(vals[x_i]), float(vals[y_i]), float(vals[z_i])
        if r_i is not None and g_i is not None and b_i is not None:
            r, g, b = int(vals[r_i]), int(vals[g_i]), int(vals[b_i])
        else:
            r, g, b = 128, 128, 128
        points.append((x, y, z, r, g, b))
    return points


def _read_ply_binary(f, vertex_count: int, prop_names: List[str],
                     prop_types: List[str]) -> List[Tuple]:
    """Read binary_little_endian PLY vertices."""
    # Build struct format
    type_map = {
        "float": "f", "float32": "f", "double": "d", "float64": "d",
        "uchar": "B", "uint8": "B", "char": "b", "int8": "b",
        "ushort": "H", "uint16": "H", "short": "h", "int16": "h",
        "uint": "I", "uint32": "I", "int": "i", "int32": "i",
    }
    fmt_chars = []
    for pt in prop_types:
        c = type_map.get(pt)
        if c is None:
            raise ValueError(f"Unknown PLY type: {pt}")
        fmt_chars.append(c)

    fmt = "<" + "".join(fmt_chars)
    row_size = struct.calcsize(fmt)

    x_i = prop_names.index("x")
    y_i = prop_names.index("y")
    z_i = prop_names.index("z")
    r_i = prop_names.index("red") if "red" in prop_names else None
    g_i = prop_names.index("green") if "green" in prop_names else None
    b_i = prop_names.index("blue") if "blue" in prop_names else None

    points = []
    for _ in range(vertex_count):
        raw = f.read(row_size)
        if len(raw) < row_size:
            break
        vals = struct.unpack(fmt, raw)
        x, y, z = float(vals[x_i]), float(vals[y_i]), float(vals[z_i])
        if r_i is not None and g_i is not None and b_i is not None:
            r, g, b = int(vals[r_i]), int(vals[g_i]), int(vals[b_i])
        else:
            r, g, b = 128, 128, 128
        points.append((x, y, z, r, g, b))
    return points


def read_ply_points(ply_path: str) -> Optional[List[Tuple]]:
    """Read PLY point cloud vertices as list of (x, y, z, r, g, b) tuples.

    Handles ASCII and binary_little_endian PLY formats.
    Returns None if file cannot be parsed.
    """
    try:
        with open(ply_path, "rb") as f:
            # Parse header
            header_lines = []
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                header_lines.append(line)
                if line == "end_header":
                    break

            vertex_count = 0
            prop_names = []
            prop_types = []
            is_binary = False
            in_vertex = False

            for line in header_lines:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "format":
                    is_binary = "binary_little_endian" in line
                elif parts[0] == "element" and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                    in_vertex = True
                elif parts[0] == "element" and parts[1] != "vertex":
                    in_vertex = False
                elif parts[0] == "property" and in_vertex and len(parts) >= 3:
                    prop_types.append(parts[1])
                    prop_names.append(parts[2])

            if vertex_count == 0:
                return None

            if is_binary:
                return _read_ply_binary(f, vertex_count, prop_names, prop_types)
            else:
                # For ASCII, reopen in text mode from after header
                pass

        # ASCII mode: reopen and skip header
        with open(ply_path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                if line.strip() == "end_header":
                    break
            return _read_ply_ascii(f, vertex_count, prop_names)

    except Exception:
        return None


def write_colmap_points3d(
    ply_path: Optional[str],
    R_cw: np.ndarray,
    cw_scale: float,
    output_path: str,
) -> int:
    """Write points3D.txt from PLY file.

    Transforms points from chunk coordinates to world coordinates.
    Handles 16-bit color by right-shifting 8 bits.

    Port of: multicam-extract.py:465-507

    Returns number of points written.
    """
    if ply_path is None or not os.path.exists(ply_path):
        write_colmap_points3d_empty(output_path)
        return 0

    raw_points = read_ply_points(ply_path)
    if raw_points is None or len(raw_points) == 0:
        write_colmap_points3d_empty(output_path)
        return 0

    count = 0
    with open(output_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        for x, y, z, rv, gv, bv in raw_points:
            coord = np.array([x, y, z], dtype=np.float64)
            coord = cw_scale * (R_cw @ coord)

            # Handle color formats (ported from multicam-extract.py:493-501)
            if max(rv, gv, bv) > 255:
                # 16-bit (0-65535) -> 8-bit (0-255)
                r, g, b = rv >> 8, gv >> 8, bv >> 8
            else:
                r, g, b = int(rv), int(gv), int(bv)

            count += 1
            f.write(
                f"{count} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} "
                f"{r} {g} {b} 0.0 \n"
            )

    return count


# =============================================================================
# RIG CONFIG
# =============================================================================

def write_rig_config(
    view_config: ViewConfig,
    output_path: str,
) -> None:
    """Generate COLMAP rig configuration JSON.

    All perspective crops from the same equirectangular parent form a rig
    with shared center but different orientations.

    Port of: equirect_to_perspectives.py:426-459
    """
    views = view_config.get_all_views()
    rig_config = [{"cameras": []}]

    for i, (yaw, pitch, fov, view_name) in enumerate(views):
        R_face = _create_rotation_matrix(yaw, pitch, 0).T  # world-to-camera
        qw, qx, qy, qz = rotation_to_quaternion(R_face)

        cam_entry = {
            "image_prefix": f"*_{view_name}.",
        }

        if i == 0:
            cam_entry["ref_sensor"] = True
        else:
            cam_entry["cam_from_rig_rotation"] = [qw, qx, qy, qz]
            cam_entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]

        rig_config[0]["cameras"].append(cam_entry)

    with open(output_path, "w") as f:
        json.dump(rig_config, f, indent=2)


# =============================================================================
# XMP WRITER (RealityCapture / RealityScan format)
# =============================================================================

def focal_length_to_35mm(f_pixels: float, image_width: int) -> float:
    """Convert pixel focal length to 35mm equivalent.
    Formula: f_35mm = f_px * 36.0 / image_width
    """
    return f_pixels * 36.0 / image_width


def write_xmp(
    output_path: str,
    R_w2c: np.ndarray,
    center_world: np.ndarray,
    focal_35mm: float,
    pose_prior: str = "locked",
) -> None:
    """Write a RealityScan-compatible XMP sidecar file.

    XMP uses xcr 1.1 namespace:
      xcr:Rotation = R_w2c row-major 9 floats
      xcr:Position = camera center in world coordinates

    Port of: realityscan-extract.py:299-333
    """
    rot_str = " ".join(f"{v:.12f}" for v in R_w2c.flatten())
    pos_str = " ".join(f"{v:.12f}" for v in center_world)

    xmp = (
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        f'  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        f'    <rdf:Description xcr:Version="3" xcr:PosePrior="{pose_prior}"\n'
        f'       xcr:Coordinates="absolute" xcr:FocalLength35mm="{focal_35mm:.6f}"\n'
        f'       xcr:Skew="0" xcr:AspectRatio="1"\n'
        f'       xcr:PrincipalPointU="0" xcr:PrincipalPointV="0"\n'
        f'       xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#">\n'
        f'      <xcr:Rotation>{rot_str}</xcr:Rotation>\n'
        f'      <xcr:Position>{pos_str}</xcr:Position>\n'
        f'    </rdf:Description>\n'
        f'  </rdf:RDF>\n'
        f'</x:xmpmeta>'
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xmp)


def write_common_xmp(
    output_dir: str,
    focal_35mm: float,
    pose_prior: str = "locked",
) -> None:
    """Write _common.xmp with shared calibration for a directory.

    This applies to all images in the directory that don't have their own XMP.
    Contains calibration only, no pose.
    """
    xmp = (
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        f'  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        f'    <rdf:Description xcr:Version="3"\n'
        f'       xcr:FocalLength35mm="{focal_35mm:.6f}"\n'
        f'       xcr:Skew="0" xcr:AspectRatio="1"\n'
        f'       xcr:PrincipalPointU="0" xcr:PrincipalPointV="0"\n'
        f'       xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#">\n'
        f'    </rdf:Description>\n'
        f'  </rdf:RDF>\n'
        f'</x:xmpmeta>'
    )

    path = os.path.join(output_dir, "_common.xmp")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xmp)


# =============================================================================
# FRAME CAMERA UNDISTORTION
# =============================================================================

def build_undistort_maps(
    sensor: SensorCalibration,
) -> Optional[Tuple[np.ndarray, np.ndarray, COLMAPCameraEntry]]:
    """Build cv2 undistortion remap tables for a frame camera sensor.

    Returns (map_x, map_y, pinhole_camera) or None if no significant distortion.
    Uses alpha=0 for no black borders in output.

    Port of: multicam-extract.py:332-373
    """
    if sensor.f is None:
        return None

    has_dist = (
        abs(sensor.k1) > 1e-8 or abs(sensor.k2) > 1e-8
        or abs(sensor.p1) > 1e-8 or abs(sensor.p2) > 1e-8
    )
    if not has_dist:
        return None

    w, h = sensor.width, sensor.height
    f = sensor.f
    # Metashape: cx/cy are offset from center; OpenCV needs absolute pixel coords
    cx = w / 2.0 + sensor.cx
    cy = h / 2.0 + sensor.cy

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([sensor.k1, sensor.k2, sensor.p1, sensor.p2], dtype=np.float64)

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (w, h), cv2.CV_32FC1
    )

    pinhole_cam = COLMAPCameraEntry(
        camera_id=0,  # assigned later
        model="PINHOLE",
        width=w,
        height=h,
        params=[new_K[0, 0], new_K[1, 1], new_K[0, 2], new_K[1, 2]],
    )
    return map1, map2, pinhole_cam


def _sensor_to_colmap_camera(sensor: SensorCalibration) -> Optional[COLMAPCameraEntry]:
    """Convert a sensor calibration to COLMAP camera entry (without undistortion).

    Port of: multicam-extract.py:292-329
    """
    if sensor.f is None:
        return None

    w, h = sensor.width, sensor.height
    f = sensor.f
    # COLMAP absolute pixel coords
    cx = w / 2.0 + sensor.cx
    cy = h / 2.0 + sensor.cy

    has_dist = (
        abs(sensor.k1) > 1e-8 or abs(sensor.k2) > 1e-8
        or abs(sensor.p1) > 1e-8 or abs(sensor.p2) > 1e-8
    )

    if has_dist:
        return COLMAPCameraEntry(
            camera_id=0,
            model="OPENCV",
            width=w, height=h,
            params=[f, f, cx, cy, sensor.k1, sensor.k2, sensor.p1, sensor.p2],
        )
    else:
        return COLMAPCameraEntry(
            camera_id=0,
            model="PINHOLE",
            width=w, height=h,
            params=[f, f, cx, cy],
        )


# =============================================================================
# MASK HELPERS
# =============================================================================

def _find_matching_mask(mask_dir: str, image_label: str) -> Optional[Path]:
    """Find mask file matching an image label.

    Strategy 1: prefixed names (mask_*, *_mask)
    Strategy 2: same-stem matching (image.jpg -> image.png in masks_dir)
    """
    masks_path = Path(mask_dir)
    stem = Path(image_label).stem

    # Strategy 1: prefixed
    for pattern in [f"mask_{stem}.*", f"{stem}_mask.*"]:
        matches = list(masks_path.glob(pattern))
        if matches:
            return matches[0]

    # Strategy 2: same-stem
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = masks_path / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


# =============================================================================
# ORCHESTRATOR
# =============================================================================

@dataclass
class ColmapExportConfig:
    """Configuration for the full COLMAP export pipeline."""
    xml_path: str
    images_dir: str
    output_dir: str
    ply_path: Optional[str] = None
    masks_dir: Optional[str] = None

    # View configuration (for spherical cameras)
    preset_name: str = "prep360_default"
    view_config: Optional[ViewConfig] = None
    output_size: int = 1920
    jpeg_quality: int = 95

    # Export toggles
    export_colmap: bool = True
    export_xmp: bool = True
    export_rig_config: bool = False

    # Processing
    undistort_frame_cameras: bool = True
    flip_yz: bool = True
    num_workers: int = 4

    # XMP settings
    xmp_pose_prior: str = "locked"

    # Incremental
    skip_existing: bool = True


class ColmapExporter:
    """Main orchestrator: Metashape XML -> perspective crops + COLMAP/XMP.

    Usage:
        config = ColmapExportConfig(xml_path=..., images_dir=..., output_dir=...)
        exporter = ColmapExporter(config)
        result = exporter.export(progress_callback=lambda cur, total, name: ...)
    """

    def __init__(self, config: ColmapExportConfig):
        self.config = config
        self.project: Optional[MetashapeProject] = None
        self._colmap_cameras: List[COLMAPCameraEntry] = []
        self._colmap_images: List[COLMAPImage] = []
        self._camera_id_map: Dict[tuple, int] = {}
        self._next_camera_id: int = 1
        self._next_image_id: int = 1
        self._xmp_count: int = 0

    def load_project(self) -> MetashapeProject:
        """Parse the Metashape XML."""
        self.project = parse_metashape_xml(self.config.xml_path)
        return self.project

    def get_view_config(self) -> ViewConfig:
        """Resolve ViewConfig from config or preset."""
        if self.config.view_config is not None:
            return self.config.view_config
        try:
            pm = PresetManager()
            preset = pm.get_preset(self.config.preset_name)
            if preset is not None:
                vc = preset.get_view_config()
                vc.output_size = self.config.output_size
                vc.jpeg_quality = self.config.jpeg_quality
                return vc
        except Exception:
            pass
        # Fallback: prep360_default equivalent
        return ViewConfig(
            rings=[
                Ring(pitch=0, count=8, fov=65),
                Ring(pitch=-20, count=4, fov=65),
            ],
            include_zenith=True,
            include_nadir=False,
            output_size=self.config.output_size,
            jpeg_quality=self.config.jpeg_quality,
        )

    def _get_or_create_erp_camera(self, fov_deg: float, size: int) -> int:
        """Get or create a shared PINHOLE camera entry for ERP perspective crops."""
        key = ("erp", fov_deg, size)
        if key in self._camera_id_map:
            return self._camera_id_map[key]

        f_px = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        cam = COLMAPCameraEntry(
            camera_id=self._next_camera_id,
            model="PINHOLE",
            width=size, height=size,
            params=[f_px, f_px, size / 2.0, size / 2.0],
        )
        self._colmap_cameras.append(cam)
        self._camera_id_map[key] = cam.camera_id
        self._next_camera_id += 1
        return cam.camera_id

    def _get_or_create_frame_camera(
        self, sensor: SensorCalibration, undistort_maps: Optional[tuple]
    ) -> int:
        """Get or create COLMAP camera entry for a frame sensor."""
        key = ("frame", sensor.sensor_id)
        if key in self._camera_id_map:
            return self._camera_id_map[key]

        if undistort_maps is not None:
            _, _, pinhole_cam = undistort_maps
            cam = COLMAPCameraEntry(
                camera_id=self._next_camera_id,
                model=pinhole_cam.model,
                width=pinhole_cam.width,
                height=pinhole_cam.height,
                params=pinhole_cam.params[:],
            )
        else:
            raw_cam = _sensor_to_colmap_camera(sensor)
            if raw_cam is None:
                return -1
            cam = COLMAPCameraEntry(
                camera_id=self._next_camera_id,
                model=raw_cam.model,
                width=raw_cam.width,
                height=raw_cam.height,
                params=raw_cam.params[:],
            )

        self._colmap_cameras.append(cam)
        self._camera_id_map[key] = cam.camera_id
        self._next_camera_id += 1
        return cam.camera_id

    def _process_spherical_camera(
        self,
        cam: MetashapeCamera,
        R_cw: np.ndarray,
        cw_scale: float,
        view_config: ViewConfig,
        images_out: Path,
        masks_out: Optional[Path],
    ) -> Tuple[int, List[str]]:
        """Process one spherical (equirectangular) camera.

        Returns (images_generated, errors).
        """
        errors = []
        generated = 0

        # Load equirectangular image
        img_path = Path(self.config.images_dir) / cam.label
        if not img_path.exists():
            # Try common extensions
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                candidate = img_path.with_suffix(ext)
                if candidate.exists():
                    img_path = candidate
                    break
            else:
                errors.append(f"Image not found: {cam.label}")
                return generated, errors

        equirect = cv2.imread(str(img_path))
        if equirect is None:
            errors.append(f"Failed to load: {img_path}")
            return generated, errors

        # Load mask if available
        mask_equirect = None
        if self.config.masks_dir:
            mask_path = _find_matching_mask(self.config.masks_dir, cam.label)
            if mask_path:
                mask_equirect = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Extract R_parent and t_parent from camera transform
        R_parent = cam.transform[:3, :3].copy()
        t_parent = cam.transform[:3, 3].copy()

        stem = Path(cam.label).stem
        views = view_config.get_all_views()

        for yaw, pitch, fov, view_name in views:
            out_name = f"{stem}_{view_name}.jpg"
            out_path = images_out / out_name

            # Compute pose (always, even if skipping image generation)
            R_w2c, t_w2c, center_world = compute_perspective_pose(
                R_parent, t_parent, R_cw, cw_scale, yaw, pitch
            )
            qw, qx, qy, qz = rotation_to_quaternion(R_w2c)

            # Get or create camera entry
            camera_id = self._get_or_create_erp_camera(fov, view_config.output_size)

            # Add to COLMAP images list
            self._colmap_images.append(COLMAPImage(
                image_id=self._next_image_id,
                qw=qw, qx=qx, qy=qy, qz=qz,
                tx=t_w2c[0], ty=t_w2c[1], tz=t_w2c[2],
                camera_id=camera_id,
                name=out_name,
            ))
            self._next_image_id += 1

            # Skip image generation if exists and skip_existing
            if self.config.skip_existing and out_path.exists():
                continue

            # Reframe perspective crop
            try:
                persp = reframe_view(
                    equirect,
                    fov_deg=fov,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    out_size=view_config.output_size,
                )
                cv2.imwrite(
                    str(out_path), persp,
                    [cv2.IMWRITE_JPEG_QUALITY, view_config.jpeg_quality],
                )
                generated += 1
            except Exception as e:
                errors.append(f"Reframe failed for {out_name}: {e}")
                continue

            # Write XMP sidecar alongside image
            if self.config.export_xmp:
                f_px = (view_config.output_size / 2.0) / math.tan(math.radians(fov) / 2.0)
                f_35mm = focal_length_to_35mm(f_px, view_config.output_size)
                xmp_path = out_path.with_suffix(".xmp")
                write_xmp(
                    str(xmp_path), R_w2c, center_world, f_35mm,
                    pose_prior=self.config.xmp_pose_prior,
                )
                self._xmp_count += 1

            # Reproject mask
            if mask_equirect is not None and masks_out is not None:
                try:
                    mask_persp = reframe_view(
                        mask_equirect,
                        fov_deg=fov,
                        yaw_deg=yaw,
                        pitch_deg=pitch,
                        out_size=view_config.output_size,
                        mode="nearest",
                    )
                    mask_out_name = f"{stem}_{view_name}.png"
                    cv2.imwrite(str(masks_out / mask_out_name), mask_persp)
                except Exception:
                    pass

        return generated, errors

    def _process_frame_camera(
        self,
        cam: MetashapeCamera,
        sensor: SensorCalibration,
        R_cw: np.ndarray,
        cw_scale: float,
        images_out: Path,
        masks_out: Optional[Path],
        undistort_maps: Optional[tuple],
    ) -> Tuple[int, List[str]]:
        """Process one frame (drone/pinhole) camera.

        Returns (images_generated, errors).
        """
        errors = []
        generated = 0

        # Get camera model
        camera_id = self._get_or_create_frame_camera(sensor, undistort_maps)
        if camera_id < 0:
            errors.append(f"No calibration for sensor '{sensor.label}', skipping {cam.label}")
            return generated, errors

        # Compute pose
        R_parent = cam.transform[:3, :3].copy()
        t_parent = cam.transform[:3, 3].copy()
        R_w2c, t_w2c, center_world = compute_frame_camera_pose(
            R_parent, t_parent, R_cw, cw_scale
        )
        qw, qx, qy, qz = rotation_to_quaternion(R_w2c)

        # Add to COLMAP images list
        out_name = cam.label
        self._colmap_images.append(COLMAPImage(
            image_id=self._next_image_id,
            qw=qw, qx=qx, qy=qy, qz=qz,
            tx=t_w2c[0], ty=t_w2c[1], tz=t_w2c[2],
            camera_id=camera_id,
            name=out_name,
        ))
        self._next_image_id += 1

        out_path = images_out / out_name

        # Skip if exists
        if self.config.skip_existing and out_path.exists():
            return generated, errors

        # Find source image
        src_path = Path(self.config.images_dir) / cam.label
        if not src_path.exists():
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                candidate = src_path.with_suffix(ext)
                if candidate.exists():
                    src_path = candidate
                    break
            else:
                errors.append(f"Image not found: {cam.label}")
                return generated, errors

        # Process image
        if undistort_maps is not None:
            map1, map2, _ = undistort_maps
            img = cv2.imread(str(src_path))
            if img is None:
                errors.append(f"Failed to load: {src_path}")
                return generated, errors
            img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(str(out_path), img,
                        [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        else:
            # Copy directly if no undistortion needed
            shutil.copy2(str(src_path), str(out_path))

        generated += 1

        # Write XMP sidecar
        if self.config.export_xmp and sensor.f is not None:
            f_35mm = focal_length_to_35mm(sensor.f, sensor.width)
            xmp_path = out_path.with_suffix(".xmp")
            write_xmp(
                str(xmp_path), R_w2c, center_world, f_35mm,
                pose_prior=self.config.xmp_pose_prior,
            )
            self._xmp_count += 1

        # Handle mask for frame camera
        if self.config.masks_dir and masks_out is not None:
            mask_path = _find_matching_mask(self.config.masks_dir, cam.label)
            if mask_path:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if undistort_maps is not None:
                        map1, map2, _ = undistort_maps
                        mask = cv2.remap(mask, map1, map2, cv2.INTER_NEAREST)
                    mask_out_path = masks_out / Path(cam.label).with_suffix(".png").name
                    cv2.imwrite(str(mask_out_path), mask)

        return generated, errors

    def export(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> ExportResult:
        """Run the full export pipeline.

        Args:
            progress_callback: Called with (current_idx, total_cameras, camera_label)

        Returns:
            ExportResult with statistics
        """
        result = ExportResult(success=True)

        # 1. Parse XML
        try:
            self.load_project()
        except Exception as e:
            result.success = False
            result.errors.append(f"XML parse error: {e}")
            return result

        project = self.project

        # 2. Compute chunk transform
        R_cw, cw_scale = get_chunk_rotation_and_scale(
            project.chunk_transform, flip_yz=self.config.flip_yz
        )

        # 3. Resolve ViewConfig
        view_config = self.get_view_config()

        # 4. Create output directories
        output = Path(self.config.output_dir)
        images_out = output / "images"
        sparse_out = output / "sparse" / "0"
        images_out.mkdir(parents=True, exist_ok=True)
        sparse_out.mkdir(parents=True, exist_ok=True)

        masks_out = None
        if self.config.masks_dir:
            masks_out = output / "masks"
            masks_out.mkdir(parents=True, exist_ok=True)

        # 5. Classify cameras
        erp_cams = project.spherical_cameras()
        frame_cams = project.frame_cameras()
        total_cams = len(erp_cams) + len(frame_cams)
        result.skipped_unaligned = len(project.cameras) - len(project.aligned_cameras())

        # 6. Build undistortion maps for frame sensors (once per sensor)
        sensor_undistort: Dict[int, Optional[tuple]] = {}
        if self.config.undistort_frame_cameras:
            for cam in frame_cams:
                sid = cam.sensor_id
                if sid not in sensor_undistort:
                    sensor = project.sensors[sid]
                    sensor_undistort[sid] = build_undistort_maps(sensor)

        # 7. Process spherical cameras
        cam_idx = 0
        for cam in erp_cams:
            if progress_callback:
                progress_callback(cam_idx, total_cams, cam.label)

            gen, errs = self._process_spherical_camera(
                cam, R_cw, cw_scale, view_config, images_out, masks_out
            )
            result.images_exported += gen
            result.errors.extend(errs)
            cam_idx += 1

        # 8. Process frame cameras
        for cam in frame_cams:
            if progress_callback:
                progress_callback(cam_idx, total_cams, cam.label)

            sensor = project.sensors[cam.sensor_id]
            ud_maps = sensor_undistort.get(cam.sensor_id)

            gen, errs = self._process_frame_camera(
                cam, sensor, R_cw, cw_scale, images_out, masks_out, ud_maps
            )
            result.images_exported += gen
            result.errors.extend(errs)
            cam_idx += 1

        # 9. Write COLMAP files
        if self.config.export_colmap:
            write_colmap_cameras(
                self._colmap_cameras, str(sparse_out / "cameras.txt")
            )
            write_colmap_images(
                self._colmap_images, str(sparse_out / "images.txt")
            )
            pts_count = write_colmap_points3d(
                self.config.ply_path, R_cw, cw_scale,
                str(sparse_out / "points3D.txt"),
            )
            result.cameras_written = len(self._colmap_cameras)
            result.points_written = pts_count
            result.colmap_dir = str(sparse_out)

        # 10. Write rig config
        if self.config.export_rig_config and erp_cams:
            write_rig_config(view_config, str(output / "rig_config.json"))

        # 11. Write common XMP
        if self.config.export_xmp and erp_cams:
            views = view_config.get_all_views()
            if views:
                fov = views[0][2]
                f_px = (view_config.output_size / 2.0) / math.tan(math.radians(fov) / 2.0)
                f_35mm = focal_length_to_35mm(f_px, view_config.output_size)
                write_common_xmp(
                    str(images_out), f_35mm,
                    pose_prior=self.config.xmp_pose_prior,
                )

        result.xmp_count = self._xmp_count
        result.success = len(result.errors) == 0 or result.images_exported > 0

        return result


def print_project_info(xml_path: str) -> None:
    """Parse XML and print project summary (for --info flag)."""
    project = parse_metashape_xml(xml_path)

    print(f"Metashape XML: {xml_path}")
    print(f"  Sensors: {len(project.sensors)}")
    for sid, s in project.sensors.items():
        print(f"    [{sid}] {s.label}: {s.sensor_type} {s.width}x{s.height}"
              + (f" f={s.f:.1f}px" if s.f else ""))

    aligned = project.aligned_cameras()
    erp = project.spherical_cameras()
    frame = project.frame_cameras()
    unaligned = len(project.cameras) - len(aligned)

    print(f"  Cameras: {len(project.cameras)} total")
    print(f"    Aligned: {len(aligned)}")
    print(f"    Spherical: {len(erp)}")
    print(f"    Frame: {len(frame)}")
    if unaligned > 0:
        print(f"    Unaligned: {unaligned}")

    if project.chunk_transform is not None:
        ct = project.chunk_transform
        det_scale = abs(np.linalg.det(ct.rotation)) ** (1.0 / 3.0)
        print(f"  Chunk transform: scale={ct.scale:.6f} (det_scale={det_scale:.6f})")
    else:
        print("  Chunk transform: identity")
