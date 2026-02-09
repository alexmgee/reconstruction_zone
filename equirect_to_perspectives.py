#!/usr/bin/env python3
"""
Equirectangular to Multi-Perspective Extraction Tool

Converts equirectangular (360°) images to multiple pinhole perspective views
for alignment with regular pinhole cameras in Metashape/COLMAP workflows.

Features:
- Configurable view layouts (8, 10, 12, or custom perspectives)
- Supports mask reprojection
- Generates COLMAP-compatible output with rig constraints
- Integrates with Metashape XML camera exports
- Uses py360convert for robust projection (with fallback)

Author: Claude (Anthropic)
License: MIT
"""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Try to import py360convert, fallback to custom implementation
try:
    import py360convert
    HAS_PY360 = True
except ImportError:
    HAS_PY360 = False
    print("Note: py360convert not found, using built-in projection")


# =============================================================================
# View Configuration Presets
# =============================================================================

@dataclass
class ViewConfig:
    """Configuration for a single perspective view."""
    name: str
    yaw: float    # Horizontal angle in degrees (0 = front, 90 = right)
    pitch: float  # Vertical angle in degrees (0 = horizon, 90 = up)
    roll: float = 0.0  # Roll angle in degrees


# Preset configurations
VIEW_PRESETS = {
    # 6 views: Standard cubemap (90° FoV recommended)
    "6-cubemap": [
        ViewConfig("front", 0, 0),
        ViewConfig("right", 90, 0),
        ViewConfig("back", 180, 0),
        ViewConfig("left", 270, 0),
        ViewConfig("top", 0, 90),
        ViewConfig("bottom", 0, -90),
    ],
    
    # 8 views: 6 horizontal (60° apart) + top/bottom (70-80° FoV recommended)
    "8-hex": [
        ViewConfig("h0", 0, 0),
        ViewConfig("h60", 60, 0),
        ViewConfig("h120", 120, 0),
        ViewConfig("h180", 180, 0),
        ViewConfig("h240", 240, 0),
        ViewConfig("h300", 300, 0),
        ViewConfig("top", 0, 90),
        ViewConfig("bottom", 0, -90),
    ],
    
    # 10 views: 8 horizontal (45° apart) + top/bottom (60-70° FoV recommended)
    "10-octa": [
        ViewConfig("h0", 0, 0),
        ViewConfig("h45", 45, 0),
        ViewConfig("h90", 90, 0),
        ViewConfig("h135", 135, 0),
        ViewConfig("h180", 180, 0),
        ViewConfig("h225", 225, 0),
        ViewConfig("h270", 270, 0),
        ViewConfig("h315", 315, 0),
        ViewConfig("top", 0, 90),
        ViewConfig("bottom", 0, -90),
    ],
    
    # 12 views: 8 horizontal + 2 elevated (±45°) + top/bottom (55-65° FoV recommended)
    "12-full": [
        ViewConfig("h0", 0, 0),
        ViewConfig("h45", 45, 0),
        ViewConfig("h90", 90, 0),
        ViewConfig("h135", 135, 0),
        ViewConfig("h180", 180, 0),
        ViewConfig("h225", 225, 0),
        ViewConfig("h270", 270, 0),
        ViewConfig("h315", 315, 0),
        ViewConfig("up45_front", 0, 45),
        ViewConfig("up45_back", 180, 45),
        ViewConfig("top", 0, 90),
        ViewConfig("bottom", 0, -90),
    ],
    
    # 14 views: For maximum overlap
    "14-dense": [
        ViewConfig("h0", 0, 0),
        ViewConfig("h45", 45, 0),
        ViewConfig("h90", 90, 0),
        ViewConfig("h135", 135, 0),
        ViewConfig("h180", 180, 0),
        ViewConfig("h225", 225, 0),
        ViewConfig("h270", 270, 0),
        ViewConfig("h315", 315, 0),
        ViewConfig("up45_0", 0, 45),
        ViewConfig("up45_90", 90, 45),
        ViewConfig("up45_180", 180, 45),
        ViewConfig("up45_270", 270, 45),
        ViewConfig("top", 0, 90),
        ViewConfig("bottom", 0, -90),
    ],
}


# =============================================================================
# Projection Math (fallback if py360convert not available)
# =============================================================================

def create_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Create rotation matrix from Euler angles (yaw, pitch, roll in degrees)."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    
    # Rotation matrices
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Rx @ Ry


def equirect_to_perspective_custom(
    equirect: np.ndarray,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0,
    out_hw: Tuple[int, int] = (1024, 1024),
    mode: str = "bilinear"
) -> np.ndarray:
    """
    Extract perspective view from equirectangular image.
    
    Custom implementation for when py360convert is not available.
    """
    h_eq, w_eq = equirect.shape[:2]
    h_out, w_out = out_hw
    
    # Focal length from FoV
    fov_rad = np.radians(fov_deg)
    f = (w_out / 2) / np.tan(fov_rad / 2)
    
    # Create pixel grid for output image
    u = np.arange(w_out) - w_out / 2
    v = np.arange(h_out) - h_out / 2
    u, v = np.meshgrid(u, v)
    
    # Convert to 3D rays (camera coordinates)
    x = u
    y = -v  # Flip y for image coordinates
    z = np.full_like(u, f)
    
    # Stack and normalize
    xyz = np.stack([x, y, z], axis=-1)
    
    # Rotate to world coordinates
    R = create_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    xyz_rot = xyz @ R.T
    
    # Convert to spherical coordinates
    x_r, y_r, z_r = xyz_rot[..., 0], xyz_rot[..., 1], xyz_rot[..., 2]
    
    # Longitude (theta) and latitude (phi)
    theta = np.arctan2(x_r, z_r)  # [-pi, pi]
    phi = np.arcsin(np.clip(y_r / np.linalg.norm(xyz_rot, axis=-1), -1, 1))  # [-pi/2, pi/2]
    
    # Map to equirectangular coordinates
    u_eq = (theta / np.pi + 1) / 2 * w_eq  # [0, w_eq]
    v_eq = (0.5 - phi / np.pi) * h_eq  # [0, h_eq]
    
    # Interpolation
    if mode == "nearest":
        u_eq = np.round(u_eq).astype(int) % w_eq
        v_eq = np.clip(np.round(v_eq).astype(int), 0, h_eq - 1)
        return equirect[v_eq, u_eq]
    else:
        # Bilinear interpolation using cv2.remap
        map_x = u_eq.astype(np.float32) % w_eq
        map_y = np.clip(v_eq.astype(np.float32), 0, h_eq - 1)
        
        return cv2.remap(
            equirect,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )


def equirect_to_perspective(
    equirect: np.ndarray,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0,
    out_hw: Tuple[int, int] = (1024, 1024),
    mode: str = "bilinear"
) -> np.ndarray:
    """
    Extract perspective view from equirectangular image.
    
    Uses py360convert if available, otherwise falls back to custom implementation.
    """
    if HAS_PY360:
        return py360convert.e2p(
            equirect,
            fov_deg=fov_deg,
            u_deg=yaw_deg,
            v_deg=pitch_deg,
            in_rot_deg=roll_deg,
            out_hw=out_hw,
            mode=mode
        )
    else:
        return equirect_to_perspective_custom(
            equirect, fov_deg, yaw_deg, pitch_deg, roll_deg, out_hw, mode
        )


# =============================================================================
# Metashape XML Parsing
# =============================================================================

def parse_metashape_xml(xml_path: str) -> Dict:
    """
    Parse Metashape camera export XML.
    
    Returns dict mapping image names to camera poses (4x4 matrices).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    cameras = {}
    
    # Find the chunk with cameras
    for chunk in root.findall('.//chunk'):
        # Get the transform if it exists
        transform = np.eye(4)
        transform_elem = chunk.find('transform')
        if transform_elem is not None:
            rotation = transform_elem.find('rotation')
            translation = transform_elem.find('translation')
            scale = transform_elem.find('scale')
            
            if rotation is not None:
                R = np.array([float(x) for x in rotation.text.split()]).reshape(3, 3)
                transform[:3, :3] = R
            if translation is not None:
                t = np.array([float(x) for x in translation.text.split()])
                transform[:3, 3] = t
            if scale is not None:
                s = float(scale.text)
                transform[:3, :3] *= s
        
        # Parse cameras
        for camera in chunk.findall('.//camera'):
            label = camera.get('label', '')
            
            # Get camera transform
            cam_transform = camera.find('transform')
            if cam_transform is not None:
                values = [float(x) for x in cam_transform.text.split()]
                cam_matrix = np.array(values).reshape(4, 4)
                
                # Apply chunk transform
                world_matrix = transform @ cam_matrix
                
                cameras[label] = {
                    'matrix': world_matrix,
                    'position': world_matrix[:3, 3],
                    'rotation': world_matrix[:3, :3]
                }
    
    return cameras


# =============================================================================
# COLMAP Output Generation
# =============================================================================

def view_to_rotation_matrix(view: ViewConfig) -> np.ndarray:
    """Convert view config to rotation matrix (world to camera)."""
    return create_rotation_matrix(view.yaw, view.pitch, view.roll).T


def generate_colmap_cameras(
    views: List[ViewConfig],
    fov_deg: float,
    image_size: int,
    output_path: str
):
    """Generate COLMAP cameras.txt file."""
    # Calculate focal length
    f = (image_size / 2) / np.tan(np.radians(fov_deg) / 2)
    cx = cy = image_size / 2
    
    with open(output_path, 'w') as f_out:
        f_out.write("# Camera list with one line of data per camera:\n")
        f_out.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f_out.write(f"# Number of cameras: {len(views)}\n")
        
        for i, view in enumerate(views):
            # PINHOLE model: fx, fy, cx, cy
            f_out.write(f"{i+1} PINHOLE {image_size} {image_size} {f:.6f} {f:.6f} {cx:.6f} {cy:.6f}\n")


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def generate_colmap_images(
    views: List[ViewConfig],
    equirect_cameras: Dict,
    output_path: str,
    yaw_offset: float = 0.0
):
    """Generate COLMAP images.txt file."""
    with open(output_path, 'w') as f_out:
        f_out.write("# Image list with two lines of data per image:\n")
        f_out.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_out.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        image_id = 1
        
        for equirect_name, cam_data in equirect_cameras.items():
            base_name = Path(equirect_name).stem
            world_R = cam_data['rotation']
            world_t = cam_data['position']
            
            for view_idx, view in enumerate(views):
                # Apply yaw offset (can vary per frame for better diversity)
                effective_yaw = view.yaw + yaw_offset
                
                # Get view rotation (relative to equirect center)
                view_R = create_rotation_matrix(effective_yaw, view.pitch, view.roll).T
                
                # Combine: camera_from_world = view_R @ world_R^T
                # Since equirect camera looks outward, we need to compose properly
                cam_R = view_R @ world_R.T
                
                # Translation: camera position in world coordinates is same as equirect
                # cam_t = -cam_R @ world_t  (COLMAP convention)
                cam_t = -cam_R @ world_t
                
                # Convert to quaternion
                quat = rotation_matrix_to_quaternion(cam_R)
                
                # Image name
                img_name = f"{base_name}_{view.name}.png"
                
                f_out.write(f"{image_id} {quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f} ")
                f_out.write(f"{cam_t[0]:.9f} {cam_t[1]:.9f} {cam_t[2]:.9f} ")
                f_out.write(f"{view_idx + 1} {img_name}\n")
                f_out.write("\n")  # Empty line for points (none yet)
                
                image_id += 1


def generate_colmap_rig_config(
    views: List[ViewConfig],
    output_path: str
):
    """
    Generate COLMAP rig configuration JSON.
    
    This allows COLMAP to treat all views from the same equirect as a rig,
    constraining their relative poses during bundle adjustment.
    """
    rig_config = [{
        "cameras": []
    }]
    
    for i, view in enumerate(views):
        # Get rotation matrix for this view
        R = create_rotation_matrix(view.yaw, view.pitch, view.roll).T
        quat = rotation_matrix_to_quaternion(R)
        
        cam_config = {
            "image_prefix": f"*_{view.name}.",  # Match pattern
        }
        
        if i == 0:
            cam_config["ref_sensor"] = True
        else:
            # Express rotation as quaternion [qw, qx, qy, qz]
            cam_config["cam_from_rig_rotation"] = quat.tolist()
            cam_config["cam_from_rig_translation"] = [0, 0, 0]  # All views same position
        
        rig_config[0]["cameras"].append(cam_config)
    
    with open(output_path, 'w') as f:
        json.dump(rig_config, f, indent=2)


# =============================================================================
# Image Processing
# =============================================================================

def process_single_equirect(args):
    """Process a single equirectangular image (for multiprocessing)."""
    (
        equirect_path,
        mask_path,
        output_dir,
        mask_output_dir,
        views,
        fov_deg,
        out_size,
        yaw_offset,
        frame_idx
    ) = args
    
    # Load equirectangular image
    equirect = cv2.imread(str(equirect_path))
    if equirect is None:
        print(f"Warning: Could not load {equirect_path}")
        return None
    
    # Load mask if provided
    mask = None
    if mask_path and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    base_name = Path(equirect_path).stem
    results = []
    
    # Calculate frame-specific yaw offset (for diversity in 3DGS training)
    frame_yaw_offset = yaw_offset * frame_idx if yaw_offset != 0 else 0
    
    for view in views:
        effective_yaw = view.yaw + frame_yaw_offset
        
        # Extract perspective view
        persp = equirect_to_perspective(
            equirect,
            fov_deg=fov_deg,
            yaw_deg=effective_yaw,
            pitch_deg=view.pitch,
            roll_deg=view.roll,
            out_hw=(out_size, out_size),
            mode="bilinear"
        )
        
        # Save perspective image
        out_name = f"{base_name}_{view.name}.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), persp)
        
        # Process mask if available
        if mask is not None:
            mask_persp = equirect_to_perspective(
                mask,
                fov_deg=fov_deg,
                yaw_deg=effective_yaw,
                pitch_deg=view.pitch,
                roll_deg=view.roll,
                out_hw=(out_size, out_size),
                mode="nearest"  # Use nearest for masks to preserve binary values
            )
            
            mask_out_path = mask_output_dir / out_name
            cv2.imwrite(str(mask_out_path), mask_persp)
        
        results.append(out_name)
    
    return base_name, results


def process_images(
    image_dir: Path,
    mask_dir: Optional[Path],
    output_dir: Path,
    views: List[ViewConfig],
    fov_deg: float,
    out_size: int,
    num_workers: int,
    max_images: int,
    yaw_offset: float
):
    """Process all equirectangular images."""
    # Create output directories
    images_out = output_dir / "images"
    masks_out = output_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    
    if mask_dir:
        masks_out.mkdir(parents=True, exist_ok=True)
    
    # Find all equirectangular images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    equirect_files = []
    for ext in extensions:
        equirect_files.extend(image_dir.glob(ext))
    
    equirect_files = sorted(equirect_files)[:max_images]
    
    print(f"Found {len(equirect_files)} equirectangular images")
    
    # Prepare arguments for multiprocessing
    args_list = []
    for i, eq_path in enumerate(equirect_files):
        mask_path = None
        if mask_dir:
            # Try to find matching mask
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask = mask_dir / (eq_path.stem + ext)
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
        
        args_list.append((
            eq_path,
            mask_path,
            images_out,
            masks_out if mask_dir else None,
            views,
            fov_deg,
            out_size,
            yaw_offset,
            i
        ))
    
    # Process with multiprocessing
    processed = {}
    
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_single_equirect, args_list))
    else:
        results = [process_single_equirect(args) for args in args_list]
    
    for result in results:
        if result:
            base_name, output_names = result
            processed[base_name] = output_names
    
    print(f"Processed {len(processed)} images into {len(processed) * len(views)} perspective views")
    
    return processed


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert equirectangular images to multiple perspective views",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
View Presets:
  6-cubemap   Standard 6-face cubemap (90° FoV recommended)
  8-hex       6 horizontal (60° apart) + top/bottom (70-80° FoV)
  10-octa     8 horizontal (45° apart) + top/bottom (60-70° FoV) [RECOMMENDED]
  12-full     8 horizontal + 2 elevated + top/bottom (55-65° FoV)
  14-dense    8 horizontal + 4 elevated + top/bottom (50-60° FoV)

Example:
  python equirect_to_perspectives.py \\
    --images ./equirect_frames \\
    --xml ./metashape_cameras.xml \\
    --output ./colmap_ready \\
    --preset 10-octa \\
    --fov-deg 65 \\
    --crop-size 1920 \\
    --num-workers 8
        """
    )
    
    parser.add_argument('--images', required=True, type=str,
                        help='Directory of equirectangular images')
    parser.add_argument('--masks', type=str, default=None,
                        help='Directory of equirectangular masks (optional)')
    parser.add_argument('--xml', type=str, default=None,
                        help='Metashape XML camera export (optional)')
    parser.add_argument('--output', required=True, type=str,
                        help='Output directory for COLMAP-ready data')
    
    parser.add_argument('--preset', type=str, default='10-octa',
                        choices=list(VIEW_PRESETS.keys()),
                        help='View preset (default: 10-octa)')
    parser.add_argument('--fov-deg', type=float, default=65.0,
                        help='Field of view in degrees (default: 65)')
    parser.add_argument('--crop-size', type=int, default=1920,
                        help='Output image size (square, default: 1920)')
    
    parser.add_argument('--yaw-offset', type=float, default=0.0,
                        help='Per-frame yaw rotation offset for training diversity (default: 0)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--max-images', type=int, default=10000,
                        help='Maximum number of equirect images to process (default: 10000)')
    
    parser.add_argument('--skip-bottom', action='store_true',
                        help='Skip bottom view (often contains tripod/operator)')
    parser.add_argument('--generate-rig-config', action='store_true',
                        help='Generate COLMAP rig configuration JSON')
    
    args = parser.parse_args()
    
    # Validate inputs
    image_dir = Path(args.images)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)
    
    mask_dir = Path(args.masks) if args.masks else None
    if mask_dir and not mask_dir.exists():
        print(f"Warning: Mask directory not found: {mask_dir}")
        mask_dir = None
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get view configuration
    views = VIEW_PRESETS[args.preset].copy()
    
    if args.skip_bottom:
        views = [v for v in views if v.name != 'bottom']
        print("Skipping bottom view")
    
    print(f"Using preset '{args.preset}' with {len(views)} views")
    print(f"FoV: {args.fov_deg}°, Output size: {args.crop_size}x{args.crop_size}")
    
    # Calculate overlap
    if 'h45' in [v.name for v in views]:
        spacing = 45
    elif 'h60' in [v.name for v in views]:
        spacing = 60
    else:
        spacing = 90
    
    overlap = max(0, args.fov_deg - spacing)
    print(f"Horizontal overlap: ~{overlap:.1f}° ({overlap/args.fov_deg*100:.0f}%)")
    
    # Process images
    processed = process_images(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        views=views,
        fov_deg=args.fov_deg,
        out_size=args.crop_size,
        num_workers=args.num_workers,
        max_images=args.max_images,
        yaw_offset=args.yaw_offset
    )
    
    # Generate COLMAP files if XML provided
    if args.xml:
        print("\nGenerating COLMAP files...")
        
        cameras = parse_metashape_xml(args.xml)
        if not cameras:
            print("Warning: No cameras found in XML, generating identity poses")
            cameras = {name: {
                'matrix': np.eye(4),
                'position': np.zeros(3),
                'rotation': np.eye(3)
            } for name in processed.keys()}
        
        # Generate cameras.txt
        generate_colmap_cameras(
            views=views,
            fov_deg=args.fov_deg,
            image_size=args.crop_size,
            output_path=output_dir / "cameras.txt"
        )
        print(f"  Created cameras.txt")
        
        # Generate images.txt
        generate_colmap_images(
            views=views,
            equirect_cameras=cameras,
            output_path=output_dir / "images.txt",
            yaw_offset=args.yaw_offset
        )
        print(f"  Created images.txt")
        
        # Generate empty points3D.txt
        with open(output_dir / "points3D.txt", 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        print(f"  Created points3D.txt (empty)")
    
    # Generate rig config
    if args.generate_rig_config:
        generate_colmap_rig_config(
            views=views,
            output_path=output_dir / "rig_config.json"
        )
        print(f"  Created rig_config.json")
        print("\nTo use rig constraints in COLMAP:")
        print(f"  colmap rig_configurator \\")
        print(f"    --database_path {output_dir}/database.db \\")
        print(f"    --rig_config_path {output_dir}/rig_config.json")
    
    print(f"\n✓ Output written to: {output_dir}")
    print(f"  - images/: {len(processed) * len(views)} perspective images")
    if mask_dir:
        print(f"  - masks/: {len(processed) * len(views)} perspective masks")
    if args.xml:
        print(f"  - cameras.txt, images.txt, points3D.txt")
    
    print("\nRecommended FoV for each preset:")
    print("  6-cubemap: 90° (exact coverage, no overlap)")
    print("  8-hex:     70-80° (~17-33% overlap)")
    print("  10-octa:   60-70° (~33-56% overlap) [RECOMMENDED]")
    print("  12-full:   55-65° (~22-44% overlap)")
    print("  14-dense:  50-60° (~11-33% overlap)")


if __name__ == '__main__':
    main()
