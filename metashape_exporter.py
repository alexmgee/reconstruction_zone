"""
metashape_exporter.py -- Export Metashape alignment data for prep360.

Run inside Agisoft Metashape Pro via Tools > Run Script.

Exports to a 'metashape_export' folder next to the .psx file:
  cameras.xml  -- Camera poses and sensor calibrations
  points.ply   -- Sparse point cloud with colors
  images/      -- Copies of source images (aligned cameras only)
  masks/       -- Exported masks (if any cameras have masks)

After export, use prep360 to generate COLMAP/XMP:
  python -m prep360 colmap --xml metashape_export/cameras.xml \\
      --images metashape_export/images \\
      --ply metashape_export/points.ply \\
      --output ./colmap_ready
"""

import os
import shutil

try:
    import Metashape
except ImportError:
    raise RuntimeError(
        "This script must be run inside Agisoft Metashape Pro "
        "(Tools > Run Script)."
    )


def main():
    doc = Metashape.app.document
    if not doc.path:
        Metashape.app.messageBox("Save the project first, then re-run.")
        return

    chunk = doc.chunk
    if not chunk:
        Metashape.app.messageBox("No active chunk found.")
        return

    output_dir = os.path.join(os.path.dirname(doc.path), "metashape_export")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting to: {output_dir}")

    # 1. Export cameras as XML
    cameras_path = os.path.join(output_dir, "cameras.xml")
    chunk.exportCameras(
        cameras_path,
        format=Metashape.CamerasFormat.CamerasFormatXML,
    )
    print(f"  cameras.xml: done")

    # 2. Export sparse point cloud as PLY
    points_path = os.path.join(output_dir, "points.ply")
    try:
        chunk.exportPointCloud(
            points_path,
            source_data=Metashape.DataSource.TiePointsData,
            format=Metashape.PointCloudFormat.PointCloudFormatPLY,
            save_colors=True,
        )
        print(f"  points.ply: done")
    except Exception as e:
        print(f"  points.ply: FAILED ({e})")

    # 3. Copy source images (aligned cameras only)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    copied = 0
    for cam in chunk.cameras:
        if cam.transform is None:
            continue
        if not cam.photo or not cam.photo.path:
            continue
        src = cam.photo.path
        if not os.path.exists(src):
            print(f"  WARNING: source not found: {src}")
            continue
        dst = os.path.join(images_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied += 1
    print(f"  images/: {copied} files")

    # 4. Export masks (if any cameras have them)
    has_masks = any(
        cam.mask is not None
        for cam in chunk.cameras
        if cam.transform is not None
    )
    if has_masks:
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        mask_count = 0
        for cam in chunk.cameras:
            if cam.transform is None or cam.mask is None:
                continue
            label = cam.label
            mask_path = os.path.join(masks_dir, f"{os.path.splitext(label)[0]}.png")
            try:
                cam.mask.image().save(mask_path)
                mask_count += 1
            except Exception:
                pass
        print(f"  masks/: {mask_count} files")
    else:
        print(f"  masks/: none (no masks found)")

    aligned = sum(1 for c in chunk.cameras if c.transform is not None)
    total = len(chunk.cameras)
    print(f"\nDone. {aligned}/{total} aligned cameras exported.")
    print(f"Output: {output_dir}")
    print(f"\nNext step:")
    print(f"  python -m prep360 colmap --xml {cameras_path} "
          f"--images {images_dir} --ply {points_path} --output ./colmap_ready")


main()
