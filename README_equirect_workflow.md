# Equirectangular to Multi-Perspective Workflow

## Overview

This workflow converts equirectangular (360°) images into multiple overlapping pinhole perspective views, enabling alignment with regular pinhole cameras in Metashape or COLMAP.

## Why This Approach?

### The Problem
- Metashape supports spherical cameras natively, but COLMAP and most 3DGS training pipelines don't
- Simple cubemap splitting (6 views at 90° FoV) creates views with minimal overlap at edges
- When mixed with pinhole cameras, feature matching suffers at cubemap seams

### The Solution
- Extract **8-12 overlapping perspective views** instead of 6 cubemap faces
- Use **COLMAP's rig constraints** to lock relative poses between views from the same 360° frame
- This provides more tie points for matching while maintaining geometric consistency

## View Configurations

| Preset | Views | Layout | Recommended FoV | Overlap |
|--------|-------|--------|-----------------|---------|
| `6-cubemap` | 6 | Standard cubemap | 90° | 0% |
| `8-hex` | 8 | 6 horizontal (60° apart) + top/bottom | 70-80° | 17-33% |
| **`10-octa`** | 10 | 8 horizontal (45° apart) + top/bottom | 60-70° | 33-56% |
| `12-full` | 12 | 8 horizontal + 2 elevated + top/bottom | 55-65° | 22-44% |
| `14-dense` | 14 | 8 horizontal + 4 elevated + top/bottom | 50-60° | 11-33% |

**Recommended:** `10-octa` with 65° FoV provides excellent overlap (~44%) for feature matching while keeping the number of images manageable.

## Installation

```bash
# Required
pip install numpy opencv-python

# Recommended (faster, better interpolation)
pip install py360convert

# Optional (GPU acceleration)
pip install pytorch360convert  # or pyequilib
```

## Basic Usage

### 1. Extract Perspectives from Equirectangular Images

```bash
python equirect_to_perspectives.py \
  --images ./360_frames \
  --output ./perspective_views \
  --preset 10-octa \
  --fov-deg 65 \
  --crop-size 1920 \
  --num-workers 8
```

### 2. With Masks (for operator/tripod removal)

```bash
python equirect_to_perspectives.py \
  --images ./360_frames \
  --masks ./360_masks \
  --output ./perspective_views \
  --preset 10-octa \
  --fov-deg 65 \
  --skip-bottom  # Skip bottom view (usually shows tripod)
```

### 3. Full Pipeline with Metashape Poses

```bash
python equirect_to_perspectives.py \
  --images ./360_frames \
  --masks ./360_masks \
  --xml ./metashape_cameras.xml \
  --output ./colmap_ready \
  --preset 10-octa \
  --fov-deg 65 \
  --crop-size 1920 \
  --num-workers 8 \
  --yaw-offset 15 \
  --generate-rig-config
```

## Integration with Metashape

### Step 1: Align 360° Images in Metashape
1. Import equirectangular images
2. **Tools → Camera Calibration → Camera Type → Spherical**
3. Align photos
4. (Optional) Add ground control points

### Step 2: Export from Metashape
```
File → Export Cameras... → Select XML format
File → Export Point Cloud... → Select PLY format (optional)
```

### Step 3: Convert to Perspectives
```bash
python equirect_to_perspectives.py \
  --images ./360_frames \
  --xml ./metashape_cameras.xml \
  --output ./colmap_ready \
  --preset 10-octa \
  --fov-deg 65 \
  --generate-rig-config
```

### Step 4: Import into COLMAP (with your pinhole images)

Option A: **Use Metashape poses directly**
```bash
# Copy your pinhole images to the same images folder
cp ./pinhole_images/*.jpg ./colmap_ready/images/

# Import the pre-computed model
colmap model_converter \
  --input_path ./colmap_ready \
  --output_path ./colmap_ready/sparse/0 \
  --output_type BIN
```

Option B: **Re-align everything in COLMAP with rig constraints**
```bash
# Feature extraction
colmap feature_extractor \
  --database_path ./colmap_ready/database.db \
  --image_path ./colmap_ready/images \
  --ImageReader.single_camera_per_folder 0

# Configure rig constraints
colmap rig_configurator \
  --database_path ./colmap_ready/database.db \
  --rig_config_path ./colmap_ready/rig_config.json

# Feature matching
colmap exhaustive_matcher \
  --database_path ./colmap_ready/database.db

# Mapping with rig constraints
colmap mapper \
  --database_path ./colmap_ready/database.db \
  --image_path ./colmap_ready/images \
  --output_path ./colmap_ready/sparse
```

## COLMAP Rig Support (Important!)

Starting with COLMAP 3.12, there's native support for **sensor rigs**. This is crucial for 360→perspective workflows because:

1. All perspective views from the same 360° frame share the same position
2. Their relative rotations are fixed and known
3. COLMAP can enforce these constraints during bundle adjustment

The `--generate-rig-config` flag creates a `rig_config.json` that tells COLMAP to treat perspectives from the same 360° image as a rigid rig.

### Without Rig Constraints
- Each perspective view is treated as an independent camera
- Bundle adjustment can drift relative poses
- Results in inconsistent alignment, especially at view boundaries

### With Rig Constraints
- Relative poses between views from same 360° frame are locked
- Only the rig position/orientation is optimized
- Much more stable alignment

## Advanced Options

### Yaw Offset for Training Diversity
The `--yaw-offset` parameter rotates the view extraction by a small amount each frame. This helps 3DGS training by ensuring different regions get sampled across frames:

```bash
--yaw-offset 15  # Rotate 15° per frame
```

### Custom View Configurations
Edit the `VIEW_PRESETS` dictionary in the script to create custom view layouts. Each view needs:
- `name`: Unique identifier
- `yaw`: Horizontal angle (0° = front, 90° = right)
- `pitch`: Vertical angle (0° = horizon, 90° = up)
- `roll`: Optional rotation around view axis

### Skip Bottom View
The bottom view often shows tripod, operator feet, or other unwanted content:

```bash
--skip-bottom
```

## Troubleshooting

### Poor Alignment at View Boundaries
- Increase FoV for more overlap
- Use `10-octa` or `12-full` preset instead of `6-cubemap`
- Enable rig constraints in COLMAP

### Memory Issues
- Reduce `--crop-size`
- Reduce `--num-workers`
- Process in batches with `--max-images`

### Masks Not Aligning
- Ensure mask filenames match image filenames
- Masks should be binary (white = keep, black = remove)
- Use `--invert-mask` if your masks are inverted

## Output Structure

```
output/
├── images/           # Perspective images
│   ├── frame001_h0.png
│   ├── frame001_h45.png
│   ├── frame001_h90.png
│   └── ...
├── masks/            # Perspective masks (if --masks provided)
│   ├── frame001_h0.png
│   └── ...
├── cameras.txt       # COLMAP camera intrinsics
├── images.txt        # COLMAP camera poses
├── points3D.txt      # Empty (populated after COLMAP reconstruction)
└── rig_config.json   # COLMAP rig configuration (if --generate-rig-config)
```

## Dependencies & Alternatives

| Library | Purpose | Notes |
|---------|---------|-------|
| **py360convert** | CPU projection | Recommended, well-maintained |
| **pytorch360convert** | GPU projection | For large datasets |
| **equilib (pyequilib)** | PyTorch batch processing | If already using PyTorch |
| **OmniCV** | General 360 processing | More features, heavier |

## References

- [COLMAP Rig Support Documentation](https://colmap.github.io/rigs.html)
- [py360convert GitHub](https://github.com/sunset1995/py360convert)
- [Metashape 360 to COLMAP (original)](https://github.com/Kotohibi/Metashape_360_to_COLMAP_plane)
- [360 Gaussian Splatting](https://github.com/inuex35/360-gaussian-splatting)

## License

MIT
