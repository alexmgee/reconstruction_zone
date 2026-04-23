# Align Tab

The Align tab runs sparse 3D reconstruction on your images using COLMAP or SphereSfM. It takes a folder of images (and optional masks from the Mask/Review workflow), runs feature extraction, matching, and reconstruction, and produces a sparse point cloud with camera poses. This is the step that turns 2D photos into 3D geometry — the foundation for dense reconstruction and Gaussian Splatting.

The tab is designed as an advanced surface: the left column exposes the full set of reconstruction parameters, while the right column shows live run status, a point cloud viewer, and alignment summary.

## Layout

| Column | Contents |
|--------|----------|
| **Left** (scrollable) | Inputs, Engine toggle, Engine Settings, Rig Configuration, Extract, Match, Reconstruct, Advanced, Actions, Run Status, Log |
| **Right** (detail panel) | Header with stage indicators, point cloud viewer, viewer controls, alignment summary |

---

## Inputs

Three path fields at the top:

- **Images** — Directory of images to reconstruct. Auto-detects the workspace when you tab out of this field.
- **Masks** — Optional directory of masks. When provided, COLMAP will ignore masked pixels during feature extraction. These should be the masks produced by the Mask tab (white = ignore, matching the COLMAP convention after remapping).
- **Workspace** — Root directory for alignment outputs. Each run gets its own timestamped subdirectory under this path, so previous runs are never overwritten.

## Engine

Radio toggle between two reconstruction backends:

| Engine | Camera model | Best for |
|--------|-------------|----------|
| **COLMAP** | Configurable (PINHOLE, OPENCV, OPENCV_FISHEYE, etc.) | Perspective images, reframed 360° views, fisheye |
| **SphereSfM** | SPHERE (fixed) | Native equirectangular images (no reframing needed) |

Switching engines swaps the settings panels and adjusts which controls are available. SphereSfM locks the camera model to SPHERE, disables rig configuration, and locks BA refinement checkboxes.

## Engine Settings

Engine-specific parameters that appear after the radio toggle:

**COLMAP:**
- **Camera model** — Lens distortion model. Use PINHOLE for reframed perspective images (no distortion). Use OPENCV or OPENCV_FISHEYE for raw fisheye or wide-angle captures where you want COLMAP to estimate distortion. Use SIMPLE_RADIAL for moderate-distortion lenses.
- **Single camera** — When checked (default), all images share one calibration. Uncheck if images come from different cameras or focal lengths.

**SphereSfM:**
- Camera model is always SPHERE. The pose path is configurable under Advanced > Camera And Reader.

## Rig Configuration

Collapsible, collapsed by default. Only relevant for multi-camera rigs where images have known relative poses (e.g., reframed 360° views extracted from the same equirectangular frame).

- **Preset** — Built-in rig configurations for common setups. Selecting a preset auto-fills the rig config file path.
- **File** — Path to a custom `rig_config.json` defining camera groups and relative poses.
- **Summary** — Read-only preview of the loaded rig configuration.

Rig configuration is disabled (greyed out) when SphereSfM is selected.

## Extract

Feature extraction settings — controls how COLMAP detects keypoints in each image.

- **Max features** — Maximum keypoints per image (default: 8192). Higher values improve matching at the cost of speed and memory. For large datasets or detailed scenes, 16384 or 32768 can help.
- **Max image size** — Images larger than this (in pixels on the longest side) are downscaled for feature extraction. Reduce this if you're running out of GPU memory.

## Match

Feature matching settings — controls how COLMAP finds correspondences between images.

- **Strategy** — How image pairs are selected for matching:
  - **Exhaustive** — Every pair. Best quality, O(n²) cost. Practical up to ~500 images.
  - **Sequential** — Adjacent images only (assumes ordered capture). Fast, good for video-derived sequences.
  - **Spatial** — Pairs based on GPS proximity. Requires geo-tagged images.
  - **Vocab tree** — Visual similarity via a vocabulary tree. Scales to thousands of images. Requires a vocabulary tree file.
- **Max matches** — Maximum verified matches per image pair (default: 32768).
- **Guided matching** — Re-matches using the estimated geometry to find additional inliers. Slower but can recover more correspondences.

Strategy-specific fields appear/hide dynamically:
- **Spatial** → GPS toggle, max distance
- **Vocab tree** → vocabulary tree file path

## Reconstruct

Controls the Structure-from-Motion mapper.

- **Mapper** — `incremental` (default, image-by-image) or `global` (all-at-once via COLMAP's global_mapper). Incremental is more robust; global is faster for well-connected datasets.
- **Min inliers** — Minimum 2D-3D correspondences to register an image (default: 15). Lower values register more images but risk bad poses.
- **Refine focal** — Allow bundle adjustment to optimize focal length. Usually on.
- **Refine principal** — Allow BA to optimize the principal point. Usually off unless you have reason to believe the principal point is off-center.
- **Refine extra** — Allow BA to optimize distortion parameters. On for OPENCV/fisheye, less important for PINHOLE.
- **Snapshots** — Optional directory + frequency for saving intermediate reconstructions. Useful for monitoring long runs.

## Advanced

Collapsed by default. Contains three nested sections:

### Tool Setup

- **Binary path** — Path to the COLMAP or SphereSfM executable. Auto-detected from known locations and environment variables (`COLMAP_BINARY`, `SPHERESFM_BINARY`), or set manually.
- **GPU index** — Which GPU to use (default: 0, or -1 for CPU). Only relevant for feature extraction and matching.

### Camera And Reader

Advanced camera configuration:

- **Camera params** — Fixed intrinsics string (e.g., `960,960,960,540` for a known pinhole). When set, COLMAP skips calibration and uses these values directly.
- **Mask type** — How masks are applied during feature extraction (`BLACK` or `WHITE`).
- **Image reader** — Database reader type (usually left at default).

### Per-Stage CLI Args

Raw command-line argument passthrough for each COLMAP stage. Three text fields:

- **Extract args** — Appended to `colmap feature_extractor`
- **Match args** — Appended to `colmap exhaustive_matcher` (or whichever matcher)
- **Reconstruct args** — Appended to `colmap mapper`

Use these for edge cases not covered by the UI (e.g., `--SiftExtraction.domain_size_pool 0`).

## Actions

Four buttons:

| Button | Action |
|--------|--------|
| **Align** | Run the full pipeline: extract → match → reconstruct |
| **Next Step** | Run only the next incomplete stage (useful for debugging or inspecting intermediate results) |
| **Cancel** | Stop the current run (sets a cancel flag; the current stage finishes before stopping) |
| **Load Model** | Load a sparse model from disk into the point cloud viewer without running alignment |

## Run Status & Model

Live diagnostics during and after a run:

- **Run Dir** — Path to the current run's output directory
- **Masks** — Whether masks were detected and how many
- **Snapshots** — Snapshot save status
- **Progress** — Current stage and status

## Detail Panel (Right Column)

### Header Bar

Shows the pipeline stages as colored labels: **Extract → Match → Reconstruct**. Colors update in real-time:

| Color | Meaning |
|-------|---------|
| Gray | Pending |
| Amber | Running |
| Green | Complete |
| Orange | Warning (completed with issues) |
| Red | Failed |

After reconstruction, a quality badge appears (based on reprojection error and image registration rate), plus inline stats: point count, registered images, and mean reprojection error.

### Point Cloud Viewer

Displays the sparse reconstruction after alignment completes (or when loaded via "Load Model"). Shows 3D points colored by their appearance, with camera frustums indicating pose.

Controls below the viewer:
- **Reset View** — Return to default orbit position
- **Roll slider** — CW/CCW rotation for correcting the up-axis
- **Open Folder** — Open the run directory in the file explorer
- **Copy Path** — Copy the sparse model path to clipboard

### Alignment Summary

A text summary of the reconstruction results: number of images registered vs. total, point count, reprojection error statistics, and any warnings about poor registration or high error.

---

## Typical Workflow

1. Set **Images** to your reframed perspective views (output of Extract tab)
2. Set **Masks** to the reviewed masks (output of Mask → Review workflow)
3. Leave engine as **COLMAP**, camera model as **PINHOLE**, strategy as **sequential** (for video-derived frames)
4. Click **Align**
5. Watch stages progress in the header bar and log
6. When complete, inspect the point cloud in the viewer and check the summary for registration rate and reprojection error
7. If the result looks good, the sparse model is ready for dense reconstruction or Gaussian Splatting training

## Tips

- **Low registration rate** (<80% of images registered) usually means matching failed. Try: increase max features, switch to exhaustive matching, or check that images have sufficient overlap.
- **High reprojection error** (>1.5 px) suggests calibration issues. Try: switch to OPENCV camera model if using raw captures, or enable "Refine principal."
- **SphereSfM** is experimental. Use it only for native equirectangular images that you don't want to reframe. For most workflows, reframe to perspectives first and use COLMAP.
- **Rig configuration** is for multi-view rigs where relative poses are known. If you reframed 360° video into perspective views with prep360, the rig preset tells COLMAP which views came from the same frame.
