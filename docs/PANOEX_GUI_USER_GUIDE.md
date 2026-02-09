# Panoex GUI User Guide

A comprehensive guide to using the Panoex 360 Video Processing Pipeline graphical interface for preparing 360-degree video footage for photogrammetry reconstruction.

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Tab: Analyze](#tab-analyze)
4. [Tab: Extract](#tab-extract)
5. [Tab: Reframe](#tab-reframe)
6. [Tab: LUT](#tab-lut)
7. [Tab: Sky Filter](#tab-sky-filter)
8. [Tab: Segment](#tab-segment)
9. [Tab: Pipeline](#tab-pipeline)
10. [Typical Workflows](#typical-workflows)
11. [Troubleshooting](#troubleshooting)

---

## Overview

Panoex is a modular 360-degree video processing pipeline designed specifically for photogrammetry workflows. It takes 360 video footage (equirectangular format) and processes it into perspective images suitable for 3D reconstruction software like COLMAP, Metashape, or Reality Capture.

### Key Capabilities

- **Frame Extraction**: Extract frames from 360 video at configurable intervals
- **Reframing**: Convert equirectangular images to multiple perspective views
- **Color Correction**: Apply LUT files and shadow/highlight adjustments
- **Sky Filtering**: Remove sky-only images that are useless for reconstruction
- **Segmentation**: Generate masks for removing photographers, equipment, and dynamic objects
- **Batch Processing**: Process multiple videos with a persistent queue

### Interface Layout

The GUI consists of:
- **Top area**: Tab navigation (Analyze, Extract, Reframe, LUT, Sky Filter, Segment, Pipeline)
- **Main area**: Tab-specific controls (left panel) and information/preview (right panel)
- **Bottom area**: Progress bar and log output

---

## Getting Started

### Launching the Application

```bash
python panoex_gui.py
```

The application opens with a dark theme interface. The log panel at the bottom displays status messages and confirms that the core library loaded successfully.

### Prerequisites

- Python 3.8+
- Required packages: `customtkinter`, `opencv-python`, `numpy`, `Pillow`
- Optional: `ultralytics` (for segmentation)

### Understanding the Workflow

A typical photogrammetry workflow using Panoex follows these steps:

1. **Analyze** your 360 video to understand its properties
2. **Extract** frames at an appropriate interval
3. **Reframe** equirectangular frames to perspective views
4. **Filter** sky-only images (optional)
5. **Apply LUT** color correction if footage is in log format (optional)
6. **Segment** to create masks for removing unwanted objects (optional)

---

## Tab: Analyze

The Analyze tab examines your 360 video file and provides detailed information to help you choose optimal extraction settings.

### Controls

| Control | Description |
|---------|-------------|
| **Video** | Path to your 360 video file (.mp4, .mov, .avi, .mkv, .360, .insv) |
| **Analyze Video** | Run the analysis |

### Analysis Results

The analysis displays:

- **File Information**: Filename, format, codec
- **Technical Specs**: Resolution, FPS, duration, frame count, bitrate, pixel format
- **360 Detection**: Whether the video is equirectangular (2:1 aspect ratio)
- **Color Profile**: Detected log format (D-Log M, I-Log, etc.)
- **Recommendations**: Suggested extraction interval and LUT file

### How to Use

1. Click the "..." button next to Video and select your 360 video file
2. Click "Analyze Video"
3. Review the results in the right panel
4. Note the recommended extraction interval for use in the Extract tab

### Tips

- A 2:1 aspect ratio (e.g., 5760x2880) indicates equirectangular format
- If log format is detected, plan to use the LUT tab for color correction
- The recommended interval is based on video FPS and typical walking speeds

---

## Tab: Extract

The Extract tab processes video files into individual equirectangular frames. It features a batch queue system for processing multiple videos.

### Left Panel: Settings

#### Output Folder
Where extracted frames will be saved. Each video creates a subfolder named after the video file.

#### Extraction Settings

| Setting | Description | Values |
|---------|-------------|--------|
| **Mode** | Extraction algorithm | `fixed` (time-based), `scene` (scene changes), `adaptive` (motion-based) |
| **Interval** | Time between frames | 0.5s - 10.0s (slider) |
| **Quality** | JPEG quality | 70 - 100 |
| **Format** | Output format | JPEG or PNG |

**Mode Recommendations:**
- `fixed`: Best for consistent walking speed. Use 2.0s for walking, 1.0s for faster movement.
- `scene`: Good for varied footage with distinct scenes.
- `adaptive`: Adjusts based on motion; useful for mixed-speed footage.

#### Color Adjustments

| Setting | Description | Values |
|---------|-------------|--------|
| **Shadows** | Lift or crush dark tones | 0-100 (50 = neutral) |
| **Highlights** | Compress or boost bright tones | 0-100 (50 = neutral) |

- Values > 50 for Shadows: Lifts (brightens) dark areas
- Values < 50 for Shadows: Crushes (darkens) dark areas
- Values > 50 for Highlights: Compresses (darkens) bright areas
- Values < 50 for Highlights: Boosts (brightens) bright areas

These adjustments are applied after frame extraction.

#### Time Range (Optional)

| Setting | Description | Format |
|---------|-------------|--------|
| **Start** | Begin extraction at this time | Seconds or `MM:SS` or `HH:MM:SS` |
| **End** | Stop extraction at this time | Seconds or `MM:SS` or `HH:MM:SS` |

Leave empty to process the entire video.

### Right Panel: Video Queue

The queue system allows batch processing of multiple videos.

#### Queue Controls

| Button | Description |
|--------|-------------|
| **Add Videos** | Select one or more video files to add to queue |
| **Add Folder** | Add all video files from a folder |
| **Remove** | Remove selected items from queue |
| **Clear Done** | Remove completed and errored items |
| **Move Up/Down** | Reorder selected item in queue |

#### Queue Display

Each video in the queue shows:
- **Status indicator** (colored dot): Gray (pending), Orange (processing), Green (done), Red (error)
- **Filename**: The video file name
- **Status text**: Current state and progress percentage
- **Progress bar**: Visual progress for processing items

Click on a queue item to select it (highlighted in blue).

#### Processing Controls

| Button | Description |
|--------|-------------|
| **Process Queue** | Start processing all pending videos |
| **Stop** | Cancel current processing |

### Queue Persistence

The queue state is automatically saved to `~/.panoex_queue.json`. If the application closes during processing:
- Completed items remain marked as done
- Processing items are reset to pending on restart
- Queue order and settings are preserved

### How to Use

1. Set the **Output Folder** where frames will be saved
2. Configure extraction settings (mode, interval, quality, format)
3. Optionally adjust shadows/highlights and time range
4. Click **Add Videos** or **Add Folder** to populate the queue
5. Review and reorder the queue as needed
6. Click **Process Queue** to begin
7. Monitor progress in the queue display and log panel

### Tips

- For photogrammetry, 2.0s interval at walking speed yields good overlap
- Use JPEG format at 95 quality for balance of size and quality
- Shadow lifting (60-70) helps recover detail in dark 360 footage
- Each video creates a subfolder: `output_folder/video_name/frame_0001.jpg`

---

## Tab: Reframe

The Reframe tab converts equirectangular (360) images into multiple perspective (pinhole camera) views. This is essential for photogrammetry software that expects standard camera images.

### Left Panel: Controls

#### Input / Output

| Control | Description |
|---------|-------------|
| **Input** | Folder containing equirectangular frames |
| **Output** | Folder where perspective views will be saved |

#### View Configuration

| Setting | Description |
|---------|-------------|
| **Preset** | Predefined view configuration |
| **Size** | Output image dimensions (512 - 3840 pixels) |
| **Quality** | JPEG quality (70-100) |
| **Workers** | Number of parallel processing threads (1-16) |

#### Available Presets

| Preset | Description | Total Views |
|--------|-------------|-------------|
| `panoex_default` | 8 horizon views + 4 below + zenith | 13 views |
| `dense_coverage` | 8 horizon + 4 above + 4 below + zenith + nadir | 18 views |
| `lightweight` | 6 horizon views only | 6 views |

Each preset defines:
- **Rings**: Groups of views at specific pitch angles
- **FOV**: Field of view for each ring
- **Zenith/Nadir**: Optional straight-up and straight-down views

#### Options

| Option | Description |
|--------|-------------|
| **Include zenith** | Add a straight-up view (for sky/ceiling) |
| **Include nadir** | Add a straight-down view (for ground/floor) |

### Right Panel: Preview

The preview displays:
- **Equirectangular image** from the input folder (first image found)
- **Colored overlays** showing where each ring's views will be extracted
- **Legend** explaining the color coding for each ring

The preview updates when you:
- Select a new input folder
- Change the preset
- Toggle zenith/nadir options

#### Legend Colors

- **Red (Ring 0)**: Horizon level views
- **Teal (Ring 1)**: First offset ring
- **Yellow (Ring 2)**: Second offset ring
- **Sky Blue**: Zenith view
- **Tan**: Nadir view

### How to Use

1. Set **Input** to your folder of equirectangular frames
2. Set **Output** to where perspective views should be saved
3. Select a **Preset** based on your coverage needs
4. Review the preview to understand view placement
5. Adjust **Size** based on your source resolution (1920 is typical)
6. Set **Workers** based on your CPU cores (4-8 is typical)
7. Click **Reframe Images**

### Output Naming

Output files follow the pattern: `original_name_viewXX.jpg`

For example, `frame_0001.jpg` with 13 views produces:
- `frame_0001_view00.jpg` through `frame_0001_view12.jpg`

### Tips

- More views = better reconstruction but more processing time
- Output size should not exceed your source resolution
- Higher worker count speeds processing but uses more RAM
- Zenith views are often just sky; consider filtering them later
- 65-75 FOV provides good overlap between adjacent views

---

## Tab: LUT

The LUT (Look-Up Table) tab applies color correction to images, converting log footage to standard color space.

### Left Panel: Controls

#### Input / Output

| Control | Description |
|---------|-------------|
| **Input** | Single image or folder of images |
| **Output** | Output folder for corrected images |

#### LUT Settings

| Setting | Description |
|---------|-------------|
| **LUT File** | Path to .cube LUT file |
| **Strength** | LUT application intensity (0-100%) |

#### Shadow / Highlight

| Setting | Description |
|---------|-------------|
| **Shadows** | Lift or crush dark tones (0-100, 50 = neutral) |
| **Highlights** | Compress or boost bright tones (0-100, 50 = neutral) |

Shadow/highlight adjustments are applied AFTER the LUT.

### Common LUT Files

| Camera | Log Format | LUT |
|--------|-----------|-----|
| DJI | D-Log M | D-Log M to Rec.709 |
| Insta360 | I-Log | I-Log to Rec.709 |
| GoPro | Protune Flat | GoPro Flat to Rec.709 |

### How to Use

1. Set **Input** to your images (folder or single file)
2. Set **Output** folder
3. Select your **.cube LUT file**
4. Adjust **Strength** (100% for full correction)
5. Optionally adjust **Shadow/Highlight** for fine-tuning
6. Click **Apply LUT**

### Workflow Tips

- Apply LUT before other processing for consistent color
- Use shadow lift (55-65) to recover detail in dark areas
- Use highlight compression (55-65) for overexposed footage
- The LUT converts log to Rec.709; adjustments refine the result

---

## Tab: Sky Filter

The Sky Filter tab removes images that show only sky, which are useless for photogrammetry reconstruction.

### Left Panel: Controls

#### Input / Output

| Control | Description |
|---------|-------------|
| **Input** | Folder of images to analyze |
| **Output** | Folder where kept images will be copied |

#### Detection Thresholds

| Setting | Description | Default |
|---------|-------------|---------|
| **Brightness** | Minimum brightness to consider as sky | 0.85 |
| **Saturation** | Maximum saturation for sky detection | 0.15 |
| **Keypoints** | Minimum keypoints for non-sky classification | 50 |

Lower brightness threshold = stricter (fewer images classified as sky)
Higher keypoint threshold = stricter (more keypoints required to keep image)

#### Options

| Option | Description |
|--------|-------------|
| **Analyze only** | Report sky detection without copying files |
| **Dry run** | Show what would happen without making changes |

### Detection Criteria

An image is classified as "sky" if it matches 3 of 4 criteria:
1. High brightness (above threshold)
2. Low saturation (below threshold)
3. Few keypoints (below threshold)
4. Low edge density

### How to Use

1. Set **Input** to your folder of perspective views
2. Enable **Analyze only** to test your thresholds first
3. Click **Filter Sky Images** and review the log
4. Adjust thresholds if needed
5. Disable **Analyze only** and set **Output** folder
6. Run again to copy non-sky images to output

### Tips

- Run analyze-only first to calibrate thresholds
- Zenith views from 360 cameras often show empty sky
- Keep the output separate from input for safety
- Adjust brightness threshold based on your footage's exposure

---

## Tab: Segment

The Segment tab uses YOLO AI to detect and create masks for objects you want to exclude from photogrammetry reconstruction (photographers, equipment, vehicles, etc.).

### Prerequisites

```bash
pip install ultralytics
```

### Left Panel: Controls

#### Input / Output

| Control | Description |
|---------|-------------|
| **Input** | Folder of images to process |
| **Output** | Folder where mask images will be saved |

#### Detection Classes

| Setting | Description |
|---------|-------------|
| **Preset** | Predefined class combinations |
| **Custom** | Comma-separated class names or IDs |

#### Available Presets

| Preset | Classes Detected |
|--------|------------------|
| `person` | People only |
| `photographer` | Person, backpack, umbrella, handbag, suitcase |
| `equipment` | Person + accessories + cell phone |
| `vehicles` | Cars, motorcycles, buses, trucks, bicycles |
| `animals` | Common animals (birds, cats, dogs, horses, etc.) |
| `all_dynamic` | People + vehicles + animals |

Leave Custom empty to use the selected preset, or enter specific classes:
- By name: `person, backpack, car`
- By COCO ID: `0, 24, 2`

#### Model Settings

| Setting | Description |
|---------|-------------|
| **Model** | YOLO model size (n/s/m/l/x) |
| **Confidence** | Detection threshold (0.1 - 0.9) |
| **Dilate** | Expand masks by N pixels |

#### Model Size Guide

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| yolo11n | Fastest | Good | Quick preview |
| yolo11s | Fast | Better | Most cases |
| yolo11m | Medium | High | Balanced |
| yolo11l | Slow | Higher | Accuracy priority |
| yolo11x | Slowest | Highest | Maximum quality |

#### Options

| Option | Description |
|--------|-------------|
| **Invert mask** | White background, black objects (COLMAP style) |

### How to Use

1. Set **Input** folder containing your images
2. Set **Output** folder for mask images
3. Select a **Preset** or enter **Custom** classes
4. Choose **Model** based on speed/accuracy needs
5. Adjust **Confidence** (lower = more detections, more false positives)
6. Set **Dilate** to expand masks (covers edges better)
7. Enable **Invert mask** if your software expects that format
8. Click **Generate Masks**

### Output

Masks are saved as PNG files with the same names as input images:
- `frame_0001.jpg` → `frame_0001.png` (mask)

Mask format:
- Standard: Black background, white objects
- Inverted: White background, black objects

### Tips

- Use `photographer` preset for typical 360 capture workflows
- Lower confidence (0.3-0.4) catches more objects but may have false positives
- Dilate 5-15 pixels helps cover object edges
- Check a few masks visually before processing the entire set

---

## Tab: Pipeline

The Pipeline tab runs the complete processing workflow in one operation, automating the sequence of Extract → Reframe → Sky Filter → LUT.

### Left Panel: Controls

#### Input / Output

| Control | Description |
|---------|-------------|
| **Video** | 360 video file to process |
| **Output** | Base folder for all outputs |

#### Pipeline Steps

| Step | Description | Enabled by Default |
|------|-------------|-------------------|
| 1. Extract frames | Video → equirectangular frames | Yes |
| 2. Reframe | Equirectangular → perspective views | Yes |
| 3. Filter sky | Remove sky-only images | No |
| 4. Apply LUT | Color correction | No |

Check/uncheck steps to customize the pipeline.

#### Settings

| Setting | Description |
|---------|-------------|
| **Interval** | Frame extraction interval (0.5 - 10.0s) |
| **Preset** | View configuration for reframing |
| **Size** | Output image dimensions |
| **Workers** | Parallel processing threads |

### Output Structure

```
output_folder/
├── frames/         # Extracted equirectangular frames
├── perspectives/   # Reframed perspective views
└── filtered/       # Sky-filtered images (if enabled)
```

### How to Use

1. Select your 360 **Video** file
2. Choose the **Output** base folder
3. Enable/disable pipeline steps as needed
4. Configure settings (interval, preset, size)
5. Click **Run Pipeline**
6. Monitor progress in the log panel

### Tips

- Use the Pipeline tab for end-to-end processing of single videos
- For batch processing multiple videos, use the Extract tab's queue
- The pipeline uses default settings for each step; use individual tabs for fine control
- Enable sky filtering for footage with many upward-facing views

---

## Typical Workflows

### Workflow 1: Basic 360 Video to Photogrammetry

1. **Analyze** → Understand video properties
2. **Extract** → 2.0s interval, JPEG 95
3. **Reframe** → `panoex_default` preset, 1920px
4. Import perspective views into reconstruction software

### Workflow 2: Log Footage with Color Correction

1. **Analyze** → Confirm log format detection
2. **Extract** → Extract frames at appropriate interval
3. **LUT** → Apply manufacturer's LUT + shadow lift (60)
4. **Reframe** → Convert to perspective views
5. Import into reconstruction software

### Workflow 3: Removing Photographer from Footage

1. **Extract** → Get frames from video
2. **Reframe** → Convert to perspective views
3. **Segment** → `photographer` preset, confidence 0.4, dilate 10px
4. Import images + masks into reconstruction software

### Workflow 4: High-Quality Dense Reconstruction

1. **Analyze** → Check resolution and format
2. **Extract** → 1.0s interval for dense frames
3. **Reframe** → `dense_coverage` preset, 2560px
4. **Sky Filter** → Remove useless sky images
5. **Segment** → Create masks for dynamic objects
6. Import with masks into reconstruction software

### Workflow 5: Batch Processing Multiple Videos

1. **Extract** tab → Add Folder (select folder with all videos)
2. Configure settings (interval, quality, output folder)
3. Click **Process Queue**
4. When complete, **Reframe** each video's frames folder
5. Continue with sky filtering/segmentation as needed

---

## Troubleshooting

### "panoex core not available"

The core library failed to load. Ensure you're running from the correct directory:
```bash
cd reframing
python panoex_gui.py
```

Or install panoex as a package.

### "ultralytics not installed"

The Segment tab requires the ultralytics package:
```bash
pip install ultralytics
```

### Slow reframing performance

- Reduce **Workers** if running out of memory
- Use smaller **Size** output
- Use `lightweight` preset for fewer views

### Poor sky detection

- Run **Analyze only** first to test thresholds
- Lower **Brightness** threshold for darker skies
- Increase **Keypoints** threshold for more selective filtering

### Masks not detecting objects

- Lower **Confidence** value (try 0.3-0.4)
- Use a larger model (yolo11m or yolo11l)
- Check that the correct **Preset** or classes are selected

### Queue not saving

- Check write permissions for `~/.panoex_queue.json`
- The queue file is created in the user's home directory

### Application freezes during processing

- Processing runs in background threads; the UI may be slow to update
- Check the log panel for progress messages
- Use the **Stop** button to cancel if needed

---

## Keyboard Shortcuts

The GUI primarily uses mouse interaction. Standard system shortcuts work:
- `Ctrl+C`: Copy selected text from log
- `Ctrl+V`: Paste into text fields

---

## Log Panel

The log panel at the bottom displays:
- Startup status and library loading
- Progress messages during operations
- Error messages and warnings
- Final results and statistics

Text accumulates during the session. The view auto-scrolls to show new messages.

---

## Support

For issues and feature requests, visit the project repository or contact the development team.
