# Extract Tab

The Extract tab is the starting point for every reconstruction project. It handles video analysis, frame extraction with quality filtering, SRT geotagging, and 360° fisheye-to-perspective reframing. Everything downstream — masking, review, coverage — depends on the frames this tab produces.

## Sections

The tab has four top-level sections — two always-visible and two collapsible:

| Section | Type | Purpose |
|---------|------|---------|
| **Video Selection** | Always visible | Load a video (or split-lens pair), set output folder |
| **Frame Extraction** | Always visible | Analysis, extraction settings, post-processing filters, batch queue |
| **360 Processing** | Collapsible | Split dual-fisheye containers or reframe to pinhole perspectives |
| **Advanced** | Collapsible (default: collapsed) | Frame quality filter for existing frames, SRT geotagging |

You'll use Video Selection + Frame Extraction for normal perspective video. Add 360 Processing when working with DJI Osmo, Insta360, or GoPro Max footage. The Advanced section contains power-user tools — a standalone frame quality filter for re-scoring existing frames, and manual SRT geotagging.

---

## Video Selection

### Input and Output

Two paths to set before anything else:

- **Input** — The source video file (`.mp4`, `.mov`, `.avi`, `.mkv`, `.360`, `.insv`, `.osv`)
- **Output** — Directory where extracted frames go. Each video creates a subfolder named after the video file (e.g. `output/my_video/frame_0001.jpg`)

### Split Lens Videos

Collapsed by default. For dual-fisheye cameras (DJI Osmo 360, Insta360 X3/X4) where you've already split the 360 container into separate front/back lens videos and graded them externally (e.g. in DaVinci Resolve).

When **Use Split Lens Videos** is checked, the Analyze and Extract buttons operate on the front/back pair instead of the single video input above.

| Field | Purpose |
|-------|---------|
| **Front** | Graded front lens video (`.mp4` / `.mov`) |
| **Back** | Graded back lens video (`.mp4` / `.mov`) |
| **Clip Folder** | Per-clip working folder (auto-filled from the filename stem). Extract writes `front/frames` + `back/frames` here. |

The workflow is: Split Lenses (in the 360 Processing section below) → grade externally → enable Split Lens Videos here → Extract.

---

## Frame Extraction

### Analysis

A collapsible subsection inside Frame Extraction (expanded by default). Click **Analyze** to probe the video with ffprobe.

**Single video** example output:

```
File:         my_video.mp4
Format:       mp4 (h264)
Resolution:   3840x2160
FPS:          29.97
Duration:     5:23 (323.4s)
Frames:       9,693
Bitrate:      45,000 kbps
Pixel Format: yuv420p

=== 360 Detection ===
Equirectangular: No
Aspect Ratio: 1.78:1

=== Color Profile ===
Log Format: D-Log M
Is Log: Yes

=== Recommendations ===
Extraction Interval: 2.0s
Estimated Frames @ 2.0s: 161
Recommended LUT: DJI_D-Log_M_to_Rec709.cube
```

**Split-lens pair** example output:

```
Source: Split lens pair
Front: front_graded.mp4
Back:  back_graded.mp4

=== Shared Pairing ===
Resolution: 1920x1920 per lens
FPS: 29.97
Duration: 5:23 (323.4s shared)
Frames: 9,693 shared

=== Recommendations ===
Extraction Interval: 2.0s
Estimated Frame Pairs @ 2.0s: 161
```

When analyzing a split pair, mismatches (resolution, FPS, duration) between front and back are flagged as warnings.

Analysis stores video metadata internally — the Coverage tab reads it for bridge extraction, and the live estimate uses it to calculate frame counts.

### Sharpness and Scene Detection

Two composable controls determine how frames are selected:

**Sharpness** (radio buttons: Tenengrad / Laplacian / None)

Controls whether the extractor scores every frame for sharpness and picks the best per interval window, or simply grabs frames at fixed intervals.

- **Tenengrad** — Scores every frame via Tenengrad (Sobel gradient energy). Better noise robustness than Laplacian — use for action cameras, high-ISO footage, or heavily compressed source video.
- **Laplacian** (default) — Scores every frame via Laplacian variance (second-derivative energy). Fast, reliable, and the ecosystem standard for 3DGS/NeRF preprocessing.
- **None** — Fixed interval extraction. One frame every N seconds, evenly spaced. Simplest and fastest.

Both Laplacian and Tenengrad analyze frames at 1920px width (configurable) and run at near-realtime speed via OpenCV.

**Scene Detection** (checkbox, default: on)

When enabled alongside a sharpness method, splits time windows at scene boundaries (detected via HSV histogram correlation) so both sides of a visual transition get a sharp representative frame. Greyed out when Sharpness is set to None, since there's no frame-by-frame scoring loop to piggyback scene detection on.

When using **Split Lens Videos**, both controls apply to paired extraction — the scoring and scene detection run on both front and back lenses simultaneously, selecting the sharpest shared frame index per window.

### Settings

| Setting | Range | Default | Effect |
|---------|-------|---------|--------|
| **Interval** | 0.1–10.0s | 2.0s | Time between extracted frames. Lower = more frames |
| **Quality** | 70–100 | 95 | JPEG compression quality (ignored for PNG) |
| **Format** | jpg / png | jpg | JPEG is smaller; PNG is lossless |
| **Start** | time | 0:00 | Skip video before this point |
| **End** | time | end | Stop extracting after this point |
| **Sharpness** | None / Laplacian / Tenengrad | Laplacian | Frame scoring method |
| **Scene Detection** | on / off | on | Split windows at scene cuts |

Time values accept seconds (`45.5`), `MM:SS` (`1:30`), or `HH:MM:SS` (`1:02:30`).

### Live estimate

Once a video is analyzed, the estimate bar updates in real-time as you adjust settings. Example:

```
~161 frames  ·  ~2.1 GB (jpg q95)  ·  5:23
```

It accounts for mode, interval, time range, format, quality, and active filters.

### Post-Processing

A collapsible section inside Frame Extraction containing four optional filters that run after frames are extracted. Each is independently collapsible with its own enable checkbox.

#### Color & LUT

> **Edition note:** This section is only visible in the `personal` edition. It is hidden in `community` and `release` builds.

Collapsed by default. Applies color correction to extracted frames in place.

| Setting | Default | Effect |
|---------|---------|--------|
| **Apply LUT after extraction** | Off | Enable/disable the entire color pipeline |
| **LUT file** | (none) | `.cube` file for color space conversion (e.g. D-Log → Rec.709) |
| **Strength** | 100% | Blend between original (0%) and full LUT (100%) |
| **Shadows** | 50 | Lift or crush shadow detail. 50 = neutral |
| **Highlights** | 50 | Lift or crush highlight detail. 50 = neutral |

#### Sky Filter

Collapsed by default. Removes images that are mostly sky — these add noise to reconstruction without contributing useful geometry.

| Setting | Default | Effect |
|---------|---------|--------|
| **Remove sky-dominated images** | Off | Enable/disable the filter |
| **Brightness** | 0.85 | Minimum brightness ratio to consider "sky-like". Lower = more aggressive |
| **Keypoints** | 50 | Minimum ORB keypoints expected. Frames below this with high brightness are flagged |

The filter uses 4 metrics: brightness, saturation, keypoint count, and edge density. A frame must fail multiple metrics to be classified as sky.

#### Blur Filter

Collapsed by default. Removes the blurriest frames from the extraction, keeping only the sharpest percentile.

| Setting | Default | Effect |
|---------|---------|--------|
| **Filter blurry frames after extraction** | Off | Enable/disable the filter |
| **Keep** | 80% | Percentage of sharpest frames to retain. 80% removes the worst 20% |

Automatically skipped when using Sharpest Frame mode (which already selects for sharpness).

#### Motion Selection

Collapsed by default. Filters extracted frames by camera movement — removes redundant near-identical frames from slow/stopped segments and blurry frames from fast motion.

| Setting | Range | Default | Effect |
|---------|-------|---------|--------|
| **Filter by sharpness + optical flow** | On/Off | Off | Enable/disable the filter |
| **Sharpness** | 10–200 | 50 | Minimum Laplacian variance. Frames below this are discarded as blurry |
| **Target Flow** | 2–30 | 10 | Optical flow magnitude between kept frames. Higher = more camera movement required between selections |

This is useful for footage where the camera moves at varying speeds. The filter examines every extracted frame pair: if two consecutive frames are too similar (low optical flow) or too blurry (low Laplacian score), the weaker frame is discarded. The same index set is applied to both front and back lenses when using split-lens extraction.

### Pipeline summary

When you click **Extract**, the selected mode pulls frames from the video, then each enabled post-processing step runs sequentially on the output:

```
 Extract frames (mode determines selection strategy)
       │
       │  Output: {output}/{video_name}/frame_NNNN.{jpg|png}
       │
       ▼
 Post-processing (each step is optional, runs in order)
       │
       ├─ 1. LUT              → color-correct in place (personal edition only)
       ├─ 2. Shadows/HL       → adjust tonality in place (personal edition only)
       ├─ 3. Sky Filter        → delete sky-dominated frames
       ├─ 4. Blur Filter       → delete blurriest frames (skipped in Sharpest mode)
       └─ 5. Motion Selection  → delete redundant/blurry frames by optical flow
       │
       ▼
 Auto-geotag (if enabled in Advanced → Metadata)
       │
       └─ Write GPS, altitude, focal length, datetime → EXIF via exiftool
```

Each post-processing step reads from disk, processes, and either overwrites (LUT, shadow/highlight) or deletes (sky, blur, motion) in place. Geotagging runs last so it only tags surviving frames. The cancel flag is checked between frames.

---

## Batch Queue

The queue lets you process multiple videos with the same settings. Collapsed by default inside Frame Extraction.

### Adding videos

Three ways to add:
- **Add to Queue** button (main action row) — adds the currently analyzed video
- **Add Videos** button (queue section) — file picker for multiple videos
- **Add Folder** button (queue section) — adds all video files from a directory

### Queue item states

| State | Color | Meaning |
|-------|-------|---------|
| Pending | Gray | Waiting to be processed |
| Processing | Orange | Currently extracting (shows progress %) |
| Done | Green | Completed (shows frame count) |
| Error | Red | Failed (check console log) |
| Cancelled | Gray | Stopped by user |

### Processing

Click **Process Queue** to run all pending videos sequentially. Each video gets the same extraction settings (mode, interval, quality, filters) and outputs to `{output}/{video_name}/`.

The queue persists to `~/.prep360_queue.json` between sessions. Use **Clear Done** to clean up completed items.

```
Queue workflow:

┌───────────┐     ┌──────────────┐     ┌───────────────┐
│ Add videos │────▶│ Configure    │────▶│ Process Queue │
│ to queue   │     │ settings     │     │ (sequential)  │
└───────────┘     └──────────────┘     └───────┬───────┘
                                                │
                              ┌─────────────────┼─────────────────┐
                              ▼                 ▼                 ▼
                        ┌──────────┐     ┌──────────┐     ┌──────────┐
                        │ video_1/ │     │ video_2/ │     │ video_3/ │
                        │ frames   │     │ frames   │     │ frames   │
                        └──────────┘     └──────────┘     └──────────┘
```

---

## 360 Processing

For dual-fisheye cameras (DJI Osmo Action 360, Insta360 X3/X4, GoPro Max). Contains two collapsible subsections: Split Lenses and Reframing.

There are two distinct 360 workflows, and they can be combined:

1. **Raw dual-fisheye → Metashape** — Split Lenses demuxes the 360 container, you grade externally, then use Split Lens Videos (in Video Selection) to extract frame pairs for direct Metashape import.
2. **360 → pinhole perspectives** — Reframing converts fisheye or equirectangular frames into standard perspective crops for COLMAP/3DGS pipelines.

### Pipeline overview

```
┌─ 360 CONTAINER ──────────────────────────────────────────────┐
│                                                               │
│  .osv / .360 / .insv file                                    │
│  Contains two video streams: front lens + back lens          │
│                                                               │
└──────────┬──────────────────────────┬────────────────────────┘
           │                          │
     ┌─────▼─────┐           ┌───────▼────────┐
     │ Split      │           │ Reframe to     │
     │ Lenses     │           │ Pinhole Views  │
     │            │           │                │
     │ Lossless   │           │ Fisheye →      │
     │ demux into │           │ perspective    │
     │ front.mp4  │           │ crops          │
     │ back.mp4   │           │                │
     └─────┬──────┘           └───────┬────────┘
           │                          │
           ▼                          ▼
     Grade externally           Output: pinhole
     (DaVinci Resolve)          perspectives for
     then use Split Lens        COLMAP / 3DGS
     Videos in Video
     Selection to extract
```

### Split Lenses

Collapsed by default. Performs a lossless demux from a 360 container (`.osv`, `.360`, `.insv`) into separate front and back lens video files. Uses the **Video Selection** Input and Output paths.

After splitting, the GUI stops at the two lens videos. This is the handoff point for any external grading workflow (DaVinci Resolve, etc.). Once graded, return to **Video Selection → Split Lens Videos** to use the graded pair as the extraction source.

### Reframing

Collapsed by default. Converts dual-fisheye frames or equirectangular (ERP) frames into standard pinhole perspective crops suitable for COLMAP and photogrammetry pipelines.

#### Inputs

The Reframing section has four path fields, giving flexibility for different source types:

| Field | Purpose | When to use |
|-------|---------|-------------|
| **Video** | `.osv` / `.360` / `.insv` file | Direct extraction from a 360 video — extracts frame pairs at the configured interval, then reframes |
| **Frames** | Existing ERP frames folder | Reframe pre-extracted equirectangular frames (skips video extraction) |
| **Masks** | Masks for ERP or fisheye frames | Optional — masks are reframed alongside images, producing per-perspective masks |
| **Output** | Perspective output directory | Where pinhole crops are saved |

You can provide a Video (extract + reframe in one step), or Frames (reframe existing images), or both. If both are set, the video is extracted first, then all frames are reframed.

#### Custom Calibration

Collapsed inside Reframing. The built-in equidistant fisheye model works for most 360 cameras (DJI, Insta360, GoPro). Only load a custom calibration JSON if you see warped straight lines in your output crops.

JSON format:
```json
{
  "front": {"K": [[...]], "D": [...], "image_size": [w, h], "rms_error": 0.123},
  "back":  {"K": [[...]], "D": [...], "image_size": [w, h], "rms_error": 0.456}
}
```

- **K** — 3×3 camera intrinsic matrix
- **D** — 4 distortion coefficients
- Click **Reset** to revert to the built-in model

#### Preset

The preset dropdown controls how many perspective views are extracted from each fisheye hemisphere. Each preset defines a set of **Rings** — groups of views at specific pitch angles.

Example preset description:
```
13 views per lens: 1 looking up (45°), 3 looking up (30°), 5 at horizon, 3 looking down (-30°), 1 looking down (-45°)
Each crop is 90° FOV — standard perspective
```

More views = better spatial coverage for reconstruction, but more output images and longer processing.

Built-in presets cover common camera configurations: `prep360_default`, `dji_osmo_360`, `insta360_x3`, `gopro_max`, `slow_motion`.

#### Settings

| Setting | Range | Default | Effect |
|---------|-------|---------|--------|
| **Crop** | 1280 / 1600 / 1920 | 1600 | Output resolution per pinhole crop (square). 1600 balances detail and file size |
| **Quality** | 70–100 | 95 | JPEG quality for output crops |
| **Interval** | 0.5–10.0s | 2.0s | Extract one frame pair every N seconds from the video |

#### Station dirs (for Metashape)

Enabled by default. Organizes output into per-source subdirectories — each subdirectory represents a Metashape **station** (shared camera position).

When enabled:
- Images: `output/images/{station}/{label}.jpg`
- Masks: `output/masks/{station}/{label}_mask.png` (separate parallel tree)
- Metadata: `output/reframe_metadata.json` with pinhole intrinsics, view configs, station list

When disabled: all perspectives are written flat into the output directory.

**Metashape import:** Drag all station subdirectories into an empty chunk, then set the group type to Station. Use the [import_masks_reframed.py](https://github.com/alexmgee/reconstruction-zone) script to load masks (Metashape's built-in mask import can't traverse subdirectories).

#### Output estimate

A live estimate updates as you change settings. Example:

```
42 frame pairs × 52 views = ~2,184 output images (1600x1600)
```

This requires a video to be probed first (selecting the input file triggers an automatic probe).

#### Process

When you click **Extract & Reframe**:

```
┌─ STEP 1: EXTRACT FISHEYE FRAMES ────────────────────────────┐
│                                                               │
│  OSVHandler extracts frame pairs at the configured interval  │
│  from both streams (front + back lens)                       │
│  (Skipped if only Frames path is set — uses existing images) │
│                                                               │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌─ STEP 2: REFRAME TO PERSPECTIVES ────────────────────────────┐
│                                                               │
│  For each frame pair:                                        │
│  • Apply fisheye calibration (undistort)                     │
│  • Project through each view in the preset layout            │
│  • Render pinhole perspective crop at configured resolution  │
│  • Skip crops with <50% content coverage                     │
│  • If masks provided, reframe masks in parallel              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Output structure

**With station dirs enabled** (default):
```
{output}/
├── images/
│   ├── frame_0001/              ← one station per source frame
│   │   ├── front_h0_y0.jpg
│   │   ├── front_h30_y0.jpg
│   │   └── ...
│   └── frame_0002/
│       └── ...
├── masks/                        ← parallel tree (if masks provided)
│   ├── frame_0001/
│   │   ├── front_h0_y0_mask.png
│   │   └── ...
│   └── ...
└── reframe_metadata.json         ← pinhole intrinsics + station mapping
```

**With station dirs disabled**:
```
{output}/
├── front_0001_h0_y0.jpg
├── front_0001_h30_y0.jpg
├── back_0001_h0_y0.jpg
└── ...
```

---

## Advanced

Collapsed by default. Contains two subsections for power-user tools: a standalone frame quality filter and SRT geotagging.

### Metadata

Collapsed inside Advanced. Handles SRT geotagging — writing GPS coordinates, altitude, focal length, and capture datetime into frame EXIF metadata using DJI SRT telemetry files.

#### Auto-geotag after extraction

Enabled by default (the checkbox is in Frame Extraction's settings row, not here). After extraction and all post-processing filters complete, the pipeline automatically geotags every surviving frame. This uses DJI SRT telemetry files — the `.SRT` file recorded alongside the video by DJI drones and action cameras.

**How it works:** prep360 writes an `extraction_manifest.json` alongside the extracted frames during extraction. This manifest records the exact video timestamp for every frame. The geotagger reads the manifest, looks up each timestamp in the SRT data, and writes the corresponding GPS/altitude/focal-length/datetime into the frame's EXIF via [exiftool](https://exiftool.org).

The SRT file is auto-detected by matching the video filename (e.g. `my_video.mp4` → `my_video.SRT` in the same directory). If auto-detection fails, use the SRT override below.

**Requirements:** [exiftool](https://exiftool.org) must be installed and on PATH. If exiftool is not found, geotagging is silently skipped — extraction still works normally.

**When to disable:** If your video has no SRT file, or if you don't need georeferenced frames. Disabling prevents the "no matching SRT file found" log message.

#### SRT file override

The auto-geotag checkbox in Frame Extraction normally auto-detects the SRT file by matching the video filename. If auto-detection fails (e.g. the SRT has a different name, or is in a different directory), you can manually specify the path here.

| Field | Purpose |
|-------|---------|
| **SRT** | Path to a `.SRT` telemetry file. Leave blank for auto-detection. Used by both auto-geotag and standalone geotagging. |

#### Apply to folder (standalone geotagging)

For when you already have extracted frames but didn't geotag them during extraction. Point it at a frames folder and an SRT file, click **Geotag**.

| Field | Purpose |
|-------|---------|
| **Frames** | Directory containing previously extracted frames |
| **Geotag** button | Runs geotagging on the selected folder |

**Reliability note:** This requires an `extraction_manifest.json` file in the frames folder. This manifest is automatically created by prep360 during extraction — it records the exact video timestamp for every frame, ensuring accurate SRT-to-frame matching. If the manifest is missing (e.g. frames were extracted by a different tool), standalone geotagging will refuse to proceed rather than guess.

**Requirements:** [exiftool](https://exiftool.org) must be installed and on PATH.

### Frame Quality Filter

Collapsed inside Advanced. A standalone tool for scoring and filtering existing extracted frames — useful when you want to thin an already-extracted dataset without re-extracting from video.

This is a two-step workflow: **Analyze** scores all frames non-destructively, then **Filter** applies the enabled filters and removes rejected frames. You can adjust thresholds between the two steps and see the effect in real-time via the analysis summary.

#### Folder

| Field | Purpose |
|-------|---------|
| **Folder** | Directory of previously extracted frames to score and filter |

#### Sharpness

Collapsible subsection (expanded by default, enabled by default). Rejects blurry frames by Laplacian variance scoring.

**Mode toggle** — two scoring strategies:

- **Percentile** (default) — Keep the sharpest N% of frames. The threshold adapts to your dataset's score distribution.
- **Absolute** — Reject any frame below a fixed Laplacian variance score. Use when you know your acceptable quality floor.

| Setting | Mode | Range | Default | Effect |
|---------|------|-------|---------|--------|
| **Keep** | Percentile | 50–99% | 80% | Keep the sharpest N%. 80% removes the worst 20% |
| **Min** | Absolute | 10–500 | 50 | Reject frames with Laplacian variance below this score |

When both Sharpness and Motion filters are enabled, the Motion filter's sharpness floor auto-syncs from the Sharpness filter settings (shown as "synced" in the UI). Disable the Sharpness filter to control the Motion filter's sharpness threshold independently.

#### Motion

Collapsible subsection (collapsed by default, disabled by default). Thins redundant frames by optical flow — removes near-identical frames from slow/stopped segments.

| Setting | Range | Default | Effect |
|---------|-------|---------|--------|
| **Sharpness** | 10–200 | 50 (synced) | Minimum Laplacian variance. Auto-synced from Sharpness filter when both are enabled |
| **Target flow** | 2–30 | 10 | Optical flow magnitude between kept frames. Higher = more camera movement required between selections |

#### Safety

| Option | Behavior |
|--------|----------|
| **Move to _rejected/** (default) | Rejected frames are moved to a `_rejected/` subdirectory inside the frames folder. Recoverable. |
| **Delete permanently** | Rejected frames are deleted from disk. Not recoverable. |

#### Workflow

1. Click **Analyze** — scores every frame for sharpness (non-destructive). The summary panel shows score range, distribution, and how many frames would be cut at current settings.
2. Adjust thresholds — the summary updates live as you move sliders.
3. Click **Filter** — applies the enabled filters. Rejected frames are moved or deleted based on the safety toggle.

---

## Typical workflow: perspective video

```
1. Video Selection  → Browse to video file, set output folder
2. (Optional)       → Expand Analysis, click Analyze to check metadata
3. Frame Extraction → Choose mode (Fixed Interval for most cases)
4.                  → Adjust interval (2.0s default, lower for dense capture)
5. (Optional)       → Enable Blur Filter at 80% to remove worst frames
6. (Optional)       → Enable Sky Filter if outdoor scene has upward-facing frames
7.                  → Click Extract
8.                  → Frames appear in {output}/{video_name}/
```

## Typical workflow: drone video with GPS

```
1. Video Selection  → Browse to drone video, set output folder
2.                  → Analyze to check metadata (confirms log format, interval)
3. Frame Extraction → Choose mode, adjust settings
4.                  → Auto-geotag is on by default (Advanced → Metadata)
5.                  → Click Extract
6.                  → Frames extracted + GPS/altitude/focal length written to EXIF
7. (If SRT not found) → Advanced → Metadata: manually set the SRT file path
```

## Typical workflow: batch processing

```
1. Video Selection  → Set output folder (shared across all videos)
2. Frame Extraction → Configure settings (applied to all queued videos)
3.                  → Expand Batch Queue
4.                  → Add Videos or Add Folder
5.                  → Click Process Queue
6.                  → Monitor progress in queue list + console log
7.                  → Each video gets its own subfolder in output
```

## Typical workflow: 360 → pinhole perspectives

```
1. 360 Processing        → Expand Reframing subsection
2.                  → Browse to .osv / .360 / .insv file (Video field)
3.                  → Set Output directory
4.                  → Select preset matching your camera
5.                  → Adjust crop size, quality, interval
6.                  → Click Extract & Reframe
7.                  → Perspective crops appear in output
8. (Next step)      → Run COLMAP on the perspectives folder
```

## Typical workflow: 360 raw → Metashape

```
1. 360 Processing        → Expand Split Lenses, click Split Lenses
2.                  → front.mp4 + back.mp4 appear in output
3. (External)       → Grade in DaVinci Resolve or similar
4. Video Selection  → Enable Split Lens Videos, load graded front/back
5.                  → Analyze to verify pair consistency
6. Frame Extraction → Configure mode + interval
7.                  → Click Extract
8.                  → Clip folder with front/frames + back/frames ready for Metashape
```

## Typical workflow: filter existing frames

```
1. Advanced           → Expand Frame Quality Filter
2.                    → Browse to folder of previously extracted frames
3. Sharpness          → Choose Percentile (keep 80%) or Absolute (min 50)
4. (Optional)         → Enable Motion filter, adjust target flow
5.                    → Select safety: Move to _rejected/ (recommended) or Delete
6.                    → Click Analyze to score all frames (non-destructive)
7.                    → Review the summary: score range, how many would be cut
8.                    → Adjust thresholds if needed (summary updates live)
9.                    → Click Filter to apply
```

## Typical workflow: 360 with coverage feedback

```
1. 360 Processing        → Extract & Reframe (steps 1-7 from "360 → pinhole" above)
2. (External)       → Run COLMAP on perspectives
3. Coverage tab     → Analyze gaps in the reconstruction
4. Coverage tab     → Extract bridges (check "Reframe fisheye → perspective")
                      Reads calibration + preset from this tab's 360 section
5. (External)       → Merge bridge perspectives into dataset, re-run COLMAP
```
