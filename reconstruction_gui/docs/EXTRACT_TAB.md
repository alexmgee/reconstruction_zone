# Extract Tab

The Extract tab is the starting point for every reconstruction project. It handles video analysis, frame extraction with quality filtering, and 360° fisheye-to-perspective reframing. Everything downstream — masking, review, coverage — depends on the frames this tab produces.

## Sections

The tab has three collapsible sections, top to bottom:

| Section | Purpose |
|---------|---------|
| **Video Selection** | Load a video, set output folder, optionally analyze metadata |
| **Frame Extraction** | Pull frames from the video with mode/interval/filter controls |
| **360 Video** | Split dual-fisheye containers, reframe to pinhole perspectives |

You'll use Video Selection + Frame Extraction for normal perspective video. Add 360 Video when working with DJI Osmo, Insta360, or GoPro Max footage.

---

## Video Selection

### Input and Output

Two paths to set before anything else:

- **Input** — The source video file (`.mp4`, `.mov`, `.avi`, `.mkv`, `.360`, `.insv`)
- **Output** — Directory where extracted frames go. Each video creates a subfolder named after the video file (e.g. `output/my_video/frame_0001.jpg`)

### Analysis

Expand the **Analysis** subsection and click **Analyze** to probe the video with ffprobe. Example output:

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

Analysis also stores `current_video_info` and `current_video_path` on the app — the Coverage tab reads these for bridge extraction.

---

## Frame Extraction

### Extraction modes

Four modes control how frames are selected from the video:

- **Fixed Interval** — Extracts one frame every N seconds, evenly spaced through the video. The simplest and most predictable mode. Use this unless you have a specific reason to try the others.

- **Scene Detection** — Starts with the same interval baseline as Fixed, but also extracts additional frames at scene cuts (large visual changes between consecutive frames). Useful for walkthroughs where the camera moves through distinct rooms or areas — the scene-cut frames capture transitions that a fixed interval might skip.

- **Adaptive Density** — Dynamically adjusts extraction density based on motion. High-motion segments get more frames; static segments get fewer. Good for varied-pace captures where the camera alternates between walking and pausing.

- **Sharpest Frame** — Divides the video into N-second windows and picks the single sharpest frame from each window, scoring every frame in the window using ffmpeg's blurdetect filter. More expensive than other modes (analyzes all frames, not just sampled ones) but produces the highest-quality dataset from shaky or handheld footage.

### Settings

| Setting | Range | Default | Effect |
|---------|-------|---------|--------|
| **Interval** | 0.1–10.0s | 2.0s | Time between extracted frames. Lower = more frames |
| **Quality** | 70–100 | 95 | JPEG compression quality (ignored for PNG) |
| **Format** | jpg / png | jpg | JPEG is smaller; PNG is lossless |
| **Start** | time | 0:00 | Skip video before this point |
| **End** | time | end | Stop extracting after this point |

Time values accept seconds (`45.5`), `MM:SS` (`1:30`), or `HH:MM:SS` (`1:02:30`).

### Live estimate

Once a video is analyzed, the estimate bar updates in real-time as you adjust settings. Example:

```
~161 frames  ·  ~2.1 GB (jpg q95)  ·  5:23
```

It accounts for mode, interval, time range, format, quality, and active filters.

### Color & LUT

Collapsed by default. Expand to apply color correction during extraction.

| Setting | Default | Effect |
|---------|---------|--------|
| **LUT file** | (none) | `.cube` file for color space conversion (e.g. D-Log → Rec.709) |
| **Strength** | 100% | Blend between original (0%) and full LUT (100%) |
| **Shadows** | 50 | Lift or crush shadow detail. 50 = neutral |
| **Highlights** | 50 | Lift or crush highlight detail. 50 = neutral |

### Sky Filter

Collapsed by default. Removes images that are mostly sky — these add noise to reconstruction without contributing useful geometry.

| Setting | Default | Effect |
|---------|---------|--------|
| **Brightness** | 0.85 | Minimum brightness ratio to consider "sky-like". Lower = more aggressive |
| **Keypoints** | 50 | Minimum ORB keypoints expected. Frames below this with high brightness are flagged |

The filter uses 4 metrics: brightness, saturation, keypoint count, and edge density. A frame must fail multiple metrics to be classified as sky.

### Blur Filter

Collapsed by default. Removes the blurriest frames from the extraction, keeping only the sharpest percentile.

| Setting | Default | Effect |
|---------|---------|--------|
| **Keep** | 80% | Percentage of sharpest frames to retain. 80% removes the worst 20% |

Automatically skipped when using Sharpest Frame mode (which already selects for sharpness).

### Pipeline summary

When you click **Extract**, the selected mode pulls frames from the video, then each enabled post-processing step runs sequentially on the output:

```
 Extract frames (mode determines selection strategy)
       │
       │  Output: {output}/{video_name}/frame_NNNN.{jpg|png}
       │
       ▼
 Post-processing (each step is optional)
       │
       ├─ 1. LUT          → color-correct in place
       ├─ 2. Shadows/HL   → adjust tonality in place
       ├─ 3. Sky Filter   → delete sky-dominated frames
       └─ 4. Blur Filter  → delete blurriest frames
                             (skipped in Sharpest mode)
```

Each post-processing step reads from disk, processes, and either overwrites (LUT, shadow/highlight) or deletes (sky, blur) in place. The cancel flag is checked between frames.

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

## 360 Video

For dual-fisheye cameras (DJI Osmo Action 360, Insta360 X3/X4, GoPro Max). This section has its own input/output paths separate from the standard extraction above.

### Input and Output

- **Input** — `.osv`, `.360`, or `.insv` container file (dual-fisheye video)
- **Output** — Directory for extracted and reframed images

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
     └────────────┘           └───────┬────────┘
                                      │
                              ┌───────▼────────┐
                              │ Output:        │
                              │ perspectives/  │
                              │   front_h0_y0  │
                              │   front_h30_y0 │
                              │   back_h0_y0   │
                              │   ...          │
                              └────────────────┘
```

**Split Lenses** and **Reframe** are independent operations. You can do either or both.

### Custom Calibration

Collapsed by default. The built-in equidistant fisheye model works for most 360 cameras. Only load a custom calibration JSON if you see warped straight lines in your output crops.

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

### Split Lenses

Lossless FFmpeg stream copy — extracts the front and back lens video streams from the 360 container without re-encoding. Use this when you want to work with raw 180° hemispheres (e.g., training on fisheye data directly).

After splitting, the front lens is auto-loaded into Video Selection for immediate analysis.

### Reframe to Pinhole Perspectives

This is the main 360 workflow. Converts dual-fisheye frames into standard perspective crops suitable for COLMAP/photogrammetry.

#### Process

```
┌─ STEP 1: EXTRACT FISHEYE FRAMES ────────────────────────────┐
│                                                               │
│  OSVHandler extracts frame pairs at the configured interval  │
│  from both streams (front + back lens)                       │
│                                                               │
│  Output: fisheye_frames/front_NNNN.jpg + back_NNNN.jpg      │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌─ STEP 1.5: MOTION SELECTION (optional) ──────────────────────┐
│                                                               │
│  If enabled, MotionSelector filters frame pairs:             │
│  • Reject frames below sharpness threshold (Laplacian)       │
│  • Keep frames with sufficient optical flow between them     │
│  • Reduces redundant/blurry frames from slow movement        │
│                                                               │
│  Same index set applied to both front and back               │
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
│                                                               │
│  Output: perspectives/front_h{yaw}_y{pitch}.jpg             │
└──────────────────────────────────────────────────────────────┘
```

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
| **Crop** | 1280 / 1600 / 1920 | 1600 | Output resolution per crop (square). 1600 balances detail and file size |
| **Quality** | 70–100 | 95 | JPEG quality for output crops |
| **Interval** | 0.5–10.0s | 2.0s | Extract one frame pair every N seconds |

#### Motion-Aware Selection

Collapsed by default. When enabled, filters extracted frame pairs before reframing.

| Setting | Range | Default | Effect |
|---------|-------|---------|--------|
| **Sharpness** | 10–200 | 50 | Minimum Laplacian variance. Frames below are discarded as blurry |
| **Target Flow** | 2–30 | 10 | Optical flow magnitude between kept frames. Higher = more motion required |

This is useful for footage where the camera moves at varying speeds — removes redundant frames from slow/stopped segments and blurry frames from fast motion.

#### Output estimate

A live estimate updates as you change settings. Example:

```
42 frame pairs × 52 views = ~2,184 output images (1600x1600)
```

This requires a video to be probed first (selecting the input file triggers an automatic probe).

### Output structure

```
{output}/
├── fisheye_frames/         ← raw extracted fisheye pairs
│   ├── front_0001.jpg
│   ├── back_0001.jpg
│   ├── front_0002.jpg
│   ├── back_0002.jpg
│   └── ...
└── perspectives/           ← reframed pinhole crops
    ├── front_0001_h0_y0.jpg
    ├── front_0001_h30_y0.jpg
    ├── front_0001_h60_y0.jpg
    ├── back_0001_h0_y0.jpg
    └── ...
```

---

## Typical workflow: perspective video

```
1. Video Selection  → Browse to video file, set output folder
2. (Optional)       → Expand Analysis, click Analyze to check metadata
3. Frame Extraction → Choose mode (Fixed Interval for most cases)
4.                  → Adjust interval (2.0s default, lower for dense capture)
5. (Optional)       → Enable Blur Filter at 80% to remove worst frames
6. (Optional)       → Enable Sky Filter if outdoor scene has upward-facing frames
7. (If log video)   → Expand Color & LUT, enable LUT, select .cube file
8.                  → Click Extract
9.                  → Frames appear in {output}/{video_name}/
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

## Typical workflow: 360 fisheye video

```
1. 360 Video        → Browse to .osv / .360 / .insv file
2.                  → Set output folder
3. (Optional)       → Split Lenses (if you need raw hemispheres)
4. Reframe section  → Select preset matching your camera
5.                  → Adjust crop size, quality, interval
6. (Optional)       → Enable Motion-Aware Selection for varied-pace footage
7.                  → Click Extract Pinhole Views
8.                  → Perspective crops appear in {output}/perspectives/
9. (Next step)      → Run COLMAP on the perspectives/ folder
```

## Typical workflow: 360 with coverage feedback

```
1. 360 Video        → Extract pinhole views (steps 1-8 above)
2. (External)       → Run COLMAP on perspectives/
3. Coverage tab     → Analyze gaps in the reconstruction
4. Coverage tab     → Extract bridges (check "Reframe fisheye → perspective")
                      Reads calibration + preset from this tab's 360 section
5. (External)       → Merge bridge perspectives into dataset, re-run COLMAP
```
