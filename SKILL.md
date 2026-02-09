---
name: video-to-frames
description: Expert guidance for extracting frames from video for photogrammetry, 3D Gaussian Splatting, and image processing workflows. Use this skill when working with ffmpeg, imagemagick, or any task involving video frame extraction, frame selection strategies, video analysis, image sequence processing, scene detection, blur detection, or preparing video footage for 3D reconstruction pipelines.
---

# Video to Frames

Extract and process frames from video for photogrammetry, 3DGS training, and image workflows.

## Quick Reference

| Task | Command |
|------|---------|
| Extract all frames | `ffmpeg -i video.mp4 frames/%05d.jpg` |
| Extract at 2 fps | `ffmpeg -i video.mp4 -vf "fps=2" frames/%05d.jpg` |
| Extract every 30th frame | `ffmpeg -i video.mp4 -vf "select=not(mod(n\,30))" -vsync vfr frames/%05d.jpg` |
| Scene detection | `ffmpeg -i video.mp4 -vf "select='gt(scene,0.3)'" -vsync vfr frames/%05d.jpg` |
| High quality JPG | `ffmpeg -i video.mp4 -qscale:v 2 frames/%05d.jpg` |
| PNG (lossless) | `ffmpeg -i video.mp4 frames/%05d.png` |
| Get video info | `ffprobe -v error -show_entries format=duration,bit_rate -show_streams video.mp4` |

## Video Analysis

Always analyze video first to plan extraction:

```bash
# Full analysis
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames,duration,codec_name \
  -of csv=p=0 video.mp4

# Get duration in seconds
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 video.mp4

# Get frame count (fast estimate)
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 video.mp4

# Get FPS
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 video.mp4
```

## Frame Extraction Strategies

### By Frame Rate (fps filter)
Best for: Consistent temporal sampling

```bash
# Extract 1 frame per second
ffmpeg -i video.mp4 -vf "fps=1" out/%05d.jpg

# Extract 0.5 fps (1 frame every 2 seconds)
ffmpeg -i video.mp4 -vf "fps=0.5" out/%05d.jpg
```

### By Frame Interval (select filter)
Best for: Precise control over which frames

```bash
# Every 30th frame
ffmpeg -i video.mp4 -vf "select=not(mod(n\,30))" -vsync vfr out/%05d.jpg

# Every 60th frame
ffmpeg -i video.mp4 -vf "select=not(mod(n\,60))" -vsync vfr out/%05d.jpg

# Frames 100-200 only
ffmpeg -i video.mp4 -vf "select=between(n\,100\,200)" -vsync vfr out/%05d.jpg
```

**Critical:** Always use `-vsync vfr` with select filter to avoid duplicate frames.

### By Time Range
Best for: Extracting specific segments

```bash
# Frames from 00:01:00 to 00:02:00
ffmpeg -ss 00:01:00 -to 00:02:00 -i video.mp4 -vf "fps=2" out/%05d.jpg

# First 30 seconds
ffmpeg -t 30 -i video.mp4 -vf "fps=2" out/%05d.jpg
```

**Note:** Place `-ss` before `-i` for fast seeking (may be less accurate), after `-i` for frame-accurate seeking.

### Scene Detection
Best for: Capturing key moments, reducing redundancy

```bash
# Extract on scene changes (threshold 0.3 = moderate sensitivity)
ffmpeg -i video.mp4 -vf "select='gt(scene,0.3)'" -vsync vfr out/%05d.jpg

# Lower threshold = more frames (0.1-0.2 for subtle changes)
ffmpeg -i video.mp4 -vf "select='gt(scene,0.15)'" -vsync vfr out/%05d.jpg

# Higher threshold = fewer frames (0.4-0.5 for major changes only)
ffmpeg -i video.mp4 -vf "select='gt(scene,0.5)'" -vsync vfr out/%05d.jpg

# Scene detection + minimum interval (at least 1 second apart)
ffmpeg -i video.mp4 -vf "select='gt(scene,0.3)*gte(t-prev_selected_t,1)'" -vsync vfr out/%05d.jpg
```

### Motion Blur Rejection
Best for: Photogrammetry quality - reject blurry frames

```bash
# Two-pass approach: extract candidates then filter
# See scripts/filter_blurry.py for Laplacian variance method

# Or use mpdecimate to skip similar/blurry frames
ffmpeg -i video.mp4 -vf "mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB" -vsync vfr out/%05d.jpg
```

## Output Quality

### JPEG Quality
```bash
# -qscale:v 2 = highest quality (2-5 recommended for photogrammetry)
ffmpeg -i video.mp4 -qscale:v 2 out/%05d.jpg

# -qscale:v 5 = good balance
ffmpeg -i video.mp4 -qscale:v 5 out/%05d.jpg
```

### PNG (Lossless)
```bash
# Best quality, larger files
ffmpeg -i video.mp4 out/%05d.png

# With compression level (0-9, higher = smaller but slower)
ffmpeg -i video.mp4 -compression_level 6 out/%05d.png
```

### Resolution Control
```bash
# Scale to width 1920, maintain aspect ratio
ffmpeg -i video.mp4 -vf "fps=2,scale=1920:-1" out/%05d.jpg

# Scale to fit within 1920x1080
ffmpeg -i video.mp4 -vf "fps=2,scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease" out/%05d.jpg
```

## 360° Video

### Equirectangular Extraction
```bash
# Standard extraction (maintains equirectangular format)
ffmpeg -i 360_video.mp4 -vf "fps=1" -qscale:v 2 out/%05d.jpg
```

### GoPro MAX / Insta360 Specifics
```bash
# GoPro MAX .360 files - extract both tracks
ffmpeg -i GS010001.360 -map 0:0 -c copy front.mp4
ffmpeg -i GS010001.360 -map 0:5 -c copy back.mp4

# Insta360 - typically already stitched, extract normally
ffmpeg -i insta360.insv -vf "fps=1" out/%05d.jpg
```

See `references/360-video.md` for detailed 360 workflows including stitching and reframing.

## Common Filter Combinations

### Photogrammetry Pipeline
```bash
# High quality, 2fps, reject near-duplicates
ffmpeg -i video.mp4 -vf "fps=2,mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB" -qscale:v 2 -vsync vfr out/%05d.jpg
```

### 3DGS Training Set
```bash
# Scene detection + fps limit + quality
ffmpeg -i video.mp4 -vf "select='gt(scene,0.2)+gte(t-prev_selected_t,0.5)',fps=4" -qscale:v 2 -vsync vfr out/%05d.jpg
```

### Timelapse Source
```bash
# Extract 1 frame every 5 seconds for timelapse input
ffmpeg -i video.mp4 -vf "fps=1/5" out/%05d.jpg
```

## ImageMagick Post-Processing

### Batch Operations
```bash
# Convert all to PNG
mogrify -format png *.jpg

# Resize all to 50%
mogrify -resize 50% *.jpg

# Add sequential numbering
ls *.jpg | cat -n | while read n f; do mv "$f" "frame_$(printf '%05d' $n).jpg"; done
```

### Quality Assessment
```bash
# Get image dimensions
identify -format "%w %h\n" *.jpg | sort | uniq -c

# Check for corrupt images
identify *.jpg 2>&1 | grep -v "\.jpg JPEG"
```

### Montage/Contact Sheet
```bash
# Create contact sheet of all frames
montage *.jpg -geometry 200x200+2+2 -tile 10x contact_sheet.jpg
```

## Calculating Extraction Parameters

For photogrammetry/3DGS, aim for 60-80% overlap between consecutive frames.

### Rule of Thumb
- Walking speed (~1.5 m/s): 2-3 fps
- Driving slow (~10 m/s): 5-10 fps  
- Drone (~5 m/s): 3-5 fps
- Static tripod rotation: 1 fps or scene detection

### Frame Count Estimation
```
total_frames = duration_seconds × extraction_fps
```

For a 5-minute video at 2 fps: `300 × 2 = 600 frames`

## Troubleshooting

### "Output file is empty"
- Check input path exists
- Ensure output directory exists: `mkdir -p out/`
- Verify video has video stream: `ffprobe -v error -show_streams video.mp4`

### Duplicate frame numbers
- Add `-vsync vfr` when using `select` filter
- Ensures output numbering matches actual extracted frames

### Color shift in output
- Add `-pix_fmt yuvj420p` for standard JPEG color
- Or use `-color_range 2` for full range

### Memory issues with long videos
- Process in segments using `-ss` and `-t`
- Or use frame interval extraction instead of fps filter

## Scripts

- `scripts/analyze_video.py` - Comprehensive video analysis
- `scripts/filter_blurry.py` - Remove blurry frames using Laplacian variance
- `scripts/extract_frames.py` - Wrapper with common presets
- `scripts/rename_sequence.py` - Renumber image sequences

## References

- `references/360-video.md` - 360° video extraction and stitching workflows
- `references/quality-metrics.md` - Frame quality assessment methods
