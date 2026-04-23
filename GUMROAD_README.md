# Reconstruction Zone

Photogrammetry preprocessing toolkit — extract frames from video, reframe 360 and fisheye into perspective views, auto-mask unwanted objects, and build clean datasets for 3D reconstruction.

## Requirements

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with CUDA support (strongly recommended — CPU works but is 10-50x slower)
- **4 GB+ VRAM** (8 GB+ recommended for SAM 3)
- [ffmpeg + ffprobe](https://ffmpeg.org/download.html) on PATH (required for video features)

## Getting Started

1. Extract the zip to any folder.
2. Double-click **ReconstructionZone.exe** to launch.
3. On first launch, the app checks for ffmpeg, CUDA, and model availability. Any issues are reported in the log console at the bottom of the window.

## Models

| Model | Type | Speed | Included |
|-------|------|-------|----------|
| **SAM 3** | Text-prompted | ~300ms/img | Downloads on first use (~2 GB). Requires [HuggingFace account](https://huggingface.co/facebook/sam3) with approved access. |
| **RF-DETR** | Transformer | ~50ms/img | Downloads on first use (~200 MB) |

SAM 3 gives the highest quality results and can detect arbitrary objects by text description ("selfie stick", "tripod shadow", "camera operator"). RF-DETR provides strong class-based detection for common objects.

### Setting up SAM 3

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Go to [facebook/sam3](https://huggingface.co/facebook/sam3) and click **Request access**.
3. Once approved (can take hours), open a terminal and run: `huggingface-cli login`
4. SAM 3 weights download automatically on first use in the app.

While waiting for approval, RF-DETR works immediately for common objects (person, backpack, car, etc.).

## Workflow

| Tab | What it does |
|-----|-------------|
| **Projects** | Track your photogrammetry projects, sources, and processing stages |
| **Extract** | Pull frames from 360/fisheye/standard video with reframing and filtering |
| **Mask** | Auto-detect and mask objects using text prompts or class selection |
| **Review** | Accept/reject/edit masks with an interactive brush editor |
| **Coverage** | Find spatial gaps in your dataset and extract bridge frames |

## Crash Logs

If the app crashes, a log is written to:
```
%USERPROFILE%\.reconstruction_zone\crash.log
```

Include this file when reporting issues.

## Support

For bug reports and feature requests, email the address provided in your Gumroad purchase confirmation.

## License

This is a licensed commercial product. See the included LICENSE file for terms.
