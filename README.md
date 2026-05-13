# Reconstruction Zone (Lite)

Automated object detection and masking for photogrammetry and 3D reconstruction. Extract frames from 360° or standard video, mask unwanted objects (people, tripods, equipment), and review results — all from one GUI.

## Requirements

- **Windows** 10 or 11
- **Python** 3.10, 3.11, or 3.12 — [download here](https://www.python.org/downloads/)
  - Check **"Add Python to PATH"** during installation
- **NVIDIA GPU** with up-to-date drivers — [download here](https://www.nvidia.com/drivers/)
- **~10 GB** free disk space (PyTorch ~2.5 GB, models ~3.5 GB, dependencies ~1 GB)

## Installation

```
git clone <repo-url> reconstruction-zone
cd reconstruction-zone
install.bat
```

The installer will:
1. Check your Python version
2. Detect your GPU and CUDA version
3. Create a virtual environment
4. Install PyTorch with the correct CUDA support
5. Install all dependencies
6. Download ffmpeg if needed
7. Create a desktop shortcut

## Usage

Double-click **launch.bat** (or the desktop shortcut).

On first launch, a setup wizard checks your environment and downloads AI model weights:
- **YOLO26** (6.5 MB) — fast object detection for known types (people, tripods, etc.)
- **RF-DETR** (129 MB) — transformer-based detection, higher accuracy
- **SAM3** (3.3 GB, optional) — text-prompted segmentation ("remove the photographer")

### Tabs

| Tab | Purpose |
|-----|---------|
| **Extract** | Load video, extract frames at intervals, reframe 360° to perspective views, apply LUTs and filters |
| **Mask** | Auto-detect and mask objects using AI models. Supports equirectangular, fisheye, and pinhole images |
| **Review** | Browse masked results as thumbnails, filter/sort, launch the OpenCV mask editor for manual fixes |

### Basic workflow

1. **Extract** — Load a video file, set extraction interval, click Extract
2. **Mask** — Set Input to your extracted frames folder, set Output, click Process
3. **Review** — Set Masks and Images folders, click Load, review results

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` not recognized | Reinstall Python with "Add to PATH" checked |
| GPU not detected | Update NVIDIA drivers, re-run `install.bat` |
| Video extraction fails | Setup wizard will offer ffmpeg download |
| `torch.cuda.is_available()` returns False | Driver/CUDA mismatch — update drivers and re-run `install.bat` |
| App crashes on launch | Check `%USERPROFILE%\.reconstruction_zone\crash.log` |

## Project Structure

```
reconstruction-zone/
├── install.bat              # One-time setup script
├── launch.bat               # Launch the app
├── setup_install.py         # Python installer (called by install.bat)
├── requirements-lite.txt    # Python dependencies
├── reconstruction-zone.ico  # App icon
├── reconstruction_gui/      # Main application
│   ├── reconstruction_zone.py   # GUI entry point
│   ├── reconstruction_pipeline.py  # Masking engine
│   ├── setup_wizard_lite.py     # First-launch wizard
│   └── tabs/source_tab.py      # Extract tab
└── prep360/                 # 360° video processing library
```
