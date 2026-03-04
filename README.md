# Reconstruction Zone

**End-to-end toolkit for photogrammetry and 3D Gaussian Splatting — from raw 360° video to reconstruction-ready datasets.**

Reconstruction Zone combines automated masking (remove photographers, tripods, drones) with 360° video preparation (frame extraction, equirect-to-perspective reframing, quality filtering) into a single workflow. The result: clean, consistent image sets ready for [COLMAP](https://colmap.github.io/), [Metashape](https://www.agisoft.com/), or [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) pipelines.

## What's inside

### Masking Studio (reconstruction_gui/)

The main application. Automated object detection and masking for photogrammetry images — 360° equirectangular, perspective, and fisheye.

- **Multi-model segmentation** — SAM 3 (text-prompted), YOLO26 (fast), RF-DETR (transformer), FastSAM, with ensemble fusion
- **360°-aware processing** — Cubemap decomposition handles pole distortion that breaks standard detectors
- **Temporal propagation** — LiVOS/Cutie propagate masks across video frames
- **Refinement pipeline** — SAM boundary refinement, ViTMatte alpha matting, shadow detection
- **Review workflow** — Thumbnail grid with accept/reject/edit, interactive mask editor

[Full documentation](reconstruction_gui/README.md)

### prep360 (prep360/)

Library and CLI for preparing 360° video for reconstruction. Used by Masking Studio's Extract and Coverage tabs.

- **Frame extraction** — Fixed interval, scene detection, or adaptive modes via ffmpeg
- **Equirect-to-perspective reframing** — Configurable rings of views with presets for common cameras
- **Quality filtering** — Sharpness scoring (Laplacian/Sobel/Brenner), sky detection, motion selection
- **Fisheye support** — Dual-fisheye calibration and reframing (DJI Osmo 360, etc.)
- **Gap detection** — Spatial coverage analysis with automatic bridge frame extraction
- **COLMAP/XMP export** — Generate COLMAP databases and XMP pose priors from Metashape alignments

```bash
python -m prep360 analyze video.mp4
python -m prep360 extract video.mp4 ./frames --interval 2.0
python -m prep360 reframe ./frames ./perspectives --preset prep360_default
python -m prep360 pipeline video.mp4 ./output --preset prep360_default
```

### Metashape Exporter (metashape_exporter.py)

Run inside Agisoft Metashape Pro to export camera poses, sparse points, images, and masks — then feed into prep360's COLMAP export pipeline.

## Quick start

```bash
# Core dependencies
pip install numpy opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics tqdm pyyaml

# Launch the GUI
pip install customtkinter
python reconstruction_gui/reconstruction_zone.py
```

See [reconstruction_gui/docs/QUICKSTART.md](reconstruction_gui/docs/QUICKSTART.md) for a full walkthrough.

## Project structure

```
reconstruction-zone/
├── reconstruction_gui/              # Masking Studio — the main application
│   ├── reconstruction_zone.py     # GUI launcher (4 tabs: Extract, Mask, Review, Coverage)
│   ├── reconstruction_pipeline.py  # Core masking pipeline
│   ├── tabs/                 # Extract, Coverage tab modules
│   └── docs/                 # Per-tab guides, QUICKSTART, ARCHITECTURE, MODELS
├── prep360/                  # 360° video preparation library + CLI
│   ├── cli.py                # python -m prep360
│   └── core/                 # Analyzer, extractor, reframer, filters, COLMAP export
├── metashape_exporter.py     # Metashape Pro export script
└── legacy/                   # Archived scripts (panoex_gui.py, etc.)
```

## Documentation

- [Quickstart](reconstruction_gui/docs/QUICKSTART.md) — First mask in 5 minutes
- [Architecture](reconstruction_gui/docs/ARCHITECTURE.md) — Pipeline internals, data flow, module map
- [Model Guide](reconstruction_gui/docs/MODELS.md) — SAM 3, YOLO26, RF-DETR, FastSAM comparison
- **Tab guides:** [Extract](reconstruction_gui/docs/EXTRACT_TAB.md) | [Mask](reconstruction_gui/docs/MASK_TAB.md) | [Review](reconstruction_gui/docs/REVIEW_TAB.md) | [Coverage](reconstruction_gui/docs/COVERAGE_TAB.md)
- [Contributing](reconstruction_gui/docs/CONTRIBUTING.md)

## Requirements

- Python 3.10+ (tested on 3.12–3.14)
- NVIDIA GPU with CUDA (strongly recommended; CPU works but is 10-50x slower)
- ffmpeg + ffprobe on PATH (for video processing)

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
