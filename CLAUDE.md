# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules
- Always use WebSearch to verify claims of any kind. Never rely solely on training data.


## Project Overview

**Reconstruction Zone** — End-to-end toolkit for photogrammetry and 3D Gaussian Splatting. Two main components:

1. **Masking Studio** (`reconstruction_gui/`) — The main application. Automated object detection and masking for 360° and perspective images, with multi-model segmentation, temporal propagation, and a full review workflow.
2. **prep360** (`prep360/`) — 360° video preparation library and CLI. Frame extraction, equirect-to-perspective reframing, quality filtering, COLMAP export. Used by Masking Studio's Extract and Coverage tabs.

## Commands

```bash
# Install dependencies
pip install numpy opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics   # YOLO26 segmentation
pip install py360convert  # recommended for better reframing quality (fallback exists)
pip install customtkinter # GUI

# External tools required: ffmpeg, ffprobe (must be on PATH)

# Launch the GUI
python reconstruction_gui/reconstruction_zone.py

# prep360 CLI subcommands
python -m prep360 analyze video.mp4
python -m prep360 extract video.mp4 ./frames --interval 2.0
python -m prep360 reframe ./frames ./perspectives --preset prep360_default
python -m prep360 sky ./perspectives ./filtered
python -m prep360 blur ./frames ./sharp_frames --percentile 80
python -m prep360 lut ./frames ./corrected --lut D-Log.cube
python -m prep360 segment ./frames ./masks --preset photographer
python -m prep360 presets --list
python -m prep360 pipeline video.mp4 ./output --preset prep360_default
```

No setup.py/pyproject.toml — run directly with Python. No test suite exists.

## Architecture

```
reconstruction-zone/
├── reconstruction_gui/                     # Masking Studio — the main application
│   ├── reconstruction_zone.py            # GUI launcher (4 tabs: Extract, Mask, Review, Coverage)
│   ├── reconstruction_pipeline.py                # Core masking pipeline — models, cubemap, postprocessing
│   ├── sam3_pipeline.py             # SAM 3 unified video predictor wrapper
│   ├── sam_refinement.py            # SAM boundary refinement
│   ├── matting.py                   # ViTMatte alpha matting
│   ├── vos_propagation.py           # LiVOS/Cutie temporal mask propagation
│   ├── shadow_detection.py          # Multi-method shadow detection
│   ├── colmap_validation.py         # Geometric mask validation against COLMAP
│   ├── review_gui.py                # Standalone review app + reusable widgets
│   ├── app_infra.py                 # GUI infrastructure — logging, threading, prefs
│   ├── widgets.py                   # Shared UI components
│   ├── review_masks.py              # Interactive OpenCV mask editor
│   ├── tabs/
│   │   ├── source_tab.py            # Extract tab: video analysis, extraction queue, fisheye
│   │   └── gaps_tab.py              # Coverage tab: spatial gap detection, bridge extraction
│   └── docs/                        # Per-tab guides, QUICKSTART, ARCHITECTURE, MODELS
│
├── prep360/                         # 360° video preparation library + CLI
│   ├── __main__.py                  # Entry: python -m prep360
│   ├── cli.py                       # argparse subcommands
│   └── core/
│       ├── analyzer.py              # VideoAnalyzer — ffprobe metadata extraction
│       ├── extractor.py             # FrameExtractor — ffmpeg frame extraction
│       ├── reframer.py              # Reframer — equirect→perspective via Ring/ViewConfig
│       ├── presets.py               # Preset/PresetManager — pipeline configs
│       ├── sky_filter.py            # SkyFilter — 4-metric sky detection
│       ├── blur_filter.py           # BlurFilter — sharpness scoring + filtering
│       ├── lut.py                   # LUTProcessor — .cube LUT trilinear interpolation
│       ├── segmenter.py             # Segmenter — YOLO instance segmentation
│       ├── colmap_export.py         # Metashape XML → COLMAP/XMP pipeline
│       ├── fisheye_reframer.py      # Dual-fisheye reframing
│       ├── gap_detector.py          # Spatial coverage gap analysis
│       ├── bridge_extractor.py      # Bridge frame extraction for gaps
│       └── queue_manager.py         # VideoQueue — persistent batch queue
│
├── metashape_exporter.py            # Metashape Pro export script
├── legacy/                          # Archived scripts (panoex_gui.py, etc.)
├── upgrade/                         # Historical spec documents
└── docs/                            # Project-level docs
```

### Key abstractions

- **Ring(pitch, count, fov)** — Defines a ring of perspective views at a pitch angle. A `ViewConfig` combines multiple Rings + optional zenith/nadir to define all extraction views.
- **Preset** — Full pipeline config: extraction settings + reframe rings + color/LUT + sky filter + masking. 5 builtins: `prep360_default`, `dji_osmo_360`, `insta360_x3`, `gopro_max`, `slow_motion`. User presets saved/loaded by `PresetManager`.
- **MaskingPipeline** — Core masking engine. Supports SAM3, YOLO26, RF-DETR, FastSAM with auto-fallback. Cubemap decomposition for equirectangular images. Two-stage postprocessing.
- **BaseSegmenter** — ABC for segmentation models. Implement `initialize()` and `segment_image()` to add new models.
- **VideoQueue** (queue_manager.py) — Persistent batch queue for the Extract tab. Persists to `~/.prep360_queue.json`.

### Masking pipeline data flow

`Input Image` → `Geometry Detection` → `Cubemap Decomposition (if equirect)` → `Per-Face Segmentation` → `Merge` → `Optional: SAM Refine, VOS Propagation, Shadow Detection, Edge Injection` → `Postprocessing (dilation, fill holes)` → `Optional: Alpha Matting` → `Quality Scoring` → `Output`

### prep360 CLI pipeline data flow

`analyze` → `extract` → (optional: `blur`) → `reframe` → (optional: `lut`, `sky`, `segment`)

### Segmentation details

- **reconstruction_pipeline.py**: Multi-model (SAM3, YOLO26, RF-DETR, FastSAM). Ensemble mode with weighted mask fusion. Masks are 0/1 uint8 (multiply by 255 only at save time).
- **prep360 segmenter.py**: YOLO-only, simpler. Class presets map to COCO class IDs.

## Key design decisions

- **Core library is GUI-independent**: All processing works via CLI or Python API; GUI is a thin wrapper
- **Masks are 0/1 uint8**: The single most common bug is treating them as 0/255. Any threshold must use `mask > 0`, never `mask > 127`
- **Two-stage postprocessing**: Lightweight cleanup at cubemap face resolution (1024px), heavy fill-holes at full equirect resolution (7680px)
- **py360convert optional**: Reframer has custom equirect→perspective fallback
- **ProcessPoolExecutor**: Used for batch operations in reframer, sky filter, adjustments
- **ffmpeg/ffprobe via subprocess**: No Python video decoding
- **No package manager**: Direct script execution, manual dependency installation
- **Lazy imports**: Optional dependencies (SAM3, RF-DETR, matting, VOS) are imported inside functions, never at module top level
