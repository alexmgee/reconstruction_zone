# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Panoex** — Modular 360° video processing pipeline for photogrammetry and 3D Gaussian Splatting. Converts equirectangular 360° video/images into perspective views ready for COLMAP, Metashape, or 3DGS workflows.

## Commands

```bash
# Install dependencies
pip install numpy opencv-python
pip install py360convert  # recommended for better quality (fallback exists)
pip install ultralytics   # for YOLO segmentation

# External tools required: ffmpeg, ffprobe (must be on PATH)

# CLI subcommands
python -m panoex analyze video.mp4
python -m panoex extract video.mp4 ./frames --interval 2.0
python -m panoex reframe ./frames ./perspectives --preset panoex_default
python -m panoex sky ./perspectives ./filtered
python -m panoex lut ./frames ./corrected --lut D-Log.cube
python -m panoex segment ./frames ./masks --preset photographer
python -m panoex presets --list
python -m panoex pipeline video.mp4 ./output --preset panoex_default

# GUI
python panoex_gui.py
```

No setup.py/pyproject.toml — run directly with Python. No test suite exists.

## Architecture

```
panoex/                          # Core library package (CLI + importable)
├── __main__.py                  # Entry: python -m panoex
├── cli.py                       # argparse subcommands → dispatches to core modules
├── __init__.py                  # Re-exports: VideoAnalyzer, FrameExtractor, Reframer,
│                                #   Ring, ViewConfig, Preset, PresetManager, SkyFilter,
│                                #   LUTProcessor, Segmenter, SegmentConfig, COCO_CLASSES
└── core/
    ├── __init__.py              # Also exports ExtractionConfig, apply_shadow_highlight,
    │                            #   batch_adjust_images (not in top-level __init__)
    ├── analyzer.py              # VideoAnalyzer — ffprobe metadata extraction
    ├── extractor.py             # FrameExtractor — ffmpeg frame extraction (fixed/scene/adaptive)
    ├── reframer.py              # Reframer — equirect→perspective via Ring/ViewConfig
    ├── presets.py               # Preset/PresetManager — full pipeline configs
    ├── sky_filter.py            # SkyFilter — 4-metric sky detection (needs 3/4 to classify)
    ├── lut.py                   # LUTProcessor — .cube LUT trilinear interpolation
    ├── segmenter.py             # Segmenter — YOLO11 instance segmentation with COCO classes
    ├── adjustments.py           # Shadow/highlight curve adjustment via LAB luminance LUT
    └── queue_manager.py         # VideoQueue — persistent batch queue (~/.panoex_queue.json)

panoex_gui.py                    # CustomTkinter GUI (7 tabs, ~3400 lines)
```

### Data flow: CLI pipeline command

`analyze` → `extract` → `reframe` → (optional: `lut`, `sky`, `segment`)

### Key abstractions

- **Ring(pitch, count, fov)** — Defines a ring of perspective views at a pitch angle. A `ViewConfig` combines multiple Rings + optional zenith/nadir to define all extraction views.
- **Preset** — Full pipeline config: extraction settings + reframe rings + color/LUT + sky filter + masking. 5 builtins: `panoex_default`, `dji_osmo_360`, `insta360_x3`, `gopro_max`, `slow_motion`. User presets saved/loaded by `PresetManager`.
- **VideoQueue** (queue_manager.py) — Persistent batch queue for the Extract tab. Items have status (PENDING/PROCESSING/DONE/ERROR/CANCELLED), 0-100 progress, reorder support. Persists to `~/.panoex_queue.json`; items marked "processing" reset to "pending" on app restart.
- **Adjustments** — Shadow/highlight sliders (0-100, 50=neutral). >50 lifts shadows/compresses highlights, <50 crushes shadows/boosts highlights. Applied via LUT on LAB luminance channel.

### GUI patterns (panoex_gui.py)

- **Threading**: All long operations run in background threads; `cancel_flag` (threading.Event) for cancellation
- **Logging**: `log_queue` (queue.Queue) polled by tkinter `after()` for thread-safe log display
- **CollapsibleSection**: Reusable widget for expandable/collapsible UI sections
- **Equirect preview** (Reframe tab): Draws colored overlay rectangles on equirect image showing extraction regions per ring. Colors: red, teal, yellow, mint, plum for rings; sky blue for zenith, tan for nadir.
- **Extract tab batch queue**: Add/remove/reorder videos, settings panel, planned frame preview table

### Segmentation details

YOLO11 models: `yolo11{n,s,m,l,x}-seg.pt` (n=nano default). SegmentConfig: model, classes, confidence, iou_threshold, dilate_pixels, invert_mask, device. Class presets map to COCO class IDs (person=0, photographer=person+gear, equipment, vehicles, animals, all_dynamic).

## Other directories

- **masking_gui/** — Separate advanced masking system (masking_v2.py). Multi-model fallback (SAM3→FastSAM→EfficientSAM→SAM2→MobileSAM→OpenCV), multi-geometry support (pinhole/fisheye/equirect/cubemap), text-based prompting, temporal consistency. Has its own README.md.
- **upgrade/** — UI/pipeline spec documents for planned restructuring (PANOEX_SPEC.md, PANOEX_UI_SPEC.md, PANOEX_GUI_UNIFIED.md)
- **docs/** — User guide (PANOEX_GUI_USER_GUIDE.md)

## Legacy scripts (root level, not part of panoex package)

- `equirect_to_perspectives.py` — Original standalone with COLMAP camera.txt/images.txt, Metashape XML, rig config, mask reprojection
- `extract_frames.py` — ffmpeg wrapper with preset profiles (photogrammetry, 3dgs, scene, timelapse, etc.)
- `filter_blurry.py` — Blur detection via Laplacian variance + Sobel + Brenner scoring, parallel processing
- `reframing_gui.py` — Legacy 3-tab GUI (superseded by panoex_gui.py)

## Key design decisions

- **Core library is GUI-independent**: All processing works via CLI; GUI is a thin wrapper
- **py360convert optional**: Reframer has custom equirect→perspective fallback
- **ProcessPoolExecutor**: Used for batch operations in reframer, sky filter, adjustments
- **ffmpeg/ffprobe via subprocess**: No Python video decoding — analyzer and extractor shell out to ffmpeg tools
- **No package manager**: Direct script execution, manual dependency installation
