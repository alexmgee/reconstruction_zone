# Reconstruction Zone — Masking Studio

**Automated object removal for photogrammetry and 3D Gaussian Splatting.**

Remove photographers, tripods, drones, and equipment from 360° and perspective images — producing clean masks ready for [COLMAP](https://colmap.github.io/), [Metashape](https://www.agisoft.com/), or [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) reconstruction pipelines.

Masking Studio is the GUI application within the [Reconstruction Zone](../README.md) project.

## Tabs

The GUI has 4 tabs:

| Tab | Purpose | Guide |
|-----|---------|-------|
| **Extract** | Video analysis, frame extraction, batch queue, fisheye calibration | [EXTRACT_TAB.md](docs/EXTRACT_TAB.md) |
| **Mask** | Model selection, detection, refinement, post-processing — the core pipeline | [MASK_TAB.md](docs/MASK_TAB.md) |
| **Review** | Thumbnail grid, accept/reject/edit, interactive OpenCV mask editor | [REVIEW_TAB.md](docs/REVIEW_TAB.md) |
| **Coverage** | Spatial gap detection, bridge frame extraction for missing angles | [COVERAGE_TAB.md](docs/COVERAGE_TAB.md) |

## Documentation

| Document | What it covers |
|----------|---------------|
| [Quickstart](docs/QUICKSTART.md) | First mask in 5 minutes — install, configure, run |
| [Architecture](docs/ARCHITECTURE.md) | Pipeline internals, data flow, module map, MaskConfig reference |
| [Model Guide](docs/MODELS.md) | SAM 3, YOLO26, RF-DETR, FastSAM — comparison, ensemble config, selection |
| [Contributing](docs/CONTRIBUTING.md) | How to add new segmenters, features, and pipeline stages |

## Models at a glance

| Model | Prompting | Speed | Quality | VRAM | Best for |
|-------|-----------|-------|---------|------|----------|
| **SAM 3** | Text descriptions | Slow | Highest | ~4 GB | Novel objects, precision work |
| **YOLO26** | COCO class IDs | Fast | Good | ~1 GB | Known objects, batch processing |
| **RF-DETR** | COCO class IDs | Medium | Good | ~2 GB | Transformer alternative to YOLO |
| **FastSAM** | Auto-segment | Very fast | Fair | ~1 GB | Quick previews |
| **Ensemble** | Combined | Slowest | Best | Sum | Final production masks |

See [MODELS.md](docs/MODELS.md) for detailed comparison, ensemble recipes, and class presets.

## Quick start

```bash
# Core dependencies
pip install numpy opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics tqdm pyyaml

# GUI
pip install customtkinter
python reconstruction_zone.py
```

See [QUICKSTART.md](docs/QUICKSTART.md) for a full walkthrough including model installation and first-mask tutorial.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Use a smaller YOLO model (`yolo_model_size="n"`), reduce batch_size, or set `device="cpu"` |
| SAM 3 returns 0 detections | Check that `remove_prompts` is populated — SAM 3 needs text prompts, not class IDs |
| Holes in mask (camera mount visible) | Enable `fill_holes=True` and set `mask_dilate_px=10` or higher |
| Poor detection at equirect poles | Ensure `geometry_aware=True` (cubemap decomposition) |
| Temporal flickering in video masks | Enable `vos_propagation=True` or increase `temporal_window` |
| Model download fails | Check internet connection; models cache to `~/.cache/` (YOLO) or `~/.prep360_sam_weights/` (SAM) |
| `pythonw` crashes silently on Windows | Run with `python` instead of `pythonw` to see error output |
| Fill holes zeros out entire mask | Masks are 0/1 uint8 — any threshold must use `mask > 0`, never `mask > 127` |

## Acknowledgments

- [SAM 3](https://github.com/facebookresearch/sam3) — Segment Anything Model 3 (Meta AI)
- [YOLO26](https://docs.ultralytics.com/) — Real-time instance segmentation (Ultralytics)
- [RF-DETR](https://github.com/roboflow/rf-detr) — Transformer-based detection (Roboflow)
- [LiVOS](https://github.com/hkchengrex/LiVOS) — Video object segmentation (CVPR 2025)
- [Cutie](https://github.com/hkchengrex/Cutie) — Video object segmentation (CVPR 2024)
- [ViTMatte](https://github.com/hustvl/ViTMatte) — Vision Transformer image matting
- [SAM](https://segment-anything.com/) — Segment Anything (Meta AI, used for boundary refinement)

## License

This project is licensed under the [GNU General Public License v3.0](../LICENSE).
