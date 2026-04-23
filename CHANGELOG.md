# Changelog

All notable changes to Reconstruction Zone will be documented in this file.

## [0.1.0] — 2026-04-21

First public release.

### Masking Studio (GUI)

- **Projects tab** — Central registry for photogrammetry projects with lifecycle tracking, multi-tool detection (Metashape/RealityScan/COLMAP), and export management.
- **Extract tab** — Frame extraction from 360, fisheye, and standard video. Equirect-to-perspective reframing with configurable view rings. Sharpest-frame selection, scene detection, blur/sky filtering, LUT application. Persistent batch queue.
- **Mask tab** — Multi-model auto-masking with SAM 3 (text-prompted), YOLO26 (class-based), RF-DETR (transformer), FastSAM, and EfficientSAM. 360-aware cubemap decomposition. Ensemble mode with weighted mask fusion. Optional SAM boundary refinement, alpha matting, VOS temporal propagation, shadow detection.
- **Review tab** — Paginated thumbnail grid with accept/reject/skip workflow. Interactive OpenCV mask editor with brush, flood fill, lasso, and zoom/pan.
- **Align tab** — COLMAP and SphereSfM integration with per-run isolation, binary validation, mask remapping, and crash recovery.
- **Coverage tab** — Spatial gap detection from COLMAP/Metashape reconstructions. Bridge frame extraction to fill coverage holes.

### prep360 (CLI library)

- Video analysis, frame extraction, equirect/fisheye reframing, blur filtering, sky filtering, LUT processing, YOLO segmentation, COLMAP export, gap detection, bridge extraction.
- 6 built-in presets: `prep360_default`, `dji_osmo_360`, `insta360_x3`, `gopro_max`, `lightweight`, `dense`.
- Dual-fisheye reframing with station-directory output for Metashape import.

### Infrastructure

- Dual-distribution build system: GitHub (GPL v3, all models) and Gumroad (proprietary, AGPL-free).
- PyInstaller spec with CUDA DLL handling and conditional YOLO exclusion.
- Rotating crash log at `~/.reconstruction_zone/crash.log`.
- Crash dialog for fatal errors under pythonw.exe.
- Startup environment checks (CUDA, ffmpeg, ffprobe).
- 36 unit tests covering reframer geometry, preset management, queue persistence, distribution flags, and version consistency.
