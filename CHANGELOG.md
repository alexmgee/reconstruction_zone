# Changelog

All notable changes to Reconstruction Zone will be documented in this file.

## [1.0.0] — 2026-07-16

First commercial release. The app ships four tabs — **Extract → Mask → Review → Align** — focused on the core preprocessing path.

### Added

- **Preset-driven ERP reframing** with rig/station/flat output layouts. Reframing now writes `reframe_metadata.json` (pinhole intrinsics per view) with every run and `rig_config.json` (Metashape Pro rig sensor definitions) for rig-layout output.
- **Static-mask library** in the Mask tab — import reusable masks (tripod, nadir rig, vehicle mounts), manage them as named layers, and stamp them onto existing mask batches in place.
- **Native COLMAP binary reader** — the Align tab reads `cameras.bin`/`images.bin`/`points3D.bin` sparse models directly, without pycolmap and without a `model_converter` round-trip.
- **Fisheye distortion fold-back guard** — corrupted rays beyond the distortion polynomial's fold boundary are excluded from useful-pixel derivation, fixing over-wide support angles on the no-mask path.
- **Setup wizard** for first-run model download and SAM 3 gated-access setup.

### Fixed

- DJI SRT geotagging now uses a single altitude datum (`rel_alt` + derived ground elevation) — multi-clip datasets no longer sink below the reconstructed ground from barometric re-zeroing.
- Paired split-video extraction: flattened output, basenames derived from video stems, per-basename manifests.
- Fisheye circle-mask dilation and temporal-flush on resolution change during masking.
- Cubeface conversion defaults to 2100 px face width and rejects non-positive values.

### Changed vs 0.1.0

- **Projects and Coverage tabs deferred** to a future release; the Adjust tab was removed. Their processing cores (project store, gap detector) remain in the codebase.
- Mask pipeline focuses on SAM 3 (text-prompted) and RF-DETR (class-based) with YOLO26/FastSAM/EfficientSAM available from source; alpha matting, VOS temporal propagation, shadow detection, and SAM boundary refinement were removed from this edition.
- Align expects a user-installed COLMAP binary (pointed to from the app); SphereSfM remains supported for ERP workflows.

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
