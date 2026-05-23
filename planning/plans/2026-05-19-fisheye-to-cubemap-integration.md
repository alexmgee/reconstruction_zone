# Fisheye-to-Cubemap Integration Plan

> **Scope:** Absorb the full capability of `D:\Projects\Fisheye-to-Cubemap` into Reconstruction Zone. This includes the cubeface conversion engine, adaptive undistortion, ERP reframing, Metashape sensor discovery, lens mapping resolution, and Metashape→COLMAP scene export. The Coverage tab is removed and replaced with an Export tab for the Metashape→COLMAP workflow.

**Source project:** `D:\Projects\Fisheye-to-Cubemap` (~18,774 lines across 14 modules + 4,345 lines GUI)
**Target project:** `D:\Projects\reconstruction-zone` (main branch)

---

## Table of Contents

1. [Current State](#1-current-state)
2. [Target State](#2-target-state)
3. [Module Inventory & Disposition](#3-module-inventory--disposition)
4. [Phase 1: Foundation — Core Engine Import](#phase-1-foundation--core-engine-import)
5. [Phase 2: Extract Tab — Cubeface & Adaptive Conversion](#phase-2-extract-tab--cubeface--adaptive-conversion)
6. [Phase 3: Export Tab — Metashape→COLMAP Workflow](#phase-3-export-tab--metashapecolmap-workflow)
7. [Phase 4: Coverage Tab Removal](#phase-4-coverage-tab-removal)
8. [Phase 5: Cross-Tab Integration](#phase-5-cross-tab-integration)
9. [Phase 6: Cleanup & Validation](#phase-6-cleanup--validation)
10. [Dependency Map](#dependency-map)
11. [Risk Register](#risk-register)
12. [Open Questions](#open-questions)

---

## 1. Current State

### Reconstruction Zone tab bar (main)
```
Projects → Adjust → Extract → Mask → Review → Align → Coverage
```

### Extract tab — existing 360° Processing section
- **Split Lenses**: Demux .osv/.360/.insv → front/back raw lens videos
- **Reframing**: Pinhole perspective extraction from OSV or existing ERP frames
  - Input: .osv file OR existing ERP frames folder + optional masks
  - Preset dropdown (4 presets: 10/14/22/26 views, all dual-fisheye Osmo 360)
  - Custom calibration (JSON override for built-in equidistant model)
  - Output: 1600×1600 JPEG perspective crops
- **Projection model**: Equidistant only (cv2.fisheye)
- **Camera support**: DJI Osmo 360 dual-fisheye only (hardcoded calibration)

### Coverage tab (being removed)
- Gap Analysis: spatial gap detection from COLMAP/Metashape/XMP reconstructions
- Bridge Extraction: extract frames from video at gap timestamps
- 437 lines UI + 843 lines gap_detector + 563 lines bridge_extractor
- No downstream consumers — leaf node in the pipeline

### Align tab
- Full COLMAP/SphereSfM runner (feature extraction → matching → reconstruction)
- Rig presets (DJI Osmo 360 dual-fisheye)
- 3D point cloud viewer
- 2,711 lines

### What Reconstruction Zone CANNOT do today
- No equisolid projection model (the primary model for DJI/Insta360 fisheye)
- No Metashape XML calibration parsing (limited to hardcoded Osmo 360)
- No single-fisheye support (only dual pairs)
- No Fourier distortion corrections
- No RBF-interpolated remapping (uses cv2.fisheye remap)
- No adaptive single-pinhole routing (cubemap-or-nothing)
- No Metashape→COLMAP scene export (must run COLMAP from scratch)
- No mixed-sensor scene support (fisheye + frame + equirect in one export)

---

## 2. Target State

### Tab bar after integration
```
Projects → Adjust → Extract → Mask → Review → Align → Export
```

### Extract tab — expanded 360° Processing section
Everything it does today, plus:
- **Cubeface conversion**: Metashape-calibrated fisheye → 5 pinhole cubefaces (equisolid, equidistant, pinhole, equirect)
- **Adaptive undistort**: Fisheye → single tighter pinhole (automatic routing based on lens characteristics)
- **ERP reframing**: F2C's equirect → perspective crops (cubemap 6-view or reframe 16-view modes)
- **Fourier corrections**: 96-coefficient distortion corrections from Metashape calibration
- **Calibration source**: Metashape XML (any sensor) in addition to existing JSON/hardcoded
- **Routing intelligence**: Automatic cubemap-vs-single-pinhole decision per sensor

### New Export tab (replaces Coverage)
- **Sensor discovery**: Parse Metashape cameras.xml → enumerate sensors with calibration details
- **Lens mapping**: Resolve which images belong to which sensor/lens (5-tier automatic + manual override)
- **Mapping validation**: Structured diagnostics (INFO/WARNING/BLOCKER) before export
- **Scene export**: Metashape alignment → COLMAP sparse model (all PINHOLE cameras)
  - Fisheye sensors → cubeface or adaptive pinhole conversion + pose composition
  - Frame sensors → passthrough with undistortion
  - Equirect sensors → perspective split + pose composition
  - Mixed scenes (multiple sensor types in one export)
- **Scene diagnostics**: Scale normalization, conversion reports, validation reports
- **Processing stamps**: Cache invalidation for incremental re-export

### Align tab — new opportunities
- "Import COLMAP Scene" button that loads an Export tab output as a starting point
- Ability to refine an exported scene with additional COLMAP BA passes
- No structural changes required — just cross-tab handoff

---

## 3. Module Inventory & Disposition

### F2C modules → where they land in Reconstruction Zone

| F2C Module | Lines | New Location | Action |
|---|---|---|---|
| `AM_ImageAndMask_to_cubemap_v4.py` | 2,332 | `prep360/core/cubeface_engine.py` | Copy + rename, preserve as stable engine |
| `gui/cubeface_processing.py` | 634 | `prep360/core/cubeface_processing.py` | Copy, update imports |
| `gui/adaptive_undistort.py` | 845 | `prep360/core/adaptive_undistort.py` | Copy, update imports |
| `gui/erp_reframe.py` | 530 | `prep360/core/erp_reframe.py` | Copy, update imports |
| `gui/routing.py` | 295 | `prep360/core/conversion_routing.py` | Copy, update imports |
| `gui/sensor_discovery.py` | 390 | `prep360/core/sensor_discovery.py` | Copy, update imports |
| `gui/mapping_resolver.py` | 1,656 | `prep360/core/mapping_resolver.py` | Copy, update imports |
| `gui/fourier_corrections.py` | 344 | `prep360/core/fourier_corrections.py` | Copy, update imports |
| `gui/corrected_rays.py` | 150 | `prep360/core/corrected_rays.py` | Copy, update imports |
| `gui/scene_manifest.py` | 295 | `prep360/core/scene_manifest.py` | Copy, update imports |
| `gui/processing_stamp.py` | 143 | `prep360/core/processing_stamp.py` | Copy, update imports |
| `gui/solid_angle.py` | 133 | `prep360/core/solid_angle.py` | Copy, update imports |
| `metashape_cameras_to_colmap.py` | 6,682 | `prep360/core/colmap_exporter.py` | Copy + rename, update imports |
| `gui/gui.py` | 4,345 | NOT imported | GUI rebuilt natively in RZ tabs |

### F2C tests → imported alongside modules

| F2C Test | Lines | New Location |
|---|---|---|
| `tests/test_cubeface_processing.py` | ~800 | `tests/test_cubeface_processing.py` |
| `tests/test_routing.py` | ~170 | `tests/test_conversion_routing.py` |
| `tests/test_erp_reframe.py` | ~560 | `tests/test_erp_reframe.py` |
| `tests/test_sensor_discovery.py` | ~440 | `tests/test_sensor_discovery.py` |
| `tests/test_fourier_corrections.py` | ~440 | `tests/test_fourier_corrections.py` |
| `tests/test_camera_records.py` | ~760 | `tests/test_colmap_exporter.py` |
| `tests/test_scene_manifest.py` | ~260 | `tests/test_scene_manifest.py` |
| `tests/test_solid_angle.py` | ~120 | `tests/test_solid_angle.py` |
| `tests/test_corrected_rays.py` | ~140 | `tests/test_corrected_rays.py` |
| `tests/fixtures/` | — | `tests/fixtures/` (merge) |

### Reconstruction Zone files removed

| File | Lines | Reason |
|---|---|---|
| `reconstruction_gui/tabs/gaps_tab.py` | 437 | Coverage tab removed |
| `prep360/core/gap_detector.py` | 843 | Coverage backend removed |
| `prep360/core/bridge_extractor.py` | 563 | Coverage backend removed |

---

## Phase 1: Foundation — Core Engine Import

**Goal:** Get all F2C library modules into `prep360/core/` with passing imports and tests. No GUI changes yet.

### Step 1.1: Copy the cubeface engine
- [ ] Copy `AM_ImageAndMask_to_cubemap_v4.py` → `prep360/core/cubeface_engine.py`
- [ ] The file is a stable, self-contained script. Do NOT refactor internals.
- [ ] Verify it has no imports from `gui/` — it should be standalone (numpy, cv2, scipy, argparse, pathlib)
- [ ] Add `cubeface_engine` to `prep360/core/__init__.py` lazy imports if needed
- [ ] Verify: `python -c "from prep360.core.cubeface_engine import compute_rays, compute_interpolation_function"`

### Step 1.2: Copy support modules (no inter-dependency)
These modules depend only on stdlib + numpy, not on each other or the engine:
- [ ] `gui/processing_stamp.py` → `prep360/core/processing_stamp.py` (hashlib, json, pathlib only)
- [ ] `gui/scene_manifest.py` → `prep360/core/scene_manifest.py` (json, dataclasses, pathlib only)
- [ ] Verify: `python -c "from prep360.core.processing_stamp import compute_stamp_digest; from prep360.core.scene_manifest import RoutingDecision"`

### Step 1.3: Copy modules that depend on cubeface_engine
These import from `AM_ImageAndMask_to_cubemap_v4` — update to `prep360.core.cubeface_engine`:
- [ ] `gui/fourier_corrections.py` → `prep360/core/fourier_corrections.py`
  - Update: no engine imports (only hashlib, xml, numpy) — verify
- [ ] `gui/solid_angle.py` → `prep360/core/solid_angle.py`
  - Update: `from AM_ImageAndMask_to_cubemap_v4 import ...` → `from prep360.core.cubeface_engine import ...`
- [ ] `gui/corrected_rays.py` → `prep360/core/corrected_rays.py`
  - Update: `from gui.fourier_corrections import ...` → `from prep360.core.fourier_corrections import ...`
  - Update: `from AM_ImageAndMask_to_cubemap_v4 import ...` → `from prep360.core.cubeface_engine import ...`
- [ ] Verify: `python -c "from prep360.core.corrected_rays import compute_rays_with_corrections"`

### Step 1.4: Copy orchestration modules
These import from the support modules above:
- [ ] `gui/adaptive_undistort.py` → `prep360/core/adaptive_undistort.py`
  - Update: `from AM_ImageAndMask_to_cubemap_v4 import ...` → `from prep360.core.cubeface_engine import ...`
  - Update: `from gui.corrected_rays import ...` → `from prep360.core.corrected_rays import ...`
- [ ] `gui/cubeface_processing.py` → `prep360/core/cubeface_processing.py`
  - Update: `from AM_ImageAndMask_to_cubemap_v4 import ...` → `from prep360.core.cubeface_engine import ...`
  - Update: `from gui.fourier_corrections import ...` → `from prep360.core.fourier_corrections import ...`
  - Update: `from gui.corrected_rays import ...` → `from prep360.core.corrected_rays import ...`
  - Update: `from gui.processing_stamp import ...` → `from prep360.core.processing_stamp import ...`
- [ ] `gui/erp_reframe.py` → `prep360/core/erp_reframe.py`
  - No engine imports — pure numpy + cv2. Verify and copy.
- [ ] `gui/routing.py` → `prep360/core/conversion_routing.py`
  - Update: `from gui.fourier_corrections import ...` → `from prep360.core.fourier_corrections import ...`
  - Update: `from gui.adaptive_undistort import ...` → `from prep360.core.adaptive_undistort import ...`
  - Update: `from gui.scene_manifest import ...` → `from prep360.core.scene_manifest import ...`
- [ ] Verify all: `python -c "from prep360.core.cubeface_processing import process_cubeface_sensor; from prep360.core.adaptive_undistort import process_sensor_adaptive; from prep360.core.erp_reframe import process_equirect_sensor; from prep360.core.conversion_routing import get_routing"`

### Step 1.5: Copy discovery & mapping modules
- [ ] `gui/sensor_discovery.py` → `prep360/core/sensor_discovery.py`
  - Update: `from gui.fourier_corrections import ...` → `from prep360.core.fourier_corrections import ...`
- [ ] `gui/mapping_resolver.py` → `prep360/core/mapping_resolver.py`
  - Check: imports `metashape_cameras_to_colmap.parse_lens_camera_map()` — this will need updating after the exporter is copied (Step 1.6)
  - Temporary: comment out or stub the cross-reference until Step 1.6
- [ ] Verify: `python -c "from prep360.core.sensor_discovery import discover_sensors"`

### Step 1.6: Copy the COLMAP exporter
- [ ] Copy `metashape_cameras_to_colmap.py` → `prep360/core/colmap_exporter.py`
- [ ] Update: `from gui.erp_reframe import register_erp_face_entries` → `from prep360.core.erp_reframe import register_erp_face_entries`
- [ ] Resolve mapping_resolver cross-reference from Step 1.5
- [ ] Verify: `python -c "from prep360.core.colmap_exporter import parse_metashape_cameras_xml, write_colmap_training_scene"`

### Step 1.7: Copy and adapt tests
- [ ] Copy `tests/` directory from F2C to `tests/` in reconstruction-zone
  - Merge fixtures into existing `tests/fixtures/` if any
  - Update all import paths (`from gui.X import ...` → `from prep360.core.X import ...`)
  - Update `from AM_ImageAndMask_to_cubemap_v4 import ...` → `from prep360.core.cubeface_engine import ...`
  - Rename test files per disposition table above
- [ ] Run: `python -m pytest tests/test_cubeface_processing.py tests/test_conversion_routing.py tests/test_erp_reframe.py tests/test_sensor_discovery.py tests/test_fourier_corrections.py -v`
- [ ] Fix any import or path failures

### Step 1.8: Dependency check
- [ ] Verify scipy is already in reconstruction-zone's dependency list (needed by adaptive_undistort for KDTree)
  - If not, document it as a new required dependency
- [ ] No new required dependencies beyond numpy, opencv-python, scipy
- [ ] Optional: colour-science (for chart correction in Adjust tab — already optional there)

**Phase 1 verification:** All 12 library modules importable from `prep360.core.*`, all copied tests pass, no GUI changes.

---

## Phase 2: Extract Tab — Cubeface & Adaptive Conversion

**Goal:** Extend the Extract tab's "360 Processing" section to support Metashape-calibrated fisheye conversion alongside the existing Osmo 360 workflow.

### Step 2.1: Understand the existing fisheye section structure

Current "360 Processing" section in `source_tab.py` (line 2898):
```
CollapsibleSection "360 Processing" (core=True)
  ├─ CollapsibleSection "Split Lenses" (demux .osv → front/back)
  └─ CollapsibleSection "Reframing" (pinhole extraction)
       ├─ Video: .osv/.360/.insv file input
       ├─ Frames: existing ERP frames folder
       ├─ Masks: masks for ERP or fisheye frames
       ├─ Output: perspective output directory
       ├─ CollapsibleSection "Custom Calibration" (JSON override)
       ├─ Preset dropdown (4 Osmo 360 presets)
       └─ Reframe button
```

### Step 2.2: Add Metashape calibration input to Reframing section
- [ ] Add a new input row after "Custom Calibration": **Metashape XML** file browser
  - Label: "Metashape Cal:"
  - Placeholder: "Metashape sensor calibration .xml (optional)..."
  - Browse button filters for `.xml`
  - Store as `app.fisheye_metashape_cal_entry`
- [ ] When a Metashape XML is loaded:
  - Parse with `sensor_discovery.classify_sensor_element()` to detect projection type
  - Display detected projection type and FOV next to the input (info label)
  - The Metashape calibration takes precedence over the JSON custom calibration and the built-in Osmo 360 defaults
- [ ] Add tooltip explaining when to use Metashape XML vs JSON calibration

### Step 2.3: Add conversion mode selector
- [ ] Add a segmented button or radio group below the calibration inputs:
  - **Dual-Fisheye Reframe** (existing behavior — presets, dual lens pairs)
  - **Cubeface Conversion** (new — Metashape-calibrated → 5 pinhole faces)
  - **Adaptive Pinhole** (new — Metashape-calibrated → single tighter pinhole)
  - **ERP Reframe** (existing behavior, but now also supports F2C's 6-view cubemap and 16-view reframe modes)
- [ ] Mode selection shows/hides relevant controls:
  - Dual-Fisheye Reframe: preset dropdown, OSV input (existing UI)
  - Cubeface Conversion: face width input, output format dropdown, layout radio (Station/Rig)
  - Adaptive Pinhole: theta-max threshold, stretch threshold, output width budget
  - ERP Reframe: split mode radio (cubemap 6-view / reframe 16-view / existing prep360 presets), split width input
- [ ] Default mode: Dual-Fisheye Reframe (preserves current behavior)

### Step 2.4: Add Cubeface Conversion mode controls
- [ ] Face width input (0 = auto-calculate from lens FOV)
  - Default: 0 (auto)
  - Tooltip: "Output resolution per cubeface. 0 = auto-calculate from calibration FOV."
- [ ] Output format dropdown: PNG / TIFF / JPEG
  - Default: PNG (masks always PNG regardless)
- [ ] Layout radio: Station (per-image subdirs) / Rig (per-face subdirs)
  - Default: Station
  - Tooltip explaining when each is appropriate
- [ ] Lens label input (optional, for multi-lens datasets)
- [ ] Force reprocess checkbox
- [ ] Lens-only mask input (single mask applied to all images)

### Step 2.5: Add Adaptive Pinhole mode controls
- [ ] Theta-max threshold slider (30°–80°, default 55°)
  - Tooltip: "Maximum half-angle before switching to cubemap split. Higher = more distortion tolerance."
- [ ] Stretch threshold slider (1.5–5.0, default 3.0)
  - Tooltip: "Maximum edge stretch factor. Projection-aware; accounts for lens model."
- [ ] Output width budget input (2000–8000, default 6000)
  - Tooltip: "Maximum output image dimension in pixels. Memory safety limit."
- [ ] Routing decision display: after loading calibration + mask, show the computed routing:
  - "Routing: SINGLE_PINHOLE (θ_max=48°, f=1240px, w_out=3600px)" or
  - "Routing: CUBEMAP_SPLIT (θ_max=72° exceeds 55° threshold)"

### Step 2.6: Add ERP Reframe mode enhancements
- [ ] Split mode radio: "Cubemap (6 views)" / "Reframe (16 views)" / "prep360 presets"
  - Cubemap and Reframe modes use `erp_reframe.process_equirect_sensor()`
  - prep360 presets mode uses existing `prep360.core.reframer` (current behavior)
- [ ] Split width input (applies to Cubemap/Reframe modes)
  - Default: auto-calculated from `sensor_discovery.recommended_equirect_width()`

### Step 2.7: Wire up conversion workers
- [ ] **Cubeface worker**: Thread that calls `cubeface_processing.process_cubeface_sensor()`
  - Input: Metashape XML path, image dirs, mask dirs, face width, format, layout, force flag
  - Progress: callback → `app.log()` + progress bar
  - Output: cubeface directory tree
  - On completion: auto-populate output path, log summary
- [ ] **Adaptive worker**: Thread that calls `adaptive_undistort.process_sensor_adaptive()`
  - Input: calibration dict, image dir, mask dir, threshold overrides
  - Progress: callback → `app.log()`
  - Output: single-pinhole directory
- [ ] **ERP worker**: Thread that calls `erp_reframe.process_equirect_sensor()`
  - Input: image dirs, mask dirs, split mode, split width
  - Progress: callback → `app.log()`
  - Output: view directory tree
- [ ] All workers follow existing source_tab threading pattern:
  - Snapshot settings before thread start
  - `app.after(0, ...)` for UI updates
  - Cancellation via `app.cancel_flag`
  - Activity recording on completion

### Step 2.8: Preserve backward compatibility
- [ ] Existing Dual-Fisheye Reframe workflow must work identically to current behavior
- [ ] Existing ERP reframe via prep360 presets must work identically
- [ ] No changes to the existing `FisheyeReframer`, `FISHEYE_PRESETS`, or `DualFisheyeCalibration` code
- [ ] New modes are purely additive

---

## Phase 3: Export Tab — Metashape→COLMAP Workflow

**Goal:** Build a new Export tab (replacing Coverage) that packages a Metashape alignment as a COLMAP training scene.

### Step 3.1: Create the Export tab file
- [ ] Create `reconstruction_gui/tabs/export_tab.py`
- [ ] Follow existing tab patterns:
  - `def build_export_tab(app, parent):` entry point
  - Left-column scrollable settings, right-column status/preview
  - `Section` and `CollapsibleSection` from `widgets.py`
  - All widget refs stored on `app` with `export_` prefix
  - Preferences persistence via `app._prefs`

### Step 3.2: Input Section
- [ ] **cameras.xml** file browser
  - Label: "Cameras XML:"
  - On load: trigger sensor discovery (Step 3.3)
- [ ] **pointcloud.ply** file browser (optional — needed for 3D points in COLMAP output)
  - Label: "Point Cloud:"
- [ ] **Output directory** browser
  - Label: "Output:"
  - Auto-suggest: `{cameras_xml_parent}/colmap_export/`

### Step 3.3: Sensor Discovery Section
- [ ] Triggered automatically when cameras.xml is loaded
- [ ] Calls `sensor_discovery.discover_sensors(xml_path)`
- [ ] Displays discovered sensors as **sensor cards** (one per unique sensor):
  - Each card shows:
    - Sensor label (from Metashape)
    - Projection type badge: `equisolid` / `equidistant` / `frame` / `equirectangular`
    - Resolution: `{width}×{height}`
    - Focal length and distortion summary
    - Camera count: "N aligned cameras"
    - Fourier corrections indicator (if present)
  - Each card has:
    - Image directory browser (required)
    - Mask directory browser (optional)
    - Per-sensor routing display: "→ Cubemap (5 faces)" or "→ Single Pinhole" or "→ Passthrough" or "→ ERP Split (16 views)"
    - Face width / output width override (0 = auto)
- [ ] Sensor cards are dynamically generated — number and content depend on the XML

### Step 3.4: Lens Mapping Section
- [ ] **Check Mapping** button
  - Calls `mapping_resolver.resolve_mapping(xml_data, lens_jobs)`
  - Displays structured diagnostics:
    - INFO items (green): confirmations
    - WARNING items (amber): potential issues
    - BLOCKER items (red): must fix before export
  - If heuristic mapping found: "Use Proposed Map" button
- [ ] Manual mapping override text field
  - Format: `lens1=1-50 lens2=51-100`
  - Validated on change via `mapping_resolver.validate_manual_map()`
- [ ] Only visible when multiple equisolid/equidistant sensors share overlapping image directories

### Step 3.5: Export Settings Section (collapsible)
- [ ] Scene normalization checkbox (recenter + rescale camera poses)
- [ ] Keep processing files checkbox (retain intermediate cubefaces/remaps after export)
- [ ] Output format for converted images: PNG / TIFF / JPEG
- [ ] Force reprocess checkbox (ignore processing stamps)

### Step 3.6: Export Actions & Progress
- [ ] **Export** button (large, primary style)
  - Validates all inputs (sensors have image directories, no BLOCKER diagnostics)
  - Snapshots settings
  - Spawns export worker thread
- [ ] **Cancel** button (appears during export)
- [ ] Progress console (real-time log with PROGRESS line parsing)
- [ ] Status section showing:
  - Export phase: "Discovering sensors..." → "Converting sensor 1/3..." → "Writing COLMAP scene..." → "Complete"
  - Output summary: # cameras, # images, # points, all-PINHOLE confirmation

### Step 3.7: Export worker implementation
- [ ] Worker thread orchestrates the full pipeline:
  1. Parse cameras.xml via `colmap_exporter.parse_metashape_cameras_xml()`
  2. For each sensor, based on routing:
     - **Equisolid/equidistant** → `cubeface_processing.process_cubeface_sensor()` or `adaptive_undistort.process_sensor_adaptive()`
     - **Frame** → passthrough (optional undistortion via `adaptive_undistort`)
     - **Equirectangular** → `erp_reframe.process_equirect_sensor()`
  3. Compose poses: `colmap_exporter.face_world_to_camera_pose()` for each virtual pinhole
  4. Write COLMAP scene: `colmap_exporter.write_colmap_training_scene()` or `write_colmap_mixed_scene()`
  5. Generate reports: conversion_report.txt, validation_report.txt, run_summary.txt
- [ ] Progress callbacks from each sub-step → console
- [ ] On completion: log summary, record activity, optionally auto-fill Align tab inputs

### Step 3.8: Right-side status panel
- [ ] Sensor overview: compact visualization of discovered sensors + their routing
- [ ] Export status: phase indicator, timing, completion badge
- [ ] Output summary: camera count, image count, all-PINHOLE badge
- [ ] Model path display (clickable to open in file explorer)

---

## Phase 4: Coverage Tab Removal

**Goal:** Cleanly remove the Coverage tab and its supporting modules.

### Step 4.1: Remove tab registration
- [ ] In `reconstruction_zone.py`: remove `self.tabs.add("Coverage")`
- [ ] In `reconstruction_zone.py`: remove `build_gaps_tab(self, self.tabs.tab("Coverage"))`
- [ ] In `reconstruction_zone.py`: remove `from tabs.gaps_tab import build_gaps_tab`

### Step 4.2: Add Export tab registration
- [ ] In `reconstruction_zone.py`: add `self.tabs.add("Export")` in Coverage's former position
- [ ] In `reconstruction_zone.py`: add `build_export_tab(self, self.tabs.tab("Export"))`
- [ ] In `reconstruction_zone.py`: add `from tabs.export_tab import build_export_tab`

### Step 4.3: Remove Coverage files
- [ ] Delete `reconstruction_gui/tabs/gaps_tab.py` (437 lines)
- [ ] Delete `prep360/core/gap_detector.py` (843 lines)
- [ ] Delete `prep360/core/bridge_extractor.py` (563 lines)
- [ ] Remove any gap_detector / bridge_extractor imports from `prep360/core/__init__.py`
- [ ] Remove any gap_detector / bridge_extractor references from `prep360/cli.py` if present

### Step 4.4: Clean up cross-references
- [ ] Search for `"Coverage"` string references across the codebase
- [ ] Search for `gap_detector`, `bridge_extractor`, `build_gaps_tab` references
- [ ] Search for `gaps_tab` imports
- [ ] Remove or update any documentation references (CLAUDE.md architecture diagram, docstrings)

---

## Phase 5: Cross-Tab Integration

**Goal:** Wire up handoffs between the new/expanded capabilities and existing tabs.

### Step 5.1: Export → Align handoff
- [ ] Add "Send to Align" button on Export tab (appears after successful export)
  - Auto-populates Align tab's image directory with export output's `images/` path
  - Auto-populates Align tab's mask directory with export output's `masks/` path
  - Switches to Align tab
- [ ] This enables the workflow: "I exported from Metashape, now I want to refine with additional COLMAP BA"

### Step 5.2: Extract → Export handoff
- [ ] After cubeface/adaptive conversion in Extract, offer "Send to Export" if the user has a Metashape XML
  - Pre-populates Export tab's sensor image directories with conversion output
- [ ] This enables: "I converted my fisheye images, now I want to package the Metashape alignment as COLMAP"

### Step 5.3: Extract → Mask handoff (existing, verify still works)
- [ ] Cubeface output (5 pinhole faces per image) should be maskable in the Mask tab
  - Verify the Mask tab handles the Station layout directory structure
  - Verify mask naming conventions are compatible

### Step 5.4: Preference persistence for new controls
- [ ] Add all new `export_*` prefs keys to `_save_prefs()` / `_restore_prefs()`
- [ ] Add all new `fisheye_metashape_*`, conversion mode, cubeface, adaptive prefs
- [ ] Verify prefs roundtrip (save → restart → restore)

### Step 5.5: Update CLAUDE.md architecture diagram
- [ ] Update the architecture tree in project CLAUDE.md:
  - Add new files under `prep360/core/`
  - Update tab list (Coverage → Export)
  - Add F2C-origin modules to the key abstractions section
  - Update masking pipeline data flow if needed

---

## Phase 6: Cleanup & Validation

### Step 6.1: Import audit
- [ ] `grep -r "from gui\." prep360/core/` — must return zero hits (no F2C gui imports surviving)
- [ ] `grep -r "AM_ImageAndMask_to_cubemap_v4" prep360/core/` — must return zero hits (all updated to cubeface_engine)
- [ ] `grep -r "gap_detector\|bridge_extractor\|gaps_tab" reconstruction_gui/` — must return zero hits

### Step 6.2: Test suite
- [ ] All imported F2C tests pass with updated imports
- [ ] Existing reconstruction-zone tests still pass (if any)
- [ ] Manual smoke test: launch GUI, verify all 7 tabs load without errors
- [ ] Manual smoke test: Extract tab → each conversion mode → verify controls show/hide correctly
- [ ] Manual smoke test: Export tab → load a cameras.xml → verify sensor cards populate

### Step 6.3: Functional validation
- [ ] End-to-end: Load Metashape XML in Extract → Cubeface conversion → verify output images
- [ ] End-to-end: Load cameras.xml in Export → export COLMAP scene → verify cameras.txt/images.txt
- [ ] End-to-end: Export → Send to Align → verify Align tab populates correctly
- [ ] Regression: Existing Osmo 360 dual-fisheye workflow in Extract → verify unchanged behavior
- [ ] Regression: Existing ERP reframe workflow in Extract → verify unchanged behavior

### Step 6.4: Documentation update
- [ ] Update `reconstruction_gui/docs/` if present (per-tab guides, QUICKSTART)
- [ ] Update project CLAUDE.md (architecture, commands, key abstractions)
- [ ] Update `docs/` (project-level docs)

---

## Dependency Map

### F2C internal dependency graph (→ = imports from)

```
cubeface_engine (standalone — numpy, cv2, scipy, argparse)
  ↑
  ├── solid_angle → cubeface_engine
  ├── corrected_rays → cubeface_engine, fourier_corrections
  ├── adaptive_undistort → cubeface_engine, corrected_rays
  └── cubeface_processing → cubeface_engine, fourier_corrections, corrected_rays, processing_stamp

fourier_corrections (standalone — hashlib, xml, numpy)
  ↑
  ├── corrected_rays → fourier_corrections
  ├── sensor_discovery → fourier_corrections
  ├── routing → fourier_corrections
  └── cubeface_processing → fourier_corrections

scene_manifest (standalone — json, dataclasses)
  ↑
  └── routing → scene_manifest

processing_stamp (standalone — hashlib, json, pathlib)
  ↑
  └── cubeface_processing → processing_stamp

erp_reframe (standalone — numpy, cv2)

routing → adaptive_undistort, fourier_corrections, scene_manifest

mapping_resolver → colmap_exporter (cross-reference for validation)

colmap_exporter → erp_reframe (optional registration)
```

### External dependencies

| Package | Required by | Already in RZ? |
|---|---|---|
| numpy | All modules | Yes |
| opencv-python | cubeface_engine, adaptive_undistort, erp_reframe, cubeface_processing | Yes |
| scipy (KDTree) | adaptive_undistort, cubeface_engine (RBF) | Needs verification |

---

## Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| **cubeface_engine is 2,332 lines of stable code** — refactoring could introduce regressions | High | Copy as-is, rename only. Do not refactor internals. Treat as a black-box engine. |
| **colmap_exporter is 6,682 lines** — largest single module | Medium | Copy as-is initially. Refactoring is a future concern, not part of this integration. |
| **Import path changes across 12+ modules** | Medium | Systematic find-replace with test verification at each step. Phase 1 is entirely about this. |
| **Extract tab source_tab.py is already 3,773 lines** — adding conversion modes increases complexity | Medium | New controls are in collapsible sub-sections. Mode selector hides irrelevant controls. Consider extracting the 360° section to its own module if it exceeds ~500 new lines. |
| **Mixed-sensor COLMAP export is complex (6,682 lines of exporter)** | Medium | Phase 3 wraps it as a worker — the exporter's internal complexity is encapsulated. |
| **Processing stamps / cache invalidation** assumes specific directory layouts | Low | Stamps are path-relative; verify they work under RZ's output directory conventions. |
| **mapping_resolver has a circular-ish dependency on colmap_exporter** | Low | Only for validation of manual mapping specs. Can be stubbed during Phase 1 and wired in Phase 3. |
| **F2C tests depend on fixture files** | Low | Copy fixtures alongside tests. Verify paths in test assertions. |

---

## Open Questions

1. **scipy dependency**: Is scipy already installed in the RZ environment? The cubeface engine and adaptive undistort need `scipy.spatial.KDTree`. If not, it becomes a new required dependency.

2. **Adjust tab interaction**: Should cubeface/adaptive conversion output go through the Adjust tab for color correction before Export? This would mean Extract → Adjust → Export as a workflow. Currently the plan treats them independently.

3. **Naming — "Export" tab**: Is "Export" the right name? Alternatives: "Convert", "Scene", "Training", "Package". "Export" is clear but generic. "Convert" emphasizes the transformation. "Scene Export" is more specific but longer.

4. **Gap detection — relocate or remove?**: The plan removes gap_detector entirely. If gap analysis is still valuable, it could move to the Align tab (analyze an alignment's coverage) or become a CLI-only tool. The bridge_extractor is tightly coupled to the Extract tab's video state and would be harder to relocate.

5. **F2C's browser demo**: The Vite+Three.js interactive preview is not included in this plan. It could be valuable as a standalone validation tool but doesn't fit naturally in the CustomTkinter GUI.

6. **cubeface_engine CLI**: The original script has argparse CLI support. Should `python -m prep360 cubeface ...` be added as a CLI subcommand in prep360, or is GUI-only sufficient?

7. **F2C test fixtures**: Some test fixtures include real Metashape XML snippets and calibration data. Verify these don't contain sensitive project-specific data before committing to reconstruction-zone.

---

## Line Count Summary

### Added to prep360/core/
| Module | Lines |
|---|---|
| cubeface_engine.py | 2,332 |
| colmap_exporter.py | 6,682 |
| cubeface_processing.py | 634 |
| adaptive_undistort.py | 845 |
| erp_reframe.py | 530 |
| conversion_routing.py | 295 |
| sensor_discovery.py | 390 |
| mapping_resolver.py | 1,656 |
| fourier_corrections.py | 344 |
| corrected_rays.py | 150 |
| scene_manifest.py | 295 |
| processing_stamp.py | 143 |
| solid_angle.py | 133 |
| **Subtotal** | **~14,429** |

### Added to reconstruction_gui/tabs/
| Module | Lines (estimated) |
|---|---|
| export_tab.py (new) | ~800–1,200 |
| source_tab.py (modifications) | ~300–500 added |
| **Subtotal** | **~1,100–1,700** |

### Removed
| Module | Lines |
|---|---|
| gaps_tab.py | 437 |
| gap_detector.py | 843 |
| bridge_extractor.py | 563 |
| **Subtotal** | **~1,843** |

### Net change
- **prep360/core/**: +14,429 lines (library modules, copied not written)
- **reconstruction_gui/**: +~600–900 net (new Export tab + Extract mods – Coverage removal)
- **tests/**: +~3,500 lines (copied F2C tests)
