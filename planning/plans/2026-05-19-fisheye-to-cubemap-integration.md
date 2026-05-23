# Fisheye-to-Cubemap Integration into Reconstruction Zone

> **Branch:** `release/v1`
> **Source:** `D:\Projects\Fisheye-to-Cubemap` (~18,774 lines across 14 modules)
> **Target:** `D:\Projects\reconstruction-zone`
> **Visual reference:** `pencil-new.pen` (Pencil design file, node `wDhnY`)

## Goal

Integrate the Fisheye-to-Cubemap cubeface conversion pipeline into Reconstruction Zone's Extract tab, inside the 360 Processing > Reframing > Fisheye subsection. The conversion is driven entirely by a Metashape `cameras.xml` file — no separate lens calibration XMLs.

This plan covers the cubeface conversion integration only. The Metashape→COLMAP scene export (Export tab) is a separate future phase.

---

## Scope

**In scope:**
- Copy F2C library modules into `prep360/core/`
- Add a Fisheye subsection inside Reframing with cubeface conversion UI
- Add an Equirectangular/Spherical subsection inside Reframing (placeholder for existing ERP workflow)
- Wire up cubeface conversion worker driven by `cameras.xml`

**Out of scope:**
- Adaptive pinhole mode (future — may become a tab within the Fisheye subsection)
- ERP cubemap/16-view modes (future)
- Export tab / COLMAP scene packaging
- Sensor discovery, mapping resolver, COLMAP exporter
- Coverage tab removal
- Cross-tab handoffs

---

## Part 1: Module Foundation

### 1.1 Dependency Graph

```
Layer 0 — Standalone (no F2C internal deps)
  processing_stamp.py    (143 lines)  — hashlib, json, pathlib
  scene_manifest.py      (295 lines)  — json, dataclasses, pathlib

Layer 1 — Depends on cubeface_engine only
  fourier_corrections.py (344 lines)  — hashlib, xml, numpy
  solid_angle.py         (133 lines)  → cubeface_engine

Layer 2 — Depends on Layer 1
  corrected_rays.py      (150 lines)  → fourier_corrections, cubeface_engine

Layer 3 — Orchestration
  cubeface_processing.py (634 lines)  → cubeface_engine, fourier_corrections,
                                        corrected_rays, processing_stamp

The cubeface engine:
  cubeface_engine.py     (2332 lines) — numpy, cv2, scipy (standalone)
```

Only the modules needed for cubeface conversion are copied in this phase. Modules for adaptive pinhole (`adaptive_undistort.py`, `conversion_routing.py`), ERP reframe (`erp_reframe.py`), and the COLMAP exporter are deferred.


### 1.2 Files to Copy

| F2C Source | RZ Destination | Import Changes |
|---|---|---|
| `AM_ImageAndMask_to_cubemap_v4.py` | `prep360/core/cubeface_engine.py` | None — standalone. Copy as-is, rename only. |
| `gui/processing_stamp.py` | `prep360/core/processing_stamp.py` | None — stdlib only |
| `gui/scene_manifest.py` | `prep360/core/scene_manifest.py` | None — stdlib only |
| `gui/fourier_corrections.py` | `prep360/core/fourier_corrections.py` | None — stdlib + numpy only |
| `gui/solid_angle.py` | `prep360/core/solid_angle.py` | `AM_ImageAndMask_to_cubemap_v4` → `prep360.core.cubeface_engine` |
| `gui/corrected_rays.py` | `prep360/core/corrected_rays.py` | `gui.fourier_corrections` → `prep360.core.fourier_corrections`; `AM_ImageAndMask_to_cubemap_v4` → `prep360.core.cubeface_engine` |
| `gui/cubeface_processing.py` | `prep360/core/cubeface_processing.py` | `AM_ImageAndMask_to_cubemap_v4` → `prep360.core.cubeface_engine`; `gui.fourier_corrections` → `prep360.core.fourier_corrections`; `gui.corrected_rays` → `prep360.core.corrected_rays`; `gui.processing_stamp` → `prep360.core.processing_stamp` |
| `gui/sensor_discovery.py` | `prep360/core/sensor_discovery.py` | `gui.fourier_corrections` → `prep360.core.fourier_corrections` |

**8 modules total.** The cubeface engine is copied as a stable black box — do not refactor its internals.

### 1.3 SystemExit Audit

F2C modules use `raise SystemExit(...)` for CLI error handling. In a GUI context, SystemExit kills the application. After copying, replace every `raise SystemExit(...)` with `raise ValueError(...)` or `raise RuntimeError(...)`.

```bash
grep -n "SystemExit" prep360/core/cubeface_engine.py prep360/core/cubeface_processing.py
```

### 1.4 Dependency Check: scipy

The cubeface engine uses `scipy.spatial.KDTree` for RBF remapping:

```bash
python -c "from scipy.spatial import KDTree; print('scipy OK')"
```

If not installed, add to requirements. Only new external dependency.

### 1.5 prep360/core/__init__.py

Do NOT add these modules to eager imports. They are imported lazily by the GUI code that needs them. The cubeface engine alone is 2,332 lines — importing it at package level would slow every `from prep360.core import ...`.

### 1.6 Import Verification

After copying all 8 modules:

```bash
python -c "
from prep360.core.cubeface_engine import compute_rays
from prep360.core.fourier_corrections import FourierCorrections
from prep360.core.corrected_rays import compute_rays_with_corrections
from prep360.core.solid_angle import compute_optimal_width
from prep360.core.processing_stamp import compute_stamp_digest
from prep360.core.scene_manifest import RoutingDecision
from prep360.core.cubeface_processing import process_cubeface_sensor
from prep360.core.sensor_discovery import discover_sensors
print('All imports OK')
"
```

### 1.7 Grep Audit

```bash
# No stale F2C gui imports
grep -rn "from gui\." prep360/core/cubeface_engine.py prep360/core/fourier_corrections.py prep360/core/corrected_rays.py prep360/core/solid_angle.py prep360/core/processing_stamp.py prep360/core/scene_manifest.py prep360/core/cubeface_processing.py prep360/core/sensor_discovery.py
# Should return zero results

# No old script name references
grep -rn "AM_ImageAndMask_to_cubemap_v4" prep360/core/
# Should return zero results
```

---

## Part 2: GUI — Fisheye Subsection

### 2.1 Current Reframing Structure (release/v1, source_tab.py:2291)

```
CollapsibleSection "Reframing"
  subtitle="extract pinhole perspectives from a 360 file or existing ERP frames"
  ├─ Video:   .osv/.360/.insv file
  ├─ Frames:  existing ERP frames folder
  ├─ Masks:   masks folder
  ├─ Output:  perspective output dir
  ├─ CollapsibleSection "Custom Calibration"
  ├─ Preset dropdown (4 Osmo 360 presets)
  ├─ Preset description, Crop, Quality, Interval
  ├─ Station dirs checkbox
  └─ "Extract & Reframe" button
```

Everything lives in one flat section. Dual-fisheye vs ERP is auto-detected at runtime by filename patterns.

### 2.2 Target Structure

The Fisheye subsection contains two tabs: **Metashape** and **Standard**.

```
CollapsibleSection "Reframing"
  subtitle="convert 360° or fisheye to pinhole perspectives"
  │
  ├─ CollapsibleSection "Fisheye"
  │    subtitle="dual-fisheye → pinhole perspectives"
  │    │
  │    ├─ Tab: "Metashape" | "Standard"
  │    │
  │    ├─ ── Metashape tab ──────────────────────────────────────────
  │    │    Calibration-aware cubeface conversion using cameras.xml.
  │    │    Produces 5 cubefaces per image via RBF-interpolated remapping.
  │    │    │
  │    │    ├─ Input:    cameras.xml file picker      (app.fisheye_xml_entry)
  │    │    ├─ Output:   pinhole output directory     (app.fisheye_output_entry)
  │    │    ├─ Images:   fisheye images directory     (app.fisheye_images_entry)
  │    │    ├─ Masks:    masks directory (optional)   (app.fisheye_masks_entry)
  │    │    │
  │    │    ├─ Face width:  entry (default 0 = auto)  (app.fisheye_face_width_var)
  │    │    │   Format:     radio (png/tiff/jpg)      (app.fisheye_format_var)
  │    │    ├─ Force reprocess checkbox               (app.fisheye_force_var)
  │    │    ├─ Station / Rig radio                    (app.fisheye_layout_var)
  │    │    │
  │    │    └─ "Convert to Cubefaces" button          (app.fisheye_convert_btn)
  │    │
  │    └─ ── Standard tab ──────────────────────────────────────────
  │         Preset-driven perspective reframing using cv2.fisheye.
  │         No Metashape XML required — uses built-in equidistant
  │         calibration. No video input.
  │         │
  │         ├─ Images:  fisheye frames directory         (app.fisheye_std_images_entry)
  │         ├─ Masks:   masks directory (optional)       (app.fisheye_std_masks_entry)
  │         ├─ Output:  perspective output directory     (app.fisheye_std_output_entry)
  │         │
  │         ├─ Preset:  dropdown                         (app.fisheye_std_preset_var)
  │         │    Default: "Pinhole 90° — 16 views (8/lens)"
  │         ├─ Crop / Quality                            (app.fisheye_std_crop_var, etc.)
  │         ├─ Station / Rig radio                       (app.fisheye_std_layout_var)
  │         │
  │         └─ "Reframe" button                          (app.fisheye_std_reframe_btn)
  │
  └─ CollapsibleSection "Equirectangular / Spherical"
       subtitle="split 360° panoramas into perspectives"
       (contains existing ERP reframe controls, moved here from the
        parent Reframing section — no new functionality)
```

**Metashape tab:** Driven by `cameras.xml`. Uses the F2C cubeface engine (`process_cubeface_sensor`) which produces 5 cubeface images per fisheye image (+Z, ±X, ±Y) via Metashape-calibrated RBF remapping. Supports equisolid, equidistant, and pinhole projection models. Supports Fourier corrections. Face width auto-calculated from calibration.

**Standard tab:** Uses the existing `FisheyeReframer` engine (`prep360/core/fisheye_reframer.py`) with `cv2.fisheye.initUndistortRectifyMap`. No Metashape XML required — uses built-in Osmo 360 equidistant calibration. Produces 8 views per lens (16 total for dual-fisheye) at 90° FOV using a ring layout. Takes fisheye image directories as input (no video input, no interval, no custom calibration). Preset dropdown for future preset additions. Station/Rig output layout choice.

### 2.3 The 8-View Preset

A new preset `"osv-pinhole-f90-dual-16"` is added to `FISHEYE_PRESETS` in `prep360/core/fisheye_reframer.py`. It uses the ring layout from the lichtfeld-360-plugin's `FISHEYE_PINHOLE_PRESET`:

```python
# 8 views per lens at 90° FOV — ring layout for 3DGS training
# Optimized geometry from fisheye-perspective-planner.html
_PINHOLE_VIEWS = [
    ("ctr_hi",  0,      -30),    # top center
    ("ring_l",  -55,    0),      # left
    ("ring_ll", -35.53, 50),     # lower-left
    ("ring_lr", 35.53,  50),     # lower-right
    ("ring_r",  55,     0),      # right
    ("ring_ur", 35.53,  -50),    # upper-right
    ("ring_ul", -35.53, -50),    # upper-left
    ("ctr_lo",  0,      30),     # bottom center
]
```

Default crop size: 1920 (vs 1600 for existing presets). The view geometry was designed for optimal 3DGS training coverage — a central ring of 6 views at the horizon with upper and lower center views.

This preset uses the same `FisheyeViewConfig` / `FisheyeReframer` / `batch_extract` API as the existing 4 presets. No engine changes needed — just a new entry in the `FISHEYE_PRESETS` dict.

### 2.4 Widget References

All existing fisheye widget references are contained within `source_tab.py` — no external files reference them. Confirmed via grep. The existing widget names (`app.fisheye_osv_entry`, `app.fisheye_reframe_frames_entry`, etc.) are preserved as-is in the Standard tab. New Metashape tab widgets use new names (`app.fisheye_xml_entry`, etc.).

### 2.5 Preference Keys

New keys for the Metashape tab:

```
fisheye_active_tab            # "metashape" | "standard"
fisheye_xml_path              # str — cameras.xml location
fisheye_cubeface_output_dir   # str — pinhole output directory
fisheye_images_dir            # str — fisheye images directory
fisheye_masks_dir             # str — masks directory
fisheye_face_width            # str ("0" = auto, computed from calibration)
fisheye_format                # "png" | "tiff" | "jpg"
fisheye_layout                # "station" | "rig"
fisheye_force_reprocess       # bool
```

Existing Standard tab preference keys are unchanged (already persisted by the current code).

---

## Part 3: Worker Function

### 3.1 Cubeface Conversion Worker

The `process_cubeface_sensor` API accepts a `calibration_xml` path. This is the `cameras.xml` — `get_calibration_with_corrections()` in `fourier_corrections.py` parses the sensor calibration from it.

```python
def _run_cubeface_convert(app):
    """Validate inputs and launch cubeface conversion thread."""
    xml_path = app.fisheye_xml_entry.get().strip()
    output_dir = app.fisheye_output_entry.get().strip()
    images_dir = app.fisheye_images_entry.get().strip()
    masks_dir = app.fisheye_masks_entry.get().strip()

    # Validation
    if not xml_path or not Path(xml_path).is_file():
        app.log("Error: Select a cameras.xml file")
        return
    if not output_dir:
        app.log("Error: Select a pinhole output directory")
        return
    if not images_dir or not Path(images_dir).is_dir():
        app.log("Error: Select a fisheye images directory")
        return
    if masks_dir and not Path(masks_dir).is_dir():
        app.log(f"Error: Masks directory not found: {masks_dir}")
        return

    face_width = int(app.fisheye_face_width_var.get() or "0")
    output_format = app.fisheye_format_var.get()
    force = app.fisheye_force_var.get()

    app._start_operation(app.fisheye_convert_btn, app.fisheye_stop_btn)
    threading.Thread(
        target=_cubeface_worker,
        args=(app, xml_path, images_dir, masks_dir or None,
              output_dir, face_width, output_format, force),
        daemon=True,
    ).start()


def _cubeface_worker(app, xml_path, images_dir, masks_dir,
                     output_dir, face_width, output_format, force):
    """Thread target: runs process_cubeface_sensor."""
    try:
        from prep360.core.cubeface_processing import process_cubeface_sensor

        app.log(f"Converting fisheye → cubefaces")
        app.log(f"  Input XML:  {xml_path}")
        app.log(f"  Images:     {images_dir}")
        if masks_dir:
            app.log(f"  Masks:      {masks_dir}")
        app.log(f"  Output:     {output_dir}")
        app.log(f"  Face width: {face_width}, Format: {output_format}")

        result = process_cubeface_sensor(
            calibration_xml=Path(xml_path),
            image_dirs=[Path(images_dir)],
            output_dir=Path(output_dir),
            face_width=face_width,
            mask_dirs=[Path(masks_dir)] if masks_dir else None,
            output_format=output_format,
            force=force,
            progress_callback=lambda msg: app.log(msg),
        )

        app.log(f"\nConversion complete")
        app.log(f"  Processed: {result['processed_count']}")
        app.log(f"  Skipped:   {result['skipped_count']}")
        app.log(f"  Face width: {result['face_width']}")
        app.log(f"  Output:    {result['output_dir']}")

        app.record_activity(
            operation="cubeface_convert",
            input_path=images_dir,
            output_path=output_dir,
            details={
                "processed": result["processed_count"],
                "skipped": result["skipped_count"],
                "face_width": result["face_width"],
            },
        )
    except Exception as e:
        import traceback
        app.log(f"Cubeface conversion error: {e}")
        app.log(traceback.format_exc())
    finally:
        app.after(0, lambda: app._stop_operation(
            app.fisheye_convert_btn, app.fisheye_stop_btn))
```

### 3.2 Stop Button

The Fisheye subsection gets its own stop button (`app.fisheye_stop_btn`), separate from the Split Lenses stop button. Both use `app.cancel_flag` for cancellation (the existing shared mechanism), but have independent button show/hide state.

### 3.3 Threading Pattern

Follows the existing source_tab pattern exactly:
1. Validate inputs in main thread
2. `app._start_operation(action_btn, stop_btn)` — disables action, shows stop
3. `threading.Thread(target=worker, daemon=True).start()`
4. Worker calls `app.log(msg)` for progress
5. Worker checks `app.cancel_flag.is_set()` for cancellation
6. `finally:` calls `app.after(0, lambda: app._stop_operation(...))`

---

## Part 4: Existing Controls — What Changes

The Standard tab is a **simplified version** of the existing Reframing section, not a direct relocation. The following are dropped:
- Video input (.osv/.360/.insv) and interval slider — Standard tab takes image directories, not video
- Custom Calibration collapsible section — Standard uses built-in calibration only
- Preset description multi-line label — unnecessary with a simpler preset list

The following are kept:
- Images/Masks/Output directory inputs (renamed from Frames/Masks/Output)
- Preset dropdown (with the new 8-view preset as default, existing presets available)
- Crop size and quality controls
- Station/Rig output layout radio (replaces the Station dirs checkbox)
- Action button (renamed from "Extract & Reframe" to "Reframe")

The existing video-based extraction workflow (OSV → extract frames → reframe) remains in the Extract tab's main extraction sections above 360 Processing. It is not part of the Fisheye subsection.

The Equirectangular/Spherical subsection is created as a collapsed placeholder with a subtitle, ready for future ERP-specific controls.

---

## Part 5: Implementation Steps

```
Step 1:  Copy cubeface_engine.py → prep360/core/cubeface_engine.py
         Fix SystemExit → ValueError/RuntimeError
         Verify: python -c "from prep360.core.cubeface_engine import compute_rays"

Step 2:  Copy Layer 0 (processing_stamp.py, scene_manifest.py)
         Verify imports

Step 3:  Copy Layer 1 (fourier_corrections.py, solid_angle.py)
         Fix solid_angle.py imports
         Verify imports

Step 4:  Copy Layer 2 (corrected_rays.py)
         Fix imports
         Verify import

Step 5:  Copy cubeface_processing.py
         Fix imports, SystemExit
         Verify: python -c "from prep360.core.cubeface_processing import process_cubeface_sensor"

Step 6:  Copy sensor_discovery.py
         Fix imports
         Verify: python -c "from prep360.core.sensor_discovery import discover_sensors"

Step 7:  Run full import verification + grep audit (Section 1.6, 1.7)

Step 8:  Add 8-view preset to prep360/core/fisheye_reframer.py
         - Add _PINHOLE_VIEWS list and "osv-pinhole-f90-dual-16" preset to FISHEYE_PRESETS
         - Verify: python -c "from prep360.core.fisheye_reframer import FISHEYE_PRESETS; print('osv-pinhole-f90-dual-16' in FISHEYE_PRESETS)"

Step 9:  Restructure source_tab.py
         - Add CollapsibleSection "Fisheye" inside Reframing
         - Add tab strip: "Metashape" | "Standard"
         - Metashape tab: build 4 input rows (Input/Output/Images/Masks)
           + settings (face width, format, force, station/rig)
           + "Convert to Cubefaces" button + stop button
         - Standard tab: build simplified reframe UI
           (Images, Masks, Output, Preset dropdown, Crop, Quality,
            Station/Rig radio, "Reframe" button)
           No video input, no interval, no custom calibration
         - Add CollapsibleSection "Equirectangular / Spherical" (collapsed placeholder)
         - Tab switching shows/hides the correct content frame

Step 10: Wire cubeface worker
         - _run_cubeface_convert (validation + thread launch)
         - _cubeface_worker (calls process_cubeface_sensor)
         - Stop button wired to app.cancel_flag

Step 11: Add preference persistence
         - Save/restore for all new fisheye_* pref keys
         - fisheye_active_tab persists which tab was last selected
         - Add to _save_prefs / _restore_prefs in source_tab

Step 12: Smoke test
         - Launch GUI — all tabs load without error
         - Fisheye subsection: Metashape and Standard tabs switch correctly
         - Standard tab: existing dual-fisheye reframe works (regression)
         - Standard tab: new 8-view preset appears in dropdown and works
         - Metashape tab: load cameras.xml + images dir + output → Convert
         - ERP subsection: collapsed placeholder visible
```

---

## Part 6: Test Plan

### 6.1 Import Tests
Run the verification battery from Section 1.6 after Step 7.

### 6.2 Grep Audits
Run the stale import checks from Section 1.7 after Step 7.

### 6.3 GUI Smoke Tests
1. Launch: `python reconstruction_gui/reconstruction_zone.py` — no import errors
2. Extract tab → 360 Processing → Reframing → Fisheye subsection visible
3. Fisheye: "Metashape" and "Standard" tabs visible, switching works
4. Metashape tab: Input/Output/Images/Masks fields present with browse buttons
5. Metashape tab: Face width, format, force reprocess, station/rig present
6. Metashape tab: "Convert to Cubefaces" button present
7. Standard tab: Images/Masks/Output fields, Preset dropdown, Crop/Quality, Station/Rig present
8. Standard tab: default preset is "Pinhole 90° — 16 views (8/lens)"
9. Standard tab: reframe produces 16 perspective crops (8 per lens) from fisheye frame pair
10. ERP subsection: collapsed placeholder visible

### 6.4 Functional Test (requires test data)
- Provide `cameras.xml` + fisheye images directory
- Set output directory
- Click "Convert to Cubefaces"
- Verify: 5 cubeface images produced per input image
- Verify: masks converted if mask directory provided
- Verify: processing stamp written
- Verify: skip works on re-run (stamps match)
- Verify: force reprocess overrides stamps

---

## Notes

- The `cameras.xml` is a full Metashape alignment export. `cubeface_processing.process_cubeface_sensor()` accepts it via `calibration_xml` parameter — `get_calibration_with_corrections()` in `fourier_corrections.py` extracts the sensor calibration from within it.

- Face width defaults to 0 (auto). The engine auto-calculates the optimal width from the lens calibration in the cameras.xml via `solid_angle.compute_optimal_width()`. The user can override with a specific pixel value if needed.

- The Fisheye subsection has two tabs: **Metashape** (calibration-driven cubeface conversion, 5 faces per image) and **Standard** (preset-driven perspective reframing, 8 views per lens). Both produce pinhole outputs from fisheye inputs, but via different engines and for different use cases. Metashape requires a cameras.xml and uses the F2C cubeface engine with RBF remapping. Standard uses the existing `FisheyeReframer` with `cv2.fisheye`, takes image directories directly (no video input), and uses the built-in equidistant calibration.

- The 8-view preset geometry comes from the lichtfeld-360-plugin (`core/fisheye_reframer.py`), which was itself ported from reconstruction-zone's `prep360/core/fisheye_reframer.py`. The view layout (`ctr_hi, ring_l, ring_ll, ring_lr, ring_r, ring_ur, ring_ul, ctr_lo`) was designed via `fisheye-perspective-planner.html` for optimal 3DGS training coverage. Default crop size is 1920 (vs 1600 for existing presets).
