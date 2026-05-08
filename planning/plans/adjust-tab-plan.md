# Adjust Tab — Feature Plan

## Overview

A full-featured image adjustment and colour correction tab for Reconstruction Zone. Sits immediately after Projects in the tab order (**Projects → Adjust → Extract → Mask → Review → Align → Coverage**). Handles video files, RAW development (Fuji RAF, iPhone ProRAW DNG, Nikon NEF, Canon CR2/CR3, Sony ARW), and standard format adjustment (JPG, PNG, TIFF, BMP). All adjustments are non-destructive and batch-applicable.

---

## Environment Setup

### Package Changes Required

| Action | Package | Version | Reason |
|--------|---------|---------|--------|
| Uninstall | `opencv-python` | 4.13.0 | Conflicts with contrib; contrib includes everything |
| Upgrade | `opencv-contrib-python` | 4.13.0.92 | Provides `cv2.xphoto` (white balance), `cv2.mcc` (chart detection) |
| Install | `rawpy` | 0.26.1 | RAW development. Has cp314-win_amd64 wheel. Supports Fuji X-Trans, iPhone ProRAW |
| Install | `scikit-image` | 0.26.0 | Histogram matching, wavelet denoise, advanced exposure. Has 3.14 wheels |
| Optional | `colour-checker-detection` | 0.2.3 | Auto chart detection (segmentation method). 50MB. No torch on 3.14 |

### Already Available (system-level, not yet in project requirements.txt)

| Package | Version | Provides | Note |
|---------|---------|----------|------|
| `colour-science` | 0.4.6 | CCT↔xy conversion, colour correction matrices (Finlayson2015/Cheung2004/Vandermonde), ColorChecker reference data, chromatic adaptation, LAB/XYZ/sRGB conversions | Installed on system but NOT imported anywhere in this project. Must be added to requirements.txt when chart correction ships. |
| `opencv-contrib-python` | (after upgrade) | CLAHE, bilateral filter, NL-means denoise, edge-preserving filter, tone mapping (Reinhard/Drago/Mantiuk), exposure fusion (Mertens), LUT, colour space conversions, `cv2.transform` (3x3 CCM), `cv2.xphoto` white balance (gray-world, learning-based, simple), `cv2.mcc` chart detection |
| `scipy` | installed | PchipInterpolator for tone curves |
| `prep360/core/lut.py` | existing | .cube 3D LUT loading + trilinear interpolation application |

### Verified Compatibility

- `rawpy` 0.26.1: cp314-win_amd64 wheel exists on PyPI
- `scikit-image` 0.26.0: Python 3.14 wheels (including free-threaded)
- `opencv-contrib-python` 4.13.0.92: contains `cv2/mcc/__init__.pyi` and `cv2/xphoto/__init__.pyi` (confirmed from wheel contents)
- `colour-checker-detection` 0.2.3: Python >=3.11,<3.15. `ultralytics` is optional extra (and restricted to <3.14 anyway). Core segmentation works without torch.
- LibRaw (underlying rawpy): Fuji X-Pro1/2/3, X-T1–5, X-H1/2, X-E1–4, X100 series all listed. iPhone 12–15 Pro via DNG.

---

## Feature Inventory

### Group 1: Tone / Exposure

| Feature | Algorithm | Source | Verified |
|---------|-----------|--------|----------|
| Exposure (EV stops) | `output = input * 2^ev` | Image Space `adjust_exposure` | Yes — 3 lines numpy |
| Brightness / Contrast | `cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)` | OpenCV core | Yes — `hasattr` confirmed |
| Shadows recovery | Exponential falloff on L-channel (LAB), `shadows_scaled * (falloff^(1-L) - 1)` | Image Space `adjust_shadows_highlights_combined` | Yes — ported to pure cv2, 15 lines, tested |
| Highlights recovery | Same algorithm applied to highlight end | Image Space (same function) | Yes — same |
| Gamma | `np.power(img, gamma)` via LUT | OpenCV `cv2.LUT` | Yes |
| Levels (black/white point) | Rescale intensity to [black, white] | `skimage.exposure.rescale_intensity` or manual `np.clip` + rescale | Yes — trivial |
| Contrast (S-curve) | Sigmoid on L-channel | Tested: 6 lines cv2 | Yes |
| CLAHE | `cv2.createCLAHE(clipLimit, tileGridSize)` | OpenCV core | Yes — `hasattr` confirmed, 0.03s on 12MP |
| Tone curves (spline) | User control points → PchipInterpolator → 256-entry LUT → `cv2.LUT` on L-channel | scipy.interpolate + cv2 | Yes — tested, 8 lines |
| Tone mapping (HDR) | Reinhard, Drago, Mantiuk | OpenCV `cv2.createTonemapReinhard/Drago/Mantiuk` | Yes — `hasattr` confirmed |
| ACES Filmic | `(x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14)` | Standard formula, 1 line numpy | Yes — well-known |
| Hable/Uncharted 2 | Standard piecewise formula | ~5 lines numpy | Not tested but formula is public |

### Group 2: Colour

| Feature | Algorithm | Source | Verified |
|---------|-----------|--------|----------|
| White balance (temperature + tint) | RGB multipliers from blackbody approximation or `colour.CCT_to_xy` | Image Space `adjust_white_balance` / `colour-science` | Yes — both approaches tested |
| White balance (eyedropper) | Sample neutral point, compute R/B ratio, estimate CCT | Image Space `sample_white_balance_from_point` | Yes — 20 lines |
| White balance (auto — gray world) | `cv2.xphoto.createGrayworldWB()` | OpenCV contrib (after fix) | Confirmed in wheel |
| White balance (auto — learning-based) | `cv2.xphoto.createLearningBasedWB()` | OpenCV contrib (after fix) | Confirmed in wheel |
| Saturation | Multiply S-channel in HSV | 4 lines cv2 | Yes — tested |
| Vibrance | Selective saturation: boost proportional to `(1 - current_S)` | 6 lines cv2 | Yes — tested |
| HSL per-hue adjustment | Circular hue distance mask in HLS, adjust S/L per band | 10 lines cv2 | Yes — tested |
| Colour correction (chart-based) | Detect chart → extract patches → `colour.matrix_colour_correction(measured, reference, 'Finlayson 2015', degree=2)` → apply to batch | `colour-science` + (`cv2.mcc` or `colour-checker-detection` or manual selection) | Yes — full pipeline tested. Degree=2 gives 3x6 matrix, mean error 0.034 |
| Colour matching (reference image) | `skimage.exposure.match_histograms(img, ref, channel_axis=2)` in LAB | scikit-image (or manual: 15 lines cv2+numpy) | Yes — manual version tested, works in LAB |
| Reinhard colour transfer | Mean/stddev matching per-channel in LAB | ~10 lines cv2+numpy | Yes — standard algorithm |
| Channel mixer | `cv2.transform(img, M)` with user-specified 3×3 matrix | OpenCV core | Yes |
| Split toning | Blend colour into shadows (low-L) and highlights (high-L) separately | ~25 lines | Not tested |

### Group 3: Detail

| Feature | Algorithm | Source | Verified |
|---------|-----------|--------|----------|
| Sharpen (unsharp mask, LAB) | Gaussian blur on L → subtract → threshold → adaptive mask → blend | Image Space `adjust_sharpen` ported to cv2 | Yes — 22 lines, tested |
| Denoise (bilateral) | `cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)` | OpenCV core | Yes — 0.06s on 12MP |
| Denoise (NL-means) | `cv2.fastNlMeansDenoisingColored(img, None, h, hColor, patch, search)` | OpenCV core | Yes — 4.07s on 12MP (search=21), 0.67s (search=11) |
| Denoise (multi-stage) | Impulse (median) + bilateral + edge-preserving, blended by strength | Image Space approach ported to cv2 | Yes — 18 lines, tested |
| Denoise (wavelet) | `skimage.restoration.denoise_wavelet` | scikit-image | Not tested (not installed) but API verified from docs |
| Detail enhance | `cv2.detailEnhance(img, sigma_s, sigma_r)` | OpenCV core | Yes — `hasattr` confirmed |
| Noise estimation | `skimage.restoration.estimate_sigma` | scikit-image | Not tested |

### Group 4: Corrections

| Feature | Algorithm | Source | Verified |
|---------|-----------|--------|----------|
| Vignette correction | Radial r² gain model, `gain = 1 + strength * (r/r_max)²` | 8 lines numpy | Yes — tested |
| Lens distortion correction | `cv2.undistort` / `cv2.fisheye.undistortImage` with camera matrix | OpenCV core | Available (used in prep360) |
| Chromatic aberration | Per-channel radial shift | ~20 lines cv2.remap | Not tested |
| Dehaze (dark channel) | Dark channel prior → atmospheric light estimation → scene recovery | ~40 lines cv2, or `cv2.xphoto` (contrib) | Not tested |

### Group 5: Batch Operations

| Feature | Algorithm | Source | Verified |
|---------|-----------|--------|----------|
| Exposure normalization | Compute median L per image, scale to batch mean | 20 lines cv2 | Yes — tested, spread 94→1 |
| Histogram matching to reference | Per-channel CDF matching in LAB | 15 lines cv2+numpy (or scikit-image) | Yes — manual version tested |
| Colour correction from chart | Fit once, apply to N images via `apply_matrix_colour_correction` | colour-science | Yes — workflow verified |
| LUT application | `prep360/core/lut.py` trilinear interpolation | Already built | Yes — existing code |
| Batch export | ThreadPoolExecutor (GUI masking pattern) or ProcessPoolExecutor (prep360 CPU pattern), configurable format/quality | Existing patterns in project | Yes |

### Group 6: RAW Development

| Feature | Algorithm | Source | Verified |
|---------|-----------|--------|----------|
| RAW loading | `rawpy.imread(path)` → `raw.postprocess(...)` | rawpy 0.26.1 | Wheel exists for 3.14. API verified from docs |
| Demosaic algorithm | AHD (default), VNG, PPG, DCB, LMMSE, AMAZE, DHT, AAHD, etc. | rawpy `demosaic_algorithm` param | Verified from API docs |
| Output colour space | sRGB, Adobe, ProPhoto, XYZ, ACES, P3D65, Rec2020 | rawpy `output_color` param | Verified |
| Bit depth | 8 or 16 bit per sample | rawpy `output_bps` | Verified |
| White balance (camera) | `use_camera_wb=True` | rawpy | Verified |
| White balance (auto) | `use_auto_wb=True` | rawpy | Verified |
| White balance (custom) | `user_wb=[r, g, b, g2]` multipliers | rawpy | Verified |
| Exposure shift | `exp_shift` (0.25–8.0 linear scale) | rawpy | Verified |
| Highlight handling | Clip, unclip, blend, rebuild | rawpy `highlight_mode` | Verified |
| Noise reduction (pre-demosaic) | `fbdd_noise_reduction` | rawpy | Verified |
| Brightness | `bright` multiplier, `no_auto_bright` | rawpy | Verified |
| Gamma | `gamma=(power, slope)`, default BT.709 (2.222, 4.5) | rawpy | Verified |
| Fuji X-Trans support | LibRaw handles X-Trans CFA natively | Verified via libraw.org supported cameras list |
| iPhone ProRAW (DNG) | DNG format supported, iPhones 12–15 Pro listed | Verified via libraw.org |

---

## Performance Constraints (Verified Benchmarks, 12MP image, RTX 3090 Ti)

| Operation | Time | Strategy |
|-----------|------|----------|
| Bilateral filter | 0.06s | Real-time slider OK |
| CLAHE | 0.03s | Real-time slider OK |
| NL-means (search=11) | 0.67s | Background thread, show on release |
| NL-means (search=21) | 4.07s | Export only, not preview |
| NL-means on 800×600 preview | 0.23s | Use downsampled preview for slider |
| Exposure/WB/contrast/saturation | <0.01s | Instant (numpy multiply/LUT) |
| Shadows/highlights (LAB) | ~0.05s est. | Real-time (LAB conversion + numpy) |
| Sharpen (LAB unsharp) | ~0.1s est. | Real-time |

**Preview strategy:** Maintain a downsampled (~2MP) preview copy for real-time slider feedback. Apply full-resolution adjustments only on export or on "Apply" action. NL-means denoising at full resolution is export-only; preview uses bilateral or smaller search window.

---

## Architecture

### New Files

```
reconstruction_gui/
├── tabs/
│   └── adjust_tab.py          # Tab builder: build_adjust_tab(app, tab_frame)
├── adjust_engine.py           # Pure processing library (no GUI deps)
│                              #   - AdjustmentState dataclass (all parameters)
│                              #   - RawDevelopState dataclass (demosaic, colourspace, bps, wb mode)
│   Sections:                  #   - apply_adjustments(img, state) → img
│                              #   - Individual functions: exposure, shadows, highlights,
│                              #     wb, saturation, vibrance, contrast, curves, hsl,
│                              #     sharpen, denoise, vignette, histogram_match
│                              #   - RAW loading: load_raw(path, raw_params) → float32 array
│                              #   - Chart correction: fit_from_chart(measured, reference) → matrix
│                              #                      apply_chart_correction(img, matrix) → img
│                              #   - Batch: normalize_exposures(images) → images
│                              #            match_to_reference(images, ref) → images
└── adjust_presets.py          # Named adjustment preset save/load (JSON)
```

**Note on state separation:** `AdjustmentState` (tone/colour/detail sliders) and `RawDevelopState` (demosaic algorithm, colour space, output_bps, WB mode) are **separate dataclasses** but nested inside a single serializable `AdjustDocument` for presets and undo. Rationale: RAW develop params are set once and rarely change per-image, while adjustment sliders are tweaked constantly. The undo stack captures the full `AdjustDocument` so both are reversible, but the UI can present them in separate collapsible sections.

### Infrastructure Reuse (from existing app)

These host-app primitives are available on the `app` instance and MUST be reused (not reimplemented):

| Primitive | Location | Purpose |
|-----------|----------|---------|
| `app.log(msg)` | `app_infra.py` | Thread-safe logging to console widget via `log_queue` |
| `app.cancel_flag` | `reconstruction_zone.py` | `threading.Event` — checked by long-running ops to abort |
| `app.is_running` | `reconstruction_zone.py` | Boolean guard preventing concurrent operations |
| `app._start_operation(run_btn, stop_btn)` | `app_infra.py:158` | Sets `is_running=True`, clears cancel flag, starts progress bar, disables `run_btn`, shows `stop_btn` |
| `app._stop_operation(run_btn, stop_btn)` | `app_infra.py:167` | Sets `is_running=False`, stops progress bar (sets to 1.0), re-enables `run_btn`, hides `stop_btn` via `pack_forget()` |
| `app.stop_operation()` | `app_infra.py:176` | Called by stop button callback — logs "Stopping..." and sets `cancel_flag` |
| `app._browse_dir_into(entry)` | `app_infra.py` | Open folder dialog, write result into CTkEntry |
| `app._browse_file_for(entry)` | `app_infra.py` | Open file dialog |
| `app._browse_video_for(entry)` | `app_infra.py` | Open video file dialog (filters for video extensions) |
| `app._entry_initialdir(entry)` | `app_infra.py` | Compute smart initial directory from entry contents |
| `app.progress_bar` | `reconstruction_zone.py:1386` | CTkProgressBar in the Mask tab's scroll area. Shared across Mask operations. Adjust can reuse it if desired (it's on the `app` instance), or add a local one in its own scroll frame for tab-local UX. |
| `app._dpi_scale` | `reconstruction_zone.py` | DPI scaling factor for CTkImage size calculations |

### Activity Store Integration

Log Adjust operations to the Projects tab's Recent Activity feed for workflow traceability. Use the `app.record_activity()` method (defined in `reconstruction_zone.py:201`), NOT the store directly:

```python
# After batch export completes:
app.record_activity(
    operation="adjust_export",
    input_path=str(input_folder),
    output_path=str(output_folder),
    status="completed",
    details={"count": count, "format": format, "preset": preset_name or None},
)

# After preset applied:
app.record_activity(
    operation="adjust_preset",
    input_path=str(input_folder),
    output_path=str(input_folder),
    status="completed",
    details={"preset": preset_name},
)
```

Signature: `app.record_activity(operation: str, input_path: str, output_path: str, status: str = "completed", details: Optional[Dict[str, Any]] = None)`. Thread-safe. No-ops if `_activity_store` is None.

### Gumroad Build Constraints

Features that depend on AGPL-licensed or heavy optional packages must gracefully degrade:

| Feature | Dependency | Gumroad behaviour |
|---------|-----------|-------------------|
| AI denoise (NAFNet/SCUNet) | torch, nafnetlib/spandrel | Dropdown shows "AI: unavailable (requires PyTorch)" when torch import fails. Bilateral/NL-means still work. |
| Content-aware exposure (sky detection) | ultralytics (YOLO) | Design intent: fall back to RF-DETR if YOLO unavailable, or disable with label. **Caveat:** RF-DETR uses COCO classes and may not include a "sky" class — verify model labels before treating as drop-in. If RF-DETR lacks sky detection, disable the feature entirely on Gumroad builds rather than shipping a broken fallback. |
| Chart auto-detection | colour-checker-detection (optional) | Shows "Auto-detect unavailable — use manual selection" when not installed |

Pattern: match existing `is_gumroad()` / try-import gates in `reconstruction_pipeline.py:51-78`.

### Processing Pipeline Order

Based on Image Space's verified correct ordering and standard photographic practice:

```
1. RAW Development (if RAW input)
   → demosaic, colour space, WB, exposure shift, gamma
   → Output: float32 [0,1] sRGB

2. Noise Reduction (before amplifying noise)
   → bilateral / NL-means / wavelet

3. White Balance (on clean data)
   → temperature + tint OR eyedropper OR auto (gray-world/learning-based)

4. Exposure (global multiplicative)
   → EV stops: img * 2^ev

5. Shadows / Highlights (targeted tonal, LAB)
   → Exponential falloff curves on L-channel

6. Contrast / Levels / Curves
   → S-curve, black/white points, custom spline LUT

7. Colour adjustments
   → Saturation, vibrance, HSL per-hue, channel mixer, split toning

8. Colour correction (chart-based or reference matching)
   → Finlayson2015 matrix, or histogram matching

9. Sharpening (last, to avoid amplifying earlier ops)
   → LAB unsharp mask with adaptive masking

10. Corrections
    → Vignette, chromatic aberration (if applicable)

11. Export
    → Format conversion, metadata copy, batch naming
```

### Integration with Existing Tabs

**Tab order:** Projects → **Adjust** → Extract → Mask → Review → Align → Coverage

| Connection | Mechanism | Widget wiring |
|------------|-----------|---------------|
| Adjust → Extract | Adjust outputs a corrected video. "Open in Extract" button writes path and switches tab. | Write `app.analyze_video_entry` with output video path, then `app.tabs.set("Extract")` |
| Adjust → Mask (direct) | Adjust outputs corrected frames. "Open in Mask" button sets Mask input. | Write `app.input_entry` with Adjust output folder path, then `app.tabs.set("Mask")` |
| Extract → Adjust | "Open in Adjust" button in Extract passes extracted frames folder. | Write `app.adjust_input_entry` (new, created by Adjust tab) with `app.extract_output_entry.get()`, then `app.tabs.set("Adjust")` |
| Adjust (video in, frames out) | Combines adjust + extraction — video decoded, per-frame adjustment, saved as image sequence. Bypasses Extract entirely. | Adjust tab's own video input entry + output folder entry |
| RAW import | Directory browser loads RAW files directly, no other tab involved. | Adjust tab's directory input entry |
| Preview panel | Shared preview column (used by Extract, Mask, Review, Coverage). Adjust module owns buffer, renders into panel. | `_preview_mode == "adjust"` |
| LUT | Reuse `prep360/core/lut.py` directly. User can browse for .cube files. | File browse dialog in Adjust's Colour Correction section |

### GUI Layout (Left Panel, Scrollable)

```
┌─ Input ──────────────────────────────────┐
│ Directory: [...........................] [Browse]
│ File types: *.jpg *.png *.dng *.nef *.raf
│ [Load RAW] [Load Images]
└──────────────────────────────────────────┘

┌─ RAW Development (collapsible, shown if RAW loaded) ─┐
│ Demosaic: [AHD ▼]  Colour space: [sRGB ▼]
│ White balance: [Camera ▼] / [Auto] / [Custom]
│ Exposure shift: [────●────] 0.0
│ Highlight mode: [Clip ▼]
│ Output bit depth: [16-bit ▼]
└──────────────────────────────────────────────────────┘

┌─ Tone ───────────────────────────────────┐
│ Exposure:   [────●────]  0.0 EV
│ Contrast:   [────●────]  0
│ Highlights: [────●────]  0
│ Shadows:    [────●────]  0
│ Whites:     [────●────]  0
│ Blacks:     [────●────]  0
│ [Curves...]  [Levels...]
└──────────────────────────────────────────┘

┌─ Colour ─────────────────────────────────┐
│ Temperature: [────●────] 5500K
│ Tint:        [────●────] 0
│ Saturation:  [────●────] 0
│ Vibrance:    [────●────] 0
│ [HSL...]  [Channel Mixer...]  [Split Tone...]
│ WB: [Auto ▼] [Eyedropper]
└──────────────────────────────────────────┘

┌─ Detail ─────────────────────────────────┐
│ Sharpen amount:    [────●────] 0
│ Sharpen radius:    [────●────] 1.0
│ Sharpen threshold: [────●────] 0
│ Denoise strength:  [────●────] 0
│ Denoise method:    [Bilateral ▼]
└──────────────────────────────────────────┘

┌─ Corrections ────────────────────────────┐
│ □ Vignette correction   Strength: [──●──]
│ □ Chromatic aberration
│ □ CLAHE                 Clip: [──●──]
└──────────────────────────────────────────┘

┌─ Colour Correction (collapsible) ────────┐
│ Mode: [None ▼] / Chart / Reference Image / Histogram Match
│ Chart: [Select image with chart] [Detect] [Manual 4-corner]
│ Reference: [Select reference image]
│ Method: [Finlayson2015 ▼]  Degree: [2 ▼]
│ [Fit Correction]  Status: "Ready — 3x6 matrix computed"
└──────────────────────────────────────────┘

┌─ Batch ──────────────────────────────────┐
│ □ Normalize exposures across batch
│ □ Match histograms to reference
│ Output format: [JPEG ▼] Quality: [92]
│ Output folder: [...................] [...]
│ Naming: [original ▼]
│ [Export All]  [Export Selected]
└──────────────────────────────────────────┘
```

### Preview Panel Behaviour

- Shows current selected image with all adjustments applied in real-time
- For fast operations (<0.1s): update on slider drag
- For slow operations (denoise, NL-means): update on slider release, show spinner
- Internal preview buffer: downsampled to ~2MP for slider interaction
- Full-resolution applied only on export
- Before/after toggle: button or drag-reveal slider
- Navigation: same slider/arrows as existing tabs

---

## Chart-Based Colour Correction — Workflow Detail

### Detection Options (in priority order)

1. **`cv2.mcc.CCheckerDetector`** (after contrib upgrade)
   - Built into OpenCV, no extra install
   - Returns detected chart as colour patches
   - Lightweight, fast

2. **`colour-checker-detection`** (optional install)
   - Segmentation-based detection (no YOLO on 3.14)
   - 50MB wheel (bundled model weights)
   - More robust than cv2.mcc in difficult lighting

3. **Manual selection**
   - User draws crop box around chart
   - Clicks 4 corners for perspective correction
   - Grid sampling at known positions
   - Always works, no model dependency

### Correction Pipeline (verified end-to-end)

```python
# 1. Extract measured patch colours (from detection or manual selection)
measured_patches = extract_patches(chart_image)  # shape: (24, 3)

# 2. Get reference values
cc = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
reference = np.array([colour.XYZ_to_sRGB(colour.xyY_to_XYZ(xyY)) for xyY in cc.data.values()])

# 3. Compute correction matrix (once per session)
M = colour.characterisation.matrix_colour_correction(
    measured_patches, reference, method='Finlayson 2015', degree=2
)  # Returns 3×6 matrix

# 4. Apply to each image in batch
for img in batch:
    corrected = colour.characterisation.apply_matrix_colour_correction(
        img, M, method='Finlayson 2015', degree=2
    )
```

**API verification note:** The above was tested against `colour-science==0.4.6`. The `colour.characterisation` module path and method string `'Finlayson 2015'` (with space) are version-sensitive. Pin `colour-science` in requirements and add a unit test that imports and calls `matrix_colour_correction` with a small fixture to catch API drift on upgrade.

---

## Export Options

| Format | Library | Options |
|--------|---------|---------|
| JPEG | cv2 / Pillow | Quality 1–100, colour space (sRGB) |
| PNG | cv2 / Pillow | Compression level, 8/16-bit |
| TIFF | cv2 / tifffile | 8/16-bit, compression (LZW/deflate/none) |
| OpenEXR | OpenImageIO (if available) | Float16/32, colour space |

Metadata (EXIF) preservation: copy from source using `Pillow` (already in requirements) + `piexif` for lossless EXIF transplant. **Note:** `piexif` and `ExifRead` are NOT in `requirements.txt` today — add when export features ship. Pillow alone can read/write basic EXIF but `piexif` handles full transplant (GPS, maker notes, etc.) without re-encoding.

---

## Additional Features (Resolved from Scope Discussion)

### AI-Powered Denoise

PyTorch + CUDA 12.6 are already installed (for SAM3/YOLO). Neural denoise adds no new framework dependency.

| Model | Package | Quality | Speed (12MP, 3090 Ti, tiled) | Use case |
|-------|---------|---------|------------------------------|----------|
| NAFNet-width64 | `pip install nafnetlib` (v0.1.0, verified on PyPI) | Excellent | ~2–4s | Fastest zero-effort option. 3-line API |
| SCUNet-GAN | `pip install spandrel` (v0.4.2, verified) + download .pth | Excellent (blind/real noise) | ~5–10s | Best for unknown ISO / mixed lighting. Blind denoiser |
| DnCNN/FFDNet | spandrel | Good (synthetic AWGN) | ~1–2s | Fast but designed for Gaussian noise, not real camera noise |
| SwinIR | spandrel | Very good | ~8–15s | Heavy transformer |
| Restormer | spandrel | Excellent | ~10–20s | Highest quality, slowest |

**Recommended approach:** Offer a "Denoise method" dropdown with:
- Bilateral (fast, real-time preview)
- NL-means (slower, better)
- AI: NAFNet (best balance of speed + quality)
- AI: SCUNet (best quality, slower)

**Implementation:** Spandrel provides a universal loader for any `.pth` restoration model. Tiling required at 12MP+ (512px tiles with overlap). NAFNetlib is a simpler single-model wrapper if we want the minimal path.

```python
# nafnetlib example (verified on PyPI)
from nafnetlib import DenoiseProcessor
processor = DenoiseProcessor("sidd_width64", device="cuda")
result = processor.process(pil_image)

# spandrel example (multiple models)
from spandrel import ModelLoader
model = ModelLoader().load_from_file("scunet_color_real_gan.pth").cuda().eval()
# + tiling loop for large images
```

### Content-Aware Adjustments

The useful feature for photogrammetry: **region-aware exposure normalization** using YOLO/SAM (already in the project).

| Feature | Value for photogrammetry | Implementation |
|---------|--------------------------|----------------|
| Sky/subject separation for independent tone | HIGH — prevents sky blow-out while lifting shadows | YOLO sky class → mask → separate exposure for sky vs ground |
| Skin tone protection during colour correction | Medium — relevant for body/portrait scanning | Face detection gate on local tone curves |
| Privacy blur for bystanders | Real need for street/site capture | YOLO person + face detection → blur. Applied POST-reconstruction to texture, not source images (blurring source destroys feature points) |

**Not useful / counterproductive:**
- "Portrait WB" — batch consistency matters more than flattering individual frames
- AI scene classification for auto-defaults — too opaque, photogrammetry needs predictable processing
- Selective artistic adjustments — creates view-inconsistency that breaks MVS matching

### Distributed Processing

Image Space's client/server TCP architecture (JSON protocol, heartbeat, auto-reconnection) is a reference for this. For batch-processing 500+ images through the full adjustment pipeline, distributing across machines on the Tailscale mesh reduces wait time significantly.

**Phase 1:** Local batch — `ProcessPoolExecutor` for CPU-bound adjustment chains (matches prep360's reframer/blur pattern), or `ThreadPoolExecutor` if GPU ops are mixed in (avoids CUDA context forking issues on Windows)
**Phase 2:** Network distribution — extract the adjustment pipeline into a stateless function that accepts (image_path, AdjustmentState) → output_path, wrap in a task protocol. Worker machines need only OpenCV + numpy + optional torch for AI denoise.

### Video Processing

The Adjust tab accepts video files as input in addition to image directories. This eliminates DaVinci Resolve from the pipeline — DJI LUTs and colour correction happen entirely within the GUI.

**Current workflow (to be replaced):**
```
Video → DaVinci Resolve (apply DJI LUT + colour grade) → Export → Extract tab → Mask
```

**New workflow options:**
```
Video → Adjust tab (LUT + full adjustment pipeline) → Export as corrected frames → Mask
Video → Adjust tab (LUT + corrections) → Export as corrected video → Extract tab (reframing) → Mask
Images → Adjust tab (full adjustments) → Mask
RAW files → Adjust tab (develop + adjust) → Mask
```

**Tab order:** Projects → **Adjust** → Extract → Mask → Review → Align → Coverage

The pipeline is not strictly linear — Adjust works on whatever is fed to it:

| Input | Output options | Use case |
|-------|---------------|----------|
| Video file (.mp4, .mov, etc.) | Corrected video file (re-encoded) | Pre-extract correction, keep as video for Extract tab's reframing |
| Video file | Corrected frame sequence (images) | Combines Adjust + Extract in one step — direct to Mask |
| Image directory (JPG/PNG/TIFF) | Corrected images to output folder | Standard post-extraction adjustment |
| RAW directory (NEF/RAF/DNG/CR3) | Developed + corrected images | Camera RAW workflow, no Extract needed |
| Image sequence (timelapse) | Temporally-smoothed corrected images | Auto-exposure normalization without flicker |

**Video implementation:**
- Decode: ffmpeg subprocess (existing pattern in prep360)
- Frame scrubbing for preview: seek to frame N, decode single frame, display with adjustments applied
- Batch export: ffmpeg pipe decode → per-frame adjustment pipeline → ffmpeg pipe encode (or save as images)
- LUT application: existing `prep360/core/lut.py` trilinear interpolation, applied per frame
- Temporal smoothing: rolling-average of exposure/WB parameters across N-frame window to prevent flicker

### Non-Destructive Edit History (Undo)

Since all adjustments are parameterized, undo is lightweight. The full state is captured in an `AdjustDocument` that nests both adjust and RAW params:

```python
@dataclass
class AdjustmentState:
    exposure: float = 0.0
    shadows: float = 0.0
    highlights: float = 0.0
    # ... ~25 tone/colour/detail parameters ...

@dataclass
class RawDevelopState:
    demosaic_algorithm: str = "AHD"
    output_color: str = "sRGB"
    output_bps: int = 16
    use_camera_wb: bool = True
    exp_shift: float = 1.0
    highlight_mode: int = 0
    # ... RAW-specific params ...

@dataclass
class AdjustDocument:
    adjustments: AdjustmentState = field(default_factory=AdjustmentState)
    raw: RawDevelopState = field(default_factory=RawDevelopState)

class UndoStack:
    """Stores full AdjustDocument snapshots — both adjust sliders and RAW params are undoable."""

    def __init__(self, max_depth=50):
        self._stack: list[AdjustDocument] = []
        self._redo: list[AdjustDocument] = []
        self._max = max_depth

    def push(self, state: AdjustDocument):
        self._stack.append(copy.deepcopy(state))
        if len(self._stack) > self._max:
            self._stack.pop(0)
        self._redo.clear()

    def undo(self) -> AdjustDocument | None:
        if len(self._stack) < 2:
            return None
        self._redo.append(self._stack.pop())
        return copy.deepcopy(self._stack[-1])

    def redo(self) -> AdjustDocument | None:
        if not self._redo:
            return None
        state = self._redo.pop()
        self._stack.append(state)
        return copy.deepcopy(state)
```

Each state is ~200 bytes (30 floats + a few strings). Stack of 50 = ~10KB. No image copies needed — re-apply parameters to regenerate the preview.

---

## Preview Architecture (Detailed)

### Design: Adjust Module Owns Preview Buffer, Renders Into Shared Panel

The shared right-side preview panel (used by Extract, Mask, Review, and Coverage — Projects and Align replace it with their own panels) displays whatever CTkImage is handed to it. The Adjust tab maintains its own processing pipeline internally:

```
┌─ adjust_tab.py ─────────────────────────────────┐
│                                                  │
│  source_image (original, full-res)               │
│       ↓                                          │
│  preview_buffer (downsampled to ~2MP)            │
│       ↓                                          │
│  apply_adjustments(preview_buffer, state)        │
│       ↓ (<10ms for most ops at 2MP)              │
│  adjusted_preview (PIL Image)                    │
│       ↓                                          │
│  Push to shared preview panel as CTkImage        │
│                                                  │
└──────────────────────────────────────────────────┘
```

**Real-time feedback rules:**
- Fast ops (exposure, WB, saturation, contrast, curves): update on slider drag (~5ms)
- Medium ops (shadows/highlights, sharpen, CLAHE): update on slider drag (~30ms at 2MP)
- Slow ops (NL-means, AI denoise): update on slider RELEASE + show spinner. Use the 2MP preview with smaller search window for NL-means (0.23s measured at 800×600)

**Full-resolution processing:**
- Only on "Export" action
- Runs in background thread with progress bar
- All the same operations but on original-size image
- For AI denoise: tiled processing with overlap at full resolution

**Before/After:**
- Two buffers: `original_preview` and `adjusted_preview`
- Toggle button swaps which is displayed
- Optional: drag-reveal slider (left = original, right = adjusted) — more complex widget but standard UX

**Navigation:**
- Same left/right arrows + slider as existing tabs
- Switching images: load new source, regenerate preview_buffer, re-apply current state
- Dedicated state attributes: `_adjust_image_list`, `_adjust_nav_idx` (NOT shared with Mask's `_nav_idx` or `input_entry`-based list)

**Preview mode integration (`_preview_mode == "adjust"`):**
- `_on_tab_change`: when `active == "Adjust"`, set `self._preview_mode = "adjust"` and call Adjust-owned loaders
- SAM click handlers on `_process_overlay_label`: must **no-op** unless mode is `"process"` AND click state is ready. Gate with: `if self._preview_mode != "process": return`
- "Mask" checkbox in preview header: **hide** when mode is `"adjust"` (not just disable — avoids confusion)
- GPU cache indicator: hide when mode is `"adjust"` (irrelevant to adjustment pipeline)

**Debounce mechanism:**
- Use the existing `self.after()` + cancel pattern (same as `_nav_debounce_id` / `_zoom_debounce_id` in `reconstruction_zone.py`)
- Fast ops: 16ms debounce (single frame at 60fps) — effectively instant
- Medium ops: 50ms debounce
- Slow ops: trigger on `<ButtonRelease>` event, no debounce needed (single fire)

---

## Resolved Decisions

| Question | Decision |
|----------|----------|
| Curves UI | Both: full bezier curve editor AND simplified preset curves (S-curve, film, etc.) with strength slider |
| HSL panel | Both: full 8-hue panel (Lightroom-style) AND simplified view |
| Chart detection | Include `cv2.mcc` from day one. Manual selection as fallback. `colour-checker-detection` as optional upgrade. |
| Preview architecture | Adjust module owns buffer, renders into shared panel. Dedicated `_preview_mode == "adjust"`. |
| Preset system | Yes — save/load named adjustment presets as JSON at `~/.reconstruction_zone/presets/adjust/`. |
| AI denoise | Include — NAFNet (via nafnetlib) as default, SCUNet (via spandrel) as quality option. Gracefully disabled when torch unavailable (Gumroad builds). |
| Content-aware | Region-aware exposure normalization via YOLO sky detection. Gumroad: RF-DETR fallback or disable. No artistic selective edits. |
| Distributed processing | Phase 2 feature — local batch first, network distribution later |
| Video input | Accept video files via ffmpeg CLI pipe (not VideoCapture). Decode + adjust + export as frames or re-encode. |
| Undo | Lightweight `AdjustDocument` stack (nests `AdjustmentState` + `RawDevelopState`). 50 states, ~10KB. |
| Export colour space | sRGB default for photogrammetry. Selectable for RAW archival export only. |
| Batch concurrency | ProcessPoolExecutor for CPU-only chains; ThreadPoolExecutor when GPU ops active. Auto-select. |
| State separation | `AdjustmentState` (sliders) + `RawDevelopState` (develop params) nested in `AdjustDocument` for undo/presets |

---

## Documentation Updates Required

When the Adjust tab ships, these files must be updated:

| File | Change |
|------|--------|
| `reconstruction_gui/docs/ARCHITECTURE.md` | Add Adjust tab to module map, update tab count, document `adjust_engine.py` and `adjust_presets.py` |
| `CLAUDE.md` (project root) | Add Adjust tab to architecture diagram, update tab list, add adjust_engine to key abstractions |
| `README.md` or equivalent user docs | Document new tab capabilities, new dependencies, RAW support |
| `reconstruction_zone.spec` | Add `rawpy` DLLs to frozen build (always). `nafnetlib`/`spandrel` only when building an AI-enabled distribution (mirror the `_is_gumroad` / env-var gating already used for ultralytics). Verify no missing native extensions. |

---

## Dependencies Summary

### Packaging Policy

Split into tiers so minimal installs stay light:

| File | Contents | When needed |
|------|----------|-------------|
| `requirements.txt` | Change `opencv-python` → `opencv-contrib-python==4.13.0.92`. Add `rawpy`, `scikit-image`, `colour-science`, `piexif`. | Always (core Adjust) |
| `requirements-ai-denoise.txt` | `nafnetlib==0.1.0`, `spandrel==0.4.2` (both depend on torch) | Only when AI denoise is desired. UI degrades gracefully without it. |
| `requirements-yolo.txt` | `ultralytics` (existing, AGPL) | Content-aware features. Already separated. |

### Must Install (core)
```
pip uninstall opencv-python
pip install opencv-contrib-python==4.13.0.92
pip install rawpy==0.26.1
pip install scikit-image==0.26.0
pip install colour-science==0.4.6
pip install piexif
```

### Optional — AI Denoise (requires torch)
```
pip install nafnetlib==0.1.0
pip install spandrel==0.4.2
```

### Already Present (in project requirements.txt)
- `scipy`
- `numpy`
- `Pillow`
- `PyTorch` 2.10.0+cu126 (installed separately via --index-url)
- `ultralytics` (YOLO11, in `requirements-yolo.txt`, AGPL — not in Gumroad builds)

### Optional
```
pip install colour-checker-detection==0.2.3   # Chart auto-detection (segmentation method, 50MB)
```

### Verified on PyPI
| Package | Version | Python 3.14 wheel? |
|---------|---------|-------------------|
| opencv-contrib-python | 4.13.0.92 | Yes (cp37-abi3) |
| rawpy | 0.26.1 | Yes (cp314-win_amd64) |
| scikit-image | 0.26.0 | Yes |
| nafnetlib | 0.1.0 | Pure Python (any) |
| spandrel | 0.4.2 | Pure Python (any) |
| colour-checker-detection | 0.2.3 | Yes (py3-none-any, requires >=3.11,<3.15) |
