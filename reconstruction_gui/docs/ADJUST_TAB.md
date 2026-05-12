# Adjust Tab

The Adjust tab is the visual colour and finishing stage for Reconstruction Zone. In the main reconstruction workflow it sits after Extract:

```text
Extract -> Adjust -> Mask -> Review -> Align
```

Extract writes raw selected frames. Adjust previews `.cube` LUTs and manual tone/color changes, then exports a new adjusted dataset with an `adjustment_recipe.json` and `adjustment_manifest.json`. The raw selected frames are not modified.

## Sections

The left panel is organized into vertically stacked sections:

| Section | Type | Purpose |
|---------|------|---------|
| **Input** | Always visible | Load images, RAW files, or video. Navigate between files. |
| **RAW Development** | Collapsible | Develop RAW files (NEF, RAF, DNG, CR2, CR3, ARW) via rawpy. Demosaic algorithm, colour space, white balance mode, exposure shift, highlight handling. |
| **LUT & Recipe** | Always visible | Enable a `.cube` LUT, set strength, reload/clear, load/save recipes, toggle before preview, choose front/back for paired datasets. |
| **Tone** | Always visible | Exposure (EV stops), contrast, highlights, shadows |
| **Colour** | Always visible | White balance (temperature + tint), saturation, vibrance |
| **Detail** | Collapsible | Sharpening (amount, radius, threshold), denoising (strength + method) |
| **Corrections** | Collapsible | CLAHE adaptive contrast, vignette correction |
| **Colour Correction** | Collapsible | Chart-based calibration (ColorChecker), histogram matching to a reference image |
| **Export** | Always visible | Output format, quality, bit depth, metadata options, safe adjusted dataset export with progress |

The right panel is the **shared Preview panel**. It shows the current image with all adjustments applied in real-time. The preview operates on a downsampled (~2 megapixel) buffer for instant feedback — full-resolution processing happens only on export.

---

## Input

### Paths

- **Input** — A paired extraction root, image folder, single image file, or video file. Supports JPG, PNG, TIFF, BMP, WebP, and RAW formats (NEF, CR2, CR3, DNG, ARW, RAF). Video files (MP4, MOV, AVI, MKV) are accepted for quick sample exports.
- **Output** — Directory where adjusted derivatives are written. Non-empty output folders are refused by default.

### Buttons

| Button | Purpose |
|--------|---------|
| **Folder** | Browse for a directory of images |
| **File** | Browse for a single image or video file |
| **Load** | Scan the input path and populate the image list (or probe a video file) |
| **< >** | Navigate to previous / next image in the loaded set |
| **Reset All** | Return every slider to its default (zero / neutral) value |

### Navigation

For image directories, the **< >** buttons step through files in alphabetical order. The counter shows `3 / 142` style position.

For video files, the **< >** buttons step through the video in 1-second increments. The counter shows `5.0s / 120.0s` style timestamps. Each step extracts a single frame via ffmpeg and displays it with current adjustments applied.

### Supported input types

| Input type | How it loads |
|------------|-------------|
| **Image directory** | Scans for all image files (including RAW), sorts alphabetically, loads the first |
| **Paired extraction root** | Detects `front/frames` + `back/frames`, loads both lists together, and applies one shared recipe to both lenses |
| **Single image** | Loads directly |
| **Video file** | Probes with ffprobe for metadata and extracts preview/sample frames on demand. Use Extract for reconstruction-grade frame selection. |
| **RAW file** | Developed via rawpy with parameters from the RAW Development section |

---

## RAW Development

Collapsed by default. Controls how RAW camera files are decoded into viewable RGB images. Requires `rawpy` (install with `pip install rawpy`). If rawpy is not installed, the section shows an install prompt and RAW files are skipped.

### Controls

| Control | Options | Default | Purpose |
|---------|---------|---------|---------|
| **Demosaic** | AHD, VNG, PPG, DCB, LMMSE, AMAZE, DHT, AAHD | AHD | Bayer/X-Trans demosaicing algorithm. AHD is a good default. AMAZE and LMMSE may produce better results on high-ISO shots. |
| **Colour** | sRGB, Adobe, ProPhoto, ACES, Rec2020 | sRGB | Output colour space. Use sRGB for photogrammetry pipelines. ProPhoto/ACES for archival export. |
| **WB** | Camera, Auto, Daylight | Camera | White balance source. Camera = use the WB the camera recorded. Auto = rawpy estimates from the image content. Daylight = D65 illuminant. |
| **Exp shift** | 0.25x – 4.0x | 1.0x | Linear exposure multiplier applied during RAW development (before demosaic). Useful for recovering under/overexposed RAW files. |
| **Highlights** | Clip, Unclip, Blend, Rebuild | Clip | How to handle clipped highlights. Clip = hard clip to white. Unclip/Blend/Rebuild attempt recovery from the raw data. |

### Re-develop button

After changing RAW settings, click **Re-develop RAW** to re-decode the current file with the new parameters. The adjustment sliders (Tone, Colour, etc.) are applied on top of the developed result.

### Supported RAW formats

| Extension | Camera |
|-----------|--------|
| `.nef` | Nikon |
| `.cr2`, `.cr3` | Canon |
| `.arw` | Sony |
| `.raf` | Fujifilm (X-Trans sensor supported natively via LibRaw) |
| `.dng` | Adobe DNG (including Apple iPhone ProRAW) |
| `.raw` | Generic RAW |

---

## LUT & Recipe

The LUT section is the first-class place to preview camera/display transforms such as DJI Osmo360 D-Log M to Rec.709.

| Control | Purpose |
|---------|---------|
| **Apply .cube LUT** | Enables the LUT in preview and export |
| **LUT path** | Selects the `.cube` file |
| **Strength** | Blends original (0%) to full LUT (100%) |
| **Reload** | Re-reads and validates the selected LUT |
| **Clear** | Disables and removes the active LUT |
| **Load Recipe / Save Recipe** | Reads or writes the full adjustment recipe JSON |
| **Before** | Temporarily shows the unadjusted preview buffer |
| **Front / Back** | Chooses which lens to preview for paired datasets |

Invalid LUTs block export until fixed. Strength updates use the same shared color engine used for full export, so preview and export follow the same processing order.

### Paired datasets

When the input root contains:

```text
front/frames
back/frames
paired_extraction_manifest.json
```

Adjust loads front and back lists as one paired dataset. Export is blocked if the counts do not match. A successful paired export writes:

```text
front/frames
back/frames
paired_extraction_manifest.json
adjustment_recipe.json
adjustment_manifest.json
```

The same recipe is applied to both lenses.

---

## Tone

Four sliders controlling overall brightness and tonal distribution. All operate in real-time on the preview.

| Slider | Range | Default | Algorithm |
|--------|-------|---------|-----------|
| **Exposure** | -5.0 to +5.0 EV | 0.0 | Multiplicative: `pixel × 2^EV`. Each stop doubles or halves brightness. |
| **Contrast** | -100 to +100 | 0 | Linear stretch on the L-channel (LAB colour space), centered at midpoint. Positive values increase the spread between darks and lights. |
| **Highlights** | -100 to +100 | 0 | Exponential falloff targeting the bright end of the L-channel. Negative values recover blown highlights, positive values lift them further. |
| **Shadows** | -100 to +100 | 0 | Exponential falloff targeting the dark end of the L-channel. Positive values lift shadows, negative values crush them. |

### Shadows / Highlights algorithm

Uses an exponential falloff curve on the L-channel in LAB colour space (ported from the Image Space `editingTools.py` approach):

```
shadow_adjustment = strength × (falloff_power^(1 - L_normalized) - 1)
highlight_adjustment = strength × (falloff_power^L_normalized - 1)
```

The `falloff_power` (default 7.0) controls how sharply the adjustment targets the extreme tones vs affecting the midtones. Higher = more targeted, lower = broader.

The LAB colour space is used so that brightness adjustments don't shift hue or saturation — only the L (lightness) channel is modified, while a and b (colour) channels are preserved.

---

## Colour

Four sliders for white balance and colour intensity.

| Slider | Range | Default | Algorithm |
|--------|-------|---------|-----------|
| **Temperature** | -100 to +100 | 0 | Simplified blackbody RGB multipliers. Positive = warmer (more red, less blue). Negative = cooler. |
| **Tint** | -100 to +100 | 0 | Green/magenta shift via the G-channel multiplier. Positive = more green, negative = more magenta. |
| **Saturation** | -100 to +100 | 0 | Global saturation in HSV colour space. Multiplies the S-channel. -100 = full grayscale, +100 = double saturation. |
| **Vibrance** | -100 to +100 | 0 | Selective saturation — pixels that are already saturated get less boost. The adjustment is proportional to `(1 - current_saturation)`, so muted colours are boosted more than vivid ones. Prevents oversaturation of already-colourful areas. |

### White balance model

The temperature slider uses a simplified blackbody approximation:
- Red multiplier: `1.0 + temperature × 0.004`
- Blue multiplier: `1.0 - temperature × 0.004`
- Multipliers are normalized so the minimum is 1.0 (prevents overall darkening)

This is a perceptual model, not a physically precise colour temperature conversion. For precise CCT-based white balance using Planckian locus chromaticity coordinates, use the Chart-based Colour Correction mode with `colour-science`.

---

## Detail

Collapsed by default. Controls sharpening and noise reduction.

### Sharpening

| Slider | Range | Default | Purpose |
|--------|-------|---------|---------|
| **Sharpen** | 0 – 100 | 0 | Sharpening strength. 0 = no sharpening. |
| **Radius** | 0.5 – 5.0 | 1.0 | Gaussian blur sigma for the unsharp mask. Larger = coarser detail enhanced. |
| **Threshold** | 0 – 100 | 0 | Noise gate. Only edges stronger than the threshold are sharpened, preventing noise amplification. |

**Algorithm:** LAB unsharp mask with adaptive edge protection. The sharpening is applied only to the L-channel (luminance) to avoid colour fringing. An adaptive mask reduces sharpening in extreme shadows (L < 0.1) and highlights (L > 0.9) to prevent artifacts.

### Denoising

| Control | Range / Options | Default | Purpose |
|---------|----------------|---------|---------|
| **Denoise** | 0 – 100 | 0 | Denoising strength. 0 = disabled. |
| **Method** | bilateral, nlmeans, ai_nafnet, ai_scunet | bilateral | Which algorithm to use |

**Methods:**

| Method | Speed (12MP) | Quality | Notes |
|--------|-------------|---------|-------|
| **bilateral** | ~0.06s | Good | Edge-preserving bilateral filter. Fast enough for real-time preview. |
| **nlmeans** | ~0.7–4s | Better | Non-local means (`cv2.fastNlMeansDenoisingColored`). Slower but handles structured noise better. Speed depends on search window size. |
| **ai_nafnet** | ~2–4s | Excellent | NAFNet neural denoiser via `nafnetlib`. Requires PyTorch + CUDA. Auto-downloads model weights on first use. Best for real camera noise (varying ISO). |
| **ai_scunet** | ~5–10s | Excellent | SCUNet via `spandrel`. Blind denoiser — handles unknown noise types without parameters. Requires downloading model `.pth` file separately. |

AI methods appear in the dropdown only when their dependencies are installed. Without PyTorch, the dropdown shows only bilateral and nlmeans.

**Pipeline position:** Denoising is applied as step 2 in the pipeline (before white balance and exposure), so noise is reduced before being amplified by other adjustments.

---

## Corrections

Collapsed by default.

| Slider | Range | Default | Algorithm |
|--------|-------|---------|-----------|
| **CLAHE** | 0 – 4.0 | 0 | Contrast-Limited Adaptive Histogram Equalization on the L-channel. Clip limit controls the maximum contrast amplification. Higher values = more local contrast. 0 = disabled. Uses `cv2.createCLAHE` with 8×8 tile grid. |
| **Vignette fix** | 0 – 2.0 | 0 | Radial brightness gain to counteract lens vignetting (dark corners). Model: `gain = 1 + strength × (r/r_max)²`. 0 = disabled, 1.0 = moderate correction, 2.0 = aggressive. |

---

## Colour Correction

Collapsed by default. Provides two modes for batch colour consistency:

### Mode selector

| Mode | Purpose |
|------|---------|
| **None** | No colour correction applied. |
| **Chart** | Detect a ColorChecker Classic 24 in one image, compute a correction matrix, apply to all images. Requires `colour-science` (for reference patch values and matrix fitting) and `opencv-contrib-python` (for `cv2.mcc` chart detection). |
| **Histogram** | Match the colour histogram of every image to a reference image. Per-channel CDF matching in LAB colour space. No additional dependencies. |

### Chart mode

1. **Select chart image** — Browse to an image containing a visible X-Rite/Calibrite ColorChecker Classic 24.
2. **Choose method and degree:**
   - Methods: `Finlayson 2015` (recommended), `Cheung 2004`, `Vandermonde`
   - Degree: 1 (3×3 matrix), 2 (3×6, recommended), 3 (3×13, risk of overfitting)
3. **Detect Chart** — Uses `cv2.mcc.CCheckerDetector` to find the chart and extract 24 measured patch colours. If detection fails, a manual fallback is needed (not yet implemented — planned).
4. **Fit** — Computes a polynomial colour correction matrix via `colour.characterisation.matrix_colour_correction()`. The matrix is fitted once and applied to every image during preview and export.

**Status label** shows: "Fitted — Finlayson 2015 deg=2 (3×6 matrix)" on success.

### Histogram mode

1. **Select reference image** — Browse to the image whose colour distribution you want all other images to match.
2. **Load Reference** — Reads the image and stores it as the matching target.
3. All images are histogram-matched per-channel in LAB colour space during preview and export.

**Algorithm:** For each channel (L, a, b), compute the cumulative distribution function (CDF) of both source and reference, then build a 256-entry lookup table mapping source values to reference values. Applied via `cv2.LUT`.

---

## Export

### Format and quality

| Control | Options | Default | Notes |
|---------|---------|---------|-------|
| **Format** | jpg, png, tiff | jpg | Output file format. The extension of every exported file is replaced with the selected format, enabling filetype conversion (e.g. DNG → JPG, PNG → TIFF). |
| **Quality** | 1 – 100 | 92 | For JPG: JPEG quality (1 = smallest/worst, 100 = largest/best). For PNG: maps to compression level (higher quality = less compression). TIFF: not applicable (uses LZW compression). |
| **Depth** | 8-bit, 16-bit | 8-bit | Bit depth per channel. 16-bit is available for PNG and TIFF only (JPG is always 8-bit). 16-bit exports preserve the full precision of the float32 processing pipeline. |

### Metadata

| Checkbox | Default | Purpose |
|----------|---------|---------|
| **Copy EXIF metadata** | On | Transfers EXIF data (camera make/model, GPS, exposure settings, lens info) from the source image to the exported file. Uses Pillow's native EXIF support. |
| **Embed sRGB ICC profile** | On | Embeds a standard sRGB ICC colour profile in the exported file. Ensures consistent colour display across applications and monitors. |

**16-bit and metadata:** 16-bit exports (PNG/TIFF) are written via OpenCV to preserve true `uint16` pixel data. EXIF and ICC are not injected into 16-bit exports because Pillow's re-save would truncate the data to 8-bit. The pixel fidelity takes priority — 16-bit exports are typically used for maximum quality in downstream processing, where embedded profiles are less important.

### Batch options

| Checkbox | Default | Purpose |
|----------|---------|---------|
| **Normalize exposures across batch** | Off | Computes the median luminance (L-channel) of every image in the batch, then scales each image so its median matches the batch mean. Reduces exposure variation across a capture session — useful for outdoor datasets or varying flash distances. |
| **Refuse non-empty output** | On | Blocks export if the output folder already contains files. This prevents accidental overwrite or mixing partial datasets. |

### Export workflow

1. Click **Export Adjusted Dataset** to begin. A background thread processes each image:
   - Load source frames one at a time when normalization is off
   - Apply the shared recipe pipeline (LUT → denoise → WB → exposure → shadows/highlights → contrast → saturation/vibrance → sharpen → CLAHE → vignette)
   - Apply colour correction (chart matrix or histogram matching, if active)
   - Optionally normalize exposures with a two-pass statistics workflow
   - Write adjusted derivatives with the selected format
2. **Progress bar** fills during processing.
3. Click **Stop** to cancel. Written output files are removed by default.
4. A successful dataset export writes `adjustment_recipe.json` and `adjustment_manifest.json`.
5. Completed exports are logged to the **Projects tab Recent Activity** feed.

### Video export

When a video file is loaded, Adjust performs a **video sample export**: one frame per second via ffmpeg, adjusted with the active recipe, and written as an image sequence. Use Extract for reconstruction-grade frame selection; video sample export is for quick visual tests and reference outputs.

---

## Processing Pipeline

Adjustments are applied in a fixed order designed to produce the best results (the same order used by professional RAW processors):

```
 1. RAW Development (if RAW input — rawpy postprocess)
 2. Noise Reduction (before amplifying noise with subsequent ops)
 3. White Balance (temperature + tint)
 4. Exposure (multiplicative EV stops)
 5. Shadows / Highlights (LAB exponential falloff)
 6. Contrast (L-channel stretch)
 7. Saturation + Vibrance (HSV)
 8. Colour Correction (chart matrix or histogram matching)
 9. Sharpening (LAB unsharp mask — last, to avoid sharpening noise)
10. Corrections (CLAHE, vignette)
```

When all sliders are at their default values (zero), the pipeline is a verified no-op — the output is identical to the input.

---

## Preview

The Adjust tab uses the shared right-side Preview panel (also used by Extract, Mask, Review, and Coverage). The tab maintains its own preview buffer internally:

1. **Source image** loaded at full resolution (float32 BGR, 0–1 range)
2. **Preview buffer** downsampled to ~2 megapixels (via `cv2.resize` with `INTER_AREA`)
3. On every slider change, the full adjustment pipeline runs on the preview buffer (~5–30ms for most operations at 2MP)
4. Result converted to PIL Image, then rendered as CTkImage in the preview panel

**Debounce:** Slider movements are debounced at 16ms (one frame at 60fps). Moving a slider rapidly produces a single preview update at the end, not one per intermediate value.

**Preview mode:** The tab sets `_preview_mode = "adjust"` when active. This prevents Mask tab's SAM click handlers from firing on the preview image, and distinguishes Adjust's image list from Mask's.

---

## Architecture

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `prep360/core/adjustment_recipe.py` | ~210 | Serializable recipe dataclasses, validation, JSON load/save |
| `prep360/core/color_pipeline.py` | ~170 | Shared recipe application, LUT handoff, image load/write helpers |
| `prep360/core/lut.py` | ~330 | `.cube` parsing and uint8/float LUT application |
| `adjust_workflow.py` | ~350 | Paired dataset detection, safe export, manifests, cancel cleanup |
| `adjust_engine.py` | ~920 | Pure processing library. No GUI dependencies. All functions accept and return numpy arrays. Usable from CLI or tests. |
| `tabs/adjust_tab.py` | ~1210 | Tab builder and UI logic. All state stored on the `app` instance via `_adjust_*` attributes. |

### Engine public API

| Function | Signature | Purpose |
|----------|-----------|---------|
| `apply_adjustments(img, state)` | `float32 BGR → float32 BGR` | Full pipeline compositor |
| `adjust_exposure(img, ev)` | `float32 BGR → float32 BGR` | Single exposure adjustment |
| `adjust_shadows_highlights(img, shadows, highlights)` | `float32 BGR → float32 BGR` | LAB shadow/highlight recovery |
| `adjust_contrast(img, amount)` | `float32 BGR → float32 BGR` | L-channel contrast |
| `adjust_white_balance(img, temperature, tint)` | `float32 BGR → float32 BGR` | RGB multiplier WB |
| `adjust_saturation(img, amount)` | `float32 BGR → float32 BGR` | HSV saturation |
| `adjust_vibrance(img, amount)` | `float32 BGR → float32 BGR` | Selective saturation |
| `adjust_sharpen(img, amount, radius, threshold)` | `float32 BGR → float32 BGR` | LAB unsharp mask |
| `adjust_clahe(img, clip_limit)` | `float32 BGR → float32 BGR` | CLAHE on L-channel |
| `adjust_vignette(img, strength)` | `float32 BGR → float32 BGR` | Radial gain |
| `denoise_image(img, strength, method)` | `float32 BGR → float32 BGR` | Denoise router |
| `load_raw(path, state)` | `path → float32 BGR` | RAW development via rawpy |
| `detect_chart_mcc(img_u8)` | `uint8 BGR → (24,3) or None` | ColorChecker detection |
| `fit_chart_correction(measured, reference)` | `(24,3) × (24,3) → matrix` | Fit correction matrix |
| `apply_chart_correction(img, ccm)` | `float32 BGR → float32 BGR` | Apply fitted matrix |
| `match_histograms(source, reference)` | `float32 BGR → float32 BGR` | LAB histogram matching |
| `normalize_exposures(images)` | `list[float32] → list[float32]` | Batch exposure normalization |
| `probe_video(path)` | `path → dict or None` | ffprobe metadata |
| `extract_single_frame(path, timestamp)` | `path + float → float32 BGR` | Single frame via ffmpeg |

### State dataclasses

| Class | Purpose |
|-------|---------|
| `AdjustmentState` | All slider values (15 parameters). Defaults are all zero/neutral. |
| `RawDevelopState` | RAW development parameters (9 parameters). Defaults are sensible (AHD, sRGB, camera WB). |
| `AdjustDocument` | Nests `AdjustmentState` + `RawDevelopState`. Used for undo stack and preset serialization. |
| `UndoStack` | 50-deep stack of `AdjustDocument` snapshots. Push on slider change, pop on undo. ~10KB total. |

### Gated dependencies

| Feature | Package | Flag | Behaviour when missing |
|---------|---------|------|----------------------|
| RAW development | `rawpy` | `HAS_RAWPY` | RAW section shows "install rawpy" message. RAW files skipped on load/export. |
| Chart correction fitting | `colour-science` | `HAS_COLOUR_SCIENCE` | Chart section shows "install colour-science" message. Detect button logs error. |
| Chart auto-detection | `cv2.mcc` (opencv-contrib) | `HAS_MCC` | `detect_chart_mcc()` returns None. Detection logs "cv2.mcc not available". |
| AI denoise (NAFNet) | `nafnetlib` | `HAS_NAFNET` | `ai_nafnet` not shown in method dropdown. |
| AI denoise (SCUNet) | `spandrel` | `HAS_SPANDREL` | `ai_scunet` not shown in method dropdown. |

---

## Integration with Other Tabs

### Main workflow

**Projects → Extract → Adjust → Mask → Review → Align → Coverage**

Extract selects raw source frames. Adjust authors and applies the visual recipe. Mask, Review, and Align should use the adjusted dataset when one exists.

### Cross-tab connections

| Direction | What happens |
|-----------|-------------|
| **Extract → Adjust** | Extract offers/logs Open in Adjust and populates Adjust with the raw selected output root. |
| **Adjust → Mask** | Mask can consume an adjusted paired root with `front/frames` and `back/frames`; masks are written under the matching lens folders. |
| **Adjust → Align** | Alignment activity records the adjustment manifest when images are inside an adjusted dataset. |

### Activity logging

Completed exports are recorded via `app.record_activity()` and appear in the **Projects tab → Recent Activity** view. Each record includes the operation type (`adjust_export`), input/output paths, image count, and output format.

---

## Performance

Benchmarks on a 12-megapixel image (RTX 3090 Ti):

| Operation | Time | Suitable for |
|-----------|------|-------------|
| Exposure, WB, saturation, contrast | < 0.01s | Real-time slider drag |
| Shadows / highlights (LAB) | ~0.05s | Real-time slider drag |
| Sharpen (LAB unsharp) | ~0.1s | Real-time slider drag |
| CLAHE | 0.03s | Real-time slider drag |
| Bilateral denoise | 0.06s | Real-time slider drag |
| NL-means denoise (search=11) | 0.67s | Slider release |
| NL-means denoise (search=21) | 4.07s | Export only |
| AI denoise (NAFNet) | ~2–4s | Export only |

Preview uses a 2-megapixel downsampled buffer, so all operations feel instant during slider interaction. Full-resolution processing happens only during export.
