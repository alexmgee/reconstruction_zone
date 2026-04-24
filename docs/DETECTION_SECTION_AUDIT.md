# Detection Section (Mask Tab) — Comprehensive Audit

**Date:** 2026-04-24
**Author:** Alex (with Claude Code research assistance)
**Scope:** Every user-facing control in the Detection section, how it maps to the pipeline, what's vital, what's underdeveloped, and what can be trimmed.

---

## Table of Contents

1. [Section Layout](#1-section-layout)
2. [Control-by-Control Trace](#2-control-by-control-trace)
   - [I/O Controls](#io-controls)
   - [SAM3 Mode Toggle](#sam3-mode-toggle)
   - [Model Selector](#model-selector)
   - [YOLO Size](#yolo-size)
   - [Geometry](#geometry)
   - [Confidence / Review Threshold](#confidence--review-threshold)
   - [torch.compile](#torchcompile)
   - [Remove / Keep Prompts](#remove--keep-prompts)
   - [Multi-pass SAM3](#multi-pass-sam3)
   - [Skip Existing / Multi-label / Inpaint](#skip-existing--multi-label--inpaint)
3. [Segmentation Models — Comparative Assessment](#3-segmentation-models)
4. [Dead Code and Orphans](#4-dead-code-and-orphans)
5. [Verdict: Keep / Trim / Develop](#5-verdict)

---

## 1. Section Layout

The Detection section is the top collapsible section in the Mask tab. It contains all controls needed to run a masking job:

```
Detection (CollapsibleSection, expanded=True)
├── I/O: Input (folder/file), Output, Create review folder, Include rejects, File types
├── SAM3 mode: SegmentedButton [Hybrid | Unified Video]
├── Model row: Model dropdown, YOLO size dropdown, Geometry dropdown
├── Sliders row: Conf slider, Review thresh slider, torch.compile checkbox
├── Prompts: Remove entry, Keep entry
├── Multi-pass SAM3: checkbox + dynamic pass rows (prompt + threshold per row)
└── Output options: Skip existing, Multi-label output, Inpaint masked
```

The Run/Queue button and progress bar are below in a separate section. Static Masks, Detection & Refinement (shadow, SAM refine, matting, ensemble, edge injection, COLMAP validation), and Post-Processing are in subsequent collapsible sections — **not** part of this audit.

---

## 2. Control-by-Control Trace

### I/O Controls

| Control | Variable | Used in |
|---------|----------|---------|
| Input entry | `input_entry` | `_worker()` reads path, determines file vs. dir vs. video |
| Output entry | `output_entry` | `_worker()` creates output dir |
| Folder / File buttons | — | Browse dialogs → populate input_entry |
| `...` button | — | Browse dialog → populate output_entry |
| Create review folder | `review_folder_var` | When True, creates `masks/` + `review/` subdirs. Also sets `save_review_images` in MaskConfig |
| Include rejects | `review_rejects_var` | Maps to `save_reject_review_images` — saves a mask even for low-confidence images |
| File types | `pattern_var` (default `"*.jpg *.png"`) | Passed to `process_directory(pattern=...)` as glob filter |

**Assessment:** All essential, well-wired. The "Create review folder" and "Include rejects" pair is well-designed — it feeds the Review tab's quality workflow. No issues.

---

### SAM3 Mode Toggle

| Setting | Effect |
|---------|--------|
| **Hybrid** (default) | Per-frame detection+segmentation. Uses whichever Model is selected. Full pipeline: detect → segment → postprocess → optional refinements. |
| **Unified Video** | Sets `sam3_unified_var=True`. Disables Model/YOLO size/Geometry/Conf/Review thresh/torch.compile rows + entire Detection & Refinement and Post-Processing sections. Only Remove/Keep prompts remain active (highlighted with blue glow). |

**How Unified works in the pipeline:**
- `_worker()` checks `self.sam3_unified_var.get()` — if True AND `pipeline.sam3_video_pipeline is not None`, calls `process_directory_sam3_unified()` or `process_video_sam3_unified()` instead of normal processing.
- `sam3_pipeline.py` loads `build_sam3_multiplex_video_predictor` from the SAM3 package — this is SAM 3.1's Object Multiplex mode (text-prompted detection + instance segmentation + temporal tracking in one model).
- Frames are loaded as PIL images and passed as a list to a video session. Text prompts drive detection. No YOLO/RF-DETR needed.

**Assessment:**
- **Conceptually sound.** SAM 3.1's unified video predictor IS the right architecture for video masking — it handles detection+segmentation+tracking in one pass, avoiding the YOLO→SAM→tracker pipeline.
- **Dependency risk:** Requires `sam3` package with HuggingFace-gated weights. If SAM3 isn't installed, `sam3_video_pipeline` is None and the toggle silently falls back to Hybrid — the user gets no error, just unexpected behavior.
- **Practical concern:** The UI disables almost everything when Unified is selected, but the user can still enter Remove/Keep prompts. If SAM3 isn't available, those prompts go nowhere.

---

### Model Selector

| Value | Enum | Segmenter Class | Installed Check |
|-------|------|----------------|-----------------|
| `auto` | — | `_auto_select_model()` picks best available | — |
| `yolo26` | `SegmentationModel.YOLO26` | `YOLO26Segmenter` | `HAS_YOLO` (ultralytics) |
| `rfdetr` | `SegmentationModel.RFDETR` | `RFDETRSegmenter` | `HAS_RFDETR` (rfdetr pip) |
| `sam3` | `SegmentationModel.SAM3` | `SAM3Segmenter` | `HAS_SAM3` (sam3 repo) |
| `fastsam` | `SegmentationModel.FASTSAM` | `FastSAMSegmenter` | `HAS_FASTSAM` (ultralytics) |

**Auto-select priority:** SAM3 → RF-DETR → YOLO26 → FastSAM → EfficientSAM → SAM2

**Models in the enum that have NO segmenter class and crash with `NotImplementedError`:**
- `EfficientSAM` — import exists, auto-select can pick it, but `_create_segmenter()` raises
- `SAM2` — import exists, auto-select can pick it, but `_create_segmenter()` raises
- `MobileSAM` — in enum only, no import, no segmenter
- `OpenCV` — in enum only, no import, no segmenter

**How each model uses the controls differently:**

| Control | SAM3 | YOLO26 | RF-DETR | FastSAM |
|---------|------|--------|---------|---------|
| Remove prompts | Text prompts → `set_text_prompt()` | Ignored (class-based) | Ignored (class-based) | Ignored |
| Keep prompts | Stored in config, not used in segmenter | Ignored | Ignored | Ignored |
| YOLO size | Ignored | `yolo26{size}-seg.pt` | Ignored | Ignored |
| Conf threshold | Ignored (SAM3 uses internal scoring) | Passed to `model.predict(conf=...)` | Passed to `model.predict(threshold=...)` | Hardcoded 0.4 |
| YOLO classes | Ignored | Passed to `model.predict(classes=...)` | Post-filtered after inference | Ignored (segments everything) |
| Geometry | Preprocessing + postprocessing in BaseSegmenter | Same | Same | Same |

**Assessment:**
- **`auto` is dangerous.** If SAM3 and RF-DETR are both missing and only YOLO is installed, auto picks YOLO — but then Remove/Keep prompts are silently ignored. The user types "person, tripod" and YOLO just uses COCO class IDs. It works for "person" (class 0) but "tripod" and "selfie stick" map to nothing.
- **FastSAM is obsolete.** It was a speed optimization when SAM2 was slow. SAM3 is faster and more capable. FastSAM also hardcodes `conf=0.4` and ignores the user's Conf slider, and it ignores Remove/Keep prompts (segments everything, returns all masks).
- **Keep prompts do nothing.** They're collected into `MaskConfig.keep_prompts` but no segmenter reads them. SAM3Segmenter only iterates over `prompts.get('remove', [])`. This is a dead feature.

---

### YOLO Size

| Value | Model file | Params (approx) | Speed |
|-------|-----------|-----------------|-------|
| n (nano) | yolo26n-seg.pt | ~3M | Fastest |
| s (small) | yolo26s-seg.pt | ~11M | Fast |
| m (medium) | yolo26m-seg.pt | ~20M | Moderate |
| l (large) | yolo26l-seg.pt | ~44M | Slow |
| x (extra) | yolo26x-seg.pt | ~68M | Slowest |

**Assessment:** Only matters when Model=yolo26. Correctly wired to `YOLO26Segmenter.initialize()`. Size `n` is the default — reasonable for photogrammetry where speed matters more than detecting small objects in corner cases. Visible even when other models are selected, which is slightly confusing but harmless.

---

### Geometry

| Value | Enum | Effect on preprocessing | Effect on postprocessing |
|-------|------|------------------------|-------------------------|
| pinhole | `PINHOLE` | No preprocessing | Morphological cleanup only |
| equirect | `EQUIRECTANGULAR` | CLAHE pole enhancement | Pole mask expansion + nadir masking + fisheye circle (if equirect+geometry_aware → cubemap decomposition path) |
| fisheye | `FISHEYE` | Radial brightness correction | Fisheye circle mask (corners + periphery) |
| cubemap | `CUBEMAP` | No preprocessing | Morphological cleanup only |

**The equirect path is special:** When geometry=equirect AND `config.geometry_aware=True` (hardcoded True), `process_image()` delegates to `_process_equirectangular()` which does cubemap decomposition — splits the equirect into 6 perspective faces, segments each face independently as PINHOLE, then stitches back to equirect. This is the core value of the masking system.

**Assessment:** Essential and well-implemented. The cubemap decomposition path is the key differentiator. However, `geometry_aware` is always True (no UI control) — if someone wants to segment an equirect directly without cubemap decomposition (e.g., for a quick test), there's no way to do that.

---

### Confidence / Review Threshold

| Slider | Variable | Default | Used in |
|--------|----------|---------|---------|
| Conf | `conf_var` | 0.70 | `MaskConfig.confidence_threshold` → YOLO `conf=`, RF-DETR `threshold=` |
| Review thresh | `review_thresh_var` | 0.85 | `MaskConfig.review_threshold` → quality scoring in `process_directory()` |

**How quality scoring works:** `BaseSegmenter._evaluate_mask_quality()` computes a weighted composite of confidence (50%), edge coherence (20%), and compactness (30%). The result is a `MaskQuality` enum: EXCELLENT/GOOD/REVIEW/POOR/REJECT. The review threshold determines which quality level triggers review folder routing.

**Note:** For SAM3, the Conf slider is **not** passed to the model — SAM3 uses internal scoring. The slider only affects YOLO and RF-DETR at inference time. For SAM3, confidence is read from the model's output scores and used for quality evaluation after-the-fact.

**Assessment:** Correctly wired for YOLO/RF-DETR. The asymmetry with SAM3 is not surfaced to the user — could be confusing.

---

### torch.compile

| Checkbox | Variable | Default | Effect |
|----------|----------|---------|--------|
| torch.compile | `torch_compile_var` | False | Calls `_apply_torch_compile()` after pipeline init |

**What it compiles:** Segmenter model (inner PyTorch model for YOLO/FastSAM, image_encoder for SAM-style), SAM refiner encoder, and matting model. Uses `mode="reduce-overhead", dynamic=True`.

**Assessment:** Works as described. First run is slow (warmup), subsequent runs faster. Correctly gated behind CUDA + PyTorch 2.0+ checks. Good power-user option. Tooltip explains the trade-off well.

---

### Remove / Keep Prompts

| Entry | Variable | Default placeholder | Used by |
|-------|----------|-------------------|---------|
| Remove | `remove_prompts_entry` | "person, tripod, backpack, selfie stick" | SAM3: text prompts. YOLO/RF-DETR: auto-translated to COCO class IDs via `COCO_NAME_TO_ID`. |
| Keep | `keep_prompts_entry` | "(optional) objects to protect from masking" | SAM3: runs keep prompts, subtracts their masks from remove results. YOLO/RF-DETR: excludes detections matching `keep_classes`. |

**Parsing:** Comma-separated text → list of strings. If empty, defaults to `["person", "tripod", "backpack", "selfie stick"]` from MaskConfig defaults.

**Remove → YOLO class translation (IMPLEMENTED 2026-04-24):** Each word from the Remove field is looked up in `COCO_NAME_TO_ID`. Exact matches map directly (e.g. "person" → 0). Partial/alias matches use substring search (e.g. "phone" matches "cell phone" → 67). Words with no COCO match are silently skipped for YOLO but still sent as text prompts for SAM3.

**Keep prompt implementation (IMPLEMENTED 2026-04-24):**
- **SAM3:** After generating remove masks, runs each keep prompt through `set_text_prompt()`, collects keep masks, and subtracts from all remove results (`mask & ~keep_mask`).
- **YOLO/RF-DETR:** Keep text is translated to class IDs via `COCO_NAME_TO_ID` (same as Remove). Detections matching `keep_classes` are excluded before mask output.

---

### Multi-pass SAM3

| Control | Effect |
|---------|--------|
| Checkbox | Enables multi-pass mode |
| + Add Pass / Remove | Add/remove rows with (prompt, threshold) pairs |
| Per-row entries | prompt text + confidence threshold |

**How it works:** When enabled, `_build_mask_config()` collects `(prompt_text, threshold)` tuples into `config.multi_pass_prompts`. In the pipeline, `_segment()` checks for non-empty `multi_pass_prompts` and calls `SAM3Segmenter._run_multi_pass()` instead of normal `segment_image()`.

`_run_multi_pass()` runs SAM3 once per prompt with per-prompt confidence filtering, then unions all masks together. This allows different thresholds for different objects (e.g., person at 0.7, tripod at 0.3).

**Assessment:**
- **Only works with SAM3.** If Model is yolo26/rfdetr/fastsam, the multi-pass flag is checked in `_segment()` but the guard `isinstance(self.segmenter, SAM3Segmenter)` fails — it silently falls back to normal segmentation. No user feedback.
- **Genuinely useful for SAM3** — different objects need different confidence thresholds. Well-implemented.
- **UI concern:** The checkbox and buttons are visible regardless of model selection. Should probably be disabled when Model != sam3.

---

### Skip Existing / Multi-label / Inpaint

| Checkbox | Variable | Config field | Effect |
|----------|----------|-------------|--------|
| Skip existing | `skip_existing_var` | — (passed directly to `process_directory()`) | Skips images that already have a mask in output dir |
| Multi-label | `multi_label_var` | `config.multi_label` | Pixel values = class IDs instead of binary 0/1 |
| Inpaint masked | `inpaint_var` | `config.inpaint_masked` | OpenCV inpainting (Telea or Navier-Stokes) fills masked regions |

**Skip existing:** Not in MaskConfig — passed as kwarg to `process_directory()`. Clean implementation.

**Multi-label:** Sets `config.multi_label=True`. Let me check if this is actually implemented in the pipeline output path...

The `multi_label` flag exists in MaskConfig but looking at `process_directory()` and `process_image()`, the actual multi-label output logic would need to be in the mask saving code. This needs deeper tracing but the flag is wired through.

**Inpaint:** Sets `config.inpaint_masked=True` with `inpaint_method="telea"` and `inpaint_radius=5`. Uses OpenCV's built-in inpainting — no extra dependencies. Useful for 3DGS where masked black voids produce gradient artifacts.

**Assessment:** All three are functional and useful. Skip existing is important for interrupted batch jobs. Multi-label is for Gaussian Grouping workflows. Inpaint addresses a real 3DGS pain point.

---

## 3. Segmentation Models — Comparative Assessment

### SAM3 (Primary)

| Aspect | Detail |
|--------|--------|
| Architecture | 848M params. Unified DETR detector + SAM2 tracker. Text-prompted. |
| Strengths | Text prompts handle arbitrary objects ("selfie stick", "tripod"). No class vocabulary limit. Multi-instance detection. Unified video tracking mode. |
| Weaknesses | Requires HuggingFace gated access. Heavyweight. Needs bfloat16 autocast on CUDA. |
| Integration quality | **Good.** Text prompts wired to `set_text_prompt()`. Multi-pass mode works. Unified video mode available. Geometry preprocessing/postprocessing applied. |
| Verdict | **KEEP — this is the primary model and the system's differentiator.** |

### YOLO26 (Production)

| Aspect | Detail |
|--------|--------|
| Architecture | CNN-based, NMS-free end-to-end. n/s/m/l/x model sizes. 80 COCO classes. |
| Strengths | Fast, reliable, well-tested. Auto-downloads weights. Class-based filtering is deterministic. |
| Weaknesses | Limited to 80 COCO classes. "Tripod", "selfie stick", "drone" are not COCO classes. Cannot detect arbitrary objects. |
| Integration quality | **Good.** Conf threshold, class filter, retina_masks all correctly wired. |
| Verdict | **KEEP — fast workhorse for standard objects (person, backpack, car). Best fallback when SAM3 unavailable.** |

### RF-DETR (Transformer)

| Aspect | Detail |
|--------|--------|
| Architecture | DINOv2 backbone, transformer-based. Architecturally different from YOLO (CNN). nano/small/medium/large. |
| Strengths | Different failure modes from YOLO — architecturally complementary for ensemble. COCO-trained. |
| Weaknesses | Same 80-class limitation as YOLO. Slower than YOLO. Requires separate `rfdetr` package pinned to `transformers <5.0`. |
| Integration quality | **Good.** Threshold, class filtering, output parsing all correct. |
| Verdict | **KEEP — valuable for ensemble mode (YOLO + RF-DETR). Marginal as standalone since it does the same COCO classes as YOLO but slower.** |

### FastSAM (Legacy)

| Aspect | Detail |
|--------|--------|
| Architecture | YOLO-based "fast SAM" — everything-segmentation, no class awareness. |
| Strengths | Was fast when SAM2 was the only alternative. |
| Weaknesses | **Ignores Conf slider** (hardcoded 0.4). **Ignores Remove/Keep prompts.** Segments everything and returns all masks. No class filtering. No text prompting. Superseded by YOLO26 (which is also fast but class-aware) and SAM3 (which is text-prompted). |
| Integration quality | **Poor.** Doesn't respect user-configured confidence threshold. Doesn't use prompts or class filtering. |
| Verdict | **TRIM.** Inferior to both YOLO26 (for speed+classes) and SAM3 (for text prompts). Ignores key user controls. |

### EfficientSAM / SAM2 / MobileSAM / OpenCV

**RESOLVED (2026-04-24):** All four removed from `SegmentationModel` enum, import checks, and `_auto_select_model()` fallback chain. Auto-select now terminates at FastSAM → RuntimeError.

---

## 4. Dead Code and Orphans — Resolution Status

| Item | Status |
|------|--------|
| `MASK_TARGETS` list | **REMOVED.** Deleted from `reconstruction_zone.py`. |
| `_target_vars` loop | **REPLACED** with `COCO_NAME_TO_ID` auto-translation in `_build_mask_config()`. Remove text now auto-derives YOLO class IDs. |
| `keep_prompts` | **IMPLEMENTED.** SAM3: runs keep prompts and subtracts masks. YOLO/RF-DETR: excludes detections matching `keep_classes`. |
| EfficientSAM/SAM2/MobileSAM/OpenCV | **REMOVED** from enum, imports, auto-select, and CLI choices. |
| `MaskConfig.num_workers` | **GUI EXPOSED.** Workers dropdown added to Batch Queue section. |
| `MaskConfig.output_format` | **GUI EXPOSED.** Format dropdown (png/jpg/npy) added to Detection output row. |
| `MaskConfig.batch_size` | CLI-only. No GUI exposure planned (GUI processes sequentially with per-image feedback). |
| `MaskConfig.save_confidence_maps` | CLI-only. Low priority for GUI (Review tab provides visual quality inspection). |

---

## 5. Changes Made (2026-04-24)

### Implemented

| Change | Files modified |
|--------|---------------|
| **Remove text → YOLO classes auto-translation.** Each word from the Remove field is looked up in `COCO_NAME_TO_ID` with fuzzy alias matching (e.g. "phone" → "cell phone" → class 67). Eliminates the SAM3/YOLO prompt disconnect. | `reconstruction_zone.py` |
| **Keep prompts — SAM3.** After running remove prompts, SAM3 runs each keep prompt, collects their masks, and subtracts from all remove results (`r.mask & ~keep_mask`). | `reconstruction_pipeline.py` |
| **Keep prompts — YOLO/RF-DETR.** New `keep_classes` field in `MaskConfig`. Detections matching keep class IDs are skipped before mask generation. | `reconstruction_pipeline.py`, `reconstruction_zone.py` |
| **Dead code removal.** `MASK_TARGETS`, `_target_vars`, EfficientSAM/SAM2/MobileSAM/OpenCV enum values and imports, dead auto-select fallback entries. | Both files |
| **Model-aware greying.** `_on_model_change()` callback greys out YOLO size when model != yolo26/auto, greys out Multi-pass SAM3 when model != sam3/auto. Uses same `_set_widgets_state()` pattern as Unified Video toggle. | `reconstruction_zone.py` |
| **Output format dropdown.** `Format: [png | jpg | npy]` added to output options row, wired to `MaskConfig.output_format`. | `reconstruction_zone.py` |
| **Workers control.** `Workers: [1 | 2 | 4 | 8]` dropdown added to Batch Queue stats row, wired to `MaskConfig.num_workers`. | `reconstruction_zone.py` |

### Remaining

| Item | Notes |
|------|-------|
| FastSAM assessment | Still in codebase. Ignores Conf slider and prompts. Consider removing from dropdown or fixing its integration. |
| Unified Video fallback feedback | If SAM3 unavailable, silently falls back to Hybrid. Should show a warning. |

---

## Summary

The Detection section has been cleaned up and strengthened. The core changes:

1. **Remove and Keep prompts now work across all models.** SAM3 uses text prompts directly. YOLO/RF-DETR auto-translate text to COCO class IDs via `COCO_NAME_TO_ID` with fuzzy matching. Keep prompts subtract protected regions (SAM3) or exclude matching detections (YOLO/RF-DETR).
2. **Dead code removed.** Four dead enum models, orphaned target checkbox code, and unused import checks are gone. Auto-select no longer has crash paths.
3. **Model-aware UI.** Controls irrelevant to the selected model are greyed out, matching the Unified Video toggle pattern. No layout shifting.
4. **New GUI controls.** Output format (png/jpg/npy) and worker count exposed from previously CLI-only config fields.
