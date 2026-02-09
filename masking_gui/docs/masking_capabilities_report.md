# Current Masking Capabilities Report

An inventory and assessment of all masking tools, methods, and techniques across the codebase.

---

## 1. Systems Overview

There are **two independent masking systems** in the project, plus a shared mask editor:

| System | Entry point | Purpose |
|--------|-------------|---------|
| **Masking Studio** (advanced) | `masking_gui/masking_studio.py` | Multi-model, multi-geometry, text-prompted, temporal, shadow-aware |
| **Panoex Segmenter** (simple) | `panoex/core/segmenter.py` | YOLO-only, perspective images, batch-with-progress |
| **Mask Editor** (shared) | `masking_gui/claude/review_masks.py` | OpenCV interactive editor — brush, flood fill, lasso |

The two segmentation systems share no code. They duplicate COCO class maps and class presets independently.

---

## 2. Masking Studio Pipeline (`masking_v2.py`)

### 2a. Segmentation Models

Six models registered; three have concrete implementations:

| Model | Class | Status | Prompting | Notes |
|-------|-------|--------|-----------|-------|
| **SAM3** | `SAM3Segmenter` | Implemented | Text prompts | Primary model; `build_sam3` + text encoder |
| **YOLO26** | `YOLO26Segmenter` | Implemented | Class IDs | Production recommendation; auto-downloads |
| **FastSAM** | `FastSAMSegmenter` | Implemented | None (auto-segment) | Fast fallback; YOLO-based |
| EfficientSAM | — | Registered only | — | Import checked, no subclass |
| SAM2 | — | Registered only | — | Import checked, no subclass |
| MobileSAM | — | Registered only | — | Enum exists, not in factory |
| OpenCV | — | Registered only | — | Enum exists, not in factory |

**Auto-select fallback chain** (line 1011–1026):
```
SAM3 → YOLO26 → FastSAM → EfficientSAM → SAM2 → RuntimeError
```

**Gap**: EfficientSAM, SAM2, MobileSAM, and OpenCV are all registered in the enum and/or import checks but have no `BaseSegmenter` subclass. If SAM3, YOLO26, and FastSAM are all unavailable, the pipeline fails despite supposedly having fallbacks.

### 2b. BaseSegmenter ABC (line 285)

Abstract interface all segmenters implement:

| Method | Purpose |
|--------|---------|
| `initialize()` | Load model/weights |
| `segment_image(image, prompts, geometry)` → `List[MaskResult]` | Core segmentation |
| `preprocess_image(image, geometry)` | Geometry-aware preprocessing |
| `postprocess_mask(mask, geometry)` | Pole expansion, morphological cleanup |
| `_evaluate_mask_quality(mask, confidence)` | Multi-metric quality scoring |
| `_merge_similar_masks(results)` | Per-prompt mask union (SAM3 only) |
| `_morphological_cleanup(mask)` | Open → Close → remove small components |
| `_expand_pole_masks(mask)` | Dilate masks in top/bottom 10% of equirect |

### 2c. Quality Scoring System (line 386–442)

Multi-metric composite score on every mask:

| Metric | Weight | How computed |
|--------|--------|--------------|
| **Model confidence** | 50% | Raw score from detector (0–1) |
| **Edge coherence** | 20% | Canny edge ratio vs expected perimeter — smoother = higher |
| **Compactness** | 30% | Isoperimetric ratio `4πA/P²` — regular shapes score higher |

Hard reject gates:
- Mask area < `min_mask_area` (default 100px) → REJECT
- Mask area > `max_mask_area_ratio` of image (default 50%) → REJECT

Score thresholds: ≥0.85 = EXCELLENT, ≥0.70 = GOOD, ≥0.55 = REVIEW, ≥0.35 = POOR, else REJECT

**Gap**: Edge coherence penalizes jagged edges, which is useful for person segmentation but may unfairly penalize legitimate complex-boundary objects (trees, hair, equipment with irregular silhouettes). No per-class quality adjustment.

### 2d. Geometry-Aware Processing

| Geometry | Preprocessing | Postprocessing | Processing path |
|----------|---------------|----------------|-----------------|
| **Pinhole** | None | Morphological cleanup | Direct segmentation |
| **Equirectangular** | CLAHE on pole regions | Pole mask expansion (1.2x) | Cubemap decomposition: 6 faces → segment each → merge |
| **Fisheye** | Radial brightness boost | Morphological cleanup | Direct segmentation |
| **Dual Fisheye** | — | — | Enum exists, no handling |
| **Cubemap** | — | — | Enum exists, no special handling |

**Cubemap strategy** (line 1106–1175): Converts equirect → 6 perspective faces → segments each face with PINHOLE geometry → merges face masks back to equirect via spherical projection. This is a significant capability — standard ML models are trained on perspective images, so this produces far better results than direct equirect segmentation.

`CubemapProjection` class (line 822–957): Full bidirectional equirect ↔ cubemap conversion with proper spherical coordinate math. 6 faces: front, back, left, right, up, down.

**Gap**: Dual fisheye geometry has an enum value but no processing path. Cubemap geometry uses the direct segmentation path rather than any special handling.

### 2e. Temporal Consistency (line 766–819)

Smooths masks across video sequences using confidence-weighted voting:

- Maintains a sliding window (default 5 frames) of mask history
- Each frame's mask weighted by its confidence score
- Pixels in the smoothed mask must exceed 50% weighted vote
- `detect_inconsistency()`: checks IoU between consecutive frames; < 0.3 triggers flag

**Gap**: The inconsistency detection method exists but is never called — nothing in the pipeline acts on it. Temporal consistency uses simple confidence weighting which doesn't account for camera motion, scene changes, or objects entering/leaving the frame.

### 2f. Shadow Detection

**Built-in heuristic** (`_detect_shadows`, line 1177–1215):
- Converts to grayscale, computes local brightness ratio: `pixel / local_median(51×51)`
- Shadow candidates = pixels where ratio < 0.7
- Morphological closing (7×7 kernel) to bridge gaps
- Dilates person mask downward (15×40 rect kernel) to find attachment zone
- Keeps only shadow pixels overlapping the downward extension of person masks
- Removes fragments smaller than 5% of person area

**Extended shadow pipeline** (`shadow_detection.py` — newly created):
- Pluggable detector framework: Brightness, C1C2C3 chromaticity, Hybrid, SDDNet, SILT, and future slots
- 4-stage pipeline: Primary detection → Verification → Spatial filtering → Cleanup
- Three spatial modes: ALL, NEAR_OBJECTS (dilation proximity), CONNECTED (flood-fill from boundary)

See `shadow_detection_analysis.md` and `shadow_detection_research.md` for full details.

### 2g. Mask Combination (line 1217–1264)

When multiple masks are detected (e.g., multiple people), they're combined via confidence-weighted accumulation:
- Each valid mask contributes proportionally to its confidence
- Pixel is included if weighted accumulation > 0.3 (low threshold — biases toward inclusion)
- Final confidence = average of all valid masks

### 2h. Batch Processing

Three batch modes:

| Mode | Method | Parallelism |
|------|--------|-------------|
| **Directory** | `process_directory()` | ThreadPoolExecutor, `num_workers` threads |
| **Video** | `process_video()` | Sequential (frame-by-frame, maintains temporal state) |
| **Image list** | `process_batch()` | ThreadPoolExecutor or sequential |

All produce per-image statistics, review images (red overlay), and summary JSON.

### 2i. Configuration (`MaskConfig`, line 156–263)

28 configuration fields covering:
- Model selection + checkpoint path
- Text prompts (remove list + keep list)
- YOLO class selection (via presets or custom class IDs)
- Quality thresholds (confidence, review, area bounds)
- Temporal consistency toggle + window size
- Geometry awareness toggle + distortion handling + pole expansion factor
- Shadow detection toggle + shadow config dict
- Output format (png/jpg/npy) + review image saving

Serializable to JSON/YAML via `to_dict()`, `save()`, `load()`.

---

## 3. Panoex Segmenter (`panoex/core/segmenter.py`)

Simpler, self-contained YOLO segmenter used by the main Panoex GUI:

| Feature | Details |
|---------|---------|
| **Model** | YOLO only (model name configurable, default `yolo11n-seg.pt`) |
| **Prompting** | Class ID list (COCO classes, same presets as masking_v2) |
| **Config** | `SegmentConfig`: model, classes, confidence, IoU, dilate_pixels, invert_mask, device |
| **Output** | Binary mask + detection list per image |
| **Post-processing** | Optional dilation, optional inversion (COLMAP-style white=exclude) |
| **Batch** | `segment_batch()` with progress callback |
| **Geometry** | None — assumes perspective/pinhole only |
| **Quality scoring** | None |
| **Temporal** | None |
| **Shadow detection** | None |

**Gap**: No quality scoring, no geometry awareness, no temporal consistency, no shadow detection, no text prompting. Essentially a thin wrapper around YOLO inference with mask dilation.

**Overlap**: Duplicates COCO_CLASSES, CLASS_PRESETS, and the basic YOLO inference pattern from masking_v2.py with no code sharing.

---

## 4. Mask Editor (`claude/review_masks.py`)

OpenCV-based interactive mask editor with rich editing tools:

### 4a. Editing Tools

| Tool | Activation | Technique |
|------|-----------|-----------|
| **Brush** | Left click = add, Right click = remove | Circle draw at cursor, configurable size |
| **Flood fill (color)** | Shift+click | Edge-guided LAB flood fill with configurable tolerance |
| **Flood fill (brightness)** | `b` toggle + Shift+click | Relative darkness ratio vs local neighborhood (51×51 median) |
| **Lasso + Otsu** | Ctrl+drag | Draw polygon, Otsu threshold within, keep largest component |
| **Undo** | `u` key | Full mask history stack |
| **Reset** | `r` key | Revert to original mask |

### 4b. View Modes

| Mode | Key | Description |
|------|-----|-------------|
| Overlay | Default | Image + red mask + green contours + yellow proximity highlight |
| Original | `v` cycle | Raw image only |
| Mask only | `v` cycle | White-on-black mask |
| A/B flicker | `a` toggle | Alternates overlay/original every ~500ms for edge verification |

### 4c. Navigation

- **Zoom**: Scroll wheel, cursor-centered, 1x–8x range
- **Pan**: Middle mouse drag
- **Coordinate transform**: Full display↔image coordinate mapping for all zoom levels

### 4d. Status Bar

Real-time display: brush size, flood tolerance, zoom %, fill mode, view mode, mask coverage %

### 4e. Workflow Integration

- Called per-image from the Review tab of Masking Studio
- Returns `save` / `skip` / `quit` status
- Saves edited mask back to the masks directory

**Gap**: The flood fill brightness mode and the `_detect_shadows` heuristic use the exact same algorithm (relative darkness ratio) but are implemented independently. The lasso Otsu tool is powerful for shadows but requires manual selection — no option to auto-apply across similar frames.

---

## 5. Review System

### 5a. Review GUI (`review_gui.py`)

Provides reusable components used by both standalone and embedded review:
- `ThumbnailWidget`: Clickable thumbnail with quality border color
- `load_overlay_thumbnail()`: Composites image + mask into red-overlay thumbnail
- `compute_mask_area_percent()`: Quick area metric for sorting
- Color maps: `QUALITY_COLORS` (green→red by quality), `STATUS_COLORS` (blue=edited, green=accepted, red=rejected)

### 5b. Review Status Manager (`review_status.py`)

Persistent review tracking via `review_status.json`:

| Field | Purpose |
|-------|---------|
| `status` | pending / accepted / edited / rejected / skipped |
| `quality` | excellent / good / review / poor / reject |
| `confidence` | Model confidence score |
| `area_percent` | Mask coverage percentage |
| `edited_at` | Timestamp of last edit |
| `action_history` | Timestamped audit trail of all review actions |

Auto-saves on every status change. Survives app restarts.

### 5c. Masking Studio Review Tab

Embedded review with:
- Path selectors for masks/images directories (+ auto-detect from output)
- Paginated thumbnail grid (60 per page)
- Sort by: filename, confidence, quality, area %
- Filter by: all, needs review, poor, unreviewed
- 2-pane preview: overlay (left) + mask-only (right)
- Info panel: stem, confidence, quality, area %, status
- Actions: Accept, Reject, launch OpenCV editor

---

## 6. Techniques Inventory

### Currently Implemented

| Category | Technique | Where | Maturity |
|----------|-----------|-------|----------|
| **Instance segmentation** | SAM3 text-prompted | masking_v2.py:483 | Full |
| **Instance segmentation** | YOLO26 class-based | masking_v2.py:683 | Full |
| **Instance segmentation** | FastSAM auto-segment | masking_v2.py:620 | Full |
| **Instance segmentation** | YOLO11 class-based | segmenter.py:94 | Full (separate system) |
| **Geometry projection** | Equirect→cubemap decomposition | masking_v2.py:822 | Full |
| **Geometry preprocessing** | Pole CLAHE enhancement | masking_v2.py:324 | Full |
| **Geometry preprocessing** | Fisheye radial enhancement | masking_v2.py:352 | Full |
| **Geometry postprocess** | Pole mask expansion | masking_v2.py:444 | Full |
| **Quality scoring** | Confidence + edge coherence + compactness | masking_v2.py:386 | Full |
| **Temporal consistency** | Confidence-weighted sliding window | masking_v2.py:766 | Basic |
| **Shadow detection** | Relative brightness heuristic | masking_v2.py:1177 | Basic |
| **Shadow detection** | C1C2C3 chromaticity invariant | shadow_detection.py | New |
| **Shadow detection** | Hybrid intensity + chromaticity | shadow_detection.py | New |
| **Shadow detection** | SDDNet / SILT ML stubs | shadow_detection.py | Stub |
| **Mask combination** | Confidence-weighted union | masking_v2.py:1217 | Full |
| **Mask cleanup** | Morphological open/close + fragment removal | masking_v2.py:463 | Full |
| **Mask editing** | Brush painting | review_masks.py:172 | Full |
| **Mask editing** | LAB flood fill | review_masks.py:231 | Full |
| **Mask editing** | Brightness flood fill | review_masks.py:236 | Full |
| **Mask editing** | Lasso + Otsu threshold | review_masks.py:177 | Full |
| **Mask dilation** | Configurable pixel dilation | segmenter.py:172 | Full (panoex only) |
| **Mask inversion** | COLMAP-style invert | segmenter.py:180 | Full (panoex only) |

### Registered but Not Implemented

| Model/Feature | Where registered | What's missing |
|---------------|-----------------|----------------|
| EfficientSAM | masking_v2.py:66–69 | `BaseSegmenter` subclass |
| SAM2 | masking_v2.py:72–76 | `BaseSegmenter` subclass |
| MobileSAM | masking_v2.py:143 | Enum only, no import check, no subclass |
| OpenCV fallback | masking_v2.py:144 | Enum only, no subclass |
| Dual fisheye geometry | masking_v2.py:131 | No processing path |
| Temporal inconsistency action | masking_v2.py:801 | Detection exists, never acted upon |
| Shadow ML detectors | shadow_detection.py | SDDNet/SILT stubs, need architecture + weights |

---

## 7. Gaps and Improvement Opportunities

### High Impact

1. **Edge/boundary refinement**: No matting, alpha estimation, or edge refinement anywhere. Masks are hard binary. Hair, fur, semi-transparent objects all get aliased hard edges. Techniques: guided filter matting, trimap-based alpha matting, deep image matting (DIM, MODNet).

2. **Mask dilation/feathering in the advanced pipeline**: The panoex segmenter has configurable dilation, but masking_v2.py does not expose any dilation parameter. Users can't easily expand masks to ensure coverage of boundary regions (critical for photogrammetry where even a few unmasked pixels cause artifacts).

3. **No adaptive/per-class confidence thresholds**: A single confidence threshold applies to all classes. People (well-trained) may work fine at 0.7 while backpacks (weaker training data) may need 0.5. Same issue with quality scoring — compactness penalizes irregular objects unfairly.

4. **Temporal consistency is primitive**: Simple confidence-weighted voting doesn't handle camera motion, entering/leaving objects, or scene cuts. Optical flow-guided propagation, tracking-based consistency (e.g., ByteTrack, BoT-SORT), or SAM2's video propagation would be far more robust.

5. **No mask propagation**: Each frame is segmented independently. If a person is missed in frame N but detected in frames N-1 and N+1, there's no mechanism to propagate/interpolate the mask. Video object segmentation (VOS) models like SAM2, XMem, or Cutie could fill this gap.

### Medium Impact

6. **Unimplemented fallback models**: EfficientSAM and SAM2 are imported but have no segmenter subclass. If YOLO and SAM3 are both unavailable, the pipeline fails. At minimum, OpenCV background subtraction (MOG2, KNN) would provide a zero-dependency fallback.

7. **No multi-scale/resolution handling**: Images are segmented at their native resolution. Very high-res images (8K equirect) may exceed VRAM or produce poor results if the model's input resolution is much smaller. No tiling, no resolution management.

8. **No inpainting/removal**: The pipeline generates masks but doesn't offer to remove the masked content. Inpainting (LaMa, MAT, ProPainter for video) could eliminate the need for masks in some workflows by directly removing unwanted content.

9. **Review workflow is per-image**: No batch operations — can't accept/reject all "excellent" masks at once, can't propagate edits from one frame to similar frames.

10. **Code duplication**: COCO classes, class presets, and YOLO inference are duplicated between `masking_v2.py` and `panoex/core/segmenter.py` with no shared code.

### Lower Priority

11. **No confidence calibration**: Model confidence scores are used raw. Different models produce differently calibrated scores. Temperature scaling or Platt scaling could normalize.

12. **No ensemble/multi-model voting**: The pipeline uses one model at a time. Running YOLO + SAM3 and intersecting/unioning results would improve precision/recall.

13. **No attention/saliency guidance**: No mechanism to tell the segmenter "focus on the bottom of the image" or "ignore the sky" beyond the keep_prompts system.

14. **No mask format standardization**: Output naming differs between systems (`mask_{stem}.png` vs `{stem}_mask.png` vs same-stem in masks dir). Discovery code must handle multiple conventions.

---

## 8. Dependency Map

| Capability | Dependencies |
|------------|-------------|
| Core pipeline | numpy, opencv-python, torch, PyYAML |
| SAM3 | sam3 (Meta), torch |
| YOLO26 | ultralytics, torch |
| FastSAM | ultralytics, torch |
| Shadow detection (classical) | numpy, opencv-python only |
| Shadow detection (ML) | torch + model-specific packages |
| GUI | customtkinter, Pillow |
| Mask editor | opencv-python, numpy |
| Review persistence | json (stdlib) |
| Panoex segmenter | ultralytics, opencv-python, numpy |

---

## 9. File Reference

| File | Lines | Role |
|------|-------|------|
| `masking_gui/masking_v2.py` | ~1500 | Pipeline library: models, quality, temporal, geometry, shadow, batch |
| `masking_gui/masking_studio.py` | ~900 | GUI: process tab, review tab, preferences |
| `masking_gui/shadow_detection.py` | ~950 | Extensible shadow detection framework |
| `masking_gui/claude/review_masks.py` | ~600 | OpenCV mask editor |
| `masking_gui/review_gui.py` | ~400 | Review GUI components (thumbnails, overlays) |
| `masking_gui/review_status.py` | ~100 | Review persistence to JSON |
| `panoex/core/segmenter.py` | ~300 | Simple YOLO segmenter for panoex GUI |
