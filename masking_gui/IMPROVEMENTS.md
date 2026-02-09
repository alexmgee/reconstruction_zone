# Masking GUI — Improvement Plan

## A) Masking Pipeline Quality

### A1. Implement YOLO26 Segmenter ✅
- Added `YOLO26Segmenter(BaseSegmenter)` to masking_v2.py
- Uses YOLO26-seg (not YOLO11): +1.9 to +3.2 mask mAP, 19% faster CPU at nano
- NMS-free end-to-end inference, identical ultralytics API
- Class-based detection with COCO presets (person, photographer, equipment, etc.)
- Mirrors the class preset system from panoex/core/segmenter.py
- Default in fallback chain: SAM3 → YOLO26 → FastSAM → EfficientSAM → SAM2

### A2. Multi-Metric Quality Scoring
- Current `_evaluate_mask_quality()` uses confidence score only
- Add: mask area ratio check (reject >50% of image area)
- Add: edge coherence (Canny on mask boundary, smooth = good)
- Add: compactness (4*pi*area/perimeter^2, regular shapes score higher)
- Add: expected-region prior (e.g., nadir region for tripods)
- Weighted composite score replaces single-metric evaluation

### A3. Automated Shadow Detection Pass
- Port the relative-brightness logic from review_masks.py `_flood_fill_add` brightness mode
- Run as post-processing step: detect persons → find dark regions adjacent to person mask base
- Use morphological association (dilate person mask downward, intersect with dark regions)
- Optional: integrate SDDNet for higher quality (heavy dependency)

### A4. Equirectangular Cubemap Strategy
- Convert equirect → 6 cubemap faces via py360convert (already a panoex dependency)
- Segment each face independently with standard pinhole models
- Merge masks back to equirect space with seam blending
- Replaces current CLAHE-at-poles approach for much better results

### A5. Fix _combine_masks Weighted Union
- Lines 783-785: weights are computed then ignored (simple logical_or)
- Implement actual weighted accumulation with threshold
- Higher-confidence masks contribute more to the final combined mask

---

## B) Review/Edit Tab

### B1. Thumbnail Grid View
- Scrollable grid of image+mask thumbnail pairs in CustomTkinter
- Color-coded borders by quality level (green/yellow/orange/red)
- Click thumbnail to jump to that mask in the editor
- Shows mask confidence score and quality label per thumbnail

### B2. Batch Status Persistence
- Write `review_status.json` sidecar tracking: reviewed, accepted, edited, rejected, skipped
- Survives app restarts — resume review sessions where you left off
- Track edit history per mask (timestamps, action taken)

### B3. Sort/Filter Controls
- Sort by: confidence score, quality level, mask area %, filename
- Filter to: needs review only, poor quality only, unreviewed only, all
- Critical for large datasets (500+ masks) where sequential is impractical

### B4. Side-by-Side Comparison View
- Three-pane: original | overlay | mask-only
- Or toggle mode: cycle through views with single keypress
- Don't lose editing state when switching views

### B5. Propagate Edits
- After user manually corrects a mask (e.g., tripod region), offer to propagate
- Use template matching or position-based heuristic for static objects
- Leverage temporal consistency engine for video sequences
- Implements "Phase 3: Propagate" from photogrammetry-masking skill

---

## C) OpenCV Editor Improvements

### C1. Fix Zoom/Coordinate System (CRITICAL — currently broken)
- Current: zoom follows mouse cursor, making zoomed editing impossible
- Current: brush cursor pins to image center when zoomed (line 351)
- Fix: scroll-wheel zooms centered on mouse position, then view stays fixed
- Fix: middle-click-drag to pan the view
- All brush/fill/lasso operations must use image coordinates, not display coordinates

### C2. Coordinate Transformation Layer
- Add `display_to_image(x, y)` and `image_to_display(x, y)` methods
- Account for initial scale-down (self.scale) AND current zoom/pan state
- Use in: `_draw_at`, flood fill click coords, lasso points, brush cursor drawing
- Eliminates entire class of "strokes land in wrong place" bugs

### C3. Embed Editor in CustomTkinter
- Replace spawned OpenCV window with CTkLabel showing PIL-rendered frames
- Mouse events via tkinter bindings instead of cv2 callbacks
- Benefits: native window integration, proper focus, dark theme, toolbar buttons
- Toolbar: brush size slider, tolerance slider, mode buttons, undo/redo
- No more memorizing keyboard shortcuts for basic operations

### C4. Mask Edge Preview / Snap
- When hovering near mask edge, highlight the edge more prominently
- In remove mode near edge, snap to nearest contour for precise cleanup
- Visual indicator showing which contour will be affected

### C5. A/B Toggle and View Modes
- Current 'm' removes mask entirely — too binary
- Add cycle: overlay → original → mask-only (white on black) → overlay
- Quick flicker comparison (hold key to show original, release to show overlay)
- Helps verify mask accuracy without overlay obscuring the image

---

## Implementation Order

Phase 1 — Fix what's broken:
  C1. Fix zoom/coordinate system
  C2. Coordinate transformation layer
  A5. Fix _combine_masks weighted union

Phase 2 — Add working segmentation:
  A1. YOLO11 segmenter
  A2. Multi-metric quality scoring

Phase 3 — Build the review infrastructure:
  B1. Thumbnail grid view
  B2. Batch status persistence
  B3. Sort/filter controls
  C5. A/B toggle and view modes

Phase 4 — Embed editor in CTk:
  C3. Embed editor in CustomTkinter
  C4. Mask edge preview/snap
  B4. Side-by-side comparison

Phase 5 — Advanced pipeline:
  A3. Automated shadow detection
  A4. Equirectangular cubemap strategy
  B5. Propagate edits
