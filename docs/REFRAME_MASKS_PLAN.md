# Mask-Aware Reframing — 360 Video Section

## Status: Implemented

## Problem

The old workflow reframed ERP/fisheye into 26+ perspectives, then masked each independently — same object detected 26 times, wasteful and inconsistent. The fix: mask the ERP first, then reframe both images and masks together.

## UI — 360 Video Subwindow

### Fields (4)
```
Video:   [.osv / .360 / .insv file...]   [...]   (optional)
Frames:  [ERP frames folder...]          [...]   (optional)
Masks:   [ERP masks folder...]           [...]   (optional)
Output:  [Output directory...]           [...]   (required)
```

### Mode auto-detection (no toggle — inferred from filled fields)

| Mode | Fields needed | Flow |
|------|-------------|-------------|
| **Extract & Reframe** | Video + Output | Extract video → `fisheye_frames/`, reframe → `output/images/` |
| **Reframe only** | Frames + Output | Reframe existing frames → `output/images/` |
| **Reframe with Masks** | Frames + Masks + Output | Reframe → `output/images/` + `output/masks/` |

### Validation
- No Output → error
- No Video and no Frames → error
- Masks without Frames → error
- Auto-detects ERP vs fisheye frames (front_*/back_* patterns = fisheye, otherwise ERP)

## Output Structure

Flat output under `images/` with parallel `masks/`:
```
output/
├── images/
│   ├── frame_00000_front_v00_y+0_p+30.jpg
│   └── ...
└── masks/          (only when masks provided)
    ├── frame_00000_front_v00_y+0_p+30.png
    └── ...
```

Masks: PNG (lossless, binary 0/255), same stem as corresponding image.

## Files Modified

### `prep360/core/reframer.py`
- `_process_single_image()` — 5th arg `mask_path`, loads grayscale, `reframe_view()` with `mode="nearest"`, threshold `> 0`, saves PNG to sibling `masks/` dir
- `reframe_batch()` — new `mask_dir` param, builds `mask_map` by stem, logs unmatched count
- `reframe_single()` — new `mask_path` param

### `prep360/core/fisheye_reframer.py`
- `extract_view()` — new `mask` param, `cv2.remap()` with same cached tables + `INTER_NEAREST`, returns `(crop, mask_crop)` tuple when mask provided
- `extract_all_views()` — new `front_mask`/`back_mask` params, returns `(view, crop, mask_or_None)` tuples
- `extract_and_save()` — new `front_mask`/`back_mask` params, saves mask PNGs to sibling `masks/` dir
- `batch_extract()` — new `mask_dir` param, builds mask lookup, loads masks by stem match (sequential path only)

### `reconstruction_gui/tabs/source_tab.py`
- `_build_fisheye_section()` — Input→Video, added Frames/Masks fields, Output stays
- `_run_fisheye()` — mode auto-detection from filled fields, new validation
- `_fisheye_worker()` — conditional extraction, auto-detects fisheye vs ERP frames, passes `mask_dir`
- `_erp_reframe_worker()` — new function for ERP reframe path (not fisheye pairs)

## Mask Matching

By stem name: `frame_00001.jpg` → `frame_00001.png`. Missing masks = warning, not error.

## Edge Cases

1. Dimension mismatch → skip with error log
2. 0/1 or 0/255 input → threshold with `> 0` handles both
3. Motion selection → masks matched by name, so filtered frames just don't use their masks
