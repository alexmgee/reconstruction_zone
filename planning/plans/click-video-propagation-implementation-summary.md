# Click–Video Propagation: Implementation Summary

Post-implementation state as of 2026-05-03. Records every change, its rationale, what was left untouched and why, known issues, orphaned code, and the testing surface.

---

## 1. Files changed

### 1.1 `reconstruction_gui/sam3_click_session.py` — full rewrite (458 lines)

**Before**: `SAM3ClickSession` (131 lines). Wrapped `build_sam3_image_model` + `predict_inst` for single-image click-to-mask. Accepted an already-loaded `(model, processor)` pair from `MaskingPipeline.get_sam3_click_handles()`.

**After**: `SAM3VideoClickSession` (458 lines). Wraps `build_sam3_multiplex_video_predictor` for multi-frame click-to-propagate. Builds its own predictor instance (independent of the batch pipeline).

| Method | Purpose |
|--------|---------|
| `start(input_dir, device)` | Build predictor, start video session on directory, enumerate frames |
| `add_click(frame_idx, x, y, positive)` | Normalise coords to [0,1], `add_prompt` with `obj_id=0`, return mask |
| `get_frame_mask(frame_idx)` | Return cached mask from last `add_prompt` on that frame |
| `propagate(output_dir, ...)` | `propagate_in_video(direction="both")`, write masks, support cancel |
| `remove_last_click(frame_idx)` | Pop last click, re-prompt or remove object |
| `reset_frame(frame_idx)` | Clear all clicks on one frame |
| `reset_all()` | Clear all clicks, `reset_session` |
| `close()` | `close_session`, delete predictor, `torch.cuda.empty_cache()` |

**Key design decisions baked in:**

| Decision | Rationale | Source |
|----------|-----------|--------|
| Pixel→[0,1] normalisation in `_send_prompt` | `rel_coordinates=True` upstream (`sam3_tracking_predictor.py:216-219`) multiplies by `image_size=1008`. Caller always works in pixel space. | Plan review §2.2 CRITICAL |
| `obj_id=0` constant | `add_prompt` asserts `obj_id is not None` for points (`sam3_multiplex_tracking.py:2706-2708`). Single merged object scope. | Plan review §2.2 CRITICAL |
| `clear_old_points=True` every call | We maintain our own click accumulator and resend the full set. Avoids state sync issues with the SAM3 session. | Plan §Phase 1 item 2 |
| `(1 - mask) * 255` on propagation write | Matches batch pipeline convention (`reconstruction_pipeline.py:2817,3047,3484`). Masked-out = black (0), keep = white (255). | Plan review §2.2, line 73 |
| Cancel keeps session open | `propagate()` breaks the stream iteration but does not call `close_session`. User can adjust clicks and re-propagate. | Plan review §2.2 MEDIUM |
| `text_str` and `boxes_xywh` always None | Point prompts and text prompts are mutually exclusive (`sam3_multiplex_tracking.py:2703`). | Plan review §2.2 MEDIUM |
| `async_loading_frames=True` + `offload_video_to_cpu=True` | Lazy frame loading + CPU tensor storage. Verified to compose correctly (`io_utils.py:197-201`). | Plan review §2.3 |
| FA3 detection + SDPA fallback patch | Replicates `sam3_pipeline.py:148-169`. Required because SAM 3.1 decoder hardcodes `sdpa_kernel(FLASH_ATTENTION)` when `use_fa3=False`. | Plan §Phase 1 item 1 |
| Frame sort replicates `io_utils.py:176-189` | Numeric stem sort, lexicographic fallback. Cross-checks `num_frames` from session state. SAM3 does not expose ordered frame list post-init. | Plan review §Phase B item 3 |

**Helper functions at module level:**

| Function | Purpose |
|----------|---------|
| `_sort_frame_names(names)` | Replicate SAM3 io_utils sort |
| `_enumerate_frames(directory)` | List + sort image files from a directory |
| `_detect_fa3()` | Check Flash Attention 3 availability |
| `_patch_sdpa_fallback()` | Monkey-patch decoder for MATH attention |

### 1.2 `reconstruction_gui/reconstruction_zone.py` — major edit

**Total file**: 5248 lines (was ~4700 before this session's changes, plus prior uncommitted work).

#### 1.2.1 State variables (line ~173)

**Removed:**
```
_click_image_bgr, _click_image_orig_size, _click_display_pil,
_click_display_size, _click_image_path, _click_last_mask, _click_last_score
```

**Replaced with:**
```python
self._click_session = None           # SAM3VideoClickSession or None
self._click_state = "idle"           # idle | loading | ready | predicting | propagating | error
self._click_current_frame_idx = 0    # currently displayed frame index
self._click_current_frame_pil = None # fit-scaled display PIL of current frame
self._click_current_display_size = None  # (H, W) of display PIL
self._click_preview_mask = None      # uint8 0/1 mask from last add_prompt
```

Note: `_click_state` expanded from 4 states (`idle|encoding|ready|predicting`) to 6 (`idle|loading|ready|predicting|propagating|error`).

#### 1.2.2 UI widgets (lines ~1218–1370)

**Removed:**
- Layer name entry (`_click_layer_name_var`, `clk_layer_row`)
- "Save as layer" button (`_click_save_btn`)
- "New session" button (`_click_new_session_btn`)
- "Clear clicks" button (`_click_clear_btn`) — replaced by two separate buttons

**Added:**
| Widget | Variable | Lines |
|--------|----------|-------|
| Frame prev button (`◀`) | `_click_frame_prev_btn` | ~1256 |
| Frame next button (`▶`) | `_click_frame_next_btn` | ~1262 |
| Frame index entry | `_click_frame_idx_var`, `_click_frame_idx_entry` | ~1268 |
| Frame total label | `_click_frame_total_label` | ~1275 |
| Frame slider | `_click_frame_slider` | ~1280 |
| Clear frame button | `_click_clear_frame_btn` | ~1290 |
| Clear all button | `_click_clear_all_btn` | ~1297 |
| Prompted frames label | `_click_prompted_label` | ~1310 |
| Propagate All button | `_click_propagate_btn` | ~1319 |
| Cancel button | `_click_cancel_btn` | ~1326 |
| Progress bar | `_click_progress_bar` | ~1334 |
| Progress label | `_click_progress_label` | ~1339 |

**Preserved:**
- Activate checkbox (`click_mode_var`) — text changed from "Activate click mode" to "Activate video click mode"
- Status label (`_click_status_label`) — reused
- Click count label (`_click_count_label`) — reused, now shows per-frame counts

#### 1.2.3 Event handlers (lines ~2586–3003)

**Removed entirely:**
| Method | Was |
|--------|-----|
| `_click_reencode_for` | Start single-image encoder worker |
| `_click_encoder_worker` | Build pipeline, get SAM3 handles, encode one image |
| `_click_decoder_worker` | Run `predict_inst` on cached points |
| `_on_click_save_layer` | Save single mask as layer PNG |
| `_click_save_worker` | Write layer PNG + review metadata |
| `_on_click_new_session` | Re-encode same image |
| `_on_click_clear` | Clear clicks (single clear button handler) |

**Added:**
| Method | Purpose | Lines |
|--------|---------|-------|
| `_click_set_state` | Updated for 6 states + new widget enable/disable | ~2589 |
| `_on_click_mode_toggle` | Release batch pipeline, start video session | ~2618 |
| `_click_session_worker` | Worker: build video predictor + start session | ~2641 |
| `_click_navigate_to` | Load frame from disk, render overlay | ~2675 |
| `_click_navigate_delta` | Prev/next by delta | ~2727 |
| `_click_navigate_to_entry` | Jump to typed frame number | ~2731 |
| `_on_click_slider` | Slider changed → navigate | ~2736 |
| `_on_click_overlay` | Click handler → dispatch decode worker | ~2739 |
| `_click_decode_worker` | Worker: `session.add_click()` under lock | ~2772 |
| `_render_click_overlay` | Rewritten for video session state | ~2810 |
| `_update_click_count` | Per-frame click count | ~2857 |
| `_update_prompted_frames_label` | Summary of prompted frame indices | ~2869 |
| `_on_click_clear_frame` | Clear current frame clicks | ~2890 |
| `_on_click_clear_all` | Clear all frames | ~2901 |
| `_on_click_propagate` | Start propagation worker | ~2914 |
| `_click_propagate_worker` | Worker: `session.propagate()` under lock | ~2938 |
| `_click_update_progress` | Update progress bar | ~2969 |
| `_on_click_cancel_propagation` | Set cancel flag (session stays open) | ~2975 |
| `_click_teardown` | Rewritten for video session state | ~2981 |

**Preserved (with modifications):**
- `_get_click_fisheye_circle` — unchanged, still generates fisheye circle masks
- `_render_click_overlay` — rewritten to use `_click_current_frame_pil` instead of `_click_display_pil`, and `session.get_frame_clicks()` instead of `session.points_record`

#### 1.2.4 Navigator integration (line ~1928)

**Before**: Re-encoded single image when navigator selection changed while click mode was active.

**After**: Looks up the selected image's filename in the video session's frame list and navigates to that index. Does not re-encode (the video session covers all frames).

#### 1.2.5 Batch run guards (lines ~3011 and ~3658)

Added click-mode guard to both `_on_run` and `_run_masking_queue`:
```python
if self._click_session is not None:
    self.log("ERROR: Deactivate interactive click mode before running batch masking.")
    return
```

#### 1.2.6 Bug fix: pyimage teardown (applied earlier in session)

In `_click_teardown`, the old code called `self._process_overlay_label.configure(image=None, text="")` followed by `_ctk_image = None`. This left a stale Tcl-level pyimage reference, causing `_tkinter.TclError: image "pyimage24" doesn't exist` on the next `configure()` call.

**Fix**: Removed the `configure(image=None)` and `_ctk_image = None` calls. The label retains its last image until the next preview or click session overwrites it. Added `_preview_mask_pil = None` alongside `_preview_overlay_pil = None` so `_zoom_render` returns early.

---

## 2. Files NOT changed (but affected)

### 2.1 `reconstruction_gui/reconstruction_pipeline.py`

**`MaskingPipeline.get_sam3_click_handles()`** at line 1678–1688 is now **orphaned dead code**. It was the bridge between the batch pipeline's SAM3 image model and the old single-image click session. No callers remain in the codebase. [grep: confirmed zero callers outside its own definition]

**Not removed** because:
- It's in a separate file from the main changes
- It causes no runtime harm (never called)
- Removing it is a one-line concern but touches a file with 567 lines of uncommitted changes from prior work
- Appropriate for a cleanup pass, not this feature implementation

**`_release_pipeline_locked()`** at line 3140–3166 still references `self._click_session.close()` and schedules `self._click_teardown`. This remains **correct** for the new video session: if the batch pipeline is forcibly released while a video click session happens to be active (shouldn't occur in normal flow since they're mutually exclusive, but defensive), calling `close()` on `SAM3VideoClickSession` properly cleans up the video predictor. The docstring ("click session holds an inference_state tied to the current model") is **stale** — the video session now holds its own predictor, not a reference to the batch model. Docstring update deferred.

### 2.2 `reconstruction_gui/sam3_pipeline.py`

Unchanged. The `SAM3VideoPipeline` class remains independent — it's the batch text-prompt video path. The new `SAM3VideoClickSession` duplicates the FA3 detection and SDPA patch as standalone functions rather than importing from `sam3_pipeline.py`. This was a deliberate choice: the pipeline module has different import patterns (lazy imports at module level via `HAS_SAM3_VIDEO` flag) and coupling them would require refactoring the import structure.

**Future cleanup**: Extract the FA3 detection + SDPA patch into a shared utility (e.g. `sam3_utils.py`) so `sam3_pipeline.py`, `sam3_click_session.py`, and `reconstruction_pipeline.py` all use one copy. Not blocking.

### 2.3 `reconstruction_gui/docs/MASK_TAB.md`

Not updated. The tab documentation still describes the old single-image click mode. Should be updated to reflect:
- Video session instead of single-image encoding
- Frame navigation
- Propagation instead of per-image layer save
- Removal of layer name field

---

## 3. SAM3 API contract (as implemented)

The session module depends on these specific SAM3 behaviours, verified against source:

| API call | Key constraints | Verified at |
|----------|-----------------|-------------|
| `start_session(resource_path=str_dir)` | Directory path triggers `load_video_frames_from_image_folder`. `async_loading_frames=True` uses lazy `AsyncImageFrameLoader`. | `io_utils.py:142-150, 197-201` |
| `add_prompt(points, point_labels, obj_id)` | Points must be [0,1] normalised. `obj_id` required for point prompts (assert). `text_str` and `boxes_xywh` must be None when points are set (assert). | `sam3_multiplex_tracking.py:2701-2708` |
| `add_prompt` return value | `outputs["out_binary_masks"]` is `(N, H, W)` bool at original resolution. `outputs["out_probs"]` is always 1.0 for user-added objects. | `sam3_multiplex_tracking.py:2863, 788-796` |
| `propagate_in_video(direction="both")` | Valid direction; runs forward then backward in one stream. | `sam3_base_predictor.py:251-284` |
| `propagate_in_video` stream | Yields `{"frame_index": int, "outputs": {"out_binary_masks": ...}}` per frame. | `sam3_base_predictor.py:272-284` |
| `close_session(run_gc_collect=True)` | Drops inference state, triggers Python GC. | `sam3_base_predictor.py:76-83` |
| Frame sort order | Numeric stem → lexicographic fallback. Extensions: `.jpg .jpeg .png .bmp .tiff .webp`. | `io_utils.py:176-189, 24` |
| Internal image size | 1008px (not 1024). | `sam3_video_inference.py:36` |
| Model checkpoint | `facebook/sam3.1` → `sam3.1_multiplex.pt` (different from image model's `facebook/sam3` → `sam3.pt`). | `model_builder.py:1212-1213 vs 644-645` |

---

## 4. Mask polarity

| Writer | Convention | Code |
|--------|-----------|------|
| Batch `process_directory` | `(1 - result.mask) * 255` — white=keep, black=masked | `reconstruction_pipeline.py:2817, 3047, 3484` |
| Video click propagation (new) | `(1 - combined) * 255` — matches batch | `sam3_click_session.py:propagate()` |
| Old click layer save (removed) | `mask * 255` — **opposite** convention | Was at `reconstruction_zone.py:2921` (deleted) |

The old click layer save used inverted polarity. The new propagation matches the batch pipeline. No legacy click-save code remains.

---

## 5. Concurrency and GPU

| Resource | Guard | Notes |
|----------|-------|-------|
| Video predictor (`add_prompt`, `propagate_in_video`) | `_pipeline_lock` | All worker threads acquire the lock before calling `handle_request` / `handle_stream_request` |
| Batch pipeline image model | Released on click mode activation via `_release_pipeline()` | Rebuilt lazily on next `_get_pipeline` call after click mode deactivates |
| `cancel_flag` | Shared `threading.Event` | Reused from batch pipeline; `_on_click_cancel_propagation` sets it, propagation loop checks between frames |
| Batch run + click mode | Mutually exclusive | `_on_run` and `_run_masking_queue` both reject with error log if `_click_session is not None` |

---

## 6. What was NOT implemented (explicit deferrals)

| Feature | Reason | Plan reference |
|---------|--------|----------------|
| Post-processing parity (shadow, SAM refine, dilation, fill holes) | Would require loading each image and running `process_image(precomputed_mask=...)` per frame. Significant complexity for v1. | Plan review §Phase D |
| Layer output for propagation | v1 writes to output root. Layer nesting (`layers/{name}/`) is a stretch goal. | Plan §Not in Scope |
| Per-object tracking (multiple obj_ids) | All clicks merge into `obj_id=0`. Multi-object needs UI for managing separate masks. | Plan §Not in Scope |
| Bidirectional propagation UI | `"both"` is the default and correct for click-on-any-frame. Separate forward/backward controls not needed. | Plan §Not in Scope |
| Review tab integration | Masks land in the standard output dir. Review tab already discovers them from any mask directory. | Plan §Not in Scope |
| Undo/redo stack | `remove_last_click` + `reset_frame` + `reset_all` cover practical needs. | Plan §Not in Scope |
| `add_prompt` score display | Always 1.0 for user-clicked objects. Not a quality signal. Omitted from status line. | Plan review §2.2 MEDIUM |

---

## 7. Orphaned / stale code

| Item | Location | Status | Action |
|------|----------|--------|--------|
| `MaskingPipeline.get_sam3_click_handles()` | `reconstruction_pipeline.py:1678-1688` | Dead code, zero callers | Remove in cleanup pass |
| `_release_pipeline_locked` docstring | `reconstruction_zone.py:3141-3146` | Stale — says "click session holds an inference_state tied to the current model" | Update to reflect video session independence |
| FA3 + SDPA patch duplication | `sam3_pipeline.py:148-169` and `sam3_click_session.py:_detect_fa3/_patch_sdpa_fallback` | Duplicated across two files | Extract to shared `sam3_utils.py` in cleanup pass |
| `MASK_TAB.md` Interactive Click Mode section | `reconstruction_gui/docs/MASK_TAB.md` | Describes old single-image flow | Update documentation |
| Verbose PVS click logging | Was in old `_on_click_overlay`, now removed | Resolved | N/A — diagnostic logs were deleted in the rewrite |

---

## 8. Testing checklist (manual)

No automated test suite exists. All verification is manual.

| # | Test | Verify |
|---|------|--------|
| 1 | **Session init** | Activate click mode with fisheye dataset. Status shows frame count. First frame displays in preview. |
| 2 | **Frame navigation** | Prev/next buttons, slider, manual index entry all load correct frames. Frame counter updates. |
| 3 | **Click → mask** | Left-click on an object. Red overlay appears. Right-click to exclude. Overlay updates. Click count updates. |
| 4 | **Multi-frame prompting** | Click on frame 1, navigate to frame 500, click again. Prompted frames label shows both. Navigate back — frame 1 still shows its overlay. |
| 5 | **Propagation** | Click "Propagate All". Progress bar advances. Output directory contains mask PNGs for all frames. |
| 6 | **Mask polarity** | Open a propagated PNG. White = background (keep), black = masked object. Matches batch pipeline output. |
| 7 | **Fisheye circle** | With fisheye circle mask enabled, propagated masks include the circular border mask. |
| 8 | **Cancel propagation** | Start propagation, click Cancel mid-run. Partial masks written. Session stays alive — can add more clicks and re-propagate. |
| 9 | **Deactivate** | Uncheck click mode. VRAM freed. Run batch masking. Confirm batch pipeline rebuilds and runs normally. |
| 10 | **Batch block** | While click mode is active, click Run or Process Queue. Confirm error message in console, not a crash. |
| 11 | **pyimage regression** | Activate click mode, add clicks, deactivate, run Preview. Confirm no `"pyimage doesn't exist"` TclError. |
| 12 | **Navigator sync** | While click mode is active, click on a different image in the file navigator. Click mode navigates to that frame. |
| 13 | **Clear frame / Clear all** | Add clicks on multiple frames. Clear frame — only current frame's clicks removed. Clear all — all clicks removed, prompted frames label resets. |
| 14 | **Session init failure** | Trigger an error (e.g. empty input dir). Confirm error shown in status, checkbox unchecked, batch pipeline still works afterward. |

---

## 9. Planning documents index

| Document | Purpose |
|----------|---------|
| `planning/plans/click-video-propagation-plan.md` | Original architecture plan (pre-review). Contains the target data flow, API spec, UI wireframe, memory budget, edge cases. Some details superseded by the review. |
| `planning/plans/click-video-propagation-plan-review.md` | Three-pass verification against SAM3 upstream source and reconstruction-zone codebase. Contains the must-fix corrections (normalised coordinates, obj_id, 1008, polarity, cancel semantics) that were incorporated into implementation. |
| `planning/plans/click-video-propagation-implementation-summary.md` | This document. Post-implementation state record. |
