# Interactive Click Mode → Video Propagation

## Problem Statement

The current Interactive Click Mode uses SAM3's single-image `predict_inst` API. The user clicks on one image, gets a mask for that one image, and can save it as a per-image layer. For a dataset of 972 frames, this is useless — the user would need to manually click on every single frame.

SAM3's multiplex video predictor supports point prompts + temporal propagation: click on a shadow in frame 0 and the model tracks it across all frames. This capability exists in the codebase (`sam3_pipeline.py`) but only for text prompts. The click session needs to be rebuilt on top of the video predictor.

---

## Current Architecture

### What exists today

| Component | File | Purpose |
|-----------|------|---------|
| `SAM3ClickSession` | `sam3_click_session.py` | Wraps `predict_inst` on `build_sam3_image_model`. Single-image only. |
| `SAM3VideoPipeline` | `sam3_pipeline.py` | Wraps `build_sam3_multiplex_video_predictor`. Text prompts → video propagation. |
| Click mode UI | `reconstruction_zone.py:1221-1291` | Checkbox, status, clear/save/new-session buttons. Layer name field. |
| Click handlers | `reconstruction_zone.py:2524-2900` | `_on_click_mode_toggle`, `_click_encoder_worker`, `_on_click_overlay`, `_render_click_overlay`, `_on_click_save_layer`, `_on_click_new_session`, `_click_teardown` |
| `SAM3Segmenter` | `reconstruction_pipeline.py:730` | Batch image model (`build_sam3_image_model`). Click mode reuses this via `get_sam3_click_handles()`. |

### Data flow (current, broken)

```
Activate click mode
  → _click_encoder_worker: get SAM3 image model from batch pipeline
  → session.set_image(bgr) — encode one image
  → user clicks → session.add_click(x, y, positive)
  → session.predict() → predict_inst → single mask on one frame
  → "Save as layer" → writes ONE mask to layers/{name}/{stem}.png
  → Repeat manually for every frame (unusable)
```

### Model distinction

- **Image model**: `build_sam3_image_model(enable_inst_interactivity=True)` — 848M params. `predict_inst` for click-to-mask on a single image. Used by batch pipeline and current click mode.
- **Video predictor**: `build_sam3_multiplex_video_predictor()` — multiplex tracker with shared-memory object buckets. Accepts both text prompts AND point prompts via `add_prompt`. Supports `propagate_in_video` across the full frame sequence.

These are **separate model instances** with different checkpoints. They cannot share weights. On a 24GB GPU (RTX 3090 Ti), both likely cannot coexist simultaneously.

---

## Target Architecture

### Data flow (new)

```
Activate click mode
  → Release batch pipeline's image model (free VRAM)
  → Build SAM3 multiplex video predictor
  → start_session(resource_path=input_directory, async_loading_frames=True,
                   offload_video_to_cpu=True)
  → User navigates frames with prev/next + slider
  → User clicks on current frame (left=include, right=exclude)
  → add_prompt(frame_index, points, point_labels) → returns mask for current frame
  → Live preview shows mask on current frame
  → User can navigate to other frames and add more clicks (multi-frame prompting)
  → "Propagate All" → propagate_in_video(direction="both")
  → Write masks for all frames to output directory
  → Close session, free video predictor VRAM
```

### Memory budget

| Resource | Estimate | Notes |
|----------|----------|-------|
| Video predictor model weights | ~3-4 GB VRAM | Similar param count to image model |
| Frame tensors (972 frames, 1024px, float16, CPU) | ~6 GB RAM | `offload_video_to_cpu=True` keeps these in system RAM |
| Async frame loader overhead | minimal | `async_loading_frames=True` loads lazily |
| Inference state per frame | ~50 MB VRAM | Cached features for prompted frames |
| **Total VRAM** | **~4-5 GB** | Well within 24 GB budget |
| **Total RAM** | **~8-10 GB** | System RAM for frame tensors + PIL overhead |

---

## Implementation Plan

### Phase 1: Rewrite `sam3_click_session.py`

Replace the single-image session with a video session wrapper.

**Current API (to remove):**
```python
session = SAM3ClickSession(model, processor)  # wraps image model
session.set_image(bgr)                         # encode one image
session.add_click(x, y, positive)              # buffer click
session.predict()                              # predict_inst → (mask, score)
session.close()                                # free state
```

**New API:**
```python
session = SAM3VideoClickSession()
session.start(input_dir, device="cuda")        # build video predictor + start_session
session.add_click(frame_idx, x, y, positive)   # add_prompt with point on frame
# Returns: mask for that frame (for live preview), score
session.get_frame_mask(frame_idx)              # get current mask for any frame
session.propagate(output_dir, callback)         # propagate_in_video → write all masks
session.close()                                # close session + free predictor
```

**Key properties:**
```python
session.num_frames          # total frames in session
session.frame_names         # list of sorted filename stems
session.prompted_frames     # dict: frame_idx → list of {x, y, label}
session.is_ready            # True after start() completes
```

**Implementation details:**

1. **`start(input_dir, device)`**:
   - Build `build_sam3_multiplex_video_predictor(use_fa3=<detect>, async_loading_frames=True)`
   - Apply the FA3/SDPA fallback patch from `sam3_pipeline.py:156-169`
   - Call `handle_request({"type": "start_session", "resource_path": str(input_dir), "offload_video_to_cpu": True})`
   - Enumerate and sort frame files from `input_dir` (same sort logic as SAM3's `io_utils.py:176-189`)
   - Store `session_id`, `frame_names`, `num_frames`

2. **`add_click(frame_idx, x, y, positive)`**:
   - Append to `prompted_frames[frame_idx]`
   - Build points list from all clicks on this frame
   - Call `handle_request({"type": "add_prompt", "session_id": sid, "frame_index": frame_idx, "points": [[x1,y1], [x2,y2], ...], "point_labels": [1, 0, ...], "clear_old_points": True})`
   - Note: `clear_old_points=True` because we resend ALL accumulated clicks each time (SAM3 replaces the prompt set per call)
   - Return `(mask, score)` from the response's `outputs["out_binary_masks"]`

3. **`get_frame_mask(frame_idx)`**:
   - If frame has been prompted, return the cached mask from the last `add_prompt` response
   - If frame hasn't been prompted, return None (mask only exists after propagation)

4. **`propagate(output_dir, fisheye_circle_fn, progress_callback, cancel_event)`**:
   - Call `handle_stream_request({"type": "propagate_in_video", "session_id": sid, "propagation_direction": "both"})`
   - Iterate streaming responses, one per frame
   - For each frame: merge all object masks (logical OR), resize to original resolution, optionally composite with fisheye circle mask, write as `{stem}.png` (0/255 uint8)
   - Call `progress_callback(current, total, message)` per frame
   - Check `cancel_event` between frames for cancellation support

5. **`close()`**:
   - `handle_request({"type": "close_session", ...})`
   - `del self.predictor`
   - `torch.cuda.empty_cache()`

6. **`remove_last_click(frame_idx)`**:
   - Pop the last click from `prompted_frames[frame_idx]`
   - If clicks remain, re-call `add_prompt` with remaining clicks
   - If no clicks remain, call `handle_request({"type": "remove_object", ...})` or `reset_session`

7. **`reset_frame(frame_idx)`**:
   - Clear all clicks on this frame
   - Call `remove_object` or re-prompt with empty points

### Phase 2: Rewrite click mode UI in `reconstruction_zone.py`

**Remove:**
- Layer name field (`_click_layer_name_var`, `clk_layer_row`)
- "Save as layer" button (`_click_save_btn`, `_on_click_save_layer`, `_click_save_worker`)
- "New session" button as currently implemented (`_click_new_session_btn`, `_on_click_new_session`)
- Single-image encoding logic (`_click_encoder_worker`, `_click_reencode_for`)
- All `_click_image_bgr`, `_click_image_orig_size`, `_click_display_pil`, `_click_display_size`, `_click_image_path` state

**Add:**

#### UI Layout (within existing collapsible section)

```
┌─ Interactive Click Mode (PVS) ─────────────────────────────┐
│ left = include, right = exclude                             │
��                                                             │
│ [x] Activate video click mode         requires SAM3         │
│                                                             │
│ ● loading 972 frames... (or: ● ready · frame 1/972)        │
│                                                             │
│ Frame: [◀] [▶]  [ 1 ]/972   [────────●──────────] slider   │
│                                                             │
│ [Clear frame]  [Clear all]          +3  -1                  │
│                                                             │
│ Prompted frames: 1, 47, 503  (3 frames with clicks)        │
│                                                             │
│ [Propagate All ▶]  [Cancel]                                 │
│                                                             │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░ 0/972                           │
└─────────────────────────────────────────────────────────────┘
```

#### Widget inventory

| Widget | Type | Variable/Attr | Purpose |
|--------|------|---------------|---------|
| Activate checkbox | CTkCheckBox | `click_mode_var` (reuse) | Start/stop video session |
| Status label | CTkLabel | `_click_status_label` (reuse) | State + frame info |
| Frame prev button | CTkButton | `_click_frame_prev_btn` | Navigate to previous frame |
| Frame next button | CTkButton | `_click_frame_next_btn` | Navigate to next frame |
| Frame index entry | CTkEntry | `_click_frame_idx_var` | Jump to specific frame (editable) |
| Frame total label | CTkLabel | `_click_frame_total_label` | Shows `/972` |
| Frame slider | CTkSlider | `_click_frame_slider` | Scrub through frames |
| Clear frame button | CTkButton | `_click_clear_frame_btn` | Clear clicks on current frame |
| Clear all button | CTkButton | `_click_clear_all_btn` | Clear ALL clicks across ALL frames |
| Click count label | CTkLabel | `_click_count_label` (reuse) | `+N  -N` for current frame |
| Prompted frames label | CTkLabel | `_click_prompted_label` | Lists which frames have clicks |
| Propagate button | CTkButton | `_click_propagate_btn` | Run propagation |
| Cancel button | CTkButton | `_click_cancel_btn` | Cancel propagation |
| Propagation progress bar | CTkProgressBar | `_click_progress_bar` | Shows propagation progress |
| Propagation progress label | CTkLabel | `_click_progress_label` | `142/972` text |

#### State variables

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_click_session` | `SAM3VideoClickSession` or None | The active video session |
| `_click_state` | str | `idle`, `loading`, `ready`, `predicting`, `propagating`, `error` |
| `_click_current_frame_idx` | int | Currently displayed frame index |
| `_click_current_frame_bgr` | ndarray or None | BGR of the currently displayed frame (for overlay rendering) |
| `_click_current_frame_pil` | PIL.Image or None | Fit-scaled display PIL of current frame |
| `_click_current_display_size` | (H, W) or None | Display dimensions of current frame |
| `_click_preview_mask` | ndarray or None | Last mask returned by `add_prompt` for current frame |

### Phase 3: Event handlers and data flow

#### Activation (`_on_click_mode_toggle`)

```
if deactivating:
    _click_teardown()  → close session, free predictor, reset UI
    return

input_dir = self.input_entry.get()
if not valid:
    show error, uncheck
    return

# Free batch pipeline's image model to make VRAM room
if self._pipeline is not None:
    self._pipeline.cleanup()
    self._pipeline = None

_click_set_state("loading", f"loading {n_frames} frames...")

# Worker thread:
    session = SAM3VideoClickSession()
    session.start(input_dir, device="cuda")
    # On main thread when done:
    self._click_session = session
    self._click_navigate_to(0)  # show first frame
    _click_set_state("ready", f"ready · frame 1/{session.num_frames}")
```

#### Frame navigation (`_click_navigate_to(frame_idx)`)

```
# Read frame from disk (input_dir / frame_names[frame_idx])
bgr = cv2.imread(str(frame_path))

# Fit-scale for display panel (same logic as current preview)
# ... resize to panel dimensions ...
display_pil = PILImage.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

self._click_current_frame_idx = frame_idx
self._click_current_frame_bgr = bgr
self._click_current_frame_pil = display_pil
self._click_current_display_size = (rh, rw)

# Update frame index widgets
self._click_frame_idx_var.set(str(frame_idx + 1))
self._click_frame_slider.set(frame_idx)

# Render overlay: show clicks on this frame + mask if available
mask = self._click_session.get_frame_mask(frame_idx)
self._render_click_overlay(mask)
self._update_click_count()  # show +N -N for THIS frame
```

#### Click handler (`_on_click_overlay`)

```
# Same coordinate mapping as current (display → original pixel coords)
# But now we pass the frame index too:

frame_idx = self._click_current_frame_idx
self._click_set_state("predicting", "decoding...")

# Worker thread:
    mask, score = self._click_session.add_click(frame_idx, ox, oy, positive)
    # On main thread:
    self._click_preview_mask = mask
    self._render_click_overlay(mask)
    self._update_click_count()
    self._update_prompted_frames_label()
    self._click_set_state("ready", f"frame {frame_idx+1} · score {score:.2f}")
```

#### Propagation (`_on_click_propagate`)

```
output_dir = self.output_entry.get()  # same output dir as batch pipeline
if not output_dir:
    error
    return

# Fisheye circle mask factory (closure over current settings)
def fisheye_circle_fn(w, h):
    return self._get_click_fisheye_circle(w, h)

self._click_set_state("propagating", "propagating...")

# Worker thread:
    self._click_session.propagate(
        output_dir=output_dir,
        fisheye_circle_fn=fisheye_circle_fn,
        progress_callback=lambda cur, total, msg:
            self.after(0, lambda c=cur, t=total: self._click_update_progress(c, t)),
        cancel_event=self.cancel_flag,
    )
    # On main thread when done:
    self._click_set_state("ready", f"done · {n} masks written to {output_dir}")
    self.log(f"Click propagation: {n} masks → {output_dir}")
```

#### Teardown (`_click_teardown`)

```
if self._click_session is not None:
    try:
        self._click_session.close()
    except Exception:
        pass
self._click_session = None
self._click_current_frame_idx = 0
self._click_current_frame_bgr = None
self._click_current_frame_pil = None
self._click_current_display_size = None
self._click_preview_mask = None
self._preview_overlay_pil = None
self._preview_mask_pil = None
# Reset UI: disable nav, clear progress, reset status
# Do NOT configure(image=None) on the overlay label (pyimage bug)
```

### Phase 4: Integration with existing workflow

#### Output location

Click propagation writes masks to the **same directory** as the batch pipeline: `{output_dir}/` (the masks entry field). This means:

- If the user already ran batch masking, click-propagated masks **overwrite** the batch masks. This is intentional — the user is choosing to replace the automatic masks with click-guided masks.
- If the user wants to keep both, they can set a different output directory or use layers.
- The Review tab picks up masks from the output directory automatically — no integration work needed.

#### Layer support (optional, deferred)

For now, click propagation writes directly to the output root. Layer nesting (writing to `layers/{name}/`) is a stretch goal once the core flow is validated. The merge-layers system already handles compositing multiple layers.

#### Batch pipeline coexistence

- Activating click mode **releases the batch pipeline** (`self._pipeline.cleanup()`) to free VRAM.
- Deactivating click mode releases the video predictor. The batch pipeline will rebuild lazily on next use.
- The `_pipeline_lock` mutex still applies: click mode operations acquire it to prevent concurrent access.
- The user cannot run batch masking while click mode is active (run button should be disabled, or we show an error).

---

## Files Changed

| File | Change | Lines (est.) |
|------|--------|-------------|
| `sam3_click_session.py` | **Full rewrite** — video session wrapper | ~200 |
| `reconstruction_zone.py` | **Major edit** — click UI + handlers | ~400 changed/added |
| `sam3_pipeline.py` | **No change** — video pipeline stays independent | 0 |
| `reconstruction_pipeline.py` | **Minor** — possibly remove `get_sam3_click_handles` (dead code after rewrite) | ~10 |

---

## SAM3 API Reference (verified)

Source: `models/sam3/sam3/model/sam3_base_predictor.py`

### `start_session`
```python
handle_request({
    "type": "start_session",
    "resource_path": str_directory_path,  # or list of PIL images
    "offload_video_to_cpu": True,         # keep frame tensors in system RAM
})
# Returns: {"session_id": "uuid-string"}
```

### `add_prompt` (point prompts)
```python
handle_request({
    "type": "add_prompt",
    "session_id": sid,
    "frame_index": 0,
    "points": [[x1, y1], [x2, y2]],      # pixel coordinates
    "point_labels": [1, 0],               # 1=foreground, 0=background
    "clear_old_points": True,             # replace previous points on this frame
    "obj_id": None,                       # optional: track as specific object
})
# Returns: {"frame_index": 0, "outputs": {"out_binary_masks": ndarray, ...}}
```

### `propagate_in_video`
```python
# Streaming API — yields one response per frame
for response in handle_stream_request({
    "type": "propagate_in_video",
    "session_id": sid,
    "propagation_direction": "both",      # "both", "forward", "backward"
    "start_frame_index": None,            # optional: propagate from this frame
}):
    frame_idx = response["frame_index"]
    outputs = response["outputs"]         # {"out_binary_masks": (N, H, W) bool}
```

### `close_session`
```python
handle_request({
    "type": "close_session",
    "session_id": sid,
    "run_gc_collect": True,
})
```

### Frame loading from directory

When `resource_path` is a directory string, SAM3's `io_utils.py` loads frames:
- Enumerates `*.jpg`, `*.jpeg`, `*.png`, etc.
- Sorts numerically by stem, falls back to lexicographic
- With `async_loading_frames=True`: uses `AsyncImageFrameLoader` (lazy, memory-efficient)
- Resizes to model's internal size (1024px) as float16 tensors
- With `offload_video_to_cpu=True`: tensors stay in system RAM

**Critical**: the frame order SAM3 uses internally (sorted filenames) must match our `frame_names` list exactly, so that `frame_index` values correspond to the correct files.

---

## Edge Cases and Risks

### Frame ordering mismatch
SAM3's `io_utils.py` sorts frames by numeric stem first, falling back to lexicographic. Our `frame_names` list must use the **identical sort order**. Solution: replicate the exact sort from `io_utils.py:176-189`.

### Large datasets (1000+ frames)
- `async_loading_frames=True` prevents memory blowup during session init
- `offload_video_to_cpu=True` keeps tensors out of VRAM
- Propagation is streaming, so we can report progress per frame
- Frame navigation reads from disk on demand (not cached)

### VRAM contention
- Batch pipeline and video predictor cannot coexist. Click mode activation releases the batch pipeline.
- If the user cancels click mode mid-propagation, we must still clean up the video predictor.

### Fisheye circle mask
- Applied during propagation output writing (same as batch pipeline).
- The circle mask is generated per-frame based on frame dimensions. For fisheye datasets, all frames should be the same size.
- Circle mask is OR'd with the object mask (same logic as current `_render_click_overlay`).

### Cancel during propagation
- `cancel_event` is checked between frames in the propagation loop.
- On cancel: stop iterating, write masks for frames processed so far, close session.
- Partial output is valid — the user can re-propagate later.

### No clicks added
- "Propagate All" button is disabled until at least one frame has clicks.

### Session init failure
- If SAM3 video predictor fails to load (missing weights, CUDA error), catch the error, show in status, uncheck the checkbox.
- Do NOT leave the batch pipeline released — rebuild it on failure.

---

## Not in Scope

| Feature | Reason |
|---------|--------|
| Bidirectional propagation UI (separate fwd/bwd) | Default "both" handles most cases. Add later if needed. |
| Per-object tracking (separate masks per object) | All objects merged (logical OR) like `sam3_pipeline.py`. Multi-object tracking adds UI complexity. |
| Review tab integration | Masks land in the standard output dir. Review tab already discovers them. No work needed. |
| Undo/redo within click session | `remove_last_click` + `reset_frame` cover the basics. Full undo stack is overkill. |
| Click mode during batch run | Mutually exclusive. Too complex to interleave. |
| Layer output for click propagation | Deferred. Direct output first, layer nesting as stretch goal. |

---

## Testing

No automated test suite exists for this project. Verification is manual:

1. **Session init**: Activate click mode with a fisheye dataset (972 frames). Confirm status shows frame count, first frame displays.
2. **Frame navigation**: Prev/next buttons, slider, manual index entry. Confirm correct frame loads.
3. **Click → mask**: Left-click on a shadow. Confirm red overlay appears on the object. Right-click to exclude. Confirm mask updates.
4. **Multi-frame prompting**: Navigate to frame 500, add a click. Navigate back to frame 0. Confirm clicks on both frames are tracked.
5. **Propagation**: Click "Propagate All". Confirm progress bar advances. Check output directory for mask PNGs.
6. **Fisheye circle**: With fisheye circle mask enabled, confirm propagated masks include the circle.
7. **Cancel**: Start propagation, cancel mid-run. Confirm partial masks written, session cleaned up.
8. **Deactivate**: Uncheck click mode. Confirm VRAM freed. Run batch masking. Confirm batch pipeline rebuilds.
9. **pyimage bug regression**: Activate click mode, add clicks, deactivate, run Preview. Confirm no `"pyimage doesn't exist"` error.
