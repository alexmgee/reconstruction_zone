# SAM3 Video Tracking Integration — Detailed Implementation Plan

**Date:** 2026-04-24
**Goal:** Replace the separate Hybrid/Unified Video modes with a single pipeline that combines SAM3's temporal tracking with the full post-processing chain (cubemap decomposition, shadow detection, SAM refine, matting, quality scoring, review workflow).

---

## Table of Contents

1. [Current State](#1-current-state)
2. [Target Architecture](#2-target-architecture)
3. [Critical Constraints Verified Against Code](#3-critical-constraints)
4. [Detailed Change Specification](#4-detailed-changes)
5. [Pitfalls and Mitigations](#5-pitfalls)
6. [Build Sequence](#6-build-sequence)
7. [Rollback Plan](#7-rollback)

---

## 1. Current State

### Hybrid mode (default)
- `_worker()` calls `pipeline.process_directory()` or `pipeline.process_video()`
- `process_directory()` loops over images, calls `process_image()` for each
- `process_image()` calls `_segment()` → `_combine_masks()` → SAM refine → edge inject → VOS/temporal → shadow → postprocess → matting
- **Directory processing resets temporal state per frame** (`reset_sequence()` at line 2720) — no tracking across frames
- **Video processing does NOT reset** — VOS/temporal state persists, but the underlying detection is still per-frame (YOLO/SAM3/RF-DETR runs fresh each frame)

### Unified Video mode
- `_worker()` calls `pipeline.process_directory_sam3_unified()` or `pipeline.process_video_sam3_unified()`
- These delegate to `SAM3VideoPipeline.process_frames()` in `sam3_pipeline.py`
- Loads ALL frames as PIL images, starts a SAM3 video session, prompts on frame 0, propagates forward
- Output: raw binary masks (N, H, W) per frame, merged via logical OR, saved as 0/255
- **No post-processing whatsoever** — no cubemap, no shadow, no refine, no quality scoring, no review folder
- **Mask convention is inverted** — writes `mask * 255` (white=object) instead of Hybrid's `(1 - mask) * 255` (white=background)

### What Unified Video gets right
- Temporal object tracking with identity persistence
- Efficient single-pass processing via Object Multiplex
- VRAM management via `offload_video_to_cpu=True`
- Flash Attention 3 auto-detection with MATH fallback

### What Unified Video gets wrong
- Bypasses the entire post-processing pipeline
- No cubemap decomposition (equirect images processed raw)
- No quality scoring or review workflow
- Mask value convention differs from Hybrid
- No per-image progress feedback in GUI
- No skip_existing, multi-label, inpaint support

---

## 2. Target Architecture

### Concept

SAM3 video tracking becomes an **optional detection source** within the existing pipeline, not a separate pipeline. The flow:

```
┌──────────────────────────────────────────────────┐
│  process_directory() / process_video()           │
│                                                  │
│  IF temporal_tracking enabled AND SAM3 available:│
│    1. Start SAM3 video session (all frames)      │
│    2. Add text prompts on frame 0                │
│    3. Propagate → collect per-frame masks         │
│    4. For each frame:                            │
│       a. Get tracked mask from propagation       │
│       b. Wrap in MaskResult(s)                   │
│       c. Feed into existing post-processing:     │
│          combine → refine → edge → shadow →      │
│          postprocess → matting                   │
│       d. Quality score → save → review           │
│    5. Close session                              │
│                                                  │
│  ELSE (current Hybrid behavior):                 │
│    For each frame:                               │
│       a. _segment() — fresh detection            │
│       b. Full post-processing chain              │
│       c. Quality score → save → review           │
└──────────────────────────────────────────────────┘
```

### What changes for the user

- The "SAM3 mode: [Hybrid | Unified Video]" segmented button becomes a **checkbox**: `□ Temporal tracking (SAM3 video)`
- When checked: detection is done once (on a configurable keyframe), tracking propagates to all other frames, full post-processing runs on every frame
- When unchecked: current Hybrid behavior (per-frame detection, no tracking)
- All other controls remain functional regardless — geometry, post-processing, quality scoring, review workflow

---

## 3. Critical Constraints Verified Against Code

### Constraint 1: SAM3 propagation runs to completion

**Verified.** The `handle_stream_request({"type": "propagate_in_video"})` call in `sam3_pipeline.py:264-269` yields responses frame-by-frame, but the propagation is a single forward pass. You cannot pause, modify a mask, and resume.

**Implication:** We cannot modify SAM3's tracked masks between frames and feed corrections back into the tracker. Post-processing runs AFTER all propagation is complete, not interleaved with it.

**Mitigation:** This is fine. The tracked masks are the detection step. Post-processing is applied independently per frame after all tracking is done. We don't need to feed post-processed masks back to the tracker — the tracker's job is object identity, not mask refinement.

### Constraint 2: SAM3 outputs are per-object binary masks

**Verified at sam3_pipeline.py:279-285.** Output format: `out_binary_masks` — shape `(N, H, W)`, dtype bool. N = number of tracked objects. Currently merged via `np.any(axis=0)` into a single combined mask.

**Implication:** We can create individual `MaskResult` objects per tracked object (preserving per-object metadata like object ID), not just one combined mask. This is better for multi-label output and quality scoring.

### Constraint 3: SAM3 needs all frame paths at session start

**Verified at sam3_pipeline.py:227-231.** `start_session` takes `resource_path` which is a list of PIL images. However, the API supports async loading (`async_loading_frames=True`) so it returns immediately after the first frame is available.

**Implication for directories:** We need to scan all frame files and load them as PIL images before starting the session. This is already done in the current Unified Video path. Memory concern: all frames are loaded as PIL images simultaneously.

**Implication for videos:** Frames must be extracted to a temp directory first (already done in `process_video_sam3_unified`).

**Mitigation for memory:** `offload_video_to_cpu=True` keeps SAM3's internal frame tensors in system RAM instead of VRAM. The PIL image list can be released after session start. For very large sequences (1000+ frames), we may need to process in chunks.

### Constraint 4: process_directory() resets temporal state per frame

**Verified at reconstruction_pipeline.py:2720.** `self.reset_sequence()` is called before every image in the directory loop.

**Implication:** When temporal tracking is enabled for directories, we must NOT call `reset_sequence()` — the tracked masks provide the temporal continuity instead of VOS/temporal averaging.

### Constraint 5: Equirect cubemap decomposition happens inside process_image()

**Verified at reconstruction_pipeline.py:1961-1965.** When geometry=EQUIRECTANGULAR, `process_image()` immediately delegates to `_process_equirectangular()`, which does cubemap decomposition and calls `_segment()` per face.

**Implication:** SAM3 video tracking operates on the original frame (equirect or pinhole). For equirect, the tracked mask is at full equirect resolution. We have two options:

- **Option A:** Feed the tracked equirect mask directly into post-processing (skip cubemap decomposition). Simpler but loses the per-face segmentation benefit.
- **Option B:** Use the tracked mask as a coarse guide, still run cubemap decomposition for per-face refinement. More complex, questionable value.
- **Recommended: Option A.** The cubemap decomposition exists because per-frame detection on equirect images is poor (objects distorted at poles). SAM3 video tracking doesn't have this problem — it tracks objects identified on a keyframe (which can be processed via cubemap). The tracked masks at equirect resolution are already good.

### Constraint 6: Mask value convention

**Verified.** Hybrid saves `(1 - mask) * 255` (white=background, Metashape convention). Unified saves `mask * 255` (white=object). The integration must use the Hybrid convention since all downstream tools (Review tab, Metashape import) expect it.

### Constraint 7: VOS propagation already exists as a similar concept

**Verified at reconstruction_pipeline.py:1967-1979.** VOS propagation (LiVOS/Cutie) already does keyframe detection + propagation between frames. It short-circuits `process_image()` on non-keyframes, returning a raw propagated mask WITHOUT post-processing.

**Implication:** VOS propagation has the same "skip post-processing" bug. When we fix this for SAM3 tracking, we should fix VOS propagation too — propagated masks should also go through post-processing.

**Decision:** SAM3 temporal tracking and VOS propagation are mutually exclusive — they solve the same problem (temporal consistency). When SAM3 tracking is enabled, VOS propagation should be disabled. The GUI already disables most controls when Unified Video is selected; we keep similar logic.

---

## 4. Detailed Change Specification

### 4.1 MaskConfig changes (`reconstruction_pipeline.py`)

**Remove:**
```python
sam3_unified: bool = False
sam3_video_config: Optional[Dict] = None
```

**Add:**
```python
temporal_tracking: bool = False        # Use SAM3 video predictor for temporal tracking
temporal_prompt_frame: int = 0         # Frame index to apply text prompts on
```

Rationale: `sam3_unified` was a mode switch. `temporal_tracking` is a feature toggle within the existing pipeline. `temporal_prompt_frame` replaces `SAM3VideoConfig.prompt_frame_index`.

### 4.2 MaskingPipeline.__init__() changes

**Remove:** The entire `sam3_video_pipeline` initialization block (lines 1621-1641 approx). The SAM3VideoPipeline class is no longer instantiated as a standalone pipeline.

**Add:** A lightweight session manager attribute:
```python
self._sam3_video_session = None  # Active SAM3 video session state
self._sam3_tracked_masks = None  # Dict[int, np.ndarray] — precomputed tracked masks per frame index
```

No model loading here — the SAM3 video predictor is loaded on demand when `temporal_tracking=True` and a directory/video is processed.

### 4.3 New method: _start_tracking_session()

```python
def _start_tracking_session(
    self,
    frame_paths: List[Path],
) -> Dict[int, np.ndarray]:
    """Start SAM3 video session, run prompts, propagate, return per-frame masks.

    Args:
        frame_paths: Ordered list of frame image paths.

    Returns:
        Dict mapping frame index → combined binary mask (0/1 uint8, H x W).
        Frames with no detections map to all-zeros mask.
    """
```

This method:
1. Loads frames as PIL images
2. Builds the SAM3 multiplex video predictor (with FA3 detection + MATH fallback, same as current `sam3_pipeline.py`)
3. Starts a session with `offload_video_to_cpu=True`
4. Combines all remove_prompts into period-separated string and calls `add_prompt` on the configured keyframe
5. Runs `propagate_in_video(direction="forward")` via streaming
6. Collects `out_binary_masks` per frame, merges per-object masks via `np.any(axis=0)` → 0/1 uint8
7. If `temporal_prompt_frame > 0`, also runs backward propagation to cover frames before the keyframe
8. Closes session, releases PIL images, returns the dict
9. If keep_prompts are set: runs a second session with keep prompts, subtracts those masks from the tracked masks

**Error handling:** If SAM3 video predictor import fails or session crashes, logs a warning and returns None. Callers fall back to per-frame detection.

### 4.4 process_directory() changes

Current flow:
```python
for img_path in image_files:
    self.reset_sequence()          # ← resets temporal state
    result = self.process_image(image, geometry)
    # save mask, review, etc.
```

New flow:
```python
# Pre-compute tracked masks if temporal tracking enabled
tracked_masks = None
if self.config.temporal_tracking:
    tracked_masks = self._start_tracking_session(image_files)
    if tracked_masks is None:
        logger.warning("Tracking session failed, falling back to per-frame detection")

for idx, img_path in enumerate(image_files):
    if tracked_masks is None:
        self.reset_sequence()  # Only reset when NOT tracking

    # Load image
    image = cv2.imread(str(img_path))

    # Get precomputed tracked mask for this frame (or None)
    precomputed = tracked_masks.get(idx) if tracked_masks is not None else None

    # Process with full post-processing pipeline
    result = self.process_image(image, geometry, precomputed_mask=precomputed)

    # save mask, review, quality scoring — UNCHANGED
```

### 4.5 process_image() changes

**New parameter:**
```python
def process_image(
    self,
    image: np.ndarray,
    geometry: ImageGeometry = ImageGeometry.PINHOLE,
    custom_prompts: Optional[Dict[str, List[str]]] = None,
    precomputed_mask: Optional[np.ndarray] = None,  # NEW
) -> MaskResult:
```

**Changed logic:**

```python
# Equirect with geometry_aware: use cubemap strategy
# ONLY when no precomputed mask (tracking provides equirect-resolution masks directly)
if geometry == ImageGeometry.EQUIRECTANGULAR and self.config.geometry_aware and precomputed_mask is None:
    result = self._process_equirectangular(image, custom_prompts)
    self._frame_counter += 1
    return result

# Skip VOS propagation when precomputed masks are provided
# (SAM3 tracking replaces VOS for temporal consistency)
if precomputed_mask is None:
    # Existing VOS propagation check (unchanged)
    if self.vos_propagator is not None and not self.vos_propagator.is_keyframe():
        ...

# Detection step
if precomputed_mask is not None:
    # Use tracked mask directly — skip _segment()
    results = [MaskResult(
        mask=precomputed_mask,
        confidence=0.85,  # tracking confidence — not model confidence
        quality=MaskQuality.GOOD,
        metadata={'method': 'sam3_tracking', 'frame': self._frame_counter}
    )]
else:
    # Normal detection (current behavior)
    results = self._segment(image, custom_prompts, geometry)

# Everything below is UNCHANGED — combine, refine, shadow, postprocess, matting
if not results:
    ...  # existing empty-mask handling

combined_mask = self._combine_masks(results)
# SAM refine, edge inject, VOS step, shadow, final postprocess, matting
# ALL unchanged
```

### 4.6 process_video() changes

Same pattern as process_directory(). Before the frame loop:

```python
tracked_masks = None
if self.config.temporal_tracking:
    # Extract frame paths from video (need temp dir for SAM3 session)
    # OR pass PIL images directly decoded from cv2.VideoCapture
    tracked_masks = self._start_tracking_session_from_video(video_path, start_frame, end_frame, skip_frames)
```

Then in the loop, pass `precomputed_mask=tracked_masks.get(frame_idx)` to `process_image()`.

**Complication:** `process_video()` reads frames with `cv2.VideoCapture` one at a time, but SAM3 needs all frames upfront. Two options:

- **Option A:** Extract all frames to a temp directory first, start SAM3 session, then re-read frames one at a time for post-processing. Double I/O but simple.
- **Option B:** Decode all frames into a list in memory, pass to SAM3, then iterate the same list for post-processing. More memory but no disk I/O.

**Recommended: Option A** for large videos (temp dir), **Option B** for small frame counts. Threshold: 500 frames. The current Unified Video path already does Option A.

### 4.7 process_directory_sam3_unified() and process_video_sam3_unified() — REMOVE

These become dead code. `process_directory()` and `process_video()` now handle temporal tracking internally.

### 4.8 GUI changes (`reconstruction_zone.py`)

**Replace** the SAM3 mode segmented button:
```python
# REMOVE:
self._sam3_mode_btn = ctk.CTkSegmentedButton(
    sam3_mode_row, values=["Hybrid", "Unified Video"],
    command=self._on_sam3_mode_change,
)

# ADD:
self.temporal_tracking_var = ctk.BooleanVar(value=False)
_tt = ctk.CTkCheckBox(sam3_mode_row, text="Temporal tracking",
                       variable=self.temporal_tracking_var, width=0)
Tooltip(_tt, "Track objects across frames using SAM3 video predictor.\n"
             "Detection runs on one keyframe, masks propagate to all others.\n"
             "Full post-processing (shadow, refine, matting) still applies.\n"
             "Requires SAM3. Best for video files and sequential frame sets.")
```

**Remove** `_on_sam3_mode_change()` method and the disable-everything logic for Unified Video mode.

**Update** `_build_mask_config()`:
```python
# REMOVE:
sam3_unified=self.sam3_unified_var.get(),
# ADD:
temporal_tracking=self.temporal_tracking_var.get(),
```

**Update** `_worker()`:
```python
# REMOVE the entire if/else branching for sam3_unified_var:
if self.sam3_unified_var.get() and pipeline.sam3_video_pipeline is not None:
    ...
else:
    ...

# REPLACE with single call (temporal_tracking is in the config):
stats = pipeline.process_directory(...)  # or process_video(...)
```

### 4.9 sam3_pipeline.py — REPURPOSE

Keep the Flash Attention detection + MATH fallback patch logic. Move the session management code into a utility function or into `_start_tracking_session()` within `MaskingPipeline`. The `SAM3VideoPipeline` class itself can be removed or reduced to a session helper.

---

## 5. Pitfalls and Mitigations

### Pitfall 1: Memory — loading all frames as PIL images

**Risk:** A 500-frame sequence of 4K equirect images = ~500 x 30MB = ~15GB RAM just for the PIL list.

**Mitigation:**
- `offload_video_to_cpu=True` keeps SAM3 tensors in system RAM (already implemented)
- Release PIL image list immediately after `start_session` returns (SAM3 encodes frames internally)
- For very large sequences, consider chunked processing: process in batches of N frames, each batch gets its own SAM3 session. This breaks cross-chunk tracking but preserves intra-chunk tracking.
- Log a warning when frame count exceeds a threshold (e.g., 300) suggesting the user process in chunks.

### Pitfall 2: Equirect tracked masks at wrong resolution

**Risk:** SAM3 processes frames at its internal resolution (typically 1024px on the longest side). The tracked masks may be smaller than the original equirect image (e.g., 7680x3840). Current Unified code handles this with `cv2.resize(INTER_NEAREST)` at line 295.

**Mitigation:** After collecting tracked masks from propagation, resize each to match the original frame dimensions before passing to `process_image()`. This is already handled — `process_image` doesn't assume mask resolution.

### Pitfall 3: Tracked mask is a single combined mask, not per-object

**Risk:** The current merge (`np.any(axis=0)`) loses per-object identity. This matters for multi-label output and per-detection quality scoring.

**Mitigation for v1:** Accept this limitation. The combined tracked mask is treated as a single detection in `process_image()`. Multi-label output won't work with temporal tracking (it needs per-class masks). Document this.

**Mitigation for v2 (future):** Preserve per-object masks from propagation. Create one `MaskResult` per tracked object with an object_id in metadata. This enables multi-label and per-object quality scoring.

### Pitfall 4: Keep prompts with temporal tracking

**Risk:** Keep prompts currently work by running the keep text through SAM3's image predictor and subtracting. With temporal tracking, the detection happens once on a keyframe — keep prompts need to work at that stage.

**Mitigation:** In `_start_tracking_session()`, after collecting tracked masks for remove prompts, run a second session with keep prompts on the same frames. Subtract the keep-tracked masks from the remove-tracked masks frame by frame. This is more expensive (two SAM3 sessions) but correct.

**Simpler alternative:** Run keep prompts only on the keyframe via the image predictor (not video), subtract from the keyframe tracked mask, and let propagation carry the subtraction forward. This works because SAM3 tracks the mask shape — if the keyframe mask has a hole where the kept object is, that hole propagates.

**Recommended: Simpler alternative.** One SAM3 video session (remove prompts) + one SAM3 image call on the keyframe (keep prompts, subtracted before propagation).

Wait — that doesn't work with the current API. Prompts are added BEFORE propagation. We can't modify the mask between prompt and propagation.

**Revised approach:** After propagation completes and we have all tracked masks, run SAM3 image predictor on each frame for keep prompts (same as Hybrid does now) and subtract. This is frame-by-frame for keep prompts only — but keep prompts are rare and the subtraction is cheap (no re-tracking needed).

### Pitfall 5: Backward propagation for mid-sequence keyframes

**Risk:** If `temporal_prompt_frame > 0`, frames before the keyframe have no tracked masks.

**Mitigation:** Run propagation in both directions:
1. Forward propagation from keyframe to end
2. Backward propagation from keyframe to start
3. Merge: for frames before keyframe use backward results, for frames at/after use forward results

The SAM3 API supports `propagation_direction: "backward"`. The current Unified Video code only runs forward from frame 0.

### Pitfall 6: The GUI's Unified Video disable-everything pattern

**Risk:** The current `_on_sam3_mode_change()` disables the entire Detection & Refinement and Post-Processing sections when Unified Video is selected. If we change to a checkbox, we need to NOT disable anything — the whole point is that temporal tracking works WITH post-processing.

**Mitigation:** The checkbox simply sets a config flag. No UI disabling needed. All controls remain active. The checkbox could be greyed out when the selected model is not SAM3 (since tracking requires SAM3).

### Pitfall 7: VOS propagation conflict

**Risk:** If both `temporal_tracking` and `vos_propagation` are enabled, they compete — VOS would try to propagate on non-keyframes but SAM3 tracking already provides masks for all frames.

**Mitigation:** When `precomputed_mask` is provided to `process_image()`, skip the VOS propagation check entirely (already specified in section 4.5). VOS and temporal tracking are mutually exclusive at the config level — if temporal_tracking is True, vos_propagation should be forced False during pipeline init. Log a warning if both are set.

### Pitfall 8: Directory images are NOT always sequential

**Risk:** `process_directory()` processes images sorted by filename. For photogrammetry datasets, images may be from different cameras, different viewpoints, or randomly ordered. SAM3 video tracking assumes temporal sequence.

**Mitigation:** Temporal tracking should only be used when the user knows their images are sequential (video frames, time-lapse). The tooltip already says "Best for video files and sequential frame sets." Additionally, when temporal_tracking is enabled for a directory, log a warning: "Temporal tracking treats images as a video sequence. For unordered photogrammetry datasets, disable this option."

---

## 6. Build Sequence

### Step 1: MaskConfig + process_image() signature
- Add `temporal_tracking`, `temporal_prompt_frame` to MaskConfig
- Add `precomputed_mask` parameter to `process_image()`
- Wire the precomputed_mask logic (skip _segment, skip VOS, skip cubemap when mask provided)
- **Verify:** existing Hybrid behavior unchanged when precomputed_mask=None

### Step 2: _start_tracking_session()
- Implement the session manager method
- Port FA3 detection + MATH fallback from sam3_pipeline.py
- Handle forward + backward propagation
- Handle keep prompts (frame-by-frame subtraction after propagation)
- **Verify:** method returns correct Dict[int, ndarray] with 0/1 uint8 masks

### Step 3: process_directory() integration
- Add tracked_masks pre-computation when temporal_tracking=True
- Skip reset_sequence() when tracking
- Pass precomputed_mask per frame
- **Verify:** masks saved with correct convention ((1 - mask) * 255), quality scoring works, review folder populated

### Step 4: process_video() integration
- Add tracked_masks pre-computation (temp dir extraction path)
- Pass precomputed_mask per frame
- **Verify:** same as Step 3 but for video input

### Step 5: Remove old Unified Video code
- Remove `process_directory_sam3_unified()` and `process_video_sam3_unified()` from MaskingPipeline
- Remove `sam3_video_pipeline` attribute from __init__
- Remove `sam3_unified` from MaskConfig
- Simplify SAM3VideoPipeline in sam3_pipeline.py or remove it entirely

### Step 6: GUI changes
- Replace SegmentedButton with checkbox
- Remove `_on_sam3_mode_change()` disable logic
- Simplify `_worker()` — remove Unified Video branching
- Wire `temporal_tracking_var` into `_build_mask_config()`
- Grey out checkbox when model is not sam3/auto

### Step 7: Testing
- Directory of pinhole frames with temporal_tracking=True
- Directory of equirect frames with temporal_tracking=True
- Video file with temporal_tracking=True
- All three with temporal_tracking=False (regression test)
- Keep prompts with temporal_tracking=True
- Large sequence memory test

---

## 7. Rollback Plan

Every step produces a working state:
- After Step 1: Hybrid works exactly as before (precomputed_mask always None)
- After Step 2: New method exists but isn't called yet
- After Step 3-4: temporal_tracking=False still works; temporal_tracking=True is the new path
- After Step 5: Old code removed, but all functionality is preserved in the new path
- After Step 6: GUI updated

If anything breaks, revert to the commit made before this work began (the current GUI-only commit).
