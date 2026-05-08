# Review: Click–Video Propagation Plan

This document responds line-by-line to `click-video-propagation-plan.md`, grounded in the current **reconstruction-zone** codebase (`reconstruction_gui/reconstruction_zone.py`, `reconstruction_pipeline.py`, `sam3_click_session.py`, `sam3_pipeline.py`) and `reconstruction_gui/docs/MASK_TAB.md`.

---

## 1. Mask tab: function, scope, and process (as implemented)

### 1.1 Role of the tab

Per `MASK_TAB.md`, the Mask tab drives **automated** detection and segmentation to produce binary masks for photogrammetry. It is organized into:

| Area | Scope |
|------|--------|
| **Detection** | Input/output, SAM3 mode (Hybrid vs Unified Video / “tracked”), model/geometry, prompts, file options, output options |
| **Detection & Refinement** | Optional stages (shadow, SAM refine, edge injection, alpha matting, ensemble, COLMAP validation) — **still applied** when Hybrid runs; when GUI selects Unified Video / temporal tracking, widgets collapse but the **batch pipeline** can still run full `process_image` after SAM3 tracking (see below) |
| **Post-Processing** | Dilate, fill holes, equirect nadir/pole, fisheye circle |
| **Static Masks** | User-painted layers under `static_masks/` (present in code; briefly mentioned in tab docs) |
| **Interactive Click Mode (PVS)** | SAM3 **image** model, single preview frame, click → `predict_inst`, optional save to `layers/{click_layer}/` |
| **Run / queue / progress** | Main worker and batch queue |

The GUI comment at `reconstruction_zone.py` (worker, ~3259) states: **the Output field is the masks folder** (`mask_out_dir = out`), optionally `out/layers/{layer_name}/` when the batch layer name is set.

### 1.2 Hybrid vs temporal (“Unified Video”) batch path

- **Hybrid**: `process_directory` → per image `process_image` → detection + combine + optional refinement + postprocess → write `(1 - result.mask) * 255` to PNG (mask convention used by the pipeline).
- **Temporal tracking** (`MaskConfig.temporal_tracking`): `_start_tracking_session` in `reconstruction_pipeline.py` (~2025–2047) issues **two** `handle_stream_request` **`propagate_in_video`** calls when needed: **`"forward"`**, then **`"backward"`** only if **`prompt_idx > 0`** (keyframe not frame 0), merging outputs so backward fills frames the forward pass missed. That is a **conditional, manual** split in batch code — not because **`"both"`** is unsupported.
- **SAM3 API (`sam3_base_predictor.py` ~251–284):** **`propagation_direction`** accepts **`"both"`**, **`"forward"`**, and **`"backward"`**. **`"both"`** runs **forward then backward inside a single** `handle_stream_request` stream. For **click mode**, where the user may prompt **any** frame, **`propagation_direction="both"`** is **valid, supported, and simpler** than reimplementing the batch keyframe optimization by hand.
- After tracking, **each frame** still goes through **`process_image(..., precomputed_mask=precomputed)`** (`reconstruction_pipeline.py` ~3020–3021 for directories; ~2806 for video path) — detection is skipped when a precomputed mask is supplied, but **combine, SAM refine, shadow, and final postprocess** still run; the **`process_image`** docstring (~2155–2156) states **“Full post-processing still applies.”** MASK_TAB’s “Unified Video bypasses refinement/post” is therefore **GUI messaging / collapsed sections**, not “pipeline output is raw SAM3 only.”

### 1.3 Current click mode (PVS)

- Activation uses **`_resolve_preview_image()`** (`reconstruction_zone.py` ~2133–2157): it prefers the **preview image** path when set, else falls back to **Input**. That is **not** the same as always using the Mask tab **Input** folder. A **video click session** should use **`self.input_entry.get()`** (or the same path rules as batch) **directly**, not `_resolve_preview_image()`, so frame indices align with the dataset being masked.
- Uses cached **`MaskingPipeline`** + **`get_sam3_click_handles()`** → **`SAM3ClickSession`** (`predict_inst`), with **`_pipeline_lock`** around `set_image` / `predict`.
- **Save** writes under **`{output}/layers/{click_layer_name}/{stem}.png`** with optional fisheye circle compositing — **not** the same path as default batch masks (which use `mask_out_dir` = output root or `output/layers/name` for batch layer name).

---

## 2. Plan review — line-by-line verification (SAM3 upstream)

Second-pass verification against SAM3 sources (`sam3_multiplex_tracking.py`, `sam3_tracking_predictor.py`, `sam3_base_predictor.py`, `sam3_video_inference.py`, `model_builder.py`, `io_utils.py`) and reconstruction-zone GUI.

### 2.1 Confirmed correct (plan vs code)

| Plan ref | What was checked | Result |
|----------|------------------|--------|
| **L17–21** (component table) | `sam3_click_session.py`, `sam3_pipeline.py`, `reconstruction_zone.py`, `reconstruction_pipeline.py` | File roles and cited locations match the codebase (GUI line numbers may still drift slightly over time). |
| **L38** | Video `add_prompt` with points | Point prompts are accepted on the video path; routing verified at **`sam3_multiplex_tracking.py` ~2686–2718** (per upstream read). |
| **L69** (RAM) | Float16 frame tensor size | **972 × 3 × 1008 × 1008 × 2 bytes ≈ 5.9 GB** (see **L1024 → 1008** correction below). |
| **L114** | `build_sam3_multiplex_video_predictor` | **`async_loading_frames` exists and defaults to `True`** (CLI `--help` / builder signature). |
| **L115** | FA3 / SDPA fallback | Patch present at **`sam3_pipeline.py` ~148–169** (slight line offset vs older “156–169” cite). |
| **L117** | Frame sort | **`io_utils.py` ~176–189**: numeric stem sort, lexicographic fallback — confirmed. |
| **L171** | Mutex | **`_pipeline_lock`** at `reconstruction_zone.py` ~168–171; batch + click paths use it — confirmed. |
| **L376–420** | API reference | **`start_session`**, **`add_prompt`**, **`propagate_in_video`**, **`close_session`** shapes match **`sam3_base_predictor.py`** (per upstream read). |
| **`propagate_in_video` / `"both"`** | `sam3_base_predictor.py` ~251–284 | **`"both"`** is a **valid** `propagation_direction`; implemented as **forward then backward** in **one** stream. Prefer for **click propagation** when prompts may sit on any frame. |
| **L453** | Fisheye + mask combine | **`np.maximum`** at `reconstruction_zone.py` ~2750 and ~2838 — binary OR equivalent — confirmed. |
| **L40** (checkpoints) | Weights | **Image**: `facebook/sam3` → **`sam3.pt`**. **Video multiplex**: `facebook/sam3.1` → **`sam3.1_multiplex.pt`** (`model_builder.py` ~644–645 vs ~1212–1213). Different checkpoints — plan wording is correct; specificity (SAM3 vs SAM3.1) matters for docs and disk. |

### 2.2 Errors, flaws, or risky assumptions

| Severity | Topic | Plan location | Issue / fix |
|----------|--------|---------------|-------------|
| **CRITICAL** | **Point coordinates** | L123, L392–393, GUI mapping L275–276 | Plan implies **pixel coordinates**. Upstream: point prompts go to **`add_sam2_new_points` → `tracker.add_new_points`** with **`rel_coordinates=True`** (default; `sam3_base_predictor` does not override). Then **`points = points * self.image_size`** with **`image_size = 1008`**. Inputs must be **normalized to [0, 1] relative to original frame width/height**: e.g. `x_norm = ox / orig_w`, `y_norm = oy / orig_h`. **GUI must convert display → original pixels → normalized** before `add_prompt`. |
| **CRITICAL** | **`obj_id` for point prompts** | L123, L395 | Plan shows **`"obj_id": None` (optional)**. Upstream **`sam3_multiplex_tracking.py` ~2706–2708** asserts: *when points are provided, **`obj_id` must be provided***. Session must allocate/track IDs. For single merged object scope, **`obj_id=0`** (or explicit small int) is the minimal fix; document if multi-object is added later. |
| **HIGH** | **Internal resolution** | L69, L428 | Plan says **1024px**; SAM3 video uses **`image_size = 1008`** (`sam3_video_inference.py` ~36; `model_builder.py` ~1205). Update memory text and any “1024” prose. |
| **MEDIUM** | **Click `add_prompt` score** | L97, L125 | Upstream sets **`obj_id_to_score[obj_id] = 1.0`** for click-added objects (`sam3_multiplex_tracking.py` ~2863). **Returned “score” is not a quality signal** — UI should omit it or label it as non-informative. |
| **MEDIUM** | **Cancel vs session** | L457–458 | **Contradiction:** “close session” on cancel vs “re-propagate later” without re-encoding. **Fix:** on cancel, **stop the stream**, optionally **write partial files**, **keep session open** so the user can edit prompts and re-run **`propagate_in_video`**. Only **full teardown** (checkbox off / tab policy) should **`close_session`**. |
| **MEDIUM** | **Points vs text** | L123 (not stated) | **`sam3_multiplex_tracking.py` ~2703**: when **`points`** are set, **`text_str` and `boxes_xywh` must be `None`** (assert). Click-video session must send **only** point-based `add_prompt` calls — do not merge batch Remove/Keep text into the same request. |
| **MEDIUM** | **`reset_session` scope** | L143–150 | **`reset_session`** (e.g. `sam3_base_predictor.py` ~73–74) clears **entire** session — too heavy for “clear one frame.” Prefer: **re-`add_prompt`** with remaining clicks and **`clear_old_points=True`**; for “no prompts left on frame,” try **empty points** if API allows, else **`remove_object`** for that **`obj_id`** only — verify assertions for empty `points`. |
| **Erratum (this review)** | **Plan L21 line numbers** | §2.2 (prior row) | An earlier version of this review **incorrectly** called the plan’s **`SAM3Segmenter` ~730** cite “stale.” **`SAM3Segmenter`** and **`get_sam3_click_handles`** (~**1678** on **`MaskingPipeline`**) are **different symbols**; the original plan table can reference **both** correctly. |
| (unchanged) | **`self._pipeline.cleanup()`** | L233, L354–355 | **`MaskingPipeline`** (`reconstruction_pipeline.py` ~1524) **has no `cleanup()`** — `self._pipeline.cleanup()` would raise **`AttributeError`**. Use **`_release_pipeline_locked()`** (`reconstruction_zone.py` ~3030) / **`_release_pipeline()`** (~3058). |
| (unchanged) | **Output directory** | L293–346 | Align mask writes with **where the user expects masks for that workflow**: main **`_worker`** uses **`mask_out_dir = Path(output_entry)`** (and `layers/...` when set, ~3261–3267). The **masking queue** path derives **`masks_dir = input_dir.parent / f"{input_dir.name}_masks"`** (~3601) plus optional layers — different entry point. Click propagation should follow the **same destination as the interactive run** (typically **`output_entry`** / `_worker` rules), not **`sam3_pipeline.py`’s** standalone **`output/masks`** layout unless deliberately unified. |
| (unchanged) | **Mask PNG polarity** | L134 | **Verified discrepancy:** batch **`process_directory`** writes **`(1 - result.mask) * 255`** (`reconstruction_pipeline.py` ~3047, ~2817, ~3484) — masked-out regions **black (0)**, keep **white (255)**. Current **click save** uses **`mask * 255`** (`reconstruction_zone.py` ~2863) — **opposite** convention. Video click propagation must match **batch** when writing beside batch masks: use **`(1 - mask) * 255`** (after defining `mask` consistently as “remove” semantics). |
| (resolved) | **`propagation_direction: "both"`** | L59, L406 | **`sam3_base_predictor.py` ~251** validates **`["both", "forward", "backward"]`**; ~**272–284** implements **`"both"`** as forward then backward in **one** stream. **Use `"both"` for click propagation** unless a narrower direction is required. Batch `_start_tracking_session` uses two calls only to **avoid backward work** when the text prompt frame is **0**. |
| (unchanged) | **`get_frame_mask` post-propagation** | L127–129 | Cache propagation outputs or reload from disk after propagate. |
| (unchanged) | **Post-processing parity** | L338–346 | Raw SAM3 write vs **`process_image(precomputed_mask=...)`** like temporal batch — product decision. |
| (unchanged) | **Input vs preview** | L226–240 | Video session should use **`input_entry`**; document UX. |
| (unchanged) | **Verbose PVS logging** | `reconstruction_zone.py` | Gate/remove diagnostic **`self.log`** in click overlay before ship. |

### 2.3 Suggestions (implementation clarity)

- **`offload_video_to_cpu` + `async_loading_frames`**: Together, **`async_loading_frames=True`** uses the lazy loader (**`io_utils.py` ~197–201**), passing **`offload_video_to_cpu`** into **`AsyncImageFrameLoader`** — frames load lazily and stay on CPU. This is the right default for large datasets; the implementation plan should state this explicitly (not only VRAM table).
- **`async_loading_frames` + directory `resource_path`**: Now **confirmed** at builder level; reconstruction-zone can still prefer a spike for end-to-end GUI + cancel + progress.

### 2.4 Minor / documentation

- **ASCII art (L167)**: Stray replacement characters in the plan — cosmetic only.
- **MASK_TAB vs tooltip**: Tab doc says Unified Video bypasses refinement/post; **pipeline** still applies post when using temporal tracking with precomputed masks. Reconcile **documentation** with `process_image` behavior when convenient — **does not block** click-video implementation.

---

## 3. Detailed response plan (implementation-oriented)

### Phase A — Spikes (before large rewrites)

1. **`propagate_in_video` + `"both"`**: **Resolved** — SAM3 **`sam3_base_predictor.py`** supports **`"both"`** (~251–284). **Prefer `propagation_direction="both"`** for click propagation (prompts on arbitrary frames). No verification spike required for direction support.
2. **`start_session`**: **`async_loading_frames`** is confirmed on the builder; still validate **directory `resource_path`** end-to-end in the GUI (ordering, cancel, progress) vs PIL-list loading if behavior differs.
3. **Coordinates (resolved)**: **`add_prompt` point coordinates must be normalized to [0, 1]** (divide by original frame width/height after display→pixel mapping). Regression-test overlay alignment on at least one non-square and one high-res frame.
4. **Mask PNG convention (resolved for v1):** When writing masks next to **batch** output, use **`(1 - mask) * 255`** to match **`process_directory`** (`reconstruction_pipeline.py` ~3047). Optionally align legacy **click-save** (`mask * 255`) in a separate cleanup PR so all GUI paths share one helper.

### Phase B — Session module (`sam3_click_session.py` or new file)

1. Introduce **`SAM3VideoClickSession`** (or rename file) wrapping **`build_sam3_multiplex_video_predictor`**, reusing **FA3 detection + decoder patch** from **`sam3_pipeline.py`** (import shared helper to avoid triplication with `reconstruction_pipeline.py`).
2. Implement **`start`**, **`add_click`**, **`propagate`**, **`close`**, **`remove_last_click` / `reset_frame`** as in the original plan, with corrections:
   - **`add_click` → `add_prompt`**: Pass **`points`** as **[[x_norm, y_norm], …]** in **[0, 1]**, never raw pixels; include a fixed **`obj_id`** (e.g. **`0`**) for all point prompts in the **single merged object** scope (multiple distinct objects on one frame would need **multiple `obj_id`s** — out of scope until multi-object UI exists). Keep **`text_str` / `boxes_xywh` absent or `None`** on those calls.
   - **`get_frame_mask`**: After propagation, populate a **`_propagated_masks`** dict or reload from disk.
   - **Cancel**: **Stop the stream**; optionally flush partial PNGs; **do not `close_session`** on cancel — keep the session so the user can adjust clicks and re-propagate. Reserve **`close_session`** for full teardown (deactivate click mode / exit).
3. **Frame list / `frame_index` alignment**: SAM3 does **not** expose an ordered filename list from the predictor after init (tensors only). **Replicate `io_utils.py` sort exactly** (~176–189) for the GUI’s `frame_names` / slider, and cross-check **`num_frames`** against session state. “Read back the predictor’s ordering” is **not** available as a first-class API — exact sort replication (or passing an explicit ordered list into `start_session` if used) is the safe approach.

### Phase C — GUI (`reconstruction_zone.py`)

1. Replace click-section widgets per the plan’s wireframe; wire **Input** directory for **video session start**; keep **preview navigation** in sync (frame index ↔ file list).
2. On activate: **`_release_pipeline_locked()`** (or equivalent) to free the **image** model — **do not call nonexistent `cleanup()`**; ensure **failure path** restores pipeline if video init fails.
3. **Propagate** output: match **the mask destination for the workflow in use** — same rules as **`_worker`** (`mask_out_dir` from **`output_entry`**, optional **`layers/{batch_layer}`**, ~3261–3267). If click propagation is ever wired from the **queue** path, align with that path’s **`masks_dir`** (~3601) instead. Do not silently use standalone **`sam3_pipeline`’s `output/masks`** layout.
4. **Mutex**: All **`handle_request` / `handle_stream_request`** calls on the video predictor from workers should run under **`_pipeline_lock`** (or merge video + image into one GPU guard).
5. **Run Masking** while click mode active: disable run button or hard-block with a clear message (plan L357–358).
6. **Status line**: Do not show **`add_prompt` “score”** as quality (upstream **1.0** for click objects); remove or relabel.

### Phase D — Optional parity with batch quality (**future enhancement**, not v1 blocker)

1. Running **`process_image(..., precomputed_mask=...)`** per frame after propagation would match **batch temporal** quality (shadow, SAM refine, dilate, fill holes, geometry masks) but **significantly** increases complexity (reload every image, hold or rebuild **`MaskingPipeline`** state, align cancel/progress). Treat as a **later** milestone unless product explicitly prioritizes parity over latency.
2. For v1, prefer **raw SAM3 (+ optional fisheye circle)** writes matching **batch PNG polarity** (`(1 - mask) * 255`), and **document in UI** that refinement/post stages were not applied unless the user runs a **batch** pass afterward.

### Phase E — Testing (extend the plan’s manual list)

1. **Output path**: Propagate with empty batch layer name vs with batch layer name — files land next to batch masks where expected.
2. **Polarity**: Open one PNG in Review / COLMAP prep — masked pixels match pipeline convention.
3. **Regression**: **`pyimage`** teardown (plan L493–494) — keep the **no `configure(image=None)`** rule.
4. **Failure**: Force video predictor init failure — **pipeline** returns on next Preview/Run.
5. **Normalized clicks**: Known fiducial at frame corners/center — overlay matches SAM mask after **`add_prompt`** with **`obj_id=0`** and **[0,1] points**.

---

## 4. Summary verdict

The original plan is **directionally correct** (move interactive prompting from image `predict_inst` to multiplex video + propagation) and aligns with where the product pain is.

**Must-fix before implementation (upstream-verified):** **`add_prompt` points in normalized [0, 1] space** (not pixels); **`obj_id` required** for point prompts (e.g. fixed `0` for single merged object — multiple **`obj_id`** only if multi-object UI); **internal image size 1008** (replace “1024” in plan copy); **do not treat `add_prompt` score as quality** (expect **1.0**); **points and text are mutually exclusive** per request; **cancel should not close the session** if re-propagate without full re-init is a goal; **PNG writes** must match batch polarity **`(1 - mask) * 255`**; use **`propagation_direction="both"`** (supported in **`sam3_base_predictor.py` ~251–284**).

**Other fixes:** correct **pipeline teardown** (`_release_pipeline_locked`, not `cleanup()`), **mask output path** ( **`_worker` `mask_out_dir`** vs queue **`masks_dir`** vs standalone `sam3_pipeline`), **PNG polarity** (`(1 - mask) * 255` to match batch, not current click-save `mask * 255`), **`get_frame_mask` after propagation**, **narrow `reset_session` / per-frame clear** semantics, and **Phase D** as optional **post-v1** parity.

**Review erratum (corrected in §2.2):** an earlier draft wrongly flagged the plan’s **`SAM3Segmenter` ~730** citation; **`SAM3Segmenter`** and **`get_sam3_click_handles`** are different entries.

---

*Generated as a structured review of `click-video-propagation-plan.md` against the reconstruction-zone codebase. Section 2.1–2.3 incorporate a second-pass line-by-line verification against SAM3 upstream sources (normalized points, `obj_id`, 1008, score behavior, text/points exclusivity, offload+async interaction, **`propagation_direction="both"`** support). Third pass: reconciliation with `sam3_base_predictor.py` / batch worker paths and correction of the SAM3Segmenter erratum.*
