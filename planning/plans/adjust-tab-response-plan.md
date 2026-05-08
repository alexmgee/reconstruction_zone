# Adjust Tab — Response & Implementation Plan

**Companion to:** [`adjust-tab-plan.md`](./adjust-tab-plan.md)

This document does **not** replace the feature plan. It records validation against the current **Reconstruction Zone** codebase, resolves contradictions found during review, and specifies **integration hooks**, **dependency packaging**, **distribution constraints**, and a suggested **delivery sequence**. Use it as the engineering checklist while implementing the Adjust tab.

---

## 1. Purpose and how to use both documents

| Document | Role |
|----------|------|
| `adjust-tab-plan.md` | Product and algorithms: feature inventory, pipeline order, UI sketch, performance targets, optional AI/video/distributed scope. |
| `adjust-tab-response-plan.md` (this file) | Engineering truth layer: what the repo actually does today, what must change, risks, and phased tasks. |

When the two conflict on **facts about the repo**, prefer **this document** until `adjust-tab-plan.md` is manually reconciled.

---

## 2. Validation summary

### 2.1 What aligns well with the codebase

- **GUI stack:** CustomTkinter main app (`reconstruction_zone.py`), left `CTkTabview`, right **shared preview** — matches the plan’s “Adjust owns buffer, renders CTkImage into shared panel” approach.
- **Tab module pattern:** `build_<name>_tab(app, parent)` under `reconstruction_gui/tabs/` — `tabs/adjust_tab.py` + `build_adjust_tab` is the right shape.
- **Engine separation:** A GUI-free `adjust_engine.py` mirrors the architectural rule that **`reconstruction_pipeline.py`** stays independent of tkinter (see `docs/ARCHITECTURE.md`).
- **LUT reuse:** `prep360/core/lut.py` (`LUTProcessor`, `.cube` load/apply) exists and is the correct integration point.
- **Video / ffmpeg:** The repo already assumes **ffmpeg on PATH** for extraction paths (`reconstruction_zone.py` warnings; `prep360/core/extractor.py`, `sharpest_extractor.py`, etc.). Video-centric Adjust workflows fit existing ops assumptions.
- **Design tokens / widgets:** `widgets.py` (`CollapsibleSection`, `slider_row`, `Tooltip`, colour/spacing constants) should be reused so Adjust visually matches Mask/Extract.

### 2.2 Corrections to statements in the initial plan

These items were flagged during review; implementation should **not** rely on the original wording alone.

| Topic | Initial plan implication | Repo reality / correction |
|-------|---------------------------|---------------------------|
| Tab order (overview) | Overview says tab sits **between Extract and Mask**. | Later sections specify **Projects → Adjust → Extract → Mask → …**. The **workflow narrative** (grade before extract) requires Adjust **before** Extract. **Fix the overview** in `adjust-tab-plan.md` when editing it, or treat this file as authoritative for order. |
| `colour-science` “already present” | Listed under already-available packages. | **Not** in root `requirements.txt` and **no** Python imports under the project today. **Must be added and pinned** when chart/CC features ship. |
| Batch concurrency | “ProcessPoolExecutor … existing pattern in masking pipeline”. | **`reconstruction_pipeline.py` uses `ThreadPoolExecutor`** for batch execution; **`ProcessPoolExecutor`** appears in **prep360** modules (e.g. reframing/blur paths). Choose **threads vs processes** for Adjust explicitly (GIL, pickling on Windows, GPU models). |
| Shared preview | “Same navigator as existing tabs.” | Preview is **shared** but behaviour is **mode-dependent** via `_preview_mode`. Adjust needs an explicit mode (see §4) so it does not collide with Mask SAM clicks or mask overlay toggles. |

---

## 3. Current application architecture (relevant subset)

### 3.1 Tab registration

Tabs are added on the main `CTkTabview` in `reconstruction_zone.py` (`Projects`, `Extract`, `Mask`, `Review`, `Align`, `Coverage`). **Adjust is not registered yet.**

Integration minimum:

1. `self.tabs.add("Adjust")` at the chosen index (plan: immediately after `Projects`).
2. `from tabs.adjust_tab import build_adjust_tab` (or equivalent package path).
3. `build_adjust_tab(self, self.tabs.tab("Adjust"))` after the tab exists.

### 3.2 Right-hand column switching

`_on_tab_change` swaps three conceptual right panels:

- **Projects** → project detail panel (`_proj_right_panel`).
- **Align** → alignment detail / viewer (`_alignment_detail_panel`) and resumes VTK pump.
- **All other tabs** (today: Extract, Mask, Review, Coverage) → **shared `_preview_panel`** when preview is visible.

Adjust will use the **preview column** like Extract/Mask/Review unless you deliberately give it a bespoke panel (not recommended).

### 3.3 Preview panel coupling

The preview stack includes:

- `_preview_mode` — `"process"` (Mask), `"review"`, or `active.lower()` for others.
- `_load_image_list` / navigator — tightly tied to **Mask** input (`input_entry`, patterns).
- **Mask-only UX** on the preview image: SAM click bindings on `_process_overlay_label`, “Mask” checkbox, GPU cache indicator tied to segmentation pipeline.

**Risk:** Reusing `_preview_mode == "process"` for Adjust would inherit Mask behaviours incorrectly.

**Recommendation:** Introduce `_preview_mode == "adjust"` (string contract) and branch:

- `_on_tab_change`: when `active == "Adjust"`, set mode to `"adjust"` and call `adjust_tab`-owned loaders (e.g. `_adjust_load_media_list()`).
- Preview update entry points: gate SAM click handlers so they **no-op** unless Mask + ready state (existing pattern partially exists; extend explicitly).
- Optionally hide or repurpose the **Mask** checkbox in the preview header when mode is `adjust` (or leave visible but disabled — cleaner to hide).

### 3.4 Infrastructure primitives (reuse)

From `app_infra.py` / existing tabs:

- **Logging:** `app.log(...)`, `log_queue`, crash-resilient file logger.
- **Long jobs:** `cancel_flag`, `is_running`, `_start_operation` / `_stop_operation` with progress bar (when wired).
- **Browse:** `_browse_dir_into`, `_browse_file_for`, `_browse_video_for`, `_entry_initialdir`.

### 3.5 Threading model expectation

The GUI polls logs and expects **heavy work off the main thread**. Preview refreshes should follow the same **debounced CTkImage** pattern Mask uses (see `_render_preview` / debounce comments in `reconstruction_zone.py`). Slow ops (NL-means, AI denoise) must **not** block Tk.

---

## 4. Integration specification (Adjust ↔ host app)

### 4.1 Required host-app touchpoints

| Location | Change |
|----------|--------|
| `reconstruction_zone.py` | Register tab; call builder; extend `_on_tab_change` for Adjust preview mode and any Coverage-style refresh hooks. |
| Preview subsystem | Mode `"adjust"`; optional suppression of mask-only controls; dedicated nav/list state attributes (`_adjust_image_list`, `_adjust_nav_idx`, etc.) **or** namespaced struct owned by adjust tab module. |
| `activity_store.py` (optional but desirable) | Log “Adjust batch export”, “Adjust preset applied”, etc., for **Projects → Recent Activity** parity. |
| `docs/ARCHITECTURE.md` | Update tab count / module map when Adjust lands (current summary lines under-count or omit **Align** in places — fix opportunistically). |

### 4.2 Data flow (recommended)

```
User picks folder/video/raw
        → adjust_tab stores paths + current index
        → adjust_engine loads/develops source → numpy/RGB float or uint8 buffer
        → downsample → apply_adjustments(preview_buffer, state) @ slider events (debounced)
        → PIL → CTkImage → shared preview label
Export
        → worker thread / pool
        → full-res pipeline + writers (JPEG/PNG/TIFF/…)
        → optional activity_store + log lines
```

### 4.3 Cross-tab handoff (from initial plan)

Implement **explicit** UX hooks rather than implicit globals:

- **Adjust → Extract:** Set Extract’s video entry to Adjust output path (if video export), or set output folder for “Open in Extract” navigation.
- **Extract → Adjust:** Mirror pattern: write Adjust input entry and switch tab.
- **Adjust → Mask:** Set `app.input_entry` (Mask tab’s input folder) to Adjust output folder.

**Canonical widget attributes on `app` (verified from source):**

| Tab | Widget | Attribute | Purpose |
|-----|--------|-----------|---------|
| Extract | Video input | `app.analyze_video_entry` | Path to video file for analysis/extraction |
| Extract | Frames output | `app.extract_output_entry` | Output directory for extracted frames |
| Extract | Split front video | `app.split_front_video_entry` | Dual-fisheye front lens |
| Extract | Split back video | `app.split_back_video_entry` | Dual-fisheye back lens |
| Extract | Split output | `app.split_output_entry` | Split extraction output dir |
| Extract | Current video path | `app.current_video_path` | Last analyzed video (state, not widget) |
| Mask | Input folder | `app.input_entry` | Images to mask |
| Mask | Output folder | `app.output_entry` | Mask output directory |

**Tab switching:** `app.tabs.set(“Adjust”)` — confirmed pattern from `alignment_tab.py:1265` (`app.tabs.set(“Align”)`) and `alignment_tab.py:2573` (`app.tabs.set(“Coverage”)`).

---

## 5. Dependencies and packaging

### 5.1 Root `requirements.txt` today (verified)

```
numpy, opencv-python, Pillow, tqdm, pyyaml, customtkinter>=5.0,
py360convert, scipy, rfdetr, supervision, transformers>=4.50,<5.0, huggingface_hub
```

Separate file: `requirements-yolo.txt` (`ultralytics` — AGPL, excluded from Gumroad builds).

**Not present:** `opencv-contrib-python`, `rawpy`, `scikit-image`, `colour-science`, `nafnetlib`, `spandrel`.

The initial plan’s uninstall/install sequence for **opencv-contrib-python** is correct for `cv2.xphoto` / `cv2.mcc`. When switching, `requirements.txt` must change `opencv-python` → `opencv-contrib-python` and add a comment warning against reinstalling the non-contrib package.

### 5.2 Proposed requirement tiers

| Tier | Packages | Notes |
|------|----------|------|
| **Core Adjust (still images, no RAW)** | `opencv-contrib-python` (pinned), `numpy`, `scipy`, `Pillow` | Replace plain opencv in root requirements; verify no transitive reintroduction of `opencv-python`. |
| **RAW** | `rawpy` (pinned) | Native wheel/LibRaw — test **PyInstaller** bundle on Windows. |
| **Colour / charts** | `colour-science` (pinned) | Add to requirements when CCM UI ships; verify API imports against pinned version. |
| **scikit-image features** | `scikit-image` | Histogram/wavelet paths per plan. |
| **Optional chart detector** | `colour-checker-detection` | Large / optional extra. |
| **AI denoise** | `nafnetlib`, `spandrel`, torch | Strongly consider **`requirements-adjust.txt`** or extras so CPU-only users are not forced to install restoration stacks. |

### 5.3 PyInstaller / Gumroad

- **Frozen builds:** Every native extension (`cv2`, `rawpy`, potentially torch DLLs) needs verification in `reconstruction_zone.spec`.
- **Gumroad distribution:** `prep360/distribution.py` — `is_gumroad()` excludes AGPL **ultralytics** path per `requirements-yolo.txt` comments.
- **Content-aware features** that assume **YOLO** must **fall back** (e.g. RF-DETR-only) or **disable** with a clear UI label on Gumroad builds.

---

## 6. Concurrency design (explicit choice)

Decide per workload:

| Workload | Suggested approach | Rationale |
|----------|-------------------|-----------|
| Preview refresh | Main thread schedules debounced updates; worker returns **small** result (PIL or ndarray) | Tk thread safety. |
| CPU batch (OpenCV heavy, no GIL release) | `ProcessPoolExecutor` **or** chunked threads | Matches prep360 for CPU-bound parallelism; watch Windows spawn cost. |
| GPU (AI denoise) | Single-threaded inference queue or limited workers | Avoid VRAM OOM; one model at a time unless engineered otherwise. |
| RAW develop + OpenCV chain | Often mixed — profile before promising `num_workers` scaling | |

**Do not** cite “masking uses ProcessPoolExecutor” — mask batch in `reconstruction_pipeline.py` is **thread-based** today.

---

## 7. Phased delivery (recommended)

Phasing limits integration risk and keeps reviewable PRs.

### Phase A — Skeleton & wiring

- New tab registered; scrollable left UI shell (sections collapsed).
- `_preview_mode == "adjust"`; preview shows static placeholder then single-file load.
- `adjust_engine.py` with `AdjustmentState` + **one** proven op (e.g. exposure EV) + pytest.

### Phase B — Core tone/colour (CPU)

- Remaining fast ops (WB, saturation, curves via LUT, CLAHE with preview rules).
- Downsampled preview + full-res export pipeline; progress + cancel.

### Phase C — RAW

- `rawpy` integration behind “RAW development” collapsible; gate imports with clear error if missing.

### Phase D — Charts / colour science

- `colour-science` integration; `cv2.mcc` detector + manual fallback UI.
- Optional package for `colour-checker-detection`.

### Phase E — Heavy denoise & AI

- NL-means rules as per plan (release-only / export-only at full res).
- NAFNet / spandrel behind optional extra; model download UX pattern like `model_downloader.py`.

### Phase F — Video & temporal smoothing

- ffmpeg decode/encode alignment with `prep360` patterns; frame scrub preview.
- Temporal smoothing for exposure/WB — separate milestone (complex).

### Phase G — Content-aware & distributed

- Sky/subject normalization: **respect Gumroad** constraints.
- Distributed workers: separate RFC; stateless `(path, serialized state) → path` contract.

---

## 8. Testing and quality gates

| Layer | Tests |
|-------|--------|
| `adjust_engine.py` | Pure functions: numeric stability, clipping, order-independence where required; golden arrays for small fixtures. |
| RAW | Smoke test on **one** RAF + one DNG sample (CI optional / local only if files large). |
| GUI | Minimal manual checklist; optional `pytest` + headless smoke if project adds harness later. |
| Packaging | One-click frozen build launches Adjust; no silent missing DLL for rawpy/opencv-contrib. |

---

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Preview mode collision with Mask | Dedicated `"adjust"` mode; guard SAM bindings. |
| OpenCV duplicate wheels | Document uninstall `opencv-python`; add CI or smoke import checking `cv2.xphoto`. |
| `colour-science` API drift | Pin version; unit-test import paths used in chart CCM snippet. |
| AI deps bloat / support burden | Optional extras; clear UI when torch unavailable. |
| Gumroad vs AGPL detectors | Feature detect `is_gumroad()` + model availability; RF-DETR fallback. |
| Undo stack vs RAW params | Separate **RAW develop state** from **adjustment state** or nest them in one serializable document for presets/undo. |

---

## 10. Resolved decisions

1. **Preset file location:** User-global (`~/.reconstruction_zone/presets/adjust/`). Presets are not project-specific — they represent camera/lighting configurations reusable across captures. A "Copy to project" export action can be added later for sharing.

2. **Export colour space:** Default locked to **sRGB** for photogrammetry exports (COLMAP and Metashape both assume sRGB input). User-selectable for RAW development only (rawpy supports ACES, ProPhoto, Adobe RGB, Rec.2020 — valid for intermediate archival export, not for reconstruction input).

3. **Process vs thread default:** `ProcessPoolExecutor` for pure-CPU adjustment batch (no GIL contention, matches prep360 pattern in `reframer.py:545`, `blur_filter.py:126`). `ThreadPoolExecutor` when AI denoise is in the pipeline (avoids CUDA context forking, which crashes on Windows). Auto-select based on whether the adjustment state includes GPU ops.

4. **Video decode/encode:** **ffmpeg CLI pipe** — not `cv2.VideoCapture`. Rationale:
   - prep360 already uses ffmpeg subprocess exclusively (`prep360/core/extractor.py`)
   - ffmpeg preserves colour metadata (matrix coefficients, transfer characteristics) that VideoCapture discards
   - ffmpeg supports hardware-accelerated decode (`-hwaccel cuda`)
   - Seeking accuracy is reliable with `-ss` before `-i` (input seeking)
   - VideoCapture has known issues with frame-accurate seeking and returns BGR with no colour space awareness

---

## 11. Checklist before marking “Adjust tab complete”

- [ ] Tab appears in correct order; persists layout across DPI/resizes.
- [ ] Preview works in Adjust without activating Mask SAM / mask overlay semantics.
- [ ] Cancel + progress behave for long exports.
- [ ] `requirements.txt` (and optional extras files) updated; no conflicting OpenCV packages.
- [ ] Gumroad build path tested **or** AI/YOLO-gated features gracefully disabled.
- [ ] Documentation: `ARCHITECTURE.md` module map + user-facing note in `README.md` / tab doc if applicable.
- [ ] Undo/preset JSON round-trip tested.

---

*Generated as an engineering companion to `adjust-tab-plan.md`. Update both documents as scope or the codebase evolves.*
