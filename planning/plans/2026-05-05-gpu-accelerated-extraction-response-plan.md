# GPU-Accelerated Extraction - Response & Implementation Plan

**Companion to:** [`2026-05-05-gpu-accelerated-extraction.md`](./2026-05-05-gpu-accelerated-extraction.md)

This document reviews the GPU extraction plan against the current Reconstruction Zone codebase and turns it into an implementation-ready engineering plan. It does **not** replace the original plan. Use the original for benchmark context and the desired target architecture; use this document for repository-grounded verification, corrections, risks, sequencing, and acceptance criteria.

When the two documents conflict on codebase facts or implementation order, prefer this response plan until the original is reconciled.

---

## 1. Executive verdict

Second-pass status: the updated source plan resolves the main issues found in the first review. It now explicitly chooses 480px BGR download plus existing CPU scene detection for v1, preserves scene-boundary ordering, fixes the sharpest manifest timing bug, treats CUDA OpenCV as optional, adds paired result metadata, and separates cancellation from GPU failure in the single-video path.

The plan is now close to implementation-ready. The current single-video sharpest-frame path is a genuine two-pass implementation, and the CUDA/NVDEC benchmark evidence supports a high-value acceleration effort for 3840x3840 HEVC fisheye clips. The highest-confidence path remains:

1. Convert `SharpestExtractor` CPU extraction to a streaming single-pass implementation.
2. Add an optional CUDA/NVDEC streaming implementation for `SharpestExtractor`.
3. Add GPU-assisted paired scoring for `PairedSplitVideoExtractor`.
4. Surface acceleration status in the Extract tab and logs.
5. Add focused CPU/fallback tests and CUDA manual validation.

The remaining corrections are narrower:

| Topic | Response |
|-------|----------|
| CUDA detection | Check actual callable functions, not only namespaces. In this local environment `hasattr(cv2, "cudacodec")` is true, but `cv2.cudacodec.createVideoReader` is absent and `cv2.cuda.createSobelFilter` is absent. |
| Benchmark traceability | New plan claims about `firstFrameIdx` pixel diffs and 480px/1920px scene-detection timings are not present in the checked-in benchmark artifacts. Add them to `benchmarks/BENCHMARK_RESULTS.md` or mark them as externally measured. |
| Paired result propagation | The plan says to add `PairedSplitResult.gpu_accelerated`, but `_select_sharpest_frame_indices()` currently returns only three values. Specify how the GPU-used flag returns to `extract()`. |
| Paired cancellation | "Return `None` from the selection method" is ambiguous because `None` is also commonly used for fallback/sentinel flow. Prefer raising/returning an explicit cancellation outcome so paired cancel cannot accidentally trigger CPU fallback or a confusing unpack error. |
| Scene detection wording | 480px histogram scene detection is a good v1 target, but "safe" is too strong. Require a parity check against CPU scene split counts/positions on representative footage. |
| Packaging wording | Standard pip OpenCV packages and CUDA wheels should not be mixed; local metadata shows multiple OpenCV distributions can coexist in a messy environment. Runtime checks must use function availability and build behavior, not package names alone. |

---

## 2. Current codebase facts

### 2.1 Single-video sharpest extraction

File: `prep360/core/sharpest_extractor.py`

Current structure:

| Component | Current behavior |
|-----------|------------------|
| `SharpestConfig` | Interval, quality, output format, start/end seconds, scoring method, scene detection, analysis width. No GPU setting. |
| `SharpestResult` | Success flag, analyzed count, extracted count, output dir, frame paths, error. No acceleration metadata. |
| `SharpestExtractor.extract()` | Always calls `_extract_opencv()`. It computes `scene_aware = config.scene_detection` and `use_ffmpeg_extract = config.scene_detection`. |
| `_extract_opencv()` | Opens `cv2.VideoCapture`, scores every frame in range, stores `(frame_idx, sharpness, is_scene)` tuples, selects winners after scoring, then runs pass 2. |
| Scene-aware pass 2 | Uses `_extract_frames()` with an ffmpeg `select` expression when scene detection is on. This is exactly the stalling path identified by benchmarks. |
| Non-scene pass 2 | Uses `_extract_opencv_pass2()` to seek each winner and write via OpenCV. |
| `_split_at_scenes()` | Boundary entry belongs to the new sub-chunk: when `entry[2]` is true, current chunk flushes before appending the boundary frame. |
| Manifest | `_write_manifest()` receives absolute frame numbers, then calculates `time_sec = start_sec + frame_num / fps`. With nonzero `start_sec`, that appears to double-count the start offset. |

The plan correctly identifies `_extract_opencv()` as the first refactor target and `_extract_frames()` as the high-risk path to remove.

### 2.2 Paired split-video extraction

File: `prep360/core/paired_split_video_extractor.py`

Current structure:

| Component | Current behavior |
|-----------|------------------|
| `PairedSplitConfig` | Mode, scoring method, scene detection, interval, quality, format, start/end, scene threshold, scale width. |
| `PairedSplitResult` | Pair count, output paths, selected times, source frame numbers, error. No acceleration metadata. |
| `_select_sharpest_frame_indices()` | Opens both videos with `cv2.VideoCapture`, seeks to `start_frame`, reads both streams in lockstep, scores both frames, optionally scene-detects both streams, stores paired entries, then selects winners. |
| `_select_from_paired_entries()` | Selects by window and scene sub-chunk using `_pair_sharpness_sort_key()`: `min(front, back), avg, -imbalance`. |
| `_extract_selected_pairs()` | Second pass reads both videos sequentially from `start_frame` to `end_frame`, writing only selected frame indices. |
| Output safety | `extract()` refuses to overwrite non-empty `front/frames`, `back/frames`, or an existing paired manifest. |

The plan's v1 paired approach, GPU scoring/decode for the selection pass plus existing CPU write pass, fits this structure well. The updated plan now frames this as a scope decision, which is the right rationale: paired selection can be streamed, but a two-pass v1 has less output and cancellation risk.

### 2.3 GUI process

File: `reconstruction_gui/tabs/source_tab.py`

Relevant paths:

| GUI path | Current behavior |
|----------|------------------|
| Single Extract | `_extract_single_worker()` builds `SharpestConfig`, calls `SharpestExtractor.extract()`, converts success to `ExtractionResult`, then runs post-processing and geotagging. |
| Queue Extract | `_extract_queue_worker()` repeats the same sharpest flow for queued items using per-item `ExtractionSettings`. |
| Split Lens Extract | `_paired_split_video_worker()` builds `PairedSplitConfig`, calls `PairedSplitVideoExtractor.extract()`, then post-processes `front/frames` and `back/frames`. |
| Settings UI | `bottom_right` currently contains Scene Detection and Auto-geotag controls. A compact passive acceleration label can fit there, but space is already tight. |
| Progress throttling | Single-video progress only throttles messages containing `"Analyzing:"`; current sharpest messages are `"Scoring:"`, so they are not throttled. The plan is correct that no current sharpest messages match that throttle, but it should not assume logging is already quiet. |

The GUI needs logs in three places: single extract, queue extract, and paired split extract.

---

## 3. Plan validation

### 3.1 What aligns well

- The benchmark-supported target is worth implementing: GPU Laplacian around 198.6s vs CPU single-pass 1219.4s on the documented 3840x3840 HEVC test is a large practical win.
- `SharpestExtractor` has a stable public API, so automatic GPU routing can be internal.
- CPU single-pass should be the first implementation step because it removes the ffmpeg select bottleneck and simplifies GPU fallback.
- `cv2.cudacodec.VideoReader` + `cv2.cuda` scoring is the right GPU stack for the measured workload.
- Keeping scene detection behavior anchored to the existing `_detect_scene_change()` for v1 is a sensible validation choice.
- Split-lens paired extraction is already isolated enough to add an acceleration path without touching the rest of the GUI.
- A passive GUI status indicator is better than exposing a toggle before fallback behavior is proven.

### 3.2 Corrections and risks

| Severity | Area | Issue | Required response |
|----------|------|-------|-------------------|
| High | CUDA availability detection | The updated plan checks `hasattr(cv2, "cudacodec")` and device count, but namespaces can exist without callable CUDA implementations. Local verification: `cv2.cudacodec` exists, `createVideoReader` does not; `cv2.cuda` exists, `createSobelFilter` does not. | Check every required callable: `cv2.cudacodec.createVideoReader`, `cv2.cudacodec.VideoReaderInitParams` when using `firstFrameIdx`, `cv2.cuda.createSobelFilter`, `cv2.cuda.createLaplacianFilter`, `cv2.cuda.resize`, `cv2.cuda.cvtColor`, and `cv2.cuda.sum`. Keep `getCudaEnabledDeviceCount() > 0` as only one part of readiness. |
| High | Paired cancellation semantics | The paired section says cancellation during GPU scoring should return `None` from selection and be handled by the caller. Current `extract()` unpacks three values from `_select_sharpest_frame_indices()` and catches exceptions; it does not have an explicit `None` contract. | Use an explicit cancellation path: raise `RuntimeError("Cancelled")` as the CPU path already does, or introduce a structured selection result with `cancelled`, `gpu_accelerated`, and selected values. Do not use `None` for both cancel and fallback-adjacent states. |
| Medium | Paired GPU metadata propagation | `PairedSplitResult.gpu_accelerated` is listed, but the plan does not specify how `_select_sharpest_frame_indices()` communicates GPU use back to `extract()`. | Change selection to return `(indices, front_scores, back_scores, gpu_used)` or a small dataclass/internal tuple. Set `PairedSplitResult(gpu_accelerated=gpu_used)` on success. |
| Medium | Benchmark traceability | `firstFrameIdx` vs `set()` pixel-diff numbers and scene-detection cost numbers are in the updated plan but not found in `benchmarks/BENCHMARK_RESULTS.md` or `benchmarks/extraction_benchmark.py`. | Add a short benchmark appendix or script output section for these numbers before implementation work depends on them. |
| Medium | Scene detection confidence | The updated plan argues 480px histogram comparison is resolution-invariant and safe for v1. It is directionally plausible, but not a mathematical guarantee for all footage. | Add manual/automated parity checks: compare scene split counts and split frame positions between CPU `scale_width` detection and GPU 480px detection on representative clips. |
| Medium | `createVideoReader` call shape | The plan uses `cv2.cudacodec.createVideoReader(video_path, params=params)`. Python OpenCV bindings are often positional and vary by wheel. | If not already verified, use positional `cv2.cudacodec.createVideoReader(video_path, params)` or add this exact keyword-call verification to the API facts. |
| Low | GUI status label | The proposed `_HAS_CUDACODEC` helper has the same namespace-only weakness as `_gpu_available()`. | Make the GUI status use the same robust helper as the extractor or call `SharpestExtractor()._gpu_available()` after hardening it. |
| Low | Top-level instruction | The source plan references `superpowers:subagent-driven-development`; that skill is not part of this repo's normal implementation contract. | Treat it as agent-workflow metadata, not a code requirement. |

---

## 4. Implementation plan

### Phase 0 - Preflight and test harness

Before code changes:

- Preserve the current public APIs for `SharpestExtractor.extract()` and `PairedSplitVideoExtractor.extract()`.
- Add or prepare small synthetic-video tests for CPU behavior. CUDA should be manually tested; CI can mock/skip it.
- Decide the output document/log wording:
  - GPU full path: `GPU (NVDEC + CUDA scoring)`
  - CPU path: `CPU`
  - GPU unavailable fallback: log the specific reason when available.

Recommended tests:

| Test | Purpose |
|------|---------|
| CPU sharpest with scene detection off | Confirms single-pass window selection and frame count. |
| CPU sharpest with synthetic scene cut | Confirms boundary frame starts a new sub-chunk. |
| CPU sharpest with `start_sec` / `end_sec` | Confirms range, manifest timestamps, and no off-by-start errors. |
| CPU sharpest cancellation | Confirms cancellation returns `Cancelled` and does not write manifest as success. |
| GPU unavailable fallback | Monkeypatch `_gpu_available()` false or `createVideoReader` failure; confirms CPU fallback and `gpu_accelerated=False`. |

### Phase 1 - CPU single-pass `SharpestExtractor`

Modify `prep360/core/sharpest_extractor.py`.

Tasks:

- Remove `use_ffmpeg_extract` routing from `extract()` and `_extract_opencv()`.
- Rewrite `_extract_opencv()` as a streaming scorer/writer.
- Keep `_laplacian_sharpness()`, `_tenengrad_sharpness()`, `_detect_scene_change()`, and `_write_manifest()`.
- Remove `_extract_frames()` once no callers remain.
- Remove `_extract_opencv_pass2()` once no callers remain.
- Remove `_split_at_scenes()` only after streaming scene logic has explicit tests.

Streaming algorithm requirements:

1. Open `cv2.VideoCapture`, read fps/frame count, compute `start_frame`, `end_frame`, `window_size`, and `frames_in_range`.
2. Use relative frame index for windows: `relative_idx = frame_idx - start_frame`.
3. For each decoded frame:
   - Downscale for scoring.
   - Compute scene boundary against previous small BGR if enabled.
   - Compute sharpness.
   - If scene boundary is true, flush the existing sub-chunk winner **before** considering the current frame as a winner.
   - Consider current frame for the active sub-chunk winner.
   - If the window closes, flush the active sub-chunk winner and reset the window state.
4. Flush the final partial sub-chunk at EOF.
5. Write the manifest using the selected absolute source frame numbers.

Implementation detail:

```text
Boundary frame handling must match _split_at_scenes():
previous chunk = frames before the boundary
new chunk      = boundary frame and following frames
```

Acceptance:

- Scene detection on/off both work.
- `start_sec` / `end_sec` selects only the intended range.
- Manifest `time_sec` maps to the true source video time.
- No ffmpeg select path remains in sharpest extraction.

### Phase 2 - GPU `SharpestExtractor`

Modify `prep360/core/sharpest_extractor.py`.

Tasks:

- Add `gpu_accelerated: bool = False` to `SharpestResult`.
- Add `_gpu_available()` with class-level caching, but make it function-level, not namespace-level. Require actual callables such as `cv2.cudacodec.createVideoReader` and the CUDA filter/ops used by scoring. Do not treat availability as the final authority; `createVideoReader` can still fail by codec/resolution/DLL.
- Add `_extract_gpu_single_pass()` returning `Optional[SharpestResult]`:
  - `SharpestResult` for success or cancellation.
  - `None` for CUDA init/runtime failure where CPU fallback should run.
- Route `extract()`:
  - Try GPU only if available.
  - On `None`, log fallback and run CPU single-pass.
  - On cancellation result, return immediately.
  - On success, return immediately.

GPU loop requirements:

- Use `cv2.cudacodec.createVideoReader()`.
- Seek to `start_frame` before processing; prefer `VideoReaderInitParams.firstFrameIdx` if stable in the local wheel, with `reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)` fallback.
- Derive fps/frame count robustly. If `reader.get()` returns `(ok, value)`, unwrap it; if not, fall back to `cv2.VideoCapture` metadata.
- Detect `GpuMat` format from the first processed frame and choose conversion codes.
- Convert `uint16` frames to `uint8` with GPU `convertTo(..., alpha=1/256)` for scoring and downloaded winners.
- Pre-create Sobel or Laplacian filters once per extraction.
- Keep only the current best full-resolution `GpuMat.clone()` for the active sub-chunk.
- Download full-resolution frames only when flushing winners.
- Track written GPU output paths for cleanup on CUDA error.

Scene detection for v1:

- Use the existing CPU `_detect_scene_change()`.
- GPU-resize a small BGR frame, convert to `uint8`, download it, and pass it to `_detect_scene_change()`.
- Choose one of these explicitly:
  - **Parity-first:** use `config.scale_width` for scene frames, matching current CPU behavior more closely.
  - **Speed-first:** use 480px, as benchmarked in the plan, and validate behavior manually.

Recommended v1 choice: use 480px only if manual tests show scene split counts and selected frames are acceptable on representative clips. Otherwise start parity-first and optimize later.

Acceptance:

- HEVC fisheye clips use GPU and write `gpu_accelerated=True`.
- H.264 ERP clips that exceed NVDEC limits fall back to CPU and write `gpu_accelerated=False`.
- Cancellation on GPU stops; it does not fall back to CPU.
- CUDA runtime errors clean up GPU-written partial files before CPU fallback.

### Phase 3 - GPU-assisted paired extraction

Modify `prep360/core/paired_split_video_extractor.py`.

Tasks:

- Add `gpu_accelerated: bool = False` to `PairedSplitResult`.
- Add a GPU selection path inside `_select_sharpest_frame_indices()` or split it into `_select_sharpest_frame_indices_gpu()` plus `_select_sharpest_frame_indices_cpu()`. Return an explicit GPU-used flag along with the selected indices/scores.
- Reuse the same GPU scoring preparation as single-video extraction where practical.
- If either front or back reader fails to initialize, log and fall back to the existing CPU selection path.
- If a CUDA error occurs during selection, restart selection with CPU. No paired output should have been written yet, so cleanup is simpler than single-video.
- Preserve existing `_extract_selected_pairs()` for v1.

Important paired details:

- Both streams must seek to the same `start_frame`.
- Both readers must advance in lockstep.
- Scene detection should still use both streams: `front_scene or back_scene`.
- Progress should keep the existing 0-80 analysis / 85-100 extraction feel.
- The returned result should indicate GPU use if the selection/scoring pass used CUDA, even though the write pass remains CPU.
- Cancellation should stay explicit. Prefer raising the same `RuntimeError("Cancelled")` used by the CPU path, or return a structured selection result with a `cancelled` flag; avoid a bare `None` that can be mistaken for fallback flow.

Acceptance:

- Paired HEVC fisheye clips use GPU scoring/decode for selection.
- Paired fallback works if either reader fails.
- Scene-aware pair selection remains based on the same pair sort key.
- GUI log can distinguish GPU-assisted paired extraction from CPU paired extraction.

### Phase 4 - GUI integration

Modify `reconstruction_gui/tabs/source_tab.py`.

Tasks:

- Add a lazy CUDA OpenCV status helper near the prep360 import block. Keep it robust:
  - no exception if `cv2` is absent or non-CUDA;
  - no import-time hard failure;
  - no assumption that `opencv-python` includes CUDA.
- Use the same hardened checks as the extractor. Namespace presence is not sufficient because some OpenCV builds expose `cv2.cuda` / `cv2.cudacodec` namespaces while omitting the actual functions needed by this plan.
- Add a compact indicator in the Extract settings row.
- Log acceleration after successful single-video sharpest extraction in:
  - `_extract_single_worker()`
  - `_extract_queue_worker()`
- Log acceleration after paired extraction in:
  - `_paired_split_video_worker()`

UI wording:

| State | Suggested label |
|-------|-----------------|
| CUDA OpenCV present | `GPU ready` |
| OpenCV present without CUDA/cudacodec | `CPU only` |
| Unknown/error | `CPU only` |

Keep it passive. Do not add a toggle until the fallback path has been used in real projects.

Acceptance:

- GUI starts without CUDA OpenCV.
- GUI status accurately reflects the current Python/OpenCV runtime.
- Logs show the path used per extraction.
- Existing fixed-interval extraction remains unchanged.

### Phase 5 - Documentation and distribution

The updated plan correctly treats CUDA OpenCV as optional local setup and keeps PyInstaller CUDA packaging out of v1. Add a small documentation pass before release:

- Update `reconstruction_gui/docs/EXTRACT_TAB.md` with GPU status and fallback behavior.
- Add a setup note for CUDA OpenCV:
  - standard `opencv-python` is CPU-only;
  - CUDA OpenCV wheel/source build is optional;
  - incompatible OpenCV wheels must not be installed together.
- Update `README.md` or a GPU setup doc only if this is intended for user-facing release.
- Review `reconstruction_zone.spec`:
  - hidden import `cv2` exists;
  - CUDA/OpenCV DLL bundling is not solved by the current spec;
  - Gumroad/release packaging may need a CPU-only default with GPU as advanced local setup.

---

## 5. Acceptance matrix

| Scenario | Expected result |
|----------|-----------------|
| Single video, CPU-only environment | Sharpest extraction succeeds via CPU single-pass; no ffmpeg select path. |
| Single video, HEVC fisheye, CUDA OpenCV | GPU path runs; `gpu_accelerated=True`; log says GPU. |
| Single video, H.264 7680x3840 ERP, CUDA OpenCV | GPU reader fails or is skipped; CPU fallback succeeds; log says CPU fallback. |
| Single video, scene detection on | Scene splits produce representatives on both sides of cuts. Boundary frame belongs to the new segment. |
| Single video, start/end range | Output frames and manifest timestamps refer to the actual source-video times. |
| Single video, cancel | Stops cleanly, no CPU fallback after cancellation. |
| Queue sharpest extraction | Each item logs GPU or CPU path independently. |
| Paired split HEVC fisheye, CUDA OpenCV | GPU-assisted selection runs for both streams; paired output and manifest remain valid. |
| Paired fallback | If either stream cannot use NVDEC, CPU paired selection runs. |
| GUI startup without CUDA OpenCV | No crash; indicator says CPU only. |

---

## 6. Suggested delivery sequence

1. **CPU single-pass sharpest extraction**
   - Biggest reliability win.
   - Removes the known ffmpeg select stall.
   - Creates the streaming boundary logic needed by GPU.

2. **SharpestExtractor GPU path**
   - Delivers the main benchmarked speedup.
   - Keeps public API stable.
   - Exercises fallback and GUI logging on the simpler single-video path.

3. **Paired GPU-assisted selection**
   - Reuses lessons from single-video GPU implementation.
   - Keeps existing paired write pass for v1.

4. **GUI status and docs**
   - Wire result metadata through single, queue, and paired paths.
   - Add user-facing explanation of optional GPU support.

5. **Packaging/release hardening**
   - Decide whether GPU support is local advanced setup only or part of packaged releases.
   - Validate PyInstaller behavior on a CUDA OpenCV environment before advertising packaged GPU support.

---

## 7. Manual validation checklist

Use real footage for final validation:

- 3840x3840 HEVC fisheye single video, scene off.
- 3840x3840 HEVC fisheye single video, scene on.
- 7680x3840 H.264 ERP single video, scene on, confirming CPU fallback.
- Front/back 3840x3840 HEVC split pair, scene off.
- Front/back 3840x3840 HEVC split pair, scene on.
- Nonzero start/end range on both single and paired workflows.
- Cancellation during GPU single-video scoring.
- Cancellation during paired scoring.
- Queue with mixed GPU-eligible and CPU-fallback videos.

For each run, check:

- output count;
- output filenames and prefixes;
- manifest source frame/time values;
- GUI progress and log readability;
- no leftover partial files after fallback/cancel;
- post-processing and geotagging still run after successful extraction.

---

## 8. Future work after v1

- GPU histogram scene detection after validating 1D histogram behavior against the current 2D HSV correlation method.
- Paired full streaming extraction that buffers best front/back frames and avoids the CPU write pass.
- Optional fast scoring width mode, e.g. 960px, if winner quality remains acceptable.
- Explicit "force CPU" debug switch if users need to compare or work around CUDA driver/OpenCV issues.
- Shared CUDA extraction helper module if single and paired implementations duplicate too much conversion/scoring code.

---

## 9. Bottom line

Implement this plan, but tighten the boundaries:

- CPU single-pass first.
- GPU single-video second.
- Paired GPU-assisted scoring third.
- Metadata/logging for both result types.
- Treat CUDA OpenCV as optional and fragile until packaging is proven.
- Preserve scene split semantics exactly before optimizing scene detection further.

That gives the project the speed win without making extraction depend on a specialized OpenCV build or letting CUDA failures leak into normal CPU workflows.
