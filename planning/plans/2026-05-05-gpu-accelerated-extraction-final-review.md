# GPU-Accelerated Extraction - Final Review

**Reviewed plan:** [`2026-05-05-gpu-accelerated-extraction.md`](./2026-05-05-gpu-accelerated-extraction.md)

This review reflects the latest revision of the GPU-accelerated extraction plan. Earlier high-risk objections have been resolved. The plan is now logically coherent, testable without CUDA for most correctness risks, and safe to implement if the guardrails below are treated as acceptance criteria rather than optional advice.

---

## 1. Final Verdict

The plan is implementation-ready in architecture.

The core design is sound:

- Convert single-video CPU sharpest extraction from two-pass to streaming single-pass.
- Add an optional GPU path using `cv2.cudacodec` for NVDEC decode and `cv2.cuda` for scoring.
- Preserve CPU fallback when CUDA OpenCV, NVDEC, codecs, or runtime GPU operations fail.
- Keep user cancellation separate from GPU failure so cancellation never triggers CPU fallback.
- Use direct GPU-memory winner writing for paired sharpest extraction, avoiding GPU/CPU decoder frame-index drift.
- Keep `_split_at_scenes()` for `sharpness_benchmark.py` while production extraction moves to inline streaming logic.
- Add result metadata so the GUI can report CPU vs GPU execution.
- Add CPU-only automated tests for the risky semantics.

I would be comfortable handing this plan to an implementation agent, with the mandatory guardrails in section 3.

---

## 2. Resolved Prior Findings

### 2.1 Paired GPU frame mismatch risk - resolved

The plan no longer scores with GPU and writes later with CPU `VideoCapture`. Paired sharpest extraction is now specified as a GPU streaming single-pass path that clones the current best front/back `GpuMat`s and writes winners directly from GPU memory. This addresses the decoder alignment issue documented in the benchmark notes.

### 2.2 `_split_at_scenes()` removal risk - resolved

The plan explicitly keeps `_split_at_scenes()` because `prep360/core/sharpness_benchmark.py` still calls it. Production extraction can use inline streaming scene logic without breaking the benchmark utility.

### 2.3 CUDA readiness checks - resolved

The readiness probe now checks the required CUDA/cudacodec callables instead of checking namespaces only. This matters because standard OpenCV wheels can expose `cv2.cuda` and `cv2.cudacodec` namespaces without usable implementations. Local inspection confirms that exact failure mode: namespaces exist, but `createVideoReader`, `VideoReaderInitParams`, and CUDA scoring callables are not callable.

The GUI now reuses `SharpestExtractor()._gpu_available()` as the single source of truth, which avoids a misleading "GPU ready" label.

### 2.4 Scene detection language - resolved

The plan now states that 480px histogram scene detection is expected to be robust, pending parity validation. It no longer treats downsampling as inherently safe. The parity check against CPU 1920px scene detection is appropriately retained.

### 2.5 Cancellation cleanup - resolved in design

The plan now defines clean cancellation across CPU, GPU, paired GPU, and paired CPU fallback paths: track written paths, delete files from the cancelled attempt, and do not write a success manifest.

### 2.6 Pre-existing output safety - resolved with a recommended policy

The plan now recognizes the overwrite/delete hazard when cancelling inside a pre-existing single-video output directory. The recommended option is to refuse non-empty sharpest output directories, matching paired extraction's existing safety contract.

### 2.7 First-frame off-by-one - resolved in wording

The plan now explicitly says that the first `nextFrame()` result used for format detection must also be scored and processed as `start_frame`. The same warning is included for paired GPU extraction.

### 2.8 Benchmark traceability - resolved

The benchmark table no longer claims a measured `~1400s` CPU two-pass result. It now says the old path stalled / was not measured, which matches the benchmark document.

### 2.9 Code hygiene - resolved in plan

The plan now includes removing `os`, `subprocess`, `tempfile`, and `_SUBPROCESS_FLAGS` after `_extract_frames()` is removed, and updating the stale `sharpest_extractor.py` module docstring.

### 2.10 Packaging wording - resolved

The packaging note now correctly says multiple OpenCV wheel variants can coexist in pip metadata while only one `cv2` is imported. It recommends uninstalling all OpenCV variants before installing the CUDA wheel and relying on runtime capability checks.

---

## 3. Mandatory Implementation Guardrails

These are not architecture blockers, but they should be treated as required acceptance criteria.

### 3.1 Make non-empty output refusal mandatory for sharpest mode

The plan lists three mitigation options for pre-existing output files and recommends option 1. For safety, implementation should make option 1 mandatory for sharpest extraction in v1:

- If the target sharpest output directory exists and contains files, return a clear failure before writing anything.
- Do this before CPU/GPU routing starts.
- This prevents cancellation cleanup from deleting files that existed before the attempt.

Staging directories are a good future improvement, but refusal is the safest and simplest v1 contract.

### 3.2 Process the first decoded GPU frame exactly once

The plan correctly warns not to discard `first_gpu`. Implementation should make the control flow unambiguous:

- Open reader with `firstFrameIdx`.
- Call `nextFrame()` once.
- If it fails, return `None` for CPU fallback.
- Detect frame format from that frame.
- Process that same frame with `frame_idx = start_frame`.
- Continue the loop at `frame_idx = start_frame + 1`.

The paired GPU path should do the same for the first front/back pair. This prevents both first-frame loss and accidental double-processing.

### 3.3 Validate first-frame format before assuming conversion rules

Benchmark evidence shows `uint16 BGRA`, and the v1 target videos appear to follow that. Still, implementation should inspect the first frame's channel count and depth/type before conversion:

- Expected: 4-channel BGRA, usually 16-bit on the tested CUDA wheel/input.
- If a frame is 8-bit, use the 8-bit path rather than scaling by `1/256`.
- If channels/depth are unsupported, log clearly and fall back to CPU instead of producing bad images.

This keeps the GPU path safe across codec or wheel differences.

### 3.4 Preserve exact streaming window semantics

The plan now defines `relative_idx = frame_idx - start_frame` and closes a window after considering a frame when `(relative_idx + 1) % window_size == 0`. Implementation should also include a final EOF flush.

For scene boundaries, preserve the existing `_split_at_scenes()` behavior: the boundary frame belongs to the new sub-chunk. If a scene boundary appears on the last frame of a window, the previous sub-chunk and the one-frame boundary sub-chunk may both produce winners, matching the current collect-then-split behavior.

### 3.5 Keep cancellation and GPU failure paths separate

Single-video GPU cancellation should return `SharpestResult(success=False, error="Cancelled")`, not `None`. GPU init/runtime failure should return `None` so `extract()` can fall back to CPU.

Paired GPU cancellation should propagate as cancellation and must not be swallowed by broad GPU error handling. The plan's `RuntimeError` before `cv2.error` ordering is sound. Keep that ordering in code.

### 3.6 Reset `_gpu_ok` in tests that monkeypatch GPU availability

The class-level `_gpu_ok` cache is reasonable at runtime. Tests that monkeypatch CUDA availability or `_gpu_available()` must reset `SharpestExtractor._gpu_ok` between cases, or they can get order-dependent false positives.

---

## 4. Automated Test Expectations

The new CPU-only test table is a strong improvement. I recommend making these tests required before merge:

- CPU single-pass scene-off output count equals `ceil(frames / window_size)`.
- CPU single-pass scene-boundary ordering keeps the boundary frame in the new sub-chunk.
- CPU single-pass start/end range only processes the requested frame range.
- Manifest `time_sec` equals `source_frame / fps` for nonzero `start_sec`.
- Cancellation before/during single-video CPU extraction returns failure and writes no manifest.
- Cancellation cleanup leaves no frames from the cancelled attempt.
- Non-empty sharpest output directory is refused before writing.
- `_gpu_available()` returns false on standard/non-CUDA OpenCV namespace stubs.
- GPU unavailable path falls back to CPU and returns `gpu_accelerated=False`.
- GPU init failure returns `None` internally and falls back to CPU.
- `SharpestResult.gpu_accelerated` defaults to `False` and is `True` only on GPU success.
- `PairedSplitResult.gpu_accelerated` defaults to `False` and is `True` only on paired GPU success.
- `sharpness_benchmark.py` still works with `_split_at_scenes()` retained.
- Paired CPU cancellation cleanup removes partial front/back files and writes no paired manifest.

CUDA-specific correctness still requires manual validation on real footage, but the above tests cover the risky deterministic behavior without requiring a GPU.

---

## 5. Manual Validation Requirements

Manual validation remains necessary for the hardware path. The plan's matrix is appropriate. Before shipping, verify at least:

- CUDA OpenCV unavailable: GUI shows CPU only; extraction succeeds via CPU.
- HEVC fisheye on the CUDA workstation: GPU path runs and logs GPU.
- H.264 7680x3840 ERP: GPU reader fails cleanly and CPU fallback succeeds.
- GPU start/end range: first processed frame is not dropped; manifest times are correct.
- GPU cancellation: no CPU fallback, no partial frames, no manifest.
- 480px GPU scene detection: split count and split positions match CPU 1920px within the stated tolerance on representative videos.
- Paired HEVC fisheye: GPU streaming writes aligned front/back winners from GPU memory.
- Paired cancellation: no CPU fallback and no partial front/back pairs remain.
- Queue with mixed videos: each item logs CPU/GPU independently.

---

## 6. Residual Assumptions

These are acceptable v1 assumptions because the plan has fallback paths:

- CUDA API behavior is based on the tested cudawarped OpenCV wheel and RTX 3090 Ti. Other wheels/GPUs may vary.
- NVDEC codec limits are hardware-specific. The code should discover them by trying the reader and falling back.
- GPU and CPU paths may select different sharp frames because decoders/resizers differ slightly. That is acceptable as long as each path produces sharp, well-distributed frames and paired GPU writes the exact frames it scored.
- 480px scene detection is a performance tradeoff, not a proven invariant. The parity check decides whether it ships as-is.

---

## 7. Sign-Off Checklist

Before implementation is considered complete, confirm:

- [ ] Single-video CPU path is streaming single-pass; `_extract_opencv_pass2()` and `_extract_frames()` are removed.
- [ ] `_split_at_scenes()` remains and is covered by a test or benchmark check.
- [ ] Manifest timing uses absolute frame timestamps with no `start_sec` double-count.
- [ ] Single-video sharpest extraction refuses non-empty output directories or otherwise prevents deleting pre-existing files on cancel.
- [ ] CPU, GPU, paired GPU, and paired CPU fallback all track written paths and clean them on cancel.
- [ ] GPU init/runtime failure falls back to CPU; user cancellation never falls back.
- [ ] First decoded GPU frame/pair is processed exactly once.
- [ ] Unexpected GPU frame format falls back safely.
- [ ] GUI readiness uses the extractor readiness helper.
- [ ] GUI logs CPU/GPU status for single, queue, and paired paths.
- [ ] CPU-only automated tests pass.
- [ ] Manual CUDA validation passes on representative fisheye, ERP fallback, paired fisheye, cancellation, and scene-detection parity cases.

---

## 8. Current Confidence

High. No major logic, scope, or safety blockers remain in the plan.

The remaining work is implementation discipline: follow the window/scene ordering precisely, make output safety non-optional, keep cancellation distinct from GPU errors, and test the deterministic CPU/fallback behavior before relying on manual CUDA runs. With those guardrails, the plan is sound enough to execute.
