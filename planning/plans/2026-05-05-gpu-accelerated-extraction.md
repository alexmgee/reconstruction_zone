# GPU-Accelerated Frame Extraction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GPU-accelerated sharpest-frame extraction to SharpestExtractor and PairedSplitVideoExtractor using NVDEC decode + GPU scoring via cv2.cudacodec, with automatic fallback to CPU when unavailable.

**Architecture:** A new single-pass GPU extraction method is added to `SharpestExtractor` alongside an improved CPU single-pass fallback. The public API (`extract()`) is unchanged — GPU acceleration is selected automatically based on hardware availability. The GPU path uses `cv2.cudacodec.VideoReader` for NVDEC hardware decode, `cv2.cuda` operations for resize/grayscale/scoring, and only downloads full-resolution frames for winners. Scene detection downloads a small 480px BGR frame and uses the existing CPU `_detect_scene_change()` for v1. Falls back to CPU single-pass when cudacodec is unavailable or the video exceeds NVDEC limits (e.g., H.264 > 4096px wide). The `PairedSplitVideoExtractor` gets a GPU streaming single-pass path using two simultaneous NVDEC sessions — scoring and writing from GPU memory to avoid GPU/CPU frame index misalignment.

**Tech Stack:** OpenCV 4.12+ with CUDA (cv2.cudacodec, cv2.cuda), existing prep360 infrastructure. CUDA OpenCV is an optional runtime capability — the tool works fully without it.

**Benchmark evidence:** All approaches benchmarked on 3840x3840 HEVC fisheye video (16,929 frames). Results in `benchmarks/BENCHMARK_RESULTS.md`.

| Path | Time | Speedup |
|------|------|---------|
| CPU two-pass (current) | stalled / not measured | baseline (ffmpeg select filter hung) |
| CPU single-pass | 1219s | measured baseline |
| GPU single-pass + GPU Tenengrad | 217s | 5.6x vs CPU single-pass |
| GPU single-pass + GPU Laplacian | 199s | 6.1x vs CPU single-pass |

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `prep360/core/sharpest_extractor.py` | Modify | Add `_extract_gpu_single_pass()`, refactor `_extract_opencv()` to CPU single-pass, update `extract()` routing, fix manifest timing bug |
| `prep360/core/paired_split_video_extractor.py` | Modify | Add GPU scoring path using two cudacodec readers, add `gpu_accelerated` to result |
| `reconstruction_gui/tabs/source_tab.py` | Modify | Add GPU status indicator to extraction settings, log GPU usage in single/queue/paired paths |

New files are allowed for tests if needed. The production changes are internal to the extractors — the public APIs do not change. The GUI detects GPU availability at startup and shows status.

---

## Verified API facts

All of these were tested during benchmarking and are confirmed working:

| API call | Verified |
|----------|----------|
| `cv2.cudacodec.createVideoReader(path)` | Yes — returns GpuMat as **uint16 BGRA (4 channels)** |
| `reader.set(cv2.CAP_PROP_POS_FRAMES, N)` | Yes — seeking works, but less frame-accurate than `firstFrameIdx` |
| `reader.get(cv2.CAP_PROP_FPS)` | Yes — returns `(True, fps_float)` |
| `reader.get(cv2.CAP_PROP_FRAME_COUNT)` | Yes — returns `(True, count_float)` |
| `cv2.cudacodec.VideoReaderInitParams().firstFrameIdx` | Yes — skips to frame N on open. Closer to CPU ground truth than `set()`. **Prefer this for start_sec seeking.** |
| `gpu_frame.size()` | Returns `(width, height)` — **opposite** of numpy's `(h, w)` |
| `gpu_frame.clone()` | Yes — required to preserve frame across `nextFrame()` calls |
| `cv2.cuda.resize(gpu_mat, (w, h))` | Yes |
| `cv2.cuda.cvtColor(gpu_mat, COLOR_BGRA2GRAY)` | Yes |
| `cv2.cuda.cvtColor(gpu_mat, COLOR_BGRA2BGR)` | Yes |
| `cv2.cuda.cvtColor(gpu_mat, COLOR_BGR2HSV)` | Yes — requires uint8 input |
| `cv2.cuda.split(gpu_mat)` | Yes — returns list of single-channel GpuMats |
| `cv2.cuda.calcHist(gpu_mat_single_channel)` | Yes — returns 256-bin histogram as GpuMat |
| `gpu_mat.convertTo(dst_type, dst_gpu, alpha=)` | Yes — used for uint16→uint8 |
| `cv2.cuda.createSobelFilter(CV_8UC1, CV_32F, ...)` | Yes — different src/dst types allowed |
| `cv2.cuda.createLaplacianFilter(CV_32FC1, CV_32FC1, ...)` | Yes — src/dst must match |
| `cv2.cuda.createLaplacianFilter(CV_8UC1, CV_32F, ...)` | **No** — assertion error |
| `cv2.cuda.sum(gpu_mat)` | Yes — returns scalar tuple |
| Two simultaneous `cudacodec.VideoReader` instances | Yes — tested with front+back fisheye |
| `cv2.cudacodec.createVideoReader(path, params=params)` | Yes — **must use keyword `params=`**. Positional form fails with argument parsing error. |

### Seeking: `firstFrameIdx` vs `set()`

Both were tested against CPU `VideoCapture` ground truth at frame 500:
- `firstFrameIdx`: mean pixel diff 19.3 from CPU (closer)
- `set(POS_FRAMES)`: mean pixel diff 33.4 from CPU (farther)

Neither is pixel-identical to CPU (NVDEC decodes differently), but `firstFrameIdx` is more accurate. **Use `firstFrameIdx` when `start_sec` is nonzero.** When `start_sec` is 0, no seeking is needed.

Important: do not read a frame for format detection before seeking. Open the reader with `firstFrameIdx` set, then detect format from the first frame read at `start_frame`.

### NVDEC hardware decode limits (tested on RTX 3090 Ti / GA102)

These limits are hardware-generation-specific. Other NVIDIA GPUs may have different limits. The implementation does not hardcode these — it tries `createVideoReader` and falls back on failure.

| Codec | Max resolution (this GPU) | Your videos |
|-------|--------------------------|-------------|
| H.264 | 4096 × 4096 | ERP 7680×3840 **exceeds** → CPU fallback |
| HEVC | 8192 × 8192 | Fisheye 3840×3840 ✓ |
| AV1 | 8192 × 8192 | Not used |

### cudacodec frame format

`nextFrame()` returns GpuMat as **uint16 BGRA** (4 channels, 16-bit per channel).
- For scoring: must convert to uint8 grayscale (`convertTo` alpha=1/256, `cvtColor`)
- For winner output: must convert to uint8 BGR (`>> 8`, `cvtColor BGRA2BGR`)
- For scene detection: must convert to uint8 BGR then HSV

---

## Known bug to fix: manifest timing double-count

`_write_manifest()` computes `time_sec = start_sec + frame_num / fps`. But `frame_num` values are **absolute** frame indices (e.g., frame 500 for a video starting at 10s). So `frame_num / fps` already equals the true timestamp. Adding `start_sec` double-counts the offset.

**Example:** `start_sec=10.0`, `fps=50`, selected frame 500:
- Current: `time_sec = 10.0 + 500/50 = 20.0` — **wrong**
- Correct: `time_sec = 500/50 = 10.0`

**Fix:** Change to `time_sec = round(frame_num / fps, 3)`. Apply this in Task 1 alongside the single-pass refactor.

---

## Task 1: Convert CPU path from two-pass to single-pass

The existing `_extract_opencv()` decodes all frames to score them (Pass 1), then re-opens the video and seeks to winners (Pass 2). This is wasteful even without GPU. Convert it to single-pass: buffer the current window's best frame in memory, write when the window closes.

**Files:**
- Modify: `prep360/core/sharpest_extractor.py` — method `_extract_opencv()`

### Changes

The current `_extract_opencv()` has two phases:
1. Lines 232-282: Score every frame, collect `(frame_idx, sharpness, is_scene)` tuples
2. Lines 297-349: Select winners from score list, re-open video, extract via Pass 2

Refactor to a single pass:

**The buffer approach:**
- Keep `best_frame: np.ndarray` (full-res pixels) and `best_score: float` for the current sub-chunk
- On scene boundary: flush current sub-chunk's best, boundary frame starts the **new** sub-chunk
- On window boundary: flush current sub-chunk's best, start new window
- Memory cost: one full-res frame (~45 MB for 3840x3840 BGR)

**Window indexing — critical:** The current two-pass code uses relative slices: `scores[0:window_size]`, `scores[window_size:2*window_size]`, etc. In the streaming single-pass, use `relative_idx = frame_idx - start_frame`. A window closes **after considering a frame** when `(relative_idx + 1) % window_size == 0`. This means the frame at `relative_idx == window_size - 1` is the last frame of window 0, and `relative_idx == window_size` is the first frame of window 1.

**Scene boundary ordering — critical:** The existing `_split_at_scenes()` puts the boundary frame in the **new** sub-chunk (line 376-379: flush current, then append boundary frame). The single-pass loop must preserve this:

```
For each frame (frame_idx from start_frame to end_frame):
  relative_idx = frame_idx - start_frame

  1. Compute sharpness and scene boundary flag
  2. IF scene boundary AND sub-chunk has content:
     → Flush sub-chunk winner (write to disk)
     → Reset sub-chunk state
  3. Consider current frame as candidate for the (possibly new) sub-chunk
  4. IF (relative_idx + 1) % window_size == 0:
     → Flush sub-chunk winner (write to disk)
     → Reset window and sub-chunk state
```

Step 2 happens BEFORE step 3 — the boundary frame is scored into the new sub-chunk, not the old one.

**Final flush at EOF:** After the loop ends (read failure or `frame_idx >= end_frame`), flush the final partial sub-chunk if it has content. This handles videos whose frame count is not a multiple of `window_size`.

**Window + scene boundary coincidence:** If a scene boundary and window boundary coincide on the same frame: step 2 flushes the previous sub-chunk, step 3 makes the boundary frame the new sub-chunk's best, then step 4 flushes it as the window winner. The `if sub_chunk_best_frame is not None` guard in step 4 handles the case where the scene flush already emptied the sub-chunk. The window counter resets regardless — tracked by `relative_idx`, not flush state.

**What to keep:**
- `_write_manifest()` — fix the `start_sec` double-count bug, otherwise unchanged
- `_laplacian_sharpness()` and `_tenengrad_sharpness()` — unchanged
- `_detect_scene_change()` — unchanged, called on the downscaled BGR frame
- `_split_at_scenes()` — production path uses inline streaming logic, but `sharpness_benchmark.py:98` still calls this method. Keep as a standalone helper.

**What to remove:**
- `_extract_opencv_pass2()` — no longer needed (single-pass writes inline)
- `_extract_frames()` (ffmpeg select filter) — the stalling problem. Confirmed broken in benchmarks. Remove entirely.

**Progress reporting** changes from two-phase (0-85% scoring, 85-100% extracting) to linear (0-100% processing). The GUI progress callback at `source_tab.py:1578` throttles on `"Analyzing:"` substring — neither the old nor new messages contain this, so no GUI changes needed.

**`prefix_source` parameter** must be threaded through to the inline write helper. The existing code uses `stem = video.stem + "_" if prefix_source else ""` and filenames like `{stem}{idx:05d}.{ext}`.

- [ ] **Step 1: Add non-empty output directory check for sharpest mode**

Before writing any frames, check if the sharpest output directory already contains image files. If so, return `SharpestResult(success=False, error="Output directory is not empty")`. This prevents cancellation cleanup from deleting pre-existing files and matches paired extraction's existing `_ensure_empty_output()` pattern. Apply this check before CPU/GPU routing in `extract()`.

- [ ] **Step 2: Refactor `_extract_opencv()` to single-pass**

Add a `_write_winner()` closure inside `_extract_opencv()` that handles format-specific imwrite, filename generation using `stem`, and the output frame counter.

Replace the two-pass body with the single-pass buffer logic. Scene boundary ordering must match the behavior described above (flush before considering boundary frame).

- [ ] **Step 3: Fix manifest timing**

Change `_write_manifest()` line: `time_sec = round(frame_num / fps, 3)` (remove `start_sec +` since frame numbers are absolute).

- [ ] **Step 4: Remove dead methods and clean up**

Remove `_extract_opencv_pass2()` and `_extract_frames()`. Keep `_split_at_scenes()` (still used by `sharpness_benchmark.py:98`).

After removal, clean up:
- Remove unused imports: `os`, `subprocess`, `tempfile`, and `_SUBPROCESS_FLAGS` (all only used by the removed `_extract_frames()`)
- Keep `json` (still used by manifest)
- Update module docstring to describe CPU single-pass streaming extraction, optional scene-aware sub-chunks. Remove references to Basic/Best tiers and ffmpeg select extraction.

- [ ] **Step 5: Verify manually**

Run the GUI, extract from a test video with:
- Scene detection ON — verify correct number of frames, scene splits logged
- Scene detection OFF — verify correct number of frames
- Start/end time range set — verify only that range is processed, manifest timestamps are correct
- Cancel mid-extraction — verify stops cleanly, no partial files remain, no manifest written
- Non-empty output dir — verify extraction is refused with clear error

- [ ] **Step 6: Commit**

```
feat(extract): convert SharpestExtractor to single-pass, fix manifest timing

Eliminates the two-pass decode (score all → re-read winners) by buffering
the current window's best frame and writing it when the window closes.
Removes the ffmpeg select filter path that was causing extraction stalls.
Fixes manifest time_sec double-counting start_sec for absolute frame indices.
```

---

## Task 2: Add GPU-accelerated extraction path

Add `_extract_gpu_single_pass()` to `SharpestExtractor`. This method uses `cv2.cudacodec.VideoReader` for NVDEC decode and `cv2.cuda` operations for scoring. The `extract()` method routes to GPU or CPU based on availability.

**Files:**
- Modify: `prep360/core/sharpest_extractor.py`

### GPU availability detection

Cache the result as a class attribute on first call (GPU availability doesn't change mid-session). Note: this is a coarse check. `createVideoReader()` can still fail per-video (codec/resolution/DLL). The class cache avoids repeating the `getCudaEnabledDeviceCount()` call, not per-video errors.

```python
_gpu_ok: Optional[bool] = None  # class-level cache

def _gpu_available(self) -> bool:
    if SharpestExtractor._gpu_ok is None:
        try:
            import cv2
            SharpestExtractor._gpu_ok = (
                cv2.cuda.getCudaEnabledDeviceCount() > 0
                and callable(getattr(cv2.cudacodec, 'createVideoReader', None))
                and callable(getattr(cv2.cudacodec, 'VideoReaderInitParams', None))
                and callable(getattr(cv2.cuda, 'createSobelFilter', None))
                and callable(getattr(cv2.cuda, 'createLaplacianFilter', None))
                and callable(getattr(cv2.cuda, 'resize', None))
                and callable(getattr(cv2.cuda, 'cvtColor', None))
                and callable(getattr(cv2.cuda, 'sum', None))
                and callable(getattr(cv2.cuda, 'multiply', None))
                and callable(getattr(cv2.cuda, 'add', None))
                and hasattr(cv2.cuda, 'GpuMat')
            )
        except Exception:
            SharpestExtractor._gpu_ok = False
    return SharpestExtractor._gpu_ok
```

This checks every CUDA callable the GPU path actually uses. Standard `opencv-python` (pip) exposes `cv2.cuda` and `cv2.cudacodec` as namespaces but without the callable implementations — namespace checks alone are insufficient. The GUI `_HAS_CUDACODEC` check must use this same function (see Task 4).

### GPU scoring helpers

**Tenengrad (GPU):**
```python
# Pre-create once before the loop:
sobel_x = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 0, 1, ksize=3)

# Per frame:
gx = sobel_x.apply(gpu_8u)
gy = sobel_y.apply(gpu_8u)
gx2 = cv2.cuda.multiply(gx, gx)
gy2 = cv2.cuda.multiply(gy, gy)
energy = cv2.cuda.add(gx2, gy2)
score = cv2.cuda.sum(energy)[0] / (w * h)
```

**Laplacian (GPU):** Must use float32→float32 (src/dst type must match):
```python
# Pre-create once:
lap_filter = cv2.cuda.createLaplacianFilter(cv2.CV_32FC1, cv2.CV_32FC1, ksize=1)

# Per frame (gpu_8u is the uint8 grayscale GpuMat):
gpu_32f = cv2.cuda.GpuMat(gpu_8u.size(), cv2.CV_32FC1)
gpu_8u.convertTo(cv2.CV_32FC1, gpu_32f)
dst = lap_filter.apply(gpu_32f)
sq = cv2.cuda.multiply(dst, dst)
sum_sq = cv2.cuda.sum(sq)
sum_val = cv2.cuda.sum(dst)
n = dst.size()[0] * dst.size()[1]
mean = sum_val[0] / n
variance = sum_sq[0] / n - mean * mean
```

### GPU scene detection

Benchmarked three approaches for scene detection on the GPU path:

| Approach | Cost per frame | PCIe transfer |
|----------|---------------|---------------|
| Download 480px BGR, CPU histogram (2D H+S) | 2.16 ms | ~700 KB |
| Download 1920px BGR, CPU histogram (2D H+S) | 11.07 ms | ~10.8 MB |
| GPU histogram (H channel), download histogram | 1.01 ms | 1 KB |
| GPU grayscale absdiff | 1.60 ms | 8 bytes |

**Use 480px download + existing CPU `_detect_scene_change()` for v1.** Rationale:
- Uses the existing `_detect_scene_change()` method unchanged — proven, tested, same algorithm as CPU path
- 2.16ms overhead is small relative to the ~11ms per-frame scoring cost
- Total per-frame with scene detection: ~13ms (vs ~11ms without) — still 5x faster than CPU

**Scene detection resolution note:** The CPU path uses `scale_width` (1920px) for scene detection because it reuses the same downscaled frame used for scoring. The GPU path uses 480px, which is a behavior change. HSV histogram comparison uses the same bin structure (50 H × 60 S) at either resolution, and the distribution is expected to be robust to downsampling. However, edge cases (hard cuts, exposure shifts, small high-saturation objects) could produce different threshold crossings at different resolutions. The 480px approach was benchmarked at 2.16ms vs 11.07ms for 1920px — a 170-second difference over 16,929 frames. **480px is the v1 target, pending parity validation** (see Step 6). If scene split counts or positions diverge on representative footage, raise to `scale_width` and accept the performance cost.

**Scene detection resize is separate from scoring resize.** The scoring path resizes to `scale_width` (1920px) on GPU. For scene detection, resize to 480px on GPU, download the small BGR frame. These are independent operations on the same source GpuMat.

### Frame preparation pipeline (per frame)

```
GPU: NVDEC decode → full-res BGRA uint16 GpuMat
  ├─ GPU resize to scale_width → cvtColor(BGRA2GRAY) → convertTo(uint8)
  │   └─ GPU scoring (Sobel/Laplacian → sum → scalar) ← 8 bytes cross PCIe
  ├─ GPU resize to 480px → cvtColor(BGRA2BGR) → convertTo(uint8) → download
  │   └─ CPU scene detection (HSV histogram comparison) ← ~700 KB cross PCIe
  └─ If new best: gpu_frame.clone() (stays on GPU)
      └─ On window/scene close: download full-res → convert → imwrite
```

### start_sec / end_sec handling

Use `VideoReaderInitParams.firstFrameIdx` when `start_sec` is nonzero (verified more frame-accurate than `set()`). For `end_sec`, stop the loop when `frame_idx >= end_frame`, same as the CPU path.

Do not read a frame for format detection and then discard it. Pass `firstFrameIdx` at reader creation time, detect format from the first frame, **and process that same frame as `start_frame`** — do not skip it.

```python
params = cv2.cudacodec.VideoReaderInitParams()
if start_frame > 0:
    params.firstFrameIdx = start_frame
try:
    reader = cv2.cudacodec.createVideoReader(video_path, params=params)
except (cv2.error, TypeError):
    # Keyword form verified on cudawarped 4.13.0 wheel; positional fails.
    # If a future wheel changes the binding, this catch handles it.
    return None  # fall back to CPU
# First nextFrame() returns the frame at start_frame
ret, first_gpu = reader.nextFrame()
channels = first_gpu.channels()  # format detection from start frame
# IMPORTANT: score and process first_gpu as frame_idx = start_frame.
# The main loop then starts with the NEXT nextFrame() call.
# Failing to process first_gpu is an off-by-one that drops the first frame.
```

The same applies to paired GPU extraction: after opening both readers and detecting format from the first front/back pair, score and process that pair before entering the main loop.

### Runtime frame format validation

Benchmark evidence shows `uint16 BGRA` (4 channels, 16-bit) from the tested cudawarped wheel and HEVC fisheye input. Do not hardcode this assumption. After reading the first frame, inspect channels and depth:

- **4-channel, uint16 (expected):** Use `convertTo(alpha=1/256)` for uint8 conversion, `BGRA2GRAY` / `BGRA2BGR` for color conversion.
- **4-channel, uint8:** Skip the `>> 8` / `convertTo` scaling. Use `BGRA2GRAY` / `BGRA2BGR` directly.
- **3-channel:** Use `BGR2GRAY` instead of `BGRA2GRAY`. Skip alpha channel handling.
- **Unsupported (1-channel, unusual depth, etc.):** Log clearly (channel count, type value) and return `None` for CPU fallback. Do not produce bad images silently.

This keeps the GPU path safe across codec, wheel, or driver differences.

### Cancellation vs GPU failure

These are **separate outcomes** with different behavior:

| Outcome | Return value | What happens |
|---------|-------------|-------------|
| GPU init failure (NVDEC limit, no device) | `None` | `extract()` falls through to CPU single-pass |
| GPU runtime error (mid-processing CUDA crash) | `None` | Clean up partial output, fall through to CPU |
| **User cancellation** | `SharpestResult(success=False, error="Cancelled")` | **Stop immediately. Do NOT fall back to CPU.** |

The `extract()` routing checks `None` for fallback. A `SharpestResult` (even a failed one) is a final answer — no fallback.

### Cancellation cleanup policy (applies to ALL paths: CPU, GPU, paired)

**Clean cancellation:** delete all frames written by the cancelled attempt and do not write a manifest.

**Pre-existing file safety:** Single-video extraction writes into `base_output / video_name` using sequential filenames (`00001.jpg`, `00002.jpg`). If the directory already contains files from a previous extraction, the new attempt may overwrite them. Cancellation cleanup that deletes tracked paths would then delete the pre-existing files too.

**Mitigation options (pick one during implementation):**
1. **Refuse non-empty output dirs** for sharpest mode, matching paired extraction's existing `_ensure_empty_output()` pattern.
2. **Write to a staging subdirectory** (e.g., `.tmp_extract/`) and rename/move on success.
3. **Record which files are newly created vs overwritten** and only delete newly created ones on cancel.

Option 1 is simplest and consistent with paired extraction. The GUI already creates per-video output subdirectories, so refusing non-empty dirs adds a clear "re-extract means delete old output first" contract.

Tracking per path:
- CPU single-pass: track `frame_paths` written so far, delete on cancel.
- GPU single-pass: track `gpu_written_paths`, delete on cancel.
- Paired GPU: track front+back paths written so far, delete on cancel.
- Paired CPU fallback: same — track and delete.

### Runtime error handling

Wrap the main decode loop in try/except. On CUDA error, clean up only the files written by the GPU attempt, then return `None`.

```python
gpu_written_paths = []  # track files this GPU run created

try:
    while True:
        if cancel_check and cancel_check():
            # Clean up GPU-written files, return cancellation (no fallback)
            for p in gpu_written_paths:
                Path(p).unlink(missing_ok=True)
            return SharpestResult(success=False, error="Cancelled")

        ret, gpu_frame = reader.nextFrame()
        # ... scoring, window logic ...
        # On winner write: gpu_written_paths.append(path)

except cv2.error as e:
    _log(f"  GPU error during processing: {e}")
    _log(f"  Falling back to CPU path")
    for p in gpu_written_paths:
        Path(p).unlink(missing_ok=True)
    return None  # triggers CPU fallback
```

### Result metadata

Add `gpu_accelerated: bool = False` to `SharpestResult` so the GUI can log which path was used.

- [ ] **Step 1: Add `_gpu_available()` with class-level caching**
- [ ] **Step 2: Add `gpu_accelerated` field to `SharpestResult`**
- [ ] **Step 3: Add `_extract_gpu_single_pass()` method**

Include: cudacodec reader with `firstFrameIdx` for seeking, format detection from first frame, pre-created GPU filters, GPU scoring (both Tenengrad and Laplacian), scene detection via 480px BGR download, scene boundary ordering (flush before considering boundary frame — same as Task 1), window boundary logic, `prefix_source` filename stem, cancel check (returns `SharpestResult`, not `None`), manifest writing.

- [ ] **Step 4: Add runtime error handling (try/except with tracked cleanup)**
- [ ] **Step 5: Update `extract()` to route GPU vs CPU**

```python
def extract(self, video_path, output_dir, config=None, ...):
    # ... existing setup ...

    # Try GPU path first
    if self._gpu_available():
        gpu_result = self._extract_gpu_single_pass(
            video_path=str(video), output_dir=str(out), config=config,
            scene_aware=scene_aware,
            progress_callback=progress_callback, cancel_check=cancel_check,
            prefix_source=prefix_source, log=log,
        )
        if gpu_result is not None:  # None = GPU failed, fall through to CPU
            return gpu_result
        # gpu_result is None → GPU init or runtime failure → CPU fallback
        if log:
            log("  Falling back to CPU extraction")

    # CPU single-pass fallback
    return self._extract_opencv(...)
```

- [ ] **Step 6: Scene detection parity check**

Before shipping 480px scene detection, compare against CPU (1920px) on representative footage:
- Run CPU single-pass with scene detection ON, record scene split count and positions
- Run GPU path with scene detection ON (480px), record scene split count and positions
- Compare: if split counts and positions match (within ±1 frame), 480px is validated
- If significant divergence: raise scene detection resolution to `scale_width` (1920px) and accept the 170s performance cost

Test on at least two videos with actual scene changes (not continuous walking footage).

- [ ] **Step 7: Verify manually**

Test with:
1. Fisheye video (3840x3840 HEVC) — should use GPU path, log confirms
2. ERP video (7680x3840 H.264) — should fall back to CPU with log message
3. Start/end time range — should seek correctly on GPU path, manifest timestamps correct
4. Cancel mid-GPU-extraction — should stop cleanly, no CPU fallback, no partial files
5. Scene detection on/off — both GPU and CPU paths

- [ ] **Step 8: Commit**

```
feat(extract): GPU-accelerated extraction via NVDEC + CUDA scoring

Uses cv2.cudacodec for hardware video decode and cv2.cuda for
resize/scoring operations. ~6x faster than CPU for HEVC fisheye
video. Falls back to CPU for H.264 > 4096px wide (NVDEC limit).
Cancellation stops immediately without CPU fallback.
```

---

## Task 3: GPU-accelerated paired extraction

The `PairedSplitVideoExtractor` processes two fisheye videos (front + back) in lockstep. It currently uses CPU decode and CPU scoring for both streams. Two simultaneous `cudacodec.VideoReader` instances have been verified working — the 3090 Ti supports multiple concurrent NVDEC sessions.

**Files:**
- Modify: `prep360/core/paired_split_video_extractor.py`

### Architecture

**Critical constraint: GPU and CPU decoders produce different frames at the same seek index.** Verified: `firstFrameIdx=100` on GPU matches CPU frame 101, not 100 (±1 offset at HEVC GOP boundaries, inconsistent). Therefore the GPU path must **write winners directly from GPU memory** — no CPU re-seek for the write pass.

The paired GPU path is a **streaming single-pass** for sharpest mode, matching the single-video GPU architecture:
1. Open both videos with `cv2.cudacodec.VideoReader`
2. GPU decode + GPU score both streams in lockstep
3. Track the best front+back pair per window/sub-chunk using the existing composite sort key, keeping cloned GpuMats for the current best
4. On window/scene close: download best front+back GpuMats → convert → write to disk
5. Scene detection: download 480px BGR from EACH stream, use existing `_detect_scene_change()` — `front_scene or back_scene` triggers split for both (existing behavior at line 281)
6. Fallback: if either reader fails to initialize, fall back to CPU for both

**Streaming selection equivalence:** Verified with 1000 random windows (0 mismatches): streaming `max()` with the paired sort key `(min(front, back), avg, -imbalance)` produces identical winners to the current collect-then-max approach. Also verified with scene boundary sub-chunking (1000 windows, 0 mismatches).

**GPU memory:** Two cloned GpuMats (best front + best back) at 3840×3840 uint16 BGRA = ~224 MB. On 24 GB VRAM this is acceptable.

**What NOT to change:**
- `_pair_sharpness_sort_key()` — pure Python, used for streaming comparison
- `_extract_selected_pairs()` — kept for fixed-interval mode and CPU fallback
- `_select_from_paired_entries()` — kept for CPU fallback path
- Output safety checks (non-empty dir refusal)

### Implementation

- [ ] **Step 1: Add `gpu_accelerated: bool = False` to `PairedSplitResult`**
- [ ] **Step 2: Add GPU scoring method to `PairedSplitVideoExtractor`**

Add `_gpu_score_frame()` method that takes a GpuMat and returns a sharpness score, using the same GPU Sobel/Laplacian approach from Task 2. Pre-create filters once per extraction call.

- [ ] **Step 3: Add GPU streaming paired extraction**

Add a new method `_extract_sharpest_gpu()` (or integrate into `extract()`) that handles the full GPU streaming pipeline for sharpest mode.

**Resource handling:** The current `extract()` opens CPU captures for metadata (fps, frame count) before calling the selection method. Release these CPU captures before opening GPU readers to avoid having 4 readers open simultaneously (2 CPU + 2 GPU), which wastes file handles and risks decoder contention on Windows. Collect metadata first, release, then open GPU readers.

Steps:
- Collect fps/frame count from CPU `VideoCapture`, release immediately
- Open both videos with `cudacodec.createVideoReader` (with `firstFrameIdx` for seeking, keyword `params=params`)
- If either fails, return `None` to signal CPU fallback
- GPU decode both streams in lockstep, GPU score both
- Track best pair per window/sub-chunk using `_pair_sharpness_sort_key()` for comparison
- Clone best front+back GpuMats when a new best is found
- On window/scene close: download cloned GpuMats → convert uint16 BGRA → uint8 BGR → imwrite to `front/frames/` and `back/frames/`
- Scene detection: download 480px BGR from each stream, use `SharpestExtractor._detect_scene_change()` on each, OR results
- Write paired manifest at the end with selected frame indices, timestamps, and scores
- Set `PairedSplitResult(gpu_accelerated=True)` on success

**First-frame processing:** After opening both readers and detecting format from the first front/back pair, score and process that pair as `frame_idx = start_frame` before entering the main loop. Do not discard it.

**Cancellation and error handling — ordering matters:**

The GPU decode loop must check cancellation **before** the broad GPU error handler catches it. Structure:

```python
gpu_written_paths = []

try:
    for frame_idx in range(start_frame, end_frame):
        # 1. Check cancel FIRST — inside the try, returns a result (no fallback)
        if cancel_check and cancel_check():
            for p in gpu_written_paths:
                Path(p).unlink(missing_ok=True)
            raise RuntimeError("Cancelled")  # caught by extract(), not by except below

        # 2. GPU decode + score + write logic
        ...

except RuntimeError:
    raise  # re-raise cancel — do NOT catch it as a GPU error

except cv2.error as e:
    # 3. GPU runtime error — clean up, return None for CPU fallback
    for p in gpu_written_paths:
        Path(p).unlink(missing_ok=True)
    return None
```

This ensures cancellation always propagates as `RuntimeError("Cancelled")` to `extract()`, which handles it as a failed `PairedSplitResult`. GPU errors return `None` for CPU fallback. The two paths never cross.

- [ ] **Step 4: Update `extract()` routing for sharpest mode**

For sharpest mode: try GPU path first, fall back to existing CPU path (which still uses `_select_sharpest_frame_indices()` + `_extract_selected_pairs()` two-pass). For fixed-interval mode: use existing CPU path (no GPU benefit since there's no scoring).

- [ ] **Step 5: Verify manually (no separate error handling step — integrated into Step 3)**

- [ ] **Step 6: Verify manually**

Test with:
1. Fountain front+back fisheye pair — should use GPU for both streams
2. Cancel mid-scoring — stops cleanly, no CPU fallback
3. Scene detection on/off — verify `front_scene or back_scene` still triggers correctly
4. Compare GPU output frames against CPU output — verify they are sharp, well-distributed, and the front/back pairs are temporally aligned

- [ ] **Step 7: Commit**

```
feat(extract): GPU-accelerated paired fisheye extraction

Uses two simultaneous NVDEC sessions to decode front+back fisheye
streams. Streaming single-pass: GPU scores both streams and writes
winners directly from GPU memory to avoid GPU/CPU frame index
misalignment. Fallback to CPU two-pass if either reader fails.
```

---

## Task 4: GUI integration

Add GPU status indicator to the Extract tab and log GPU usage across all extraction paths. No new tabs, no popout dialogs — inline in existing settings.

**Files:**
- Modify: `reconstruction_gui/tabs/source_tab.py`

- [ ] **Step 1: Add GPU availability check at tab build time**

After the prep360 import block, detect GPU. Use lazy import pattern (consistent with codebase — no import-time hard failure if `cv2` is absent or non-CUDA):

```python
_HAS_CUDACODEC = False
try:
    from prep360.core.sharpest_extractor import SharpestExtractor
    _HAS_CUDACODEC = SharpestExtractor()._gpu_available()
except Exception:
    pass
```

Reuses the extractor's hardened check directly. Do not duplicate the callable list — a single source of truth prevents the GUI from showing "GPU ready" when the extractor would immediately fall back.

- [ ] **Step 2: Add GPU indicator to extraction settings**

In the extraction settings grid (row 1, col 1, `bottom_right` frame), add a passive status label:

```python
gpu_label = ctk.CTkLabel(
    bottom_right,
    text="GPU ready" if _HAS_CUDACODEC else "CPU only",
    text_color="#4CAF50" if _HAS_CUDACODEC else COLOR_TEXT_DIM,
    font=ctk.CTkFont(size=11),
)
gpu_label.pack(side="right", padx=(8, 0))
```

No toggle — GPU is used automatically when available. Do not add a toggle until the fallback path has been used in real projects.

- [ ] **Step 3: Log GPU acceleration status in all extraction paths**

Three GUI logging sites:

1. **`_extract_single_worker()`** — after `SharpestResult` returned
2. **Queue extraction loop in `_extract_queue_worker()`** — after each item's `SharpestResult`

```python
if sharp_result.gpu_accelerated:
    app.log("  Acceleration: GPU (NVDEC + CUDA scoring)")
else:
    app.log("  Acceleration: CPU")
```

3. **`_paired_split_video_worker()`** — after `PairedSplitResult` returned (uses `result`, not `sharp_result`)

```python
if result.gpu_accelerated:
    app.log("  Acceleration: GPU (NVDEC + CUDA scoring)")
else:
    app.log("  Acceleration: CPU")
```

- [ ] **Step 4: Verify end-to-end**

Launch the GUI. Verify:
- GPU indicator shows "GPU ready" in green (or "CPU only" if no CUDA OpenCV)
- GUI starts without error when CUDA OpenCV is not installed
- Extract from fisheye video — log shows "GPU (NVDEC + CUDA scoring)"
- Extract from ERP video — log shows fallback message and "CPU"
- Paired extraction — log shows GPU for scoring pass
- Queue with mixed GPU-eligible and CPU-fallback videos — each item logs independently
- Cancel works in both paths, no fallback after cancel

- [ ] **Step 5: Commit**

```
feat(gui): show GPU acceleration status in Extract tab
```

---

## Implementation Notes

### Lazy imports

All `cv2.cuda` and `cv2.cudacodec` imports must be inside functions, never at module top level. This is a codebase convention — optional dependencies are imported lazily so the tool works without CUDA OpenCV installed. The `_gpu_available()` check itself uses a lazy import.

### Packaging

`opencv-python` (pip) will never include `cv2.cudacodec`. GPU acceleration requires the `opencv-contrib-python` CUDA wheel from [cudawarped](https://github.com/cudawarped/opencv-python-cuda-wheels). Multiple OpenCV wheel variants (`opencv-python`, `opencv-contrib-python`, `opencv-python-headless`) can coexist in pip metadata while only one `cv2` is actually imported. To avoid confusion: **uninstall all OpenCV wheel variants before installing the CUDA wheel.** Do not rely on package names to determine capability — use runtime callable checks (as `_gpu_available()` does).

For v1, treat CUDA OpenCV as an optional local setup:
- The tool works fully without it (CPU fallback is always available)
- Setup instructions in `benchmarks/BENCHMARK_RESULTS.md` (already documented)
- PyInstaller packaging with CUDA OpenCV is out of scope for this plan

### What NOT to change

- `SharpestConfig` — no new fields needed. GPU routing is automatic.
- `MotionSelector` — separate component, not part of this plan
- Post-processing filters (LUT, sky, blur, motion) — not part of this plan
- `FrameExtractor` (fixed-interval mode) — uses ffmpeg directly, no scoring involved

### Fallback chain

```
SharpestExtractor.extract() called
  → _gpu_available()?
    → Yes: try _extract_gpu_single_pass()
      → User cancelled: return SharpestResult(success=False, error="Cancelled")
         → extract() returns it immediately (NO CPU fallback)
      → cudacodec.createVideoReader() fails (NVDEC limit, DLL): return None
         → extract() falls through to CPU
      → Runtime CUDA error: clean up written files, return None
         → extract() falls through to CPU
      → Success: return SharpestResult(gpu_accelerated=True)
         → extract() returns it immediately
    → No: skip
  → _extract_opencv() (CPU single-pass, always works)
```

---

## Acceptance matrix

| Scenario | Expected result |
|----------|-----------------|
| Single video, CPU-only environment | Sharpest extraction succeeds via CPU single-pass; no ffmpeg select path. |
| Single video, HEVC fisheye, CUDA OpenCV | GPU path runs; `gpu_accelerated=True`; log says GPU. |
| Single video, H.264 7680×3840 ERP, CUDA OpenCV | GPU reader fails; CPU fallback succeeds; log says fallback. |
| Single video, scene detection on | Scene splits produce representatives on both sides of cuts. Boundary frame belongs to the new segment. |
| Single video, start/end range | Output frames and manifest timestamps refer to actual source-video times. No double-counting. |
| Single video, cancel during GPU | Stops cleanly, no CPU fallback, no partial files. |
| Queue extraction | Each item logs GPU or CPU independently. |
| Paired HEVC fisheye, CUDA OpenCV | GPU streaming single-pass for both streams; winners written from GPU memory; paired output and manifest valid. |
| Paired fallback | If either stream cannot use NVDEC, CPU selection runs. |
| GUI startup without CUDA OpenCV | No crash; indicator says "CPU only". |

---

## Automated tests (CPU-only, no CUDA hardware required)

These tests verify CPU streaming logic, fallback behavior, and metadata correctness without needing a GPU. They can use short synthetic videos generated with ffmpeg.

| Test | What it verifies |
|------|-----------------|
| CPU single-pass scene-off output count | Window count matches `ceil(frames / window_size)` for a known video |
| CPU single-pass scene boundary ordering | Boundary frame belongs to new sub-chunk (synthesize a video with a hard cut, verify winners on both sides) |
| CPU single-pass `start_sec` / `end_sec` range | Only frames in range are processed; output count matches expected windows |
| Manifest `time_sec` correctness | `time_sec == frame_num / fps` (no double-count of `start_sec`) for nonzero `start_sec` |
| Cancellation returns failure | Cancel flag set before extraction → `SharpestResult(success=False, error="Cancelled")`, no manifest written |
| Cancellation cleanup | After cancel, no partial frames remain in output dir |
| `_gpu_available()` returns False on standard OpenCV | Monkeypatch `cv2.cuda.getCudaEnabledDeviceCount` to return 0 → `_gpu_available()` is False |
| GPU unavailable fallback | Monkeypatch `_gpu_available()` to False → `extract()` uses CPU path, `gpu_accelerated=False` |
| GPU init failure fallback | Monkeypatch `createVideoReader` to raise `cv2.error` → CPU fallback runs |
| `SharpestResult.gpu_accelerated` default | Default is `False`, explicitly set to `True` only by GPU path |
| `PairedSplitResult.gpu_accelerated` default | Default is `False`, explicitly set to `True` only by GPU path |
| `sharpness_benchmark.py` still runs | `_split_at_scenes()` exists and produces expected sub-chunks for known input |
| Non-empty output directory refused | Sharpest extraction into a dir with existing files → `SharpestResult(success=False)` before writing anything |
| Paired CPU cancellation cleanup | Cancel during paired CPU extraction → no partial front/back frames remain, no paired manifest written |

**Test infrastructure note:** Tests that monkeypatch `_gpu_available()` or CUDA callables must reset `SharpestExtractor._gpu_ok = None` between test cases. The class-level cache persists across calls and can cause order-dependent false positives if not cleared.

CUDA-specific behavior (GPU decode, GPU scoring, NVDEC limits) should be validated manually on real footage.

---

## Future optimizations (not in this plan)

1. **GPU histogram scene detection** — Replace 480px BGR download + CPU HSV histogram with GPU-computed 1D H-channel histogram + download tiny histogram array (1 KB). Saves ~1ms per frame. Requires validating that 1D histogram scene detection matches 2D histogram results across diverse footage.

2. **GPU Laplacian via uint8 path** — `createLaplacianFilter` fails for mixed src/dst types, but `createLinearFilter` with a custom kernel also fails the same way. A potential workaround is to use `createDerivFilter` or compute Laplacian as `(Sobel_xx + Sobel_yy)` using two Sobel filters. Would eliminate the float32 conversion overhead.

3. **Scoring resolution reduction** — Benchmarks used 1920px for scoring. Dropping to 960px would halve the Sobel/Laplacian compute cost with negligible impact on relative sharpness ranking. Could be offered as a "fast" mode.

4. **"Force CPU" debug switch** — An explicit toggle for users to bypass GPU when comparing results or working around CUDA driver issues.

5. **Shared CUDA scoring helper** — If single-video and paired implementations duplicate too much GPU conversion/scoring code, extract into a shared internal module. Defer until duplication is proven problematic.
