# GPU-Accelerated Extraction — Test Plan

**Implementation:** `prep360/core/sharpest_extractor.py`, `prep360/core/paired_split_video_extractor.py`
**GUI:** `reconstruction_gui/tabs/source_tab.py`
**Benchmark:** `benchmarks/extraction_benchmark.py`, `benchmarks/BENCHMARK_RESULTS.md`
**Architecture plan:** `planning/plans/2026-05-05-gpu-accelerated-extraction.md`

---

## Part 1: CPU-Only Automated Tests

These tests verify the streaming single-pass logic, fallback behavior, metadata correctness, and cancellation safety without requiring a GPU. They should run in CI and on any machine with standard `opencv-python`.

### Test infrastructure

- Use short synthetic videos generated with ffmpeg in a pytest fixture (e.g., 5 seconds at 30fps = 150 frames, 640x480, H.264)
- For scene detection tests, generate a video with a hard cut (first half solid blue, second half solid red)
- All tests use `SharpestExtractor` and `PairedSplitVideoExtractor` directly — no GUI involvement
- Tests that monkeypatch `_gpu_available()` or CUDA callables must reset `SharpestExtractor._gpu_ok = None` in teardown to avoid order-dependent failures

### Synthetic video fixture

```python
import subprocess, tempfile
from pathlib import Path

def create_test_video(output_path, duration=5.0, fps=30, width=640, height=480):
    """Create a simple test video with frame-count text overlay."""
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output_path),
    ], check=True, capture_output=True)

def create_scene_cut_video(output_path, fps=30, width=640, height=480):
    """Create a video with a hard scene cut at the midpoint."""
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:s={width}x{height}:d=2.5:r={fps}",
        "-f", "lavfi", "-i", f"color=c=red:s={width}x{height}:d=2.5:r={fps}",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output_path),
    ], check=True, capture_output=True)
```

---

### Test 1: CPU single-pass scene-off output count

**What it proves:** The streaming single-pass selects the correct number of winners.

```
Given: 5s video at 30fps (150 frames), interval=2.0s, scene detection OFF
Expected windows: ceil(150 / 60) = 3  (window_size = 30fps * 2.0s = 60)
Assert: result.frames_extracted == 3
Assert: result.success == True
Assert: len(output_dir.glob("*.jpg")) == 3
```

### Test 2: CPU single-pass scene boundary ordering

**What it proves:** The boundary frame belongs to the NEW sub-chunk, not the old one. This is the behavior defined by `_split_at_scenes()`.

```
Given: scene-cut video (blue→red at midpoint), interval=10.0s (one big window), scene detection ON
Expected: 2 winners — one from the blue segment, one from the red segment
Assert: result.frames_extracted == 2
Bonus: read the two output frames, verify one is predominantly blue and the other red
```

### Test 3: CPU single-pass start_sec / end_sec range

**What it proves:** Only the requested frame range is processed.

```
Given: 5s video, interval=1.0s, start_sec=1.0, end_sec=3.0
Expected: processing 2 seconds = 60 frames at 30fps
Expected windows: ceil(60 / 30) = 2
Assert: result.frames_extracted == 2
Assert: result.total_frames_analyzed == 60 (approximately — may be ±1 due to rounding)
```

### Test 4: Manifest time_sec correctness

**What it proves:** `time_sec = frame_num / fps` with no double-count of `start_sec`.

```
Given: 5s video at 30fps, interval=2.0s, start_sec=2.0
Extract, read manifest JSON
For each frame entry:
  Assert: entry["time_sec"] == entry["source_frame"] / 30.0
  Assert: entry["time_sec"] >= 2.0 (not less than start_sec)
  Assert: no entry has time_sec >= 2 * start_sec (would indicate double-count)
```

### Test 5: Cancellation returns failure

**What it proves:** Cancel flag before extraction starts → immediate failure, no manifest.

```
Given: cancel flag already set before calling extract()
Assert: result.success == False
Assert: result.error == "Cancelled"
Assert: manifest file does NOT exist in output dir
```

### Test 6: Cancellation cleanup

**What it proves:** No partial frames remain after cancel mid-extraction.

```
Given: 5s video, cancel flag set after a short delay (e.g., 0.1s via threading.Timer)
Run extract() — it should cancel partway through
Assert: result.success == False
Assert: no .jpg files remain in output dir
```

### Test 7: Non-empty output directory refused

**What it proves:** Extraction into a dir with existing files fails before writing.

```
Given: output dir with a pre-existing file (any file, not just images)
Assert: result.success == False
Assert: "not empty" in result.error
Assert: pre-existing file still exists (not deleted)
Assert: no new files created
```

### Test 8: `_gpu_available()` returns False on standard OpenCV

**What it proves:** The hardened callable check correctly detects non-CUDA OpenCV.

```
Monkeypatch cv2.cuda.getCudaEnabledDeviceCount to return 0
Reset SharpestExtractor._gpu_ok = None
Assert: SharpestExtractor()._gpu_available() == False
Teardown: reset _gpu_ok = None
```

### Test 9: GPU unavailable fallback

**What it proves:** When GPU is unavailable, CPU path runs and result reflects it.

```
Monkeypatch SharpestExtractor._gpu_available to return False
Reset SharpestExtractor._gpu_ok = None
Run extract() on the test video
Assert: result.success == True
Assert: result.gpu_accelerated == False
Teardown: reset _gpu_ok = None
```

### Test 10: GPU init failure fallback

**What it proves:** When `createVideoReader` throws, CPU fallback runs.

```
Monkeypatch cv2.cudacodec.createVideoReader to raise cv2.error
Reset SharpestExtractor._gpu_ok = None
Force _gpu_available to return True (so GPU path is attempted)
Run extract()
Assert: result.success == True
Assert: result.gpu_accelerated == False (fell back to CPU)
Teardown: reset _gpu_ok = None
```

### Test 11: SharpestResult.gpu_accelerated default

**What it proves:** The field defaults correctly.

```
Assert: SharpestResult(success=True).gpu_accelerated == False
Assert: SharpestResult(success=True, gpu_accelerated=True).gpu_accelerated == True
```

### Test 12: PairedSplitResult.gpu_accelerated default

**What it proves:** The field defaults correctly.

```
Assert: PairedSplitResult(success=True).gpu_accelerated == False
Assert: PairedSplitResult(success=True, gpu_accelerated=True).gpu_accelerated == True
```

### Test 13: `_split_at_scenes()` still works

**What it proves:** The helper method is preserved and produces correct sub-chunks.

```
Given: [(0, 10.0, False), (1, 20.0, False), (2, 15.0, True), (3, 25.0, False)]
Assert: result == [
    [(0, 10.0, False), (1, 20.0, False)],
    [(2, 15.0, True), (3, 25.0, False)],
]
```

### Test 14: Non-empty output directory refused (manifest counts as non-empty)

**What it proves:** A leftover manifest from a previous run also triggers refusal.

```
Given: output dir with only an extraction_manifest.json (no images)
Assert: result.success == False
Assert: "not empty" in result.error
```

### Test 15: Paired CPU cancellation cleanup

**What it proves:** No partial front/back frames remain after paired CPU cancel.

```
Given: two short test videos, cancel flag set after short delay
Run PairedSplitVideoExtractor.extract() with sharpest mode
Assert: result.success == False
Assert: no .jpg files in front/frames/ or back/frames/
Assert: no paired_extraction_manifest.json
```

### Test 16: Window boundary at exact frame count multiple

**What it proves:** No off-by-one when frame count is an exact multiple of window_size.

```
Given: video with exactly 120 frames at 30fps, interval=2.0s (window_size=60)
Expected: exactly 2 windows, no partial flush
Assert: result.frames_extracted == 2
```

---

## Part 2: CUDA Hardware Tests

These tests require a machine with an NVIDIA GPU, CUDA 13.1+ runtime, and the cudawarped OpenCV CUDA wheel installed. They verify GPU decode correctness, NVDEC limits, frame alignment, and scene detection parity.

**Skip condition:** All tests should be decorated with `@pytest.mark.skipif(not _has_cuda(), ...)` so they skip gracefully in CI or on machines without CUDA OpenCV.

```python
def _has_cuda():
    try:
        from prep360.core.sharpest_extractor import SharpestExtractor
        return SharpestExtractor()._gpu_available()
    except Exception:
        return False
```

---

### Test C1: GPU path runs on HEVC fisheye

**What it proves:** The GPU path activates and produces correct output for the primary use case.

**Test video:** Any 3840x3840 HEVC fisheye clip (e.g., `CAM_20251114115500_0002_D_back.mp4`, first 10s)

```
Config: interval=2.0, scoring_method="tenengrad", scene_detection=False, end_sec=10.0
Assert: result.success == True
Assert: result.gpu_accelerated == True
Assert: result.frames_extracted == 5 (10s / 2.0s)
Assert: output frames exist and are valid JPEGs (can be opened with cv2.imread)
Assert: log contains "GPU: NVDEC decode"
```

### Test C2: CPU fallback on H.264 ERP exceeding NVDEC limits

**What it proves:** The GPU path fails gracefully and CPU takes over for oversized H.264.

**Test video:** Any 7680x3840 H.264 ERP clip (e.g., `circlePass2.mp4`, first 4s)

```
Config: interval=2.0, scoring_method="laplacian", scene_detection=False, end_sec=4.0
Assert: result.success == True
Assert: result.gpu_accelerated == False
Assert: log contains "cudacodec reader failed" or similar
Assert: log contains "Falling back to CPU"
```

### Test C3: GPU cancellation does not fall back to CPU

**What it proves:** Cancel during GPU processing stops immediately.

**Test video:** Any HEVC fisheye clip (full length, no end_sec — gives time to cancel)

```
Set cancel flag after 0.5s via threading.Timer
Assert: result.success == False
Assert: result.error == "Cancelled"
Assert: log does NOT contain "Falling back to CPU"
Assert: no partial frames in output dir
```

### Test C4: GPU start_sec / end_sec seeking

**What it proves:** `firstFrameIdx` correctly positions the reader.

**Test video:** Any HEVC fisheye clip

```
Config: interval=2.0, start_sec=10.0, end_sec=20.0, scene_detection=False
Assert: result.success == True
Assert: result.gpu_accelerated == True
Assert: result.frames_extracted == 5 (10s / 2.0s)
Read manifest: all time_sec values are between 10.0 and 20.0
Assert: first frame's time_sec >= 10.0
Assert: last frame's time_sec < 20.0
```

### Test C5: GPU scene detection (functional)

**What it proves:** Scene detection runs without error on the GPU path.

**Test video:** Any HEVC fisheye clip (first 10s)

```
Config: interval=2.0, scene_detection=True, end_sec=10.0
Assert: result.success == True
Assert: result.gpu_accelerated == True
Assert: log contains "scene detection: True"
Assert: log contains "Scene changes detected:" (with a count, even if 0)
```

### Test C6: Scene detection parity — 480px GPU vs 1920px CPU

**What it proves:** The GPU's 480px scene detection produces the same split positions as CPU's 1920px detection on footage with actual scene changes.

**Test video:** A video with known scene changes (or the synthetic scene-cut video if it's HEVC-encoded). Ideally test with 2+ real-world clips.

```
Run 1: CPU single-pass (monkeypatch _gpu_available=False), scene detection ON
  Record: scene_count, winner frame numbers

Run 2: GPU path, scene detection ON
  Record: scene_count, winner frame numbers

Assert: scene counts match (or differ by at most 1)
Assert: winner frame numbers match within ±2 frames per window
If significant divergence: flag for investigation (may need to raise scene detection
  to scale_width instead of 480px)
```

### Test C7: GPU paired extraction — both streams use GPU

**What it proves:** Two simultaneous NVDEC sessions work for paired fisheye.

**Test video:** Front + back HEVC fisheye pair (e.g., fountain clips, first 10s)

```
Config: mode="sharpest", interval=2.0, scene_detection=False, end_sec=10.0
Assert: result.success == True
Assert: result.gpu_accelerated == True
Assert: result.pair_count == 5
Assert: front/frames/ has 5 files
Assert: back/frames/ has 5 files
Assert: paired_extraction_manifest.json exists and is valid JSON
Assert: manifest has "gpu_accelerated": true
Assert: log contains "GPU: paired NVDEC decode"
```

### Test C8: GPU paired scene detection — synced across lenses

**What it proves:** `front_scene OR back_scene` triggers splits for both streams.

**Test video:** Front + back HEVC fisheye pair (first 20s, scene detection ON)

```
Config: mode="sharpest", interval=2.0, scene_detection=True, end_sec=20.0
Assert: result.success == True
Assert: log contains "Scene changes detected:"
Assert: front and back frame counts are equal (paired winners are always synced)
```

### Test C9: GPU paired cancellation

**What it proves:** Paired GPU cancel cleans up and doesn't fall back.

**Test video:** Front + back HEVC fisheye pair (full length)

```
Set cancel flag after 1.0s via threading.Timer
Assert: result.success == False
Assert: "Cancelled" in result.error
Assert: no files in front/frames/ or back/frames/
Assert: no paired_extraction_manifest.json
```

### Test C10: GPU frame format validation

**What it proves:** The GPU path correctly detects uint16 BGRA and handles it.

**Test video:** Any HEVC fisheye clip (first 4s)

```
Assert: result.success == True
Assert: log contains "4ch uint16" (or "3ch uint8" — depends on codec/wheel)
Assert: output frames are valid uint8 JPEGs (not 16-bit, not BGRA)
  Verify: cv2.imread(frame) returns a (H, W, 3) uint8 array
```

### Test C11: GPU output frame quality

**What it proves:** GPU-extracted frames are sharp and not corrupted.

**Test video:** HEVC fisheye clip (first 10s)

```
Extract with GPU path
For each output frame:
  img = cv2.imread(path)
  Assert: img is not None
  Assert: img.shape[2] == 3 (BGR, not BGRA)
  Assert: img.dtype == np.uint8
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
  Assert: sharpness > 100 (not a blank/corrupt frame)
```

---

## Running the tests

### CPU tests (any machine)
```bash
pytest tests/test_sharpest_extraction.py -v
```

### CUDA tests (GPU workstation only)
```bash
pytest tests/test_sharpest_extraction.py -v -m cuda
```

Or run only CPU tests:
```bash
pytest tests/test_sharpest_extraction.py -v -m "not cuda"
```

### Test video paths

CUDA tests need real video files. Configure via environment variable or a test config:

```python
FISHEYE_VIDEO = os.environ.get(
    "TEST_FISHEYE_VIDEO",
    r"D:\Edgeworks\deliverables\york-crossing_the_plains\input\fisheye\CAM_20251114115500_0002_D_back.mp4",
)
ERP_VIDEO = os.environ.get(
    "TEST_ERP_VIDEO",
    r"D:\Capture\sunkenGardens2025\circlePass2.mp4",
)
FISHEYE_FRONT = os.environ.get(
    "TEST_FISHEYE_FRONT",
    r"D:\Capture\fountain_may3\osmo\raw_fisheyes\CAM_20260503185323_0026_D_front.mp4",
)
FISHEYE_BACK = os.environ.get(
    "TEST_FISHEYE_BACK",
    r"D:\Capture\fountain_may3\osmo\raw_fisheyes\CAM_20260503185323_0026_D_back.mp4",
)
```

---

## Acceptance criteria

**CPU tests pass** = streaming logic, fallback, cancellation, manifest timing are correct. Safe to merge.

**CUDA tests pass** = GPU decode, scoring, fallback, scene detection, paired extraction are correct. Safe to ship GPU acceleration to users.

**Scene detection parity (C6) passes** = 480px histogram scene detection is validated. If it fails, raise to `scale_width` (1920px) and accept the performance cost documented in `benchmarks/BENCHMARK_RESULTS.md`.
