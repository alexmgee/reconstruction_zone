# Extraction Benchmark Results

## Date: 2026-05-05

## Test Video
- **File:** `CAM_20251114115500_0002_D_back.mp4`
- **Resolution:** 3840x3840 (square fisheye)
- **Codec:** HEVC (H.265)
- **FPS:** 49.57
- **Frames:** 16,929
- **Duration:** 341.5s
- **Note:** Corrupt reference frame at POC 27 (non-fatal decode warning)

## Test Environment
- **GPU:** NVIDIA RTX 3090 Ti (compute 8.6, NVDEC 5th gen)
- **CPU:** Intel 13th Gen
- **CUDA:** 13.1 runtime (side-by-side with 12.9)
- **OpenCV:** 4.13.0 with CUDA/cudacodec (cudawarped wheel)
- **cuDNN:** 9.21.1
- **Python:** 3.14

## Extraction Parameters
- **Interval:** 2.0s (~170 windows of ~99 frames each)
- **Analysis width:** 1920px
- **Quality:** 95 JPEG
- **Scoring:** Laplacian variance (methods 1-5), Tenengrad (method 6)

---

## Method Descriptions

### Method 1: current_twopass
The original `SharpestExtractor` approach.
- Pass 1: CPU decodes every frame, resizes, scores, discards pixels
- Pass 2: Re-opens video, seeks to winners via OpenCV or ffmpeg select filter
- The ffmpeg select filter (`eq(n,X)+eq(n,Y)+...`) re-decodes the entire video
- **This is what stalled** — ffmpeg stuck at frame 45/165 for 15+ minutes

### Method 2: single_pass
Same as method 1, but buffers the current window's best frame in CPU RAM.
When window closes, writes the buffered frame. No Pass 2.
- Total CPU decodes: 16,929 (once, no re-read)
- Extra RAM: ~45 MB (one 3840x3840 BGR frame)

### Method 3: nvdec_pipe
ffmpeg decodes on GPU via NVDEC, pipes raw grayscale frames to Python for scoring.
Winners extracted via OpenCV seek.
- **Broken:** ffmpeg emitted 17,077 frames vs OpenCV's 16,929 (HEVC B-frame/keyframe
  duplication). Window boundaries shift, wrong frames selected. 2.9% winner agreement.
- **Abandoned:** Two decoders disagreeing on frame count is a design flaw, not a bug.

### Method 4: cudacodec_single_pass
`cv2.cudacodec.VideoReader` decodes on GPU. Downloads full-res frame to CPU every
iteration for scoring. Single-pass (buffers winner).
- 99.4% winner agreement with method 2 (identical scoring, just GPU decode)
- Slow because every frame downloads full 3840x3840 uint16 BGRA (~42 MB) to CPU
- The `imwrite` depth warning was caused by uint16 frames not being converted to uint8

### Method 5: cudacodec_optimized
Same as method 4, but resizes and converts to grayscale ON the GPU before downloading.
Only the small grayscale frame (~3.5 MB) crosses PCIe for scoring.
Full-res download only for the ~171 winners.
- Scoring: Laplacian on CPU (still the bottleneck at this point)

### Method 6: cudacodec_full_gpu
Same architecture as method 5, but scoring also happens on GPU:
- GPU Sobel → GPU multiply → GPU add → `cv2.cuda.sum()` → single scalar to CPU
- Only 8 bytes cross PCIe per frame for scoring
- Full-res download only for winners

---

## Results

### York fisheye — optimization progression (3840×3840 HEVC, 16,929 frames)
| Method | Time (s) | vs baseline | Notes |
|--------|----------|-------------|-------|
| single_pass (CPU Laplacian) | 1219.4 | baseline | CPU decode bottleneck |
| nvdec_pipe | 893.0 | 1.37x | **Broken** — frame count mismatch |
| cudacodec_single_pass | 1068.9 | 1.14x | PCIe bandwidth wasted |
| cudacodec_optimized (GPU decode + CPU Laplacian) | 608.8 | 2.00x | CPU Laplacian bottleneck |
| cudacodec_full_gpu_tenengrad | 214.5 | 5.7x | |
| **cudacodec_full_gpu_laplacian** | **198.6** | **6.1x** | Fastest |

### Cross-video comparison — GPU full pipeline
| Video | Resolution | Codec | Frames | GPU Tenengrad | GPU Laplacian | FPS (Lap) |
|-------|-----------|-------|--------|---------------|---------------|-----------|
| york fisheye | 3840×3840 | HEVC | 16,929 | 214.5s | 198.6s | 85 |
| fountain fisheye | 3840×3840 | HEVC | 39,746 | 515.5s | 472.1s | 84 |
| **ERP (circlePass2)** | **7680×3840** | **H.264** | **4,951** | **FAILED** | **FAILED** | — |

### NVDEC hardware limit
The ERP video fails because NVDEC's H.264 decoder is limited to **4096px max width**
(hardware ASIC constraint, not a software limitation). HEVC supports up to 8192×8192.
Most 360° cameras encode equirectangular video as H.264 at 7680×3840, so this will
always require the CPU fallback path. **The GPU path is for fisheye/perspective only
unless the source is HEVC-encoded ERP.**

### Laplacian vs Tenengrad scoring — GPU comparison
| Video | Tenengrad | Laplacian | Laplacian advantage | Winner agreement |
|-------|-----------|-----------|--------------------:|-----------------|
| york fisheye | 214.5s | 198.6s | 8% faster | 34.5% exact, 47.4% within 2 |
| fountain fisheye | 515.5s | 472.1s | 9% faster | 26.6% exact, 38.9% within 2 |

GPU Laplacian is consistently ~8-9% faster than GPU Tenengrad. Both produce valid
sharp frame selections. Agreement between the two methods is ~35% exact / ~43% within
2 frames — they often disagree on which frame is "sharpest" in a window because
Laplacian measures second-derivative edge response while Tenengrad measures first-
derivative gradient energy.

### Winner agreement (frame content comparison via NCC)
| Comparison | Identical (>0.999) | Similar (>0.95) | Different (≤0.95) |
|------------|-------------------|-----------------|-------------------|
| single_pass vs cudacodec_optimized (both Laplacian) | 31.0% | 16.4% | 52.6% |
| single_pass vs cudacodec_full_gpu (Laplacian vs Tenengrad) | 25.1% | 20.5% | 54.4% |

**Why GPU and CPU paths pick different frames (same scoring method):** NVDEC and
OpenCV's CPU HEVC decoder produce slightly different pixel values for the same frame.
`cv2.cuda.resize` and `cv2.resize` also differ by small rounding amounts. These tiny
differences change sharpness scores enough to tip the balance when two frames in a
window have similar quality. Neither set is "wrong" — both paths produce sharp,
well-distributed frames.

---

## Per-Operation Profiling (per frame, 1920x1920)

### Pipeline breakdown (method 5, cudacodec_optimized)
| Operation | Time (ms) | % of total |
|-----------|-----------|-----------|
| Laplacian (CPU) | 20.78 | 64% |
| Download gray (GPU→CPU) | 5.26 | 16% |
| Decode (GPU NVDEC) | 4.13 | 13% |
| Resize (GPU) | 1.37 | 4% |
| CvtColor (GPU) | 0.59 | 2% |
| Clone (amortized) | 0.39 | 1% |
| **Total** | **36.0** | — |

### GPU vs CPU scoring comparison
| Scoring Method | CPU (ms) | GPU (ms) | Speedup | Score Match |
|---------------|----------|----------|---------|-------------|
| Laplacian variance | 20.95 | 3.15 | 6.7x | Exact (5625.8) |
| Tenengrad (Sobel energy) | 41.23 | 4.66 | 8.8x | Exact (14276.2) |

### GPU Laplacian — what works and what doesn't
| Approach | Works? | Score | Notes |
|----------|--------|-------|-------|
| `createLaplacianFilter(CV_32FC1, CV_32FC1)` | **Yes** | 5625.8 (exact) | Must convert input to float32 first |
| `createLaplacianFilter(CV_8UC1, CV_32F)` | No | — | Assertion: `dstType == srcType` |
| `createLaplacianFilter(CV_8UC1, CV_8UC1)` | Runs | 2111.0 (wrong) | Negatives clamped to 0 |
| `createLinearFilter(CV_8UC1, CV_32FC1, kernel)` | No | — | Same `dstType == srcType` restriction |

### Projected pipeline (method 6, all-GPU Tenengrad)
| Operation | Time (ms) |
|-----------|-----------|
| Decode (GPU NVDEC) | 4.13 |
| Resize (GPU) | 1.37 |
| CvtColor (GPU) | 0.59 |
| Convert uint16→uint8 (GPU) | ~0.5 |
| Tenengrad scoring (GPU) | 4.66 |
| **Total** | **~11.2** |
| **Projected FPS** | **~89** |
| **Projected 16929 frames** | **~190s** |

---

## Key Findings

### cudacodec returns uint16 BGRA
`cv2.cudacodec.VideoReader.nextFrame()` returns frames as **uint16 BGRA (4 channels)**.
- Must right-shift by 8 (`>> 8`) and cast to uint8 for consistent scoring with CPU path
- Must convert BGRA→BGR before JPEG imwrite (JPEG has no alpha channel)
- The "Unsupported depth" warning from method 4's first run was this issue

### GpuMat API notes
- `gpu_frame.size()` returns `(width, height)` — opposite of numpy's `(height, width)`
- `gpu_frame.clone()` is necessary to preserve a frame across `nextFrame()` calls
- `gpu_frame.channels()` returns 4 for cudacodec output (BGRA)
- `gpu_frame.type()` returns OpenCV type enum (2 = CV_16UC1, 0 = CV_8UC1, etc.)
- `convertTo(dst_type, dst_gpumat, alpha=)` — alpha scales values during conversion

### ffmpeg NVDEC pipe is unreliable for frame-accurate work
ffmpeg's NVDEC decode path can emit different frame counts than OpenCV's VideoCapture
for the same HEVC file (17,077 vs 16,929). This makes pipe-based approaches unsuitable
when frame-number accuracy matters (sharpest-frame selection, scene detection).

### Seeking: firstFrameIdx vs set(POS_FRAMES)
Both tested against CPU `VideoCapture` ground truth at frame 500 on york fisheye HEVC:
- `VideoReaderInitParams(firstFrameIdx=500)`: mean pixel diff 19.3 from CPU (closer)
- `reader.set(CAP_PROP_POS_FRAMES, 500)`: mean pixel diff 33.4 from CPU (farther)

Neither is pixel-identical (NVDEC decodes differently from CPU), but `firstFrameIdx` is
more accurate. Prefer `firstFrameIdx` for start_sec seeking. Note: `createVideoReader`
requires keyword form `params=params` — positional fails with argument parsing error.

### Scene detection at different resolutions (GPU path)
Benchmarked on york fisheye HEVC, 50 iterations per approach:
| Approach | Cost per frame | PCIe transfer per frame |
|----------|---------------|------------------------|
| Download 480px BGR, CPU histogram (2D H+S) | 2.16 ms | ~700 KB |
| Download 1920px BGR, CPU histogram (2D H+S) | 11.07 ms | ~10.8 MB |
| GPU histogram (H channel only), download histogram | 1.01 ms | ~1 KB |
| GPU grayscale absdiff, no download | 1.60 ms | 8 bytes |

Over 16,929 frames: 480px adds ~37s, 1920px adds ~187s. GPU histogram adds ~17s.
The CPU path uses 1920px (reuses the scoring downscale). The GPU path uses 480px for v1 —
HSV histograms use the same bin structure at either resolution (50 H × 60 S), and the
distribution comparison is expected to be robust to downsampling. However, edge cases
(hard cuts, exposure shifts, small high-saturation objects) could shift threshold
crossings. Parity check required on footage with actual scene changes before shipping.

### DLL setup for cudacodec on Windows
- CUDA 13.1 runtime installed side-by-side with 12.9 (only runtime component needed)
- cuDNN 9.x DLLs must be in the CUDA bin directory (`bin\x64\`) — pip installs them
  to site-packages where Windows DLL loader can't find them
- System PATH must include `CUDA\v13.1\bin\x64` — terminals opened before PATH update
  won't see the DLLs (restart required)

---

## Setup Reference

### Install OpenCV with CUDA
```powershell
# Wheel from cudawarped (cp37-abi3 = works with Python 3.7+)
C:\Python314\python.exe -m pip uninstall opencv-python opencv-contrib-python -y
C:\Python314\python.exe -m pip install https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.13.0.90/opencv_contrib_python-4.13.0.90-cp37-abi3-win_amd64.whl

# cuDNN (needed at runtime)
C:\Python314\python.exe -m pip install nvidia-cudnn-cu12

# Copy cuDNN DLLs to CUDA bin (one-time)
Copy-Item "$env:APPDATA\Python\Python314\site-packages\nvidia\cudnn\bin\*.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\"
```

### Verify
```powershell
C:\Python314\python.exe -c "import cv2; print(cv2.__version__); print('cudacodec:', hasattr(cv2, 'cudacodec'))"
# Expected: 4.13.0 / cudacodec: True
```
