# FrameDistill — Technical Analysis

> Scraped and reverse-engineered from https://framedistill.vk.land/ on 2026-03-29.
> Author: VoxelKei ([X](https://x.com/VoxelKei), [YouTube](https://www.youtube.com/@VoxelKei))

## What It Does

Client-side web app that extracts optimized video frames for 3DGS & photogrammetry.
Upload a video (or image set), interactively filter out blurry and near-duplicate frames
using histogram sliders, then export the selection as PNG/JPEG (ZIP or direct-to-folder).

All processing runs in the browser — no server, no uploads. Requires Chrome/Edge with
WebCodecs and File System Access API support.

---

## Architecture

```
Main Thread (vanilla JS, Vite-bundled)
├── UI: cyberpunk HUD aesthetic, vanilla HTML/CSS, no framework
├── Video upload → File blob
├── Spawns Web Worker for decode + analysis
├── Receives per-frame: {index, timestamp, blurScore, phash, thumbnail}
├── Interactive filtering via two sliders (blur threshold, similarity threshold)
├── Re-spawns worker for export of selected frames
└── ZIP via fflate, or folder via File System Access API

Web Worker (decode-worker-BrStp4SR.js)
├── MP4 demux via mp4box.js
├── Video decode via WebCodecs VideoDecoder
├── Per-frame analysis on OffscreenCanvas:
│   ├── RGBA → grayscale conversion (BT.601)
│   ├── Blur scoring (center-weighted Laplacian variance)
│   └── Perceptual hashing (aHash, 64-bit)
├── Thumbnail generation (ImageBitmap transfer)
└── Export: re-decode selected frames → PNG/JPEG blobs
```

### Dependencies

| Library | Purpose | License |
|---------|---------|---------|
| [mp4box.js](https://github.com/gpac/mp4box.js) | MP4 container demuxing | BSD-3-Clause |
| [fflate](https://github.com/101arrowz/fflate) | ZIP compression | MIT |
| [Vite](https://vitejs.dev) | Build tool | MIT |
| [Vercel Analytics](https://vercel.com/analytics) | Web analytics | MIT |

Browser APIs used: WebCodecs (VideoDecoder), File System Access (showDirectoryPicker),
OffscreenCanvas, Web Workers, ImageBitmap transfers.

---

## Algorithm 1: Blur Detection — Center-Weighted Laplacian Variance

### Overview

Measures image sharpness using the variance of the Laplacian operator, with a center-bias
weight that downweights edge pixels. Higher score = sharper image.

### Grayscale Conversion (`eo`)

Standard BT.601 luminance from RGBA ImageData:

```
Y = 0.299 * R + 0.587 * G + 0.114 * B
```

Stored as a flat `Uint8Array` of width*height grayscale values.

### Laplacian Kernel

4-connected cross pattern (not full 8-connected 3x3):

```
     [0,  1, 0]
     [1, -4, 1]
     [0,  1, 0]
```

For pixel at position (x, y) in the grayscale buffer:

```
L(x,y) = pixel[y-1][x] + pixel[y][x-1] + pixel[y][x+1] + pixel[y+1][x] - 4 * pixel[y][x]
```

Only interior pixels (1..width-2, 1..height-2) are evaluated.

### Subsampling Strategy

For performance (this runs per-frame in a Web Worker):

```
interior_pixels = (width - 2) * (height - 2)
if interior_pixels > 1000:
    stride = floor(sqrt(interior_pixels / 1000))
else:
    stride = 1
```

This caps the sample count at ~1000 regardless of resolution.

### Center Weighting

Each sampled pixel gets a weight based on distance from the image center:

```
cx = (width - 1) / 2       # center x
cy = (height - 1) / 2      # center y
dx = |x - cx| / cx          # normalized x distance [0, 1]
dy = |y - cy| / cy          # normalized y distance [0, 1]
weight = 2 - max(dx, dy)    # range [1, 2]: center=2, corner=1
```

This means center pixels contribute ~2x more than corner pixels. The weighting uses
Chebyshev distance (max of dx, dy), not Euclidean — so it forms a diamond/square falloff
pattern rather than circular.

### Variance Calculation

Weighted variance of Laplacian values:

```
weighted_sum = 0
weighted_sum_sq = 0
total_weight = 0

for each sampled pixel:
    L = laplacian(x, y)
    w = center_weight(x, y)
    total_weight += w
    weighted_sum += w * L
    weighted_sum_sq += w * L * L

mean = weighted_sum / total_weight
blurScore = weighted_sum_sq / total_weight - mean * mean
```

This is the standard one-pass variance formula: `E[X^2] - E[X]^2`.

### Complete Deminified Implementation

```javascript
// Grayscale conversion (BT.601)
function toGrayscale(imageData, grayBuffer) {
    const rgba = imageData.data;
    for (let i = 0; i < grayBuffer.length; i++) {
        const j = i * 4;
        grayBuffer[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
    }
}

// Plain Laplacian variance (no weighting, used when stride=1)
function laplacianVariance(gray, width, height, stride = 1) {
    let sum = 0, sumSq = 0, count = 0;
    for (let y = 1; y < height - 1; y += stride) {
        for (let x = 1; x < width - 1; x += stride) {
            const idx = y * width + x;
            const L = gray[idx - width] + gray[idx - 1] + gray[idx + 1] + gray[idx + width] - 4 * gray[idx];
            sum += L;
            sumSq += L * L;
            count++;
        }
    }
    if (count === 0) return 0;
    const mean = sum / count;
    return sumSq / count - mean * mean;
}

// Center-weighted Laplacian variance (used when stride > 1)
function centerWeightedBlurScore(gray, width, height) {
    const interiorPixels = (width - 2) * (height - 2);
    const stride = interiorPixels > 1000 ? Math.floor(Math.sqrt(interiorPixels / 1000)) : 1;

    if (stride === 1) return laplacianVariance(gray, width, height, 1);

    const cx = (width - 1) / 2;
    const cy = (height - 1) / 2;
    const invCx = 1 / cx;
    const invCy = 1 / cy;

    let totalWeight = 0, weightedSum = 0, weightedSumSq = 0;

    for (let y = 1; y < height - 1; y += stride) {
        const dy = Math.abs(y - cy) * invCy;
        for (let x = 1; x < width - 1; x += stride) {
            const dx = Math.abs(x - cx) * invCx;
            const weight = 2 - Math.max(dx, dy);  // [1, 2]
            const idx = y * width + x;
            const L = gray[idx - width] + gray[idx - 1] + gray[idx + 1] + gray[idx + width] - 4 * gray[idx];

            totalWeight += weight;
            weightedSum += weight * L;
            weightedSumSq += weight * L * L;
        }
    }

    if (totalWeight === 0) return 0;
    const mean = weightedSum / totalWeight;
    return weightedSumSq / totalWeight - mean * mean;
}
```

### Design Notes

- Center weighting is valuable for wide-angle and 360 imagery where lens distortion
  softens edges. Prevents corner blur from dragging down the sharpness score.
- The ~1000 sample cap is a JS/Worker performance optimization. In Python with numpy,
  vectorized Laplacian (via cv2.Laplacian or ndimage) doesn't need subsampling.
- The 4-connected kernel is cheaper than 8-connected and sufficient for blur detection.
- Score is unbounded (not normalized). Higher = sharper. Default threshold = 20.

---

## Algorithm 2: Similarity Detection — Average Hash (aHash)

### Overview

Computes a 64-bit perceptual hash per frame. Similar-looking frames produce similar hashes.
Hamming distance between hashes measures visual difference.

### Hash Computation (`io`)

```
1. Divide grayscale image into 8x8 grid
2. For each cell: compute mean pixel value (area-average downscale)
3. Compute global mean of all 64 cell values
4. For each cell: bit = 1 if cell_mean > global_mean, else 0
5. Pack 64 bits into Uint8Array(8), MSB-first per byte
```

### Deminified Implementation

```javascript
function computeAHash(gray, width, height) {
    const cells = new Float64Array(64);
    const cellW = width / 8;
    const cellH = height / 8;

    // Area-average each of the 8x8 cells
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            let sum = 0, count = 0;
            const y0 = Math.floor(row * cellH);
            const y1 = Math.floor((row + 1) * cellH);
            const x0 = Math.floor(col * cellW);
            const x1 = Math.floor((col + 1) * cellW);
            for (let y = y0; y < y1; y++) {
                for (let x = x0; x < x1; x++) {
                    sum += gray[y * width + x];
                    count++;
                }
            }
            cells[row * 8 + col] = count > 0 ? sum / count : 0;
        }
    }

    // Global mean
    let globalMean = 0;
    for (let i = 0; i < 64; i++) globalMean += cells[i];
    globalMean /= 64;

    // Pack into 8 bytes, MSB-first
    const hash = new Uint8Array(8);
    for (let i = 0; i < 64; i++) {
        if (cells[i] > globalMean) {
            hash[Math.floor(i / 8)] |= 1 << (7 - i % 8);
        }
    }
    return hash;
}
```

### Hamming Distance (`Xo`)

Counts differing bits between two 8-byte hashes:

```javascript
function hammingDistance(hashA, hashB) {
    let dist = 0;
    for (let i = 0; i < 8; i++) {
        let xor = hashA[i] ^ hashB[i];
        while (xor) {
            dist += xor & 1;
            xor >>= 1;
        }
    }
    return dist;  // 0 = identical, 64 = maximally different
}
```

### Design Notes

- aHash is the simplest perceptual hash. Less discriminating than dHash (gradient-based)
  or pHash (DCT-based), but fast and adequate for sequential video dedup.
- The 8x8 grid acts as a very aggressive low-pass filter, making the hash invariant to
  small shifts, minor brightness changes, and noise.
- Default similarity threshold = 8 (reject if Hamming distance < 8, i.e., < 12.5% bits differ).
- Max similarity slider value is 32 (50% of bits).

---

## Frame Selection Logic (`wc`)

Two-pass sequential filter:

```javascript
function selectFrames(allFrames, { blurThreshold, similarityThreshold }) {
    // Pass 1: Keep only frames at or above blur threshold
    const sharp = allFrames.filter(f => f.blurScore >= blurThreshold);

    // Pass 2: Sequential dedup against last-kept frame
    const selected = [];
    let lastKeptHash = null;

    for (const frame of sharp) {
        if (lastKeptHash !== null
            && similarityThreshold > 0
            && hammingDistance(lastKeptHash, frame.phash) < similarityThreshold) {
            continue;  // too similar to last kept frame, skip
        }
        selected.push(frame.index);
        lastKeptHash = frame.phash;
    }

    return { selected, total: allFrames.length };
}
```

### Design Notes

- **Blur first, then dedup** — correct ordering; no point deduplicating blurry rejects.
- **Sequential comparison only** — compares each frame to the LAST KEPT frame, not all
  kept frames. This is O(n) and intentionally allows revisited viewpoints (good for
  photogrammetry loop closure).
- Both thresholds are interactive (slider + histogram visualization). The user sees
  real-time count updates as they drag.

---

## Video Decode Pipeline

### Demuxing (mp4box.js)

- Parses MP4/MOV containers, extracts H.264/H.265 encoded chunks
- Keyframe-only mode: filters `sample.is_sync` for faster analysis
- Caches keyframe offsets per file (by name+size) to avoid re-scanning
- Streams in 1MB chunks (4MB for files > 512MB)

### Decoding (WebCodecs)

- `VideoDecoder` API decodes `EncodedVideoChunk` → `VideoFrame`
- Frames drawn to `OffscreenCanvas` → `getImageData()` for pixel access
- Rotation handling: 90/180/270 degree transforms via canvas
- Queue throttle: pauses at 30 queued frames to prevent memory blowout

### Analysis Flow

```
For each decoded frame:
    1. Draw to OffscreenCanvas
    2. getImageData() → RGBA pixels
    3. RGBA → grayscale (Uint8Array)
    4. blurScore = centerWeightedBlurScore(gray, w, h)
    5. phash = computeAHash(gray, w, h)
    6. thumbnail = canvas.transferToImageBitmap()
    7. postMessage({index, timestamp, blurScore, phash, thumbnail})
```

### Export Flow

- Re-decodes only selected frame indices from original file
- Supports resolution scaling (custom long edge, aspect-preserved)
- Handles EXIF rotation
- Two output modes:
  - ZIP download (fflate in-memory compression)
  - Folder export (File System Access API, `showDirectoryPicker`)
- JPEG quality configurable (default 90)

### Image Set Mode

Also accepts pre-existing images (not just video):
- Creates `ImageBitmap` from each file
- Draws to OffscreenCanvas, runs same blur+phash analysis
- Export re-reads originals, applies resolution scaling

---

## UI / UX Design

### Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Blur Threshold | 20 | dynamic | Min blur score to keep (higher = stricter) |
| Similarity Threshold | 8 | 0-32 | Min Hamming distance to keep (higher = more dedup) |
| Export Format | JPEG | PNG/JPEG | Output format |
| JPEG Quality | 90 | 1-100 | Compression quality |
| Resolution | Original | Original/Custom | Custom = max long edge in px |
| Analysis Mode | All frames | All/Keyframes only | Keyframes = faster |

### Interactive Histograms

Both blur and similarity have canvas-drawn histograms showing the distribution of scores.
A vertical line shows the current threshold. As you drag the slider, the histogram updates
and the frame count changes in real-time.

### Preview Grid

- Virtual scrolling (position: absolute, calculated from row/column)
- Three thumbnail sizes: S, M, L
- Color-coded by blur score (green=sharp → red=blurry)
- Click to open full-resolution modal
- Deselected (filtered-out) frames are visually dimmed

### Design Aesthetic

Cyberpunk/HUD theme. Fonts: Orbitron (headings), Share Tech Mono (mono), Jura (UI),
Noto Sans JP (Japanese). Corner decorations, scanline overlays, spectrum glow effects.
Bilingual (English/Japanese toggle).

---

## Source Files

| Asset | URL | Size |
|-------|-----|------|
| HTML | https://framedistill.vk.land/ | ~16KB |
| JS bundle | /assets/index-D2jNq5uG.js | ~221KB |
| Worker | /assets/decode-worker-BrStp4SR.js | ~190KB |
| CSS | /assets/index-DMm-POd8.css | ~25KB |
| OG image | /og-image.png | — |
| Favicon | /favicon.svg | — |

Note: JS bundle includes mp4box.js (~120KB) and fflate (~30KB) inlined. The worker
includes a copy of mp4box.js for demuxing in the worker thread.

---

## Relevance to Reconstruction Zone

### Center-Weighted Blur Detection

prep360's `BlurFilter` uses plain Laplacian variance (via `cv2.Laplacian`). Adding
center-weighting would improve accuracy for:
- 360 equirectangular frames (poles are distorted)
- Wide-angle/fisheye frames (edge softness from lens)
- Reframed perspective crops (edges may be interpolation-softened)

### Perceptual Hash Dedup

prep360 currently selects frames by time interval only. Adding similarity-based dedup
would handle:
- Slow panning (many near-identical frames despite time gap)
- Stationary segments (camera not moving)
- Redundant coverage in dense extraction settings

Both features would integrate at the `BlurFilter` / post-extraction stage in prep360.
