# Coverage Tab

The Coverage tab analyzes spatial gaps in existing 3D reconstructions and extracts targeted "bridge" frames from source video to fill them. It sits at the end of the tab bar because it operates on reconstruction output — you use it *after* your first COLMAP or Metashape run, not before.

## When to use this tab

You've extracted frames, run them through COLMAP or Metashape, and the reconstruction has problems:
- Disconnected components (two chunks that should be one model)
- Sparse regions (thin corridors where cameras are too far apart)
- Failed images (frames that didn't align at all)

The Coverage tab finds these gaps, maps them back to timestamps in your source video, and extracts additional frames from exactly those moments.

## Prerequisites

Before using Coverage, you need:
1. **A video loaded on the Extract tab** — the source video your frames came from
2. **An output folder set on the Extract tab** — bridge frames go to `{output}/bridges/`
3. **A completed reconstruction** — COLMAP output, Metashape XML, or XMP sidecars

The Coverage tab reads video and output paths from the Extract tab directly. You don't re-enter them.

---

## Gap Analysis

### Supported sources

The gap detector reads camera positions from three formats. All three feed into the same analysis pipeline — the only difference is how positions are parsed.

| Source | Input | What it reads |
|--------|-------|---------------|
| **COLMAP directory** | `sparse/0/` or similar | `images.txt` — quaternion + translation per image, converted to world-space camera centers via `-R^T @ t` |
| **Metashape XML** | `cameras.xml` | Camera transforms (4x4 camera-to-chunk matrices), scaled by chunk transform to world coordinates |
| **XMP sidecars** | Directory of `.xmp` files | `xcr:Position` tags — camera centers already in world space (RealityScan/RealityCapture export) |

### Analysis pipeline

All three sources produce the same thing: a list of `(camera_name, 3D_position, timestamp?)` entries. From there:

```
Camera positions
       │
       ▼
┌─────────────────────────────────────────────────┐
│  1. DBSCAN CLUSTERING                           │
│                                                  │
│  Groups cameras into spatial components.         │
│  eps = max distance between neighbors (def 5.0)  │
│  min_samples = 3 cameras to form a cluster       │
│                                                  │
│  Falls back to BFS connected-components          │
│  if scikit-learn is not installed.               │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
┌──────────────────┐    ┌──────────────────────────┐
│ 2a. INTER-COMP   │    │ 2b. SPARSE REGIONS       │
│     GAPS          │    │     (largest component)   │
│                   │    │                           │
│ For each pair of  │    │ Sort cameras by trajectory │
│ components:       │    │ (timestamps or PCA axis)   │
│                   │    │                           │
│ • Min pairwise    │    │ Consecutive distances >    │
│   distance        │    │ median + thresh × MAD      │
│ • K nearest       │    │ → "sparse" gap             │
│   cameras per     │    │                           │
│   side            │    │ severity = dist / median   │
│ • gap_type =      │    │                           │
│   "disconnected"  │    │                           │
│                   │    │                           │
│ severity =        │    │                           │
│ dist × total_cams │    │                           │
└────────┬─────────┘    └────────────┬─────────────┘
         │                           │
         └─────────┬─────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│  3. FAILED IMAGE DETECTION (optional)            │
│                                                   │
│  If "Images" directory provided:                  │
│  diff source images against aligned camera names  │
│  → list of frames not in the reconstruction       │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   GapReport     │
              │                 │
              │ • gaps[]        │
              │ • failed_images │
              │ • camera_pos[]  │
              │ • components    │
              └─────────────────┘
```

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| **Cluster eps** | 5.0 | DBSCAN radius. Larger = more lenient clustering (fewer disconnected gaps). Units match your reconstruction's world scale. |
| **Sparse threshold** | 2.0x | Multiplier on median absolute deviation. Lower = more sensitive to thin spots. |

### Gap types

**Disconnected** — Two camera clusters that should be connected but aren't. This is the most severe type. The reconstruction literally has a hole where no geometry exists between two groups.

**Sparse** — Cameras within a single component that are unusually far apart relative to the rest of the trajectory. The reconstruction might still connect here, but the geometry will be weak or distorted.

### Timestamp extraction

The bridge extractor needs timestamps to know *where* in the source video each gap corresponds. Timestamps are extracted from filenames using these patterns:

| Pattern | Example | Timestamp |
|---------|---------|-----------|
| `t{seconds}s` | `frame_00123_t45.67s.jpg` | 45.67s |
| Sequential number | `00042.jpg` | 84.0s (index × 2.0) |

If filenames don't match either pattern, gaps are detected spatially but **cannot be bridged** — the extractor skips them because it can't map back to the video.

---

## Bridge Extraction

### Workflow

Bridge extraction is a feedback loop. The full cycle:

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│ Extract tab  │────▶│ COLMAP /     │────▶│ Coverage tab    │
│ pull frames  │     │ Metashape    │     │ find gaps       │
└─────────────┘     └──────────────┘     └───────┬────────┘
       ▲                                          │
       │                                          ▼
       │                                 ┌────────────────┐
       │                                 │ Bridge extract  │
       │                                 │ fill the gaps   │
       └─────────────────────────────────┴────────────────┘
                    merge into dataset,
                    re-run reconstruction
```

### What happens when you click "Extract Bridges"

```
┌─ PLAN ────────────────────────────────────────────────────┐
│                                                            │
│  For each gap with estimated_timestamps:                   │
│                                                            │
│  • "disconnected" gaps → target 8 bridge frames            │
│  • "sparse" gaps       → target 3 bridge frames            │
│  • Add ±2s padding around each gap's time window           │
│  • Clamp to video bounds                                   │
│                                                            │
│  Gaps WITHOUT timestamps are SKIPPED.                      │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌─ EXTRACT (per gap) ───────────────────────────────────────┐
│                                                            │
│  MotionSelector samples the video at 5 fps within          │
│  the gap's time window, then picks frames that are:        │
│                                                            │
│  • Sharp enough (Laplacian variance > 50.0)                │
│  • Spread apart (optical flow > 8.0 between selections)    │
│                                                            │
│  Output: {output}/bridges/gap_00/, gap_01/, etc.           │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌─ OPTIONAL: REFRAME ───────────────────────────────────────┐
│                                                            │
│  If "Reframe fisheye → perspective" is checked:            │
│                                                            │
│  • Reads calibration + preset from the Extract tab's       │
│    360 Video section                                       │
│  • Each bridge frame → N perspective crops                 │
│  • Saved to gap_XX/reframed/                               │
│  • Crops with <50% content coverage are skipped            │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌─ REPORT ──────────────────────────────────────────────────┐
│                                                            │
│  bridge_result.json saved to {output}/bridges/             │
│                                                            │
│  Per frame: path, timestamp, sharpness, flow, gap_index    │
│  Per gap: frames extracted vs. target count                │
│  Summary logged to the console                             │
└───────────────────────────────────────────────────────────┘
```

### Bridge extractor parameters

These are hardcoded in `BridgeExtractor.__init__()` — not exposed in the GUI:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `sample_fps` | 5.0 | How densely to sample video in gap windows |
| `min_sharpness` | 50.0 | Laplacian blur rejection threshold |
| `target_flow` | 8.0 | Optical flow between selected frames (lower than normal extraction for denser coverage) |
| `padding_sec` | 2.0 | Extra seconds before/after each gap window |
| `frames_per_disconnected` | 8 | Target frames for severe gaps |
| `frames_per_sparse` | 3 | Target frames for mild gaps |

### Output structure

```
{extract_output}/
├── video_name/          ← normal extracted frames
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── bridges/             ← bridge frames (from Coverage tab)
    ├── gap_00/
    │   ├── bridge_001_t23.4s.jpg
    │   ├── bridge_002_t25.1s.jpg
    │   └── reframed/    ← (if fisheye reframe enabled)
    │       ├── bridge_001_t23.4s_front_h0_y0.jpg
    │       └── ...
    ├── gap_01/
    │   └── ...
    ├── bridge_report.json
    └── bridge_result.json
```

---

## Typical workflow: perspective video

```
1. Extract tab    → Load video, set output, extract frames at 2s interval
2. (External)     → Run COLMAP on the extracted frames
3. Coverage tab   → Select "COLMAP directory", point to sparse/0/
4. Coverage tab   → Optionally set "Images" dir to detect failed frames
5. Coverage tab   → Click "Analyze Gaps"
6. Coverage tab   → Review gap summary and details
7. Coverage tab   → Click "Extract Bridges" (reads video from Extract tab)
8. (External)     → Copy bridge frames into your COLMAP images/ directory
9. (External)     → Re-run COLMAP with the augmented dataset
```

## Typical workflow: 360 fisheye video

```
1. Extract tab    → 360 Video section: load .osv, split lenses, reframe
2. (External)     → Run COLMAP on the perspective crops
3. Coverage tab   → Select "COLMAP directory", point to sparse/0/
4. Coverage tab   → Click "Analyze Gaps"
5. Coverage tab   → Check "Reframe fisheye → perspective"
6. Coverage tab   → Click "Extract Bridges"
                    (reads calibration + preset from Extract tab's 360 section)
7. (External)     → Merge gap_XX/reframed/ into your dataset
8. (External)     → Re-run COLMAP
```

## Typical workflow: Metashape

```
1. Extract tab    → Load video, extract frames
2. (External)     → Import into Metashape, align cameras
3. (External)     → File → Export → Export Cameras (cameras.xml)
4. Coverage tab   → Select "Metashape XML", point to cameras.xml
5.                → Same as steps 5-9 above
```
