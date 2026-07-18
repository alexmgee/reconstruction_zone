# Align Tab

The Align tab runs sparse 3D reconstruction on your images using COLMAP 4.1 or the
legacy SphereSfM fork. It takes a folder of images (and optional masks from the
Mask/Review workflow), runs feature extraction, matching, and reconstruction, and
produces a sparse point cloud with camera poses. This is the step that turns 2D
photos into 3D geometry — the foundation for dense reconstruction and Gaussian
Splatting.

Highlights of the current stack:

- **Caspar GPU bundle adjustment** — COLMAP 4.1's GPU solver, in a measured-optimal
  hybrid configuration (see *Bundle adjustment backends* below).
- **Native 360° reconstruction** — equirectangular images reconstruct directly via
  the `EQUIRECTANGULAR` camera model; no fork, no reframing required.
- **Backend qualification** — the app probes its reconstruction backends in a
  separate process before using them, so a broken installation is excluded with a
  clear message instead of crashing the app or silently degrading your run.
- **Capability refusal** — a run that asks for something its backend cannot do
  (e.g. a 360° camera model on an old COLMAP binary) refuses up front, naming the
  problem, instead of failing minutes later or producing a corrupt result.

## Layout

| Column | Contents |
|--------|----------|
| **Left** (scrollable) | Inputs, Engine toggle, Engine Settings, Rig Configuration, Extract, Match, Reconstruct, Advanced, Actions, Run Status, Log |
| **Right** (detail panel) | Header with stage indicators, point cloud viewer, viewer controls, alignment summary |

---

## Inputs

Three path fields at the top:

- **Images** — Directory of images to reconstruct. Auto-detects the workspace when
  you tab out of this field.
- **Masks** — Optional directory of masks. When provided, COLMAP ignores masked
  pixels during feature extraction. Use the masks produced by the Mask → Review
  workflow.
- **Workspace** — Root directory for alignment outputs. Each run gets its own
  timestamped subdirectory, so previous runs are never overwritten. Every run also
  records its full provenance (engine, backend, binary hash, camera-model
  namespace) in `run_info.json`.

## Engine

Radio toggle between two reconstruction engines:

| Engine | Camera models | Best for |
|--------|--------------|----------|
| **COLMAP** | PINHOLE, OPENCV, fisheye variants, **EQUIRECTANGULAR**, **EUCM**, … | Everything — perspective, reframed 360°, fisheye, and native 360° |
| **SphereSfM (legacy)** | SPHERE (fixed) | Kept for older projects. New 360° work should use COLMAP's EQUIRECTANGULAR model instead. |

Switching engines swaps the settings panels. SphereSfM locks the camera model to
SPHERE, disables rig configuration and the bundle-adjustment controls, and always
runs its external binary.

### Execution backend

The COLMAP engine can run two ways, selectable under **Advanced → Tool Setup →
Backend**:

- **auto** (default) — use the in-process pycolmap library when its wheel passes
  qualification, otherwise fall back to the external CLI binary.
- **pycolmap** — require the in-process library. Runs refuse to start if it fails
  qualification (no silent downgrade).
- **cli** — always run the external binary configured in Tool Setup.

Backends are qualified once per app session in a fresh process. A backend that
fails qualification stays excluded until the app restarts; the log tells you what
was excluded and why. A CPU-only pycolmap wheel is deliberately excluded — this
app expects the CUDA build (see *Installation notes* at the end).

## Engine Settings

**COLMAP:**

- **Camera model** — how lens geometry is modeled:

  | Model | Use for |
  |-------|---------|
  | `PINHOLE` / `SIMPLE_PINHOLE` | Reframed perspective views, undistorted images |
  | `SIMPLE_RADIAL` / `RADIAL` | Typical photos from real lenses (default choice) |
  | `OPENCV` / `FULL_OPENCV` | Wide-angle captures where distortion should be estimated |
  | `OPENCV_FISHEYE` | Raw fisheye lenses |
  | `EQUIRECTANGULAR` | **Native 360° panoramas** (2:1 equirect frames) — requires a COLMAP 4.1+ backend |
  | `EUCM` | Wide-angle fisheye / catadioptric systems — requires a COLMAP 4.1+ backend |

  If the selected backend is too old for a model, the run refuses to start with a
  message naming the incapable component.

- **Single camera** — when checked, all images share one calibration. Uncheck for
  mixed cameras, focal lengths, or images with varying dimensions.

**SphereSfM (legacy):** camera model is always SPHERE; pose path is configurable
under Advanced → Camera And Reader.

## Rig Configuration

Collapsible, collapsed by default. For multi-camera rigs where images have known
relative poses (e.g. reframed 360° views extracted from the same frame). Select a
built-in preset or point to a custom `rig_config.json`. Disabled under SphereSfM.

## Extract

Feature extraction — how keypoints are detected in each image.

- **Feature type** — `SIFT` (default), or the learned `ALIKED_N16ROT` /
  `ALIKED_N32` detectors (pair them with an ALIKED matcher type under Match).
- **Max features** — target keypoints per image (default 8192). Note: COLMAP
  treats this as a *soft* cap — the final per-image count can slightly exceed it
  because keypoints get multiple orientations. Raising/lowering it still works
  exactly as expected; identical behavior on both backends.
- **Max image size** — images larger than this (longest side, pixels) are
  downscaled for extraction. Lower it if you run out of GPU memory; consider
  raising it for very high-resolution 360° frames.

Extraction and matching run on the GPU automatically when CUDA is available.

## Match

Feature matching — how correspondences are found between image pairs.

- **Strategy**
  - **Exhaustive** — every pair. Best quality, O(n²) cost. Practical up to ~500
    images. Required for reframed 360° view sets (views of the same frame face
    different directions, so sequence order means little).
  - **Sequential** — adjacent images (ordered captures / video frames). Fast.
  - **Spatial** — pairs by GPS proximity (geo-tagged images).
  - **Vocab tree** — visual similarity; scales to thousands of images. Needs a
    vocabulary tree file.
- **Max matches** — verified matches per pair (default 32768).
- **Guided matching** — re-match using estimated geometry. Slower, more inliers.

Strategy-specific fields (GPS toggle / max distance / vocab tree path) appear when
relevant.

## Reconstruct

Controls the Structure-from-Motion mapper.

- **Mapper** — three modes:
  - **`global`** (default) — solves all camera poses simultaneously (GLOMAP-style).
    Fast on well-connected sets, no drift; can fail on weakly-connected ones.
  - **`incremental`** — registers one image at a time. Most robust; required for
    Caspar bundle adjustment.
  - **`pose_prior`** — incremental, constrained by GPS positions from EXIF.
    Reveals extra fields: **Position std (X/Y/Z)** in meters (≈1.0 for consumer
    GPS, 0.01–0.1 for RTK/PPK) and **Override prior covariance**.
- **Min inliers** — minimum correspondences to register an image (default 15).
- **BA backend / GPU BA** — see the next section.
- **Refine focal / principal / extra** — what bundle adjustment may optimize.
  Locked automatically under SphereSfM; harmless no-ops for `EQUIRECTANGULAR`
  (the model has no such parameters to refine).
- **Snapshots** — optionally save intermediate reconstructions during long runs.

## Bundle adjustment backends

Bundle adjustment (BA) is the heavy optimization inside reconstruction. The
**BA backend** dropdown and **GPU BA** checkbox control it:

| Setting | What happens |
|---------|--------------|
| **AUTO** (default) | Caspar when eligible (see below), otherwise Ceres. The log shows what was observed and chosen. |
| **CERES** | The classic solver, always compatible. |
| **CASPAR** | COLMAP 4.1's GPU solver. Eligible only for `PINHOLE`/`SIMPLE_RADIAL` cameras with the incremental mapper. |

**When Caspar runs, it uses a measured hybrid:** the large *global* adjustments go
to Caspar on the GPU while the many small *local* adjustments stay on Ceres — on a
430-image benchmark this was 2.1× faster than running Caspar everywhere and ~25%
faster than Ceres, with identical quality.

**Guardrails you will see:**

- Explicitly selecting CASPAR with an ineligible camera model or mapper **aborts
  the reconstruction stage immediately** with a message naming the offending
  models. This is deliberate: Caspar would otherwise silently ignore those
  cameras' observations and quietly degrade your model.
- AUTO never aborts — it logs the observed camera models and falls back to Ceres.

**Honest limitations:**

- Ceres GPU acceleration (the cuDSS sparse solver) works on both backends, but
  COLMAP deliberately routes small problems (under ~1000 images by default) to
  its CPU solvers, because they're faster at that scale — so "GPU BA" with
  Ceres often runs on CPU for typical datasets. Caspar always runs on the GPU.
- Global-mapper GPU BA is cli-only.

## Advanced

### Tool Setup

- **Binary** — the COLMAP / SphereSfM executable. The saved preference wins over
  auto-discovery; use "Change" to override.
- **Backend** — the execution-backend selector described under *Engine* above.
- **GPU index** — which GPU to use (default 0; −1 for CPU).

### Camera And Reader

- **Camera params** — fixed intrinsics string (skips calibration when set).
- **Pose path / camera mask path** — pose-aware and SphereSfM workflows.

### Per-Stage CLI Args

Raw argument passthrough appended to each stage's command — for edge cases the UI
doesn't cover.

## Actions

| Button | Action |
|--------|--------|
| **Align** | Run extract → match → reconstruct |
| **Next Step** | Run only the next incomplete stage |
| **Cancel** | Stop the current run (current stage finishes first) |
| **Load Model** | Load an existing sparse model into the viewer |

## Run Status & Detail Panel

Stage indicators (gray → amber → green/orange/red) track progress; after
reconstruction a quality badge appears with point count, registered images, and
mean reprojection error.

The **point cloud viewer** shows the sparse points colored by appearance, with
camera markers: perspective cameras get frustum wireframes; 360° cameras
(`SPHERE`, `EQUIRECTANGULAR`) get sphere markers; models without implemented
marker geometry (e.g. `EUCM`) render their points with markers skipped (noted in
the log).

Note: the geometric *mask validation* tools support pinhole-family cameras only —
they refuse `EQUIRECTANGULAR`/`EUCM`/`SPHERE` models explicitly rather than
producing incorrect results.

## 360° workflows

You have two supported paths, plus one legacy:

1. **Native (simplest):** point Images at your equirectangular frames, camera
   model `EQUIRECTANGULAR`, sequential matching, incremental mapper. Bundle
   adjustment uses Ceres (Caspar doesn't support 360° cameras).
2. **Reframe → Caspar (fastest BA, most compatible outputs):** convert frames to
   pinhole views first (prep360 reframe / Extract tab), then align the views with
   camera model `PINHOLE`, exhaustive matching, incremental mapper, BA backend
   AUTO. Downstream tools that expect perspective cameras prefer this path.
3. **SphereSfM (legacy):** the original fork, kept working for old projects.
   Marked "(legacy)" in the engine toggle.

On the same 76-frame test set, the native path registered 72/76 frames at 1.17 px
mean reprojection error — slightly better than the legacy fork on identical input.

Global mapping combined with `EQUIRECTANGULAR` is experimental and unvalidated;
the run log warns when you select that combination. Use incremental.

## Typical workflow (perspective)

1. **Images** → your photos or reframed views; **Masks** → reviewed masks.
2. Engine **COLMAP**, camera model `SIMPLE_RADIAL` (real lenses) or `PINHOLE`
   (reframed views), strategy **sequential** for ordered captures.
3. Mapper **incremental**, BA backend **AUTO**, GPU BA on.
4. Click **Align**; watch the stages and the log (it reports which backend and
   BA solver were selected, and why).
5. Inspect the point cloud and summary. Good signs: >90% of images registered,
   mean reprojection error under ~1.5 px.

## Tips

- **Low registration** (<80–90%) usually means matching failed: increase max
  features, switch to exhaustive matching, or check image overlap.
- **High reprojection error** (>1.5–2 px): try a distortion-estimating camera
  model (`OPENCV`), or enable principal-point refinement.
- **"…failed qualification" in the log**: the in-process backend was excluded for
  this session. Runs continue via the CLI binary. Restart the app after fixing
  the installation.
- **"Camera model X is not supported by…"**: your backend predates COLMAP 4.1.
  Point Tool Setup at a 4.1+ binary or wheel.
- **Varying image dimensions** in one folder: uncheck **Single camera**.

## Installation notes (for anyone setting this up fresh)

The Align tab expects a **CUDA build of pycolmap** and a **COLMAP 4.1+ binary**
(the app verifies both at runtime and says so if they're missing). Two warnings:

- Do **not** run a generic `pip install --upgrade pycolmap`: the standard PyPI
  Windows wheel is CPU-only and would silently replace a CUDA build. The app will
  detect and exclude a CPU wheel, but you'd lose in-process GPU alignment until
  you reinstall the CUDA wheel.
- CUDA-enabled Windows wheels and binaries (with Caspar and cuDSS) are published
  by the [build_gpu_colmap](https://github.com/lyehe/build_gpu_colmap/releases)
  project; verify downloads against the release checksums.
