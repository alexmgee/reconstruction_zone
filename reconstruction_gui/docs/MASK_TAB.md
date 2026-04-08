# Mask Tab

The Mask tab is the core of Masking Studio. It runs automated object detection and segmentation on your images, producing binary masks that tell COLMAP/Metashape/3DGS which pixels to ignore during reconstruction. The tab is organized into three collapsible sections — Detection, Detection & Refinement, and Post-Processing — followed by Run/Cancel controls, a Batch Queue, and a progress bar.

## Sections

| Section | Default | Purpose |
|---------|---------|---------|
| **Detection** | Expanded | Input/output paths, model selection, prompts, file options, output modes |
| **Detection & Refinement** | Expanded | Optional pipeline stages that improve detection quality |
| **Post-Processing** | Expanded | Mask cleanup, geometry-specific corrections |

Below the sections: **Preview** / **Run Masking** / **Stop** buttons, a collapsible **Batch Queue**, progress bar, and status label.

---

## Detection

Expanded by default. Contains everything needed for a basic masking run.

### Input and Output

- **Input** — A folder of images, a single image file, or a video file. When given a folder, the pipeline processes every image matching the file type filter. When given a video, frames are extracted first.
- **Output** — Directory where mask files are saved. Each mask has the same filename as its source image but in PNG format (always lossless, regardless of input format).

### SAM3 Mode

Two modes control how SAM3 processes a directory of images:

- **Hybrid** (default) — Each frame is processed independently: the selected detector (YOLO, RF-DETR, etc.) finds objects per frame, then SAM segments them. Each frame's detections are independent.

- **Unified Video** — Loads the entire directory as a video sequence into SAM3's video predictor. You provide text prompts (the Remove/Keep fields), SAM3 detects objects on the first frame using Promptable Concept Segmentation, then propagates those detections forward through all frames with built-in temporal tracking. Objects only need to be found once — SAM3 tracks them across frames automatically. This bypasses the per-frame detector entirely. Requires the SAM3 model.

Use Hybrid for most work. Use Unified Video when you have a sequential image set (frames from a video) and want consistent tracking without per-frame detection noise.

**What changes in the GUI when you select Unified Video:**

Unified Video runs SAM3's own end-to-end pipeline — it bypasses per-frame detection, all Detection & Refinement stages, and all Post-Processing. The GUI reflects this with three visual cues:

1. **Dimmed widgets** — Model, YOLO size, Geometry, and Conf dropdowns turn grey. Mask target checkboxes turn grey. All are disabled (non-interactive).
2. **Sections collapse** — Detection & Refinement and Post-Processing auto-collapse to signal they're not used. Switching back to Hybrid re-expands them.
3. **Prompt glow** — The Remove and Keep entry rows get a subtle blue highlight background, emphasizing that text prompts are the primary input for Unified mode.

| Greyed out (not used) | Still active |
|-----------------------|-------------|
| Model, YOLO size, Geometry, Conf | Input / Output paths |
| Mask target checkboxes | Remove / Keep text prompts (highlighted) |
| All of Detection & Refinement (collapsed) | File types, Skip existing, Review thresh |
| All of Post-Processing (collapsed) | Multi-label output, Inpaint masked |

### Model and Geometry

These settings are used in Hybrid mode only. They are dimmed and disabled when Unified Video is selected.

| Setting | Options | Default | Purpose |
|---------|---------|---------|---------|
| **Model** | auto, yolo26, rfdetr, sam3, fastsam | auto | Which detector to use. `auto` picks YOLO26 if available, falls back through the list |
| **YOLO size** | n, s, m, l, x | n | YOLO model variant. `n` (nano) is fastest; `x` is most accurate but slowest |
| **Geometry** | pinhole, equirect, fisheye, cubemap | pinhole | Image projection type. `equirect` triggers cubemap decomposition — the image is split into 6 cube faces, each face is segmented independently, then masks are merged back to equirectangular |
| **Conf** | 0.0–1.0 | 0.70 | Detection confidence threshold. Lower = more detections (more recall, more false positives) |
| **torch.compile** | On/Off | Off | JIT-compile the detection model for ~20-50% faster GPU inference. Adds startup time (warmup). Requires PyTorch 2.0+ and CUDA |

### Prompts and Classes

Two ways to specify what to mask:

**Text prompts** (used by SAM3 in Unified Video mode):
- **Remove** — Objects to mask out. Default: `person, tripod, backpack, selfie stick`
- **Keep** — Objects to protect from masking (optional). Useful when "remove person" would accidentally catch mannequins or statues you want to keep.

**Mask targets** (used by YOLO/RF-DETR in Hybrid mode):
Checkboxes for common classes: person, backpack, car, camera, selfie stick. Defaults: person, backpack, and selfie stick are checked. Checked targets are passed as class filters to the detector. Custom prompts in the Remove field are also sent as text prompts to SAM3 if the model supports it. Dimmed and disabled in Unified Video mode.

### Multi-pass SAM3

When enabled, SAM3 runs once per prompt with individual confidence thresholds, then unions all masks together. This is useful when different objects need different sensitivity — e.g. a person at 0.7 confidence but a thin selfie stick at 0.4.

Each row specifies a prompt string and a confidence threshold. Click **+ Add Pass** to add rows, **Remove** to delete the last row.

This bypasses the global Conf slider for SAM3 — each pass uses its own threshold.

### File Types, Skip Existing, Review Threshold

| Setting | Default | Purpose |
|---------|---------|---------|
| **File types** | `*.jpg *.png` | Glob pattern for input images. Change to `*.tiff *.bmp` or similar if your dataset uses other formats |
| **Skip existing** | Off | When enabled, skips images that already have a mask in the output directory. Useful for resuming interrupted runs |
| **Review thresh** | 0.85 | Quality score threshold. Masks scoring below this are flagged for manual review on the Review tab. The score is based on mask area, edge quality, and detection confidence |

### Output Options

| Setting | Default | Purpose |
|---------|---------|---------|
| **Create review folder** | Off | Creates `masks/` and `review/` subdirectories inside the output folder |
| **Include rejects** | On | Generate a mask for every image regardless of confidence score (low-confidence masks still get quality flags) |
| **Multi-label output** | Off | Save per-class segmentation maps alongside binary masks. Each pixel value is a class ID (0 = background, 1 = person, 2 = backpack, etc.). Saved to `{output}/segmaps/` |
| **Inpaint masked** | Off | Fill masked regions with plausible texture using OpenCV inpainting (Telea algorithm). Saves separate files to `{output}/inpainted/`. For 3DGS training where black holes in masked regions cause dark splat artifacts |

---

## Detection & Refinement

Expanded by default. Contains optional pipeline stages that run after the initial detection to improve mask quality. Auto-collapses when Unified Video is selected. Each stage is independently collapsible and has its own Enable checkbox.

The stages run in this order in the pipeline:

```
Detect (YOLO/RF-DETR/SAM3)
    |
    v
Shadow Detection ── add shadow regions to mask
    |
    v
SAM Mask Refinement ── tighten edges with SAM point prompts
    |
    v
Edge Injection ── add thin structures via Canny edges
    |
    v
Alpha Matting ── generate soft edges
    |
    v
COLMAP Geometric Validation ── cross-check against 3D geometry
```

### Shadow Detection

Detects shadows cast by masked objects and includes them in the mask. Shadows from photographers, tripods, and equipment can confuse reconstruction if left unmasked.

| Setting | Options | Default | Purpose |
|---------|---------|---------|---------|
| **Detector** | targeted_person, brightness, c1c2c3, hybrid | targeted_person | Shadow detection algorithm. `targeted_person` detects shadows beneath detected persons using depth-aware ground plane estimation; `brightness` is a fast heuristic; `c1c2c3` uses chromaticity-invariant color space; `hybrid` combines brightness + chromaticity |
| **Verify** | none, c1c2c3, hybrid, brightness | none | Optional second-pass verification to reduce false positives |
| **Spatial** | all, near_objects, connected | near_objects | Where to look for shadows. `near_objects` only checks regions near detected objects; `connected` requires shadows to touch the object mask |
| **Dilation** | 0–200 px | 50 | Expand shadow search region around objects |
| **Confidence** | 0.0–1.0 | 0.50 | Minimum confidence to accept a shadow detection |
| **Darkness** | 0.0–1.0 | 0.70 | How dark a region must be to qualify as shadow |
| **Chromaticity** | 0.0–0.5 | 0.15 | Color shift threshold for shadow vs. non-shadow |

### SAM Mask Refinement

Takes the coarse mask from the detector and refines edges using SAM (Segment Anything Model) with automatically placed point prompts along the mask boundary. Produces tighter, more accurate edges — especially useful for complex silhouettes.

| Setting | Options | Default | Purpose |
|---------|---------|---------|---------|
| **SAM model** | vit_b, vit_l, vit_h | vit_b | Model size. vit_b (375MB) is fastest; vit_h is most accurate |
| **Box margin** | 0.0–0.5 | 0.15 | How much to expand the bounding box around each detection before sending to SAM |
| **IoU threshold** | 0.0–1.0 | 0.50 | Minimum overlap between SAM's output and the original mask to accept the refinement |

### Alpha Matting (ViTMatte)

Generates soft alpha edges instead of hard binary boundaries. Useful for hair, fur, transparent fabric, and any semi-transparent boundary. The output mask contains gradual transitions (0–255) rather than sharp 0/1 edges.

| Setting | Options | Default | Purpose |
|---------|---------|---------|---------|
| **Model** | small, base | small | ViTMatte model size. small (~100MB) is recommended |
| **Erode** | 0–50 px | 10 | Shrink the mask edge to create the trimap's definite-foreground region |
| **Dilate** | 0–50 px | 10 | Expand the mask edge to create the trimap's definite-background region. The band between erode and dilate becomes the "unknown" region where matting operates |

### Ensemble Detection (WMF)

Runs multiple detectors on each image and merges their masks using Weighted Mask Fusion. Higher recall than any single detector — catches objects that one model misses but another finds.

| Setting | Options | Default | Purpose |
|---------|---------|---------|---------|
| **RF-DETR size** | nano, small, medium, large | small | RF-DETR model variant for the ensemble |
| **Models** | text | yolo26, rfdetr | Comma-separated list of detectors to run |
| **IoU threshold** | 0.0–1.0 | 0.50 | Minimum overlap between detectors' masks to consider them the same object |

### Edge Injection

Adds Canny edge detection to catch thin structures that object detectors typically miss — wires, cables, antennas, thin poles. The edge map is combined with the object mask before post-processing. Only a checkbox to enable/disable; no additional parameters.

### COLMAP Geometric Validation

Cross-checks masks against an existing COLMAP sparse reconstruction. Projects 3D points into each camera view and checks whether masked pixels coincide with reconstructed geometry. Flags masks that contradict the 3D model.

| Setting | Default | Purpose |
|---------|---------|---------|
| **Sparse dir** | (browse) | Path to COLMAP sparse reconstruction (e.g. `sparse/0/`) |
| **Agreement** | 0.70 | Minimum fraction of projected 3D points that must agree with the mask |
| **Flag above** | 0.15 | Fraction of disagreeing points above which the mask is flagged for review |

> **Note:** VOS temporal propagation (LiVOS/Cutie) is available via the Python API (`vos_propagation=True` in MaskConfig) but is not currently exposed in the GUI.

---

## Post-Processing

Expanded by default. Mask cleanup and equirectangular-specific corrections applied after the detection pipeline completes. Auto-collapses when Unified Video is selected.

### Mask Dilate and Fill Holes

| Setting | Default | Purpose |
|---------|---------|---------|
| **Mask dilate** | 0 px | Expand the final mask by N pixels in all directions. Adds a safety margin around detected objects |
| **Fill holes** | Off | Fill enclosed holes inside the mask using morphological close + flood fill. Catches gaps from camera mounts, equipment between legs, etc. |

### 360° / Equirect

Only relevant when Geometry is set to `equirect`.

| Setting | Default | Purpose |
|---------|---------|---------|
| **Nadir mask** | 0% | Mask the bottom N% of the equirectangular image. Covers the nadir pole where the camera mount / tripod head is always visible |
| **Pole expand** | 1.2x | Expansion factor for masks near the poles. Objects near the poles are stretched in equirectangular projection, so masks need to be expanded to compensate |
| **Cubemap overlap** | 0° | Overlap angle between adjacent cubemap faces. Non-zero values reduce seam artifacts at face boundaries but increase computation |

### Fisheye

Only relevant when processing raw fisheye images (not reframed perspectives).

| Setting | Default | Purpose |
|---------|---------|---------|
| **Fisheye circle mask** | On | Mask the corners and periphery outside the fisheye image circle |
| **Margin** | 0% | Extra margin to trim from the fisheye circle edge. Increase if the lens has soft edges or chromatic aberration at the periphery |

---

## Batch Queue

Collapsed by default. Process multiple image folders with the same masking settings.

| Button | Purpose |
|--------|---------|
| **Add Current** | Add the current Input folder to the queue |
| **Add Subfolders** | Scan a parent directory and add each image-containing subfolder |
| **Remove** | Remove the selected queue item |
| **Clear Done** | Remove all completed items from the queue |
| **Process Queue** | Run masking on all pending items sequentially |

Each queue item shows its folder name, status (pending / processing / done / error), and progress percentage. The queue persists between sessions.

---

## Pipeline Summary

```
Input images
      |
      v
Detection: detect objects per frame (or SAM3 unified tracking)
      |
      v
Detection & Refinement (each optional):
      |-- Shadow Detection
      |-- SAM Mask Refinement
      |-- Edge Injection
      |-- Alpha Matting
      |-- COLMAP Geometric Validation
      |
      v
Post-Processing:
      |-- Mask dilation
      |-- Fill holes
      |-- Nadir mask / pole expand (equirect)
      |-- Fisheye circle mask (fisheye)
      |
      v
Output:
      |-- masks/         binary masks (always)
      |-- segmaps/       per-class maps (if multi-label)
      |-- inpainted/     filled images (if inpaint)
      |-- Quality scores + review flags
```

---

## Typical workflow: perspective images

```
1. Detection     -> Set input folder and output folder
2.               -> Model = auto, Geometry = pinhole (default)
3.               -> Check target classes (person, backpack, etc.)
4.               -> File types = *.jpg *.png
5.               -> Click Run Masking
6. (Optional)    -> Enable Skip existing, re-run if interrupted
7. Review tab    -> Check flagged masks, fix with brush editor
```

## Typical workflow: 360° equirectangular images

```
1. Detection     -> Set input folder and output folder
2.               -> Model = auto, Geometry = equirect
3.               -> Check target classes
4. Post-Proc     -> Set Nadir mask to ~5-10% (covers tripod head)
5.               -> Set Pole expand to 1.2-1.5 (compensates for stretching)
6. (Optional)    -> Detection & Refinement: enable Shadow Detection
7.               -> Click Run Masking
8. Review tab    -> Check flagged masks
```

## Typical workflow: video sequence with temporal consistency

```
1. Detection     -> Set input folder (frames from a video)
2.               -> SAM3 mode = Unified Video
3.               -> Type prompts in Remove field: person, tripod
4.               -> Click Run Masking
```

Unified Video mode uses SAM3's built-in temporal tracking — objects are detected on the first frame and tracked through all subsequent frames automatically.

## Typical workflow: maximum quality (slow)

```
1. Detection     -> Model = auto, Geometry = pinhole
2. Det & Refine  -> Enable Shadow Detection (hybrid detector)
3.               -> Enable SAM Mask Refinement (vit_b)
4.               -> Enable Ensemble Detection (YOLO26 + RF-DETR)
5.               -> Enable Alpha Matting (soft edges)
6. Post-Proc     -> Mask dilate = 2-3 px (safety margin)
7.               -> Fill holes = on
8.               -> Click Run Masking
```
