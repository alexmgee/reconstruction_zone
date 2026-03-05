# Quickstart: First Mask in 5 Minutes

This guide gets you from zero to a working mask on a fresh machine. By the end, you'll have processed a 360° equirectangular image and produced a clean binary mask.

## Prerequisites

- **Python 3.10+** (3.12–3.14 tested)
- **NVIDIA GPU** with CUDA drivers installed (check with `nvidia-smi`)
- **Git** (for cloning SAM 3)

> **CPU-only?** Everything works on CPU — just set `device="cpu"` in MaskConfig. Expect 10-50x slower processing. YOLO26 on CPU is still usable for small batches.

## Step 1: Install everything

Request access to SAM 3 model weights before starting — approval can take hours. Go to [facebook/sam3 on HuggingFace](https://huggingface.co/facebook/sam3), create a free account if needed, and click **Request access**.

```bash
# PyTorch with CUDA (adjust cu126 to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Core + GUI + models
pip install numpy opencv-python Pillow ultralytics tqdm pyyaml customtkinter
pip install rfdetr supervision py360convert
pip install huggingface_hub "transformers>=4.50,<5.0"

# SAM 3 — text-prompted segmentation (primary masking model)
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .

# Authenticate with HuggingFace (once your access request is approved)
huggingface-cli login
```

SAM 3 weights (~2 GB) download automatically on first run once authenticated. While waiting for approval, the app falls back to YOLO26 (class-based detection, works immediately).

Verify CUDA:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

<details>
<summary><strong>Optional extras</strong> — temporal propagation and shadow detection</summary>

**Temporal propagation** (masks propagate across video frames):

```bash
# LiVOS (recommended)
git clone https://github.com/hkchengrex/LiVOS.git
cd LiVOS && pip install -e .

# Or Cutie (alternative)
git clone https://github.com/hkchengrex/Cutie.git
cd Cutie && pip install -e .
```

**Shadow detection** (extends masks to cover cast shadows):

```bash
pip install efficientnet-pytorch
```

</details>

**Common issues:**

| Symptom | Fix |
|---------|-----|
| `torch.cuda.is_available()` returns `False` | PyTorch CUDA version doesn't match your driver. Run `nvidia-smi` to check, then install the matching wheel from [pytorch.org](https://pytorch.org/get-started/locally/) |
| `No module named 'ultralytics'` | `pip install ultralytics` — this provides YOLO26 |

## Step 2: Choose your model config

**SAM 3** (recommended — text-prompted, highest quality):
```python
from reconstruction_pipeline import MaskConfig, SegmentationModel

config = MaskConfig(
    model=SegmentationModel.SAM3,
    remove_prompts=["person", "tripod", "backpack", "selfie stick"],
    confidence_threshold=0.5,
)
```

**YOLO26** (fast, COCO classes only):
```python
from reconstruction_pipeline import MaskConfig, SegmentationModel

config = MaskConfig(
    model=SegmentationModel.YOLO26,
    yolo_classes=[0, 24, 25, 26, 28],  # person, backpack, umbrella, handbag, suitcase
    confidence_threshold=0.5,
)
```

**Auto-select** — if you don't specify a model, the pipeline tries them in priority order and uses the first one available:
```
SAM 3 → RF-DETR → YOLO26 → FastSAM → EfficientSAM → SAM 2
```

## Step 3: Process your first image

```python
from reconstruction_pipeline import MaskingPipeline, MaskConfig, ImageGeometry, SegmentationModel
import cv2

# Configure the pipeline (using SAM3 or YOLO26 — see Step 3)
config = MaskConfig(
    model=SegmentationModel.SAM3,
    remove_prompts=["person", "tripod", "backpack"],
    confidence_threshold=0.5,
    geometry_aware=True,                     # cubemap decomposition for 360°
    fill_holes=True,                         # fill interior gaps
    mask_dilate_px=10,                       # grow mask 10px outward
)

# Initialize (loads model, ~2-5 seconds)
pipeline = MaskingPipeline(config)

# Process a single image
image = cv2.imread("your_360_image.jpg")
result = pipeline.process_image(image, ImageGeometry.EQUIRECTANGULAR)

# Save the mask (multiply by 255 — internal masks are 0/1)
cv2.imwrite("mask.png", result.mask * 255)

# Check quality
print(f"Quality: {result.quality.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Needs review: {result.needs_review}")
```

**What each step does:**

1. `MaskingPipeline(config)` — Loads the segmentation model onto GPU. First run downloads weights.
2. `process_image()` with `EQUIRECTANGULAR` geometry — Decomposes the 360° image into 6 cubemap faces, runs detection on each face, merges results back to equirectangular space, then applies postprocessing (dilation, fill holes).
3. `result.mask` — A numpy array of 0s and 1s. White (1) = masked region, Black (0) = keep.

### For perspective (non-360°) images

```python
result = pipeline.process_image(image, ImageGeometry.PINHOLE)
```

No cubemap decomposition — the detector runs directly on the image.

## Step 4: Batch process a directory

```python
stats = pipeline.process_directory(
    input_dir="./frames",
    output_dir="./masks",
    geometry=ImageGeometry.EQUIRECTANGULAR,
    pattern="*.jpg",                         # also supports "*.jpg *.png"
    skip_existing=True,                      # resume interrupted runs
)

print(f"Processed: {stats['processed']}")
print(f"Skipped (already exist): {stats['skipped']}")
print(f"Failed: {stats['failed']}")
```

Masks are saved with the same stem as the input: `frame_0001.jpg` → `masks/frame_0001.png`.

## Step 5: Review the results

### Quick check in Python

```python
from review_gui import load_overlay_thumbnail, compute_mask_area_percent

# Create a visual overlay (red tint on masked regions)
thumb = load_overlay_thumbnail("frames/frame_0001.jpg", "masks/frame_0001.png", size=600)
cv2.imshow("Overlay", thumb)
cv2.waitKey(0)

# Check mask coverage
area = compute_mask_area_percent("masks/frame_0001.png")
print(f"Mask covers {area:.1f}% of the image")
```

### Launch the GUI for full review

```bash
python reconstruction_zone.py
```

The **Review tab** shows a thumbnail grid of all mask overlays, filterable by quality. You can accept, reject, skip, or open the interactive editor (brush, flood fill, lasso tools) for manual touch-up.

## What's next

- **Video sequences?** Enable temporal propagation: `vos_propagation=True` — masks propagate between keyframes using LiVOS/Cutie. See [ARCHITECTURE.md](ARCHITECTURE.md#temporal-propagation).
- **Soft edges?** Enable alpha matting: `matting=True` — converts binary masks to soft alpha mattes via ViTMatte.
- **Multiple models?** Enable ensemble: `ensemble=True, ensemble_models=["yolo26", "rfdetr"]` — runs multiple detectors and fuses results. See [MODELS.md](MODELS.md#ensemble-mode).
- **Shadow removal?** Enable shadow detection: `detect_shadows=True` — extends person masks to cover cast shadows.
- **COLMAP validation?** After reconstruction, validate mask consistency: `colmap_validate=True, colmap_dir="./sparse/0"`. See [ARCHITECTURE.md](ARCHITECTURE.md#colmap-validation).

## Dependency reference

| Package | What it provides |
|---------|------------------|
| `torch`, `torchvision`, `torchaudio` | GPU compute, model loading |
| `numpy`, `opencv-python` | Array operations, image I/O |
| `ultralytics` | YOLO26 + FastSAM segmentation |
| `rfdetr`, `supervision` | RF-DETR transformer detection (ensemble partner) |
| `py360convert` | Equirect-to-perspective reframing |
| `transformers` | ViTMatte alpha matting |
| `huggingface_hub` | SAM 3 weight authentication |
| `sam3` | Text-prompted segmentation ([HF access](https://huggingface.co/facebook/sam3) required) |
| `tqdm`, `pyyaml` | Progress bars, config parsing |
| `customtkinter` | GUI framework |
| ffmpeg, ffprobe | Video frame extraction ([ffmpeg.org](https://ffmpeg.org/download.html), must be on PATH) |
