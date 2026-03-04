# Model Selection Guide

Choosing the right segmentation model (or combination) for your masking task.

## Quick decision tree

```
What are you masking?
│
├─ Known objects (people, backpacks, tripods, vehicles)
│  ├─ Speed matters → YOLO26
│  ├─ Best quality → SAM 3
│  └─ Both matter → Ensemble (YOLO26 + RF-DETR)
│
├─ Unusual objects ("selfie stick", "camera rig", "drone shadow")
│  └─ SAM 3 (only model that accepts text descriptions)
│
├─ Quick preview of a few images
│  └─ FastSAM (fastest, "good enough" for previewing)
│
└─ Final production masks for reconstruction
   └─ Ensemble mode (multiple models, weighted fusion)
```

## Model comparison

| | SAM 3 | YOLO26 | RF-DETR | FastSAM |
|---|---|---|---|---|
| **Architecture** | Vision Transformer + text encoder | YOLOv8 derivative, NMS-free | DETR transformer | YOLO-based auto-segmenter |
| **Prompting** | Text descriptions | COCO class IDs | COCO class IDs | None (auto-segment) |
| **Speed** (3090 Ti, 1024×1024) | ~800ms/image | ~40ms/image | ~120ms/image | ~25ms/image |
| **Mask quality** | Highest — precise boundaries | Good — occasional edge artifacts | Good — transformer accuracy | Fair — coarser boundaries |
| **VRAM** | ~4 GB | ~1 GB (nano) | ~2 GB (small) | ~1 GB |
| **Novel objects** | Yes (any text description) | No (80 COCO classes only) | No (80 COCO classes only) | Auto — no class control |
| **Weights** | ~2 GB (auto-download) | 7 MB nano, 25 MB small | ~100 MB small | ~140 MB |
| **Install** | `git clone` + `pip install -e .` | `pip install ultralytics` | `pip install rfdetr supervision` | `pip install ultralytics` |

### SAM 3 — Segment Anything Model 3

**When to use:** When you need the highest quality masks, or when your targets aren't in COCO's 80 classes.

SAM 3 is the only model that accepts natural language descriptions. You can say "person holding a camera" or "selfie stick" — it understands semantic concepts, not just class labels. This makes it uniquely valuable for photogrammetry where the objects to remove (camera rigs, custom equipment) may not match standard detection classes.

**Trade-off:** It's 20x slower than YOLO26. For a 500-frame dataset with 6 cubemap faces each, that's ~40 minutes vs ~2 minutes.

```python
config = MaskConfig(
    model=SegmentationModel.SAM3,
    remove_prompts=["person", "tripod", "backpack", "selfie stick", "camera rig"],
    keep_prompts=["building", "tree"],  # optional: protect these from masking
    confidence_threshold=0.5,
)
```

**Paper:** [Segment Anything Model 3](https://github.com/facebookresearch/sam3) (Meta AI)

### YOLO26 — Real-time instance segmentation

**When to use:** Batch processing, known COCO objects, when speed matters.

YOLO26 is the production workhorse. NMS-free end-to-end inference means consistent, fast results. The nano model (7 MB) is good enough for most photogrammetry masking; step up to small/medium for better accuracy on small or distant objects.

**Trade-off:** Limited to COCO's 80 classes. Can't detect "selfie stick" or "camera rig" — but covers person (0), backpack (24), umbrella (25), handbag (26), suitcase (28), and other common capture gear.

```python
config = MaskConfig(
    model=SegmentationModel.YOLO26,
    yolo_model_size="n",                     # n=nano, s=small, m=medium, l=large, x=xlarge
    yolo_classes=[0, 24, 25, 26, 28],        # person + common gear
    confidence_threshold=0.5,
)
```

**Model sizes:**

| Size | Params | Speed (3090 Ti) | Use case |
|------|--------|-----------------|----------|
| `n` (nano) | 3.2M | ~40ms | Default — fast, good enough |
| `s` (small) | 11.2M | ~55ms | Better for small/distant objects |
| `m` (medium) | 25.9M | ~80ms | Higher accuracy |
| `l` (large) | 43.7M | ~110ms | Diminishing returns for masking |
| `x` (xlarge) | 68.2M | ~150ms | Rarely needed |

**Paper:** [YOLO — Ultralytics](https://docs.ultralytics.com/)

### RF-DETR — Transformer detection

**When to use:** As a YOLO alternative in ensembles, or when transformer accuracy matters more than speed.

RF-DETR is a DETR-family transformer detector. It can find objects that YOLO misses (different failure modes), making it valuable as an ensemble partner. Uses the same COCO class IDs as YOLO.

```python
config = MaskConfig(
    model=SegmentationModel.RFDETR,
    rfdetr_model_size="small",               # nano, small, medium, large
    yolo_classes=[0, 24, 25, 26, 28],        # shares COCO class config with YOLO
    confidence_threshold=0.5,
)
```

**Paper:** [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow)

### FastSAM — Quick auto-segmentation

**When to use:** Quick previews, checking if masking will work on a dataset before committing to a full run.

FastSAM auto-segments everything in the image — no prompts needed. It's the fastest option but gives you less control. Masks tend to be coarser than SAM 3 or YOLO26.

```python
config = MaskConfig(
    model=SegmentationModel.FASTSAM,
    confidence_threshold=0.4,                # lower threshold since auto-segment is less precise
)
```

## Ensemble mode

Run multiple models on the same image and fuse their results. This catches objects that any single model might miss — YOLO and RF-DETR have different failure modes, so their union is more complete than either alone.

### How it works

1. Each model in `ensemble_models` runs independently on the image
2. Results are clustered by IoU (detections with IoU > `ensemble_iou_threshold` are considered the same object)
3. Matched detections are fused via Weighted Mask Fusion: pixel-wise `(confidence × model_weight)` weighted average, thresholded at 0.5
4. Unmatched detections (found by only one model) are included if their confidence exceeds `confidence_threshold`

### Configuration

```python
config = MaskConfig(
    ensemble=True,
    ensemble_models=["yolo26", "rfdetr"],    # any combination of: yolo26, rfdetr, fastsam, sam3
    ensemble_weights=[0.6, 0.4],             # YOLO26 weighted higher (None = equal weights)
    ensemble_iou_threshold=0.5,              # IoU threshold for matching detections
    confidence_threshold=0.5,
)
```

### Practical ensemble recipes

**Fast + reliable** (recommended for production):
```python
ensemble_models=["yolo26", "rfdetr"]
ensemble_weights=[0.6, 0.4]
# ~160ms/image — YOLO catches most things, RF-DETR catches edge cases
```

**Maximum coverage** (when you can't afford misses):
```python
ensemble_models=["sam3", "yolo26", "rfdetr"]
ensemble_weights=[0.5, 0.3, 0.2]
# ~950ms/image — SAM3's text prompts + YOLO's speed + RF-DETR's precision
```

**Quick sanity check**:
```python
ensemble_models=["yolo26", "fastsam"]
ensemble_weights=None  # equal
# ~65ms/image — two fast models with different detection strategies
```

## Auto-fallback chain

If your preferred model isn't available (missing package, import error), the pipeline automatically tries the next model:

```
SAM 3 → RF-DETR → YOLO26 → FastSAM → EfficientSAM → SAM 2 → RuntimeError
```

Set `auto_select_model=False` in `MaskingPipeline()` to disable fallback and fail immediately if the requested model can't load.

## COCO class reference

Common classes used in photogrammetry masking:

| ID | Class | Typical use |
|----|-------|-------------|
| 0 | person | The photographer, bystanders |
| 24 | backpack | Gear on the ground |
| 25 | umbrella | Shade equipment |
| 26 | handbag | Camera bags |
| 28 | suitcase | Equipment cases |
| 56 | chair | Folding chairs at capture sites |
| 63 | laptop | Tethered capture control |
| 66 | keyboard | — |
| 67 | cell phone | Remote triggers |

**Built-in presets** (used by the prep360 CLI's `--preset` option):
- **person** — `[0]` — Just the photographer
- **photographer** — `[0, 24, 25, 26, 28]` — Person + common gear
- **equipment** — `[0, 24, 25, 26, 27, 28, 67]` — Person + accessories + phone
- **vehicles** — `[1, 2, 3, 5, 6, 7, 8]` — Car, bicycle, motorbike, bus, train, truck, boat
- **animals** — `[14, 15, 16, 17, 18, 19, 20, 21, 22, 23]` — All COCO animals
- **all_dynamic** — `[0, 1, 2, 3, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19]` — Union of person + vehicles + animals

> **Note:** The GUI's Mask tab uses a separate target list (person, backpack, car, camera, selfie stick) — see [MASK_TAB.md](MASK_TAB.md#prompts-and-classes).
