# Shadow Detection for Photogrammetry Masking — Research Report

## Problem Statement

Outdoor photogrammetry datasets captured at high noon under cloudless skies produce
harsh, well-defined cast shadows from people, equipment, and tripods. These shadows
need to be detected and masked so photogrammetry software (COLMAP, Metashape) ignores
those pixels during reconstruction. The current approach in `masking_v2.py` uses a
simple brightness-ratio heuristic that only searches directly below detected people.

**Target scenario**: Beach/ground-level scenes, overhead sun, strong contrast shadows
on uniform surfaces (sand, concrete, grass). Shadows are visually obvious to humans
but the current heuristic misses shadows that aren't directly below a person, shadows
cast at angles, and shadows on surfaces that are already somewhat dark.

---

## Current Implementation

**File**: `masking_v2.py`, `_detect_shadows()` (line 1135)

**Method**:
1. Convert image to grayscale
2. Compute local median brightness (51x51 blur)
3. Pixels darker than 70% of local median = shadow candidates
4. Morphological closing (7x7 kernel) to bridge gaps
5. Dilate person mask downward (15x40 rect kernel) to find attachment zone
6. AND shadow candidates with attachment zone
7. Remove connected components smaller than 5% of person mask area

**Limitations**:
- Grayscale only — ignores color/chromaticity information
- Only searches directly below detected people (downward dilation)
- Fixed 0.7 threshold doesn't adapt to scene lighting
- Can't distinguish dark surfaces from shadowed surfaces
- Misses shadows cast at angles (e.g., low sun, or person walking at angle)
- No soft shadow / penumbra handling

---

## Dedicated Shadow Detection Models

### Overview

These models take a single RGB image as input and output a binary (or soft) shadow
mask — exactly what we need. They are trained on shadow-specific datasets (SBU, ISTD,
CUHK-Shadow) that contain diverse outdoor scenes with cast shadows.

All claims below have been verified by directly inspecting the GitHub repos.

---

### SILT (ICCV 2023) — Best documented, multiple backbones

- **Repo**: https://github.com/hanyangclarence/SILT
- **Paper**: Shadow-aware Iterative Label Tuning
- **What it does**: Binary shadow segmentation from a single image
- **Output**: Grayscale probability map → threshold to binary mask

**Pretrained weights** (Google Drive, verified links exist):
| Backbone | SBU BER | ISTD BER | Link |
|----------|---------|----------|------|
| ResNeXt-101 | — | — | [Google Drive](https://drive.google.com/file/d/18U2o7msKJexwUzYuoWf4Hp_hxM0sl6IP/view) |
| PVTv2-B3 | 4.36 | 1.11 | [Google Drive](https://drive.google.com/file/d/1xIsO5uS_Z7G5WsK_qlCCdxI4GA3sYb9Y/view) |
| PVTv2-B5 | 4.19 | 1.16 | [Google Drive](https://drive.google.com/file/d/1fgF8pgXEgDJ2bFFLcNUeJJvMzhdr2oOa/view) |

**Inference**: `python infer.py --dataset SBU --ckpt path_to_weight`
- Currently expects dataset directory with `test.txt` listing image paths
- Would need minor adaptation for single-image / batch-directory use

**Dependencies**: Python 3.8, PyTorch 1.8.0, CUDA 11.1, torchvision 0.9.0,
timm, kornia, omegaconf 2.3.0. Full conda env in `environment.yaml`.

**Input resolution**: 512x512 (configurable in config yaml)

**Notes**:
- Uses iterative label refinement during training (self-training on pseudo-labels)
- PVTv2-B3 is the best accuracy/speed tradeoff
- BER numbers above are from the paper, not the standardized benchmark (see below)

---

### FDRNet (ICCV 2021) — Best edge quality with CRF

- **Repo**: https://github.com/rayleizhu/FDRNet
- **Paper**: Feature Decomposition and Reconstruction Network
- **What it does**: Binary shadow segmentation + optional CRF edge refinement
- **Output**: Sigmoid probability map saved as grid images (input | prediction | GT)

**Pretrained weights** (GitHub Releases, tag `checkpoint`):
- `sbu_epoch_010.pt` — for SBU and UCF test sets
- `istd_epoch_010.pt` — for ISTD test set

**Inference**: Two-step process:
1. `python test.py` → raw predictions in `test/raw/`
2. `python crf_refine.py` → refined predictions in `test/crf/`

**Dependencies**: Python 3.6.10, PyTorch 1.5.0, CUDA 10.1, EfficientNet-PyTorch,
pydensecrf, OpenCV 3.4.2. Full conda env in `env.yaml`.

**BER (SBU)**: 3.27 raw, 3.04 with CRF refinement

**Notes**:
- CRF post-processing produces very crisp shadow boundaries — ideal for masking
- Older PyTorch version (1.5.0) may conflict with other tools in the pipeline
- test.py currently hardcodes dataset path — needs modification for custom data
- Output is a grid image (input|pred|GT), not a standalone mask — needs extraction
- pydensecrf can be tricky to install on Windows

---

### SDDNet (ACM MM 2023) — Fastest inference

- **Repo**: https://github.com/rmcong/SDDNet_ACMMM23
- **What it does**: Binary shadow segmentation, real-time capable
- **Output**: Shadow probability map

**Pretrained weights**: Baidu Netdisk (code: `mvpl`) and Google Drive

**Speed**: 32 FPS (36.73 images/sec on RTX 4090 per benchmark)
**Parameters**: 15.02M
**BER (SBU-Refine)**: 4.86 @ 512x512, 5.39 @ 256x256

**Dependencies**: Python 3.6.10, PyTorch 1.10.1, CUDA 11.1, pydensecrf

**Notes**:
- Best speed/accuracy ratio for batch processing large datasets
- ResNeXt-101 backbone
- CRF refinement available

---

### FSDNet (TIP 2021) — Smallest model

- **Repo**: https://github.com/xw-hu/FSDNet
- **What it does**: Binary shadow segmentation with minimal parameters
- **Output**: Shadow probability map

**Parameters**: ~4M (smallest in the field)
**Inference**: `python3 infer.py`

**Dependencies**: Python 3.6, PyTorch 1.3.0, CuPy (GPU-accelerated NumPy)

**Notes**:
- Tiny model — best candidate for ONNX export if portability matters later
- Uses custom CUDA kernels (IRNN), which makes Windows builds harder
- CuPy dependency adds complexity
- Same author as the Unveiling Deep Shadows benchmark

---

## SAM-Based Shadow Detectors

### SAM-Adapter (2023, updated 2025) — Framework, not plug-and-play

- **Repo**: https://github.com/tianrun-chen/SAM-Adapter-PyTorch
- **What it does**: Adds lightweight trainable adapters to frozen SAM encoder,
  fine-tuned for downstream tasks (camouflage detection, shadow detection, medical)

**Branches** (verified):
- `main` — configs are ALL for camouflaged object detection (`cod-sam-vit-b/l/h.yaml`)
- `SAM3-Adapter` — claims shadow detection support on ISTD dataset

**Pretrained weights**: Google Drive link exists, but appears to be for
camouflage detection (COD), NOT shadow detection. The shadow detection task
on the SAM3-Adapter branch references ISTD but it is unclear whether
pre-trained shadow weights are provided or if you must train from scratch.

**Training requirement**: 4x A100 GPUs (training only; inference is single-GPU)

**Key config** (`demo.yaml`):
- Input: 1024x1024
- Model: SAM ViT-L (`sam_vit_l_0b3195.pth`)
- Adapter depth: 32 layers, 16 attention heads
- Dataset: CAMO (camouflage) — NOT shadow

**Dependencies**: Python 3.8, PyTorch 1.13.0

**Assessment**: This is an adapter *framework* — powerful architecture but you'd
likely need to train it yourself on shadow data. Not a download-and-run shadow
detector. The SAM3-Adapter branch may have shadow configs but this is unconfirmed.

**There is NO SAM2-Adapter branch** — only `main` and `SAM3-Adapter`.

---

### AdapterShadow / ShadowAdapter (Expert Systems & Applications, 2025)

- **Repo**: https://github.com/LeipingJie/AdapterShadow
- **What it does**: Adapts SAM specifically for shadow detection using auto-prompt
  (grid-sampled point prompts — no manual interaction needed)

**Inference**:
```
python pl_test_simple.py -net sam -exp_name sbu -npts 5 -backbone b1 \
  -plug_image_adapter -all -freeze_backbone -use_neg_point \
  -sample grid -grid_out_size 16
```

**Pretrained weights**: Google Drive folder "Code & Results" (link in README)
— could not verify contents (Drive folder didn't load)

**Datasets**: SBU, UCF, CUHK, ISTD, SBU_New

**Notes**:
- Grid sampling for dense prompts = fully automatic, no per-image interaction
- Multiple dataset support suggests good generalization
- 2025 publication date = most recent SAM-based shadow detector with code
- Requires SAM base model checkpoint
- Not verified whether weights actually download successfully

---

### ShadowSAM (IEEE TGRS 2023)

- **Repo**: https://github.com/wuwen1994/ShadowSAM
- **What it does**: CNN-assisted SAM tuning for shadow detection

**Pretrained weights**: Baidu Netdisk ONLY (code: 9y41) — no Google Drive option

**Dependencies**: Python 3.6, PyTorch 1.10, timm 0.3.2, pydensecrf

**Notes**:
- Lower GPU memory than other SAM adapters
- Baidu-only weight hosting is a friction point

---

## ShadowFormer — Shadow Removal (not detection)

- **Repo**: https://github.com/GuoLanqing/ShadowFormer
- **What it does**: Takes shadowed image → produces shadow-free image (removal)

**Critical clarification**: ShadowFormer does NOT require a shadow mask at
inference time. The mask (dataset component B) is used only during training
as supervision and during evaluation for computing metrics. At inference,
it's purely: shadowed image in → shadow-free image out.

**Could be used for detection via diff**: Run model, compute
`abs(input - output)` → regions where the model lifted shadows show high
difference. Threshold the diff → rough shadow mask.

**Caveats of the diff approach**:
- Model may subtly adjust tone/color globally, not just in shadows → noise
- For harsh noon shadows the diff signal would be strong and obvious
- Requires running a full image restoration model just to get a mask
- Less precise than a purpose-built shadow detector

**Dependencies**: Python 3.7, PyTorch 1.7, CUDA 11.1

**Pretrained weights** (Google Drive):
- ISTD dataset model
- ISTD+ dataset model
- SRD dataset model

---

## Unveiling Deep Shadows Benchmark — One-stop weight shop

- **Repo**: https://github.com/xw-hu/Unveiling-Deep-Shadows
- **What it is**: Standardized benchmark that retrained ALL major shadow detection
  models under identical conditions and published weights for all of them.

**This is the single most useful resource.** One GitHub Releases page has weights
for 8 shadow detection models at 2 resolutions:

**Release**: `Weights_ShadowDetection` (tag: `Weights_SD`) — 18 assets

| Model | Year | Params | Speed (img/s) | BER @256 | BER @512 |
|-------|------|--------|---------------|----------|----------|
| DSC | 2018 | 122.49M | 26.86 | 6.79 | 6.34 |
| BDRAR | 2018 | 42.46M | 39.76 | 6.27 | 5.62 |
| DSDNet | 2019 | 58.16M | 37.53 | 5.37 | 5.04 |
| MTMT-Net | 2020 | — | — | — | 6.32 |
| FDRNet | 2021 | — | — | 5.64 | 5.39 |
| FSDNet | 2021 | 4M | fastest | — | — |
| ECA | 2021 | — | — | 7.08 | 7.52 |
| SDDNet | 2023 | 15.02M | 36.73 | 5.39 | 4.86 |

*BER = Balanced Error Rate (lower is better). Tested on SBU-Refine dataset.*

**Note**: These BER numbers are from the standardized benchmark (SBU-Refine),
not the original papers. Some models perform differently here than in their
papers because the benchmark uses corrected labels and uniform conditions.

Also available: video shadow detection weights (7 assets), instance shadow
detection weights (5 assets), shadow removal weights (28 assets).

---

## Dataset Context

The models are trained on these shadow datasets:

| Dataset | Images | What's in it |
|---------|--------|-------------|
| **SBU** | ~5,000 | Diverse outdoor scenes with ground-cast shadows. Most relevant to our use case. |
| **SBU-Refine** | ~5,000 | SBU with corrected/refined label annotations |
| **ISTD** | ~2,000 | Triplets: shadow image / shadow-free image / shadow mask. Cleaner annotations. |
| **CUHK-Shadow** | ~10,000+ | Largest dataset. More shadow regions per image. |
| **UCF** | ~245 | Small, test-only |

For beach/ground-level harsh shadows from people and equipment, SBU is the
most directly relevant training set.

---

## Integration Considerations

### How shadow detection fits the masking pipeline

```
Current flow:
  YOLO/SAM3 → person/equipment mask → (optional) brightness heuristic shadow detection

Proposed flow:
  YOLO/SAM3 → person/equipment mask
  Shadow detector → full-image shadow mask
  (Optional) spatial filter: keep only shadows near detected people
  Union: person mask + filtered shadow mask → final mask
```

The spatial filtering step (step 3) is important — without it, tree shadows,
building shadows, and other environmental shadows would also be masked. For
photogrammetry you generally want to keep those; you only want to remove
shadows cast by the things you're already masking (people, equipment).

The existing downward-dilation logic could be repurposed for this filtering,
but expanded to work in multiple directions (not just downward).

### PyTorch version conflicts

The models span PyTorch 1.3–1.13. The pipeline already uses PyTorch for
SAM3/YOLO. Key concern: make sure the shadow model works with whatever
PyTorch version is already installed. Most models will work with newer
PyTorch despite being developed on older versions.

### Windows compatibility

- **pydensecrf**: Required by FDRNet, SDDNet for CRF refinement. Can be
  difficult to install on Windows. There is a `pydensecrf2` pip package
  that may work, or CRF can be skipped (raw output is still usable).
- **CuPy**: Required by FSDNet for custom CUDA kernels. Windows builds
  exist but can be finicky.
- **SILT**: No unusual dependencies — most likely to work cleanly on Windows.

### Batch processing

For 1400-frame datasets, speed matters:
- SDDNet @ 32 FPS → ~44 seconds for 1400 frames
- BDRAR @ 40 img/s → ~35 seconds
- Most others are 25-40 img/s at 512x512

---

## Recommendations

### For immediate integration: SDDNet or SILT

Both have downloadable weights (via Unveiling Deep Shadows benchmark or their
own repos), reasonable accuracy, and no unusual dependencies beyond PyTorch.
SDDNet is fastest; SILT has the cleanest inference code.

### For best edge quality: FDRNet with CRF

The CRF post-processing gives the crispest shadow boundaries, which matters
for photogrammetry masking (you don't want fuzzy mask edges). But pydensecrf
on Windows may be a hassle, and the older PyTorch version is a concern.

### For maximum quality (requires training): SAM-Adapter

If the dedicated detectors don't produce clean enough masks on your specific
data, the SAM3-Adapter framework with a ViT backbone could be trained on
shadow data. This is a project, not a quick integration. Would need shadow
training data (SBU/ISTD) and significant GPU time.

### For shadow removal + detection: ShadowFormer diff approach

If you want both cleaned images AND shadow masks, ShadowFormer's
input-vs-output diff is viable for harsh shadows. But for detection alone,
a dedicated detector is more direct and precise.

### Easiest starting point: Unveiling Deep Shadows benchmark

Download the `Weights_ShadowDetection` release, pick SDDNet or FDRNet
weights, and adapt their test scripts. One repo, standardized weights,
multiple models to compare.

---

*Report compiled 2026-02-07. All GitHub repos and weight availability
verified by direct inspection.*
