# Reconstruction Zone

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)
![Platform: Windows](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)
![CUDA](https://img.shields.io/badge/CUDA-Recommended-76B900.svg)

![Before and after masking](reconstruction_gui/docs/assets/hero_before_after.jpg)

> Photogrammetry preprocessing toolkit — extract frames from video, reframe 360° and fisheye into perspective views, auto-mask unwanted objects, and build clean datasets for 3D reconstruction.

This GUI prepares camera captures for 3D reconstruction. Extract and reframe perspectives from 360° or fisheye video, filter for sharpness, auto-detect and mask photographers/tripods/equipment across hundreds of images, review every mask with an interactive editor, then analyze spatial coverage and fill gaps. Outputs datasets ready for reconstruction pipelines.

## Requirements

- Python 3.10+ (tested on 3.12–3.14)
- NVIDIA GPU with CUDA (strongly recommended)
- [ffmpeg + ffprobe](https://ffmpeg.org/download.html) on PATH (for video features)

## Installation

**1. Request SAM 3 model access** (do this first — approval may take hours)

SAM 3 is the primary masking model. It accepts natural language prompts ("photographer with tripod", "selfie stick", "camera rig") and produces the highest quality masks. Weights are gated on HuggingFace — request access now so it's ready when you need it.

Go to [facebook/sam3 on HuggingFace](https://huggingface.co/facebook/sam3) and click **Request access**. You'll need a free HuggingFace account.

**2. Install dependencies**

```bash
# PyTorch with CUDA (CPU works but is 10-50x slower)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Core + GUI
pip install numpy opencv-python ultralytics tqdm pyyaml customtkinter

# SAM 3
pip install huggingface_hub
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .
```

**3. Authenticate with HuggingFace** (once your access request is approved)

```bash
huggingface-cli login
```

SAM 3 weights (~2 GB) download automatically on first run once authenticated. While waiting for approval, the app falls back to YOLO26 (class-based detection, works immediately).

Verify CUDA:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

<details>
<summary><strong>Optional extras</strong></summary>

| Package | Feature | Install |
|---------|---------|---------|
| `rfdetr`, `supervision` | RF-DETR transformer detection (ensemble partner for YOLO26) | `pip install rfdetr supervision` |
| `py360convert` | Higher-quality equirect-to-perspective reframing | `pip install py360convert` |
| `transformers` | ViTMatte alpha matting (soft mask edges) | `pip install "transformers>=4.50,<5.0"` |
| LiVOS / Cutie | Temporal mask propagation across video frames | Clone repos + `pip install -e .` |
| `efficientnet-pytorch` | Shadow detection (SDDNet model) | `pip install efficientnet-pytorch` |

</details>

## Launch

Double-click `reconstruction_gui/ReconstructionStudio.bat` to start the app (no console window).

Or from the command line:

```bash
python reconstruction_gui/reconstruction_zone.py
```

## The four tabs

The GUI is organized into four tabs that follow the photogrammetry preprocessing workflow:

| Tab | What it does | Guide |
|-----|-------------|-------|
| **Extract** | Pull frames from 360° video, fisheye, or standard video. Equirect-to-perspective reframing with configurable view rings. | [Extract Guide](reconstruction_gui/docs/EXTRACT_TAB.md) |
| **Mask** | Auto-detect and mask objects using text prompts or class selection. Supports 360°-aware cubemap decomposition. | [Mask Guide](reconstruction_gui/docs/MASK_TAB.md) |
| **Review** | Thumbnail grid with accept/reject/skip workflow. Open any mask in the interactive editor for brush, flood fill, and lasso touch-ups. | [Review Guide](reconstruction_gui/docs/REVIEW_TAB.md) |
| **Coverage** | Analyze spatial coverage gaps in your dataset and extract bridge frames to fill them. | [Coverage Guide](reconstruction_gui/docs/COVERAGE_TAB.md) |

<details>
<summary><strong>Supported models</strong></summary>

| Model | Type | Speed | Best for |
|-------|------|-------|----------|
| **SAM 3** | Text-prompted | ~300ms/img | Highest quality, arbitrary objects ("selfie stick", "tripod") |
| **YOLO26** | Class-based | ~15ms/img | Fast batch processing, COCO objects (person, backpack, car) |
| **RF-DETR** | Transformer | ~50ms/img | Strong detection + segmentation in one pass |
| **FastSAM** | Real-time SAM | ~30ms/img | Quick previews, lightweight |
| **EfficientSAM** | Lightweight SAM | ~40ms/img | Fallback when others unavailable |

YOLO26, FastSAM, and RF-DETR weights auto-download on first use. SAM 3 requires [HuggingFace access](#installation) first. See the full [Model Guide](reconstruction_gui/docs/MODELS.md) for configuration, model sizes, and comparison.

</details>

## Documentation

- [Quickstart](reconstruction_gui/docs/QUICKSTART.md) — First mask in 5 minutes
- [Architecture](reconstruction_gui/docs/ARCHITECTURE.md) — Pipeline internals, data flow, module map
- [Model Guide](reconstruction_gui/docs/MODELS.md) — Model comparison, configuration, COCO class reference
- [Contributing](reconstruction_gui/docs/CONTRIBUTING.md) — Adding new models and modules

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
