# Reconstruction Zone — Dependency License Audit

> **Last updated:** 2026-03-03
> **Audited by:** Claude Code (Opus 4.6)
> **Project license:** GPL-3.0

This document provides a comprehensive audit of every third-party tool, library, and model used by the Reconstruction Zone project, along with their licenses, compatibility notes, and actionable flags.

---

## Table of Contents

- [This Project](#this-project)
- [Core Python Dependencies](#core-python-dependencies)
  - [Deep Learning Framework](#deep-learning-framework)
  - [Computer Vision & Image Processing](#computer-vision--image-processing)
  - [GUI Framework](#gui-framework)
  - [Utilities & Support Libraries](#utilities--support-libraries)
- [AI/ML Model Repositories](#aiml-model-repositories-cloned-into-models)
- [Detection & Segmentation Libraries](#detection--segmentation-libraries-pip-installed)
- [External CLI Tools](#external-cli-tools)
- [Transitive Dependencies of Note](#transitive-dependencies-of-note)
- [License Compatibility Matrix](#license-compatibility-matrix)
- [Flags Requiring Attention](#flags-requiring-attention)
  - [Ultralytics (AGPL-3.0)](#1-ultralytics-agpl-30)
  - [SAM 3 (SAM License)](#2-sam-3-sam-license)
  - [FFmpeg (LGPL/GPL)](#3-ffmpeg-lgplgpl)
  - [Metashape Pro (Proprietary)](#4-metashape-pro-proprietary)
- [Clean Dependencies](#clean-dependencies-no-concerns)
- [Recommendations](#recommendations)
- [How to Verify](#how-to-verify)
- [Sources](#sources)

---

## This Project

| Field | Value |
|-------|-------|
| **Project** | Reconstruction Zone |
| **License** | GNU General Public License v3.0 (GPL-3.0) |
| **License file** | [`LICENSE`](../LICENSE) |
| **Copyright** | See LICENSE file |

The GPL-3.0 is a strong copyleft license. All dependencies must be compatible with GPL-3.0 for the combined work to be distributable.

---

## Core Python Dependencies

### Deep Learning Framework

| Package | PyPI Name | License | OSI Approved | Copyleft | Used By | Import Path |
|---------|-----------|---------|:------------:|:--------:|---------|-------------|
| PyTorch | `torch` | BSD-3-Clause | Yes | No | All ML pipelines | `import torch` |
| TorchVision | `torchvision` | BSD-3-Clause | Yes | No | Image transforms, model loading | `import torchvision` |
| TorchAudio | `torchaudio` | BSD-2-Clause | Yes | No | SAM3 dependency | `import torchaudio` |
| Triton | `triton-windows` | MIT | Yes | No | GPU kernel compilation (Windows) | `import triton` |

**Install:** `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`

**Notes:**
- PyTorch is installed with CUDA 12.6 support on this system (RTX 3090 Ti).
- `triton-windows` is the Windows-compatible fork of OpenAI's Triton compiler. The upstream `triton` package does not support Windows.

---

### Computer Vision & Image Processing

| Package | PyPI Name | License | OSI Approved | Copyleft | Used By | Import Path |
|---------|-----------|---------|:------------:|:--------:|---------|-------------|
| NumPy | `numpy` | BSD-3-Clause | Yes | No | Everywhere — array ops, mask manipulation | `import numpy` |
| OpenCV | `opencv-python` | Apache 2.0 | Yes | No | Image I/O, morphology, contours, drawing | `import cv2` |
| Pillow | `Pillow` | HPND (Historical Permission Notice and Disclaimer) | Yes | No | Image I/O for model APIs (SAM, RF-DETR) | `from PIL import Image` |
| scikit-image | `scikit-image` | BSD-3-Clause | Yes | No | Advanced image processing (optional) | `import skimage` |
| SciPy | `scipy` | BSD-3-Clause | Yes | No | Distance transforms, morphology, interpolation | `import scipy` |

**Notes:**
- Pillow's HPND license is functionally equivalent to MIT — fully permissive, no copyleft.
- OpenCV is the `opencv-python` headless or GUI package (not `opencv-contrib-python`, which bundles modules with potentially different licenses).

---

### GUI Framework

| Package | PyPI Name | License | OSI Approved | Copyleft | Used By | Import Path |
|---------|-----------|---------|:------------:|:--------:|---------|-------------|
| CustomTkinter | `customtkinter` | MIT | Yes | No | Masking Studio GUI | `import customtkinter` |
| Tkinter | (stdlib) | PSF License | Yes | No | Underlying toolkit (bundled with Python) | `import tkinter` |

**Notes:**
- CustomTkinter wraps Tkinter with modern widget styling. The entire GUI stack is MIT/PSF — no copyleft concerns.
- Repository: [TomSchimansky/CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)

---

### Utilities & Support Libraries

| Package | PyPI Name | License | OSI Approved | Copyleft | Used By | Import Path |
|---------|-----------|---------|:------------:|:--------:|---------|-------------|
| py360convert | `py360convert` | MIT | Yes | No | Equirect-to-cubemap conversion (optional, fallback exists) | `import py360convert` |
| matplotlib | `matplotlib` | PSF-based (BSD-compatible) | Yes | No | Plotting, visualization | `import matplotlib` |
| tqdm | `tqdm` | MPL-2.0 + MIT | Yes | No | Progress bars | `from tqdm import tqdm` |
| PyYAML | `pyyaml` | MIT | Yes | No | Config file parsing | `import yaml` |
| requests | `requests` | Apache 2.0 | Yes | No | HTTP requests (model downloads) | `import requests` |
| einops | `einops` | MIT | Yes | No | Tensor reshaping (model code) | `import einops` |

**Notes:**
- `py360convert` is optional — the reframer has a custom pure-NumPy fallback for equirect-to-perspective conversion.
- `tqdm` dual-licenses under MPL-2.0 and MIT. The MIT license alone is sufficient for GPL-3.0 compatibility.

---

## AI/ML Model Repositories (cloned into `models/`)

These are full Git repositories cloned locally and installed in development mode (`pip install -e .`).

### SAM 3 (Segment Anything Model 3)

| Field | Value |
|-------|-------|
| **Repository** | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) |
| **License** | SAM License (custom Meta license, updated November 19, 2025) |
| **OSI Approved** | **No** |
| **Copyleft** | Partial — redistribution must preserve SAM License |
| **Commercial Use** | Permitted with conditions |
| **Local Path** | `models/sam3/` |
| **Used By** | `masking_v2.py`, `sam3_pipeline.py` |

**Key License Terms:**
- Grants non-exclusive, worldwide, royalty-free license to use, reproduce, distribute, and create derivative works.
- **Redistribution requirement:** Any distribution of SAM Materials or derivatives must include a copy of the SAM License.
- **No trademark rights:** Cannot use Meta's name or trademarks to endorse derivatives.
- **Trade controls:** Subject to US export/sanctions restrictions.
- **No warranty:** Provided "as is."
- **Meta can update terms:** The license may be updated; continued use constitutes acceptance.

**Compatibility with GPL-3.0:** The SAM License is permissive enough for personal/research use alongside GPL-3.0 code. However, the redistribution clause (must keep SAM License) and Meta's ability to update terms create tension with GPL-3.0's requirement that no additional restrictions be imposed. For personal use this is not an issue; for public distribution of the combined work, legal review is recommended.

---

### EfficientSAM

| Field | Value |
|-------|-------|
| **Repository** | [yformer/EfficientSAM](https://github.com/yformer/EfficientSAM) |
| **License** | Apache 2.0 |
| **OSI Approved** | Yes |
| **Copyleft** | No |
| **Commercial Use** | Yes |
| **Local Path** | `models/EfficientSAM/` |
| **Used By** | `masking_v2.py` (fallback segmenter) |

**Compatibility with GPL-3.0:** Fully compatible. Apache 2.0 is a permissive license that is one-way compatible with GPL-3.0.

---

### LiVOS

| Field | Value |
|-------|-------|
| **Repository** | [uncbiag/LiVOS](https://github.com/uncbiag/LiVOS) |
| **License** | MIT |
| **OSI Approved** | Yes |
| **Copyleft** | No |
| **Commercial Use** | Yes |
| **Local Path** | `models/LiVOS/` |
| **Used By** | `vos_propagation.py` (temporal mask propagation) |
| **Paper** | CVPR 2025 — "Light Video Object Segmentation with Gated Linear Matching" |

**Compatibility with GPL-3.0:** Fully compatible. MIT is the most permissive common license.

**Import Note:** The correct import path is `livos.model.livos_wrapper` (not `livos.model.livos`) and `livos.eval.InferenceCore` (not `livos.inference.InferenceCore`).

---

### Cutie

| Field | Value |
|-------|-------|
| **Repository** | [hkchengrex/Cutie](https://github.com/hkchengrex/Cutie) |
| **License** | MIT |
| **OSI Approved** | Yes |
| **Copyleft** | No |
| **Commercial Use** | Yes |
| **Local Path** | `models/Cutie/` |
| **Used By** | `vos_propagation.py` (alternative VOS backend) |
| **Paper** | CVPR 2024 Highlight — "Putting the Object Back Into Video Object Segmentation" |

**Compatibility with GPL-3.0:** Fully compatible.

**Note:** The `gui/ritm` subdirectory follows RITM's own license. This code is not used by Reconstruction Zone (only the core VOS inference is used).

---

## Detection & Segmentation Libraries (pip-installed)

### Ultralytics (YOLO26, FastSAM)

| Field | Value |
|-------|-------|
| **Package** | [ultralytics](https://pypi.org/project/ultralytics/) |
| **License** | **AGPL-3.0** (open source) or Ultralytics Enterprise License (commercial) |
| **OSI Approved** | Yes (AGPL-3.0 is OSI-approved) |
| **Copyleft** | **Yes — strong copyleft with network clause** |
| **Commercial Use** | Requires Enterprise License for proprietary use |
| **Used By** | `masking_v2.py` (YOLO26 segmentation), `prep360/core/segmenter.py` |
| **Import Path** | `from ultralytics import YOLO` |

**Key License Terms (AGPL-3.0):**
- Any distribution of the software or derivative works must be under AGPL-3.0.
- **Network use = distribution:** If you run AGPL software as a network service, users interacting over the network must be offered the source code. (Not currently relevant — this is a desktop app.)
- Modified works must be open-sourced under AGPL-3.0.
- All code that forms a "combined work" with AGPL code inherits the AGPL obligation.

**Enterprise License Alternative:**
Ultralytics offers a paid Enterprise License that removes the AGPL-3.0 obligations. Required for:
- Proprietary/closed-source commercial products
- Internal company use without open-sourcing
- SaaS deployment without source disclosure

**Compatibility with GPL-3.0:** AGPL-3.0 and GPL-3.0 are compatible per the FSF, but the combined work must be distributed under **AGPL-3.0** (the stricter license). This means your project, when distributed, technically should carry the AGPL-3.0 notice for the combined work — or ultralytics must be made optional/separable.

---

### RF-DETR

| Field | Value |
|-------|-------|
| **Package** | [rfdetr](https://github.com/roboflow/rf-detr) |
| **License** | Apache 2.0 |
| **OSI Approved** | Yes |
| **Copyleft** | No |
| **Commercial Use** | Yes |
| **Used By** | `masking_v2.py` (RF-DETR detector) |
| **Import Path** | `from rfdetr import RFDETRSegSmall` |

**Note:** The base RF-DETR models are Apache 2.0. Roboflow also offers **RF-DETR+** under a proprietary "Platform Model License 1.0" (PML-1.0) — this project does **not** use RF-DETR+.

**Compatibility with GPL-3.0:** Fully compatible.

**Dependency Note:** RF-DETR requires `transformers < 5.0` (pinned to 4.57.x). The HuggingFace Transformers library itself is Apache 2.0.

---

### supervision

| Field | Value |
|-------|-------|
| **Package** | [supervision](https://github.com/roboflow/supervision) |
| **License** | MIT |
| **OSI Approved** | Yes |
| **Copyleft** | No |
| **Used By** | RF-DETR result handling |

---

## External CLI Tools

### FFmpeg / FFprobe

| Field | Value |
|-------|-------|
| **Tool** | [FFmpeg](https://ffmpeg.org/) (includes ffprobe) |
| **License** | **LGPL 2.1+** (default) or **GPL 2.0+** (if compiled with `--enable-gpl`) |
| **Copyleft** | Yes (LGPL or GPL depending on build) |
| **Used By** | `prep360/core/extractor.py`, `prep360/core/analyzer.py`, `prep360/core/osv.py`, `prep360/core/sharpest_extractor.py` |
| **Invocation** | `subprocess.run()` — process isolation, no linking |

**Key Details:**
- FFmpeg's license depends on **how it was compiled:**
  - Default build: **LGPL 2.1+** — linking restrictions apply, but subprocess invocation is not "linking."
  - With `--enable-gpl`: **GPL 2.0+** — stronger copyleft, but again subprocess-isolated.
  - With `--enable-nonfree`: Not redistributable at all (includes proprietary codec support).
- Most pre-built Windows binaries (gyan.dev, BtbN GitHub releases) are **GPL builds** because they enable x264, x265, and other GPL-licensed codecs.
- Since Reconstruction Zone invokes FFmpeg via `subprocess.run()` (not linking or embedding), **license isolation applies** — FFmpeg's license does not propagate to the calling Python code. This is the standard interpretation per the FSF.

**To verify your FFmpeg build's license:** Run `ffmpeg -version` and check the `configuration:` line for `--enable-gpl` or `--enable-nonfree` flags.

---

### COLMAP

| Field | Value |
|-------|-------|
| **Tool** | [COLMAP](https://colmap.github.io/) |
| **License** | BSD-3-Clause (new BSD) |
| **Copyleft** | No |
| **Used By** | `prep360/core/colmap_export.py` (data export target — COLMAP is not invoked directly) |

**Notes:** Reconstruction Zone writes data files **for** COLMAP but does not invoke COLMAP as a subprocess. COLMAP is an optional downstream consumer of the pipeline output.

---

### Metashape Pro

| Field | Value |
|-------|-------|
| **Tool** | [Agisoft Metashape Professional](https://www.agisoft.com/) |
| **License** | **Proprietary** (commercial, per-seat) |
| **Used By** | `metashape_exporter.py` (runs inside Metashape via Tools > Run Script) |

**Notes:**
- `metashape_exporter.py` is a Metashape Python script that runs **inside** the Metashape application — it is not imported by or bundled with Reconstruction Zone's pipeline.
- Metashape is entirely optional. The COLMAP export path (`prep360/core/colmap_export.py`) provides a free alternative for camera pose extraction.
- No Metashape libraries are imported into the main codebase. The exporter script uses Metashape's internal `Metashape` Python module, which is only available inside the application.

---

## Transitive Dependencies of Note

These are not directly imported by Reconstruction Zone but are pulled in by the model repositories or pip packages listed above.

| Package | License | Pulled In By | Notes |
|---------|---------|-------------|-------|
| HuggingFace Transformers | Apache 2.0 | RF-DETR, ViTMatte (matting.py) | Pinned to < 5.0 for RF-DETR |
| timm | Apache 2.0 | SAM3 | PyTorch Image Models registry |
| OmegaConf | BSD-3-Clause | LiVOS, Cutie | Configuration management |
| Hydra | MIT | LiVOS, Cutie | Configuration framework |
| huggingface_hub | Apache 2.0 | transformers, timm | Model download utilities |
| iopath | MIT | SAM3 | Facebook's I/O path abstraction |
| fvcore | Apache 2.0 | SAM3 | Facebook Vision Core utilities |
| albumentations | MIT | EfficientSAM | Data augmentation (training only) |
| pycocotools | BSD-2-Clause | Ultralytics, RF-DETR | COCO evaluation metrics |
| pandas | BSD-3-Clause | Ultralytics | Data analysis |
| scikit-learn | BSD-3-Clause | Various | ML utilities |
| tensorboard | Apache 2.0 | Training scripts | Logging (not used at inference) |

All transitive dependencies listed above use permissive licenses fully compatible with GPL-3.0.

---

## License Compatibility Matrix

This matrix shows whether each dependency's license is compatible with the project's GPL-3.0 license.

| License | Compatible with GPL-3.0? | Direction | Dependencies Using It |
|---------|:-----------------------:|-----------|----------------------|
| MIT | Yes | One-way (MIT → GPL OK) | CustomTkinter, LiVOS, Cutie, py360convert, einops, PyYAML, supervision, Hydra, iopath |
| BSD-2-Clause | Yes | One-way | torchaudio, pycocotools |
| BSD-3-Clause | Yes | One-way | PyTorch, torchvision, NumPy, SciPy, scikit-image, OmegaConf, pandas, scikit-learn |
| Apache 2.0 | Yes | One-way (Apache 2.0 → GPL-3.0 OK) | OpenCV, RF-DETR, EfficientSAM, Transformers, timm, requests, fvcore, tensorboard |
| HPND | Yes | One-way | Pillow |
| PSF | Yes | One-way | Python stdlib, matplotlib |
| **AGPL-3.0** | **Yes, but...** | Combined work must be AGPL-3.0 | **Ultralytics** |
| **SAM License** | **Uncertain** | Custom terms may conflict | **SAM 3** |
| LGPL 2.1+ | Yes | Subprocess isolation | FFmpeg (default build) |
| GPL 2.0+ | Yes | Compatible with GPL-3.0 | FFmpeg (GPL build, subprocess-isolated) |
| **Proprietary** | **N/A** | Not bundled | Metashape Pro (external tool, optional) |

---

## Flags Requiring Attention

### 1. Ultralytics (AGPL-3.0)

**Risk Level: Medium**

This is the most restrictive open-source dependency in the project.

**What AGPL-3.0 means for you:**
- If you distribute Reconstruction Zone (e.g., publish on GitHub, share binaries), the combined work that includes Ultralytics must be available under AGPL-3.0.
- GPL-3.0 is compatible with AGPL-3.0, but the **combined work** must carry AGPL-3.0 obligations. In practice, this means:
  - Your GPL-3.0 LICENSE file covers your own code.
  - But the distributed whole (your code + ultralytics) is governed by AGPL-3.0 terms.
- The **network clause** (AGPL Section 13) means if you ever serve this as a web service, users must be offered source access. Currently irrelevant for a desktop app.
- For proprietary/commercial use, an [Ultralytics Enterprise License](https://www.ultralytics.com/license) is required.

**Mitigation options:**
1. Switch project license to AGPL-3.0 (simplest, most honest).
2. Make ultralytics optional with a runtime check and fallback to RF-DETR or other Apache-2.0 detectors.
3. Purchase an Enterprise License for commercial use.

---

### 2. SAM 3 (SAM License)

**Risk Level: Low-Medium**

**What the SAM License means for you:**
- You can use, modify, and redistribute SAM 3 freely.
- Redistribution must include the SAM License text.
- Meta retains the right to update the license terms.
- The license is **not OSI-approved** and is not a standard open-source license.

**Potential GPL-3.0 tension:**
- GPL-3.0 Section 7 states you cannot impose "additional restrictions" beyond what the GPL allows.
- The SAM License's requirement to redistribute under the SAM License (not GPL) and Meta's right to update terms could be considered "additional restrictions."
- For a personal/research project, this is academic. For formal distribution, this warrants legal review.

**Mitigation:** SAM 3 is already optional (lazy-imported, only loaded if the user selects it). The pipeline works with YOLO alone. Document SAM 3 as a separate, independently-licensed component.

---

### 3. FFmpeg (LGPL/GPL)

**Risk Level: Low**

**Why it's low risk:**
- Reconstruction Zone invokes FFmpeg via `subprocess.run()` — this is **process isolation**, not linking.
- The FSF and legal consensus hold that invoking a separate process does not create a "combined work" under GPL/LGPL terms.
- FFmpeg is a system tool the user installs independently; it is not bundled with or distributed by this project.

**Action:** No changes needed. Document that FFmpeg must be installed separately by the user.

---

### 4. Metashape Pro (Proprietary)

**Risk Level: None**

- `metashape_exporter.py` runs **inside** Metashape's own Python environment.
- No Metashape code or libraries are imported into Reconstruction Zone.
- The script is a standalone helper, not a core dependency.

**Action:** None needed.

---

## Clean Dependencies (No Concerns)

All of the following use permissive licenses (MIT, BSD, Apache 2.0) that are fully compatible with GPL-3.0 and impose no copyleft or redistribution obligations beyond attribution:

- PyTorch, TorchVision, TorchAudio (BSD)
- NumPy, SciPy, scikit-image, scikit-learn, pandas (BSD)
- OpenCV (Apache 2.0)
- Pillow (HPND)
- CustomTkinter (MIT)
- py360convert (MIT)
- matplotlib (PSF/BSD)
- tqdm (MIT)
- PyYAML (MIT)
- requests (Apache 2.0)
- einops (MIT)
- supervision (MIT)
- RF-DETR (Apache 2.0)
- EfficientSAM (Apache 2.0)
- LiVOS (MIT)
- Cutie (MIT)
- HuggingFace Transformers (Apache 2.0)
- timm (Apache 2.0)
- Triton (MIT)
- COLMAP (BSD-3-Clause)

---

## Recommendations

### For Personal/Research Use (current state)

No changes required. All dependencies are usable as-is for personal projects and academic research.

### For Public Distribution on GitHub

1. **Consider switching to AGPL-3.0** for the project license. Since Ultralytics (AGPL-3.0) is a core dependency, the combined distributed work is technically governed by AGPL-3.0 terms anyway. Making this explicit avoids confusion.

2. **Add a `THIRD_PARTY_LICENSES` notice** listing all dependencies and their licenses. This satisfies attribution requirements for Apache 2.0, BSD, and MIT licensed dependencies.

3. **Document SAM 3 as independently licensed.** Note in the README or a NOTICE file that `models/sam3/` is covered by Meta's SAM License, not the project's GPL/AGPL license.

### For Commercial Distribution

1. **Purchase an Ultralytics Enterprise License** — required for any proprietary use of YOLO models.
2. **Review SAM 3 License terms** with legal counsel before commercial distribution.
3. **Verify FFmpeg build** — ensure you're not using a `--enable-nonfree` build if redistributing.
4. **RF-DETR base models are fine** (Apache 2.0) — do not use RF-DETR+ without a Roboflow account and PML-1.0 compliance.

---

## How to Verify

### Check FFmpeg license on your system

```bash
ffmpeg -version
# Look for: --enable-gpl, --enable-nonfree in the configuration line
```

### Check installed Python package licenses

```bash
pip install pip-licenses
pip-licenses --format=table --with-license-file --no-license-path
```

### Check local model repository licenses

```bash
cat models/sam3/LICENSE        # SAM License (custom)
cat models/EfficientSAM/LICENSE  # Apache 2.0
cat models/LiVOS/LICENSE       # MIT
cat models/Cutie/LICENSE       # MIT
```

### Verify ultralytics license

```bash
pip show ultralytics
# License: AGPL-3.0
```

---

## Sources

- [Ultralytics Licensing](https://www.ultralytics.com/license)
- [AGPL-3.0 Full Text](https://www.gnu.org/licenses/agpl-3.0.en.html)
- [GPL-3.0 Full Text](https://www.gnu.org/licenses/gpl-3.0.en.html)
- [SAM 3 License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- [SAM 3 License Explainer](https://sam3ai.com/license/)
- [FFmpeg Legal](https://ffmpeg.org/legal.html)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [EfficientSAM GitHub](https://github.com/yformer/EfficientSAM)
- [LiVOS GitHub](https://github.com/uncbiag/LiVOS)
- [Cutie GitHub](https://github.com/hkchengrex/Cutie)
- [CustomTkinter GitHub](https://github.com/TomSchimansky/CustomTkinter)
- [py360convert PyPI](https://pypi.org/project/py360convert/)
- [FSF License Compatibility](https://www.gnu.org/licenses/license-compatibility.en.html)
- [COLMAP GitHub](https://github.com/colmap/colmap)
- [Agisoft Metashape](https://www.agisoft.com/)
