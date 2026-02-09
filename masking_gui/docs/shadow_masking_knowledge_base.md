# Shadow Detection & Masking for Photogrammetry — Definitive Knowledge Base

*Synthesized from eight independent research reports across two rounds of cross-model polling (original survey, Claude, Gemini, Grok, ChatGPT/PDF Round 1; ChatGPT/PDF Round 2, Gemini Round 2, Grok Round 2) and cross-validated against primary sources. Compiled 2026-02-07, updated 2026-02-08.*

---

## Executive Summary

Eight independent research efforts converge on a clear consensus: **no major photogrammetry tool has built-in shadow detection**, but all accept external masks, making the integration gap the primary practical problem rather than the absence of good detection models. The field has matured rapidly since 2023, with viable pretrained models (SILT, SDDNet, ShadowAdapter), foundation-model adaptations (SAM2-Adapter), and neural reconstruction methods (appearance-embedding 3DGS) all offering practical paths forward. For the specific scenario of harsh noon shadows from people and equipment on uniform outdoor surfaces, the strongest validated approach combines a learned detector with classical chromaticity verification, applied through the reconstruction software's existing mask import interface.

### Consensus Points (All Sources Agree)

- **SDDNet** is the best speed/convenience choice for batch processing (~32 FPS, 15M params, pretrained weights available)
- **SILT** offers the best accuracy/documentation balance for single-image detection (BER ~4.2% on SBU)
- **Unveiling Deep Shadows benchmark** is the single most useful resource (unified weights for 8 models under identical conditions)
- **ShadowAdapter / AdapterShadow** (ESWA 2025) is the most production-ready SAM-based approach with automatic prompting
- **MetaShadow** (CVPR 2025) is the most notable new entry but code availability is uncertain
- **SoftShadow** (CVPR 2025) introduces penumbra-aware soft masks — cutting edge but complex to deploy
- **S-EO dataset** (CVPR Workshop 2025) is the most important new dataset for geometry-aware shadow detection
- **Domain gap** between benchmark training data and real photogrammetry scenes is the critical practical challenge
- Shadows affect **texturing most**, dense reconstruction moderately, and feature matching/alignment least
- Legacy heuristic scripts (grayscale thresholding, fixed dilations below person masks) are **confirmed obsolete** — they cannot distinguish dark albedo from shadows, miss angled shadows, and fail on penumbras
- **Hybrid approaches** (learned model + classical verification) outperform either method alone
- No fully automated end-to-end shadow masking → reconstruction pipeline exists as a packaged tool

### Key Differences Across Sources

- **Gemini** (both rounds) uniquely emphasizes acquisition-phase prevention (cross-polarization, chrome/grey ball rigs, wandering light) and post-reconstruction de-lighting (Blender division, Substance 3D Sampler, Agisoft Texture De-Lighter)
- **Claude** alone covers intrinsic image decomposition (Careaga & Aksoy) as a practical alternative path
- **ChatGPT/PDF Round 2** provides the most detailed treatment of classical chromaticity methods with a concrete implementation recipe, the removal-based masking "diff trick," soft mask weighted contributions, and environmental shadow preservation strategies
- **ChatGPT/PDF Round 2** confirms FastInstShadow as **CVPR 2025** (not just arXiv), with full architectural detail (Association Transformer Decoder)
- **Gemini Round 2** uniquely identifies Unity ArtEngine as discontinued (2025/2026) and SSD-GS as a new 4-component relightable 3DGS variant
- **Grok Round 2** adds three additional 3DGS variants (VA-GS, 3D-HGS, RadSplat) and notes global shutter camera relevance (Sony Pregius S)
- Both Round 2 PDFs now cover TICA test-time adaptation and classical methods — previously only Claude did

---

## Part 1: Why Shadows Corrupt Photogrammetry

### 1.1 The Physics of Shadow Formation

Shadows arise when an occluder blocks direct illumination from reaching a surface. The resulting region receives only indirect (ambient/skylight) illumination. For outdoor scenes under sunlight:

- **Umbra**: Fully occluded from direct sun. Receives only skylight (blue-shifted, diffuse). Intensity drops 60–90% relative to sunlit surface.
- **Penumbra**: Partially occluded (extended light source or atmospheric scattering). Gradual intensity transition at shadow boundaries. Width depends on occluder-surface distance and sun angular diameter (~0.5°).
- **Color shift**: Outdoor shadows are illuminated primarily by blue skylight rather than direct yellowish sunlight. This causes a measurable hue shift toward blue, increased saturation, and decreased value in HSV space.

The mathematical relationship: a camera captures Radiance L, which is the product of Albedo ρ and Irradiance E. In shadow, E drops and shifts spectrally:

```
L(x) ≈ ρ(x) · E(x)
```

Photogrammetry software projects raw Radiance onto the mesh. To recover true Albedo requires inverting this equation — which requires knowing E(x), the irradiance field.

### 1.2 How Shadows Break Each Pipeline Stage

**Feature Matching / SfM (Moderate Impact)**

All sources agree SIFT/RootSIFT features have inherent partial illumination invariance through gradient-based description with brightness normalization. COLMAP's RootSIFT further improves this. However:

- Shadow boundaries create strong false edges generating unreliable keypoints
- In multi-temporal datasets (drone surveys over hours), shadow positions shift — matching shadow corners across images creates false 3D tie points that stress bundle adjustment ("transient feature problem")
- In turntable scenarios, stationary shadows are treated as valid static features, potentially collapsing alignment ("stationary shadow artifact" — the solver concludes the camera never moved)
- RANSAC outlier rejection catches some but not all shadow-induced mismatches

**Dense Reconstruction / MVS (Significant Impact)**

Photo-consistency (typically measured via NCC) is directly violated by shadows:

- A surface patch shadowed in one view but sunlit in another produces dramatically different pixel intensities
- NCC drops below acceptance threshold → holes in the depth map or high-frequency noise on what should be smooth surfaces
- Persistent shadows (same position across all similar viewpoints) are worse — they are accepted as valid and create incorrect depth estimates or phantom geometry
- A 2024 study confirmed that even a basic shadow detector improved MVS results, reducing holes in reconstructed surfaces

**Texturing (Most Severe Impact)**

All sources agree this is the primary damage vector:

- Shadows are unconditionally baked into texture maps
- The resulting asset is locked to the capture lighting environment
- Multi-view blending can partially mitigate this (mosaic mode selects best-exposed pixels), but large shadows visible from all angles have no shadow-free alternative view
- Removing baked shadows post-hoc requires de-lighting — a complex additional step

### 1.3 Metashape Mask Application Strategy

Metashape's "Apply Mask to..." setting has two importantly different modes:

- **Apply Mask to Key Points**: Prevents SIFT from detecting features in masked areas. Risk: if mask is loose, you lose valid edge features → poor alignment.
- **Apply Mask to Tie Points** (recommended): Allows feature detection everywhere, but discards matches inside the mask after detection. Preserves high-contrast boundary features while filtering shadow-caused noise.

The "Mask from Model" recursion workflow (validated by community sources):
1. Draft alignment (Low accuracy, no masks)
2. Build rough mesh
3. Delete shadow/background geometry in 3D viewport
4. Generate masks from the cleaned model (projects clean 3D shape back to 2D)
5. Re-align on High accuracy with masks applied to tie points

### 1.4 Transient vs. Environmental Shadow Discrimination

A critical operational consideration from the Round 2 PDF: not all shadows should be masked. Static environmental shadows (building shadows, tree shadows) that appear consistently across all views may actually be helpful for reconstruction and should be preserved. Only transient shadows from moving objects (people, equipment) need masking.

**Spatial filtering approaches:**
- Take the convex hull around the foreground object mask and intersect with shadow mask
- Apply distance thresholds: discard shadow regions more than X pixels from any masked object
- Use instance shadow detection (FastInstShadow) which directly pairs shadows with their casters

**Formalized mask composition** (from Gemini Round 2):
```
Final Mask = (Shadow Mask ∩ Spatial Filter) ∪ Object Mask
```
Where the spatial filter is a dilated region around detected objects, ensuring only shadows cast by the subject are masked.

---

## Part 2: Acquisition-Phase Shadow Prevention

### 2.1 Cross-Polarization

The gold standard for studio/controlled photogrammetry. Polarizing film on lights + circular polarizer on lens (90° offset) blocks specular reflections. For shadow management:

- Eliminates bright specular highlights, allowing exposure for shadows without clipping
- Enables aggressive "Shadow Lifting" in RAW processing: Shadows +100, Blacks +50 (Lightroom/ACR) to neutralize ambient shadows without clipping risk, since specular highlights have already been optically removed
- Produces flat, diffuse-only images ideal for de-lighting algorithms
- **Cost**: 1.5–2 stops of light loss; fails on metallic objects (which preserve polarization and appear black)
- **Hardware**: Ring flash (Godox AR400) preferred — aligns light axis with lens axis, minimizing cast shadow length. High-transmission polarizing film for lights, high-extinction CPL for lens.
- **Heat sensitivity**: Polarizing film can melt under continuous modeling lights — use strobes (flash) instead

### 2.2 Grey Ball / Chrome Ball Reference Rig (Epic Games Method)

For outdoor scans where lighting cannot be controlled:

- **50% chrome sphere**: Captures sun direction and environmental reflection → unwrap to HDRI using Nuke, PTGui, or specialized scripts
- **18% grey sphere**: Measures lit/shadow contrast ratio ("lighting delta")
- **Macbeth ColorChecker**: Color constancy across captures
- The HDRI enables mathematically subtracting the exact captured lighting in post (Blender/Nuke division method)

### 2.3 Wandering Light / Light Painting

For static objects in dark studios:

- Long exposure (2–5 seconds) with operator moving handheld light in hemispherical pattern around object
- Shadows average out across the exposure, simulating ambient dome light (illuminated from hundreds of angles during single frame integration)
- Produces exceptionally flat, shadow-free textures
- Requires perfectly dark room and static subject

### 2.4 Practical Outdoor Mitigation (Community Consensus)

- **Overcast days** are universally preferred for photogrammetry capture
- **RAW shadow recovery**: Shadows +50 to +100, Highlights -50 to -100 in Lightroom as preprocessing
- **Capture timing**: Avoid high noon for horizontal surfaces; noon is actually best for vertical surfaces (facades)
- **Consistent lighting**: COLMAP documentation explicitly warns to "capture images at similar illumination conditions"
- **Global shutter cameras** (e.g. Sony Pregius S sensors, noted by Grok Round 2): Eliminate rolling shutter distortion for 2026 capture workflows — relevant for dynamic outdoor scenes

---

## Part 3: Shadow Detection Models — Consensus Taxonomy

### 3.1 Dedicated Single-Image Shadow Detectors

These models take RGB input → output binary or probability shadow mask. Trained on shadow-specific datasets.

**BER calibration context**: Balanced Error Rate (BER) is the average of false positive rate and false negative rate — lower is better, 0.0 is perfect. Human-level performance is estimated at ~1–2%. Early deep models from 2018 achieved ~6–7%. The best current models are in the 4–5% range, representing substantial progress where each 1% improvement is significant.

#### Full Benchmark Table (Unveiling Deep Shadows, SBU-Refine)

| Model | Year | Venue | Params | Speed (img/s) | BER @256 | BER @512 |
|-------|------|-------|--------|---------------|----------|----------|
| **DSC** | 2018 | CVPR | 122.5M | 26.86 | 6.79% | 6.34% |
| **BDRAR** | 2018 | ECCV | 42.5M | 39.76 | 6.27% | 5.62% |
| **DSDNet** | 2019 | ICCV | 58.2M | 37.53 | 5.37% | 5.04% |
| **FDRNet** | 2021 | ICCV | 37.5M | 30–40 | 5.64% | 5.39% |
| **FSDNet** | 2021 | TIP | ~4M | fastest | — | — |
| **SDDNet** | 2023 | ACM MM | 15.0M | 36.73 | 5.39% | 4.86% |
| **SILT** | 2023 | ICCV | varies | ~25 | — | ~4.2%* |

*SILT BER from its own paper (PVTv2-B5 backbone). FDRNet speed depends on whether CRF is used. Note the trend: 2018 models ~6% → 2023 models ~4%, with SDDNet matching older model accuracy at 1.5–2× speed.*

#### Tier 1: Recommended for Immediate Use

**SDDNet** (Style-guided Dual-layer Disentanglement Network) — Speed champion. Disentangles the image into "shadow layer" and "background layer" using style-based feature separation, then detects shadows via a dual-decoder architecture. ResNeXt-101 backbone. ~32 FPS on RTX 4090 (~36 img/s). ~44 seconds for 1400 frames vs. 2–3 minutes for typical 10–20 FPS models. Optional CRF for sharper edges. PyTorch 1.10.1. All sources agree this is the best starting point for large datasets.

**SILT** — Accuracy champion among easy-to-deploy models. U-Net variant with CNN and transformer components. Iterative label refinement during training explicitly teaches the network to reject dark objects as false shadows — adds augmented "fake shadows" and removes mislabeled regions during self-training. PVTv2-B3 is the best accuracy/speed tradeoff. PyTorch 1.8. All sources agree on cleanest documentation and most robust inference code. Expects dataset file structures by default, but only minor tweaks needed for arbitrary image folders.

**FDRNet** — Edge quality champion via CRF post-processing. EfficientNet-based encoder that explicitly models illumination features. Produces the crispest shadow boundaries — minimal "halo" of penumbra pixels. Ideal for masking where fuzzy edges waste valid pixels. PyTorch 1.5 (old), pydensecrf problematic on Windows. **Practical tip**: The Unveiling Deep Shadows benchmark provides an FDRNet implementation in PyTorch 1.8 that sidesteps the original legacy codebase, though the CRF step may need separate implementation. All sources recommend using a Conda environment or Docker container for the original code.

#### Tier 2: Specialized or Emerging

| Model | Year | Venue | Distinguishing Feature | Status |
|-------|------|-------|----------------------|--------|
| **FSDNet** | 2021 | TIP | Smallest (~4M params), MobileNet-v2/TinyNet base, best for ONNX export | Custom CUDA IRNN ops complicate Windows; can run without them at quality cost |
| **BDRAR** | 2018 | ECCV | Fast (40 img/s), bidirectional feature extraction | Older but well-tested, 42.5M params |
| **DSDNet** | 2019 | ICCV | Good accuracy (BER 5.04 @512), 58.2M params | Available via benchmark |
| **STNet** | 2025 | IET-IP | Swin Transformer + edge-guided multi-task decoder + semantic affinity loss | Code limited (journal repo); impact more conceptual |
| **FastInstShadow** | 2025 | **CVPR** | Instance-level: Association Transformer Decoder pairs shadows with casters | Code on GitHub; 30+ FPS; surpasses SSISv2 on SOBA |

**STNet** (Structure-Aware Transformer Network) — Explicitly tackles the hard case of dark non-shadow objects vs. shadows. Uses an edge-guided multi-task decoder to impose sharp structure on shadow boundaries, plus a semantic affinity loss that ensures shadow regions are learned distinctly from black surfaces. Trains with an auxiliary edge map providing boundary supervision. Reduced false detections on dark regions compared to prior art. Underscores the broader theme that boundary-aware training is critical — also seen in FDRNet's CRF and SILT's label tuning.

**FastInstShadow** (Inoue & Miyamoto, CVPR 2025) — A query-based one-stage instance shadow detector. Introduces the Association Transformer Decoder that simultaneously predicts shadow masks and object boxes using query embeddings to learn shadow-object relationships. Eliminates the slow two-step "detect then associate" pipeline of SSIS/SSISv2. On the SOBA benchmark, outperforms all prior instance methods while running in real-time. For photogrammetry, this enables automatically masking "only shadows cast by people" — run person detection, then use the paired shadow mask for those instances. This directly supports the environmental shadow preservation strategy described in Section 1.4.

### 3.2 SAM-Based Shadow Detectors

#### Consensus: ShadowAdapter is the Most Production-Ready

All sources converge on **AdapterShadow / ShadowAdapter** (ESWA 2025, github.com/LeipingJie/AdapterShadow) as the best SAM-based option:

- Automatic grid-based point prompts — no manual interaction needed for batch processing
- Trained on SBU, UCF, ISTD, CUHK — broad generalization
- Leverages SAM's powerful pretrained visual representations
- Tested on ~1,400-frame video sequences without human interaction
- Inference command: `python pl_test_simple.py -net sam -exp_name sbu -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point -sample grid -grid_out_size 16`
- Requires SAM ViT-B or ViT-L checkpoint + PyTorch Lightning
- Weights via Google Drive (one user noted manual download intervention needed)

#### Other SAM Adaptations

| Model | Architecture | Shadow-Specific? | Weights Available? | Notes |
|-------|-------------|-----------------|-------------------|-------|
| **SAM-Adapter** | SAM1 + adapters | Framework only | **Shadow weights NOT released** (camouflage only confirmed) | 4×A100 training required. More recipe than plug-and-play. |
| **SAM2-Adapter** | SAM2 + adapters | Yes (ICLR 2025 Workshop) | TBD | Vanilla SAM2 has zero shadow understanding without adaptation |
| **ShadowSAM** | SAM1 + CNN refinement | Yes (TGRS 2023) | Baidu only | Lower VRAM; somewhat superseded by AdapterShadow |
| **OPSeg** | SAM2 + Grounding DINO | Instance shadow | Yes (GitHub) | GDINO localizes shadow regions + Hungarian matching for shadow-object pairing |
| **SoftShadow** | SAM + penumbra loss | Yes (CVPR 2025) | Part of removal pipeline | Gets initial hard mask from SAM, learns soft edge via physical penumbra formation models |

**Critical finding confirmed across all rounds**: Vanilla SAM (any version) produces meaningless results on shadow detection without task-specific fine-tuning. SAM "fails to segment anything" for shadows without adaptation.

#### Test-Time Adaptation: TICA (ICONIP 2024)

Now confirmed across multiple sources (originally Claude, now also ChatGPT PDF Round 2). TICA introduces test-time intensity consistency adaptation — at inference, it adjusts the model to minimize intensity distribution inconsistencies, under the assumption that shadows cause specific, predictable intensity changes. Using a base HRNet-18 model, TICA achieved BER ~1.17 on ISTD — a 24% improvement over FDRNet on the same dataset. This is a research prototype (not widely packaged), but it points toward future **self-calibrating shadow detectors** that adapt to each new lighting environment without additional labeled data. Highly valuable for out-of-distribution photogrammetry images.

### 3.3 Object-Centric and Multi-Task Models

**MetaShadow** (CVPR 2025, Adobe/CUHK) — Three-in-one framework: shadow detection, removal, and synthesis. The Shadow Analyzer component focuses on shadows attached to specific objects via a GAN-based approach that jointly detects and removes an object's shadow. mIoU improvement from 55.8 to 71.0 on object-shadow benchmark vs. previous best SSISv2. Currently paper-only; code may be released. For photogrammetry: feed person masks from YOLO → MetaShadow → person-specific shadow masks. Heavyweight (two networks: analyzer + synthesizer) but represents the state of the art for object-centric shadow handling.

**ShadowFormer** — Shadow removal model (input image → shadow-free image). The "diff" approach (|input - output| → threshold → rough shadow mask) is viable but indirect. Does NOT require a shadow mask at inference. Pretrained weights for ISTD, ISTD+, SRD on Google Drive.

### 3.4 Shadow Removal Models (with Detection Byproduct)

The Round 2 PDF formalizes the concept of **removal-based masking**: running a shadow removal network and deriving masks from the difference between input and output. This offers a two-in-one benefit — both masks for the 3D pipeline AND pre-delighted images for texturing.

**RASM** (Regional Attention Shadow Removal, NTIRE 2025 Winner) — Lightweight U-net architecture with a novel regional attention mechanism that lets each shadow region "see" its surrounding non-shadow neighborhood. Unlike global attention approaches (which lose high-frequency texture detail), RASM uses a "divide-and-conquer" strategy. Outputs both a shadow-free image and a binary shadow mask. The mask is directly usable for photogrammetry. Code: github.com/CalcuLuUus/RASM.

**ReHiT** (Retinex-guided Histogram Transformer, NTIRE 2025 top performer) — Two-branch CNN-Transformer architecture: one branch for reflectance, one for illumination, guided by Retinex theory. Operates mask-free — no external shadow mask needed at inference. Uses a recurrent structure to progressively refine shadow removal. Users report better feature matching in complex urban areas when ReHiT is used to flatten images before reconstruction. Paper: arxiv.org/html/2504.14092v1.

**The "diff trick"**: For any removal model, `|input - output|` → threshold → shadow mask. For hard shadows where the removed area changes dramatically, this produces decent masks. Caveat: removal networks may also subtly alter non-shadow tones, introducing noise in the difference map. Running a full removal model solely for masks is computationally heavier than dedicated detectors. But as removal models improve, the two-in-one approach becomes increasingly practical.

**DeS3** (AAAI 2024) — First diffusion-based shadow removal robust to hard, soft, and self shadows. github.com/jinyeying/DeS3_Deshadow.

### 3.5 Soft Masks: Beyond Binary

SoftShadow (CVPR 2025) represents a paradigm shift from binary to probabilistic masks. Instead of hard 0/1 classification, it predicts continuous values distinguishing umbra (fully shadowed) from penumbra (gradient). It uses SAM to get an initial hard mask, then learns a soft edge around it using physical penumbra formation models.

**Implication for photogrammetry** (from Round 2 PDF): Soft masks could be used to **weight pixel contributions** rather than making a hard cut — down-weighting half-shadow regions instead of omitting them entirely. This would avoid abrupt texture cut-offs at mask boundaries, potentially reducing visible masking artifacts in the final texture. Whether photogrammetry software can accept weighted masks (rather than binary) is an open integration question.

---

## Part 4: Classical and Physics-Based Approaches

### 4.1 Chromaticity-Invariant Color Spaces

These exploit the physics of shadow formation: shadows change illumination intensity but largely preserve chromaticity ratios on Lambertian surfaces. Now covered across multiple sources (originally Claude, confirmed by ChatGPT PDF Round 2).

**c1c2c3 Color Space** (Finlayson et al.)
```
C1 = arctan(R / max(G, B))
C2 = arctan(G / max(R, B))  
C3 = arctan(B / max(G, R))
```
Under a shadow, overall intensity drops but these normalized color angles remain nearly the same as in light. On a c1c2c3 image, shadow regions look similar to lit regions of the same material while differing dramatically in RGB. Simple threshold suffices on uniform surfaces. Implementation: ~2 lines of NumPy. Known weakness: fails when one channel dominates (e.g. lush green grass confuses the invariants). Also fails on very dark surfaces where ratios become noise-dominated.

**Normalized RGB**: r = R/(R+G+B). Even simpler invariant that ratios out intensity. In practice, makes shadowed pixels only slightly bluer (skylight fill). Sensitive to noise at low intensities where R+G+B approaches zero. Doesn't fully account for shadow blue-shift.

**Finlayson Entropy Minimization** (ECCV 2004, IJCV 2009): Finds a specific linear projection of log-RGB values that maximally flattens illumination changes. Rotates the color space to find an axis along which the image's color distribution has minimum entropy — the resulting single-channel image is (ideally) completely illumination-invariant, meaning shadows disappear entirely. Assumes Planckian (blackbody) illuminant spectrum and Lambertian surfaces. Solution can be unstable without calibration. MATLAB implementation at github.com/hangong/gfinvim. **Algorithm is patented**, which has limited adoption.

**Salvador et al. (2004)**: Generate edge maps from original image AND from invariant image (c1c2c3). Shadow edges appear in normal image but NOT in invariant image. Material edges appear in both. Subtract to get candidate shadow edges. Then apply geometric heuristics (shadow edges connect to casting objects, have characteristic lengths) to eliminate false positives. This multi-stage hypothesis-verification framework reduces false positives significantly — literature reports ~65% reduction near boundaries.

**HSI space for aerial imagery**: A 2012 paper (Springer, cited by Grok Round 2) demonstrated shadow detection in color aerial images based on HSI space and color attenuation relationships.

### 4.2 Practical Hybrid Intensity-Chromaticity Filter

For harsh shadows on sand/concrete (the target scenario), the Round 2 PDF provides a concrete implementation recipe:

1. Convert image to HSV and to normalized RGB
2. Identify pixels where V (value) is dark — specifically, ~50% lower than the median in a local window
3. Check that the chromatic difference (r,g values in normalized RGB) between the pixel and the local window median is below a threshold
4. Pixels meeting both criteria (dark but chromatically stable) are classified as shadow
5. Those that are dark but have different chromaticity from surroundings are likely dark objects, not shadows

**Strengths**: Zero training data, perfect generalization, sub-second per frame, no GPU needed. Can pick up hard shadows on ground reliably with few false alarms.

**Weaknesses**: Misses very soft shadows (where color temperature shifts or intensity drop is minor). Can mistake truly black objects. Fails on non-uniform surfaces (grass, textured stone).

**Hybrid strategy** (confirmed across multiple sources): Use classical invariant to get obvious shadow regions quickly, then feed to a learned model to refine boundaries and detect trickier parts. Or use as validation — regions that a learned model labels as shadow but that have unstable chromaticity are likely false positives (dark objects, not shadows). This leverages the strengths of both approaches.

### 4.3 Intrinsic Image Decomposition

Identified by Claude as the "highest-leverage underexplored approach." Now acknowledged by Gemini Round 2 as well.

**Careaga & Aksoy Pipeline** (ACM TOG 2023, SIGGRAPH Asia 2024 Best Paper Honorable Mention):
- github.com/compphoto/Intrinsic
- pip-installable: `pip install https://github.com/compphoto/Intrinsic/archive/main.zip`
- Separates image into Albedo (reflectance) and Shading components
- Shadows fall entirely into the shading layer
- On uniform surfaces, albedo should be constant → any shading variation = shadow/lighting
- Shadow mask = threshold(shading_map), where S = I / R
- No shadow-specific training needed — bypasses domain gap entirely

**IntrinsicAnything** (2024): Diffusion-model priors for inverse rendering. github.com/zju3dv/IntrinsicAnything.

**IDArb** (ICLR 2025): Multi-view intrinsic decomposition with cross-view attention — potentially ideal for photogrammetry datasets.

### 4.4 Depth-Based Shadow Prediction

Uniquely practical for photogrammetry since the pipeline already produces 3D geometry:

1. Reconstruct initial point cloud / mesh
2. Compute sun position from GPS + timestamp (using `pvlib` or `suncalc`)
3. Ray-trace shadows from sun against reconstructed geometry
4. Predicted shadow regions = where sun rays are occluded by geometry
5. Compare predicted vs. observed shadows to identify person/equipment shadows (which aren't in the reconstruction geometry)

Validated by the S-EO dataset methodology (LiDAR DSMs + sun position → shadow masks for satellite photogrammetry). Also referenced in the Oxford "Dealing with Shadows" paper on capturing intrinsic scene appearance for image-based outdoor localization.

---

## Part 5: Neural Reconstruction Approaches (Bypass Masking)

### 5.1 NeRF-Based Methods

| Method | Mechanism | Shadow Handling |
|--------|-----------|----------------|
| **NeRF-W** | Per-image appearance (48D) + transient (16D) embeddings | "Explains away" lighting variation; transient embedding models image-dependent content with uncertainty |
| **RobustNeRF** (CVPR 2023) | Trimmed robust loss | Treats shadow pixels as outliers, maintains >31 dB PSNR with increasing distractors |
| **NeRF-HuGS** (CVPR 2024) | SfM features as static prompts for SAM | Uses the photogrammetry pipeline's own output to detect transient content including shadows |

### 5.2 3D Gaussian Splatting Variants

| Method | Key Innovation | Performance | Reference |
|--------|---------------|-------------|-----------|
| **WildGaussians** (NeurIPS 2024) | Appearance modeling + DINO uncertainty | Same inference speed as vanilla 3DGS | github.com/jkulhanek/wild-gaussians |
| **GS-W** (ECCV 2024) | Separate intrinsic vs. dynamic appearance features | Physically motivated decomposition | — |
| **SpotLessSplats** (Google DeepMind, ToG 2024) | Stable Diffusion features for distractor detection | Robust optimization excluding distractor pixels | github.com/lilygoli/SpotLessSplats |
| **Splatfacto-W** (Nerfstudio) | Per-Gaussian neural color + per-image appearance | +5.3 dB PSNR over vanilla 3DGS, 150× faster than NeRF | — |
| **ShadowGS** (2026) | Per-splat shadow opacity, sun-direction conditioning | Real-time sun position changes; implicit de-lighting | arxiv.org/abs/2601.00939 |
| **SSD-GS** (2026) | **Diffuse + Specular + Shadow + Scattering** decomposition | Four-component relightable 3DGS | openreview.net/forum?id=7m2Dqz9g05 |
| **VA-GS** | View-aligned density control | Enhanced geometric representation | arxiv.org/abs/2510.11473 |
| **3D-HGS** (CVPR 2025) | Half-Gaussians for opacity modeling | Improved surface representation | CVPR 2025 poster |
| **RadSplat** | NeRF-3DGS hybrid | Claims 900 FPS rendering | — |

**ShadowGS** learns to decouple Albedo from Illumination during training, introducing a "Shadow Opacity" parameter per Gaussian splat. Uses sun position data (from metadata or estimated) to predict occlusion. The output allows changing sun position in real-time — de-lighting becomes implicit by turning off the shadow layer. Currently satellite-focused but potentially adaptable to ground-level photogrammetry via sun-position priors.

**SSD-GS** (new in Round 2) goes further than ShadowGS by decomposing the scene into four radiometric components — Diffuse, Specular, Shadow, and Scattering — enabling full relighting during the reconstruction process itself. This represents the cutting edge of implicit shadow handling.

### 5.3 Practical Implications

For users whose end goal is a neural 3D representation:
- Appearance embeddings make explicit shadow masking **substantially less critical**
- RobustNeRF and SpotLessSplats require zero shadow detection — they treat shadows as outliers automatically
- WildGaussians and Splatfacto-W handle most lighting variation with per-image embeddings
- SSD-GS and ShadowGS offer physics-based shadow decomposition for full relighting

For users targeting traditional mesh + texture output (COLMAP/Metashape):
- These tools lack equivalent mechanisms
- Explicit shadow masking remains important
- Post-hoc de-lighting (Section 7) becomes the fallback

---

## Part 6: Video and Temporal Shadow Detection

*Relevant for 360° video photogrammetry workflows.*

### 6.1 Dedicated Video Shadow Detection

| Method | Speed | Mechanism | Code |
|--------|-------|-----------|------|
| **SCOTCH and SODA** (CVPR 2023) | — | Deformation attention tracks shadow changes across frames; BER 10.36 on ViSha | github.com/lihaoliu-cambridge/scotch-and-soda |
| **ShadowSAM / Detect-AnyShadow** (TCSVT 2023) | — | Detect in keyframes, propagate via LSTN | github.com/harrytea/Detect-AnyShadow |
| **STICT** (CVPR 2022) | **91.34 FPS** | Converts image detectors to video via temporal consistency | 1400 frames in ~15 seconds |

### 6.2 SAM2 Video Mode for Shadows

Important caveats from Jie et al. (December 2024, github.com/LeipingJie/SAM2Video):

- Mask prompts work satisfactorily; point prompts perform significantly worse
- **Accuracy degrades beyond ~100 frames** — quality drops notably
- For 1400 frames: need to re-prompt every 30–50 frames (30–47 keyframes required)
- Not a "prompt once, propagate all" solution for large datasets

### 6.3 Recommended Video Pipeline

1. Run strong single-image detector (ShadowAdapter or SILT) on keyframes every 30–50 frames
2. Propagate masks between keyframes using ShadowSAM's LSTN or SAM2 video mode
3. Apply optical flow-based consistency checking as post-processing validation
4. Total processing: ~2–3 minutes for 1400 frames (detection + propagation)

---

## Part 7: Post-Reconstruction De-Lighting

### 7.1 Blender Division Method (Most Physically Accurate)

```
Albedo = Scan_Texture / Irradiance_Map
```

1. Import mesh into Blender with Cycles renderer
2. Bake geometric maps (AO, bent normals, curvature) to UV space
3. Match directional light to shadow direction seen in texture (use Sun Position addon or captured HDRI from chrome ball)
4. Bake illumination to texture → Irradiance_Map
5. Divide scan texture by irradiance map in compositor
6. Apply high-pass filter to restore contrast lost in division (texture often looks "washed out" after division — overlay blend mode or Curve adjustment needed)

### 7.2 Agisoft Texture De-Lighter

Free standalone tool with two modes:
- **Remove Shading** (automatic): Reverses calculated AO. Works for uniform/cloudy scans.
- **Remove Cast Shadows** (semi-automatic): User paints "Lit" (yellow) and "Shadowed" (blue) strokes. Algorithm computes color histogram transfer function matching shadow areas to lit areas. Less accurate than Blender method for complex materials — relies on color statistics rather than light physics. Can produce "muddy" textures if shadow has strong color cast (blue skylight).

### 7.3 Adobe Substance 3D Sampler

AI-based "Delight" filter in Image to Material pipeline. CNN trained on lit/unlit material pairs detects low-frequency lighting gradients. Best for soft gradients; less effective at removing hard, complex cast shadows where geometric-aware methods (Blender/Agisoft) excel.

### 7.4 Unity ArtEngine (Discontinued)

**Unity ArtEngine** has been officially discontinued as of 2025/2026 (confirmed by Gemini Round 2, create.unity.com/unity-artengine-sunset). Previously a staple for "Example-Based" material creation and delighting (Texture Mutation, Seam Removal). Its features are NOT being directly ported to a Unity successor in the same form. This creates a "void" for delighting and seam removal workflows that Substance 3D Sampler and Metashape Texture De-Lighter are now filling.

### 7.5 Tool Comparison (from Gemini Round 2)

| Feature | Agisoft Metashape Pro (v2.3) | Adobe Substance 3D Sampler |
|---------|----------------------------|---------------------------|
| Primary Role | Full photogrammetry & geometry engine | Material authoring & texturing tool |
| Performance (326 images) | ~12 minutes | ~45 minutes |
| Light Sensitivity | Robust in varied lighting; high-detail meshes | "Craves light"; struggles with low-contrast/overcast |
| Delighting | Texture De-Lighter (cast shadow removal) | Integrated "Image to Material" AI delighting |
| Best For | Geometry/scale, surveyors, high-end VFX | Ease of use, texture artists, quick PBR creation |

### 7.6 Community Insights on De-Lighting

From Polycount forum: Removing AO from photogrammetry textures is a long-standing challenge. The Blender division method is preferred by game studios (Epic Games Kite Demo workflow), but requires accurately matching the lighting environment.

Reddit r/photogrammetry consensus: No fully automated AI de-lighting tool works reliably on complex outdoor scans as of 2026. Best results combine automated tools with manual touch-up.

Agisoft forum discussion: Comparison of Agisoft De-Lighter vs. Unity ArtEngine vs. Substance Alchemist (now Sampler) reveals no clear winner — each excels in different scenarios.

---

## Part 8: Software Integration Details

### 8.1 COLMAP

- Mask format: PNG, same filename as image, pixel value 0 = extract features, 255 = masked
- Integration: `--ImageReader.mask_path /path/to/masks/` during feature extraction
- Built-in robustness: RootSIFT descriptors, RANSAC outlier rejection, NCC-based PatchMatchStereo with brightness normalization, geometric consistency checks
- **No built-in shadow detection** — all sources confirm
- Documentation explicitly warns: "capture images at similar illumination conditions"

### 8.2 Agisoft Metashape

- Mask import: `File → Import Masks → From File` (GUI) or `chunk.importMasks(path='masks/{filename}.png')` (Python API)
- AI masking (v2.3): Automatic foreground/background separation using rembg/U²-Net — **NOT shadow-specific** (targets background removal, not shadow detection)
- Mask application modes: Key Points vs. Tie Points (see Section 1.3)
- Texture De-Lighter: Free standalone post-processing tool
- Scripts repository: github.com/agisoft-llc/metashape-scripts — includes `masking_by_color_dialog.py`
- Supports mask persistence across processing steps
- Performance: ~12 minutes for 326 images (from Gemini Round 2 comparison)

### 8.3 RealityCapture / RealityScan

- **RealityScan 2.0** (mid-2025): Native AI masking via "Detect Masks" tool (likely distilled SAM variant)
- Effective for turntable backgrounds and floor shadows
- Less effective for continuous terrain (drone) where no clear foreground exists
- CLI supports importing external masks via `-importMasks` — enables shadow-free reconstruction at scale
- **Image Layers** feature: Create separate geometry layer (shadow-masked) and texture layer
- No direct shadow detection

### 8.4 Meshroom / AliceVision

- MeshMasking node: Applies masks to mesh after generation but before texturing
- 2025 updates allow ML segmentation nodes directly in the node graph before SfM
- Semantic segmentation for classes like "sky" or "shadow" can feed into feature extraction

### 8.5 Pix4D

- PIX4Dmatic has integrated SAM for general semantic segmentation
- No shadow-specific detection
- Supports external mask import

### 8.6 OpenDroneMap

- No shadow handling at all
- Accepts masks through standard preprocessing

---

## Part 9: The Domain Gap Problem

All sources identify domain gap as the critical challenge.

### 9.1 Benchmark Limitations

- **SBU** (~5000 images): Diverse outdoor scenes, most relevant to photogrammetry. SBU-Refine has corrected annotations. Most representative for photogrammetry use cases.
- **ISTD** (~2000 images): Only 135 distinct backgrounds, mostly hard building/pole shadows. Clean triplets (shadow/shadow-free/mask) but poor diversity. Often used for both detection and removal evaluation.
- **CUHK-Shadow** (~10000+ images): Largest, more shadow regions per image.
- **UCF** (~245 images): Test-only, too small.
- **SRD**: Shadow removal dataset, useful for cross-dataset evaluation.

Cross-dataset BER roughly doubles: SDDNet achieves 4.86 on SBU but 7.65 on SRD.

**Important benchmark note** (Round 2 PDF): When reading papers, check whether they report on SBU (large, diverse, shadow masks only) vs. ISTD (smaller, with ground-truth shadow-free images). For photogrammetry usage, SBU and SBU-Refine are most representative.

### 9.2 Photogrammetry-Specific Challenges

- **Resolution mismatch**: Benchmarks use 256–640px; photogrammetry frames are 4000–8000px+
- **Dark surface confusion**: Most common failure — dark sand, wet concrete, dark grass misclassified as shadow
- **Shadow shapes**: Articulated human shadows and complex equipment shadows poorly represented in training data
- **Boundary errors**: 65% of false negatives within 10 pixels of shadow boundaries
- **Penumbra**: Harsh noon shadow penumbral regions poorly represented in training data
- **Aerial-specific**: Fine-grained shadow detection in aerial imagery remains a distinct challenge (DAGM GCPR 2023 paper)

### 9.3 Mitigation Strategies (Ranked by Practicality)

1. **Classical chromaticity verification**: Bypass domain gap entirely using physics-based invariants. Free, instant, zero training. Works well on uniform surfaces.

2. **TICA test-time adaptation**: Adapts any pretrained detector during inference by exploiting intensity consistency. No additional training data needed. BER 1.17 on ISTD. Points toward self-calibrating field detectors.

3. **Intrinsic image decomposition**: Careaga & Aksoy pipeline separates albedo from shading without shadow-specific training. pip-installable.

4. **Fine-tune on 20–50 annotated frames**: S-EO paper finding: "domain-appropriate training data matters more than model architecture." Even a simple U-Net with domain-appropriate data matches specialized architectures.

5. **Foundation model generalization**: SAM2-Adapter and ShadowAdapter leverage rich pretrained representations for better out-of-distribution robustness than smaller CNN detectors.

6. **Self-supervised labeling from reconstruction**: After initial reconstruction, use sun position + reconstructed geometry to ray-trace predicted shadows → generate training labels automatically → fine-tune detector.

---

## Part 10: Datasets and Benchmarks Reference

| Dataset | Size | Content | Primary Use | Year |
|---------|------|---------|-------------|------|
| **SBU** | ~5,000 | Diverse outdoor shadows | Detection training | 2012 |
| **SBU-Refine** | ~5,000 | SBU with corrected labels | Benchmark evaluation | 2024 |
| **ISTD** | ~2,000 | Triplets: shadow/shadow-free/mask | Detection + removal | 2018 |
| **CUHK-Shadow** | ~10,000+ | Dense shadow annotations | Detection training | 2020 |
| **UCF** | ~245 | Small test set | Evaluation only | 2013 |
| **SOBA** | ~1,000 | Shadow-object pairs | Instance detection | 2021 |
| **S-EO** | ~19,162 | Satellite + LiDAR-derived masks | Geometry-aware detection | 2025 |
| **ADSP** | 1,220 | Outdoor, occluder visible, high-res | Detection training | 2024 |
| **ViSha** | 120 videos / 11,685 frames | Video shadows | Video detection | 2021 |
| **CVSD** | 196 clips / 19,757 frames | Complex overlapping shadows | Video detection | 2024 |

**Unveiling Deep Shadows Benchmark** (github.com/xw-hu/Unveiling-Deep-Shadows): The one-stop resource. Provides standardized weights for 8 detection models at 2 resolutions, 7 video detection models, 5 instance detection models, and 28 shadow removal models — all retrained under identical conditions. Also provides a unified PyTorch 1.8 codebase that sidesteps original authors' older dependency requirements.

**NTIRE 2025 Image Shadow Removal Challenge** (CodaLab): Pushed the frontier for removal methods. Top entries (RASM, ReHiT) demonstrated mask-free approaches achieving excellent results, with emphasis on perceptual MOS alongside PSNR/SSIM.

**Validation metrics for photogrammetry** (from Grok Round 2): BER for detection quality; PSNR/SSIM for rendering/texture quality; Chamfer distance and F1 score for geometric accuracy of the reconstruction.

---

## Part 11: Recommended Workflows by Scenario

### Workflow A: Large Outdoor Dataset (Beach/Ground, ~1400 Frames from 360° Video)

**This is the primary target scenario.**

```
Phase 1: Detection
├── Primary: SDDNet @512×512 (~44s for 1400 frames)
│   └── Output: probability maps → threshold at 0.4 (err toward recall)
├── Verification: c1c2c3 chromaticity stability check
│   └── Reject candidates with unstable chromaticity (dark surfaces, not shadows)
├── Environmental shadow preservation:
│   └── Spatial filter: Final Mask = (Shadow Mask ∩ Spatial Filter) ∪ Object Mask
│   └── Keep shadows >X pixels from any person/equipment mask
├── Video consistency: STICT (91 FPS) or propagate from keyframes
│   └── Every 30-50 frames: full ShadowAdapter inference
│   └── Between keyframes: SAM2 video propagation or optical flow warp
└── Union with person/equipment masks from YOLO + SAM

Phase 2: Mask Application
├── COLMAP: --ImageReader.mask_path with binary PNGs
├── Metashape: importMasks() + Apply Mask to Tie Points
└── For texture: use mosaic blending mode (selects best-exposed pixels)

Phase 3: Validation
├── Compare reconstructions with/without masks
├── Check point cloud density in previously shadowed areas
├── Inspect texture maps for residual shadow artifacts
├── Metrics: Chamfer distance for geometry, PSNR for texture quality
└── If residual shadows in texture: Agisoft Texture De-Lighter or Blender division
```

### Workflow B: Controlled Studio / Turntable

```
Prevention: Cross-polarization + ring flash
Shadow Lifting: RAW processing (Shadows +100, Blacks +50)
Masking: Difference masking (clean plate subtraction via Python script)
Application: Metashape Mask from Model recursion
De-lighting: Agisoft Texture De-Lighter (Remove Shading mode)
```

### Workflow C: Aerial / Drone Survey

```
Pre-processing: RASM or ReHiT for image flattening before reconstruction
    (two-in-one: provides masks AND de-lighted images)
Masking: SDDNet or SILT on each frame
Application: COLMAP masks or Metashape Mask to Tie Points
Post-processing: Adobe Substance 3D Sampler for texture de-lighting
```

### Workflow D: Neural Reconstruction (NeRF/3DGS Target)

```
Option 1 (minimal effort): Use WildGaussians or Splatfacto-W — appearance 
    embeddings handle most shadow variation automatically
Option 2 (maximum quality): Shadow mask → SpotLessSplats with explicit 
    distractor identification
Option 3 (research frontier): ShadowGS with per-splat shadow opacity
Option 4 (bleeding edge): SSD-GS with 4-component radiometric decomposition
    (Diffuse + Specular + Shadow + Scattering)
```

### Workflow E: Highest Quality (Hybrid)

```
1. Capture with chrome/grey ball reference rig
2. SDDNet detection → c1c2c3 verification → spatial filter → union with person masks
3. Metashape reconstruction with Mask to Tie Points
4. Texture baking with mosaic blending
5. Blender division de-lighting using captured HDRI
6. Substance 3D Sampler for residual gradient cleanup
```

### Workflow F: Instance-Selective Masking (New)

```
For scenes where environmental shadows should be preserved:
1. Run YOLO/SAM for person/equipment detection → object masks
2. Run FastInstShadow for shadow-object pairing
3. Keep only shadows paired with transient objects
4. OR: run SDDNet globally → intersect with convex hull of object masks
5. Apply filtered masks to reconstruction pipeline
```

---

## Part 12: Key Repositories

### Detection Models
| Repository | What | URL |
|-----------|------|-----|
| Unveiling Deep Shadows | Benchmark + unified weights for 8+ models (PyTorch 1.8) | github.com/xw-hu/Unveiling-Deep-Shadows |
| SILT | Best documented detector | github.com/hanyangclarence/SILT |
| SDDNet | Fastest detector | github.com/rmcong/SDDNet_ACMMM23 |
| FDRNet | Best edge quality (CRF) | github.com/rayleizhu/FDRNet |
| FSDNet | Smallest model (~4M) | github.com/xw-hu/FSDNet |
| ShadowAdapter | SAM-based with auto-prompts | github.com/LeipingJie/AdapterShadow |
| SAM-Adapter | Framework (needs training; no shadow weights) | github.com/tianrun-chen/SAM-Adapter-PyTorch |
| OPSeg | GDINO + SAM2 instance detection | github.com/deyang2000/OpSeg |
| FastInstShadow | Instance detection, CVPR 2025, 30+ FPS | arxiv.org/abs/2503.07517 |

### Shadow Removal
| Repository | What | URL |
|-----------|------|-----|
| ShadowFormer | Shadow-free image generation | github.com/GuoLanqing/ShadowFormer |
| DeS3 | Diffusion-based removal | github.com/jinyeying/DeS3_Deshadow |
| RASM | NTIRE 2025 winner, regional attention | github.com/CalcuLuUus/RASM |
| ReHiT | Retinex-guided, mask-free removal | arxiv.org/html/2504.14092v1 |

### Video Shadow Detection
| Repository | What | URL |
|-----------|------|-----|
| SCOTCH and SODA | Temporal deformation attention | github.com/lihaoliu-cambridge/scotch-and-soda |
| Detect-AnyShadow | SAM + LSTN video propagation | github.com/harrytea/Detect-AnyShadow |
| SAM2 Video Shadow | SAM2 evaluation for video shadows | github.com/LeipingJie/SAM2Video |

### Alternative Approaches
| Repository | What | URL |
|-----------|------|-----|
| Intrinsic Decomposition | Albedo/shading separation | github.com/compphoto/Intrinsic |
| Finlayson Invariant | Illumination-invariant image (MATLAB) | github.com/hangong/gfinvim |
| Grounded SAM 2 | Open-vocabulary + segmentation | github.com/IDEA-Research/Grounded-Segment-Anything |
| IntrinsicAnything | Diffusion-based inverse rendering | github.com/zju3dv/IntrinsicAnything |

### Neural Reconstruction
| Repository | What | URL |
|-----------|------|-----|
| WildGaussians | 3DGS with appearance modeling | github.com/jkulhanek/wild-gaussians |
| SpotLessSplats | 3DGS distractor handling | github.com/lilygoli/SpotLessSplats |
| IC-Light | Diffusion-based relighting | github.com/lllyasviel/IC-Light |
| ShadowGS | Shadow-aware satellite 3DGS | arxiv.org/abs/2601.00939 |
| SSD-GS | 4-component relightable 3DGS | openreview.net/forum?id=7m2Dqz9g05 |

### Photogrammetry Tools
| Repository | What | URL |
|-----------|------|-----|
| Metashape Scripts | Official script collection | github.com/agisoft-llc/metashape-scripts |
| SlicerMorph Photogrammetry | SAM + ODM pipeline | github.com/SlicerMorph/SlicerPhotogrammetry |

---

## Part 13: Environment and Dependency Notes

### PyTorch Version Matrix

| Model | Python | PyTorch | CUDA | Special Dependencies |
|-------|--------|---------|------|---------------------|
| SILT | 3.8 | 1.8.0 | 11.1 | timm, kornia, omegaconf |
| SDDNet | 3.6.10 | 1.10.1 | 11.1 | pydensecrf (optional) |
| FDRNet | 3.6.10 | 1.5.0 | 10.1 | EfficientNet-PyTorch, pydensecrf |
| FDRNet (via benchmark) | ~3.8 | 1.8 | — | CRF step separate |
| FSDNet | 3.6 | 1.3.0 | — | CuPy (custom CUDA kernels) |
| ShadowAdapter | 3.8 | 1.13.0 | — | PyTorch Lightning, SAM checkpoint |
| Benchmark (unified) | ~3.8 | 1.8 | — | environment.yaml provided |

### Windows Compatibility Summary

- **Safe**: SILT, SDDNet (without CRF), ShadowAdapter, ShadowFormer
- **Difficult**: FDRNet (pydensecrf), FSDNet (CuPy/custom CUDA ops), FastInstShadow (Detectron2 C++ compilation)
- **Workaround**: pydensecrf2 pip package, or skip CRF (raw output still usable)
- **Alternative**: Run inference on WSL2 Ubuntu, transfer masks to Windows pipeline
- **Practical tip**: Most PyTorch models run on newer versions with minor edits (replacing deprecated functions). Favor models with minimal non-Python dependencies (SILT, SDDNet) for Windows studio pipelines.

### RTX 3090 Ti (24GB) Capacity

All models in this survey fit comfortably within 24GB VRAM at 512×512 input:
- SDDNet: ~2–4GB
- SILT (PVTv2-B5): ~6–8GB
- ShadowAdapter (SAM ViT-L): ~10–14GB
- FastInstShadow: ~4–6GB (Detectron2-based)

For higher-resolution inference (1024×1024 or full photogrammetry resolution), tile-based processing may be needed for SAM-based models. SDDNet and SILT should handle up to ~2048×2048 on 24GB. Generally, a 12–24GB GPU handles all common 2D shadow models at 4K by processing a few images at a time.

---

## Part 14: Open Questions and Future Directions

1. **Validated end-to-end benchmarks**: A 2024 study showed basic shadow detection improved MVS (reduced holes), and Epic Games' reality capture team reports significant manual cleanup reduction. But no source provides controlled experiments with specific metrics (Chamfer distance improvement, texture PSNR gain) comparing masked vs. unmasked reconstruction across multiple scenes. This is the critical missing validation.

2. **Soft mask integration**: SoftShadow produces penumbra-aware continuous masks, but can photogrammetry software accept weighted masks rather than binary? If so, weighted contribution could avoid abrupt texture cut-offs. Testing this with Metashape/COLMAP is straightforward.

3. **Self-supervised loop**: Using initial reconstruction geometry + sun position to generate shadow training labels → fine-tune detector → re-mask → re-reconstruct. No source reports actually implementing this closed loop, though all the components exist.

4. **Multi-view shadow consensus**: Across overlapping views, the same surface point may be shadowed in some views and sunlit in others. A proper multi-view shadow detector would exploit this consistency. No current tool implements this.

5. **Integration with Panoex / 360° video tools**: For the 360° video photogrammetry workflow, shadow detection should ideally operate on equirectangular frames or be integrated into the frame extraction pipeline.

6. **Metashape Python API automation**: A complete script that runs SDDNet inference → generates mask PNGs → imports into Metashape chunk → processes with mask-to-tie-points would close the integration gap entirely. This is a weekend project, not a research problem.

7. **Unity ArtEngine void**: With ArtEngine discontinued, the niche of example-based texture mutation and delighting has no direct successor. Custom AI workflows or Substance 3D Sampler are partial fills, but the texture mutation capability specifically is lost.

8. **Self-calibrating field detectors**: TICA points toward shadow detectors that adapt automatically to each new scene's lighting. Packaging this into a usable tool (rather than research prototype) would be transformative for field photogrammetry.

9. **Removal as pre-processing**: Using RASM or ReHiT to flatten images before reconstruction (not just for masks) represents a promising but underexplored workflow. The combination of mask generation AND de-lighted images from a single pass could simplify the pipeline significantly.

---

*This knowledge base synthesizes research from eight independent reports across two rounds of cross-model polling. Cross-validated against primary repositories and papers. Areas of disagreement are explicitly flagged. Every uploaded document was read in full before integration. Last updated 2026-02-08.*
