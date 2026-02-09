# Shadow Detection & Masking for Photogrammetry — Definitive Knowledge Base

*Synthesized from five independent research reports (original survey, Claude, Gemini, Grok, ChatGPT/PDF) and cross-validated against primary sources. Compiled 2026-02-07.*

---

## Executive Summary

Five independent research efforts converge on a clear consensus: **no major photogrammetry tool has built-in shadow detection**, but all accept external masks, making the integration gap the primary practical problem rather than the absence of good detection models. The field has matured rapidly since 2023, with viable pretrained models (SILT, SDDNet, ShadowAdapter), foundation-model adaptations (SAM2-Adapter), and neural reconstruction methods (appearance-embedding 3DGS) all offering practical paths forward. For the specific scenario of harsh noon shadows from people and equipment on uniform outdoor surfaces, the strongest validated approach combines a learned detector with classical chromaticity verification, applied through the reconstruction software's existing mask import interface.

### Consensus Points (All 5 Sources Agree)

- **SDDNet** is the best speed/convenience choice for batch processing (~32 FPS, 15M params, pretrained weights available)
- **SILT** offers the best accuracy/documentation balance for single-image detection (BER ~4.2% on SBU)
- **Unveiling Deep Shadows benchmark** is the single most useful resource (unified weights for 8 models)
- **ShadowAdapter / AdapterShadow** (ESWA 2025) is the most production-ready SAM-based approach with automatic prompting
- **MetaShadow** (CVPR 2025) is the most notable new entry but code availability is uncertain
- **SoftShadow** (CVPR 2025) introduces penumbra-aware masks — cutting edge but complex to deploy
- **S-EO dataset** (CVPR Workshop 2025) is the most important new dataset for geometry-aware shadow detection
- **Domain gap** between benchmark training data and real photogrammetry scenes is the critical practical challenge
- Shadows affect **texturing most**, dense reconstruction moderately, and feature matching/alignment least

### Key Disagreements and Gaps

- **Gemini** uniquely emphasizes acquisition-phase prevention (cross-polarization, chrome/grey ball rigs, wandering light) — the others focus exclusively on computational detection
- **Claude** alone covers classical chromaticity methods (c1c2c3, Finlayson entropy minimization) and intrinsic image decomposition as practical alternatives
- **Gemini** uniquely covers post-reconstruction de-lighting workflows (Blender division method, Substance 3D Sampler, Agisoft Texture De-Lighter)
- **ChatGPT/PDF** uniquely covers FastInstShadow (2025) for instance-level shadow detection
- **Grok** is the only source claiming Metashape v2.3 has "AI masking" for shadows specifically — this requires clarification (Metashape's AI masking targets foreground/background separation, not shadow detection per se)
- **Claude** alone identifies TICA (test-time adaptation) as a domain-gap mitigation strategy
- **Gemini** alone covers Meshroom/AliceVision's node-based masking workflow and RealityScan 2.0's native AI masking
- No source provides validated end-to-end benchmarks showing reconstruction quality improvement from shadow masking

---

## Part 1: Why Shadows Corrupt Photogrammetry

### 1.1 The Physics of Shadow Formation

Shadows arise when an occluder blocks direct illumination from reaching a surface. The resulting region receives only indirect (ambient/skylight) illumination. For outdoor scenes under sunlight:

- **Umbra**: Fully occluded from direct sun. Receives only skylight (blue-shifted, diffuse). Intensity drops 60–90% relative to sunlit surface.
- **Penumbra**: Partially occluded (extended light source or atmospheric scattering). Gradual intensity transition at shadow boundaries. Width depends on occluder-surface distance and sun angular diameter (~0.5°).
- **Color shift**: Outdoor shadows are illuminated primarily by blue skylight rather than direct yellowish sunlight. This causes a measurable hue shift toward blue, increased saturation, and decreased value in HSV space.

The mathematical relationship (from Gemini's analysis): a camera captures Radiance L, which is the product of Albedo ρ and Irradiance E. In shadow, E drops and shifts spectrally:

```
L(x) ≈ ρ(x) · E(x)
```

Photogrammetry software projects raw Radiance onto the mesh. To recover true Albedo requires inverting this equation — which requires knowing E(x), the irradiance field.

### 1.2 How Shadows Break Each Pipeline Stage

**Feature Matching / SfM (Moderate Impact)**

All sources agree SIFT/RootSIFT features have inherent partial illumination invariance through gradient-based description with brightness normalization. COLMAP's RootSIFT further improves this. However:

- Shadow boundaries create strong false edges generating unreliable keypoints
- In multi-temporal datasets (drone surveys over hours), shadow positions shift — matching shadow corners across images creates false 3D tie points that stress bundle adjustment (Gemini's "transient feature problem")
- In turntable scenarios, stationary shadows are treated as valid static features, potentially collapsing alignment (Gemini's "stationary shadow artifact")
- RANSAC outlier rejection catches some but not all shadow-induced mismatches

**Dense Reconstruction / MVS (Significant Impact)**

Photo-consistency (typically measured via NCC) is directly violated by shadows:

- A surface patch shadowed in one view but sunlit in another produces dramatically different pixel intensities
- NCC drops below acceptance threshold → holes in the depth map or high-frequency noise on what should be smooth surfaces
- Persistent shadows (same position across all similar viewpoints) are worse — they are accepted as valid and create incorrect depth estimates or phantom geometry

**Texturing (Most Severe Impact)**

All sources agree this is the primary damage vector:

- Shadows are unconditionally baked into texture maps
- The resulting asset is locked to the capture lighting environment
- Multi-view blending can partially mitigate this (mosaic mode selects best-exposed pixels), but large shadows visible from all angles have no shadow-free alternative view
- Removing baked shadows post-hoc requires de-lighting — a complex additional step

### 1.3 Metashape Mask Application Strategy

Gemini uniquely provides the critical operational detail: Metashape's "Apply Mask to..." setting has two importantly different modes:

- **Apply Mask to Key Points**: Prevents SIFT from detecting features in masked areas. Risk: if mask is loose, you lose valid edge features → poor alignment.
- **Apply Mask to Tie Points** (recommended): Allows feature detection everywhere, but discards matches inside the mask after detection. Preserves high-contrast boundary features while filtering shadow-caused noise.

The "Mask from Model" recursion workflow (validated by Gemini and community sources):
1. Draft alignment (Low accuracy, no masks)
2. Build rough mesh
3. Delete shadow/background geometry in 3D viewport
4. Generate masks from the cleaned model (projects clean 3D shape back to 2D)
5. Re-align on High accuracy with masks applied to tie points

---

## Part 2: Acquisition-Phase Shadow Prevention

*Exclusively covered by Gemini. The other four sources focus on computational post-capture approaches.*

### 2.1 Cross-Polarization

The gold standard for studio/controlled photogrammetry. Polarizing film on lights + circular polarizer on lens (90° offset) blocks specular reflections. For shadow management:

- Eliminates bright specular highlights, allowing exposure for shadows without clipping
- Enables +50 to +100 Shadow recovery in RAW processing (Lightroom/ACR)
- Produces flat, diffuse-only images ideal for de-lighting algorithms
- **Cost**: 1.5–2 stops of light loss; fails on metallic objects
- **Hardware**: Ring flash (Godox AR400), high-transmission polarizing film, CPL for lens

### 2.2 Grey Ball / Chrome Ball Reference Rig (Epic Games Method)

For outdoor scans where lighting cannot be controlled:

- **50% chrome sphere**: Captures sun direction and environmental reflection → unwrap to HDRI
- **18% grey sphere**: Measures lit/shadow contrast ratio
- **Macbeth ColorChecker**: Color constancy across captures
- The HDRI enables mathematically subtracting the exact captured lighting in post (Blender/Nuke division method)

### 2.3 Wandering Light / Light Painting

For static objects in dark studios:

- Long exposure (2–5 seconds) with operator moving handheld light around object
- Shadows average out across the exposure, simulating ambient dome light
- Produces exceptionally flat, shadow-free textures
- Requires perfectly dark room and static subject

### 2.4 Practical Outdoor Mitigation (Community Consensus)

From multiple community sources across the reports:

- **Overcast days** are universally preferred for photogrammetry capture
- **RAW shadow recovery**: Shadows +50 to +100, Highlights -50 to -100 in Lightroom as preprocessing (from 80.lv photogrammetry guide, cited by Claude)
- **Capture timing**: Avoid high noon for horizontal surfaces; noon is actually best for vertical surfaces (facades)
- **Consistent lighting**: COLMAP documentation explicitly warns to "capture images at similar illumination conditions"

---

## Part 3: Shadow Detection Models — Consensus Taxonomy

### 3.1 Dedicated Single-Image Shadow Detectors

These models take RGB input → output binary or probability shadow mask. Trained on shadow-specific datasets.

#### Tier 1: Recommended for Immediate Use

| Model | Year | Params | Speed | BER (SBU) | Weights | Windows OK | Sources Recommending |
|-------|------|--------|-------|-----------|---------|------------|---------------------|
| **SDDNet** | ACM MM 2023 | 15M | 32 FPS (4090) | 4.86 @512 | Google Drive + Baidu | Yes (CRF optional) | All 5 |
| **SILT** | ICCV 2023 | varies | moderate | 4.19 (PVTv2-B5) | Google Drive | Yes | All 5 |
| **FDRNet** | ICCV 2021 | — | moderate | 3.04 (with CRF) | GitHub Releases | Difficult (pydensecrf) | All 5 |

**SDDNet** — Speed champion. ResNeXt-101 backbone. ~44 seconds for 1400 frames. Optional CRF for sharper edges. PyTorch 1.10.1. All sources agree this is the best starting point for large datasets.

**SILT** — Accuracy champion among easy-to-deploy models. Iterative label refinement during training explicitly teaches the network to reject dark objects as false shadows. PVTv2-B3 is the best accuracy/speed tradeoff. PyTorch 1.8. All sources agree on cleanest documentation and most robust inference code. The ChatGPT/PDF report notes that SILT "explicitly counteracts the tendency to label dark objects as shadow."

**FDRNet** — Edge quality champion via CRF post-processing. Produces the crispest shadow boundaries — critical for masking where fuzzy edges waste valid pixels. PyTorch 1.5 (old), pydensecrf problematic on Windows. All sources recommend using a Conda environment or Docker container.

#### Tier 2: Specialized or Emerging

| Model | Year | Distinguishing Feature | Status |
|-------|------|----------------------|--------|
| **FSDNet** | TIP 2021 | Smallest (~4M params), best for ONNX export | Custom CUDA ops complicate Windows |
| **BDRAR** | ECCV 2018 | Fast (40 img/s) | Older but well-tested |
| **DSDNet** | 2019 | Good accuracy (BER 5.04 @512) | Available via benchmark |
| **STNet** | IET IP 2025 | Swin Transformer + semantic affinity loss for dark surface rejection | Claude only |
| **FastInstShadow** | arXiv 2025 | Instance-level: pairs each shadow with its caster, 30+ FPS | ChatGPT/PDF only |

**FastInstShadow** is uniquely identified by the ChatGPT/PDF report. It uses a DETR-inspired query transformer to directly learn shadow-object pairs without two-step detect-then-associate. Outputs per-object shadow masks — combine all instances for a binary union mask. Built on Detectron2. Surpasses SSISv2 on the SOBA benchmark. For photogrammetry, the ability to selectively mask only person/equipment shadows while keeping environmental shadows is highly valuable.

### 3.2 SAM-Based Shadow Detectors

#### Consensus: ShadowAdapter is the Most Production-Ready

All five sources converge on **AdapterShadow / ShadowAdapter** (ESWA 2025, github.com/LeipingJie/AdapterShadow) as the best SAM-based option:

- Automatic grid-based point prompts — no manual interaction needed for batch processing
- Trained on SBU, UCF, ISTD, CUHK — broad generalization
- Leverages SAM's powerful pretrained visual representations
- Inference command: `python pl_test_simple.py -net sam -exp_name sbu -npts 5 -backbone b1 -plug_image_adapter -all -freeze_backbone -use_neg_point -sample grid -grid_out_size 16`
- Requires SAM ViT-B or ViT-L checkpoint + PyTorch Lightning
- Weights via Google Drive (one user noted manual download intervention needed)

#### Other SAM Adaptations

| Model | Architecture | Shadow-Specific? | Weights Available? | Notes |
|-------|-------------|-----------------|-------------------|-------|
| **SAM-Adapter** | SAM1 + adapters | Camouflage focus, shadow unclear | Camouflage only confirmed | Framework, not plug-and-play. 4×A100 training. |
| **SAM2-Adapter** | SAM2 + adapters | Yes (ICLR 2025 Workshop) | TBD | Vanilla SAM2 has zero shadow understanding without adaptation |
| **ShadowSAM** | SAM1 + CNN refinement | Yes (TGRS 2023) | Baidu only | Lower VRAM, satellite focus |
| **OPSeg** | SAM2 + Grounding DINO | Instance shadow | Yes (GitHub) | Uses GDINO for auto-prompting + Hungarian matching |
| **SoftShadow** | SAM + penumbra loss | Yes (CVPR 2025) | Part of removal pipeline | Produces soft masks with umbra/penumbra distinction |

**Critical finding confirmed by Claude and ChatGPT/PDF**: Vanilla SAM (any version) produces meaningless results on shadow detection without task-specific fine-tuning. SAM "fails to segment anything" for shadows without adaptation.

### 3.3 Object-Centric and Multi-Task Models

**MetaShadow** (CVPR 2025, Adobe/CUHK) — All five sources cover this. Three-in-one framework: shadow detection, removal, and synthesis. Given an object mask, detects and removes that object's shadow. mIoU improvement from 55.8 to 71.0. Currently paper-only; code may be released. For photogrammetry: would allow feeding person masks from YOLO → MetaShadow → person-specific shadow masks. ChatGPT/PDF notes it's "quite involved" with two networks (analyzer + synthesizer).

**ShadowFormer** — Shadow removal model (input image → shadow-free image). All sources note the diff approach (|input - output| → threshold → rough shadow mask) as a viable but indirect detection path. Does NOT require a shadow mask at inference. Best for when you want both cleaned images AND masks. Pretrained weights for ISTD, ISTD+, SRD on Google Drive.

### 3.4 New Models and Competition Results

**RASM** (NTIRE 2025 Shadow Removal Challenge Winner) — Covered only by Gemini. Regional Attention Shadow Removal Model. Shifts from global to regional attention, correlating shadowed regions with adjacent non-shadow neighborhoods. Outputs both a shadow-free image and a binary shadow mask. The binary mask is directly usable for photogrammetry masking.

**ReHiT** (Recurrent Hierarchical Transformer) — Covered only by Gemini. Mask-free shadow removal using progressive recurrent refinement. Users report better feature matching in complex urban areas when used to flatten images before reconstruction.

**DeS3** (AAAI 2024) — Covered by Claude. First diffusion-based shadow removal robust to hard, soft, and self shadows. github.com/jinyeying/DeS3_Deshadow.

**TICA** (ICONIP 2024) — Covered only by Claude. Test-time intensity consistency adaptation. BER 1.17 on ISTD (24.5% improvement over FDRNet). Adapts at inference without retraining — uniquely valuable for bridging domain gaps to photogrammetry imagery.

---

## Part 4: Classical and Physics-Based Approaches

*Primarily covered by Claude. Largely absent from other sources.*

### 4.1 Chromaticity-Invariant Color Spaces

These exploit the physics of shadow formation: shadows change illumination intensity but largely preserve chromaticity ratios on Lambertian surfaces.

**c1c2c3 Color Space**
```
C1 = arctan(R / max(G, B))
C2 = arctan(G / max(R, B))  
C3 = arctan(B / max(G, R))
```
Shadow and non-shadow regions on uniform surfaces appear nearly identical in c1c2c3 while differing in RGB. Implementation: ~2 lines of NumPy. Known weakness: greenish objects misclassified.

**Normalized RGB**: r = R/(R+G+B). Even simpler. Sensitive to noise at low intensities. Doesn't account for shadow blue-shift.

**Finlayson Entropy Minimization** (ECCV 2004, IJCV 2009): Projects log-chromaticity orthogonal to illumination change direction. Produces genuinely shadow-free grayscale images for Planckian illumination on Lambertian surfaces. MATLAB implementation at github.com/hangong/gfinvim. **Algorithm is patented.**

**Salvador et al. (2004)**: Multi-stage approach: c1c2c3 edge maps identify shadow-caused edges (present in RGB, absent in c1c2c3), then geometric verification reduces false positives.

### 4.2 Practical Classical Pipeline for Photogrammetry

For harsh shadows on sand/concrete (the target scenario), the recommended classical approach:

1. Convert to HSV — shadows decrease V, increase S, shift H toward blue
2. Compute c1c2c3 chromaticity
3. Shadow candidates: pixels where V is low relative to local median BUT c1c2c3 is stable relative to neighbors
4. Edge verification: shadow edges appear in intensity but not in chromaticity
5. Morphological cleanup

**Strengths**: Zero training data, perfect generalization, sub-second per frame, no GPU needed.
**Weaknesses**: Fails on non-uniform surfaces (grass, textured stone), doesn't handle soft shadows, requires tuning per scene.

### 4.3 Intrinsic Image Decomposition

Claude uniquely identifies this as the "highest-leverage underexplored approach" for photogrammetry shadow handling.

**Careaga & Aksoy Pipeline** (ACM TOG 2023, SIGGRAPH Asia 2024 Best Paper Honorable Mention):
- github.com/compphoto/Intrinsic
- pip-installable: `pip install https://github.com/compphoto/Intrinsic/archive/main.zip`
- Separates image into Albedo (reflectance) and Shading components
- Shadows fall entirely into the shading layer
- On uniform surfaces, albedo should be constant → any shading variation = shadow/lighting
- Shadow mask = threshold(shading_map)
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

This approach is validated by the S-EO dataset methodology (which uses LiDAR DSMs + sun position to generate shadow masks for satellite photogrammetry).

---

## Part 5: Neural Reconstruction Approaches (Bypass Masking)

### 5.1 NeRF-Based Methods

| Method | Mechanism | Shadow Handling |
|--------|-----------|----------------|
| **NeRF-W** | Per-image appearance + transient embeddings | "Explains away" lighting variation; transient embedding models image-dependent content with uncertainty |
| **RobustNeRF** (CVPR 2023) | Trimmed robust loss | Treats shadow pixels as outliers, maintains >31 dB PSNR with increasing distractors |
| **NeRF-HuGS** (CVPR 2024) | SfM features as static prompts for SAM | Uses the photogrammetry pipeline's own output to detect transient content including shadows |

### 5.2 3D Gaussian Splatting Variants

| Method | Key Innovation | Performance |
|--------|---------------|-------------|
| **WildGaussians** (NeurIPS 2024) | Appearance modeling + DINO uncertainty | Same inference speed as vanilla 3DGS |
| **GS-W** (ECCV 2024) | Separate intrinsic vs. dynamic appearance features | Physically motivated decomposition |
| **SpotLessSplats** (Google DeepMind, ToG 2024) | Stable Diffusion features for distractor detection | Robust optimization excluding distractor pixels |
| **Splatfacto-W** (Nerfstudio) | Per-Gaussian neural color + per-image appearance | +5.3 dB PSNR over vanilla 3DGS, 150× faster than NeRF |
| **ShadowGS** (2026) | Per-splat shadow opacity, sun-direction conditioning | Can change sun position in real-time; implicit de-lighting |

**ShadowGS** (covered by Gemini) is the most directly relevant: it learns to decouple Albedo from Illumination during training, introducing a "Shadow Opacity" parameter per Gaussian splat. The output allows changing sun position in real-time — de-lighting becomes implicit by turning off the shadow layer.

### 5.3 Practical Implications

For users whose end goal is a neural 3D representation:
- Appearance embeddings make explicit shadow masking **substantially less critical**
- RobustNeRF and SpotLessSplats require zero shadow detection — they treat shadows as outliers automatically
- WildGaussians and Splatfacto-W handle most lighting variation with per-image embeddings

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
| **SCOTCH and SODA** (CVPR 2023) | — | Deformation attention tracks shadow changes across frames | github.com/lihaoliu-cambridge/scotch-and-soda |
| **ShadowSAM / Detect-AnyShadow** (TCSVT 2023) | — | Detect in keyframes, propagate via LSTN | github.com/harrytea/Detect-AnyShadow |
| **STICT** (CVPR 2022) | **91.34 FPS** | Converts image detectors to video via temporal consistency | 1400 frames in ~15 seconds |

### 6.2 SAM2 Video Mode for Shadows

Claude's research reveals important caveats from Jie et al. (December 2024, github.com/LeipingJie/SAM2Video):

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

*Primarily covered by Gemini. Addresses shadows already baked into textures.*

### 7.1 Blender Division Method (Most Physically Accurate)

```
Albedo = Scan_Texture / Irradiance_Map
```

1. Import mesh into Blender with Cycles renderer
2. Bake geometric maps (AO, bent normals, curvature) to UV space
3. Match directional light to shadow direction seen in texture (use Sun Position addon or captured HDRI from chrome ball)
4. Bake illumination to texture → Irradiance_Map
5. Divide scan texture by irradiance map in compositor
6. Apply high-pass filter to restore contrast lost in division

### 7.2 Agisoft Texture De-Lighter

Free standalone tool with two modes:
- **Remove Shading** (automatic): Reverses calculated AO. Works for uniform/cloudy scans.
- **Remove Cast Shadows** (semi-automatic): User paints "Lit" (yellow) and "Shadowed" (blue) strokes. Algorithm computes color histogram transfer function. Less accurate than Blender method for complex materials — relies on statistics rather than physics.

### 7.3 Adobe Substance 3D Sampler

AI-based "Delight" filter in Image to Material pipeline. CNN trained on lit/unlit material pairs detects low-frequency lighting gradients. Best for soft gradients; poor at hard cast shadows.

### 7.4 Community Insights on De-Lighting

From Polycount forum (cited by Gemini): Removing AO from photogrammetry textures is a long-standing community challenge. The division method is preferred by game studios (Epic Games Kite Demo workflow), but requires accurately matching the lighting environment, which is time-consuming.

Reddit r/photogrammetry consensus (cited by Gemini): No fully automated AI de-lighting tool exists that works reliably on complex outdoor scans as of 2026. The best results still combine automated tools with manual touch-up.

---

## Part 8: Software Integration Details

### 8.1 COLMAP

- Mask format: PNG, same filename as image, pixel value 0 = extract features, 255 = masked
- Integration: `--ImageReader.mask_path /path/to/masks/` during feature extraction
- Built-in robustness: RootSIFT descriptors, RANSAC outlier rejection, NCC-based PatchMatchStereo with brightness normalization, geometric consistency checks
- **No built-in shadow detection** — all sources confirm
- Documentation explicitly warns: "capture images at similar illumination conditions"

### 8.2 Agisoft Metashape

- Mask import: `File → Import Masks → From File` (GUI) or `chunk.importMask(path, source=MaskSourceFile)` (Python API)
- AI masking (v2.3): Automatic foreground/background separation using rembg/U²-Net — **NOT shadow-specific** (Grok's claim of "AI masking for shadows" is imprecise; it targets background removal)
- Mask application modes: Key Points vs. Tie Points (see Section 1.3)
- Texture De-Lighter: Free standalone post-processing tool
- Scripts repository: github.com/agisoft-llc/metashape-scripts — includes `masking_by_color_dialog.py`
- Supports mask persistence across processing steps

### 8.3 RealityCapture / RealityScan

- **RealityScan 2.0** (mid-2025): Native AI masking via "Detect Masks" tool (Gemini only source)
- Effective for turntable backgrounds and floor shadows
- Less effective for continuous terrain (drone) where no clear foreground exists
- CLI supports importing external masks via `-importMasks`
- **Image Layers** feature: Create separate geometry layer (shadow-masked) and texture layer
- No direct shadow detection

### 8.4 Meshroom / AliceVision

- MeshMasking node: Applies masks to mesh after generation but before texturing
- 2025 updates allow ML segmentation nodes in the graph before SfM
- Semantic segmentation for classes like "sky" or "shadow" can feed into feature extraction (Gemini source)

### 8.5 Pix4D

- PIX4Dmatic has integrated SAM for general semantic segmentation
- No shadow-specific detection
- Supports external mask import

### 8.6 OpenDroneMap

- No shadow handling at all
- Accepts masks through standard preprocessing

---

## Part 9: The Domain Gap Problem

All sources identify domain gap as the critical challenge. The detailed analysis:

### 9.1 Benchmark Limitations

- **SBU** (~5000 images): Diverse outdoor scenes, most relevant to photogrammetry. SBU-Refine has corrected annotations.
- **ISTD** (~2000 images): Only 135 distinct backgrounds, mostly hard building/pole shadows. Clean triplets (shadow/shadow-free/mask) but poor diversity.
- **CUHK-Shadow** (~10000+ images): Largest, more shadow regions per image.
- **UCF** (~245 images): Test-only, too small.

Cross-dataset BER roughly doubles: SDDNet achieves 4.86 on SBU but 7.65 on SRD.

### 9.2 Photogrammetry-Specific Challenges

- **Resolution mismatch**: Benchmarks use 256–640px; photogrammetry frames are 4000–8000px+
- **Dark surface confusion**: Most common failure — dark sand, wet concrete, dark grass misclassified as shadow
- **Shadow shapes**: Articulated human shadows and complex equipment shadows poorly represented in training data
- **Boundary errors**: 65% of false negatives within 10 pixels of shadow boundaries
- **Penumbra**: Harsh noon shadow penumbral regions poorly represented in training data

### 9.3 Mitigation Strategies (Ranked by Practicality)

1. **Classical chromaticity verification** (Claude): Bypass domain gap entirely using physics-based invariants. Free, instant, zero training. Works well on uniform surfaces.

2. **TICA test-time adaptation** (Claude): Adapts any pretrained detector during inference by exploiting intensity consistency. No additional training data needed. BER 1.17 on ISTD.

3. **Intrinsic image decomposition** (Claude): Careaga & Aksoy pipeline separates albedo from shading without shadow-specific training. pip-installable.

4. **Fine-tune on 20–50 annotated frames** (Claude, ChatGPT/PDF): S-EO paper finding: "domain-appropriate training data matters more than model architecture." Even a simple U-Net with domain-appropriate data matches specialized architectures.

5. **Foundation model generalization** (all sources): SAM2-Adapter and ShadowAdapter leverage rich pretrained representations for better out-of-distribution robustness than smaller CNN detectors.

6. **Self-supervised labeling from reconstruction** (Claude): After initial reconstruction, use sun position + reconstructed geometry to ray-trace predicted shadows → generate training labels automatically → fine-tune detector.

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

**Unveiling Deep Shadows Benchmark** (github.com/xw-hu/Unveiling-Deep-Shadows): The one-stop resource. Provides standardized weights for 8 detection models at 2 resolutions, 7 video detection models, 5 instance detection models, and 28 shadow removal models — all retrained under identical conditions.

---

## Part 11: Recommended Workflows by Scenario

### Workflow A: Large Outdoor Dataset (Beach/Ground, ~1400 Frames from 360° Video)

**This is the primary target scenario.**

```
Phase 1: Detection
├── Primary: SDDNet @512×512 (~44s for 1400 frames)
│   └── Output: probability maps → threshold at 0.4 (lower for recall)
├── Verification: c1c2c3 chromaticity stability check
│   └── Reject candidates with unstable chromaticity (dark surfaces, not shadows)
├── Video consistency: STICT (91 FPS) or propagate from keyframes
│   └── Every 30-50 frames: full ShadowAdapter inference
│   └── Between keyframes: SAM2 video propagation or optical flow warp
└── Union with person/equipment masks from YOLO + SAM

Phase 2: Mask Application
├── COLMAP: --ImageReader.mask_path with binary PNGs
├── Metashape: importMask() + Apply Mask to Tie Points
└── For texture: use mosaic blending mode (selects best-exposed pixels)

Phase 3: Validation
├── Compare reconstructions with/without masks
├── Check point cloud density in previously shadowed areas
├── Inspect texture maps for residual shadow artifacts
└── If residual shadows in texture: Agisoft Texture De-Lighter or Blender division
```

### Workflow B: Controlled Studio / Turntable

```
Prevention: Cross-polarization + ring flash
Masking: Difference masking (clean plate subtraction via Python script)
Application: Metashape Mask from Model recursion
De-lighting: Agisoft Texture De-Lighter (Remove Shading mode)
```

### Workflow C: Aerial / Drone Survey

```
Pre-processing: RASM or ReHiT for image flattening before reconstruction
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
```

### Workflow E: Highest Quality (Hybrid)

```
1. Capture with chrome/grey ball reference rig
2. SDDNet detection → c1c2c3 verification → union with person masks
3. Metashape reconstruction with Mask to Tie Points
4. Texture baking with mosaic blending
5. Blender division de-lighting using captured HDRI
6. Substance 3D Sampler for residual gradient cleanup
```

---

## Part 12: Key Repositories

### Detection Models
| Repository | What | URL |
|-----------|------|-----|
| Unveiling Deep Shadows | Benchmark + unified weights for 8 models | github.com/xw-hu/Unveiling-Deep-Shadows |
| SILT | Best documented detector | github.com/hanyangclarence/SILT |
| SDDNet | Fastest detector | github.com/rmcong/SDDNet_ACMMM23 |
| FDRNet | Best edge quality (CRF) | github.com/rayleizhu/FDRNet |
| FSDNet | Smallest model (~4M) | github.com/xw-hu/FSDNet |
| ShadowAdapter | SAM-based with auto-prompts | github.com/LeipingJie/AdapterShadow |
| SAM-Adapter | Framework (needs training) | github.com/tianrun-chen/SAM-Adapter-PyTorch |
| OPSeg | GDINO + SAM2 instance detection | github.com/deyang2000/OpSeg |
| FastInstShadow | Instance detection, 30+ FPS | github.com search "FastInstShadow" |

### Shadow Removal
| Repository | What | URL |
|-----------|------|-----|
| ShadowFormer | Shadow-free image generation | github.com/GuoLanqing/ShadowFormer |
| DeS3 | Diffusion-based removal | github.com/jinyeying/DeS3_Deshadow |
| RASM | NTIRE 2025 winner | github.com/CalcuLuUus/RASM |

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
| Finlayson Invariant | Illumination-invariant image | github.com/hangong/gfinvim |
| Grounded SAM 2 | Open-vocabulary + segmentation | github.com/IDEA-Research/Grounded-Segment-Anything |
| IntrinsicAnything | Diffusion-based inverse rendering | github.com/zju3dv/IntrinsicAnything |

### Neural Reconstruction
| Repository | What | URL |
|-----------|------|-----|
| WildGaussians | 3DGS with appearance modeling | github.com/jkulhanek/wild-gaussians |
| SpotLessSplats | 3DGS distractor handling | github.com/lilygoli/SpotLessSplats |
| IC-Light | Diffusion-based relighting | github.com/lllyasviel/IC-Light |

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
| FSDNet | 3.6 | 1.3.0 | — | CuPy (custom CUDA kernels) |
| ShadowAdapter | 3.8 | 1.13.0 | — | PyTorch Lightning, SAM checkpoint |
| Benchmark (unified) | ~3.8 | 1.8 | — | environment.yaml provided |

### Windows Compatibility Summary

- **Safe**: SILT, SDDNet (without CRF), ShadowAdapter, ShadowFormer
- **Difficult**: FDRNet (pydensecrf), FSDNet (CuPy/custom CUDA ops)
- **Workaround**: pydensecrf2 pip package, or skip CRF (raw output still usable)
- **Alternative**: Run inference on WSL2 Ubuntu, transfer masks to Windows pipeline

### RTX 3090 Ti (24GB) Capacity

All models in this survey fit comfortably within 24GB VRAM at 512×512 input:
- SDDNet: ~2–4GB
- SILT (PVTv2-B5): ~6–8GB
- ShadowAdapter (SAM ViT-L): ~10–14GB
- FastInstShadow: ~4–6GB (Detectron2-based)

For higher-resolution inference (1024×1024 or full photogrammetry resolution), tile-based processing may be needed for SAM-based models. SDDNet and SILT should handle up to ~2048×2048 on 24GB.

---

## Part 14: Open Questions and Future Directions

1. **No validated end-to-end benchmark**: No source provides controlled experiments showing how much shadow masking actually improves reconstruction quality metrics (Chamfer distance, texture PSNR, etc.) compared to no masking. This is the critical missing validation.

2. **Penumbra handling**: SoftShadow (CVPR 2025) is the only model addressing soft shadow boundaries. For harsh noon shadows, penumbra is narrow but exists. Whether binary or soft masks perform better for photogrammetry masking is untested.

3. **Self-supervised loop**: Using initial reconstruction geometry + sun position to generate shadow training labels → fine-tune detector → re-mask → re-reconstruct. No source reports actually implementing this closed loop.

4. **Multi-view shadow consensus**: Across overlapping views, the same surface point may be shadowed in some views and sunlit in others. A proper multi-view shadow detector would exploit this consistency — identifying regions that change appearance across views despite being the same 3D point. No current tool implements this.

5. **Integration with Panoex / 360° video tools**: For the 360° video photogrammetry workflow, shadow detection should ideally operate on equirectangular frames or be integrated into the frame extraction pipeline.

6. **Metashape Python API automation**: A complete script that runs SDDNet inference → generates mask PNGs → imports into Metashape chunk → processes with mask-to-tie-points would close the integration gap entirely. This is a weekend project, not a research problem.

---

*This knowledge base synthesizes research from five independent sources: the original survey document, expanded reports from Claude, Gemini, Grok, and ChatGPT/PDF. Cross-validated against primary repositories and papers. Areas of disagreement are explicitly flagged. Last updated 2026-02-07.*
