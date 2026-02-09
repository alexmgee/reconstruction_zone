# Shadow detection and masking for photogrammetry: an expanded research guide

**The most actionable gap in current photogrammetry shadow handling is not better detection models — it is the absence of any end-to-end pipeline connecting shadow detectors to reconstruction software.** No major photogrammetry tool (Metashape, COLMAP, RealityCapture, Pix4D, ODM) has built-in shadow detection, yet all accept external masks. The research landscape since 2024 has advanced significantly across foundation-model adaptation (SAM2-Adapter, ShadowAdapter), video shadow detection (SCOTCH and SODA, ShadowSAM for video), and intrinsic image decomposition — all of which offer practical paths forward. For harsh outdoor shadows on uniform surfaces, a hybrid approach combining classical chromaticity invariants with learned models likely outperforms either alone, and neural reconstruction methods (3DGS with appearance embeddings) can substantially reduce the criticality of perfect shadow masking.

---

## New models push beyond the user's existing survey

Several significant shadow detection and removal models have appeared since the models already covered (SILT, FDRNet, SDDNet, FSDNet, SAM-Adapter, AdapterShadow, ShadowSAM, ShadowFormer).

**MetaShadow** (CVPR 2025, Adobe Research) introduces a three-in-one framework combining shadow detection, removal, and synthesis. Its object-centered formulation is particularly relevant for photogrammetry because it can handle shadows from specific objects like people and equipment. The Shadow Analyzer uses a GAN-based detection+removal pipeline, while the Shadow Synthesizer leverages diffusion models. It improves mIoU from **55.8 to 71.0** for object-centered shadow mask detection.

**SAM2-Adapter** (ICLR 2025 Workshop) extends the original SAM-Adapter concept to SAM2's more powerful architecture. It inserts lightweight trainable adapters into SAM2's frozen encoder to inject shadow-specific knowledge, achieving competitive or better results than the original SAM-Adapter. The critical finding is that **vanilla SAM2 has no understanding of the "shadow" concept** without proper adaptation — it produces meaningless results on shadow detection tasks without fine-tuning.

**ShadowAdapter** (Expert Systems with Applications, 2025) solves a key practical problem: automatic prompt generation. Instead of requiring manual point prompts for SAM, it uses an auxiliary network to automatically generate high-quality point prompts, making it viable for batch processing **~1,400 frames** without human interaction. Code is available at github.com/LeipingJie/AdapterShadow.

**STNet** (IET Image Processing, 2025) combines a Swin Transformer backbone with edge-guided multi-task learning and semantic-aware learning with semantic affinity loss specifically designed to distinguish shadows from shadow-like dark regions — a critical capability for photogrammetry scenes with dark surfaces.

**TICA** (ICONIP 2024) introduces test-time intensity consistency adaptation, the first TTA framework for shadow detection. It exploits light-intensity inconsistencies during inference and requires no additional training data. This is especially promising for bridging domain gaps: it achieves **BER 1.17 on ISTD** with HRNet-18, claiming a 24.5% BER reduction versus FDRNet. For photogrammetry images that differ substantially from training data, TICA's ability to adapt at test time without retraining is highly valuable.

**OPSeg** (IJCNN 2025) fine-tunes SAM2 for instance shadow detection on the SOBA dataset, using **Grounding DINO to automatically generate prompts** followed by Hungarian algorithm matching for shadow-object association. This confirms that Grounding DINO can localize shadows to some degree, which matters for the alternative approaches discussed below. Code: github.com/deyang2000/OpSeg.

Two new datasets deserve attention. **S-EO** (CVPR EarthVision Workshop 2025) provides ~19,162 satellite images with geometry-derived shadow masks from LiDAR DSMs and sun position — the first dataset enabling geometry-aware shadow detection training. **ADSP** (ACCV 2024) offers 1,220 image pairs specifically designed for outdoor scenes where both shadow and shadow-casting object are visible, with the highest resolution among current shadow benchmarks.

On the shadow removal front, **DeS3** (AAAI 2024) is the first diffusion-based method robust to hard, soft, and self shadows using ViT similarity for adaptive attention (github.com/jinyeying/DeS3_Deshadow). **HomoFormer** (CVPR 2024) provides homogenized transformer-based shadow removal, while **SoftShadow** (CVPR 2025) introduces penumbra-aware soft masks that could help with the gradual shadow boundaries common in outdoor scenes.

---

## Classical chromaticity methods remain surprisingly powerful

For harsh shadows on uniform surfaces like sand and concrete, classical color-invariance approaches offer advantages that learned models cannot: zero training data requirements, perfect generalization, and sub-second processing per frame. These methods exploit the physics of shadow formation — shadows change illumination intensity but largely preserve chromaticity ratios.

**The c1c2c3 color space** computes shadow-invariant chrominance as C1 = arctan(R/max(G,B)), C2 = arctan(G/max(R,B)), C3 = arctan(B/max(G,R)). The arctan normalization produces values invariant to shadow/shading under Lambertian assumptions. On uniform surfaces, shadow and non-shadow regions appear nearly identical in c1c2c3 while differing dramatically in RGB — making shadow detection a simple thresholding problem. Implementation requires just two lines of NumPy. The known weakness is **greenish objects being misclassified**, which matters for grass surfaces.

**Normalized RGB** (r = R/(R+G+B), etc.) offers even simpler shadow invariance. On a uniform surface, shadow pixels have nearly identical normalized chromaticity to non-shadow pixels while differing in total intensity. The approach is sensitive to noise at low intensity values where R+G+B approaches zero, and it does not account for the bluer color temperature of outdoor shadows — a real limitation for sunlit scenes.

**Finlayson's entropy minimization** (ECCV 2004, IJCV 2009) is the most mathematically principled approach. It projects log-chromaticity values onto the direction orthogonal to the illumination change direction, producing a 1D grayscale image that is fully invariant to illumination color and intensity. The invariant angle is found by minimizing the entropy of the projected distribution. For outdoor scenes under sunlight (approximately Planckian illumination) on uniform surfaces (approximately Lambertian), this approach produces genuinely shadow-free images. A MATLAB implementation is available at github.com/hangong/gfinvim, though the algorithm is patented.

**Salvador et al. (2004)** demonstrated the practical combination: use c1c2c3 edge maps to identify which edges in the RGB image are shadow-caused versus material-caused (shadow edges appear in RGB but not in c1c2c3), then verify candidates using geometric properties. This multi-stage hypothesis-verification framework reduces false positives significantly.

**For photogrammetry specifically**, the most practical classical approach is computing shadow masks by comparing intensity (V channel in HSV, or luminance) against chromaticity stability (c1c2c3 or normalized RGB). Regions with low intensity but stable chromaticity relative to their neighbors are shadows. This works exceptionally well on **uniform surfaces like sand and concrete** where the main variation between shadow and non-shadow is pure illumination. On grass, the approach needs more care due to inherent chromaticity variation. HSV-based thresholding — shadows decrease V, increase S, and shift H toward blue — offers the simplest implementation through standard OpenCV operations.

---

## Multi-view geometry and neural reconstruction change the equation

Photogrammetry's multi-view nature creates both unique opportunities for shadow detection and built-in mechanisms that reduce the need for explicit masking.

**Multi-view photo-consistency** is the foundational principle that shadows violate. A correct 3D surface point should appear the same across all views, but a shadowed patch in one view won't match the sunlit version in another. This directly causes holes in dense reconstruction and incorrect depth estimates. The 2025 paper "Fusing Feature Consistency Across Views" introduces a **color invariance module** deriving properties independent of illumination from physics-based reflection models, directly addressing this problem for MVS.

**NeRF-HuGS** (CVPR 2024) provides the most directly relevant multi-view shadow handling insight: SfM feature points belonging to static scene elements are reliably matched and triangulated, while shadow-affected features fail matching. HuGS exploits this to identify transient content by using SfM feature points as "static" prompts for SAM, generating dense masks that exclude shadows and moving objects. This is essentially **using the photogrammetry pipeline's own output to detect shadows**.

**NeRF-W's appearance and transient embeddings** remain the gold standard for handling lighting variation in neural reconstruction. Each training image gets a learnable 48-dimensional appearance embedding that allows the color prediction MLP to "explain away" photometric variations including shadows. The separate 16-dimensional transient embedding models image-dependent content with a data-dependent uncertainty field, allowing the model to discount shadow pixels during optimization.

**For 3D Gaussian Splatting**, multiple wild-scene variants now exist. **WildGaussians** (NeurIPS 2024) extends 3DGS with an appearance modeling module and DINO-based uncertainty predictor, achieving the same inference speed as vanilla 3DGS (github.com/jkulhanek/wild-gaussians). **GS-W** (ECCV 2024) introduces separated per-Gaussian intrinsic and dynamic appearance features — intrinsic features capture material properties while dynamic features capture illumination variation, offering a more physically motivated decomposition. **SpotLessSplats** (Google DeepMind, ACM ToG 2024) uses Stable Diffusion features for semantic clustering to identify transient distractors including shadows, with robust optimization that excludes distractor pixels (github.com/lilygoli/SpotLessSplats). **Splatfacto-W** integrates into the Nerfstudio framework with per-Gaussian neural color features, per-image appearance embeddings, and heuristic transient object masking, improving PSNR by **5.3 dB** over vanilla 3DGS while maintaining 150× faster training than NeRF methods.

**RobustNeRF** (CVPR 2023) takes the simplest approach: treat shadow pixels as outliers and downweight them using a trimmed robust loss. It maintains over **31 dB PSNR** even with increasing fractions of distractor-contaminated images, while vanilla mip-NeRF 360 drops from 33 to 25 dB. This requires no shadow detection at all.

**The practical upshot**: for users whose end goal is a neural 3D representation (NeRF or 3DGS), appearance embeddings make explicit shadow masking substantially less critical. However, for traditional photogrammetry pipelines producing meshes and orthophotos via COLMAP or Metashape, shadow masking remains important because these tools lack equivalent mechanisms.

---

## Video shadow detection enables temporal consistency across frames

Since photogrammetry datasets from 360° video are sequential frames, video shadow detection methods can enforce temporal consistency — ensuring shadow masks don't flicker between frames, which would cause reconstruction artifacts.

**SCOTCH and SODA** (CVPR 2023) is the strongest dedicated video shadow detection method. Its **SODA** (Shadow Deformation Attention Trajectory) module computes pointwise spatial-temporal similarity to track shadow shape changes across frames, while **SCOTCH** (Shadow Contrastive Learning) learns a unified shadow representation from shadow pairs across different videos. The deformation attention is well-suited for photogrammetry video where shadows change gradually with camera motion. Code: github.com/lihaoliu-cambridge/scotch-and-soda.

**ShadowSAM / Detect-AnyShadow** (IEEE TCSVT 2023) is the closest existing method to an ideal photogrammetry workflow. It fine-tunes SAM on shadow data, detects shadows in a key frame with bounding box prompts, then propagates masks across the video using a **Long Short-Term Network (LSTN)** that learns spatial correlations between distant frames. This "detect in key frames + propagate" paradigm is directly applicable to photogrammetry. Code: github.com/harrytea/Detect-AnyShadow.

**STICT** (CVPR 2022) offers the fastest processing at **91.34 fps** — fast enough to process 1,400 frames in under 16 seconds. It converts pre-trained image shadow detectors to video by training predictions to be temporally consistent through interpolation consistency, making it the most practical option for large frame counts.

The critical evaluation of SAM2's video propagation mode for shadows (Jie et al., December 2024, github.com/LeipingJie/SAM2Video) reveals important caveats. Mask prompts work satisfactorily but point prompts perform significantly worse. Most critically, **accuracy degrades as frame count increases** — beyond ~100 frames, mask quality drops notably. For 1,400 frames, you would need to re-prompt every 30–50 frames, requiring 30–47 key frames rather than just one or two.

The **ViSha** dataset (120 videos, 11,685 frames) and the newer **CVSD** dataset (196 clips, 19,757 frames with complex overlapping shadow patterns from ECCV 2024) provide training data for video shadow detection.

A practical pipeline for photogrammetry video would use three tiers: run a strong single-image shadow detector (SAM2-Adapter) on key frames every 30–50 frames, propagate masks between key frames using ShadowSAM's LSTN or SAM2's video mode, and apply optical flow-based consistency checking as a post-processing validation step.

---

## Intrinsic decomposition and alternative approaches offer fresh angles

**Intrinsic image decomposition** may be the most underexplored approach for photogrammetry shadow handling. By separating an image into reflectance (albedo) and shading components, shadows naturally fall entirely into the shading layer.

The **Careaga & Aksoy pipeline** (ACM TOG 2023, extended in SIGGRAPH Asia 2024 with Best Paper Honorable Mention) represents the current state of the art. It estimates ordinal shading maps at multiple resolutions, producing robust albedo/shading separation on arbitrary in-the-wild images. The tool is pip-installable (`pip install https://github.com/compphoto/Intrinsic/archive/main.zip`) and works without any training or fine-tuning. For outdoor photogrammetry on uniform surfaces, the albedo should be highly consistent, so **any variation in the shading layer directly corresponds to shadow and lighting effects**. Shadow mask extraction becomes: compute shading S = I/R, threshold the result.

**IntrinsicAnything** (2024) uses diffusion model priors for inverse rendering under unknown illumination, estimating albedo from single images (github.com/zju3dv/IntrinsicAnything). **IDArb** (ICLR 2025) extends intrinsic decomposition to arbitrary numbers of input views under varying illuminations with cross-view, cross-domain attention — potentially ideal for multi-view photogrammetry datasets. The AAAI 2025 paper "When Shadow Removal Meets Intrinsic Image Decomposition" directly couples shadow removal with IID using unpaired training data.

**Grounding DINO with "shadow" as text prompt** has indirect validation. The OPSeg paper explicitly uses Grounding DINO to generate bounding box prompts for SAM2-based instance shadow detection, confirming that Grounding DINO can localize shadows. The **Grounded-SAM pipeline** (github.com/IDEA-Research/Grounded-Segment-Anything) combines Grounding DINO bounding boxes with SAM2 pixel masks. This offers a zero-shot baseline — prompt with "shadow" or "cast shadow" and evaluate. Expect moderate performance on obvious hard shadows on uniform surfaces but poor results on soft shadows. This works best as a **quick coarse annotation tool** rather than a production detector.

**Depth-based shadow prediction** from reconstructed geometry is uniquely viable for photogrammetry. Since the pipeline already produces 3D point clouds or meshes, you can ray-trace shadows from the known sun position (computed from GPS coordinates, timestamp, and date using libraries like `pvlib` or `suncalc`) against the reconstructed geometry. This predicts where shadows should fall without any learned model. The **OutCast** paper (CGF 2024) demonstrates a learned image-space ray-marching layer that converts approximate monocular depth into shadow predictions, but for photogrammetry the existing high-quality geometry makes classical ray-tracing more appropriate.

**IC-Light** (ICLR 2025, github.com/lllyasviel/IC-Light) is the most accessible relighting tool, using diffusion models to relight images with text conditioning. However, relighting approaches are fundamentally generative — they synthesize new pixel values rather than identifying shadows, making them **problematic for photogrammetry** where preserving original pixel data matters for measurement accuracy. The best use is indirect: relight to uniform illumination, compute the difference with the original, and threshold that difference as a shadow mask.

---

## No photogrammetry tool has built-in shadow detection

A systematic review of all major photogrammetry software confirms a universal gap.

**Agisoft Metashape** has no shadow detection. It supports importing external masks via Python API (`chunk.importMask(path, source=MaskSourceFile)`) and includes a Texture De-Lighter for post-hoc shadow reduction in textures. The Metashape 2.2 release integrated automatic background masking (previously the `automatic_masking.py` script using rembg/U²-Net), but this targets background removal, not shadows. The official scripts repository (github.com/agisoft-llc/metashape-scripts) includes `masking_by_color_dialog.py` for color-based masking with tolerance.

**COLMAP** explicitly warns users to "capture images at similar illumination conditions" and "avoid high dynamic range scenes." It supports masks during feature extraction via `--ImageReader.mask_path` — PNG images where pixel value 0 means features are extracted and 255 means masked. COLMAP's **RootSIFT** descriptors (the default normalization) provide some illumination invariance, and RANSAC-based geometric verification rejects some shadow-caused mismatches, but these are partial measures.

**RealityCapture** (now RealityScan 2.0) has no shadow detection but offers the useful **Image Layers** feature: users can create a "geometry layer" with shadow-reduced images for alignment and a separate "texture layer" for final texturing. **Pix4D** has integrated SAM into PIX4Dmatic for general semantic segmentation but not shadow-specific detection. **OpenDroneMap** has no shadow handling at all.

The most complete practical integration found is **SlicerMorph Photogrammetry** (github.com/SlicerMorph/SlicerPhotogrammetry, Biology Open 2025), which uses SAM for mask generation and NodeODM for reconstruction, reducing mean surface error by **10–15%** compared to unmasked workflows. However, it targets background removal for turntable scanning, not shadow masking. The **S-EO project** (centreborelli.github.io/shadow-eo/) demonstrates the most advanced shadow+3D integration but targets satellite photogrammetry, using sun position and LiDAR DSMs to generate shadow masks that improve EO-NeRF reconstruction.

**The missing pipeline component** is straightforward to build: run a shadow detection model on each frame, save the output as binary PNG masks, and pass them to COLMAP via `--ImageReader.mask_path` or Metashape via `importMask()`. Community workflows from 3DFlow Zephyr show a manual version using Photoshop's Select > Color Range > Shadows action batched across images. Vlad Kuzmin's photogrammetry guide (80.lv) recommends RAW preprocessing to flatten shadows (Shadows +50 to +100, Highlights -50 to -100) as a pragmatic alternative to masking.

---

## Domain gap is the critical practical challenge

Standard shadow detection benchmarks (SBU, ISTD, CUHK-Shadow) have significant limitations for photogrammetry applications. Cross-dataset evaluation from the Unveiling Deep Shadows benchmark shows **BER roughly doubles** when moving between datasets — SDDNet achieves 4.86 BER on SBU but 7.65 on SRD, demonstrating substantial domain gap even between similar benchmarks.

The specific challenges for outdoor photogrammetry shadows are well-documented. **Dark albedo materials are frequently misclassified as shadows** — the most common failure case across all models, directly relevant for dark-colored sand, wet concrete, or dark grass. Shadow boundary errors account for **65% of false negatives** within 10 pixels of boundaries, and the penumbral regions of harsh noon shadows are poorly represented in training data. ISTD contains only 135 distinct backgrounds with mostly hard building/pole shadows, while photogrammetry scenes feature articulated human shadows and equipment shadows with complex shapes.

Resolution mismatch compounds the problem. Standard benchmarks use **256–640px** images, while photogrammetry frames are typically 4,000–8,000px or higher. The Detail-Preserving Latent Diffusion paper (CVPR 2025) specifically addresses this, noting that current methods are "prone to overfitting training data, often leading to reduced performance on unseen cases."

The S-EO paper identifies a crucial insight: **domain-appropriate training data matters more than model architecture**. Even a simple U-Net baseline, when properly trained on domain-appropriate data, "performs on par with specialized shadow detection models." This suggests that fine-tuning any reasonable architecture on a small set of annotated photogrammetry shadow images would likely outperform running a sophisticated model trained only on SBU/ISTD.

Three mitigation strategies stand out. TICA's test-time adaptation exploits intensity consistency without retraining. Foundation-model approaches (SAM2-Adapter) offer better generalization through rich pretrained representations. And classical chromaticity methods bypass the domain gap entirely by using physics-based invariants that generalize perfectly.

---

## Shadows affect texturing most, geometry moderately, alignment least

The impact of shadows varies substantially across the photogrammetry pipeline stages. **Texturing is most affected** — shadows are unconditionally baked into texture maps, creating permanent artifacts visible in the final 3D model. Paul Bourke (2019) summarizes the issue: photogrammetry captures "not just the geometry and texture, but also the lighting conditions."

**Dense reconstruction (MVS) is significantly affected.** Photo-consistency is directly broken by shadows — a shadowed patch in one view won't match the sunlit version in another. The MCSP paper (2025) demonstrates that shadow occlusion areas cause poor disparity estimation in urban stereo matching. The Shading-Aware MVS paper (ECCV 2016) confirms that "cast shadows can be matched well with stereo matching" in some cases, but "on sparse datasets, our shading term causes holes in the shadowed areas."

**Feature matching (SfM) is moderately affected.** SIFT features are gradient-based descriptors with inherent brightness normalization, providing partial illumination invariance. COLMAP's default RootSIFT further improves this. However, shadow boundaries create false edges generating unreliable keypoints, and the NeRF-HuGS paper confirms that shadow-affected features fail triangulation.

COLMAP's existing robustness mechanisms — RANSAC outlier rejection, NCC-based PatchMatchStereo with brightness normalization, geometric consistency checks in stereo fusion — provide **partial but insufficient** shadow handling. Persistent shadows (same position across all views from similar viewpoints) will be accepted as valid geometry and texture. Shadow boundaries create systematic matching errors that can pass geometric verification.

For traditional pipelines, the practical recommendation is a tiered approach: shadow masking before SfM for large, inconsistent shadows; mosaic blending mode for texturing (which selects best-exposed pixels across views); and Agisoft's Texture De-Lighter or manual de-shadowing as post-processing. For neural pipelines, appearance embeddings handle most lighting variation automatically.

---

## Conclusion: a recommended integrated pipeline

The research reveals that an effective photogrammetry shadow masking pipeline should combine multiple complementary approaches rather than relying on any single method.

**For detection**, the strongest practical approach pairs a learned model with classical verification: run SAM2-Adapter or ShadowAdapter on each frame for pixel-level shadow masks, then validate using c1c2c3 chromaticity stability — regions that the model labels as shadow but have unstable chromaticity are likely false positives (dark objects, not shadows). For video sequences, process key frames every 30–50 frames with the learned detector and propagate masks using ShadowSAM's LSTN or SAM2's video mode.

**For reconstruction**, the choice depends on the target pipeline. If using NeRF or 3DGS, appearance embedding variants (WildGaussians, Splatfacto-W, SpotLessSplats) substantially reduce the need for explicit masking. If using COLMAP or Metashape, feed shadow masks through `--ImageReader.mask_path` or `importMask()` respectively, and use mosaic blending for texturing.

**The highest-leverage underexplored approach** is intrinsic image decomposition via the Careaga & Aksoy pipeline (github.com/compphoto/Intrinsic). On uniform outdoor surfaces, the shading layer directly reveals shadows without any shadow-specific training, bypassing the domain gap problem entirely. Computing shadow masks as thresholded shading maps may prove more robust than any benchmark-trained detector for this specific use case.

**For domain gap mitigation**, annotating even 20–50 frames from the target photogrammetry domain and fine-tuning an existing model would likely yield the single largest quality improvement. Alternatively, TICA's test-time adaptation adapts models during inference without any annotated target data. The S-EO project demonstrates that for geometry-known scenes, sun position + 3D geometry can generate training labels automatically — directly applicable to photogrammetry datasets where both are available from the reconstruction pipeline itself.

Key repositories for implementation:
- Shadow detection benchmark and weights: github.com/xw-hu/Unveiling-Deep-Shadows
- SAM2-Adapter for shadow detection: github.com/tianrun-chen/SAM-Adapter-PyTorch
- ShadowAdapter with auto-prompting: github.com/LeipingJie/AdapterShadow
- Video shadow detection: github.com/lihaoliu-cambridge/scotch-and-soda
- ShadowSAM video propagation: github.com/harrytea/Detect-AnyShadow
- Intrinsic decomposition: github.com/compphoto/Intrinsic
- Wild 3DGS: github.com/jkulhanek/wild-gaussians
- SpotLessSplats: github.com/lilygoli/SpotLessSplats
- Finlayson illumination invariant: github.com/hangong/gfinvim
- OPSeg (Grounding DINO + SAM2): github.com/deyang2000/OpSeg
- Grounded SAM 2: github.com/IDEA-Research/Grounded-Segment-Anything