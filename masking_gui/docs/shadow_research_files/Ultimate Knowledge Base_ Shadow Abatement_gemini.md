# **The Radiometric Paradox and the Consensus on Shadow Abatement: A Comprehensive Technical Analysis of Photogrammetric De-lighting Architectures (2025–2026)**

## **1\. Introduction: The Divergence of Geometry and Radiometry**

The discipline of photogrammetry currently stands at a critical juncture, defined by a fundamental conflict known as the **Radiometric Paradox**. To reconstruct 3D geometry via Structure-from-Motion (SfM), algorithms require high-contrast features to resolve depth and triangulate tie points. However, the very lighting phenomena that produce this necessary contrast—specifically cast shadows and specular highlights—act as destructive noise during the subsequent phases of dense matching (Multi-View Stereo) and texture generation.1  
For decades, the industry standard accepted "baked-in" lighting as an unavoidable artifact of the capture process. However, the 2025–2026 consensus, derived from a synthesis of technical benchmarks, user adoption patterns ("polls"), and academic literature, indicates a paradigm shift toward **Radiometric Fidelity**. The modern objective is the creation of "Digital Twins"—assets possessing intrinsic material properties (Albedo, Roughness, Metallicity) capable of reacting physically to virtual lighting environments, rather than static meshes locked to the illumination conditions of their capture.1  
This report synthesizes the "ultimate knowledge base" requested, aggregating data from the **NTIRE 2025 Shadow Removal Challenge**, the **Unveiling Deep Shadows Benchmark**, and the operational realities of software ecosystems like **Agisoft Metashape**, **Adobe Substance 3D Sampler**, and the now-discontinued **Unity ArtEngine**. It serves as a definitive guide to the workflows, tools, and mathematical foundations required to bridge the gap between geometric accuracy and radiometric purity.

### **1.1 The Mechanisms of Shadow Corruption**

To address the problem effectively, one must quantify the specific mechanisms by which shadows corrupt the reconstruction pipeline. The consensus identifies three primary failure modes:

1. **Transient Feature Triangulation (Phantom Geometry):** In dynamic capture scenarios, such as drone surveys conducted over several hours or time-lapse datasets, shadow boundaries move relative to the physical surface. Feature detectors (e.g., SIFT, SuperPoint) identify these high-contrast shadow edges as keypoints. Because these points shift between frames, the SfM algorithm creates "outliers"—3D points that float in space or create false depressions/spikes in Digital Surface Models (DSMs).1  
2. **Stationary Shadow Artifacts (Alignment Collapse):** In turntable photogrammetry, the object rotates while the camera and lights remain fixed. Consequently, shadows cast by the object onto the turntable background remain static in the camera's frame of reference. Unmasked, the software interprets these shadows as valid static features, leading to "alignment collapse," where the solver incorrectly concludes the camera never moved relative to the subject.1  
3. **Photo-Consistency Violation (Dense Matching Failure):** Multi-View Stereo (MVS) relies on the assumption of photo-consistency—that a surface point looks radiometrically similar across multiple views. This similarity is often measured via **Normalized Cross-Correlation (NCC)**. When a shadow falls across a surface in one view but not another, the histogram of the pixel patch changes drastically, dropping the NCC score below the acceptance threshold. This results in data rejection, manifesting as "holes" in the mesh or high-frequency surface noise.1

## **2\. Acquisition-Phase Paradigms: The Physics of Suppression**

The "poll" of industry experts and high-end practitioners reveals a unanimous preference: prevention is superior to cure. Physical suppression of shadows and highlights during acquisition remains the "Gold Standard" for high-fidelity assets, as it preserves the original data signal without the potential hallucinations introduced by AI inpainting.1

### **2.1 Cross-Polarization: The Radiometric Gold Standard**

Cross-polarization is the primary method for capturing shadow-free albedo and suppressing specular highlights. It is essential for generating PBR-ready textures where albedo must be devoid of lighting information.

#### **2.1.1 Optical Mechanics**

The workflow requires placing a **Linear Polarizing Film** over the light sources (strobes or LEDs) and a **Circular Polarizer (CPL)** on the camera lens. The critical operational parameter is **orthogonal alignment**: the transmission axis of the lens polarizer must be rotated 90 degrees relative to the axis of the light source polarizer.1  
The physics of this interaction allows for the separation of reflectance components:

* **Specular Reflection:** Light bounces off the surface interface (glare) retaining its original polarization state. Because the lens polarizer is orthogonal, this light is absorbed (blocked).  
* **Diffuse Reflection:** Light penetrates the material, scatters efficiently (subsurface scattering), and is re-emitted as the object's true color. This scattering process "depolarizes" the light. Since this light is now unpolarized, the lens polarizer permits it to pass.1

#### **2.1.2 The "Shadow Lifting" Workflow**

By eliminating specular highlights, cross-polarization significantly compresses the dynamic range of the image. This allows practitioners to utilize a "Shadow Lifting" workflow in RAW processing (e.g., Adobe Lightroom or Camera RAW). Exposure settings can be adjusted aggressively (Shadows \+100, Blacks \+50) to neutralize ambient occlusion and cast shadows without the risk of "clipping" or "blowing out" specular highlights, as those highlights have already been optically removed. This results in a flat, high-fidelity Albedo map.1  
**Table 1: Cross-Polarization Limitations and Requirements**

| Constraint | Description | Mitigation Strategy |
| :---- | :---- | :---- |
| **Light Loss** | The polarizing filters absorb approximately 1.5 to 2 stops of light ($\~75\\%$ transmission loss). | Use high-output strobes (e.g., Godox AR400) or high-ISO sensors. 1 |
| **Metallicity Failure** | Metallic surfaces reflect light without subsurface scattering, preserving polarization. They appear black in cross-polarized views. | Use standard unpolarized capture for metallic maps; blend in post. 1 |
| **Heat Sensitivity** | Linear polarizing film can melt or warp under continuous modeling lights. | Use strobes (flash) rather than continuous hot lights; use heat-resistant film. |

### **2.2 The Reference Rig Workflow (Epic Games Standard)**

For uncontrolled outdoor environments (e.g., cliff faces, large-scale terrain), physical light modification is impossible. The industry consensus, popularized by Epic Games for the *Kite Demo* and Quixel Megascans, relies on a "Reference Rig" to capture ground truth lighting data for later mathematical subtraction.2

#### **2.2.1 Rig Composition and Data Capture**

The standard rig comprises:

1. **50% Chrome Sphere:** Captures the directionality, sharpness, and environmental context of the incident light. It acts as a physical light probe.  
2. **18% Grey Sphere:** Provides a verified mid-tone reference to measure contrast ratios between lit and shadowed areas (the "lighting delta").  
3. **Macbeth Color Checker:** Ensures color constancy across exposure brackets.1

#### **2.2.2 The Delighting Equation**

The captured chrome ball reflection is unwrapped (using tools like Nuke, PTGui, or specialized scripts) into a spherical **HDRI**. This HDRI serves as the "Ground Truth" for the lighting environment. The delighting process is then a mathematical inversion of the rendering equation:

$$\\text{Albedo}(\\rho) \\approx \\frac{\\text{Captured Radiance}(L)}{\\text{Calculated Irradiance}(E)}$$  
By projecting the unwrapped HDRI onto the reconstructed mesh in a 3D package (Blender/Maya) and rendering the *irradiance* (light) pass, artists create a map of "pure light." Dividing the original texture by this light map cancels out shadows and gradients, leaving only the albedo.2

### **2.3 The Wandering Light (Light Painting) Method**

For static studio subjects, the "Wandering Light" method offers a cost-effective alternative to multi-light rigs.

* **Technique:** The camera is set to a long exposure (typically 2–5 seconds). The operator moves a handheld light source continuously in a hemispherical pattern around the subject during the exposure.  
* **Radiometric Result:** By illuminating the object from hundreds of angles during a single frame integration, cast shadows are "averaged out" to the point of invisibility. The resulting image simulates a perfectly uniform ambient dome light.1

## **3\. The 2026 Software Landscape: Consensus and "Abandonware"**

A significant portion of the user inquiry relates to the status of specific software tools. The analysis reveals a volatile market where once-dominant tools have been deprecated, creating a "void" that is being filled by a mix of commercial updates and open-source innovation.

### **3.1 The Fall of Unity ArtEngine**

**Unity ArtEngine**, once a staple for "Example-Based" material creation and delighting (Mutation, Seam Removal), has been officially **discontinued** as of 2025/2026.

* **Status:** Development has ceased. The desktop application is no longer maintained, and its features are not being directly ported to a single successor within the Unity ecosystem in the same form.3  
* **The Void:** This has left a gap for "delighting and seam removal" workflows. Users who relied on ArtEngine for "Texture Mutation" are now forced to migrate to alternative tools or custom AI workflows.3

### **3.2 The Successor Consensus: Metashape vs. Sampler**

In the wake of ArtEngine's decline, the industry consensus has bifurcated between **Agisoft Metashape** and **Adobe Substance 3D Sampler**, depending on the user's specific needs.6  
**Table 2: Industry Consensus – Metashape vs. Substance 3D Sampler (2025)**

| Feature | Agisoft Metashape Pro (v2.3) | Adobe Substance 3D Sampler |
| :---- | :---- | :---- |
| **Primary Role** | Full Photogrammetry & Geometry Engine | Material Authoring & Texturing Tool |
| **Masking Approach** | Built-in "Automatic (AI)" masking; Python API. | "Black Box" AI; limited user control. |
| **Performance** | High speed on large datasets (e.g., 326 images in \~12 mins). | Slower on large sets (\~45 mins for same set). |
| **Light Sensitivity** | Robust in varied lighting; creates high-detail meshes. | "Craves Light"; struggles with low-contrast/overcast data. |
| **Delighting** | "Texture De-Lighter" tool (Cast shadow removal). | Integrated "Image to Material" delighting. |
| **User Base** | Surveyors, High-End VFX, Large Scale. | Texture Artists, Game Devs, Quick PBR creation. |
| **Consensus** | **Winner for Geometry/Scale.** | **Winner for Ease of Use/Material Integration.** |

6

### **3.3 The Integration Gap**

A critical finding of this report is the **Integration Gap**. Despite the availability of advanced detection models (Section 4), no major photogrammetry software (Metashape, COLMAP, RealityCapture) currently features *native*, *deep-learning-based* shadow detection pipelines that utilize state-of-the-art transformers. Users are currently forced to bridge this gap using custom Python scripts and intermediate mask files.1

## **4\. Algorithmic Shadow Detection: The 2025–2026 "Poll" Winners**

When physical suppression is impossible (e.g., aerial mapping, sunlight), the workflow shifts to 2D pre-processing. The "poll" results—interpreted here as the outcomes of the **Unveiling Deep Shadows Benchmark** and the **NTIRE 2025 Challenge**—provide a clear hierarchy of models based on accuracy (BER) and speed (FPS).

### **4.1 Failure of Heuristics (masking\_v2.py)**

The report confirms that legacy heuristic scripts (like masking\_v2.py) are obsolete. These scripts typically rely on:

* Converting images to grayscale (ignoring chromaticity).  
* Thresholding pixels below 70% of local median brightness.  
* Dilating person-masks downward to find "attachment zones." **Consensus:** These methods fail catastrophically in complex scenes because they cannot distinguish dark albedo (asphalt, black clothing) from shadows, miss angled shadows (low sun), and cannot handle penumbras.1

### **4.2 Validated Deep Learning Models**

The following models are validated as the current leaders for integration into photogrammetry pipelines.

#### **4.2.1 SDDNet (Segment Dark and Difficult Net)**

* **Rank:** **\#1 for Speed/Throughput.**  
* **Architecture:** Lightweight ResNeXt-101 backbone with dual-layer disentanglement.  
* **Performance:** Processes \~32 FPS on an RTX 4090\. A dataset of 1,400 images can be masked in approximately 44 seconds.  
* **Accuracy:** Competitive BER of **4.86** on SBU (512x512).  
* **Use Case:** The "Workhorse" for large-scale aerial or drone datasets where processing time is the bottleneck.1

#### **4.2.2 SILT (Shadow-aware Iterative Label Tuning)**

* **Rank:** **\#1 for Noise Robustness.**  
* **Architecture:** Transformer-CNN hybrid (PVTv2 backbone).  
* **Innovation:** Uses iterative self-training to refine noisy labels in training data, teaching the network to reject "false shadows" (e.g., dark vegetation).  
* **Performance:** BER of **4.19** on SBU.  
* **Use Case:** High-fidelity terrestrial scanning where false positives (masking out parts of the object) must be minimized.1

#### **4.2.3 FDRNet (Feature Decomposition and Reconstruction)**

* **Rank:** **\#1 for Edge Precision.**  
* **Architecture:** Explicitly models illumination and applies **Conditional Random Fields (CRF)** as a core layer.  
* **Performance:** Lowest BER of **3.04** (with CRF).  
* **Constraint:** Requires legacy dependencies (pydensecrf, PyTorch 1.5) that are difficult to compile on Windows, often requiring Docker/Linux.1

### **4.3 Foundation Model Adaptations (SAM)**

The **Segment Anything Model (SAM)** is "shadow-blind" in its vanilla form. 2025 research has focused on adapting it.

* **AdapterShadow (ShadowAdapter):** Solves the manual prompting issue by using an auxiliary network to generate an automatic grid of point prompts based on shadow features. It allows for batch processing of thousands of frames without human interaction.1  
* **SoftShadow (CVPR 2025):** Represents the shift from binary masking to **Penumbra-Aware** masking. It predicts a "soft mask" distinguishing Umbra (hard shadow) from Penumbra (gradient). This is critical for de-lighting, as it prevents the "halo artifacts" common when removing shadows using binary masks.1

## **5\. Shadow Removal (Delighting) Architectures**

Detection creates the mask; Delighting performs the removal.

### **5.1 RASM and ReHiT (NTIRE 2025 Winners)**

The **NTIRE 2025 Shadow Removal Challenge** highlighted two dominant architectures:

* **RASM (Regional Attention Shadow Removal Model):** Uses a "divide-and-conquer" strategy. Instead of global attention (which loses high-frequency texture), it predicts shadow pixel values by analyzing the immediate non-shadowed neighborhood. It outputs a shadow-free albedo map.1  
* **ReHiT (Recurrent Hierarchical Transformer):** A "mask-free" workflow designed for complex urban environments. It uses a recurrent structure to progressively "lift" shadows, preserving structural details in complex scenes like alleyways.1

### **5.2 Intrinsic Image Decomposition (IID)**

An underexplored workflow for uniform surfaces involves the **Careaga & Aksoy Pipeline**.

* **Concept:** Decomposes an image into **Reflectance (Albedo)** and **Shading** layers without training on shadow datasets.  
* **Shadow Extraction:** Since shadows are purely illumination, they reside entirely in the Shading layer. A mask can be extracted by thresholding the shading map ($S \= I / R$). This is robust for outdoor scenes where training data might not generalize.1

### **5.3 ShadowFormer "Diff" Workflow**

A practical workflow using **ShadowFormer**:

1. Input shadowed image \-\> ShadowFormer \-\> Output "Clean" image.  
2. Calculate Difference: $\\text{Diff} \= | \\text{Input} \- \\text{Output} |$.  
3. Threshold the Difference map to create a mask.  
* **Assessment:** Effective for harsh noon shadows but prone to noise if the model hallucinates texture details.1

## **6\. Neural Reconstruction: The Shift Away from Masking**

A critical finding in the 2025/2026 literature is that for **Neural Radiance Fields (NeRF)** and **3D Gaussian Splatting (3DGS)**, explicit shadow masking is becoming obsolete.

### **6.1 Appearance Embeddings**

Models like **WildGaussians** and **Splatfacto-W** (Nerfstudio) utilize per-image **Appearance Embeddings**.

* **Mechanism:** The network learns a latent vector for each image that explains photometric variations (exposure, shadows) separate from the static geometry.  
* **Result:** The model "learns" that the shadow is a transient effect specific to that view, not a permanent geometric feature. This allows for high-fidelity reconstruction (improving PSNR by \~5.3 dB) without manual masking.1

### **6.2 ShadowGS and SSD-GS**

Emerging frameworks explicitly model light physics:

* **SSD-GS (2026):** Decomposes the scene into **Diffuse, Specular, Shadow, and Scattering** components. It effectively "relights" the scene during the reconstruction process.1  
* **ShadowGS:** Tailored for satellite imagery, it uses sun position data to disentangle shadows from albedo, solving the multi-date consistency problem.

## **7\. Operational Guide: Implementing the "Best Consensus" Pipeline**

To operationalize these findings, the report recommends the following integrated pipeline for professional photogrammetry.

### **7.1 Step-by-Step Implementation**

1. **Detection (The Hybrid Approach):**  
   * Run **SDDNet** (for speed) or **SILT** (for accuracy) on the dataset to generate initial binary masks.  
   * Run **Object Detection** (YOLO/SAM) to mask the subject (person/tripod).  
   * **Logic:** $ \\text{Final Mask} \= (\\text{Shadow Mask} \\cap \\text{Spatial Filter}) \\cup \\text{Object Mask} $. The spatial filter (e.g., dilated radius around the object) ensures only shadows *cast by the subject* are masked, preserving environmental shadows.1  
2. **Verification:**  
   * Use **Chromaticity Stability** (c1c2c3 space) to verify the masks. If a masked region shows stable chromaticity but low luminance, it is a confirmed shadow. If chromaticity varies, it may be a dark object (False Positive).1  
3. **Import:**  
   * **Metashape:** Use chunk.importMasks(path='masks/{filename}.png').  
   * **COLMAP:** Use \--ImageReader.mask\_path during feature extraction.  
4. **Reconstruction:**  
   * Execute SfM/MVS with masks enabled to prevent "phantom geometry."

### **7.2 Code and Dependencies**

* **SDDNet:** Requires Python 3.6, PyTorch 1.10. Code at github.com/rmcong/SDDNet\_ACMMM23.  
* **SILT:** Requires Python environment with omegaconf. Code at github.com/hanyangclarence/SILT.  
* **SpotLessSplats:** For 3DGS, use pip install git+https://github.com/lilygoli/SpotLessSplats.git.1

## **8\. Conclusion**

The "ultimate knowledge base" for shadow masking in 2026 confirms that while physical suppression (Cross-Polarization) remains the gold standard, digital workflows have matured significantly. The consensus favors dedicated Deep Learning models (**SDDNet**, **SILT**) over heuristics, and highlights the rapid emergence of "mask-free" neural reconstruction (**ShadowGS**, **WildGaussians**) as the future of the field. By adopting the tiered workflows outlined above, practitioners can effectively resolve the Radiometric Paradox and achieve production-grade Digital Twins.

### **Summary of Recommended Tools**

| Tool | Primary Strength | Best Use Case | Key Constraint |
| :---- | :---- | :---- | :---- |
| **Cross-Polarization** | Physical Perfection | Studio/Texture Capture | Fails on Metals; Light Loss |
| **SDDNet** | Speed (32 FPS) | Large Aerial/Drone Sets | Windows compilation (pydensecrf) |
| **SILT** | Noise Robustness | Ground-level / Complex Scenes | Batch setup required |
| **AdapterShadow** | Automation | Batch processing (No clicks) | Requires SAM weights |
| **ShadowGS** | Physics-Based | Satellite/Aerial 3DGS | Research code (Satellite focus) |
| **Metashape AI** | Ease of Use | Quick Commercial Turnaround | "Black Box" (Low control) |

6

#### **Works cited**

1. Shadow Detection for Photogrammetry Masking (2025–2026 Update).pdf  
2. Removing AO from photogrammetry — polycount, accessed February 7, 2026, [https://polycount.com/discussion/161470/removing-ao-from-photogrammetry](https://polycount.com/discussion/161470/removing-ao-from-photogrammetry)  
3. ArtEngine (legacy) \- Unity, accessed February 7, 2026, [https://create.unity.com/unity-artengine-sunset](https://create.unity.com/unity-artengine-sunset)  
4. What's Next: Unity Engine 2025 Roadmap, accessed February 7, 2026, [https://unity.com/blog/unity-engine-2025-roadmap](https://unity.com/blog/unity-engine-2025-roadmap)  
5. Unity ArtEngine \- How To \- Seam Removal Structure \- YouTube, accessed February 7, 2026, [https://www.youtube.com/watch?v=OqxQunzrM88](https://www.youtube.com/watch?v=OqxQunzrM88)  
6. After messing with Metashape over the past few days, I pulled the trigger on it. Overall very impressed with it. Here my results comparing with Adobe over two datasets. : r/photogrammetry \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/photogrammetry/comments/1aj2pyc/after\_messing\_with\_metashape\_over\_the\_past\_few/](https://www.reddit.com/r/photogrammetry/comments/1aj2pyc/after_messing_with_metashape_over_the_past_few/)  
7. Agisoft Delighting vs Unity Delighting Vs Substance Alchemist Delighting?, accessed February 7, 2026, [https://www.agisoft.com/forum/index.php?topic=12504.0](https://www.agisoft.com/forum/index.php?topic=12504.0)  
8. SSD-GS: Scattering and Shadow Decomposition for Relightable 3D Gaussian Splatting, accessed February 7, 2026, [https://openreview.net/forum?id=7m2Dqz9g05](https://openreview.net/forum?id=7m2Dqz9g05)