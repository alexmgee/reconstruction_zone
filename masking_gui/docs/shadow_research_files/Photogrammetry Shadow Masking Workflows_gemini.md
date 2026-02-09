# **Advanced Workflows for Shadow Abatement and Delighting in Photogrammetry: A 2026 Technical Report**

## **1\. Introduction: The Radiometric Paradox in 3D Reconstruction**

The pursuit of photorealism in three-dimensional reconstruction has fundamentally shifted the requirements of photogrammetry from mere geometric accuracy to complete radiometric fidelity. In the domain of modern digital preservation, game development, and visual effects, the "Digital Twin" is no longer satisfied by a mesh that simply matches the shape of the physical object; it must also possess surface properties—specifically albedo, roughness, and metallicity—that react physically to virtual lighting. Herein lies the central paradox of the discipline: the very shadows that provide the contrast necessary for algorithms to perceive depth are the primary contaminants that degrade the utility of the final asset.  
Shadows, in the context of photogrammetry, represent a dualistic phenomenon. To the feature detection algorithm—searching for high-frequency gradients to establish correspondence between images—a sharp shadow edge is a robust feature, often more distinct than the texture of the object itself. However, to the texture generation and dense matching pipeline, shadows are destructive noise. They introduce transient radiometric inconsistencies where a single point on a surface exhibits drastically different luminance values across different viewing angles, violating the assumption of photo-consistency that underpins Multi-View Stereo (MVS). Furthermore, when these shadows are "baked" into the final texture map, they lock the asset into a static lighting environment, rendering it unusable in dynamic virtual worlds where the sun position or artificial lighting may change.1  
This report validates and expands upon the existing workflows for masking and removing shadows from photogrammetry datasets. By synthesizing data from the NTIRE 2025 Shadow Removal Challenge 2, documentation from industry-standard software like Agisoft Metashape and RealityCapture 4, and production methodologies from studios such as Epic Games 6, we establish a comprehensive taxonomy of shadow management. The analysis is structured to guide the practitioner through the entire pipeline: from physical intervention during acquisition, through algorithmic detection and masking during reconstruction, to mathematical de-lighting in post-processing.

### **1.1 The Economic and Technical Imperative**

The failure to adequately manage shadows results in significant downstream costs. In the context of large-scale aerial mapping, moving shadows from clouds or the sun can cause "phantom geometry"—false depressions or spikes in Digital Surface Models (DSMs)—that require hours of manual sculpting to correct.7 In e-commerce and virtual production, assets with baked-in shadows require manual texture painting to restore the albedo, a process that defeats the efficiency gains of scanning.1 Therefore, the validation of automated or semi-automated shadow masking workflows is not merely an academic exercise but a critical production requirement.  
The following sections detail the physics of shadow corruption, the hardware solutions to prevent it, and the software pipelines to repair it, creating a definitive reference for high-fidelity 3D reconstruction in 2026\.

## ---

**2\. The Physics and Mathematics of Shadow Corruption**

To validate a workflow for shadow removal, one must first understand the specific mechanisms by which shadows corrupt photogrammetric data. The corruption occurs at two distinct levels: the geometric level (affecting the mesh) and the radiometric level (affecting the texture).

### **2.1 Geometric Corruption: Feature Matching and False Tie Points**

The foundation of Structure-from-Motion (SfM) lies in the detection of invariant features. Algorithms such as SIFT (Scale-Invariant Feature Transform) or its modern variants detect "key points" by analyzing local contrast gradients in the image. A cast shadow, particularly one cast by a hard light source like the sun, creates a strong gradient boundary.8

#### **2.1.1 The Transient Feature Problem**

In datasets where the lighting changes over time—such as a drone survey conducted over several hours or a time-lapse capture—the position of cast shadows shifts relative to the static geometry. SfM algorithms, however, assume that the scene is static. If a feature detector locks onto a shadow corner in Image $t\_0$ and matches it to a shadow corner in Image $t\_1$, the triangulation process will attempt to solve for a 3D point that satisfies both projections. Since the shadow has moved across the surface, the triangulated point will not lie on the physical surface of the object. Instead, it will float in space or be buried beneath the mesh, creating "outliers" in the sparse point cloud. These false tie points introduce stress into the bundle adjustment, potentially warping the entire camera alignment solution.9

#### **2.1.2 The Stationary Shadow Artifact**

Conversely, in turntable photogrammetry, the object rotates while the camera and lights remain fixed. In this scenario, shadows cast by the object onto the turntable or background are stationary in the camera's reference frame. To the SfM algorithm, these stationary shadows appear as valid, high-quality features that obey the epipolar geometry of a *static* scene. If the algorithm prioritizes these background features over the rotating object features (which appear to be moving and thus invalid in a static-camera solve), the alignment will collapse. The software will calculate that the camera never moved, resulting in all images aligning to a single location.10 This necessitates the rigorous masking workflows detailed in Section 5\.

### **2.2 Radiometric Corruption: Dense Matching Failure**

Once camera poses are established, the pipeline moves to Multi-View Stereo (MVS) to generate a dense point cloud or depth maps. MVS relies on "photo-consistency," the assumption that a patch of surface $P$ looks radiometrically similar in Image $A$ and Image $B$.  
The similarity is often measured using metrics like Normalized Cross-Correlation (NCC).

$$NCC(P\_A, P\_B) \= \\frac{\\sum (A\_i \- \\bar{A})(B\_i \- \\bar{B})}{\\sqrt{\\sum (A\_i \- \\bar{A})^2 \\sum (B\_i \- \\bar{B})^2}}$$  
When a shadow falls across a surface in one view but not another (due to occlusion or light movement), the histogram of the pixel patch changes dramatically. The NCC score drops below the acceptance threshold, causing the algorithm to discard valid geometry. This manifests as holes in the mesh or high-frequency noise on surface areas that should be smooth.8

### **2.3 The "Baked-In" Lighting Equation**

The ultimate goal of PBR texturing is to derive the **Albedo** ($\\rho$), which represents the inherent color of the material. However, a camera captures **Radiance** ($L$), which is the product of Albedo and **Irradiance** ($E$):

$$L(x, \\omega\_o) \\approx \\rho(x) \\cdot E(x, \\omega\_i)$$  
Where $E$ is the sum of direct illumination (blocked by shadows) and indirect bounce light. In a shadowed region, $E$ drops significantly and shifts in color temperature (usually becoming bluer due to skylight fill). Photogrammetry software projects this raw Radiance $L$ onto the mesh. To restore the Albedo, the workflow must mathematically invert this equation:

$$\\rho(x) \\approx \\frac{L(x)}{E(x)}$$  
The validation of any de-lighting workflow depends on how accurately it can estimate the Irradiance term $E(x)$ to perform this division.11

## ---

**3\. Acquisition-Phase Workflows: Physical Shadow Suppression**

The most effective "masking" is to prevent the shadow from being captured in the first place. While digital removal has advanced, physical suppression remains the gold standard for high-fidelity preservation and material scanning.

### **3.1 Cross-Polarization Photogrammetry**

Cross-polarization is widely cited in the research as the critical technique for capturing shadow-free, glare-free albedo textures.12 It leverages the wave nature of light to filter out specular reflections, which indirectly aids in shadow management by compressing the dynamic range of the scene.

#### **3.1.1 Physics of the Workflow**

Light emitted from a flash is unpolarized (vibrating in all planes). By placing a **Linear Polarizing Film** over the light source, the light is restricted to a single vibration plane (e.g., vertical).

* **Specular Reflection:** When this polarized light bounces off a non-metallic surface (dielectric) directly into the lens, it retains its polarization. This creates the "shine" or "hotspot" on the object.  
* **Diffuse Reflection:** Some light penetrates the surface, bounces around the microstructure (subsurface scattering), and re-emerges. This process "scrambles" the polarization, making the exiting light unpolarized (vibrating in all planes).12

By placing a **Circular Polarizer (CPL)** on the camera lens and rotating it 90 degrees relative to the light source filter (orthogonal alignment), the camera blocks the vertically polarized specular return. It only admits the horizontally component of the depolarized diffuse reflection.

#### **3.1.2 Validation for Shadow Removal**

While cross-polarization is famous for removing glare, its role in shadow removal is equally vital.

1. **Dynamic Range Compression:** By eliminating bright specular highlights, the photographer can expose the image for the shadows without clipping the highlights. This allows for significant "shadow lifting" in RAW processing (e.g., Lightroom) effectively neutralizing ambient shadows before they enter the photogrammetry pipeline.14  
2. **Flatness for Delighting:** Automated delighting algorithms (discussed in Section 6\) often mistake specular highlights for light sources or white paint. Removing specularity ensures that the software correctly identifies brightness gradients as shading information, making the calculated AO maps significantly more accurate.13

**Hardware Requirements:**

* **Light Source:** Ring flashes (e.g., Godox AR400) are preferred because they align the light axis with the lens axis, minimizing cast shadow length.13  
* **Filtration:** High-transmission polarizing film for the lights and high-extinction CPLs for the lens.  
* **Limitations:** This method causes a light loss of 1.5 to 2 stops, requiring powerful strobes. It also fails on metallic objects, which preserve polarization (appearing black in cross-polarized views).

### **3.2 The Epic Games "Grey Ball / Chrome Ball" Workflow**

For environments where lighting cannot be controlled (e.g., outdoor rock scanning), the workflow shifts from *suppression* to *capture and subtraction*. This method, popularized by Epic Games for the *Kite Demo*, remains the industry reference for creating PBR-ready assets from natural environments.6

#### **3.2.1 The Reference Rig**

The workflow mandates carrying a reference rig containing:

* **50% Chrome Sphere:** To capture the direction and sharpness of the primary light source (sun) and the environmental reflection.  
* **18% Grey Sphere:** To measure the precise intensity of the mid-tones and the contrast ratio between lit and shadowed surfaces.  
* **Macbeth Color Checker:** To ensure color constancy across different times of day.

#### **3.2.2 Capture Protocol**

For every scanned asset (e.g., a cliff face), a reference photo of the rig is taken in the *exact* lighting conditions. This is not used for the 3D reconstruction itself but for the **Texture De-Lighting** phase.

* **HDRI Generation:** In post-production, the reflection on the chrome ball is "unwrapped" (using software like Nuke or PTGui) to create a spherical HDRI (High Dynamic Range Image).  
* **Validation:** This HDRI serves as the "Ground Truth" for the lighting environment. It allows the artist to re-light the scanned mesh in a 3D software (Blender/Maya) with the *exact* light that was present on set. The resulting render is used to mathematically subtract the light from the texture.17

### **3.3 The "Wandering Light" Method**

For static objects in a dark studio, the "Wandering Light" or "Light Painting" technique effectively removes shadows by averaging them out.

* **Technique:** The camera is set to a long exposure (e.g., 2-5 seconds). During the exposure, the operator walks around the object with a handheld light source, constantly moving it.  
* **Result:** Because the light source moves, shadows are "softened" to the point of invisibility. The light wraps around the object from multiple angles during a single frame, simulating a perfect ambient dome light.  
* **Validation:** This produces incredibly flat, shadow-free textures ideal for photogrammetry, but requires a perfectly dark room and a static subject.18

## ---

**4\. 2D Pre-Processing Workflows: Automated Detection and Masking**

When physical suppression is impossible—such as in drone mapping or archival footage processing—the shadows are captured in the images. The first digital line of defense is to identify and mask these shadows in the 2D domain *before* 3D reconstruction begins.

### **4.1 AI-Driven Shadow Detection (2025 State of the Art)**

The *New Trends in Image Restoration and Enhancement (NTIRE) 2025* challenge has catalyzed a leap in Deep Learning (DL) models specifically for shadow detection.2 These models offer a powerful pre-processing step for photogrammetry.

#### **4.1.1 Regional Attention Shadow Removal Model (RASM)**

The winner of the NTIRE 2025 challenge, RASM, represents a shift from global attention mechanisms to regional ones.19

* **Architecture:** Unlike previous models (e.g., ShadowFormer) that process the entire image context—often losing high-frequency details—RASM focuses on the correlation between a shadowed region and its immediate non-shadowed neighborhood. It uses a "divide-and-conquer" strategy to predict what the pixels *under* the shadow should look like based on the adjacent lit pixels.  
* **Workflow Integration:**  
  1. **Inference:** Users run the raw image dataset through the RASM Python inference script.  
  2. **Output:** The model generates two outputs: a **Shadow-Free Image** (reconstructed albedo) and a **Binary Shadow Mask**.  
  3. **Validation:** The Binary Shadow Mask is the critical asset for photogrammetry. It is imported into the photogrammetry software to mask out shadow pixels during alignment. The Shadow-Free Image can be used as a secondary input for texture generation, ensuring that the final map has no baked shadows.19

#### **4.1.2 ReHiT: Mask-Free Removal**

For datasets where generating masks is computationally expensive or storage-intensive, the **ReHiT (Recurrent Hierarchical Transformer)** model offers a "mask-free" workflow.20

* **Mechanism:** ReHiT is trained to output a delighting image directly from the input without an intermediate mask stage. It uses a recurrent structure to progressively refine the shadow removal, handling complex self-shadowing (form shadows) better than single-pass GANs.  
* **Application:** This is particularly useful for pre-processing drone footage of urban areas where shadows are complex and fragmented. By "flattening" the images with ReHiT before feeding them into RealityCapture, users report significantly better feature matching in alleyways and between buildings.20

### **4.2 Script-Based Automation (Agisoft Metashape)**

For controlled scenarios like turntables, Deep Learning is often unnecessary. A deterministic approach using Python scripting within Agisoft Metashape provides a perfect, mathematically generated mask.21

#### **4.2.1 Difference Masking Workflow**

This workflow assumes the camera is locked on a tripod and only the object moves (or is swapped).

1. **Clean Plate Capture:** Capture one image of the empty background/turntable.  
2. **Python Implementation:** The script iterates through every image in the chunk, compares it to the Clean Plate, and generates a mask based on the pixel difference.

Python

\# Conceptual logic for Agisoft Difference Masking  
import Metashape  
chunk \= Metashape.app.document.chunk  
background\_image \= Metashape.Image()  
background\_image.load("path/to/clean\_plate.jpg")

for camera in chunk.cameras:  
    current\_image \= camera.photo.image()  
    \# Calculate difference  
    diff \= current\_image.difference(background\_image)  
    \# Thresholding to create binary mask  
    mask \= diff.threshold(10)   
    \# Apply mask to camera  
    camera.mask \= mask

* **Validation:** This method is superior to AI for turntables because it is pixel-perfect. It masks out not just the background, but also the *static shadows* cast by the object onto the turntable, which are the primary cause of alignment failure in rotation scanning.10

### **4.3 Satellite and Aerial: The S-EO Dataset Impact**

For satellite photogrammetry, the 2025 release of the **S-EO (Shadow-Earth Observation)** dataset has standardized shadow detection.23

* **Data:** It provides ground-truth shadow masks derived from LiDAR DSMs and sun positioning for over 20,000 satellite tiles.  
* **Workflow:** Operators training models on this dataset can deploy specific "Satellite Shadow Detectors" that account for cloud shadows and building cast shadows. These masks are then used to drive "Shadow-Aware" reconstruction algorithms like EO-NeRF, which treat shadowed regions as transient data to be ignored during density optimization.23

## ---

**5\. 3D Reconstruction Phase: Strategic Masking Workflows**

Once masks are generated (via AI or scripts), they must be integrated into the 3D reconstruction software. The strategy for application is nuanced: masking for *alignment* is different from masking for *texturing*.

### **5.1 Agisoft Metashape: The Granular Control**

Metashape offers the industry's most specific controls for how masks influence the SfM pipeline. The critical distinction lies in the **"Apply Mask to..."** settings in the **Align Photos** dialog.25

#### **5.1.1 Workflow: Masking for Tie Points (Geometric Fidelity)**

This is the most misunderstood setting in Metashape but is vital for shadow management.

* **Option A: Apply Mask to Key Points:** This prevents the feature detector (SIFT) from even looking at the masked area.  
  * *Risk:* If the mask is slightly "loose" (covering valid object edge), you lose critical edge features. This can result in poor alignment or camera drift.  
* **Option B: Apply Mask to Tie Points:** This allows feature detection everywhere, but *discards* matches (tie points) that fall within the mask *after* detection.  
  * *Validation:* This is the recommended workflow for shadow masking. It ensures that the algorithm finds features near the shadow boundary (which are often high contrast and good) but prevents it from creating "false" tie points inside the shadow itself. This preserves the geometric integrity of the sparse cloud while filtering out the noise caused by moving shadows.26

#### **5.1.2 Workflow: The "Mask from Model" Recursion**

A powerful technique validated by power users is the recursive masking loop.22

1. **Draft Alignment:** Align photos with "Low" accuracy and no masks.  
2. **Draft Mesh:** Build a rough mesh. The background and shadows will appear as "junk" geometry.  
3. **Manual Cleaning:** Delete the shadow geometry and background noise in the 3D viewport.  
4. **Generate Masks:** Right-click on the clean model \-\> **Import Masks \-\> From Model**.  
   * *Mechanism:* Metashape projects the "clean" 3D shape back onto the 2D images. Everything that is *not* the model (i.e., the shadows and background you deleted) becomes a black mask.  
5. **Re-Alignment:** Reset the alignment. Run "Align Photos" again on "High" accuracy, this time checking "Apply mask to tie points".  
   * *Result:* The software now ignores all the shadow areas perfectly, resulting in a pristine alignment and dense cloud.22

### **5.2 RealityCapture / RealityScan: Automated Efficiency**

RealityCapture (RC) prioritizes speed and automation. With the release of **RealityScan 2.0** (mid-2025), it introduced native AI masking.5

#### **5.2.1 Native AI Masking Workflow**

* **Tool:** Alignment \-\> Detect Masks.  
* **Capabilities:** This tool uses a built-in segmentation network (likely a distilled version of SAM \- Segment Anything Model) to identify the "subject" of the scan.  
* **Validation:**  
  * *Turntables:* It effectively masks out the background and the *floor shadows*, solving the stationary shadow problem without external scripts.  
  * *Aerial:* It is less effective for continuous terrain (drone maps) where there is no clear "foreground object." In these cases, RC relies on importing external masks via the CLI.  
* **Cli Workflow:** For large datasets (e.g., city mapping), the RC Command Line Interface allows injecting masks generated by external AI (like RASM) using the \-importMasks command. This enables "Shadow-Free" city reconstruction at scale.28

### **5.3 Meshroom / AliceVision: The Node-Based Approach**

Meshroom offers a modular approach via its node graph.

* **MeshMasking Node:** This node allows applying masks to the mesh *after* generation but before texturing.  
* **Semantic Segmentation:** New updates in 2025 allow integrating machine learning nodes directly into the graph. A user can insert a "Segmentation" node before the "StructureFromMotion" node, automatically generating masks for specific classes (e.g., "sky" or "shadow") and feeding them into the feature extraction process.29

## ---

**6\. Post-Processing Workflows: Texture De-Lighting**

Even with perfect masking during alignment, the shadows captured in the photographs will be projected onto the final texture. Removing them from the texture map—"De-Lighting"—is the final and most complex step.

### **6.1 The "Divide and Conquer" Workflow (Blender/Nuke)**

This is the scientifically rigorous method for de-lighting. It treats the texture as a mathematical product of Albedo and Light, and solves for Albedo by dividing the Texture by a simulated Light Map.

#### **6.1.1 Step 1: Baking Geometric Maps**

Import the high-resolution mesh into Blender. Using the Cycles render engine, bake the following maps to the UV space 11:

1. **Ambient Occlusion (AO):** Captures "form shadows" (crevices).  
2. **Bent Normal:** Captures the direction of greatest light accessibility.  
3. **Curvature/Thickness:** Useful for masking edges.

#### **6.1.2 Step 2: Simulating the Light Map**

The goal is to create a black-and-white image that matches the lighting pattern in the original scan *exactly*.

* **Directional Matching:** Use the **Sun Position** addon 32 or manually position a directional light in Blender to match the shadows seen in the texture.  
* **Environment Matching:** If you captured a Chrome Ball HDRI (from Section 3.2), use it to light the object.  
* **Baking:** Bake this illumination to a texture. Let's call this the Irradiance\_Map.

#### **6.1.3 Step 3: The Division (Compositing)**

In Blender's Shader Editor or Compositor:

$$Albedo \= \\frac{Scan\\\_Texture}{Irradiance\\\_Map}$$

* **Mechanism:** If a pixel in the scan is dark gray (0.2) because it is in shadow, and the corresponding pixel in the Irradiance Map is also dark gray (0.2), the division yields $0.2 / 0.2 \= 1.0$ (White/Pure Color). The shadow is effectively canceled out.33  
* **Correction:** This process often leaves the texture looking "washed out" or desaturated. A "High Pass" filter (overlay blend mode) or Curve adjustment is required to restore local contrast and color saturation.

### **6.2 Agisoft Texture De-Lighter**

For users avoiding complex node graphs, Agisoft's free standalone tool offers a simplified workflow.4

#### **6.2.1 Two-Pronged Algorithm**

1. **Remove Shading (AO Correction):** This is automatic. It reverses the calculated ambient occlusion. It works well for uniform, cloudy-day scans.  
2. **Remove Cast Shadows (User-Guided):**  
   * **Input:** The user paints "Lit" (yellow) and "Shadowed" (blue) strokes on the model.  
   * **Processing:** The algorithm calculates the color histogram of the lit vs. shadowed areas and computes a transfer function (curve) to match them. It effectively "color corrects" the shadow areas to match the lit areas.  
   * **Validation:** While faster, this method is less accurate than the Blender Division method for complex materials. It relies on color statistics rather than light physics, which can lead to "muddy" textures if the shadow has a strong color cast (e.g., blue skylight).34

### **6.3 Adobe Substance 3D Sampler**

Adobe leverages AI for de-lighting in the **Image to Material** filter (formerly Alchemist).36

* **Workflow:** Import the texture \-\> Apply "Delight" filter.  
* **Mechanism:** A CNN trained on pairs of lit/unlit materials detects low-frequency lighting gradients and removes them.  
* **Best Use:** This is excellent for removing large, soft gradients across a surface (e.g., uneven flash coverage). It is less effective at removing hard, complex cast shadows from geometry, where the geometric-aware methods (Blender/Agisoft) excel.

## ---

**7\. Emerging Frontiers: Neural Rendering and Implicit Shadows**

The field is rapidly moving towards methods that do not "remove" shadows but "model" them. In 2025/2026, Neural Radiance Fields (NeRF) and Gaussian Splatting have introduced workflows that fundamentally differ from the classic photogrammetry pipeline.

### **7.1 Shadow-Aware 3D Gaussian Splatting (ShadowGS)**

**ShadowGS** 38 adapts 3D Gaussian Splatting (3DGS) to handle variable illumination.

* **Concept:** Standard 3DGS assumes the scene color is view-dependent but time-independent. ShadowGS introduces a "Shadow Opacity" parameter for each Gaussian splat.  
* **Decoupling:** During training, the network learns to separate the **Albedo** (constant color) from the **Illumination** (transient shadow). It uses the sun direction (derived from metadata or estimated) to predict which splats are occluded.  
* **Result:** The output is a 3D model where the user can *change the sun position* in real-time. The "Delighting" is implicit; you simply turn off the shadow rendering layer. This eliminates the need for manual masking entirely, as the network learns that shadows are "transient" features.38

### **7.2 Neural De-lighting with S-EO Data**

For satellite/aerial data, pipelines like **EO-NeRF** utilize the **S-EO dataset** masks to supervise training.23

* **Workflow:**  
  1. Use S-EO masks to identify shadow regions in input images.  
  2. Train the NeRF with a "Transient Loss" function that down-weights pixels inside the shadow masks.  
  3. **Validation:** This produces "Shadow-Free" NeRFs of cities, where the geometry is sharp (because shadows didn't confuse the density estimation) and the texture is flat (because shadow pixels were ignored).

## ---

**8\. Workflow Matrix and Recommendations**

Based on the research, we categorize validated workflows by project type.

### **Table 1: Shadow Masking Workflow Matrix**

| Application | Acquisition | Pre-Processing | Reconstruction | Post-Processing |
| :---- | :---- | :---- | :---- | :---- |
| **Cultural Heritage / Product** | **Cross-Polarization** \+ Ring Flash | **Difference Masking** (Python Script) | Agisoft: Apply Mask to **Tie Points** | Agisoft De-Lighter (Remove Shading only) |
| **Game Assets (Rocks/Props)** | Grey/Chrome Ball Reference | None (or basic cleanup) | RealityCapture: **AI Masking** | **Blender Division Method** (Albedo \= Tex/Light) |
| **Aerial / Drone Mapping** | Nadir/Oblique Flight (Sunny) | **RASM / ReHiT** (AI Shadow Detection) | Agisoft: **Mask from Model** Recursion | Adobe Sampler (AI Delighting) |
| **City-Scale / Satellite** | Multi-date Satellite | **S-EO** based Shadow Detectors | **ShadowGS** (Neural Reconstruction) | N/A (Implicit De-lighting) |

### **8.1 Validation Summary**

* **For Geometry:** The most critical step is **Masking Tie Points** in Agisoft Metashape. This prevents the "phantom geometry" caused by moving shadows.  
* **For Texture:** The most accurate method is the **Blender Division Workflow**, as it respects the physics of light transport.  
* **For Speed:** **RealityCapture AI Masking** combined with **Adobe Sampler** offers the fastest "good enough" result for non-critical assets.

## **9\. Conclusion**

The management of shadows in photogrammetry has matured from a manual "cleanup" task to a rigorous engineering discipline. In 2026, the practitioner has access to a tiered arsenal of tools. Physical suppression via cross-polarization remains the baseline for quality. Where that fails, AI detection models like RASM and ReHiT provide robust 2D masking. In reconstruction, the strategic masking of tie points prevents geometric corruption. Finally, mathematical de-lighting in post-production restores the albedo. The emerging era of Shadow-Aware Gaussian Splatting promises to eventually automate this entire chain, but for the present, the "Hybrid Workflow"—combining physical rigour, algorithmic masking, and mathematical compositing—remains the validated path to ground-truth 3D reconstruction.

#### **Works cited**

1. Remove Shadow: A Modern Editing Guide for 2025 \- BeatColor CO.,LTD, accessed February 7, 2026, [https://beatcolor.com/blog/photo-video-editting/remove-shadow-a-modern-editing-guide-for-2025/](https://beatcolor.com/blog/photo-video-editting/remove-shadow-a-modern-editing-guide-for-2025/)  
2. \[2506.15524\] NTIRE 2025 Image Shadow Removal Challenge Report \- arXiv, accessed February 7, 2026, [https://arxiv.org/abs/2506.15524](https://arxiv.org/abs/2506.15524)  
3. \[Literature Review\] NTIRE 2025 Image Shadow Removal Challenge Report \- Moonlight, accessed February 7, 2026, [https://www.themoonlight.io/en/review/ntire-2025-image-shadow-removal-challenge-report](https://www.themoonlight.io/en/review/ntire-2025-image-shadow-removal-challenge-report)  
4. Agisoft Texture De-Lighter: General Workflow and Best Practices ..., accessed February 7, 2026, [https://www.agisoftmetashape.com/agisoft-texture-de-lighter-general-workflow-and-best-practices/](https://www.agisoftmetashape.com/agisoft-texture-de-lighter-general-workflow-and-best-practices/)  
5. RealityScan 2.0: New release brings powerful new features to a rebranded RealityCapture, accessed February 7, 2026, [https://www.realityscan.com/en-US/news/realityscan-20-new-release-brings-powerful-new-features-to-a-rebranded-realitycapture](https://www.realityscan.com/en-US/news/realityscan-20-new-release-brings-powerful-new-features-to-a-rebranded-realitycapture)  
6. The Tech & Beauty behind Epic's UE4 Open World Demo | Moritz ..., accessed February 7, 2026, [https://moritzweller.wordpress.com/2015/03/13/the-tech-behind-epics-open-world-demo-in-unreal-engine-4/](https://moritzweller.wordpress.com/2015/03/13/the-tech-behind-epics-open-world-demo-in-unreal-engine-4/)  
7. Metric Error Assessment Regarding Geometric 3D Reconstruction of Transparent Surfaces via SfM Enhanced by 2D and 3D Gaussian Splatting \- PMC, accessed February 7, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12300396/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12300396/)  
8. The 3D Creation Playbook: How AI, Scanning & Photogrammetry Collide \- FRANKI T, accessed February 7, 2026, [https://www.francescatabor.com/articles/2025/9/9/the-3d-creation-playbook-how-ai-scanning-amp-photogrammetry-collide](https://www.francescatabor.com/articles/2025/9/9/the-3d-creation-playbook-how-ai-scanning-amp-photogrammetry-collide)  
9. Masking and its effects for point cloud and mesh generation \- Agisoft Metashape, accessed February 7, 2026, [https://www.agisoft.com/forum/index.php?topic=3024.0](https://www.agisoft.com/forum/index.php?topic=3024.0)  
10. r/photogrammetry \- Agisoft Metashape \- how to get a good alignment? \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/photogrammetry/comments/11v2jjr/agisoft\_metashape\_how\_to\_get\_a\_good\_alignment/](https://www.reddit.com/r/photogrammetry/comments/11v2jjr/agisoft_metashape_how_to_get_a_good_alignment/)  
11. The Workflows of Creating Game Ready Textures and Assets using Photogrammetry, accessed February 7, 2026, [https://www.gamedeveloper.com/art/the-workflows-of-creating-game-ready-textures-and-assets-using-photogrammetry](https://www.gamedeveloper.com/art/the-workflows-of-creating-game-ready-textures-and-assets-using-photogrammetry)  
12. How to Use Cross Polarization for Texture Photography and Material ..., accessed February 7, 2026, [https://www.lotpixel.com/blog/how-to-use-cross-polarization-for-texture-photography-and-material-scanning](https://www.lotpixel.com/blog/how-to-use-cross-polarization-for-texture-photography-and-material-scanning)  
13. The Key to Cleaner 3D Scans: Cross-Polarization \- YouTube, accessed February 7, 2026, [https://www.youtube.com/watch?v=GJ2gtQ0WxTU](https://www.youtube.com/watch?v=GJ2gtQ0WxTU)  
14. When shooting photos for photogrammetry should I use or avoid polarized camera lenses? What effect do they have positive or negative? \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/photogrammetry/comments/mjt3k4/when\_shooting\_photos\_for\_photogrammetry\_should\_i/](https://www.reddit.com/r/photogrammetry/comments/mjt3k4/when_shooting_photos_for_photogrammetry_should_i/)  
15. Create Textures for revopoint scans by using photogrammetry, accessed February 7, 2026, [https://forum.revopoint3d.com/t/create-textures-for-revopoint-scans-by-using-photogrammetry/25598](https://forum.revopoint3d.com/t/create-textures-for-revopoint-scans-by-using-photogrammetry/25598)  
16. Imperfection for Perfection: How To Create Photogrammetric Game-Ready Assets in UE4, accessed February 7, 2026, [https://www.unrealengine.com/en-US/blog/imperfection-for-perfection](https://www.unrealengine.com/en-US/blog/imperfection-for-perfection)  
17. Everything You Need to Know about Photogrammetryï¿½ I hope \- Game Developer, accessed February 7, 2026, [https://www.gamedeveloper.com/business/everything-you-need-to-know-about-photogrammetry-i-hope](https://www.gamedeveloper.com/business/everything-you-need-to-know-about-photogrammetry-i-hope)  
18. Large-scale Photogrammetry for 3D Model Creation – No LiDAR \- Mosaic 51, accessed February 7, 2026, [https://www.mosaic51.com/technology/large-scale-photogrammetry-create-3d-models-reconstructions/](https://www.mosaic51.com/technology/large-scale-photogrammetry-create-3d-models-reconstructions/)  
19. CalcuLuUus/RASM: Impl of Regional Attention For Shadow ... \- GitHub, accessed February 7, 2026, [https://github.com/CalcuLuUus/RASM](https://github.com/CalcuLuUus/RASM)  
20. Final ranking of NTIRE 2025 Shadow Removal Challenge. While maintaining... \- ResearchGate, accessed February 7, 2026, [https://www.researchgate.net/figure/Final-ranking-of-NTIRE-2025-Shadow-Removal-Challenge-While-maintaining-a-certain-level\_tbl1\_390991235](https://www.researchgate.net/figure/Final-ranking-of-NTIRE-2025-Shadow-Removal-Challenge-While-maintaining-a-certain-level_tbl1_390991235)  
21. Automatic background masking using custom script \- Helpdesk Portal, accessed February 7, 2026, [https://agisoft.freshdesk.com/support/solutions/articles/31000161958-automatic-background-masking-using-custom-script](https://agisoft.freshdesk.com/support/solutions/articles/31000161958-automatic-background-masking-using-custom-script)  
22. How to Automatically Mask Backgrounds in Agisoft Metashape, accessed February 7, 2026, [https://www.agisoftmetashape.com/how-to-automatically-mask-backgrounds-in-agisoft-metashape/](https://www.agisoftmetashape.com/how-to-automatically-mask-backgrounds-in-agisoft-metashape/)  
23. S-EO: A Large-Scale Dataset for Geometry-Aware Shadow Detection in Remote Sensing Applications \- arXiv, accessed February 7, 2026, [https://arxiv.org/html/2504.06920v1](https://arxiv.org/html/2504.06920v1)  
24. \[Literature Review\] S-EO: A Large-Scale Dataset for Geometry-Aware Shadow Detection in Remote Sensing Applications \- Moonlight, accessed February 7, 2026, [https://www.themoonlight.io/en/review/s-eo-a-large-scale-dataset-for-geometry-aware-shadow-detection-in-remote-sensing-applications](https://www.themoonlight.io/en/review/s-eo-a-large-scale-dataset-for-geometry-aware-shadow-detection-in-remote-sensing-applications)  
25. Agisoft Metashape User Manual \- Standard Edition, Version 2.0, accessed February 7, 2026, [https://www.agisoft.com/pdf/metashape\_2\_0\_en.pdf](https://www.agisoft.com/pdf/metashape_2_0_en.pdf)  
26. Agisoft Metashape User Manual \- Professional Edition, Version 1.5, accessed February 7, 2026, [https://www.agisoft.com/pdf/metashape-pro\_1\_5\_en.pdf](https://www.agisoft.com/pdf/metashape-pro_1_5_en.pdf)  
27. Mask Images \- RealityScan Help, accessed February 7, 2026, [https://rshelp.capturingreality.com/en-US/tools/mask.htm](https://rshelp.capturingreality.com/en-US/tools/mask.htm)  
28. RealityCapture Automated CLI pipeline for Large Datasets | Epic Developer Community, accessed February 7, 2026, [https://dev.epicgames.com/community/learning/tutorials/9nO5/unreal-engine-realityscan-realitycapture-automated-cli-pipeline-for-large-datasets](https://dev.epicgames.com/community/learning/tutorials/9nO5/unreal-engine-realityscan-realitycapture-automated-cli-pipeline-for-large-datasets)  
29. Meshroom 2025.1.0 Release \- Google Groups, accessed February 7, 2026, [https://groups.google.com/g/alicevision/c/a9xGujkF-6c](https://groups.google.com/g/alicevision/c/a9xGujkF-6c)  
30. meshroom.nodes.aliceVision.MeshMasking.MeshMasking \- Meshroom documentation, accessed February 7, 2026, [https://meshroom.readthedocs.io/en/stable/generated/meshroom.nodes.aliceVision.MeshMasking.MeshMasking.html](https://meshroom.readthedocs.io/en/stable/generated/meshroom.nodes.aliceVision.MeshMasking.MeshMasking.html)  
31. Photogrammetry Blender Workflow With Stock Footage? \- YouTube, accessed February 7, 2026, [https://www.youtube.com/watch?v=52Uqq1T2Cuc](https://www.youtube.com/watch?v=52Uqq1T2Cuc)  
32. Sun Position \- Blender 4.1 Manual, accessed February 7, 2026, [https://docs.blender.org/manual/en/4.1/addons/lighting/sun\_position.html](https://docs.blender.org/manual/en/4.1/addons/lighting/sun_position.html)  
33. Removing AO from photogrammetry \- Polycount, accessed February 7, 2026, [https://polycount.com/discussion/161470/removing-ao-from-photogrammetry](https://polycount.com/discussion/161470/removing-ao-from-photogrammetry)  
34. Photogrammetry Breakdown: De-lighting \- Inu Games, accessed February 7, 2026, [https://inu-games.com/2022/03/23/photogrammetry-breakdown-de-lighting/](https://inu-games.com/2022/03/23/photogrammetry-breakdown-de-lighting/)  
35. Difference between StandAlone Delighter and "Remove Lighting" from MetaShape?, accessed February 7, 2026, [https://www.agisoft.com/forum/index.php?topic=12500.0](https://www.agisoft.com/forum/index.php?topic=12500.0)  
36. Photogrammetry software for 3D capture \- Adobe Substance 3D, accessed February 7, 2026, [https://www.adobe.com/products/substance3d/apps/sampler.html](https://www.adobe.com/products/substance3d/apps/sampler.html)  
37. Any new AI Delighting Tools? : r/photogrammetry \- Reddit, accessed February 7, 2026, [https://www.reddit.com/r/photogrammetry/comments/1egw989/any\_new\_ai\_delighting\_tools/](https://www.reddit.com/r/photogrammetry/comments/1egw989/any_new_ai_delighting_tools/)  
38. ShadowGS: Shadow-Aware 3D Gaussian Splatting for Satellite Imagery \- arXiv, accessed February 7, 2026, [https://arxiv.org/html/2601.00939v1](https://arxiv.org/html/2601.00939v1)  
39. ShadowGS: Shadow-Aware 3D Gaussian Splatting for Satellite Imagery \- Semantic Scholar, accessed February 7, 2026, [https://www.semanticscholar.org/paper/ShadowGS%3A-Shadow-Aware-3D-Gaussian-Splatting-for-Luo-Pan/eb676aa602313c0710a3f8ab1d0e0f8f915416a0](https://www.semanticscholar.org/paper/ShadowGS%3A-Shadow-Aware-3D-Gaussian-Splatting-for-Luo-Pan/eb676aa602313c0710a3f8ab1d0e0f8f915416a0)