# Automated masking for photogrammetry: edge refinement, temporal consistency, and model ensembles

**The most impactful upgrades to your pipeline are a YOLO→SAM2 cascade with majority-vote fusion, DEVA-style temporal propagation between keyframe detections, and CascadePSP or ViTMatte-based edge refinement — all feasible on your 24GB RTX 3090 Ti.** These three areas interact: better per-frame masks from ensembles reduce the error that temporal propagation must correct, and edge refinement applied to the final fused mask produces cleaner binary boundaries for COLMAP and Metashape. The research landscape has shifted dramatically since 2023, with MatAnyone (CVPR 2025) bridging the gap between binary segmentation and alpha matting for video, SAM2's video mode enabling mask propagation from keyframes, and Grounded-SAM-2 providing a production-ready detect-then-segment cascade. For equirectangular content, no methods yet operate natively on the projection — your cubemap decomposition approach remains the standard, though adding overlapping face FOVs (100–110° instead of 90°) significantly improves cross-boundary handling.

---

## Area 1: Edge refinement transforms coarse binary masks into precise boundaries

Your binary masks from YOLO and SAM cut through hair, clothing edges, and equipment boundaries with hard pixel steps. The refinement pipeline is straightforward: take your existing mask, use it to generate a trimap or guide a matting network, produce a soft alpha matte, then re-binarize at a threshold that captures fine boundary detail the original mask missed. The alpha matte's value isn't the softness itself — it's that the matting network uses image evidence to find the *true* object boundary, producing a more accurate binary mask after thresholding.

### Trimap-based matting: ViTMatte and DiffMatte lead the field

The best trimap-based matting models as of early 2026 are **DiffMatte** (ECCV 2024) and **ViTMatte** (2024), with DiffMatte achieving ~5–8% lower SAD and ~15% lower MSE than ViTMatte on Composition-1k through iterative diffusion refinement. ViTMatte uses a plain Vision Transformer backbone with a lightweight Detail Capture Module; DiffMatte adds a decoupled diffusion decoder that progressively recovers fine structures like hair strands. Both accept RGB + trimap as input and have pretrained weights on GitHub (`github.com/hustvl/ViTMatte`, `github.com/YihanHu-2022/DiffMatte`).

**Automatic trimap generation from your binary masks** is well-established: erode and dilate the mask with an elliptical kernel to create definite foreground, definite background, and an "unknown" band:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
eroded = cv2.erode(mask, kernel, iterations=2)
dilated = cv2.dilate(mask, kernel, iterations=2)
trimap = np.full(mask.shape, 128, dtype=np.uint8)
trimap[eroded >= 254] = 255   # definite foreground
trimap[dilated <= 1] = 0       # definite background
```

Standard parameters use kernel sizes of **10–20 pixels** (elliptical) with 1–3 iterations, producing an unknown band of 10–30 pixels total. Some approaches scale kernel size adaptively: `ω = η × √(mask_area)` with η ≈ 12. The ViTMatte paper trains with random kernel sizes in [1, 30] for robustness.

Two newer models deserve attention: **MEMatte** (2024) reduces memory by ~88% and latency by ~50% versus ViTMatte on the RTX 3090, making it the most practical choice for batch processing (`github.com/linyiheng123/MEMatte`). **AEMatter** claims 25% SAD reduction over MatteFormer (`github.com/QLYoo/AEMatter`). Older models like GCA Matting (SAD 30.3 on Composition-1k) and FBA Matting (~SAD 26) remain functional baselines but have been convincingly surpassed.

### Trimap-free matting and the most relevant discovery: MatAnyone

**MatAnyone** (CVPR 2025, `github.com/pq-yang/MatAnyone`) is the single most relevant model for your pipeline. It takes exactly what you produce — a **binary segmentation mask on the first frame** — and generates temporally consistent alpha mattes across an entire video sequence. It uses consistent memory propagation with region-adaptive fusion and was specifically trained with erosion/dilation augmentation (kernel 1–5) to be robust to inaccurate input masks. This collapses the entire edge-refinement and temporal-consistency problems into one model.

**BiRefNet** (`github.com/ZhengPeng7/BiRefNet`) is another strong option. While primarily a dichotomous image segmentation model, its `BiRefNet-matting` and `BiRefNet-HR-matting` variants produce trimap-free alpha mattes from RGB images alone at **17 FPS at 1024×1024 using ~3.45GB VRAM** in FP16. It has an extensive model zoo on HuggingFace with 18+ variants including portrait-specific and high-resolution (2048×2048) models.

Other trimap-free models have more limited applicability to your pipeline. **MODNet** and **Robust Video Matting (RVM)** both work from images/video alone without accepting mask guidance — they're purely automatic portrait matting models. RVM is remarkably fast (4K@76FPS) but human-only and cannot leverage your existing YOLO/SAM detections. **Background Matting V2** requires a pre-captured clean background plate, which isn't available in photogrammetry fieldwork.

The key distinction for your pipeline: models that **accept a coarse mask as input** (MatAnyone, MaGGIe, Matte Anything, trimap-based methods via auto-trimap generation) are far more useful than fully automatic matting models, because you already have good detection results.

### CascadePSP: the fastest path to better binary masks

For immediate, practical improvement with minimal integration effort, **CascadePSP** (`github.com/hkchengrex/CascadePSP`) is the strongest recommendation. It's a class-agnostic segmentation refinement module available as a pip package (`pip install segmentation-refinement`) that takes any coarse binary mask plus the RGB image and outputs a refined binary mask directly — no alpha-to-binary conversion needed. It improved DeepLabV3+ boundary accuracy (mBA) by **10.4%** on PASCAL VOC and **14.3%** on the BIG dataset. Processing takes ~0.5 seconds in fast mode or ~5 seconds in full mode per image.

```python
import segmentation_refinement as refine
refiner = refine.Refiner(device='cuda:0')
refined_mask = refiner.refine(image, mask, fast=False, L=900)
```

**Guided image filtering** (`cv2.ximgproc.guidedFilter`) is the lightweight alternative — it aligns mask boundaries with image edges in milliseconds without any neural network. Use the RGB image as guide, the binary mask (converted to float) as input, with `radius=40` and `eps=1e-4×255²`. However, research confirms guided filtering produces "obviously coarse" results compared to neural matting: it can shift boundaries to image edges but cannot hallucinate missing hair strands or recover fine detail. It's best as a fallback when neural methods are too slow.

### Converting alpha mattes to binary masks for photogrammetry

Since COLMAP and Metashape require binary masks, the conversion step matters. Simple thresholding at 0.5 is the standard approach, but **for photogrammetry you should threshold lower, at 0.3–0.4**, because unmasked person fragments cause worse reconstruction artifacts than slight over-masking. A more sophisticated approach uses hysteresis thresholding (two thresholds at 0.3 and 0.7 — definite foreground above 0.7, extend connected regions down to 0.3, similar to Canny edge logic). Follow thresholding with a small morphological cleanup (opening/closing with kernel 3–5) to remove isolated pixels.

The fundamental insight: **the matte's value is giving you a more accurate boundary location**, even though you discard the alpha values. The pipeline binary mask → auto-trimap → neural matting → alpha matte → threshold → improved binary mask consistently produces better boundaries than the original segmentation alone.

### Equirectangular compatibility

No matting methods work on equirectangular projections directly. All are trained on standard perspective images and break under severe polar distortion. Your cubemap decomposition approach is correct — process each face as a perspective image, run matting/refinement per face, then merge back. For the merge step with alpha mattes, use overlapping faces (slight FOV > 90°) and blend in overlap regions before the final binarization.

---

## Area 2: Temporal propagation turns independent frames into coherent sequences

Your current per-frame segmentation with basic sliding-window smoothing misses the core problem: segmentation models make different errors on different frames, causing mask boundaries to flicker even when the object moves smoothly. The solution is to propagate masks from high-confidence keyframes rather than re-detecting independently on every frame.

### The DEVA paradigm best matches your pipeline architecture

**DEVA** (ICCV 2023, `github.com/hkchengrex/Tracking-Anything-with-DEVA`) implements exactly the architecture your pipeline needs: it decouples task-specific image-level segmentation from class-agnostic temporal propagation. You run your existing detector (YOLO) periodically as keyframe detections, and DEVA propagates masks between keyframes using a temporal model trained once for all tasks. It works with any image segmentor — SAM, Grounded SAM, YOLO — and handles objects entering and leaving the scene naturally through periodic detection-propagation-merge cycles.

DEVA's author (Ho Kei Cheng) also created **Cutie** and **XMem**, forming an integrated ecosystem. **Cutie** (CVPR 2024 Highlight, `github.com/hkchengrex/Cutie`) is the current best-in-class VOS model, outperforming XMem by **+8.7 J&F on MOSE** while being **3× faster** than DeAOT. Its object-level memory reading with query-based transformers makes it particularly robust to distractors — critical for outdoor scenes with similar-looking people. Cutie maintains a ~15-frame memory for temporal correlations and supports FIFO memory for long sequences.

A practical caveat with DEVA: it can produce higher false-positive rates on very long videos when targets exit the scene, due to its training augmentation strategy. Mitigate this by using higher detection thresholds and reducing `max_missed_detection_count` for more aggressive object deletion.

### SAM2 video mode: powerful but constrained for long sequences

SAM2's video propagation uses a streaming memory architecture with an image encoder (Hiera), memory attention via transformer blocks, and a memory bank storing spatial features from recent frames (6 recent + prompted frames). It supports multiple objects simultaneously, each with independent tracking via unique object IDs. Initialize from YOLO detections by feeding bounding boxes as prompts through `add_new_points()`.

The critical limitation is **error accumulation**: SAM2 uses greedy mask selection (highest IoU per frame), and single-frame errors propagate and compound. The **SAM2Long** paper confirmed that the performance gap between SAM2 and SAM2Long widens over time, validating long-sequence degradation. On your **RTX 3090 Ti with 24GB VRAM**, stock SAM2 handles approximately **200 frames with 6–7 objects** before running out of memory. For your 1400-frame sequences, you must either:

- Enable `offload_state_to_cpu=True` (22% speed penalty, reduces VRAM to ~21GB for a 2:30 video)
- Use Det-SAM2's `release_old_frames()` to cap memory growth
- Process in overlapping windows of ~200 frames with re-initialization

**Re-prompt every 50–100 frames** or when confidence drops. SAM2.1 adds better distractor handling through `SAM2.1++`. The recommended integration pattern: run YOLO on the first frame and periodically, feed YOLO boxes as SAM2 prompts, let SAM2 propagate between prompt frames, re-inject YOLO detections for drift correction.

Several SAM2 variants address long-video limitations: **SAM2Long** (training-free memory tree exploring multiple segmentation pathways, +2.1–5.3 J&F on SA-V), **SAMURAI** (motion-aware memory scoring for fast-moving objects), and **SAM2Plus** (Kalman filter + dynamic quality thresholds). **EntitySAM** (CVPR 2025) extends SAM2 with automatic entity discovery, removing the need for explicit prompts.

### Multi-object tracking gives your existing YOLO pipeline immediate improvement

The fastest path to better temporal consistency is adding a tracker to your existing YOLO detections. **BoT-SORT** is recommended for your use case because it includes camera motion compensation via sparse optical flow — essential for photogrammetry where the camera moves continuously. Integration is trivial with Ultralytics:

```python
results = model.track(frame, persist=True, tracker="botsort.yaml")
```

**ByteTrack** (30 FPS, 80.3 MOTA on MOT17) is simpler and faster but lacks camera motion compensation. **OC-SORT** (700+ FPS on CPU) excels at occlusion-heavy scenarios with its observation-centric re-update mechanism. For your photogrammetry masking where you just need "any person masked" regardless of identity, all trackers work — new IDs assigned when objects re-enter are acceptable since you aren't tracking specific individuals.

The enhanced pipeline: YOLO-seg every frame → BoT-SORT assigns track IDs → smooth masks per-track-ID instead of globally → fill gaps with morphological interpolation. This requires **~4–5GB VRAM** and runs at **25+ FPS**.

### Optical flow for frame interpolation, not primary propagation

RAFT optical flow (`github.com/princeton-vl/RAFT`, also in torchvision) enables mask warping between frames at ~10 FPS for 1080p pairs using 4–8GB VRAM. The approach is simple: compute flow from frame N to N+1, warp the mask using grid sampling, threshold to binary. However, **error accumulation makes flow-based propagation impractical as the primary method for 1400 frames** — small warping errors compound into complete mask drift.

The practical application is **interpolation between keyframe detections**: detect every 3–5 frames with YOLO, warp masks for intermediate frames using forward/backward RAFT flow, merge the two warped masks (union for over-coverage). This fills 1–2 frame gaps cheaply and reliably.

### 360° equirectangular video VOS remains a frontier problem

This is the least mature area in your pipeline's requirements. **No VOS or tracking method handles equirectangular projections natively.** Standard VOS models trained on planar data suffer **10–18 point J&F drops** when applied directly to equirectangular input, as demonstrated by the **Leader360V** dataset (2025) — the first large-scale 360° video dataset with dense annotations for segmentation and tracking.

Your cubemap decomposition approach is the standard solution, but objects crossing face boundaries break tracking continuity. Three strategies mitigate this:

- **Overlapping face FOV**: Expand each face to 100–110° instead of 90°, creating overlap zones where objects appear on two faces simultaneously. Merge masks in overlap regions using confidence weighting.
- **Global detection pass**: Run YOLO on a downscaled equirectangular image as a "meta-detector" to catch objects near seams, then use those detections to inform per-face processing.
- **Spherical coordinate tracking**: Map per-face detections back to angular (spherical) coordinates for tracking association, avoiding the discontinuities of pixel-space matching across faces.

A specialized approach from **UPDT360** (2025) converts equirectangular images to gnomonic projections centered on tracked objects, runs standard 2D tracking, then transforms back. This is more complex but handles boundary crossings naturally. The **360Tracking** repo (`github.com/VitaAmbroz/360Tracking`) implements equirectangular rotation to keep tracked objects centered in the projection.

### Video instance segmentation is overkill for this use case

VIS methods like DVIS++, Mask2Former-VIS, and VITA are designed for closed-vocabulary instance tracking across fixed categories. They either process entire sequences in memory (offline mode — infeasible for 1400 frames) or operate in short clip windows (typically 5–36 frames). While DVIS++ achieves impressive results (57.1 AP on OVIS, only ~11GB VRAM), the detection-then-propagation approach from DEVA or the YOLO+tracker pattern is more practical and flexible for your "mask all people/equipment" scenario.

### Recommended tiered integration

- **Tier 1 (immediate, minimal code change)**: Add BoT-SORT tracking to existing YOLO pipeline. Smooth per-track-ID. Fill gaps with morphological interpolation. ~4–5GB VRAM, 25+ FPS.
- **Tier 2 (best quality/effort ratio)**: DEVA with YOLO as detection source every 5 frames, temporal propagation between keyframes. ~8–12GB VRAM, 10–15 FPS.
- **Tier 3 (highest mask quality)**: YOLO-initialized SAM2 with re-prompting every 50–100 frames, processed in 200-frame overlapping windows with CPU offloading. ~16–20GB VRAM, 15–30 FPS.

---

## Area 3: Multi-model ensembles exploit complementary strengths for more reliable masks

Running YOLO, SAM2, and FastSAM independently wastes their complementary strengths. YOLO excels at class-aware detection but produces coarser boundaries. SAM2 generates precise boundaries but lacks class semantics. FastSAM offers a middle ground. Combining them systematically produces masks that are both more complete and more accurate than any single model.

### The YOLO→SAM2 cascade is the highest-impact single change

The most well-established and impactful pattern is using YOLO bounding boxes as SAM2 prompts. YOLO provides class-aware detection ("this is a person"), and SAM2 generates pixel-precise segmentation within that region. The accuracy improvement over YOLO-seg alone is substantial — SAM2 was trained on 1B+ masks with an architecture specifically optimized for boundary precision. An IEEE study on aerial segmentation (2024) confirmed YOLO+SAM produces more reliable segmentations than either model alone, and a MICCAI 2025 paper (YoloSAM) demonstrated the pipeline outperforming nnU-Net.

Ultralytics provides built-in integration:

```python
from ultralytics import YOLO, SAM
yolo = YOLO("yolo11x-seg.pt")
sam = SAM("sam2.1_b.pt")
results = yolo(image, classes=[0])  # detect people
for box in results[0].boxes.xyxy:
    sam_result = sam(image, bboxes=[box.tolist()])
```

**Grounded-SAM-2** (`github.com/IDEA-Research/Grounded-SAM-2`, 10K+ stars) provides a production-ready pipeline combining Grounding DINO (text-guided detection) with SAM2 for open-vocabulary segmentation and video tracking. It supports text prompts like "person. equipment. backpack." — eliminating the need for a fixed class list. The tradeoff: Grounding DINO is slower than YOLO for detection (~50–100ms vs. ~5ms), but the open-vocabulary capability adds flexibility.

### Fusion strategies: union for photogrammetry, majority vote for precision

For your photogrammetry application, **union (logical OR) is the recommended default** because unmasked person fragments cause worse reconstruction artifacts than slight over-masking. When any model detects a region as person/equipment, mask it.

**Majority voting (2-of-3 agreement)** provides a balanced alternative when false positives matter — for instance, when important scene features might be incorrectly flagged. A pixel masked by 2 of 3 models is almost certainly a real detection.

**Confidence-weighted fusion** offers the most principled approach but requires access to soft probability outputs and calibration across models. For binary masks, this means: compute `vote_sum = w₁ × mask_yolo + w₂ × mask_sam + w₃ × mask_fastsam` where weights reflect each model's empirical reliability, then threshold.

The **STAPLE algorithm** (Statistical Fusion via Simultaneous Truth and Performance Level Estimation) provides a more sophisticated framework: it treats each model as an "annotator" with unknown reliability, then uses an EM algorithm to simultaneously estimate the true segmentation and each model's sensitivity/specificity. Available in SimpleITK as `sitk.STAPLE()`, it's proven in multi-annotator medical imaging and directly applicable here.

For practical mask-level fusion, the **Weighted-Masks-Fusion** repository (`github.com/chrise96/Weighted-Masks-Fusion`) extends the well-known Weighted-Boxes-Fusion approach to instance segmentation masks. It matches masks across models using IoU, then fuses matched masks using confidence-weighted averaging. This originated from an 8th-place Kaggle competition solution and accepts mask arrays + confidence scores from multiple models.

### Conflict resolution favors SAM2 at boundaries and YOLO for detection

When models disagree, apply **hierarchical trust**:

- **For detection decisions** (is this a person?): Trust YOLO most — it's trained specifically for object detection with strong class discrimination.
- **For boundary precision** (where exactly is the person edge?): Trust SAM2 most — it was trained on 1B+ masks with architecture optimized for boundary accuracy.
- **For speed/fallback**: Trust FastSAM.

A practical boundary-weighted fusion approach: erode all masks slightly to find the high-confidence core where models agree, identify the boundary zone as the dilation minus the core, and in that boundary zone use SAM2's prediction exclusively. This leverages each model's strength at the appropriate spatial scale.

Model agreement itself serves as a confidence signal: pixels where all 3 models agree are high-confidence (keep as-is), pixels where 2-of-3 agree are medium-confidence (majority vote), and pixels where only 1 model flags are low-confidence. For photogrammetry, even low-confidence detections should typically be masked.

### Confidence calibration matters less than you'd expect for binary output

Different models produce incomparable confidence scores: YOLO outputs detection confidence (P(class|box)), SAM2 outputs mask IoU prediction scores, and FastSAM outputs YOLOv8 detection confidence. These measure fundamentally different quantities with different distributions.

**Temperature scaling** (`github.com/gpleiss/temperature_scaling`) is the simplest calibration — divide logits by a learned scalar T before softmax, learned on a validation set. **Local Temperature Scaling** (`github.com/uncbiag/LTS`, ICCV 2021) applies spatially varying calibration via a small CNN, which better handles the fact that boundary pixels are calibrated differently than interior pixels.

However, since your final output is binary (0/255), full calibration adds complexity with marginal benefit. The practical approach is to set **independent per-model confidence thresholds** empirically tuned on a small labeled set (e.g., YOLO conf > 0.5, SAM2 IoU > 0.7, FastSAM conf > 0.4), then apply your chosen fusion strategy to the thresholded binary masks.

### All three models fit comfortably in 24GB VRAM

Running YOLO11x-seg (~4GB), SAM2-base+ (~5–6GB), and FastSAM (~3–4GB) simultaneously with PyTorch overhead totals **~14–17GB** — fitting within your 24GB with headroom. Using lighter variants (YOLO11n-seg, SAM2-tiny, FastSAM-s) drops this to ~8–10GB. Sequential loading with `torch.cuda.empty_cache()` between models allows any combination at any size.

The recommended processing strategy is **adaptive, not exhaustive**: run YOLO-seg + YOLO→SAM2 cascade as the default (covering ~90% of cases), trigger FastSAM or SAM2 auto-mode as a fallback when YOLO detects fewer objects than expected based on temporal context, and run the full 3-model ensemble only for key frames or quality-critical images.

### The supervision library provides the unified interface

Roboflow's **supervision** library (`github.com/roboflow/supervision`) offers a `sv.Detections` class that unifies outputs from YOLO, SAM, FastSAM, and Transformers models. While it doesn't implement pixel-level mask fusion directly, its `.merge()` operation and built-in F1Score/MeanAveragePrecision metrics provide the scaffolding for a custom fusion pipeline. For evaluation without manual annotation, use cross-model agreement as a proxy for correctness, or measure downstream photogrammetry quality (reprojection error, point cloud density) as the ultimate ground truth.

---

## Practical integration roadmap for your pipeline

Given the interdependencies between these three areas, implementation should follow this sequence:

**Phase 1 — Immediate wins (1–2 weeks)**. Add BoT-SORT tracking to your existing YOLO pipeline for per-track temporal smoothing. Implement YOLO→SAM2 box-prompt cascade for higher-quality per-frame masks. Add CascadePSP edge refinement as a post-processing step (`pip install segmentation-refinement`). These changes are additive and don't require architectural changes.

**Phase 2 — Temporal propagation (2–4 weeks)**. Integrate DEVA for detect-then-propagate architecture, using YOLO as the detection source every 5 frames with Cutie-based temporal propagation between keyframes. Add flow-based mask interpolation (RAFT) for filling single-frame detection gaps. For equirectangular content, expand cubemap face FOV to 100–110° and implement overlap-aware mask merging.

**Phase 3 — Advanced refinement (4–6 weeks)**. Evaluate MatAnyone for combined edge-refinement and temporal-consistency in a single model — it could replace separate refinement and propagation steps. Implement majority-vote or STAPLE fusion across YOLO-seg, SAM2, and FastSAM outputs. Add SAM2 video mode as an alternative to DEVA for highest mask quality, processing in 200-frame windows with YOLO re-prompting every 50 frames.

**Phase 4 — Optimization and evaluation**. Tune per-model confidence thresholds on a small labeled set. Compare fusion strategies (union vs. majority vote) by measuring COLMAP/Metashape reconstruction quality. Profile VRAM and speed to optimize the processing pipeline for batch throughput.

---

## Conclusion

The research reveals three key insights that reshape how to approach this pipeline. First, **MatAnyone (CVPR 2025) collapses edge refinement and temporal consistency into a single model** by accepting a binary segmentation mask on the first frame and producing temporally consistent alpha mattes for the entire video — this is the most architecturally elegant solution for your video workflows. Second, **the DEVA paradigm of periodic detection + temporal propagation perfectly matches your existing pipeline structure** — you already have the detection stage (YOLO/SAM), and adding propagation between keyframes eliminates the flickering from per-frame independence without requiring full VOS integration. Third, **union fusion is the correct default for photogrammetry masking** because the reconstruction cost of under-masking (ghost artifacts from residual person pixels) far exceeds the cost of over-masking (losing a few edge pixels of scene geometry).

For equirectangular content, the field remains immature — Leader360V (2025) is the first large-scale dataset, suggesting 360°-native VOS models may emerge within 1–2 years. Until then, your cubemap decomposition with overlapping FOVs remains the correct approach. The practical bottleneck is not model quality but integration complexity: a YOLO→SAM2 cascade with BoT-SORT tracking and CascadePSP refinement, all running within ~15GB VRAM, would deliver substantial quality improvements over independent per-frame segmentation with minimal architectural disruption.