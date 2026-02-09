# Shadow Detection Options — Effectiveness vs Complexity Analysis

*Derived from shadow_masking_knowledge_base.md. All approaches ranked relative
to the target scenario: harsh noon shadows from people/equipment on outdoor
ground surfaces (beach, concrete, grass) across ~1400-frame datasets.*

---

## Effectiveness Spectrum

Ranked by expected real-world shadow masking quality for photogrammetry.
Accounts for accuracy, domain robustness, edge quality, and false positive rate.

```
MOST EFFECTIVE
│
│  ██ Hybrid: Learned detector + chromaticity verification + spatial filter
│  ██    (Best of both worlds — ML catches shadows, physics rejects false positives,
│  ██     spatial filter preserves environmental shadows)
│
│  █▓ TICA test-time adaptation (BER 1.17 on ISTD)
│  █▓    (Self-calibrates to each scene's lighting — ideal for domain gap,
│  █▓     but research prototype, not packaged)
│
│  █▓ FastInstShadow (CVPR 2025, mIoU 0.71 on SOBA)
│  █▓    (Instance-level: directly pairs shadows with casters — eliminates
│  █▓     the spatial filter heuristic entirely. 30+ FPS.)
│
│  █▓ MetaShadow (CVPR 2025, mIoU 71.0)
│  █▓    (Object-centric detection+removal+synthesis. Feed person mask → get
│  █▓     that person's shadow. Highest reported instance accuracy. No code.)
│
│  ██ ShadowAdapter / AdapterShadow (ESWA 2025)
│  ██    (SAM backbone = strong generalization. Trained on SBU+UCF+ISTD+CUHK.
│  ██     Auto grid prompts = fully automatic. Best SAM-based option.)
│
│  ██ SILT with PVTv2-B3 (ICCV 2023, BER ~4.2% on SBU)
│  ██    (Iterative label tuning explicitly trains against dark-surface false
│  ██     positives. Best single-model accuracy with available weights.)
│
│  █▓ SoftShadow (CVPR 2025)
│  █▓    (Penumbra-aware continuous masks — theoretically superior for noon
│  █▓     shadows with sharp umbra/soft penumbra. Part of removal pipeline.)
│
│  ██ SDDNet (ACM MM 2023, BER 4.86% on SBU-Refine)
│  ██    (Dual-layer disentanglement. Strong accuracy at high speed.
│  ██     Best practical choice for large batch processing.)
│
│  ██ FDRNet + CRF (ICCV 2021, BER 3.04% on SBU with CRF)
│  ██    (Best shadow boundary quality. CRF = crispest mask edges.
│  ██     Low BER but on original SBU, not SBU-Refine.)
│
│  ██ RASM removal diff (NTIRE 2025 winner)
│  ██    (Two-in-one: mask + de-lighted image. Regional attention preserves
│  ██     texture detail. Strong on hard shadows.)
│
│  █▓ Careaga & Aksoy intrinsic decomposition
│  █▓    (Physics-based albedo/shading split. No shadow training needed.
│  █▓     Excellent on uniform surfaces like sand. Bypasses domain gap.)
│
│  ██ DSDNet (ICCV 2019, BER 5.04% on SBU-Refine)
│  ██    (Solid mid-tier. Available via benchmark.)
│
│  ██ BDRAR (ECCV 2018, BER 5.62% on SBU-Refine)
│  ██    (Fast at 40 img/s. Older but well-tested.)
│
│  █▓ ShadowFormer diff trick
│  █▓    (Viable for harsh shadows where diff signal is strong. Noisy on
│  █▓     subtle shadows. Requires running full removal model for just a mask.)
│
│  ██ Hybrid chromaticity filter (c1c2c3 + intensity)
│  ██    (Strong on uniform surfaces — sand, concrete. Zero domain gap.
│  ██     Fails on textured/non-uniform surfaces and soft shadows.)
│
│  █▓ FSDNet (TIP 2021, ~4M params)
│  █▓    (Tiny but surprisingly capable. Custom CUDA ops limit portability.)
│
│  ██ Depth-based ray-tracing (sun position + geometry)
│  ██    (Physically correct for static geometry shadows. Needs reconstruction
│  ██     first = chicken-and-egg. Can't detect people shadows.)
│
│  ██ c1c2c3 chromaticity alone
│  ██    (Catches obvious hard shadows on uniform surfaces. High false negative
│  ██     rate on textured surfaces. No penumbra handling.)
│
│  ██ Current heuristic (masking_v2.py brightness ratio)
│  ██    (Only searches below people. Fixed threshold. Grayscale only.
│  ██     Misses angled shadows, dark surface confusion, no penumbra.)
│
LEAST EFFECTIVE
```

### Effectiveness notes

- **BER numbers aren't directly comparable** across datasets. SBU vs SBU-Refine
  vs ISTD have different characteristics. Cross-dataset BER roughly doubles.
- **Domain gap is the wildcard.** A model with BER 4% on SBU may perform at
  8-10% on your beach data. The hybrid approach and intrinsic decomposition
  are more robust here because they rely on physics, not learned patterns.
- **For your specific scenario** (harsh noon, uniform sand): chromaticity
  methods will overperform their general ranking because sand is nearly
  Lambertian and uniform. Learned models may slightly underperform because
  beach scenes are underrepresented in SBU/ISTD.
- The **hybrid approach** tops the list because it combines the best of both:
  learned model catches shadows that chromaticity misses (textured surfaces,
  soft shadows), chromaticity rejects false positives the model makes
  (dark surfaces), and spatial filtering preserves environmental shadows.

---

## Complexity Spectrum

Ranked by total effort to get running in your Windows pipeline.
Accounts for dependencies, setup difficulty, integration work, and maintenance.

```
SIMPLEST
│
│  ██ Current heuristic (already implemented)
│  ██    Effort: 0. Already in masking_v2.py line 1135.
│
│  ██ c1c2c3 chromaticity filter
│  ██    Effort: ~20 lines of NumPy. No dependencies beyond what you have.
│  ██    Just arctan ratios on RGB channels + threshold.
│
│  ██ Hybrid chromaticity + intensity filter
│  ██    Effort: ~50 lines. HSV conversion + normalized RGB + local window
│  ██    comparison. Pure OpenCV/NumPy.
│
│  █▓ Careaga & Aksoy intrinsic decomposition
│  █▓    Effort: pip install + ~30 lines of integration. Outputs albedo +
│  █▓    shading arrays. Threshold shading channel = shadow mask.
│  █▓    Needs PyTorch (already have it). pip-installable from GitHub.
│
│  ██ SDDNet (via Unveiling Deep Shadows benchmark)
│  ██    Effort: Clone benchmark repo, download weights from GitHub Releases,
│  ██    adapt test script for custom image directory. ~100 lines of wrapper.
│  ██    PyTorch 1.8+ (compatible with modern versions). No exotic deps.
│  ██    Windows-safe.
│
│  ██ SILT
│  ██    Effort: Clone repo, download weights from Google Drive, modify
│  ██    infer.py to accept arbitrary image paths (currently expects dataset
│  ██    directory with test.txt). ~100-150 lines of wrapper.
│  ██    timm + kornia + omegaconf dependencies. Windows-safe.
│
│  ██ DSDNet / BDRAR / MTMT (via benchmark)
│  ██    Effort: Same as SDDNet — benchmark provides unified codebase.
│  ██    Download different weight files, same inference path.
│
│  █▓ ShadowFormer diff trick
│  █▓    Effort: Clone repo, download weights, run inference, compute diff,
│  █▓    threshold. ~80 lines of wrapper + diff logic.
│  █▓    PyTorch 1.7+. Moderate — you're running a full restoration model
│  █▓    just to get a byproduct.
│
│  █▓ RASM removal + mask
│  █▓    Effort: Clone repo, download weights, adapt inference. Need to verify
│  █▓    whether mask is a direct output or needs diff extraction.
│  █▓    U-Net architecture = should be straightforward.
│
│  ██ FDRNet + CRF
│  ██    Effort: Clone repo, download weights from GitHub Releases.
│  ██    PROBLEM: pydensecrf is painful on Windows. Options:
│  ██    (a) skip CRF and use raw output, (b) install pydensecrf2,
│  ██    (c) use benchmark's version instead. Output is grid image
│  ██    (input|pred|GT) — needs extraction, not standalone masks.
│  ██    Older PyTorch (1.5) may need compatibility fixes.
│
│  █▓ ShadowAdapter / AdapterShadow
│  █▓    Effort: Clone repo, download SAM ViT-B/L checkpoint (~2.5GB),
│  █▓    download adapter weights from Google Drive (unverified contents),
│  █▓    install PyTorch Lightning + SAM dependencies.
│  █▓    Complex inference command with many flags.
│  █▓    Google Drive download may need manual intervention.
│  █▓    Moderate-high setup, but fully automatic once running.
│
│  █▓ FSDNet
│  █▓    Effort: Custom CUDA kernels (IRNN) require compilation.
│  █▓    CuPy dependency. Windows builds possible but finicky.
│  █▓    Tiny model but annoying setup.
│
│  █▓ Hybrid: learned + chromaticity + spatial filter
│  █▓    Effort: Sum of SDDNet/SILT setup + chromaticity code + spatial
│  █▓    filter logic. ~200-300 lines total integration. Conceptually
│  █▓    straightforward but most integration code to write.
│
│  █▓ Depth-based ray-tracing
│  █▓    Effort: Need reconstructed geometry first (chicken-and-egg).
│  █▓    Need sun position calculation (pvlib/suncalc).
│  █▓    Need ray-tracing against mesh (trimesh/embree).
│  █▓    Multi-step pipeline. Only works after initial reconstruction.
│
│  █▓ FastInstShadow (CVPR 2025)
│  █▓    Effort: Detectron2-based = C++ compilation required.
│  █▓    Detectron2 on Windows is notoriously difficult.
│  █▓    May need WSL2. Code availability not fully verified.
│
│  █▓ SoftShadow (CVPR 2025)
│  █▓    Effort: Part of a removal pipeline, not standalone detection.
│  █▓    Would need to extract the mask component. Penumbra loss
│  █▓    training is novel = less community support.
│
│  ██ SAM-Adapter (train from scratch)
│  ██    Effort: HIGH. Need shadow training data (SBU/ISTD download),
│  ██    SAM checkpoint, 4x A100 equivalent for training.
│  ██    No shadow weights provided — must train yourself.
│  ██    Framework, not solution.
│
│  █▓ MetaShadow (CVPR 2025)
│  █▓    Effort: NO CODE AVAILABLE. Paper only.
│  █▓    Cannot be implemented without significant reproduction effort.
│
│  ██ TICA test-time adaptation
│  ██    Effort: Research prototype. Not packaged for general use.
│  ██    Would need to reproduce from paper. HRNet-18 base model.
│  ██    Conceptually elegant but significant implementation work.
│
│  ██ Neural reconstruction bypass (WildGaussians, SpotLessSplats, etc.)
│  ██    Effort: Requires changing your entire reconstruction pipeline
│  ██    from COLMAP/Metashape to NeRF/3DGS. Different output format
│  ██    (neural representation vs mesh+texture). Not a shadow masking
│  ██    solution — it's a different workflow entirely.
│
MOST COMPLEX
```

---

## Two-Dimensional View

Plotting effectiveness (Y) against complexity (X):

```
                         EFFECTIVENESS
                              ▲
                         High │
                              │
          Hybrid learned+     │                    TICA ○
          classical+spatial ● │           MetaShadow ○
                              │  FastInstShadow ○
              ShadowAdapter ◐ │                    SAM-Adapter (trained) ○
                              │
                    SILT ●    │  SoftShadow ○
                   SDDNet ●   │
              FDRNet+CRF ●    │
                              │  RASM ◐
         Intrinsic decomp ◐   │
                              │
                   DSDNet ●   │
                    BDRAR ●   │        Depth ray-trace ○
                              │
                              │  ShadowFormer diff ◐
       Chromaticity hybrid ●  │
                              │                    Neural recon bypass ○
                    FSDNet ◐  │
                              │
           c1c2c3 alone ●     │
                              │
     Current heuristic ●      │
                              │
                         Low  │
                              └──────────────────────────────────►
                              Simple                    Complex
                                       COMPLEXITY

     ● = Pretrained weights verified, Windows-compatible
     ◐ = Weights exist but unverified, or minor compatibility concern
     ○ = No weights / no code / requires training / major platform issue
```

---

## Sweet Spots

Three clusters emerge from this analysis:

### 1. Quick Win (days to integrate)

**SDDNet or SILT + c1c2c3 verification**

- Download weights from Unveiling Deep Shadows benchmark or SILT repo
- Write ~150 lines of wrapper code in masking_v2.py
- Add c1c2c3 chromaticity check as false-positive filter (~20 lines)
- Use existing person mask dilation for spatial filtering
- All Windows-safe, all dependencies already satisfied or trivially installed
- Expected result: dramatically better than current heuristic, especially
  on your beach dataset where chromaticity verification will shine

### 2. Best Practical Integration (weeks to integrate)

**Full hybrid pipeline: SDDNet + chromaticity + spatial filter + SILT fallback**

- SDDNet as primary detector (speed for 1400 frames)
- c1c2c3 chromaticity verification to reject dark-surface false positives
- Spatial filter: `Final Mask = (Shadow Mask ∩ dilated_person_region) ∪ Person Mask`
- SILT as optional second opinion on ambiguous frames
- Careaga & Aksoy intrinsic decomposition as third verification layer
- ~300 lines of integration code
- Produces the highest-confidence masks with the lowest false positive rate

### 3. Research Frontier (months, or wait for releases)

**ShadowAdapter + FastInstShadow + TICA**

- ShadowAdapter for best SAM-based generalization (if weights work)
- FastInstShadow for automatic shadow-object pairing (if code works on Windows)
- TICA for self-calibrating domain adaptation (if someone packages it)
- These collectively solve the remaining hard problems (domain gap,
  instance association, penumbra) but none are fully plug-and-play today

---

## Recommendation for Your Beach Dataset

Your scenario (high noon, cloudless, uniform sand, harsh defined shadows)
is actually the **easiest case** for shadow detection. The shadows have:
- Maximum contrast (direct overhead sun)
- Sharp boundaries (minimal penumbra at noon)
- Uniform background (sand = nearly Lambertian)
- Strong chromaticity signal (sunlit sand vs blue-shifted shadow on sand)

**Start with cluster 1.** SDDNet + c1c2c3 will likely handle 95%+ of your
beach shadows correctly. If edge cases remain, escalate to cluster 2.
Cluster 3 is for when you encounter scenes that genuinely challenge the
proven models — complex urban environments, mixed lighting, textured
ground surfaces where chromaticity invariants break down.
