# Reconstruction Zone — .exe Packaging: Complete Record

**Created:** 2026-05-22
**Status:** Planning — ready to execute
**Branch:** `release/v1` (primary), `release/v1-lite` (fallback/reference)
**Goal:** Ship Reconstruction Zone as a standalone Windows .exe that protects source code, bundles all dependencies, and works on any NVIDIA GPU machine without Python installed.

This document is the single source of truth for the .exe packaging effort. It records what was tried, what worked, what failed, the current state of all artifacts, and the forward plan. Hand this to any agent or collaborator to loop them in completely.

---

## End Result

When this roadmap is complete, the deliverable is:

**A zip file (~8-9 GB)** uploaded to Google Drive (then Gumroad for sale). Inside:

```
ReconstructionZone/
├── ReconstructionZone.exe        (~910 MB, native compiled binary)
├── [Nuitka dist files]           (torch, CUDA DLLs, compiled modules, data files)
├── ffmpeg/
│   ├── ffmpeg.exe
│   └── ffprobe.exe
├── QUICKSTART.txt
├── LICENSE.txt
└── THIRD_PARTY_LICENSES.txt
```

**User experience:**
1. Download zip, extract anywhere
2. Double-click `ReconstructionZone.exe`
3. First launch: setup wizard downloads AI model weights (~1.2 GB for RF-DETR, optional 3.3 GB for SAM3)
4. App is fully functional. No Python, no terminal, no dependency management.

**No Python required. No install step. Extract and run.**

### Two distribution channels

| Channel | Branch | Format | Tabs | Models | YOLO? | Price |
|---------|--------|--------|------|--------|-------|-------|
| **Gumroad** (paid) | `release/v1` | Compiled .exe (zip) | 3 (Extract, Mask, Review) | RF-DETR + SAM3 | **NO** (AGPL) | Paid |
| **GitHub** (free) | `main` | Source code (work in progress) | All (whatever is current) | All | **YES** | Free |

**The paid product is a polished, scoped .exe** — 3 tabs, strongest features only, compiled to native code. Built from `release/v1`, which is a frozen, curated subset of the codebase.

**The GitHub repo is just `main`** — the ongoing work-in-progress. All tabs, all features, all experiments. Not packaged, not polished, not a "release." Users who find it can clone and run from source. No compiled binary provided, no install script, no hand-holding. It's the open workshop, not the product.

The .exe is the product. GitHub is the repo.

### Build flag system

A constant in `prep360/distribution.py` controls which models are available:

```python
DISTRIBUTION = "github"  # or "gumroad" — set by build script before Nuitka compilation
```

When `DISTRIBUTION == "gumroad"`:
- YOLO and FastSAM imports are skipped (AGPL compliance)
- CLI entry point prints "CLI not available in this build" and exits
- GUI hides YOLO/FastSAM model options from dropdowns

When `DISTRIBUTION == "github"`:
- All models available
- CLI works
- Full feature set

---

## Table of Contents

1. [Why .exe (Not Install Script)](#1-why-exe-not-install-script)
2. [Product Context](#2-product-context)
3. [Tool Evaluation: Why Nuitka](#3-tool-evaluation-why-nuitka)
4. [Build Environment](#4-build-environment)
5. [The Singledispatch Problem (and Its Fix)](#5-the-singledispatch-problem-and-its-fix)
6. [History of Build Attempts](#6-history-of-build-attempts)
7. [Current State of All Artifacts](#7-current-state-of-all-artifacts)
8. [Runtime Issues Catalog](#8-runtime-issues-catalog)
9. [Release Branch Comparison](#9-release-branch-comparison)
10. [Scope for v1.0](#10-scope-for-v10)
11. [Execution Plan](#11-execution-plan)
12. [Build Commands Reference](#12-build-commands-reference)
13. [Verification Checklist](#13-verification-checklist)
14. [Risk Register](#14-risk-register)
15. [Fallback: Python 3.12 + Nuitka](#15-fallback-python-312--nuitka)
16. [Post-v1.0 Roadmap](#16-post-v10-roadmap)
17. [Open Questions](#17-open-questions)
18. [Appendix: Prior Documents Index](#18-appendix-prior-documents-index)

---

## 1. Why .exe (Not Install Script)

Two distribution approaches were explored:

| Approach | Branch | How it works | Source protection |
|----------|--------|-------------|-------------------|
| **Compiled .exe** | `release/v1` | Nuitka compiles Python → C → native binary | **Strong** — reverse engineering requires C decompilation |
| **Install script** | `release/v1-lite` | User installs Python, runs `install.bat`, launches via `launch.bat` | **None** — raw .py files on disk |

The install script approach (v1-lite) was completed and works. But it ships raw Python source code, making the product trivially pirateable. The .exe approach is chosen specifically for **source code protection** — this is a commercial product intended for sale.

This is not a hypothetical concern. Competitors (e.g., kotohibi's "Extract Sharpest Frame" tool) sell packaged .exe builds on BOOTH marketplace. Our tool is more capable but not yet distributable.

---

## 2. Product Context

**What:** Windows desktop app for photogrammetry preprocessing — frame extraction from video, 360° reframing, automated AI masking, mask review/editing.

**Who:** Hobbyist Gaussian splat creators, drone photogrammetry users, anyone preparing datasets for Metashape/COLMAP/RealityScan/etc.

**Selling points:**
- Intelligent frame extraction (sharpest frame selection, configurable ring-based reframing)
- Multi-model AI masking (SAM3, YOLO26, RF-DETR) with cubemap decomposition for 360° images
- Full review workflow (thumbnail gallery, interactive OpenCV editor)

**Competition:** kotohibi_3d sells "Extract Sharpest Frame" on BOOTH — a single-purpose tool that extracts sharp frames and is adding SAM3 text-prompt masking. Reconstruction Zone covers the full pipeline with more models and more features.

---

## 3. Tool Evaluation: Why Nuitka

### Tools evaluated

| Tool | What it produces | Source protection | PyTorch compat | Verdict |
|------|-----------------|-------------------|----------------|---------|
| **Nuitka 4.0.8** | Python → C → native .exe | **Strong** (C binary) | Works with YAML patches for singledispatch | **CHOSEN** |
| PyInstaller 6.20 | Bundles .pyc in zip archive | **None** (trivially extractable with pyinstxtractor + uncompyle6) | Circular import on 3.14+torch 2.10 (6 fix attempts failed) | Rejected — no code protection |
| cx_Freeze 8.6 | Bundles .pyc alongside .exe | **None** (same as PyInstaller) | Multiple closed CUDA issues, untested | Rejected — no code protection |
| Cython + PyInstaller | .py → .pyd (native) + PyInstaller bundle | **Partial** (.pyd files are native, but framework is PyInstaller) | Two tools to debug | Rejected — complexity, still PyInstaller underneath |

**Key finding:** PyInstaller's own documentation states ".pyc files could in principle be decompiled to reveal the logic of your code" and recommends Cython as mitigation. Nuitka eliminates the problem entirely by compiling to C.

### Nuitka version timeline

| Version | Python 3.14 support | Relevant changes |
|---------|--------------------|--------------------|
| 4.0.8 (current) | "Experimental" | All prior builds used this. Works with YAML patches. |
| 4.1 | "Experimental, barely made the cut" | Decorator fixes for deferred annotations. No singledispatch fix. |
| 4.1.1 (latest) | Unknown | Bugfix release. No singledispatch-specific changes found in changelog. |

**Recommendation:** Upgrade to 4.1.1 before the next build — it may improve 3.14 decorator handling. But the YAML patches remain necessary regardless.

---

## 4. Build Environment

### Development machine

| Component | Value | Verified |
|-----------|-------|----------|
| Python | 3.14.0 at `C:\Python314` | `python --version` |
| PyTorch | 2.10.0+cu126 (CUDA 12.6) | `torch.__version__` |
| TorchVision | 0.25.0+cu126 | `torchvision.__version__` |
| GPU | NVIDIA RTX 3090 Ti | `torch.cuda.get_device_name(0)` |
| OS | Windows 11 Pro 10.0.26200 | `platform.platform()` |
| Nuitka | 4.0.8 | `nuitka --version` |
| MSVC | cl 14.5 (Visual Studio 2022) | Used by Nuitka for C compilation |

### Python 3.12 fallback (installed, partially configured)

| Component | Value |
|-----------|-------|
| Python | 3.12.6 at `C:\Python312` |
| PyTorch | 2.6.0+cu124 (CUDA available) |
| Nuitka | **NOT installed** |
| Missing packages | customtkinter, rfdetr, py360convert |

### Key PATH hazards

- **Miniconda** at `C:\Users\alexm\miniconda3` — can shadow Python, contains `tiff.dll` that poisons COLMAP subprocess calls
- Build commands MUST use explicit interpreter paths (`C:\Python314\python.exe`)
- Subprocess calls MUST sanitize PATH to exclude miniconda

### Installed packages relevant to the build (Python 3.14)

Core: torch 2.10.0+cu126, torchvision 0.25.0+cu126, opencv-contrib-python, customtkinter, numpy, scipy, Pillow, tqdm, pyyaml, py360convert

Detection/segmentation: ultralytics (YOLO26), rfdetr, supervision, transformers (<5.0), huggingface_hub

Optional/lazy: sam3, ViTMatte, LiVOS, Cutie (imported inside functions, not at module level)

---

## 5. The Singledispatch Problem (and Its Fix)

This is the central technical challenge of the entire packaging effort. Understanding it is essential.

### What happens

PyTorch uses `functools.singledispatch` with implicit type dispatch (type inferred from function annotations) in 6 places loaded during `import torch`:

| Module | Line | Decorator | Dispatch type |
|--------|------|-----------|---------------|
| `torch.library` | ~556 | `@define.register` | `Library` |
| `torch.library` | ~670 | `@impl.register` | `Library` |
| `torch._refs` | ~2131 | `@_to_dispatch.register` | `torch.device` |
| `torch._refs` | ~2149 | `@_to_dispatch.register` | `str` |
| `torch._refs` | ~2167 | `@_to_dispatch.register` | `torch.dtype` |
| `torch._refs` | ~2183 | `@_to_dispatch.register` | `Tensor` |

All 6 are loaded at import time — `'torch._refs' in sys.modules` is `True` after `import torch`.

### Why it breaks in Nuitka (Python 3.14 only)

Python 3.14 introduced PEP 649 (deferred evaluation of annotations). Functions now have `__annotate__` (a callable) instead of `__annotations__` (a dict). The stdlib `functools.singledispatch.register()` on 3.14 calls `__annotate__` to discover the dispatch type:

```python
# Python 3.14 singledispatch.register():
ann = getattr(cls, '__annotate__', None)  # <-- uses __annotate__
if ann is None:
    raise TypeError(...)
```

Nuitka compiles Python functions to C. The compiled functions do NOT preserve `__annotate__`. So `getattr(func, '__annotate__', None)` returns `None`, and `register()` raises `TypeError: Invalid first argument to register()`.

### Why it does NOT break on Python 3.12

Python 3.12's `singledispatch.register()` reads `__annotations__` (a plain dict):

```python
# Python 3.12 singledispatch.register():
ann = getattr(cls, '__annotations__', {})  # <-- uses __annotations__
```

Nuitka DOES preserve `__annotations__` dicts. So this works on 3.12 without patches.

### The YAML fix

File: `scripts/nuitka-torch-fix.yml`

Nuitka's anti-bloat system can rewrite source code before compilation. The YAML patches convert implicit `@register` (needs annotations) to explicit `@register(Type)` (no annotations needed):

**`torch.library` (2 decorators) — `replacements_plain` (simple string replacement):**
```yaml
- module-name: 'torch.library'
  anti-bloat:
    - replacements_plain:
        '@define.register': '@define.register(Library)'
        '@impl.register': '@impl.register(Library)'
```

**`torch._refs` (4 decorators) — `replacements_re` (regex, needed because all 4 share `@_to_dispatch.register` text):**
```yaml
- module-name: 'torch._refs'
  anti-bloat:
    - replacements_re:
        '@_to_dispatch\.register\ndef _to_device\(\n    device: torch\.device,':
          '@_to_dispatch.register(torch.device)\ndef _to_device(\n    device,'
        # ... (3 more similar patterns for str, torch.dtype, Tensor)
```

**`torch._dynamo._trace_wrapped_higher_order_op` (annotation compat):**
```yaml
- module-name: 'torch._dynamo._trace_wrapped_higher_order_op'
  anti-bloat:
    - replacements_plain:
        'from typing import Any, Optional': 'from typing import Any, Optional, List'
        'shape: list[int],': 'shape: List[int],'
        'indices: list[Tensor],': 'indices: List[Tensor],'
```

### Checksum mismatch warning

The build log shows: `Detected 2 module(s) with mismatching checksum in scripts/nuitka-torch-fix.yml: torch.library,torch._refs`

This means torch source changed since the YAML was authored (e.g., `pip install --upgrade torch`). The patches use string matching, not line numbers, so they may still work — but verification is required. The `--show-anti-bloat-changes` flag confirms whether all 6 replacements fired.

**Despite this warning, the torch smoke exe works correctly** (verified 2026-05-22). This suggests the string patterns still match the current torch source, and the checksum mismatch is cosmetic.

---

## 6. History of Build Attempts

### Timeline

| Date | What | Tool | Result |
|------|------|------|--------|
| 2026-03-24 | Gumroad release design written | — | Product vision, feature inventory |
| 2026-05-08 | Release/v1 branch created, trimmed to 5 tabs | — | Removed Adjust, Coverage tabs |
| 2026-05-08 | PyInstaller build attempted (Python 3.14 + torch 2.10) | PyInstaller 6.20 | **FAILED** — circular import `torch.autograd` |
| 2026-05-08 | 6 PyInstaller fix attempts | PyInstaller | All failed — each fix revealed deeper circular deps |
| 2026-05-08 | **PyInstaller abandoned for Python 3.14** | — | Decision: try Nuitka |
| 2026-05-08 | First Nuitka build | Nuitka 4.0.8 | Built successfully, 917 MB exe, GUI launches |
| 2026-05-08 | Nuitka build tested | — | **FAILED** — `TypeError: Invalid first argument to register()` (singledispatch) |
| 2026-05-10 | Exe packaging plan + review written (371 lines) | — | Detailed analysis of Options A/B/C |
| 2026-05-10 | YAML singledispatch fix designed | — | 6 decorators across 2 modules identified |
| 2026-05-11 | Torch smoke test compiled and run | Nuitka + YAML | **PASSED** — prints torch version, CUDA True, OK |
| 2026-05-11 | GUI smoke test compiled | Nuitka + YAML | Built, but fails on `_version` import |
| 2026-05-11 | Full GUI build completed (×1) | Nuitka + YAML | Build succeeds, 910 MB exe, runtime issues found |
| 2026-05-11 | Build issues documented (4 runtime issues) | — | cv2 recursion, wizard errors, model paths |
| 2026-05-11 | Source-level fixes applied to setup_wizard.py | — | Queue-based downloads, multi-path model checks |
| 2026-05-11 | Phantom code incident discovered | — | Commit `2741e55` introduced fabricated code referencing non-existent modules |
| 2026-05-11 | Remediation plan written | — | Cleanup, audit, prevention |
| 2026-05-12 | Phase 6b: 4 more full GUI builds | Nuitka + YAML | All compiled successfully (~910 MB each) |
| 2026-05-12 | Phase 6b: RF-DETR smoke builds (×3) | Nuitka + YAML | Mixed results — rfdetr imports are complex |
| 2026-05-12 | Multiple validation/verification scripts run | — | Pycache, syntax, model path checks |
| 2026-05-13 | v1-lite branch created (install script approach) | — | 3-tab GUI, install.bat, setup_install.py |
| 2026-05-13 | v1-lite completed and verified | — | All phases done, tested from scratch |
| 2026-05-14–22 | Work shifted to main branch (lint fixes, feature work) | — | 3 commits landed on main that release branches don't have |
| **2026-05-22** | **This document created** | — | Audit of all artifacts, forward plan |

### PyInstaller failure details

6 attempts to fix `AttributeError: partially initialized module 'torch' has no attribute 'autograd'`:

1. Runtime hook pre-importing `torch.autograd` — revealed `torch.nested` dependency
2. Pre-importing `torch.nested` — revealed `torch._C` loading order issue
3. Explicit `torch._C` load before autograd — revealed `torch.utils` dependency
4. Custom module loader reordering — too deep in PyInstaller internals
5. `module_collection_mode = pyz+py` — no effect
6. `module_collection_mode = py` — no effect

**Conclusion:** PyInstaller's custom module loader (`pyimod02_importers`) triggers torch modules in a different order than CPython's native loader. The circular dependency graph in torch 2.10's `__init__` is too complex to patch externally. Nuitka uses CPython's native loader and doesn't have this problem.

### Phantom code incident (2026-05-11)

Commit `2741e55` ("release(v1): trim experimental features for v1 release") introduced ~200 lines of fabricated code referencing three Python modules that never existed (`adjustment_recipe`, `color_pipeline`, `dual_fisheye_dataset` as imports in the wrong locations). This caused Nuitka builds to stall for hours analyzing imports that didn't resolve. Multiple sessions were spent debugging before the root cause was identified. Documented in `planning/plans/2026-05-11-phantom-code-incident-report.md` on the release/v1 branch.

---

## 7. Current State of All Artifacts

### Build outputs (verified 2026-05-22)

| Artifact | Path | Size | Status |
|----------|------|------|--------|
| **Torch smoke exe** | `dist_test/smoke_import_torch.dist/smoke_import_torch.exe` | 902 MB | **WORKS** — run verified, prints `2.10.0+cu126`, `True`, `OK` |
| GUI smoke exe | `dist_test/smoke_import_gui.dist/smoke_import_gui.exe` | 519 MB | **FAILS** — `ModuleNotFoundError: _version` |
| Full GUI exe (finalvacuum) | `dist_test/phase-6b-full-gui-finalvacuum-20260512/.../ReconstructionZone.exe` | 954 MB | **COMPILED** (exit 0) — runtime untested |
| Full GUI exe (maskpreviewfix) | `dist_test/phase-6b-full-gui-maskpreviewfix-20260512/.../ReconstructionZone.exe` | 954 MB | **COMPILED** (exit 0) — runtime untested |
| Full GUI exe (newinstallfix) | `dist_test/phase-6b-full-gui-newinstallfix-20260512/.../ReconstructionZone.exe` | 954 MB | **COMPILED** (exit 0) — runtime untested |
| Full GUI exe (workspacecache) | `dist_test/phase-6b-full-gui-workspacecache-20260512/.../ReconstructionZone.exe` | 954 MB | **COMPILED** (exit 0) — runtime untested |
| RF-DETR smoke exe | `dist_test/phase-6b-rfdetr-a2-workspacecache-20260512/.../smoke_import_rfdetr.exe` | 928 MB | Compiled — status unknown |
| Latest Nuitka build | `dist_nuitka/reconstruction_zone.dist/` | ~8.5 GB folder | **INCOMPLETE** — no .exe produced (build errored) |

### Build infrastructure files

| File | Location | Purpose |
|------|----------|---------|
| `scripts/nuitka-torch-fix.yml` | main + release/v1 | YAML patches for singledispatch (3 modules, 6+ decorators) |
| `scripts/smoke_import_torch.py` | release/v1 | Minimal torch compile test |
| `scripts/smoke_import_gui.py` | release/v1 | GUI import compile test |
| `scripts/pyinstaller_runtime_hook.py` | release/v1 | ffmpeg PATH injection (reusable) |
| `scripts/copy_cv2_cuda_dlls.ps1` | release/v1 | Post-build DLL copy for OpenCV |
| `scripts/copy_ultralytics_runtime_data.ps1` | release/v1 | Post-build data copy for YOLO |
| `reconstruction_zone.spec` | main (stale) | PyInstaller spec — **abandoned**, not current |
| `scripts/build_gumroad.py` | main (stale) | Gumroad build script — references old PyInstaller approach |

### App infrastructure for packaging (on release/v1)

| File | Purpose |
|------|---------|
| `reconstruction_gui/app_paths.py` | Centralized path resolution (crash log, prefs, cache dirs) |
| `reconstruction_gui/model_paths.py` | Model weight location resolution (multi-path search) |
| `reconstruction_gui/setup_wizard.py` | First-launch wizard: health checks, model downloads, SAM3 gate |
| `reconstruction_gui/sam3_setup.py` | SAM3 gated model state machine (HuggingFace token flow) |
| `reconstruction_gui/subprocess_env.py` | Subprocess environment isolation (PATH sanitization) |

### Disk usage

| Directory | Size | Action needed |
|-----------|------|---------------|
| `dist_test/` | ~30 GB (estimated) | Contains all smoke + phase 6b builds. Keep for reference. |
| `dist_nuitka/` | ~17 GB | Incomplete build + cache. Can reuse `reconstruction_zone.build/` for faster rebuilds. |
| `dist/` (PyInstaller) | ~8.7 GB | **Dead.** Safe to delete. |
| `build/` (PyInstaller) | ~248 MB | **Stale.** Safe to delete. |
| `bench_output/` | ~7 GB | Benchmark images. Not related to packaging. |

---

## 8. Runtime Issues Catalog

These are bugs in the compiled .exe that prevent it from working correctly. The build succeeds — these are runtime failures.

### Issue 1: `_version` module not found

- **Error:** `ModuleNotFoundError: No module named '_version'`
- **Where:** `reconstruction_zone.py` line 74: `from _version import __version__`
- **Why:** Bare import relies on `reconstruction_gui/` being on sys.path. Nuitka standalone resolves modules differently.
- **Fix:** Use explicit import: `from reconstruction_gui._version import __version__` with try/except fallback.
- **Status:** Not fixed. Trivial.
- **Effort:** 5 minutes.

### Issue 2: cv2 CUDA DLL loading

- **Error:** `ImportError: DLL load failed while importing cv2` or `recursion is detected during loading of "cv2" binary extensions`
- **Where:** Any code that imports cv2 (masking pipeline, setup wizard model downloads)
- **Why:** OpenCV's `.pyd` depends on CUDA DLLs (`npp*64_13.dll`, `cublas64_13.dll`, etc.) that Nuitka doesn't copy. Also, Nuitka may produce duplicate `cv2.pyd` files at different paths causing recursive loading.
- **Fix options:**
  - A) Post-build script copies required CUDA DLLs (`scripts/copy_cv2_cuda_dlls.ps1` exists)
  - B) Switch to `opencv-python-headless` (no CUDA deps). App doesn't use OpenCV CUDA ops — all GPU work is through PyTorch.
  - C) Setup wizard download functions were rewritten to avoid cv2 imports entirely (fix applied on release/v1 in `setup_wizard.py`). But masking pipeline still needs cv2 at inference time.
- **Status:** Partially mitigated (wizard downloads avoid cv2). Not verified for inference path.
- **Effort:** 30-60 minutes.

### Issue 3: COLMAP subprocess DLL contamination

- **Error:** Native Windows dialog: `colmap.exe - Ordinal Not Found: The ordinal 127 could not be located in tiff.dll`
- **Where:** `ColmapRunner.validate_binary()` → `subprocess.run([colmap.exe, "help"])`
- **Why:** Miniconda's `tiff.dll` is on PATH before COLMAP's own DLL.
- **Fix:** `_isolated_subprocess_env(binary_path)` helper that sanitizes PATH. File `subprocess_env.py` exists on release/v1 but may not be wired into all call sites.
- **Status:** Helper exists, integration incomplete.
- **Note:** If Align tab is deferred to v1.0, COLMAP code doesn't run. But ffmpeg/ffprobe subprocess calls also need sanitization.
- **Effort:** 30 minutes.

### Issue 4: Setup wizard threading crash

- **Error:** `RuntimeError: main thread is not in main loop`
- **Where:** Background download thread calls `self._root.after()`
- **Why:** tkinter requires all UI calls from the main thread. `after()` internally calls `_register()` which checks the calling thread.
- **Fix:** Queue-based pattern: background thread → `queue.Queue()` → main thread polls with `after(100, _poll)`. This was designed and partially implemented in `setup_wizard.py` on release/v1.
- **Status:** Fix designed, implementation unclear. Needs verification.
- **Effort:** 30-60 minutes.

### Issue 5: Wizard reports success despite download failures

- **Error:** Wizard shows "Setup Complete!" even when RF-DETR and YOLO26 downloads failed (due to cv2 recursion).
- **Fix:** Download worker now tracks failures and sends `("all_done_with_errors", failures)` instead of unconditional `("all_done",)`. Applied in `setup_wizard.py` on release/v1.
- **Status:** Fixed in source on release/v1. Not verified in build.
- **Effort:** Done.

### Issue 6: Model path resolution in packaged app

- **Error:** Wizard shows models as "missing" even when cached on disk.
- **Why:** Check functions looked in CWD (which differs between source and packaged) or only checked one cache location.
- **Fix:** Multi-path search: torch hub cache, HF cache, ultralytics cache, CWD, app bundle dir. Applied in `setup_wizard.py` on release/v1.
- **Status:** Fixed in source on release/v1. Not verified in build.
- **Effort:** Done.

### Issue 7: CTkToplevel crash after wizard closes

- **Error:** `TypeError: 'ReconstructionZone' object is not callable` (in tkinter `nametowidget`)
- **Where:** After wizard CTkToplevel is destroyed, main app's mainloop crashes.
- **Why:** Possibly stale callback reference from destroyed wizard, or Nuitka-specific CTkToplevel destruction issue.
- **Status:** Hypothesis: caused by error cascade from Issues 2+5. May resolve when those are fixed. Needs testing.
- **Effort:** Unknown — may be 0 (resolved by other fixes) or significant (Nuitka+CTk interaction).

### Issue 8: Nuitka YAML checksum mismatch

- **Warning:** `Detected 2 module(s) with mismatching checksum`
- **Why:** Torch source changed after YAML was written.
- **Impact:** Patches may still work (string matching), but must verify with `--show-anti-bloat-changes`.
- **Status:** Warning present, but torch smoke exe works. Patches are likely still valid.
- **Effort:** 15 minutes to verify, 30 minutes if YAML needs updating.

---

## 9. Release Branch Comparison

### Lineage

```
main ──┬── release/v1       (4 commits ahead, 3 behind main)
       └── release/v1-lite   (4 shared + 5 unique = 9 ahead, 3 behind main)
```

### Commits behind main (both branches)

| Commit | Description |
|--------|-------------|
| `c8afa2c` | fix(lint): resolve all 397 ruff errors failing CI |
| `77e6eac` | fix: restore all work lost during release/v1 branch reset |
| `31cbadc` | fix: restore Adjust tab files lost during release/v1 branch reset |

### release/v1 — compiled .exe approach

- **Tabs:** All 6 (Projects, Extract, Mask, Review, Align, Coverage) — or 5 after trim
- **Packaging:** Nuitka standalone → native .exe
- **New modules:** app_paths, model_paths, setup_wizard, sam3_setup, subprocess_env, adjust_engine
- **Testing:** 15+ smoke test scripts, validation scripts
- **Status:** Builds compile. Runtime issues identified and partially fixed. Not usable yet.

### release/v1-lite — install script approach

- **Tabs:** 3 only (Extract, Mask, Review) — Projects, Align, Coverage removed
- **Packaging:** install.bat + launch.bat + setup_install.py (user needs Python 3.10-3.12)
- **New modules:** setup_wizard_lite, setup_install.py
- **Compat fixes:** Python 3.10-3.12, f-string fix for 3.11, cu128 index, deferred ultralytics import
- **Status:** Complete, verified, all phases done.

### What v1-lite got right (lessons for v1)

1. **3-tab scope** — Extract, Mask, Review covers the core workflow. Ship less, ship sooner.
2. **Deferred ultralytics import** — `import ultralytics` at module level stalls startup by seconds. Must be lazy.
3. **CUDA index mapping** — Default PyPI torch is CPU-only. Installer must use `--index-url` for CUDA wheels.
4. **Python 3.10-3.12 target** — Broader compatibility than 3.14.
5. **setup_wizard_lite.py** — Simpler than setup_wizard.py, but same core flow (health checks + model downloads).

---

## 10. Scope for v1.0

### Tabs

| Tab | Ship in v1.0? | Rationale |
|-----|--------------|-----------|
| **Extract** | YES | Core workflow step 1 |
| **Mask** | YES | Primary selling point |
| **Review** | YES | Core workflow step 3 |
| Projects | DEFER | Project management is nice-to-have |
| Align | DEFER | Requires COLMAP (external binary), adds complexity |
| Coverage | DEFER | Depends on COLMAP data, niche use case |

### Models and downloads

| Model | License | Size | Bundled in .exe? | Distribution |
|-------|---------|------|-----------------|--------------|
| YOLO26 | AGPL-3.0 | 6.5 MB | No | First-launch download (automatic) |
| RF-DETR | Apache 2.0 | ~1.2 GB | No | First-launch download (automatic) |
| SAM3 | SAM License | 3.3 GB | No | First-launch download (HuggingFace gated — requires user approval) |

**Total first-launch download:** ~4.5 GB. User must have internet on first run. SAM3 requires free HuggingFace account + model access request.

**AGPL note:** YOLO26 is AGPL-3.0. The paid .exe (Gumroad) must NOT include YOLO — excluded via the distribution flag system (`prep360/distribution.py`). The free GitHub release is source code only (no .exe), so YOLO is included under GPL v3. For initial friend-testing of the .exe, YOLO can be included (private use, no distribution).

### External tools

| Tool | Bundled? | Notes |
|------|----------|-------|
| ffmpeg + ffprobe | **YES** (~250 MB) | Core functionality (video extraction) |
| COLMAP | NO | Align tab deferred |
| exiftool | NO | Optional (SRT geotagging). Document as prerequisite if used. |

### Distribution format

**Paid (.exe):** Zip archive → user extracts folder → runs `ReconstructionZone.exe`. Add Inno Setup installer later for a more professional experience. Distributed via Gumroad. Built from `release/v1`.

**Free (GitHub):** Just the `main` branch as-is. Work in progress. No packaging, no install scripts, no curated release. If someone finds the repo and can figure out how to run it, great.

---

## 11. Execution Plan

### Phase 0: Preparation (1-2 hours)

| Step | Action | Verification |
|------|--------|-------------|
| 0.1 | Upgrade Nuitka 4.0.8 → 4.1.1 | `nuitka --version` shows 4.1.1 |
| 0.2 | Merge main into release/v1 (pick up 3 missing commits) | `git log` shows merged commits, conflicts resolved |
| 0.3 | Trim to 3 tabs (if not already done) — remove Projects, Align, Coverage | App launches with 3 tabs from source |
| 0.4 | Delete dead build artifacts: `dist/`, `build/` | Recovered ~9 GB disk space |
| 0.5 | Add to .gitignore: `dist_nuitka/`, `dist_test/`, `bench_output/`, `dist/`, `build/` | `git status` clean of build artifacts |

### Phase 1: Fix runtime issues (2-3 hours)

| Step | Action | Verification |
|------|--------|-------------|
| 1.1 | Fix `_version` import (bare → explicit) | `python -c "from reconstruction_gui.reconstruction_zone import __version__"` |
| 1.2 | Fix subprocess env isolation (PATH sanitization) | Launch with miniconda on PATH, no DLL popups |
| 1.3 | Fix/verify setup wizard threading (queue pattern) | Run wizard, download YOLO26, no crash |
| 1.4 | Verify YAML patches match current torch source | grep confirms all 6 decorator patterns present |
| 1.5 | Decide OpenCV variant: keep opencv-contrib-python or switch to headless | Import test passes |
| 1.6 | Run full app from source with all fixes | 3 tabs work, wizard works, masking works |

### Phase 2: Smoke builds (1 hour)

| Step | Action | Verification |
|------|--------|-------------|
| 2.1 | Compile torch smoke test | Exe prints torch version, CUDA True, OK |
| 2.2 | Compile GUI import smoke test | Exe prints version, GUI_IMPORT_OK |
| 2.3 | If either fails → diagnose before proceeding | — |

### Phase 3: Full GUI build + testing (3-4 hours)

| Step | Action | Verification |
|------|--------|-------------|
| 3.1 | Run full Nuitka build (~30-45 min) | Exit code 0, exe exists, anti-bloat log shows 6 replacements |
| 3.2 | Post-build: copy CUDA DLLs for cv2 (if needed) | Exe launches without cv2 errors |
| 3.3 | Test: app launches, 3 tabs render | Visual check |
| 3.4 | Test: setup wizard appears, downloads work | YOLO, RF-DETR download, SAM3 gate works |
| 3.5 | Test: extract frames from video | Frames appear in output dir |
| 3.6 | Test: run masking on frames | Masks generated |
| 3.7 | Test: review tab loads masks | Thumbnails, overlay, editor |
| 3.8 | Test: close cleanly | No crash on exit |
| 3.9 | Fix failures, rebuild if needed (~15-20 min per rebuild with cache) | All tests pass |

### Phase 4: Distribution packaging (1-2 hours)

| Step | Action | Verification |
|------|--------|-------------|
| 4.1 | Assemble distributable zip | Contains: dist folder, QUICKSTART.txt, LICENSE.txt, THIRD_PARTY_LICENSES.txt |
| 4.2 | Size optimization (optional) | Exclude sklearn datasets, pandas templates, matplotlib extras |
| 4.3 | Test on clean machine (no Python installed) | Exe launches, wizard runs, workflow works |

### Phase 5: Documentation + ship (1-2 hours)

| Step | Action | Verification |
|------|--------|-------------|
| 5.1 | Write TESTING.md (prerequisites, SAM3 HF access instructions) | — |
| 5.2 | Final build with `--windows-console-mode=disable` | No console window on launch |
| 5.3 | Commit all packaging changes to release/v1 | Clean git status |
| 5.4 | Upload zip to Google Drive | Shareable link works |
| 5.5 | Share with tester | Tester confirms it works |

### Total estimated effort: 10-15 hours

This includes buffer for unexpected issues. The build itself is 30-45 minutes; the rest is fixing, testing, and iterating.

---

## 12. Build Commands Reference

### Torch smoke test

```powershell
& 'C:\Python314\python.exe' -m nuitka `
  --standalone --assume-yes-for-downloads `
  --include-package=torch --include-package=torchvision `
  --user-package-configuration-file=scripts/nuitka-torch-fix.yml `
  --module-parameter=torch-disable-jit=yes `
  --show-anti-bloat-changes `
  --show-source-changes=torch.library `
  --show-source-changes=torch._refs `
  --report=dist_test/nuitka-report.xml `
  --output-dir=dist_test `
  scripts/smoke_import_torch.py
```

### Full GUI build

```powershell
& 'C:\Python314\python.exe' -m nuitka `
  --standalone `
  --assume-yes-for-downloads `
  --windows-console-mode=attach `
  --windows-icon-from-ico=reconstruction-zone.ico `
  --enable-plugin=tk-inter `
  --enable-plugin=no-qt `
  --nofollow-import-to=triton `
  --nofollow-import-to=triton.* `
  --user-package-configuration-file=scripts/nuitka-torch-fix.yml `
  --include-package=prep360 `
  --include-package=reconstruction_gui `
  --include-package=torch `
  --include-package=torchvision `
  --include-package=customtkinter `
  --include-data-dir=reconstruction_gui/docs=reconstruction_gui/docs `
  --include-data-files=reconstruction-zone.ico=reconstruction-zone.ico `
  --include-data-files="C:/Users/alexm/ffmpeg/bin/ffmpeg.exe=ffmpeg/ffmpeg.exe" `
  --include-data-files="C:/Users/alexm/ffmpeg/bin/ffprobe.exe=ffmpeg/ffprobe.exe" `
  --output-dir=dist_nuitka `
  --output-filename=ReconstructionZone.exe `
  --module-parameter=torch-disable-jit=yes `
  --show-anti-bloat-changes `
  --report=dist_nuitka/nuitka-report.xml `
  reconstruction_gui/reconstruction_zone.py
```

### Final release build (console disabled)

Same as above but replace `--windows-console-mode=attach` with `--windows-console-mode=disable`.

---

## 13. Verification Checklist

Run after every build. Mark each item pass/fail.

```
BUILD
[ ] Nuitka exits with code 0
[ ] Anti-bloat log shows all 6 singledispatch replacements fired
[ ] No "experimental Python" warning (if Nuitka 4.1.1+ with 3.14 support)
[ ] Exe file exists and is ~900+ MB

LAUNCH
[ ] ReconstructionZone.exe launches (GUI window appears)
[ ] No native DLL popup dialogs (tiff.dll, etc.)
[ ] Console shows version number (if console=attach)
[ ] No import errors in console output

SETUP WIZARD
[ ] Wizard appears on first launch (no cached models)
[ ] YOLO26 downloads successfully (~6.5 MB)
[ ] RF-DETR downloads successfully (~1.2 GB)
[ ] SAM3 gate: token prompt appears, skip works
[ ] Wizard closes cleanly, app continues

EXTRACT TAB
[ ] Tab renders without errors
[ ] Can browse to a video file
[ ] Frame extraction runs and produces output

MASK TAB
[ ] Tab renders without errors
[ ] Can point to extracted frames folder
[ ] YOLO masking runs (GPU inference)
[ ] Masks are saved to output directory

REVIEW TAB
[ ] Tab renders without errors
[ ] Thumbnails load from mask output
[ ] OpenCV editor opens on click
[ ] Mask edits save correctly

EXIT
[ ] App closes cleanly (no crash, no hung process)
[ ] No error dialogs on close
```

---

## 14. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | YAML patches don't match after torch update | Medium | High | Verify with `--show-anti-bloat-changes`. Pin torch version for release builds. |
| 2 | cv2 DLL loading fails in compiled build | High | High | Switch to opencv-python-headless (no CUDA DLLs needed). |
| 3 | New singledispatch sites in future torch updates | Medium | High | Pin torch version. Test before upgrading. |
| 4 | Nuitka 4.1.1 introduces regressions | Low | High | Keep 4.0.8 as known-good fallback. |
| 5 | CTkToplevel destruction crash persists after fixing other issues | Medium | Medium | May need to restructure wizard as a frame within the main window instead of a toplevel. |
| 6 | SAM3 HuggingFace gate confuses users | High | Medium | Clear wizard instructions, skip button, app works without SAM3. |
| 7 | 8.5 GB distribution size | Low | Low | Normal for CUDA apps (ComfyUI is similar). Document clearly. |
| 8 | Miniconda on user's PATH causes DLL issues | High | Medium | subprocess_env.py sanitizes PATH for all external binary calls. |
| 9 | rfdetr import graph pulls in unexpected deps at Nuitka compile time | Medium | Medium | Use `--nofollow-import-to` for rfdetr's non-essential transitive deps. |
| 10 | Build cache invalidation after code changes | Low | Low | Nuitka handles incremental rebuilds. Delete `.build/` dir for clean rebuild if needed. |

---

## 15. Fallback: Python 3.12 + Nuitka

If the Python 3.14 path hits an unfixable blocker, Python 3.12 is available:

**Advantages:**
- No singledispatch YAML patches needed (3.12 uses `__annotations__`, preserved by Nuitka)
- Full Nuitka production support (no "experimental" warning)
- Proven stable combination in the broader community

**Disadvantages:**
- Torch 2.6 vs 2.10 — SAM3 may need compatibility testing
- Need to install: nuitka, customtkinter, rfdetr, py360convert on 3.12
- Separate build environment to maintain
- transformers 5.3 is installed but rfdetr pins `<5.0` — version conflict

**Steps to activate:**
```powershell
C:\Python312\python.exe -m pip install nuitka customtkinter py360convert
C:\Python312\python.exe -m pip install "transformers>=4.50,<5.0"  # downgrade for rfdetr
C:\Python312\python.exe -c "import torch, customtkinter; print('OK')"
# Then run smoke test with C:\Python312\python.exe -m nuitka (no YAML patches)
```

**Current 3.12 package state:**
- torch 2.6.0+cu124: installed, CUDA working
- ultralytics: installed
- opencv-contrib-python: installed
- transformers: 5.3.0 (**needs downgrade to <5.0 for rfdetr**)
- sam3: installed (editable from models/sam3)
- Missing: nuitka, customtkinter, rfdetr (has supervision), py360convert

---

## 16. Post-v1.0 Roadmap

### v1.1 — Feature expansion
- Restore Projects tab
- Restore Align tab (COLMAP/SphereSfM integration)
- Restore Coverage tab (gap detection, bridge frames)
- Inno Setup installer (proper install/uninstall)
- Fisheye-to-cubemap integration (planning doc exists: `2026-05-19-fisheye-to-cubemap-integration.md`)

### v1.2 — Distribution + marketing
- Gumroad listing (paid .exe: RF-DETR + SAM3 only, no YOLO, compiled binary)
- GitHub `main` stays as the public work-in-progress (no special release packaging)
- Landing page, demo videos, community presence

### v2.0 — Advanced features
- LiVOS/Cutie temporal mask propagation
- Alpha matting (ViTMatte)
- Learned shadow detectors
- CLI mode

---

## 17. Open Questions

These require decisions before starting execution:

1. **Model download strategy:** The paid .exe has no YOLO. RF-DETR (~1.2 GB) and SAM3 (~3.3 GB, gated) download on first launch. Should RF-DETR be bundled inside the .exe for a faster first experience, or is a first-launch download acceptable?

2. **OpenCV variant:** Keep `opencv-contrib-python` (has CUDA features, needs DLL copying) or switch to `opencv-python-headless` (no CUDA, no DLL issues, smaller)? The app uses OpenCV for CPU image processing only — all GPU inference is through PyTorch.

3. **Nuitka upgrade:** Go to 4.1.1 (better 3.14 support) or stay on 4.0.8 (proven to compile)?

4. **Feature freeze:** Should the fisheye-to-cubemap integration (`2026-05-19` plan) land before packaging work starts, or ship in v1.1?

5. **Testing strategy:** Friend-test only (ship to 1-2 people), or broader beta (needs more polish)?

6. **Pricing:** What price point for the Gumroad .exe? (kotohibi sells on BOOTH — research comparable pricing)

---

## 18. Appendix: Prior Documents Index

All documents related to packaging/release, with locations:

### On `release/v1` branch (not on main)

| Document | Description |
|----------|-------------|
| `planning/plans/exe-packaging-plan.md` | 371-line plan with Options A/B/C analysis, build commands, execution phases |
| `planning/plans/exe-packaging-checklist.md` | Detailed audit trail: changes made, opinions, decisions, corrections (26 entries) |
| `planning/plans/2026-05-11-build-issues-and-fixes.md` | 4 runtime issues found in Nuitka build with root causes and fixes |
| `planning/plans/2026-05-11-build-issues-remediation-instructions.md` | Conservative fix sequence for build issues |
| `planning/plans/2026-05-11-build-issues-remediation-instructions-report.md` | Implementation report for remediation |
| `planning/plans/2026-05-11-nuitka-build-diagnostics-handoff-report.md` | Codex→Claude handoff: diagnostic findings, failed builds, evidence-backed plan |
| `planning/plans/2026-05-11-phantom-code-incident-report.md` | Fabricated code in release trim commit |
| `planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md` | Cleanup and prevention |
| `planning/plans/2026-05-11-setup-wizard-plan.md` | Setup wizard design |
| `planning/plans/2026-05-12-lite-github-fallback-plan.md` | v1-lite design (install script approach) |
| `planning/plans/2026-05-12-*-result.md` (×8) | Build attempt results for phase 6b |
| `planning/plans/adjust-tab-plan.md` | Adjust tab design (deferred feature) |
| `AGENTS.md` | Agent configuration for the release branch |

### On main branch

| Document | Description |
|----------|-------------|
| `archive/docs/specs/2026-03-24-gumroad-release-design.md` | Product vision, 2-build architecture, licensing, feature inventory |
| `planning/plans/2026-05-13-lite-build-progress.md` | v1-lite build progress tracker (all phases complete) |
| `planning/plans/2026-05-22-exe-packaging-roadmap.md` | **This document** |

### Brainstorm artifacts

| File | Description |
|------|-------------|
| `.superpowers/brainstorm/6014-*/release-plan-overview.html` | Visual brainstorm output (2026-03-24) |
| `.superpowers/brainstorm/6298-*/release-plan-overview.html` | Visual brainstorm output (2026-03-24) |

---

## Corrections Log

Inherited from prior documents plus new findings:

| # | Claim | Correction | Source |
|---|-------|-----------|--------|
| 1 | `torch._refs` is lazy/not imported at startup | **WRONG.** `'torch._refs' in sys.modules` is True after `import torch`. | Verified via Python REPL |
| 2 | RF-DETR download is ~100 MB | **WRONG.** 129 MB seg weights + 1.1 GB DINOv2 backbone = ~1.2 GB total. | File size inspection |
| 3 | PyInstaller is a viable option for code protection | **WRONG.** PyInstaller bundles extractable .pyc files. Its own docs say so. | PyInstaller documentation |
| 4 | Full GUI Nuitka build never completed | **WRONG.** 4 builds compiled successfully (exit code 0, ~954 MB each). Runtime untested. | `dist_test/phase-6b-*` inspection, 2026-05-22 |
| 5 | Torch smoke exe status unknown | **WRONG.** It runs and works. Verified 2026-05-22. | Direct execution |
| 6 | Nuitka 4.1 fixes singledispatch | **NOT FOUND.** Changelog mentions decorator fixes but no singledispatch-specific fix. YAML patches still needed. | Nuitka 4.1 changelog review |
| 7 | cx_Freeze compiles to native code | **WRONG.** It bundles bytecode like PyInstaller. | cx_Freeze documentation review |
