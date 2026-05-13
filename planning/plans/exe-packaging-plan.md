# Reconstruction Zone v1 — .exe Packaging Plan (revised)

**Date:** 2026-05-10
**Revised:** 2026-05-10 (incorporating review from `2026-05-10-exe-packaging-plan-review.md`)
**Branch:** `release/v1`
**Goal:** Package Masking Studio as a standalone Windows .exe installer for friend-testing, deliverable via Google Drive.

---

## 1. Build Audit — Current State

### 1.1 Development Environment

| Component | Value | Verified |
|---|---|---|
| Python | 3.14.0 (CPython Official, MSC v.1944, 64-bit) at `C:\Python314` | `python --version` |
| PyTorch | 2.10.0+cu126 (CUDA 12.6) | `torch.__version__` |
| TorchVision | 0.25.0+cu126 | `torchvision.__version__` |
| GPU | NVIDIA GeForce RTX 3090 Ti | `torch.cuda.get_device_name(0)` |
| OS | Windows 11 Pro 10.0.26200 | `platform.platform()` |
| PyInstaller | 6.20.0 (installed on 3.14 and 3.12) | `PyInstaller --version` |
| Nuitka | 4.0.8 (Python 3.14 only, experimental support) | `nuitka --version` |
| Python 3.12 | 3.12.6 at `C:\Python312` with torch 2.6.0+cu124 | Available as fallback |
| Miniconda | 3.12.12 at `C:\Users\alexm\miniconda3` (CPU torch 2.8.0) | On PATH — can contaminate builds |

**IMPORTANT:** Bare `python` resolves to `C:\Python314` in bash, but may resolve to miniconda in cmd.exe or PowerShell depending on PATH ordering. All build commands MUST use explicit interpreter paths: `C:\Python314\python.exe` for Option B, `.venv312\Scripts\python.exe` for Option A.

No venv in use — global pip install. The repo has a `pyproject.toml` with package metadata and console/GUI script entries, but no built wheel or installed-app workflow is used. The app is launched via direct script execution.

### 1.2 Release Branch (`release/v1`)

Three commits on top of `main` (`9418a0e`):
```
00c8b73 feat: first-launch model downloader, bundle ffmpeg, fix prep360 init for v1
c945879 fix: remove deleted module references from prep360 init and spec file
2741e55 release(v1): trim experimental features for v1 release
```

`release/v1` is 2 commits ahead of `origin/release/v1` (only `2741e55` has been pushed).

Uncommitted changes (4 tracked files):
| File | Change | Status |
|---|---|---|
| `reconstruction_gui/model_downloader.py` | Rewritten with SAM3/RF-DETR/YOLO26 check+download | Threading broken — uses `_root.after()` from background thread |
| `reconstruction_gui/reconstruction_zone.py` | Model check completely removed (no import, no call) | Working but no first-launch UX |
| `reconstruction_zone.spec` | PyInstaller experiments (pathex, module_collection_mode, excludes) | Stale — only relevant if we use Option A |
| `scripts/pyinstaller_runtime_hook.py` | ffmpeg PATH injection + torch pre-import | ffmpeg PATH logic reusable |

Untracked (NOT gitignored — need explicit rules added to `.gitignore`):
- `dist_nuitka/` (17 GB) — Nuitka build artifacts
- `bench_output/` (7 GB) — benchmark images
- `planning/plans/exe-packaging-plan.md` — this document

**Note:** `git check-ignore -v` reports these match `.gitignore` line 62 (a blank line), but `git status` shows them as `??` untracked. The `.gitignore` does NOT contain rules for these directories. They would be swept up by `git add .` and must have explicit ignore rules added before any bulk staging.

### 1.3 Build Artifacts on Disk
| Directory | Size | Status |
|---|---|---|
| `dist/` (PyInstaller) | 8.7 GB | **Dead.** Delete. |
| `dist_nuitka/reconstruction_zone.dist/` | 8.5 GB (exe: 917 MB) | **Partial.** GUI launches but torch crashes. |
| `dist_nuitka/reconstruction_zone.build/` | ~8.5 GB | Nuitka C compilation cache. Can speed up rebuilds. |
| `build/` (PyInstaller) | 248 MB | **Stale.** Delete. |

### 1.4 What Works in the Current Nuitka Build
- `ReconstructionZone.exe` (917 MB) launches
- All 5 tabs render: Projects, Extract, Mask, Review, Align
- COLMAP detected at `C:\Users\alexm\COLMAP\COLMAP-4.0.3-x64-windows-cuda\bin\colmap.exe`
- SphereSfM detected at `D:\Tools\SphereSfM\bin\colmap.exe`
- ffmpeg/ffprobe bundled at `reconstruction_zone.dist/ffmpeg/` within the Nuitka output
- CustomTkinter themes and assets included
- App icon set

### 1.5 What Fails in the Current Nuitka Build
| Feature | Error | Severity |
|---|---|---|
| torch import | `TypeError: Invalid first argument to register()` in `torch/library.py:556` | **BLOCKER** — all ML features dead |
| prep360 core | `Error: prep360 core not available` | Cascading from torch failure |
| Model downloader | Disabled (code removed) | **BLOCKER** — no first-launch model UX |
| COLMAP validation | Loads miniconda `tiff.dll` → native "Ordinal 127" popup | **BLOCKER** — DLL contamination at startup |
| Console logging | No output from terminal | Built with `--windows-console-mode=disable` |

---

## 2. Problem Analysis

### 2.1 PROBLEM: PyInstaller Circular Import (ABANDONED)

**Tool:** PyInstaller 6.20.0 + Python 3.14 + PyTorch 2.10
**Error:** `AttributeError: partially initialized module 'torch' has no attribute 'autograd'`

PyInstaller's custom module loader (`pyimod02_importers`) triggers `torch.nested` before `torch.autograd` is set as an attribute during `torch.__init__` execution. CPython's native loader handles this correctly. Six different fix attempts all failed — each fix revealed the next circular dependency deeper in torch's import graph. **Not worth further time on Python 3.14 + torch 2.10 for this release.**

PyInstaller remains viable with Python 3.12 + PyTorch 2.6 (Option A), where the torch import graph is simpler.

### 2.2 PROBLEM: Nuitka singledispatch Annotation Stripping (ACTIVE BLOCKER)

**Tool:** Nuitka 4.0.8 + Python 3.14 + PyTorch 2.10
**Error:** `TypeError: Invalid first argument to register()`
**Root cause:** Nuitka compiles Python functions to C code. Compiled functions no longer present annotation metadata in the form that `functools.singledispatch.register()` expects (on Python 3.14 this involves `__annotate__` per PEP 649, not just `__annotations__`). The practical result: `register()` receives a compiled function, cannot determine the dispatch type, and raises `TypeError`.

**Affected locations (all loaded during normal `import torch`):**

| Module | Line | Decorator | Dispatch Type | Loaded at import? |
|---|---|---|---|---|
| `torch.library` | 556 | `@define.register` | `Library` | **YES** |
| `torch.library` | 670 | `@impl.register` | `Library` | **YES** |
| `torch._refs` | 2131 | `@_to_dispatch.register` | `torch.device` | **YES** |
| `torch._refs` | 2149 | `@_to_dispatch.register` | `str` | **YES** |
| `torch._refs` | 2167 | `@_to_dispatch.register` | `torch.dtype` | **YES** |
| `torch._refs` | 2183 | `@_to_dispatch.register` | `Tensor` | **YES** |

**CORRECTION from review:** The original plan claimed `torch._refs` was lazy/not imported during startup. This was WRONG. Verified: `'torch._refs' in sys.modules` returns `True` immediately after `import torch`. All 6 decorators must be fixed, not just 2.

**`torch._subclasses.fake_tensor`** has `@atexit.register` but this is stdlib `atexit`, not singledispatch. Safe.

### 2.3 PROBLEM: COLMAP DLL Contamination at Startup (BLOCKER — NEW)

**Discovered by:** Review document, independently reproduced.
**Error:** Native Windows dialog: `colmap.exe - Ordinal Not Found: The ordinal 127 could not be located in the dynamic link library C:\Users\alexm\miniconda3\Library\bin\tiff.dll`

**Root cause:** `_check_external_tools()` runs during `__init__()`, calling `_refresh_alignment_binary_cache()` → `_validate_alignment_binary_candidate()` → `ColmapRunner.validate_binary()` → `subprocess.run([colmap.exe, "help"])`. No `env` or `cwd` is set. Miniconda's `tiff.dll` is on PATH before COLMAP's own `tiff.dll`, and COLMAP loads the wrong one.

**Fix:** Sanitize subprocess environment for external binary calls. Set `cwd` to the binary's parent directory. Prepend binary directory to PATH. Remove/demote miniconda directories.

### 2.4 PROBLEM: Model Downloader Threading (BLOCKER)

**Error:** `RuntimeError: main thread is not in main loop`
**Root cause:** Background download thread calls `self._root.after()` which internally calls tkinter's `_register()` — this checks the calling thread is the main thread.

**Fix:** Queue-based pattern. Background thread puts messages on `queue.Queue()`. Main thread polls with `after(100, _poll)`. No tkinter calls from any non-main thread.

**Additional concern:** SAM3 is a **gated model** on HuggingFace (`model_info('facebook/sam3').gated == 'manual'`). Users must request and be approved for access before downloading. The downloader must handle this gracefully (clear error message, not a cryptic HTTP 403).

### 2.5 PROBLEM: Console Output Suppression (COSMETIC)

Built with `--windows-console-mode=disable`. Fix: use `--windows-console-mode=attach` for testing builds. Also: `app_infra.py` replaces stdout/stderr with queue/file log writers after startup, and `crash.log` is always written. So the early-startup devnull redirect is not the whole story — the app does have file-based logging after initialization.

---

## 3. Response Plan

### 3.0 Options Summary (REVISED risk levels)

| Option | Compiler | Python | PyTorch | Risk | Notes |
|---|---|---|---|---|---|
| **B** | Nuitka 4.0.8 | 3.14 | 2.10+cu126 | **Medium-High** | Must fix 6 singledispatch decorators across 2 modules. `replacements_plain` for library, `replacements_re` for _refs. |
| **A** | PyInstaller 6.20 | 3.12 | 2.6+cu124 | **Medium** | Proven toolchain combo, but deps not verified in clean venv. rfdetr not installed on 3.12. SAM3 missing einops. |

**Option C (patch stdlib functools) is ELIMINATED.** Patching a global stdlib file is too fragile and contaminates the system Python.

**Strategy: Try B with a minimal `import torch` compile first (5 min, not 30 min). If it fails, fall back to A after proving deps in a clean venv.**

### 3.1 Option B — Nuitka YAML Source Rewriting

**`torch.library` fix (2 decorators):** Use `replacements_plain` — simple `str.replace`:
```yaml
- module-name: 'torch.library'
  anti-bloat:
    - description: 'fix singledispatch for Nuitka compiled functions'
      replacements_plain:
        '@define.register': '@define.register(Library)'
        '@impl.register': '@impl.register(Library)'
```

**`torch._refs` fix (4 decorators):** Use `replacements_re` — regex `re.sub` with multi-line context:
```yaml
- module-name: 'torch._refs'
  anti-bloat:
    - description: 'fix singledispatch for Nuitka compiled functions'
      replacements_re:
        '@_to_dispatch\.register\ndef _to_device\(\n    device: torch\.device,':
          '@_to_dispatch.register(torch.device)\ndef _to_device(\n    device,'
        '@_to_dispatch\.register\ndef _to_device_str\(\n    device: str,':
          '@_to_dispatch.register(str)\ndef _to_device_str(\n    device,'
        '@_to_dispatch\.register\ndef _to_dtype\(\n    dtype: torch\.dtype,':
          '@_to_dispatch.register(torch.dtype)\ndef _to_dtype(\n    dtype,'
        '@_to_dispatch\.register\ndef _to_other\(\n    other: Tensor,':
          '@_to_dispatch.register(Tensor)\ndef _to_other(\n    other,'
```

**Step 1: Minimal compile test (5 min)**

Use the checked-in smoke script `scripts/smoke_import_torch.py`:
```python
# scripts/smoke_import_torch.py
import torch
import torch._refs
import torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())
print("OK")
```

Compile and run (PowerShell):
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

& .\dist_test\smoke_import_torch.dist\smoke_import_torch.exe
```

**Pass criteria:**
- Anti-bloat log shows all 6 source replacements fired
- Compiled exe prints torch version, torchvision version, CUDA status, and "OK"

**Fail criteria → diagnose or fall back to Option A:**
- Anti-bloat log shows fewer than 6 replacements (regex mismatch)
- Exe crashes with new singledispatch error (7th decorator site not covered)
- Exe crashes with different error entirely

**Step 2: Full GUI build (~30 min)**
```powershell
& 'C:\Python314\python.exe' -m nuitka `
  --standalone --assume-yes-for-downloads `
  --windows-console-mode=attach `
  --windows-icon-from-ico=reconstruction-zone.ico `
  --enable-plugin=tk-inter `
  --user-package-configuration-file=scripts/nuitka-torch-fix.yml `
  --include-package=prep360 --include-package=reconstruction_gui `
  --include-package=torch --include-package=torchvision `
  --include-package=customtkinter `
  --include-data-dir=reconstruction_gui/docs=reconstruction_gui/docs `
  --include-data-files=reconstruction-zone.ico=reconstruction-zone.ico `
  --include-data-files="C:/Users/alexm/ffmpeg/bin/ffmpeg.exe=ffmpeg/ffmpeg.exe" `
  --include-data-files="C:/Users/alexm/ffmpeg/bin/ffprobe.exe=ffmpeg/ffprobe.exe" `
  --output-dir=dist_nuitka --output-filename=ReconstructionZone.exe `
  --module-parameter=torch-disable-jit=yes `
  reconstruction_gui/reconstruction_zone.py
```

### 3.2 Option A — Python 3.12 Venv + PyInstaller (FALLBACK)

**Pre-requisites before building:**
1. Create `.venv312` from `C:\Python312\python.exe`
2. Install ALL deps including `einops`, `rfdetr`, `ultralytics`
3. Verify source-run imports AND one inference path:
   - `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
   - `python -c "from sam3.model_builder import build_sam3_image_model"`
   - `python -c "import rfdetr; print('rfdetr OK')"`
   - `python -c "from ultralytics import YOLO; print('yolo OK')"`
   - `python reconstruction_gui/reconstruction_zone.py` (from source)
4. Only then run PyInstaller

---

## 4. App-Level Fixes Required Before Packaging

These must be done BEFORE any packaging attempt. The review correctly identified that fixing app-level blockers after packaging wastes 30-minute build cycles.

### 4.1 Fix Model Downloader (queue-based threading)
Queue pattern: worker thread → `queue.Queue()` → main thread `after(100, _poll)`. No tkinter calls from non-main thread. Handle SAM3 gated-model errors gracefully.

### 4.2 Isolate External Binary Subprocess Environment
For COLMAP, SphereSfM, ffmpeg, exiftool — create a dedicated helper (e.g., `_isolated_subprocess_env(binary_path)`) separate from the existing `_subprocess_window_kwargs()` (which only handles Windows console flags). The helper should:
- Set `cwd` to the binary's parent directory
- Copy `os.environ` and prepend binary directory to `PATH`
- Remove/demote miniconda and conda directories from `PATH`
- Preserve non-PATH environment variables
- Merge with `_subprocess_window_kwargs()` on Windows

Apply to all subprocess call sites, not just COLMAP validation:
- `ColmapRunner.validate_binary()` and `_run_cli_command()`
- `prep360/core/analyzer.py`, `extractor.py`, `sharpest_extractor.py` (ffmpeg/ffprobe)
- `prep360/core/geotagger.py` (exiftool)
- `prep360/core/osv.py` (ffmpeg/ffprobe)
- `reconstruction_gui/tabs/alignment_tab.py` (COLMAP)

### 4.3 Re-enable Model Check
After 4.1 is fixed, re-add the deferred model check to `reconstruction_zone.py`:
```python
self.after(500, self._check_model_weights)
```

---

## 5. External Dependencies

| Dependency | Required For | Size | Bundled? | Notes |
|---|---|---|---|---|
| NVIDIA GPU drivers | CUDA inference | varies | No | User responsibility |
| COLMAP 3.x/4.x | Align tab | ~200 MB | No | Detected from PATH |
| SphereSfM (optional) | Align tab (360° ERP) | ~200 MB | No | Detected from PATH |
| exiftool (optional) | SRT geotagging | ~25 MB | No | Checked at startup, missing from prior doc |
| SAM3 weights | Masking (SAM3 model) | 3.3 GB | No | First-use download. **GATED MODEL** — requires HuggingFace access approval |
| RF-DETR seg weights | Masking (RF-DETR) | 129 MB | No | First-use download from Google Storage |
| DINOv2 backbone | RF-DETR backbone | 1.1 GB | No | First-use download via torch.hub as `romav2.pt` |
| YOLO26-n-seg weights | Masking (YOLO) | 6.5 MB | No | First-use download from HuggingFace |
| ffmpeg + ffprobe | Video extraction | 250 MB total | **YES** | Bundled in packaged app's `ffmpeg/` folder |

**Total first-use model downloads: ~4.5 GB** (requires internet + SAM3 HF approval)

### License Considerations
- Project: GPL-3.0
- Ultralytics/YOLO26: AGPL-3.0 — included for friend-testing, must be stripped for commercial release
- SAM3: Meta license (research use)
- RF-DETR: Apache 2.0
- ffmpeg: LGPL/GPL depending on build configuration
- PyTorch: BSD-3-Clause
- This is a **private friend-test build**, not a public release. License compliance is not blocking but should be documented before any wider distribution.

---

## 6. Execution Order (REVISED)

```
Phase 0: Stabilize workspace
  ├── Work in D:\Projects\reconstruction-zone on release/v1
  ├── Preserve current dirty diff
  ├── Delete dist/ and build/ (dead PyInstaller artifacts)

Phase 1: Fix app-level blockers BEFORE packaging
  ├── 1a. Add subprocess env isolation for COLMAP/SphereSfM/ffmpeg/exiftool
  │       (fixes DLL contamination popup — blocks all other testing)
  ├── 1b. Rewrite model_downloader.py with queue pattern
  ├── 1c. Re-enable model check in reconstruction_zone.py
  ├── 1d. Decide YOLO inclusion (yes for friend-test)
  ├── 1e. Add dist_nuitka/ and bench_output/ to .gitignore
  ├── 1f. Test from source: model check + download + one masking inference
  ├── Commit these fixes

Phase 2: Prove torch compiles (minimal test, not full build)
  ├── 2a. Create scripts/nuitka-torch-fix.yml (library + _refs)
  ├── 2b. Create scripts/smoke_import_torch.py
  ├── 2c. Compile with Nuitka + YAML + --show-anti-bloat-changes (~5 min)
  ├── 2d. Verify anti-bloat log shows all 6 replacements fired
  ├── 2e. Run the compiled exe
  │   ├── PASS (prints torch version + "OK") → proceed to Phase 3
  │   └── FAIL → diagnose or fall back to Option A
  │       ├── Create .venv312
  │       ├── Install all deps
  │       ├── Verify source-run + inference
  │       └── Build with PyInstaller

Phase 3: Full GUI build (~30 min)
  ├── Build with Nuitka + YAML fix + console=attach
  ├── Launch and verify:
  │   ├── Console output visible
  │   ├── No "PyTorch not installed" warning
  │   ├── No native DLL popups at startup
  │   ├── All 5 tabs load
  │   ├── Model download dialog appears (if models missing)
  │   ├── Extract tab works with video
  │   ├── Mask tab works (triggers model download + GPU inference)
  │   ├── Review tab loads mask output
  │   ├── Align tab detects COLMAP

Phase 4: Installer + distribution
  ├── Install Inno Setup 6
  ├── Create scripts/installer.iss (with AppId, install-size, per-user install)
  ├── Compile installer
  ├── Test install → launch → uninstall cycle
  ├── Test launch with hostile PATH (miniconda first) → no native DLL popups

Phase 5: Ship
  ├── Write TESTING.md
  ├── Build final installer (console=disable for release)
  ├── Commit all packaging changes to release/v1
  ├── Push to GitHub
  ├── Upload installer to Google Drive
  ├── Share link + TESTING.md with tester
```
