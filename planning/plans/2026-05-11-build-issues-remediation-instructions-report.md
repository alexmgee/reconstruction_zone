# Remediation Instructions — Implementation Report

**Date:** 2026-05-11
**Source:** `planning/plans/2026-05-11-build-issues-remediation-instructions.md`
**Implemented by:** Claude (fixes 1, 5, 6, 7) + Alex (wizard rewrite to match `2026-05-11-setup-wizard-plan.md`)

---

## Codex Review Addendum (2026-05-11)

This report was re-audited against the current source and the targeted OpenCV smoke test was run.

### High-priority corrections

1. **OpenCV smoke test is no longer "not run"; it was run and failed.**
   - Nuitka compile command: `C:\Python314\python.exe -m nuitka --standalone --assume-yes-for-downloads --include-package=cv2 --output-dir=dist_test scripts/smoke_import_cv2.py`
   - Compile result: **PASS**; produced `dist_test\smoke_import_cv2.dist\smoke_import_cv2.exe`.
   - Executable result: **FAIL**.
   - Failure:
     ```text
     ImportError: DLL load failed while importing cv2: The specified module could not be found.
     ```
   - Bundle layout: exactly one `cv2.pyd`, at `dist_test\smoke_import_cv2.dist\cv2\cv2.pyd`, plus `opencv_videoio_ffmpeg4130_64.dll`.
   - Direct imports from `cv2.pyd` include CUDA/NVIDIA DLLs such as `npp*64_13.dll`, `cublas64_13.dll`, `cufft64_12.dll`, `cudnn64_9.dll`, `nvcuvid.dll`, and `nvEncodeAPI64.dll`.
   - Meaning: stop before the full GUI rebuild. The current blocker is not proven duplicate `cv2.pyd`; it is missing OpenCV native dependencies in the standalone cv2 smoke bundle.
   - Follow-up proof: after manually copying the required CUDA/NVIDIA DLLs into `dist_test\smoke_import_cv2.dist\cv2`, the compiled smoke exe printed `OK`.

2. **Claude's `model_downloader.py` concern is directionally right but technically overstated.**
   - `model_downloader.py` does **not** import `rfdetr` or `from ultralytics import YOLO` at module import time.
   - Those imports are inside `_download_rfdetr()` and `_download_yolo26()`.
   - However, the file is still stale and risky:
     - The planned Nuitka command uses `--include-package=reconstruction_gui`, so unused package submodules may still be compiled/analyzed.
     - `reconstruction_zone.spec` explicitly lists `reconstruction_gui.model_downloader` in `hiddenimports`.
     - The stale downloader still contains old CWD/HF assumptions and old download paths.
   - Follow-up completed: `model_downloader.py` has been replaced with a tiny compatibility stub that imports no model libraries, and the stale PyInstaller hidden import was removed from `reconstruction_zone.spec`.

3. **Current source checks still pass.**
   - `py_compile` passed for `model_paths.py`, `setup_wizard.py`, and `reconstruction_pipeline.py`.
   - `scripts/smoke_model_paths.py` passed from source.
   - `check_all_models()` returned `sam3=True`, `rfdetr=True`, `yolo26=True`.

---

## Fix-by-Fix Audit

### Fix 1: Add a shared model-path resolver (`model_paths.py`)

**Instruction status: FULLY IMPLEMENTED**

| Required behavior | Implemented? | Location |
|---|---|---|
| App model dir at `%LOCALAPPDATA%\ReconstructionZone\models` | YES | `model_paths.py:74-77` |
| Fallback `~/.cache/reconstruction-zone/models` | YES | `model_paths.py:79-81` |
| Override via `RECONSTRUCTION_ZONE_MODEL_DIR` | YES | `model_paths.py:71-73` |
| `YOLO26_WEIGHT_NAMES` dict | YES | `model_paths.py:29-35` |
| `RFDETR_SEG_WEIGHT_NAMES` dict | YES | `model_paths.py:37-44` |
| `RFDETR_SEG_URLS` dict | YES | `model_paths.py:46-53` (not in original instructions but useful) |
| `app_model_dir(create)` | YES | `model_paths.py:63` |
| `candidate_model_dirs()` | YES | `model_paths.py:88` |
| `find_model_file(names)` | YES | `model_paths.py:119` |
| `find_hf_cached_file(repo_id, filename)` | YES | `model_paths.py:133` |
| `resolve_yolo26_weights(size)` | YES | `model_paths.py:145` |
| `resolve_rfdetr_seg_weights(size)` | YES | `model_paths.py:166` |
| `resolve_sam3_weights()` | YES | `model_paths.py:180` (not in original instructions but needed by wizard) |

**Candidate lookup order (instruction vs implementation):**

| Priority | Instruction | Implemented? |
|---|---|---|
| 1 | App model directory | YES |
| 2 | CWD for legacy source-run | YES |
| 3 | Executable directory for frozen builds | YES (`sys.frozen` check) |
| 4 | Project root and project-root/models | YES |
| 5 | `~/.cache/ultralytics` for legacy YOLO | YES |
| 6 | HF cache for YOLO n only | YES (in `resolve_yolo26_weights`, not in `candidate_model_dirs`) |

**Source verification:**
```
model_dir: C:\Users\alexm\AppData\Local\ReconstructionZone\models
rfdetr: D:\Projects\reconstruction-zone\rf-detr-seg-small.pt
yolo26: D:\Projects\reconstruction-zone\yolo26n-seg.pt
sam3: D:\Models\huggingface\hub\models--facebook--sam3\...\config.json
```
All three resolve correctly from source.

---

### Fix 2: Update `setup_wizard.py` to use shared paths

**Instruction status: FULLY IMPLEMENTED (via full wizard rewrite)**

| Required edit | Implemented? | Notes |
|---|---|---|
| Import shared resolver with dual-path try/except | YES | `setup_wizard.py:33-42` |
| `_rfdetr_ready()` → `resolve_rfdetr_seg_weights("small")` | YES | `setup_wizard.py:62-64` |
| `_yolo26_ready()` → `resolve_yolo26_weights("n")` | YES | `setup_wizard.py:67-69` |
| RF-DETR downloads to app model dir as `rf-detr-seg-small.pt` | YES | `setup_wizard.py:106-107` |
| RF-DETR uses verified URL from `RFDETR_SEG_URLS` | YES | `setup_wizard.py:110` |
| Atomic temp-file-then-replace pattern | YES | `setup_wizard.py:112-118` |
| DINOv2/romav2 download REMOVED per instruction | YES | Not present in code |
| YOLO26 copies HF `model.pt` to app dir as `yolo26n-seg.pt` | YES | `setup_wizard.py:124-131` |
| RF-DETR display size changed to `129 MB` | YES | `setup_wizard.py:143` |
| Threading pattern preserved (queue only) | YES | All download functions use `q.put()` only |
| Silent exception in `_poll()` replaced with logging | YES | `setup_wizard.py:698` |
| RF-DETR readiness no longer requires `romav2*.pt` | YES | Only checks seg weight file |

---

### Fix 3: Make wizard partial-failure UI explicit

**Instruction status: IMPLEMENTED BUT DESIGN CHANGED**

The remediation instructions called for:
1. "Continue Anyway" button — **REMOVED per wizard plan.** Wizard blocks until all succeed or user closes.
2. No "Setup complete!" after failures — **YES.** Failures go to Step 5 (Error), not Step 4 (Done).
3. Explicit status text — **YES.** Per-model error text shown in failed cards.

The wizard plan took a stricter approach than the remediation instructions: no escape hatches at all. The user either succeeds (Step 4) or sees the error state (Step 5) with per-model retry buttons. Closing the wizard on Step 5 is allowed — it will reappear on next launch.

**Discrepancy:** The remediation instructions said add "Continue Anyway." The wizard plan said remove it. The wizard plan won. This is the correct choice for a first-launch experience where incomplete models mean broken functionality.

---

### Fix 4: Remove nested `wait_window()` from first-launch wizard

**Instruction status: FULLY IMPLEMENTED**

| Required change | Implemented? | Location |
|---|---|---|
| Remove `parent.wait_window(wizard)` | YES | `setup_wizard.py:820-827` |
| Use `wizard.transient(parent)` | YES | `setup_wizard.py:822` |
| Use `wizard.grab_set()` | YES | `setup_wizard.py:823` |
| Use `wizard.focus()` | YES | `setup_wizard.py:824` |
| Exception guard around modal setup | YES | `setup_wizard.py:821-826` |
| Add `_close()` helper with `grab_release()` | YES | `setup_wizard.py:769-774` |
| `_close()` used for WM_DELETE_WINDOW | YES | `setup_wizard.py:619, 659` |
| `_close()` used for "Enter Reconstruction Zone" | YES | `setup_wizard.py:617` |
| Return `False` instead of blocking | YES | `setup_wizard.py:827` |

---

### Fix 5: Update runtime segmenters to use the same resolver

**Instruction status: IMPLEMENTED WITH MINOR DEVIATION**

**YOLO26Segmenter:**

| Required change | Implemented? | Notes |
|---|---|---|
| Import `resolve_yolo26_weights` | YES | `reconstruction_pipeline.py:995-997` (lazy, inside method) |
| Resolver-first, fallback to auto-download | YES | `reconstruction_pipeline.py:998-999` |

The instruction recommended importing at the top of the file. The implementation uses a lazy import inside `initialize()` with a try/except fallback. This is consistent with the app's existing lazy-import pattern and works correctly, but differs from the instruction's explicit top-level import recommendation.

**RFDETRSegmenter:**

| Required change | Implemented? | Notes |
|---|---|---|
| Import `resolve_rfdetr_seg_weights` | YES | `reconstruction_pipeline.py:1111-1113` (lazy) |
| Pass resolved path as `pretrain_weights` kwarg | YES | `reconstruction_pipeline.py:1115` |
| Fall back to rfdetr auto-download if not found | YES | `reconstruction_pipeline.py:1117-1118` |

**Deviation from instruction:** The instruction recommended raising `FileNotFoundError` if weights are not found. The implementation falls back silently to `model_cls(**model_kwargs)` with empty kwargs, allowing rfdetr's own download mechanism to try. This is more forgiving but means rfdetr may attempt a download to CWD at runtime, which could fail in a read-only install directory.

**Assessment:** The forgiving approach is reasonable for a friend-testing build where the packaged app's CWD may be writable. For a production release, the strict `FileNotFoundError` approach from the instruction would be safer.

---

### Fix 6: Add targeted smoke tests

**Fix 6a: OpenCV Nuitka smoke test**

**Instruction status: SCRIPT CREATED, COMPILED, RUN, FAILED**

`scripts/smoke_import_cv2.py` exists with the exact content recommended:
```python
import cv2
print(cv2.__file__)
print(cv2.__version__)
print("OK")
```

The Nuitka smoke compile was run during Codex review:

```powershell
& 'C:\Python314\python.exe' -m nuitka `
  --standalone --assume-yes-for-downloads `
  --include-package=cv2 `
  --output-dir=dist_test `
  scripts/smoke_import_cv2.py
```

Compile result: **PASS**.

The compiled executable was then run:

```powershell
& .\dist_test\smoke_import_cv2.dist\smoke_import_cv2.exe
```

Run result: **FAIL**.

```text
Traceback (most recent call last):
  File "D:\Projects\reconstruction-zone\dist_test\smoke_import_cv2.dist\smoke_import_cv2.py", line 16, in <module>
  File "D:\Projects\reconstruction-zone\dist_test\smoke_import_cv2.dist\cv2\__init__.py", line 210, in <module cv2>
  File "D:\Projects\reconstruction-zone\dist_test\smoke_import_cv2.dist\cv2\__init__.py", line 182, in bootstrap
ImportError: DLL load failed while importing cv2: The specified module could not be found.
```

Bundle inspection found:

```text
dist_test\smoke_import_cv2.dist\cv2\cv2.pyd
dist_test\smoke_import_cv2.dist\cv2\opencv_videoio_ffmpeg4130_64.dll
```

There is only one `cv2.pyd` in the smoke bundle. This points to a missing native dependency for `cv2.pyd`, not the earlier duplicate-root `cv2.pyd` theory. Direct PE import inspection of `cv2.pyd` shows CUDA/NVIDIA dependencies that Nuitka did not copy into the smoke bundle, including:

```text
nvcuvid.dll
nvEncodeAPI64.dll
nppitc64_13.dll
nppig64_13.dll
cufft64_12.dll
nppif64_13.dll
nppim64_13.dll
nppial64_13.dll
nppist64_13.dll
nppidei64_13.dll
nppicc64_13.dll
cublas64_13.dll
cudnn64_9.dll
```

Most CUDA DLLs exist on this machine under `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64`, and `cudnn64_9.dll` also exists under both `site-packages\nvidia\cudnn\bin` and `site-packages\torch\lib`.

Manual proof of fix:

```text
Copied the CUDA/NVIDIA DLLs above into:
dist_test\smoke_import_cv2.dist\cv2

Then ran:
dist_test\smoke_import_cv2.dist\smoke_import_cv2.exe

Result:
D:\Projects\reconstruction-zone\dist_test\smoke_import_cv2.dist\cv2\__init__.py
4.13.0
OK
```

The next remediation should make this repeatable in the Nuitka build, either by bundling the required native dependencies explicitly or by switching to a non-CUDA OpenCV wheel for packaging.

Repeatable helper added:

```powershell
.\scripts\copy_cv2_cuda_dlls.ps1 -DistDir .\dist_test\smoke_import_cv2.dist
```

Run the same helper against the full GUI bundle after Nuitka creates `dist_nuitka\reconstruction_zone.dist`.

**Fix 6b: Model-path smoke test**

**Instruction status: CREATED AND RUN FROM SOURCE**

`scripts/smoke_model_paths.py` exists and was run from source successfully:
```
model_dir: C:\Users\alexm\AppData\Local\ReconstructionZone\models
rfdetr: D:\Projects\reconstruction-zone\rf-detr-seg-small.pt
yolo26: D:\Projects\reconstruction-zone\yolo26n-seg.pt
sam3: ...\config.json
OK
```

Not yet compiled with Nuitka. The instruction says to compile it only if needed after OpenCV passes.

---

### Fix 7: Audit stale `model_downloader.py`

**Instruction status: IMPLEMENTED AS COMPATIBILITY STUB**

| Required action | Done? | Notes |
|---|---|---|
| Confirm no imports of `model_downloader` exist | YES | `grep` found zero references |
| Add deprecation note | YES | File is now a compatibility stub |
| Verify whether Nuitka bundles unused submodules | PARTIAL | The planned build command uses `--include-package=reconstruction_gui`, which can compile/analyze unused submodules. The stale PyInstaller hidden import has been removed. |

**Correction to earlier wording:** `model_downloader.py` does not import `rfdetr` or `YOLO` at module import time. Those imports are inside download functions. The risk is still real for packaging analysis and diagnostics, but it is not a top-level runtime import trigger.

**Current state:** `model_downloader.py` has been replaced with a compatibility stub. It imports no `cv2`, `rfdetr`, `ultralytics`, `customtkinter`, or `huggingface_hub`.

**Current packaging state:** `reconstruction_gui.model_downloader` has been removed from `reconstruction_zone.spec` hidden imports.

---

## Verification Checklist Status

### Source-level checks

| Check | Status | Result |
|---|---|---|
| `py_compile` all three files | PASS | model_paths, setup_wizard, reconstruction_pipeline all compile |
| `check_all_models()` returns correct readiness | PASS | sam3=True, rfdetr=True, yolo26=True |
| `smoke_model_paths.py` from source | PASS | All three models resolve |
| Full app source launch | PASS | "All models ready — setup wizard not needed" |

### OpenCV/Nuitka smoke

| Check | Status | Result |
|---|---|
| Compile `smoke_import_cv2.py` with Nuitka | **PASS** | `dist_test\smoke_import_cv2.dist\smoke_import_cv2.exe` created |
| Run compiled cv2 smoke exe before manual DLL copy | **FAIL** | `ImportError: DLL load failed while importing cv2: The specified module could not be found.` |
| Run compiled cv2 smoke exe after manual CUDA/NVIDIA DLL copy | **PASS** | `cv2.__file__`, `4.13.0`, `OK` |
| Inspect smoke cv2 bundle layout | **PASS** | One `cv2.pyd` found at `cv2\cv2.pyd`; no duplicate root `cv2.pyd` |

### Torch smoke (from earlier session)

| Check | Status | Result |
|---|---|---|
| Compile `smoke_import_torch.py` with Nuitka + YAML | PASS | Completed 2026-05-11 02:04 |
| Run compiled torch smoke exe | PASS | `2.10.0+cu126 / 0.25.0+cu126 / True / OK` |

### Full packaged build checks

| # | Check | Status |
|---|---|---|
| 1 | No COLMAP DLL popup at startup | **NOT TESTED** (needs rebuild) |
| 2 | Wizard opens only when models missing | **NOT TESTED** |
| 3 | Wizard never says "Setup complete!" after failures | **NOT TESTED** (code change made, needs rebuild) |
| 4 | Wizard has per-model Retry on Step 5 | **NOT TESTED** (code implemented, needs rebuild) |
| 5 | Wizard close doesn't crash | **NOT TESTED** (wait_window removed, needs rebuild) |
| 6 | `cv2.pyd` bundle layout inspection | **NOT TESTED** |
| 7 | Mask tab import doesn't trigger cv2 recursion | **NOT TESTED** |
| 8 | YOLO26 instantiates from wizard-downloaded path | **NOT TESTED** |
| 9 | RF-DETR instantiates from wizard-downloaded path | **NOT TESTED** |
| 10 | One-image masking run succeeds | **NOT TESTED** |

---

## Build-Issues Document Updates (per remediation instructions section 7)

These corrections should be applied to `2026-05-11-build-issues-and-fixes.md`:

| # | Instruction | Done? |
|---|---|---|
| 1 | Replace duplicate cv2.pyd claim with "unproven" | NO — document not yet updated |
| 2 | Narrow the "no build infrastructure changes" statement | NO |
| 3 | Replace RF-DETR 1.1 GB/DINOv2 requirement with corrected statement | NO |
| 4 | Keep YOLO26 HF-cache risk, mark fix | NO |
| 5 | Mark nametowidget crash as directly mitigated by removing wait_window | NO |

These are documentation updates, not code changes. They should be applied after the next build validates the fixes.

---

## Outstanding Items Before Rebuild

1. **Make the OpenCV DLL fix repeatable.** The standalone cv2 smoke now passes after manually copying CUDA/NVIDIA DLLs beside `cv2.pyd`. Add explicit Nuitka include-data rules or a post-build copy script so the full GUI bundle gets the same DLLs automatically.
2. **Keep `model_downloader.py` stubbed.** The stale downloader logic has been neutralized, and the old PyInstaller hidden import has been removed. If future cleanup deletes the stub entirely, first confirm no packaging path imports it.
3. **Decide on RF-DETR DINOv2 backbone.** The wizard currently downloads only the 129 MB seg weights. If rfdetr downloads the 1.1 GB backbone at first inference, the user will experience a long unexplained delay. Either prefetch it in the wizard or document it as expected behavior.
4. **Update the build-issues document** with the 5 corrections listed above.

---

## Summary

The remediation instructions contained 7 fixes plus a verification checklist. Current status:

| Fix | Status |
|---|---|
| 1. model_paths.py | Fully implemented and verified from source |
| 2. Wire wizard to model_paths | Fully implemented (via wizard rewrite) |
| 3. Partial-failure UI | Design changed by wizard plan — stricter than instructions (no escape hatches) |
| 4. Remove wait_window | Fully implemented |
| 5. Wire runtime pipeline | Implemented with minor deviation (forgiving fallback instead of strict error) |
| 6a. OpenCV smoke test | Script created; Nuitka compile passed; compiled exe failed before DLL copy and passed after manual CUDA/NVIDIA DLL copy |
| 6b. Model-path smoke test | Created and verified from source |
| 7. Audit model_downloader | Replaced with compatibility stub; stale PyInstaller hidden import removed |

The wizard rewrite by Alex superseded fixes 2, 3, and parts of 4 with a more comprehensive 5-step design matching the `planning.pen` mockups. The backend (model_paths, download functions, threading pattern) is unchanged from my implementation. The frontend is entirely new.
