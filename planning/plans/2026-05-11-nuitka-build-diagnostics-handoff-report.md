# Nuitka Build Diagnostics and Handoff Report

Date: 2026-05-11  
Project: Reconstruction Zone  
Prepared by: Codex  
Build executor going forward: Claude

## Purpose

This report captures the diagnostic work, failed build attempts, source changes,
and current evidence-backed plan for the Reconstruction Zone standalone Windows
build. It is intended as a handoff document so the next build attempts are
deliberate, bounded, and based on confirmed causes instead of broad trial and
error.

Per user direction, Codex should stay in a diagnostics/planning role from this
point forward. Claude should run future build attempts.

## Current Position

The full app executable has not been successfully produced in the current
session. The work did, however, uncover and fix several concrete blockers:

- The OpenCV failure is not currently proven to be duplicate `cv2.pyd`; the
  standalone OpenCV smoke bundle had exactly one `cv2.pyd`.
- The OpenCV standalone failure was reproduced and fixed by copying missing
  CUDA/NVIDIA DLLs beside the bundled `cv2` package.
- `model_downloader.py` was made safe as a compatibility stub so stale packaging
  inputs do not pull obsolete model imports into analysis.
- The ffmpeg include path was corrected from a non-existent source directory to
  explicit `ffmpeg.exe` and `ffprobe.exe` file includes.
- The latest Nuitka stall was traced to an eager import path that pulls
  `ultralytics` and then `transformers` during GUI shell import.

The key unresolved blocker is now source-level import hygiene before packaging:
the GUI shell import path is too broad, and it lets optional ML dependencies
enter Nuitka analysis before the user has triggered masking/model work.

## High-Confidence Root Cause of the Latest Stall

The latest small GUI import smoke compile stalled in Nuitka's Transformers
plugin. The active child process was running a command equivalent to:

```text
from transformers.utils.import_utils import define_import_structure
define_import_structure('...site-packages\\transformers\\models\\deprecated\\mega\\__init__.py')
```

That means Nuitka was not merely compiling the app shell. It was walking the
Transformers model package tree.

The source-level import path that explains this is:

```text
reconstruction_gui/reconstruction_zone.py
  -> from tabs.source_tab import build_source_tab
reconstruction_gui/tabs/source_tab.py
  -> from prep360.core import (...)
prep360/core/__init__.py
  -> from .segmenter import Segmenter, SegmentConfig, COCO_CLASSES, CLASS_PRESETS
prep360/core/segmenter.py
  -> from ultralytics import YOLO
ultralytics
  -> transformers
Nuitka transformers plugin
  -> attempts to enumerate/analyze many transformers.models.* packages
```

Source references:

- `reconstruction_gui/reconstruction_zone.py:87` imports `build_source_tab`.
- `reconstruction_gui/tabs/source_tab.py:39` imports from `prep360.core`.
- `prep360/core/__init__.py:13` imports `.segmenter`.
- `prep360/core/segmenter.py:27` imports `YOLO` from `ultralytics`.
- `reconstruction_gui/reconstruction_pipeline.py:70`, `:80`, and `:87` also
  import FastSAM, YOLO, and RF-DETR, but those are inside the lazy
  `_import_pipeline()` path and are not the first suspected GUI-shell leak.

This is the main reason the "small" GUI smoke was not actually small.

## Additional Source Bug Found During Diagnostics

`source_tab.py` imports `AdjustmentRecipe` from `prep360.core`:

```text
reconstruction_gui/tabs/source_tab.py:43: AdjustmentRecipe
```

But a repo-wide search found no `AdjustmentRecipe` definition. This means the
`from prep360.core import (...)` block is likely failing with `ImportError` after
it has already imported `prep360.core` and paid the cost of eager imports.

Impact:

- `HAS_PREP360` may be set to `False` even when the underlying prep360 modules
  are present.
- Nuitka still sees the broad import graph before the optional import fails.
- The Source tab may have disabled behavior unrelated to packaging.

This should be fixed before the next serious build attempt. It is a source bug,
not merely a packaging flag problem.

## Work Already Completed

### Setup Wizard Plan Review

Reviewed and improved:

```text
planning/plans/2026-05-11-setup-wizard-plan.md
```

Notable corrections:

- Added verified current-state notes.
- Corrected model path/download details to use `model_paths.py` and
  `app_model_dir(create=True)`.
- Clarified RF-DETR DINOv2 mismatch risks.
- Clarified YOLO Hugging Face endpoint behavior.
- Clarified queue polling in error state.
- Clarified readiness and `ready_to_install` behavior.

### Build Issues Remediation Report Review

Reviewed and improved:

```text
planning/plans/2026-05-11-build-issues-remediation-instructions-report.md
```

The main correction was that the OpenCV duplicate-`cv2.pyd` theory was not
confirmed by the current smoke bundle. The actual smoke failure was missing
native CUDA/NVIDIA dependencies.

### OpenCV Smoke Test

Command shape used:

```powershell
C:\Python314\python.exe -m nuitka --standalone --assume-yes-for-downloads --include-package=cv2 --output-dir=dist_test scripts/smoke_import_cv2.py
```

Result:

- Compile: PASS.
- Initial exe run: FAIL.
- Failure text:

```text
ImportError: DLL load failed while importing cv2: The specified module could not be found.
```

Bundle inspection:

- Found exactly one `cv2.pyd`.
- Found `opencv_videoio_ffmpeg4130_64.dll`.
- The `cv2.pyd` imported CUDA/NVIDIA DLLs that were not present beside `cv2`.

Manual proof:

- After copying the required CUDA/NVIDIA DLLs into
  `dist_test\smoke_import_cv2.dist\cv2`, the smoke exe printed:

```text
4.13.0
OK
```

Conclusion:

- Do not spend more time on duplicate `cv2.pyd` unless a future bundle actually
  contains duplicates.
- Treat CUDA/NVIDIA DLL staging as a required post-build step for this local
  CUDA-enabled OpenCV wheel.

### OpenCV CUDA DLL Copy Helper

Created and tested:

```text
scripts/copy_cv2_cuda_dlls.ps1
```

The script copies required DLLs from:

- `C:\Windows\System32`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64`

into the bundle's `cv2` directory.

Known copied dependency families:

- `nvcuvid.dll`
- `nvEncodeAPI64.dll`
- `npp*64_13.dll`
- `cublas64_13.dll`
- `cublasLt64_13.dll`
- `cufft64_12.dll`
- `cudart64_13.dll`
- `cudnn64_9.dll`

### Deprecated Model Downloader Neutralized

Updated:

```text
reconstruction_gui/model_downloader.py
```

Current behavior:

- Imports no OpenCV, RF-DETR, Ultralytics, Hugging Face, CustomTkinter, or model
  libraries.
- `check_missing_models()` returns an empty list for deprecated compatibility.
- `ModelDownloadDialog.__init__` raises a clear error directing callers to
  `setup_wizard.py`.

Also removed the stale hidden import from:

```text
reconstruction_zone.spec
```

This addresses Claude's concern that stale `model_downloader.py` could pull
`rfdetr` or `ultralytics` into packaging analysis.

### Source Checks That Passed

These source-level checks passed after the downloader/model-path work:

```powershell
C:\Python314\python.exe -m py_compile .\reconstruction_gui\model_downloader.py .\reconstruction_gui\model_paths.py .\reconstruction_gui\setup_wizard.py .\reconstruction_gui\reconstruction_pipeline.py
```

```powershell
C:\Python314\python.exe scripts/smoke_model_paths.py
```

`check_all_models()` reported:

```text
sam3 True
rfdetr True
yolo26 True
```

### ffmpeg Include Path Correction

The stale command used:

```text
--include-data-dir=ffmpeg=ffmpeg
```

That failed because the repo does not contain a local `ffmpeg` directory.

Correct Nuitka file includes should be:

```text
--include-data-files="C:/Users/alexm/ffmpeg/bin/ffmpeg.exe=ffmpeg/ffmpeg.exe"
--include-data-files="C:/Users/alexm/ffmpeg/bin/ffprobe.exe=ffmpeg/ffprobe.exe"
```

Runtime expectation:

- Users should not need ffmpeg on their system PATH if the packaged app prepends
  the bundled `ffmpeg/` folder to PATH or resolves those binaries explicitly.
- The issue was not that ffmpeg was "in the wrong spot" for users; the attempted
  Nuitka command referenced the wrong source location during packaging.

## Build Attempts and Outcomes

### Attempt 1: Full Build With Stale ffmpeg Directory

Outcome:

- Failed immediately.

Cause:

- `--include-data-dir=ffmpeg=ffmpeg` pointed at a non-existent repo directory.

Status:

- Do not repeat.
- Use explicit file includes for `ffmpeg.exe` and `ffprobe.exe`.

### Attempt 2: Corrected Full Build Drafts

Outcome:

- Several proposed full-build command attempts were rejected or stopped before a
  useful result.

Cause:

- The command still had broad package includes and unresolved warning strategy.

Status:

- Do not resume broad full-build commands until import graph is constrained.

### Attempt 3: Full Build With Warning Fixes and `dill-compat`

Added options included:

```text
--enable-plugin=tk-inter
--enable-plugin=no-qt
--enable-plugin=dill-compat
--nofollow-import-to=triton
```

Outcome:

- Failed.

Fatal error:

```text
FATAL: dill-compat: Error, decision True does not match other plugin 'anti-bloat' decision.
```

Conclusion:

- Do not enable `dill-compat` in the current environment without first resolving
  the anti-bloat conflict.
- The warning about missing dill support is not enough by itself to justify
  enabling a plugin that causes a fatal conflict.

### Attempt 4: Full Build Without `dill-compat`

Options retained included:

```text
--enable-plugin=tk-inter
--enable-plugin=no-qt
--nofollow-import-to=triton
```

Outcome:

- Ran for a while.
- Emitted the known warning:

```text
Use '--enable-plugin=dill-compat' for: Missing 'dill' support.
```

- Eventually exited without a useful success/failure artifact and no final exe.

Conclusion:

- The command was still too broad.
- A warning-only approach is insufficient; the import graph must be narrowed
  before another full build.

### Attempt 5: "Small" GUI Import Nuitka Smoke

Created:

```text
scripts/smoke_import_gui.py
```

Purpose:

- Import the GUI app module without constructing a CTk root.
- Confirm Nuitka can analyze the GUI shell before spending time on a full build.

Outcome:

- The compile did not behave like a small GUI compile.
- It stalled in Nuitka's Transformers plugin while walking
  `transformers.models.deprecated.mega`.

Process evidence:

- Main build process spawned a child Nuitka process.
- A further child process was running a Transformers import-structure command.
- Build folders remained effectively empty.
- No `SmokeGuiImport.exe` was produced.

Conclusion:

- The smoke itself was valid in concept, but the source import graph made it
  invalid in practice.
- Do not rerun this smoke until the app shell import is guarded against
  `ultralytics`, `transformers`, and `rfdetr`.

## Warning Policy

### `tk-inter`

Action:

- Keep `--enable-plugin=tk-inter`.

Reason:

- The app uses Tk/CustomTkinter heavily.

### Matplotlib GUI Toolkit Warning

Observed warning:

- Nuitka warned about matplotlib GUI toolkit selection.

Source usage found:

- `reconstruction_gui/pointcloud_viewer.py` imports `matplotlib.cm` for colormap
  mapping.
- No `pyplot`, `FigureCanvas`, or matplotlib GUI backend usage was found in the
  relevant app source search.

Action:

- Keep `--enable-plugin=no-qt` for now.
- Consider setting/forcing a non-GUI backend only if a later source check shows
  actual pyplot/backend usage.

### `dill-compat`

Action:

- Do not enable unless a later runtime test proves a dill-dependent feature is
  broken and the anti-bloat conflict has a specific fix.

Reason:

- Enabling it caused a fatal Nuitka plugin conflict.

Scope:

- This warning likely relates to optional serialization paths in dependencies,
  not to the core GUI shell, ffmpeg, OpenCV, or basic model path readiness.
- It should not be treated as a build-stopping issue until a packaged runtime
  smoke demonstrates a feature that actually requires dill support.

### `triton` / bitsandbytes

Action:

- Keep no-follow exclusions for optional acceleration/integration paths when
  they appear:

```text
--nofollow-import-to=triton
--nofollow-import-to=triton.*
--nofollow-import-to=bitsandbytes
--nofollow-import-to=bitsandbytes.*
--nofollow-import-to=transformers.integrations.bitsandbytes
```

But note:

- These exclusions did not solve the latest stall because the larger problem was
  the Transformers plugin walking `transformers.models.*`.

## Recommended Source Fix Before Claude Builds Again

The next source change should be small and testable.

Recommended first fix:

1. Stop importing `prep360.core` wholesale from `source_tab.py`.
2. Import only the exact modules/classes used by the Source tab.
3. Do not import `prep360.core.segmenter` as part of Source tab startup.
4. Resolve or remove the invalid `AdjustmentRecipe` import.

Alternative acceptable fix:

1. Keep the `from prep360.core import ...` API.
2. Make `prep360.core.__init__` lazy for segmentation exports:
   `Segmenter`, `SegmentConfig`, `COCO_CLASSES`, `CLASS_PRESETS`.
3. Also export or remove/fix `AdjustmentRecipe`.

Preferred fix:

- Direct imports in `source_tab.py` are more explicit and less risky for the
  library API.
- Lazy exports in `prep360.core.__init__` are still useful if other callers rely
  on `from prep360.core import Segmenter`, but they are broader than strictly
  needed for the GUI shell.

Success criterion:

- Importing `reconstruction_gui.reconstruction_zone` in normal Python should not
  leave any of these modules in `sys.modules`:

```text
ultralytics
transformers
rfdetr
```

## Proposed Guard Script Before Any Nuitka Build

Create or temporarily run a Python guard equivalent to:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = ROOT / "reconstruction_gui"
sys.path[:0] = [str(ROOT), str(GUI_DIR)]

import reconstruction_gui.reconstruction_zone as reconstruction_zone

blocked = ("ultralytics", "transformers", "rfdetr")
loaded = [
    name for name in blocked
    if any(module == name or module.startswith(name + ".") for module in sys.modules)
]

print(reconstruction_zone.__version__)
print("GUI_IMPORT_OK")
if loaded:
    raise SystemExit(f"Heavy modules imported during GUI shell import: {loaded}")
```

Do not run a Nuitka GUI smoke until this guard passes in normal Python.

Note:

- One inline guarded run hit a local permission issue writing
  `C:\Users\alexm\.reconstruction_zone\crash.log`. That is an environment/file
  permission issue, not proof about the import graph. Use the same elevated or
  environment-clean path that allowed the earlier `smoke_import_gui.py` source
  run to pass.

## Claude Build Protocol

Claude should not start with a full 2.5-hour build.

Recommended sequence:

1. Confirm no Python/Nuitka build processes are running.
2. Apply the source import fix.
3. Run the guarded normal-Python GUI shell import.
4. If it imports `ultralytics`, `transformers`, or `rfdetr`, stop and fix source.
5. Run the constrained Nuitka GUI import smoke.
6. If the smoke enters Transformers model-tree analysis, stop immediately and
   inspect the import graph; do not let it run for hours.
7. Only after GUI smoke passes, run the full build.
8. After full build, run the OpenCV CUDA DLL copy helper against the final
   `.dist` directory.
9. Run packaged runtime smokes.

Suggested constrained GUI smoke options:

```text
--standalone
--assume-yes-for-downloads
--windows-console-mode=attach
--enable-plugin=tk-inter
--enable-plugin=no-qt
--nofollow-import-to=bitsandbytes
--nofollow-import-to=bitsandbytes.*
--nofollow-import-to=triton
--nofollow-import-to=triton.*
--nofollow-import-to=transformers.integrations.bitsandbytes
--user-package-configuration-file=scripts/nuitka-torch-fix.yml
--include-package=customtkinter
--include-data-dir=reconstruction_gui/docs=reconstruction_gui/docs
--include-data-dir=resources=resources
--include-data-files=reconstruction-zone.ico=reconstruction-zone.ico
```

Do not add broad includes such as:

```text
--include-package=reconstruction_gui
--include-package=torch
--include-package=torchvision
```

until the smaller smoke proves the import surface is controlled. Broad includes
can force Nuitka to follow packages that normal app startup would not load.

## Full Build Inputs Once Source Smokes Pass

Known required inclusions:

```text
--enable-plugin=tk-inter
--enable-plugin=no-qt
--user-package-configuration-file=scripts/nuitka-torch-fix.yml
--include-package=customtkinter
--include-data-dir=reconstruction_gui/docs=reconstruction_gui/docs
--include-data-dir=resources=resources
--include-data-files=reconstruction-zone.ico=reconstruction-zone.ico
--include-data-files="C:/Users/alexm/ffmpeg/bin/ffmpeg.exe=ffmpeg/ffmpeg.exe"
--include-data-files="C:/Users/alexm/ffmpeg/bin/ffprobe.exe=ffmpeg/ffprobe.exe"
```

Known post-build step:

```powershell
.\scripts\copy_cv2_cuda_dlls.ps1 -DistDir <final .dist directory>
```

Known command to avoid:

```text
--enable-plugin=dill-compat
```

unless the anti-bloat conflict is specifically resolved first.

## What Not To Repeat

- Do not run another broad full build before the guarded GUI import passes.
- Do not treat the `dill-compat` warning as actionable until runtime evidence
  proves it matters.
- Do not assume `--nofollow-import-to=triton` or bitsandbytes exclusions are
  enough to control Transformers.
- Do not rerun the GUI Nuitka smoke if normal Python import still loads
  `ultralytics`, `transformers`, or `rfdetr`.
- Do not use the stale `--include-data-dir=ffmpeg=ffmpeg` command.
- Do not spend time deleting `cv2.pyd` files unless a future bundle inspection
  actually shows duplicate `cv2.pyd` locations.

## Current Open Questions

1. What is the intended replacement for `AdjustmentRecipe`?
   - It is referenced in `source_tab.py`, but no definition was found.
   - This needs a source fix independent of packaging.

2. Should `prep360.core.__init__` continue exporting segmentation symbols?
   - If yes, implement lazy exports so importing `prep360.core` does not import
     Ultralytics.
   - If no, remove segmentation from top-level core exports and update callers.

3. Does RF-DETR actually require broad `transformers.models.*` at packaged
   runtime?
   - Do not answer this with a full app build.
   - Answer with a targeted RF-DETR import/model-load smoke after the GUI shell
     smoke is clean.

4. Should the packaged app use the CUDA OpenCV wheel?
   - The local wheel works after DLL staging.
   - For public distribution, this creates a larger and more fragile dependency
     story than CPU OpenCV.

## Recommended Next Action

Before Claude runs any build:

1. Fix `source_tab.py` imports so the GUI shell does not import `prep360.core`
   segmentation.
2. Fix or remove `AdjustmentRecipe`.
3. Run the guarded normal-Python GUI import.
4. Only if the guard passes, run the constrained Nuitka GUI smoke.

That is the shortest reliable path out of the current failure mode.
