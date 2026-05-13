# Phase 6B Full GUI Final-Vacuum Build Result

Date: 2026-05-12  
Owner: Codex  
Repository: `D:\Projects\reconstruction-zone`  
Status: **final parity rebuild passed; visible blank-vacuum setup wizard passed**

## 1. Purpose

This was a parity rebuild after the source fix that moved the Projects tab's
default tracker store away from the legacy global path:

```text
D:\tracker.json
```

The previous rebuilt artifact already proved the missing-model setup wizard was
visible and stable, but it was built before the project-store vacuum fix. This
build packages the current source so a blank new-install test covers:

- empty model directory;
- isolated app-home;
- isolated HuggingFace cache;
- app-home prefs/logs;
- app-home project tracker path.

## 2. Build Directory

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512
```

Runner:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\run-full-gui-build.ps1
```

Artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

## 3. Build Command Family

Same proven Phase 6B command family:

```text
NUITKA_CACHE_DIR=D:\Projects\reconstruction-zone\dist_test\nuitka-cache
no --include-package=transformers
no --nofollow-import-to=transformers.models.*
torch YAML fix retained
bitsandbytes/triton excluded
prep360/reconstruction_gui/torch/torchvision/rfdetr/ultralytics/customtkinter included
```

## 4. Build Result

The build exited successfully:

```text
started: 2026-05-12T19:26:04-05:00
finished: 2026-05-12T20:46:49-05:00
exit code: 0
duration: about 1h 21m
report completion: yes
```

Nuitka markers:

```text
Completed Python level compilation and optimization.
Backend C compiler: cl (cl 14.5).
Backend C linking with 9494 files.
Compiled 9494 C files using clcache with 9486 cache hits and 8 cache misses.
Successfully created ReconstructionZone.exe.
Compilation report written.
```

Artifact metadata:

```text
ReconstructionZone.exe: 954,292,736 bytes
full-gui-finalvacuum-report.xml: 22,643,674 bytes
```

## 5. OpenCV CUDA/NVIDIA DLL Copy

Post-build command:

```powershell
.\scripts\copy_cv2_cuda_dlls.ps1 `
  -DistDir D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist
```

Result:

```text
Copied 16 OpenCV CUDA/NVIDIA DLLs to:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\cv2
```

## 6. Visible Blank-Vacuum Setup Wizard Probe

Command shape:

```powershell
.\scripts\run_visual_vacuum_setup_probe.ps1 `
  -ExePath D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ReconstructionZone.exe `
  -CaseName visual-empty-vacuum-finalvacuum-20260512 `
  -WarmupSeconds 18 `
  -Reset
```

Captured isolated environment:

```text
APP_HOME=D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-finalvacuum-20260512\app-home
MODEL_DIR=D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-finalvacuum-20260512\app-home\models
STRICT_MODEL_DIRS=1
HF_HOME=D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-finalvacuum-20260512\hf-home
TRANSFORMERS_CACHE=D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-finalvacuum-20260512\hf-home\transformers
```

Probe lifecycle:

```text
PROCESS_STILL_RUNNING after warmup
MAIN_WINDOW_HANDLE present
desktop screenshot captured
main-window screenshot captured
PROCESS_STOPPED by harness
```

Visual result:

- setup wizard visible;
- `SAM3`, `RF-DETR-Seg`, and `YOLO26-n-seg` listed;
- readiness showed `0/3 ready`;
- `Begin Setup` button visible;
- Projects tab behind the wizard showed `0 projects`;
- Projects tab store path used the isolated app-home project tracker, not
  `D:\tracker.json`.

Evidence:

```text
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-finalvacuum-20260512\screenshots\desktop-after-warmup.png
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-finalvacuum-20260512\screenshots\main-window-after-warmup.png
```

Log marker:

```text
Models need setup: SAM3, RF-DETR-Seg, YOLO26-n-seg
```

Sandbox app-home contents after passive first launch:

```text
logs
models
prefs
logs\crash.log
prefs\studio_prefs.json
```

Text search inside the final vacuum probe folder found no `D:\tracker.json`.

Classification:

```text
PASS
```

## 7. Final Artifact Against Real Model Sandbox

The final artifact was also launched against the existing isolated sandbox that
contains real SAM3, RF-DETR, and YOLO model assets:

```text
case:
sam3-download-isolated-20260512
```

Result:

```text
PROCESS_STILL_RUNNING after 20 seconds
PROCESS_STOPPED by harness
log marker:
All models ready — setup wizard not needed
```

Classification:

```text
PASS
```

## 8. Process Hygiene

Final process checks found no lingering:

```text
ReconstructionZone
smoke_setup_wizard
python
cl
link
scons
git
powershell from the build runner
```

## 9. Current Release-Gate Classification

Validated in the final artifact:

```text
full GUI Nuitka build: PASS
Nuitka XML completion: PASS
OpenCV CUDA/NVIDIA DLL copy: PASS
visible blank model/setup wizard state: PASS
app-home project-store vacuum: PASS
no D:\tracker.json leakage in final vacuum probe: PASS
all-real-models-ready recognition: PASS
```

Still not fully exercised by this final probe:

```text
manual click-through of the visible setup wizard's Begin Setup flow
normal-tab visual sweep after dismissing/bypassing setup
```

Those are UI acceptance gates, not evidence of a packaging compile failure.

## 10. Mask Preview Runtime Packaging Finding

During manual user testing after the setup wizard passed, the Mask tab Preview
button failed with:

```text
FileNotFoundError: [Errno 2] No such file or directory:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ultralytics\cfg\default.yaml
```

Verification:

```text
local installed file exists:
C:\Users\alexm\AppData\Roaming\Python\Python314\site-packages\ultralytics\cfg\default.yaml

packaged file was missing:
reconstruction_zone.dist\ultralytics\cfg\default.yaml
```

Classification:

```text
packaging data-file miss, not setup-wizard failure
```

Immediate hot patch applied to the final artifact:

```text
copied:
C:\Users\alexm\AppData\Roaming\Python\Python314\site-packages\ultralytics\cfg

to:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ultralytics\cfg

files copied: 107
bytes copied: 376,345
default.yaml present: yes
```

Repeatable post-build helper added:

```text
scripts\copy_ultralytics_runtime_data.ps1
```

Final-vacuum build runner was also updated to include:

```text
--include-data-dir="$UltralyticsCfg=ultralytics/cfg"
```

Required follow-up:

```text
Relaunch the patched final artifact and rerun Mask tab Preview.
If this specific error is gone, continue Mask-tab runtime testing for the next
missing package-data or model-runtime issue.
```

## 11. Mask Preview Torch/Ultralytics Import Failure

After the Ultralytics `default.yaml` hot patch, manual testing reached a deeper
Mask Preview failure:

```text
Preview error:
infer_schema(func): Parameter shape has unsupported type list[int].
Got func with signature:
(shape: list[int], indices: list[torch.Tensor], vals: torch.Tensor) -> torch.Tensor
```

Traceback path:

```text
reconstruction_pipeline.py
from ultralytics import FastSAM / YOLO
ultralytics.__getattr__
ultralytics.models
ultralytics.models.sam
ultralytics.models.sam.sam3.geometry_encoders
torchvision
torch._dynamo._trace_wrapped_higher_order_op
torch.library.custom_op schema inference
```

Classification:

```text
packaged runtime import failure, not setup-wizard failure
```

Source/build fixes applied:

```text
reconstruction_gui\reconstruction_pipeline.py:
  optional FastSAM/YOLO imports now catch Exception, not only ImportError.
  This prevents optional Ultralytics failures from crashing pipeline import.

prep360\core\segmenter.py:
  optional YOLO import now catches Exception.

scripts\nuitka-torch-fix.yml:
  added torch._dynamo._trace_wrapped_higher_order_op annotation rewrite:
  list[...] -> typing.List[...]

scripts\smoke_import_mask_pipeline.py:
  added focused import smoke for the Mask pipeline path.
```

Source verification:

```text
py_compile: pass
source mask-pipeline import smoke: pass
```

Focused packaged smoke attempt:

```text
dist_test\mask-pipeline-import-smoke-20260512
```

Result:

```text
INCONCLUSIVE
```

Nuitka wrote a report with:

```text
completion="exception"
OSError: [Errno 22] Invalid argument
```

The exception occurred while Nuitka was printing plugin output through
Colorama after the command timeout/closed output stream. This does not prove
or disprove the original Torch `infer_schema` fix.

Packaging state:

```text
Current final-vacuum artifact does not contain these compiled source fixes.
New artifact pending rebuild.
```

Prepared but not run:

```text
dist_test\phase-6b-full-gui-maskpreviewfix-20260512\run-full-gui-build.ps1
```

The user declined the rebuild launch at this point.

## 12. Superseded By Mask Preview Fix Rebuild

Update, 2026-05-13:

The rebuild was later approved, launched, and completed successfully. For any
Mask tab Preview testing, use the newer artifact rather than this final-vacuum
artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

New build result:

```text
exit code: 0
Nuitka XML completion: yes
compiled C files: 9494
clcache hits/misses: 9490 / 4
Ultralytics cfg inclusion: pass
OpenCV CUDA/NVIDIA DLL copy: pass
source Mask pipeline import smoke: pass
packaged blank new-install launch smoke: pass
```

Detailed report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-maskpreviewfix-result.md
```
