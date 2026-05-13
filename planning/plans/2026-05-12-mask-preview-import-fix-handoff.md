# Mask Preview Import Fix Handoff

Date: 2026-05-12  
Owner: Codex  
Repository: `D:\Projects\reconstruction-zone`  
Status: **source/build fixes prepared; full GUI rebuild pending/running depending on current process state**

## 1. Why This Document Exists

The setup wizard itself was manually tested by the user and passed. The Extract
tab was also manually exercised successfully.

The current blocker is now the Mask tab Preview path in the packaged GUI.

This document is a handoff in case Codex cannot continue after launching the
next rebuild.

## 2. Current Good Artifact Before This Fix

The previous final-vacuum artifact is:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

It validated:

```text
setup wizard visible in blank state: pass
manual setup wizard click-through: pass, per user
project-store vacuum: pass
Extract tab operations: pass, per user
real model sandbox recognition: pass
```

It is **not** sufficient for Mask tab Preview because it was built before the
Mask Preview import fix below.

## 3. Mask Preview Failures Observed

### 3.1 Missing Ultralytics Runtime Config

First Mask Preview failure:

```text
FileNotFoundError:
reconstruction_zone.dist\ultralytics\cfg\default.yaml
```

Verified local source:

```text
C:\Users\alexm\AppData\Roaming\Python\Python314\site-packages\ultralytics\cfg\default.yaml
```

Immediate patch applied to final-vacuum artifact:

```text
copied ultralytics\cfg into:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ultralytics\cfg

files: 107
bytes: 376,345
default.yaml present: yes
```

Repeatable helper:

```text
scripts\copy_ultralytics_runtime_data.ps1
```

### 3.2 Torch/Ultralytics Import Failure

Second Mask Preview failure after `default.yaml` was fixed:

```text
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
packaged runtime import failure
not a setup-wizard failure
not a model-download failure
```

## 4. Source/Build Fixes Already Applied

### 4.1 Optional Ultralytics Import Guard

File:

```text
reconstruction_gui\reconstruction_pipeline.py
```

Change:

```text
FastSAM and YOLO optional imports now catch Exception, not only ImportError.
```

Reason:

If Ultralytics imports fail inside its optional SAM/SAM3 side tree, the whole
Mask pipeline import should not crash. The app can still use RF-DETR when
available.

### 4.2 prep360 YOLO Guard

File:

```text
prep360\core\segmenter.py
```

Change:

```text
optional YOLO import now catches Exception.
```

### 4.3 Torch Nuitka Annotation Rewrite

File:

```text
scripts\nuitka-torch-fix.yml
```

Added module:

```text
torch._dynamo._trace_wrapped_higher_order_op
```

Rewrite:

```text
from typing import Any, Optional
->
from typing import Any, Optional, List

shape: list[int],
->
shape: List[int],

indices: list[Tensor],
->
indices: List[Tensor],
```

Reason:

The runtime schema error explicitly says `list[int]` was rejected while
`typing.List[int]` is accepted. This follows the same pattern as the existing
Torch/Nuitka annotation rewrites in `scripts\nuitka-torch-fix.yml`.

### 4.4 Focused Smoke Script

File:

```text
scripts\smoke_import_mask_pipeline.py
```

Source result:

```text
MASK_PIPELINE_IMPORT_OK
{"HAS_FASTSAM": true, "HAS_RFDETR": true, "HAS_SAM3": true, "HAS_YOLO": true}
```

### 4.5 Mask Preview Fix Build Runner

Prepared runner:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\run-full-gui-build.ps1
```

This runner uses the same proven Phase 6B command family and includes:

```text
--user-package-configuration-file=scripts/nuitka-torch-fix.yml
--include-data-dir="$UltralyticsCfg=ultralytics/cfg"
```

## 5. Focused Packaged Smoke Attempt

Attempted:

```text
dist_test\mask-pipeline-import-smoke-20260512
```

Result:

```text
INCONCLUSIVE
```

Nuitka report:

```text
completion="exception"
OSError: [Errno 22] Invalid argument
```

Cause:

The exception happened while Nuitka/Colorama was printing plugin output after
the shell command timeout/closed output stream. It did not produce an exe. It
does not prove or disprove the original Torch `infer_schema` fix.

## 6. Rebuild To Launch

Launch command:

```powershell
$root='D:\Projects\reconstruction-zone'
$out=Join-Path $root 'dist_test\phase-6b-full-gui-maskpreviewfix-20260512'
$stdout=Join-Path $out 'full-gui-build.stdout.log'
$stderr=Join-Path $out 'full-gui-build.stderr.log'
$runner=Join-Path $out 'run-full-gui-build.ps1'
$p=Start-Process -FilePath 'powershell.exe' `
  -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$runner) `
  -WorkingDirectory $root `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru
"BUILD_PROCESS_ID`t$($p.Id)" | Set-Content -LiteralPath (Join-Path $out 'full-gui-build-process-id.txt') -Encoding ascii
"BUILD_STARTED`t$(Get-Date -Format o)" | Set-Content -LiteralPath (Join-Path $out 'full-gui-build-monitor.txt') -Encoding utf8
```

Expected output artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Expected report:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\full-gui-maskpreviewfix-report.xml
```

Expected exit code file:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\full-gui-build-exit-code.txt
```

## 7. Monitoring Commands

Check process state:

```powershell
Get-Process python,cl,link,scons,git,powershell -ErrorAction SilentlyContinue |
  Select-Object Id,ProcessName,CPU,Path,StartTime
```

Read logs:

```powershell
Get-Content D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\full-gui-build.stderr.log -Tail 120
Get-Content D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\full-gui-build.stdout.log -Tail 80
```

Check generated C/object progress:

```powershell
Get-ChildItem D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.build -Filter *.c -File | Measure-Object
Get-ChildItem D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.build -Filter *.obj -File | Measure-Object
```

Check report completion:

```powershell
Select-String -Path D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\full-gui-maskpreviewfix-report.xml -Pattern 'completion="yes"' -SimpleMatch
```

Expected successful markers:

```text
Nuitka: Completed Python level compilation and optimization.
Nuitka-Scons: Backend C compiler: cl
Nuitka-Scons: Backend C linking with 9494 files
Nuitka: Successfully created ReconstructionZone.exe
Nuitka-Reports: Compilation report written
PHASE_6B_FULL_GUI_MASKPREVIEWFIX_BUILD_EXIT 0
```

## 8. Post-Build Steps

Run both post-build copies even though Ultralytics cfg is included at compile
time. The copy script is a cheap guardrail.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\Projects\reconstruction-zone\scripts\copy_cv2_cuda_dlls.ps1 `
  -DistDir D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist

powershell -NoProfile -ExecutionPolicy Bypass -File D:\Projects\reconstruction-zone\scripts\copy_ultralytics_runtime_data.ps1 `
  -DistDir D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist
```

Verify:

```powershell
Test-Path D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ultralytics\cfg\default.yaml
```

## 9. Manual Retest Command

Use the same manual sandbox if the user wants to continue where they left off:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\Projects\reconstruction-zone\scripts\launch_manual_blank_setup_wizard.ps1 `
  -ExePath D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe `
  -CaseName manual-blank-setup-wizard-20260512-210921
```

Do **not** pass `-Reset` unless intentionally starting over with empty models.

## 10. What To Test Next

1. Open the Mask tab.
2. Use the same input image that failed before.
3. Click Preview.
4. Confirm the previous error is gone:

```text
infer_schema(func): Parameter shape has unsupported type list[int]
```

5. If Preview succeeds, test one actual Mask operation.
6. If Preview fails, copy the full traceback and classify:
   - missing data file;
   - missing DLL;
   - model weight path;
   - PyTorch/Torchvision import/runtime issue;
   - RF-DETR runtime issue;
   - YOLO runtime issue.

## 11. If The Same Error Persists

If `infer_schema(... list[int] ...)` still appears after the rebuild:

1. Confirm the new artifact is being launched, not the final-vacuum artifact.
2. Confirm the build used the updated `scripts\nuitka-torch-fix.yml`.
3. Inspect the report for `torch._dynamo._trace_wrapped_higher_order_op`.
4. Consider an Ultralytics anti-bloat patch that prevents `ultralytics.models`
   from importing `.sam` when only YOLO is required.

Candidate fallback patch idea:

```text
module-name: 'ultralytics.models'
anti-bloat replacement:
  remove or neutralize:
    from .sam import SAM
```

Do not apply that blindly without a focused source/package smoke, because it
may affect FastSAM/SAM-related code paths.

## 12. Process Hygiene

Watch and stop unexpected blank-command-line `git.exe` processes if they appear
and are not part of a requested Git operation.

After build/test, check:

```powershell
Get-Process ReconstructionZone,python,cl,link,scons,git,powershell -ErrorAction SilentlyContinue |
  Select-Object Id,ProcessName,CPU,Path,CommandLine
```

Do not run `git add`, `git status`, or other Git commands unless the user asks.

## 13. Documentation To Update After Rebuild

Update:

```text
planning/plans/2026-05-12-phase-6b-full-gui-finalvacuum-result.md
planning/plans/2026-05-11-next-codex-handoff.md
planning/plans/2026-05-11-remediation-progress-report.md
```

If the rebuild succeeds, add a new final report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-maskpreviewfix-result.md
```

Include:

- build start/finish times;
- exit code;
- report completion;
- artifact metadata;
- OpenCV DLL copy result;
- Ultralytics cfg copy result;
- manual Mask Preview retest result.

## 14. Verified References

Nuitka official user manual confirms standalone builds are built artifacts and
that data files are a separate packaging concern:

```text
https://nuitka.net/user-documentation/user-manual.html
```

## 15. Rebuild Executed, 2026-05-13

The prepared full-GUI rebuild was launched and completed.

Result:

```text
artifact:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe

build start: 2026-05-13T04:53:37.1528181-05:00
build exit:  2026-05-13T06:42:13.7296411-05:00
exit code:   0
report:      completion="yes"
compiled C files: 9494
clcache hits/misses: 9490 / 4
```

Post-build helpers:

```text
OpenCV CUDA/NVIDIA DLL copy: pass
Ultralytics cfg runtime copy: pass
ultralytics cfg files after helper: 107
ultralytics cfg bytes after helper: 376,345
```

Smoke verification:

```text
source Mask pipeline import: pass
packaged blank new-install launch smoke: pass
fatal startup traceback: none observed
```

Detailed result report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-maskpreviewfix-result.md
```

Remaining gate:

```text
manual Mask tab Preview against the new maskpreviewfix artifact
```

Recommended manual launch:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\Projects\reconstruction-zone\scripts\launch_manual_blank_setup_wizard.ps1 `
  -ExePath D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe `
  -CaseName manual-blank-setup-wizard-20260512-210921
```
