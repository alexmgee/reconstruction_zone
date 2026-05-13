# Phase 6B Full GUI Mask Preview Fix Result

Date: 2026-05-13

Workspace:

```text
D:\Projects\reconstruction-zone
```

## 1. Purpose

This rebuild was launched after manual Mask tab testing reached two packaged
runtime failures in the prior `phase-6b-full-gui-finalvacuum-20260512`
artifact:

```text
1. Missing Ultralytics runtime data:
   reconstruction_zone.dist\ultralytics\cfg\default.yaml

2. Deeper packaged import failure:
   torch.library infer_schema rejected list[int] annotations while importing
   torch._dynamo._trace_wrapped_higher_order_op through the Ultralytics SAM
   import path.
```

This build is intended to include:

```text
Mask Preview source import hardening
Torch annotation rewrite in scripts\nuitka-torch-fix.yml
Ultralytics cfg runtime data inclusion
The same proven Phase 6B full-GUI Nuitka packaging strategy
```

## 2. Artifact

Build output folder:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512
```

Executable:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Executable metadata:

```text
size: 954,295,296 bytes
created: 2026-05-13 06:36:52 local
last written: 2026-05-13 06:36:54 local
```

Build logs:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\full-gui-build.stdout.log
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\full-gui-build.stderr.log
```

Nuitka XML report:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\full-gui-maskpreviewfix-report.xml
```

Build runner:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\run-full-gui-build.ps1
```

## 3. Build Result

Build command family:

```text
C:\Python314\python.exe -m nuitka
--standalone
--assume-yes-for-downloads
--windows-console-mode=attach
--windows-icon-from-ico=reconstruction-zone.ico
--enable-plugin=tk-inter
--enable-plugin=no-qt
--user-package-configuration-file=scripts/nuitka-torch-fix.yml
--module-parameter=torch-disable-jit=yes
--nofollow-import-to=bitsandbytes
--nofollow-import-to=bitsandbytes.*
--nofollow-import-to=triton
--nofollow-import-to=triton.*
--include-package=prep360
--include-package=reconstruction_gui
--include-package=torch
--include-package=torchvision
--include-package=rfdetr
--include-package=ultralytics
--include-package=customtkinter
--include-data-dir=resources=resources
--include-data-dir=reconstruction_gui/docs=reconstruction_gui/docs
--include-data-dir=<installed ultralytics cfg path>=ultralytics/cfg
--include-data-files=reconstruction-zone.ico=reconstruction-zone.ico
--include-data-files=C:/Users/alexm/ffmpeg/bin/ffmpeg.exe=ffmpeg/ffmpeg.exe
--include-data-files=C:/Users/alexm/ffmpeg/bin/ffprobe.exe=ffmpeg/ffprobe.exe
--output-dir=dist_test/phase-6b-full-gui-maskpreviewfix-20260512
--output-filename=ReconstructionZone.exe
--report=<full-gui-maskpreviewfix-report.xml>
reconstruction_gui/reconstruction_zone.py
```

Observed build result:

```text
build start: 2026-05-13T04:53:37.1528181-05:00
build exit:  2026-05-13T06:42:13.7296411-05:00
exit code:   0
report:      completion="yes"
```

Compiler/cache result:

```text
compiled C files: 9494
clcache hits:     9490
clcache misses:   4
```

Classification:

```text
full GUI rebuild: pass
Nuitka report completion: pass
new executable created: pass
```

## 4. Runtime Data Inclusion

The Nuitka report confirms `ultralytics\cfg\default.yaml` was included as a
specified data-directory file:

```text
ultralytics\cfg\default.yaml
source: C:\Users\alexm\AppData\Roaming\Python\Python314\site-packages\ultralytics\cfg\default.yaml
reason: specified data dir on command line
```

Direct filesystem verification after build:

```text
default.yaml present: yes
ultralytics cfg file count after initial build: 105
ultralytics cfg byte count after initial build: 287,590
```

The repeatable post-build helper was also run:

```text
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\copy_ultralytics_runtime_data.ps1 `
  -DistDir dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist
```

Helper result:

```text
files copied: 107
bytes copied: 376,345
destination:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ultralytics\cfg
```

Classification:

```text
Ultralytics default.yaml failure should be fixed in this artifact.
```

## 5. CUDA/OpenCV Runtime Copy

The repeatable OpenCV CUDA/NVIDIA helper was run:

```text
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\copy_cv2_cuda_dlls.ps1 `
  -DistDir dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist
```

Helper result:

```text
copied DLL count reported by helper: 16
cv2 DLL count after copy: 17
cv2 DLL byte count after copy: 1,029,215,216
```

Classification:

```text
OpenCV CUDA/NVIDIA DLL post-build copy: pass
```

## 6. Source Mask Pipeline Smoke

Command:

```text
C:\Python314\python.exe scripts\smoke_import_mask_pipeline.py
```

Result:

```text
MASK_PIPELINE_IMPORT_OK
{"HAS_FASTSAM": true, "HAS_RFDETR": true, "HAS_SAM3": true, "HAS_YOLO": true}
```

Classification:

```text
source-side Mask pipeline import: pass
```

Important limitation:

```text
This verifies the source environment, not the compiled executable's Mask Preview
runtime path. The real compiled Mask Preview gate still requires manual GUI
testing or a future automation harness that can drive the Mask tab.
```

## 7. Packaged New-Install Vacuum Smoke

Command:

```text
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_packaged_new_install_probe.ps1 `
  -ExePath dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe `
  -CaseName phase-6b-maskpreviewfix-vacuum-smoke-20260513-0457 `
  -DurationSeconds 12 `
  -SetupTestMode
```

Result:

```text
PROCESS_STILL_RUNNING 6756
PROCESS_STOPPED       6756
```

Probe folder:

```text
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\phase-6b-maskpreviewfix-vacuum-smoke-20260513-0457
```

Relevant stderr/log tail:

```text
Session started
Detected colmap binary: C:\Users\alexm\COLMAP\COLMAP-4.0.3-x64-windows-cuda\bin\colmap.exe
Detected spheresfm binary: D:\Tools\SphereSfM\bin\colmap.exe
Models need setup: SAM3, RF-DETR-Seg, YOLO26-n-seg
```

Classification:

```text
packaged executable starts in a blank app-home and reaches missing-model setup state: pass
fatal startup traceback during short smoke: none observed
```

## 8. Process Hygiene

Pre-launch process check found no relevant active `git`, compiler, Python, or
ReconstructionZone processes.

During launch, three transient `git.exe` processes briefly appeared. Follow-up
inspection showed no remaining `git.exe` processes. Final process check after
build and smoke found no active `ReconstructionZone`, `python`, `cl`, `link`,
`scons`, `git`, or extra `powershell` processes from this work.

No Git commands were intentionally run for this task.

No files were written to `C:\tmp`.

## 9. Current Status

Passed:

```text
full GUI rebuild
Nuitka completion report
new executable creation
Ultralytics cfg inclusion
OpenCV CUDA/NVIDIA DLL post-build copy
source Mask pipeline import smoke
packaged blank new-install startup smoke
```

Still not proven:

```text
compiled Mask tab Preview execution from the GUI
compiled Mask operation after Preview
whether the torch infer_schema/list[int] packaged runtime error is eliminated
```

Why this remains unproven:

```text
The failure is triggered by a real Mask tab Preview action inside the GUI. The
existing automated smoke scripts can verify source import and packaged app
startup, but they do not currently drive the Mask tab Preview workflow.
```

## 10. Manual Retest Command

Use the same sandbox the user was testing if possible, so model downloads and
project data do not need to be recreated. From any PowerShell location:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\Projects\reconstruction-zone\scripts\launch_manual_blank_setup_wizard.ps1 `
  -ExePath D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe `
  -CaseName manual-blank-setup-wizard-20260512-210921
```

Do not pass `-Reset` unless intentionally starting the sandbox from scratch.

Manual retest checklist:

```text
1. Confirm the title/window belongs to the newly launched artifact.
2. Confirm setup wizard state still looks correct if it appears.
3. Open or create the same test project used for Extract and Mask testing.
4. Go to Mask tab.
5. Select the same image/frame that previously failed.
6. Press Preview.
7. Record whether the Ultralytics default.yaml error is gone.
8. Record whether the torch infer_schema/list[int] error is gone.
9. If Preview succeeds, run one actual Mask operation.
10. Save stdout/stderr/log output from:
    D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\manual-blank-setup-wizard-20260512-210921\launch-logs
```

## 11. If Preview Still Fails

Classify the new traceback before changing anything:

```text
missing package data
missing DLL
missing model weight
torch/torchvision runtime import problem
Ultralytics optional SAM import problem
RF-DETR runtime problem
YOLO runtime problem
application logic bug
```

If the same `infer_schema(... list[int] ...)` error appears:

```text
1. Confirm the executable path is the maskpreviewfix artifact, not finalvacuum.
2. Confirm scripts\nuitka-torch-fix.yml includes the
   torch._dynamo._trace_wrapped_higher_order_op rewrite.
3. Confirm the Nuitka report includes torch._dynamo._trace_wrapped_higher_order_op.
4. Treat the torch anti-bloat rewrite as insufficient for this runtime path.
5. Consider a focused Ultralytics anti-bloat patch that prevents importing
   ultralytics.models.sam during optional YOLO/FastSAM availability checks.
```

Do not apply a broad Ultralytics exclusion blindly. A broad exclusion may hide
the error but could also break model paths that the application needs.

## 12. Verified Reference

Official Nuitka documentation confirms that standalone builds require explicit
data-file/data-directory inclusion for files that packages load at runtime:

```text
https://nuitka.net/user-documentation/user-manual.html
```

Relevant verified point:

```text
--include-data-dir=<source>=<target> recursively includes a directory in the
distribution, which is why ultralytics\cfg is now included rather than copied
only by hand after the fact.
```
