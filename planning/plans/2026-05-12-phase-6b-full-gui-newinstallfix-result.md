# Phase 6B Full GUI New-Install-Fix Build Result

Date started: 2026-05-12  
Date completed: 2026-05-12  
Owner: Codex  
Repository: `D:\Projects\reconstruction-zone`  
Status: **full GUI rebuild passed; isolated new-install setup-wizard smoke passed**

## 1. Purpose

This run rebuilds the full GUI after the setup-wizard `_root` fix and app-home
path refactor.

It uses the same proven Phase 6B command family as the previous successful full
GUI build:

- workspace `NUITKA_CACHE_DIR`;
- no `--include-package=transformers`;
- no `--nofollow-import-to=transformers.models.*`;
- torch YAML fix retained;
- OpenCV CUDA/NVIDIA DLL copy to follow if build succeeds.

## 2. Pre-Build State

The known setup-wizard crash was fixed and validated before this build:

```text
SetupWizard.self._root -> SetupWizard.self._app_root
```

Validation already passed:

- source direct setup-wizard smoke;
- source `run_setup_wizard_if_needed` smoke;
- packaged direct setup-wizard smoke;
- packaged `run_setup_wizard_if_needed` smoke.

Detailed pre-build proof:

```text
planning/plans/2026-05-12-new-install-setup-wizard-validation-result.md
```

## 3. Build Directory

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512
```

## 4. Exact Build Runner

```text
dist_test\phase-6b-full-gui-newinstallfix-20260512\run-full-gui-build.ps1
```

## 5. Result

### 5.1 Build Result

The full GUI Nuitka rebuild exited `0`.

```text
started: 2026-05-12T14:38:16-05:00
finished: 2026-05-12T17:18:06-05:00
duration: about 2h 40m
exit code: 0
```

Nuitka reported:

```text
report completion: yes
compiled/linked C files: 9494
clcache hits: 4367
clcache misses: 5127
```

Artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Artifact metadata:

```text
ReconstructionZone.exe: 954,277,376 bytes
full-gui-newinstallfix-report.xml: 22,643,061 bytes
```

The XML report begins with:

```text
<nuitka-compilation-report ... mode="standalone" completion="yes">
```

### 5.2 Post-Build OpenCV DLL Copy

Command:

```powershell
.\scripts\copy_cv2_cuda_dlls.ps1 `
  -DistDir dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist
```

Result:

```text
Copied 16 OpenCV CUDA/NVIDIA DLLs to:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist\cv2
```

### 5.3 Isolated New-Install Probe: Empty App-Home

Command shape:

```powershell
.\scripts\run_packaged_new_install_probe.ps1 `
  -ExePath D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe `
  -CaseName full-gui-empty-apphome-20260512 `
  -DurationSeconds 20
```

Harness result:

```text
PROCESS_STILL_RUNNING
PROCESS_STOPPED by harness after 20 seconds
```

Captured log markers:

```text
Session started
Detected colmap binary:
C:\Users\alexm\COLMAP\COLMAP-4.0.3-x64-windows-cuda\bin\colmap.exe
Detected spheresfm binary:
D:\Tools\SphereSfM\bin\colmap.exe
Models need setup: SAM3, RF-DETR-Seg, YOLO26-n-seg
```

Crash log tail:

```text
Session started
```

No fatal traceback was written to the isolated crash log.

Classification:

```text
PASS for empty new-install setup-wizard liveness.
```

This verifies that the previous first-launch Tk fatal is fixed in the rebuilt
full GUI.

Artifacts:

```text
dist_test\new-install-sandbox\full-gui-empty-apphome-20260512\
```

### 5.4 Isolated New-Install Probe: RF-DETR / YOLO Fixture Models

Command shape:

```powershell
.\scripts\run_packaged_new_install_probe.ps1 `
  -ExePath D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe `
  -CaseName full-gui-rf-yolo-fixtures-20260512 `
  -DurationSeconds 20 `
  -WithFixtureModels
```

Harness result:

```text
PROCESS_STILL_RUNNING
PROCESS_STOPPED by harness after 20 seconds
```

Captured log markers:

```text
Session started
Detected colmap binary:
C:\Users\alexm\COLMAP\COLMAP-4.0.3-x64-windows-cuda\bin\colmap.exe
Detected spheresfm binary:
D:\Tools\SphereSfM\bin\colmap.exe
Models need setup: SAM3
```

Isolated model directory contained:

```text
rf-detr-seg-small.pt
yolo26n-seg.pt
```

Crash log tail:

```text
Session started
```

No fatal traceback was written to the isolated crash log.

Classification:

```text
PASS for isolated app-home RF-DETR/YOLO model discovery.
```

This verifies that the rebuilt packaged app no longer relies on the repository
root to discover RF-DETR and YOLO weights.

Artifacts:

```text
dist_test\new-install-sandbox\full-gui-rf-yolo-fixtures-20260512\
```

## 6. Current Classification

The rebuilt full GUI is now in this state:

```text
compile/package: PASS
OpenCV DLL post-copy: PASS
empty new-install setup-wizard liveness: PASS
isolated RF-DETR/YOLO model discovery: PASS
SAM3 real gated/token/download flow: PASS (see section 9)
RF-DETR/YOLO real download flow: PASS (see section 9)
visual setup-wizard inspection: PASS for initial missing-model screen
complete blank app-state vacuum: FAIL in this artifact; fixed in source after build
```

The specific previous blocker:

```text
TypeError: 'ReconstructionZone' object is not callable
```

did not recur in the isolated full-GUI probes.

## 7. Remaining Release Gates

Still required before calling this friend-test/release ready:

1. Visual inspection of the setup wizard screens in the rebuilt full GUI.
2. Real RF-DETR download path test, or a deliberate decision to ship with
   pre-provisioned RF-DETR weights.
3. Real YOLO26 download path test, or a deliberate decision to ship with
   pre-provisioned YOLO26 weights.
4. SAM3 gated/token UX test:
   - no token;
   - bad/no-access token;
   - approved token/cache path.
5. Relaunch-after-setup test once all required models are installed or
   deliberately bypassed.
6. Confirm no app data is written beside the executable during normal use.

## 9. Real Model Install Workflow Update

Codex validated the real RF-DETR, YOLO26, and SAM3 model workflows in isolated
app homes.

Summary:

```text
RF-DETR real download: pass
YOLO26 real download: pass
SAM3 no-token state: pass
SAM3 saved-token verification: pass
SAM3 gated access: pass
SAM3 real weight download: pass
packaged GUI all-models-ready recognition: pass
```

Detailed report:

```text
planning/plans/2026-05-12-real-model-install-workflow-result.md
```

## 10. Visual Vacuum Probe And New App-State Finding

Codex ran a visible launch of this packaged artifact with an empty isolated
app-home, model directory, and HuggingFace cache:

```text
case:
visual-empty-vacuum-setup-wizard-20260512

artifact:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Result:

```text
PROCESS_STILL_RUNNING after 18 seconds
setup wizard visible
models listed as missing: SAM3, RF-DETR-Seg, YOLO26-n-seg
wizard readiness: 0/3 ready
app log: Models need setup: SAM3, RF-DETR-Seg, YOLO26-n-seg
fatal traceback: none
```

Evidence:

```text
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-setup-wizard-20260512\screenshots\desktop-after-warmup.png
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-setup-wizard-20260512\launch-logs\stderr.txt
```

Classification:

```text
PASS for visible packaged missing-model setup-wizard display.
```

However, the same screenshot showed the Projects tab behind the wizard reading:

```text
D:\tracker.json
```

That means this artifact is not a complete blank app-state vacuum. Model setup
was isolated and correct; project-library state was still reading the legacy
default tracker path.

Source fix applied after this build:

```text
reconstruction_gui\app_paths.py:
  project_store_file()
  activity_store_file()

reconstruction_gui\tabs\projects_tab.py:
  default tracker path now uses app-home

reconstruction_gui\project_store.py:
  ProjectStore() default now uses app-home
```

Verification of the source fix:

```text
project_store_file(): isolated app-home\projects\tracker.json
activity_store_file(): isolated app-home\projects\activity_log.json
ProjectStore().store_path: isolated app-home\projects\tracker.json
py_compile: PASS with project-local PYTHONPYCACHEPREFIX
```

Consequence:

```text
This packaged artifact remains good evidence for the fixed setup-wizard display
and real model workflow behavior, but it should not be treated as the final
new-install candidate. A final rebuild is required to include the project-store
vacuum fix and the strict model-dir testing toggle added after this artifact.
```

## 12. Superseded By Final-Vacuum Artifact

Codex ran the required final rebuild after the project-store vacuum fix.

Final artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Final result:

```text
build exit code: 0
report completion: yes
OpenCV CUDA/NVIDIA DLL copy: pass
visible blank-vacuum setup wizard: pass
project-store vacuum: pass
all-real-models-ready recognition: pass
```

Detailed report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-finalvacuum-result.md
```

## 11. Process Hygiene

After the build and probes, no lingering processes were expected. Final process
checks should continue to include:

```text
ReconstructionZone
smoke_setup_wizard
python
cl
link
scons
git
```
