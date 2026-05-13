# Real Model Install Workflow Result

Date: 2026-05-12  
Owner: Codex  
Repository: `D:\Projects\reconstruction-zone`  
Status: **RF-DETR, YOLO26, SAM3 token/access/download workflows validated in isolated app homes**

## 1. Purpose

This report records the release-gate work for real model install and setup
wizard workflows after the full GUI new-install build passed.

Validated workflows:

- RF-DETR-Seg real download;
- YOLO26 real download through HuggingFace Hub;
- SAM3 no-token state;
- SAM3 saved-token verification;
- SAM3 gated model access;
- SAM3 real weight download through HuggingFace Hub;
- packaged GUI launch against an isolated app-home containing all required
  model assets.

No HuggingFace token values were printed or written into this report.

## 2. Safety Boundaries

All workflow tests used isolated directories under:

```text
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox
```

The tests did not delete real user app data.

The tests did not delete:

```text
%LOCALAPPDATA%\ReconstructionZone
C:\Users\alexm\.reconstruction_zone
the user's default HuggingFace cache
```

## 3. Scripts Added Or Updated

Added:

```text
scripts\validate_model_download_workflows.py
scripts\validate_sam3_workflow.py
```

Updated:

```text
reconstruction_gui\model_paths.py
scripts\run_packaged_new_install_probe.ps1
```

Important test-hardening change:

```text
RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS=1
```

When set, source-level model resolution ignores development fallbacks such as
the repository root. This prevents repo-root model files from masking missing
models during new-install validation.

Note:

- this strict test toggle was added after the current full GUI
  `phase-6b-full-gui-newinstallfix-20260512` build;
- the tested rebuilt GUI artifact still passed `.dist` / isolated app-home
  launch probes because it used explicit app-home/model/HF paths;
- a final release build should be run after this source change if exact
  source/artifact parity is required.

## 4. RF-DETR + YOLO26 Real Download Workflow

Command:

```powershell
C:\Python314\python.exe scripts\validate_model_download_workflows.py `
  --case-name real-rfdetr-yolo-downloads-strict-20260512 `
  --reset
```

Case root:

```text
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\real-rfdetr-yolo-downloads-strict-20260512
```

Initial model readiness:

```text
{'sam3': False, 'rfdetr': False, 'yolo26': False}
```

RF-DETR result:

```text
RFDETR_DOWNLOAD_START
QUEUE ('status', 'Downloading RF-DETR-Seg-Small (129 MB)...')
rf-detr-seg-small.pt: 135,042,342 bytes
RFDETR_DOWNLOAD_OK
```

YOLO26 result:

```text
YOLO26_DOWNLOAD_START
QUEUE ('status', 'Downloading YOLO26-n-seg (6.5 MB)...')
yolo26n-seg.pt: 6,719,965 bytes
YOLO26_DOWNLOAD_OK
```

Final model readiness:

```text
{'sam3': False, 'rfdetr': True, 'yolo26': True}
```

Final SAM3 state at this stage:

```text
overall_stage='needs_token'
message='SAM 3 requires a HuggingFace token.'
```

Classification:

```text
PASS
```

## 5. Packaged GUI Recognition Of Real RF-DETR + YOLO26 Downloads

Command:

```powershell
.\scripts\run_packaged_new_install_probe.ps1 `
  -ExePath D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe `
  -CaseName real-rfdetr-yolo-downloads-strict-20260512 `
  -DurationSeconds 20
```

Harness result:

```text
PROCESS_STILL_RUNNING
PROCESS_STOPPED by harness after 20 seconds
```

Captured app log:

```text
Models need setup: SAM3
```

Isolated model directory:

```text
rf-detr-seg-small.pt: 135,042,342 bytes
yolo26n-seg.pt: 6,719,965 bytes
```

Crash log:

```text
no fatal traceback
```

Classification:

```text
PASS
```

## 6. SAM3 No-Token Workflow

Command:

```powershell
C:\Python314\python.exe scripts\validate_sam3_workflow.py `
  --case-name sam3-no-token-isolated-20260512 `
  --reset
```

Result:

```text
overall_stage='needs_token'
message='SAM 3 requires a HuggingFace token.'
next_action='Paste a HuggingFace token, then click Verify Access.'
```

Classification:

```text
PASS
```

## 7. SAM3 Saved-Token Verification Workflow

Command:

```powershell
C:\Python314\python.exe scripts\validate_sam3_workflow.py `
  --case-name sam3-token-verify-isolated-20260512 `
  --reset `
  --use-saved-token
```

The validator reads the saved token without printing it, then resets `HF_HOME`
to the isolated test case before importing HuggingFace Hub APIs.

Result:

```text
SAVED_TOKEN_PRESENT True
VERIFY_SAVED_TOKEN_REPORT:
  token_status='verified'
  access_status='granted'
  runtime_status='installed'
  weights_status='missing'
  overall_stage='needs_weights'
  message='SAM 3 runtime installed but model weights are missing.'
```

Classification:

```text
PASS
```

## 8. SAM3 Real Weight Download Workflow

Command:

```powershell
C:\Python314\python.exe scripts\validate_sam3_workflow.py `
  --case-name sam3-download-isolated-20260512 `
  --reset `
  --use-saved-token `
  --download
```

Result:

```text
SAVED_TOKEN_PRESENT True
SAM3_DOWNLOAD_START
SAM3_PROGRESS Downloading SAM 3 weights (~3.5 GB)...
SAM3_PROGRESS SAM 3 weights downloaded.
SAM3_DOWNLOAD_RESULT True
FINAL_SAM3_SETUP:
  token_status='verified'
  access_status='granted'
  runtime_status='installed'
  weights_status='present'
  overall_stage='ready'
  message='SAM 3 is ready to use.'
SAM3_WORKFLOW_OK
```

HuggingFace snapshot result:

```text
Fetching 12 files: 100%
```

Classification:

```text
PASS
```

## 9. Packaged GUI Recognition Of All Real Model Assets

After SAM3 weights were downloaded, the real RF-DETR and YOLO26 files from the
strict download case were copied into the same isolated app-home model directory:

```text
sam3-download-isolated-20260512\app-home\models
```

Then the rebuilt packaged GUI was launched against that isolated app-home:

```powershell
.\scripts\run_packaged_new_install_probe.ps1 `
  -ExePath D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe `
  -CaseName sam3-download-isolated-20260512 `
  -DurationSeconds 20
```

Harness result:

```text
PROCESS_STILL_RUNNING
PROCESS_STOPPED by harness after 20 seconds
```

Captured app log:

```text
All models ready — setup wizard not needed
```

Crash log:

```text
no fatal traceback
```

Classification:

```text
PASS
```

## 10. Current Release-Gate Classification

The following are now validated:

```text
full GUI build: PASS
OpenCV CUDA/NVIDIA DLL post-copy: PASS
empty new-install setup wizard liveness: PASS
RF-DETR real download: PASS
YOLO26 real download: PASS
SAM3 no-token state: PASS
SAM3 token/access verification: PASS
SAM3 real weight download: PASS
packaged GUI all-models-ready recognition: PASS
```

Still pending before calling this truly friend-test ready:

```text
visual setup-wizard inspection
manual click-through of wizard screens
final full rebuild after source changes made after the current artifact
```

## 12. Follow-Up Visual Vacuum Finding

A visible launch of the rebuilt packaged app in an empty isolated model/cache
state did show the setup wizard correctly:

```text
case:
visual-empty-vacuum-setup-wizard-20260512

wizard state:
SAM3, RF-DETR-Seg, YOLO26-n-seg missing
0/3 ready
Begin Setup visible
fatal traceback: none
```

However, the screenshot also showed the Projects tab behind the wizard reading
the legacy project tracker:

```text
D:\tracker.json
```

That is unrelated to model download readiness, but it is relevant to the
"blank new install" release gate.

Source fix applied after the current packaged artifact:

```text
project tracker default -> app-home\projects\tracker.json
activity log default -> app-home\projects\activity_log.json
```

Consequence:

```text
The model workflow result remains valid.
The current packaged artifact is no longer the final release candidate.
A final rebuild is required to include the project-store vacuum fix and the
strict-model-dir test toggle.
```

## 13. Final Artifact Follow-Up

Codex rebuilt the full GUI after the project-store vacuum fix.

Final artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

The final artifact was launched against the existing real-model sandbox:

```text
case:
sam3-download-isolated-20260512

log marker:
All models ready — setup wizard not needed
```

Classification:

```text
PASS
```

Detailed final artifact report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-finalvacuum-result.md
```

## 11. Notes

The workflow tests produced a non-blocking warning from `timm`:

```text
FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
```

This warning did not cause any test failure.

Final process hygiene checks should continue to include:

```text
ReconstructionZone
smoke_setup_wizard
python
cl
link
scons
git
```
