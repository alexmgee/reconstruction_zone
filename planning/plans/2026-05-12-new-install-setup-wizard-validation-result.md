# New-Install Setup Wizard Validation Result

Date: 2026-05-12  
Owner: Codex  
Repository: `D:\Projects\reconstruction-zone`  
Status: **setup-wizard root cause fixed; final packaged blank-vacuum probe passed**

## 1. What Was Fixed

The fatal setup-wizard error was not primarily a Nuitka compiler bug.

Root cause:

```text
SetupWizard.__init__ assigned self._root = parent
```

Tkinter widgets already rely on an internal `_root()` method. Assigning
`self._root` on a `CTkToplevel` instance shadowed that method. When Tkinter later
handled an event and tried to call `_root()`, it instead found the parent
window object and raised:

```text
TypeError: 'CTk' object is not callable
```

In the full packaged GUI, the same bug appeared as:

```text
TypeError: 'ReconstructionZone' object is not callable
```

Fix:

```text
self._root -> self._app_root
```

Changed file:

```text
reconstruction_gui/setup_wizard.py
```

## 2. App-Home Path Work Completed

Added a shared app path resolver:

```text
reconstruction_gui/app_paths.py
```

It provides:

```text
app_home()
model_dir()
log_dir()
crash_log_file()
prefs_dir()
prefs_file()
bundled_model_dirs()
project_store_file()
activity_store_file()
```

Default Windows app-home:

```text
%LOCALAPPDATA%\ReconstructionZone
```

Isolated-test override:

```text
RECONSTRUCTION_ZONE_APP_HOME
```

Model-only compatibility override:

```text
RECONSTRUCTION_ZONE_MODEL_DIR
```

Updated:

```text
reconstruction_gui/app_infra.py
reconstruction_gui/model_paths.py
reconstruction_gui/reconstruction_zone.py
reconstruction_gui/docs/ARCHITECTURE.md
```

Behavioral changes:

- crash logs now resolve through app-home logs;
- prefs now resolve through app-home prefs;
- model paths resolve through app-home models unless explicitly overridden;
- project tracker defaults now resolve through app-home projects instead of
  `D:\tracker.json`;
- activity-log defaults now resolve through app-home projects instead of beside
  the legacy `D:\tracker.json` path;
- packaged model lookup now prefers an intentional bundled `models` directory
  rather than misidentifying the compiled bundle as the source checkout;
- file logger failure no longer crashes app import; it falls back to a
  `NullHandler`.

## 3. New Scripts Added

### 3.1 Source / packaged setup-wizard smoke

```text
scripts/smoke_setup_wizard.py
```

Modes:

```text
--mode direct
--mode if-needed
```

`direct` verifies:

- import;
- root creation;
- `SetupWizard(parent, models)`;
- `transient(parent)`;
- `grab_set()`;
- `focus()`;
- short `mainloop()`.

`if-needed` verifies:

- import;
- root creation;
- `run_setup_wizard_if_needed(parent)`;
- missing-model wizard path;
- short `mainloop()`.

### 3.2 Packaged new-install probe harness

```text
scripts/run_packaged_new_install_probe.ps1
```

Purpose:

- launch a packaged exe with isolated app-home/model/cache directories;
- optionally copy fixture models;
- capture stdout/stderr, crash-log tail, model directory contents, and process
  lifecycle;
- avoid touching the real user app data.

### 3.3 Visual vacuum setup-wizard probe harness

```text
scripts/run_visual_vacuum_setup_probe.ps1
```

Purpose:

- launch the GUI visibly with isolated app-home/model/HF cache directories;
- capture desktop and main-window screenshots after warmup;
- capture stdout/stderr, crash-log tail, model directory listing, and process
  lifecycle;
- reset only the named case directory under
  `dist_test\new-install-sandbox`.

## 4. Source-Level Verification

### 4.1 Syntax checks

Command shape:

```powershell
$env:PYTHONPYCACHEPREFIX='D:\Projects\reconstruction-zone\dist_test\pycache-syntax'
python -m py_compile `
  reconstruction_gui\app_paths.py `
  reconstruction_gui\app_infra.py `
  reconstruction_gui\model_paths.py `
  reconstruction_gui\setup_wizard.py `
  reconstruction_gui\reconstruction_zone.py `
  scripts\smoke_setup_wizard.py
```

Result:

```text
PASS
```

### 4.2 Local import audit

Command:

```powershell
C:\Python314\python.exe scripts\audit_local_imports.py
```

Result:

```text
No unresolved local imports found.
```

### 4.3 Source direct setup-wizard smoke

Command:

```powershell
C:\Python314\python.exe scripts\smoke_setup_wizard.py --mode direct
```

Result:

```text
exit code: 0
SETUP_WIZARD_IMPORT_OK
SETUP_WIZARD_ROOT_OK
SETUP_WIZARD_CONSTRUCTED
SETUP_WIZARD_MODAL_APPLIED
SETUP_WIZARD_MAINLOOP_OK
```

### 4.4 Source API-path setup-wizard smoke

Command:

```powershell
C:\Python314\python.exe scripts\smoke_setup_wizard.py --mode if-needed
```

Result:

```text
exit code: 0
SETUP_WIZARD_IMPORT_OK
SETUP_WIZARD_API_ROOT_OK
SETUP_WIZARD_IF_NEEDED_RETURNED False
SETUP_WIZARD_IF_NEEDED_MAINLOOP_OK
```

Interpretation:

- `False` is expected in the missing-model path.
- The important result is that the wizard opens, runs its event loop briefly,
  and exits without the Tk `_root()` fatal.

## 5. Packaged Setup-Wizard Smoke

### 5.1 First packaging attempt

The first attempt included:

```text
--include-package=huggingface_hub
```

Result:

```text
TIMEOUT after 60 minutes
```

Observed issue:

- including the full HuggingFace Hub package dragged in unwanted ML-adjacent
  packages including `bitsandbytes`, `peft`, `torch`, and `triton`;
- no leftover `python`, compiler, linker, SCons, or Git processes remained after
  the timeout.

Classification:

```text
Rejected as an over-broad setup-wizard smoke command.
```

### 5.2 Corrected lightweight packaged smoke

Command shape:

```powershell
$env:NUITKA_CACHE_DIR='D:\Projects\reconstruction-zone\dist_test\nuitka-cache'

C:\Python314\python.exe -m nuitka `
  --standalone `
  --assume-yes-for-downloads `
  --windows-console-mode=attach `
  --enable-plugin=tk-inter `
  --enable-plugin=no-qt `
  --module-parameter=torch-disable-jit=yes `
  --nofollow-import-to=huggingface_hub `
  --nofollow-import-to=huggingface_hub.* `
  --nofollow-import-to=torch `
  --nofollow-import-to=torch.* `
  --nofollow-import-to=transformers `
  --nofollow-import-to=transformers.* `
  --nofollow-import-to=peft `
  --nofollow-import-to=peft.* `
  --nofollow-import-to=bitsandbytes `
  --nofollow-import-to=bitsandbytes.* `
  --nofollow-import-to=triton `
  --nofollow-import-to=triton.* `
  --include-package=customtkinter `
  --include-module=reconstruction_gui.setup_wizard `
  --include-module=reconstruction_gui.model_paths `
  --include-module=reconstruction_gui.app_paths `
  --output-dir=dist_test\setup-wizard-smoke-pkg-20260512 `
  --report=dist_test\setup-wizard-smoke-pkg-20260512\setup-wizard-smoke-pkg-report.xml `
  scripts\smoke_setup_wizard.py
```

Result:

```text
exit code: 0
report completion: yes
compiled/linked C files: 144
clcache hits: 136
clcache misses: 8
```

Artifacts:

```text
dist_test\setup-wizard-smoke-pkg-20260512\smoke_setup_wizard.dist\smoke_setup_wizard.exe
dist_test\setup-wizard-smoke-pkg-20260512\setup-wizard-smoke-pkg-report.xml
```

Artifact sizes:

```text
smoke_setup_wizard.exe: 16,395,776 bytes
setup-wizard-smoke-pkg-report.xml: 742,595 bytes
```

## 6. Packaged Smoke Runtime Results

The packaged smoke had to be run outside the sandbox because the sandbox cannot
reliably host the Tk executable window. The runs were bounded and exited on
their own.

### 6.1 Direct mode

Capture files:

```text
dist_test\setup-wizard-smoke-pkg-20260512\direct-capture\stdout.txt
dist_test\setup-wizard-smoke-pkg-20260512\direct-capture\stderr.txt
```

Result:

```text
exit code: 0
stderr: 0 bytes
SETUP_WIZARD_IMPORT_OK
SETUP_WIZARD_ROOT_OK
SETUP_WIZARD_CONSTRUCTED
SETUP_WIZARD_MODAL_APPLIED
SETUP_WIZARD_MAINLOOP_OK
```

### 6.2 API `run_setup_wizard_if_needed` mode

Capture files:

```text
dist_test\setup-wizard-smoke-pkg-20260512\if-needed-capture\stdout.txt
dist_test\setup-wizard-smoke-pkg-20260512\if-needed-capture\stderr.txt
```

Result:

```text
exit code: 0
stderr: 0 bytes
SETUP_WIZARD_IMPORT_OK
SETUP_WIZARD_API_ROOT_OK
SETUP_WIZARD_IF_NEEDED_RETURNED False
SETUP_WIZARD_IF_NEEDED_MAINLOOP_OK
```

## 7. Current Classification

The setup-wizard `_root` crash is fixed and validated in:

1. source direct wizard construction;
2. source `run_setup_wizard_if_needed`;
3. packaged direct wizard construction;
4. packaged `run_setup_wizard_if_needed`.

This does **not** mean the full GUI package is release-ready yet.

It means the next full GUI build is now justified, because the known
first-launch setup-wizard blocker has been fixed in a small bounded proof.

## 8. Still Pending

Before claiming new-install release readiness:

1. Rebuild the full GUI with the proven Phase 6B command family.
2. Run OpenCV CUDA/NVIDIA DLL post-copy.
3. Use `scripts\run_packaged_new_install_probe.ps1` against the rebuilt exe.
4. Run at least these isolated cases:
   - empty app-home;
   - fixture models present;
   - `.dist` working-directory launch;
   - app-home logs/prefs/models creation;
   - relaunch after fixture models.
5. Validate real model download paths separately:
   - RF-DETR;
   - YOLO26;
   - SAM3 gated/token path.

## 8A. Full GUI Rebuild Follow-Up

Codex rebuilt the full GUI after the setup-wizard fix.

Result:

```text
build exit code: 0
report completion: yes
artifact:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Post-build OpenCV CUDA/NVIDIA DLL copy passed.

Isolated new-install probes also passed:

1. Empty app-home:
   - process stayed alive for 20 seconds;
   - logged `Models need setup: SAM3, RF-DETR-Seg, YOLO26-n-seg`;
   - no fatal traceback in isolated crash log.

2. RF-DETR/YOLO fixture app-home:
   - process stayed alive for 20 seconds;
   - logged `Models need setup: SAM3`;
   - isolated model dir contained `rf-detr-seg-small.pt` and `yolo26n-seg.pt`;
   - no fatal traceback in isolated crash log.

Detailed rebuilt-artifact report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-newinstallfix-result.md
```

## 8B. Real Model Workflow Follow-Up

Codex validated real model installation workflows after the full GUI rebuild.

Validated:

- RF-DETR-Seg real download into isolated app-home;
- YOLO26 real download into isolated app-home;
- SAM3 no-token state;
- SAM3 saved-token verification without printing token values;
- SAM3 gated access;
- SAM3 real weight download into isolated HF home;
- rebuilt packaged GUI launch with all model assets ready.

Final packaged GUI marker:

```text
All models ready — setup wizard not needed
```

Detailed report:

```text
planning/plans/2026-05-12-real-model-install-workflow-result.md
```

## 8C. Visual Vacuum Probe Follow-Up

Codex then ran a visible packaged GUI launch with a completely empty isolated
model directory and isolated HuggingFace cache:

```text
case:
visual-empty-vacuum-setup-wizard-20260512
artifact:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Captured environment:

```text
APP_HOME=D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-setup-wizard-20260512\app-home
MODEL_DIR=D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-setup-wizard-20260512\app-home\models
STRICT_MODEL_DIRS=1
HF_HOME=D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-setup-wizard-20260512\hf-home
TRANSFORMERS_CACHE=D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-setup-wizard-20260512\hf-home\transformers
```

Harness result:

```text
PROCESS_STILL_RUNNING
MAIN_WINDOW_HANDLE present
desktop screenshot captured
main-window screenshot captured
PROCESS_STOPPED by harness
```

Visual result:

- the packaged setup wizard is visible;
- it lists `SAM3`, `RF-DETR-Seg`, and `YOLO26-n-seg`;
- it shows `0/3 ready`;
- it offers `Begin Setup`;
- the app log says `Models need setup: SAM3, RF-DETR-Seg, YOLO26-n-seg`;
- no fatal traceback was written.

Evidence:

```text
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-setup-wizard-20260512\screenshots\desktop-after-warmup.png
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-setup-wizard-20260512\screenshots\main-window-after-warmup.png
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-setup-wizard-20260512\launch-logs\stderr.txt
```

Classification:

```text
PASS for visible missing-model setup-wizard display in the rebuilt packaged app.
```

Important caveat:

The screenshot also showed the Projects tab behind the wizard reading the legacy
default project tracker:

```text
D:\tracker.json
```

That means the model/setup-wizard state was vacuum-isolated, but the packaged
artifact was not a complete new-user vacuum across all app state.

## 8D. Project-Store Vacuum Fix

Root cause:

```text
reconstruction_gui\tabs\projects_tab.py defaulted tracker_store_path to D:\tracker.json
reconstruction_gui\project_store.py defaulted ProjectStore() to D:\tracker.json
```

Fix applied in source:

```text
reconstruction_gui\app_paths.py:
  project_store_file()
  activity_store_file()

reconstruction_gui\tabs\projects_tab.py:
  default tracker path now uses project_store_file()
  default activity path now uses activity_store_file()

reconstruction_gui\project_store.py:
  ProjectStore() default now uses project_store_file()
```

Compatibility:

- existing users with explicit `tracker_store_path` in preferences keep that
  path;
- new users without that preference get an app-home project store.

Source verification:

```text
RECONSTRUCTION_ZONE_APP_HOME=D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\source-path-check-20260512\app-home

project_store_file():
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\source-path-check-20260512\app-home\projects\tracker.json

activity_store_file():
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\source-path-check-20260512\app-home\projects\activity_log.json

ProjectStore().store_path:
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\source-path-check-20260512\app-home\projects\tracker.json
```

Syntax verification:

```text
py_compile with project-local PYTHONPYCACHEPREFIX: PASS
```

Source visible vacuum launch after the fix:

```text
case:
source-visual-empty-vacuum-setup-wizard-20260512

log marker:
Models need setup: SAM3, RF-DETR-Seg, YOLO26-n-seg

app-home contents:
logs
models
prefs
logs\crash.log
prefs\studio_prefs.json
```

No `projects` store was created during this passive first-launch probe because
no project was saved.

Current consequence:

```text
The current full GUI artifact predates this project-store fix.
Another full GUI rebuild is required before claiming complete blank-new-install
parity for the packaged app.
```

## 8E. Final Packaged Vacuum Rebuild Follow-Up

Codex ran the final parity rebuild after the project-store vacuum fix.

Artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Build result:

```text
exit code: 0
report completion: yes
compiled C files: 9494
clcache hits: 9486
clcache misses: 8
```

Visible blank-vacuum probe:

```text
case:
visual-empty-vacuum-finalvacuum-20260512

setup wizard visible: pass
missing models shown: SAM3, RF-DETR-Seg, YOLO26-n-seg
ready count: 0/3
Projects tab: 0 projects
D:\tracker.json leakage: none found
fatal traceback: none
```

The final artifact also recognized the existing real-model sandbox:

```text
All models ready — setup wizard not needed
```

Detailed final report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-finalvacuum-result.md
```

## 9. Process Hygiene

Final process checks after this work found no lingering:

```text
ReconstructionZone
smoke_setup_wizard
python
cl
link
scons
git
```

## 10. Files Changed Or Added

Added:

```text
reconstruction_gui/app_paths.py
scripts/smoke_setup_wizard.py
scripts/run_packaged_new_install_probe.ps1
scripts/run_visual_vacuum_setup_probe.ps1
planning/plans/2026-05-12-new-install-setup-wizard-validation-result.md
```

Updated:

```text
reconstruction_gui/app_infra.py
reconstruction_gui/model_paths.py
reconstruction_gui/project_store.py
reconstruction_gui/setup_wizard.py
reconstruction_gui/reconstruction_zone.py
reconstruction_gui/tabs/projects_tab.py
reconstruction_gui/docs/ARCHITECTURE.md
```
