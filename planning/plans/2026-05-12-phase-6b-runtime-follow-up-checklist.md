# Phase 6B Runtime Follow-Up Checklist

Date: 2026-05-12  
Scope: follow-up work after the full GUI Nuitka build succeeded but the
compiled first-launch/setup-wizard path failed.

## 1. Current Ground Truth

Do not rerun the full GUI build as the next step.

Verified:

- The full GUI Nuitka compile completed successfully.
- The produced XML report has `completion="yes"`.
- The packaged executable exists at:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

- OpenCV CUDA/NVIDIA DLL post-copy completed.
- Outside-sandbox startup stays alive when model lookup is satisfied.

Blocking runtime issue:

- A `.dist` working-directory launch with missing RF-DETR/YOLO26 weights enters
  the setup-wizard path and writes:

```text
TypeError: 'ReconstructionZone' object is not callable
```

## 2. Immediate User-Test Path

For local testing of the current artifact, satisfy model lookup first:

```powershell
$env:RECONSTRUCTION_ZONE_MODEL_DIR='D:\Projects\reconstruction-zone'
& 'D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist\ReconstructionZone.exe'
```

This is a local testing workaround only. It is not a release solution.

## 3. Reproduce The Setup-Wizard Failure Cheaply

Create a small smoke script that:

1. imports `customtkinter`;
2. imports `SetupWizard` and `ModelInfo`;
3. creates a minimal parent `ctk.CTk`;
4. creates synthetic missing models:

```python
[
    ModelInfo("RF-DETR-Seg", "rfdetr", "129 MB", ready=False),
    ModelInfo("YOLO26-n-seg", "yolo26", "6.5 MB", ready=False),
]
```

5. constructs `SetupWizard(parent, models)`;
6. calls:

```python
wizard.transient(parent)
wizard.grab_set()
wizard.focus()
```

7. runs a short `after()`-based lifecycle so the smoke exits by itself.

## 4. Package The Minimal Smoke

Compile that smoke with:

```text
NUITKA_CACHE_DIR=D:\Projects\reconstruction-zone\dist_test\nuitka-cache
--standalone
--windows-console-mode=attach
--enable-plugin=tk-inter
--enable-plugin=no-qt
--include-package=customtkinter
--include-package=reconstruction_gui
```

Do not include `torch`, `rfdetr`, `ultralytics`, or full GUI packages unless the
smoke proves they are required. The point is to isolate Tk/CustomTkinter wizard
behavior.

## 5. Interpret The Smoke

If the minimal smoke reproduces:

- fix the setup-wizard/Tk interaction before rebuilding the full app;
- likely suspects to inspect:
  - `wizard.transient(parent)`;
  - `wizard.grab_set()`;
  - event bindings created during wizard construction;
  - any callback that passes the `ReconstructionZone` instance where Tk expects
    a Tcl widget name.

If the minimal smoke does not reproduce:

- add the next smallest part of the real app startup path:
  - `run_setup_wizard_if_needed(parent)`;
  - model readiness check;
  - parent app class shape;
  - startup `after(500, self._run_setup_wizard)` scheduling.

## 6. Release-Correct Model Provisioning

Choose one release path deliberately:

1. Installer/download path:
   - weights live under `%LOCALAPPDATA%\ReconstructionZone\models`;
   - setup wizard downloads or verifies them;
   - missing-model `.dist` launch must show a working wizard without traceback.

2. Bundled/testing path:
   - include or copy weights into a documented bundle/model directory;
   - update `candidate_model_dirs()` or installer layout so `.dist` launch finds
     them without relying on the source checkout as current working directory.

Do not rely on `D:\Projects\reconstruction-zone` as a model path for release.

## 7. Pass Criteria Before Another Full Build

The next full GUI build is justified only after:

- the minimal setup-wizard smoke passes, or the setup wizard is intentionally
  bypassed by a release-correct model provisioning path;
- a `.dist` working-directory launch with missing or app-dir models does not
  write the Tk `TypeError`;
- the persistent crash log has no new fatal tracebacks from the probe;
- no leftover `ReconstructionZone`, `python`, `cl`, `link`, `scons`, or `git`
  processes are running from the test.

## 8. Documentation To Update After The Fix

After fixing or bypassing the setup-wizard path, update:

```text
planning/plans/2026-05-12-phase-6b-full-gui-build-result.md
planning/plans/2026-05-11-remediation-progress-report.md
planning/plans/2026-05-11-next-codex-handoff.md
planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md
```

## 9. Execution Update, 2026-05-12

The setup-wizard `_root` collision has been fixed:

```text
SetupWizard.self._root -> SetupWizard.self._app_root
```

Validated:

- source direct setup-wizard smoke: pass;
- source `run_setup_wizard_if_needed` smoke: pass;
- packaged direct setup-wizard smoke: pass;
- packaged `run_setup_wizard_if_needed` smoke: pass.

The full GUI has not yet been rebuilt with this fix. The next appropriate step
is a full GUI rebuild using the proven Phase 6B command family, followed by the
new isolated install harness.

Detailed result:

```text
planning/plans/2026-05-12-new-install-setup-wizard-validation-result.md
```
