# New-Install Setup Wizard Validation Plan

Date: 2026-05-12  
Owner: Codex  
Repository: `D:\Projects\reconstruction-zone`  
Purpose: make the packaged app testable from a safe, ground-up "new user"
state and prove the first-launch setup wizard works as intended.

## 1. Current Ground Truth

The full GUI Nuitka build has crossed the compile barrier.

Verified artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Verified build facts:

- Nuitka exit code: `0`
- XML report: `completion="yes"`
- Build strategy:
  - workspace `NUITKA_CACHE_DIR`;
  - no `--include-package=transformers`;
  - no `--nofollow-import-to=transformers.models.*`;
  - torch YAML fix retained.
- OpenCV CUDA/NVIDIA DLL copy completed.
- The app stays alive outside the sandbox when model lookup is satisfied.

Verified blocker:

- A `.dist` working-directory launch without RF-DETR/YOLO26 model weights enters
  the setup wizard path.
- That path writes a fatal Tk callback error:

```text
TypeError: 'ReconstructionZone' object is not callable
```

Therefore:

- **Compile/package strategy:** passed.
- **New-install release readiness:** not passed.
- **Next work:** setup wizard + clean app-data test harness, not another blind
  full GUI build.

## 2. Definition Of A Real "New Install" Test

A valid new-install test must not depend on:

- the source checkout as the current working directory;
- model weights sitting in `D:\Projects\reconstruction-zone`;
- existing files in `%LOCALAPPDATA%\ReconstructionZone`;
- existing HuggingFace cache contents unless the test explicitly says it is a
  cache-hit test;
- existing `C:\Users\alexm\.reconstruction_zone\crash.log`;
- existing `.studio_prefs.json` in the source tree or bundle.

A valid new-install test must prove:

1. The packaged app starts from its `.dist` directory.
2. It can create and write all required user data/log/model files.
3. Missing model weights are detected accurately.
4. The setup wizard opens without Tk/Tcl callback errors.
5. The wizard can handle:
   - already-ready models;
   - missing downloadable models;
   - failed downloads;
   - retry;
   - SAM3 gated/token flow.
6. After setup succeeds, a relaunch sees the models as ready and does not show
   the wizard again unless intentionally reset.

## 3. Safety Requirements

Do not test "new install" by deleting or mutating the user's real app data.

Do not delete:

```text
C:\Users\alexm\.reconstruction_zone
%LOCALAPPDATA%\ReconstructionZone
%USERPROFILE%\.cache\huggingface
%USERPROFILE%\.cache\ultralytics
```

unless the user explicitly asks for that exact destructive cleanup.

Instead, tests should use isolated directories under the repo:

```text
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\
```

Suggested layout:

```text
dist_test\new-install-sandbox\
  appdata\
    ReconstructionZone\
      models\
      logs\
      prefs\
  hf-home\
  launch-logs\
  probe-results\
```

The test harness should set environment variables for the launched process so
the packaged app behaves like a new user without touching the real user profile.

## 4. Required App-Data Refactor

The current code has split storage behavior:

- model weights:
  - `RECONSTRUCTION_ZONE_MODEL_DIR`, then `%LOCALAPPDATA%\ReconstructionZone\models`;
- crash log:
  - `Path.home() / ".reconstruction_zone" / "crash.log"`;
- GUI preferences:
  - `_this_dir / ".studio_prefs.json"`;
- HuggingFace cache:
  - normal HuggingFace defaults unless `HF_HOME`/cache args are set.

This makes clean new-install testing unnecessarily risky and uneven.

### 4.1 Add a single app-home resolver

Create a small shared module, for example:

```text
reconstruction_gui/app_paths.py
```

Responsibilities:

- expose `app_home(create=False)`;
- expose `model_dir(create=False)`;
- expose `log_dir(create=False)`;
- expose `prefs_file(create=False)`;
- expose `diagnostics_dir(create=False)` if useful.

Proposed priority:

1. `RECONSTRUCTION_ZONE_APP_HOME`, for isolated tests and portable builds.
2. `%LOCALAPPDATA%\ReconstructionZone`, for normal Windows installs.
3. `Path.home() / ".reconstruction_zone"`, as a fallback.

Keep `RECONSTRUCTION_ZONE_MODEL_DIR` as a narrower model-only override for
compatibility, but prefer `app_home() / "models"` when it is not set.

### 4.2 Move crash logging to app-home

Update:

```text
reconstruction_gui/app_infra.py
reconstruction_gui/reconstruction_zone.py
```

so crash logs use:

```text
app_home(create=True) / "logs" / "crash.log"
```

This fixes two things:

- normal release behavior becomes consistent;
- test harnesses can safely redirect logs to a disposable directory.

### 4.3 Move preferences to app-home

Current source:

```text
ReconstructionZone._PREFS_FILE = _this_dir / ".studio_prefs.json"
```

This is risky in a packaged app because the bundle directory may be read-only,
or may be shared by multiple users.

Move preferences to:

```text
app_home(create=True) / "prefs" / "studio_prefs.json"
```

Keep a one-time compatibility import from the old source/bundle prefs file only
if needed, but do not write new prefs beside the executable.

### 4.4 Make model lookup release-oriented

Update `candidate_model_dirs()` so the default release search order is:

1. explicit `RECONSTRUCTION_ZONE_MODEL_DIR`;
2. `app_home() / "models"`;
3. packaged `models` directory beside the executable, if intentionally present;
4. HuggingFace/Ultralytics caches, if supported.

Avoid relying on `Path.cwd()` for release behavior. It may remain as a
development-only fallback, but it should not mask release failures in tests.

## 5. Required Setup Wizard Fix

The current fatal traceback happens after the setup wizard path is entered.

Known source path:

```text
reconstruction_gui/reconstruction_zone.py
  self.after(500, self._run_setup_wizard)

reconstruction_gui/setup_wizard.py
  run_setup_wizard_if_needed(parent)
  wizard = SetupWizard(parent, models)
  wizard.transient(parent)
  wizard.grab_set()
  wizard.focus()
```

The immediate plan is not to guess the final fix. It is to isolate the failure.

### 5.1 Create a source-level setup wizard smoke

Add:

```text
scripts/smoke_setup_wizard.py
```

The smoke should:

1. set a disposable `RECONSTRUCTION_ZONE_APP_HOME`;
2. import `customtkinter`;
3. import `SetupWizard` and `ModelInfo`;
4. create a minimal `ctk.CTk` parent;
5. create synthetic missing model objects;
6. construct the wizard;
7. call `transient`, `grab_set`, and `focus`;
8. run `mainloop()` briefly using `after()` to close cleanly;
9. print explicit markers before and after each step.

Pass criteria:

```text
SETUP_WIZARD_CONSTRUCTED
SETUP_WIZARD_MODAL_APPLIED
SETUP_WIZARD_MAINLOOP_OK
exit code 0
no fatal crash log
```

### 5.2 Package the setup wizard smoke

Compile only the smoke:

```powershell
$env:NUITKA_CACHE_DIR='D:\Projects\reconstruction-zone\dist_test\nuitka-cache'

& 'C:\Python314\python.exe' -m nuitka `
  --standalone `
  --assume-yes-for-downloads `
  --windows-console-mode=attach `
  --enable-plugin=tk-inter `
  --enable-plugin=no-qt `
  --include-package=customtkinter `
  --include-package=reconstruction_gui `
  --output-dir=dist_test\setup-wizard-smoke-YYYYMMDD `
  --report=dist_test\setup-wizard-smoke-YYYYMMDD\setup-wizard-smoke-report.xml `
  scripts\smoke_setup_wizard.py
```

Do not include `torch`, `rfdetr`, or `ultralytics` unless the smoke proves they
are needed. The purpose is Tk/CustomTkinter isolation.

### 5.3 Interpret the setup wizard smoke

If the packaged smoke reproduces the Tk fatal:

- inspect event bindings and widget parent handling inside `SetupWizard`;
- try removing or changing only one modal call at a time:
  - `transient(parent)`;
  - `grab_set()`;
  - `focus()`;
  - `protocol("WM_DELETE_WINDOW", ...)`;
- prove the minimal safe modal pattern under Nuitka before changing the full app.

If the packaged smoke does not reproduce:

- expand the smoke to call `run_setup_wizard_if_needed(parent)`;
- then expand it to use the real `ReconstructionZone` parent class;
- then expand it to use the real `after(500, self._run_setup_wizard)` scheduling.

Do not run another full GUI build until the smallest reproducing case is known
or the smoke proves the setup wizard path safe.

## 6. Model Download And Verification Plan

The setup wizard is not merely a dialog. It is the install path for core assets.
It needs its own validation matrix.

### 6.1 Model readiness checks

For each model:

- ready file exists in app-home model dir;
- missing file is reported as missing;
- corrupt zero-byte file is rejected or at least detected as invalid;
- partially downloaded `.tmp` file is ignored or cleaned;
- successful download writes the final expected filename.

Expected files:

```text
models\rf-detr-seg-small.pt
models\yolo26n-seg.pt
SAM3 HuggingFace cache sentinel / configured local state
```

### 6.2 Download behavior

Validate these cases:

1. RF-DETR successful download to app-home model dir.
2. YOLO26 successful download to app-home model dir.
3. RF-DETR network failure:
   - wizard shows error;
   - `.tmp` file is cleaned;
   - retry works.
4. YOLO26 HuggingFace failure:
   - wizard shows error;
   - no corrupt final file remains;
   - retry works.
5. SAM3 no token:
   - wizard shows gated-model explanation;
   - does not crash;
   - does not block RF-DETR/YOLO26 state from being explainable.
6. SAM3 bad token / no access:
   - user sees actionable status;
   - no traceback.
7. SAM3 approved token:
   - wizard proceeds to download or verify cache.

Use official HuggingFace behavior as the reference:

- `hf_hub_download()` downloads and returns a local file path;
- HuggingFace supports `HF_HOME`, `cache_dir`, and `local_dir` flows;
- `try_to_load_from_cache()` returns `None` when a file is not cached and should
  not raise just because the cache is empty.

### 6.3 Avoid accidental real downloads during UI tests

Add a test mode for setup-wizard smokes, such as:

```text
RECONSTRUCTION_ZONE_SETUP_TEST_MODE=1
```

In test mode:

- do not contact HuggingFace or Google Storage;
- use tiny local fixture files;
- exercise the same UI/queue/status flow;
- write dummy files with the real target names;
- clearly label logs as test mode.

This makes setup wizard UI behavior testable without consuming bandwidth or
requiring credentials.

## 7. Clean New-Install Harness

Create a launch harness script:

```text
scripts\run_packaged_new_install_probe.ps1
```

Inputs:

- `-ExePath`
- `-CaseName`
- `-DurationSeconds`
- `-WithFixtureModels`
- `-SetupTestMode`

Harness behavior:

1. Creates an isolated test root:

```text
dist_test\new-install-sandbox\<case-name>\
```

2. Sets environment:

```powershell
RECONSTRUCTION_ZONE_APP_HOME=<case>\app-home
RECONSTRUCTION_ZONE_MODEL_DIR=<case>\app-home\models
HF_HOME=<case>\hf-home
TRANSFORMERS_CACHE=<case>\hf-home\transformers
```

3. Optionally copies fixture or real model files into the isolated model dir.
4. Launches `ReconstructionZone.exe` from its `.dist` working directory.
5. Captures stdout/stderr to:

```text
<case>\launch-logs\stdout.txt
<case>\launch-logs\stderr.txt
```

6. Waits a fixed observation window.
7. Records:

```text
process exited or stayed alive
exit code if exited
crash log tail
created files
model dir contents
```

8. Stops only the process it launched if it is still alive after the observation
window.
9. Checks for leftover `ReconstructionZone`, `python`, `cl`, `link`, `scons`,
and `git` processes.

This harness must not run Git commands.

## 8. New-Install Test Matrix

### Case A: Truly empty app-home

Purpose:

- prove the app starts from a blank profile;
- prove the setup wizard opens without fatal Tk errors.

Expected:

- app creates app-home/log directories;
- model readiness reports missing RF-DETR/YOLO26/SAM3 as appropriate;
- wizard appears;
- no `TypeError`;
- no fatal crash log.

### Case B: Empty app-home with setup test mode

Purpose:

- prove the wizard can progress through model installation without network.

Expected:

- dummy files are created in isolated model dir;
- wizard reaches done state;
- relaunch reports all test-mode models ready.

### Case C: Fixture model files already present

Purpose:

- prove a model-satisfied new install skips the wizard.

Expected:

- `All models ready — setup wizard not needed`;
- process stays alive;
- no setup wizard crash.

### Case D: RF-DETR missing, YOLO26 present

Purpose:

- prove partial readiness is shown correctly.

Expected:

- only RF-DETR appears missing;
- wizard can install or fail/retry RF-DETR without disturbing YOLO26.

### Case E: YOLO26 missing, RF-DETR present

Purpose:

- same as Case D, reversed.

Expected:

- only YOLO26 appears missing;
- wizard can install or fail/retry YOLO26.

### Case F: Corrupt model files

Purpose:

- prevent a zero-byte or partial file from being treated as ready.

Expected:

- corrupt files are rejected or clearly flagged;
- setup wizard offers repair/re-download.

### Case G: SAM3 no token

Purpose:

- prove gated-model UX works for a genuine new user.

Expected:

- clear SAM3 access explanation;
- no crash;
- RF-DETR/YOLO26 state remains understandable.

### Case H: Network unavailable

Purpose:

- prove diagnostics and retry screens work.

Expected:

- no traceback;
- diagnostic panel reports connectivity failure;
- retry remains available.

### Case I: Existing user upgrade

Purpose:

- prove existing real users are not broken by the app-home refactor.

Expected:

- old prefs can be imported if needed;
- existing model directory is still honored;
- no duplicate downloads if weights already exist.

### Case J: Double-click-style launch

Purpose:

- mimic a user launching the executable from the bundle folder.

Expected:

- launch from `.dist` working directory works;
- no reliance on repo root or shell-provided environment except deliberate
  installer/test variables.

## 9. Visual Verification

Automated process liveness is not enough. The setup wizard is a visible user
workflow.

After the first crash is fixed, perform visual checks:

- welcome screen fits at default size;
- model cards show correct ready/missing states;
- SAM3 gated step is understandable;
- token entry does not expose token text;
- download screen shows active/waiting/completed states;
- error screen shows diagnostics and retry;
- done screen closes cleanly into the app;
- text does not overlap at normal Windows scaling;
- app remains usable after wizard closes.

If possible, capture screenshots of:

```text
welcome
SAM3 gate
downloading/test-mode install
error + diagnostics
done
post-setup app
```

Store screenshots under:

```text
dist_test\new-install-sandbox\<case>\screenshots\
```

## 10. Full GUI Rebuild Criteria

Do not run another full GUI build until all of these are true:

- app-home path refactor is implemented or explicitly deferred with a reason;
- setup wizard source smoke passes;
- setup wizard packaged smoke passes;
- new-install harness exists;
- at least Case A, Case B, and Case C pass against the smaller smoke or current
  packaged artifact;
- no new fatal traceback is written to the isolated crash log;
- no real user app-data directories were deleted or modified;
- no leftover build/test processes remain.

When a full rebuild is justified, use the proven command family from:

```text
dist_test\phase-6b-full-gui-workspacecache-20260512\run-full-gui-build.ps1
```

with the same protections:

- workspace `NUITKA_CACHE_DIR`;
- no `--include-package=transformers`;
- no `--nofollow-import-to=transformers.models.*`;
- torch YAML fix;
- preserve report and artifacts;
- run OpenCV CUDA/NVIDIA DLL copy after build.

## 11. Final Release Gate

A build is release-candidate quality only when:

1. Clean new-install launch from `.dist` opens the setup wizard without fatal
   errors.
2. Setup wizard can install/verify RF-DETR and YOLO26 into app-home models.
3. SAM3 gated path is understandable and non-fatal.
4. Relaunch after setup skips the wizard.
5. App logs, prefs, and models are written under app-home, not beside the exe or
   source checkout.
6. The app can be tested with an isolated app-home and with the real default
   `%LOCALAPPDATA%\ReconstructionZone` path.
7. No test requires `D:\Projects\reconstruction-zone` as current working
   directory.
8. The bundle can be copied to a different folder and still perform the same
   new-install workflow.
9. Crash log contains no new fatal traceback from setup.
10. Full GUI visual smoke confirms the main app opens after setup.

## 12. Documentation Updates Required

After implementing and validating this plan, update:

```text
planning/plans/2026-05-12-phase-6b-full-gui-build-result.md
planning/plans/2026-05-12-phase-6b-runtime-follow-up-checklist.md
planning/plans/2026-05-11-remediation-progress-report.md
planning/plans/2026-05-11-next-codex-handoff.md
planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md
```

Also add a user-facing install note that says where model files and logs live:

```text
%LOCALAPPDATA%\ReconstructionZone\
```

## 13. References Checked

- Nuitka user manual:
  `https://nuitka.net/user-documentation/user-manual.html`
  - standalone packaging, Windows console mode, data files, reports.
- Nuitka tips:
  `https://nuitka.net/user-documentation/tips.html`
  - `NUITKA_CACHE_DIR` cache-location override.
- Python tkinter documentation:
  `https://docs.python.org/3/library/tkinter.html`
  - confirms tkinter is the standard Python interface to Tcl/Tk and is bundled
    with official Python builds on Windows.
- HuggingFace Hub download guide:
  `https://huggingface.co/docs/huggingface_hub/main/guides/download`
  - `hf_hub_download`, cache behavior, `HF_HOME`, `cache_dir`, and `local_dir`.
- HuggingFace Hub cache reference:
  `https://huggingface.co/docs/huggingface_hub/main/package_reference/cache`
  - `try_to_load_from_cache` returns `None` when a file is not cached.

## 14. Execution Update, 2026-05-12

Codex started executing this plan.

Completed:

- added `reconstruction_gui/app_paths.py`;
- moved crash logging and prefs resolution onto app-home paths;
- updated model lookup to prefer app-home and intentional bundled model
  locations;
- fixed the setup-wizard Tk root-collision bug:

```text
self._root -> self._app_root
```

- added `scripts/smoke_setup_wizard.py`;
- added `scripts/run_packaged_new_install_probe.ps1`;
- validated source setup-wizard direct and API-path smokes;
- validated lightweight packaged setup-wizard direct and API-path smokes.

Key result:

```text
The known TypeError setup-wizard crash is fixed in bounded source and packaged
proofs.
```

Detailed result report:

```text
planning/plans/2026-05-12-new-install-setup-wizard-validation-result.md
```

Still pending:

- visual inspection of the rebuilt full GUI setup wizard;
- validate real model download flows.

Full GUI rebuild follow-up is complete:

```text
planning/plans/2026-05-12-phase-6b-full-gui-newinstallfix-result.md
```
