# Phase 6B Full GUI Nuitka Build Result

Date started: 2026-05-12  
Date completed: 2026-05-12  
Owner: Codex  
Repository: `D:\Projects\reconstruction-zone`  
Status: **build passed; runtime smoke partially passed; first-launch/setup-wizard path is a blocker**

## 1. Purpose

This report records the full GUI Nuitka build attempt requested after the RF-DETR
Phase 6B packaging smoke passed with a workspace cache.

The user-approved build constraints are:

- use `NUITKA_CACHE_DIR=D:\Projects\reconstruction-zone\dist_test\nuitka-cache`;
- do not use `--include-package=transformers`;
- do not use `--nofollow-import-to=transformers.models.*`;
- keep the torch Nuitka YAML fix;
- preserve the report and produced artifacts.

## 2. Pre-Build Verification

All local inputs required by the command were checked before launch:

```text
C:\Users\alexm\ffmpeg\bin\ffmpeg.exe: present
C:\Users\alexm\ffmpeg\bin\ffprobe.exe: present
reconstruction-zone.ico: present
scripts\nuitka-torch-fix.yml: present
scripts\copy_cv2_cuda_dlls.ps1: present
```

Interpreter and package baseline:

```text
Python: 3.14.0
Nuitka: 4.0.8
torch: 2.10.0+cu126
torchvision: 0.25.0+cu126
customtkinter: 5.2.2
rfdetr: importable, package does not expose __version__
ultralytics: 8.4.14
transformers: 4.57.6
```

Process hygiene:

```text
git.exe processes visible before launch: none
```

Two pre-existing Miniconda `python.exe` processes were visible but had no command
line available and were not identified as part of this build. They were not
stopped.

## 3. Official Documentation Checked

The command shape is aligned with the current Nuitka documentation:

- Nuitka user manual:
  `https://nuitka.net/user-documentation/user-manual.html`
  - documents running Nuitka with the intended Python interpreter via
    `python -m nuitka`;
  - documents Windows console modes, including `attach`;
  - documents data-file inclusion and XML compilation reports.
- Nuitka tips:
  `https://nuitka.net/user-documentation/tips.html`
  - documents `NUITKA_CACHE_DIR` as the base cache-location override.
- Nuitka package configuration docs:
  `https://nuitka.net/user-documentation/nuitka-package-config.html`
  - documents `--user-package-configuration-file` for local YAML package
    configuration.

## 4. Build Directory

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-workspacecache-20260512
```

Expected preserved files:

```text
run-full-gui-build.ps1
full-gui-build-transcript.txt
full-gui-build-exit-code.txt
full-gui-workspacecache-report.xml
reconstruction_zone.build\
reconstruction_zone.dist\
```

The exact generated `.dist` directory was:

```text
dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist
```

## 5. Exact Build Command

The command is preserved in:

```text
dist_test\phase-6b-full-gui-workspacecache-20260512\run-full-gui-build.ps1
```

Command body:

```powershell
$env:NUITKA_CACHE_DIR = "D:\Projects\reconstruction-zone\dist_test\nuitka-cache"

& "C:\Python314\python.exe" -m nuitka `
  --standalone `
  --assume-yes-for-downloads `
  --windows-console-mode=attach `
  --windows-icon-from-ico=reconstruction-zone.ico `
  --enable-plugin=tk-inter `
  --enable-plugin=no-qt `
  --user-package-configuration-file=scripts/nuitka-torch-fix.yml `
  --module-parameter=torch-disable-jit=yes `
  --nofollow-import-to=bitsandbytes `
  --nofollow-import-to=bitsandbytes.* `
  --nofollow-import-to=triton `
  --nofollow-import-to=triton.* `
  --include-package=prep360 `
  --include-package=reconstruction_gui `
  --include-package=torch `
  --include-package=torchvision `
  --include-package=rfdetr `
  --include-package=ultralytics `
  --include-package=customtkinter `
  --include-data-dir=resources=resources `
  --include-data-dir=reconstruction_gui/docs=reconstruction_gui/docs `
  --include-data-files=reconstruction-zone.ico=reconstruction-zone.ico `
  --include-data-files="C:/Users/alexm/ffmpeg/bin/ffmpeg.exe=ffmpeg/ffmpeg.exe" `
  --include-data-files="C:/Users/alexm/ffmpeg/bin/ffprobe.exe=ffmpeg/ffprobe.exe" `
  --output-dir=dist_test/phase-6b-full-gui-workspacecache-20260512 `
  --output-filename=ReconstructionZone.exe `
  --report=dist_test/phase-6b-full-gui-workspacecache-20260512/full-gui-workspacecache-report.xml `
  reconstruction_gui/reconstruction_zone.py
```

## 6. Rationale For Included And Excluded Options

Kept from the proven RF-DETR A2 packaging smoke:

- workspace `NUITKA_CACHE_DIR`;
- torch YAML fix;
- `--module-parameter=torch-disable-jit=yes`;
- no broad `--include-package=transformers`.

Promoted for the full GUI build:

- `--include-package=reconstruction_gui`;
- `--include-package=prep360`;
- `--include-package=customtkinter`;
- `--include-package=ultralytics`;
- `--include-package=rfdetr`;
- GUI/data inputs: icon, resources, GUI docs, ffmpeg, ffprobe.

Intentionally not used:

- `--include-package=transformers`, because the RF-DETR A1 smoke failed with a
  Nuitka fatal in `transformers.commands.add_new_model_like`;
- `--nofollow-import-to=transformers.models.*`, because it overlaps with
  eagerly imported RF-DETR/Transformers modules and was never promoted from
  optional optimization to correctness requirement.

Kept as targeted bloat exclusions:

- `bitsandbytes`;
- `bitsandbytes.*`;
- `triton`;
- `triton.*`.

These are not the rejected Transformers model-tree exclusion.

## 7. Runtime Verification Plan

If the build exits `0`, the next checks are:

1. inspect the Nuitka XML report for `completion="yes"`;
2. inspect output artifact names and sizes;
3. run `scripts\copy_cv2_cuda_dlls.ps1` against the produced `.dist` directory;
4. run the packaged executable from the repository root;
5. run the packaged executable from the `.dist` directory;
6. classify stdout/stderr carefully, remembering that GUI startup may redirect
   output after early initialization;
7. preserve all logs and artifacts.

If the build fails or times out:

1. record exit code;
2. inspect transcript tail;
3. inspect partial report/build directory, if present;
4. check for leftover `python`, `cl`, `link`, or `scons` processes tied to this
   build;
5. classify the result from evidence only.

## 8. Result

### 8.1 Build Outcome

The Nuitka build exited `0`.

```text
started: 2026-05-12T08:45:53-05:00
finished: 2026-05-12T11:49:51-05:00
duration: about 3h 04m
exit code: 0
```

Nuitka reported:

```text
completion="yes"
compiled/linked C files: 9492
clcache hits: 0
clcache misses: 9492
```

Nuitka also reported that the command used:

```text
--include-package=rfdetr
--include-package=ultralytics
--include-package=torch
--include-package=torchvision
```

and did **not** use:

```text
--include-package=transformers
--nofollow-import-to=transformers.models.*
```

### 8.2 Preserved Artifact Metadata

```text
full-gui-workspacecache-report.xml: 22,639,674 bytes
full-gui-build-transcript.txt: 1,099 bytes
ReconstructionZone.exe: 954,236,928 bytes
```

The XML report begins with:

```text
<nuitka-compilation-report ... mode="standalone" completion="yes">
```

Important logging caveat:

- the PowerShell transcript preserved the wrapper start/end metadata and exit
  marker;
- the transcript did **not** capture the native Nuitka stream;
- the authoritative preserved build artifacts are therefore the XML report,
  build directory, `.dist` directory, executable, wrapper script, and exit-code
  file.

The console stream observed during the build showed:

- Python-level compilation completed;
- C backend generation completed;
- SCons compilation/linking completed;
- data files were included for:
  - `reconstruction-zone.ico`;
  - `ffmpeg\ffmpeg.exe`;
  - `ffmpeg\ffprobe.exe`;
  - `resources\...`;
  - `reconstruction_gui\docs\...`;
  - CustomTkinter assets;
  - matplotlib/Tk/Tcl/pytz/tzdata/sklearn/torch data files.

### 8.3 OpenCV CUDA DLL Post-Build Step

The post-build helper succeeded:

```powershell
.\scripts\copy_cv2_cuda_dlls.ps1 `
  -DistDir dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist
```

Result:

```text
Copied 16 OpenCV CUDA/NVIDIA DLLs to:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist\cv2
```

Spot-checked copied files:

```text
nvcuvid.dll
nvEncodeAPI64.dll
cudart64_13.dll
cudnn64_9.dll
```

### 8.4 Sandbox Runtime Probe

Running the packaged executable inside the Codex sandbox failed before GUI
startup:

```text
exit code: 1
stderr: PermissionError: [Errno 13] Permission denied:
'C:\\Users\\alexm\\.reconstruction_zone\\crash.log'
```

Interpretation:

- this is consistent with sandbox write restrictions, because the app's
  `app_infra.py` initializes a rotating file logger under
  `Path.home() / ".reconstruction_zone" / "crash.log"` at import time;
- it is **not** evidence that the packaged app cannot start in the normal user
  environment.

### 8.5 Outside-Sandbox Runtime Probes

Because the sandbox blocked the normal profile log path, Codex ran bounded
outside-sandbox launch probes with explicit 20-second observation windows and
then stopped only the processes started by the probes.

#### Repo-root working directory

Command shape:

```text
FilePath: ReconstructionZone.exe
WorkingDirectory: D:\Projects\reconstruction-zone
WindowStyle: Hidden
Observation window: 20 seconds
```

Observed:

```text
process stayed alive past 20 seconds
process was stopped by Codex after the observation window
stdout: 0 bytes
stderr: 838 bytes
```

Log markers:

```text
Detected colmap binary:
C:\Users\alexm\COLMAP\COLMAP-4.0.3-x64-windows-cuda\bin\colmap.exe

Detected spheresfm binary:
D:\Tools\SphereSfM\bin\colmap.exe

All models ready — setup wizard not needed
```

Classification:

```text
PASS as a bounded startup/liveness smoke when launched from the repo root.
```

This pass is dependent on the repo root being in the model search path, because
the current RF-DETR and YOLO26 weights live in:

```text
D:\Projects\reconstruction-zone\rf-detr-seg-small.pt
D:\Projects\reconstruction-zone\yolo26n-seg.pt
```

#### `.dist` working directory without model override

Command shape:

```text
FilePath: ReconstructionZone.exe
WorkingDirectory:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist
WindowStyle: Hidden
Observation window: 20 seconds
```

Observed:

```text
process stayed alive past 20 seconds
process was stopped by Codex after the observation window
stdout: 0 bytes
stderr: 1089 bytes
```

Log markers:

```text
Models need setup: RF-DETR-Seg, YOLO26-n-seg
```

Persistent app log additionally recorded:

```text
FATAL CRASH: 2026-05-12T12:26:44.047794
TypeError: 'ReconstructionZone' object is not callable
```

Traceback location:

```text
tkinter.__init__._substitute
tkinter.__init__.nametowidget
tkinter.__init__._report_exception
customtkinter.windows.ctk_tk.mainloop
reconstruction_zone.py main()
```

Classification:

```text
FAIL for first-launch / missing-model setup-wizard path.
```

The app did not immediately disappear, but the setup-wizard path wrote a fatal
Tk callback error to the persistent log. That cannot be called a clean runtime
pass.

#### `.dist` working directory with explicit model directory override

Command shape:

```text
Environment:
RECONSTRUCTION_ZONE_MODEL_DIR=D:\Projects\reconstruction-zone

FilePath: ReconstructionZone.exe
WorkingDirectory:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist
WindowStyle: Hidden
Observation window: 20 seconds
```

Observed:

```text
process stayed alive past 20 seconds
process was stopped by Codex after the observation window
stdout: 0 bytes
stderr: 1089 bytes
```

Log markers:

```text
All models ready — setup wizard not needed
```

No new fatal traceback appeared after this probe.

Classification:

```text
PASS as a bounded startup/liveness smoke when model lookup is satisfied.
```

### 8.6 Model Lookup Finding

The root-versus-`.dist` difference is explained by
`reconstruction_gui/model_paths.py`.

`candidate_model_dirs()` searches:

1. app model dir:
   `%LOCALAPPDATA%\ReconstructionZone\models`;
2. current working directory;
3. executable directory, for frozen builds;
4. the project root inferred from `__file__`;
5. inferred `project_root\models`;
6. Ultralytics cache, if present.

In this compiled standalone bundle, `__file__` points inside:

```text
...\reconstruction_zone.dist\
```

so the inferred project root is the build-output directory, not the original
source checkout. A `.dist` launch does not naturally search
`D:\Projects\reconstruction-zone`.

The existing local model weights were found by repo-root launch because they are
in the current working directory. They were not found by plain `.dist` launch.

### 8.7 Current Phase 6B Classification

Phase 6B has split into two conclusions:

1. **Packaging compile strategy: PASS**
   - workspace cache worked;
   - no `--include-package=transformers`;
   - no `transformers.models.*` exclusion;
   - torch YAML fix held;
   - full GUI executable was produced.

2. **Packaged runtime readiness: PARTIAL / BLOCKED**
   - startup works when model lookup is satisfied;
   - first-launch setup-wizard path is broken in the compiled bundle when
     RF-DETR/YOLO26 weights are missing from the packaged app's model search
     locations.

This means the next work should **not** be another blind full build. The next
work should target model lookup / first-launch setup wizard behavior.

## 9. Recommended Next Actions

### 9.1 For local user testing now

The current artifact can be tested if model lookup is satisfied.

Known-good launch shape from this report:

```powershell
$env:RECONSTRUCTION_ZONE_MODEL_DIR='D:\Projects\reconstruction-zone'
& 'D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist\ReconstructionZone.exe'
```

Alternatively, launching with working directory set to the repository root also
found the current local weights in this environment.

Do **not** use this as the final release arrangement. It is a local test
convenience.

### 9.2 For release correctness

Fix one or both of these paths:

1. Ensure model weights are discoverable in a user/app location:
   - `%LOCALAPPDATA%\ReconstructionZone\models\rf-detr-seg-small.pt`;
   - `%LOCALAPPDATA%\ReconstructionZone\models\yolo26n-seg.pt`;
   - or a deliberate installer/bundle location documented in
     `candidate_model_dirs()`.

2. Fix the compiled first-launch setup-wizard path:
   - reproduce the `SetupWizard(parent, models)` / `transient()` / `grab_set()`
     path in a small CustomTkinter Nuitka smoke;
   - determine whether the `TypeError: 'ReconstructionZone' object is not
     callable` is caused by a Nuitka/Tkinter event-substitution issue, a parent
     argument shape, or a project-side callback binding;
   - rerun a missing-model `.dist` launch and require no fatal Tk traceback.

### 9.3 Suggested next bounded proof

Create a small packaging smoke that imports and exercises only:

```text
customtkinter root
SetupWizard parent/top-level construction
transient(parent)
grab_set()
focus()
```

with synthetic missing-model data. This should be much cheaper than a full GUI
rebuild and should isolate the Tk callback failure.

Only after that path is fixed should another full GUI build be launched.

## 10. Final Assessment

The full GUI Nuitka build finally crossed the compile barrier under the
approved constraints.

It is **not yet a releasable friend-test artifact** because a realistic
first-launch/missing-model path still writes a fatal Tk callback error in the
compiled bundle.

It **is** a useful local test artifact for model-satisfied GUI startup and for
the next packaging investigations.
