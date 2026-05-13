# Build Issues Remediation Instructions (2026-05-11)

This document is a source-fix instruction plan for the issues recorded in
`planning/plans/2026-05-11-build-issues-and-fixes.md`.

No source code is changed by this document. Its purpose is to give the next
implementer a conservative, reviewable sequence of fixes and verification
steps for the next Nuitka build attempt.

## Executive Summary

The current source-level fixes are directionally useful, but they still leave
three important risks unresolved:

1. The claimed duplicate `cv2.pyd` root cause is not proven in the current
   `dist_nuitka` folder. The inspected bundle contains one `cv2.pyd`, at
   `dist_nuitka/reconstruction_zone.dist/cv2/cv2.pyd`. Treat the OpenCV
   failure as real, but treat the duplicate-root explanation as unverified.

2. The setup wizard's model-ready checks and downloads do not fully match the
   runtime loading paths used by `YOLO26Segmenter` and `RFDETRSegmenter`.
   A wizard can therefore report success while runtime still tries to load a
   different file or downloads to a non-stable current working directory.

3. The `wait_window()`/`CTkToplevel` close crash should be fixed directly. Do
   not assume it disappears merely because the download errors are handled
   better.

Recommended next move: add one small shared model-path resolver, update the
wizard and runtime segmenters to use it, make wizard close handling non-nested
and explicit, then run targeted source and packaged smoke tests before spending
another 2.5 hours on a full GUI build.

## Re-Verified Facts

These checks were performed against `D:\Projects\reconstruction-zone` on
2026-05-11.

### Setup wizard references

Current relevant locations in `reconstruction_gui/setup_wizard.py`:

| Concern | Current location |
|---|---|
| RF-DETR readiness check | line 56 |
| YOLO26 readiness check | line 83 |
| RF-DETR download | line 116 |
| YOLO26 download | line 155 |
| RF-DETR display size | line 179 |
| `all_done_with_errors` queue send | line 439 |
| `all_done_with_errors` handler | line 490 |
| public wizard entrypoint | line 551 |
| `parent.wait_window(wizard)` | line 570 |

### Runtime segmenter references

Current relevant locations in `reconstruction_gui/reconstruction_pipeline.py`:

| Concern | Current location |
|---|---|
| module-level `import cv2` | line 23 |
| `from ultralytics import YOLO` | line 80 |
| `import rfdetr as _rfdetr_module` | line 87 |
| second `import cv2` block | line 94 |
| `YOLO26Segmenter` class | line 974 |
| YOLO checkpoint fallback to `model_name` | line 992 |
| YOLO runtime constructor | line 995 |
| `RFDETRSegmenter` class | line 1062 |
| RF-DETR runtime constructor | line 1098 |

### OpenCV bundle layout

The current `dist_nuitka` folder shows only one `cv2.pyd`:

```powershell
Get-ChildItem dist_nuitka\reconstruction_zone.dist -Recurse -Filter cv2.pyd -Force
```

Observed result:

```text
D:\Projects\reconstruction-zone\dist_nuitka\reconstruction_zone.dist\cv2\cv2.pyd
```

This contradicts the build-issues document's current claim that both of these
exist:

```text
dist_nuitka/reconstruction_zone.dist/cv2.pyd
dist_nuitka/reconstruction_zone.dist/cv2/cv2.pyd
```

Instruction: update the issue document after implementation to say the
recursive OpenCV import is confirmed, but the duplicate-root `cv2.pyd` root
cause is not confirmed by the current bundle.

### RF-DETR installed package behavior

The installed `rfdetr` package contains its own hosted-model mapping:

| Package file | Fact |
|---|---|
| `rfdetr/config.py:173` | `RFDETRSegSmallConfig` exists |
| `rfdetr/config.py:183` | default `pretrain_weights` is `rf-detr-seg-small.pt` |
| `rfdetr/detr.py:77` | constructor calls `maybe_download_pretrain_weights()` |
| `rfdetr/detr.py:585` | `RFDETRSegSmall` exists |
| `rfdetr/detr.py:588` | `RFDETRSegSmall` returns `RFDETRSegSmallConfig(**kwargs)` |
| `rfdetr/main.py:73` | `rf-detr-seg-small.pt` maps to `https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth` |
| `rfdetr/main.py:83` | `download_pretrain_weights()` downloads by filename key |

Conservative interpretation: the wizard's separate `romav2.pt` download is not
proven necessary for `RFDETRSegSmall()` runtime. Remove or defer it unless a
targeted runtime test proves RF-DETR still attempts that additional download.

## Root-Cause Corrections

### Correction 1: OpenCV recursion is real, but the duplicate-file explanation is unverified

The popup/error should still be treated as a blocking packaged-build issue. But
do not anchor the fix on deleting a root-level `cv2.pyd` unless the next bundle
actually contains one.

Required action:

1. Add a direct OpenCV smoke test for Nuitka.
2. Inspect `dist_nuitka/reconstruction_zone.dist` after every build:

```powershell
Get-ChildItem .\dist_nuitka\reconstruction_zone.dist -Recurse -Filter cv2.pyd -Force |
  Select-Object FullName,Length,LastWriteTime
```

3. If the recursion persists with only `cv2/cv2.pyd`, investigate Nuitka/OpenCV
   bootstrap behavior, not duplicate-file cleanup.

### Correction 2: Avoiding cv2 during downloads does not prove runtime cv2 works

The wizard no longer importing `rfdetr` or `ultralytics` during download can
avoid the first crash point. However, normal masking still imports OpenCV and
model libraries at runtime. `reconstruction_pipeline.py` imports `cv2` at module
load, and YOLO/RF-DETR imports happen near the top of the same module.

Required action: test `cv2` and model loading separately in the packaged app.
Do not treat a successful wizard as proof that masking will work.

### Correction 3: Current model-ready checks are stricter or looser than runtime in the wrong places

RF-DETR currently checks for a torch hub DINOv2 file before reporting ready, but
the runtime constructor's primary required checkpoint is `rf-detr-seg-small.pt`.

YOLO26 currently accepts a HuggingFace cache file as ready, but runtime falls
back to `YOLO("yolo26n-seg.pt")`, which may not look at that HuggingFace cache
path.

Required action: make setup checks, setup downloads, and runtime constructors
use the same resolver.

### Correction 4: The wizard close crash needs a direct mitigation

The current code uses `parent.wait_window(wizard)` from a call scheduled by
`self.after(500, self._run_setup_wizard)`. The reported packaged crash happens
after wizard close inside Tk event substitution/name resolution. Even if the
download errors are improved, the nested wait and toplevel destruction remain a
plausible trigger.

Required action: remove `wait_window()` for the first-launch wizard. Use a
non-nested modal-ish toplevel with `transient()`, `grab_set()`, `focus()`, and a
close helper that releases the grab before destroy.

## Recommended Fix Set

Apply these changes in this order.

## Fix 1: Add a shared model-path resolver

Add new file:

```text
reconstruction_gui/model_paths.py
```

Purpose: one source of truth for where model files live and where runtime code
looks for them.

Required behavior:

1. Define a user-writable app model directory.

Recommended default on Windows:

```text
%LOCALAPPDATA%\ReconstructionZone\models
```

Recommended fallback when `LOCALAPPDATA` is absent:

```text
~/.cache/reconstruction-zone/models
```

2. Support an override environment variable for testing:

```text
RECONSTRUCTION_ZONE_MODEL_DIR
```

3. Expose constants:

```python
YOLO26_WEIGHT_NAMES = {
    "n": "yolo26n-seg.pt",
    "s": "yolo26s-seg.pt",
    "m": "yolo26m-seg.pt",
    "l": "yolo26l-seg.pt",
    "x": "yolo26x-seg.pt",
}

RFDETR_SEG_WEIGHT_NAMES = {
    "nano": "rf-detr-seg-nano.pt",
    "small": "rf-detr-seg-small.pt",
    "medium": "rf-detr-seg-medium.pt",
    "large": "rf-detr-seg-large.pt",
    "xlarge": "rf-detr-seg-xlarge.pt",
    "2xlarge": "rf-detr-seg-xxlarge.pt",
}
```

4. Expose functions:

```python
app_model_dir(create: bool = False) -> Path
candidate_model_dirs() -> list[Path]
find_model_file(names: Iterable[str]) -> Optional[Path]
find_hf_cached_file(repo_id: str, filename: str) -> Optional[Path]
resolve_yolo26_weights(size: str = "n") -> Optional[Path]
resolve_rfdetr_seg_weights(size: str = "small") -> Optional[Path]
```

5. Candidate lookup order should be conservative:

```text
1. app model directory
2. current working directory, for legacy source-run weights
3. executable directory, for ad hoc unpacked test builds
4. project root and project-root/models, for source checkouts
5. ~/.cache/ultralytics, for legacy YOLO downloads
6. HuggingFace cache only for YOLO n, if using openvision/yolo26-n-seg
```

Do not rely on writing model files next to the packaged executable. The app may
eventually be installed in a read-only location.

## Fix 2: Update `setup_wizard.py` to use shared paths

Target file:

```text
reconstruction_gui/setup_wizard.py
```

Required edits:

1. Import the shared resolver with both source-run and package import styles:

```python
try:
    from model_paths import ...
except ImportError:
    from reconstruction_gui.model_paths import ...
```

2. Replace `_rfdetr_ready()` with a runtime-equivalent check:

```python
return resolve_rfdetr_seg_weights("small") is not None
```

Do not require `romav2*.pt` for readiness unless a runtime test proves that
file is independently required.

3. Replace `_yolo26_ready()` with:

```python
return resolve_yolo26_weights("n") is not None
```

4. Change `_download_rfdetr()` to download only the segmentation checkpoint to
the app model directory:

```text
%LOCALAPPDATA%\ReconstructionZone\models\rf-detr-seg-small.pt
```

Use the already verified package URL:

```text
https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth
```

Use a temporary file plus atomic replace:

```python
tmp_dest = seg_dest.with_suffix(seg_dest.suffix + ".tmp")
urllib.request.urlretrieve(seg_url, tmp_dest)
tmp_dest.replace(seg_dest)
```

5. Remove the separate DINOv2/`romav2.pt` download block from the wizard for
now. Reasons:

- It was not verified from `rfdetr` source as required for `RFDETRSegSmall`.
- It adds approximately 1.1 GB to first-launch setup.
- It makes `_rfdetr_ready()` stricter than the runtime constructor's explicit
  small-seg checkpoint path.

If RF-DETR later proves to download another backbone during runtime, document
that with the exact stack trace and add it as a separate targeted fix.

6. Change `_download_yolo26()` to copy the HuggingFace cache artifact into the
shared app model directory under the runtime filename:

```text
%LOCALAPPDATA%\ReconstructionZone\models\yolo26n-seg.pt
```

The HuggingFace file can still be downloaded with:

```python
hf_hub_download("openvision/yolo26-n-seg", "model.pt")
```

But the app should not rely on runtime `YOLO("yolo26n-seg.pt")` discovering a
HuggingFace cache file named `model.pt`.

7. Update the RF-DETR display size from `~1.2 GB` to `129 MB` if the wizard no
longer downloads `romav2.pt`.

8. Keep the mailbox/threading pattern. Background threads must only put queue
messages. All CustomTkinter/Tk updates must remain in `_poll()` and
`_handle_message()` on the main thread.

9. Replace the broad silent exception in `_poll()` with logging:

```python
except Exception:
    logger.exception("Setup wizard queue handler failed")
```

The silent `pass` hides exactly the class of packaged-only GUI failure that is
being debugged.

## Fix 3: Make wizard partial-failure UI explicit

Target file:

```text
reconstruction_gui/setup_wizard.py
```

Current behavior is improved enough not to call `_show_done()` after failures,
but the user still needs a visible escape hatch.

Required edits:

1. In the `all_done_with_errors` handler, add a visible button:

```text
Continue Anyway
```

2. The button should close the wizard without setting `success=True`.

3. Keep the status text explicit:

```text
N model(s) failed to download
Failed: <names>. The app will still launch; these models will retry on first use.
```

4. Do not show `Setup complete!` unless all required downloads succeeded.

5. Consider changing `The app will still launch` wording if runtime does not
actually retry on first use after the path resolver change. If runtime simply
fails clearly, say that instead.

## Fix 4: Remove nested `wait_window()` from first-launch wizard

Target file:

```text
reconstruction_gui/setup_wizard.py
```

Current code:

```python
wizard = SetupWizard(parent, models)
parent.wait_window(wizard)
return wizard.success
```

Recommended replacement behavior:

```python
wizard = SetupWizard(parent, models)
try:
    wizard.transient(parent)
    wizard.grab_set()
    wizard.focus()
except Exception:
    logger.exception("Could not make setup wizard modal")
return False
```

Add a close helper on the wizard:

```python
def _close(self):
    try:
        self.grab_release()
    except Exception:
        pass
    self.destroy()
```

Use `_close` for:

- `WM_DELETE_WINDOW` when close is allowed
- `Start Using Reconstruction Zone`
- `Continue Anyway`

Rationale: the caller in `reconstruction_zone.py` currently ignores the return
value from `run_setup_wizard_if_needed(self)`, so there is no need to block the
Tk event loop inside a nested `wait_window()`. Removing that nested wait is a
direct mitigation for the packaged `nametowidget`/`ReconstructionZone object is
not callable` crash.

## Fix 5: Update runtime segmenters to use the same resolver

Target file:

```text
reconstruction_gui/reconstruction_pipeline.py
```

Required edits:

1. Import shared model path helpers near the other imports:

```python
try:
    from model_paths import resolve_rfdetr_seg_weights, resolve_yolo26_weights
except ImportError:
    from reconstruction_gui.model_paths import resolve_rfdetr_seg_weights, resolve_yolo26_weights
```

2. In `YOLO26Segmenter.initialize()`, replace the automatic fallback path:

```python
model_path = model_name  # ultralytics auto-downloads
```

with resolver-first behavior:

```python
resolved_path = resolve_yolo26_weights(size)
model_path = str(resolved_path) if resolved_path else model_name
```

This preserves Ultralytics auto-download as a fallback, but makes
wizard-downloaded weights deterministic.

3. In `RFDETRSegmenter.initialize()`, replace:

```python
self.model = model_cls()
```

with explicit path loading:

```python
weights_path = resolve_rfdetr_seg_weights(size)
if not weights_path:
    expected_name = f"rf-detr-seg-{size}.pt"
    raise FileNotFoundError(
        f"RF-DETR-Seg weights not found for size '{size}'. "
        f"Run the setup wizard or place {expected_name} in the app model directory."
    )
self.model = model_cls(pretrain_weights=str(weights_path))
```

Rationale: the `rfdetr` constructor accepts `pretrain_weights` through
`RFDETRSegSmallConfig(**kwargs)`. Passing an absolute user-cache path avoids
implicit download into whatever the process current working directory happens
to be.

Important follow-up: this only addresses model path determinism. It does not
prove `rfdetr` and OpenCV load correctly in the Nuitka bundle.

## Fix 6: Add targeted smoke tests before the full rebuild

### 6a. Add OpenCV smoke script

Add:

```text
scripts/smoke_import_cv2.py
```

Suggested content:

```python
import cv2

print(cv2.__file__)
print(cv2.__version__)
print("OK")
```

Compile and run with PowerShell:

```powershell
& 'C:\Python314\python.exe' -m nuitka `
  --standalone --assume-yes-for-downloads `
  --include-package=cv2 `
  --output-dir=dist_test `
  scripts/smoke_import_cv2.py

& .\dist_test\smoke_import_cv2.dist\smoke_import_cv2.exe
```

Expected result:

```text
<path to bundled cv2 package>
<OpenCV version>
OK
```

If this fails with the same recursion error, fix OpenCV packaging before another
full GUI build.

### 6b. Add optional model-path smoke script

Add:

```text
scripts/smoke_model_paths.py
```

Suggested content:

```python
from reconstruction_gui.model_paths import (
    app_model_dir,
    resolve_rfdetr_seg_weights,
    resolve_yolo26_weights,
)

print("model_dir", app_model_dir(create=False))
print("rfdetr", resolve_rfdetr_seg_weights("small"))
print("yolo26", resolve_yolo26_weights("n"))
print("OK")
```

Run from source first:

```powershell
& 'C:\Python314\python.exe' .\scripts\smoke_model_paths.py
```

Then compile if needed after OpenCV passes.

## Fix 7: Audit stale `model_downloader.py`

Current search result shows `reconstruction_gui/reconstruction_zone.py` imports
`setup_wizard`, not `model_downloader`.

Required action:

```powershell
rg -n "model_downloader|setup_wizard|run_setup_wizard_if_needed" reconstruction_gui -S
```

If `model_downloader.py` is truly unused:

1. Add a short deprecation note at the top, or remove it in a separate cleanup
   commit after the build is stable.
2. Do not leave it looking like the active first-launch downloader. It still
   imports `rfdetr`/`ultralytics` during download and contains misleading
   thread-safety comments.

If Nuitka is using `--include-package=reconstruction_gui`, verify whether unused
submodules are compiled into the bundle. That is not necessarily a runtime
problem, but it can confuse diagnostics.

## Verification Checklist After Implementing Fixes

Run these in order.

### Source-level checks


```powershell
& 'C:\Python314\python.exe' -m py_compile `
  .\reconstruction_gui\model_paths.py `
  .\reconstruction_gui\setup_wizard.py `
  .\reconstruction_gui\reconstruction_pipeline.py
```

```powershell
& 'C:\Python314\python.exe' -c "from reconstruction_gui.setup_wizard import check_all_models; [print(m.key, m.ready) for m in check_all_models()]"
```

```powershell
& 'C:\Python314\python.exe' .\scripts\smoke_model_paths.py
```

### OpenCV/Nuitka smoke before full GUI rebuild

```powershell
& 'C:\Python314\python.exe' -m nuitka `
  --standalone --assume-yes-for-downloads `
  --include-package=cv2 `
  --output-dir=dist_test `
  scripts/smoke_import_cv2.py

& .\dist_test\smoke_import_cv2.dist\smoke_import_cv2.exe
```

Stop here if OpenCV recursion still reproduces.

### Existing torch smoke remains required

Keep using the existing torch smoke script before the full build:

```powershell
& 'C:\Python314\python.exe' -m nuitka `
  --standalone --assume-yes-for-downloads `
  --include-package=torch --include-package=torchvision `
  --user-package-configuration-file=scripts/nuitka-torch-fix.yml `
  --module-parameter=torch-disable-jit=yes `
  --show-anti-bloat-changes `
  --show-source-changes=torch.library `
  --show-source-changes=torch._refs `
  --report=dist_test/nuitka-report.xml `
  --output-dir=dist_test `
  scripts/smoke_import_torch.py

& .\dist_test\smoke_import_torch.dist\smoke_import_torch.exe
```

### Full packaged build checks

After the full build, verify:

1. Startup produces no COLMAP DLL popup with miniconda on PATH.
2. The setup wizard opens only when models are missing.
3. The setup wizard never says `Setup complete!` after download failures.
4. The setup wizard has a visible `Continue Anyway` path after partial failure.
5. Closing the wizard does not crash with `TypeError: 'ReconstructionZone' object is not callable`.
6. `Get-ChildItem -Recurse -Filter cv2.pyd` shows the actual OpenCV bundle layout.
7. Opening the Mask tab and importing the masking pipeline does not trigger OpenCV recursion.
8. YOLO26 can instantiate from the wizard-downloaded path.
9. RF-DETR can instantiate from the wizard-downloaded path.
10. A one-image masking run succeeds or fails with a clear model-specific error, not with setup/path confusion.

## Diagnostics To Capture If Problems Persist

If the packaged app still fails, capture these before changing more code:

```powershell
Get-ChildItem .\dist_nuitka\reconstruction_zone.dist -Recurse -Filter cv2.pyd -Force |
  Select-Object FullName,Length,LastWriteTime
```

```powershell
Get-ChildItem .\dist_nuitka\reconstruction_zone.dist\cv2 -Force |
  Select-Object Name,Length,Mode,LastWriteTime
```

```powershell
$env:RECONSTRUCTION_ZONE_MODEL_DIR
Get-ChildItem "$env:LOCALAPPDATA\ReconstructionZone\models" -Force -ErrorAction SilentlyContinue |
  Select-Object Name,Length,LastWriteTime
```

Also save:

- Full console output from `ReconstructionZone.exe`.
- Any native Windows popup text, exactly.
- Whether the failure occurred during startup, setup wizard, Mask tab import, model constructor, or actual inference.

## Priority Order

1. Add `model_paths.py`.
2. Wire `setup_wizard.py` readiness/download functions to `model_paths.py`.
3. Fix setup wizard partial-failure UI.
4. Remove `wait_window()` and add explicit close/grab handling.
5. Wire `reconstruction_pipeline.py` YOLO/RF-DETR constructors to `model_paths.py`.
6. Add and run the OpenCV smoke test.
7. Add and run the model-path smoke test.
8. Rebuild only after the smoke tests pass.

## What Not To Do Yet

Do not delete or move bundled `cv2.pyd` files based solely on the current build issue document. Re-check the actual bundle after every build.

Do not spend another full rebuild cycle before a minimal OpenCV Nuitka smoke test exists.

Do not keep requiring `romav2.pt` in the wizard unless runtime RF-DETR produces a verified error showing that it is required.

Do not rely on current working directory for model files in packaged builds.

Do not leave the wizard's queue handler swallowing exceptions silently.

## Recommended Update To Existing Build-Issues Document

After applying and verifying the fixes, update `planning/plans/2026-05-11-build-issues-and-fixes.md` with these corrections:

1. Replace the duplicate `cv2.pyd` root-cause claim with:

```text
The current bundle inspection found only dist_nuitka/reconstruction_zone.dist/cv2/cv2.pyd, so the OpenCV recursion is confirmed but the duplicate-root cause is unproven.
```

2. Replace `The fixes do NOT require changes to the Nuitka YAML config, the spec file, or any other build infrastructure` with a narrower statement:

```text
The model path and wizard fixes are source changes only. OpenCV packaging remains separately unproven and may require build-infrastructure changes after a targeted smoke test.
```

3. Replace the RF-DETR 1.1 GB/DINOv2 requirement with:

```text
The installed rfdetr package maps rf-detr-seg-small.pt directly to the Roboflow hosted checkpoint; no separate DINOv2 download has been proven necessary for first-launch setup.
```

4. Keep the YOLO26 HF-cache risk, but mark the recommended fix: copy or resolve the HF `model.pt` artifact as `yolo26n-seg.pt` in the shared app model directory.

5. Mark the `nametowidget` crash as directly mitigated by removing nested `wait_window()`, not merely expected to disappear after download fixes.
