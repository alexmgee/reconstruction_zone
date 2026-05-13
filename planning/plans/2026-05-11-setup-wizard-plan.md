# Setup Wizard — Comprehensive Implementation Plan

**Date:** 2026-05-11
**File:** `reconstruction_gui/setup_wizard.py`
**Design reference:** `docs/planning/planning.pen` (5 wizard step mockups at bottom of canvas)
**Verified against:** `setup_wizard.py`, `model_paths.py`, `sam3_setup.py`, `reconstruction_pipeline.py`

---

## Overview

The Setup Wizard is a first-launch CTkToplevel dialog that ensures all AI models are downloaded and ready before the user enters the app. It replaces the broken `model_downloader.py` and handles the SAM3 gated model flow.

If all models are already cached, the wizard never appears.

The implementation should use `model_paths.py` as the single source of truth for model locations. Runtime code and setup code must agree on the same resolver paths:

- App model directory: `%LOCALAPPDATA%\ReconstructionZone\models` on Windows, or `~/.cache/reconstruction-zone/models` fallback
- Optional override: `RECONSTRUCTION_ZONE_MODEL_DIR`
- Legacy/source fallback locations: CWD, executable dir in frozen builds, project root, project `models/`, and the Ultralytics cache

---

## Verified Current State (2026-05-11)

The current `setup_wizard.py` is a functional first-pass, but it does **not** yet match this plan or the `planning.pen` mockups.

### Already implemented
- `run_setup_wizard_if_needed(parent)` is called by `reconstruction_zone.py` via `self.after(500, self._run_setup_wizard)`.
- Model readiness checks call the shared resolvers:
  - `_sam3_ready()` → `resolve_sam3_weights()`
  - `_rfdetr_ready()` → `resolve_rfdetr_seg_weights("small")`
  - `_yolo26_ready()` → `resolve_yolo26_weights("n")`
- Downloads run on a background thread and UI updates are delivered through `queue.Queue()`.
- YOLO26 downloads from HuggingFace cache and copies to the app model directory as `yolo26n-seg.pt`, matching the runtime filename expected by Ultralytics.
- RF-DETR seg weights download to the app model directory as `rf-detr-seg-small.pt`.

### Mismatches to fix
- Current UI is a simple status/progress dialog, not the five-step card-based wizard from `planning.pen`.
- Current SAM3 gate includes a **Skip SAM3** button. The target design says there is no skip option.
- Current failure path adds **Continue Anyway** and marks the wizard as done. The target design says failure goes to Step 5 and blocks until all required models succeed or the user closes.
- `run_diagnostics()` is not implemented.
- Per-model retry is not implemented.
- Current `_download_rfdetr()` downloads only the 129 MB RF-DETR segmentation weights. It does **not** explicitly prefetch the DINOv2/`romav2` backbone. If the first inference still triggers a 1.1 GB backbone download through `rfdetr`, this plan must either implement that prefetch or adjust the stated total download size.
- Current readiness for RF-DETR only checks the seg weight file, not the DINOv2 backbone.
- Current close behavior blocks during downloads, closes during SAM3 gate, and closes after done. This should be made explicit in UI state handling.

## Wizard Steps

### Step 1: Welcome
- **Header:** "Reconstruction Zone" / "Setup Wizard"
- **Context box:** Explains what the models do and that this is a one-time setup
- **Model cards** (3 cards, consistent height):
  - SAM3 — Text-Prompted Segmentation (3.3 GB) — tagged "gated" in yellow
  - RF-DETR — Object Detection (129 MB confirmed; ~1.2 GB only if DINOv2 backbone is explicitly included)
  - YOLO26 — Fast Detection (6.5 MB)
- **Footer:** Total download estimate, X/3 ready, "Begin Setup" button
- **Behavior:** Checks `_sam3_ready()`, `_rfdetr_ready()`, `_yolo26_ready()`. If all ready → never show wizard. If SAM3 not ready → go to Step 2. If SAM3 ready but others missing → skip to Step 3.

**Download total rule:** Show `~3.4 GB + 135 MB` unless RF-DETR DINOv2 prefetch is implemented and verified. Only show `~4.5 GB` after the wizard actually downloads/checks the 1.1 GB backbone.

### Step 2: SAM3 Access (only if SAM3 not ready)
- **Header:** "Reconstruction Zone" / "SAM3 Access"
- **Context box:** Explains SAM3 is the primary engine, developed by Meta, gated on HuggingFace. "Approval is usually instant but can take up to a few hours."
- **Numbered steps:** 4 steps with circle indicators
  1. Create account at huggingface.co
  2. Visit huggingface.co/facebook/sam3 and click Request Access
  3. Create token at huggingface.co/settings/tokens
  4. Paste token below and click Verify
- **Token input:** Label + password field + Verify button
- **Privacy note:** "Your token is saved locally and never shared."
- **No skip option.** The wizard blocks until SAM3 access is resolved or the user closes the window (wizard reappears on next launch).
- **Behavior:** Calls `verify_hf_token_detailed(token)` from `sam3_setup.py`. On success → Step 3. On 403/pending → show message, stay on Step 2. On invalid token → show message, stay on Step 2.

### Step 3: Downloading
- **Header:** "Reconstruction Zone" / "Downloading Models"
- **Model cards** (same height as Steps 4/5):
  - Completed models: faded (opacity 0.6), green checkmark, green "X.X GB downloaded"
  - Active model: full opacity, ▶ icon, description, progress bar + "Downloading..." grouped at bottom of card
  - Waiting models: faded (opacity 0.4), grey dot, "Waiting..."
- **Behavior:** Downloads run in background thread via queue pattern. Worker puts messages on `queue.Queue()`, main thread polls with `after(100, _poll)`. No tkinter calls from background thread.
- **Download functions (no cv2 imports):**
  - SAM3: `huggingface_hub.snapshot_download("facebook/sam3")`
  - RF-DETR: `urllib.request.urlretrieve()` for seg weights to `app_model_dir(create=True) / "rf-detr-seg-small.pt"`
  - RF-DETR DINOv2 backbone: explicitly prefetch only after verifying the stable URL/cache path used by `rfdetr`
  - YOLO26: `huggingface_hub.hf_hub_download("openvision/yolo26-n-seg", "model.pt")`, then copy to `app_model_dir(create=True) / "yolo26n-seg.pt"`
- **On all succeed → Step 4. On any failure → Step 5.**

### Step 4: Done
- **Header:** "Reconstruction Zone" / "Setup Complete" (green)
- **Model cards:** All faded, all green checkmarks, green "downloaded" text
- **Footer:** "All models are installed and ready." + blue "Enter Reconstruction Zone" button
- **Behavior:** Button calls `self.destroy()`. Wizard never appears again (all models cached).

### Step 5: Error
- **Header:** "Reconstruction Zone" / "Download Issue" (red, using #EF4444)
- **Model cards:** Same layout as Steps 3/4
  - Succeeded: faded, green checkmark, green "downloaded" text
  - Failed: full opacity, red ✗, red error text (e.g., "Connection timed out"), blue "Retry" button aligned right in status row
- **Diagnostics section** (no surrounding box, text aligned with card content):
  - Runs automatically when downloads fail
  - Shows pass/fail for each relevant check
  - Only checks endpoints relevant to the failed models
- **Per-model Retry buttons:** Each failed model has its own Retry button. On click, retries just that model's download.
- **No "Continue Anyway" option.** Wizard stays until all models succeed or user closes (reappears on next launch).

---

## Diagnostics Function

### Purpose
When a download fails, run concrete checks instead of guessing. Show the user exactly what's wrong.

### Implementation: `run_diagnostics(failed_model_keys)`

```python
def run_diagnostics(failed_model_keys: list[str]) -> list[tuple[str, str]]:
    """Run connectivity and environment checks for failed model downloads.
    
    Args:
        failed_model_keys: List of model keys that failed ("sam3", "rfdetr", "yolo26")
    
    Returns:
        List of (status, message) tuples where status is "pass" or "fail"
    """
    import urllib.request
    import shutil
    
    results = []
    
    # 1. General internet connectivity
    try:
        urllib.request.urlopen("https://www.google.com", timeout=5)
        results.append(("pass", "Internet connection active"))
    except Exception:
        results.append(("fail", "No internet connection detected"))
        return results  # No point checking endpoints if offline
    
    # 2. Per-model endpoint checks (only for models that failed)
    MODEL_ENDPOINTS = {
        "sam3": [
            ("https://huggingface.co", "huggingface.co", "SAM3 weights"),
        ],
        "rfdetr": [
            ("https://storage.googleapis.com", "storage.googleapis.com", "RF-DETR seg weights"),
            # Add the DINOv2/romav2 endpoint here only after the stable source is verified.
        ],
        "yolo26": [
            ("https://huggingface.co", "huggingface.co", "YOLO26 weights"),
        ],
    }
    
    checked_hosts = set()
    for key in failed_model_keys:
        for url, host, label in MODEL_ENDPOINTS.get(key, []):
            if host in checked_hosts:
                continue
            checked_hosts.add(host)
            try:
                urllib.request.urlopen(url, timeout=5)
                results.append(("pass", f"{host} reachable ({label})"))
            except Exception:
                results.append(("fail", f"{host} unreachable ({label})"))
    
    # 3. Disk space
    try:
        try:
            from model_paths import app_model_dir
        except ImportError:
            from reconstruction_gui.model_paths import app_model_dir
        model_dir = app_model_dir(create=True)
        free_bytes = shutil.disk_usage(model_dir).free
        free_gb = free_bytes / (1024 ** 3)
        if free_gb > 6:
            results.append(("pass", f"Disk space: {free_gb:.0f} GB free"))
        else:
            results.append(("fail", f"Low disk space: {free_gb:.1f} GB free (need at least 6 GB)"))
    except Exception:
        results.append(("fail", "Could not check disk space"))
    
    # 4. SAM3-specific: HF token check (only if SAM3 failed)
    if "sam3" in failed_model_keys:
        try:
            from huggingface_hub import get_token
            token = get_token()
            if token and len(token) > 0:
                results.append(("pass", "HuggingFace token saved"))
            else:
                results.append(("fail", "No HuggingFace token found"))
        except Exception:
            results.append(("fail", "Could not check HuggingFace token"))
    
    return results
```

### Integration with wizard
- Called automatically in `_download_worker()` after failures, before sending `all_done_with_errors`
- Results sent as `("diagnostics", results)` message on the queue
- Main thread renders each result as a check/cross + text line
- Failure messages should store failed model **keys** (`"sam3"`, `"rfdetr"`, `"yolo26"`), not display names, so retry and diagnostics do not need string matching.

---

## Model Check Functions

### `_sam3_ready()`
Use `resolve_sam3_weights()` from `model_paths.py`. Today this checks HuggingFace cache for `facebook/sam3` `config.json` via `try_to_load_from_cache()`.

### `_rfdetr_ready()`
Use `resolve_rfdetr_seg_weights("small")` from `model_paths.py`.

Current resolver checks `rf-detr-seg-small.pt` in the shared candidate dirs, including the app model directory and legacy/source locations. It does **not** check DINOv2/`romav2` backbone readiness yet.

**Decision needed:** If RF-DETR must be fully ready before entering the app, add a DINOv2 backbone resolver/check. If it is acceptable for `rfdetr` to download its backbone at first inference, remove DINOv2 from the wizard size estimate and progress UI.

### `_yolo26_ready()`
Use `resolve_yolo26_weights("n")` from `model_paths.py`.

Current resolver checks the runtime filename `yolo26n-seg.pt` in shared candidate dirs first, then HuggingFace cache via `try_to_load_from_cache("openvision/yolo26-n-seg", "model.pt")`.

---

## Download Functions (cv2-free)

The wizard downloads weights WITHOUT importing cv2. This avoids the Nuitka cv2 recursion bug.

### `_download_sam3(queue)`
```python
from huggingface_hub import snapshot_download
snapshot_download("facebook/sam3")
```

### `_download_rfdetr(queue)`
Current verified behavior:
1. Seg weights via `urllib.request.urlretrieve("https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth", app_model_dir(create=True) / "rf-detr-seg-small.pt")`
2. No explicit DINOv2 backbone download yet

Target behavior, if keeping the ~1.2 GB RF-DETR estimate:
1. Download seg weights to app model dir.
2. Prefetch the DINOv2/`romav2` backbone using the same URL/cache path that `rfdetr` uses internally.
3. Add a resolver/check for the backbone so readiness is honest.

### `_download_yolo26(queue)`
```python
import shutil
from huggingface_hub import hf_hub_download

hf_path = hf_hub_download("openvision/yolo26-n-seg", "model.pt")
shutil.copy2(hf_path, app_model_dir(create=True) / "yolo26n-seg.pt")
```

---

## Threading Pattern

Hard rule: **the worker thread never calls tkinter.**

### Message types on the queue:
| Type | Args | Meaning |
|---|---|---|
| `("progress", idx, total, name)` | int, int, str | Starting download for model at index |
| `("status", text)` | str | Status update text (e.g., "Downloading DINOv2 backbone...") |
| `("model_done", key)` | str | Model download succeeded |
| `("model_error", key, error)` | str, str | Model download failed with error message |
| `("all_done",)` | — | All downloads succeeded |
| `("all_done_with_errors", failures)` | list[str] | Some downloads failed; list of failed model keys |
| `("diagnostics", results)` | list[tuple] | Diagnostic check results after failures |
| `("sam3_verify", report)` | Sam3SetupReport | SAM3 token verification result |
| `("retry_done", key)` | str | Single-model retry succeeded |
| `("retry_error", key, error)` | str, str | Single-model retry failed |

### Polling:
```python
def _poll(self):
    while not self._queue.empty():
        msg = self._queue.get_nowait()
        self._handle_message(msg)
    if self._current_step not in ("done",):
        self._root.after(100, self._poll)
```

Step 5 is an active error state with retry buttons, so queue polling must continue while retries are running. Do not stop polling just because `_current_step == "error"`.

---

## Card Layout Specification

All model cards across Steps 3, 4, and 5 are **101px tall** with identical structure:

```
[14px top padding]
[Header row: icon + model name]       — 15px tall
[6px gap]
[Description text]                    — fill_container (absorbs extra height)
[6px gap]  
[Status row: downloaded/error/retry]  — 13-20px tall
[14px bottom padding]
```

### Card states:
| State | Opacity | Icon | Status row |
|---|---|---|---|
| Completed | 0.6 | Green ✓ | Green "X.X GB downloaded" |
| Active | 1.0 | Blue ▶ | Progress bar + "Downloading..." grouped at bottom |
| Waiting | 0.4 | Grey dot | Grey "Waiting..." |
| Failed | 1.0 | Red ✗ | Red error text (left) + blue Retry button (right) |

---

## SAM3 Gate Integration

Uses `sam3_setup.py` (extracted from LichtFeld plugin). Key functions:

- `check_sam3_setup()` → returns `Sam3SetupReport` with `overall_stage`
- `verify_hf_token_detailed(token)` → saves token, checks access, returns report
- Stage machine: `needs_token → needs_access → ready_to_install → needs_weights → ready`

The wizard maps `overall_stage` to UI state:
- `needs_token` / `needs_access` → show Step 2
- `ready_to_install` / `needs_weights` → skip to Step 3, include SAM3 in downloads
- `ready` → SAM3 already good, skip gate

Implementation note: current `_show_sam3_gate()` treats `ready` and `needs_weights` as access-verified. It should also explicitly handle `ready_to_install` to match the stage machine documented here.

---

## Files Involved

| File | Role |
|---|---|
| `reconstruction_gui/setup_wizard.py` | Main wizard dialog + model registry + download/check functions + diagnostics |
| `reconstruction_gui/model_paths.py` | Single source of truth for model storage directories, filenames, and readiness resolution |
| `reconstruction_gui/sam3_setup.py` | SAM3 gated model state machine (token, access, weights) |
| `reconstruction_gui/reconstruction_zone.py` | Calls `run_setup_wizard_if_needed()` via `self.after(500, ...)` |
| `reconstruction_gui/reconstruction_pipeline.py` | Runtime loading path for YOLO26 and RF-DETR; must stay aligned with `model_paths.py` |
| `reconstruction_gui/model_downloader.py` | **Superseded** — old broken downloader, no longer called |

---

## Design System Tokens (from planning.pen)

| Token | Value | Usage |
|---|---|---|
| Background | `#1a1a2e` | Wizard frame background |
| Surface | `#1e293b` | Model cards, context boxes |
| Active accent | `#2563eb` | Buttons, active indicators, progress bar |
| Text primary | `#ffffff` | Titles |
| Text secondary | `#e2e8f0` | Model names, descriptions, diagnostics text |
| Text muted | `#94a3b8` | Descriptions, secondary info |
| Text dim | `#64748b` / `#475569` | Hints, waiting state |
| Success | `#22c55e` | Checkmarks, "downloaded" text, "Setup Complete" |
| Error | `#EF4444` | ✗ icons, error text, "Download Issue" |
| Warning | `#eab308` | "gated" tag, timing warnings |
| Card corner radius | 4px | All model cards |
| Button corner radius | 3-4px | All buttons |
| Card height | 101px | All model cards in Steps 3-5 |
| Card padding | 14px all sides | All model cards |
| Card gap | 6px | Between card children |
| Card spacing | 8px | Between cards in the list |

---

## Open Issues to Resolve Before Build

1. **RF-DETR DINOv2 readiness** — current setup wizard only downloads/checks the 129 MB seg weights. Decide whether to prefetch the 1.1 GB DINOv2/`romav2` backbone. If yes, verify the stable URL/cache path from `rfdetr` internals and add both download and readiness checks. If no, correct all wizard totals to remove the backbone.

2. **YOLO26 file location** — mostly resolved. Current wizard copies HF `model.pt` to `app_model_dir(create=True) / "yolo26n-seg.pt"`, and runtime uses `resolve_yolo26_weights("n")`. Still test in source and packaged builds.

3. **RF-DETR seg weights location** — mostly resolved. Current wizard downloads to `app_model_dir(create=True) / "rf-detr-seg-small.pt"`, and runtime passes that path as `pretrain_weights`. Still test in packaged builds.

4. **cv2 recursion in Nuitka** — the download functions avoid cv2, but the main masking pipeline imports cv2 at runtime. If the Nuitka cv2 recursion bug affects all cv2 usage (not just during downloads), masking itself will break. Needs testing.

5. **Wizard close behavior** — target behavior: Step 2 can close and reappear next launch; Step 3 ignores close attempts while downloads are active; Step 5 can close and reappear next launch. Document in TESTING.md.

6. **Per-model Retry** — Step 5 shows individual Retry buttons per failed model. The current `setup_wizard.py` code sends `all_done_with_errors` as a batch. Need to implement per-model retry that re-downloads just the failed model and updates its card individually.

7. **Remove escape hatches** — current code has `Skip SAM3` and `Continue Anyway`. Remove or hide both to match the blocking first-launch setup contract.

8. **Design fidelity** — current UI is not the `planning.pen` five-screen card layout. Build card components once and reuse across Welcome, Downloading, Done, and Error states.

9. **Diagnostics** — implement `run_diagnostics(failed_model_keys)` and render results in Step 5. Use `app_model_dir()` for disk-space checks and HuggingFace, not GitHub, for YOLO26 endpoint checks.

10. **Tests/smoke checks** — add source-run smoke coverage for:
    - all models ready → wizard does not appear
    - SAM3 missing/token missing → Step 2
    - SAM3 ready but YOLO/RF-DETR missing → Step 3
    - failed download → Step 5, diagnostics shown, no Continue Anyway
    - per-model retry succeeds and transitions to Step 4 when all ready
