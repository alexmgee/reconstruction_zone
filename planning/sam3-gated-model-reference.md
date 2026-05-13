# SAM 3 Gated Model Onboarding — Reference Document

**Source:** LichtFeld Studio 360 plugin (`core/setup_checks.py` + `core/sam3_compat.py`)
**Extracted to:** `planning/sam3_gated_setup.py` (standalone, ready to adapt)
**Date:** 2026-05-10

## The Problem

SAM 3 (`facebook/sam3`) is a gated model on HuggingFace. Unlike YOLO or RF-DETR, you can't just `pip install` and download weights — users must:

1. Create a HuggingFace account and generate an access token
2. Visit `huggingface.co/facebook/sam3` and request access
3. Wait for approval (can be instant or take hours)
4. Only then can they download the model weights

If you skip this and just call `hf_hub_download("facebook/sam3", ...)`, the user gets a cryptic 403 error with no guidance on what to do.

## The Solution: A Stage Machine

The extracted code models the onboarding as a state machine with six stages:

```
needs_token ──→ needs_access ──→ ready_to_install ──→ needs_weights ──→ ready
                     │                                                    ↑
                     └── error (at any point) ────────────────────────────┘
                              (retry loops back)
```

Each stage maps to a UI state:

| Stage | UI Shows | User Action |
|---|---|---|
| `needs_token` | Text input for HF token | Paste token, click Verify |
| `needs_access` | Instructions to request access on HF | Visit HF page, wait for approval, click Re-check |
| `ready_to_install` | "Install" button | Click Install |
| `needs_weights` | "Download" button | Click Download |
| `ready` | Green checkmark | Nothing — SAM 3 is operational |
| `error` | Error message + retry button | Fix issue, click Retry |

## Key Data Structure

```python
@dataclass
class Sam3SetupReport:
    token_status: str    # missing | saved | verified | invalid | network_error
    access_status: str   # unknown | pending | granted | network_error
    runtime_status: str  # missing | installed | broken
    weights_status: str  # missing | present | failed
    overall_stage: str   # needs_token | needs_access | ready_to_install | needs_weights | ready | error
    message: str         # "SAM 3 requires a HuggingFace token."
    next_action: str     # "Paste a HuggingFace token, then click Verify Access."
    detail: str          # Technical detail for logs
```

The `overall_stage` is computed from the individual status fields by `_build_sam3_setup_report()`. Your UI code never needs to inspect the individual fields — just switch on `overall_stage` and display `message` + `next_action`.

## Integration with Reconstruction Zone

### Current state (model_downloader.py)

`model_downloader.py` has `_sam3_cached()` and `_download_sam3()` which check/download weights but **don't handle gating**. The download will fail with a 403 if the user hasn't set up token + access.

### How to integrate

The gated setup flow should run **before** the weight download. Two integration points:

**Option A: Gating check inside model_downloader.py**

Before `_download_sam3()` runs, check gating:

```python
from sam3_gated_setup import check_sam3_setup

def _download_sam3(status_callback=None):
    report = check_sam3_setup()
    if report.overall_stage != "ready" and report.overall_stage != "ready_to_install":
        # Can't download yet — need token/access first
        if status_callback:
            status_callback(f"SAM 3: {report.message}")
        return False
    # ... proceed with download
```

**Option B: Separate setup tab/dialog (recommended)**

Add a SAM 3 setup section to the UI that walks the user through the stages. This is what the LichtFeld plugin does — it has a dedicated onboarding panel. The flow:

1. App launch → `check_sam3_setup()` → if not `ready`, show setup UI
2. User pastes token → `verify_hf_token_detailed(token)` → update UI
3. If `ready_to_install` → show Install button → run your install logic
4. If `needs_weights` → `download_model_weights()` → update UI

### Error classification

The most valuable piece for your 403 handling is `_classify_hf_access_exception()`. It maps HuggingFace HTTP errors to user-friendly states:

| HF Error | Classified As | User Sees |
|---|---|---|
| 401 Unauthorized | `token_status="invalid"` | "HuggingFace rejected this token." |
| 403 Forbidden / gated | `access_status="pending"` | "SAM 3 model access is still pending approval." |
| Connection timeout | `access_status="network_error"` | "Could not reach HuggingFace to verify SAM 3 access." |
| Unknown error | `access_status="unknown"` | "Could not verify SAM 3 access." |

This is what turns a wall of Python traceback into an actionable message.

## Import Compatibility (Windows/.exe)

SAM 3's package has eager imports that pull in video/training modules at import time. On Windows and in frozen .exe builds, these fail even though the image API works fine.

The `enable_sam3_image_import_compat()` function installs three lightweight stubs:

```python
# These modules are imported eagerly by sam3.model_builder but only used
# for video/training. The stubs satisfy the import without pulling in
# the full video/training dependency chain.
sam3.train.data.collator          → stub with BatchedDatapoint=object
sam3.model.sam3_video_inference   → stub with empty class
sam3.model.sam3_video_predictor   → stub with empty class
```

**Important:** This is only needed if you use the **image API** (`build_sam3_image_model` + `Sam3Processor`). If Reconstruction Zone uses the **video API** (`build_sam3_multiplex_video_predictor` from `sam3_pipeline.py`), you do NOT want these stubs — the video API needs the real modules.

Since Reconstruction Zone uses `sam3_pipeline.py` with the full video API, the import compat shims are likely **not needed** for the main masking path. However, they're useful if you add a lightweight "is SAM 3 importable?" check that doesn't need to load the full video stack.

## BPE Vocabulary Path

SAM 3's tokenizer needs a BPE vocabulary file (`bpe_simple_vocab_16e6.txt.gz`). The `get_sam3_bpe_path()` function resolves it from several candidate locations. In a frozen .exe, you'll need to either:

1. Bundle it alongside the exe and add a candidate path for your bundle layout
2. Rely on the editable install (`pip install -e .`) which resolves it via package resources

Since Reconstruction Zone uses `pip install -e .` from a git clone of SAM 3, the package-relative path should work. In a Nuitka build, verify that Nuitka includes the `sam3/assets/` directory.

## Token Persistence

`huggingface_hub.login(token=...)` saves the token to `~/.cache/huggingface/token`. This persists across sessions and app restarts. The user only needs to do the token setup once per machine.

`forget_hf_token()` calls `huggingface_hub.logout()` to remove the saved token.

## Dependencies

The gated setup flow only needs `huggingface_hub` (which Reconstruction Zone already has as a transitive dependency of SAM 3). No additional packages required.

```
huggingface_hub  — get_token, login, logout, model_info, try_to_load_from_cache, snapshot_download
```

## Files in this extraction

| File | Purpose |
|---|---|
| `planning/sam3_gated_setup.py` | Standalone module — drop into your source tree and adapt |
| `planning/sam3-gated-model-reference.md` | This document — context and integration guidance |
