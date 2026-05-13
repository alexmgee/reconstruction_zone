# Reconstruction Zone Lite — GitHub Fallback Plan

**Date:** 2026-05-12
**Status:** Sketch (not yet implemented)
**Branch:** `release/v1-lite` (to be created from `release/v1` if needed)
**Trigger:** The .exe build fails and we need to share release/v1 via GitHub instead.

---

## Purpose

A pared-down version of Reconstruction Zone that a tester with a CUDA GPU and Python installed can clone from GitHub and run with minimal friction. Focused on the core workflow: extract frames, mask objects, review results.

---

## Scope

### Tabs: Extract, Mask, Review only

Remove Projects, Align, and Coverage tabs. Everything within the three kept tabs remains unchanged — no modifications to the Mask tab model selector, the Extract tab workflow, or the Review tab.

### Two-layer setup

**Layer 1: Install script (`install.bat` + `setup_install.py`)**
Runs once before first launch. Handles everything that can't be done from inside the app.

**Layer 2: Setup wizard (`setup_wizard_lite.py`)**
Runs on every launch. Full environment health check + model downloader. New file — does not modify the existing `setup_wizard.py`.

---

## Layer 1: Install Script

### install.bat (thin wrapper)

Creates the venv, then hands off to the Python installer script running inside the venv.

```batch
@echo off
echo Reconstruction Zone — Setup
echo.
where python >nul 2>&1 || (echo ERROR: Python not found on PATH. & pause & exit /b 1)
python -m venv .venv
.venv\Scripts\python setup_install.py %*
pause
```

### setup_install.py (the real work)

Runs inside the venv. Steps:

1. **Check Python version** — must be 3.10-3.12. If outside range, print specific guidance and stop.
2. **Check disk space** — warn if <10 GB free on the drive.
3. **Detect GPU via nvidia-smi** — parse CUDA driver version from output. If nvidia-smi not found, report "no NVIDIA GPU detected or drivers not installed" and stop with guidance.
4. **Select PyTorch CUDA index** — map driver CUDA version to the correct `--index-url`:
   - CUDA 13.0+ → default PyPI (no `--index-url` needed, PyTorch 2.11+ ships cu130 by default)
   - CUDA 12.6-12.9 → `cu128` (`https://download.pytorch.org/whl/cu128`)
   - CUDA 12.1-12.5 → `cu124` (`https://download.pytorch.org/whl/cu124`)
   - CUDA 11.8-12.0 → `cu118` (`https://download.pytorch.org/whl/cu118`)
   - Below 11.8 → unsupported, tell user to update driver
5. **Install PyTorch** — `pip install torch torchvision torchaudio` (with `--index-url` only if CUDA < 13.0)
6. **Verify CUDA works** — `import torch; torch.cuda.is_available()`. If False, diagnose: driver version mismatch, no GPU detected, etc.
7. **Install remaining deps** — `pip install -r requirements-lite.txt`
8. **Check ffmpeg** — if not on system PATH, download gyan.dev essentials build, unpack to `tools/ffmpeg/`.
9. **Create desktop shortcut** — PowerShell one-liner to create `.lnk` pointing to `launch.bat` with the app icon.
10. **Print summary** — what succeeded, what to do next.

Each step prints clear status. Failures stop with specific diagnostic messages, not cryptic errors.

### Why Python instead of pure batch

- Version comparison (parsing "12.6" > "12.1") is one line in Python, fifteen fragile lines in batch.
- `subprocess.run` with proper output capture vs batch's `for /f` parsing that breaks on unexpected whitespace.
- `try/except` around every step with specific diagnostics vs batch's silent failures.
- Won't break on localized Windows installs where nvidia-smi outputs in a different language.

---

## Layer 2: Setup Wizard (Lite)

### New file: `setup_wizard_lite.py`

Separate from the existing `setup_wizard.py` (which belongs to `release/v1` and is not modified).

### Behavior

- **First launch:** Wizard always appears, even if everything is healthy. Shows a confirmation screen with green checkmarks so the user sees their setup is verified. User clicks through to enter the app. Sets `"setup_wizard_completed": true` in `.studio_prefs.json`.
- **Subsequent launches:** If all checks pass and the completed flag exists, wizard skips silently. If any check fails, wizard reappears regardless of the flag.

### Checks (in order)

1. **CUDA available** — `torch.cuda.is_available()`, GPU name
2. **ffmpeg/ffprobe on PATH** — offer to download if missing (same gyan.dev flow as install script)
3. **Disk space** — adequate for model downloads
4. **Network connectivity** — can reach model download endpoints
5. **Model status + downloads:**
   - SAM3 (gated — HuggingFace token flow, same as current wizard)
   - RF-DETR (ungated, 129 MB)
   - YOLO26 (ungated, 6.5 MB)

If a runtime check fails (ffmpeg missing, model deleted, CUDA broke), the wizard reappears with specific guidance for the failing check.

---

## launch.bat

```batch
@echo off
set PATH=%~dp0tools\ffmpeg\bin;%PATH%
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0reconstruction_zone\reconstruction_zone.py"
```

- `%~dp0` resolves to the repo directory regardless of where the user puts it.
- `pythonw.exe` runs without a console window.
- Prepends local ffmpeg to PATH so the app finds it without system-wide install.

---

## Code Changes to reconstruction_zone.py

Surgical removals only:

### _build_ui()
- Remove `self.tabs.add("Projects")`, `self.tabs.add("Align")`
- Remove `build_projects_tab(...)`, `build_alignment_tab(...)` calls
- Remove Coverage tab if it exists

### __init__()
- Remove alignment binary state (`_alignment_binary_paths`, `_alignment_binary_status`, `_SPHERESFM_*` constants)
- Remove `_activity_store`

### Top-level imports
- Remove `from tabs.alignment_tab import build_alignment_tab`
- Remove `from tabs.projects_tab import build_projects_tab`

### _run_setup_wizard()
- Point to `setup_wizard_lite` instead of `setup_wizard`

### Everything else unchanged
- Mask tab (`_build_process_tab`) — unchanged
- Review tab (`_build_review_tab`) — unchanged
- Preview panel — unchanged
- All masking pipeline code — unchanged

---

## File Layout

```
reconstruction-zone/
├── install.bat                     # Thin wrapper
├── setup_install.py                # Python-based installer
├── launch.bat                      # Runs app via venv
├── requirements-lite.txt           # All deps incl YOLO, excl torch
├── QUICKSTART.txt                  # Plain text guide
├── README.md                       # Formatted guide
├── reconstruction-zone.ico         # App icon (existing)
├── tools/                          # Created by installer
│   └── ffmpeg/                     # Downloaded if needed
├── .venv/                          # Created by installer
│
├── reconstruction_gui/
│   ├── reconstruction_zone.py      # Modified: 3 tabs only, points to lite wizard
│   ├── setup_wizard_lite.py        # NEW: expanded health check + models
│   ├── setup_wizard.py             # Untouched (not used in lite)
│   ├── reconstruction_pipeline.py  # Unchanged
│   ├── review_gui.py               # Unchanged
│   ├── review_status.py            # Unchanged
│   ├── review_masks.py             # Unchanged
│   ├── app_infra.py                # Unchanged
│   ├── widgets.py                  # Unchanged
│   ├── model_paths.py              # Unchanged
│   ├── model_downloader.py         # Unchanged (already a stub)
│   ├── sam3_setup.py               # Unchanged
│   ├── _version.py                 # Unchanged
│   └── tabs/
│       └── source_tab.py           # Unchanged
│
└── prep360/                        # Unchanged
```

### Files excluded from lite (can be deleted from branch)

```
reconstruction_gui/
├── colmap_runner.py                # Align tab only
├── colmap_validation.py            # Align tab only
├── alignment_profiles.py           # Align tab only
├── tabs/alignment_tab.py           # Align tab
├── tabs/gaps_tab.py                # Coverage tab
├── tabs/projects_tab.py            # Projects tab

prep360/core/
├── gap_detector.py                 # Coverage tab
├── bridge_extractor.py             # Coverage tab
```

Optional advanced features (can be excluded if they add install complexity):
```
reconstruction_gui/
├── sam3_pipeline.py                # SAM3 video predictor
├── sam_refinement.py               # SAM boundary refinement
├── matting.py                      # ViTMatte alpha matting
├── vos_propagation.py              # LiVOS/Cutie temporal propagation
├── shadow_detection.py             # Shadow detection
```

These are all lazy-imported — their absence won't crash the app, just makes those features unavailable.

---

## Branch Strategy

- Create `release/v1-lite` from `release/v1`
- Make the changes described above
- Parallel branch — does not merge back

---

## What the Tester Gets vs. Full Version

| Feature | Full v1 | Lite |
|---------|---------|------|
| Extract tab (video analysis, frame extraction, fisheye, queue) | Yes | Yes |
| Mask tab (all models, all modes, all settings) | Yes | Yes |
| Review tab (thumbnails, filter/sort, OpenCV editor, layers) | Yes | Yes |
| Setup wizard with environment health checks | Models only | Full: CUDA, ffmpeg, disk, network, models |
| Project registry | Yes | No |
| COLMAP/SphereSfM alignment | Yes | No |
| Coverage gap detection | Yes | No |
| One-click install script | No (.exe) | Yes |
| Desktop shortcut | No (.exe) | Yes |
