# Reconstruction Zone v1 — Packaging Checklist

**Created:** 2026-05-10
**Tracks:** `exe-packaging-plan.md` (revised) + `2026-05-10-exe-packaging-plan-review.md`
**Branch:** `release/v1`

This document tracks every change made, opinions formed, decisions taken, and issues remaining during the .exe packaging effort. It serves as a living audit trail.

---

## Changes Made (completed work)

### Release branch trimming (2026-05-08)
- [x] Created `release/v1` from `main`, reset to `9418a0e`
- [x] Removed Adjust tab (tab registration, import, build call, 5 backend files)
- [x] Removed Coverage tab (tab registration, import, build call, `gaps_tab.py`)
- [x] Removed Detection & Refinement section from Mask tab (shadow, SAM refine, matting, ensemble, edge injection, COLMAP validation)
- [x] Removed Interactive Click Mode section from Mask tab (`sam3_click_session.py`)
- [x] Simplified `MaskConfig` — removed 20+ experimental fields
- [x] Simplified `MaskingPipeline` — removed init blocks, processing steps, methods for all removed features
- [x] Deleted 13 files (~9,400 lines)
- [x] Fixed `prep360/__init__.py` and `prep360/core/__init__.py` — removed imports of deleted `adjustment_recipe` and `color_pipeline`
- [x] Pushed `release/v1` to GitHub (commit `2741e55`)

### Model downloader rewrite (2026-05-08)
- [x] Rewrote `model_downloader.py` with SAM3/RF-DETR/YOLO26 cache detection functions
- [x] Added download functions using each library's native download path
- [x] Created `ModelDownloadDialog` with CTkToplevel UI
- [x] **FIXED** — old `model_downloader.py` replaced by `setup_wizard.py` with queue-based threading. Source smoke test confirmed no threading errors.
- [x] Model check re-enabled via `setup_wizard.py` (replaces the old disabled downloader)

### PyInstaller spec updates (2026-05-08)
- [x] Removed deleted modules from `hiddenimports` (shadow_detection, sam_refinement, matting, vos_propagation, gaps_tab, adjust_tab)
- [x] Added `activity_store` to hiddenimports
- [x] Added `pathex=['reconstruction_gui']` to resolve bare module imports (`_version`, `widgets`, etc.)
- [x] Added ffmpeg/ffprobe as binaries from `C:\Users\alexm\ffmpeg\bin\`
- [x] Created `scripts/pyinstaller_runtime_hook.py` for ffmpeg PATH injection
- [x] Removed `torch.testing` from excludes (torch needs it during import)
- [x] Tried `module_collection_mode` = `pyz+py` and `py` — neither fixed circular import
- [x] **ABANDONED** PyInstaller path for Python 3.14 + torch 2.10

### Nuitka builds (2026-05-08 to 2026-05-10)
- [x] Installed Nuitka 4.0.8 on Python 3.14
- [x] First build: succeeded but `--windows-console-mode=disable` + missing `--assume-yes-for-downloads`
- [x] Second build: added `--assume-yes-for-downloads`, Dependency Walker auto-downloaded
- [x] Build produces 917 MB exe, 8.5 GB dist folder, ~30 min build time
- [x] GUI launches, all 5 tabs render
- [x] ffmpeg/ffprobe bundled and present in output
- [ ] **PENDING** — `torch.library` singledispatch fix via YAML rewrite (6 decorators). Smoke compile in progress. Verdict when exe runs.
- [x] Tried `--nofollow-import-to=torch.library` — excluded module entirely, wrong approach
- [x] Tried `--nofollow-import-to=torch.backends.mps` — irrelevant

---

## Changes Planned (not yet done)

### Phase 1: App-level fixes (reordered: subprocess isolation first, per review)

- [x] **1a. Isolate external binary subprocess environments** (moved up — blocks all other testing because COLMAP DLL popups interrupt startup)
  - Create a dedicated helper (e.g., `_isolated_subprocess_env(binary_path)`) — do NOT overload `_subprocess_window_kwargs()` which is only for console window flags
  - Helper sets `cwd=binary_path.parent`, copies `os.environ`, prepends binary dir to PATH, removes/demotes conda/miniconda dirs
  - Apply to ALL subprocess call sites, not just COLMAP:
    - `ColmapRunner.validate_binary()` and `_run_cli_command()`
    - `prep360/core/analyzer.py` (ffprobe)
    - `prep360/core/extractor.py` (ffmpeg)
    - `prep360/core/sharpest_extractor.py` (ffmpeg/ffprobe)
    - `prep360/core/geotagger.py` (exiftool)
    - `prep360/core/osv.py` (ffmpeg/ffprobe)
    - `reconstruction_gui/tabs/alignment_tab.py` (COLMAP)
  - Prevents miniconda `tiff.dll` contaminating COLMAP startup

- [x] **1b. First-launch setup wizard** (replaces model_downloader.py)
  - Created `reconstruction_gui/setup_wizard.py` — complete wizard with queue-based threading
  - Created `reconstruction_gui/sam3_setup.py` — SAM3 gated model state machine
  - Handles all 3 models: SAM3 (gated), RF-DETR (free), YOLO26 (free)
  - SAM3 gate flow: token entry → verify → access check → download
  - Skip SAM3 option for users who can't get HF access
  - Queue-based threading: background threads never touch tkinter

- [x] **1c. Re-enable model check in `reconstruction_zone.py`**
  - Added `self.after(500, self._run_setup_wizard)` calling `run_setup_wizard_if_needed()`
  - Wizard appears only if models are missing; silent if everything is cached

- [x] **1d. Decide YOLO inclusion** → YES for friend-test (AGPL OK for private testing)

- [x] **1e. Add `dist_nuitka/` and `bench_output/` to `.gitignore`**

- [x] **1f. Source-run smoke test** — wizard appeared, detected missing YOLO26, downloaded 6.4MB, app launched. No COLMAP DLL popups. SAM3 and RF-DETR already cached.

### Phase 2: Minimal torch compile test

- [ ] **2a. Create `scripts/nuitka-torch-fix.yml`**
  - `torch.library`: `replacements_plain` for `@define.register` and `@impl.register`
  - `torch._refs`: `replacements_re` for 4 `@_to_dispatch.register` decorators with different types
  - Verify YAML syntax

- [x] **2b. Create `scripts/smoke_import_torch.py`**
  - Must import `torch`, `torch._refs`, and `torchvision`
  - Print version, CUDA status, and "OK"

- [ ] **2c. Compile smoke script with Nuitka** (~5 min, not 30 min)
  - Use explicit `C:\Python314\python.exe -m nuitka`
  - Include `--show-anti-bloat-changes` and `--show-source-changes=torch.library` and `--show-source-changes=torch._refs`
  - Include `--report=dist_test/nuitka-report.xml`

- [ ] **2d. Verify anti-bloat log shows all 6 replacements fired**
  - 2 in `torch.library` (`@define.register`, `@impl.register`)
  - 4 in `torch._refs` (`_to_device`, `_to_device_str`, `_to_dtype`, `_to_other`)
  - If fewer than 6 → regex mismatch, fix YAML before proceeding

- [ ] **2e. Run compiled exe**
  - PASS = prints torch version + "OK"
  - FAIL = diagnose (new singledispatch site? different error?) or fall back to Option A

- [ ] **2f. If minimal test fails → fall back to Option A**
  - Create `.venv312` from `C:\Python312\python.exe`
  - Install torch 2.6+cu124, rfdetr, ultralytics, einops, all requirements.txt deps
  - Verify source-run + inference
  - Build with PyInstaller

### Phase 3: Full GUI build

- [ ] Build with Nuitka (or PyInstaller if Option A) + console=attach
- [ ] Launch and verify: console output, no warnings, no DLL popups, all tabs, model download, masking, extraction

### Phase 4: Installer

- [ ] Install Inno Setup 6
- [ ] Create `scripts/installer.iss` with AppId, version, install-size, per-user install option
- [ ] Compile installer
- [ ] Test install → launch → uninstall
- [ ] Test launch with hostile PATH (miniconda first)

### Phase 5: Ship

- [ ] Write `TESTING.md`
- [ ] Final build with `--windows-console-mode=disable`
- [ ] Commit all packaging changes
- [ ] Push to GitHub
- [ ] Upload to Google Drive
- [ ] Share with tester

---

## Opinions and Judgments

### On the Nuitka YAML approach (Option B)
**My assessment: Worth one controlled attempt, but riskier than I initially thought.**

The `replacements_plain` mechanism is well-established (319 uses in Nuitka's own config, verified as simple `str.replace`). The `torch.library` fix is clean — 2 unique decorators, no collision risk.

The `torch._refs` fix is harder. All 4 decorators share the same `@_to_dispatch.register` text but dispatch on different types. `replacements_re` (regex `re.sub`) can handle this with multi-line context matching, but the patterns must exactly match the torch 2.10 source, including whitespace. If torch updates these files, the regex silently fails to match and the error returns.

**My opinion:** The minimal compile test (Phase 2b) is the right move. If `import torch` compiles and runs cleanly in 5 minutes, we've validated the approach without wasting 30 minutes on a full build. If it fails, we immediately know whether the `_refs` regex didn't match, or whether there's a 7th singledispatch site we didn't find.

### On the Python 3.12 fallback (Option A)
**My assessment: More work than it appears, but probably works.**

The review correctly identified that `rfdetr` and `einops` are not installed on 3.12, and SAM3 was only partially verified. Building a clean `.venv312` and proving the full dependency chain is ~1 hour of work before the actual PyInstaller build. But once deps are proven, PyInstaller + torch 2.6 is a well-trodden path.

**My concern:** Torch 2.6 → 2.10 is a significant gap. SAM3 was developed against torch 2.10. If there are subtle API incompatibilities, debugging them under a packaging constraint is painful. I'd rather make the Nuitka path work if possible.

### On the COLMAP DLL contamination
**My assessment: Real issue, easy fix, should be done first.**

The fix is straightforward — sanitize the subprocess `env` dict for external binary calls. This is ~20 lines of code in `colmap_runner.py`. It's independent of the packaging tool choice and improves the app for all users, not just packaged builds.

**My opinion:** This should have been caught earlier. Any machine with miniconda (common for ML users) would hit this. It's not specific to packaging.

### On the model downloader
**My assessment: The queue pattern is the correct fix and should be done before packaging.**

The review was right to call this a release blocker. A friend-testing build where the first masking attempt hangs for 10 minutes with no progress indication (because the downloader is disabled) is a bad experience. The queue pattern is ~40 lines of code and well-understood.

**Additional concern I want to flag:** SAM3 is a gated model on HuggingFace. The tester needs to have an HuggingFace account, request access to `facebook/sam3`, and be approved BEFORE the downloader will work. This needs to be in `TESTING.md` with clear instructions. If the tester doesn't have HF access, SAM3 masking won't work at all.

### On Option C (patching functools)
**Eliminated.** The review was right — patching `C:\Python314\Lib\functools.py` is global, fragile, and the wrong blast radius. Even as a last resort, a venv-local copy or runtime monkey-patch would be better than mutating the system Python.

### On the build size (8.5 GB dist)
**My assessment: Normal for a PyTorch CUDA app, but worth investigating.**

8.5 GB is large but not unusual. PyTorch + CUDA DLLs alone account for ~3-4 GB. torchvision adds ~500 MB. The remaining ~4 GB is the compiled app, other dependencies (scipy, sklearn, pandas, etc.), and data files. Some of these (sklearn, pandas, matplotlib) are pulled in transitively but may not be needed at runtime.

**Future optimization:** After the packaging works, we could trim unnecessary packages from the Nuitka includes. But this is polish, not a blocker.

### On the `pyproject.toml` discrepancy
The plan originally said "No pyproject.toml." The review correctly noted it exists with full package metadata. My plan's intent was "no installed-package workflow" — the app runs via direct script execution, not `pip install -e .` → `reconstruction-zone` CLI. The `pyproject.toml` exists for metadata but isn't used in the packaging flow. I've reworded this in the revised plan.

---

## Remaining Issues and Risks

### BLOCKING — must fix before any build is release-candidate

| # | Issue | Owner | Status | Estimated Effort | Order |
|---|---|---|---|---|---|
| 1 | COLMAP subprocess DLL contamination | App code | Not started | ~30 min (14 call sites) | **First** — blocks all other testing |
| 2 | Model downloader threading broken | App code | Not started | ~30 min | Second |
| 3 | Model check disabled in GUI | App code | Not started (blocked by #2) | ~5 min | Third |
| 4 | Nuitka singledispatch fix YAML | Build config | Not started | ~30 min (including smoke test) | Fourth |
| 5 | `dist_nuitka/` and `bench_output/` not in `.gitignore` | Git config | Not started | ~2 min | Anytime |

### HIGH PRIORITY — should fix before friend-test distribution

| # | Issue | Owner | Status | Risk if unfixed |
|---|---|---|---|---|
| 6 | SAM3 gated model — tester needs HF access | Documentation | Not started | Tester can't download SAM3, masking fails silently |
| 7 | Console mode = disable (no log output) | Build flag | Identified | Testing builds should use `attach` |
| 8 | exiftool not documented as dependency | Documentation | Identified | SRT geotagging silently fails |
| 9 | RF-DETR total download is ~1.2 GB not ~100 MB | Documentation | Corrected in plan | Tester surprised by download size |

### MEDIUM PRIORITY — should address eventually

| # | Issue | Owner | Status | Notes |
|---|---|---|---|---|
| 10 | Inno Setup installer needs AppId, install-size, per-user option | Build config | Not started | Functional without it, just less professional |
| 11 | License documentation (THIRD_PARTY_LICENSES.md) | Documentation | Not started | Not blocking friend-test, blocks wider release |
| 12 | Build size optimization (trim sklearn, pandas, matplotlib) | Build config | Not started | 8.5 GB is normal for CUDA app |
| 13 | `release/v1` 2 commits ahead of `origin/release/v1` | Git | Identified | Push after fixes are committed |
| 14 | Clean-cache model download test matrix | Testing | Not started | Verify downloads work from scratch |

### LOW PRIORITY / DEFERRED

| # | Issue | Owner | Notes |
|---|---|---|---|
| 15 | Nuitka warns Python 3.14 is "experimental" | External | Nuitka 4.0.8 message, may be resolved in newer Nuitka |
| 16 | COLMAP/SphereSfM not bundled | Feature | Detected from PATH, user must install separately |
| 17 | No auto-update mechanism | Feature | Out of scope for v1 friend-test |
| 18 | Hostile-PATH regression test | Testing | Launch with miniconda first on PATH, verify no native popups |
| 19 | Nuitka `re.sub` passes `re.S` as `count` not `flags` | External | Bug in `AntiBloatPlugin.py:496`. Does not affect our fix (we use `\n` not `.`). Note for awareness only. |

---

## Corrections Log

Tracking every factual correction made during the plan/review cycle.

| # | Original Claim | Correction | Source |
|---|---|---|---|
| 1 | `torch._refs` is lazy, not imported during startup | **WRONG.** `'torch._refs' in sys.modules` is `True` after `import torch` | Verified via `C:\Python314\python.exe` |
| 2 | RF-DETR download is ~100 MB | **WRONG.** 129 MB seg weights + 1.1 GB DINOv2 backbone = ~1.2 GB total | `requests.head()` for seg weights, `ls ~/.cache/torch/hub/checkpoints/romav2.pt` for backbone |
| 3 | Total model downloads ~3.5 GB | **WRONG.** ~4.5 GB (SAM3 3.3 + RF-DETR 1.2 + YOLO 0.006) | Sum of corrected sizes |
| 4 | YOLO26 weights 6 MB | Minor: **6.5 MB** | `du -sh` of HF cache blob |
| 5 | 30 singledispatch `.register` uses in torch.library | **WRONG.** 2 actual `@decorator.register` decorators (29 was all `.register()` calls) | `grep -c "@.*\.register$"` |
| 6 | `dist_nuitka/` and `bench_output/` not gitignored | **CORRECT** (review claim). Despite `git check-ignore -v` reporting a match on line 62 (a blank line — likely a git bug), `git status` shows them as `??` untracked. They need explicit ignore rules added. | `git status --porcelain` shows `??`, contradicting `git check-ignore` |
| 7 | Bare `python` resolves to miniconda | **Context-dependent.** Resolves to `C:\Python314` in bash on this machine. Resolves to miniconda in the Codex PowerShell sessions used for the review. Either way, build commands should use explicit paths. | `which python` varies by shell |
| 8 | No `pyproject.toml` | **WRONG.** Exists with full metadata. But no installed-package workflow is used. | `cat pyproject.toml` |
| 9 | exiftool not mentioned as dependency | **CORRECT** (review finding). It's checked at startup but was missing from the dependency table. | `grep -rn exiftool reconstruction_gui/` |
| 10 | SAM3 is freely downloadable | **WRONG.** Gated model on HuggingFace, requires manual access approval. | `model_info('facebook/sam3').gated == 'manual'` |
| 11 | Model downloader threading cause is console mode | **Overfit** (review finding). The real cause is tkinter's thread check in `_register()`, regardless of console mode. | Review analysis, agreed upon reflection |
| 12 | Total singledispatch decorators needing fix: 2 | **WRONG.** 6 total across 2 modules (2 in library, 4 in _refs), all loaded at import time | `grep` + `sys.modules` check |
| 13 | `app_infra.py` logging only goes to devnull | **WRONG** (my implied claim). After init, `_setup_console_redirect()` replaces stdout/stderr with queue/file writers. `crash.log` is always written. | Review finding, verified in source |
| 14 | Plan said "Untracked (all gitignored)" for dist_nuitka/bench_output | **WRONG.** `git status` shows `??`. `.gitignore` has no rules for these. `git check-ignore -v` reports line 62 (blank) — likely a git quirk, not real coverage. | Third-pass review + `git status --porcelain` |
| 15 | Plan said ffmpeg bundled at `dist/ffmpeg/` | **STALE.** Nuitka output is at `reconstruction_zone.dist/ffmpeg/`, not `dist/ffmpeg/`. | Third-pass review |
| 16 | PyInstaller conclusion: "structural impossibility" | **OVERSTRONG.** Softened to "not worth further time on Python 3.14 + torch 2.10 for this release." PyInstaller remains viable for Python 3.12 fallback. | Third-pass review |
| 17 | Smoke test used Unix `/tmp` path and bash syntax | **WRONG for target.** Plan targets Windows. Replaced with checked-in `scripts/smoke_import_torch.py` and PowerShell commands. | Third-pass review |
| 18 | Plan said to modify `_subprocess_window_kwargs()` for env isolation | **WRONG approach.** That helper is specifically for console window flags. Use a separate helper for env isolation. | Third-pass review |
| 19 | Phase 1 order: downloader first, subprocess later | **SUBOPTIMAL.** COLMAP DLL popup blocks all testing. Subprocess isolation must come first. | Third-pass review recommendation |
| 20 | Nuitka `AntiBloatPlugin.py:496` passes `re.S` as `count` to `re.sub` | **Nuitka bug** (count=16, not flags=DOTALL). Does not affect our fix — our patterns use `\n` (literal newline), not `.` (dot), so DOTALL is irrelevant. | Verified via `re.sub` signature inspection |
| 21 | Dependency table said ffmpeg bundled in `dist/ffmpeg/` | **STALE.** Nuitka output path is `reconstruction_zone.dist/ffmpeg/`. Changed to generic "packaged app's ffmpeg/ folder." | Fourth-pass review |
| 22 | Full GUI build command used bash `\` continuations | **WRONG for target.** Converted to PowerShell backtick continuations. | Fourth-pass review |
| 23 | Checklist subprocess detail was under task 1c (model check) | **MISPLACED.** Moved to task 1a (subprocess isolation) where it belongs. | Fourth-pass review |
| 24 | Checklist had duplicate `2c` numbering | **BUG.** Fallback item renamed to `2f`. | Fourth-pass review |
| 25 | Decision record #2 said "Structural incompatibility" | **OVERSTRONG.** Changed to "Not worth further time for this release." | Fourth-pass review |
| 26 | `scripts/smoke_import_torch.py` referenced but not created | **MISSING FILE.** Now created with torch, torch._refs, torchvision imports. | Fourth-pass review |

---

## Decision Record

| # | Decision | Rationale | Date |
|---|---|---|---|
| 1 | Keep YOLO/ultralytics in friend-test build | Private testing, AGPL OK. Strip for commercial/Gumroad release later. | 2026-05-08 |
| 2 | Abandon PyInstaller on Python 3.14 + torch 2.10 | 6 fix attempts, all failed. Not worth further time for this release. PyInstaller remains viable for Python 3.12 fallback. | 2026-05-08 |
| 3 | Try Nuitka as primary build tool | Compiles with CPython's native import system, avoids PyInstaller circular import issue. | 2026-05-08 |
| 4 | Bundle ffmpeg/ffprobe (250 MB) | Video extraction is core functionality. Can't ask testers to install ffmpeg separately. | 2026-05-08 |
| 5 | Models download on first use, not bundled | SAM3 alone is 3.3 GB. Bundling would make installer ~12+ GB. First-use download is standard pattern. | 2026-05-08 |
| 6 | Eliminate Option C (patch stdlib functools) | Too fragile, global side effects, wrong blast radius for a friend-test build. | 2026-05-10 |
| 7 | Fix app blockers BEFORE packaging (revised execution order) | Review correctly identified that packaging before fixing app issues wastes 30-min build cycles. | 2026-05-10 |
| 8 | Use minimal `import torch` compile test before full build | Avoids 30-min feedback loops. Validates YAML fix in ~5 min. | 2026-05-10 |
| 9 | Use explicit interpreter paths in all build commands | Miniconda can shadow `C:\Python314\python.exe` depending on shell. | 2026-05-10 |
| 10 | Reorder Phase 1: subprocess isolation before downloader | COLMAP DLL popups block all testing. Fix the environment first. | 2026-05-10 |
| 11 | Use separate helper for subprocess env isolation | `_subprocess_window_kwargs()` is for console flags only. Don't overload it. | 2026-05-10 |
| 12 | Add `--show-anti-bloat-changes` to smoke test | Must verify all 6 replacements fired, not just that the exe runs. | 2026-05-10 |
| 13 | Use checked-in smoke script, not ad-hoc `/tmp` | Auditable, cross-platform, committed to repo. | 2026-05-10 |
