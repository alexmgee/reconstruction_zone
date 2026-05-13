# Phantom Code Remediation — Full Progress Report (Updated)

Date: 2026-05-11T19:55
Author: Claude (self-assessment)
Plan: `planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md`

## Purpose

Exhaustive progress report against every item in the remediation plan, plus a
full audit of all 36 files modified by commit `2741e55`. Written to guide Codex
in identifying remaining issues and verifying Claude's work.

## Current Packaging Status After Phase 6A Reconciliation

The preserved Phase 6A retry artifact has now been reconciled against the report:

- The retry compile completed successfully.
- The preserved retry executable exits `0` from both the repo root and its
  `.dist` directory.
- The `_version` runtime failure previously attributed to that retry artifact is
  withdrawn as misattributed carry-forward from an earlier attempt.
- `--include-package=reconstruction_gui` is currently supported by the preserved
  retry outcome.
- A broad GUI import refactor is deferred because the preserved retry artifact
  does not justify it.

Therefore:

- **Phase 6A compiled import smoke: PASSED**
- **Next packaging concern: Phase 6B / full-build strategy, not a Phase 6A
  `_version` blocker**

---

## Phase 0: Stabilize Before Editing — COMPLETE

| Item | Status |
|------|--------|
| No build processes running | DONE |
| Working directory confirmed | DONE — `d:\Projects\reconstruction-zone` |
| Branch captured | DONE — `release/v1`, ahead of origin by 2 commits |
| Branch decision | Working on `release/v1` directly (no new branch) |
| Unrelated work preserved | DONE — surgical edits only |

---

## Phase 1: Remove Phantom Adjust Code From Source Tab — COMPLETE

Target: `reconstruction_gui/tabs/source_tab.py`

### 1A. Fix imports — DONE

- [x] Removed `AdjustmentRecipe` from `prep360.core` import block
- [x] Removed entire `adjust_workflow` import block (both paths + fallback)
- [x] No stub added
- [x] No `tabs.adjust_tab` import remains

Verification: `rg -n "AdjustmentRecipe|adjust_workflow|tabs\.adjust_tab|detect_adjust_input|export_adjusted_dataset"` → no matches

### 1B. Remove Adjust Handoff UI — DONE

- [x] Deleted `Adjust Handoff` CollapsibleSection
- [x] Deleted `app.extract_open_adjust_var`
- [x] Deleted `app.extract_adjust_recipe_enabled_var`
- [x] Deleted `app.extract_adjust_recipe_entry`
- [x] Deleted recipe browse row
- [x] Restored LUT section wording

Current LUT wording (line 956-957):
```
Section title: "Color & LUT"
Subtitle: ".cube LUT, shadow/highlight adjustment"
Checkbox: "Apply LUT after extraction"
```

Note: plan recommended subtitle `optional quick post-process after extraction`.
Claude used the original main-branch wording instead. This is functionally
correct and matches pre-phantom state.

Verification: `rg -n "Adjust Handoff|Open in Adjust|Adjust recipe|extract_open_adjust_var|extract_adjust_recipe"` → no matches

### 1C. Remove Adjust settings from `_snapshot_settings()` — DONE

- [x] Removed `adjust_recipe_enabled=...`
- [x] Removed `adjust_recipe_path=...`
- [x] Removed `open_adjust_after_extraction=...`
- [x] Real settings preserved: LUT, shadow/highlight, sky, blur, motion, sharpness, scene

Verification: `rg -n "adjust_recipe_enabled|adjust_recipe_path|open_adjust_after_extraction"` → no matches

### 1D. Remove Adjust-only helper functions — DONE

- [x] Deleted `_validate_adjust_recipe_before_extraction()`
- [x] Deleted `_export_adjusted_derivative()`
- [x] Deleted `_offer_open_in_adjust()`
- [x] Deleted `_run_post_processing_safely()` — restored direct `_run_post_processing()` calls

**`PostProcessingError` — KEPT, requires Codex review:**

`PostProcessingError` was introduced by commit `2741e55`. It is currently used
by real LUT validation code in `_run_post_processing()`:

- Line 1436: `raise PostProcessingError("LUT post-processing is enabled but no LUT path is set")`
- Line 1438: `raise PostProcessingError(f"LUT file not found: {settings.lut_path}")`
- Line 1445: `raise PostProcessingError(f"Could not load LUT '{settings.lut_path}': {e}")`

The original main-branch `_run_post_processing()` did not validate LUT paths —
it silently skipped if the path didn't exist (`if settings.lut_enabled and
settings.lut_path and Path(settings.lut_path).exists()`). The phantom commit
replaced this with explicit validation that raises `PostProcessingError`.

Claude kept this because it's a functional improvement (fail loudly vs silently
skip). However, this was code from a commit with proven fabrication, and Claude
did not flag the decision to the user. Codex should verify or reject.

**LUT API change — also from `2741e55`, requires Codex review:**

The phantom commit also changed the LUT application call from:
```python
# Original (main branch):
processed = processor.apply_lut(img, settings.lut_path, settings.lut_strength)

# Current (after phantom commit):
lut_3d, _lut_info = processor.load_cube(settings.lut_path)
processed = processor.apply_uint8(img, lut_3d, settings.lut_strength)
```

`apply_uint8` was added to `prep360/core/lut.py` by the same phantom commit
(+202 lines to lut.py). Verified that the method exists and the signatures
match:
- `LUTProcessor.load_cube(cube_path: str) -> Tuple[ndarray, LUTInfo]` ✓
- `LUTProcessor.apply_uint8(image_bgr: ndarray, lut_3d: ndarray, strength: float) -> ndarray` ✓

**Correction (from Codex oversight):** `apply_lut` still exists in `lut.py` as
a compatibility wrapper. Claude's original claim that "the old method no longer
exists" was false. The `apply_uint8` choice is a performance decision (load the
LUT once, apply per-image) rather than forced by absence of `apply_lut`.

### 1E. Remove call sites — DONE

- [x] 3x `_validate_adjust_recipe_before_extraction()` calls removed (single, queue, split-pair)
- [x] 3x `_export_adjusted_derivative()` calls removed
- [x] 3x `_offer_open_in_adjust()` calls removed
- [x] All extraction paths now call `_run_post_processing()` directly
- [x] Paired worker "Next:" line restored to original wording

Verification: `rg -n "validate_adjust|adjusted_derivative|offer_open_in_adjust|Open in Adjust|Adjusted derivative"` → no matches

---

## Phase 2: Remove Phantom Adjust Fields From Queue Settings — COMPLETE

Target: `prep360/core/queue_manager.py`

- [x] Removed `adjust_recipe_enabled` field
- [x] Removed `adjust_recipe_path` field
- [x] Removed `open_adjust_after_extraction` field
- [x] Removed from `from_dict()` deserialization
- [x] Removed `Adjust recipe` and `open Adjust` summary entries

Verification: `rg -n "adjust_recipe|open_adjust|Adjust recipe|open Adjust"` → no matches

---

## Phase 3: Remove Phantom Coverage/Gaps Wiring — COMPLETE

Target: `reconstruction_gui/tabs/alignment_tab.py`

- [x] Removed `Send To Coverage` button creation
- [x] Removed `app.alignment_send_to_coverage_btn` references
- [x] Deleted `_send_alignment_to_coverage()` function entirely
- [x] Removed button enable on completion (line ~2646)
- [x] Removed auto-send on success (line ~2649)

Verification: `rg -n "Coverage|gaps_tab|send_to_coverage|Send To Coverage"` → no matches

---

## Phase 4: Fix `prep360` Import Hygiene — COMPLETE

### 4A. Source tab imports

- [x] Removed `AdjustmentRecipe` from broad import block (this was the poison pill)
- [ ] Did NOT replace broad `from prep360.core import (...)` with direct imports

Assessment: The broad import block now succeeds because `AdjustmentRecipe` is
removed. All remaining symbols resolve. The plan says to prefer direct imports
"where practical" — Claude deferred this.

### 4B. Make segmentation exports lazy — DONE

- [x] Removed `from .segmenter import Segmenter, SegmentConfig, COCO_CLASSES, CLASS_PRESETS` from `prep360/core/__init__.py`
- [x] Added `__getattr__` for lazy loading of those 4 symbols
- [x] Kept all 4 in `__all__`
- [x] Removed eager segmenter re-exports from `prep360/__init__.py`
- [x] Added matching `__getattr__` in `prep360/__init__.py`

Verification:
```python
import prep360.core
# {'ultralytics': False, 'transformers': False, 'rfdetr': False}
```
PASS.

### 4C. CLI segmentation behavior — ALREADY CORRECT

`cli.py:476` imports `Segmenter` inside `cmd_segment()` function, not at module level.

---

## Phase 5: Source Verification Gates — ALL PASS

| Gate | Result |
|------|--------|
| 1: No phantom terms in live source | PASS |
| 2: Local missing import audit | PASS (inline AST audit, no persistent script created) |
| 3: `HAS_PREP360=True` | PASS |
| 4: GUI shell import clean (no blocked packages) | PASS |
| 5: Source tab import clean | PASS |
| 6: `py_compile` all modified files | PASS |
| 7: Focused tests (18/18) | PASS |
| 8: GUI smoke launch | PASS — app launched, detected binaries, wizard confirmed models ready |

Update: The persistent `scripts/audit_local_imports.py` audit now exists and has
been rerun successfully. Codex also strengthened it to cover plain
`import local.module` statements and relative import resolution, bringing it
closer to the remediation plan's stated requirements.
Gap: Gate 8 was log-based, not a visual UI inspection.

---

## Phase 6: Resume Packaging — PHASE 6A PASSED; PHASE 6B REMAINS

### Earlier Constrained GUI Attempt — FAILED (Transformers stall; historical context)

The build was launched as background task `bo7ueau3m` at 7:23 PM. After ~25
minutes, process inspection revealed:

- PID 89172: Nuitka main process, 713s CPU, 4.8GB working set, **CPU stalled**
- Child PID 67276: Running `from transformers.utils.import_utils import
  define_import_structure` on `transformers/models/qwen2_5_vl/__init__.py`

**This is the exact Transformers plugin stall that Codex warned about.**

Root cause: The source-level gates prove that `ultralytics` and `transformers`
do not load during normal Python import. However, Nuitka performs static
dependency analysis — it sees `ultralytics` in the Python package metadata
(installed as a pip dependency), activates its Transformers plugin, and walks
the entire `transformers.models.*` tree regardless of whether the code actually
imports it at runtime.

The lazy `__getattr__` in `prep360/core/__init__.py` prevents runtime loading
but does NOT prevent Nuitka's static analysis from following the dependency
graph.

**Fix needed:** Add `--nofollow-import-to` exclusions for `transformers` and
possibly `ultralytics` to the Nuitka command. These packages should only be
loaded at runtime when the user actually triggers masking, not bundled into the
standalone executable's import graph.

**Correction (from Codex oversight):** `--noinclude-transformers-mode` was not
observed in local `nuitka --help` or `--help-all` output. This option is
speculative and should not be relied upon.

The build was killed (both PIDs) after diagnosis.

This historical stall does not override the later artifact-reconciled Phase 6A
retry result, which passed at the compiled import smoke level.

### Build Command Issues

1. `| tail -5` on the build command prevented all output monitoring (tail
   buffers until EOF)
2. `tasklist /FO CSV | grep python` returned 0 lines due to shell encoding —
   PowerShell `Get-Process` was needed instead
3. Pre-existing `smoke_import_gui.build/` and `.dist/` directories from Codex's
   prior attempts were not cleared, making timestamp analysis unreliable

---

## Phase 7: Full-Scale Audit — COMPLETE

### Audit A: Deleted-file reference audit — PASS

Searched all `.py` and `.spec` files for references to modules deleted by
`2741e55`:

```
rg -n "adjust_tab|gaps_tab|adjust_workflow|color_pipeline|adjustment_recipe|shadow_detection|sam_refinement|vos_propagation" -g "*.py" -g "*.spec"
```

Result: No matches.

### Audit B: Missing local import audit — PASS (inline)

AST-based audit of all Python files under `reconstruction_gui/` and `prep360/`
found no unresolved local imports.

Persistent `scripts/audit_local_imports.py`: present and passing on the current
tree (`No unresolved local imports found.`).

### Audit C: Package `__init__.py` eager import audit — DONE

Import weight classification of all `prep360/core/__init__.py` exports:

| Module | Weight | Loads |
|--------|--------|-------|
| `prep360.core.analyzer` | medium | `cv2` |
| `prep360.core.extractor` | light | — |
| `prep360.core.reframer` | light | — |
| `prep360.core.presets` | light | — |
| `prep360.core.sky_filter` | light | — |
| `prep360.core.lut` | light | — |
| `prep360.core.adjustments` | light | — |
| `prep360.core.blur_filter` | light | — |
| `prep360.core.colmap_export` | light | — |
| `prep360.core.osv` | light | — |
| `prep360.core.fisheye_calibration` | light | — |
| `prep360.core.motion_selector` | light | — |
| `prep360.core.gap_detector` | light | — |
| `prep360.core.fisheye_reframer` | light | — |
| `prep360.core.bridge_extractor` | light | — |
| `prep360.core.sharpest_extractor` | light | — |
| `prep360.core.paired_split_video_extractor` | light | — |
| `prep360.core.segmenter` | **heavy** | `ultralytics` → `transformers` |

Only `analyzer` pulls `cv2`. Segmenter is already lazy via `__getattr__`.
No `torch` from any module at import time.

### Audit D: Optional import block audit — PASS

All `HAS_*` flags in the codebase guard real external optional dependencies:

| Flag | Module | Guards |
|------|--------|--------|
| `_HAS_VTK` | pointcloud_viewer.py | vtk (3D viewer) |
| `HAS_PY360` | reframer.py | py360convert |
| `HAS_YOLO` | segmenter.py | ultralytics |
| `HAS_SAM3_VIDEO` | sam3_pipeline.py | sam3 video API |
| `HAS_TORCH` | sam3_pipeline.py | torch |
| `HAS_SAM3` | reconstruction_pipeline.py | sam3 |
| `HAS_FASTSAM` | reconstruction_pipeline.py | FastSAM |
| `HAS_YOLO` | reconstruction_pipeline.py | ultralytics |
| `HAS_RFDETR` | reconstruction_pipeline.py | rfdetr |
| `HAS_OPENCV` | reconstruction_pipeline.py | cv2 |
| `HAS_PREP360` | source_tab.py | prep360.core |
| `_HAS_CUDACODEC` | source_tab.py | CUDA video decode |
| `HAS_PIL` | source_tab.py | PIL/Pillow |

None reference deleted modules. All guard real external packages.

### Audit E: UI references to removed tabs/features — PASS

```
rg -n "Click Mode|SAM refinement|alpha matting|shadow detection|edge injection|COLMAP validation" reconstruction_gui/ -g "*.py"
rg -n 'tabs\.set.*Adjust|tabs\.set.*Coverage|tabs\.set.*Gaps' reconstruction_gui/ -g "*.py"
```

No matches. No UI buttons, tab switches, or labels reference removed features.

Note: Two comments in `reconstruction_zone.py` mention "click-PVS" (lines 620,
1304). These are comments describing mouse click bindings for the preview
overlay, not references to the removed Click Mode feature. The bindings
themselves call `_on_click_overlay` which exists in the file.

### Audit F: Spec and packaging hook audit — PASS

```
rg -n "adjust|gaps|shadow_detection|sam_refinement|matting|vos_propagation|coverage|model_downloader" reconstruction_zone.spec scripts/ -g "*.py" -g "*.ps1" -g "*.spec"
```

No matches. Spec file and scripts are clean.

### Audit G: Planning-doc leakage audit — PASS (source clean, docs remain)

No live `.py` or `.spec` source references planned-but-unimplemented APIs.

5 planning documents from the phantom commit still exist on disk:
- `planning/plans/adjust-tab-plan.md` (709 lines)
- `planning/plans/adjust-tab-response-plan.md` (289 lines)
- `planning/plans/click-video-propagation-plan.md` (494 lines)
- `planning/plans/click-video-propagation-plan-review.md` (146 lines)
- `planning/plans/click-video-propagation-implementation-summary.md` (291 lines)

These describe future features and contain no executable code. Per the plan:
"Planning documents may remain, but source code must not import or call planned
APIs that are not implemented." Source is clean.

Recommendation: Add a status header to each doc noting "Future plan only. Not
implemented in v1."

### Audit H: Import side-effect matrix — DONE

| Module | Blocked | Discouraged | HAS_* flags |
|--------|---------|-------------|-------------|
| `prep360` | none | `cv2` | — |
| `prep360.core` | none | none | — |
| `tabs.source_tab` | none | none | `HAS_PIL=True`, `HAS_PREP360=True` |
| `tabs.alignment_tab` | none | none | — |
| `reconstruction_gui.reconstruction_zone` | none | none | — |
| `reconstruction_pipeline` | `ultralytics`, `transformers`, `rfdetr` | `torch`, `cv2` | Expected — this is the masking runtime |

`reconstruction_pipeline` correctly loads heavy packages because it IS the
masking runtime. All other modules are clean at import time.

### Audit I: Commit-intent sanity check

Applied to `2741e55`:
- Commit says "trim" → net adds 919 lines. **RED FLAG confirmed.**
- 3 phantom modules imported. **Confirmed and fixed.**
- 1,929 lines of planning docs for features being "removed." **Anomalous but
  docs are harmless.**

Future prevention: Any commit that says "remove/trim/simplify" should be
checked with `git show --stat` to verify net line count is negative.

### Audit J: Documentation and handover cleanup

```
rg -n "adjust_tab\.py|gaps_tab\.py|adjust_workflow\.py|adjustment_recipe\.py|color_pipeline\.py" -g "*.md"
```

References found only in incident/audit/remediation reports (this is expected
and correct — historical references in reports).

No current implementation docs describe deleted modules as present.

---

## Remaining Files From `2741e55` — Audit Results

### Files with no phantom code found

| File | Lines added | Assessment |
|------|-------------|------------|
| `reconstruction_gui/reconstruction_zone.py` | +1400 | Scanned for phantom imports and feature references. Found only `adjustment_manifest.json` (a data file check, not a module import) and "click-PVS" comments (describing real mouse bindings). No phantom code. The large additions are real features: preview pan/zoom, pipeline caching, layer discovery, paired dataset masking. |
| `reconstruction_gui/colmap_runner.py` | +412 | Found `adjustment_manifest_path: ""` — a metadata field in run info dict, not a module import. Stores path to manifest file if it exists on disk. Benign. |
| `reconstruction_gui/review_gui.py` | +197 | No phantom references. |
| `reconstruction_gui/review_masks.py` | +332 | No phantom references. |
| `reconstruction_gui/review_status.py` | +98 | No phantom references. |
| `reconstruction_gui/app_infra.py` | +14 | No phantom references. |
| `reconstruction_gui/colmap_validation.py` | +16 | No phantom references. |
| `prep360/core/sharpest_extractor.py` | +207 | No phantom references. Real feature additions. |
| `prep360/core/paired_split_video_extractor.py` | +57 | No phantom references. |
| `prep360/core/osv.py` | +2 | Trivial change, no phantom references. |
| `AGENTS.md` | +121 | Codex instruction file. No phantom code. |
| `README.md` | +11 | Documentation update. |
| `docs/SAM3_INTEGRATION_PLAN.md` | +165 | Documentation. |
| `.gitignore` | +1 | Trivial. |
| `tests/conftest.py` | +10 | Test config. |
| `tests/test_review_status.py` | +86 | Test additions. |
| `tests/test_sharpest_extraction.py` | +84 | New test file. |
| `reconstruction_gui/docs/EXTRACT_TAB.md` | +50 | Doc update. |

### Files requiring Codex review

| File | Lines added | Issue |
|------|-------------|-------|
| `prep360/core/lut.py` | +202 | New methods added: `apply_float`, `_apply_float_cuda`, `_apply_float_numpy`, `apply_uint8`, `apply_lut_uint8`, `apply_lut_float`, `apply_lut`, `_has_torch_cuda`. These are real functional additions (verified signatures match call sites). But they came from a commit with proven fabrication and were not audited for correctness. |

---

## Earlier Packaging Blocker Observed Before The Successful Phase 6A Retry

### What happened

The source-level fix (lazy `__getattr__` for segmenter) prevents runtime loading
of `ultralytics`/`transformers`. All 8 source gates pass — normal Python import
does not load these packages.

However, Nuitka performs static analysis. It follows ALL import statements in
ALL reachable code, including imports inside functions that are never called
during startup. `reconstruction_pipeline.py` contains lazy imports inside
`_import_pipeline()`:

```python
def _import_pipeline():
    from reconstruction_pipeline import (...)  # includes YOLO, RF-DETR, SAM3
```

Nuitka sees these imports statically, follows `ultralytics` into its dependency
graph, discovers `transformers` as a dependency, activates its Transformers
plugin, and walks every model in `transformers.models.*`. The child process was
found running `define_import_structure` on `transformers/models/qwen2_5_vl/`.

### Root cause

The lazy `__getattr__` in `prep360/core/__init__.py` solves the wrong layer of
the problem. It prevents Python's runtime import machinery from loading
`ultralytics`, but Nuitka's static analyzer doesn't use `__getattr__` — it
parses AST import statements directly. The actual import statements in
`reconstruction_pipeline.py` are the entry point Nuitka follows.

### Decisions made

**PostProcessingError**: KEEPING. The original main-branch code silently skipped
broken LUT configs (`if settings.lut_enabled and settings.lut_path and
Path(settings.lut_path).exists()`). The current code fails with a clear error
message. That's better UX — the user learns their LUT path is wrong instead of
silently getting no color correction. The class is used exclusively by real LUT
validation code, not by any Adjust phantom code.

**LUT API** (`apply_lut` → `apply_uint8`): KEEPING. `apply_uint8` exists in
`lut.py` with verified signatures matching the call sites. `apply_lut` still
exists as a compatibility wrapper, so this is not a forced migration; it is a
deliberate choice to use the newer load-once / apply-many flow. Reverting would
be churn without a demonstrated benefit.

---

## Proposed Build Plan

### Phase 6A: Constrained GUI Nuitka Smoke (completed via retry)

**Pre-build cleanup:**

```powershell
# Clear stale build artifacts from prior attempts
Remove-Item -Recurse -Force dist_test\smoke_import_gui.build, dist_test\smoke_import_gui.dist -ErrorAction SilentlyContinue
```

**Command (revised):**

The previous command stalled because Nuitka followed `reconstruction_pipeline.py`'s
lazy imports into `ultralytics` → `transformers`. For the GUI smoke, these
packages are not needed — the smoke only tests that the GUI shell imports and
renders version info.

Add `--nofollow-import-to` exclusions for all packages that are lazy-loaded at
masking runtime:

```
--nofollow-import-to=ultralytics
--nofollow-import-to=ultralytics.*
--nofollow-import-to=transformers
--nofollow-import-to=transformers.*
--nofollow-import-to=rfdetr
--nofollow-import-to=rfdetr.*
--nofollow-import-to=sam3
--nofollow-import-to=sam3.*
```

These are safe for the smoke because the smoke script does not trigger masking.
It only imports the GUI module and prints the version.

**Do NOT use `| tail -5`.** Stream output directly so progress is visible:

```powershell
C:\Python314\python.exe -m nuitka `
  --standalone `
  --assume-yes-for-downloads `
  --windows-console-mode=attach `
  --enable-plugin=tk-inter `
  --enable-plugin=no-qt `
  --nofollow-import-to=bitsandbytes `
  --nofollow-import-to="bitsandbytes.*" `
  --nofollow-import-to=triton `
  --nofollow-import-to="triton.*" `
  --nofollow-import-to=ultralytics `
  --nofollow-import-to="ultralytics.*" `
  --nofollow-import-to=transformers `
  --nofollow-import-to="transformers.*" `
  --nofollow-import-to=rfdetr `
  --nofollow-import-to="rfdetr.*" `
  --nofollow-import-to=sam3 `
  --nofollow-import-to="sam3.*" `
  --nofollow-import-to="transformers.integrations.bitsandbytes" `
  --user-package-configuration-file=scripts/nuitka-torch-fix.yml `
  --include-package=customtkinter `
  --include-data-dir=reconstruction_gui/docs=reconstruction_gui/docs `
  --include-data-dir=resources=resources `
  --include-data-files=reconstruction-zone.ico=reconstruction-zone.ico `
  --output-dir=dist_test `
  scripts/smoke_import_gui.py
```

**Monitoring:**

- Run in foreground with a long timeout (Codex's torch smoke took ~2.5h)
- If using background execution, do NOT pipe through `tail` — let output stream
  directly to the output file
- Check for the Transformers stall pattern: child process running
  `define_import_structure` on `transformers/models/*/`. If seen, kill
  immediately — the exclusions didn't work
- Use PowerShell `Get-Process` to check for python processes, not `tasklist`
  with grep (which failed due to encoding issues)

**Success criteria:**

- `dist_test/smoke_import_gui.dist/smoke_import_gui.exe` exists
- Running the preserved executable exits `0` with no traceback from:
  - the repo root;
  - the `.dist` directory.
- Visible `GUI_IMPORT_OK` output is not required for this artifact because
  stdout/stderr are suppressed by the packaged runtime path under the observed
  launch conditions.

**Stop conditions:**

- Child process enters Transformers model-tree analysis → kill, investigate
  which import path bypassed the exclusions
- Build exceeds 3 hours → kill, reassess approach
- Fatal Nuitka error → read error, do not retry blindly

### Phase 6B: Full Build

Only after the GUI smoke passes. The full build needs a different exclusion
strategy because the packaged app DOES need `ultralytics`, `transformers`, and
`rfdetr` at masking runtime.

**Original proposal:** Use `--nofollow-import-to=transformers.models.*` instead
of excluding all of `transformers`.

**Codex oversight update, 2026-05-11:** Do not treat that proposal as approved
yet. Local probes in the current environment show that `import rfdetr`, importing
`RFDETRSegSmall`, and importing `rfdetr.models.backbone.dinov2` each load ten
`transformers.models.*` modules already. That means a blanket
`transformers.models.*` exclusion overlaps with modules the present import path
actually touches, not just with irrelevant deep model-tree traversal.

The full build command will need:

```
# Remove the blanket transformers/ultralytics/rfdetr exclusions from the smoke
# Replace with targeted model-tree exclusion:
--nofollow-import-to="transformers.models.*"

# Keep these from the smoke:
--nofollow-import-to=bitsandbytes
--nofollow-import-to="bitsandbytes.*"
--nofollow-import-to=triton
--nofollow-import-to="triton.*"

# Add ML package includes so they're bundled for runtime:
--include-package=ultralytics
--include-package=rfdetr
--include-package=transformers
--include-package=torch
--include-package=torchvision
```

Risk: `--nofollow-import-to="transformers.models.*"` might exclude model
classes that RF-DETR imports at runtime (e.g., DINOv2 lives under
`transformers.models.dinov2`).

```
# Rejected sketch, retained only as historical context:
# do not promote blanket or guessed model-tree exclusions
# into the next full build command without a smaller
# RF-DETR-specific packaging proof first.
```

This is fragile. A better approach is to treat the blanket exclusion as an
unapproved hypothesis and validate any candidate strategy against a smaller
RF-DETR-specific packaging smoke before promoting it into the full GUI build.
That bounded smoke now exists as:

```text
scripts/smoke_import_rfdetr.py
```

and is specified in:

```text
planning/plans/2026-05-11-phase-6b-rfdetr-packaging-smoke-plan.md
```

**Import-footprint probe already run:**

```python
# Capture the eager Transformers model modules imported immediately by rfdetr
import rfdetr
import sys
used_models = [m for m in sys.modules if m.startswith('transformers.models.')]
print(used_models)
```

Observed result in the current environment:

```text
transformers.models.auto
transformers.models.auto.auto_factory
transformers.models.auto.configuration_auto
transformers.models.auto.modeling_auto
transformers.models.auto.tokenization_auto
transformers.models.bloom
transformers.models.bloom.configuration_bloom
transformers.models.bloom.modeling_bloom
transformers.models.encoder_decoder
transformers.models.encoder_decoder.configuration_encoder_decoder
```

This tells us what eager import reaches today. It does **not** fully determine
what RF-DETR model construction or inference will need. See
`2026-05-11-phase-6b-strategy-oversight-report.md` for the recommended bounded
next proof.

Source-run baseline for the new bounded smoke has also been executed:

```text
RFDETR_VERSION 1.4.1
TRANSFORMERS_VERSION 4.57.6
RFDETR_CLASSES_OK RFDETRSegNano,RFDETRSegSmall,RFDETRSegMedium,RFDETRSegLarge
TRANSFORMERS_MODEL_MODULE_COUNT 10
RFDETR_IMPORT_OK
```

Execution update, 2026-05-12:

- The first bounded packaging control that force-included all of
  `transformers` failed immediately with a Nuitka fatal in
  `transformers.commands.add_new_model_like`.
- The narrower static-follow control without `--include-package=transformers`
  is not yet classified. A 30-minute direct run did not complete before timeout,
  and the runner failed to preserve usable telemetry.
- The optional `--nofollow-import-to="transformers.models.*"` trial has **not**
  been run, because the control baseline is still inconclusive.
- A follow-up 60-second direct foreground diagnostic confirmed that the A2
  command does emit live Nuitka progress text when run without the silent wrapper
  path. That narrows the telemetry defect to the earlier launcher/log-capture
  method, not to Nuitka itself.
- A later authoritative direct A2 run was intentionally interrupted by the user
  after about 44 minutes and 44 seconds. No executable or A2 report was produced.
  The interruption left build-era Python processes alive; Codex stopped only
  those build-era processes. A2 remains inconclusive/interrupted, not pass/fail.
- A subsequent long A2 run failed on a Nuitka AppData module-cache
  `PermissionError`, not on RF-DETR or Transformers import resolution.
- `--disable-cache=bytecode` did not avoid that module-cache write path.
- Setting `NUITKA_CACHE_DIR` to
  `D:\Projects\reconstruction-zone\dist_test\nuitka-cache` solved the cache
  permission blocker. A2 then compiled successfully and the packaged
  `smoke_import_rfdetr.exe` exited `0` from both repo-root and `.dist` launches.
  Marker text remained suppressed/inconclusive, but runtime exit status was
  clean.
- See
  `planning/plans/2026-05-12-phase-6b-rfdetr-packaging-smoke-result.md`
  for the detailed result ledger.

### Phase 6C: Post-Build Steps

1. Copy OpenCV CUDA DLLs:
   ```powershell
   .\scripts\copy_cv2_cuda_dlls.ps1 -DistDir dist_test\smoke_import_gui.dist
   ```

2. Run the packaged exe:
   ```powershell
   .\dist_test\smoke_import_gui.dist\smoke_import_gui.exe
   ```
   Expected: exit code `0` with no traceback. Visible version /
   `GUI_IMPORT_OK` text may be suppressed by the runtime stdout redirect.

3. For the full build, additionally test:
   - GUI launches and shows all 5 tabs
   - Extract tab is functional (HAS_PREP360=True equivalent)
   - No Adjust handoff UI visible
   - No Send To Coverage button visible
   - Setup wizard appears or confirms models ready

### Phase 6B Full GUI Build Result, 2026-05-12

Codex ran the full GUI Nuitka build requested by the user using:

```text
NUITKA_CACHE_DIR=D:\Projects\reconstruction-zone\dist_test\nuitka-cache
no --include-package=transformers
no --nofollow-import-to=transformers.models.*
scripts\nuitka-torch-fix.yml retained
```

Result:

```text
build exit code: 0
Nuitka report completion: yes
compiled/linked C files: 9492
artifact:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Post-build OpenCV CUDA/NVIDIA DLL copy also passed:

```text
Copied 16 DLLs into reconstruction_zone.dist\cv2
```

Runtime smoke classification:

- **Sandbox launch:** fails with `PermissionError` opening
  `C:\Users\alexm\.reconstruction_zone\crash.log`; this is consistent with
  sandbox write restrictions, not with a normal-user launch failure.
- **Outside-sandbox repo-root launch:** process stayed alive past 20 seconds,
  detected COLMAP/SphereSfM, and logged `All models ready — setup wizard not
  needed`.
- **Outside-sandbox `.dist` launch without model override:** process stayed
  alive but logged `Models need setup: RF-DETR-Seg, YOLO26-n-seg` and wrote a
  fatal Tk callback traceback:
  `TypeError: 'ReconstructionZone' object is not callable`.
- **Outside-sandbox `.dist` launch with
  `RECONSTRUCTION_ZONE_MODEL_DIR=D:\Projects\reconstruction-zone`:** process
  stayed alive past 20 seconds and logged `All models ready — setup wizard not
  needed`; no new fatal traceback appeared after that probe.

Current classification:

- **Phase 6B compile/package strategy:** passed under the approved constraints.
- **Packaged runtime readiness:** partial. The model-satisfied launch path is
  viable, but the first-launch/missing-model setup-wizard path is a blocker.

Do not start another blind full build for this issue. The next bounded proof
should isolate the compiled `SetupWizard`/CustomTkinter top-level path and/or
fix model weight discovery for packaged launches.

Detailed report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-build-result.md
```

### Setup Wizard Runtime Fix, 2026-05-12

Codex executed the next bounded proof and found the setup-wizard crash root
cause.

The problem was not primarily a Nuitka compiler failure. `SetupWizard` assigned:

```text
self._root = parent
```

on a `CTkToplevel` instance. That shadowed Tkinter's internal `_root()` method.
When Tkinter later handled an event and attempted to call `_root()`, it raised:

```text
TypeError: 'ReconstructionZone' object is not callable
```

Fix applied:

```text
self._root -> self._app_root
```

Also added:

- `reconstruction_gui/app_paths.py`;
- `scripts/smoke_setup_wizard.py`;
- `scripts/run_packaged_new_install_probe.ps1`.

Validated:

- source direct setup-wizard smoke: pass;
- source `run_setup_wizard_if_needed` smoke: pass;
- packaged direct setup-wizard smoke: pass;
- packaged `run_setup_wizard_if_needed` smoke: pass.

The next full GUI rebuild is now justified, but the existing full GUI artifact
was built before this fix. Do not use that old artifact to judge the fixed
first-launch path.

Detailed result:

```text
planning/plans/2026-05-12-new-install-setup-wizard-validation-result.md
```

### Full GUI Rebuild After Setup Wizard Fix, 2026-05-12

Codex rebuilt the full GUI with the setup-wizard fix and app-home path changes.

Result:

```text
build exit code: 0
report completion: yes
artifact:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Post-build:

```text
scripts\copy_cv2_cuda_dlls.ps1 copied 16 OpenCV CUDA/NVIDIA DLLs into the bundle.
```

Isolated new-install probes:

- **Empty app-home:** process stayed alive for 20 seconds, logged
  `Models need setup: SAM3, RF-DETR-Seg, YOLO26-n-seg`, no fatal traceback in
  isolated crash log.
- **RF-DETR/YOLO fixture app-home:** process stayed alive for 20 seconds, logged
  `Models need setup: SAM3`, no fatal traceback in isolated crash log.

Current state:

```text
full GUI compile/package: pass
first-launch setup-wizard liveness: pass
isolated RF-DETR/YOLO model discovery: pass
real download/token flows: pass
visual setup-wizard initial screen: pass
complete app-state vacuum in current artifact: fail, fixed in source after build
```

Detailed rebuilt-artifact report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-newinstallfix-result.md
```

### Real Model Install Workflow Result, 2026-05-12

Codex exercised the real model install/download/token workflows in isolated app
homes.

Result:

```text
RF-DETR real download: pass
YOLO26 real download: pass
SAM3 no-token state: pass
SAM3 saved-token verification: pass
SAM3 access granted: pass
SAM3 real weight download: pass
packaged GUI all-models-ready recognition: pass
```

The final packaged GUI probe logged:

```text
All models ready — setup wizard not needed
```

No token values were printed.

Detailed report:

```text
planning/plans/2026-05-12-real-model-install-workflow-result.md
```

### Visual Vacuum Probe And App-State Leak, 2026-05-12

Codex ran the visible packaged app in an isolated empty app-home/model/cache
state.

Result:

```text
setup wizard visible: pass
missing models listed: SAM3, RF-DETR-Seg, YOLO26-n-seg
ready count: 0/3
fatal traceback: none
```

This confirms the missing-model setup wizard is visibly present in the rebuilt
packaged artifact.

However, the screenshot also showed the Projects tab reading:

```text
D:\tracker.json
```

That is a first-launch app-state isolation failure. It does not invalidate the
model download/token results, but it means the artifact is not a complete blank
new-install candidate.

Source fix applied after that artifact:

- project tracker default now resolves through
  `app-home\projects\tracker.json`;
- activity log default now resolves through
  `app-home\projects\activity_log.json`;
- explicit legacy/custom `tracker_store_path` preferences are still preserved.

Verified:

```text
ProjectStore().store_path points at isolated app-home
py_compile passes with project-local PYTHONPYCACHEPREFIX
```

Updated classification:

```text
model/setup-wizard visible vacuum: pass
complete app-state vacuum in current artifact: fail
complete app-state vacuum in source: fixed, pending final rebuild
```

### Final-Vacuum Rebuild Result, 2026-05-12

Codex ran the final rebuild after the app-state/project-store vacuum fix.

Final artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Result:

```text
build exit code: 0
Nuitka XML completion: yes
compiled C files: 9494
clcache hits/misses: 9486 / 8
OpenCV CUDA/NVIDIA DLL copy: pass
```

Final visible blank-vacuum probe:

```text
setup wizard visible: pass
missing models listed: SAM3, RF-DETR-Seg, YOLO26-n-seg
ready count: 0/3
Projects tab: 0 projects
D:\tracker.json leakage: none found
fatal traceback: none
```

Final real-model sandbox probe:

```text
All models ready — setup wizard not needed
```

Updated classification:

```text
full GUI compile/package: pass
first-launch setup-wizard liveness: pass
visible blank model/setup state: pass
complete app-state vacuum: pass
real model ready recognition: pass
manual setup-wizard click-through: pending
normal-tab visual sweep: pending
```

Detailed report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-finalvacuum-result.md
```

### Mask Tab Preview Packaging Fix, 2026-05-12

Manual testing found:

```text
FileNotFoundError:
reconstruction_zone.dist\ultralytics\cfg\default.yaml
```

Verified:

```text
source package has ultralytics\cfg\default.yaml
final-vacuum bundle did not include ultralytics\cfg
```

Applied:

```text
hot-patched final-vacuum artifact with ultralytics\cfg
added scripts\copy_ultralytics_runtime_data.ps1
updated final-vacuum build runner to include ultralytics/cfg data
```

Current classification:

```text
setup wizard: manually passed by user
Extract tab: user ran operations successfully
Mask Preview: blocked by missing Ultralytics cfg, patched, pending user retest
```

### Mask Preview Fix Rebuild, 2026-05-13

The prior final-vacuum artifact is now superseded for Mask Preview testing by:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

This artifact includes:

```text
Mask Preview optional import hardening in source
torch._dynamo._trace_wrapped_higher_order_op annotation rewrite
Ultralytics cfg runtime data inclusion
OpenCV CUDA/NVIDIA DLL post-build copy
```

Verified result:

```text
build exit code: 0
Nuitka XML completion: yes
compiled C files: 9494
clcache hits/misses: 9490 / 4
source Mask pipeline import smoke: pass
packaged blank new-install launch smoke: pass
```

Updated classification:

```text
setup wizard: manually passed by user on prior artifact
Extract tab: user ran operations successfully on prior artifact
Mask Preview compile/package fix: built successfully
Mask Preview manual runtime retest on new artifact: pending
```

Detailed report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-maskpreviewfix-result.md
```

---

## Remaining Gaps

1. **Persistent `scripts/audit_local_imports.py`** — now present and passing.
   Keep it as a repeatable pre-build gate rather than falling back to inline
   one-shot audits.

2. **Planning doc headers** — 5 planning docs from the phantom commit describe
   unimplemented features. They should get a status header: "Future plan only.
   Not implemented in v1. Do not reference these APIs from live source."

3. **Visual UI verification** — the setup wizard initial missing-model screen
   has now been visually inspected. Still inspect the normal tabs to confirm:
   no Adjust handoff section, no Send To Coverage button, LUT controls present,
   Extract tab not disabled.

4. **`prep360/core/lut.py` correctness** — the +202 lines of new methods were
   verified by signature matching but not by running actual LUT processing on
   images. A functional test (apply a .cube LUT to a test image) would confirm
   the implementation is correct.
