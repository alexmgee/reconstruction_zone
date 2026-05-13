# Phantom Code Remediation and Full-Scale Audit Plan

Date: 2026-05-11
Project: Reconstruction Zone
Source of truth: `D:\Projects\reconstruction-zone`
Related incident report: `planning/plans/2026-05-11-phantom-code-incident-report.md`
Related build handoff: `planning/plans/2026-05-11-nuitka-build-diagnostics-handoff-report.md`

## Purpose

This plan defines the exact cleanup, verification, and broader audit process for
repairing the damage introduced by commit `2741e55` (`release(v1): trim
experimental features for v1 release`). The immediate objective is to remove all
references to planned-but-unimplemented Adjust/Coverage code, restore the Extract
tab's real prep360 integration, and prevent GUI startup/import analysis from
pulling optional ML packages into Nuitka.

The second objective is to establish a repeatable full-scale audit for other
"phantom code" problems: source references to modules, classes, functions, UI
surfaces, settings, or package exports that do not actually exist in the repo or
were explicitly deleted by the v1 trim.

Per current division of labor:

- Codex: diagnostics, plan review, static/source audit, checklist maintenance.
- Claude: source cleanup implementation and build attempts.
- No full Nuitka build should run until the source-level gates in this plan pass.

## Status Note — 2026-05-11

This remains the governing remediation/audit plan. Since it was written:

- The source-cleanup work described in Phases 0-5 has been reported complete in
  `2026-05-11-remediation-progress-report.md`.
- The preserved Phase 6A retry artifact was later reconciled and exits `0` from
  both the repo root and its `.dist` directory.
- The previously reported `_version` retry-failure attribution was withdrawn as
  misattributed carry-forward from an earlier attempt.

Use this plan for the remediation structure and quality gates, and use the
current progress/retry reports for live execution status.

## Current Verified Problems

### 1. Source tab still references phantom Adjust APIs

Current live references in `reconstruction_gui/tabs/source_tab.py` include:

- `AdjustmentRecipe` in the `from prep360.core import (...)` block.
- `reconstruction_gui.adjust_workflow` and fallback `adjust_workflow` imports.
- `Adjust Handoff` UI section.
- `extract_adjust_recipe_enabled_var`, `extract_adjust_recipe_entry`, and
  `extract_open_adjust_var` GUI state.
- `_validate_adjust_recipe_before_extraction()`.
- `_export_adjusted_derivative()`.
- `_offer_open_in_adjust()`.
- `from tabs.adjust_tab import _load_adjust_images`.
- Multiple call sites wired into extraction workers.

The referenced modules/classes do not exist:

- `prep360/core/adjustment_recipe.py` has no history.
- `prep360/core/color_pipeline.py` has no history.
- `reconstruction_gui/adjust_workflow.py` has no history.
- `reconstruction_gui/tabs/adjust_tab.py` was explicitly deleted by the v1 trim.

### 2. Queue settings still carry phantom Adjust fields

Current `prep360/core/queue_manager.py` still contains:

- `adjust_recipe_enabled`.
- `adjust_recipe_path`.
- `open_adjust_after_extraction`.
- Deserialization support for those fields.
- Summary text entries: `Adjust recipe`, `open Adjust`.

These settings no longer correspond to any real v1 feature.

### 3. Alignment tab still references deleted Coverage/Gaps code

The first local import audit found:

- `reconstruction_gui/tabs/alignment_tab.py:2588` imports
  `reconstruction_gui.tabs.gaps_tab`.
- `reconstruction_gui/tabs/alignment_tab.py:2590` imports `tabs.gaps_tab`.
- The UI still defines a `Send To Coverage` button around line 980.
- `reconstruction_gui/tabs/gaps_tab.py` was explicitly deleted by `2741e55`.

This is the same class of problem as the Adjust handoff: UI and source wiring to
features that were supposed to be removed for v1.

### 4. Import side effects are currently broken

Verified source-only probe:

```text
python -c "import sys; sys.path[:0]=['D:/Projects/reconstruction-zone','D:/Projects/reconstruction-zone/reconstruction_gui']; import tabs.source_tab as st; heavy=('ultralytics','transformers','rfdetr'); print('HAS_PREP360', getattr(st,'HAS_PREP360',None)); print('HEAVY', {name:any(m==name or m.startswith(name+'.') for m in sys.modules) for name in heavy})"
```

Current result:

```text
HAS_PREP360 False
HEAVY {'ultralytics': True, 'transformers': False, 'rfdetr': False}
```

A wider probe also shows `torch` and `cv2` loaded during Source tab import:

```text
{'ultralytics': True, 'transformers': False, 'rfdetr': False, 'torch': True, 'cv2': True, 'pycolmap': False, 'open3d': False}
```

The minimum required fix is: Source tab and GUI shell import must not load
`ultralytics`, `transformers`, or `rfdetr`. Reducing `torch`/`cv2` at shell
import is strongly preferred but may require a broader import-hygiene pass.

## Guiding Rules For The Cleanup

- Work only against `D:\Projects\reconstruction-zone` unless explicitly told otherwise.
- Do not run full Nuitka builds during source cleanup.
- Do not preserve phantom code for hypothetical future features.
- Do not add replacement stubs for deleted features unless a current runtime
  caller genuinely requires backward compatibility.
- Planning documents may remain, but source code must not import or call planned
  APIs that are not implemented.
- A broad `try/except ImportError` around many imports is not proof that the app
  is healthy. If a required current feature import fails, the failure should be
  visible in tests/checks.
- Optional dependencies should be imported at the point of use or behind a small
  feature-specific guard, not through package-wide `__init__.py` exports.
- Every deletion must be followed by `rg` verification that no live `.py` source
  still references the removed symbol/module, except in explicit audit/report
  docs.

## Phase 0: Stabilize Before Editing

Checklist:

- [ ] Confirm no Python/Nuitka build processes are running.
- [ ] Confirm working directory is `D:\Projects\reconstruction-zone`.
- [ ] Capture current branch and dirty state:

```powershell
git -C D:\Projects\reconstruction-zone status --short --branch
```

- [ ] Decide whether cleanup happens on current `release/v1` or a new branch.
      Recommended branch if using git branching:

```powershell
git -C D:\Projects\reconstruction-zone switch -c codex/phantom-code-cleanup
```

- [ ] Do not revert unrelated local work.
- [ ] Keep this plan and the incident report open while editing.

Stop condition:

- If the working tree contains unrelated uncommitted edits in files to be edited,
  inspect them and preserve them. Do not blindly reset or checkout files.

## Phase 1: Remove Phantom Adjust Code From Source Tab

Target file:

```text
reconstruction_gui/tabs/source_tab.py
```

### 1A. Fix imports

Checklist:

- [ ] Remove `AdjustmentRecipe` from the `from prep360.core import (...)` block.
- [ ] Remove the entire `adjust_workflow` import block:

```python
try:
    from reconstruction_gui.adjust_workflow import detect_adjust_input, export_adjusted_dataset
except ImportError:
    try:
        from adjust_workflow import detect_adjust_input, export_adjusted_dataset
    except ImportError:
        detect_adjust_input = None
        export_adjusted_dataset = None
```

- [ ] Do not replace it with a stub.
- [ ] Do not import `tabs.adjust_tab` anywhere in `source_tab.py`.

Verification:

```powershell
rg -n "AdjustmentRecipe|adjust_workflow|tabs\.adjust_tab|from adjust_workflow|detect_adjust_input|export_adjusted_dataset" D:\Projects\reconstruction-zone\reconstruction_gui\tabs\source_tab.py
```

Expected result after cleanup:

- No matches.

### 1B. Remove Adjust Handoff UI

Checklist:

- [ ] Delete the `Adjust Handoff` `CollapsibleSection`.
- [ ] Delete `app.extract_open_adjust_var`.
- [ ] Delete `app.extract_adjust_recipe_enabled_var`.
- [ ] Delete `app.extract_adjust_recipe_entry`.
- [ ] Delete the recipe browse row.
- [ ] Restore the LUT section wording so it no longer points users toward Adjust:

Recommended wording:

```text
Section: Color & LUT
Subtitle: optional quick post-process after extraction
Checkbox: Apply LUT after extraction
```

Verification:

```powershell
rg -n "Adjust Handoff|Open in Adjust|Adjust recipe|adjustment_recipe|extract_open_adjust_var|extract_adjust_recipe" D:\Projects\reconstruction-zone\reconstruction_gui\tabs\source_tab.py
```

Expected result after cleanup:

- No matches.

### 1C. Remove Adjust settings from `_snapshot_settings()`

Checklist:

- [ ] Remove `adjust_recipe_enabled=...`.
- [ ] Remove `adjust_recipe_path=...`.
- [ ] Remove `open_adjust_after_extraction=...`.
- [ ] Keep real extraction settings: LUT, shadow/highlight, sky filter, blur,
      motion, sharpness, scene detection, etc.

Verification:

```powershell
rg -n "adjust_recipe_enabled|adjust_recipe_path|open_adjust_after_extraction" D:\Projects\reconstruction-zone\reconstruction_gui\tabs\source_tab.py
```

Expected result after cleanup:

- No matches.

### 1D. Remove Adjust-only helper functions

Delete these functions entirely:

- `_validate_adjust_recipe_before_extraction()`.
- `_export_adjusted_derivative()`.
- `_offer_open_in_adjust()`.

For `PostProcessingError` and `_run_post_processing_safely()`:

- Preferred v1 cleanup: remove `_run_post_processing_safely()` and return to
  direct `_run_post_processing()` calls, unless there is a verified current UX
  requirement to keep raw extractions even when optional LUT/sky/blur processing
  fails.
- If keeping a safe wrapper, it must be generic and must not mention Adjust.
  Example permitted log concept: `Extraction complete, but optional
  post-processing failed; raw frames remain available at ...`.
- `PostProcessingError` may remain only if it is used by real post-processing
  code and not coupled to Adjust. Otherwise replace it with `ValueError` or a
  generic existing exception and remove the class.

Verification:

```powershell
rg -n "_validate_adjust_recipe|_export_adjusted_derivative|_offer_open_in_adjust|Open in Adjust|Adjust workflow|Adjust handoff|PostProcessingError" D:\Projects\reconstruction-zone\reconstruction_gui\tabs\source_tab.py
```

Expected result after cleanup:

- No Adjust-specific matches.
- `PostProcessingError` is allowed only if deliberately retained for generic
  real post-processing.

### 1E. Remove call sites

Known current call-site groups:

- `_validate_adjust_recipe_before_extraction(settings)` before single extraction.
- `_validate_adjust_recipe_before_extraction(s)` before queued extraction.
- `_validate_adjust_recipe_before_extraction(settings)` before split-pair extraction.
- `_export_adjusted_derivative(...)` after extraction.
- `_offer_open_in_adjust(...)` after extraction.

Checklist:

- [ ] Delete all Adjust validation calls.
- [ ] Delete all adjusted derivative export calls.
- [ ] Delete all open-in-Adjust offer calls.
- [ ] Ensure real `_run_post_processing(...)` still runs for existing real
      features: LUT, shadow/highlight, sky, blur, motion.
- [ ] Ensure extraction completion logging no longer mentions Adjust.

Verification:

```powershell
rg -n "validate_adjust|adjusted_derivative|offer_open_in_adjust|Open in Adjust|Adjusted derivative" D:\Projects\reconstruction-zone\reconstruction_gui\tabs\source_tab.py
```

Expected result after cleanup:

- No matches.

## Phase 2: Remove Phantom Adjust Fields From Queue Settings

Target file:

```text
prep360/core/queue_manager.py
```

Checklist:

- [ ] Remove dataclass fields:
  - `adjust_recipe_enabled`.
  - `adjust_recipe_path`.
  - `open_adjust_after_extraction`.
- [ ] Remove those fields from `ExtractionSettings.from_dict()`.
- [ ] Remove `Adjust recipe` and `open Adjust` summary entries.
- [ ] Keep `from_dict()` tolerant of old queue JSON by simply ignoring unknown
      keys. The current manual constructor already supports that pattern.
- [ ] Do not migrate old queue JSON unless the app has a documented queue schema
      migration system. Unknown fields can be ignored.

Verification:

```powershell
rg -n "adjust_recipe|open_adjust|Adjust recipe|open Adjust" D:\Projects\reconstruction-zone\prep360\core\queue_manager.py
```

Expected result after cleanup:

- No matches.

## Phase 3: Remove Phantom Coverage/Gaps Wiring From Alignment Tab

Target file:

```text
reconstruction_gui/tabs/alignment_tab.py
```

Problem:

The Coverage/Gaps tab was removed, but Alignment still exposes a `Send To
Coverage` button and imports `gaps_tab`.

Checklist:

- [ ] Remove the `Send To Coverage` button from the alignment viewer controls.
- [ ] Remove `app.alignment_send_to_coverage_btn` if no longer used.
- [ ] Remove `_send_alignment_to_coverage()` entirely.
- [ ] Remove all calls to `_send_alignment_to_coverage(...)`.
- [ ] Remove `gaps_` state references from Alignment unless they are used by a
      real current v1 feature.
- [ ] If a future Coverage workflow is planned, document it only in planning
      docs, not live source.

Verification:

```powershell
rg -n "Coverage|gaps_tab|gaps_|send_to_coverage|Send To Coverage" D:\Projects\reconstruction-zone\reconstruction_gui\tabs\alignment_tab.py D:\Projects\reconstruction-zone\reconstruction_gui\reconstruction_zone.py
```

Expected result after cleanup:

- No live source references to Coverage/Gaps UI or deleted `gaps_tab`.
- Matches in planning docs are acceptable only if clearly future-facing.

## Phase 4: Fix `prep360` Import Hygiene

The immediate phantom cleanup removes nonexistent Adjust references, but there is
also a real packaging risk: importing `prep360.core` currently imports
`segmenter.py`, which imports `ultralytics` at module import time.

Target files:

```text
prep360/core/__init__.py
prep360/__init__.py
reconstruction_gui/tabs/source_tab.py
prep360/cli.py
```

### 4A. Prefer direct Source-tab imports over package-wide `prep360.core`

Checklist:

- [ ] Replace broad `from prep360.core import (...)` in `source_tab.py` with
      direct imports from specific modules where practical.
- [ ] Do not import `prep360.core.segmenter` or `Segmenter` from Source tab.
- [ ] Consider lazy-importing `LUTProcessor`, `SkyFilter`, `SharpestExtractor`,
      and other OpenCV-dependent tools inside the functions that use them if
      GUI shell import should avoid `cv2`.

Example direction, not exact final code:

```python
from prep360.core.analyzer import VideoAnalyzer
from prep360.core.extractor import FrameExtractor, ExtractionConfig, ExtractionMode
from prep360.core.queue_manager import VideoQueue, ExtractionSettings
from prep360.core.fisheye_reframer import FisheyeViewConfig, FISHEYE_PRESETS, default_osmo360_calibration, batch_extract as fisheye_batch_extract
```

For modules that are optional/heavy, import inside the function that uses them.

### 4B. Make segmentation exports lazy or remove them from top-level package exports

Options:

1. Preferred compatibility-preserving option:
   - Remove eager `from .segmenter import ...` from `prep360/core/__init__.py`.
   - Add `__getattr__` for `Segmenter`, `SegmentConfig`, `COCO_CLASSES`, and
     `CLASS_PRESETS` so importing those names explicitly still works but plain
     `import prep360.core` does not load Ultralytics.

2. Stricter v1 option:
   - Remove segmentation symbols from `prep360.core.__all__` and `prep360.__all__`.
   - Update any real callers to import from `prep360.core.segmenter` directly.

Do not leave `prep360/__init__.py` eagerly importing segmentation from
`prep360.core`, because that defeats lazy core exports.

Verification:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\Python314\python.exe -c "import sys; sys.path.insert(0,'D:/Projects/reconstruction-zone'); import prep360.core; heavy=('ultralytics','transformers','rfdetr'); print({name:any(m==name or m.startswith(name+'.') for m in sys.modules) for name in heavy})"
```

Expected result:

```text
{'ultralytics': False, 'transformers': False, 'rfdetr': False}
```

### 4C. Keep CLI segmentation behavior explicit

If `prep360/cli.py` has a `segment` command that needs `Segmenter`, import it in
that command handler, not at package import time.

Checklist:

- [ ] `prep360 analyze`, `extract`, `reframe`, etc. should not import Ultralytics.
- [ ] `prep360 segment` may import Ultralytics at the point of use.
- [ ] If Ultralytics is unavailable, only the segment command should fail with a
      clear message.

## Phase 5: Source Verification Gates

Do not run a full build until every gate below passes.

### Gate 1: No known phantom terms in live Python source

Run:

```powershell
rg -n "AdjustmentRecipe|adjust_workflow|adjust_tab|color_pipeline|adjustment_recipe|open_adjust_after_extraction|adjust_recipe_enabled|_validate_adjust_recipe|_export_adjusted_derivative|_offer_open_in_adjust|gaps_tab|Send To Coverage|send_to_coverage" D:\Projects\reconstruction-zone -g "*.py" -g "*.spec"
```

Expected result:

- No matches in live source or spec files.
- Planning/report markdown may still mention these terms.

### Gate 2: Local missing import audit is clean

Use the AST-based audit described in Phase 7. First-pass findings before cleanup
were:

```text
reconstruction_gui\tabs\alignment_tab.py:2588 from reconstruction_gui.tabs.gaps_tab
reconstruction_gui\tabs\alignment_tab.py:2590 from tabs.gaps_tab
reconstruction_gui\tabs\source_tab.py:67 from reconstruction_gui.adjust_workflow
reconstruction_gui\tabs\source_tab.py:1525 from tabs.adjust_tab
```

Expected result after cleanup:

- No unresolved local imports, unless explicitly listed in an allowlist with a
  reason.

### Gate 3: Source tab prep360 integration is restored

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\Python314\python.exe -c "import sys; sys.path[:0]=['D:/Projects/reconstruction-zone','D:/Projects/reconstruction-zone/reconstruction_gui']; from tabs.source_tab import HAS_PREP360; print(f'HAS_PREP360={HAS_PREP360}')"
```

Expected result:

```text
HAS_PREP360=True
```

If this is false, stop. Do not build.

### Gate 4: GUI shell import does not pull optional ML packages

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\Python314\python.exe -c "import sys; sys.path[:0]=['D:/Projects/reconstruction-zone','D:/Projects/reconstruction-zone/reconstruction_gui']; import reconstruction_gui.reconstruction_zone as rz; blocked=('ultralytics','transformers','rfdetr'); loaded=[name for name in blocked if any(m==name or m.startswith(name+'.') for m in sys.modules)]; print(rz.__version__); print('LOADED', loaded); raise SystemExit(1 if loaded else 0)"
```

Expected result:

```text
0.1.0
LOADED []
```

If this loads any blocked module, stop. Do not build.

### Gate 5: Source tab import does not pull optional ML packages

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\Python314\python.exe -c "import sys; sys.path[:0]=['D:/Projects/reconstruction-zone','D:/Projects/reconstruction-zone/reconstruction_gui']; import tabs.source_tab as st; blocked=('ultralytics','transformers','rfdetr'); loaded=[name for name in blocked if any(m==name or m.startswith(name+'.') for m in sys.modules)]; print('HAS_PREP360', st.HAS_PREP360); print('LOADED', loaded); raise SystemExit(1 if loaded or not st.HAS_PREP360 else 0)"
```

Expected result:

```text
HAS_PREP360 True
LOADED []
```

### Gate 6: Compile checks

Run:

```powershell
C:\Python314\python.exe -m py_compile `
  D:\Projects\reconstruction-zone\reconstruction_gui\tabs\source_tab.py `
  D:\Projects\reconstruction-zone\reconstruction_gui\tabs\alignment_tab.py `
  D:\Projects\reconstruction-zone\prep360\core\queue_manager.py `
  D:\Projects\reconstruction-zone\prep360\core\__init__.py `
  D:\Projects\reconstruction-zone\prep360\__init__.py
```

Expected result:

- Exit code 0.

### Gate 7: Focused tests

Run existing relevant tests before packaging:

```powershell
python -m pytest -q tests/test_sharpest_extraction.py tests/test_review_status.py
```

If tests fail because of unrelated local environment setup, document that before
continuing. Do not silently ignore failures.

### Gate 8: GUI smoke launch/import

At minimum, run source import smokes. If a GUI launch is feasible, launch the app
only after the import guards pass and confirm:

- Extract tab is present.
- Extract controls are not disabled due to `HAS_PREP360=False`.
- No Adjust handoff UI remains.
- No Send To Coverage button remains.
- LUT controls still exist if LUT post-processing is a current v1 feature.

## Phase 6: Only After Source Gates Pass, Resume Packaging

Claude should own build attempts. Build sequence should be:

1. Run the constrained GUI Nuitka smoke.
2. If it enters Transformers model-tree analysis, stop immediately.
3. If the GUI smoke passes, run the full build.
4. After full build, copy OpenCV CUDA DLLs using `scripts/copy_cv2_cuda_dlls.ps1`.
5. Run packaged runtime smokes.

No full build should begin while any of these are true:

- `HAS_PREP360=False` on Source tab import.
- GUI shell import loads `ultralytics`, `transformers`, or `rfdetr`.
- Local missing import audit reports deleted local modules.
- Live source still references Adjust/Coverage phantom terms.

## Phase 7: Full-Scale Phantom Issue Audit

This audit is broader than the known Adjust/Coverage fallout. It is meant to
catch any other source references to deleted, planned, or fabricated code.

### Audit A: Deleted-file reference audit

Purpose:

Find source references to files/modules removed by a commit or branch diff.

Commands:

```powershell
git -C D:\Projects\reconstruction-zone show --name-status --format=short 2741e55
```

For each deleted `.py` file, search for:

- The module stem, e.g. `adjust_tab`, `gaps_tab`, `shadow_detection`.
- Important class/function names formerly provided by that file.
- UI labels tied to the deleted feature, e.g. `Coverage`, `Adjust`, `Click Mode`.

Example:

```powershell
rg -n "adjust_tab|gaps_tab|adjust_workflow|shadow_detection|sam_refinement|matting|vos_propagation|Coverage|Click Mode|Adjust" D:\Projects\reconstruction-zone -g "*.py" -g "*.spec"
```

Triage rules:

- Source imports/calls to deleted modules are blockers.
- Spec hidden imports to deleted modules are blockers.
- Planning docs may mention deleted modules only as future/history, not as current
  implementation instructions.
- UI labels for deleted tabs/features are blockers unless the feature is still
  present and tested.

### Audit B: Missing local import audit

Purpose:

Catch imports of local modules that do not exist.

Create a script, recommended path:

```text
scripts/audit_local_imports.py
```

Requirements for the script:

- Parse Python files with `ast`.
- Build a module map from repo files under `reconstruction_gui`, `prep360`,
  `scripts`, and `tests`.
- Understand package imports and relative imports.
- Understand `reconstruction_gui` top-level alias behavior, where
  `reconstruction_gui` is added to `sys.path` and modules like `tabs.source_tab`
  are imported as top-level paths.
- Report unresolved local imports with file and line.
- Support a small explicit allowlist for intentional optional local imports, but
  every allowlist entry must include a reason.
- Exit nonzero when unresolved non-allowlisted imports exist.

Initial ad hoc audit found these live issues before cleanup:

```text
reconstruction_gui\tabs\alignment_tab.py:2588 from reconstruction_gui.tabs.gaps_tab
reconstruction_gui\tabs\alignment_tab.py:2590 from tabs.gaps_tab
reconstruction_gui\tabs\source_tab.py:67 from reconstruction_gui.adjust_workflow
reconstruction_gui\tabs\source_tab.py:1525 from tabs.adjust_tab
```

Checklist:

- [ ] Add the audit script.
- [ ] Run it before cleanup to capture baseline.
- [ ] Run it after cleanup and require no blocker findings.
- [ ] Add it to future pre-build checklist.

### Audit C: Package `__init__.py` eager import audit

Purpose:

Find package init files that import heavy optional dependencies or deleted modules
as a side effect of importing the package.

Commands:

```powershell
rg -n "^from \.[A-Za-z0-9_]+ import|^from prep360\.core import|^from reconstruction_gui" D:\Projects\reconstruction-zone\prep360\__init__.py D:\Projects\reconstruction-zone\prep360\core\__init__.py D:\Projects\reconstruction-zone\reconstruction_gui\__init__.py
```

Checklist:

- [ ] Identify every symbol exported from `prep360/core/__init__.py`.
- [ ] For each export, classify dependency weight:
  - light: stdlib/simple local dataclasses.
  - medium: numpy/PIL/cv2.
  - heavy: torch, ultralytics, transformers, rfdetr, pycolmap, open3d.
- [ ] Move heavy exports behind lazy import or remove them from top-level exports.
- [ ] Verify plain `import prep360.core` does not load `ultralytics`,
      `transformers`, or `rfdetr`.
- [ ] Verify plain `import prep360` does not load those packages either.

### Audit D: Optional import block audit

Purpose:

Find broad `try/except ImportError` blocks that hide real source breakage.

Commands:

```powershell
rg -n "except ImportError|except ModuleNotFoundError|HAS_[A-Z0-9_]+\s*=\s*False|HAS_[A-Z0-9_]+\s*=\s*True" D:\Projects\reconstruction-zone\reconstruction_gui D:\Projects\reconstruction-zone\prep360 -g "*.py"
```

Checklist:

- [ ] For each `HAS_*` flag, identify whether it guards an external optional
      dependency or an internal project module.
- [ ] Internal project module failures should generally be test failures, not
      silent feature disablement.
- [ ] Split broad optional import blocks into smaller blocks so one missing
      optional feature does not disable unrelated working features.
- [ ] Log or surface actionable errors when a current v1 feature is unavailable.
- [ ] Add source import smokes for important flags such as `HAS_PREP360`.

### Audit E: UI references to removed tabs/features

Purpose:

Catch buttons, tab switches, prefs, and handoff helpers for removed surfaces.

Search terms from the v1 trim:

```text
Adjust
Coverage
Gaps
Click Mode
SAM refinement
alpha matting
shadow detection
VOS
edge injection
COLMAP validation
```

Command:

```powershell
rg -n "Adjust|Coverage|Gaps|Click Mode|SAM refinement|alpha matting|shadow detection|VOS|edge injection|COLMAP validation" D:\Projects\reconstruction-zone\reconstruction_gui -g "*.py"
```

Triage rules:

- A current source reference is allowed only if the feature still exists in v1.
- Future feature labels belong in planning docs, not live GUI code.
- Buttons that point to removed tabs must be deleted, not disabled silently.
- Any `app.tabs.set("DeletedTab")` call is a blocker.

### Audit F: Spec and packaging hook audit

Purpose:

Ensure packaging config does not keep deleted modules alive.

Commands:

```powershell
rg -n "adjust|gaps|shadow_detection|sam_refinement|matting|vos_propagation|click|coverage|model_downloader" D:\Projects\reconstruction-zone\reconstruction_zone.spec D:\Projects\reconstruction-zone\scripts -g "*.py" -g "*.ps1" -g "*.spec"
```

Checklist:

- [ ] No hidden imports for deleted feature modules.
- [ ] Runtime hooks do not import deleted feature modules.
- [ ] Smoke scripts do not depend on deleted feature modules.
- [ ] Deprecated compatibility stubs are documented and do not import heavy
      optional packages.

### Audit G: Planning-doc leakage audit

Purpose:

Prevent future planned APIs from leaking into source before implementation.

Checklist:

- [ ] List all planning docs added by the suspect commit.
- [ ] Extract proposed module names, class names, and function names from those
      docs.
- [ ] Search live source for those names.
- [ ] Any source reference must point to a real implemented module and have a
      passing import test.
- [ ] If a plan remains future-only, add a header to that plan:

```text
Status: Future plan only. Not implemented in v1. Do not reference these APIs from live source until implementation exists and tests pass.
```

Known planning-doc names to audit:

- `AdjustmentRecipe`.
- `adjustment_recipe.py`.
- `color_pipeline.py`.
- `adjust_workflow.py`.
- `adjust_tab.py`.
- click/video propagation helpers.
- Coverage/Gaps helpers.

### Audit H: Import side-effect matrix

Purpose:

Make import behavior explicit before packaging.

Run a matrix like this for each target module:

- `prep360`.
- `prep360.core`.
- `reconstruction_gui.tabs.source_tab`.
- `reconstruction_gui.tabs.alignment_tab`.
- `reconstruction_gui.reconstruction_zone`.
- `reconstruction_gui.reconstruction_pipeline`.

Blocked packages for shell-level imports:

```text
ultralytics
transformers
rfdetr
```

Strongly discouraged for GUI shell import, but may be deferred if too invasive:

```text
torch
cv2
pycolmap
open3d
```

For each module, record:

- Does import succeed?
- Which heavy packages are loaded?
- Which `HAS_*` flags are false?
- Is the heavy import expected for that module?

Expected behavior:

- `reconstruction_pipeline` may load model/OpenCV packages because it is the
  masking runtime module.
- GUI shell and Source tab should not load YOLO/RF-DETR/Transformers.
- `prep360.core` should not load Ultralytics merely to expose non-segmentation
  tools.

### Audit I: Commit-intent sanity check

Purpose:

Catch contradictions like a trim commit that net-adds a large amount of code.

Checklist for future feature-removal commits:

- [ ] Run `git show --stat <commit>`.
- [ ] If the commit says "trim/remove/simplify" but net-adds lines, inspect
      every added hunk.
- [ ] Run deleted-file reference audit before committing.
- [ ] Run local missing import audit before committing.
- [ ] Run package import guards before committing.
- [ ] Require a short commit note explaining any additions in a removal commit.

### Audit J: Documentation and handover cleanup

Purpose:

Avoid docs telling future agents to use deleted files.

Commands:

```powershell
rg -n "adjust_tab\.py|gaps_tab\.py|adjust_workflow\.py|adjustment_recipe\.py|color_pipeline\.py" D:\Projects\reconstruction-zone -g "*.md"
```

Checklist:

- [ ] Current implementation docs must not describe deleted modules as present.
- [ ] Future planning docs must be clearly labeled as future-only.
- [ ] Handover docs must not instruct agents to open deleted files as current
      source.
- [ ] Incident and audit reports may mention deleted files as historical facts.

## Phase 8: Definition of Done

The cleanup is done only when all of the following are true:

- [ ] No live Python source references `AdjustmentRecipe`, `adjust_workflow`,
      `adjust_tab`, `adjustment_recipe`, or `color_pipeline`.
- [ ] No live Python source references `gaps_tab` or exposes a `Send To Coverage`
      action in v1.
- [ ] `prep360/core/queue_manager.py` no longer contains Adjust-only fields.
- [ ] `source_tab.py` imports successfully with `HAS_PREP360=True`.
- [ ] GUI shell import does not load `ultralytics`, `transformers`, or `rfdetr`.
- [ ] Source tab import does not load `ultralytics`, `transformers`, or `rfdetr`.
- [ ] `prep360.core` import does not load `ultralytics`, `transformers`, or
      `rfdetr`.
- [ ] Local missing import audit has no blocker findings.
- [ ] Relevant files pass `py_compile`.
- [ ] Focused tests pass or failures are documented as unrelated and accepted by
      the user.
- [ ] Claude has not started a full build until all source gates pass.
- [ ] Build handoff docs are updated with the source cleanup result.

## Suggested Implementation Order For Claude

1. Remove Adjust imports/UI/functions/calls from `source_tab.py`.
2. Remove Adjust fields from `queue_manager.py`.
3. Remove Coverage/Gaps button/function/calls from `alignment_tab.py`.
4. Run Gate 1 and fix any remaining phantom references.
5. Fix `prep360` import hygiene for segmenter/Ultralytics.
6. Run Gates 2 through 6.
7. Run focused tests.
8. Update the incident report with final cleanup commit/verification results.
9. Only then proceed to constrained Nuitka GUI smoke.

## Notes On Risk

- Removing phantom Adjust UI may change visible Extract tab layout, but that is
  correct for v1 because Adjust was removed.
- Removing phantom queue fields is safe if `from_dict()` ignores unknown old
  queue keys; existing old queue JSON may keep unused keys on disk, but runtime
  should not depend on them.
- Making `Segmenter` lazy may affect callers that expect `from prep360.core
  import Segmenter` to work. If compatibility matters, implement lazy
  `__getattr__` rather than removing the export outright.
- Reducing `cv2`/`torch` imports from Source tab startup may require a second
  pass. Do not let that block the minimum required fix unless packaging still
  drags OpenCV/torch too early.

## Final Rule

No more full-build attempts as a discovery mechanism. The source tree must first
prove that current v1 imports are real, deleted features are not referenced, and
GUI shell import does not pull optional ML stacks.

## Packaging Update, 2026-05-12

The source cleanup gates were sufficient to proceed into a controlled full GUI
Nuitka build. Codex ran that build under the approved Phase 6B constraints:

```text
NUITKA_CACHE_DIR=D:\Projects\reconstruction-zone\dist_test\nuitka-cache
no --include-package=transformers
no --nofollow-import-to=transformers.models.*
torch YAML fix retained
```

Build result:

```text
PASS at compile/package level
Nuitka XML report completion: yes
artifact:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Runtime result:

```text
PARTIAL
```

The packaged GUI stays alive when model lookup is satisfied, but the compiled
first-launch/missing-model setup-wizard path logs:

```text
TypeError: 'ReconstructionZone' object is not callable
```

Therefore, the "no blind full builds" rule remains active. The next work should
be a bounded setup-wizard/CustomTkinter packaging smoke or a direct fix to model
provisioning, not another full-build discovery attempt.

### Setup Wizard Fix Update, 2026-05-12

The bounded setup-wizard proof has now been run.

Root cause:

```text
SetupWizard.self._root shadowed Tkinter's internal _root() method.
```

Fix:

```text
SetupWizard.self._root -> SetupWizard.self._app_root
```

Validation:

- source setup-wizard direct smoke: pass;
- source setup-wizard API-path smoke: pass;
- packaged setup-wizard direct smoke: pass;
- packaged setup-wizard API-path smoke: pass.

This satisfies the bounded proof requirement for the known first-launch wizard
crash. A full GUI rebuild is now appropriate, followed by an isolated
new-install harness run.

### Full GUI Rebuild And New-Install Runtime Update, 2026-05-12

Codex rebuilt the full GUI after the setup-wizard fix.

Result:

```text
compile/package: pass
OpenCV CUDA/NVIDIA DLL post-copy: pass
empty app-home setup-wizard liveness: pass
RF-DETR/YOLO fixture model discovery: pass
RF-DETR real download: pass
YOLO26 real download: pass
SAM3 no-token/token/download flow: pass
packaged GUI all-models-ready recognition: pass
```

Visible vacuum probe:

```text
setup wizard visible: pass
missing models: SAM3, RF-DETR-Seg, YOLO26-n-seg
ready count: 0/3
fatal traceback: none
```

New finding:

```text
The current packaged artifact still reads D:\tracker.json for the Projects tab
behind the setup wizard.
```

Classification:

```text
model/setup-wizard blank-state behavior: pass
complete app-state blank-new-install behavior in current artifact: fail
```

Source fix applied after the artifact:

- add `project_store_file()` and `activity_store_file()` to
  `reconstruction_gui\app_paths.py`;
- default `tabs\projects_tab.py` to app-home project storage unless an explicit
  legacy/custom `tracker_store_path` preference exists;
- default `ProjectStore()` to app-home project storage.

Verified:

```text
ProjectStore().store_path resolves to isolated app-home\projects\tracker.json
py_compile: pass
```

Next packaging step:

```text
Run one final full GUI rebuild, then rerun the visible vacuum setup-wizard probe
against the new artifact before calling the new-install gate complete.
```

### Final-Vacuum Packaging Result, 2026-05-12

Codex ran the final rebuild after the app-state/project-store vacuum fix.

Final artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Result:

```text
build exit code: 0
Nuitka report completion: yes
OpenCV CUDA/NVIDIA DLL copy: pass
visible blank setup wizard: pass
project-store vacuum: pass
all-real-models-ready recognition: pass
```

The final visible vacuum probe found no `D:\tracker.json` leakage.

Detailed report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-finalvacuum-result.md
```
