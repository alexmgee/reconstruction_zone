# Phantom Code Incident Report

Date: 2026-05-11
Author: Claude (self-audit)
Commit under review: `2741e55` ("release(v1): trim experimental features for v1 release")
Branch: `release/v1`

---

## Executive Summary

Commit `2741e55`, dated 2026-05-08, was supposed to trim experimental features
from the `release/v1` branch to prepare a minimal, shippable build. Instead, it
introduced approximately 200 lines of fabricated code referencing three Python
modules that have never existed in the repository. This phantom code silently
broke the Extract tab's prep360 integration, caused Nuitka builds to stall for
hours, and consumed multiple sessions of debugging time from both Claude and
Codex before the root cause was identified.

---

## What the Commit Was Supposed to Do

The commit message states:

> release(v1): trim experimental features for v1 release
>
> GUI tabs removed:
> - Adjust tab (color correction, LUT, RAW dev, chart matching)
> - Coverage tab (spatial gap detection, bridge extraction)
>
> Mask tab sections removed:
> - Detection & Refinement (shadow detection, SAM refinement,
>   alpha matting, ensemble detection, edge injection,
>   COLMAP geometric validation)
> - Interactive Click Mode (SAM3 video click session)

The intent was to *remove* code. The commit was expected to produce a smaller,
simpler codebase.

## What the Commit Actually Did

The commit touched 36 files, added 5,679 lines, and removed 4,760 lines — a
**net addition of 919 lines** in a commit whose purpose was reduction.

It did perform some legitimate trimming:
- Deleted `matting.py`, `shadow_detection.py`, `sam_refinement.py`,
  `vos_propagation.py`, `gaps_tab.py` (4,522 lines removed)
- Removed experimental MaskConfig fields and pipeline methods
- Removed Coverage tab and Click Mode UI sections

But it simultaneously introduced:

### 1. Three Phantom Modules

The commit added import statements for three Python modules that have **never
existed** in the repository — not on `main`, not on any branch, not in any
prior commit:

| Module | Referenced From | Purpose (Fabricated) |
|--------|----------------|---------------------|
| `prep360.core.adjustment_recipe` | `prep360/core/__init__.py` | `AdjustmentRecipe`, `LUTSettings`, `ToneSettings`, `WhiteBalanceSettings`, `ColorSettings`, `DetailSettings`, `CorrectionSettings`, `OutputSettings` |
| `prep360.core.color_pipeline` | `prep360/core/__init__.py` | `ImageLoadResult`, `apply_adjustment_recipe`, `load_image_float`, `write_image_float` |
| `reconstruction_gui.adjust_workflow` | `reconstruction_gui/tabs/source_tab.py` | `detect_adjust_input`, `export_adjusted_dataset` |

Verification:

```
$ ls prep360/core/adjustment_recipe.py 2>/dev/null   → (does not exist)
$ ls prep360/core/color_pipeline.py 2>/dev/null      → (does not exist)
$ ls reconstruction_gui/adjust_workflow.py 2>/dev/null → (does not exist)
$ git log --all --oneline -- "prep360/core/adjustment_recipe.py"   → (no history)
$ git log --all --oneline -- "prep360/core/color_pipeline.py"      → (no history)
$ git log --all --oneline -- "reconstruction_gui/adjust_workflow.py" → (no history)
```

These files were never created. The commit introduced imports for code that was
fabricated during the trim operation.

### 2. ~200 Lines of Dead Feature Code in source_tab.py

The commit added 212 lines to `source_tab.py` (and removed only 9). The
additions include:

- **Import of `AdjustmentRecipe`** from `prep360.core` (line 43) — a class that
  doesn't exist
- **Import of `adjust_workflow`** module (lines 63-71) — a module that doesn't
  exist
- **"Adjust Handoff" UI section** (lines 977-1023) — a complete collapsible
  section with checkboxes, entry fields, and browse button for a recipe
  workflow that has no backend
- **`_validate_adjust_recipe_before_extraction()`** function — calls
  `AdjustmentRecipe.load()` on a nonexistent class
- **`_run_post_processing_safely()`** wrapper function
- **`_export_adjusted_derivative()`** function — calls both phantom modules
- **`_offer_open_in_adjust()`** function — references `tabs.adjust_tab` which
  was supposedly removed by this same commit
- **`PostProcessingError`** exception class
- **Wiring into 5 extraction code paths** — `_validate_adjust_recipe_before_extraction()`
  is called at lines 1761, 2140, and 3395; `_run_post_processing_safely()` replaces
  direct `_run_post_processing()` calls; `_export_adjusted_derivative()` and
  `_offer_open_in_adjust()` are called after each extraction type

### 3. Phantom Exports in prep360 Package Init Files

- `prep360/core/__init__.py`: Added 16 lines importing from `adjustment_recipe`
  and `color_pipeline` (neither exists), plus 12 `__all__` entries
- `prep360/__init__.py`: Added 9 re-export lines from `prep360.core` for the
  same phantom symbols, plus 9 `__all__` entries

### 4. Phantom Fields in ExtractionSettings

`prep360/core/queue_manager.py`: Added 3 dataclass fields
(`adjust_recipe_enabled`, `adjust_recipe_path`,
`open_adjust_after_extraction`), 3 deserialization entries, and 4 summary
lines — all for a feature that doesn't exist.

### 5. Planning Documents for Unimplemented Features

The commit added 1,929 lines of planning documents:
- `planning/plans/adjust-tab-plan.md` (709 lines) — detailed plan for an
  Adjust tab
- `planning/plans/adjust-tab-response-plan.md` (289 lines) — response/review
  of the adjust plan
- `planning/plans/click-video-propagation-plan.md` (494 lines)
- `planning/plans/click-video-propagation-plan-review.md` (146 lines)
- `planning/plans/click-video-propagation-implementation-summary.md` (291
  lines)

These planning documents describe features that were explicitly listed in the
commit message as being *removed*.

---

## Cascade of Consequences

### Consequence 1: HAS_PREP360 Set to False

`source_tab.py` lines 38-62 import from `prep360.core` inside a `try/except
ImportError` block. Because the `__init__.py` now tries to import from the
nonexistent `adjustment_recipe` module, the entire block fails:

```python
try:
    from prep360.core import (
        VideoAnalyzer,
        FrameExtractor, ExtractionConfig, ExtractionMode,
        LUTProcessor,
        AdjustmentRecipe,    # ← doesn't exist, kills the whole block
        SkyFilter, SkyFilterConfig,
        ...
    )
    HAS_PREP360 = True
except ImportError:
    HAS_PREP360 = False      # ← always taken
```

Verified:
```
$ python -c "from prep360.core import AdjustmentRecipe"
→ ImportError: cannot import name 'AdjustmentRecipe' from 'prep360.core'
```

With `HAS_PREP360 = False`, the Extract tab loses access to `VideoAnalyzer`,
`FrameExtractor`, `SkyFilter`, `LUTProcessor`, and every other working prep360
component — none of which are actually broken.

### Consequence 2: Eager Loading of ultralytics During Import

The `source_tab.py` import block and the `prep360/core/__init__.py` eager
exports create a two-part problem:

1. `prep360/core/__init__.py` eagerly imports `.segmenter` (line 13), which
   does `from ultralytics import YOLO`. This has always been there on `main`,
   but it means **any** import from `prep360.core` pays the cost of loading
   the entire ultralytics package tree.

2. `source_tab.py` uses a broad `from prep360.core import (...)` block wrapped
   in `try/except ImportError`. This block triggers the full `__init__.py`
   execution — including the eager segmenter import — regardless of whether
   the import ultimately succeeds or fails.

The phantom `AdjustmentRecipe` import makes the block fail (Consequence 1),
but the damage is already done: Python has already executed `__init__.py` far
enough to import `.segmenter` and load ultralytics. The `try/except` cannot
prevent this — `HAS_PREP360` is *determined by* attempting the import, so it
cannot protect against side effects that occur during that attempt.

The actual bug is the combination of:
- Eager exports in `prep360/core/__init__.py` (segmenter loads ultralytics
  at import time)
- A broad optional import block in `source_tab.py` that triggers the full
  init regardless

Verified — even a failing import loads 65 ultralytics modules:

```
$ python -c "
import sys; sys.path.insert(0, '.')
try:
    from prep360.core import AdjustmentRecipe
except ImportError:
    pass
loaded = [k for k in sys.modules if k.startswith('ultralytics')]
print(f'{len(loaded)} ultralytics modules loaded')
"
→ 65 ultralytics modules loaded
```

Note: the phantom `adjustment_recipe` and `color_pipeline` imports in
`__init__.py` were partially cleaned up by commit `c945879`, but the eager
segmenter import was NOT fixed because it wasn't recognized as part of the
same problem.

### Consequence 3: Nuitka Build Stall

Nuitka sees `ultralytics` in the import graph and follows its dependency tree
into `transformers`. In plain Python, importing `ultralytics` does not
immediately load `transformers` — but Nuitka's Transformers plugin follows the
ultralytics package metadata and dependency graph into `transformers`, then
attempts to enumerate every model in `transformers.models.*`. The "small GUI
smoke" build stalled indefinitely while walking hundreds of Transformer model
packages. This was the primary blocker that Codex spent an entire diagnostic
session investigating, culminating in the handoff report at
`planning/plans/2026-05-11-nuitka-build-diagnostics-handoff-report.md`.

Codex correctly traced the import chain:
```
reconstruction_zone.py
  → source_tab.py
    → prep360.core
      → segmenter.py
        → ultralytics
          → (Nuitka follows dependency graph into transformers)
            → Nuitka Transformers plugin walks transformers.models.*
              → STALL
```

The eager segmenter import in `prep360/core/__init__.py` has always been there
on `main`, but it was never a problem before because it only fires when
something imports from `prep360.core` — which `source_tab.py` does
unconditionally at module level.

### Consequence 4: Wasted Diagnostic Time

The following work was performed to diagnose and work around problems caused
directly or indirectly by this phantom code:

- Codex's full diagnostic session producing the 660-line handoff report
- 5 failed Nuitka build attempts (each taking 30+ minutes)
- The OpenCV smoke test investigation (which was valid but secondary to the
  real problem)
- The `model_downloader.py` neutralization (partially related — Codex was
  trying to eliminate all eager import paths)
- Multiple rounds of Claude's own debugging in the prior session
- The remediation instructions document and its audit report

Conservative estimate: 8+ hours of combined Claude/Codex/user time spent on
consequences of this single commit.

---

## Why This Happened

This is fabrication. The commit was supposed to remove the Adjust tab, but
instead of simply deleting the code, Claude generated new code that *used* the
Adjust tab's planned API — an API that was designed in the planning documents
but never implemented. The planning documents (`adjust-tab-plan.md`,
`adjust-tab-response-plan.md`) describe classes like `AdjustmentRecipe` with
methods like `.load()` and `.validate()`, and the phantom code in
`source_tab.py` calls exactly those methods.

The most likely explanation: during the v1 trim, Claude confused "features
planned for the future" with "features that exist and should be wired up." The
planning documents were included in the same commit, suggesting Claude was
working from the plans and generating integration code for them, rather than
working from the actual codebase.

This violates the project's verification protocol in multiple ways:
- Code was written referencing modules without checking that they exist
- The commit message described removal, but the diff shows addition
- No import test was run after the changes
- The net line count (919 added) contradicts the stated purpose (trimming)

---

## Full Inventory of Phantom Code to Remove

### Files to clean (on current release/v1 working tree):

**`prep360/core/__init__.py`** — current state:
- Line 13: `from .segmenter import Segmenter, SegmentConfig, COCO_CLASSES, CLASS_PRESETS`
  (this is legitimate but should be lazy, not eager — it's the ultralytics trigger)
- The `adjustment_recipe` and `color_pipeline` imports were already removed by
  commit `c945879`

**`prep360/__init__.py`** — check for remaining phantom re-exports

**`reconstruction_gui/tabs/source_tab.py`**:
- Line 43: `AdjustmentRecipe` in the `prep360.core` import block
- Lines 63-71: `adjust_workflow` import block
- Lines 977-1023: "Adjust Handoff" UI section
- Lines 194-206: `adjust_recipe_enabled`, `adjust_recipe_path`,
  `open_adjust_after_extraction` in `_snapshot_settings()`
- Lines 1436-1535: `PostProcessingError`, `_validate_adjust_recipe_before_extraction()`,
  `_run_post_processing_safely()`, `_export_adjusted_derivative()`,
  `_offer_open_in_adjust()`
- All call sites for these functions (lines 1761, 2140, 3395, 1860, 2232,
  3476, 1886, 2256, 3494+)
- LUT subtitle change ("legacy quick post-process; prefer Adjust for LUTs")
- LUT checkbox text change ("Apply LUT after extraction (advanced)")

**`prep360/core/queue_manager.py`**:
- `adjust_recipe_enabled`, `adjust_recipe_path`, `open_adjust_after_extraction`
  fields and their deserialization/summary code

**Planning documents** (these describe unimplemented features — keep or delete
at user's discretion):
- `planning/plans/adjust-tab-plan.md`
- `planning/plans/adjust-tab-response-plan.md`
- `planning/plans/click-video-propagation-plan.md`
- `planning/plans/click-video-propagation-plan-review.md`
- `planning/plans/click-video-propagation-implementation-summary.md`

---

## Verification After Cleanup

After removing all phantom code, the following must pass:

1. **`prep360.core` imports cleanly:**
   ```
   python -c "from prep360.core import VideoAnalyzer, FrameExtractor; print('OK')"
   ```

2. **`HAS_PREP360` is True in source_tab:**
   ```
   python -c "
   import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'reconstruction_gui')
   from tabs.source_tab import HAS_PREP360; print(f'HAS_PREP360 = {HAS_PREP360}')
   "
   ```

3. **No heavy modules loaded during GUI shell import** (Codex's guard test):
   ```python
   import reconstruction_gui.reconstruction_zone
   blocked = ("ultralytics", "transformers", "rfdetr")
   loaded = [n for n in blocked if any(m == n or m.startswith(n + ".") for m in sys.modules)]
   assert not loaded, f"Heavy modules imported during GUI shell: {loaded}"
   ```

4. **GUI launches and Extract tab functions normally**

---

## Lessons

1. **A "trim" commit that net-adds lines is a red flag.** The diff stats alone
   (5,679 added / 4,760 removed) should have prompted review before committing.

2. **Planning documents are not source code.** The existence of a plan for
   `AdjustmentRecipe` does not mean the class exists. Code that references
   planned-but-unimplemented APIs is fabrication.

3. **Import tests catch this immediately.** A single `python -c "from
   prep360.core import AdjustmentRecipe"` would have caught this before commit.

4. **Silent `try/except ImportError` blocks mask cascading failures.** The
   `HAS_PREP360 = False` fallback silently disabled working functionality,
   making the root cause invisible at runtime.

5. **Downstream diagnostics inherit upstream fabrication.** Codex's handoff
   report correctly identified the import chain problem but never questioned
   *why* `prep360.core.__init__` was structured that way — because it appeared
   to be a pre-existing design choice, not a bug introduced three days earlier.
