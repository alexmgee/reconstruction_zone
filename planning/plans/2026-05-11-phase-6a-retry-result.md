# Phase 6A Retry Result

Date: 2026-05-11
Build executor: Claude
Attempt: Second Phase 6A build (first with `--include-package=reconstruction_gui`)

## Current Reconciled Status

This retry report has been artifact-reconciled. The preserved executable at:

```text
D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe
```

was rerun from both the repository root and its `.dist` directory and exited
`0` with no traceback in both cases.

Current classification:

| Aspect | Status |
|--------|--------|
| Nuitka compile | **PASS** |
| Preserved executable import smoke | **PASS** |
| Earlier `_version` retry-failure attribution | **WITHDRAWN / MISATTRIBUTED** |
| `--include-package=reconstruction_gui` | **Supported by preserved retry result** |
| Broad import refactor | **Deferred; not justified by this artifact** |

The original runtime-failure narrative remains in the historical sections below
only to preserve the incident trail. It must not be treated as the current Phase
6A retry result.

## Build Command (exact)

```
C:\Python314\python.exe -m nuitka
  --standalone
  --assume-yes-for-downloads
  --windows-console-mode=attach
  --enable-plugin=tk-inter
  --enable-plugin=no-qt
  --nofollow-import-to=bitsandbytes
  --nofollow-import-to=bitsandbytes.*
  --nofollow-import-to=triton
  --nofollow-import-to=triton.*
  --nofollow-import-to=ultralytics
  --nofollow-import-to=ultralytics.*
  --nofollow-import-to=transformers
  --nofollow-import-to=transformers.*
  --nofollow-import-to=rfdetr
  --nofollow-import-to=rfdetr.*
  --nofollow-import-to=sam3
  --nofollow-import-to=sam3.*
  --nofollow-import-to=transformers.integrations.bitsandbytes
  --user-package-configuration-file=scripts/nuitka-torch-fix.yml
  --include-package=customtkinter
  --include-package=reconstruction_gui
  --include-data-dir=reconstruction_gui/docs=reconstruction_gui/docs
  --include-data-dir=resources=resources
  --include-data-files=reconstruction-zone.ico=reconstruction-zone.ico
  --output-dir=dist_test
  scripts/smoke_import_gui.py
```

Difference from first Phase 6A attempt: added `--include-package=reconstruction_gui`.
Did NOT add `--include-package=prep360` (per Codex oversight recommendation).

## Build Environment

- Nuitka: 4.0.8
- Python: 3.14.0 (experimentally supported)
- C compiler: cl 14.5
- OS: Windows 11 Pro
- Working directory: `D:\Projects\reconstruction-zone`
- Branch: `release/v1`

## Preflight Checks (passed before build)

1. No Python/Nuitka/Git build processes running (PowerShell `Get-Process`)
2. No `.git/index.lock`
3. No SCons cache lock warning (`nuitka --version` clean)
4. Previous build artifacts archived to `dist_test/archive/2026-05-11-pre-6a-retry/`
5. `dist_test/` confirmed gitignored

## Build Result: SUCCESS (Nuitka exit 0)

### Warnings (3)

1. `Python version '3.14' is only experimentally supported by Nuitka '4.0.8'`
2. `Detected 2 module(s) with mismatching checksum in 'scripts/nuitka-torch-fix.yml': torch.library, torch._refs`
   — This means the YAML config was engaged and found the target modules. It
   does not prove the singledispatch replacements are semantically correct at
   runtime (per Codex oversight: the exe failed before any torch code path was
   reached).
3. `Module 'torch' has parameter: Torch JIT is disabled by default in standalone mode`

### Compilation stats

- C files compiled: **4,960** (vs 4,764 in the first attempt without
  `--include-package=reconstruction_gui` — the 196 additional files are the
  `reconstruction_gui/` package modules)
- Cache hits: 4,762 (reused from first attempt)
- Cache misses: 198 (the new `reconstruction_gui` modules)

### DLLs included

- cv2: 1 DLL
- sklearn: 2 DLLs
- torch: 37 DLLs
- torchvision: 3 DLLs
- Windows Runtime DLLs: included (default)

### Data files included

- customtkinter assets: 8 files
- matplotlib: 183 files
- pandas templates: 7 files
- pytz zoneinfo: 604 files
- scipy: 1 file
- sklearn: 2 files
- Tk: 89 files
- Tcl: 837 files
- torch include: 9,317 files
- tzdata: 604 files
- reconstruction_gui/docs: 12 files
- resources: 2 files
- reconstruction-zone.ico: 1 file

### Produced executable

```
D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe
```

### No Transformers stall observed

Process inspection at 2 minutes into the build (PowerShell `Get-CimInstance`):
- No child process matching `define_import_structure` or `transformers`
- Main Nuitka process PID 48680: 692s CPU, 4.2GB working set
- Build was actively compiling

This confirms `--nofollow-import-to=transformers.*` holds even when
`--include-package=reconstruction_gui` is present (which includes
`reconstruction_pipeline.py`, a file that contains lazy imports of
`ultralytics`, `rfdetr`, and `transformers` inside functions).

Note: this observation is from a single process check, not continuous
monitoring. It is consistent with the build completing without stall, but is
not a continuous proof.

### Wall-clock build time

Not precisely recorded. The build was launched as a background task and
completed before the next user interaction. Based on the cache hit ratio
(4,762/4,960 = 96%), the incremental compile was substantially faster than the
first attempt. Exact duration unknown.

## Original Runtime Interpretation: FAILURE — Superseded

This section records the retry result as it was first reported. Later
artifact-specific reverification contradicted it. The preserved executable does
not currently reproduce this failure; see the reconciliation sections above and
below.

```
$ dist_test/smoke_import_gui.dist/smoke_import_gui.exe

Traceback (most recent call last):
  File "D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.py",
    line 21, in <module>
  File "D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\
    reconstruction_gui\reconstruction_zone.py", line 74,
    in <module reconstruction_gui.reconstruction_zone>
ModuleNotFoundError: No module named '_version'
```

Exit code: 1.

**Superseded interpretation:** this was reported as the same failure as the
first Phase 6A attempt, but later reruns of the preserved retry artifact exited
`0` from both launch locations. This paragraph is historical, not current.

## Historical Hypothesis: Why `--include-package=reconstruction_gui` Did Not Help

The analysis in this section was written before the preserved retry executable
was rerun and should be read as a disproven working hypothesis, not as the
current diagnosis.

`--include-package=reconstruction_gui` tells Nuitka to compile and bundle all
`.py` files under `reconstruction_gui/`. This was verified — the build compiled
196 additional C files (4,960 total vs 4,764 without the flag).

However, the runtime import statement that fails is:

```python
from _version import __version__
```

This is a bare top-level import. Python's import machinery looks for a module
named `_version` in `sys.path`. It does not look for
`reconstruction_gui._version`. That would require:

```python
from reconstruction_gui._version import __version__
```

In the source development environment, this works because
`reconstruction_zone.py` lines 32-37 add `reconstruction_gui/` to `sys.path`:

```python
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
```

In the Nuitka standalone bundle, compiled modules are resolved through Nuitka's
own import system. Even though `reconstruction_gui/_version.py` was compiled
into the bundle as `reconstruction_gui._version`, the bare import
`from _version import` searches for a top-level module named `_version`, not a
submodule of `reconstruction_gui`. No top-level `_version` exists in the
bundle.

### Verification

```
$ find dist_test/smoke_import_gui.dist -name "*_version*"
```

Found only torch C headers (`ATen/ops/_version.h` etc.). No `_version.py` or
compiled `_version` module at the top level.

## Historical Root-Cause Hypothesis — Not Supported By Preserved Retry Artifact

`reconstruction_zone.py` uses bare-name imports that depend on runtime
`sys.path` manipulation. This pattern is incompatible with Nuitka's standalone
module resolution, which resolves compiled modules by their package-qualified
names, not filesystem paths.

Affected bare imports in `reconstruction_zone.py`:

| Line | Import | Bare name | Fully qualified equivalent |
|------|--------|-----------|---------------------------|
| 74 | `from _version import __version__` | `_version` | `reconstruction_gui._version` |
| 75 | `from widgets import (...)` | `widgets` | `reconstruction_gui.widgets` |
| 85 | `from app_infra import AppInfrastructure` | `app_infra` | `reconstruction_gui.app_infra` |
| 86 | `from tabs.alignment_tab import ...` | `tabs.alignment_tab` | `reconstruction_gui.tabs.alignment_tab` |
| 87 | `from tabs.source_tab import ...` | `tabs.source_tab` | `reconstruction_gui.tabs.source_tab` |
| 88 | `from tabs.projects_tab import ...` | `tabs.projects_tab` | `reconstruction_gui.tabs.projects_tab` |

The same pattern exists in other files within `reconstruction_gui/` that import
siblings using bare names. For example, `source_tab.py` does
`from widgets import (...)`, `tabs/alignment_tab.py` does
`from colmap_runner import ...`, etc. The full scope of bare sibling imports
across the package has not been enumerated.

## Previously Considered Fix Options — Deferred After Reconciliation

### Option A: Source fix — convert bare imports to fully-qualified

Change `reconstruction_zone.py` lines 74-88 to use fully-qualified imports:

```python
from reconstruction_gui._version import __version__
from reconstruction_gui.widgets import (...)
from reconstruction_gui.app_infra import AppInfrastructure
from reconstruction_gui.tabs.alignment_tab import build_alignment_tab
from reconstruction_gui.tabs.source_tab import build_source_tab
from reconstruction_gui.tabs.projects_tab import build_projects_tab
```

The same change would need to be applied to every file in `reconstruction_gui/`
that uses bare sibling imports, for every module in the smoke's import graph.

**Pros:** Fixes the root cause. Works in both source and packaged environments.
Nuitka can resolve these statically without special flags.

**Cons:** Broader source change. Every file in `reconstruction_gui/` that uses
bare sibling imports would need updating. However, the `sys.path` manipulation
at lines 32-37 already adds `_project_root` (line 37), so
`reconstruction_gui.*` is importable as a package in both direct-launch and
package-import modes.

### Option B: Nuitka fix — `--include-module` for each bare import

```
--include-module=reconstruction_gui._version
--include-module=reconstruction_gui.widgets
--include-module=reconstruction_gui.app_infra
...
```

**Does not fix the problem.** Even if the modules are force-included, the bare
import `from _version import` still resolves against top-level module names,
not submodules. This is the same reason `--include-package` didn't work.

### Option C: Nuitka runtime hook to add package dir to sys.path

Use a Nuitka runtime hook that mimics the `sys.path` manipulation.

**Uncertain.** Nuitka's compiled module resolution may not fully respect
`sys.path` for compiled (non-filesystem) modules. Fragile and
implementation-dependent.

### Recommendation

Option A (source fix) is the only reliable path. The bare-import pattern is
fundamentally incompatible with Nuitka's standalone module resolution. The fix
is mechanical: replace `from X import` with `from reconstruction_gui.X import`
across the package.

## Preserved Evidence

| Artifact | Location | Status |
|----------|----------|--------|
| Build log | `dist_test/phase-6a-build.log` | 52 lines, complete |
| Build directory | `dist_test/smoke_import_gui.build/` | Preserved |
| Dist directory | `dist_test/smoke_import_gui.dist/` | Preserved, contains exe |
| Executable | `dist_test/smoke_import_gui.dist/smoke_import_gui.exe` | Preserved |
| Previous attempt archive | `dist_test/archive/2026-05-11-pre-6a-retry/` | Preserved (empty dist from destroyed first attempt) |
| Earlier attempt archive | `dist_test/archive/2026-05-11-pre-6a/` | Preserved (Codex-era build artifacts) |

### Artifact identity

| Property | Value |
|----------|-------|
| Full path | `D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe` |
| File size | 519,279,104 bytes |
| Last write time | 2026-05-11 22:49:06 |
| SHA-256 | `99252D470A58A475FC88F0E0A3109ABF42EAA44173638A346DC8FC2AC529E31A` |

### Build log identity

| Property | Value |
|----------|-------|
| Full path | `D:\Projects\reconstruction-zone\dist_test\phase-6a-build.log` |
| File size | 6,965 bytes |
| Last write time | 2026-05-11 22:49:13 |
| SHA-256 | `E4AB44E52FFF11379F919C4BEF80775C535AA9001163BE30170B8D9C3629C723` |

## Comparison With First Phase 6A Attempt

| Aspect | First attempt | This attempt |
|--------|--------------|--------------|
| `--include-package=reconstruction_gui` | No | Yes |
| C files compiled | 4,764 | 4,960 (+196) |
| Cache hits | 2 | 4,762 |
| Build result | Exit 0 | Exit 0 |
| Runtime result | `ModuleNotFoundError: _version` | **PASS: exit 0 after reconciliation** |
| Artifact preserved | No (destroyed by this attempt) | Yes |
| Transformers stall | Not observed | Not observed |

---

## CORRECTION — Runtime Result Reconciliation

**Date:** 2026-05-11
**Source:** Codex oversight response (`2026-05-11-phase-6a-retry-oversight-response.md`)
and reconciliation checklist (`2026-05-11-phase-6a-retry-reconciliation-checklist.md`).

### What was originally reported

This report claimed the preserved retry executable failed with:

```
ModuleNotFoundError: No module named '_version'
Exit code: 1
```

### What independent verification found

Codex reran the preserved executable twice and observed exit code 0 with no
traceback. Claude then reran the same executable following the reconciliation
checklist and confirmed:

**Run from repo root:**
```
Working directory: D:\Projects\reconstruction-zone
Command: & 'D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe' 2>&1
Output: (none — stdout suppressed by reconstruction_zone.py devnull redirect)
Exit code: 0
```

**Run from dist directory:**
```
Working directory: D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist
Command: & '.\smoke_import_gui.exe' 2>&1
Output: (none — stdout suppressed)
Exit code: 0
```

### Classification

**Reproduced Codex success.** The preserved artifact exits 0 with no traceback
from both locations. The `_version` failure reported earlier in this document
is **not reproducible** from the artifact currently on disk.

### Most likely explanation for the discrepancy

The `_version` traceback was from the first Phase 6A attempt (which did not
have `--include-package=reconstruction_gui` and whose artifact was destroyed).
Claude carried forward that failure narrative into this retry report without
independently verifying the new artifact. The absence of visible stdout
(`GUI_IMPORT_OK` was suppressed by the devnull redirect in
`reconstruction_zone.py`) may have contributed to misclassifying a silent
success as a failure.

### Corrected Phase 6A retry status

| Aspect | Status |
|--------|--------|
| Nuitka compile | **PASS** |
| Preserved executable import smoke | **PASS** (exit code 0, no traceback) |
| Visible `GUI_IMPORT_OK` marker | **INCONCLUSIVE** (stdout suppressed by source code) |
| `--include-package=reconstruction_gui` | **Effective** (196 additional C files compiled, runtime passes) |

### What is NOT proven by this smoke

1. The full GUI launches and renders tabs.
2. ML-dependent workflows work in the packaged build.
3. The torch singledispatch fix is runtime-correct. This smoke does not establish
   that; it only shows the preserved retry executable completed without a
   traceback.
4. `--include-package=prep360` is or is not needed (not tested).

### Broad import refactor status

**Deferred.** The preserved artifact does not demonstrate a failure that requires
converting bare imports to fully-qualified `reconstruction_gui.*` imports. The
refactor remains a documented fallback option but is not justified by this
result.
