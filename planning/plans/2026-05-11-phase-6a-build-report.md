# Phase 6A Build Report — Constrained GUI Nuitka Smoke

> Update, 2026-05-11: Later artifact-specific reverification showed the
> preserved Phase 6A retry executable exits `0` from both the repo root and its
> `.dist` directory. The `_version` runtime-failure discussion in this report is
> historical context from the earlier Phase 6A line of investigation and should
> not be treated as the confirmed result of the preserved retry artifact.

Date: 2026-05-11
Build executor: Claude
Build status: COMPILED SUCCESSFULLY, RUNTIME FAILURE

## Build Command

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
  --include-data-dir=reconstruction_gui/docs=reconstruction_gui/docs
  --include-data-dir=resources=resources
  --include-data-files=reconstruction-zone.ico=reconstruction-zone.ico
  --output-dir=dist_test
  scripts/smoke_import_gui.py
```

## Build Environment

- Nuitka 4.0.8
- Python 3.14.0 (experimentally supported)
- C compiler: cl 14.5
- OS: Windows 11 Pro
- Working directory: `D:\Projects\reconstruction-zone`
- Branch: `release/v1`

## Build Output (complete, 52 lines)

```
Nuitka-Options: Used command line options: [as above]
Nuitka:WARNING: The Python version '3.14' is only experimentally supported by
  Nuitka '4.0.8', but an upcoming release will change that.
Nuitka: Starting Python compilation with:
  Version '4.0.8' on Python 3.14 (flavor 'CPython Official')
  commercial grade 'not installed'.
Nuitka: Detected 2 module(s) with mismatching checksum in
  'scripts/nuitka-torch-fix.yml': torch.library, torch._refs
Nuitka-Plugins:WARNING: options-nanny: Module 'torch' has parameter:
  Torch JIT is disabled by default in standalone mode
Nuitka-Plugins:anti-bloat: Not including 'pandas.core._numba.extensions'
  automatically in order to avoid bloat
Nuitka-Plugins:matplotlib: Using configuration file or default backend 'tkagg'.
Nuitka-Plugins:dll-files: Found 1 file DLLs from cv2 installation.
Nuitka-Plugins:dll-files: Found 2 files DLLs from sklearn._distributor_init.
Nuitka-Plugins:dll-files: Found 37 files DLLs from torch installation.
Nuitka-Plugins:dll-files: Found 3 files DLLs from torchvision installation.
Nuitka-Inclusion: Including Windows Runtime DLLs.
Nuitka: Completed Python level compilation and optimization.
Nuitka: Generating source code for C backend compiler.
Nuitka: Running data composer tool for optimal constant value handling.
Nuitka: Running C compilation via Scons.
Nuitka-Scons: Backend C compiler: cl (cl 14.5).
Nuitka-Scons: Backend C linking with 4764 files.
Nuitka-Scons: Compiled 4764 C files using clcache with 2 cache hits
  and 4762 cache misses.
Nuitka: Successfully created
  'D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe'.
```

## What Worked

1. Nuitka compiled and linked successfully (exit 0).
2. The `--nofollow-import-to` exclusions for ultralytics, transformers, rfdetr,
   and sam3 prevented the Transformers model-tree stall. No child process
   entered `define_import_structure` at any point.
3. The `scripts/nuitka-torch-fix.yml` singledispatch patch applied correctly to
   `torch.library` and `torch._refs` (2 modules detected).
4. Torch (37 DLLs), torchvision (3 DLLs), cv2 (1 DLL), sklearn (2 DLLs) were
   all included.
5. 4,764 C files compiled and linked.

## Runtime Failure

```
$ dist_test/smoke_import_gui.dist/smoke_import_gui.exe

Traceback (most recent call last):
  File ".../smoke_import_gui.py", line 21, in <module>
  File ".../reconstruction_gui/reconstruction_zone.py", line 74,
    in <module reconstruction_gui.reconstruction_zone>
ModuleNotFoundError: No module named '_version'
```

## Root Cause

`reconstruction_zone.py` adds `reconstruction_gui/` to `sys.path` at runtime
(lines 32-37) and then uses bare-name imports:

```python
# Lines 32-37: sys.path manipulation
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

# Line 74: bare import — works at runtime, invisible to Nuitka
from _version import __version__

# Line 75-85: more bare imports
from widgets import (...)
from app_infra import AppInfrastructure
from tabs.alignment_tab import build_alignment_tab
from tabs.source_tab import build_source_tab
from tabs.projects_tab import build_projects_tab
```

Nuitka's static analysis follows imports from the smoke script's entry point:

```
smoke_import_gui.py
  → import reconstruction_gui.reconstruction_zone
    → from _version import __version__   ← Nuitka cannot resolve this
```

Nuitka sees `_version` as a top-level module name. It does not know that
`reconstruction_gui/` will be on `sys.path` at runtime. Since no top-level
`_version` package exists in the Python environment, Nuitka silently skips it
during compilation. At runtime, the bundled `reconstruction_zone.py` tries the
bare import, `reconstruction_gui/` is not on `sys.path` in the standalone
bundle's module resolution, and the import fails.

The same issue would occur for `widgets`, `app_infra`, `tabs.alignment_tab`,
`tabs.source_tab`, and `tabs.projects_tab` — all are bare imports that depend
on `reconstruction_gui/` being on `sys.path`.

## Fix Options

### Option A: `--include-package=reconstruction_gui`

Add `--include-package=reconstruction_gui` to the Nuitka command. This tells
Nuitka to compile and bundle every `.py` file in the `reconstruction_gui/`
package, making `_version`, `widgets`, `app_infra`, `tabs/*` etc. available.

**Risk:** `reconstruction_gui/reconstruction_pipeline.py` imports `ultralytics`,
`rfdetr`, `sam3`, and `transformers` inside lazy functions. With
`--include-package`, Nuitka will compile `reconstruction_pipeline.py` and
encounter those imports. Whether `--nofollow-import-to` prevents following them
in this case needs verification.

**Evidence so far:** A partial second build attempt used this flag combination.
Process inspection showed no Transformers stall after 376s CPU / 3GB working
set. The build was killed before completion, so this is not proof, but it is
a positive signal.

Similarly, `--include-package=prep360` would be needed because `source_tab.py`
imports from `prep360.core`. Same risk profile — `prep360/core/segmenter.py`
imports `ultralytics` at module level, but `--nofollow-import-to` should
prevent following it.

### Option B: Rewrite bare imports to fully-qualified

Change `reconstruction_zone.py` to use fully-qualified imports:

```python
from reconstruction_gui._version import __version__
from reconstruction_gui.widgets import (...)
from reconstruction_gui.app_infra import AppInfrastructure
from reconstruction_gui.tabs.alignment_tab import build_alignment_tab
from reconstruction_gui.tabs.source_tab import build_source_tab
from reconstruction_gui.tabs.projects_tab import build_projects_tab
```

**Risk:** This is a source change that affects all runtime paths, not just
packaging. Every other module in `reconstruction_gui/` that uses bare imports
would also need updating for consistency. This is a larger change with more
surface area for breakage.

### Option C: Individual `--include-module` flags

```
--include-module=reconstruction_gui._version
--include-module=reconstruction_gui.widgets
--include-module=reconstruction_gui.app_infra
--include-module=reconstruction_gui.tabs.alignment_tab
--include-module=reconstruction_gui.tabs.source_tab
--include-module=reconstruction_gui.tabs.projects_tab
```

Plus every transitive bare import those modules make. This is precise but
fragile — any new bare import would need a new flag.

### Recommendation

Option A (`--include-package=reconstruction_gui --include-package=prep360`)
is the most practical. The partial evidence from the killed second build
suggests `--nofollow-import-to` holds even with `--include-package`, but this
needs to be verified to completion.

## Build Artifact Status

The successful build artifact (`smoke_import_gui.exe`) was destroyed when a
second build attempt was launched into the same output directory without
archiving the first. The build must be re-run.

## What Must Happen Before Next Attempt

1. Archive any existing `dist_test/smoke_import_gui.build/` and `.dist/`
   directories to `dist_test/archive/<timestamp>/` before starting.
2. Add `--include-package=reconstruction_gui --include-package=prep360` to
   the command.
3. Run in foreground or with visible output streaming (no `| tail`).
4. Monitor for Transformers stall via PowerShell `Get-Process` and child
   process inspection.
5. If build succeeds, test the exe IMMEDIATELY before doing anything else.
6. Do not start another build until the exe has been tested and results
   recorded.
