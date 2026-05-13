# Phase 6A Oversight Follow-Up Report

> Update, 2026-05-11: The later preserved retry artifact was independently
> rechecked and exits `0` from both the repo root and its `.dist` directory. The
> `_version` runtime-failure framing below should be read as historical analysis
> from before that reconciliation, not as the current Phase 6A retry result.

Date: 2026-05-11
Author: Codex
Scope: conservative review of Claude's `2026-05-11-phase-6a-build-report.md`
Source of truth: `D:\Projects\reconstruction-zone`

Related governing documents:

1. `planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md`
2. `planning/plans/2026-05-11-codex-oversight-response-plan.md`
3. `planning/plans/2026-05-11-phase-6a-build-report.md`

## Executive Summary

Claude's Phase 6A report is directionally credible:

1. A constrained GUI Nuitka smoke apparently compiled successfully.
2. The runtime then failed on `ModuleNotFoundError: No module named '_version'`.
3. The live source supports the diagnosis that `reconstruction_gui/reconstruction_zone.py`
   relies on bare sibling imports after runtime `sys.path` manipulation.

However, the report should not be treated as fully closed or fully proven:

1. The successful executable was destroyed by a second build attempt, so Codex could
   not independently rerun it or reproduce the exact failure.
2. The recommendation to add both `--include-package=reconstruction_gui` and
   `--include-package=prep360` is broader than the evidence currently supports.
3. Several claims in the report are plausible but not durably evidenced, because
   the second attempt's live monitoring was not preserved as logs or snapshots.

Codex recommendation:

1. Next attempt should add **only** `--include-package=reconstruction_gui`.
2. Do **not** add `--include-package=prep360` unless the next build/runtime result
   proves it is required.
3. Preserve artifacts and logs before any retry.

## 1. What Codex Verified

### 1.1 Claude's build environment matches the local toolchain

Codex verified:

1. Nuitka version: `4.0.8`
2. Python version: `3.14.0`
3. Platform: Windows 11 / CPython official install

This matches the build report.

Additional note:

Running `C:\Python314\python.exe -m nuitka --version` during verification also
emitted an `SConsLockFailure` timeout against Nuitka's MSVC cache file:

```text
C:\Users\alexm\AppData\Local\Nuitka\Nuitka\Cache\scons-msvc-config\content-14.5.json
```

That does not invalidate Claude's earlier build, but it is a preflight warning for
any next attempt. A retry should confirm the Nuitka/SCons cache lock is no longer
contended before starting another long compile.

### 1.2 The report's main runtime diagnosis matches live source

Claude attributes the runtime failure to bare imports in:

`reconstruction_gui/reconstruction_zone.py`

Codex verified the live file does this:

1. Adds `reconstruction_gui/` and project root to `sys.path` at runtime.
2. Uses bare imports immediately afterward:
   - `from _version import __version__`
   - `from widgets import (...)`
   - `from app_infra import AppInfrastructure`
   - `from tabs.alignment_tab import build_alignment_tab`
   - `from tabs.source_tab import build_source_tab`
   - `from tabs.projects_tab import build_projects_tab`

Codex also verified all of those source files exist:

1. `reconstruction_gui/_version.py`
2. `reconstruction_gui/widgets.py`
3. `reconstruction_gui/app_infra.py`
4. `reconstruction_gui/tabs/alignment_tab.py`
5. `reconstruction_gui/tabs/source_tab.py`
6. `reconstruction_gui/tabs/projects_tab.py`

Therefore:

1. The runtime failure is not a missing source file.
2. It is a packaging/import-resolution failure in the compiled artifact.

### 1.3 The exact runtime failure could not be independently replayed

Claude's report states:

```text
dist_test/smoke_import_gui.dist/smoke_import_gui.exe
```

was successfully produced and then destroyed by a subsequent build attempt into
the same output directory.

Codex verified:

1. `dist_test/smoke_import_gui.dist/` currently exists.
2. It is currently empty.

Therefore:

1. Codex could not rerun the built executable.
2. Codex could not independently reproduce the exact `_version` traceback from
   the artifact.
3. The traceback in Claude's report should be treated as credible but not
   independently revalidated after the artifact was lost.

## 2. Findings Requiring Correction Or Caution

### 2.1 The diagnosis is directionally right but slightly overstated

Claude's report says the standalone runtime did not have `reconstruction_gui/`
on `sys.path`.

Codex recommends narrower wording:

1. The observed failure proves `_version` was not bundled/resolvable at runtime.
2. Because the `sys.path` insertion code executes before the bare import line,
   the traceback alone does not prove path insertion never occurred.
3. The safer diagnosis is:

```text
Nuitka did not bundle the bare-imported sibling modules in a way that satisfied
runtime resolution inside the standalone executable.
```

### 2.2 `--include-package=reconstruction_gui` is justified

The next attempt needs to address bare sibling imports under
`reconstruction_gui/`.

Adding:

```text
--include-package=reconstruction_gui
```

is justified because:

1. The verified missing import was `_version`.
2. The same file also imports multiple sibling modules using bare names.
3. Including the package is the most practical low-scope packaging-side response
   before considering source rewrites.

### 2.3 `--include-package=prep360` is not yet justified

Claude recommends:

```text
--include-package=reconstruction_gui --include-package=prep360
```

Codex considers the second half premature.

Reason:

1. The observed runtime failure is entirely within `reconstruction_gui`.
2. `reconstruction_gui/tabs/source_tab.py` imports `prep360.core` using fully
   qualified package imports.
3. The Phase 6A report does not show a runtime failure related to unresolved
   `prep360` modules.

Conservative recommendation:

1. Add only `--include-package=reconstruction_gui` on the next retry.
2. Re-run the smoke.
3. Add `--include-package=prep360` only if the next artifact fails in a way that
   specifically requires it.

### 2.4 Claims about the partial second build are not independently verifiable

Claude states that a second attempt with broader includes showed:

1. No Transformers stall after `376s CPU`.
2. About `3GB` working set.

Codex could not independently verify this because:

1. No preserved process snapshot was found.
2. No durable monitoring log for that second attempt was found.
3. The build was killed before completion.

Status:

1. Plausible.
2. Useful as anecdotal evidence.
3. Not strong enough to settle the include-package strategy.

### 2.5 The torch package-config claim is stronger than the evidence

Claude writes that:

```text
scripts/nuitka-torch-fix.yml
```

applied correctly.

What the build excerpt actually proves:

1. Nuitka read the config file.
2. Nuitka reported two modules with mismatching checksum:
   - `torch.library`
   - `torch._refs`

What it does **not** prove:

1. The transformed code behaved correctly at runtime.
2. Torch import/runtime paths reached by the smoke were healthy after the patch.

Reason:

The built executable failed earlier on `_version`, before any meaningful
torch-path runtime validation occurred.

Better wording:

```text
The user package configuration was detected and engaged by Nuitka; runtime
effectiveness remains unverified by this smoke due to earlier `_version` failure.
```

## 3. Packaging Strategy Recommendation

### 3.1 Next attempt should stay narrow

Use the same Phase 6A command as Claude's successful compile attempt, plus:

```text
--include-package=reconstruction_gui
```

Do not add:

```text
--include-package=prep360
```

unless a subsequent runtime failure proves it is necessary.

### 3.2 Preserve evidence before retrying

Before a new attempt:

1. Move existing:
   - `dist_test/smoke_import_gui.build/`
   - `dist_test/smoke_import_gui.dist/`
2. Into:
   - `dist_test/archive/<timestamp>/`
3. Do not destroy the prior directories.

The Phase 6A report already recommends this. Codex agrees and considers it now
mandatory.

### 3.3 Preserve build/runtime logs

The next attempt must produce durable evidence:

1. Full Nuitka stdout/stderr log.
2. Process-monitor notes if stall detection is needed.
3. The runtime output of the produced executable.
4. Exact exit codes for:
   - Nuitka build.
   - packaged executable run.

Do not start a second build before testing and recording the first executable.

## 4. Compatibility With Governing Plans

This follow-up is consistent with:

1. `2026-05-11-phantom-code-remediation-and-audit-plan.md`
2. `2026-05-11-codex-oversight-response-plan.md`

In particular:

1. No full build should begin until source gates remain green.
2. Phase 6A should be treated as a constrained smoke, not proof of final package
   readiness.
3. Evidence preservation is required before iterating.
4. Build attempts should stop immediately if Transformers model-tree analysis
   returns.

## 5. Recommended Next-Step Checklist

1. Confirm no active Git/Nuitka/Python build processes are running.
2. Confirm no `.git/index.lock`.
3. Confirm the Nuitka/SCons cache lock warning is no longer active.
4. Archive current `smoke_import_gui.build/` and `.dist/`.
5. Re-run Phase 6A with:
   - existing nofollow exclusions,
   - existing torch package config,
   - existing data includes,
   - `--include-package=reconstruction_gui`,
   - no `--include-package=prep360` yet.
6. Capture a full log.
7. Run the executable immediately if build exits 0.
8. Record exact stdout/stderr and exit code.
9. Only then decide whether `prep360` needs explicit package inclusion.

## 6. Bottom Line

Claude's Phase 6A report identifies a real next blocker:

```text
compiled artifact failed on a packaged-runtime import resolution problem for
bare `reconstruction_gui` sibling imports.
```

The next attempt should address exactly that blocker and no more:

```text
add --include-package=reconstruction_gui
```

Codex does **not** recommend adding `--include-package=prep360` yet. That is a
larger graph-expanding change without direct supporting failure evidence from the
Phase 6A artifact that was actually observed.
