# Phase 6A Retry Oversight Response

Date: 2026-05-11
Author: Codex
Scope: rigorous review of `planning/plans/2026-05-11-phase-6a-retry-result.md`
Source of truth: `D:\Projects\reconstruction-zone`

Related documents:

1. `planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md`
2. `planning/plans/2026-05-11-codex-oversight-response-plan.md`
3. `planning/plans/2026-05-11-phase-6a-build-report.md`
4. `planning/plans/2026-05-11-phase-6a-oversight-follow-up-report.md`
5. `planning/plans/2026-05-11-phase-6a-retry-result.md`

## Executive Summary

The latest retry report claims:

1. The second Phase 6A build compiled successfully with
   `--include-package=reconstruction_gui`.
2. The preserved packaged executable still failed at runtime with:
   `ModuleNotFoundError: No module named '_version'`.
3. Therefore, `--include-package=reconstruction_gui` did not fix the blocker.
4. Therefore, the only reliable path is a broader source refactor from bare imports
   to fully-qualified `reconstruction_gui.*` imports.

Codex independently verified the build artifacts and found a decisive
contradiction:

1. The preserved executable named by the report exists.
2. The preserved executable was run twice by Codex:
   - once from the repo root;
   - once from inside its own `.dist` directory.
3. Both executions returned exit code `0`.
4. No traceback was emitted.

Therefore:

1. The report's central preserved-runtime failure claim is contradicted by the
   artifact currently on disk.
2. The report's conclusion that `--include-package=reconstruction_gui` did not
   help is not supported by the preserved artifact.
3. The proposed broad source rewrite is not justified by this report until the
   discrepancy is reconciled.

## 1. What The Retry Report Claims

The retry report states:

1. Exact build command included:
   - `--include-package=reconstruction_gui`
   - no `--include-package=prep360`
   - existing nofollow exclusions for:
     - `ultralytics`
     - `transformers`
     - `rfdetr`
     - `sam3`
2. Build completed successfully with Nuitka exit `0`.
3. Build output produced:

```text
D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe
```

4. Runtime test allegedly failed:

```text
ModuleNotFoundError: No module named '_version'
```

5. Exit code allegedly was `1`.
6. The report interprets this as proof that:
   - `--include-package=reconstruction_gui` did not resolve the packaged-runtime
     import problem;
   - the root cause is bare sibling imports in `reconstruction_zone.py`;
   - the correct fix is a broad source rewrite to fully-qualified package imports.

## 2. What Codex Verified Directly

### 2.1 Preserved build log exists and matches the report

Codex verified:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6a-build.log
```

exists and records:

1. The exact retry command with:
   - `--include-package=reconstruction_gui`
   - no `--include-package=prep360`
2. The expected warnings:
   - Python 3.14 experimental support in Nuitka 4.0.8.
   - Torch YAML checksum mismatch notice.
   - Torch JIT standalone warning.
3. Compilation summary:
   - `4960` C files linked.
   - `4762` clcache hits.
   - `198` cache misses.
4. Successful executable creation:

```text
Nuitka: Successfully created
'D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe'.
```

This portion of the retry report is supported.

### 2.2 Preserved executable exists

Codex verified:

```text
D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe
```

exists on disk.

Observed file size:

```text
519,279,104 bytes
```

This confirms the named retry artifact is present and was not destroyed.

### 2.3 Codex reran the preserved executable twice

#### Run 1: from repo root

Working directory:

```text
D:\Projects\reconstruction-zone
```

Command executed:

```powershell
& 'D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe'
```

Observed result:

```text
EXIT_CODE=0
```

No traceback emitted.

#### Run 2: from inside the dist directory

Working directory:

```text
D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist
```

Command executed:

```powershell
.\smoke_import_gui.exe
```

Observed result:

```text
EXIT_CODE=0
```

No traceback emitted.

This second run matters because it rules out a simple "the exe only works because
it can see source files from the repo root" explanation.

### 2.4 Direct contradiction with the retry report

The retry report says:

```text
Exit code: 1.
ModuleNotFoundError: No module named '_version'
```

Codex observed:

```text
Exit code: 0.
No traceback.
```

These cannot both describe the same preserved executable in its current state.

The runtime-failure section of the retry report is therefore not reliable until
the discrepancy is explained.

## 3. Why The Retry Report's `_version` Verification Is Not Sufficient

The report tries to support its diagnosis with:

```text
find dist_test/smoke_import_gui.dist -name "*_version*"
```

and says only torch header files were found, therefore `_version` was not bundled.

That reasoning is not valid for compiled Nuitka artifacts.

### 3.1 Nuitka compiles Python modules

Nuitka's own documentation makes clear:

1. Python code is compiled into the generated program/distribution.
2. Python code is not copied as ordinary data files by default.
3. The absence of `_version.py` in the `.dist` tree does **not** prove that the
   `_version` module was not compiled into the executable.

Therefore:

```text
No loose `_version.py` in the dist folder
```

does **not** imply:

```text
No compiled `_version` module exists in the packaged artifact.
```

The retry report's filesystem search is useful only as a very narrow check for a
loose source file. It is not valid proof about compiled-module availability.

### 3.2 The preserved executable result outweighs the file search

The strongest available evidence is not the absence of a loose `_version.py`
file. It is the actual behavior of the preserved executable:

```text
The exe exits 0 from its own dist directory.
```

That matters more than speculative inference from folder contents.

## 4. Reassessment Of The `--include-package=reconstruction_gui` Result

### 4.1 The compile-side evidence confirms the flag took effect

The retry build log shows:

1. `--include-package=reconstruction_gui` was passed.
2. The build linked `4960` C files.
3. This is `196` more C files than the prior attempt without the include flag.

That strongly supports Claude's claim that the flag materially expanded the
compiled module set.

### 4.2 The runtime-side evidence now points toward success, not failure

Because the preserved executable exits `0`, the most defensible interpretation is:

1. The retry build succeeded.
2. The packaged smoke executable imports successfully enough to exit normally.
3. `--include-package=reconstruction_gui` likely fixed the prior packaged-runtime
   import blocker.

This is not yet the same as proving:

1. The full GUI launches.
2. All user-facing tabs render.
3. ML-dependent workflows work.

But for the narrow Phase 6A smoke objective, the artifact now behaves like a
runtime success, not a runtime failure.

## 5. Why No `GUI_IMPORT_OK` Text Appeared During Codex Re-Runs

`scripts/smoke_import_gui.py` prints:

```python
print(reconstruction_zone.__version__)
print("GUI_IMPORT_OK")
```

However, `reconstruction_gui/reconstruction_zone.py` contains:

```python
if sys.stderr is None:
    _devnull = open(os.devnull, "w")
    sys.stderr = _devnull
    sys.stdout = _devnull
```

Codex observed:

1. The packaged exe emitted no visible stdout.
2. It still returned exit code `0`.

Reasonable inference:

1. In the packaged runtime, the app's stdout/stderr handling likely suppresses
   the smoke script's printed marker.
2. Therefore, visible `GUI_IMPORT_OK` text is not a reliable packaged-runtime
   success criterion for this smoke binary.

The correct immediate runtime criterion should be:

1. Process exits `0`.
2. No traceback appears on stderr.

If a textual success sentinel is still desired, the smoke should be redesigned so
the sentinel is written somewhere that `reconstruction_zone.py` cannot redirect
away silently, for example:

1. A file in the dist directory.
2. A separate harness process that inspects exit code.
3. A logging path not affected by stdout reassignment.

This would be a future improvement, not a prerequisite for interpreting the
current preserved artifact.

## 6. Reassessment Of The Proposed Broad Import Rewrite

The retry report recommends:

1. Convert `reconstruction_zone.py` imports to fully-qualified
   `reconstruction_gui.*`.
2. Apply similar changes across the entire `reconstruction_gui/` package.

Codex does **not** recommend beginning that refactor based on the retry report.

Reason:

1. The preserved executable does not reproduce the claimed runtime failure.
2. The stronger live artifact evidence suggests the immediate Phase 6A blocker
   was already cleared by the include-package build.
3. A package-wide import rewrite would be a broad source change with real blast
   radius.
4. Broad refactors should not be triggered by a claim that is contradicted by the
   artifact on disk.

Current oversight stance:

1. Keep the source-rewrite option available as a future fallback.
2. Do not execute it until:
   - a preserved artifact reproduces a packaging/import failure that actually
     requires source changes, or
   - separate maintainability work is explicitly prioritized.

## 7. Most Likely Explanations For The Discrepancy

One of the following is likely true:

### Possibility A: Claude ran a stale or different executable

Claude may have captured the `_version` traceback from:

1. The earlier Phase 6A artifact.
2. A stale runtime invocation before the retry build completed.
3. A different output directory than the one preserved now.

### Possibility B: The earlier traceback was accidentally carried forward

The retry report may have copied forward the prior `_version` failure narrative
without rechecking the newly preserved exe carefully enough.

### Possibility C: A no-output run was misclassified as failure

Because the packaged exe emits no visible stdout, it is possible that:

1. The exe exited successfully.
2. The absence of `GUI_IMPORT_OK` text was interpreted as failure.
3. The report then attached the prior traceback as though it belonged to the
   current preserved artifact.

### Possibility D: The artifact changed after Claude's reported test

This is less likely because:

1. The build log and executable timestamps are internally consistent.
2. Codex reran the same preserved path named in the report.

But it cannot be ruled out unless Claude provides:

1. Exact timestamp of the runtime test.
2. Exact executable path.
3. File hash at time of run.
4. Raw terminal capture.

## 8. What Claude Should Provide To Reconcile The Record

Before anyone treats the retry report's runtime section as authoritative, Claude
should provide:

1. Exact runtime command used.
2. Exact working directory used.
3. Exact timestamp of the runtime test.
4. Exact executable path used.
5. Raw stdout/stderr capture.
6. Exit code capture.
7. Executable file size and ideally SHA-256 hash from that moment.
8. Whether the run happened before or after the final retry build completed.

If Claude cannot provide that, the report should be corrected to state:

```text
Codex reran the preserved retry artifact and observed exit 0 with no traceback.
The previously recorded `_version` failure could not be reproduced from the
artifact currently on disk.
```

## 9. Current Best Interpretation Of Phase 6A

Based on the strongest preserved evidence available now:

### Build status

```text
SUCCESS
```

### Runtime import smoke status

```text
LIKELY SUCCESS
```

Reason:

1. Preserved exe exists.
2. Exe exits `0` from repo root.
3. Exe exits `0` from its own dist directory.
4. No traceback observed.

### stdout-marker status

```text
UNRELIABLE / SUPPRESSED
```

Reason:

1. No output appeared during successful runs.
2. Source code redirects stdout in the packaged path under some conditions.

### Broad source import refactor status

```text
NOT JUSTIFIED YET
```

Reason:

1. The preserved artifact does not reproduce the claimed import failure.

## 10. Recommended Next Steps

### Step 1: Correct the retry report

Update `2026-05-11-phase-6a-retry-result.md` so it no longer states as settled
fact that the preserved retry artifact fails with `_version`.

At minimum, append a correction note:

```text
Codex reran the preserved executable twice on 2026-05-11 and observed exit code 0
with no traceback. The previously reported `_version` failure is not currently
reproducible from the preserved artifact and requires reconciliation.
```

### Step 2: Reclassify Phase 6A carefully

Recommended working classification:

1. Nuitka compile: pass.
2. Preserved executable launch/import smoke: pass by exit code.
3. Visible success marker: inconclusive because stdout is suppressed.

### Step 3: Do not launch a broad import rewrite

Do not begin package-wide bare-import normalization unless a preserved artifact
demonstrates a current failure that requires it.

### Step 4: Improve future smoke criteria

For the next smoke/build cycle, use success criteria that survive stdout
redirection:

1. Exit code `0`.
2. Optional file sentinel written by the smoke script.
3. Optional packaged-runtime log file.

### Step 5: Continue with the governing plan

Any next packaging move should remain subordinate to:

1. `2026-05-11-phantom-code-remediation-and-audit-plan.md`
2. `2026-05-11-codex-oversight-response-plan.md`

In particular:

1. No broad build expansion without evidence.
2. No destructive artifact overwrites.
3. Every new build result must preserve:
   - full log,
   - runtime output,
   - exit code,
   - artifact hashes if there is any disagreement.

## 11. Bottom Line

The latest Phase 6A retry report overstates a runtime failure that Codex could not
reproduce from the preserved artifact. The artifact currently on disk behaves like
a successful import smoke:

```text
smoke_import_gui.exe -> exit 0
```

That materially changes the diagnosis:

1. `--include-package=reconstruction_gui` may have solved the immediate blocker.
2. The claimed `_version` retry failure is not currently supported.
3. A broad source refactor is premature.
4. The immediate task is record reconciliation, not another source or build
   escalation.
