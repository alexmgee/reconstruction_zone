# Phase 6A Retry Reconciliation Checklist

Date: 2026-05-11
Author: Codex
Purpose: actionable follow-through checklist for
`planning/plans/2026-05-11-phase-6a-retry-oversight-response.md`
Source of truth: `D:\Projects\reconstruction-zone`

Related governing documents:

1. `planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md`
2. `planning/plans/2026-05-11-codex-oversight-response-plan.md`
3. `planning/plans/2026-05-11-phase-6a-retry-result.md`
4. `planning/plans/2026-05-11-phase-6a-retry-oversight-response.md`

## Objective

Resolve the contradiction between:

1. Claude's retry report, which claims the preserved Phase 6A retry executable
   failed with:

```text
ModuleNotFoundError: No module named '_version'
Exit code: 1
```

2. Codex's direct verification, which found that the same preserved executable
   currently returns:

```text
EXIT_CODE=0
```

from both:

1. the repo root; and
2. the `.dist` directory itself.

Do not proceed to new source refactors or build escalations until this record is
reconciled.

## Operating Constraints

1. Work only in `D:\Projects\reconstruction-zone`.
2. Preserve artifacts. Do not overwrite, delete, or rebuild over the current
   Phase 6A retry output until reconciliation is complete.
3. Do not begin a package-wide import rewrite based on the current retry report.
4. Do not launch another Nuitka build until the current artifact's truth state is
   settled and the retry report is corrected or corroborated.
5. Treat stdout markers as unreliable until proven otherwise, because packaged
   runtime stdout may be redirected to devnull.
6. Use explicit command output, timestamps, and exit codes. Do not rely on memory.

## Phase 1: Freeze And Identify The Artifact

### Checklist

- [ ] Confirm the retry executable exists:

```powershell
Get-Item 'D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe'
```

- [ ] Record:
  - Full path.
  - File size.
  - Last write time.
  - SHA-256 hash.

Suggested command:

```powershell
Get-Item 'D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe' |
  Select-Object FullName,Length,LastWriteTime

Get-FileHash 'D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe' -Algorithm SHA256
```

- [ ] Confirm the preserved build log exists:

```powershell
Get-Item 'D:\Projects\reconstruction-zone\dist_test\phase-6a-build.log'
```

- [ ] Record:
  - Full path.
  - File size.
  - Last write time.
  - SHA-256 hash.

### Acceptance Criteria

Proceed only when:

1. The executable and build log are uniquely identified.
2. Their timestamps/hashes are recorded in a durable note or corrected report.

## Phase 2: Re-Run The Preserved Executable Reproducibly

### 2A. Run From Repo Root

- [ ] Set working directory:

```powershell
Set-Location 'D:\Projects\reconstruction-zone'
```

- [ ] Run:

```powershell
$output = & 'D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist\smoke_import_gui.exe' 2>&1
'OUTPUT_BEGIN'
$output
'OUTPUT_END'
'EXIT_CODE=' + $LASTEXITCODE
```

- [ ] Preserve exact terminal text.

### 2B. Run From Dist Directory

- [ ] Set working directory:

```powershell
Set-Location 'D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist'
```

- [ ] Run:

```powershell
$output = & '.\smoke_import_gui.exe' 2>&1
'OUTPUT_BEGIN'
$output
'OUTPUT_END'
'EXIT_CODE=' + $LASTEXITCODE
```

- [ ] Preserve exact terminal text.

### Acceptance Criteria

Record both runs exactly.

Classify the result as one of:

1. **Reproduced Claude failure**
   - traceback appears;
   - exit code `1`.
2. **Reproduced Codex success**
   - no traceback;
   - exit code `0`.
3. **Mixed / unstable**
   - result differs by working directory or run-to-run.

Do not continue until one of those three classifications is explicitly chosen.

## Phase 3: Reconcile The Retry Report Against Artifact Reality

### If The Executable Reproduces Codex Success

- [ ] Append or patch `2026-05-11-phase-6a-retry-result.md` with a correction note:

```text
Codex reran the preserved retry executable from both the repo root and the dist
directory and observed exit code 0 with no traceback. The previously reported
`ModuleNotFoundError: No module named '_version'` is not currently reproducible
from the preserved artifact and should not be treated as the settled Phase 6A
runtime result.
```

- [ ] Reclassify Phase 6A retry status:
  - Build: PASS.
  - Preserved executable import smoke: PASS by exit code.
  - Visible `GUI_IMPORT_OK` marker: INCONCLUSIVE / stdout-suppressed.

- [ ] Mark the broad import-refactor recommendation as:
  - Deferred.
  - Not justified by the preserved retry artifact.

### If The Executable Reproduces Claude Failure

- [ ] Preserve the exact traceback and exit code.
- [ ] Confirm the executable hash matches the one described in the retry report.
- [ ] If hashes match, treat the failure as reproducible and continue with targeted
  root-cause analysis.
- [ ] If hashes differ, document that multiple artifacts existed and the retry
  report needs artifact identity correction.

### If The Result Is Mixed Or Unstable

- [ ] Stop.
- [ ] Do not refactor or rebuild.
- [ ] Capture:
  - working directory;
  - command;
  - stdout/stderr;
  - exit code;
  - exe hash;
  - timestamps.
- [ ] Repeat until the instability pattern is understood.

## Phase 4: Validate The Meaning Of "Success"

### Problem

The smoke script prints:

```python
print(reconstruction_zone.__version__)
print("GUI_IMPORT_OK")
```

But packaged runtime may suppress stdout because
`reconstruction_gui/reconstruction_zone.py` redirects stdout/stderr to devnull
when `sys.stderr is None`.

### Checklist

- [ ] Confirm whether packaged runtime suppresses stdout.
- [ ] Do not treat absent `GUI_IMPORT_OK` text as automatic failure unless stdout
  is known to remain visible.
- [ ] Define Phase 6A success criteria as:
  - executable launches;
  - no traceback;
  - exit code `0`.

### Optional Improvement

If a durable success sentinel is still desired:

- [ ] Modify the smoke harness in a future intentional source change so it writes a
  marker file outside redirected stdout, for example:

```text
dist_test/smoke_import_gui.dist/gui_import_ok.marker
```

or another controlled location.

Do not make that change merely to reconcile the current report unless the user
explicitly asks for the harness improvement.

## Phase 5: Reassess The Need For Source Refactor

### Decision Gate

Only consider a package-wide bare-import refactor if one of these is true:

1. The preserved retry artifact reproducibly fails because of unresolved bare
   sibling imports.
2. A new preserved artifact after reconciliation shows a current import failure
   that `--include-package=reconstruction_gui` does not solve.
3. The user explicitly prioritizes import hygiene refactoring independent of the
   Phase 6A result.

### If The Refactor Is Not Yet Justified

- [ ] Do not change:
  - `reconstruction_zone.py`
  - `source_tab.py`
  - `alignment_tab.py`
  - other `reconstruction_gui/` files
  solely to normalize import style.

- [ ] Keep the refactor as a documented fallback option, not the next action.

## Phase 6: Decide The Next Packaging Move

### If Phase 6A Retry Is Reclassified As Success

Recommended next action:

- [ ] Treat Phase 6A as passed at the "compiled import smoke" level.
- [ ] Do not rerun the same smoke just to regenerate identical evidence.
- [ ] Update the packaging handoff to say:
  - `--include-package=reconstruction_gui` succeeded;
  - stdout marker is unreliable;
  - the executable itself exits `0`.

Next step after that should be decided against the governing plans, likely:

1. stronger packaged smoke verification; or
2. preparation for the next bounded packaging phase,

but not a reactive source rewrite based on a failure that no longer reproduces.

### If Phase 6A Retry Truly Fails

Recommended next action:

- [ ] Keep the failure narrow.
- [ ] Identify the exact unresolved import.
- [ ] Choose the smallest remedy that directly addresses that import.
- [ ] Avoid package-wide rewrites until a narrow remedy is shown insufficient.

## Phase 7: Required Report Updates

### Retry Result Report

- [ ] Correct or qualify the runtime result section.
- [ ] State whether the preserved artifact currently reproduces the claimed failure.
- [ ] State exact verification commands and exit codes.

### Oversight Response Report

- [ ] If reconciliation changes the current conclusion, update:
  - `2026-05-11-phase-6a-retry-oversight-response.md`
  with a short addendum.

### Handoff / Progress Reports

- [ ] Update any central handoff or progress report that summarizes Phase 6A so it
  does not continue to repeat the stale `_version` retry-failure claim without
  qualification.

## Phase 8: Stop Conditions

Stop immediately if:

1. The executable hash differs from the artifact claimed in the retry report and
   nobody can identify when the change occurred.
2. Another tool overwrites or deletes the current `.dist` directory before
   reconciliation is complete.
3. A new Nuitka build is started before the current discrepancy is documented.
4. Codex or another tool starts a background Git staging storm again.
5. Runtime behavior differs across repeated runs and nobody has recorded the exact
   environment delta.

## Phase 9: Definition Of Done

This reconciliation task is complete only when all are true:

1. The exact retry executable and build log are uniquely identified by path,
   timestamp, size, and hash.
2. The executable has been rerun from:
   - repo root;
   - dist directory.
3. The exact stdout/stderr and exit codes are preserved.
4. The retry report is corrected or corroborated.
5. Phase 6A retry status is explicitly classified as one of:
   - runtime success;
   - runtime failure;
   - unstable/inconclusive.
6. Any next packaging/source action is chosen from that corrected classification,
   not from a stale or contradicted report.

## Bottom Line

The immediate action is **record reconciliation**, not another build and not a
package-wide source refactor.

The checklist outcome should answer one narrow question definitively:

```text
Does the preserved Phase 6A retry executable actually fail, or does it pass?
```

Until that is settled and the report record is corrected, no broader conclusion
about `_version`, bare imports, or package-wide refactoring should be treated as
final.
