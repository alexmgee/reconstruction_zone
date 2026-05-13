# Codex Oversight Response Plan

Date: 2026-05-11
Project: Reconstruction Zone
Source of truth: `D:\Projects\reconstruction-zone`
Reviewed report: `planning/plans/2026-05-11-remediation-progress-report.md`
Purpose: conservative oversight of the updated remediation report and proposed Nuitka build plan.

## Governing Context

This document is an oversight supplement to:

`planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md`

It does not replace that plan. The remediation/audit plan remains the governing checklist for:

1. Phase 0 stabilization.
2. Phases 1-4 source cleanup.
3. Phase 5 source verification gates.
4. Phase 6 packaging sequencing.
5. Phase 7 full-scale phantom issue audit, Audits A-J.
6. Phase 8 definition of done.

Any build recommendation in this document is subordinate to the original plan's rule:

> No full Nuitka build should run until the source-level gates in the remediation/audit plan pass.

The most important original-plan requirements that remain active are:

1. Keep this plan and the incident report open while editing.
2. Run Gate 1 phantom-term search before packaging.
3. Run Gate 2 local missing import audit before packaging.
4. Ensure `HAS_PREP360=True` for Source tab import before packaging.
5. Ensure GUI shell and Source tab imports do not load `ultralytics`, `transformers`, or `rfdetr`.
6. Create the persistent `scripts/audit_local_imports.py` described in Phase 7 Audit B.
7. Complete Phase 7 Audits A-J, including documentation/handover cleanup.
8. Do not use full builds as a discovery mechanism.

## 0. Operating Rules For This Oversight Pass

1. Work only in `D:\Projects\reconstruction-zone` unless the user explicitly authorizes another path.
2. Do not stage files, commit, push, clean, reset, or run whole-repo `git add`.
3. Do not run Nuitka builds unless the user explicitly redirects build execution to Codex.
4. Treat Claude's report as a claim set, not as proof.
5. Separate:
   - Verified locally.
   - Verified from official documentation.
   - Plausible but unverified.
   - Contradicted by live source or local tool output.
6. Avoid destructive cleanup by default. Prefer timestamped rename/move of build outputs. Use `Remove-Item -Recurse` only with explicit user approval.
7. Before any build-related work, verify no active `git.exe`, `python.exe`, `pythonw.exe`, or Nuitka process is already running.
8. Stop immediately if Codex or any tool starts spawning background `git add -A` again.

## 1. Inputs Reviewed

1. Updated remediation report:
   - `planning/plans/2026-05-11-remediation-progress-report.md`
   - Last observed timestamp: 2026-05-11 19:58 local.

2. Governing remediation/audit plan:
   - `planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md`
   - Key active sections: Phase 5 gates, Phase 6 packaging sequencing, Phase 7 Audits A-J, Phase 8 definition of done.

3. Live source files inspected:
   - `reconstruction_gui/tabs/source_tab.py`
   - `prep360/core/lut.py`
   - `scripts/smoke_import_gui.py`
   - `scripts/nuitka-torch-fix.yml`
   - `reconstruction_gui/reconstruction_zone.py`
   - `reconstruction_gui/reconstruction_pipeline.py`

4. Local tool output inspected:
   - `C:\Python314\python.exe -m nuitka --help`
   - `C:\Python314\python.exe -m nuitka --help-all`
   - `C:\Python314\python.exe -m nuitka --plugin-list`

5. Official documentation checked:
   - Nuitka user manual: `https://nuitka.net/user-documentation/user-manual.html`
   - Nuitka unwanted module guidance: `https://nuitka.net/info/unwanted-module.html`
   - Nuitka use cases: `https://nuitka.net/user-documentation/use-cases.html`

## 2. Immediate Status From Codex

1. Codex did not run a Nuitka build during this review.
2. Codex did not stage, commit, reset, clean, or push.
3. A prior Codex-side `git add -A` respawn loop was observed and stopped.
4. `.gitignore` was updated to ignore local Nuitka smoke outputs:
   - `dist_test/`
   - `*.build/`
   - `*.dist/`
   - `nuitka-report.xml`
5. The last verified Git process state after that intervention was clear, with no `.git/index.lock`.

## 3. High-Level Oversight Verdict

The updated report is substantially better than the earlier handoff because it makes explicit decisions about `PostProcessingError`, LUT API behavior, and build monitoring. However, it should not be accepted as build-ready without corrections.

Main reasons:

1. The report contains at least one direct contradiction about `LUTProcessor.apply_lut`.
2. The Phase 6A smoke success criteria likely names the wrong executable unless `--output-filename=SmokeGuiImport.exe` is added.
3. Phase 6B's `--nofollow-import-to="transformers.models.*"` strategy is high risk for RF-DETR runtime because Nuitka's documented behavior is that excluded imports can raise `ImportError` when reached at runtime.
4. The proposed RF-DETR pre-build probe is too shallow because importing `rfdetr` does not prove which `transformers.models.*` modules are needed when an RF-DETR model class is instantiated.
5. The proposed pre-build cleanup uses destructive deletion. Given recent process/build instability, cleanup should default to timestamped quarantine/move.
6. The persistent `scripts/audit_local_imports.py` gate is still missing and should be created before any further build attempt.

## 4. Verified Local Facts

### 4.1 `PostProcessingError` usage

Live source confirms:

1. `PostProcessingError` exists in `reconstruction_gui/tabs/source_tab.py`.
2. It is used only inside `_run_post_processing()` for LUT validation failures:
   - LUT enabled but no LUT path.
   - LUT path does not exist.
   - LUT file load failure.
3. No Adjust-specific call sites were seen around those uses.

Oversight conclusion:

Keeping `PostProcessingError` is reasonable as a UX decision, provided the extraction flow catches the exception at the correct level and reports it clearly instead of crashing a worker silently.

Required follow-up:

1. Verify each `_run_post_processing()` caller catches/logs exceptions acceptably.
2. Verify GUI state is restored after a raised `PostProcessingError`.
3. Verify queued extraction does not abort the entire queue unexpectedly unless that is intended.

### 4.2 LUT API state

Live source confirms these methods exist in `prep360/core/lut.py`:

1. `LUTProcessor.load_cube(cube_path: str)`
2. `LUTProcessor.apply_float(image_bgr_float, lut_3d, strength=1.0)`
3. `LUTProcessor.apply_uint8(image_bgr, lut_3d, strength=1.0)`
4. `LUTProcessor.apply_lut_uint8(image_bgr, lut_path, strength=1.0)`
5. `LUTProcessor.apply_lut_float(image_bgr_float, lut_path, strength=1.0)`
6. `LUTProcessor.apply_lut(image_bgr, lut_path, strength=1.0)`
7. `LUTProcessor.apply(image, lut_3d, strength=1.0)`

Contradiction in Claude report:

1. Lines 114-115 say the old `apply_lut` method no longer exists.
2. Line 430 lists `apply_lut` among methods added to `lut.py`.
3. Live source confirms `apply_lut` does exist as a compatibility wrapper.

Oversight conclusion:

The decision to use `apply_uint8` in the post-processing loop remains reasonable because it avoids reloading the LUT per image. However, the report should be corrected: `apply_lut` exists, so "old method doesn't exist" is false in the current live repo.

Required follow-up:

1. Add a small functional LUT test using a repo-local temporary fixture or checked-in minimal `.cube` fixture.
2. Test at least:
   - identity LUT preserves BGR pixels.
   - invert LUT produces expected BGR inversion.
   - `strength=0.0` preserves input.
   - `strength=0.5` blends between input and transformed output.
   - malformed LUT raises a clear error.
3. Test both `apply_uint8` and the compatibility wrapper `apply_lut`.

### 4.3 Nuitka local option availability

Installed Nuitka help confirms:

1. `--nofollow-import-to=MODULE/PACKAGE` exists.
2. `--include-package=PACKAGE` exists.
3. `--user-package-configuration-file=YAML_FILENAME` exists.
4. `--output-filename=FILENAME` exists.
5. Plugins include:
   - `tk-inter`
   - `no-qt`
   - `transformers`
   - `anti-bloat`
6. `--noinclude-custom-mode=PACKAGE:MODE` exists.
7. No `--noinclude-transformers-mode` option was observed in local `--help` or `--help-all` output.

Oversight conclusion:

References to a possible `--noinclude-transformers-mode` option should be treated as speculative unless proven against this exact installed Nuitka version.

### 4.4 Smoke script executable naming

Live `scripts/smoke_import_gui.py` has filename base `smoke_import_gui`.

Nuitka help says standalone output defaults to the base filename of the main module unless `--output-filename` changes it.

Claude report Phase 6A success criteria expects:

`dist_test/smoke_import_gui.dist/SmokeGuiImport.exe`

But the proposed command does not include:

`--output-filename=SmokeGuiImport.exe`

Oversight conclusion:

The success criteria and command are inconsistent. Either:

1. Add `--output-filename=SmokeGuiImport.exe`, or
2. Change expected executable to `dist_test/smoke_import_gui.dist/smoke_import_gui.exe`.

Conservative recommendation:

Add `--output-filename=SmokeGuiImport.exe` if the existing reports and scripts expect that executable name.

## 5. Official Documentation Constraints

Official Nuitka documentation says standalone mode follows imports by default. It also says `--nofollow-import-to` can selectively exclude modules, but excluded imports may raise `ImportError` when attempted at runtime.

Implication for this project:

1. Phase 6A can safely nofollow all ML packages only if the smoke executable never reaches masking imports at runtime.
2. Phase 6B cannot assume `--nofollow-import-to="transformers.models.*"` is safe for RF-DETR runtime.
3. If RF-DETR actually imports `transformers.models.dinov2` or similar at model construction/inference time, the full build may compile faster but fail at runtime.

Official Nuitka unwanted-module guidance also says `--nofollow-import-to` may not be enough in some cases and anti-bloat configuration can be needed.

Implication:

If the Transformers model-tree walk continues despite nofollow exclusions, the next move should be a targeted Nuitka package configuration/anti-bloat rule, not repeated blind builds.

## 6. Phase 6A Review: Constrained GUI Nuitka Smoke

### 6A.1 What is reasonable

1. Running a constrained smoke before full build is the correct ordering.
2. Blanket nofollow of runtime-only ML packages is reasonable for a GUI import smoke if the smoke only imports the shell and prints version.
3. Avoiding `tail` is mandatory.
4. Foreground execution or real-time log streaming is correct.
5. Stall detection for `transformers/models/*/define_import_structure` is correct.

### 6A.2 Required corrections before execution

1. Fix executable name mismatch:
   - Add `--output-filename=SmokeGuiImport.exe`, or update expected path.

2. Make cleanup non-destructive by default:
   - Do not use `Remove-Item -Recurse -Force` as the default.
   - Prefer moving old outputs to `dist_test/archive/<timestamp>/`.
   - If deletion is desired, ask the user explicitly.

3. Add a preflight process check:
   - Confirm no active `git.exe add -A`.
   - Confirm no active `python.exe -m nuitka`.
   - Confirm no stale `.git/index.lock`.

4. Add a preflight import check:
   - Run `scripts/smoke_import_gui.py` with normal Python first.
   - Confirm it prints version and `GUI_IMPORT_OK`.
   - Confirm it does not load `ultralytics`, `transformers`, `rfdetr`, or `sam3`.

5. Keep the smoke purpose narrow:
   - It is an import/analyzer test only.
   - Passing it does not prove masking works in packaged form.

### 6A.3 Conservative revised success criteria

1. Nuitka exits 0.
2. Expected executable exists.
3. Running executable exits 0.
4. Executable output includes:
   - version string.
   - `GUI_IMPORT_OK`.
5. Process tree never enters Transformers model-tree analysis.
6. No new `git add -A` storm is observed.

### 6A.4 Stop conditions

1. Any child process starts processing `transformers/models/*`.
2. CPU remains active but log output has no meaningful phase change for an agreed timeout window.
3. Working set grows without output and without known Nuitka phase progress.
4. Any `git add -A` respawn loop appears.
5. Nuitka emits a fatal error.
6. Build exceeds the agreed hard timeout.

## 7. Phase 6B Review: Full Build

### 7.1 Main risk

The proposed strategy:

`--nofollow-import-to="transformers.models.*"`

is high risk because RF-DETR may need modules under `transformers.models.*` at runtime. Official Nuitka docs warn that a nofollowed import can become an `ImportError` when attempted at program runtime.

### 7.2 Problem with the proposed RF-DETR probe

Claude's proposed probe:

```python
import rfdetr
import sys
used_models = [m for m in sys.modules if m.startswith("transformers.models.")]
print(used_models)
```

This is insufficient because it only captures modules imported during `import rfdetr`.

It does not prove:

1. Which modules load when constructing `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, or `RFDETRLarge`.
2. Which modules load when loading local weights.
3. Which modules load during first inference.
4. Which modules are imported via dynamic import helpers.
5. Which package data/config files are required by Transformers.

### 7.3 Required full-build probe before 6B

Before any full build, run a runtime probe in normal Python that exercises the same RF-DETR path the GUI will use.

Minimum probe:

1. Import `rfdetr`.
2. List available RF-DETR classes.
3. Instantiate the configured v1 class if this can be done without downloading.
4. If local weights are configured, load them.
5. Run a tiny inference on a generated or repo-local test image if feasible.
6. Print all loaded modules matching:
   - `transformers`
   - `transformers.models.*`
   - `rfdetr`
   - `torch`
   - `torchvision`
7. Record whether network access or model download would be required.

If instantiation requires downloads or credentials, stop and document that the probe cannot establish packaging requirements.

### 7.4 Safer alternatives to blanket model-tree nofollow

Preferred options, in order:

1. Use Nuitka anti-bloat/user package configuration to block the specific Transformers model-tree enumeration path that causes the static walk, while preserving modules actually needed by RF-DETR.
2. Exclude only known-unneeded top-level integrations and optional backends first:
   - `bitsandbytes`
   - `triton`
   - optional integrations that are confirmed unused.
3. Build a positive allowlist from a functional RF-DETR runtime probe, then nofollow only confirmed unused model families.
4. As a last resort, nofollow `transformers.models.*` and accept that RF-DETR must be tested immediately after build with a real packaged runtime test.

Do not claim a full build is RF-DETR-capable until packaged RF-DETR model construction and a small inference pass have been tested.

### 7.5 Full build success criteria

A full build is not successful merely because Nuitka exits 0.

Required runtime checks:

1. Packaged GUI launches.
2. Extract tab loads and reports `HAS_PREP360=True` equivalent.
3. No Adjust handoff UI is visible.
4. No Send To Coverage button is visible.
5. Setup wizard behavior is correct.
6. OpenCV imports in packaged executable.
7. Torch imports in packaged executable.
8. At least one non-ML extraction workflow works.
9. If RF-DETR is claimed packaged, RF-DETR model construction and a tiny inference test pass.
10. If YOLO/FastSAM/SAM3 are claimed packaged, each has a separate runtime test or is explicitly marked deferred.

## 8. Phase 6C Review: Post-Build Steps

### 8.1 OpenCV CUDA DLL copy

This step is reasonable only after the target dist directory exists.

Required checks:

1. Confirm source DLL locations exist.
2. Confirm target dist directory is the correct one.
3. Run the copy script with the exact dist path.
4. Re-run packaged OpenCV smoke after copy.
5. Confirm exactly one `cv2.pyd` remains in the bundle unless there is a documented reason for more.

### 8.2 Runtime verification

Runtime verification should be split into tiers:

1. Tier 1: smoke import executable.
2. Tier 2: full GUI shell launch.
3. Tier 3: Extract tab/non-ML workflow.
4. Tier 4: ML model imports.
5. Tier 5: ML model functional inference.

Do not let a Tier 1 pass imply Tier 4 or Tier 5 readiness.

## 9. Required Source Gates Before Any Build

This section is a condensed restatement of Phase 5 from the governing remediation/audit plan. If there is a conflict, follow the governing plan and update this oversight document.

These should run before Phase 6A:

1. Persistent local import audit exists:
   - `scripts/audit_local_imports.py`
2. Persistent audit passes.
3. Phantom source terms absent from live `.py` and `.spec`.
4. `scripts/smoke_import_gui.py` passes under normal Python.
5. `tabs.source_tab` imports with:
   - `HAS_PREP360=True`
   - no `ultralytics`
   - no `transformers`
   - no `rfdetr`
6. `prep360` import does not load heavy ML packages.
7. `prep360.core` import does not load heavy ML packages.
8. No active build/Git process conflicts.
9. `dist_test/` is ignored by Git.

## 10. Persistent `audit_local_imports.py` Requirements

The missing script should:

1. Parse Python files with `ast`, not regex.
2. Build a map of local modules from:
   - `reconstruction_gui`
   - `prep360`
   - `scripts`
   - `tests`
3. Understand top-level aliases created by adding `reconstruction_gui` to `sys.path`, especially:
   - `tabs.*`
   - `widgets`
   - `app_infra`
   - other existing GUI-local imports.
4. Distinguish external dependencies from missing local imports.
5. Emit machine-readable or stable text output.
6. Exit nonzero on unresolved local imports.
7. Support an explicit allowlist for intentional dynamic imports.
8. Be safe to run repeatedly before builds.

## 11. Documentation Oversight

Claude reports current implementation docs are clean and planning docs remain.

Codex should verify before accepting:

1. Search current docs for deleted modules:
   - `adjust_tab.py`
   - `gaps_tab.py`
   - `adjust_workflow.py`
   - `adjustment_recipe.py`
   - `color_pipeline.py`
2. Confirm any remaining matches are in historical incident/audit/planning reports only.
3. Add future-only headers to old planning docs that describe unimplemented features.

Recommended header:

```markdown
> Status: Future plan only. Not implemented in v1. Do not reference these APIs from live source.
```

## 12. Corrections Required In Claude's Updated Report

Claude should update the report or a follow-up note with these corrections:

1. `apply_lut` exists in live `prep360/core/lut.py`; the statement that the old method no longer exists is false.
2. The `apply_uint8` decision should be justified as a performance/API choice, not as forced by absence of `apply_lut`.
3. Phase 6A command and success criteria disagree on executable name.
4. `--noinclude-transformers-mode` was not observed in local Nuitka help and should be marked speculative unless proven.
5. Phase 6B `transformers.models.*` nofollow should be marked high risk, not a settled strategy.
6. RF-DETR import-only probe should be expanded to model construction and ideally tiny inference.
7. Pre-build cleanup should avoid destructive deletion by default.

## 13. Recommended Next Sequence

This sequence preserves the implementation order in the governing remediation/audit plan: source gates first, Phase 7 audit completion second, constrained smoke only after those gates are green.

### Step 1: Stabilize

1. Verify no Git/Nuitka/Python build processes are running.
2. Verify no `.git/index.lock`.
3. Confirm `dist_test/` remains ignored.

### Step 2: Finish source gates

1. Create `scripts/audit_local_imports.py`.
2. Run it.
3. Run phantom term search.
4. Run normal Python smoke import.
5. Run import side-effect probes.

### Step 3: Patch report inaccuracies

1. Correct `apply_lut` statement.
2. Correct Phase 6A executable name.
3. Reclassify Phase 6B as high risk until RF-DETR runtime probe proves otherwise.

### Step 4: LUT functional test

1. Add a focused LUT test or one-off script.
2. Verify identity/invert/strength behavior.
3. Verify malformed LUT error handling.

### Step 5: Prepare Phase 6A only

1. Build command should include `--output-filename=SmokeGuiImport.exe` if that name is expected.
2. Old build outputs should be moved aside, not deleted, unless the user approves deletion.
3. Build output should stream visibly or write to a full timestamped log.
4. Process monitoring should use PowerShell process inspection.
5. Stop immediately on Transformers model-tree analysis.

### Step 6: Defer Phase 6B until proof exists

1. Run expanded RF-DETR runtime probe.
2. Decide whether full build can exclude any `transformers.models.*` safely.
3. Prefer targeted anti-bloat rules over blanket model-family nofollow if runtime requirements are unclear.

## 14. Commands Allowed Without Additional User Approval

Read-only/source-audit commands:

1. `rg` searches.
2. `Get-Content` reads.
3. `git --no-optional-locks status --short --branch`.
4. `git --no-optional-locks diff --name-status`.
5. `git check-ignore -v` for specific paths.
6. Python import probes that do not download, build, stage, or write outside the repo.

Commands requiring explicit user approval:

1. Nuitka builds.
2. Long-running model import/instantiation probes if they may download weights.
3. Any destructive cleanup.
4. Killing processes.
5. Writing outside `D:\Projects\reconstruction-zone`.
6. Staging, committing, pushing, cleaning, resetting.

## 15. Final Oversight Position

Phase 6A is acceptable after correcting the executable naming and cleanup strategy.

Phase 6B is not yet acceptable as written. It must be treated as experimental until RF-DETR runtime module requirements are proven. The official Nuitka documentation makes the risk explicit: nofollowed imports can produce runtime `ImportError` when reached.

The highest-value immediate work is not another build. It is:

1. Create the persistent import audit script.
2. Correct the report inaccuracies.
3. Add a small LUT functional test.
4. Run source gates.
5. Only then let Claude run the constrained 6A smoke with visible monitoring.
