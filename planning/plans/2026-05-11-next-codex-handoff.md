# Next Codex Conversation Handoff

Date: 2026-05-11
Project: Reconstruction Zone
Live repo/source of truth: `D:\Projects\reconstruction-zone`
Branch: `release/v1`
User state: frustrated by wasted build time and sandbox/worktree mismatch. Be direct, evidence-based, and do not run speculative builds.

## Current Phase 6A State

The Phase 6A retry artifact has been reconciled:

- The retry build compiled successfully.
- The preserved executable exits `0` from both the repository root and its
  `.dist` directory.
- The previously reported `_version` failure was carried forward without
  artifact-specific reverification and should not be treated as the retry result.
- `--include-package=reconstruction_gui` is currently supported by the preserved
  retry outcome.
- The broad import refactor is deferred because the preserved artifact does not
  justify it.

## Critical Access Instruction

The user chose `D:\Projects\reconstruction-zone` as the actual workspace. Treat that path as the only source of truth.

If the next Codex session reports writable roots under a `.codex\worktrees\...` path instead of `D:\Projects\reconstruction-zone`, stop and tell the user immediately. Do not silently work in the Codex worktree and then copy files back. That mismatch caused major friction in the prior conversation.

Required behavior:

- Read from `D:\Projects\reconstruction-zone`.
- Edit `D:\Projects\reconstruction-zone` directly.
- Run git commands against `D:\Projects\reconstruction-zone`.
- Do not rely on `C:\Users\alexm\.codex\worktrees\...` as project state.
- If sandbox approval is needed to write to `D:\Projects\reconstruction-zone`, ask once and explain that the sandbox is misconfigured relative to the intended workspace.

## Division Of Labor

The user explicitly assigned roles:

- Codex: diagnostics, planning, verification, audits, clear handoff instructions.
- Claude: source implementation and Nuitka build attempts.

Do not start another long Nuitka build unless the user explicitly asks Codex to take that over. The user has been very clear that speculative build attempts wasted time.

## Relevant Documents Already Created

Read these first, in this order:

1. `planning/plans/2026-05-11-phantom-code-incident-report.md`
   - Claude self-audit of the phantom-code incident from commit `2741e55`.
   - Main finding: a v1 trim commit referenced planned/deleted Adjust APIs and left phantom wiring.

2. `planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md`
   - Codex-created comprehensive cleanup and full-scale audit plan.
   - Defines Phases 0-8, source gates, audit categories A-J, and definition of done.

3. `planning/plans/2026-05-11-remediation-progress-report.md`
   - Claude progress report after implementing the main remediation.
   - Claims Phases 0-5 complete and all 8 source gates pass.

4. `planning/plans/2026-05-11-phase-6a-retry-result.md`
   - Canonical Phase 6A retry report after artifact reconciliation.
   - Records the corrected exit-0 runtime smoke result and withdrawn `_version`
     attribution.

5. `planning/plans/2026-05-11-nuitka-build-diagnostics-handoff-report.md`
   - Codex report summarizing failed build attempts, OpenCV smoke findings, ffmpeg correction, model_downloader neutralization, and the import-chain diagnosis.

6. `planning/plans/2026-05-11-build-issues-remediation-instructions-report.md`
   - Earlier reviewed report about OpenCV/Nuitka issues.

## Incident Summary

Commit under suspicion:

```text
2741e55 release(v1): trim experimental features for v1 release
```

It was supposed to remove experimental features for v1, but it also introduced references to planned/deleted APIs. The most damaging phantom areas were:

- Adjust handoff code in `reconstruction_gui/tabs/source_tab.py`.
- Phantom imports of `AdjustmentRecipe` and `adjust_workflow`.
- Queue fields for nonexistent Adjust recipe behavior.
- Alignment "Send To Coverage" wiring to deleted `gaps_tab.py`.
- Eager `prep360.core` segmenter export causing `ultralytics` to load during GUI/Source import.

Consequences:

- `HAS_PREP360` was false during Source tab import.
- `ultralytics` loaded during Source tab import.
- Nuitka GUI smoke wandered into Transformers plugin analysis and stalled.
- Multiple build attempts wasted time.

## What Claude Says Is Done

From `2026-05-11-remediation-progress-report.md`, Claude says:

- Phases 0-5 are complete.
- All phantom Adjust code removed from `source_tab.py`.
- Adjust fields removed from `prep360/core/queue_manager.py`.
- Coverage/Gaps wiring removed from `reconstruction_gui/tabs/alignment_tab.py`.
- `Segmenter`, `SegmentConfig`, `COCO_CLASSES`, and `CLASS_PRESETS` made lazy in `prep360/core/__init__.py` and `prep360/__init__.py`.
- Known phantom Adjust/Coverage terms no longer appear in live `.py`/`.spec` files.
- `HAS_PREP360=True`.
- Source tab import no longer loads `ultralytics`, `transformers`, or `rfdetr`.
- `prep360` and `prep360.core` import without loading those packages.
- Focused tests: `18 passed in 0.17s` per Claude.
- GUI launch reportedly clean, but Claude did not visually inspect the UI.

## What Codex Independently Verified In This Thread

These checks were run against `D:\Projects\reconstruction-zone` after Claude's update.

### Phase 6A retry reconciliation

Codex reran the preserved executable from:

- `D:\Projects\reconstruction-zone`
- `D:\Projects\reconstruction-zone\dist_test\smoke_import_gui.dist`

Observed result in both cases:

```text
Exit code: 0
Traceback: none
```

The missing visible `GUI_IMPORT_OK` marker is explained by stdout/stderr
suppression in the packaged runtime path, not by a failed import smoke.

### Known phantom terms gone from live Python/spec

Command:

```powershell
rg -n "AdjustmentRecipe|adjust_workflow|adjust_tab|color_pipeline|adjustment_recipe|open_adjust_after_extraction|adjust_recipe_enabled|_validate_adjust_recipe|_export_adjusted_derivative|_offer_open_in_adjust|gaps_tab|Send To Coverage|send_to_coverage" 'D:\Projects\reconstruction-zone' -g "*.py" -g "*.spec"
```

Result: no matches.

### `PostProcessingError` current state

Command:

```powershell
rg -n "class PostProcessingError|PostProcessingError|_run_post_processing_safely|Apply LUT|Color & LUT|legacy quick|prefer Adjust" 'D:\Projects\reconstruction-zone\reconstruction_gui\tabs\source_tab.py'
```

Result:

```text
955:    # -- Color & LUT (collapsible) --
956:    lut_sec = CollapsibleSection(pp, "Color & LUT",
967:    ctk.CTkCheckBox(lut_sec.content, text="Apply LUT after extraction",
1376:class PostProcessingError(RuntimeError):
1436:            raise PostProcessingError("LUT post-processing is enabled but no LUT path is set")
1438:            raise PostProcessingError(f"LUT file not found: {settings.lut_path}")
1445:            raise PostProcessingError(f"Could not load LUT '{settings.lut_path}': {e}") from e
```

Interpretation: `PostProcessingError` remains, but only for real LUT validation. It is not currently Adjust-coupled. The next assistant should review whether this stricter LUT behavior is desirable, because it was introduced in the suspect commit.

### `prep360` import hygiene

Commands:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\Python314\python.exe -c "import sys; sys.path.insert(0,'D:/Projects/reconstruction-zone'); import prep360; heavy=('ultralytics','transformers','rfdetr'); print('prep360', {name:any(m==name or m.startswith(name+'.') for m in sys.modules) for name in heavy})"
```

Result:

```text
prep360 {'ultralytics': False, 'transformers': False, 'rfdetr': False}
```

Command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\Python314\python.exe -c "import sys; sys.path.insert(0,'D:/Projects/reconstruction-zone'); import prep360.core; heavy=('ultralytics','transformers','rfdetr'); print('prep360.core', {name:any(m==name or m.startswith(name+'.') for m in sys.modules) for name in heavy})"
```

Result:

```text
prep360.core {'ultralytics': False, 'transformers': False, 'rfdetr': False}
```

### Source tab import hygiene

Command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\Python314\python.exe -c "import sys; sys.path[:0]=['D:/Projects/reconstruction-zone','D:/Projects/reconstruction-zone/reconstruction_gui']; import tabs.source_tab as st; blocked=('ultralytics','transformers','rfdetr'); loaded=[name for name in blocked if any(m==name or m.startswith(name+'.') for m in sys.modules)]; print('HAS_PREP360', st.HAS_PREP360); print('LOADED', loaded); raise SystemExit(1 if loaded or not st.HAS_PREP360 else 0)"
```

Result:

```text
HAS_PREP360 True
LOADED []
```

### Local missing import audit

Codex ran an inline AST local-import audit after Claude's cleanup.

Result:

```text
LOCAL_IMPORT_AUDIT_OK
```

Update: `scripts/audit_local_imports.py` now exists and was rerun successfully.
Codex strengthened it to cover plain `import local.module` statements and
relative import resolution in addition to the original `from ... import ...`
handling.

### GUI shell import gate caveat

Codex attempted the GUI shell import gate without escalation:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\Python314\python.exe -c "import sys; sys.path[:0]=['D:/Projects/reconstruction-zone','D:/Projects/reconstruction-zone/reconstruction_gui']; import reconstruction_gui.reconstruction_zone as rz; blocked=('ultralytics','transformers','rfdetr'); loaded=[name for name in blocked if any(m==name or m.startswith(name+'.') for m in sys.modules)]; print(rz.__version__); print('LOADED', loaded); raise SystemExit(1 if loaded else 0)"
```

It failed before reaching the import-heavy check because sandbox permissions blocked writing:

```text
PermissionError: [Errno 13] Permission denied: 'C:\Users\alexm\.reconstruction_zone\crash.log'
```

This was a sandbox artifact, not evidence of an import-chain failure. The user rejected the escalated rerun. Therefore, Codex did not independently verify the GUI shell gate after Claude's cleanup, but Claude claims it passed.

### `py_compile` caveat

Codex attempted:

```powershell
C:\Python314\python.exe -m py_compile ...
```

It failed because sandbox permissions blocked writing pyc files under:

```text
D:\Projects\reconstruction-zone\reconstruction_gui\tabs\__pycache__\...
```

Again, this was a sandbox artifact. Claude claims `py_compile` passed.

## Current Git State Observed By Codex

Command:

```powershell
git -C 'D:\Projects\reconstruction-zone' status --short --branch
```

Observed state after Claude cleanup:

```text
## release/v1...origin/release/v1 [ahead 2]
 M .gitignore
 M prep360/__init__.py
 M prep360/core/__init__.py
 M prep360/core/analyzer.py
 M prep360/core/extractor.py
 M prep360/core/geotagger.py
 M prep360/core/osv.py
 M prep360/core/queue_manager.py
 M prep360/core/sharpest_extractor.py
 M reconstruction_gui/colmap_runner.py
 M reconstruction_gui/model_downloader.py
 M reconstruction_gui/reconstruction_pipeline.py
 M reconstruction_gui/reconstruction_zone.py
 M reconstruction_gui/tabs/alignment_tab.py
 M reconstruction_gui/tabs/source_tab.py
 M reconstruction_zone.spec
 M scripts/pyinstaller_runtime_hook.py
?? dist_test/
?? planning/plans/2026-05-11-build-issues-and-fixes.md
?? planning/plans/2026-05-11-build-issues-remediation-instructions-report.md
?? planning/plans/2026-05-11-build-issues-remediation-instructions.md
?? planning/plans/2026-05-11-nuitka-build-diagnostics-handoff-report.md
?? planning/plans/2026-05-11-phantom-code-incident-report.md
?? planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md
?? planning/plans/2026-05-11-remediation-progress-report.md
?? planning/plans/2026-05-11-setup-wizard-plan.md
?? planning/plans/exe-packaging-checklist.md
?? planning/plans/exe-packaging-plan.md
?? planning/sam3-gated-model-reference.md
?? planning/sam3_gated_setup.py
?? prep360/core/subprocess_utils.py
?? reconstruction_gui/model_paths.py
?? reconstruction_gui/sam3_setup.py
?? reconstruction_gui/setup_wizard.py
?? reconstruction_gui/subprocess_env.py
?? scripts/copy_cv2_cuda_dlls.ps1
?? scripts/nuitka-torch-fix.yml
?? scripts/smoke_import_cv2.py
?? scripts/smoke_import_gui.py
?? scripts/smoke_import_torch.py
?? scripts/smoke_model_paths.py
```

Warnings about `C:\Users\alexm/.config/git/ignore` and `.pytest_cache` permissions appeared; do not treat those as source failures.

## Open Issues / Things The Next Conversation Should Do

### 1. Do not rerun Nuitka with `| tail -5`

Claude tried a constrained GUI smoke with:

```text
... scripts/smoke_import_gui.py 2>&1 | tail -5
```

That was wrong because `tail -5` buffers until EOF, making progress invisible.

If Claude runs the next build, recommended monitoring pattern:

- Do not pipe to `tail`.
- Send full output to a timestamped log file with unbuffered/tee style logging if needed.
- Ensure old `dist_test/smoke_import_gui.build` and `.dist` are moved aside or clearly timestamped before retrying.
- Check process tree if output stalls.
- Stop immediately if the process returns to Transformers model-tree analysis.

Codex should not run the build unless the user explicitly redirects that work to Codex.

### 2. Keep `scripts/audit_local_imports.py` as an explicit gate

The reusable audit script now exists and currently reports:

```text
No unresolved local imports found.
```

Use it as a repeatable pre-build check. Do not let this slip back into a
one-off inline audit.

### 3. Audit remaining `2741e55` changes

Claude cleaned only the files named by the first remediation plan. The suspect commit changed 36 files. Remaining unaudited/high-risk files include:

- `reconstruction_gui/reconstruction_zone.py` with large net additions.
- `reconstruction_gui/colmap_runner.py`.
- `reconstruction_gui/review_gui.py`.
- `reconstruction_gui/review_masks.py`.
- `reconstruction_gui/review_status.py`.
- `prep360/core/lut.py`.
- `prep360/core/sharpest_extractor.py`.
- `prep360/core/paired_split_video_extractor.py`.
- Documentation files that may still describe removed features as current.

Priority audit methods:

- Search for references to deleted modules/features in live source.
- Compare `2741e55^..2741e55` hunks in each file and classify additions as real, deleted-feature residue, or suspicious.
- Run import side-effect matrix from the audit plan.
- Update docs that still describe Adjust/Coverage as current.

### 4. Documentation still references removed features

Codex found markdown references after Claude cleanup:

- `AGENTS.md` still lists `gaps_tab.py` as Coverage tab.
- `CLAUDE.md` still lists `gaps_tab.py` as Coverage tab.
- `HANDOVER.md` mentions deleted `gaps_tab.py` and `adjust_tab.py` paths.
- `reconstruction_gui/docs/EXTRACT_TAB.md` still references Open in Adjust / Adjust workflow as current.
- `reconstruction_gui/docs/ARCHITECTURE.md` still lists `tabs/gaps_tab.py`.

These are docs, not live source, but they will mislead future agents. The next conversation should decide whether to clean docs now or after source/build stabilization. At minimum, mark future-only plans clearly.

### 5. Review `PostProcessingError` and LUT behavior

`PostProcessingError` is no longer phantom-coupled, but it came from the suspect commit.

Need to decide:

- Keep strict LUT validation, or revert to older silent-skip behavior.
- Verify `LUTProcessor.apply_uint8()` exists and has the signature used by `_run_post_processing()`.
- Verify `prep360/core/lut.py` additions from `2741e55` are real and correct.

### 6. `cv2` and `torch` may still import during Source tab import

Before Claude's cleanup, Source tab import loaded `cv2` and `torch`. Claude deferred reducing that because the critical gate was no `ultralytics`/`transformers`/`rfdetr`.

The next conversation should check current state:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\Python314\python.exe -c "import sys; sys.path[:0]=['D:/Projects/reconstruction-zone','D:/Projects/reconstruction-zone/reconstruction_gui']; import tabs.source_tab as st; heavy=('ultralytics','transformers','rfdetr','torch','cv2','pycolmap','open3d'); print({name:any(m==name or m.startswith(name+'.') for m in sys.modules) for name in heavy})"
```

If `torch`/`cv2` still load, decide whether acceptable for packaging. It may be okay if Nuitka GUI smoke finishes, but if builds remain slow, trace these imports next.

### 7. Before any future build, confirm no Python/Nuitka process is still active

The earlier `tail -5` monitoring mistake is historical, but the precaution
remains sound: check for live Python/Nuitka processes before starting any new
Phase 6B or later build work.

Suggested command:

```powershell
Get-Process python,nuitka -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path
```

If access denied for process details, ask the user before escalating.

## OpenCV / ffmpeg / model downloader Context

These were already addressed before the phantom cleanup:

### OpenCV smoke

- Nuitka standalone cv2 smoke compiled.
- First run failed with missing native DLLs, not duplicate `cv2.pyd`.
- Bundle had exactly one `cv2.pyd`.
- Copying CUDA/NVIDIA DLLs into bundle `cv2` directory made the smoke pass.
- Helper script exists: `scripts/copy_cv2_cuda_dlls.ps1`.

### ffmpeg

Bad include:

```text
--include-data-dir=ffmpeg=ffmpeg
```

Correct includes:

```text
--include-data-files="C:/Users/alexm/ffmpeg/bin/ffmpeg.exe=ffmpeg/ffmpeg.exe"
--include-data-files="C:/Users/alexm/ffmpeg/bin/ffprobe.exe=ffmpeg/ffprobe.exe"
```

### model_downloader

`reconstruction_gui/model_downloader.py` was replaced by a compatibility stub that imports no heavy model libraries. `setup_wizard.py` and `model_paths.py` are now the active model onboarding path.

## Recommended Immediate Next Steps

1. Confirm the next Codex session has direct write access to `D:\Projects\reconstruction-zone`. If not, stop and resolve workspace configuration.
2. Read the three main reports listed above.
3. Verify no Python/Nuitka build processes are running.
4. Re-run `scripts/audit_local_imports.py` as part of the pre-build gate set.
5. Run Audit A-J from `2026-05-11-phantom-code-remediation-and-audit-plan.md`, starting with docs/spec/source references to deleted features.
6. Review `PostProcessingError` and `prep360/core/lut.py` additions for correctness.
7. Read `planning/plans/2026-05-11-phase-6b-strategy-oversight-report.md`
   before launching any Phase 6B build attempt.
8. Read and follow
   `planning/plans/2026-05-11-phase-6b-rfdetr-packaging-smoke-plan.md`.
   The source smoke target already exists at
   `scripts/smoke_import_rfdetr.py` and currently emits `RFDETR_IMPORT_OK`.
9. Read
   `planning/plans/2026-05-12-phase-6b-rfdetr-packaging-smoke-result.md`
   before attempting another RF-DETR packaging run. It records that:
   - broad `--include-package=transformers` control is rejected;
   - the narrower static-follow control passes when `NUITKA_CACHE_DIR` is
     redirected into `dist_test\nuitka-cache`;
   - the blanket `transformers.models.*` exclusion trial has not yet been run and
     is now optional/optimization-oriented rather than required for the RF-DETR
     import smoke.
   It also records that a 60-second direct foreground diagnostic does produce
   live Nuitka telemetry, so the next A2 attempt should avoid the earlier silent
   wrapper path.
   After the user-interrupted authoritative A2 run, Codex found and stopped the
   leftover build-era Python processes. Future interrupted/timeout builds must
   get the same immediate process check.
10. Treat Phase 6A as passed. Any next packaging work should proceed from the
   Phase 6B / full-build strategy in the updated progress report, with proper
   logging and without resurrecting the withdrawn `_version` retry-failure claim.

## Phase 6B Full GUI Build Update, 2026-05-12

Codex ran the approved full GUI Nuitka build with:

```text
NUITKA_CACHE_DIR=D:\Projects\reconstruction-zone\dist_test\nuitka-cache
no --include-package=transformers
no --nofollow-import-to=transformers.models.*
scripts\nuitka-torch-fix.yml retained
```

Compile/package result:

```text
PASS
report completion: yes
compiled/linked C files: 9492
exe:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-workspacecache-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Post-build:

```text
scripts\copy_cv2_cuda_dlls.ps1 copied 16 OpenCV CUDA/NVIDIA DLLs into the bundle.
```

Runtime result:

- Sandbox execution fails at app import because the sandbox blocks writes to
  `C:\Users\alexm\.reconstruction_zone\crash.log`.
- Outside the sandbox, repo-root launch stayed alive for 20 seconds and logged
  `All models ready — setup wizard not needed`.
- Outside the sandbox, `.dist` launch without model override hit the
  first-launch/missing-model branch and wrote a fatal Tk callback traceback:
  `TypeError: 'ReconstructionZone' object is not callable`.
- Outside the sandbox, `.dist` launch with
  `RECONSTRUCTION_ZONE_MODEL_DIR=D:\Projects\reconstruction-zone` stayed alive
  for 20 seconds and logged `All models ready — setup wizard not needed`.

Interpretation:

- The compile strategy is now proven for a full GUI build.
- The artifact is usable for local model-satisfied smoke testing.
- It is **not** releasable as a clean first-launch artifact until the packaged
  setup wizard / missing-model path is fixed.

Next best move:

1. Do not run another blind full GUI build.
2. Create a small CustomTkinter/Nuitka setup-wizard smoke that exercises
   `SetupWizard(parent, models)`, `transient(parent)`, `grab_set()`, and
   `focus()` with synthetic missing models.
3. Fix the compiled Tk callback failure or adjust model provisioning so
   first-launch does not enter a broken wizard path.
4. After the bounded fix, rerun only the missing-model `.dist` launch smoke.

Detailed report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-build-result.md
```

## Setup Wizard Fix Update, 2026-05-12

Codex identified the first-launch setup-wizard crash root cause.

Cause:

```text
SetupWizard assigned self._root = parent
```

That shadowed Tkinter's internal `_root()` method on the `CTkToplevel`
instance. Tkinter later attempted to call `_root()` during event handling and
raised:

```text
TypeError: 'ReconstructionZone' object is not callable
```

Fix:

```text
self._root -> self._app_root
```

Additional path cleanup:

- added `reconstruction_gui/app_paths.py`;
- moved logs/prefs/model defaults toward a redirectable app-home;
- added setup-wizard source and packaged smokes;
- added packaged new-install probe harness.

Validated:

- source direct setup-wizard smoke: pass;
- source `run_setup_wizard_if_needed` smoke: pass;
- packaged direct setup-wizard smoke: pass;
- packaged `run_setup_wizard_if_needed` smoke: pass.

The full GUI executable from
`phase-6b-full-gui-workspacecache-20260512` was built before this fix and should
not be used to judge the fixed first-launch path.

Next step:

```text
Rebuild full GUI with the fixed source, then run the isolated new-install probe.
```

Detailed result:

```text
planning/plans/2026-05-12-new-install-setup-wizard-validation-result.md
```

### Full GUI Rebuild After Setup Fix

Codex completed the rebuild after the setup-wizard fix.

Artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-newinstallfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Result:

```text
build exit code: 0
report completion: yes
OpenCV CUDA/NVIDIA DLL copy: pass
```

Isolated new-install probes:

- empty app-home: process stayed alive for 20 seconds, logged all three models
  need setup, no fatal crash log;
- RF-DETR/YOLO fixture app-home: process stayed alive for 20 seconds, logged
  only SAM3 needs setup, no fatal crash log.

Detailed rebuilt-artifact report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-newinstallfix-result.md
```

## Real Model Workflow Update, 2026-05-12

Codex validated the real model installation/download workflows in isolated app
homes.

Passes:

- RF-DETR-Seg real download;
- YOLO26 real download through HuggingFace Hub;
- SAM3 no-token state;
- SAM3 saved-token verification, with no token value printed;
- SAM3 gated access;
- SAM3 real weight download;
- packaged GUI launch against all real model assets, logging:

```text
All models ready — setup wizard not needed
```

Remaining release work:

- visual setup-wizard inspection: initial missing-model screen passed in the
  packaged app;
- manual click-through of wizard screens;
- final full rebuild to include source changes made after the current rebuilt
  artifact.

Detailed report:

```text
planning/plans/2026-05-12-real-model-install-workflow-result.md
```

## Visual Vacuum / Project Store Update, 2026-05-12

Codex launched the rebuilt packaged app visibly with an isolated empty app-home,
empty model directory, and isolated HuggingFace cache.

Result:

```text
setup wizard visible
missing models shown: SAM3, RF-DETR-Seg, YOLO26-n-seg
readiness: 0/3 ready
fatal traceback: none
```

Evidence:

```text
D:\Projects\reconstruction-zone\dist_test\new-install-sandbox\visual-empty-vacuum-setup-wizard-20260512\screenshots\desktop-after-warmup.png
```

Important finding:

```text
The Projects tab behind the wizard still read D:\tracker.json.
```

This means the current packaged artifact passed model/setup-wizard vacuum
display, but failed complete blank app-state isolation.

Source fix applied after the artifact:

- `reconstruction_gui\app_paths.py` now exposes `project_store_file()` and
  `activity_store_file()`;
- `reconstruction_gui\tabs\projects_tab.py` defaults to app-home project
  storage unless an explicit `tracker_store_path` preference exists;
- `reconstruction_gui\project_store.py` defaults `ProjectStore()` to app-home
  project storage.

Verified in source:

```text
project_store_file(): isolated app-home\projects\tracker.json
activity_store_file(): isolated app-home\projects\activity_log.json
ProjectStore().store_path: isolated app-home\projects\tracker.json
py_compile: pass
```

Next best move:

```text
Run one final full GUI rebuild, then rerun the visible vacuum setup-wizard probe
against the new artifact.
```

## Final-Vacuum Artifact Update, 2026-05-12

Codex ran the final full GUI rebuild after the project-store vacuum fix.

Artifact:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Build result:

```text
exit code: 0
report completion: yes
compiled C files: 9494
clcache hits/misses: 9486 / 8
OpenCV CUDA/NVIDIA DLL copy: pass
```

Final visible blank-vacuum probe:

```text
case:
visual-empty-vacuum-finalvacuum-20260512

setup wizard visible: pass
missing models: SAM3, RF-DETR-Seg, YOLO26-n-seg
ready count: 0/3
Projects tab: 0 projects
D:\tracker.json leakage: none found
fatal traceback: none
```

Final real-model sandbox probe:

```text
All models ready — setup wizard not needed
```

Detailed report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-finalvacuum-result.md
```

Remaining work:

```text
manual click-through of Begin Setup screens
normal-tab visual sweep after dismissing/bypassing setup
```

## Mask Preview Runtime Packaging Finding, 2026-05-12

Manual testing found a Mask tab Preview failure in the final-vacuum artifact:

```text
FileNotFoundError:
reconstruction_zone.dist\ultralytics\cfg\default.yaml
```

Cause:

```text
Ultralytics runtime config data was not included in the Nuitka bundle.
```

Immediate fix applied to the current final-vacuum artifact:

```text
copied ultralytics\cfg from the installed Python package into:
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-finalvacuum-20260512\reconstruction_zone.dist\ultralytics\cfg

files copied: 107
default.yaml present: yes
```

Repeatable helper added:

```text
scripts\copy_ultralytics_runtime_data.ps1
```

Final-vacuum build runner updated to include:

```text
--include-data-dir="$UltralyticsCfg=ultralytics/cfg"
```

Next test:

```text
Relaunch the patched final-vacuum artifact and retry Mask tab Preview.
```

## Mask Preview Fix Rebuild, 2026-05-13

The final-vacuum artifact is no longer the preferred test target for Mask
Preview. A fresh full-GUI rebuild was completed with the Mask Preview import
fixes and runtime data inclusion compiled into the artifact.

Use this executable for the next manual test:

```text
D:\Projects\reconstruction-zone\dist_test\phase-6b-full-gui-maskpreviewfix-20260512\reconstruction_zone.dist\ReconstructionZone.exe
```

Build result:

```text
exit code: 0
report completion: yes
compiled C files: 9494
clcache hits/misses: 9490 / 4
OpenCV CUDA/NVIDIA DLL copy: pass
Ultralytics cfg runtime data: pass
source Mask pipeline import smoke: pass
packaged blank new-install launch smoke: pass
```

Detailed report:

```text
planning/plans/2026-05-12-phase-6b-full-gui-maskpreviewfix-result.md
```

Remaining work:

```text
manual Mask tab Preview against the new maskpreviewfix artifact
actual Mask operation if Preview succeeds
```

## User Communication Guidance

The user is upset for good reasons. Avoid vague reassurance. Use concrete status:

- What was checked.
- What passed.
- What failed.
- What remains unaudited.
- Whether a build is safe to run.

Do not say "running ok?" or imply a build is progressing unless process/output evidence supports it.

Do not ask the user to trust another speculative attempt. The pattern now is: prove source gates first, then run bounded build attempts with visible logs.
