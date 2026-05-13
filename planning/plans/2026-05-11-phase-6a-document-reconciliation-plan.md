# Phase 6A Documentation Reconciliation Plan

Date: 2026-05-11  
Project: Reconstruction Zone  
Scope: Phase 6A / remediation documentation only  
Governing context: `planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md`

## Purpose

This plan defines a narrow documentation reconciliation pass after the Phase 6A
retry artifact was rechecked and the prior `_version` runtime-failure claim was
withdrawn.

The goal is to make sure the documents that actively steer the remediation/build
thread agree on the current truth:

- Phase 6A compile succeeded.
- The preserved Phase 6A retry executable exits `0` from both the repository
  root and its `.dist` directory.
- The previously reported `_version` failure does not belong to the preserved
  retry artifact and must not continue to guide next-step decisions.
- `--include-package=reconstruction_gui` is currently supported by the observed
  retry result.
- A broad GUI import refactor is deferred because the preserved artifact does not
  justify it.
- Missing visible `GUI_IMPORT_OK` text is explained by stdout suppression in the
  packaged runtime path, not by a failed import smoke.

This is a documentation alignment task only. It is not a source refactor, build
retry, packaging redesign, or repository-wide markdown cleanup.

## Explicit Scope Boundary

Only documents directly involved in the active Phase 6A/remediation thread are in
scope.

### Primary documents to reconcile

1. `planning/plans/2026-05-11-phase-6a-retry-result.md`
2. `planning/plans/2026-05-11-remediation-progress-report.md`
3. `planning/plans/2026-05-11-next-codex-handoff.md`
4. `planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md`

### Secondary documents to annotate only if useful

These may receive a concise supersession or correction note if they could mislead
someone reading the Phase 6A history in isolation:

1. `planning/plans/2026-05-11-phase-6a-build-report.md`
2. `planning/plans/2026-05-11-phase-6a-oversight-follow-up-report.md`
3. `planning/plans/2026-05-11-phase-6a-retry-oversight-response.md`
4. `planning/plans/2026-05-11-phase-6a-retry-reconciliation-checklist.md`

### Out of scope

- Any unrelated file in `planning/plans/`.
- Broader roadmap, product, architecture, or unrelated audit documents.
- Source code changes.
- Build attempts.
- New diagnosis work unless the documents themselves contain a contradiction that
  cannot be resolved from the already-verified Phase 6A evidence.

## Current Truth Set To Preserve

Every reconciled document should align with the following statements unless it
is explicitly historical and labeled as such.

### Confirmed Phase 6A status

- The Phase 6A retry compile completed successfully.
- The preserved retry executable:
  - exits `0` when launched from the repository root;
  - exits `0` when launched from its `.dist` directory;
  - does not reproduce the earlier reported `_version` traceback.
- Therefore, the preserved retry artifact supports classifying Phase 6A as:
  - **compile smoke: passed**
  - **compiled import/runtime smoke: passed at the preserved artifact level**

### Corrected interpretation

- The `_version` failure previously associated with the retry artifact was
  carried forward without artifact-specific reverification.
- That failure must be treated as a stale or misattributed prior observation, not
  as the current Phase 6A retry result.
- The evidence does not currently justify a broad import refactor of
  `reconstruction_zone.py`.

### Packaging implication

- `--include-package=reconstruction_gui` is supported by the preserved retry
  outcome and should remain recorded as the relevant Phase 6A packaging lesson.
- Any future refactor or build strategy change must be justified by new evidence,
  not by the withdrawn `_version` interpretation.

### Output interpretation

- Absence of visible `GUI_IMPORT_OK` text is not itself a failure signal because
  the packaged runtime path suppresses stdout/stderr under the observed launch
  conditions.
- Exit status and traceback presence are the reliable smoke indicators for this
  specific retry artifact.

## Reconciliation Method

### Step 1: Freeze the exact reconciliation target

Before editing, record the current truth set above as the standard. Do not expand
the task into a broader Phase 6 review.

### Step 2: Read only the scoped document set

Review the four primary documents in full. Review secondary documents only when
they are likely to mislead future readers about the corrected Phase 6A result.

### Step 3: Search only for stale Phase 6A claims

Search within the scoped document set for terms that could preserve the withdrawn
interpretation or imply an obsolete next step:

```powershell
rg -n "_version|ModuleNotFoundError|Phase 6A failed|runtime failure|retry failure|broad import refactor|include-package=reconstruction_gui|GUI_IMPORT_OK" `
  D:\Projects\reconstruction-zone\planning\plans\2026-05-11-phase-6a-retry-result.md `
  D:\Projects\reconstruction-zone\planning\plans\2026-05-11-remediation-progress-report.md `
  D:\Projects\reconstruction-zone\planning\plans\2026-05-11-next-codex-handoff.md `
  D:\Projects\reconstruction-zone\planning\plans\2026-05-11-phantom-code-remediation-and-audit-plan.md `
  D:\Projects\reconstruction-zone\planning\plans\2026-05-11-phase-6a-build-report.md `
  D:\Projects\reconstruction-zone\planning\plans\2026-05-11-phase-6a-oversight-follow-up-report.md `
  D:\Projects\reconstruction-zone\planning\plans\2026-05-11-phase-6a-retry-oversight-response.md `
  D:\Projects\reconstruction-zone\planning\plans\2026-05-11-phase-6a-retry-reconciliation-checklist.md
```

This search is intentionally constrained. Do not widen it to all planning files
unless a later, explicit request requires that.

### Step 4: Classify each scoped document

For each document, assign exactly one of these roles:

1. **Canonical current-state document**
   - Must state the corrected Phase 6A truth plainly.
   - Must not preserve stale failure language as current guidance.

2. **Historical report**
   - May retain its original findings if they are part of the timeline.
   - Must receive a brief correction or supersession note if the original text
     now risks misleading a future operator.

3. **Governing plan**
   - Should remain stable unless the plan's stated next step became outdated due
     to the corrected Phase 6A result.
   - Any update must be minimal and clearly framed as status/context alignment,
     not a rewrite of the full remediation plan.

## Document-Specific Editing Rules

### 1. `2026-05-11-phase-6a-retry-result.md`

Target role: **Canonical current-state retry report**

Required qualities:

- Clearly state that the preserved executable exits `0` from both tested launch
  locations.
- Clearly retract or correct the stale `_version` failure attribution.
- State that Phase 6A passed at the compiled import smoke level.
- Explain the missing `GUI_IMPORT_OK` text through stdout suppression rather than
  import failure.
- Avoid hedging language that implies the retry result is still unresolved.

### 2. `2026-05-11-remediation-progress-report.md`

Target role: **Canonical current remediation status**

Required qualities:

- Update the Phase 6A section so it no longer says the retry result is failed or
  blocked on `_version`.
- Record that Phase 6A is passed and that the broad import refactor is deferred.
- Make sure the next active build concern is Phase 6B / later work, not a phantom
  Phase 6A runtime blocker.
- Preserve already-made decisions about `PostProcessingError` and LUT API unless
  separately revisited.

### 3. `2026-05-11-next-codex-handoff.md`

Target role: **Operator handoff**

Required qualities:

- Reflect the corrected state a new operator should inherit.
- Remove any instruction to chase the stale `_version` retry failure.
- Point next work toward Phase 6B preparation, documentation alignment, or other
  still-open verified tasks.
- Keep the division of labor intact unless the user explicitly changes it.

### 4. `2026-05-11-phantom-code-remediation-and-audit-plan.md`

Target role: **Governing plan**

Required qualities:

- Continue to serve as the controlling remediation/audit plan.
- Only change it if a status note or next-step reference became misleading after
  Phase 6A was proven passed.
- Do not turn it into a Phase 6A incident diary.
- If updated, keep the note concise and oriented toward present execution state.

## Optional Historical Supersession Notes

If a secondary document now contains a materially outdated conclusion, prepend or
append a compact note such as:

```text
Update, 2026-05-11: Later artifact-specific reverification showed the preserved
Phase 6A retry executable exits 0 from both the repo root and its `.dist`
directory. The `_version` runtime failure discussed below should not be treated as
the confirmed result of that preserved retry artifact.
```

Use this only where it prevents confusion. Do not rewrite every historical note
for stylistic consistency.

## Prohibited During This Pass

- Do not edit unrelated plans in `planning/plans/`.
- Do not perform a repo-wide markdown audit.
- Do not run a new Nuitka build.
- Do not restart the Phase 6A investigation unless a scoped document contradicts
  the already-reverified artifact result.
- Do not introduce a new import refactor recommendation unless it is backed by
  separate, current evidence.
- Do not stage, commit, or otherwise mutate git state unless the user explicitly
  asks.

## Verification After Edits

After updating the scoped documents:

1. Re-run the same constrained `rg` search from Step 3.
2. Confirm no primary document still presents the withdrawn `_version` retry
   failure as current truth.
3. Confirm no primary document says Phase 6A is still failed, unresolved, or
   blocked if it is referring to the preserved retry artifact.
4. Confirm the remediation progress report and handoff point to the same next
   work.
5. Confirm the governing remediation plan still reads coherently with the
   corrected Phase 6A status.

## Definition Of Done

This reconciliation pass is complete only when:

- The four primary documents agree on the corrected Phase 6A state.
- Any secondary document likely to mislead a future reader has a concise
  supersession/correction note.
- No unrelated planning document was touched.
- No new build or source-code work was introduced under the banner of doc cleanup.
- The resulting documentation makes it clear that Phase 6A is passed and that
  the next work belongs to the remaining build/remediation sequence rather than
  to a disproven retry failure.

## Resulting Deliverable

The output of this plan should be a small, surgical doc update set that:

- preserves the historical record;
- corrects the active operational record;
- prevents stale failure claims from steering future work;
- leaves unrelated planning material undisturbed.
