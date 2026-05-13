# Phase 6B RF-DETR Packaging Smoke Plan

Date: 2026-05-11  
Project: Reconstruction Zone  
Scope: bounded RF-DETR packaging proof before any full GUI build  
Related oversight report:
`planning/plans/2026-05-11-phase-6b-strategy-oversight-report.md`

## Purpose

This plan defines the smaller Phase 6B proof recommended by Codex before any
full application build adopts or rejects a Transformers exclusion strategy.

The goal is not to prove the entire packaged GUI works. The goal is to answer a
much narrower packaging question:

> Can a standalone Nuitka artifact successfully import the exact RF-DETR
> segmentation-class surface the app depends on, and what happens if we try to
> apply a `transformers.models.*` exclusion in that smaller context?

## Execution Status Update — 2026-05-12

The first packaging control variant has now been tested and rejected:

- `--include-package=transformers` causes an immediate Nuitka fatal in
  `transformers.commands.add_new_model_like`.
- That variant is no longer the recommended control path.

The next control to run is the narrower **static-follow Transformers** variant
that keeps `rfdetr`, `torch`, and `torchvision` included, but does **not**
force-include the entire `transformers` package.

Later update:

- A direct authoritative A2 static-follow run was started with visible telemetry.
- The user interrupted it after about 44 minutes and 44 seconds.
- The run left build-era `python` processes alive; Codex stopped only those
  build-era processes.
- No executable or A2 report was produced.
- A2 remains **inconclusive / interrupted**, not pass/fail.

Final A2 update:

- A2 with the default AppData Nuitka cache failed after about an hour with a
  `PermissionError` writing to
  `C:\Users\alexm\AppData\Local\Nuitka\Nuitka\Cache\module-cache\...`.
- A2 with `--disable-cache=bytecode` still failed on the same AppData
  module-cache path.
- A2 with `NUITKA_CACHE_DIR` redirected to
  `D:\Projects\reconstruction-zone\dist_test\nuitka-cache` compiled
  successfully and produced `smoke_import_rfdetr.exe`.
- The packaged executable exited `0` from both the repo root and `.dist`
  directory, with stdout/stderr empty.
- A2 is now classified as **PASS with workspace cache override**.

See:

```text
planning/plans/2026-05-12-phase-6b-rfdetr-packaging-smoke-result.md
```

## New Smoke Script

The bounded proof now has a dedicated source target:

```text
scripts/smoke_import_rfdetr.py
```

That script deliberately:

- imports `rfdetr`;
- verifies presence of:
  - `RFDETRSegNano`
  - `RFDETRSegSmall`
  - `RFDETRSegMedium`
  - `RFDETRSegLarge`
- prints the installed `rfdetr` and `transformers` versions;
- prints every eager `transformers.models.*` module already loaded by the import
  path;
- exits nonzero if any required segmentation class is absent.

It deliberately does **not**:

- instantiate a model;
- download weights;
- run inference;
- import the full GUI;
- claim to validate the final Phase 6B full-build path.

## Why This Scope Is Correct

The application code currently resolves RF-DETR model classes dynamically inside
`RFDETRSegmenter.initialize()` and eventually constructs the selected model class.
For the immediate packaging question, the first thing to prove is that the
package import/class surface survives packaging.

That is smaller, cheaper, and more diagnostic than jumping directly into a full
GUI build.

## Source-Run Baseline

Run:

```powershell
& 'C:\Python314\python.exe' D:\Projects\reconstruction-zone\scripts\smoke_import_rfdetr.py
```

Expected shape:

```text
RFDETR_VERSION ...
TRANSFORMERS_VERSION ...
RFDETR_CLASSES_OK RFDETRSegNano,RFDETRSegSmall,RFDETRSegMedium,RFDETRSegLarge
TRANSFORMERS_MODEL_MODULE_COUNT ...
TRANSFORMERS_MODEL_MODULE ...
RFDETR_IMPORT_OK
```

For the current environment already inspected by Codex:

```text
rfdetr 1.4.1
transformers 4.57.6
```

and the eager import footprint included ten `transformers.models.*` modules.

## Packaging Experiment A: Control Smoke Without Blanket Model-Tree Exclusion

Purpose:

- Establish whether the bounded RF-DETR import/class smoke packages at all.
- Avoid confounding the result with an unproven exclusion flag.

### Rejected A1 Command Shape

The original control command force-included the entire `transformers` package:

```powershell
& 'C:\Python314\python.exe' -m nuitka `
  --standalone `
  --assume-yes-for-downloads `
  --windows-console-mode=attach `
  --include-package=rfdetr `
  --include-package=transformers `
  --include-package=torch `
  --include-package=torchvision `
  --user-package-configuration-file=scripts/nuitka-torch-fix.yml `
  --module-parameter=torch-disable-jit=yes `
  --output-dir=dist_test `
  scripts\smoke_import_rfdetr.py
```

This A1 variant has now been tested and rejected. Do not rerun it as the default
control.

### Recommended A2 Command Shape

Use the narrower static-follow variant instead:

```powershell
& 'C:\Python314\python.exe' -m nuitka `
  --standalone `
  --assume-yes-for-downloads `
  --windows-console-mode=attach `
  --include-package=rfdetr `
  --include-package=torch `
  --include-package=torchvision `
  --user-package-configuration-file=scripts/nuitka-torch-fix.yml `
  --module-parameter=torch-disable-jit=yes `
  --output-dir=dist_test `
  scripts\smoke_import_rfdetr.py
```

Then, only if the build completes, run:

```powershell
& 'D:\Projects\reconstruction-zone\dist_test\smoke_import_rfdetr.dist\smoke_import_rfdetr.exe'
```

Success criteria:

- build completes;
- executable exits `0`;
- output contains `RFDETR_IMPORT_OK`;
- output contains `RFDETR_CLASSES_OK ...`;
- any `transformers.models.*` footprint is recorded, not ignored.

Stop conditions:

- Nuitka enters a long Transformers model-tree walk and shows the same stall
  signature seen in the earlier GUI context;
- build exceeds the agreed bounded window;
- executable fails import/class resolution.

Additional A2 discipline:

- Preserve logs and an explicit exit-code file.
- Do not treat a shell-wrapper problem as a Nuitka result.
- If a runner times out without telemetry, classify it as **inconclusive**, not
  failed and not passed.
- Prefer direct foreground execution with visible output for the next
  authoritative A2 run, since a 60-second diagnostic confirmed Nuitka streams
  live telemetry correctly in that mode.
- Agree on a time budget before re-running A2.
- After any interruption or timeout, immediately check for leftover build-era
  `python`, compiler, or shell processes and stop only the processes tied to the
  interrupted build.
- For any future A2-derived command, set:

```powershell
$env:NUITKA_CACHE_DIR='D:\Projects\reconstruction-zone\dist_test\nuitka-cache'
```

before invoking Nuitka.

## Packaging Experiment B: Optional Exclusion Trial

Only after Experiment A is understood, test:

```text
--nofollow-import-to="transformers.models.*"
```

in the same bounded RF-DETR smoke, not in the full GUI build.

Interpretation rules:

1. If the control smoke succeeds but the exclusion smoke fails:
   - reject the blanket exclusion as unsafe for Phase 6B;
   - do not promote it into a full build command.

2. If both control and exclusion smokes succeed:
   - record that result carefully;
   - do **not** automatically conclude model construction or inference will also
     succeed;
   - decide whether a second-stage runtime smoke is justified.

3. If both smokes stall:
   - the issue is broader than the current exclusion hypothesis;
   - investigate Nuitka package configuration / anti-bloat handling rather than
     guessing more command-line patterns.

## Optional Experiment C: Minimal Constructor Smoke

This is explicitly **not** part of the first bounded proof, but it may become the
next layer if import/class packaging is clean.

A constructor smoke would need to be designed carefully so it:

- does not download weights;
- uses an explicit local weight path only if one is already present and approved
  for the test;
- distinguishes package/import breakage from model-weight availability;
- records any additional `transformers.models.*` modules loaded during actual
  constructor execution.

Do not create or run this automatically just because Experiment A succeeds.

## Required Logging Discipline

For any Nuitka packaging smoke:

- Do not pipe output through `tail`.
- Preserve the command used.
- Preserve the build log.
- Preserve the produced `.dist` artifact until the report is reconciled.
- Record:
  - exit code;
  - whether the executable ran;
  - exact success/failure markers;
  - whether the `transformers.models.*` footprint changed.

## Documentation Deliverable After Running The Smoke

Once the bounded smoke is actually executed, write a short result report that
records:

1. exact command;
2. package versions;
3. source smoke output;
4. packaged smoke output;
5. whether Experiment B was attempted;
6. decision:
   - proceed,
   - reject blanket exclusion,
   - or escalate to a deeper packaging strategy review.

## Definition Of Done

This Phase 6B micro-plan is complete when:

- `scripts/smoke_import_rfdetr.py` exists;
- the source-run baseline is known;
- the control packaging command is clearly documented;
- the optional exclusion trial is framed as a test, not as an assumed fix;
- no one can reasonably confuse this bounded smoke with a full application build
  or with proof of RF-DETR inference readiness.
