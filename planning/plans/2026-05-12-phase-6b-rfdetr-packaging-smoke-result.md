# Phase 6B RF-DETR Packaging Smoke Result

Date: 2026-05-12  
Project: Reconstruction Zone  
Scope: bounded RF-DETR packaging experiments only  
Related plan:
`planning/plans/2026-05-11-phase-6b-rfdetr-packaging-smoke-plan.md`

## Executive Summary

The RF-DETR source smoke is good. The first packaging control was not.

Current status:

1. **Source smoke:** PASS.
2. **Packaging control A1, force-including all of `transformers`:** FAILS
   immediately and is rejected.
3. **Packaging control A2, letting Nuitka follow only the statically reached
   Transformers imports:** PASS when `NUITKA_CACHE_DIR` is redirected into the
   workspace. Earlier A2 attempts were inconclusive until the cache-location
   issue was isolated. A2 has now had:
   - one 30-minute direct runner attempt with bad telemetry;
   - one 60-second direct telemetry diagnostic that proved live output works;
   - one authoritative direct run interrupted by the user after about 44 minutes;
   - one AppData cache-permission failure;
   - one failed `--disable-cache=bytecode` retry;
   - one successful workspace-cache retry.
4. **Optional `--nofollow-import-to="transformers.models.*"` exclusion trial:**
   **NOT RUN**. It is now optional rather than blocked.

The most important conclusion is:

> Do not promote `--include-package=transformers` into the Phase 6B full-build
> command. The viable A2 control is: do not force-include all of Transformers;
> redirect `NUITKA_CACHE_DIR` into the workspace; let Nuitka follow the statically
> reached Transformers imports.

## 1. Source Smoke Baseline

Script:

```text
scripts/smoke_import_rfdetr.py
```

Command:

```powershell
& 'C:\Python314\python.exe' D:\Projects\reconstruction-zone\scripts\smoke_import_rfdetr.py
```

Observed result:

```text
RFDETR_VERSION 1.4.1
TRANSFORMERS_VERSION 4.57.6
RFDETR_CLASSES_OK RFDETRSegNano,RFDETRSegSmall,RFDETRSegMedium,RFDETRSegLarge
TRANSFORMERS_MODEL_MODULE_COUNT 10
TRANSFORMERS_MODEL_MODULE transformers.models.auto
TRANSFORMERS_MODEL_MODULE transformers.models.auto.auto_factory
TRANSFORMERS_MODEL_MODULE transformers.models.auto.configuration_auto
TRANSFORMERS_MODEL_MODULE transformers.models.auto.modeling_auto
TRANSFORMERS_MODEL_MODULE transformers.models.auto.tokenization_auto
TRANSFORMERS_MODEL_MODULE transformers.models.bloom
TRANSFORMERS_MODEL_MODULE transformers.models.bloom.configuration_bloom
TRANSFORMERS_MODEL_MODULE transformers.models.bloom.modeling_bloom
TRANSFORMERS_MODEL_MODULE transformers.models.encoder_decoder
TRANSFORMERS_MODEL_MODULE transformers.models.encoder_decoder.configuration_encoder_decoder
RFDETR_IMPORT_OK
```

Classification:

- **Source import/class surface:** PASS.
- The source smoke confirms the same ten eager `transformers.models.*` modules
  previously observed through smaller one-line probes.

## 2. Packaging Control A1 — Rejected `--include-package=transformers` Variant

### Command shape

The first packaging control used:

```text
--include-package=rfdetr
--include-package=transformers
--include-package=torch
--include-package=torchvision
```

inside:

```text
dist_test/phase-6b-rfdetr-control/
```

Artifacts preserved:

```text
dist_test/phase-6b-rfdetr-control/build.stderr.log
dist_test/phase-6b-rfdetr-control/build.exitcode.txt
dist_test/phase-6b-rfdetr-control/nuitka-report.xml
```

### Result

Observed exit code:

```text
1
```

Observed fatal:

```text
FATAL: transformers: Making changes to
'transformers.commands.add_new_model_like' that cause SyntaxError
'f-string: valid expression required before '}' (add_new_model_like.py, line 328)'
```

The compilation report records:

```text
completion="error exit message (1)"
```

with the same fatal message in the report metadata.

### Interpretation

This failure is **not** evidence that RF-DETR itself is impossible to package.
It is evidence that **force-including the entire `transformers` package** pulls
Nuitka into a problematic Transformers command module unrelated to the bounded
RF-DETR smoke target.

Decision:

- Reject `--include-package=transformers` for this bounded control.
- Do not carry it forward into the Phase 6B full-build proposal without a much
  stronger reason.

## 3. Packaging Control A2 — Static-Follow Transformers Variant

### Revised intent

After A1 failed, Codex attempted a narrower control:

- keep:
  - `--include-package=rfdetr`
  - `--include-package=torch`
  - `--include-package=torchvision`
- remove:
  - `--include-package=transformers`

The intent was to let Nuitka follow only the Transformers modules statically
reached by the smoke graph, rather than eagerly packaging the entire
Transformers distribution.

### Execution notes

Several detached-run wrappers were attempted first. They were not reliable
enough to treat as build evidence because they exited without preserving useful
logs or exit markers.

Codex then ran the durable runner directly in the foreground with a 30-minute
timeout:

```text
dist_test/phase-6b-rfdetr-control-static-transformers/run-control-static-transformers.ps1
```

### Observed result

- The foreground command timed out after 30 minutes.
- No executable was produced.
- No exit-code marker was written.
- No useful stdout/stderr build transcript was preserved by the runner before the
  timeout.
- The generated build tree contained only:

```text
smoke_import_rfdetr.build/.gitignore
```

- The `.dist` directory existed but contained no executable.

### Interpretation

This control is **inconclusive**, not a confirmed failure and not a confirmed
success.

What it does show:

- Removing `--include-package=transformers` avoids the immediate A1 fatal.
- The narrower control can run materially longer than the rejected A1 variant.

What it does **not** show:

- that the smoke would eventually build successfully;
- that it is stuck in the same Transformers stall as the full GUI attempt;
- that the later optional `transformers.models.*` exclusion is safe to test now.

## 4. Exclusion Trial Status

The optional experiment:

```text
--nofollow-import-to="transformers.models.*"
```

was **not run**.

Reason:

- the required comparison baseline, Packaging Control A2, is not yet classified;
- testing the exclusion before the control is understood would create an
  uninterpretable result.

## 5. Process-State Check After The Attempt

After the 30-minute A2 timeout, Codex checked for lingering compiler/build
processes.

Observed:

- no `python` process was returned;
- no `cl` process was returned.

Only unrelated `pwsh` processes were visible in the general process query.

## 5A. Authoritative A2 Direct Run — Interrupted By User

After the 60-second telemetry diagnostic confirmed visible output works, Codex
started the authoritative A2 static-follow control directly:

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
  --report=dist_test/phase-6b-rfdetr-control-static-transformers/a2-authoritative-report.xml `
  --output-dir=dist_test/phase-6b-rfdetr-control-static-transformers `
  scripts\smoke_import_rfdetr.py
```

Observed at launch:

```text
Nuitka-Options: Used command line options:
...
Nuitka: Starting Python compilation with:
Nuitka:   Version '4.0.8' on Python 3.14 ...
Nuitka: Detected 2 module(s) with mismatching checksum in
'scripts/nuitka-torch-fix.yml': torch.library,torch._refs
```

The user then intentionally interrupted the turn after approximately 44 minutes
and 44 seconds of wall-clock runtime.

### Post-interruption artifact state

Codex checked:

```text
dist_test/phase-6b-rfdetr-control-static-transformers/a2-authoritative-report.xml
```

Result:

```text
not present
```

Codex checked:

```text
dist_test/phase-6b-rfdetr-control-static-transformers/smoke_import_rfdetr.dist/smoke_import_rfdetr.exe
```

Result:

```text
not present
```

The `.dist` directory existed but contained no executable. The build directory
still contained only its placeholder `.gitignore` when inspected through the
top-level listing available after interruption.

### Post-interruption process state

After the user interruption, Codex found build-era processes still alive:

```text
PID 73256  python  C:\Python314\python.exe  started 2026-05-12 00:38:34
PID 87980  python  C:\Python314\python.exe  started 2026-05-12 00:38:34
PID 28360  pwsh    C:\Program Files\PowerShell\7\pwsh.exe  started 2026-05-12 00:38:33
```

Codex stopped only those build-era processes:

```powershell
Stop-Process -Id 73256,87980,28360 -Force
```

Follow-up `Get-Process python,pwsh` showed no remaining `python` processes.

### Classification

This authoritative A2 run is **interrupted / inconclusive**.

It is not a pass because no executable was produced and no packaged smoke ran.
It is not a normal Nuitka failure because no fatal error, report, or exit code was
produced before interruption.
It is not yet a proven recognizable stall because the visible telemetry captured
in the conversation only reached early Nuitka startup text before later output
was interrupted.

## 6. Current Decision

The Phase 6B RF-DETR packaging thread should now stand at:

1. Source smoke: **PASS**
2. A1 broad Transformers include: **REJECTED**
3. A2 static-follow control with AppData cache: **FAILED** due to Nuitka
   module-cache `PermissionError`
4. A2 static-follow control with `--disable-cache=bytecode`: **FAILED** with
   the same AppData module-cache `PermissionError`
5. A2 static-follow control with workspace `NUITKA_CACHE_DIR`: **PASS**
6. Blanket `transformers.models.*` exclusion trial: **OPTIONAL / NOT RUN**

## 7. Recommended Next Move

Before any further Nuitka comparison:

1. Agree on a time budget for the next A2 attempt.
2. Re-run **only** the A2 static-follow control with:
   - visible or reliably captured logs;
   - an explicit exit-code file;
   - a hard time limit;
   - artifact preservation.
3. Immediately after any interruption or timeout, check for and stop only the
   build-era `python` / compiler processes before doing more work.
4. Only if A2 produces a classifiable result should the optional
   `transformers.models.*` exclusion trial be considered.

### Telemetry diagnostic added after the 30-minute attempt

Codex then ran the A2 command directly, without the silent wrapper/redirection
path, under a short 60-second diagnostic timeout.

Observed live output:

```text
Nuitka-Options: Used command line options:
...
Nuitka: Starting Python compilation with:
Nuitka:   Version '4.0.8' on Python 3.14 ...
Nuitka: Detected 2 module(s) with mismatching checksum in
'scripts/nuitka-torch-fix.yml': torch.library,torch._refs
```

Interpretation:

- direct invocation does stream usable Nuitka telemetry;
- the earlier silence was a runner/log-capture issue, not proof that Nuitka was
  idle;
- the next authoritative A2 attempt should therefore run directly with visible
  output, or use a proven streaming capture mechanism, rather than the silent
  wrapper approach used earlier.

## 7A. A2 Long Run — AppData Nuitka Cache Permission Failure

Codex then ran the A2 static-follow control directly with a long timeout:

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
  --report=dist_test/phase-6b-rfdetr-a2-long-20260512/a2-long-report.xml `
  --output-dir=dist_test/phase-6b-rfdetr-a2-long-20260512 `
  scripts\smoke_import_rfdetr.py
```

Observed result:

```text
exit code: 1
report: dist_test/phase-6b-rfdetr-a2-long-20260512/a2-long-report.xml
report completion: exception
```

Failure:

```text
PermissionError: [Errno 13] Permission denied:
C:\Users\alexm\AppData\Local\Nuitka\Nuitka\Cache\module-cache\email.mime@...
```

Interpretation:

- This was not an RF-DETR import failure.
- This was not the previous broad Transformers command-module failure.
- This was Nuitka failing to write its own AppData module cache.

## 7B. A2 Retry With `--disable-cache=bytecode` — Still Failed

Codex then tested:

```text
--disable-cache=bytecode
```

in:

```text
dist_test/phase-6b-rfdetr-a2-nobytecodecache-20260512
```

Observed result:

```text
exit code: 1
report: dist_test/phase-6b-rfdetr-a2-nobytecodecache-20260512/a2-nobytecodecache-report.xml
```

Failure:

```text
PermissionError: [Errno 13] Permission denied:
C:\Users\alexm\AppData\Local\Nuitka\Nuitka\Cache\module-cache\__future__@...
```

Interpretation:

- `--disable-cache=bytecode` did not disable the module-cache write path that was
  failing.
- The cache-location problem remained.

## 7C. A2 Retry With Workspace `NUITKA_CACHE_DIR` — PASS

Codex then used Nuitka's documented cache-location override:

```powershell
$env:NUITKA_CACHE_DIR='D:\Projects\reconstruction-zone\dist_test\nuitka-cache'
```

and reran A2 in:

```text
dist_test/phase-6b-rfdetr-a2-workspacecache-20260512
```

Command shape:

```powershell
$env:NUITKA_CACHE_DIR='D:\Projects\reconstruction-zone\dist_test\nuitka-cache'
& 'C:\Python314\python.exe' -m nuitka `
  --standalone `
  --assume-yes-for-downloads `
  --windows-console-mode=attach `
  --include-package=rfdetr `
  --include-package=torch `
  --include-package=torchvision `
  --user-package-configuration-file=scripts/nuitka-torch-fix.yml `
  --module-parameter=torch-disable-jit=yes `
  --report=dist_test/phase-6b-rfdetr-a2-workspacecache-20260512/a2-workspacecache-report.xml `
  --output-dir=dist_test/phase-6b-rfdetr-a2-workspacecache-20260512 `
  scripts\smoke_import_rfdetr.py
```

Build result:

```text
exit code: 0
report completion: yes
executable: dist_test/phase-6b-rfdetr-a2-workspacecache-20260512/smoke_import_rfdetr.dist/smoke_import_rfdetr.exe
```

Artifact identity:

```text
smoke_import_rfdetr.exe: 927,674,880 bytes
a2-workspacecache-report.xml: 21,715,447 bytes
```

Notable build facts:

- Nuitka compiled and linked **9,308 C files**.
- Nuitka found and bundled:
  - bitsandbytes DLLs;
  - cv2 DLL;
  - sklearn DLLs;
  - torch DLLs;
  - torchvision DLLs.
- The `transformers.commands.add_new_model_like` fatal did not recur.
- The AppData module-cache permission error did not recur.

Runtime checks:

1. Run from repository root:

```text
exit code: 0
stdout length: 0
stderr length: 0
```

2. Run from `.dist` directory:

```text
exit code: 0
stdout length: 0
stderr length: 0
```

Interpretation:

- A2 is now **compiled import-smoke PASS**.
- The packaged executable exits cleanly from both tested launch locations.
- The expected text markers are not visible in captured stdout/stderr. As with
  the Phase 6A GUI smoke, marker visibility is therefore **inconclusive /
  suppressed**, but the no-traceback exit-0 result is the meaningful packaged
  smoke signal.

## 7D. Revised Recommended Next Move

The A2 control is now classified.

Next choices:

1. **Conservative next step:** carry the A2 workspace-cache strategy forward into
   the next Phase 6B planning layer without running the optional
   `transformers.models.*` exclusion trial.
2. **Optimization experiment:** run the optional
   `--nofollow-import-to="transformers.models.*"` trial in the same RF-DETR smoke
   context, also with `NUITKA_CACHE_DIR` redirected into the workspace, to learn
   whether it still exits `0` and whether it materially reduces build size/time.

Codex recommendation:

- Do not use `--include-package=transformers`.
- Keep the workspace `NUITKA_CACHE_DIR` override.
- Treat the blanket `transformers.models.*` exclusion as optional optimization,
  not required for correctness.

## 8. Documentation Impact

The prior Phase 6B smoke plan must no longer imply that the
`--include-package=transformers` control is the recommended first packaging
attempt. That variant has now been tested and rejected.

The live next-step documentation should instead point toward:

- A2 static-follow with workspace `NUITKA_CACHE_DIR` as the currently proven
  RF-DETR packaging-smoke strategy;
- then, conditionally, the optional exclusion trial.

## Final Assessment

This was still useful progress.

We learned something real without spending a full GUI build:

- **full Transformers package inclusion is a bad control strategy here;**
- **the narrower RF-DETR control works when Nuitka's cache is redirected into
  the workspace;**
- **the next open question is optimization, not basic RF-DETR import packaging.**
