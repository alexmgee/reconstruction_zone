# Phase 6B Strategy Oversight Report

Date: 2026-05-11  
Project: Reconstruction Zone  
Scope: Phase 6B full-build preparation only  
Governing context:

1. `planning/plans/2026-05-11-phantom-code-remediation-and-audit-plan.md`
2. `planning/plans/2026-05-11-remediation-progress-report.md`
3. `planning/plans/2026-05-11-phase-6a-retry-result.md`

## Executive Conclusion

Phase 6A is closed as passed. The next unresolved packaging question is Phase 6B.

The current Phase 6B proposal in the progress report:

```text
--nofollow-import-to="transformers.models.*"
```

should **not** be treated as approved or build-ready yet.

Local evidence from the installed environment shows that `rfdetr` import paths
already load several `transformers.models.*` modules before any full RF-DETR
runtime workflow is exercised. That makes a blanket
`transformers.models.*` exclusion risky: it may cut modules that are touched by
the actual import path, not merely avoid dead-weight model discovery.

The right next step is a smaller, RF-DETR-specific packaging proof before any
full application build adopts that exclusion strategy.

That bounded proof is now defined in:

```text
planning/plans/2026-05-11-phase-6b-rfdetr-packaging-smoke-plan.md
```

with the executable source target:

```text
scripts/smoke_import_rfdetr.py
```

## 1. Persistent Local-Import Audit Status

The previously missing reusable audit script now exists:

```text
scripts/audit_local_imports.py
```

During this oversight pass it was strengthened to align more closely with the
governing remediation plan:

- It now audits plain `import local.module` statements, not only
  `from local.module import ...`.
- It now resolves relative `from .module import ...` paths against the current
  package context.
- It continues to understand the `reconstruction_gui` top-level alias pattern
  (`tabs.*`, `widgets`, `app_infra`, etc.).
- It preserves the explicit allowlist structure with reason strings.
- It suppresses unrelated `SyntaxWarning` noise while parsing files for import
  structure.

Verification command:

```powershell
C:\Python314\python.exe D:\Projects\reconstruction-zone\scripts\audit_local_imports.py
```

Observed result:

```text
No unresolved local imports found.
```

Status:

- **Persistent local-import audit: PRESENT**
- **Current run: PASS**

## 2. Local RF-DETR Environment Snapshot

Observed local package versions:

```text
rfdetr 1.4.1
transformers 4.57.6
```

The new source smoke script was run successfully after creation and emitted:

```text
RFDETR_IMPORT_OK
```

## Packaging Experiment Update — 2026-05-12

See:

```text
planning/plans/2026-05-12-phase-6b-rfdetr-packaging-smoke-result.md
```

Current packaging findings:

1. The broad control variant that force-included all of `transformers` is
   rejected. It fails fast with a Nuitka fatal in
   `transformers.commands.add_new_model_like`.
2. The narrower static-follow control that omits
   `--include-package=transformers` remains worth testing, but the first attempt
   is **inconclusive** because the 30-minute run did not return before timeout
   and the runner did not preserve usable telemetry.
3. The optional blanket `transformers.models.*` exclusion trial remains deferred
   until the narrower control produces a classifiable result.

Observed installed RF-DETR package root:

```text
C:\Users\alexm\AppData\Roaming\Python\Python314\site-packages\rfdetr
```

## 3. Import Footprint Probes

### Probe A: `import rfdetr`

Command:

```powershell
C:\Python314\python.exe -c "import rfdetr, sys; mods=sorted(m for m in sys.modules if m.startswith('transformers.models.')); print('COUNT', len(mods)); print('\n'.join(mods[:80]))"
```

Observed:

```text
COUNT 10
transformers.models.auto
transformers.models.auto.auto_factory
transformers.models.auto.configuration_auto
transformers.models.auto.modeling_auto
transformers.models.auto.tokenization_auto
transformers.models.bloom
transformers.models.bloom.configuration_bloom
transformers.models.bloom.modeling_bloom
transformers.models.encoder_decoder
transformers.models.encoder_decoder.configuration_encoder_decoder
```

### Probe B: `from rfdetr import RFDETRSegSmall`

Command:

```powershell
C:\Python314\python.exe -c "import sys; from rfdetr import RFDETRSegSmall; mods=sorted(m for m in sys.modules if m.startswith('transformers.models.')); print('COUNT', len(mods)); print('\n'.join(mods[:120]))"
```

Observed:

```text
COUNT 10
transformers.models.auto
transformers.models.auto.auto_factory
transformers.models.auto.configuration_auto
transformers.models.auto.modeling_auto
transformers.models.auto.tokenization_auto
transformers.models.bloom
transformers.models.bloom.configuration_bloom
transformers.models.bloom.modeling_bloom
transformers.models.encoder_decoder
transformers.models.encoder_decoder.configuration_encoder_decoder
```

### Probe C: `import rfdetr.models.backbone.dinov2`

Command:

```powershell
C:\Python314\python.exe -c "import sys; import rfdetr.models.backbone.dinov2 as d; mods=sorted(m for m in sys.modules if m.startswith('transformers.models.')); print('COUNT', len(mods)); print('\n'.join(mods[:120]))"
```

Observed:

```text
COUNT 10
transformers.models.auto
transformers.models.auto.auto_factory
transformers.models.auto.configuration_auto
transformers.models.auto.modeling_auto
transformers.models.auto.tokenization_auto
transformers.models.bloom
transformers.models.bloom.configuration_bloom
transformers.models.bloom.modeling_bloom
transformers.models.encoder_decoder
transformers.models.encoder_decoder.configuration_encoder_decoder
```

## 4. Source-Level RF-DETR / Transformers Dependencies

The installed RF-DETR package directly imports non-model-tree Transformers APIs
and also uses `AutoBackbone`:

```text
rfdetr\models\backbone\dinov2.py
  from transformers import AutoBackbone
```

```text
rfdetr\models\backbone\dinov2_with_windowed_attn.py
  from transformers.activations import ACT2FN
  from transformers.configuration_utils import PretrainedConfig
  from transformers.modeling_outputs import ...
  from transformers.modeling_utils import PreTrainedModel
  from transformers.pytorch_utils import ...
  from transformers.utils import ...
  from transformers.utils.backbone_utils import ...
```

Additionally, the local app's RF-DETR segmenter eventually constructs the
selected RF-DETR model class:

```text
reconstruction_gui\reconstruction_pipeline.py
  self.model = model_cls(**model_kwargs)
```

That means an import-only probe is informative, but not sufficient to describe
the eventual runtime footprint of RF-DETR model construction or inference.

## 5. Why The Current Phase 6B Proposal Is Not Ready

The current progress report says:

```text
Use --nofollow-import-to="transformers.models.*"
```

and suggests a pre-build probe based on:

```python
import rfdetr
used_models = [m for m in sys.modules if m.startswith("transformers.models.")]
```

This needs refinement for two reasons.

### Concern 1: The exclusion overlaps with modules already imported today

The local probes show `import rfdetr` itself currently loads:

- `transformers.models.auto.*`
- `transformers.models.bloom.*`
- `transformers.models.encoder_decoder.*`

So the proposed blanket exclusion is not merely skipping distant, irrelevant
subtrees. It overlaps with modules already touched by the present import path.

### Concern 2: The proposed probe is eager-import-only

The suggested `import rfdetr` probe captures what appears during package import,
not what appears during:

- RF-DETR model-class construction;
- backbone initialization;
- AutoBackbone resolution;
- inference on a real image.

It is useful, but it cannot by itself justify a full-build exclusion policy.

## 6. Revised Phase 6B Recommendation

Do **not** move directly from the current progress-report strategy to a full GUI
build.

Use this narrower sequence instead:

### Step 1: Preserve the observed eager-import footprint

Record the 10 `transformers.models.*` modules already loaded by the three probes
above. Treat them as required-to-explain inputs, not as disposable noise.

### Step 2: Add a dedicated RF-DETR packaging smoke

Create or define a minimal smoke target that:

- imports `rfdetr`;
- imports the exact RF-DETR segmentation class the app uses;
- avoids model download;
- avoids full GUI startup;
- exits nonzero on import failure.

This smoke should be much cheaper than a full app build and gives a safer place
to test Nuitka exclusion flags.

### Step 3: Test any `transformers.models.*` exclusion only against that smoke

If experimenting with:

```text
--nofollow-import-to="transformers.models.*"
```

validate it first on the RF-DETR-specific smoke. Do not promote it into the full
application build until that smoke passes.

### Step 4: If blanket exclusion breaks the RF-DETR smoke, stop using it

If the dedicated RF-DETR smoke raises import errors or otherwise fails because of
the exclusion, reject the blanket `transformers.models.*` strategy. Do not try to
force it into the GUI build by optimism.

### Step 5: Only then choose the full-build path

After the smaller smoke is understood, choose one of:

1. **No blanket Transformers model-tree exclusion** if correctness demands it.
2. **A narrower, evidence-backed exclusion set** if specific model families are
   proven irrelevant.
3. **A Nuitka package-config / anti-bloat route** if the issue is primarily
   unwanted static analysis rather than runtime code needs.

The correct branch should be selected from observed smoke behavior, not from the
current assumption that the full `transformers.models.*` tree can be safely
blocked.

## 7. Proposed Status Update To The Build Thread

The build thread should now read:

- **Phase 6A:** passed.
- **Persistent local-import audit:** present and passing.
- **Phase 6B:** strategy under review; current blanket
  `transformers.models.*` nofollow idea is **not yet approved**.
- **Immediate next technical step:** smaller RF-DETR packaging smoke or an
  equivalent bounded proof before any full build adopts that exclusion strategy.

## 8. Stop Conditions

Stop and reassess if any of the following occurs:

- A Phase 6B plan reintroduces the blanket `transformers.models.*` exclusion as
  if it were already proven safe.
- A full build is launched before the RF-DETR-specific packaging question is
  resolved.
- A probe result is promoted beyond what it actually proves, especially if an
  import-only result is used to justify runtime inference packaging.

## Final Assessment

The repository is in better shape than it was before the Phase 6A reconciliation,
but Phase 6B still needs disciplined narrowing.

The important correction is:

- **6A is settled.**
- **The audit script gap is closed.**
- **6B needs a smaller, evidence-backed RF-DETR packaging proof before the next
  expensive build attempt.**
