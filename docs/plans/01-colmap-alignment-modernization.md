# Plan 01 — COLMAP Alignment Modernization (Align tab) — v1.2

v1.1 folded in the gpt5.6-sol xhigh ledger interrogation (verdict EXECUTE-AFTER-EDITS; 8
findings; two Codex claims corrected by probe). v1.2 (2026-07-16, post-A0) adopts the
**release-download amendment** per gpt5.6-sol review .collab-logs/codex-a0-amendment-review.md
(verdict ADOPT-WITH-CONDITIONS): Phase A acquires CI-built artifacts from the build repo's
GitHub release v4.1.0 instead of rebuilding locally; local build kit retained as fallback.
Conditions folded in below: wheel metadata/filename consistency check, provenance archive,
pythonw+user-site real-CUDA probe, cuDSS-BA validation, artifact+hash pinning.

## Scope
Bring the Align tab's alignment stack to the most capable, current GPU COLMAP available:
fresh CUDA CLI + CUDA cp314 pycolmap wheel (both with Caspar, EQUIRECTANGULAR, EUCM) built
from `D:\Data\build_gpu_colmap`; runner/adapter fixed to actually use GPU BA; camera-aware
backend gating; explicit backend selector with per-run qualification; transactional
promotion with full rollback; SphereSfM retained as legacy-marked engine. Explicitly NOT:
dense reconstruction, meshing, other-tab changes, prep360 reframing changes, global-ERP
validation (G3 DEFERRED), full geometric validation/viewer projection for EUCM/ERP
(v1 = parse-safe + validate-refuse + points-only render; see Design), doc rewrites until after.

## User decisions (locked 2026-07-16)
1. Coordinator (Claude) drives the build via Bash with checkpoints + documentation.
2. SphereSfM → legacy-marked, contingent on native-ERP incremental validation.
3. Caspar ships as a GUI option — hard-gated, never silent.
4. Both 360 paths maintained: reframe→pinhole→Caspar AND native ERP (incremental).

## Gates — deterministic branches (no "surface first" stalls; user informed at checkpoint)
- **G1 — cp314 CUDA wheel: ANSWERED 2026-07-16.** Prebuilt
  `pycolmap-4.1.0+cuda.cudss-cp314-cp314-win_amd64.whl` exists in release v4.1.0
  (API digest sha256:54c61ca5…ecd1699). Local-build FAIL branches retained below only for
  the fallback path.
- **G2 — Caspar patch: MOOT on download path** (CI built from the 4.1.0 tag). If fallback
  local build is ever invoked: test `git -C third_party/colmap-for-pycolmap grep -l CASPAR
  -- src/pycolmap/estimators/`; PASS → delete patch + apply-step; FAIL → `git apply --check`;
  apply-failure → STOP wheel track, checkpoint with user.
- **G4 (new) — wheel filename/metadata consistency.** Codex claim (plausible, locally
  unverifiable — local workflow clone predates release): release wheels were RENAMED to
  `+cuda.cudss` without updating embedded METADATA `Version:`, which current pip may reject.
  Test at download: read METADATA from the wheel zip; attempt venv install with pip 25.2.
  PASS → proceed. FAIL (pip rejects) → rename file back to `pycolmap-4.1.0-cp314-cp314-
  win_amd64.whl` (matching embedded version), re-hash, record BOTH names + hashes in
  ACQUISITION-RECORD, install-test the renamed artifact. Precedent: live 4.0.2 wheel had
  the same filename scheme (`4.0.2+cuda`) with METADATA `Version: 4.0.2` and installed.
- **G3 — global-ERP:** DEFERRED out of this plan. GUI marks global mapper + EQUIRECTANGULAR
  combination "experimental — unvalidated"; no Phase-E lane. Revisit in a follow-up plan.
- **A0 FAIL branch:** escrow reinstall-test fails → STOP everything (no safety net exists);
  investigate wheel rebuild from site-packages metadata before any other phase.

## Design decisions (pre-resolved — do NOT re-litigate)
- **Camera-model ID namespace:** `prep360/core/colmap_binary.py` `COLMAP_CAMERA_MODELS` →
  upstream table (0–17; 11=RAD_TAN_THIN_PRISM_FISHEYE, 16=EUCM, 17=EQUIRECTANGULAR).
  Legacy fork table (11=SPHERE,3) retained as `SPHERESFM_CAMERA_MODELS`; reader takes
  `variant: Literal["colmap","spheresfm"]`, NO default — every call site passes explicitly
  (forcing the audit). `colmap_binary_validation_adapter.py` gains the same param and
  threads it through.
- **run_info schema v2 + migration:** adds `schema_version:2`, `engine`, `binary_path`,
  `binary_sha256`, `backend` ("pycolmap"|"cli"), `pycolmap_version`+`pycolmap_pyd_sha256`
  (when backend=pycolmap), `colmap_commit` (when known), `camera_model_namespace`
  ("colmap"|"spheresfm"). Reading a run: v2 → use recorded namespace; v1/absent →
  namespace inferred from recorded engine name if present, else REFUSE load with message
  "legacy run — re-run alignment or set namespace manually"; never guess from model bytes.
  Cross-engine resume → hard reject.
- **Caspar gate (evaluated immediately before mapping, when the db exists):** query DISTINCT
  camera models from `database.db` cameras table. Explicit CASPAR + any model ∉
  {PINHOLE, SIMPLE_RADIAL} → ABORT stage with explicit error naming offending models.
  AUTO → log the observed model set; select CASPAR iff eligible ∧ mapper=incremental ∧
  engine=colmap, else CERES. Backend NEVER switches mid-run.
- **BA controls in GUI:** Reconstruct section: "BA backend" dropdown (AUTO|CERES|CASPAR,
  default AUTO) + "GPU BA" checkbox (default ON → `ba_use_gpu` / CLI
  `Mapper.ba_use_gpu=1`; Ceres path additionally `BundleAdjustmentCeres.use_gpu=1` where the
  surface exists — CLI verified; pycolmap global-BA surface must be PROBED during Phase C:
  pip 4.1.0 `GlobalPipelineOptions` has NO `.bundle_adjustment` attr [probed 2026-07-16];
  if no surface exists, GPU-BA for global mapping is CLI-backend-only and the GUI says so).
- **Enum fixes:** `FeatureExtractor.ALIKED` → `FeatureExtractorType.ALIKED_N16ROT`;
  `FeatureMatchingType` → `FeatureMatcherType`. Extract plumbing via
  `FeatureExtractionOptions`: `.max_image_size` (top-level) + `.sift.max_num_features`
  [both probed present in 4.1.0]; ALIKED feature-limit surface probed at Phase C
  (no `.aliked` attr in pip 4.1.0 — locate actual surface or document limit as SIFT-only).
- **Backend qualification (per-run, fresh-process):** at run start, qualify each backend in
  a FRESH subprocess (`PYTHONNOUSERSITE` left default for GUI env probe; shim tests use
  `PYTHONNOUSERSITE=1` to exclude user-site) running the contract test against the exact
  interpreter+env the run will use. A backend failing qualification is excluded FOR THE
  SESSION (never mid-run fallback). ERP camera model requires an ERP-qualified backend
  (new CLI or new wheel) — else run refuses to start, naming the incapable backend.
  Wheel/shim changes require app restart (imports cache in `sys.modules`); the GUI does not
  attempt hot-reload.
- **Validation/viewer for EUCM/EQUIRECTANGULAR (v1):** `colmap_validation.py` REFUSES
  geometric validation for camera models it has no projection for (explicit message), never
  silently pinhole-projects. `pointcloud_viewer.py` renders points for any model (model-
  agnostic XYZ/RGB) and SKIPS camera frustums for models without implemented frustum
  geometry (log line). Full ERP/EUCM projection = follow-up plan.
- **New camera models in GUI:** dropdown adds EQUIRECTANGULAR + EUCM under COLMAP engine.
- **SphereSfM UI:** engine radio label → "SphereSfM (legacy)".
- **New CLI install location:** `C:\Users\alexm\COLMAP\COLMAP-<version>-<shortsha>-x64-windows-cuda-cuDSS\`
  — a NEW directory; the three existing installs are untouched. Discovery pref outranks the
  reverse-sorted scan [verified reconstruction_zone.py:314-323], so promotion (Phase E) must
  explicitly update the `alignment_colmap_binary` pref; the build itself changes nothing live.

## Files
CREATE: `tests/test_pycolmap_contract.py` · `docs/plans/PROGRESS.md` (exists).
MODIFY (this repo): `prep360/core/colmap_binary.py` ·
  `reconstruction_gui/colmap_binary_validation_adapter.py` ·
  `reconstruction_gui/colmap_runner.py` · `reconstruction_gui/tabs/alignment_tab.py` ·
  `reconstruction_gui/colmap_validation.py` · `reconstruction_gui/pointcloud_viewer.py` ·
  `reconstruction_gui/reconstruction_zone.py` (discovery source labels + qualification hooks) ·
  existing tests under `tests/` that assert the old ID table or reader signature.
ALLOWED-CHANGE MANIFEST for `D:\Data\build_gpu_colmap` (nothing else there moves):
  submodule gitlinks `third_party/colmap` + `third_party/colmap-for-pycolmap` (recorded:
  both at fa8e3b3f pre-bump; record target SHA before checkout) · `patches/pycolmap-caspar-
  bindings.patch` (delete iff G2 PASS) · the script line(s) applying that patch ·
  `build/` stamps + install tree + `wheelhouse/` (purged for clean rebuild) · provenance
  sidecar `BUILD-RECORD-<date>.md` (new). Preserve any pre-existing dirty state: `git -C`
  status recorded to PROGRESS.md before touching; no commits there (human-only).
DO NOT TOUCH: Mask/Review/Extract tab code · prep360 reframing/extraction ·
  `static_mask_library/` · git commits anywhere (human-only) · live user-site pycolmap until
  Phase E promotion · `D:\Tools\SphereSfM\` · the three existing `~/COLMAP/*` installs.

## Standing rules (restated for any executor — zero context assumed)
- Masks are 0/1 uint8; threshold `mask > 0` never `mask > 127`.
- No popout dialogs; inline collapsible sections only.
- Match existing CustomTkinter patterns; surgical diffs; no unrequested tests/refactors/scope.
- Behavior verification by Claude/human only — exit codes and stdout claims are not proof.
- Windows file reads: explicit encoding="utf-8".

## Budget
≤4 external dispatches · gpt5.6 consults: UNLIMITED (user directive 2026-07-16 — "ignore
the budget for consults, do them freely") · ≤2 correction rounds/phase on dispatched builds.
Dispatch overrun = stop and surface.

## Contract test spec (`tests/test_pycolmap_contract.py`) — exact assertions
Surfaces below verified against pip 4.1.0 probe 2026-07-16; test runs against whichever
pycolmap the target env exposes and is the Phase-D qualification probe.
1. `pycolmap.__version__` importable; record value.
2. `pycolmap.has_cuda is True` (GUI-env qualification only; shim/venv variants may expect False).
3. `CameraModelId` members: PINHOLE, SIMPLE_RADIAL, OPENCV_FISHEYE, EUCM, EQUIRECTANGULAR;
   integer values of EUCM==16, EQUIRECTANGULAR==17; member with value 11 is
   RAD_TAN_THIN_PRISM_FISHEYE (upstream-namespace canary vs fork's SPHERE).
4. `FeatureExtractorType` members SIFT, ALIKED_N16ROT, ALIKED_N32; `FeatureMatcherType`
   members SIFT_BRUTEFORCE, SIFT_LIGHTGLUE, ALIKED_BRUTEFORCE, ALIKED_LIGHTGLUE;
   `pycolmap.FeatureMatchingType` does NOT exist (regression canary for runner bug).
5. `FeatureExtractionOptions()`: has `max_image_size`; `.sift.max_num_features` settable and
   round-trips a written value.
6. `IncrementalPipelineOptions()`: has `ba_use_gpu`, `ba_gpu_index`, `ba_local_backend`,
   `ba_global_backend`, `min_num_matches`, the three `ba_refine_*`; backend attrs accept
   `BundleAdjustmentBackend.CASPAR` and `.CERES`.
7. `BundleAdjustmentBackend` members exactly ⊇ {CERES, CASPAR}.
8. Call-plumbing (monkeypatch `pycolmap.extract_features` / `match_*` / mappers): invoking
   the RUNNER's pycolmap paths with max_features=2048, max_image_size=1000, guided=True,
   backend=CASPAR must show those values in the captured pycolmap call objects — proves the
   adapter passes them, without needing images.

## Phases
- **A0 — Escrow** · Route: coordinator inline → verify: user-site pycolmap 4.0.2 (package +
  dist-info + pycolmap.libs) archived to `D:\Data\pycolmap-wheel-escrow\` with SHA256 manifest;
  repacked as wheel; INSTALL-TESTED in throwaway venv: import OK, has_cuda=True, version 4.0.2.
  FAIL → STOP (see Gates).
- **A (AMENDED v1.2) — Acquire release artifacts** · Route: coordinator inline (Bash,
  checkpointed) · user download approval GIVEN 2026-07-16 ("WORD") · location amended by
  user: NOT D:\Data (curated) — artifacts live under `C:\Users\alexm\COLMAP\` (established
  COLMAP home; precedent: original 4.0.2 wheel lived there per direct_url.json) → steps:
  download to `C:\Users\alexm\COLMAP\release-v4.1.0\`:
  `pycolmap-4.1.0+cuda.cudss-cp314-cp314-win_amd64.whl` (1.21 GB) and
  `COLMAP-4.1.0-windows-2022-CUDA-cuDSS-Caspar.zip` (1.88 GB) + `SHA256SUMS.txt` + the
  cp314 build_info sidecars → verify local SHA256 vs API digests
  (whl 54c61ca5…ecd1699 · zip 795eb85f…37b3b3c) AND vs SHA256SUMS.txt → **G4 test** →
  archive provenance (release API JSON, SHA256SUMS, sidecars, workflow ref + run id if
  published, upstream tag SHA fa8e3b3f) into ACQUISITION-RECORD-2026-07-16.md → unzip CLI
  to NEW dir `C:\Users\alexm\COLMAP\COLMAP-4.1.0-fa8e3b3f-x64-windows-cuda-cuDSS-Caspar\`
  → verify: CLI `feature_extractor --ImageReader.camera_model EQUIRECTANGULAR` passes
  model-exists check; `mapper --help` shows ba_local_backend; cudss DLL present in bin;
  wheel in ISOLATED venv: has_cuda=True + contract-test items 3,4,7 green + ONE REAL CUDA
  operation (extract features on a tiny image set, GPU evidence in log).
  **Pinning rule:** future installs of this wheel come ONLY from the archived artifact by
  hash — a later generic `pip install -U pycolmap` would silently replace it with the
  CPU PyPI wheel (record warning in ACQUISITION-RECORD and PROGRESS).
  **Fallback:** if artifacts fail verification → revert to v1.1 local-build Phase A
  (build kit at D:\Data\build_gpu_colmap retained untouched).
  User checkpoint at end of A.
- **B — Downstream camera-model compatibility** · Route: coordinator inline → verify:
  contract-test fixtures: synthetic upstream model dir (cameras 16,17) parses under
  variant="colmap" with correct names/params; real or synthetic SphereSfM model (11=SPHERE)
  parses under variant="spheresfm"; wrong-variant read RAISES (both directions);
  validation refuses EUCM/ERP geometric validation with explicit message; viewer points-only
  path exercised. Old-run regression: an existing pre-plan run dir still loads (v1 migration
  rule) or refuses with the specified message — no silent misread.
- **C — Adapter + backend controls** · Route: gpt5.6-sol frozen-spec build (effort=medium,
  fence = this plan's DO-NOT-TOUCH + standing rules verbatim) — dispatch only on user go;
  fallback inline → verify (coordinator): full contract test green incl. item 8 plumbing;
  Caspar+OPENCV_FISHEYE db → explicit abort naming model; AUTO+fisheye → CERES with logged
  model set; probe-first resolution of pycolmap global-BA + ALIKED-limit surfaces recorded
  in PROGRESS.md.
- **D — Selector + qualification** · Route: coordinator inline → verify: with a broken-wheel
  shim (PYTHONNOUSERSITE=1 subprocess), qualification excludes pycolmap, CLI qualifies, runs
  proceed backend=cli; ERP run with only-old-CLI qualified → refusal naming the binary;
  qualification results visible in GUI/log.
- **E — E2E matrix → transactional promotion** · Route: coordinator runs, human eyeballs →
  **Matrix (pre-promotion, isolated venv w/ new wheel + new CLI, fixture datasets declared
  in PROGRESS.md with path+SHA256+image count at execution time; numeric bounds set per
  fixture when declared):**
  (1) pinhole+AUTO → log asserts backend=CASPAR, Caspar summary present, residual count>0,
  ≥90% images registered, mean reproj < declared bound; wall-clock recorded.
  (2) same dataset backend=CERES+GPU-BA → solver log shows CUDA/cuDSS evidence line (not
  inference from GPU activity); wall-clock recorded vs (1).
  (3) fisheye(OPENCV_FISHEYE)+explicit CASPAR → abort message; +AUTO → CERES-CUDA completes.
  (4) ERP incremental (EQUIRECTANGULAR) → completes; model parses variant="colmap";
  validation refuses geometrically (v1 rule); viewer shows points.
  (5) reframed-360 pinhole+CASPAR → completes.
  (6) SphereSfM legacy regression → completes; parses variant="spheresfm".
  (7) max_features: count(2048) ≤ 2048 AND count(8192) > count(2048) from database keypoints.
  Matrix item (2) additionally asserts cuDSS-specific solver evidence (cuDSS named in the
  solver log or cudss DLL loaded during BA) — validates the cuda.cudss variant choice.
  **Promotion (only after matrix green):** close GUI → backup prefs file (SHA256) → escrow
  re-verified present → `pip install <new wheel from archived artifact>` into GUI env →
  fresh-process probe of GUI interpreter: version+has_cuda+CASPAR → **pythonw.exe probe
  with user-site ENABLED replicating GUI import order, asserting pycolmap.__file__ + one
  real CUDA op** (Codex A0-finding: clean-venv import ≠ GUI-env import) → set
  `alignment_colmap_binary` pref to new CLI path (old value recorded) → launch GUI → one
  live run per engine (colmap-pinhole-Caspar; spheresfm-legacy) → record run_info v2 fields.
  **Rollback (single documented procedure):** close GUI → `pip install` escrow wheel →
  restore prefs backup → verify fresh-process probe shows 4.0.2+has_cuda=True → launch.
  Covers wheel, pref pin, CLI selection; new CLI dir may remain on disk (inert — pref
  outranks scan).

## Acceptance (observations)
- [ ] New CLI accepts EQUIRECTANGULAR (fails today) and exposes Mapper.ba_local_backend.
- [ ] GUI env post-promotion: has_cuda=True, CASPAR present, EQUIRECTANGULAR present,
      version + pyd SHA256 recorded in run_info of a live run.
- [ ] Matrix items (1)–(7) green with their numeric assertions; wall-clocks for (1) vs (2)
      recorded in PROGRESS.md.
- [ ] Caspar refusal (explicit) and AUTO-fallback (logged model set) both demonstrated.
- [ ] Pre-plan run dir: loads under migration rule or refuses with specified message.
- [ ] Broken-wheel shim → CLI-only session, ERP-on-old-CLI refusal, GUI functional.
- [ ] Rollback procedure executed once end-to-end during Phase E (dry-run before promotion):
      restores 4.0.2 + prefs byte-identical (hash match).
- [ ] SphereSfM legacy run green; engine radio shows "(legacy)".

## Project "done" checks
- [ ] PROGRESS.md: gate branches, SHAs, artifact+prefs hashes, timings, probe results.
- [ ] BUILD-RECORD sidecar written in build repo.
- [ ] ALIGN_TAB.md + align-tab-tooling memory updated to the NEW stack (post-landing).
- [ ] Escrow + new wheel + old prefs backup retained with hashes.

## Handoff prompt → gpt5.6-sol (Phase C only; requires user go)
Assembled at dispatch from Design decisions + Files fence + Standing rules verbatim +
contract-test spec + "report files changed + verification output; no unrequested tests,
refactors, or scope expansion."
