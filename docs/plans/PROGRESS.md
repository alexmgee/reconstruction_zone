# Progress Ledger — Plan 01 (COLMAP Alignment Modernization)

Posture: **GUARDED** (build/install/code by explicit user go; consults free).
Plan: [01-colmap-alignment-modernization.md](01-colmap-alignment-modernization.md)
Dispatch logs: `../../.collab-logs/` (gitignored 2026-07-16).

## Phases
- [x] A0 — Escrow CUDA 4.0.2 wheel (backup + reinstall-test) ✅ 2026-07-16
- [x] A — Acquire 4.1.0 CLI + cp314 wheel (amended: release download) ✅ 2026-07-16
- [x] B — Downstream camera-model compatibility (ID tables, validation/viewer audit) ✅ 2026-07-17 (incl. Codex diff-review fixes)
- [x] C — Adapter fixes + BA backend controls + Caspar gate ✅ 2026-07-17 (Codex build, coordinator-verified)
- [x] D — Backend selector + qualification gate ✅ 2026-07-17 (inline, 14 new tests)
- [x] E — E2E matrix → atomic wheel promotion ✅ 2026-07-17 (7/7 lanes + rehearsal +
  promotion + live acceptance both engines; gpt5.6 final audit dispatched)

## Gates
- G1 cp314 wheel: **ANSWERED 2026-07-16** — prebuilt `pycolmap-4.1.0+cuda.cudss-cp314-cp314-
  win_amd64.whl` (1.21 GB) exists in build repo's GitHub release v4.1.0 (gh API-confirmed).
- G2 Caspar patch: **MOOT on download path** (CI built from 4.1.0 tag; no local patch step).
  Still applies if we ever rebuild locally.
- G3 global-ERP support: **UNRESOLVED** (ships experimental; incremental-ERP is the v1 claim)

## Phase E fixtures (declared 2026-07-17, user-provided; hashes → .collab-logs/phaseE-matrix/fixtures-manifest.json)
- **F1 pinhole:** D:\Capture\garage_test\archive — 430 JPG, ~3972×2655 (dims vary → camera
  mode AUTO, single_camera off), 831 MB. Lanes L1 (Caspar) / L2 (Ceres-GPU timing).
  Bounds: ≥90% registered, mean reproj < 1.5 px.
- **F2 refusal-lane db source:** D:\Capture\chasing_rabbits\fake_head — 200 JPG 3840×2160
  16:9 video frames + masks. Lane L3a only (Caspar refusal via OPENCV_FISHEYE extraction —
  gate is model-driven).
- **F4 production stills+masks:** D:\Capture\chasing_rabbits\chasing_rabbits_fountain —
  174 JPG 6000×4000 (3:2) + 174 reviewed mask PNG (review_status.json present). Lane L3b
  (Ceres-GPU + masks, production workflow). Bounds: ≥90% registered, mean reproj < 1.5 px.
- **Fisheye lane: N/A** — user's library holds no raw fisheye frames (both offered sets
  verified perspective). True-fisheye reconstruction quality stays UNPROVEN; refusal +
  AUTO-fallback proofs are camera-model-driven and fully covered.
- **F3 native ERP:** D:\Capture\testing_fisheye\osmo\ERP\frames — 76 JPG 7680×3840 (2:1)
  + 76 mask PNG, 854 MB. Lanes L4 (EQUIRECTANGULAR incremental) / L5 (reframe→Caspar) /
  L6 (SphereSfM legacy regression). Bounds (L4): ≥80% registered, mean reproj < 2.0 px.
- L7 (max_features 2048 vs 8192 keypoint counts) runs on a 5-image subset of F2.

## Activity log (newest first)
- 2026-07-17 · **FINAL AUDIT (gpt5.6 xhigh-lineage, verdict HOLD) → all findings resolved
  or dispositioned; suite 264/264.** (.collab-logs/codex-final-audit.md)
  F1 provenance-enforcement: FIXED — `_pycolmap_available` now requires has_cuda; CPU
  PyPI wheel is excluded with a logged pointer to the archived artifact (+2 tests).
  F2 evidence gaps: L1 all_ok:false explained + waived (runs/L1/PASS-ADDENDUM.md — cv2
  harness gap, stages green, re-parse 426/430@0.693px); **cuDSS execution PROVEN on the
  CLI backend** (live module snapshot of colmap.exe bundle_adjuster under forced GPU
  routing: cudss64_0.dll + full CUDA family loaded). **Counter-finding: the pycolmap
  wheel BUNDLES cudss but never loads it** (module enumeration under forced routing:
  zero CUDA modules) — likely delvewheel name-mangling breaks Ceres' runtime dlopen →
  in-process Ceres is CPU-only; Caspar is the in-process GPU solver. Documented in GUI
  tooltip; investigation deferred. L7 waiver already ledgered.
  F3: GUI smoke = user's launch step (headless can't drive Tk/VTK); ERP sphere-marker is
  a DOCUMENTED Phase B design decision (correct geometry, same as SPHERE), not a
  points-only violation — plan text deviation noted here.
  F4: FIXED — run_info gains pycolmap_pyd_sha256; BA controls now restore from prefs;
  tooltip states Ceres-GPU/global-GPU are cli-only; global+EQUIRECTANGULAR now logs an
  EXPERIMENTAL warning at run start (G3 promise honored). spheresfm recording
  pycolmap_version = benign environment record, kept.
  F5: untracked junk (`-` [May 15], scripts/x.py, work/, static_mask_library/) is
  PRE-EXISTING, not this unit's. COMMIT CURATION REQUIRED: stage Plan-01 files
  selectively, never `git add -A`.
  F6 deferred list EXPANDED (below).

## Deferred / follow-up (consolidated)
- GUI smoke by user: launch, Align tab, Load Model (ERP + pinhole), Next Step flow,
  cancel mid-run, viewer render of ERP model.
- Ceres-GPU in-process: investigate delvewheel dlopen mangling (wheel bundles cudss but
  never loads it); consider routing large-Ceres jobs to CLI automatically.
- Cross-engine resume rejection: designed (run_info v2) but NOT implemented — implement
  + test.
- G3 global-ERP validation (warning now shown); upstream global panoramic mapping
  integration unvalidated.
- EUCM + true-fisheye real-imagery validation (no fisheye frames in user library).
- Old-run migration proof (no pre-plan run dirs existed; rule untested on real v1 data).
- Doc-debt: ALIGN_TAB.md rewrite; project CLAUDE.md prep360 reframe flags stale
  (--layout removed); colmap skill refresh (3.14→4.1.0 era, new paths).
- 2026-07-17 · **LIVE ACCEPTANCE PASS — PLAN 01 EXECUTION COMPLETE.** Promoted env runs:
  colmap engine 426/430, 0.678px, backend=pycolmap 4.1.0, ba_backend_resolved=CASPAR
  (hybrid), namespace=colmap, run_info schema v2 ✓ · spheresfm legacy 72/76, 1.181px,
  backend=subprocess, namespace=spheresfm, schema v2 ✓. Outstanding: user GUI launch +
  eyeball; deferred doc-debt (ALIGN_TAB.md + CLAUDE.md prep360 reframe flags + colmap
  skill refresh); G3 global-ERP validation = follow-up plan. gpt5.6 final audit → 
  .collab-logs/codex-final-audit.md.
- 2026-07-17 · **ROLLBACK REHEARSAL PASS + PROMOTION EXECUTED** (user go; GUI verified
  closed — no pythonw). Rehearsal: escrow wheel force-reinstalled into REAL user-site →
  fresh probe 4.0.2/has_cuda/GPU ✓, prefs hash byte-identical (3d9295e7…) ✓. Promotion:
  prefs backed up byte-exact to release-v4.1.0/prefs-backup-pre-promotion.json → artifact
  re-hash-verified at install time → `pip install --user` 4.1.0+cuda.cudss → live probes:
  python (4.1.0, CUDA, CASPAR+EQUIRECTANGULAR+EUCM) ✓, **pythonw.exe user-site probe with
  real forced-CUDA extraction PASS** (Codex A0 condition) ✓ → contract test in live env
  **9/9 zero skips** ✓ → full suite **262/262** ✓ → pref `alignment_colmap_binary`
  flipped: OLD `…COLMAP-4.0.3…\colmap.exe` (recorded) → NEW
  `…COLMAP-4.1.0-fa8e3b3f-…-cuDSS-Caspar\bin\colmap.exe` (resolves, reports 4.1.0
  fa8e3b3) ✓. Live acceptance runs (both engines, promoted env) in flight.
  Rollback = documented procedure: escrow wheel + prefs-backup restore.
- 2026-07-17 · **L6 + L7 PASS — MATRIX COMPLETE (7/7 lanes green).**
  L6 (SphereSfM legacy): 72/76, 1.21 px, 50.5s, backend=subprocess, namespace=spheresfm,
  model parses as SPHERE — legacy engine fully working through all new code; native ERP
  (L4) matches its registration with better reproj (1.17 vs 1.21 px).
  L7 (max_features): investigation trail — db rows exceeded cap on GPU (2478>2048),
  orientation-1 hypothesis REFUTED (still over), CPU also over, **CLI parity check:
  IDENTICAL per-image rows [2341,2420,2125,2478,2293] on subprocess path** → upstream
  soft-cap semantics (keypoint truncation pre-orientation-expansion; SiftGPU soft
  target), not an adapter bug. Setting effect proven (2048-run vs 8192-run: 11,657 vs
  14,420 rows). Original acceptance wording (rows ≤ cap) corrected to parity + effect.
  Also: 4.1.0 CLI renamed SiftExtraction.max_image_size → FeatureExtraction.max_image_size
  (runner already handles it; my probe initially used the stale name).
  → REMAINING: rollback rehearsal + promotion (both touch live env — awaiting user go).
- 2026-07-17 · **L5 (reframed-360 → Caspar) PASS.** prep360 reframe: 76 ERP → 1216 pinhole
  views 1920×1920 (16/frame; NOTE: `--layout` flag no longer exists in CLI — project doc
  stale, deferred doc-debt). Deterministic subset frames 1–30 (480 views), exhaustive:
  **475/480 (99%)**, 63,206 pts, **1.10 px**; gate {1} → CASPAR; shipped hybrid ran
  in-pipeline (mapping 220.5s/480 views). Modern 360 path (ERP→reframe→pinhole→Caspar)
  fully proven.
- 2026-07-17 · **L4 (native EQUIRECTANGULAR) PASS — headline capability proven.** 76 ERP
  frames 7680×3840 + masks → **72/76 registered (95%)** (≥80% ✓), 5,725 pts, **1.17 px**
  (<2.0 ✓), 68.6s total. Gate observed {17} → CERES (Caspar-ineligible, correct). Model
  parses under variant="colmap" with cameras=['EQUIRECTANGULAR'] (Phase B namespace holds
  on real output). Validator REFUSED the real ERP model at load with the specified
  message (validate-refuse rule verified on production data, not just fixtures).
  Native 360 no longer requires the SphereSfM fork.
- 2026-07-17 · **L3b (production stills + masks) PASS.** fountain 174 imgs, OPENCV,
  single camera, reviewed masks: **174/174 registered (100%)**, 26,166 pts, **1.19 px**
  (<1.5 ✓); extract 30.3s / match 77.5s / reconstruct 314.3s. AUTO fallback proven live
  (observed {4} → CERES); masks active (174 loads in solver log).
- 2026-07-17 · **L3a (Caspar refusal) PASS.** fake_head × OPENCV_FISHEYE extraction (16.8s)
  + matching (6.4s) OK; explicit CASPAR reconstruction REFUSED in 0.0s with the exact
  specified error naming model [5] and the eligible set {1,2}. No silent observation
  skipping possible.
- 2026-07-17 · **L2 PASS + MAJOR TUNING FINDING → hybrid BA shipped.** L2 (explicit
  CERES+GPU, same 430 imgs): 426/430 (99%), 0.677px, mapping 208.9s — quality equals L1
  but FASTER than both-Caspar's 341.1s. Hypothesis: Caspar GPU-launch overhead loses on
  hundreds of small LOCAL BAs, wins on large GLOBAL BAs. **L1b experiment** (same db,
  mapping only, local=CERES global=CASPAR): **159.4s**, 426/430, 0.677px — 2.1× faster
  than both-Caspar, 24% faster than Ceres, identical quality. CHANGED: resolved CASPAR
  now = global CASPAR + local CERES in both pycolmap and CLI paths (comment carries the
  measurements); contract test + GUI tooltip updated; 260+2 suite green.
  Matrix wall-clocks (mapping, 430 imgs): both-Caspar 341s · Ceres-GPU 209s ·
  **hybrid 159s** ← shipped default when Caspar resolves.
- 2026-07-17 · **L1 (pinhole+Caspar flagship) PASS.** 430 imgs → extract 37.9s (GPU SIFT)
  · match 22.5s (sequential) · reconstruct 341.1s · **426/430 registered (99%)**,
  138,237 pts, **mean reproj 0.693 px** (bounds: ≥90%, <1.5 ✓). Gate: observed {2} →
  AUTO selected CASPAR; run_info v2 recorded backend=pycolmap 4.1.0, namespace=colmap,
  ba_backend_resolved=CASPAR. **Caspar execution PROVEN directly** (runs/L1/
  caspar-execution-proof.md): distinct summary types per backend; same model BA =
  Caspar 0.9s vs Ceres-CPU 6.6s (1.33M residuals, identical 0.693px result). Learned:
  Caspar logs nothing at INFO — engagement evidence = summary type, never log-grep.
  Harness venv needed opencv-python-headless (installed in throwaway venv only).
- 2026-07-17 · **PHASE D COMPLETE** (coordinator inline). colmap_runner.py:
  `qualify_pycolmap_backend()` — fresh-subprocess JSON probe (importable/version/cuda/
  caspar/camera_models), session-cached, runs BEFORE any in-process import so a crashing
  wheel can't take down the GUI; `cli_supports_camera_model()` — per-binary probe via the
  empty-dir feature_extractor trick, session-cached. Runner gains `backend_preference`
  (auto|pycolmap|cli): cli forces subprocess; pycolmap raises if unqualified (no silent
  downgrade); auto falls back to CLI; spheresfm always subprocess.
  `ensure_camera_model_supported()` at start_run: EQUIRECTANGULAR/EUCM refuse BEFORE run-
  dir creation when backend incapable, naming wheel version or binary path.
  reconstruction_zone.py: create_alignment_runner reads pref `alignment_colmap_backend`.
  alignment_tab.py: Backend dropdown in Tool Setup (auto/pycolmap/cli) + tooltip, pref
  persisted, threaded through snapshot → runner. Tests: tests/test_backend_qualification.py
  (14: probe-vs-in-process match, PYTHONNOUSERSITE broken-shim exclusion, session cache,
  all preference semantics, ERP refusal both backends, refusal-precedes-run-dir, capable-
  backend proceeds, pinhole-no-probe, real-CLI PINHOLE integration probe).
  **260 passed + 2 gated skips**; ruff clean on my files (reconstruction_zone.py's 11
  errors pre-existing on HEAD, untouched); GUI modules compile + import.
  → NEXT: Phase E (E2E matrix + promotion) — needs fixture datasets from user.
- 2026-07-17 · **PHASE C COMPLETE** — gpt5.6-sol frozen-spec build returned; coordinator
  verified by artifact: fence CLEAN (all hunks in restricted files are Phase B's own);
  full suite 246 passed + 2 version-gated skips (4.1.0-only items, correct on 4.0.2);
  ruff clean on changed files. Implementation review: enum fixes in;
  FeatureExtractionOptions plumbing in (max_image_size + sift.max_num_features +
  aliked.max_num_features); `_resolve_ba_backend` exactly to spec (explicit CASPAR raises
  on ineligible models {≠1,2}/non-incremental/spheresfm/missing surface; AUTO logs
  observed db model set and falls back CERES; resolved once, never mid-run); CLI path
  plumbs Mapper.ba_use_gpu/ba_gpu_index/ba_local_backend/ba_global_backend; run_info v2
  (schema_version 2, engine, binary_sha256 cached, pycolmap_version, namespace,
  ba_backend_resolved); GUI: BA row + GPU BA checkbox (disabled under spheresfm),
  EQUIRECTANGULAR/EUCM in dropdown, "SphereSfM (legacy)" label, prefs persisted, args
  threaded through run snapshot. Contract test: 9 tests (7 pass, 2 gated) incl.
  call-plumbing via monkeypatch + explicit-CASPAR-rejects-fisheye-db.
  Findings: CUDA wheel HAS FeatureExtractionOptions.aliked.max_num_features (CPU wheel
  does NOT — my earlier probe was CPU; Codex right for target). pycolmap ba_gpu_index is
  STRING-valued. pycolmap global_mapping has no BA-GPU surface → GPU BA for global =
  CLI-only (per plan). → NEXT: Phase D (selector + qualification gate).
- 2026-07-17 · **Phase C DISPATCHED** to gpt5.6-sol (frozen-spec build, effort=medium,
  user go given). Handoff: .collab-logs/phaseC-handoff-prompt.md — 6 tasks: enum fixes,
  extraction-limit plumbing, BA backend controls + Caspar db-gate, run_info v2, GUI
  controls (BA backend dropdown, GPU BA, EQUIRECTANGULAR/EUCM, "SphereSfM (legacy)"),
  contract test. Fence: only colmap_runner.py + alignment_tab.py + new test + probe notes;
  no git writes; no pip installs. Verification after return = coordinator (behavior, not
  transcript claims).
- 2026-07-17 · **Phase B diff review (gpt5.6-sol high) → FIX-BEFORE-PROCEED, 3 findings,
  ALL verified against code and FIXED** (.collab-logs/codex-phaseB-diff-review.md):
  (P1) `_is_spheresfm_like()` returned False pre-validation → now falls back to explicit
  engine/camera-model config (`_binary_requires_sphere_support()`); parse_model gate
  changed `_pycolmap_available()` → `_use_pycolmap()` so SphereSfM models never route
  through pycolmap's upstream namespace. (P1) "wrong variant always raises" overclaim
  softened to trusted-provenance + fail-fast-defense-in-depth in comments/docstrings —
  Codex constructed a coincidentally-aligned 4-camera file that parses under both tables.
  (P2) GeometricValidator now REFUSES unsupported camera models AT LOAD (both loaders,
  `_reject_unsupported_cameras()`) — zero-point EQUIRECTANGULAR model can no longer "pass".
  +2 regression tests (load-time refusal; pre-validation engine detection).
  13/13 variant tests, 239/239 full suite. Codex also independently confirmed: all param
  counts match COLMAP 4.1.0 source; no missed production variant call sites; pycolmap
  model_name compatible; OPENCV_FISHEYE frustum handling unchanged (pre-existing).
- 2026-07-16 · **PHASE B COMPLETE** (code + tests green; adversarial review dispatched).
  colmap_binary.py: upstream table 0–17 (authoritative param counts probed from 4.1.0 wheel
  — CORRECTION vs review: id 11 RAD_TAN_THIN_PRISM_FISHEYE = **16** params, not 12) +
  SPHERESFM_CAMERA_MODELS (11=SPHERE,3) + required keyword `variant` on both readers, no
  default; unknown-id error names the variant. Adapter threads variant; runner passes it
  from `_is_spheresfm_like()` and stamps `camera_model_namespace` into parse output.
  colmap_validation.py: silent "first param is focal" fallback REPLACED with explicit
  ValueError + VALIDATION_SUPPORTED_MODELS={SIMPLE_PINHOLE,PINHOLE,SIMPLE_RADIAL,RADIAL,
  OPENCV} + supports_geometric_validation(). pointcloud_viewer.py: EQUIRECTANGULAR added
  to spherical marker set; unknown models → points render, frustum skipped, logged once.
  Tests: new tests/test_camera_model_variants.py (12 tests: both namespaces parse, wrong-
  variant RAISES both directions, variant required, adapter threads, validation refusals);
  29 existing reader-test call sites updated to variant="colmap". **45/45 targeted,
  237/237 full suite.** Ruff: only pre-existing E741 (untouched line, left per surgical
  rule). Old-run regression: NO pre-plan alignment run dirs exist on disk (workspace pref
  empty; searched) — nothing to migrate; wrong-variant raise covers misread risk.
- 2026-07-16 · **PHASE A COMPLETE** (amended download path; user approved "WORD"; location
  amended by user → `C:\Users\alexm\COLMAP\release-v4.1.0\`; escrow also relocated to
  `C:\Users\alexm\COLMAP\pycolmap-wheel-escrow\` and re-verified 63/63 + wheel hash).
  Both artifacts hash-verified (local == API digest == SHA256SUMS.txt). **G4 RESOLVED:**
  Codex's rename claim factually TRUE (filename +cuda.cudss vs embedded 4.1.0) but pip 25.2
  installs cleanly — no rename needed. Wheel probe ALL GREEN incl. forced-CUDA extraction
  ("Creating SIFT GPU feature extractor", 45,164 kp/3 imgs); CLI probe ALL GREEN: 4.1.0
  fa8e3b3 · cudss64_0.dll · EQUIRECTANGULAR+EUCM ACCEPTED · Mapper.ba_local/global_backend
  present. CLI installed at `~/COLMAP/COLMAP-4.1.0-fa8e3b3f-x64-windows-cuda-cuDSS-Caspar\`
  (inert until pref update at Phase E promotion). Provenance archived in
  ACQUISITION-RECORD-2026-07-16.md (build_info: CUDA 12.8.1, cuDSS 0.7.1.4, caspar=true).
  → NEXT: Phase B (downstream camera-model compatibility).
- 2026-07-16 · **Plan → v1.2.** gpt5.6-sol review of A0 + download amendment
  (.collab-logs/codex-a0-amendment-review.md): A0 internally sound (it independently
  verified 63/63 files + wheel hash + RECORD validity); amendment verdict
  **ADOPT-WITH-CONDITIONS** — all 5 conditions folded into plan v1.2: new gate G4
  (wheel filename-vs-METADATA rename check at download; claim plausible-unverified locally
  — local workflow clone predates release), provenance archive → ACQUISITION-RECORD,
  pythonw+user-site+real-CUDA promotion probe, cuDSS-BA evidence in matrix (2),
  artifact+hash pinning (generic `pip install -U pycolmap` would clobber the custom wheel
  with the CPU PyPI wheel — never upgrade generically).
  API digests recorded: whl cp314 cuda.cudss sha256:54c61ca5dc2458d23656a44004eecd1b12d119
  4a7a1b3555b7c2669bdecd1699 · CLI zip sha256:795eb85f24a2da5b963b722b30f6e650524415eb943d
  af7794408a2ec37b3b3c. SHA256SUMS.txt asset exists as cross-check.
  **AWAITING user download approval (~3.1 GB) to execute amended Phase A.**
- 2026-07-16 · **A0 COMPLETE.** Escrow at `D:\Data\pycolmap-wheel-escrow\`: (1) site-packages
  copy of pycolmap/, pycolmap.libs/, dist-info — 63 files, SHA256 manifest, 0 mismatches;
  (2) repacked installable wheel `pycolmap-4.0.2-cp314-cp314-win_amd64.whl` (1.12 GB,
  sha256 425c783b…3ec2d9) — install-tested in fresh venv w/ PYTHONNOUSERSITE=1: import from
  venv confirmed, v4.0.2, has_cuda=True, 1 CUDA device. Rollback path EXISTS.
- 2026-07-16 · **Provenance discovery** (dist-info/direct_url.json + DELVEWHEEL): live wheel
  was installed from `C:\Users\alexm\COLMAP\pycolmap-4.0.2+cuda-...whl` (file now GONE;
  recorded sha256 e3132c19…) and was built by **lyehe/build_gpu_colmap GitHub Actions CI**
  (delvewheel 1.12.0, CUDA 12.8, hostedtoolcache py3.14.3).
- 2026-07-16 · **Phase-A alternative found:** build repo's GitHub releases include
  **COLMAP Build v4.1.0 (2026-06-26)**: Windows CUDA+Caspar CLI zip (1.69 GB) + pycolmap
  4.1.0+cuda wheels py3.10–3.14. Could replace the local rebuild with CI artifacts of the
  SAME provenance as the proven 4.0.2 wheel. Download requires user approval. G1 (cp314)
  effectively answered by CI matrix; cuDSS-in-release + exact cp314 asset name to verify.
- 2026-07-16 · Plan v1.1: gpt5.6-sol xhigh ledger interrogation (verdict EXECUTE-AFTER-EDITS,
  8 findings, .collab-logs/codex-plan-ledger-interrogation.md) folded in: transactional
  promotion+rollback, allowed-change manifest for build repo, contract-test spec from
  probed surfaces, pre-mapping Caspar gate, run_info schema v2 + migration, G3 deferred,
  validate-refuse/points-only rule for EUCM/ERP, deterministic FAIL branches. Two Codex
  claims corrected by probe (see findings). Awaiting go for Phase A0.
- 2026-07-16 · Plan + ledger written; user locked decisions (Claude drives build; SphereSfM
  legacy-marked pending ERP validation; Caspar ships gated; both 360 paths).
- 2026-07-16 · gpt5.6-sol xhigh plan review (.collab-logs/codex-plan-v2-review.md): 7
  findings; #3 ID-collision and #2 build-tree risks VERIFIED against artifacts; "atomic
  promotion gate replaces early wheel install" adopted (Phase E).
- 2026-07-16 · Verified: PyPI Windows pycolmap = CPU-only (isolated venv, no CUDA DLLs);
  live 4.0.2 CUDA wheel is custom, irreplaceable, wheelhouse gone → escrow phase added.
- 2026-07-16 · Verified: pip 4.1.0 has CASPAR + EQUIRECTANGULAR + EUCM; on-disk CLI dev2
  (2026-05-10) rejects all three camera models → rebuild is the freshness root.
- 2026-07-16 · gpt5.6-sol initial review (.collab-logs/codex-last.md): runner enum bugs +
  ba_use_gpu-never-set CONFIRMED by direct probes.

## Findings (durable)
- `prep360/core/colmap_binary.py:24-37` fork ID table: 11=SPHERE(3) vs upstream
  11=RAD_TAN_THIN_PRISM_FISHEYE, 16=EUCM, 17=EQUIRECTANGULAR. Unknown IDs raise; model-11
  misreads silently. Namespace design pre-resolved in plan.
- `build_gpu_colmap`: TWO colmap submodules (colmap, colmap-for-pycolmap) currently both at
  fa8e3b3f — must bump in lockstep. `patches/pycolmap-caspar-bindings.patch` (163 ln) adds
  Caspar pybind bindings; upstream now ships these → expect retire at G2.
- Caspar: PINHOLE/SIMPLE_RADIAL only; other models' observations SILENTLY SKIPPED — gate on
  db+mode tuple, assert backend + residual coverage in logs.
- Runner today: ba_use_gpu never set (CPU BA everywhere); `FeatureExtractor.ALIKED` and
  `FeatureMatchingType` don't exist in any pycolmap ≥4.0.2; pycolmap extract path drops
  max_features/max_image_size.
- pip rollback trap: `pip install pycolmap==4.0.2` restores the CPU wheel, NOT the custom
  CUDA wheel. Escrow is the only rollback.
- **Prefs outrank all discovery** (reconstruction_zone.py:314 — prefs before env/PATH/scan),
  and live prefs PIN `alignment_colmap_binary` = COLMAP-4.0.3, `alignment_spheresfm_binary`
  = D:\Tools legacy. GUI is NOT on 4.1.0-dev2 today; promotion must update the pref or the
  rebuild changes nothing live. (Codex finding 1 — CONFIRMED live.)
- Codex overstatements corrected by 4.1.0-venv probe: `GlobalPipelineOptions` has NO
  `.bundle_adjustment` attr; `FeatureExtractionOptions` has NO `.aliked` attr (`.sift.
  max_num_features` + top-level `.max_image_size` DO exist). Contract test specs from
  probed surfaces; global-BA + ALIKED-limit surfaces = probe-first at Phase C.

## Needs Claude / user
- USER GO required to start Phase A0 (escrow) → then A (rebuild, ~30–90 min).
- If G1 FAILS (no cp314): choose script patch vs 3.12-venv route (branch pre-written in plan).
