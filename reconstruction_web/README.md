# Reconstruction Zone — Local Web Interface

A local-only web interface for Reconstruction Zone. It runs a small Python server (standard library
only — no extra runtime dependencies) that serves a single-page monitoring shell and a read-only JSON
API, bound strictly to `127.0.0.1` on your machine. Nothing is exposed to the network and nothing is
fetched from the internet.

**What it is today (Phase 1):** the foundation — a typed React shell that shows server health,
version, and a live job panel, plus the API skeleton the future app builds on. It is deliberately
read-only in the browser. **What it will become:** the Project Control Room — a project dashboard
that answers "what state is this project in and what should happen next," with workflow modules
(Extract, Mask review, Align) and a 3D viewer. See "Where this is going" below.

---

## Quick start

1. **Create a state directory** (anywhere *outside* the source tree, your home directory, and
   production data — the server refuses unsafe locations):

   ```powershell
   mkdir D:\rz-web-state
   ```

2. **Start the server** from the repository root:

   ```powershell
   python -m reconstruction_web --state-root D:\rz-web-state
   ```

   You should see: `reconstruction_web listening on http://127.0.0.1:8765`

3. **Open** <http://127.0.0.1:8765/> in any browser on the same machine.

4. **Stop** with `Ctrl+C`.

## What you'll see

Two views, switched by the header links (hash routing — `#status` and `#jobs`):

- **Runtime status** — a health card (service up, local-only mode, state configured) refreshed every
  5 seconds, and a version card fetched once.
- **Representative jobs** — a read-only list of jobs with a detail pane (progress bar, timestamps,
  step, result/error, logs) refreshed every 2 seconds. Click a job (or use the keyboard) to select it.

If the server goes away, the shell keeps the last data visible and marks it **Stale.** with a note
that the API is unavailable; when the server returns it announces "Local API reconnected." Polling
pauses automatically while the tab is hidden and resumes when you return.

### Seeing the job panel do something

"Representative" jobs are synthetic 20-step demo jobs — they prove the job pipeline (queue, progress,
logs, failure, cancellation) before real reconstruction work lands. The Phase-1 UI has no create
button, so a fresh server shows an empty list. Create one from a terminal:

```powershell
# a job that completes
Invoke-RestMethod -Method Post -ContentType "application/json" `
  -Body '{"job_type": "representative", "behavior": "complete"}' http://127.0.0.1:8765/api/jobs

# a job that fails halfway (to see error + log rendering)
Invoke-RestMethod -Method Post -ContentType "application/json" `
  -Body '{"job_type": "representative", "behavior": "fail"}' http://127.0.0.1:8765/api/jobs
```

Watch the browser: the list and detail update as the job runs.

## Command-line reference

```text
python -m reconstruction_web --state-root <dir> [--port N] [--root label=path ...] [--project-store file.json]
```

| Option | Meaning |
|---|---|
| `--state-root <dir>` | **Required.** Existing directory for web-track state. Refused if it is the source tree, a filesystem root, your home directory, or a known production location. |
| `--port N` | Localhost port (default `8765`; `0` = OS-assigned). |
| `--root label=path` | Register a read-only content root, exposed via the file-listing API as an opaque token (repeatable). |
| `--project-store file.json` | An existing project-store JSON file for the read-only projects API. |

## API summary

All endpoints are same-origin JSON under `/api/`. The browser shell consumes only the first four.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Service health (shell: 5 s poll) |
| `/api/version` | GET | Service version (shell: fetched once) |
| `/api/jobs` | GET | Job list (shell: 2 s poll) |
| `/api/jobs/{id}` | GET | Job detail + logs (shell: 2 s poll while selected) |
| `/api/jobs` | POST | Create a representative job (`{"job_type","behavior"}`; no UI yet) |
| `/api/jobs/{id}/cancel` | POST | Request cancellation (no UI yet) |
| `/api/roots`, `/api/files/list`, `/api/files/stat` | GET | Token-scoped read-only file browsing over `--root` registrations |
| `/api/projects`, `/api/projects/{id}` | GET | Read-only project summaries from `--project-store` |

The four shell endpoints have authoritative JSON Schemas in `frontend/contracts/`; a contract test
validates real server responses against them.

## Security model (why it's shaped this way)

- **Loopback only.** The server binds `127.0.0.1` and validates the `Host` header (DNS-rebinding
  defense). There is no remote access and none is planned.
- **One compiled-in page.** The entire UI is a single self-contained HTML file embedded in the
  Python package (`shell_html.py`) and served only at exactly `GET /`. There is no static-file
  serving, no directory handler, and no external asset request — a strict Content-Security-Policy
  enforces this in the browser.
- **Read-only browser.** Phase 1 ships no mutating UI. The write endpoints above exist for tooling
  and tests; browser-initiated mutation is gated behind a separate CSRF/origin-hardening step.
- **State isolation.** The server refuses to run against production state; everything it writes goes
  under your explicit `--state-root`.

## Development (changing the UI)

The UI source lives in `frontend/` (TypeScript + React + Vite, exact-pinned toolchain). The server
never runs Node — the build output is committed as `reconstruction_web/shell_html.py`.

```powershell
cd frontend
npm ci --ignore-scripts
npm test                # Vitest unit/component suite
npm run build           # vite build -> verify-single-file -> regenerate shell_html.py
npm run test:e2e        # one Playwright test against the real Python server
```

Live development loop (rebuild + regenerate + restart the real server on every change):

```powershell
node frontend/scripts/watch-shell.mjs --state-root D:\rz-web-state --port 8765
```

Python-side tests: `python -m pytest tests/test_web_*.py -q` (the full suite and the frontend gates
also run in CI on every push to `main` and `main-web`).

Never hand-edit `reconstruction_web/shell_html.py` or `frontend/src/api/generated.ts` — both are
generated (`npm run build` / `npm run codegen`).

## Where this is going

The product direction is the **Project Control Room** (see
`docs/planning/pointport_integration/web/WEB1-architecture-decision-record.md` §12 and
`WEB1-ADR-ADDENDUM-A-frontend-build.md` §15): a project dashboard over real data — media, frame
sets, masks, review state, alignment runs, coverage, exports, jobs — followed by read-only Review
triage, the first narrow mutation, real workflow jobs (extraction, masking, alignment), and an
interactive 3D viewer. Each step lands as its own reviewed packet; this Phase-1 shell is the typed,
tested, CI-guarded foundation they build on.
