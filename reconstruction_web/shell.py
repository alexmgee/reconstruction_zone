"""Compiled-in browser shell for the localhost-only web service."""

from typing import Final


SHELL_HTML: Final[bytes] = b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reconstruction Zone Local</title>
  <link rel="icon" href="data:,">
  <style>
    :root {
      color-scheme: dark;
      font-family: system-ui, sans-serif;
      background: #10161d;
      color: #e7edf4;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 18rem; background: #10161d; }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      padding: 1rem clamp(1rem, 4vw, 3rem);
      border-bottom: 1px solid #314052;
      background: #16212c;
    }
    .header-row, nav, .split, .job-heading, .status-line {
      display: flex;
      align-items: center;
      gap: .75rem;
    }
    .header-row { justify-content: space-between; flex-wrap: wrap; }
    .product { margin: 0; font-size: 1.2rem; }
    .badge, .pill {
      display: inline-block;
      padding: .2rem .55rem;
      border: 1px solid #56708b;
      border-radius: 999px;
      color: #c9def2;
      font-size: .78rem;
      letter-spacing: .03em;
    }
    nav { margin-top: .8rem; }
    nav a {
      padding: .45rem .7rem;
      border-radius: .4rem;
      color: #b8c9da;
      text-decoration: none;
    }
    nav a[aria-current="page"] { background: #27445d; color: #fff; }
    a:focus-visible, button:focus-visible { outline: 3px solid #79b8ef; outline-offset: 2px; }
    main { max-width: 76rem; margin: 0 auto; padding: 1.5rem clamp(1rem, 4vw, 3rem) 3rem; }
    h2, h3 { margin-top: 0; }
    .muted { color: #9fb0c2; }
    .status-line { min-height: 1.6rem; color: #b8c9da; }
    .status-line.error { color: #ffb3ad; }
    .status-line.stale { color: #ffd88a; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }
    .card {
      padding: 1rem;
      border: 1px solid #314052;
      border-radius: .7rem;
      background: #18232f;
    }
    dl { margin: 0; }
    .field {
      display: grid;
      grid-template-columns: minmax(7rem, 1fr) 1.4fr;
      gap: .75rem;
      padding: .35rem 0;
      border-bottom: 1px solid #263646;
    }
    .field:last-child { border-bottom: 0; }
    dt { color: #9fb0c2; }
    dd { margin: 0; overflow-wrap: anywhere; }
    .split { align-items: stretch; margin-top: 1rem; }
    .job-list { flex: 0 1 23rem; }
    .job-detail { flex: 1 1 32rem; }
    #job-list { display: grid; gap: .55rem; }
    .job-button {
      width: 100%;
      padding: .75rem;
      border: 1px solid #40556a;
      border-radius: .55rem;
      background: #1c2a37;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }
    .job-button.selected { border-color: #79b8ef; background: #223b50; }
    .job-heading { justify-content: space-between; }
    .job-id { margin-top: .3rem; color: #9fb0c2; font-family: ui-monospace, monospace; font-size: .8rem; }
    progress { width: 100%; height: .8rem; accent-color: #64c28b; }
    .tone-good { border-color: #4f966d; color: #a9e7bf; }
    .tone-warn { border-color: #aa7e32; color: #ffda91; }
    .tone-bad { border-color: #9e514c; color: #ffb3ad; }
    .tone-neutral { border-color: #56708b; color: #c9def2; }
    .logs { max-height: 18rem; overflow: auto; margin: .75rem 0 0; padding-left: 1.25rem; }
    .logs li { margin: .4rem 0; overflow-wrap: anywhere; }
    .log-warning { color: #ffda91; }
    .log-error { color: #ffb3ad; }
    [hidden] { display: none !important; }
    @media (max-width: 48rem) {
      .split { flex-direction: column; }
      .field { grid-template-columns: 1fr; gap: .15rem; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-row">
      <div class="header-row">
        <h1 class="product">Reconstruction Zone</h1>
        <span class="badge">LOCAL ONLY</span>
      </div>
      <div class="muted">Last successful refresh: <span id="last-refresh">Never</span></div>
    </div>
    <nav aria-label="Local views">
      <a id="status-link" href="#status">Runtime status</a>
      <a id="jobs-link" href="#jobs">Representative jobs</a>
    </nav>
  </header>
  <main>
    <section id="status-view" aria-labelledby="status-heading">
      <h2 id="status-heading">Runtime status</h2>
      <p id="health-status" class="status-line" aria-live="polite">Loading local status...</p>
      <div class="cards">
        <section class="card" aria-labelledby="health-heading">
          <h3 id="health-heading">Local service</h3>
          <dl id="health-fields" aria-busy="true"></dl>
        </section>
        <section class="card" aria-labelledby="version-heading">
          <h3 id="version-heading">Version</h3>
          <p id="version-status" class="status-line">Loading version...</p>
          <dl id="version-fields" aria-busy="true"></dl>
        </section>
      </div>
    </section>

    <section id="jobs-view" aria-labelledby="jobs-heading" hidden>
      <h2 id="jobs-heading">Representative jobs</h2>
      <p id="jobs-status" class="status-line" aria-live="polite">Loading jobs...</p>
      <div class="split">
        <section class="card job-list" aria-labelledby="retained-heading">
          <h3 id="retained-heading">Retained jobs</h3>
          <div id="job-list" aria-busy="true"></div>
        </section>
        <section id="job-detail" class="card job-detail" aria-labelledby="detail-heading" hidden>
          <h3 id="detail-heading">Selected job</h3>
          <p id="detail-status" class="status-line">Loading job detail...</p>
          <progress id="job-progress" max="100" value="0">0%</progress>
          <dl id="detail-fields"></dl>
          <h3>Logs</h3>
          <ol id="job-logs" class="logs"></ol>
        </section>
      </div>
    </section>
    <p id="live-feedback" class="muted" aria-live="polite"></p>
  </main>
  <script>
    "use strict";

    const HEALTH_INTERVAL_MS = 5000;
    const JOB_INTERVAL_MS = 2000;
    const FETCH_TIMEOUT_MS = 5000;
    const HEALTH_PATH = "/api/health";
    const VERSION_PATH = "/api/version";
    const JOBS_PATH = "/api/jobs";
    const EMPTY_JOBS_TEXT = "No representative jobs are available. WEB5 is read-only; jobs must be created outside this page.";
    const UNAVAILABLE_TEXT = "Local API unavailable. Is the localhost server still running?";
    const UNEXPECTED_TEXT = "Local API returned an unexpected response.";
    const JOB_STATES = ["queued", "running", "cancel_requested", "completed", "failed", "cancelled"];
    const STATE_CLASSES = {
      queued: "tone-neutral",
      running: "tone-neutral",
      cancel_requested: "tone-warn",
      completed: "tone-good",
      failed: "tone-bad",
      cancelled: "tone-warn"
    };
    const LOG_CLASSES = { info: "", warning: "log-warning", error: "log-error" };

    const elements = {
      statusView: document.getElementById("status-view"),
      jobsView: document.getElementById("jobs-view"),
      statusLink: document.getElementById("status-link"),
      jobsLink: document.getElementById("jobs-link"),
      lastRefresh: document.getElementById("last-refresh"),
      healthStatus: document.getElementById("health-status"),
      healthFields: document.getElementById("health-fields"),
      versionStatus: document.getElementById("version-status"),
      versionFields: document.getElementById("version-fields"),
      jobsStatus: document.getElementById("jobs-status"),
      jobList: document.getElementById("job-list"),
      jobDetail: document.getElementById("job-detail"),
      detailStatus: document.getElementById("detail-status"),
      detailFields: document.getElementById("detail-fields"),
      jobProgress: document.getElementById("job-progress"),
      jobLogs: document.getElementById("job-logs"),
      liveFeedback: document.getElementById("live-feedback")
    };

    let healthData = null;
    let versionData = null;
    let jobSummaries = null;
    let jobDetailData = null;
    let selectedJobId = null;
    let healthTimer = null;
    let jobTimer = null;
    let healthInFlight = false;
    let jobsInFlight = false;
    let versionInFlight = false;
    let healthRestartRequested = false;
    let jobsRestartRequested = false;
    let pollingStopped = false;
    let hadConnectivityFailure = false;
    const controllers = new Set();

    function isObject(value) {
      return typeof value === "object" && value !== null && !Array.isArray(value);
    }

    function isStringOrNull(value) {
      return typeof value === "string" || value === null;
    }

    function normalizeHealth(value) {
      if (!isObject(value) || typeof value.ok !== "boolean" ||
          typeof value.service !== "string" || typeof value.mode !== "string" ||
          typeof value.local_only !== "boolean" || typeof value.state_configured !== "boolean") {
        return null;
      }
      return {
        ok: value.ok,
        service: value.service,
        mode: value.mode,
        local_only: value.local_only,
        state_configured: value.state_configured
      };
    }

    function normalizeVersion(value) {
      if (!isObject(value) || typeof value.service !== "string" ||
          typeof value.version !== "string" || typeof value.version_source !== "string") {
        return null;
      }
      return {
        service: value.service,
        version: value.version,
        version_source: value.version_source
      };
    }

    function normalizeSummary(value) {
      if (!isObject(value) || typeof value.job_id !== "string" ||
          !/^[0-9a-f]{32}$/.test(value.job_id) || typeof value.job_type !== "string" ||
          typeof value.state !== "string" || !JOB_STATES.includes(value.state) ||
          typeof value.progress !== "number" || !Number.isFinite(value.progress) ||
          !isStringOrNull(value.current_step) || typeof value.created_at !== "string" ||
          !isStringOrNull(value.started_at) || !isStringOrNull(value.finished_at)) {
        return null;
      }
      return {
        job_id: value.job_id,
        job_type: value.job_type,
        state: value.state,
        progress: Math.min(100, Math.max(0, Math.round(value.progress))),
        current_step: value.current_step,
        created_at: value.created_at,
        started_at: value.started_at,
        finished_at: value.finished_at
      };
    }

    function normalizeJobs(value) {
      if (!isObject(value) || !Array.isArray(value.jobs)) {
        return null;
      }
      const jobs = value.jobs.map(normalizeSummary);
      return jobs.every((job) => job !== null) ? jobs : null;
    }

    function normalizeLog(value) {
      if (!isObject(value) || typeof value.time !== "string" ||
          typeof value.level !== "string" || !(value.level in LOG_CLASSES) ||
          typeof value.code !== "string" || typeof value.message !== "string") {
        return null;
      }
      return { time: value.time, level: value.level, code: value.code, message: value.message };
    }

    function normalizeResult(value) {
      if (value === null) {
        return null;
      }
      if (!isObject(value) || typeof value.kind !== "string" ||
          typeof value.steps_completed !== "number" || !Number.isInteger(value.steps_completed)) {
        return undefined;
      }
      return { kind: value.kind, steps_completed: value.steps_completed };
    }

    function normalizeError(value) {
      if (value === null) {
        return null;
      }
      if (!isObject(value) || typeof value.code !== "string" || typeof value.message !== "string") {
        return undefined;
      }
      return { code: value.code, message: value.message };
    }

    function normalizeDetail(value) {
      const summary = normalizeSummary(value);
      if (summary === null || !Array.isArray(value.logs)) {
        return null;
      }
      const logs = value.logs.map(normalizeLog);
      const result = normalizeResult(value.result);
      const error = normalizeError(value.error);
      if (!logs.every((log) => log !== null) || result === undefined || error === undefined) {
        return null;
      }
      return {
        job_id: summary.job_id,
        job_type: summary.job_type,
        state: summary.state,
        progress: summary.progress,
        current_step: summary.current_step,
        created_at: summary.created_at,
        started_at: summary.started_at,
        finished_at: summary.finished_at,
        logs: logs,
        result: result,
        error: error
      };
    }

    function appendField(list, label, value) {
      const row = document.createElement("div");
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      row.className = "field";
      term.textContent = label;
      description.textContent = value;
      row.replaceChildren(term, description);
      list.append(row);
    }

    function booleanText(value) {
      return value ? "Yes" : "No";
    }

    function optionalText(value) {
      return value === null ? "Not available" : value;
    }

    function setRegionStatus(element, text, kind) {
      element.textContent = text;
      element.className = "status-line" + (kind ? " " + kind : "");
    }

    function renderHealth() {
      if (healthData === null) {
        return;
      }
      const fragment = document.createDocumentFragment();
      appendField(fragment, "Healthy", booleanText(healthData.ok));
      appendField(fragment, "Service", healthData.service);
      appendField(fragment, "Mode", healthData.mode);
      appendField(fragment, "Local only", booleanText(healthData.local_only));
      appendField(fragment, "State configured", booleanText(healthData.state_configured));
      elements.healthFields.replaceChildren(fragment);
      elements.healthFields.setAttribute("aria-busy", "false");
    }

    function renderVersion() {
      if (versionData === null) {
        return;
      }
      const fragment = document.createDocumentFragment();
      appendField(fragment, "Service", versionData.service);
      appendField(fragment, "Version", versionData.version);
      appendField(fragment, "Source", versionData.version_source);
      elements.versionFields.replaceChildren(fragment);
      elements.versionFields.setAttribute("aria-busy", "false");
    }

    function renderJobList() {
      const fragment = document.createDocumentFragment();
      if (jobSummaries !== null) {
        jobSummaries.forEach((job) => {
          const button = document.createElement("button");
          const heading = document.createElement("span");
          const state = document.createElement("span");
          const identifier = document.createElement("span");
          button.type = "button";
          button.className = "job-button" + (job.job_id === selectedJobId ? " selected" : "");
          heading.className = "job-heading";
          heading.textContent = job.job_type;
          state.className = "pill " + STATE_CLASSES[job.state];
          state.textContent = job.state;
          identifier.className = "job-id";
          identifier.textContent = job.job_id;
          heading.append(state);
          button.replaceChildren(heading, identifier);
          button.addEventListener("click", () => selectJob(job.job_id));
          fragment.append(button);
        });
      }
      elements.jobList.replaceChildren(fragment);
      elements.jobList.setAttribute("aria-busy", "false");
    }

    function renderDetail() {
      if (jobDetailData === null) {
        return;
      }
      const detail = jobDetailData;
      const fields = document.createDocumentFragment();
      appendField(fields, "Job ID", detail.job_id);
      appendField(fields, "Type", detail.job_type);
      appendField(fields, "State", detail.state);
      appendField(fields, "Progress", String(detail.progress) + "%");
      appendField(fields, "Current step", optionalText(detail.current_step));
      appendField(fields, "Created", detail.created_at);
      appendField(fields, "Started", optionalText(detail.started_at));
      appendField(fields, "Finished", optionalText(detail.finished_at));
      if (detail.result !== null) {
        appendField(fields, "Result kind", detail.result.kind);
        appendField(fields, "Steps completed", String(detail.result.steps_completed));
      }
      if (detail.error !== null) {
        appendField(fields, "Error code", detail.error.code);
        appendField(fields, "Error", detail.error.message);
      }
      elements.detailFields.replaceChildren(fields);
      elements.jobProgress.value = detail.progress;
      elements.jobProgress.textContent = String(detail.progress) + "%";
      const logs = document.createDocumentFragment();
      detail.logs.forEach((log) => {
        const item = document.createElement("li");
        item.className = LOG_CLASSES[log.level];
        item.textContent = log.time + " [" + log.level + "] " + log.code + ": " + log.message;
        logs.append(item);
      });
      elements.jobLogs.replaceChildren(logs);
      elements.jobDetail.hidden = false;
    }

    function updateLastRefresh() {
      elements.lastRefresh.textContent = new Date().toLocaleTimeString();
      if (hadConnectivityFailure) {
        elements.liveFeedback.textContent = "Local API reconnected.";
        hadConnectivityFailure = false;
      }
    }

    function failureText(failure) {
      if (isObject(failure) && Number.isInteger(failure.httpStatus)) {
        return "Local API returned HTTP " + String(failure.httpStatus) + ".";
      }
      if (isObject(failure) && failure.unexpected === true) {
        return UNEXPECTED_TEXT;
      }
      hadConnectivityFailure = true;
      return UNAVAILABLE_TEXT;
    }

    async function requestJson(path) {
      const controller = new AbortController();
      controllers.add(controller);
      const timeout = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
      try {
        const response = await fetch(path, {
          method: "GET",
          cache: "no-store",
          credentials: "omit",
          signal: controller.signal
        });
        if (!response.ok) {
          throw { httpStatus: response.status };
        }
        try {
          return await response.json();
        } catch (failure) {
          if (controller.signal.aborted) {
            throw { unavailable: true };
          }
          throw { unexpected: true };
        }
      } finally {
        window.clearTimeout(timeout);
        controllers.delete(controller);
      }
    }

    async function loadVersion() {
      if (versionInFlight) {
        return;
      }
      versionInFlight = true;
      setRegionStatus(elements.versionStatus, versionData === null ? "Loading version..." : "Refreshing...", "");
      elements.versionFields.setAttribute("aria-busy", "true");
      try {
        const value = normalizeVersion(await requestJson(VERSION_PATH));
        if (value === null) {
          throw { unexpected: true };
        }
        versionData = value;
        renderVersion();
        setRegionStatus(elements.versionStatus, "Version information is current.", "");
        updateLastRefresh();
      } catch (failure) {
        if (pollingStopped || document.hidden) {
          return;
        }
        setRegionStatus(elements.versionStatus, failureText(failure) + (versionData === null ? "" : " Stale."), versionData === null ? "error" : "stale");
      } finally {
        versionInFlight = false;
        elements.versionFields.setAttribute("aria-busy", "false");
      }
    }

    async function runHealthCycle() {
      if (pollingStopped || document.hidden || healthInFlight) {
        return;
      }
      healthInFlight = true;
      setRegionStatus(elements.healthStatus, healthData === null ? "Loading local status..." : "Refreshing...", "");
      elements.healthFields.setAttribute("aria-busy", "true");
      try {
        const value = normalizeHealth(await requestJson(HEALTH_PATH));
        if (value === null) {
          throw { unexpected: true };
        }
        healthData = value;
        renderHealth();
        setRegionStatus(elements.healthStatus, value.ok ? "Local service is healthy." : "Local service reported unhealthy.", value.ok ? "" : "error");
        updateLastRefresh();
        if (versionData === null) {
          await loadVersion();
        }
      } catch (failure) {
        if (!pollingStopped && !document.hidden) {
          setRegionStatus(elements.healthStatus, failureText(failure) + (healthData === null ? "" : " Stale."), healthData === null ? "error" : "stale");
        }
      } finally {
        healthInFlight = false;
        elements.healthFields.setAttribute("aria-busy", "false");
        if (!pollingStopped && !document.hidden) {
          const delay = healthRestartRequested ? 0 : HEALTH_INTERVAL_MS;
          healthRestartRequested = false;
          healthTimer = window.setTimeout(runHealthCycle, delay);
        }
      }
    }

    function selectedSummaryExists() {
      return jobSummaries !== null && jobSummaries.some((job) => job.job_id === selectedJobId);
    }

    async function loadSelectedDetail() {
      if (selectedJobId === null || !/^[0-9a-f]{32}$/.test(selectedJobId)) {
        return;
      }
      setRegionStatus(elements.detailStatus, jobDetailData === null ? "Loading job detail..." : "Refreshing detail...", "");
      elements.jobDetail.hidden = false;
      try {
        const path = JOBS_PATH + "/" + encodeURIComponent(selectedJobId);
        const value = normalizeDetail(await requestJson(path));
        if (value === null || value.job_id !== selectedJobId) {
          throw { unexpected: true };
        }
        jobDetailData = value;
        renderDetail();
        setRegionStatus(elements.detailStatus, "Job detail is current.", "");
        updateLastRefresh();
      } catch (failure) {
        if (pollingStopped || document.hidden) {
          return;
        }
        if (isObject(failure) && failure.httpStatus === 404) {
          setRegionStatus(elements.detailStatus, "Job detail is no longer available. Refreshing list...", jobDetailData === null ? "error" : "stale");
        } else {
          setRegionStatus(elements.detailStatus, failureText(failure) + (jobDetailData === null ? "" : " Stale."), jobDetailData === null ? "error" : "stale");
        }
      }
    }

    async function runJobsCycle() {
      if (pollingStopped || document.hidden || jobsInFlight) {
        return;
      }
      jobsInFlight = true;
      setRegionStatus(elements.jobsStatus, jobSummaries === null ? "Loading jobs..." : "Refreshing...", "");
      elements.jobList.setAttribute("aria-busy", "true");
      try {
        const summaries = normalizeJobs(await requestJson(JOBS_PATH));
        if (summaries === null) {
          throw { unexpected: true };
        }
        jobSummaries = summaries;
        if (!selectedSummaryExists()) {
          selectedJobId = summaries.length === 0 ? null : summaries[0].job_id;
          jobDetailData = null;
        }
        renderJobList();
        updateLastRefresh();
        if (summaries.length === 0) {
          elements.jobDetail.hidden = true;
          setRegionStatus(elements.jobsStatus, EMPTY_JOBS_TEXT, "");
          elements.liveFeedback.textContent = EMPTY_JOBS_TEXT;
        } else {
          setRegionStatus(elements.jobsStatus, "Retained jobs are current.", "");
          await loadSelectedDetail();
        }
      } catch (failure) {
        if (!pollingStopped && !document.hidden) {
          setRegionStatus(elements.jobsStatus, failureText(failure) + (jobSummaries === null ? "" : " Stale."), jobSummaries === null ? "error" : "stale");
        }
      } finally {
        jobsInFlight = false;
        elements.jobList.setAttribute("aria-busy", "false");
        if (!pollingStopped && !document.hidden) {
          const delay = jobsRestartRequested ? 0 : JOB_INTERVAL_MS;
          jobsRestartRequested = false;
          jobTimer = window.setTimeout(runJobsCycle, delay);
        }
      }
    }

    function selectJob(jobId) {
      if (!/^[0-9a-f]{32}$/.test(jobId) || jobId === selectedJobId) {
        return;
      }
      selectedJobId = jobId;
      jobDetailData = null;
      renderJobList();
      if (jobsInFlight) {
        jobsRestartRequested = true;
      } else {
        window.clearTimeout(jobTimer);
        jobTimer = window.setTimeout(runJobsCycle, 0);
      }
    }

    function applyHashView() {
      const showJobs = window.location.hash === "#jobs";
      elements.statusView.hidden = showJobs;
      elements.jobsView.hidden = !showJobs;
      elements.statusLink.setAttribute("aria-current", showJobs ? "false" : "page");
      elements.jobsLink.setAttribute("aria-current", showJobs ? "page" : "false");
    }

    function abortReads() {
      controllers.forEach((controller) => controller.abort());
      controllers.clear();
    }

    function pausePolling() {
      window.clearTimeout(healthTimer);
      window.clearTimeout(jobTimer);
      healthTimer = null;
      jobTimer = null;
      abortReads();
    }

    function resumePolling() {
      if (pollingStopped || document.hidden) {
        return;
      }
      if (healthInFlight) {
        healthRestartRequested = true;
      } else {
        healthTimer = window.setTimeout(runHealthCycle, 0);
      }
      if (jobsInFlight) {
        jobsRestartRequested = true;
      } else {
        jobTimer = window.setTimeout(runJobsCycle, 0);
      }
      if (versionData === null) {
        loadVersion();
      }
    }

    function handleVisibility() {
      if (document.hidden) {
        pausePolling();
      } else {
        resumePolling();
      }
    }

    window.addEventListener("DOMContentLoaded", () => {
      window.addEventListener("hashchange", applyHashView);
      document.addEventListener("visibilitychange", handleVisibility);
      window.addEventListener("pagehide", () => {
        pollingStopped = true;
        pausePolling();
      });
      applyHashView();
      elements.liveFeedback.textContent = "Loading local status and jobs...";
      resumePolling();
    });
  </script>
</body>
</html>
"""
