import type {
  HealthResponse,
  JobDetailResponse,
  JobError,
  JobResult,
  JobSummary,
  LogEntry,
  VersionResponse,
} from "./generated";
import { JOB_STATES, LOG_CLASSES, type JobState } from "../constants";

export type DecodedHealth = Omit<
  HealthResponse,
  "ok" | "service" | "mode" | "local_only" | "state_configured"
> & {
  ok: boolean;
  service: string;
  mode: string;
  local_only: boolean;
  state_configured: boolean;
};

export type DecodedVersion = Omit<VersionResponse, "service" | "version_source"> & {
  service: string;
  version_source: string;
};

export type DecodedSummary = Omit<JobSummary, "job_type" | "state"> & {
  job_type: string;
  state: JobState;
};

export type DecodedLog = Omit<LogEntry, "level"> & {
  level: keyof typeof LOG_CLASSES;
};

export type DecodedResult = Omit<JobResult, "kind"> & { kind: string };
export type DecodedError = JobError;
export type DecodedDetail = Omit<
  JobDetailResponse,
  "job_type" | "state" | "logs" | "result" | "error"
> & {
  job_type: string;
  state: JobState;
  logs: DecodedLog[];
  result: DecodedResult | null;
  error: DecodedError | null;
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringOrNull(value: unknown): value is string | null {
  return typeof value === "string" || value === null;
}

function isJobState(value: unknown): value is JobState {
  return typeof value === "string" && JOB_STATES.some((state) => state === value);
}

export function normalizeHealth(value: unknown): DecodedHealth | null {
  if (
    !isObject(value) ||
    typeof value.ok !== "boolean" ||
    typeof value.service !== "string" ||
    typeof value.mode !== "string" ||
    typeof value.local_only !== "boolean" ||
    typeof value.state_configured !== "boolean"
  ) {
    return null;
  }
  return {
    ok: value.ok,
    service: value.service,
    mode: value.mode,
    local_only: value.local_only,
    state_configured: value.state_configured,
  };
}

export function normalizeVersion(value: unknown): DecodedVersion | null {
  if (
    !isObject(value) ||
    typeof value.service !== "string" ||
    typeof value.version !== "string" ||
    typeof value.version_source !== "string"
  ) {
    return null;
  }
  return {
    service: value.service,
    version: value.version,
    version_source: value.version_source,
  };
}

export function normalizeSummary(value: unknown): DecodedSummary | null {
  if (
    !isObject(value) ||
    typeof value.job_id !== "string" ||
    !/^[0-9a-f]{32}$/.test(value.job_id) ||
    typeof value.job_type !== "string" ||
    !isJobState(value.state) ||
    typeof value.progress !== "number" ||
    !Number.isFinite(value.progress) ||
    !isStringOrNull(value.current_step) ||
    typeof value.created_at !== "string" ||
    !isStringOrNull(value.started_at) ||
    !isStringOrNull(value.finished_at)
  ) {
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
    finished_at: value.finished_at,
  };
}

export function normalizeJobs(value: unknown): DecodedSummary[] | null {
  if (!isObject(value) || !Array.isArray(value.jobs)) {
    return null;
  }
  const jobs: DecodedSummary[] = [];
  for (const candidate of value.jobs) {
    const job = normalizeSummary(candidate);
    if (job === null) {
      return null;
    }
    jobs.push(job);
  }
  return jobs;
}

export function normalizeLog(value: unknown): DecodedLog | null {
  if (
    !isObject(value) ||
    typeof value.time !== "string" ||
    (value.level !== "info" && value.level !== "warning" && value.level !== "error") ||
    typeof value.code !== "string" ||
    typeof value.message !== "string"
  ) {
    return null;
  }
  return { time: value.time, level: value.level, code: value.code, message: value.message };
}

export function normalizeResult(value: unknown): DecodedResult | null | undefined {
  if (value === null) {
    return null;
  }
  if (
    !isObject(value) ||
    typeof value.kind !== "string" ||
    typeof value.steps_completed !== "number" ||
    !Number.isInteger(value.steps_completed)
  ) {
    return undefined;
  }
  return { kind: value.kind, steps_completed: value.steps_completed };
}

export function normalizeError(value: unknown): DecodedError | null | undefined {
  if (value === null) {
    return null;
  }
  if (!isObject(value) || typeof value.code !== "string" || typeof value.message !== "string") {
    return undefined;
  }
  return { code: value.code, message: value.message };
}

export function normalizeDetail(value: unknown): DecodedDetail | null {
  const summary = normalizeSummary(value);
  if (summary === null || !isObject(value) || !Array.isArray(value.logs)) {
    return null;
  }
  const logs: DecodedLog[] = [];
  for (const candidate of value.logs) {
    const log = normalizeLog(candidate);
    if (log === null) {
      return null;
    }
    logs.push(log);
  }
  const result = normalizeResult(value.result);
  const error = normalizeError(value.error);
  if (result === undefined || error === undefined) {
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
    logs,
    result,
    error,
  };
}

export function isFailure(value: unknown): value is Record<string, unknown> {
  return isObject(value);
}
