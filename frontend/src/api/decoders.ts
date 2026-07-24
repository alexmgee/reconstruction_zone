import type {
  HealthResponse,
  JobDetailResponse,
  JobError,
  JobResult,
  JobSummary,
  LogEntry,
  ProjectDetailResponse,
  ProjectSourceDetail,
  ProjectSummary,
  ProjectWorkDirDetail,
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
export type DecodedProjectSummary = ProjectSummary;
export type DecodedProjectSourceDetail = ProjectSourceDetail;
export type DecodedProjectWorkDirDetail = ProjectWorkDirDetail;
export type DecodedProjectDetail = ProjectDetailResponse;

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

export function normalizeProjectSummary(value: unknown): DecodedProjectSummary | null {
  if (
    !isObject(value) ||
    typeof value.id !== "string" ||
    typeof value.title !== "string" ||
    !Array.isArray(value.tags) ||
    !value.tags.every((tag) => typeof tag === "string") ||
    typeof value.source_count !== "number" ||
    !Number.isInteger(value.source_count) ||
    value.source_count < 0 ||
    typeof value.work_dir_count !== "number" ||
    !Number.isInteger(value.work_dir_count) ||
    value.work_dir_count < 0 ||
    typeof value.created_at !== "string" ||
    typeof value.updated_at !== "string"
  ) {
    return null;
  }
  return {
    id: value.id,
    title: value.title,
    tags: value.tags,
    source_count: value.source_count,
    work_dir_count: value.work_dir_count,
    created_at: value.created_at,
    updated_at: value.updated_at,
  };
}

export function normalizeProjectList(value: unknown): DecodedProjectSummary[] | null {
  if (!isObject(value) || !Array.isArray(value.projects)) {
    return null;
  }
  const projects: DecodedProjectSummary[] = [];
  for (const candidate of value.projects) {
    const project = normalizeProjectSummary(candidate);
    if (project === null) {
      return null;
    }
    projects.push(project);
  }
  return projects;
}

export function normalizeProjectSourceDetail(
  value: unknown,
): DecodedProjectSourceDetail | null {
  if (
    !isObject(value) ||
    typeof value.label !== "string" ||
    typeof value.path !== "string" ||
    typeof value.media_type !== "string" ||
    typeof value.notes !== "string" ||
    typeof value.file_count !== "number" ||
    !Number.isInteger(value.file_count) ||
    value.file_count < 0 ||
    typeof value.exists !== "boolean"
  ) {
    return null;
  }
  return {
    label: value.label,
    path: value.path,
    media_type: value.media_type,
    file_count: value.file_count,
    notes: value.notes,
    exists: value.exists,
  };
}

export function normalizeProjectWorkDirDetail(
  value: unknown,
): DecodedProjectWorkDirDetail | null {
  if (
    !isObject(value) ||
    typeof value.label !== "string" ||
    typeof value.path !== "string" ||
    typeof value.stage !== "string" ||
    typeof value.derived_from !== "string" ||
    typeof value.file_count !== "number" ||
    !Number.isInteger(value.file_count) ||
    value.file_count < 0 ||
    typeof value.exists !== "boolean"
  ) {
    return null;
  }
  return {
    label: value.label,
    path: value.path,
    stage: value.stage,
    file_count: value.file_count,
    derived_from: value.derived_from,
    exists: value.exists,
  };
}

export function normalizeProjectDetail(value: unknown): DecodedProjectDetail | null {
  if (
    !isObject(value) ||
    typeof value.id !== "string" ||
    typeof value.title !== "string" ||
    typeof value.created_at !== "string" ||
    typeof value.updated_at !== "string" ||
    typeof value.notes !== "string" ||
    typeof value.root_dir !== "string" ||
    typeof value.static_masks_dir !== "string" ||
    !Array.isArray(value.tags) ||
    !value.tags.every((tag) => typeof tag === "string") ||
    !Array.isArray(value.sources) ||
    !Array.isArray(value.work_dirs)
  ) {
    return null;
  }
  const sources: DecodedProjectSourceDetail[] = [];
  for (const candidate of value.sources) {
    const source = normalizeProjectSourceDetail(candidate);
    if (source === null) {
      return null;
    }
    sources.push(source);
  }
  const workDirs: DecodedProjectWorkDirDetail[] = [];
  for (const candidate of value.work_dirs) {
    const workDir = normalizeProjectWorkDirDetail(candidate);
    if (workDir === null) {
      return null;
    }
    workDirs.push(workDir);
  }
  return {
    id: value.id,
    title: value.title,
    created_at: value.created_at,
    updated_at: value.updated_at,
    sources,
    work_dirs: workDirs,
    notes: value.notes,
    tags: value.tags,
    root_dir: value.root_dir,
    static_masks_dir: value.static_masks_dir,
  };
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
