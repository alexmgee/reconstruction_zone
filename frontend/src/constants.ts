export const HEALTH_INTERVAL_MS = 5000;
export const JOB_INTERVAL_MS = 2000;
export const FETCH_TIMEOUT_MS = 5000;
export const PROJECT_ID_PATTERN = /^[A-Za-z0-9._~-]+$/;

export const EMPTY_JOBS_TEXT =
  "No representative jobs are available. WEB5 is read-only; jobs must be created outside this page.";
export const UNAVAILABLE_TEXT = "Local API unavailable. Is the localhost server still running?";
export const UNEXPECTED_TEXT = "Local API returned an unexpected response.";

export const JOB_STATES = [
  "queued",
  "running",
  "cancel_requested",
  "completed",
  "failed",
  "cancelled",
] as const;

export type JobState = (typeof JOB_STATES)[number];
export type StatusTone = "" | "error" | "stale";

export const STATE_CLASSES: Record<JobState, string> = {
  queued: "tone-neutral",
  running: "tone-neutral",
  cancel_requested: "tone-warn",
  completed: "tone-good",
  failed: "tone-bad",
  cancelled: "tone-warn",
};

export const LOG_CLASSES = {
  info: "",
  warning: "log-warning",
  error: "log-error",
} as const;
