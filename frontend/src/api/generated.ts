/* generated — do not edit
 * Source: frontend/contracts/*.schema.json
 * Canonical schema SHA-256: 82775ff5649b9065453dc9bb9452230998e4ea4b5ff3da54b8d5915fe0ff5b28
 */
export const SCHEMA_SHA256 = "82775ff5649b9065453dc9bb9452230998e4ea4b5ff3da54b8d5915fe0ff5b28" as const;

export type ApiResponse = ErrorResponse | HealthResponse | JobDetailResponse | JobsResponse | VersionResponse;
export type Timestamp = string;
export type TimestampOrNull = Timestamp | null;

export interface ErrorResponse {
  error: string;
  message: string;
}
export interface HealthResponse {
  ok: true;
  service: "reconstruction_web";
  mode: "local";
  local_only: true;
  state_configured: true;
}
export interface JobDetailResponse {
  job_id: string;
  job_type: "representative";
  state: "queued" | "running" | "cancel_requested" | "completed" | "failed" | "cancelled";
  progress: number;
  current_step: string | null;
  created_at: Timestamp;
  started_at: TimestampOrNull;
  finished_at: TimestampOrNull;
  logs: LogEntry[];
  result: JobResult | null;
  error: JobError | null;
}
export interface LogEntry {
  time: Timestamp;
  level: "info" | "warning" | "error";
  code: string;
  message: string;
}
export interface JobResult {
  kind: "representative";
  steps_completed: number;
}
export interface JobError {
  code: string;
  message: string;
}
export interface JobsResponse {
  jobs: JobSummary[];
}
export interface JobSummary {
  job_id: string;
  job_type: "representative";
  state: "queued" | "running" | "cancel_requested" | "completed" | "failed" | "cancelled";
  progress: number;
  current_step: string | null;
  created_at: Timestamp;
  started_at: TimestampOrNull;
  finished_at: TimestampOrNull;
}
export interface VersionResponse {
  service: "reconstruction_web";
  version: string;
  version_source: "reconstruction_web";
}
