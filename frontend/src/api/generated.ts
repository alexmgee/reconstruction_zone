/* generated — do not edit
 * Source: frontend/contracts/*.schema.json
 * Canonical schema SHA-256: dffbd13d387877edfeccf4337624f194d4bacdf93be273fe4bd7a869cbf47b7b
 */
export const SCHEMA_SHA256 = "dffbd13d387877edfeccf4337624f194d4bacdf93be273fe4bd7a869cbf47b7b" as const;

export type ApiResponse =
  | ErrorResponse
  | HealthResponse
  | JobDetailResponse
  | JobsResponse
  | ProjectDetailResponse
  | ProjectsResponse
  | VersionResponse;
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
export interface ProjectDetailResponse {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  sources: ProjectSourceDetail[];
  work_dirs: ProjectWorkDirDetail[];
  notes: string;
  tags: string[];
  root_dir: string;
  static_masks_dir: string;
}
export interface ProjectSourceDetail {
  label: string;
  path: string;
  media_type: string;
  file_count: number;
  notes: string;
  exists: boolean;
}
export interface ProjectWorkDirDetail {
  label: string;
  path: string;
  stage: string;
  file_count: number;
  derived_from: string;
  exists: boolean;
}
export interface ProjectsResponse {
  projects: ProjectSummary[];
}
export interface ProjectSummary {
  id: string;
  title: string;
  tags: string[];
  source_count: number;
  work_dir_count: number;
  created_at: string;
  updated_at: string;
}
export interface VersionResponse {
  service: "reconstruction_web";
  version: string;
  version_source: "reconstruction_web";
}
