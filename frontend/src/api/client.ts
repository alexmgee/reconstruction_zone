import { FETCH_TIMEOUT_MS, PROJECT_ID_PATTERN } from "../constants";

const HEALTH_PATH = "/api/health" as const;
const VERSION_PATH = "/api/version" as const;
const JOBS_PATH = "/api/jobs" as const;
const PROJECTS_PATH = "/api/projects" as const;
const JOB_ID_PATTERN = /^[0-9a-f]{32}$/;
const DETAIL_PATH_PATTERN = /^\/api\/jobs\/[0-9a-f]{32}$/;
const PROJECT_DETAIL_PATH_PATTERN = /^\/api\/projects\/[A-Za-z0-9._~-]+$/;

type ApiPath =
  | typeof HEALTH_PATH
  | typeof VERSION_PATH
  | typeof JOBS_PATH
  | typeof PROJECTS_PATH
  | `/api/jobs/${string}`
  | `/api/projects/${string}`;

export interface ApiClient {
  getHealth(): Promise<unknown>;
  getVersion(): Promise<unknown>;
  getJobs(): Promise<unknown>;
  getJobDetail(jobId: string): Promise<unknown>;
  getProjects(): Promise<unknown>;
  getProjectDetail(id: string): Promise<unknown>;
  abortAll(): void;
}

export function createApiClient(): ApiClient {
  const controllers = new Set<AbortController>();

  async function requestJson(path: ApiPath): Promise<unknown> {
    const controller = new AbortController();
    controllers.add(controller);
    const timeout = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    try {
      const response = await fetch(path, {
        method: "GET",
        cache: "no-store",
        credentials: "omit",
        signal: controller.signal,
      });
      if (!response.ok) {
        throw { httpStatus: response.status };
      }
      try {
        return await response.json();
      } catch {
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

  return {
    getHealth: () => requestJson(HEALTH_PATH),
    getVersion: () => requestJson(VERSION_PATH),
    getJobs: () => requestJson(JOBS_PATH),
    getProjects: () => requestJson(PROJECTS_PATH),
    getJobDetail: (jobId: string) => {
      if (!JOB_ID_PATTERN.test(jobId)) {
        return Promise.reject(new Error("Invalid job ID"));
      }
      const path: `/api/jobs/${string}` = `${JOBS_PATH}/${encodeURIComponent(jobId)}`;
      if (!DETAIL_PATH_PATTERN.test(path)) {
        return Promise.reject(new Error("Invalid job detail path"));
      }
      return requestJson(path);
    },
    getProjectDetail: (id: string) => {
      if (
        !PROJECT_ID_PATTERN.test(id) ||
        id === "." ||
        id === ".." ||
        id.length > 200
      ) {
        return Promise.reject(new Error("Invalid project ID"));
      }
      const path: `/api/projects/${string}` = `${PROJECTS_PATH}/${encodeURIComponent(id)}`;
      if (!PROJECT_DETAIL_PATH_PATTERN.test(path)) {
        return Promise.reject(new Error("Invalid project detail path"));
      }
      return requestJson(path);
    },
    abortAll: () => {
      controllers.forEach((controller) => controller.abort());
      controllers.clear();
    },
  };
}
