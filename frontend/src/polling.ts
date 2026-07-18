import { createApiClient, type ApiClient } from "./api/client";
import {
  isFailure,
  normalizeDetail,
  normalizeHealth,
  normalizeJobs,
  normalizeVersion,
  type DecodedDetail,
  type DecodedHealth,
  type DecodedSummary,
  type DecodedVersion,
} from "./api/decoders";
import {
  EMPTY_JOBS_TEXT,
  HEALTH_INTERVAL_MS,
  JOB_INTERVAL_MS,
  UNAVAILABLE_TEXT,
  UNEXPECTED_TEXT,
  type StatusTone,
} from "./constants";

export interface RegionStatus {
  text: string;
  tone: StatusTone;
}

export interface ShellState {
  healthData: DecodedHealth | null;
  versionData: DecodedVersion | null;
  jobSummaries: DecodedSummary[] | null;
  jobDetailData: DecodedDetail | null;
  selectedJobId: string | null;
  healthStatus: RegionStatus;
  versionStatus: RegionStatus;
  jobsStatus: RegionStatus;
  detailStatus: RegionStatus;
  healthBusy: boolean;
  versionBusy: boolean;
  jobsBusy: boolean;
  detailVisible: boolean;
  lastRefresh: string;
  liveFeedback: string;
}

const initialState = (): ShellState => ({
  healthData: null,
  versionData: null,
  jobSummaries: null,
  jobDetailData: null,
  selectedJobId: null,
  healthStatus: { text: "Loading local status...", tone: "" },
  versionStatus: { text: "Loading version...", tone: "" },
  jobsStatus: { text: "Loading jobs...", tone: "" },
  detailStatus: { text: "Loading job detail...", tone: "" },
  healthBusy: true,
  versionBusy: true,
  jobsBusy: true,
  detailVisible: false,
  lastRefresh: "Never",
  liveFeedback: "Loading local status and jobs...",
});

function status(text: string, tone: StatusTone = ""): RegionStatus {
  return { text, tone };
}

export class ShellController {
  private state = initialState();
  private readonly listeners = new Set<() => void>();
  private healthTimer: ReturnType<typeof window.setTimeout> | null = null;
  private jobTimer: ReturnType<typeof window.setTimeout> | null = null;
  private healthInFlight = false;
  private jobsInFlight = false;
  private versionInFlight = false;
  private healthRestartRequested = false;
  private jobsRestartRequested = false;
  private pollingStopped = false;
  private hadConnectivityFailure = false;
  private started = false;

  constructor(private readonly api: ApiClient = createApiClient()) {}

  getState(): ShellState {
    return this.state;
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    window.addEventListener("hashchange", this.emitCurrent);
    document.addEventListener("visibilitychange", this.handleVisibility);
    window.addEventListener("pagehide", this.handlePageHide);
    this.resumePolling();
  }

  dispose(): void {
    if (!this.started) return;
    this.pollingStopped = true;
    this.pausePolling();
    window.removeEventListener("hashchange", this.emitCurrent);
    document.removeEventListener("visibilitychange", this.handleVisibility);
    window.removeEventListener("pagehide", this.handlePageHide);
    this.listeners.clear();
    this.started = false;
  }

  selectJob(jobId: string): void {
    if (!/^[0-9a-f]{32}$/.test(jobId) || jobId === this.state.selectedJobId) {
      return;
    }
    this.state = {
      ...this.state,
      selectedJobId: jobId,
      jobDetailData: null,
      detailStatus: status("Loading job detail..."),
      detailVisible: true,
    };
    this.emit();
    if (this.jobsInFlight) {
      this.jobsRestartRequested = true;
    } else {
      this.clearJobTimer();
      this.jobTimer = window.setTimeout(this.runJobsCycle, 0);
    }
  }

  private readonly emitCurrent = (): void => this.emit();

  private emit(): void {
    this.listeners.forEach((listener) => listener());
  }

  private updateLastRefresh(): void {
    let liveFeedback = this.state.liveFeedback;
    if (this.hadConnectivityFailure) {
      liveFeedback = "Local API reconnected.";
      this.hadConnectivityFailure = false;
    }
    this.state = {
      ...this.state,
      lastRefresh: new Date().toLocaleTimeString(),
      liveFeedback,
    };
  }

  private failureText(failure: unknown): string {
    if (isFailure(failure) && Number.isInteger(failure.httpStatus)) {
      return `Local API returned HTTP ${String(failure.httpStatus)}.`;
    }
    if (isFailure(failure) && failure.unexpected === true) {
      return UNEXPECTED_TEXT;
    }
    this.hadConnectivityFailure = true;
    return UNAVAILABLE_TEXT;
  }

  private readonly loadVersion = async (): Promise<void> => {
    if (this.versionInFlight) return;
    this.versionInFlight = true;
    this.state = {
      ...this.state,
      versionStatus: status(this.state.versionData === null ? "Loading version..." : "Refreshing..."),
      versionBusy: true,
    };
    this.emit();
    try {
      const value = normalizeVersion(await this.api.getVersion());
      if (value === null) throw { unexpected: true };
      this.state = {
        ...this.state,
        versionData: value,
        versionStatus: status("Version information is current."),
      };
      this.updateLastRefresh();
    } catch (failure) {
      if (this.pollingStopped || document.hidden) return;
      const stale = this.state.versionData !== null;
      this.state = {
        ...this.state,
        versionStatus: status(this.failureText(failure) + (stale ? " Stale." : ""), stale ? "stale" : "error"),
      };
    } finally {
      this.versionInFlight = false;
      this.state = { ...this.state, versionBusy: false };
      this.emit();
    }
  };

  private readonly runHealthCycle = async (): Promise<void> => {
    if (this.pollingStopped || document.hidden || this.healthInFlight) return;
    this.healthInFlight = true;
    this.state = {
      ...this.state,
      healthStatus: status(this.state.healthData === null ? "Loading local status..." : "Refreshing..."),
      healthBusy: true,
    };
    this.emit();
    try {
      const value = normalizeHealth(await this.api.getHealth());
      if (value === null) throw { unexpected: true };
      this.state = {
        ...this.state,
        healthData: value,
        healthStatus: status(
          value.ok ? "Local service is healthy." : "Local service reported unhealthy.",
          value.ok ? "" : "error",
        ),
      };
      this.updateLastRefresh();
      this.emit();
      if (this.state.versionData === null) {
        await this.loadVersion();
      }
    } catch (failure) {
      if (!this.pollingStopped && !document.hidden) {
        const stale = this.state.healthData !== null;
        this.state = {
          ...this.state,
          healthStatus: status(this.failureText(failure) + (stale ? " Stale." : ""), stale ? "stale" : "error"),
        };
      }
    } finally {
      this.healthInFlight = false;
      this.state = { ...this.state, healthBusy: false };
      this.emit();
      if (!this.pollingStopped && !document.hidden) {
        const delay = this.healthRestartRequested ? 0 : HEALTH_INTERVAL_MS;
        this.healthRestartRequested = false;
        this.healthTimer = window.setTimeout(this.runHealthCycle, delay);
      }
    }
  };

  private selectedSummaryExists(): boolean {
    return (
      this.state.jobSummaries !== null &&
      this.state.jobSummaries.some((job) => job.job_id === this.state.selectedJobId)
    );
  }

  private readonly loadSelectedDetail = async (): Promise<void> => {
    const selectedJobId = this.state.selectedJobId;
    if (selectedJobId === null || !/^[0-9a-f]{32}$/.test(selectedJobId)) return;
    this.state = {
      ...this.state,
      detailStatus: status(this.state.jobDetailData === null ? "Loading job detail..." : "Refreshing detail..."),
      detailVisible: true,
    };
    this.emit();
    try {
      const value = normalizeDetail(await this.api.getJobDetail(selectedJobId));
      if (value === null || value.job_id !== this.state.selectedJobId) throw { unexpected: true };
      this.state = {
        ...this.state,
        jobDetailData: value,
        detailStatus: status("Job detail is current."),
      };
      this.updateLastRefresh();
    } catch (failure) {
      if (this.pollingStopped || document.hidden) return;
      const stale = this.state.jobDetailData !== null;
      const text =
        isFailure(failure) && failure.httpStatus === 404
          ? "Job detail is no longer available. Refreshing list..."
          : this.failureText(failure) + (stale ? " Stale." : "");
      this.state = {
        ...this.state,
        detailStatus: status(text, stale ? "stale" : "error"),
      };
    } finally {
      this.emit();
    }
  };

  private readonly runJobsCycle = async (): Promise<void> => {
    if (this.pollingStopped || document.hidden || this.jobsInFlight) return;
    this.jobsInFlight = true;
    this.state = {
      ...this.state,
      jobsStatus: status(this.state.jobSummaries === null ? "Loading jobs..." : "Refreshing..."),
      jobsBusy: true,
    };
    this.emit();
    try {
      const summaries = normalizeJobs(await this.api.getJobs());
      if (summaries === null) throw { unexpected: true };
      this.state = { ...this.state, jobSummaries: summaries };
      if (!this.selectedSummaryExists()) {
        this.state = {
          ...this.state,
          selectedJobId: summaries.length === 0 ? null : summaries[0].job_id,
          jobDetailData: null,
        };
      }
      this.updateLastRefresh();
      if (summaries.length === 0) {
        this.state = {
          ...this.state,
          detailVisible: false,
          jobsStatus: status(EMPTY_JOBS_TEXT),
          liveFeedback: EMPTY_JOBS_TEXT,
        };
      } else {
        this.state = { ...this.state, jobsStatus: status("Retained jobs are current.") };
        this.emit();
        await this.loadSelectedDetail();
      }
    } catch (failure) {
      if (!this.pollingStopped && !document.hidden) {
        const stale = this.state.jobSummaries !== null;
        this.state = {
          ...this.state,
          jobsStatus: status(this.failureText(failure) + (stale ? " Stale." : ""), stale ? "stale" : "error"),
        };
      }
    } finally {
      this.jobsInFlight = false;
      this.state = { ...this.state, jobsBusy: false };
      this.emit();
      if (!this.pollingStopped && !document.hidden) {
        const delay = this.jobsRestartRequested ? 0 : JOB_INTERVAL_MS;
        this.jobsRestartRequested = false;
        this.jobTimer = window.setTimeout(this.runJobsCycle, delay);
      }
    }
  };

  private clearHealthTimer(): void {
    if (this.healthTimer !== null) window.clearTimeout(this.healthTimer);
    this.healthTimer = null;
  }

  private clearJobTimer(): void {
    if (this.jobTimer !== null) window.clearTimeout(this.jobTimer);
    this.jobTimer = null;
  }

  private pausePolling(): void {
    this.clearHealthTimer();
    this.clearJobTimer();
    this.api.abortAll();
  }

  private resumePolling(): void {
    if (this.pollingStopped || document.hidden) return;
    if (this.healthInFlight) {
      this.healthRestartRequested = true;
    } else {
      this.healthTimer = window.setTimeout(this.runHealthCycle, 0);
    }
    if (this.jobsInFlight) {
      this.jobsRestartRequested = true;
    } else {
      this.jobTimer = window.setTimeout(this.runJobsCycle, 0);
    }
    if (this.state.versionData === null) {
      void this.loadVersion();
    }
  }

  private readonly handleVisibility = (): void => {
    if (document.hidden) {
      this.pausePolling();
    } else {
      this.resumePolling();
    }
  };

  private readonly handlePageHide = (): void => {
    this.pollingStopped = true;
    this.pausePolling();
  };
}
