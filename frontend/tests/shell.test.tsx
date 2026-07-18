import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { App } from "../src/App";
import type { ApiClient } from "../src/api/client";
import {
  normalizeDetail,
  normalizeHealth,
  normalizeSummary,
  type DecodedDetail,
  type DecodedSummary,
} from "../src/api/decoders";
import { EMPTY_JOBS_TEXT, UNAVAILABLE_TEXT, UNEXPECTED_TEXT } from "../src/constants";
import { ShellController, type RegionStatus, type ShellState } from "../src/polling";
import { JobsView, StatusLine, StatusView } from "../src/views/LazyViews";

const jobId = "0123456789abcdef0123456789abcdef";
const summary: DecodedSummary = {
  job_id: jobId,
  job_type: "representative",
  state: "failed",
  progress: 75,
  current_step: null,
  created_at: "2026-01-01T00:00:00Z",
  started_at: null,
  finished_at: null,
};
const detail: DecodedDetail = {
  ...summary,
  logs: [{ time: "now", level: "error", code: "E_TEST", message: "failed safely" }],
  result: { kind: "representative", steps_completed: 3 },
  error: { code: "failed", message: "Expected failure" },
};

function makeState(overrides: Partial<ShellState> = {}): ShellState {
  return {
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
    ...overrides,
  };
}

function status(text: string, tone: RegionStatus["tone"] = ""): RegionStatus {
  return { text, tone };
}

function settledApi(): ApiClient {
  return {
    getHealth: vi.fn().mockResolvedValue({
      ok: true,
      service: "reconstruction_web",
      mode: "local",
      local_only: true,
      state_configured: true,
    }),
    getVersion: vi.fn().mockResolvedValue({
      service: "reconstruction_web",
      version: "1.0.0",
      version_source: "reconstruction_web",
    }),
    getJobs: vi.fn().mockResolvedValue({ jobs: [] }),
    getJobDetail: vi.fn(),
    abortAll: vi.fn(),
  };
}

afterEach(() => {
  cleanup();
  window.location.hash = "";
});

describe("React shell parity", () => {
  it("renders the full visible state catalog with exact copy and tones", () => {
    const catalog = [
      "Loading local status...",
      "Loading version...",
      "Loading jobs...",
      "Loading job detail...",
      "Refreshing...",
      "Refreshing detail...",
      "Local service is healthy.",
      "Local service reported unhealthy.",
      "Version information is current.",
      "Retained jobs are current.",
      "Job detail is current.",
      EMPTY_JOBS_TEXT,
      UNAVAILABLE_TEXT,
      "Local API returned HTTP 503.",
      UNEXPECTED_TEXT,
      `${UNAVAILABLE_TEXT} Stale.`,
      "Job detail is no longer available. Refreshing list...",
      "Local API reconnected.",
      "Never",
    ];
    render(
      <div>
        {catalog.map((text, index) => (
          <StatusLine
            id={`catalog-${index}`}
            value={status(text, text === "Local service reported unhealthy." ? "error" : "")}
            live={false}
            key={text}
          />
        ))}
        <JobsView
          state={makeState({
            jobSummaries: [summary],
            selectedJobId: jobId,
            jobDetailData: detail,
            detailVisible: true,
          })}
          hidden={false}
          onSelectJob={vi.fn()}
        />
      </div>,
    );

    for (const text of catalog) {
      expect(screen.getAllByText(text)[0]).toBeVisible();
    }
    expect(screen.getByText("Local service reported unhealthy.")).toHaveClass("error");
    expect(screen.getAllByText("Not available")).toHaveLength(3);
    expect(screen.getByText("3")).toBeVisible();
  });

  it("keeps the unhealthy health decoder/render branch with error tone", () => {
    const unhealthy = normalizeHealth({
      ok: false,
      service: "reconstruction_web",
      mode: "local",
      local_only: true,
      state_configured: true,
    });
    expect(unhealthy).not.toBeNull();
    render(
      <StatusView
        state={makeState({
          healthData: unhealthy,
          healthStatus: status("Local service reported unhealthy.", "error"),
          healthBusy: false,
        })}
        hidden={false}
      />,
    );

    expect(screen.getByText("Local service reported unhealthy.")).toHaveClass("error");
    expect(screen.getByText("No")).toBeVisible();
  });

  it("renders exactly the three required live-region IDs across both lazy views", async () => {
    const controller = new ShellController(settledApi());
    render(<App controller={controller} />);

    await waitFor(() => expect(document.querySelectorAll("[aria-live]")).toHaveLength(3));
    const ids = Array.from(document.querySelectorAll("[aria-live]"), (element) => element.id).sort();
    expect(ids).toEqual(["health-status", "jobs-status", "live-feedback"]);
    expect(document.querySelector("#version-status")).not.toHaveAttribute("aria-live");
    expect(document.querySelector("#detail-status")).not.toHaveAttribute("aria-live");
    expect(document.querySelector("main > #live-feedback")).toBeInTheDocument();
  });

  it("renders named decoded fields only and keeps hostile or unknown properties inert", () => {
    const secret = "UNKNOWN_SENTINEL_MUST_NOT_RENDER";
    const hostile = '<img src=x onerror="globalThis.pwned=true">';
    const decoratedDetail = normalizeDetail({
      ...detail,
      current_step: hostile,
      secret,
      result: { kind: "representative", steps_completed: 3, secret },
      logs: [{ time: "now", level: "error", code: hostile, message: hostile, secret }],
      error: { code: hostile, message: hostile, secret },
    });
    expect(decoratedDetail).not.toBeNull();
    expect(normalizeDetail({ ...detail, result: { kind: "representative", steps_completed: 1.5 } })).toBeNull();
    expect(normalizeSummary({ ...summary, job_type: "runtime-loose-job-type" })?.job_type).toBe(
      "runtime-loose-job-type",
    );
    render(
      <JobsView
        state={makeState({
          jobSummaries: [summary],
          selectedJobId: jobId,
          jobDetailData: decoratedDetail,
          detailVisible: true,
        })}
        hidden={false}
        onSelectJob={vi.fn()}
      />,
    );

    expect(screen.queryByText(secret)).not.toBeInTheDocument();
    expect(screen.getAllByText(hostile).length).toBeGreaterThan(0);
    expect(document.querySelector("img")).not.toBeInTheDocument();
  });

  it("keeps job selection keyboard-operable", async () => {
    const user = userEvent.setup();
    const onSelectJob = vi.fn();
    render(
      <JobsView
        state={makeState({ jobSummaries: [summary], jobsBusy: false })}
        hidden={false}
        onSelectJob={onSelectJob}
      />,
    );
    const button = screen.getByRole("button", { name: new RegExp(jobId) });
    button.focus();
    await user.keyboard("{Enter}");
    expect(onSelectJob).toHaveBeenCalledWith(jobId);
  });

  it("exposes aria-busy, progress semantics, and focus-visible styling", () => {
    const { rerender } = render(<StatusView state={makeState({ healthBusy: true })} hidden={false} />);
    expect(document.querySelector("#health-fields")).toHaveAttribute("aria-busy", "true");
    rerender(<StatusView state={makeState({ healthBusy: false })} hidden={false} />);
    expect(document.querySelector("#health-fields")).toHaveAttribute("aria-busy", "false");
    cleanup();

    render(
      <JobsView
        state={makeState({
          jobSummaries: [summary],
          selectedJobId: jobId,
          jobDetailData: detail,
          detailVisible: true,
          jobsBusy: true,
        })}
        hidden={false}
        onSelectJob={vi.fn()}
      />,
    );
    const progress = document.querySelector("#job-progress");
    expect(progress?.tagName).toBe("PROGRESS");
    expect(progress).toHaveAttribute("max", "100");
    expect(progress).toHaveTextContent("75%");
    expect(document.querySelector("#job-list")).toHaveAttribute("aria-busy", "true");

    expect(readFileSync("src/styles.css", "utf8")).toMatch(/:focus-visible/);
  });
});
