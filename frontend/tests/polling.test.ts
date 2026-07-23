import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApiClient, type ApiClient } from "../src/api/client";
import { ShellController } from "../src/polling";

const jobId = "0123456789abcdef0123456789abcdef";
const health = {
  ok: true,
  service: "reconstruction_web",
  mode: "local",
  local_only: true,
  state_configured: true,
};
const version = {
  service: "reconstruction_web",
  version: "1.2.3",
  version_source: "reconstruction_web",
};
const summary = {
  job_id: jobId,
  job_type: "representative",
  state: "running",
  progress: 25,
  current_step: "step",
  created_at: "2026-01-01T00:00:00Z",
  started_at: null,
  finished_at: null,
};
const detail = { ...summary, logs: [], result: null, error: null };

function mockApi(): ApiClient {
  return {
    getHealth: vi.fn().mockResolvedValue(health),
    getVersion: vi.fn().mockResolvedValue(version),
    getJobs: vi.fn().mockResolvedValue({ jobs: [summary] }),
    getJobDetail: vi.fn().mockResolvedValue(detail),
    abortAll: vi.fn(),
  };
}

function setHidden(value: boolean): void {
  Object.defineProperty(document, "hidden", { configurable: true, value });
}

beforeEach(() => {
  vi.useFakeTimers();
  setHidden(false);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  setHidden(false);
});

describe("polling lifecycle", () => {
  it("uses recursive 5000/2000 cycles and requests version exactly once", async () => {
    const api = mockApi();
    const controller = new ShellController(api);
    controller.start();

    expect(api.getVersion).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(0);
    expect(api.getHealth).toHaveBeenCalledTimes(1);
    expect(api.getJobs).toHaveBeenCalledTimes(1);
    expect(api.getJobDetail).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(1999);
    expect(api.getJobs).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(api.getJobs).toHaveBeenCalledTimes(2);
    expect(api.getJobDetail).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(3000);
    expect(api.getHealth).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(10000);
    expect(api.getHealth).toHaveBeenCalledTimes(4);
    expect(api.getVersion).toHaveBeenCalledTimes(1);
    controller.dispose();
  });

  it("enforces the 5000 ms fetch timeout with an AbortController signal", async () => {
    let capturedSignal: AbortSignal | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn((_path: string, options: RequestInit) => {
        capturedSignal = options.signal as AbortSignal;
        return new Promise<Response>(() => undefined);
      }),
    );
    const client = createApiClient();
    void client.getHealth();

    expect(capturedSignal?.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(4999);
    expect(capturedSignal?.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(capturedSignal?.aborted).toBe(true);
  });

  it("does not overlap a cycle whose request is still pending", async () => {
    let resolveHealth!: (value: unknown) => void;
    const api = mockApi();
    vi.mocked(api.getHealth).mockImplementation(
      () => new Promise((resolve) => (resolveHealth = resolve)),
    );
    vi.mocked(api.getJobs).mockResolvedValue({ jobs: [] });
    const controller = new ShellController(api);
    controller.start();
    await vi.advanceTimersByTimeAsync(0);

    await vi.advanceTimersByTimeAsync(20000);
    expect(api.getHealth).toHaveBeenCalledTimes(1);
    resolveHealth(health);
    await Promise.resolve();
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(4999);
    expect(api.getHealth).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(api.getHealth).toHaveBeenCalledTimes(2);
    controller.dispose();
  });

  it("aborts and pauses while hidden, resumes immediately, and cleans up", async () => {
    const api = mockApi();
    const controller = new ShellController(api);
    controller.start();
    await vi.advanceTimersByTimeAsync(0);
    const healthCalls = vi.mocked(api.getHealth).mock.calls.length;
    const jobsCalls = vi.mocked(api.getJobs).mock.calls.length;

    setHidden(true);
    document.dispatchEvent(new Event("visibilitychange"));
    expect(api.abortAll).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(10000);
    expect(api.getHealth).toHaveBeenCalledTimes(healthCalls);
    expect(api.getJobs).toHaveBeenCalledTimes(jobsCalls);

    setHidden(false);
    document.dispatchEvent(new Event("visibilitychange"));
    await vi.advanceTimersByTimeAsync(0);
    expect(api.getHealth).toHaveBeenCalledTimes(healthCalls + 1);
    expect(api.getJobs).toHaveBeenCalledTimes(jobsCalls + 1);

    controller.dispose();
    expect(api.abortAll).toHaveBeenCalledTimes(2);
    const callsAfterCleanup = vi.mocked(api.getHealth).mock.calls.length;
    document.dispatchEvent(new Event("visibilitychange"));
    await vi.advanceTimersByTimeAsync(10000);
    expect(api.getHealth).toHaveBeenCalledTimes(callsAfterCleanup);
  });
});
