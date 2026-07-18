import { afterEach, describe, expect, it, vi } from "vitest";
import { createApiClient } from "../src/api/client";

function jsonResponse(value: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(value),
  } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("read-only same-origin API client", () => {
  it("uses only the four relative allowlisted GET path shapes with hardened options", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);
    const client = createApiClient();
    const jobId = "0123456789abcdef0123456789abcdef";

    await client.getHealth();
    await client.getVersion();
    await client.getJobs();
    await client.getJobDetail(jobId);

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/api/health",
      "/api/version",
      "/api/jobs",
      `/api/jobs/${jobId}`,
    ]);
    for (const [path, options] of fetchMock.mock.calls) {
      expect(String(path)).toMatch(/^\/api\/(?:health|version|jobs(?:\/[0-9a-f]{32})?)$/);
      expect(options).toMatchObject({ method: "GET", credentials: "omit", cache: "no-store" });
      expect(options.signal).toBeInstanceOf(AbortSignal);
      expect(options.body).toBeUndefined();
    }
  });

  it.each(["not-hex", "0123456789abcdef", "gggggggggggggggggggggggggggggggg", "../secret"])(
    "rejects invalid detail id %s before fetch",
    async (jobId) => {
      const fetchMock = vi.fn();
      vi.stubGlobal("fetch", fetchMock);
      const client = createApiClient();

      await expect(client.getJobDetail(jobId)).rejects.toThrow("Invalid job ID");
      expect(fetchMock).toHaveBeenCalledTimes(0);
    },
  );

  it("constructs the valid detail path exactly", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);
    const client = createApiClient();
    const jobId = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    await client.getJobDetail(jobId);

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toBe(`/api/jobs/${jobId}`);
  });
});
