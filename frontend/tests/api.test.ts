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
  it("uses only the relative allowlisted GET path shapes with hardened options", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);
    const client = createApiClient();
    const jobId = "0123456789abcdef0123456789abcdef";
    const projectId = "project-one";

    await client.getHealth();
    await client.getVersion();
    await client.getJobs();
    await client.getJobDetail(jobId);
    await client.getProjects();
    await client.getProjectDetail(projectId);

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      "/api/health",
      "/api/version",
      "/api/jobs",
      `/api/jobs/${jobId}`,
      "/api/projects",
      `/api/projects/${projectId}`,
    ]);
    for (const [path, options] of fetchMock.mock.calls) {
      expect(String(path)).toMatch(
        /^\/api\/(?:health|version|jobs(?:\/[0-9a-f]{32})?|projects(?:\/[A-Za-z0-9._~-]+)?)$/,
      );
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

  it.each(["550e8400-e29b-41d4-a716-446655440000", "project-one"])(
    "constructs the valid project detail path for %s",
    async (projectId) => {
      const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
      vi.stubGlobal("fetch", fetchMock);
      const client = createApiClient();

      await client.getProjectDetail(projectId);

      expect(fetchMock).toHaveBeenCalledOnce();
      expect(fetchMock.mock.calls[0][0]).toBe(`/api/projects/${projectId}`);
    },
  );

  it.each(["folder/project", ".", "..", "", "project one"])(
    "rejects invalid project detail id %s before fetch",
    async (projectId) => {
      const fetchMock = vi.fn();
      vi.stubGlobal("fetch", fetchMock);
      const client = createApiClient();

      await expect(client.getProjectDetail(projectId)).rejects.toThrow("Invalid project ID");
      expect(fetchMock).toHaveBeenCalledTimes(0);
    },
  );
});
