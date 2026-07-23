import { expect, test, type Request, type Route } from "@playwright/test";

import healthFixture from "./fixtures/health.json" with { type: "json" };
import jobDetailFixture from "./fixtures/job-detail.json" with { type: "json" };
import jobsFixture from "./fixtures/jobs.json" with { type: "json" };
import versionFixture from "./fixtures/version.json" with { type: "json" };

const EXACT_CSP =
  "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src data:; font-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";
const JOB_ID = jobsFixture.jobs[0].job_id;
const API_PATHS = new Set([
  "/api/health",
  "/api/version",
  "/api/jobs",
  `/api/jobs/${JOB_ID}`,
]);
const UNAVAILABLE = "Local API unavailable. Is the localhost server still running?";

test("compiled local shell preserves its real-server and browser contracts", async ({ page, baseURL }) => {
  if (baseURL === undefined) throw new Error("Playwright baseURL is required.");

  const observedRequests: Request[] = [];
  const websocketUrls: string[] = [];
  const requestedAt = {
    health: [] as number[],
    version: [] as number[],
    jobs: [] as number[],
    detail: [] as number[],
  };
  let routesFailing = false;
  let detailFailedDuringOutage = false;
  let initialImgCount = -1;
  let initialScriptCount = -1;

  const fulfillJson = async (route: Route, body: unknown) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });

  const assertNetworkContract = () => {
    const documents = observedRequests.filter((request) => request.resourceType() === "document");
    expect(documents).toHaveLength(1);
    expect(documents[0].method()).toBe("GET");
    expect(documents[0].postData()).toBeNull();
    expect(new URL(documents[0].url()).pathname).toBe("/");

    const apiRequests = observedRequests.filter((request) => request.resourceType() !== "document");
    expect(new Set(apiRequests.map((request) => new URL(request.url()).pathname))).toEqual(API_PATHS);
    for (const request of apiRequests) {
      const url = new URL(request.url());
      expect(url.origin).toBe(new URL(baseURL).origin);
      expect(API_PATHS.has(url.pathname)).toBe(true);
      expect(url.search).toBe("");
      expect(request.method()).toBe("GET");
      expect(request.postData()).toBeNull();
      expect(request.resourceType()).toBe("fetch");
    }
    expect(websocketUrls).toEqual([]);
  };

  await test.step("1. Load the real document and verify every hardening header", async () => {
    page.on("request", (request) => observedRequests.push(request));
    page.on("websocket", (socket) => websocketUrls.push(socket.url()));
    await page.addInitScript(() => {
      Object.defineProperty(window, "__e2eSentinel", {
        value: "intact",
        writable: true,
        configurable: false,
      });
    });

    await page.route(`${baseURL}api/health`, async (route) => {
      requestedAt.health.push(Date.now());
      if (routesFailing) await route.abort("failed");
      else await fulfillJson(route, healthFixture);
    });
    await page.route(`${baseURL}api/version`, async (route) => {
      requestedAt.version.push(Date.now());
      if (routesFailing) await route.abort("failed");
      else await fulfillJson(route, versionFixture);
    });
    await page.route(`${baseURL}api/jobs`, async (route) => {
      requestedAt.jobs.push(Date.now());
      if (routesFailing && detailFailedDuringOutage) await route.abort("failed");
      else await fulfillJson(route, jobsFixture);
    });
    await page.route(`${baseURL}api/jobs/${JOB_ID}`, async (route) => {
      requestedAt.detail.push(Date.now());
      if (routesFailing) {
        detailFailedDuringOutage = true;
        await route.abort("failed");
      } else {
        await fulfillJson(route, jobDetailFixture);
      }
    });

    const response = await page.goto("/");
    if (response === null) throw new Error("The document navigation returned no response.");
    expect(response.status()).toBe(200);
    await expect(response.headerValue("content-type")).resolves.toBe("text/html; charset=utf-8");
    await expect(response.headerValue("cache-control")).resolves.toBe("no-store");
    await expect(response.headerValue("x-content-type-options")).resolves.toBe("nosniff");
    await expect(response.headerValue("referrer-policy")).resolves.toBe("no-referrer");
    await expect(response.headerValue("x-frame-options")).resolves.toBe("DENY");
    await expect(response.headerValue("content-security-policy")).resolves.toBe(EXACT_CSP);
    const shellHtml = await response.text();
    expect(shellHtml).toContain("Local service reported unhealthy.");
    expect(shellHtml).toContain("Local service is healthy.");
    initialImgCount = await page.locator("img").count();
    initialScriptCount = await page.locator("script").count();
    expect(initialImgCount).toBe(0);
    expect(initialScriptCount).toBe(1);
  });

  await test.step("2. Prove the initial network is only the document and four same-origin GET APIs", async () => {
    await expect.poll(() => requestedAt.health.length).toBeGreaterThanOrEqual(1);
    await expect.poll(() => requestedAt.version.length).toBe(1);
    await expect.poll(() => requestedAt.jobs.length).toBeGreaterThanOrEqual(1);
    await expect.poll(() => requestedAt.detail.length).toBeGreaterThanOrEqual(1);
    await expect(page.locator("#health-status")).toHaveText("Local service is healthy.");
    assertNetworkContract();
  });

  await test.step("3. Verify React mount, default status routing, and both navigation directions", async () => {
    await expect(page.locator("#root #status-view")).toBeVisible();
    await expect(page.locator("#root #jobs-view")).toBeHidden();
    await expect(page.locator("#status-link")).toHaveAttribute("aria-current", "page");
    await expect(page.locator("#jobs-link")).toHaveAttribute("aria-current", "false");

    await page.evaluate(() => {
      window.location.hash = "#unknown";
    });
    await expect(page.locator("#status-view")).toBeVisible();
    await expect(page.locator("#status-link")).toHaveAttribute("aria-current", "page");

    await page.locator("#jobs-link").click();
    await expect(page.locator("#jobs-view")).toBeVisible();
    await expect(page.locator("#status-view")).toBeHidden();
    await expect(page.locator("#jobs-link")).toHaveAttribute("aria-current", "page");
    await expect(page.locator("#status-link")).toHaveAttribute("aria-current", "false");

    await page.locator("#status-link").click();
    await expect(page.locator("#status-view")).toBeVisible();
    await expect(page.locator("#jobs-view")).toBeHidden();
    await expect(page.locator("#status-link")).toHaveAttribute("aria-current", "page");
  });

  await test.step("4. Observe real-time health, jobs, and selected-detail polling with one-shot version", async () => {
    await expect.poll(() => requestedAt.jobs.length, { timeout: 3_500 }).toBeGreaterThanOrEqual(2);
    await expect.poll(() => requestedAt.detail.length, { timeout: 3_500 }).toBeGreaterThanOrEqual(2);
    await expect.poll(() => requestedAt.health.length, { timeout: 7_000 }).toBeGreaterThanOrEqual(2);

    expect(requestedAt.jobs[1] - requestedAt.jobs[0]).toBeGreaterThanOrEqual(1_750);
    expect(requestedAt.jobs[1] - requestedAt.jobs[0]).toBeLessThan(3_500);
    expect(requestedAt.detail[1] - requestedAt.detail[0]).toBeGreaterThanOrEqual(1_750);
    expect(requestedAt.detail[1] - requestedAt.detail[0]).toBeLessThan(3_500);
    expect(requestedAt.health[1] - requestedAt.health[0]).toBeGreaterThanOrEqual(4_500);
    expect(requestedAt.health[1] - requestedAt.health[0]).toBeLessThan(7_000);
    expect(requestedAt.version).toHaveLength(1);
  });

  await test.step("5. Select the retained job and require its exact detail request path", async () => {
    await page.locator("#jobs-link").click();
    const button = page.locator("#job-list .job-button", { hasText: JOB_ID });
    await expect(button).toBeVisible();
    await expect(button).toHaveClass(/selected/);

    const before = requestedAt.detail.length;
    const detailRequest = page.waitForRequest(
      (request) => new URL(request.url()).pathname.match(/^\/api\/jobs\/[0-9a-f]{32}$/) !== null,
      { timeout: 3_500 },
    );
    await button.click();
    expect(new URL((await detailRequest).url()).pathname).toBe(`/api/jobs/${JOB_ID}`);
    await expect.poll(() => requestedAt.detail.length).toBeGreaterThan(before);
  });

  await test.step("6. Render every hostile free-text field as inert visible text", async () => {
    await page.locator("#status-link").click();
    const versionFields = page.locator("#version-fields");
    await expect(versionFields).toBeVisible();
    await expect(versionFields).toContainText(versionFixture.version);

    await page.locator("#jobs-link").click();
    const detailFields = page.locator("#detail-fields");
    const logs = page.locator("#job-logs");
    await expect(detailFields).toBeVisible();
    await expect(logs).toBeVisible();
    await expect(detailFields).toContainText(jobDetailFixture.current_step);
    await expect(detailFields).toContainText(jobDetailFixture.error.code);
    await expect(detailFields).toContainText(jobDetailFixture.error.message);
    await expect(logs).toContainText(jobDetailFixture.logs[0].code);
    await expect(logs).toContainText(jobDetailFixture.logs[0].message);

    expect(await page.locator("img").count()).toBe(initialImgCount);
    expect(await page.locator("script").count()).toBe(initialScriptCount);
    expect(await page.locator("[onerror], [onload], [onclick], [srcdoc]").count()).toBe(0);
    expect(
      await page.locator('[href^="javascript:" i], [src^="javascript:" i], [action^="javascript:" i]').count(),
    ).toBe(0);
    expect(await page.evaluate(() => (window as Window & { __e2eSentinel?: string }).__e2eSentinel)).toBe(
      "intact",
    );
    expect(observedRequests.some((request) => /e2e-(?:version|error)-probe/.test(request.url()))).toBe(false);
    assertNetworkContract();
  });

  await test.step("7. Keep exactly the three intended live regions", async () => {
    const liveIds = await page.locator("[aria-live]").evaluateAll((nodes) =>
      nodes.map((node) => node.id).sort(),
    );
    expect(liveIds).toEqual(["health-status", "jobs-status", "live-feedback"]);
    expect(liveIds).toHaveLength(3);
    await expect(page.locator("#version-status")).not.toHaveAttribute("aria-live");
    await expect(page.locator("#detail-status")).not.toHaveAttribute("aria-live");
  });

  await test.step("8. Fail polling, retain cached data, and surface the stale unavailable catalog", async () => {
    routesFailing = true;

    await expect(page.locator("#detail-status")).toHaveText(`${UNAVAILABLE} Stale.`, { timeout: 4_000 });
    await expect(page.locator("#jobs-status")).toHaveText(`${UNAVAILABLE} Stale.`, { timeout: 6_500 });
    await expect(page.locator("#health-status")).toHaveText(`${UNAVAILABLE} Stale.`, { timeout: 7_000 });
    await expect(page.locator("#detail-status")).toHaveClass(/stale/);
    await expect(page.locator("#jobs-status")).toHaveClass(/stale/);
    await expect(page.locator("#health-status")).toHaveClass(/stale/);

    await expect(page.locator("#health-fields")).toContainText("reconstruction_web");
    await expect(page.locator("#version-fields")).toContainText(versionFixture.version);
    await expect(page.locator("#job-list")).toContainText(JOB_ID);
    await expect(page.locator("#detail-fields")).toContainText(jobDetailFixture.error.message);
    await expect(page.locator("#job-logs")).toContainText(jobDetailFixture.logs[0].message);
  });

  await test.step("9. Restore API success and announce reconnection", async () => {
    routesFailing = false;
    await expect(page.locator("#live-feedback")).toHaveText("Local API reconnected.", { timeout: 4_000 });
    await expect(page.locator("#jobs-status")).toHaveText("Retained jobs are current.");
    await expect(page.locator("#detail-status")).toHaveText("Job detail is current.");
    expect(requestedAt.version).toHaveLength(1);
  });

  await test.step("10. Finish with the closed network surface; Playwright owns server teardown", async () => {
    assertNetworkContract();
    expect(await page.locator("img").count()).toBe(0);
    expect(await page.evaluate(() => (window as Window & { __e2eSentinel?: string }).__e2eSentinel)).toBe(
      "intact",
    );
  });
});
