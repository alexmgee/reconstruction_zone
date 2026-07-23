import { resolve } from "node:path";

import { defineConfig } from "@playwright/test";

const E2E_PORT = "18765";
const BASE_URL = `http://127.0.0.1:${E2E_PORT}/`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  retries: 0,
  workers: 1,
  reporter: "list",
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
  use: {
    baseURL: BASE_URL,
    trace: "off",
    video: "off",
  },
  webServer: {
    command: "python -B -m frontend.e2e.serve_shell",
    cwd: resolve(import.meta.dirname, ".."),
    env: { E2E_PORT },
    url: BASE_URL,
    reuseExistingServer: false,
  },
});
