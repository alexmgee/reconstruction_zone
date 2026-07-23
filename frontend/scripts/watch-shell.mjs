import { spawn } from "node:child_process";
import { lstat, realpath } from "node:fs/promises";
import { resolve } from "node:path";
import { build } from "vite";

const frontendRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(frontendRoot, "..");
const verifyScript = resolve(import.meta.dirname, "verify-single-file.mjs");
const generateScript = resolve(import.meta.dirname, "generate-shell-html.mjs");

function fail(message) {
  console.error(`watch-shell: ${message}`);
  process.exitCode = 1;
}

function parseArguments(argv) {
  let stateRoot;
  let port;

  for (let index = 0; index < argv.length; index += 1) {
    const option = argv[index];
    const value = argv[index + 1];
    if ((option !== "--state-root" && option !== "--port") || value === undefined || value.startsWith("--")) {
      throw new Error("usage: node frontend/scripts/watch-shell.mjs --state-root <path> --port <port>");
    }
    if (option === "--state-root") {
      if (stateRoot !== undefined) throw new Error("--state-root must be specified exactly once");
      stateRoot = value;
    } else {
      if (port !== undefined) throw new Error("--port must be specified exactly once");
      port = value;
    }
    index += 1;
  }

  if (!stateRoot?.trim()) throw new Error("--state-root is required");
  if (!port || !/^\d+$/.test(port) || Number(port) > 65535) {
    throw new Error("--port must be an integer from 0 through 65535");
  }
  return { stateRoot, port };
}

function run(command, args, options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { stdio: "inherit", ...options });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) resolvePromise();
      else reject(new Error(`${command} failed (${signal ?? `exit ${code}`})`));
    });
  });
}

async function validateStateRoot(candidate) {
  const stats = await lstat(candidate).catch(() => null);
  if (!stats?.isDirectory()) throw new Error("--state-root must name an existing directory");
  const resolvedRoot = await realpath(candidate);
  const validator =
    "import sys; from reconstruction_web.state import resolve_state_root; " +
    "print(resolve_state_root(sys.argv[1]))";
  await run("python", ["-B", "-c", validator, resolvedRoot], { cwd: repositoryRoot });
  return resolvedRoot;
}

let serverChild;
let shuttingDown = false;
const stoppingServers = new WeakSet();

async function stopServer() {
  const child = serverChild;
  serverChild = undefined;
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  stoppingServers.add(child);

  await new Promise((resolvePromise, reject) => {
    const timeout = setTimeout(() => reject(new Error("Python server did not stop within five seconds")), 5_000);
    child.once("exit", () => {
      clearTimeout(timeout);
      resolvePromise();
    });
    if (!child.kill("SIGINT")) {
      clearTimeout(timeout);
      reject(new Error("could not signal the Python server to stop"));
    }
  });
}

function startServer(stateRoot, port) {
  const child = spawn(
    "python",
    ["-m", "reconstruction_web", "--state-root", stateRoot, "--port", port],
    { cwd: repositoryRoot, stdio: "inherit" },
  );
  serverChild = child;
  child.once("error", (error) => fail(`Python server failed to start: ${error.message}`));
  child.once("exit", (code, signal) => {
    if (serverChild === child) serverChild = undefined;
    if (!shuttingDown && !stoppingServers.has(child)) {
      fail(`Python server exited unexpectedly (${signal ?? `exit ${code}`})`);
    }
  });
}

async function main() {
  const { stateRoot: stateRootArgument, port } = parseArguments(process.argv.slice(2));
  const stateRoot = await validateStateRoot(stateRootArgument);

  const devLoopPlugin = {
    name: "watch-shell-dev-loop",
    apply: "build",
    async closeBundle() {
      await run(process.execPath, [verifyScript], { cwd: frontendRoot });
      await run(process.execPath, [generateScript], { cwd: frontendRoot });
      await stopServer();
      startServer(stateRoot, port);
    },
  };

  const watcher = await build({
    root: frontendRoot,
    configFile: resolve(frontendRoot, "vite.config.ts"),
    build: { watch: {} },
    plugins: [devLoopPlugin],
  });

  const shutdown = async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    await watcher.close();
    await stopServer();
  };
  process.once("SIGINT", () => shutdown().catch((error) => fail(error.message)));
  process.once("SIGTERM", () => shutdown().catch((error) => fail(error.message)));
}

main().catch((error) => fail(error.message));
