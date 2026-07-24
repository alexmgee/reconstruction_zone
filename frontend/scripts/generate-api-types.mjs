import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { compile } from "json-schema-to-typescript";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const FRONTEND_DIR = resolve(SCRIPT_DIR, "..");
const CONTRACTS_DIR = join(FRONTEND_DIR, "contracts");
const GENERATED_PATH = join(FRONTEND_DIR, "src", "api", "generated.ts");
const SCHEMA_FILES = [
  "common.schema.json",
  "error-response.schema.json",
  "health-response.schema.json",
  "job-detail-response.schema.json",
  "jobs-response.schema.json",
  "project-detail-response.schema.json",
  "projects-response.schema.json",
  "version-response.schema.json",
];

function stableStringify(value) {
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    const entries = Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`);
    return `{${entries.join(",")}}`;
  }
  return JSON.stringify(value);
}

async function loadSchemaSet() {
  const discovered = (await readdir(CONTRACTS_DIR))
    .filter((name) => name.endsWith(".schema.json"))
    .sort();
  if (stableStringify(discovered) !== stableStringify(SCHEMA_FILES)) {
    throw new Error(
      `Expected exactly these schema files: ${SCHEMA_FILES.join(", ")}; found: ${discovered.join(", ")}`,
    );
  }

  const schemaSet = {};
  for (const name of SCHEMA_FILES) {
    schemaSet[name] = JSON.parse(await readFile(join(CONTRACTS_DIR, name), "utf8"));
  }
  return schemaSet;
}

async function generate() {
  const schemaSet = await loadSchemaSet();
  const digest = createHash("sha256").update(stableStringify(schemaSet), "utf8").digest("hex");
  const responseSchemas = SCHEMA_FILES.filter((name) => name !== "common.schema.json");
  const aggregateSchema = {
    $schema: "https://json-schema.org/draft/2020-12/schema",
    title: "ApiResponse",
    anyOf: responseSchemas.map((name) => ({ $ref: name })),
  };
  const declarations = await compile(aggregateSchema, "ApiResponse", {
    bannerComment: "",
    cwd: CONTRACTS_DIR,
    style: {
      endOfLine: "lf",
      printWidth: 120,
      semi: true,
      singleQuote: false,
      tabWidth: 2,
      useTabs: false,
    },
  });
  return (
    "/* generated — do not edit\n" +
    " * Source: frontend/contracts/*.schema.json\n" +
    ` * Canonical schema SHA-256: ${digest}\n` +
    " */\n" +
    `export const SCHEMA_SHA256 = "${digest}" as const;\n\n` +
    declarations.replace(/^\s+/, "").replace(/\r\n?/g, "\n")
  );
}

async function writeGenerated(outputPath) {
  const output = await generate();
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, output, "utf8");
}

async function checkGenerated() {
  const temporaryDirectory = await mkdtemp(join(tmpdir(), "web7-api-types-"));
  const temporaryPath = join(temporaryDirectory, "generated.ts");
  try {
    await writeGenerated(temporaryPath);
    const [expected, actual] = await Promise.all([
      readFile(temporaryPath),
      readFile(GENERATED_PATH).catch(() => Buffer.alloc(0)),
    ]);
    if (!expected.equals(actual)) {
      throw new Error("frontend/src/api/generated.ts is stale; run npm run codegen");
    }
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
}

const arguments_ = process.argv.slice(2);
if (arguments_.length === 0) {
  await writeGenerated(GENERATED_PATH);
} else if (arguments_.length === 1 && arguments_[0] === "--check") {
  await checkGenerated();
} else {
  throw new Error(`Unsupported arguments: ${arguments_.join(" ")}`);
}
