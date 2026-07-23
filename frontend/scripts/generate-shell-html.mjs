import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const frontendRoot = resolve(import.meta.dirname, "..");
const inputPath = resolve(frontendRoot, "dist", "index.html");
const outputPath = resolve(frontendRoot, "..", "reconstruction_web", "shell_html.py");
const bytesPerLiteral = 24;

function fail(message) {
  throw new Error(`shell HTML generation failed: ${message}`);
}

function validateInput(bytes) {
  if (bytes.includes(0x0d)) {
    fail("dist/index.html contains a CR byte; CRLF and lone CR are forbidden");
  }

  try {
    new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    fail("dist/index.html is not valid UTF-8");
  }
}

function pythonBytesLiteral(bytes) {
  const lines = [];
  for (let offset = 0; offset < bytes.length; offset += bytesPerLiteral) {
    const chunk = bytes.subarray(offset, offset + bytesPerLiteral);
    const escaped = [...chunk].map((byte) => `\\x${byte.toString(16).padStart(2, "0")}`).join("");
    lines.push(`    b"${escaped}"`);
  }
  if (lines.length === 0) lines.push('    b""');
  return lines.join("\n");
}

const inputBytes = await readFile(inputPath);
validateInput(inputBytes);

const moduleSource =
  "# generated — do not edit\n" +
  "from typing import Final\n\n" +
  "SHELL_HTML: Final[bytes] = (\n" +
  `${pythonBytesLiteral(inputBytes)}\n` +
  ")\n";

await writeFile(outputPath, moduleSource, { encoding: "utf-8" });
console.log(`generate-shell-html PASS (${inputBytes.byteLength} input bytes)`);
