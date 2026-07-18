import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { describe, expect, it } from "vitest";

const frontendRoot = resolve(import.meta.dirname, "..");
const sourceRoot = join(frontendRoot, "src");

function sourceFiles(directory: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) {
      files.push(...sourceFiles(path));
    } else if (path !== join(sourceRoot, "api", "generated.ts")) {
      files.push(path);
    }
  }
  return files;
}

const inspectedFiles = [...sourceFiles(sourceRoot), join(frontendRoot, "index.html")];

describe("application source security", () => {
  it("contains no raw markup sinks, alternate transports, persistence, or background execution", () => {
    const forbidden: Array<[string, RegExp]> = [
      ["raw React markup sink", /dangerouslySetInnerHTML/],
      ["direct markup assignment", /\.innerHTML\b/],
      ["adjacent markup insertion", /insertAdjacentHTML/],
      ["document writer", /document\.write/],
      ["contextual fragment", /createContextualFragment/],
      ["DOM parser", /\bDOMParser\b/],
      ["web socket", /\bWebSocket\b/],
      ["XML HTTP transport", /\bXMLHttpRequest\b/],
      ["event stream transport", /\bEventSource\b/],
      ["beacon transport", /\bsendBeacon\b/],
      ["worker", /\b(?:SharedWorker|Worker)\b/],
      ["service worker", /serviceWorker/],
      ["local persistence", /\b(?:localStorage|sessionStorage|indexedDB)\b/],
      ["cookies", /document\.cookie|cookieStore/],
    ];
    for (const file of inspectedFiles) {
      const source = readFileSync(file, "utf8");
      for (const [label, pattern] of forbidden) {
        expect(source, `${label} in ${relative(frontendRoot, file)}`).not.toMatch(pattern);
      }
    }
  });

  it("uses closed named-field rendering and no decoded-object JSX prop spread", () => {
    for (const file of sourceFiles(sourceRoot)) {
      const source = readFileSync(file, "utf8");
      expect(source, `Object.keys in ${relative(frontendRoot, file)}`).not.toMatch(/Object\.keys\s*\(/);
      expect(source, `Object.entries in ${relative(frontendRoot, file)}`).not.toMatch(/Object\.entries\s*\(/);
      expect(source, `for-in in ${relative(frontendRoot, file)}`).not.toMatch(/\bfor\s*\([^)]*\bin\b[^)]*\)/);
      expect(source, `JSX prop spread in ${relative(frontendRoot, file)}`).not.toMatch(
        /<[A-Za-z][^>]*\{\s*\.\.\./s,
      );
    }
  });

  it("keeps fetch ownership and protocol surface read-only", () => {
    const fetchOwners: string[] = [];
    for (const file of sourceFiles(sourceRoot)) {
      const source = readFileSync(file, "utf8");
      if (/\bfetch\s*\(/.test(source)) fetchOwners.push(relative(frontendRoot, file).replaceAll("\\", "/"));
      expect(source).not.toMatch(/\/cancel\b/);
      expect(source).not.toMatch(/["'](?:POST|PUT|PATCH|DELETE)["']/);
    }
    expect(fetchOwners).toEqual(["src/api/client.ts"]);
  });

  it("keeps the mount document free of inline handlers and executable URLs", () => {
    const documentSource = readFileSync(join(frontendRoot, "index.html"), "utf8");
    expect(documentSource).not.toMatch(/\son[a-z]+\s*=/i);
    expect(documentSource).not.toMatch(/javascript\s*:/i);
  });
});
