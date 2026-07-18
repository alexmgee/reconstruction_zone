import { lstat, readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const dist = resolve(root, "dist");
const indexPath = resolve(dist, "index.html");

// Keep synchronized with tests/test_web_server_skeleton.py:24-40,379-395 and
// tests/test_web_file_access.py:29-45. These are the exact package-wide source
// guard tokens whose appearance in generated Python would fail pytest.
const packageGuardExactTokens = [
  "Path.home(",
  "expanduser",
  ".studio_prefs",
  "SimpleHTTPRequestHandler",
  "tracker.json",
  "activity_log.json",
  "D:\\",
  "D:/",
  "C:\\",
  "C:/",
  ".prep360",
  ".reconstruction_zone",
];
const packageGuardCaseInsensitiveTokens = [
  "fastapi",
  "flask",
  "aiohttp",
  "uvicorn",
  "asyncio",
  "multiprocessing",
  "subprocess",
  "concurrent.futures",
  "SimpleHTTPRequestHandler",
];

function fail(message) {
  throw new Error(`single-file verification failed: ${message}`);
}

async function assertOnlyIndexHtml() {
  const distStat = await lstat(dist);
  if (!distStat.isDirectory() || distStat.isSymbolicLink()) {
    fail("dist must be a real directory");
  }

  const entries = await readdir(dist, { withFileTypes: true });
  if (entries.length !== 1 || entries[0].name !== "index.html") {
    fail(`dist must contain only index.html; found: ${entries.map((entry) => entry.name).join(", ")}`);
  }

  const entry = entries[0];
  const indexStat = await lstat(indexPath);
  if (!entry.isFile() || entry.isSymbolicLink() || !indexStat.isFile() || indexStat.isSymbolicLink()) {
    fail("dist/index.html must be one regular, non-symlink file");
  }
}

function decodeUtf8(bytes) {
  if (bytes.includes(0x0d)) {
    fail("dist/index.html contains a CR byte; output must be LF-only");
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    fail("dist/index.html is not valid UTF-8");
  }
}

function assertNoPackageGuardTokens(source) {
  for (const token of packageGuardExactTokens) {
    if (source.includes(token)) fail(`package-source guard token is forbidden: ${JSON.stringify(token)}`);
  }

  const lowered = source.toLowerCase();
  for (const token of packageGuardCaseInsensitiveTokens) {
    if (lowered.includes(token.toLowerCase())) {
      fail(`case-insensitive package-source guard token is forbidden: ${JSON.stringify(token)}`);
    }
  }
}

function attributes(source) {
  const result = [];
  const pattern = /([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  for (const match of source.matchAll(pattern)) {
    result.push({ name: match[1].toLowerCase(), value: match[2] ?? match[3] ?? match[4] ?? "" });
  }
  return result;
}

function maskJavaScriptLiteralsAndComments(source) {
  let result = "";
  let index = 0;
  let state = "code";
  let quote = "";

  while (index < source.length) {
    const char = source[index];
    const next = source[index + 1];
    if (state === "code") {
      if (char === "/" && next === "/") {
        result += "  ";
        index += 2;
        state = "line-comment";
      } else if (char === "/" && next === "*") {
        result += "  ";
        index += 2;
        state = "block-comment";
      } else if (char === "\"" || char === "'" || char === "`") {
        quote = char;
        result += " ";
        index += 1;
        state = "string";
      } else {
        result += char;
        index += 1;
      }
    } else if (state === "line-comment") {
      result += char === "\n" ? "\n" : " ";
      index += 1;
      if (char === "\n") state = "code";
    } else if (state === "block-comment") {
      if (char === "*" && next === "/") {
        result += "  ";
        index += 2;
        state = "code";
      } else {
        result += char === "\n" ? "\n" : " ";
        index += 1;
      }
    } else if (char === "\\") {
      result += "  ";
      index += 2;
    } else if (char === quote) {
      result += " ";
      index += 1;
      state = "code";
    } else {
      result += char === "\n" ? "\n" : " ";
      index += 1;
    }
  }

  return result;
}

await assertOnlyIndexHtml();
const bytes = await readFile(indexPath);
const html = decodeUtf8(bytes);
assertNoPackageGuardTokens(html);

const scriptMatches = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)];
const styleMatches = [...html.matchAll(/<style\b([^>]*)>([\s\S]*?)<\/style\s*>/gi)];
const documentMarkup = html
  .replace(/(<script\b[^>]*>)[\s\S]*?(<\/script\s*>)/gi, "$1$2")
  .replace(/(<style\b[^>]*>)[\s\S]*?(<\/style\s*>)/gi, "$1$2");
const scriptOpenTags = [...documentMarkup.matchAll(/<script\b/gi)].length;
const styleOpenTags = [...documentMarkup.matchAll(/<style\b/gi)].length;
if (scriptMatches.length !== 1 || scriptOpenTags !== 1) {
  fail(`expected exactly one inline script; found ${scriptOpenTags}`);
}
if (styleMatches.length !== 1 || styleOpenTags !== 1) {
  fail(`expected exactly one inline style; found ${styleOpenTags}`);
}
if (attributes(scriptMatches[0][1]).some(({ name }) => name === "src")) fail("script src is forbidden");

const tags = [...documentMarkup.matchAll(/<([a-z][a-z0-9:-]*)\b([^>]*)>/gi)];
const forbiddenUrlAttributes = new Set([
  "action",
  "background",
  "cite",
  "data",
  "formaction",
  "manifest",
  "poster",
  "src",
  "srcset",
  "xlink:href",
]);
let faviconCount = 0;
for (const tag of tags) {
  const tagName = tag[1].toLowerCase();
  const attrs = attributes(tag[2]);
  const attrMap = new Map(attrs.map(({ name, value }) => [name, value]));
  for (const { name, value } of attrs) {
    if (/^on[a-z]+$/.test(name)) fail(`inline event attribute ${name} is forbidden`);
    if (/^\s*javascript:/i.test(value)) fail(`javascript: value in ${name} is forbidden`);
    if (forbiddenUrlAttributes.has(name)) fail(`URL-bearing attribute ${name} is forbidden`);
    if (name === "href") {
      const isFavicon = tagName === "link" && attrMap.get("rel")?.toLowerCase() === "icon" && value === "data:,";
      const isHashAnchor = tagName === "a" && /^#[^\s]*$/.test(value);
      if (!isFavicon && !isHashAnchor) fail(`external or unapproved href is forbidden: ${value}`);
      if (isFavicon) faviconCount += 1;
    }
  }
}
if (faviconCount !== 1) fail(`expected exactly one data: favicon; found ${faviconCount}`);

const css = styleMatches[0][2];
if (/@import\b/i.test(css)) fail("CSS @import is forbidden");
if (/url\s*\(/i.test(css)) fail("CSS url() is forbidden");

const script = scriptMatches[0][2];
const executable = maskJavaScriptLiteralsAndComments(script);
if (/(?<![A-Za-z0-9_$])import(?![A-Za-z0-9_$])\s*(?:\(|["'{*]|[A-Za-z_$])/m.test(executable)) {
  fail("residual JavaScript import is forbidden");
}
if (/\bexport\s+(?:default|\{|\*|const|let|var|function|class)\b/m.test(executable)) fail("residual JavaScript export is forbidden");
if (/\bimport\.meta\b/.test(executable)) fail("residual import.meta is forbidden");
if (/\bnew\s+(?:SharedWorker|Worker)\s*\(/.test(executable)) fail("worker construction is forbidden");
if (/\bnew\s+URL\s*\(/.test(executable)) fail("un-inlined URL construction is forbidden");
if (/sourceMappingURL|sourceURL|\.map(?:[?#\"']|$)/i.test(html)) fail("source-map reference is forbidden");
if (/(?:^|[\"'(/])assets\//i.test(html)) fail("un-inlined asset path is forbidden");

console.log(`verify-single-file PASS (${bytes.byteLength} bytes)`);
