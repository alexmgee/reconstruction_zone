"""WEB5 compiled-in browser shell tests."""

from __future__ import annotations

import ast
import builtins
import http.client
import json
import re
import subprocess
import sys
import threading
import time
from html.parser import HTMLParser
from pathlib import Path

import pytest

from reconstruction_web import __version__
from reconstruction_web.jobs import JobRegistry
from reconstruction_web.server import HOST, make_server, shutdown_server
from reconstruction_web.shell_html import SHELL_HTML
from reconstruction_web.state import build_state_config


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "reconstruction_web"
SHELL_PATH = PACKAGE_ROOT / "shell_html.py"
FIXED_404 = b'{"error": "not_found", "message": "Route not found."}'
EXACT_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "connect-src 'self'; img-src data:; font-src 'none'; object-src 'none'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
SUMMARY_FIELDS = {
    "job_id",
    "job_type",
    "state",
    "progress",
    "current_step",
    "created_at",
    "started_at",
    "finished_at",
}
DETAIL_FIELDS = SUMMARY_FIELDS | {"logs", "result", "error"}


def _shell_source_tree_and_assignment() -> tuple[str, ast.Module, ast.AnnAssign]:
    source = SHELL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SHELL_PATH))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "SHELL_HTML"
    ]
    assert len(assignments) == 1
    return source, tree, assignments[0]


def _literal_span(source: str, value: ast.expr) -> tuple[str, str, str]:
    assert value.lineno is not None
    assert value.col_offset is not None
    assert value.end_lineno is not None
    assert value.end_col_offset is not None

    lines = source.splitlines(keepends=True)
    start = sum(len(line.encode("utf-8")) for line in lines[: value.lineno - 1]) + value.col_offset
    end = sum(len(line.encode("utf-8")) for line in lines[: value.end_lineno - 1]) + value.end_col_offset
    source_bytes = source.encode("utf-8")
    before = source_bytes[:start].decode("utf-8", errors="strict")
    literal = source_bytes[start:end].decode("utf-8", errors="strict")
    after = source_bytes[end:].decode("utf-8", errors="strict")
    assert before + literal + after == source
    return before, literal, after


class ShellInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self._capture: str | None = None
        self._chunks: dict[str, list[str]] = {"style": [], "script": []}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        self.tags.append((lowered, dict(attrs)))
        if lowered in self._chunks:
            self._capture = lowered

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == self._capture:
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._chunks[self._capture].append(data)

    @property
    def style(self) -> str:
        return "".join(self._chunks["style"])

    @property
    def script(self) -> str:
        return "".join(self._chunks["script"])


@pytest.fixture
def shell_server(tmp_path):
    state_root = tmp_path / "web_state"
    state_root.mkdir()
    registry = JobRegistry(_step_wait=0)
    server = make_server(build_state_config(state_root), port=0, job_registry=registry)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, registry
    finally:
        shutdown_server(server, thread, join_timeout=3)


def _request(server, method: str, target: str) -> tuple[int, dict[str, str], bytes]:
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request(method, target)
        response = connection.getresponse()
        headers = {name.lower(): value for name, value in response.getheaders()}
        return response.status, headers, response.read()
    finally:
        connection.close()


def _served_shell(server) -> tuple[dict[str, str], bytes, ShellInspector]:
    status, headers, body = _request(server, "GET", "/")
    assert status == 200
    parser = ShellInspector()
    parser.feed(body.decode("utf-8"))
    return headers, body, parser


def test_root_serves_exact_compiled_shell_with_hardening_headers(shell_server):
    server, _ = shell_server
    headers, body, parser = _served_shell(server)

    assert body == SHELL_HTML
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert headers["content-length"] == str(len(body))
    assert headers["cache-control"] == "no-store"
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["x-frame-options"] == "DENY"
    assert headers["content-security-policy"] == EXACT_CSP
    assert sum(tag == "style" for tag, _ in parser.tags) == 1
    assert sum(tag == "script" for tag, _ in parser.tags) == 1


@pytest.mark.parametrize(
    "target",
    [
        "/index.html",
        "/favicon.ico",
        "/static",
        "/static/",
        "/static/app.js",
        "/assets/app.js",
        "/nested/arbitrary/path",
        "/?query=1",
        "/?",
        "/;x",
        "/;x?y=1",
        "/./",
        "/%2e%2e/secret",
        "/static/%2e%2e/server.py",
    ],
)
def test_only_exact_raw_root_target_serves_shell(shell_server, target: str):
    server, _ = shell_server
    status, headers, body = _request(server, "GET", target)
    assert status == 404
    assert headers["content-type"] == "application/json; charset=utf-8"
    assert body == FIXED_404
    assert b"<!doctype html>" not in body.lower()


def test_post_root_keeps_fixed_json_404(shell_server):
    server, _ = shell_server
    status, headers, body = _request(server, "POST", "/")
    assert status == 404
    assert headers["content-type"] == "application/json; charset=utf-8"
    assert body == FIXED_404
    assert b"text/html" not in body


def test_shell_assignment_is_plain_ast_literal_and_has_no_file_loader():
    source, _, assignment = _shell_source_tree_and_assignment()
    value = assignment.value
    assert isinstance(value, ast.Constant)
    assert isinstance(value.value, bytes)

    before, literal, after = _literal_span(source, value)
    literal_expression = ast.parse(f"({literal})", mode="eval").body
    assert isinstance(literal_expression, ast.Constant)
    assert literal_expression.value == value.value

    excised_source = before + after
    for forbidden in ("open(", "read_bytes", "read_text", "Path(", "importlib", "__file__"):
        assert forbidden not in excised_source

    replacement_tree = ast.parse(before + 'b""' + after)
    replacement_assignments = [
        node for node in replacement_tree.body if isinstance(node, ast.AnnAssign)
    ]
    assert len(replacement_assignments) == 1
    replacement_value = replacement_assignments[0].value
    assert isinstance(replacement_value, ast.Constant)
    assert replacement_value.value == b""


def test_shell_delivery_uses_no_filesystem_read_after_construction(shell_server, monkeypatch):
    server, _ = shell_server

    def blocked(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("filesystem read attempted while serving shell")

    monkeypatch.setattr(builtins, "open", blocked)
    monkeypatch.setattr(Path, "read_bytes", blocked)
    monkeypatch.setattr(Path, "read_text", blocked)
    status, _, body = _request(server, "GET", "/")
    assert status == 200
    assert body == SHELL_HTML


def test_no_static_tree_or_non_python_package_asset_exists():
    package_files = [
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    assert package_files
    assert all(path.suffix == ".py" for path in package_files)
    assert not any(
        path.is_dir() and path.name.lower() in {"static", "assets", "templates"}
        for path in PACKAGE_ROOT.rglob("*")
    )


def test_served_html_has_no_external_resource_or_runtime_reference(shell_server):
    server, _ = shell_server
    headers, _, parser = _served_shell(server)
    assert headers["content-security-policy"] == EXACT_CSP

    resource_attributes = {"src", "href", "action", "formaction", "poster", "data"}
    observed_references: list[tuple[str, str, str]] = []
    for tag, attrs in parser.tags:
        for name, value in attrs.items():
            if name.lower() in resource_attributes and value is not None:
                observed_references.append((tag, name.lower(), value))
    assert observed_references == [("link", "href", "data:,")]
    # React creates these anchors at runtime; frontend/tests/shell.test.tsx owns
    # their rendered-DOM behavior. The compiled declarations remain closed here.
    assert re.findall(r'\bhref:\s*"(#[^"]+)"', parser.script) == ["#status", "#jobs"]
    assert not any(tag in {"img", "iframe", "object", "embed", "audio", "video"} for tag, _ in parser.tags)
    assert "@import" not in parser.style.lower()
    assert re.search(r"url\s*\(", parser.style, flags=re.IGNORECASE) is None


def test_frontend_fetches_are_same_origin_get_only(shell_server):
    server, _ = shell_server
    _, _, parser = _served_shell(server)
    script = parser.script
    for path in ("/api/health", "/api/version", "/api/jobs"):
        assert f'"{path}"' in script
    assert "`${JOBS_PATH}/${encodeURIComponent(jobId)}`" in script
    assert r"/^\/api\/jobs\/[0-9a-f]{32}$/" in script
    assert re.search(
        r"\bmethod\s*:\s*['\"](?:POST|PUT|PATCH|DELETE)['\"]",
        script,
    ) is None
    assert "/cancel" not in script
    # GET options and job-ID rejection-before-fetch are owned by
    # frontend/tests/api.test.ts; this test covers the compiled served artifact.


def test_rendering_uses_closed_fields_and_text_nodes_only(shell_server):
    server, _ = shell_server
    headers, _, parser = _served_shell(server)
    assert headers["content-security-policy"] == EXACT_CSP
    assert re.findall(r'\blabel:\s*"([^"]+)"', parser.script) == [
        "Healthy",
        "Service",
        "Mode",
        "Local only",
        "State configured",
        "Service",
        "Version",
        "Source",
        "Job ID",
        "Type",
        "State",
        "Progress",
        "Current step",
        "Created",
        "Started",
        "Finished",
        "Result kind",
        "Steps completed",
        "Error code",
        "Error",
    ]
    for visible_heading in (
        "Runtime status",
        "Representative jobs",
        "Local service",
        "Version",
        "Retained jobs",
        "Selected job",
        "Logs",
    ):
        assert visible_heading in parser.script
    # Closed DOM rendering and injection-sink bans are owned by
    # frontend/tests/shell.test.tsx and frontend/tests/source-security.test.ts.


def test_polling_lifecycle_and_visible_state_catalog_are_present(shell_server):
    server, _ = shell_server
    _, body, _ = _served_shell(server)
    text = body.decode("utf-8")
    for state_text in (
        "Loading local status...",
        "Loading version...",
        "Loading jobs...",
        "Loading job detail...",
        "Refreshing...",
        "Refreshing detail...",
        "Local service is healthy.",
        "Local service reported unhealthy.",
        "Version information is current.",
        "Retained jobs are current.",
        "Job detail is current.",
        "Stale.",
        "Local API unavailable. Is the localhost server still running?",
        "Local API returned HTTP ",
        "Local API returned an unexpected response.",
        "No representative jobs are available. WEB5 is read-only; jobs must be created outside this page.",
        "Job detail is no longer available. Refreshing list...",
        "Local API reconnected.",
        "Never",
    ):
        assert state_text in text
    # Timers, non-overlap, abort/pause/resume, cleanup, and one-shot version
    # loading are owned by frontend/tests/polling.test.ts.


def test_api_routes_keep_json_precedence_and_web2_contracts(shell_server):
    server, _ = shell_server
    status, headers, body = _request(server, "GET", "/api/health")
    assert status == 200
    assert headers["content-type"] == "application/json; charset=utf-8"
    assert json.loads(body) == {
        "ok": True,
        "service": "reconstruction_web",
        "mode": "local",
        "local_only": True,
        "state_configured": True,
    }
    status, headers, body = _request(server, "GET", "/api/version")
    assert status == 200
    assert headers["content-type"] == "application/json; charset=utf-8"
    assert json.loads(body) == {
        "service": "reconstruction_web",
        "version": __version__,
        "version_source": "reconstruction_web",
    }
    status, headers, body = _request(server, "GET", "/api/roots")
    assert status == 200
    assert headers["content-type"] == "application/json; charset=utf-8"
    assert json.loads(body) == {"roots": []}
    assert b"<!doctype html>" not in body.lower()


def test_job_list_and_detail_keep_web4_contracts(tmp_path):
    state_root = tmp_path / "web_state"
    state_root.mkdir()
    registry = JobRegistry(_step_wait=0)
    job_id = registry.create_representative(behavior="complete")["job_id"]
    deadline = time.monotonic() + 2
    while registry.get_snapshot(job_id)["state"] != "completed" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert registry.get_snapshot(job_id)["state"] == "completed"
    server = make_server(build_state_config(state_root), port=0, job_registry=registry)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, headers, body = _request(server, "GET", "/api/jobs")
        listing = json.loads(body)
        assert status == 200
        assert headers["content-type"] == "application/json; charset=utf-8"
        assert set(listing) == {"jobs"}
        assert set(listing["jobs"][0]) == SUMMARY_FIELDS
        assert listing["jobs"][0]["job_id"] == job_id
        status, headers, body = _request(server, "GET", f"/api/jobs/{job_id}")
        detail = json.loads(body)
        assert status == 200
        assert headers["content-type"] == "application/json; charset=utf-8"
        assert set(detail) == DETAIL_FIELDS
        assert detail["job_id"] == job_id
        assert detail["result"] == {"kind": "representative", "steps_completed": 20}
        assert b"<!doctype html>" not in body.lower()
    finally:
        shutdown_server(server, thread, join_timeout=3)


def test_shell_server_binding_and_clean_import_isolation(tmp_path):
    state_root = tmp_path / "web_state"
    state_root.mkdir()
    server = make_server(build_state_config(state_root), port=0)
    try:
        assert HOST == "127.0.0.1"
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()
        server.job_registry.shutdown(join_timeout=1)

    probe = """
import json
import sys
import reconstruction_web.shell_html
import reconstruction_web.server
print(json.dumps({
    "customtkinter": "customtkinter" in sys.modules,
    "desktop": "reconstruction_gui.reconstruction_zone" in sys.modules,
    "app_infra": "reconstruction_gui.app_infra" in sys.modules,
    "prep360": any(name == "prep360" or name.startswith("prep360.") for name in sys.modules),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(PACKAGE_ROOT.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {
        "customtkinter": False,
        "desktop": False,
        "app_infra": False,
        "prep360": False,
    }


def test_server_and_shell_sources_have_no_generic_static_serving():
    source = (PACKAGE_ROOT / "server.py").read_text(encoding="utf-8")
    shell_source, _, assignment = _shell_source_tree_and_assignment()
    before, _, after = _literal_span(shell_source, assignment.value)
    combined = source + before + after
    for forbidden in (
        "SimpleHTTPRequestHandler",
        "os.chdir",
        "static_root",
        "template_loader",
        "package_resources",
        "importlib.resources",
    ):
        assert forbidden not in combined
    assert 'if self.path == "/":' in source
    assert source.index('if route == "/api/health":') < source.index('if self.path == "/":')
    assert source.index('if route == "/api/jobs":') < source.index('if self.path == "/":')
