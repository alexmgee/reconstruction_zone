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
from reconstruction_web.shell import SHELL_HTML
from reconstruction_web.state import build_state_config


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "reconstruction_web"
SHELL_PATH = PACKAGE_ROOT / "shell.py"
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
    assert b'id="status-view"' in body
    assert b'id="jobs-view"' in body


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
    source = SHELL_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "SHELL_HTML"
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Constant)
    assert isinstance(value.value, (bytes, str))
    for forbidden in ("open(", "read_bytes", "read_text", "Path(", "importlib", "__file__"):
        assert forbidden not in source


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
    headers, body, parser = _served_shell(server)
    text = body.decode("utf-8")
    lowered = text.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert headers["content-security-policy"] == EXACT_CSP

    resource_attributes = {"src", "href", "action", "formaction", "poster", "data"}
    observed_references: list[tuple[str, str, str]] = []
    for tag, attrs in parser.tags:
        for name, value in attrs.items():
            if name.lower() in resource_attributes and value is not None:
                observed_references.append((tag, name.lower(), value))
    assert observed_references == [
        ("link", "href", "data:,"),
        ("a", "href", "#status"),
        ("a", "href", "#jobs"),
    ]
    assert not any(tag in {"img", "iframe", "object", "embed", "audio", "video"} for tag, _ in parser.tags)
    assert "@import" not in parser.style.lower()
    assert re.search(r"url\s*\(", parser.style, flags=re.IGNORECASE) is None
    assert "WebSocket" not in parser.script
    assert "EventSource" not in parser.script
    assert "sendBeacon" not in parser.script
    assert "XMLHttpRequest" not in parser.script


def test_frontend_fetches_are_same_origin_get_only(shell_server):
    server, _ = shell_server
    _, _, parser = _served_shell(server)
    script = parser.script
    assert script.count("fetch(") == 1
    assert 'method: "GET"' in script
    assert 'const HEALTH_PATH = "/api/health";' in script
    assert 'const VERSION_PATH = "/api/version";' in script
    assert 'const JOBS_PATH = "/api/jobs";' in script
    assert 'JOBS_PATH + "/" + encodeURIComponent(selectedJobId)' in script
    assert re.search(r'method\s*:\s*"(?:POST|PUT|PATCH|DELETE)"', script) is None
    assert "request body" not in script.lower()
    assert "/cancel" not in script
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "document.cookie" not in script
    assert "serviceWorker" not in script


def test_rendering_uses_closed_fields_and_text_nodes_only(shell_server):
    server, _ = shell_server
    _, _, parser = _served_shell(server)
    script = parser.script
    assert "document.createElement" in script
    assert "replaceChildren" in script
    assert "textContent" in script
    for forbidden in (
        "innerHTML",
        "insertAdjacentHTML",
        "document.write",
        "JSON.stringify",
        "Object.keys",
        "Object.entries",
    ):
        assert forbidden not in script
    assert re.search(r"\bfor\s*\([^)]*\bin\b", script) is None

    response_fields = set(re.findall(r"\bvalue\.([A-Za-z_][A-Za-z0-9_]*)", script))
    assert response_fields == DETAIL_FIELDS | {
        "jobs",
        "ok",
        "service",
        "mode",
        "local_only",
        "state_configured",
        "version",
        "version_source",
        "time",
        "level",
        "code",
        "message",
        "kind",
        "steps_completed",
    }
    assert "encodeURIComponent(selectedJobId)" in script


def test_polling_lifecycle_and_visible_state_catalog_are_present(shell_server):
    server, _ = shell_server
    _, body, parser = _served_shell(server)
    script = parser.script
    text = body.decode("utf-8")
    assert "const HEALTH_INTERVAL_MS = 5000;" in script
    assert "const JOB_INTERVAL_MS = 2000;" in script
    assert "const FETCH_TIMEOUT_MS = 5000;" in script
    assert "setInterval" not in script
    assert "healthTimer = window.setTimeout(runHealthCycle, delay);" in script
    assert "jobTimer = window.setTimeout(runJobsCycle, delay);" in script
    assert "AbortController" in script
    assert 'document.addEventListener("visibilitychange"' in script
    assert 'window.addEventListener("pagehide"' in script
    for state_text in (
        "Loading local status...",
        "Loading jobs...",
        "Refreshing...",
        "Stale.",
        "Local API unavailable. Is the localhost server still running?",
        "Local API returned HTTP ",
        "Local API returned an unexpected response.",
        "No representative jobs are available.",
        "Local API reconnected.",
    ):
        assert state_text in text


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
import reconstruction_web.shell
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
    shell_source = SHELL_PATH.read_text(encoding="utf-8")
    combined = source + shell_source
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
