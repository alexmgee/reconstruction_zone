"""WEB2 local server skeleton tests."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from reconstruction_web import __version__
from reconstruction_web.server import HOST, make_server, main
from reconstruction_web.state import WebStateConfigError, build_state_config, resolve_state_root


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "reconstruction_web"

FORBIDDEN_SOURCE_SUBSTRINGS = (
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
)

STATE_PY_ONLY_ALLOWED_LITERALS = (
    ".prep360",
    ".reconstruction_zone",
)


@pytest.fixture
def allowed_state_root(tmp_path):
    state_root = tmp_path / "web_state"
    state_root.mkdir()
    return state_root


def _start_server(state_root: Path, port: int = 0):
    config = build_state_config(state_root)
    server = make_server(config, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _fetch_json(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)


def test_build_state_config_none_raises():
    with pytest.raises(WebStateConfigError, match="Missing explicit non-production state root"):
        build_state_config(None)


def test_build_state_config_missing_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(WebStateConfigError, match="State root does not exist"):
        build_state_config(missing)


def test_build_state_config_does_not_create_directory(tmp_path):
    missing = tmp_path / "should_not_be_created"
    assert not missing.exists()
    with pytest.raises(WebStateConfigError):
        build_state_config(missing)
    assert not missing.exists()


def test_build_state_config_existing_directory_succeeds(allowed_state_root):
    config = build_state_config(allowed_state_root)
    assert config.state_root == allowed_state_root.resolve()


def test_main_without_state_root_exits_nonzero(capsys):
    code = main([])
    captured = capsys.readouterr()
    assert code == 1
    assert "Missing explicit non-production state root." in captured.err


@pytest.mark.parametrize(
    "unsafe_path",
    [
        pytest.param("D:\\", id="drive-root-backslash"),
        pytest.param("D:/", id="drive-root-slash"),
        pytest.param("D:\\Projects", id="projects-root"),
        pytest.param("D:\\Projects\\reconstruction-zone", id="desktop-workspace"),
        pytest.param("D:\\Projects\\reconstruction-zone-web", id="web-worktree"),
        pytest.param("D:\\Capture", id="capture-root"),
    ],
)
def test_unsafe_state_roots_are_rejected(unsafe_path):
    path = Path(unsafe_path)
    if not path.exists() and unsafe_path not in {"D:\\", "D:/"}:
        pytest.skip(f"{unsafe_path} does not exist on this machine")
    with pytest.raises(WebStateConfigError, match="not allowed|too broad"):
        resolve_state_root(path)


@pytest.mark.parametrize(
    "descendant_path",
    [
        pytest.param("D:\\Projects\\reconstruction-zone\\tests", id="desktop-workspace-descendant"),
        pytest.param("D:\\Projects\\reconstruction-zone-web\\reconstruction_web", id="web-workspace-descendant"),
    ],
)
def test_source_workspace_descendants_are_rejected(descendant_path):
    path = Path(descendant_path)
    if not path.exists():
        pytest.skip(f"{descendant_path} does not exist on this machine")
    with pytest.raises(WebStateConfigError, match="not allowed"):
        resolve_state_root(path)


def test_fake_source_workspace_descendants_are_rejected(tmp_path, monkeypatch):
    main_anchor = tmp_path / "main-workspace"
    web_anchor = tmp_path / "web-workspace"
    main_descendant = main_anchor / "nested" / "web_state"
    web_descendant = web_anchor / "nested" / "web_state"
    main_descendant.mkdir(parents=True)
    web_descendant.mkdir(parents=True)

    monkeypatch.setattr(
        "reconstruction_web.state._forbidden_workspace_anchors",
        lambda: (main_anchor.resolve(), web_anchor.resolve()),
    )

    with pytest.raises(WebStateConfigError, match="not allowed"):
        resolve_state_root(main_descendant)
    with pytest.raises(WebStateConfigError, match="not allowed"):
        resolve_state_root(web_descendant)


def test_package_workspace_root_rejected_on_any_checkout_path(monkeypatch):
    """The derived package anchor must reject the source tree even when the checkout
    lives outside the known developer workspace paths (e.g. a CI runner)."""
    import reconstruction_web

    monkeypatch.setattr("reconstruction_web.state._owner_workspace_anchors", lambda: ())
    package_dir = Path(reconstruction_web.__file__).resolve().parent
    with pytest.raises(WebStateConfigError, match="not allowed"):
        resolve_state_root(package_dir)


def test_capture_copy_style_path_allowed_when_present(tmp_path):
    capture_root = tmp_path / "Capture"
    copy_path = capture_root / "_rz_web_test_copies" / "web_state"
    copy_path.mkdir(parents=True)

    config = build_state_config(copy_path)
    assert config.state_root == copy_path.resolve()


def test_home_directory_is_rejected():
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not home:
        pytest.skip("No home directory env var available")
    home_path = Path(home)
    if not home_path.exists():
        pytest.skip("Home directory does not exist")
    with pytest.raises(WebStateConfigError, match="not allowed"):
        resolve_state_root(home_path)


def test_forbidden_home_children_are_rejected():
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not home:
        pytest.skip("No home directory env var available")
    dot = "."
    candidates = (
        Path(home) / f"{dot}reconstruction_zone",
        Path(home) / f"{dot}prep360_models",
        Path(home) / f"{dot}prep360_thumb_cache",
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        with pytest.raises(WebStateConfigError, match="not allowed"):
            resolve_state_root(candidate)


def test_drive_relative_path_is_rejected():
    if os.name != "nt":
        pytest.skip("Drive-relative paths are Windows-specific")
    with pytest.raises(WebStateConfigError, match="Drive-relative paths are not allowed"):
        resolve_state_root("D:foo")


def test_path_home_and_expanduser_monkeypatch_respected(allowed_state_root, monkeypatch):
    sentinel = allowed_state_root / "sentinel_home"
    sentinel.mkdir()

    monkeypatch.setenv("USERPROFILE", os.fspath(sentinel))
    monkeypatch.setenv("HOME", os.fspath(sentinel))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: sentinel))
    monkeypatch.setattr(os.path, "expanduser", lambda value: os.fspath(sentinel / value.lstrip("~/")))

    with pytest.raises(WebStateConfigError, match="not allowed"):
        resolve_state_root(sentinel)


def _run_import_probe(probe: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    return json.loads(completed.stdout.strip())


def test_import_reconstruction_web_does_not_import_desktop_modules():
    probe = '''
import json
import sys

import reconstruction_web  # noqa: F401

print(json.dumps({
    "customtkinter": "customtkinter" in sys.modules,
    "reconstruction_zone": "reconstruction_gui.reconstruction_zone" in sys.modules,
    "app_infra": "reconstruction_gui.app_infra" in sys.modules,
}))
'''
    result = _run_import_probe(probe)
    assert result["customtkinter"] is False
    assert result["reconstruction_zone"] is False
    assert result["app_infra"] is False


def test_import_reconstruction_web_does_not_import_prep360():
    probe = '''
import json
import sys

import reconstruction_web  # noqa: F401

print(json.dumps({"prep360": "prep360" in sys.modules}))
'''
    result = _run_import_probe(probe)
    assert result["prep360"] is False


def test_import_jobs_does_not_import_desktop_or_workflow_modules():
    probe = '''
import json
import sys

import reconstruction_web.jobs  # noqa: F401

print(json.dumps({
    "customtkinter": "customtkinter" in sys.modules,
    "reconstruction_zone": "reconstruction_gui.reconstruction_zone" in sys.modules,
    "app_infra": "reconstruction_gui.app_infra" in sys.modules,
    "prep360": any(name == "prep360" or name.startswith("prep360.") for name in sys.modules),
}))
'''
    result = _run_import_probe(probe)
    assert result == {
        "customtkinter": False,
        "reconstruction_zone": False,
        "app_infra": False,
        "prep360": False,
    }


def test_desktop_startup_path_does_not_import_reconstruction_web():
    probe = '''
import json
import sys

import reconstruction_gui.reconstruction_zone  # noqa: F401

print(json.dumps({
    "web": any(name == "reconstruction_web" or name.startswith("reconstruction_web.") for name in sys.modules),
}))
'''
    result = _run_import_probe(probe)
    assert result["web"] is False


def test_version_route_does_not_import_desktop_modules_subprocess():
    runner = '''
import json
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

from reconstruction_web.server import make_server
from reconstruction_web.state import build_state_config

state_root = Path(tempfile.mkdtemp())
config = build_state_config(state_root)
httpd = make_server(config, port=0)
thread = threading.Thread(target=httpd.serve_forever, daemon=True)
thread.start()
host, port = httpd.server_address
try:
    with urllib.request.urlopen(f"http://{host}:{port}/api/version", timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    modules = sorted(sys.modules)
    print(json.dumps({"payload": payload, "modules": modules}))
finally:
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)
'''
    completed = subprocess.run(
        [sys.executable, "-c", runner],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    result = json.loads(completed.stdout.strip())
    modules = result["modules"]
    assert "customtkinter" not in modules
    assert "reconstruction_gui.reconstruction_zone" not in modules
    assert result["payload"]["version_source"] == "reconstruction_web"


def test_source_guard_forbidden_literals():
    state_path = PACKAGE_ROOT / "state.py"
    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_SOURCE_SUBSTRINGS:
            assert forbidden not in text, f"{forbidden!r} found in {path}"
        if path == state_path:
            continue
        for forbidden in STATE_PY_ONLY_ALLOWED_LITERALS:
            assert forbidden not in text, f"{forbidden!r} found in {path}"


def test_source_guard_state_py_allowlist_is_limited():
    state_text = (PACKAGE_ROOT / "state.py").read_text(encoding="utf-8")
    for literal in STATE_PY_ONLY_ALLOWED_LITERALS:
        assert literal in state_text, f"{literal!r} should appear in state.py for rejection rules"

    other_modules = [path for path in PACKAGE_ROOT.rglob("*.py") if path.name != "state.py"]
    assert other_modules, "expected reconstruction_web modules besides state.py"
    for path in other_modules:
        text = path.read_text(encoding="utf-8")
        for literal in STATE_PY_ONLY_ALLOWED_LITERALS:
            assert literal not in text, f"{literal!r} must stay out of {path}"


def test_no_forbidden_boundary_imports_in_source():
    project_reader_path = PACKAGE_ROOT / "projects.py"
    forbidden_modules = (
        "reconstruction_gui.project_store",
        "reconstruction_gui.activity_store",
        "reconstruction_gui.review_status",
        "reconstruction_gui.thumb_cache",
        "reconstruction_gui.review_gui",
        "prep360",
        "prep360.core",
    )
    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for module in forbidden_modules:
            if path == project_reader_path and module == "reconstruction_gui.project_store":
                continue
            assert f"import {module}" not in text, f"{module!r} import found in {path}"
            assert f"from {module}" not in text, f"{module!r} import found in {path}"


def test_no_forbidden_web4_runtime_or_serving_technology_in_source():
    forbidden = (
        "fastapi",
        "flask",
        "aiohttp",
        "uvicorn",
        "asyncio",
        "multiprocessing",
        "subprocess",
        "concurrent.futures",
        "SimpleHTTPRequestHandler",
    )
    for path in PACKAGE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        for value in forbidden:
            assert value.lower() not in lowered, f"{value!r} found in {path}"


def test_sentinel_home_not_modified_during_server_run(allowed_state_root, monkeypatch):
    sentinel = allowed_state_root / "sentinel_home"
    sentinel.mkdir()
    before = list(sentinel.iterdir())

    monkeypatch.setenv("USERPROFILE", os.fspath(sentinel))
    monkeypatch.setenv("HOME", os.fspath(sentinel))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: sentinel))
    monkeypatch.setattr(os.path, "expanduser", lambda value: os.fspath(sentinel / value.lstrip("~/")))

    server, thread = _start_server(allowed_state_root, port=0)
    host, port = server.server_address
    try:
        status, payload = _fetch_json(f"http://{host}:{port}/api/health")
        assert status == 200
        assert payload["ok"] is True
        status, version_payload = _fetch_json(f"http://{host}:{port}/api/version")
        assert status == 200
        assert version_payload["version"] == __version__
    finally:
        _stop_server(server, thread)

    after = list(sentinel.iterdir())
    assert before == after


def test_health_endpoint(allowed_state_root):
    server, thread = _start_server(allowed_state_root, port=0)
    host, port = server.server_address
    try:
        status, payload = _fetch_json(f"http://{host}:{port}/api/health")
        assert status == 200
        assert payload == {
            "local_only": True,
            "mode": "local",
            "ok": True,
            "service": "reconstruction_web",
            "state_configured": True,
        }
    finally:
        _stop_server(server, thread)


def test_version_endpoint(allowed_state_root):
    server, thread = _start_server(allowed_state_root, port=0)
    host, port = server.server_address
    try:
        status, payload = _fetch_json(f"http://{host}:{port}/api/version")
        assert status == 200
        assert payload == {
            "service": "reconstruction_web",
            "version": __version__,
            "version_source": "reconstruction_web",
        }
    finally:
        _stop_server(server, thread)


def test_unknown_route_returns_404(allowed_state_root):
    server, thread = _start_server(allowed_state_root, port=0)
    host, port = server.server_address
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _fetch_json(f"http://{host}:{port}/api/unknown")
        assert exc_info.value.code == 404
    finally:
        _stop_server(server, thread)


def test_api_routes_use_json_content_type(allowed_state_root):
    server, thread = _start_server(allowed_state_root, port=0)
    host, port = server.server_address
    try:
        request = urllib.request.Request(f"http://{host}:{port}/api/health")
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.headers.get_content_type() == "application/json"
    finally:
        _stop_server(server, thread)


def test_server_binds_localhost_only(allowed_state_root):
    server, thread = _start_server(allowed_state_root, port=0)
    host, port = server.server_address
    try:
        assert host == HOST == "127.0.0.1"
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            assert sock.connect_ex(("127.0.0.1", port)) == 0
        finally:
            sock.close()
    finally:
        _stop_server(server, thread)


def test_port_zero_assigns_available_port(allowed_state_root):
    server, thread = _start_server(allowed_state_root, port=0)
    try:
        _, port = server.server_address
        assert isinstance(port, int)
        assert port > 0
    finally:
        _stop_server(server, thread)


def test_server_shutdown_releases_port(allowed_state_root):
    server, thread = _start_server(allowed_state_root, port=0)
    _, port = server.server_address
    _stop_server(server, thread)

    rebound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    rebound.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        rebound.bind((HOST, port))
    finally:
        rebound.close()


def test_configurable_port_works(allowed_state_root):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        chosen_port = sock.getsockname()[1]

    server, thread = _start_server(allowed_state_root, port=chosen_port)
    try:
        assert server.server_address[1] == chosen_port
    finally:
        _stop_server(server, thread)


def test_no_production_state_files_written_during_tests(allowed_state_root):
    marker = allowed_state_root / "only_marker.txt"
    marker.write_text("ok", encoding="utf-8")
    server, thread = _start_server(allowed_state_root, port=0)
    host, port = server.server_address
    try:
        _fetch_json(f"http://{host}:{port}/api/health")
        _fetch_json(f"http://{host}:{port}/api/version")
    finally:
        _stop_server(server, thread)

    assert marker.read_text(encoding="utf-8") == "ok"
    assert list(allowed_state_root.iterdir()) == [marker]
