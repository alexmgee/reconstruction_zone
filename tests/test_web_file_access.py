"""WEB3 folder-token file access tests."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from reconstruction_web.file_access import (
    FileAccessError,
    FolderTokenRegistry,
    is_same_or_descendant,
    response_has_path_leak,
)
from reconstruction_web.server import HOST, build_root_registry, make_server, parse_root_argument
from reconstruction_web.state import WebStateConfigError, build_state_config, resolve_state_root

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "reconstruction_web"

FORBIDDEN_SOURCE_SUBSTRINGS = (
    "Path.home(",
    "expanduser",
    ".studio_prefs",
    "SimpleHTTPRequestHandler",
    "D:\\",
    "D:/",
    "C:\\",
    "C:/",
)

STATE_PY_ONLY_ALLOWED_LITERALS = (
    "tracker.json",
    "activity_log.json",
    ".prep360",
    ".reconstruction_zone",
)


@pytest.fixture
def file_root(tmp_path):
    root = tmp_path / "content_root"
    root.mkdir()
    images = root / "images"
    images.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"\xff\xd8\xff" + b"x" * 100)
    (images / "nested").mkdir()
    (images / "nested" / "deep.txt").write_text("deep", encoding="utf-8")
    (root / "readme.txt").write_text("hello", encoding="utf-8")
    (root / "masks").mkdir()
    return root


@pytest.fixture
def allowed_state_root(tmp_path):
    state_root = tmp_path / "web_state"
    state_root.mkdir()
    return state_root


def _registry_with_root(file_root: Path, label: str = "qa") -> tuple[FolderTokenRegistry, str]:
    registry = FolderTokenRegistry()
    registered = registry.register_root(file_root, label=label)
    return registry, registered.token


def _start_server(state_root: Path, registry: FolderTokenRegistry | None = None, port: int = 0):
    config = build_state_config(state_root)
    server = make_server(config, port=port, root_registry=registry)
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


def _fetch_json_allow_error(url: str) -> tuple[int, dict, str]:
    try:
        status, payload = _fetch_json(url)
        return status, payload, json.dumps(payload, sort_keys=True)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, json.loads(body), body


def _assert_no_leaks(body_text: str, *, root_path: Path | None = None) -> None:
    assert not response_has_path_leak(body_text, root_path=root_path)
    assert "tracker.json" not in body_text


# --- Registry and token tests ---


def test_register_existing_directory_succeeds(file_root):
    registry, token = _registry_with_root(file_root)
    assert token
    assert token != os.fspath(file_root)
    assert token != "qa"


def test_token_is_opaque_not_path_or_label(file_root):
    registry = FolderTokenRegistry()
    registered = registry.register_root(file_root, label="qa")
    assert registered.token != os.fspath(file_root.resolve())
    assert registered.token != "qa"
    assert "/" not in registered.token
    assert "\\" not in registered.token


def test_api_roots_has_no_absolute_path(file_root, allowed_state_root):
    registry, token = _registry_with_root(file_root)
    server, thread = _start_server(allowed_state_root, registry=registry)
    host, port = server.server_address
    try:
        status, payload, body = _fetch_json_allow_error(f"http://{host}:{port}/api/roots")
        assert status == 200
        assert payload["roots"] == [{"label": "qa", "read_only": True, "token": token}]
        _assert_no_leaks(body, root_path=file_root)
    finally:
        _stop_server(server, thread)


def test_multiple_roots_get_distinct_tokens(file_root, tmp_path):
    second = tmp_path / "second_root"
    second.mkdir()
    registry = FolderTokenRegistry()
    first = registry.register_root(file_root, label="alpha")
    second_reg = registry.register_root(second, label="beta")
    assert first.token != second_reg.token


def test_duplicate_labels_rejected(file_root):
    registry = FolderTokenRegistry()
    registry.register_root(file_root, label="qa")
    with pytest.raises(FileAccessError, match="already registered") as exc_info:
        registry.register_root(file_root, label="qa")
    assert exc_info.value.code == "duplicate_label"


@pytest.mark.parametrize(
    "label",
    [
        "",
        "bad/label",
        "bad\\label",
        "bad:label",
        "bad\x00label",
        "label*",
    ],
)
def test_invalid_labels_rejected(file_root, label):
    registry = FolderTokenRegistry()
    with pytest.raises(FileAccessError, match="not allowed"):
        registry.register_root(file_root, label=label)


def test_unknown_token_rejected(file_root):
    registry, _ = _registry_with_root(file_root)
    with pytest.raises(FileAccessError, match="not recognized") as exc_info:
        registry.resolve("unknown-token", "")
    assert exc_info.value.code == "unknown_root"
    assert exc_info.value.status == 404


# --- Root validation tests ---


def test_missing_root_refused(tmp_path):
    missing = tmp_path / "missing"
    registry = FolderTokenRegistry()
    with pytest.raises(FileAccessError, match="not allowed"):
        registry.register_root(missing, label="qa")


def test_file_path_root_refused(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    registry = FolderTokenRegistry()
    with pytest.raises(FileAccessError, match="not allowed"):
        registry.register_root(file_path, label="qa")


@pytest.mark.parametrize(
    "unsafe_path",
    [
        pytest.param("D:\\", id="drive-root"),
        pytest.param("D:\\Projects", id="projects-root"),
        pytest.param("D:\\Projects\\reconstruction-zone", id="desktop-workspace"),
        pytest.param("D:\\Projects\\reconstruction-zone-web", id="web-worktree"),
        pytest.param("D:\\Capture", id="capture-root"),
    ],
)
def test_broad_roots_refused(unsafe_path):
    path = Path(unsafe_path)
    if not path.exists() and unsafe_path not in {"D:\\", "D:/"}:
        pytest.skip(f"{unsafe_path} does not exist")
    registry = FolderTokenRegistry()
    with pytest.raises(FileAccessError, match="not allowed"):
        registry.register_root(path, label="qa")


@pytest.mark.parametrize(
    "descendant_path",
    [
        pytest.param("D:\\Projects\\reconstruction-zone\\tests", id="desktop-descendant"),
        pytest.param("D:\\Projects\\reconstruction-zone-web\\reconstruction_web", id="web-descendant"),
    ],
)
def test_workspace_descendants_refused(descendant_path):
    path = Path(descendant_path)
    if not path.exists():
        pytest.skip(f"{descendant_path} does not exist")
    registry = FolderTokenRegistry()
    with pytest.raises(FileAccessError, match="not allowed"):
        registry.register_root(path, label="qa")


def test_home_root_refused():
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not home or not Path(home).exists():
        pytest.skip("Home directory unavailable")
    registry = FolderTokenRegistry()
    with pytest.raises(FileAccessError, match="not allowed"):
        registry.register_root(Path(home), label="qa")


def test_home_children_refused():
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not home:
        pytest.skip("Home directory unavailable")
    registry = FolderTokenRegistry()
    for child in (".reconstruction_zone", ".prep360_models", ".prep360_thumb_cache"):
        candidate = Path(home) / child
        if not candidate.exists():
            continue
        with pytest.raises(FileAccessError, match="not allowed"):
            registry.register_root(candidate, label="qa")


def test_content_root_rejects_home_child_descendants(tmp_path, monkeypatch):
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    nested_roots = (
        fake_home / ".reconstruction_zone" / "nested_root",
        fake_home / ".prep360_models" / "nested_root",
        fake_home / ".prep360_thumb_cache" / "nested_root",
    )
    for nested in nested_roots:
        nested.mkdir(parents=True)

    monkeypatch.setenv("USERPROFILE", os.fspath(fake_home))
    monkeypatch.setenv("HOME", os.fspath(fake_home))

    registry = FolderTokenRegistry()
    for nested in nested_roots:
        with pytest.raises(FileAccessError, match="not allowed"):
            registry.register_root(nested, label="qa")


def test_capture_copy_path_allowed(tmp_path):
    copy_path = tmp_path / "Capture" / "_rz_web_test_copies" / "some_project"
    copy_path.mkdir(parents=True)
    registry = FolderTokenRegistry()
    registered = registry.register_root(copy_path, label="copy")
    assert registered.root == copy_path.resolve()


def test_drive_relative_root_refused():
    if os.name != "nt":
        pytest.skip("Windows-specific")
    registry = FolderTokenRegistry()
    with pytest.raises(FileAccessError, match="not allowed"):
        registry.register_root("D:foo", label="qa")


def test_unc_root_refused():
    if os.name != "nt":
        pytest.skip("Windows-specific")
    unc = Path("\\\\server\\share")
    registry = FolderTokenRegistry()
    with pytest.raises(FileAccessError, match="not allowed"):
        registry.register_root(unc, label="qa")


# --- Relative path resolver tests ---


def test_empty_and_dot_relative_paths_resolve_to_root(file_root):
    registry, token = _registry_with_root(file_root)
    assert registry.resolve(token, "") == file_root.resolve()
    assert registry.resolve(token, ".") == file_root.resolve()


def test_nested_relative_path_resolves(file_root):
    registry, token = _registry_with_root(file_root)
    resolved = registry.resolve(token, "images/nested/deep.txt")
    assert resolved.name == "deep.txt"
    assert is_same_or_descendant(resolved, file_root)


@pytest.mark.parametrize(
  "bad_path",
  [
      pytest.param("D:\\tracker.json", id="windows-absolute"),
      pytest.param("C:/Users/alexm/file.txt", id="posix-style-absolute"),
      pytest.param("\\\\server\\share\\file.txt", id="unc"),
      pytest.param("/tmp/file.txt", id="leading-posix"),
      pytest.param("\\windows\\file.txt", id="leading-backslash"),
      pytest.param("file.jpg:stream", id="ads-colon"),
      pytest.param("\\\\?\\C:\\secret", id="extended-prefix"),
      pytest.param("\\\\.\\COM1", id="device-prefix"),
      pytest.param("CON", id="reserved-con"),
      pytest.param("images\\CON.txt", id="reserved-component"),
      pytest.param("D:foo", id="drive-relative"),
      pytest.param("..", id="dot-dot"),
      pytest.param("../tracker.json", id="parent-traversal"),
      pytest.param("images/../../tracker.json", id="nested-traversal"),
      pytest.param("images\\..\\..\\tracker.json", id="mixed-traversal"),
      pytest.param("file://host/path", id="file-url"),
      pytest.param("bad\x00path", id="nul-byte"),
      pytest.param("bad\x01path", id="control-char"),
      pytest.param("~", id="tilde-only"),
      pytest.param("~/anything", id="tilde-slash"),
      pytest.param("~\\.prep360_queue.json", id="tilde-backslash-queue"),
      pytest.param("~/.prep360_queue.json", id="tilde-posix-queue"),
      pytest.param("images//frame_0001.jpg", id="double-slash"),
      pytest.param("images\\\\frame_0001.jpg", id="double-backslash"),
      pytest.param("images/\\frame_0001.jpg", id="mixed-double-separator"),
      pytest.param("images\\//frame_0001.jpg", id="mixed-triple-separator"),
  ],
)
def test_unsafe_relative_paths_refused(file_root, bad_path):
    registry, token = _registry_with_root(file_root)
    with pytest.raises(FileAccessError, match="not allowed|outside|not found"):
        registry.resolve(token, bad_path)


def test_missing_path_not_found_without_creation(file_root):
    registry, token = _registry_with_root(file_root)
    with pytest.raises(FileAccessError, match="not found") as exc_info:
        registry.resolve(token, "does_not_exist.txt")
    assert exc_info.value.code == "path_not_found"
    assert not (file_root / "does_not_exist.txt").exists()


def test_symlink_escape_refused(file_root):
    outside = file_root.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = file_root / "escape_link"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation not permitted: {exc}")
    registry, token = _registry_with_root(file_root)
    with pytest.raises(FileAccessError, match="outside|not found"):
        registry.resolve(token, "escape_link")


def test_inside_root_symlink_allowed(file_root):
  try:
      target = file_root / "images" / "frame_0001.jpg"
      link = file_root / "inside_link"
      link.symlink_to(target)
  except OSError as exc:
      pytest.skip(f"symlink creation not permitted: {exc}")
  registry, token = _registry_with_root(file_root)
  resolved = registry.resolve(token, "inside_link")
  assert resolved.name == "frame_0001.jpg"
  assert is_same_or_descendant(resolved, file_root)


def test_is_same_or_descendant_direct(file_root, tmp_path):
    child = file_root / "images"
    outside = tmp_path / "outside"
    outside.mkdir()
    assert is_same_or_descendant(file_root, file_root)
    assert is_same_or_descendant(child, file_root)
    assert not is_same_or_descendant(outside, file_root)


# --- Endpoint tests ---


def test_api_roots_empty_without_registry(allowed_state_root):
    server, thread = _start_server(allowed_state_root)
    host, port = server.server_address
    try:
        status, payload, _ = _fetch_json_allow_error(f"http://{host}:{port}/api/roots")
        assert status == 200
        assert payload == {"roots": []}
    finally:
        _stop_server(server, thread)


def test_files_list_top_level_deterministic(file_root, allowed_state_root):
    registry, token = _registry_with_root(file_root)
    server, thread = _start_server(allowed_state_root, registry=registry)
    host, port = server.server_address
    try:
        url = f"http://{host}:{port}/api/files/list?{urllib.parse.urlencode({'root': token, 'path': '.'})}"
        status, payload, body = _fetch_json_allow_error(url)
        assert status == 200
        names = [entry["name"] for entry in payload["entries"]]
        assert names == sorted(names, key=lambda name: (0 if name in {"images", "masks"} else 1, name.casefold(), name))
        kinds = {entry["name"]: entry["kind"] for entry in payload["entries"]}
        assert kinds["images"] == "directory"
        assert kinds["masks"] == "directory"
        assert kinds["readme.txt"] == "file"
        _assert_no_leaks(body, root_path=file_root)
    finally:
        _stop_server(server, thread)


def test_files_list_nested_directory(file_root, allowed_state_root):
    registry, token = _registry_with_root(file_root)
    server, thread = _start_server(allowed_state_root, registry=registry)
    host, port = server.server_address
    try:
        url = f"http://{host}:{port}/api/files/list?{urllib.parse.urlencode({'root': token, 'path': 'images'})}"
        status, payload, _ = _fetch_json_allow_error(url)
        assert status == 200
        names = {entry["name"] for entry in payload["entries"]}
        assert "frame_0001.jpg" in names
        assert "nested" in names
    finally:
        _stop_server(server, thread)


def test_files_list_refuses_file_target(file_root, allowed_state_root):
    registry, token = _registry_with_root(file_root)
    server, thread = _start_server(allowed_state_root, registry=registry)
    host, port = server.server_address
    try:
        url = f"http://{host}:{port}/api/files/list?{urllib.parse.urlencode({'root': token, 'path': 'readme.txt'})}"
        status, payload, body = _fetch_json_allow_error(url)
        assert status == 400
        assert payload["error"] == "not_a_directory"
        _assert_no_leaks(body, root_path=file_root)
    finally:
        _stop_server(server, thread)


def test_files_stat_file_metadata(file_root, allowed_state_root):
    registry, token = _registry_with_root(file_root)
    server, thread = _start_server(allowed_state_root, registry=registry)
    host, port = server.server_address
    try:
        url = f"http://{host}:{port}/api/files/stat?{urllib.parse.urlencode({'root': token, 'path': 'readme.txt'})}"
        status, payload, body = _fetch_json_allow_error(url)
        assert status == 200
        entry = payload["entry"]
        assert entry["kind"] == "file"
        assert entry["name"] == "readme.txt"
        assert entry["size_bytes"] == len("hello")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", entry["modified_time_utc"])
        _assert_no_leaks(body, root_path=file_root)
    finally:
        _stop_server(server, thread)


def test_files_stat_directory_metadata(file_root, allowed_state_root):
    registry, token = _registry_with_root(file_root)
    server, thread = _start_server(allowed_state_root, registry=registry)
    host, port = server.server_address
    try:
        url = f"http://{host}:{port}/api/files/list?{urllib.parse.urlencode({'root': token, 'path': 'images'})}"
        status, payload, _ = _fetch_json_allow_error(url)
        assert status == 200
        url = f"http://{host}:{port}/api/files/stat?{urllib.parse.urlencode({'root': token, 'path': 'images'})}"
        status, payload, body = _fetch_json_allow_error(url)
        assert status == 200
        assert payload["entry"]["kind"] == "directory"
        assert payload["entry"]["name"] == "images"
        _assert_no_leaks(body, root_path=file_root)
    finally:
        _stop_server(server, thread)


def test_endpoints_reject_missing_root(file_root, allowed_state_root):
    server, thread = _start_server(allowed_state_root, registry=_registry_with_root(file_root)[0])
    host, port = server.server_address
    try:
        for route in ("/api/files/list", "/api/files/stat"):
            url = f"http://{host}:{port}{route}?path=."
            status, payload, body = _fetch_json_allow_error(url)
            assert status == 400
            assert payload["error"] == "missing_root"
            _assert_no_leaks(body, root_path=file_root)
    finally:
        _stop_server(server, thread)


def test_endpoints_reject_unknown_token(allowed_state_root):
    server, thread = _start_server(allowed_state_root)
    host, port = server.server_address
    try:
        url = f"http://{host}:{port}/api/files/list?{urllib.parse.urlencode({'root': 'bad-token', 'path': '.'})}"
        status, payload, body = _fetch_json_allow_error(url)
        assert status == 404
        assert payload["error"] == "unknown_root"
        _assert_no_leaks(body)
    finally:
        _stop_server(server, thread)


def test_endpoint_errors_do_not_leak_paths(file_root, allowed_state_root):
    registry, token = _registry_with_root(file_root)
    server, thread = _start_server(allowed_state_root, registry=registry)
    host, port = server.server_address
    cases = [
        f"/api/files/list?{urllib.parse.urlencode({'root': token, 'path': '../outside'})}",
        f"/api/files/stat?{urllib.parse.urlencode({'root': token, 'path': 'missing.txt'})}",
        f"/api/files/list?{urllib.parse.urlencode({'root': 'missing', 'path': '.'})}",
    ]
    try:
        for route in cases:
            _, _, body = _fetch_json_allow_error(f"http://{host}:{port}{route}")
            _assert_no_leaks(body, root_path=file_root)
            assert "FileNotFoundError" not in body
            assert "PermissionError" not in body
    finally:
        _stop_server(server, thread)


def test_no_file_content_read_in_responses(file_root, allowed_state_root):
    registry, token = _registry_with_root(file_root)
    server, thread = _start_server(allowed_state_root, registry=registry)
    host, port = server.server_address
    try:
        url = f"http://{host}:{port}/api/files/stat?{urllib.parse.urlencode({'root': token, 'path': 'readme.txt'})}"
        _, payload, body = _fetch_json_allow_error(url)
        assert "hello" not in body
        assert "content" not in payload
    finally:
        _stop_server(server, thread)


def test_unknown_route_still_404(allowed_state_root):
    server, thread = _start_server(allowed_state_root)
    host, port = server.server_address
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _fetch_json(f"http://{host}:{port}/api/unknown")
        assert exc_info.value.code == 404
    finally:
        _stop_server(server, thread)


def test_parse_root_argument_accepts_label_equals_path():
    assert parse_root_argument("qa=C:\\temp\\root") == ("qa", "C:\\temp\\root")


def test_build_root_registry_from_cli_specs(file_root):
    registry = build_root_registry([f"qa={file_root}"])
    assert len(registry.roots_for_api()) == 1


def test_startup_rejects_invalid_root_before_bind(allowed_state_root, tmp_path):
    missing = tmp_path / "missing_root"
    with pytest.raises(WebStateConfigError):
        build_root_registry([f"qa={missing}"])


def test_web2_health_and_version_still_work(file_root, allowed_state_root):
    registry, _ = _registry_with_root(file_root)
    server, thread = _start_server(allowed_state_root, registry=registry)
    host, port = server.server_address
    try:
        status, health = _fetch_json(f"http://{host}:{port}/api/health")
        assert status == 200
        assert health["ok"] is True
        status, version = _fetch_json(f"http://{host}:{port}/api/version")
        assert status == 200
        assert version["version_source"] == "reconstruction_web"
    finally:
        _stop_server(server, thread)


# --- Source guards ---


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


# --- Production-state no-write proof ---


def test_no_production_state_modified_during_file_access(allowed_state_root, file_root, monkeypatch):
    sentinel = allowed_state_root / "sentinel_home"
    sentinel.mkdir()
    before = list(sentinel.iterdir())

    monkeypatch.setenv("USERPROFILE", os.fspath(sentinel))
    monkeypatch.setenv("HOME", os.fspath(sentinel))

    marker = file_root / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")
    before_root = {p.name for p in file_root.iterdir()}

    registry, token = _registry_with_root(file_root)
    server, thread = _start_server(allowed_state_root, registry=registry)
    host, port = server.server_address
    try:
        _fetch_json(f"http://{host}:{port}/api/roots")
        _fetch_json(
            f"http://{host}:{port}/api/files/list?{urllib.parse.urlencode({'root': token, 'path': '.'})}"
        )
        _fetch_json(
            f"http://{host}:{port}/api/files/stat?{urllib.parse.urlencode({'root': token, 'path': 'readme.txt'})}"
        )
    finally:
        _stop_server(server, thread)

    assert list(sentinel.iterdir()) == before
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert {p.name for p in file_root.iterdir()} == before_root


def test_import_file_access_does_not_import_desktop_modules():
    probe = '''
import json
import sys

import reconstruction_web.file_access  # noqa: F401

print(json.dumps({
    "customtkinter": "customtkinter" in sys.modules,
    "reconstruction_zone": "reconstruction_gui.reconstruction_zone" in sys.modules,
    "app_infra": "reconstruction_gui.app_infra" in sys.modules,
    "prep360": "prep360" in sys.modules,
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    result = json.loads(completed.stdout.strip())
    assert result["customtkinter"] is False
    assert result["reconstruction_zone"] is False
    assert result["app_infra"] is False
    assert result["prep360"] is False


# --- WEB3 review-required fixes (2026-07-13) ---


def test_unc_root_refused_before_existence_check():
    """UNC roots must be rejected as a class, not merely because the share is unreachable."""
    for unc in ("\\\\rz-nonexistent\\share", "\\\\rz-nonexistent\\share\\sub", "//rz-nonexistent/share/sub"):
        with pytest.raises(WebStateConfigError, match="not allowed"):
            resolve_state_root(unc)


def test_containment_rejects_sibling_with_shared_prefix(tmp_path):
    """Guards against startswith-style containment: /a/foo must not contain /a/foobar."""
    root = tmp_path / "content_root"
    root.mkdir()
    sibling = tmp_path / "content_root_sibling"
    sibling.mkdir()
    assert is_same_or_descendant(root, root)
    assert not is_same_or_descendant(sibling, root)


@pytest.mark.parametrize(
    "bad_path",
    [
        pytest.param("CON ", id="reserved-trailing-space"),
        pytest.param("CON.", id="reserved-trailing-dot"),
        pytest.param("images/NUL .txt", id="reserved-component-trailing-space"),
    ],
)
def test_reserved_device_trailing_chars_refused(file_root, bad_path):
    registry, token = _registry_with_root(file_root)
    with pytest.raises(FileAccessError, match="not allowed"):
        registry.resolve(token, bad_path)


def _make_junction(link: Path, target: Path) -> bool:
    import subprocess
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", os.fspath(link), os.fspath(target)],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and link.exists()


def test_listing_does_not_follow_outward_junction(file_root, tmp_path):
    """A Windows junction inside the root pointing outside must classify as 'other', not follow it."""
    if os.name != "nt":
        pytest.skip("junctions are Windows-only")
    outside = tmp_path / "SECRET_OUTSIDE"
    outside.mkdir()
    (outside / "secret.txt").write_text("classified", encoding="utf-8")
    junction = file_root / "escape_junction"
    if not _make_junction(junction, outside):
        pytest.skip("junction creation not permitted")
    try:
        registry, token = _registry_with_root(file_root)
        listing = registry.list_dir(token, ".")
        entry = next(e for e in listing["entries"] if e["name"] == "escape_junction")
        assert entry["kind"] == "other", f"junction leaked as {entry!r}"
        assert "modified_time_utc" not in entry
        # and stat on the junction must be refused outright
        with pytest.raises(FileAccessError, match="outside|not found"):
            registry.stat(token, "escape_junction")
    finally:
        import subprocess
        subprocess.run(["cmd", "/c", "rmdir", os.fspath(junction)], capture_output=True, text=True)


def test_listing_follows_inward_junction(file_root, tmp_path):
    """A junction inside the root pointing to an in-root directory may classify as 'directory'."""
    if os.name != "nt":
        pytest.skip("junctions are Windows-only")
    inward = file_root / "inward_junction"
    target = file_root / "images"
    if not _make_junction(inward, target):
        pytest.skip("junction creation not permitted")
    try:
        registry, token = _registry_with_root(file_root)
        listing = registry.list_dir(token, ".")
        entry = next(e for e in listing["entries"] if e["name"] == "inward_junction")
        assert entry["kind"] == "directory"
    finally:
        import subprocess
        subprocess.run(["cmd", "/c", "rmdir", os.fspath(inward)], capture_output=True, text=True)
