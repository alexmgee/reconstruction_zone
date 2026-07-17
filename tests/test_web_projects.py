"""WEB6 fail-closed, read-only project API tests."""

from __future__ import annotations

import ast
import builtins
import hashlib
import http.client
import io
import json
import ntpath
import os
import subprocess
import sys
import threading
from pathlib import Path, PureWindowsPath
from unittest.mock import Mock, patch

import pytest

from reconstruction_gui import project_store as project_store_leaf
import reconstruction_web.server as web_server
from reconstruction_web.projects import (
    PROJECT_NOT_FOUND,
    PROJECT_STORE_UNAVAILABLE,
    ProjectReader,
    ProjectStoreConfigError,
    build_project_reader,
    resolve_project_store_path,
)
from reconstruction_web.server import make_server, shutdown_server
from reconstruction_web.state import build_state_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_SOURCE = REPOSITORY_ROOT / "reconstruction_web" / "projects.py"
JSON_CONTENT_TYPE = "application/json; charset=utf-8"


def _write_store(path: Path, projects: list[dict], **extra: object) -> Path:
    payload = {
        "version": 1,
        "store_path": r"Z:\private\top-level-only.json",
        "projects": projects,
        "unrelated_path": r"Y:\must-never-leak\secret.txt",
        **extra,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def copied_store(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    existing_source = tmp_path / "source_exists"
    existing_source.mkdir()
    existing_work = tmp_path / "work_exists"
    existing_work.mkdir()
    missing_source = tmp_path / "source_MiSsInG"
    missing_work = tmp_path / "work_missing"
    unc_source = r"\\web6.invalid\share\Capture"
    device_work = r"\\?\C:\WEB6\device"

    detail_project = {
        "id": "project-detail",
        "title": "Stored Detail Title",
        "created_at": "2026-07-14T08:30:00",
        "updated_at": "2026-07-16T12:00:00",
        "sources": [
            {
                "label": "Existing source",
                "path": os.fspath(existing_source),
                "media_type": "images",
                "file_count": 7,
                "notes": "Stored source notes",
                "future_source_field": "hidden",
            },
            {
                "label": "Duplicate source",
                "path": os.fspath(existing_source),
                "media_type": "images",
                "file_count": 7,
                "notes": "Duplicate path",
            },
            {
                "label": "Missing source",
                "path": os.fspath(missing_source),
                "media_type": "video",
                "file_count": 0,
                "notes": "Missing by design",
            },
            {
                "label": "UNC source",
                "path": unc_source,
                "media_type": "other",
                "file_count": 0,
                "notes": "Must not be probed",
            },
        ],
        "work_dirs": [
            {
                "label": "Existing work",
                "path": os.fspath(existing_work),
                "stage": "masked",
                "file_count": 5,
                "derived_from": "Existing source",
                "future_work_field": "hidden",
            },
            {
                "label": "Missing work",
                "path": os.fspath(missing_work),
                "stage": "aligned",
                "file_count": 0,
                "derived_from": "Missing source",
            },
            {
                "label": "Device work",
                "path": device_work,
                "stage": "",
                "file_count": 0,
                "derived_from": "",
            },
        ],
        "notes": "Stored project notes",
        "tags": ["fixture", "read-only"],
        "root_dir": r"R:\DisplayOnly\ProjectRoot",
        "static_masks_dir": r"R:\DisplayOnly\ProjectRoot\StaticMasks",
        "future_project_field": "hidden",
    }
    projects = [
        detail_project,
        {
            "id": "project-tie",
            "title": "Tie Wins By Reverse Insertion",
            "created_at": "2026-07-13T09:00:00",
            "updated_at": "2026-07-16T12:00:00",
            "sources": [],
            "work_dirs": [],
            "notes": "",
            "tags": ["tie"],
            "root_dir": "",
            "static_masks_dir": "",
        },
        {
            "id": "project-created-order",
            "title": "Blank Updated Timestamp",
            "created_at": "2026-07-15T10:00:00",
            "updated_at": "",
            "sources": [],
            "work_dirs": [],
            "notes": "",
            "tags": [],
            "root_dir": "",
            "static_masks_dir": "",
        },
    ]
    store_path = _write_store(tmp_path / "copied-project-store.json", projects)
    expected = {
        "detail_project": detail_project,
        "existing_source": os.fspath(existing_source),
        "existing_work": os.fspath(existing_work),
        "missing_source": os.fspath(missing_source),
        "missing_work": os.fspath(missing_work),
        "unc_source": unc_source,
        "device_work": device_work,
    }
    return store_path, expected


@pytest.fixture
def server_factory(tmp_path: Path):
    state_root = tmp_path / "web_state"
    state_root.mkdir()
    state_config = build_state_config(state_root)
    running: list[tuple[object, threading.Thread]] = []

    def start(project_store: str | Path | None = None):
        server = make_server(state_config, port=0, project_store=project_store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        running.append((server, thread))
        return server

    yield start

    for server, thread in reversed(running):
        shutdown_server(server, thread, join_timeout=3)


def _request(
    server: object,
    route: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
) -> tuple[int, str | None, bytes]:
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request(method, route, body=body)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type"), response.read()
    finally:
        connection.close()


def _json_request(server: object, route: str) -> tuple[int, dict, bytes]:
    status, content_type, body = _request(server, route)
    assert content_type == JSON_CONTENT_TYPE
    return status, json.loads(body), body


def _snapshot(path: Path) -> tuple[bytes, str, int, int]:
    content = path.read_bytes()
    stat = path.stat()
    return content, hashlib.sha256(content).hexdigest(), stat.st_size, stat.st_mtime_ns


@pytest.mark.parametrize(
    "candidate",
    [None, "", "   ", r"D:tracker-copy.json", r"\\server\share\copy.json", r"\\?\D:\copy.json"],
)
def test_lexically_invalid_store_paths_are_rejected_before_resolution(candidate):
    with patch.object(Path, "resolve", autospec=True) as resolve_spy:
        with pytest.raises(ProjectStoreConfigError):
            resolve_project_store_path(candidate)
        resolve_spy.assert_not_called()


def test_store_path_validation_accepts_only_safe_existing_file(tmp_path: Path, monkeypatch):
    safe = (tmp_path / "safe.json")
    safe.write_text('{"projects": []}', encoding="utf-8")
    assert resolve_project_store_path(safe) == safe.resolve()

    missing = tmp_path / "missing.json"
    with pytest.raises(ProjectStoreConfigError):
        resolve_project_store_path(missing)
    with pytest.raises(ProjectStoreConfigError):
        resolve_project_store_path(tmp_path)
    with pytest.raises(ProjectStoreConfigError):
        resolve_project_store_path(Path(tmp_path.anchor))
    with pytest.raises(ProjectStoreConfigError):
        resolve_project_store_path(PROJECTS_SOURCE)

    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    home_store = fake_home / "copy.json"
    home_store.write_text('{"projects": []}', encoding="utf-8")
    forbidden_store = fake_home / ".reconstruction_zone" / "nested" / "copy.json"
    forbidden_store.parent.mkdir(parents=True)
    forbidden_store.write_text('{"projects": []}', encoding="utf-8")
    monkeypatch.setenv("USERPROFILE", os.fspath(fake_home))
    monkeypatch.setenv("HOME", os.fspath(fake_home))
    with pytest.raises(ProjectStoreConfigError):
        resolve_project_store_path(home_store)
    with pytest.raises(ProjectStoreConfigError):
        resolve_project_store_path(forbidden_store)


def test_project_store_cli_argument_is_optional_and_independent():
    missing = web_server._parse_args(["--state-root", "state-only"])
    assert missing.state_root == "state-only"
    assert missing.project_store is None

    separate = web_server._parse_args(
        ["--state-root", "state-location", "--project-store", "copied-store.json"]
    )
    assert separate.state_root == "state-location"
    assert separate.project_store == "copied-store.json"


def test_symlink_to_forbidden_location_is_rejected(tmp_path: Path, monkeypatch):
    forbidden_anchor = tmp_path / "forbidden_workspace"
    forbidden_anchor.mkdir()
    target = forbidden_anchor / "copy.json"
    target.write_text('{"projects": []}', encoding="utf-8")
    link = tmp_path / "safe-looking.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    monkeypatch.setattr(
        "reconstruction_web.state._forbidden_workspace_anchors",
        lambda: (forbidden_anchor.resolve(),),
    )
    with pytest.raises(ProjectStoreConfigError):
        resolve_project_store_path(link)


def test_all_drive_root_production_names_are_lexically_rejected():
    for drive_letter in ("A", "D", "z"):
        candidate = f"{drive_letter}:{os.sep}" + "tracker.json"
        with patch.object(Path, "resolve", autospec=True) as resolve_spy:
            with pytest.raises(ProjectStoreConfigError):
                resolve_project_store_path(candidate)
            resolve_spy.assert_not_called()


def test_constructor_receives_exact_validated_nonproduction_path(copied_store):
    store_path, _ = copied_store
    validated = resolve_project_store_path(store_path)
    with patch(
        "reconstruction_web.projects.ProjectStore",
        wraps=project_store_leaf.ProjectStore,
    ) as constructor:
        reader = build_project_reader(store_path)
    assert isinstance(reader, ProjectReader)
    constructor.assert_called_once_with(store_path=str(validated))
    supplied = constructor.call_args.kwargs["store_path"]
    assert ntpath.normcase(supplied) != ntpath.normcase(r"D:\tracker.json")


def test_integrity_gate_reads_only_the_validated_nonproduction_store(copied_store, monkeypatch):
    store_path, _ = copied_store
    validated = store_path.resolve()
    original = Path.read_text
    calls: list[Path] = []

    def record_read(path: Path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_read)
    assert isinstance(build_project_reader(store_path), ProjectReader)
    assert calls == [validated, validated]


def test_production_path_primitives_are_recorded_and_never_called(
    tmp_path: Path, server_factory, monkeypatch
):
    production = PureWindowsPath(r"D:\tracker.json")
    recorders = {
        name: Mock(name=name)
        for name in ("open", "read_text", "read_bytes", "stat", "exists", "resolve")
    }

    def is_production(value: object) -> bool:
        try:
            text = os.fspath(value)
        except TypeError:
            return False
        return ntpath.normcase(ntpath.normpath(text)) == ntpath.normcase(
            ntpath.normpath(os.fspath(production))
        )

    original_open = builtins.open
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    original_stat = Path.stat
    original_exists = Path.exists
    original_resolve = Path.resolve

    def open_spy(file, *args, **kwargs):
        if is_production(file):
            recorders["open"](file, *args, **kwargs)
            return io.StringIO('{"projects": []}')
        return original_open(file, *args, **kwargs)

    def read_text_spy(path: Path, *args, **kwargs):
        if is_production(path):
            recorders["read_text"](path, *args, **kwargs)
            return '{"projects": []}'
        return original_read_text(path, *args, **kwargs)

    def read_bytes_spy(path: Path, *args, **kwargs):
        if is_production(path):
            recorders["read_bytes"](path, *args, **kwargs)
            return b'{"projects": []}'
        return original_read_bytes(path, *args, **kwargs)

    def stat_spy(path: Path, *args, **kwargs):
        if is_production(path):
            recorders["stat"](path, *args, **kwargs)
            return os.stat_result((0,) * 10)
        return original_stat(path, *args, **kwargs)

    def exists_spy(path: Path, *args, **kwargs):
        if is_production(path):
            recorders["exists"](path, *args, **kwargs)
            return False
        return original_exists(path, *args, **kwargs)

    def resolve_spy(path: Path, *args, **kwargs):
        if is_production(path):
            recorders["resolve"](path, *args, **kwargs)
            return path
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_spy)
    monkeypatch.setattr(Path, "read_text", read_text_spy)
    monkeypatch.setattr(Path, "read_bytes", read_bytes_spy)
    monkeypatch.setattr(Path, "stat", stat_spy)
    monkeypatch.setattr(Path, "exists", exists_spy)
    monkeypatch.setattr(Path, "resolve", resolve_spy)

    missing = tmp_path / "missing-store.json"
    assert build_project_reader(None) is None
    assert build_project_reader(missing) is None
    assert build_project_reader(os.fspath(production)) is None
    for candidate in (None, missing, os.fspath(production)):
        server = server_factory(candidate)
        assert _json_request(server, "/api/projects")[0] == 503
        assert _json_request(server, "/api/projects/project-id")[0] == 503

    for spy in recorders.values():
        spy.assert_not_called()


@pytest.mark.parametrize(
    "content",
    [
        "{",
        "[]",
        '{}',
        '{"projects": {}}',
        '{"projects": [null]}',
    ],
)
def test_malformed_or_failed_store_load_returns_fixed_503_without_leakage(
    tmp_path: Path, server_factory, content: str
):
    store = tmp_path / "bad-store.json"
    store.write_text(content, encoding="utf-8")
    server = server_factory(store)
    for route in ("/api/projects", "/api/projects/project-id"):
        status, payload, body = _json_request(server, route)
        assert status == 503
        assert payload == PROJECT_STORE_UNAVAILABLE
        text = body.decode("utf-8")
        assert os.fspath(store) not in text
        assert "JSON" not in text
        assert "Warning" not in text
        assert "Traceback" not in text


def test_unreadable_store_returns_fixed_503(tmp_path: Path, server_factory, monkeypatch):
    store = _write_store(tmp_path / "unreadable-store.json", [])
    original = Path.read_text

    def fail_store_read(path: Path, *args, **kwargs):
        if path == store.resolve():
            raise PermissionError(r"X:\private\parser-detail.txt")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_store_read)
    server = server_factory(store)
    for route in ("/api/projects", "/api/projects/project-id"):
        status, payload, body = _json_request(server, route)
        assert status == 503
        assert payload == PROJECT_STORE_UNAVAILABLE
        assert os.fspath(store).encode("utf-8") not in body
        assert b"parser-detail" not in body


def test_missing_store_returns_exact_fixed_503_on_both_routes(server_factory):
    server = server_factory()
    expected = json.dumps(PROJECT_STORE_UNAVAILABLE, sort_keys=True).encode("utf-8")
    for route in ("/api/projects", "/api/projects/anything"):
        status, payload, body = _json_request(server, route)
        assert status == 503
        assert payload == PROJECT_STORE_UNAVAILABLE
        assert body == expected


def test_project_list_exact_schema_values_counts_and_store_order(copied_store, server_factory):
    store_path, expected = copied_store
    server = server_factory(store_path)
    status, payload, _ = _json_request(server, "/api/projects")
    assert status == 200
    assert list(payload) == ["projects"]
    assert [project["id"] for project in payload["projects"]] == [
        "project-tie",
        "project-detail",
        "project-created-order",
    ]
    summary = payload["projects"][1]
    assert set(summary) == {
        "id",
        "title",
        "tags",
        "source_count",
        "work_dir_count",
        "created_at",
        "updated_at",
    }
    fixture = expected["detail_project"]
    assert summary["title"] == fixture["title"]
    assert summary["created_at"] == fixture["created_at"]
    assert summary["updated_at"] == fixture["updated_at"]
    assert summary["source_count"] == 4
    assert summary["work_dir_count"] == 3
    rendered = json.dumps(summary)
    for forbidden in ("sources", "work_dirs", "notes", "root_dir", "store_path", "hidden"):
        assert forbidden not in rendered


def test_valid_empty_store_is_200_not_unavailable(tmp_path: Path, server_factory):
    store = _write_store(tmp_path / "empty-store.json", [])
    status, payload, _ = _json_request(server_factory(store), "/api/projects")
    assert status == 200
    assert payload == {"projects": []}


def test_project_detail_exact_schema_values_paths_and_fixed_404(copied_store, server_factory):
    store_path, expected = copied_store
    server = server_factory(store_path)
    status, detail, _ = _json_request(server, "/api/projects/project-detail")
    assert status == 200
    assert set(detail) == {
        "id",
        "title",
        "created_at",
        "updated_at",
        "sources",
        "work_dirs",
        "notes",
        "tags",
        "root_dir",
        "static_masks_dir",
    }
    fixture = expected["detail_project"]
    for field in ("title", "created_at", "updated_at", "notes", "tags", "root_dir", "static_masks_dir"):
        assert detail[field] == fixture[field]
    assert all(
        set(source) == {"label", "path", "media_type", "file_count", "notes", "exists"}
        for source in detail["sources"]
    )
    assert all(
        set(work_dir) == {"label", "path", "stage", "file_count", "derived_from", "exists"}
        for work_dir in detail["work_dirs"]
    )
    assert [source["path"] for source in detail["sources"]] == [
        source["path"] for source in fixture["sources"]
    ]
    assert [work_dir["path"] for work_dir in detail["work_dirs"]] == [
        work_dir["path"] for work_dir in fixture["work_dirs"]
    ]
    rendered = json.dumps(detail)
    assert "store_path" not in rendered
    assert "future_" not in rendered
    assert "unrelated_path" not in rendered

    expected_not_found = json.dumps(PROJECT_NOT_FOUND, sort_keys=True).encode("utf-8")
    for project_id in ("unknown-id", "%5C%5Cattacker%5Cshare", ""):
        status, payload, body = _json_request(server, f"/api/projects/{project_id}")
        assert status == 404
        assert payload == PROJECT_NOT_FOUND
        assert body == expected_not_found
        if project_id:
            assert project_id.encode("utf-8") not in body
    status, payload, _ = _json_request(server, "/api/projects/a/b")
    assert status == 404
    assert payload == {"error": "not_found", "message": "Route not found."}


def test_existence_bits_short_circuit_unc_and_use_no_content_or_mutation_operations(
    copied_store, server_factory, monkeypatch
):
    store_path, expected = copied_store
    server = server_factory(store_path)
    original_exists = Path.exists
    exists_calls: list[str] = []

    def exists_spy(path: Path) -> bool:
        exists_calls.append(os.fspath(path))
        return original_exists(path)

    def forbidden(*args, **kwargs):
        raise AssertionError(f"forbidden operation reached: {args!r} {kwargs!r}")

    monkeypatch.setattr(Path, "exists", exists_spy)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(os, "walk", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    for name in ("save", "create_project", "delete_project", "relocate_source", "relocate_work_dir"):
        monkeypatch.setattr(project_store_leaf.ProjectStore, name, forbidden)
    for name in ("add_source", "add_work_dir"):
        monkeypatch.setattr(project_store_leaf.Project, name, forbidden)
    monkeypatch.setattr(project_store_leaf, "count_media_files", forbidden)

    status, detail, _ = _json_request(server, "/api/projects/project-detail")
    assert status == 200
    source_bits = [(item["path"], item["exists"]) for item in detail["sources"]]
    work_bits = [(item["path"], item["exists"]) for item in detail["work_dirs"]]
    assert source_bits == [
        (expected["existing_source"], True),
        (expected["existing_source"], True),
        (expected["missing_source"], False),
        (expected["unc_source"], False),
    ]
    assert work_bits == [
        (expected["existing_work"], True),
        (expected["missing_work"], False),
        (expected["device_work"], False),
    ]
    assert exists_calls.count(expected["existing_source"]) == 1
    assert expected["unc_source"] not in exists_calls
    assert expected["device_work"] not in exists_calls
    assert detail["root_dir"] not in exists_calls
    assert detail["static_masks_dir"] not in exists_calls

    before = list(exists_calls)
    assert _json_request(server, "/api/projects/%5C%5Cattacker%5Cshare")[0] == 404
    assert exists_calls == before


def test_gets_are_disk_and_memory_read_only_and_posts_remain_unrouted(
    copied_store, server_factory
):
    store_path, _ = copied_store
    before = _snapshot(store_path)
    server = server_factory(store_path)
    list_first = _request(server, "/api/projects")
    list_second = _request(server, "/api/projects")
    detail_first = _request(server, "/api/projects/project-detail")
    detail_second = _request(server, "/api/projects/project-detail")
    assert list_first == list_second
    assert detail_first == detail_second
    assert _json_request(server, "/api/projects/not-found")[0] == 404

    for route in (
        "/api/projects",
        "/api/projects/project-detail",
        "/api/projects/create",
        "/api/projects/project-detail/delete",
        "/api/projects/project-detail/relocate",
        "/api/projects/save",
    ):
        status, _, _ = _request(server, route, method="POST", body=b"")
        assert status == 404
    for method in ("PUT", "PATCH", "DELETE"):
        status, _, _ = _request(server, "/api/projects/project-detail", method=method)
        assert status == 501
    assert _snapshot(store_path) == before


def test_internal_failure_is_fixed_and_does_not_leak_exception(copied_store, server_factory, monkeypatch):
    store_path, _ = copied_store
    server = server_factory(store_path)
    sentinel = r"X:\private\exception-only.txt"

    def fail(_self):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(ProjectReader, "list_payload", fail)
    status, payload, body = _json_request(server, "/api/projects")
    assert status == 500
    assert payload == {"error": "internal_error", "message": "Internal server error."}
    assert sentinel.encode("utf-8") not in body
    assert b"RuntimeError" not in body
    assert b"Traceback" not in body


def test_projects_source_enforces_import_call_and_serializer_boundaries():
    source = PROJECTS_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    gui_imports: list[tuple[str | None, tuple[str, ...]]] = []
    heavy_roots = {"customtkinter", "tkinter", "torch", "cv2", "numpy", "PIL"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] not in heavy_roots
                if alias.name.startswith("reconstruction_gui"):
                    gui_imports.append((alias.name, ()))
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".", 1)[0] not in heavy_roots
            if (node.module or "").startswith("reconstruction_gui"):
                gui_imports.append((node.module, tuple(alias.name for alias in node.names)))
    assert gui_imports == [("reconstruction_gui.project_store", ("ProjectStore",))]

    constructors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ProjectStore"
    ]
    assert len(constructors) == 1
    assert not constructors[0].args
    assert [keyword.arg for keyword in constructors[0].keywords] == ["store_path"]
    store_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr == "_store"
    }
    assert store_calls == {"list_projects", "get_project"}
    assert store_calls <= {"list_projects", "get_project", "validate_paths"}
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
        for node in ast.walk(tree)
    )
    assert "tracker.json" not in source
    assert "D:\\" not in source

    server_tree = ast.parse((REPOSITORY_ROOT / "reconstruction_web" / "server.py").read_text("utf-8"))
    post_handlers = [
        node for node in ast.walk(server_tree) if isinstance(node, ast.FunctionDef) and node.name == "do_POST"
    ]
    assert len(post_handlers) == 1
    assert "/api/projects" not in ast.unparse(post_handlers[0])


def test_only_projects_module_imports_the_gui_leaf():
    web_root = REPOSITORY_ROOT / "reconstruction_web"
    imports: list[tuple[Path, str]] = []
    for path in web_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("reconstruction_gui"):
                imports.append((path, node.module or ""))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("reconstruction_gui"):
                        imports.append((path, alias.name))
    assert imports == [(PROJECTS_SOURCE, "reconstruction_gui.project_store")]


def test_projects_module_clean_import_allowlist():
    probe = r'''
import json
import sys

before = set(sys.modules)
import reconstruction_web.projects  # noqa: F401
new_gui = sorted(
    name for name in set(sys.modules) - before
    if name == "reconstruction_gui" or name.startswith("reconstruction_gui.")
)
heavy = sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + ".")
           for root in ("customtkinter", "tkinter", "torch", "cv2", "numpy", "PIL"))
)
print(json.dumps({"new_gui": new_gui, "heavy": heavy}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert set(result["new_gui"]) == {
        "reconstruction_gui",
        "reconstruction_gui._version",
        "reconstruction_gui.project_store",
    }
    assert result["heavy"] == []
