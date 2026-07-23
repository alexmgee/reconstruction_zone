"""WEB4 in-memory representative job API tests."""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from reconstruction_web.jobs import (
    MAX_ACTIVE_JOBS,
    MAX_LOG_ENTRIES,
    MAX_TERMINAL_JOBS,
    JobRegistry,
    JobRegistryError,
)
from reconstruction_web.server import make_server, shutdown_server
from reconstruction_web.state import build_state_config


@pytest.fixture
def state_root(tmp_path: Path) -> Path:
    root = tmp_path / "web_state"
    root.mkdir()
    return root


def _start_server(state_root: Path, registry: JobRegistry | None = None):
    server = make_server(build_state_config(state_root), port=0, job_registry=registry)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request_json(
    server,
    method: str,
    route: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object], str]:
    host, port = server.server_address
    request = urllib.request.Request(
        f"http://{host}:{port}{route}",
        data=body,
        method=method,
        headers=headers or {},
    )
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8")
        return exc.code, json.loads(text), exc.headers.get("Content-Type", "")
    with response:
        text = response.read().decode("utf-8")
        return response.status, json.loads(text), response.headers.get("Content-Type", "")


def _create_http(server, behavior: str = "complete"):
    return _request_json(
        server,
        "POST",
        "/api/jobs",
        body=json.dumps({"job_type": "representative", "behavior": behavior}).encode(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )


def _wait_for_state(
    registry: JobRegistry,
    job_id: str,
    states: set[str],
    *,
    timeout: float = 5,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = registry.get_snapshot(job_id)
        if snapshot["state"] in states:
            return snapshot
        threading.Event().wait(0.002)
    raise AssertionError("job did not reach the expected state")


def _wait_for_no_workers(registry: JobRegistry, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not registry.has_live_workers():
            return
        threading.Event().wait(0.002)
    raise AssertionError("representative worker remained alive")


def _raw_http(server, request: bytes) -> tuple[int, dict[str, object]]:
    host, port = server.server_address
    with socket.create_connection((host, port), timeout=2) as client:
        client.settimeout(2)
        client.sendall(request)
        chunks = []
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    response = b"".join(chunks)
    head, body = response.split(b"\r\n\r\n", 1)
    status = int(head.split(b"\r\n", 1)[0].split()[1])
    return status, json.loads(body.decode("utf-8"))


def test_complete_lifecycle_progress_result_and_safe_snapshots():
    registry = JobRegistry(_step_wait=0.003)
    created = registry.create_representative(behavior="complete")
    job_id = created["job_id"]
    observed = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = registry.get_snapshot(job_id)
        observed.append(snapshot["progress"])
        if snapshot["state"] == "completed":
            break
        threading.Event().wait(0.001)
    assert snapshot["state"] == "completed"
    assert observed == sorted(observed)
    assert any(0 < value < 100 for value in observed)
    assert snapshot["progress"] == 100
    assert snapshot["result"] == {"kind": "representative", "steps_completed": 20}
    assert snapshot["error"] is None
    assert snapshot["finished_at"] is not None
    assert len(snapshot["logs"]) <= MAX_LOG_ENTRIES
    summary = registry.list_summaries()[0]
    assert "logs" not in summary and "result" not in summary and "error" not in summary
    registry.shutdown(join_timeout=2)


def test_controlled_failure_is_predictable_and_registry_remains_usable():
    registry = JobRegistry(_step_wait=0)
    failed_id = registry.create_representative(behavior="fail")["job_id"]
    failed = _wait_for_state(registry, failed_id, {"failed"})
    assert failed["progress"] == 50
    assert failed["result"] is None
    assert failed["error"] == {
        "code": "representative_failure",
        "message": "Representative job failed as requested.",
    }
    _wait_for_no_workers(registry)
    completed_id = registry.create_representative(behavior="complete")["job_id"]
    assert _wait_for_state(registry, completed_id, {"completed"})["state"] == "completed"
    registry.shutdown(join_timeout=2)


def test_unexpected_runner_exception_is_sanitized_and_handle_removed():
    sentinel = "C:/Users/sentinel/private/project.txt"

    def raising_runner(registry, job_id, behavior, cancel_event):
        _ = registry, job_id, behavior, cancel_event
        raise RuntimeError(sentinel)

    registry = JobRegistry(_runner=raising_runner)
    job_id = registry.create_representative(behavior="complete")["job_id"]
    snapshot = _wait_for_state(registry, job_id, {"failed"})
    _wait_for_no_workers(registry)
    encoded = json.dumps(snapshot)
    assert snapshot["error"]["code"] == "internal_job_error"
    assert sentinel not in encoded
    assert "RuntimeError" not in encoded and "Traceback" not in encoded
    registry.shutdown(join_timeout=1)


def test_thread_start_failure_returns_failed_snapshot_without_leak():
    sentinel = "C:/Users/sentinel/thread-start.txt"

    def fail_start(thread):
        _ = thread
        raise RuntimeError(sentinel)

    registry = JobRegistry(_thread_start=fail_start)
    snapshot = registry.create_representative(behavior="complete")
    assert snapshot["state"] == "failed"
    assert snapshot["error"]["code"] == "job_start_failure"
    assert sentinel not in json.dumps(snapshot)
    assert not registry.has_live_workers()


def test_cancel_while_queued_then_worker_acknowledges():
    captured: list[threading.Thread] = []
    registry = JobRegistry(_thread_start=captured.append)
    created = registry.create_representative(behavior="complete")
    job_id = created["job_id"]
    assert created["state"] == "queued"
    accepted, requested = registry.request_cancel(job_id)
    assert accepted is True and requested["state"] == "cancel_requested"
    again, unchanged = registry.request_cancel(job_id)
    assert again is True and unchanged == requested
    captured[0].start()
    cancelled = _wait_for_state(registry, job_id, {"cancelled"})
    assert cancelled["progress"] == 0 and cancelled["result"] is None
    registry.shutdown(join_timeout=2)


def test_cancel_while_running_and_cancel_cleanup_failure_transition():
    entered = threading.Event()

    def cancellable(registry, job_id, behavior, cancel_event):
        _ = behavior
        entered.set()
        cancel_event.wait(2)
        registry._acknowledge_cancel(job_id)

    registry = JobRegistry(_runner=cancellable)
    job_id = registry.create_representative(behavior="complete")["job_id"]
    assert entered.wait(1)
    accepted, snapshot = registry.request_cancel(job_id)
    assert accepted is True and snapshot["state"] == "cancel_requested"
    assert _wait_for_state(registry, job_id, {"cancelled"})["progress"] < 100
    registry.shutdown(join_timeout=2)

    cleanup_entered = threading.Event()

    def broken_cleanup(registry, job_id, behavior, cancel_event):
        _ = registry, job_id, behavior
        cleanup_entered.set()
        cancel_event.wait(2)
        raise RuntimeError("C:/Users/sentinel/cancel-cleanup.txt")

    broken = JobRegistry(_runner=broken_cleanup)
    broken_id = broken.create_representative(behavior="complete")["job_id"]
    assert cleanup_entered.wait(1)
    broken.request_cancel(broken_id)
    failed = _wait_for_state(broken, broken_id, {"failed"})
    assert failed["error"]["code"] == "internal_job_error"
    assert "sentinel" not in json.dumps(failed)
    _wait_for_no_workers(broken)
    broken.shutdown(join_timeout=2)


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_cancel_on_terminal_state_is_not_accepted(terminal: str):
    registry = JobRegistry(_step_wait=0)
    behavior = "fail" if terminal == "failed" else "complete"
    job_id = registry.create_representative(behavior=behavior)["job_id"]
    if terminal == "cancelled":
        registry.request_cancel(job_id)
    before = _wait_for_state(registry, job_id, {terminal})
    accepted, after = registry.request_cancel(job_id)
    assert accepted is False
    assert after == before
    registry.shutdown(join_timeout=2)


def test_cancel_versus_completion_first_locked_outcome_wins():
    ready = threading.Event()
    release = threading.Event()

    def gated_complete(registry, job_id, behavior, cancel_event):
        _ = behavior
        ready.set()
        release.wait(2)
        if cancel_event.is_set():
            registry._acknowledge_cancel(job_id)
        else:
            registry._complete(job_id)

    registry = JobRegistry(_runner=gated_complete)
    job_id = registry.create_representative(behavior="complete")["job_id"]
    assert ready.wait(1)
    accepted, _ = registry.request_cancel(job_id)
    assert accepted is True
    release.set()
    assert _wait_for_state(registry, job_id, {"cancelled"})["state"] == "cancelled"
    registry.shutdown(join_timeout=2)

    ready2 = threading.Event()
    release2 = threading.Event()

    def complete_first(registry, job_id, behavior, cancel_event):
        _ = behavior, cancel_event
        ready2.set()
        release2.wait(2)
        registry._complete(job_id)

    registry2 = JobRegistry(_runner=complete_first)
    second_id = registry2.create_representative(behavior="complete")["job_id"]
    assert ready2.wait(1)
    release2.set()
    completed = _wait_for_state(registry2, second_id, {"completed"})
    accepted, unchanged = registry2.request_cancel(second_id)
    assert accepted is False and unchanged == completed
    registry2.shutdown(join_timeout=2)


def test_progress_validation_and_terminal_immutability():
    entered = threading.Event()
    release = threading.Event()

    def gated(registry, job_id, behavior, cancel_event):
        _ = behavior, cancel_event
        registry.update_progress(job_id, progress=40, step_code="representative_work")
        entered.set()
        release.wait(2)
        registry._complete(job_id)

    registry = JobRegistry(_runner=gated)
    job_id = registry.create_representative(behavior="complete")["job_id"]
    assert entered.wait(1)
    with pytest.raises(ValueError):
        registry.update_progress(job_id, progress=39, step_code="representative_work")
    with pytest.raises(ValueError):
        registry.update_progress(job_id, progress=101, step_code="representative_work")
    release.set()
    completed = _wait_for_state(registry, job_id, {"completed"})
    with pytest.raises(RuntimeError):
        registry.update_progress(job_id, progress=50, step_code="representative_work")
    assert registry.get_snapshot(job_id) == completed
    registry.shutdown(join_timeout=2)


def test_active_cap_concurrent_creation_and_fifth_refusal():
    captured: list[threading.Thread] = []
    captured_lock = threading.Lock()

    def hold_start(thread):
        with captured_lock:
            captured.append(thread)

    registry = JobRegistry(_thread_start=hold_start)
    results: list[dict[str, object]] = []
    errors: list[JobRegistryError] = []
    result_lock = threading.Lock()

    def create_one():
        try:
            value = registry.create_representative(behavior="complete")
            with result_lock:
                results.append(value)
        except JobRegistryError as exc:
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=create_one) for _ in range(MAX_ACTIVE_JOBS + 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == MAX_ACTIVE_JOBS
    assert len({item["job_id"] for item in results}) == MAX_ACTIVE_JOBS
    assert [error.code for error in errors] == ["job_limit_reached"]
    assert len(captured) == MAX_ACTIVE_JOBS
    registry.begin_shutdown()
    for worker in captured:
        worker.start()
    registry.shutdown(join_timeout=2)


def test_concurrent_list_detail_cancel_has_no_deadlock_or_torn_snapshot():
    entered = threading.Event()

    def wait_for_cancel(registry, job_id, behavior, cancel_event):
        _ = behavior
        entered.set()
        cancel_event.wait(2)
        registry._acknowledge_cancel(job_id)

    registry = JobRegistry(_runner=wait_for_cancel)
    job_id = registry.create_representative(behavior="complete")["job_id"]
    assert entered.wait(1)
    failures: list[BaseException] = []

    def reader():
        try:
            for _ in range(100):
                detail = registry.get_snapshot(job_id)
                summary = registry.list_summaries()[0]
                assert detail["job_id"] == summary["job_id"]
                assert 0 <= detail["progress"] <= 100
                assert isinstance(detail["logs"], list)
        except BaseException as exc:
            failures.append(exc)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for thread in readers:
        thread.start()
    cancel_threads = [threading.Thread(target=registry.request_cancel, args=(job_id,)) for _ in range(4)]
    for thread in cancel_threads:
        thread.start()
    for thread in readers + cancel_threads:
        thread.join(3)
    assert all(not thread.is_alive() for thread in readers + cancel_threads)
    assert failures == []
    assert _wait_for_state(registry, job_id, {"cancelled"})["state"] == "cancelled"
    registry.shutdown(join_timeout=2)


def test_retention_prunes_oldest_terminal_and_never_live_jobs():
    registry = JobRegistry(_step_wait=0)
    ids = []
    for _ in range(MAX_TERMINAL_JOBS + 2):
        job_id = registry.create_representative(behavior="complete")["job_id"]
        ids.append(job_id)
        _wait_for_state(registry, job_id, {"completed"})
        _wait_for_no_workers(registry)
    retained = registry.list_summaries()
    assert len(retained) == MAX_TERMINAL_JOBS
    assert [item["job_id"] for item in retained] == list(reversed(ids[2:]))
    for pruned in ids[:2]:
        with pytest.raises(JobRegistryError) as exc_info:
            registry.get_snapshot(pruned)
        assert exc_info.value.code == "job_not_found"
    registry.shutdown(join_timeout=2)


def test_log_cap_and_snapshot_copy_isolation():
    def noisy(registry, job_id, behavior, cancel_event):
        _ = behavior, cancel_event
        for _ in range(MAX_LOG_ENTRIES + 5):
            registry.append_log(job_id, level="info", code="representative_progress")
        registry._complete(job_id)

    registry = JobRegistry(_runner=noisy)
    job_id = registry.create_representative(behavior="complete")["job_id"]
    snapshot = _wait_for_state(registry, job_id, {"completed"})
    assert len(snapshot["logs"]) == MAX_LOG_ENTRIES
    snapshot["logs"][0]["message"] = "mutated"
    snapshot["result"]["steps_completed"] = -1
    fresh = registry.get_snapshot(job_id)
    assert fresh["logs"][0]["message"] != "mutated"
    assert fresh["result"]["steps_completed"] == 20
    registry.shutdown(join_timeout=2)


def test_registry_is_process_local_and_writes_no_state(state_root: Path):
    registry = JobRegistry(_step_wait=0)
    complete = registry.create_representative(behavior="complete")["job_id"]
    fail = registry.create_representative(behavior="fail")["job_id"]
    cancel = registry.create_representative(behavior="complete")["job_id"]
    registry.request_cancel(cancel)
    _wait_for_state(registry, complete, {"completed"})
    _wait_for_state(registry, fail, {"failed"})
    _wait_for_state(registry, cancel, {"cancelled"})
    registry.shutdown(join_timeout=2)
    assert list(state_root.iterdir()) == []
    assert JobRegistry().list_summaries() == []


def test_sentinel_home_and_state_root_untouched_during_job(state_root: Path, monkeypatch):
    sentinel_home = state_root.parent / "sentinel_home"
    sentinel_home.mkdir()
    monkeypatch.setenv("HOME", os.fspath(sentinel_home))
    monkeypatch.setenv("USERPROFILE", os.fspath(sentinel_home))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: sentinel_home))
    monkeypatch.setattr(os.path, "expanduser", lambda value: os.fspath(sentinel_home / "blocked"))
    registry = JobRegistry(_step_wait=0)
    job_id = registry.create_representative(behavior="complete")["job_id"]
    _wait_for_state(registry, job_id, {"completed"})
    registry.shutdown(join_timeout=2)
    assert list(state_root.iterdir()) == []
    assert list(sentinel_home.iterdir()) == []
    # Import-isolation (jobs imports no desktop/workflow/prep360 module) is covered
    # robustly by test_import_jobs_does_not_import_desktop_or_workflow_modules in
    # tests/test_web_server_skeleton.py (clean subprocess). Asserting on process-global
    # sys.modules here is fragile under full-suite import pollution, so it is omitted.


def test_http_all_four_routes_and_json_content_type(state_root: Path):
    registry = JobRegistry(_step_wait=0.01)
    server, thread = _start_server(state_root, registry)
    try:
        status, created, content_type = _create_http(server)
        assert status == 202 and content_type.startswith("application/json")
        job_id = created["job_id"]
        assert set(created) == {
            "job_id", "job_type", "state", "progress", "current_step",
            "created_at", "started_at", "finished_at", "logs", "result", "error",
        }
        assert "/" not in job_id and "\\" not in job_id
        status, listing, content_type = _request_json(server, "GET", "/api/jobs")
        assert status == 200 and content_type.startswith("application/json")
        assert listing["jobs"][0]["job_id"] == job_id
        assert "logs" not in listing["jobs"][0]
        status, detail, _ = _request_json(server, "GET", f"/api/jobs/{job_id}")
        assert status == 200 and detail["job_id"] == job_id
        status, cancelled, _ = _request_json(
            server, "POST", f"/api/jobs/{job_id}/cancel", body=b""
        )
        assert status in {200, 202}
        assert isinstance(cancelled["cancel_accepted"], bool)
    finally:
        shutdown_server(server, thread, join_timeout=3)


@pytest.mark.parametrize(
    ("raw_request", "expected"),
    [
        (
            b"POST /api/jobs HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: -1\r\n"
            b"Content-Type: application/json\r\nConnection: close\r\n\r\n",
            400,
        ),
        (
            b"POST /api/jobs HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: 5000\r\n"
            b"Content-Type: application/json\r\nConnection: close\r\n\r\n{}",
            400,
        ),
        (
            b"POST /api/jobs HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nTransfer-Encoding: chunked\r\n"
            b"Content-Type: application/json\r\nConnection: close\r\n\r\n0\r\n\r\n",
            400,
        ),
        (
            b"POST /api/jobs HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n"
            b"Connection: close\r\n\r\n{}",
            400,
        ),
    ],
)
def test_http_rejects_unbounded_body_forms_promptly(
    state_root: Path, raw_request: bytes, expected: int
):
    server, thread = _start_server(state_root)
    try:
        started = time.monotonic()
        raw_request = raw_request.replace(
            b"{port}", str(server.server_address[1]).encode("ascii")
        )
        status, payload = _raw_http(server, raw_request)
        assert time.monotonic() - started < 2
        assert status == expected
        assert payload == {"error": "invalid_request", "message": "Invalid request."}
    finally:
        shutdown_server(server, thread)


def test_http_rejects_short_body_after_eof(state_root: Path):
    server, thread = _start_server(state_root)
    host, port = server.server_address
    try:
        with socket.create_connection((host, port), timeout=2) as client:
            client.settimeout(2)
            client.sendall(
                (
                    f"POST /api/jobs HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                    "Content-Length: 10\r\nContent-Type: application/json\r\n"
                    "Connection: close\r\n\r\n{}"
                ).encode("ascii")
            )
            client.shutdown(socket.SHUT_WR)
            response = b""
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                response += chunk
        assert b" 400 " in response.split(b"\r\n", 1)[0]
        assert b"invalid_request" in response
    finally:
        shutdown_server(server, thread)


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json-C:/Users/sentinel/private.txt",
        b"{}",
        b'{"job_type":"representative"}',
        b'{"job_type":"representative","behavior":"complete","extra":true}',
        b'{"job_type":1,"behavior":"complete"}',
        b'{"job_type":"unknown","behavior":"complete"}',
        b'{"job_type":"representative","behavior":"unknown"}',
    ],
)
def test_http_create_rejects_invalid_closed_requests(state_root: Path, payload: bytes):
    server, thread = _start_server(state_root)
    try:
        status, response, _ = _request_json(
            server,
            "POST",
            "/api/jobs",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        assert status == 400 and response["error"] == "invalid_request"
        assert "sentinel" not in json.dumps(response)
    finally:
        shutdown_server(server, thread)


def test_http_content_type_and_cancel_body_rules(state_root: Path):
    server, thread = _start_server(state_root)
    try:
        body = b'{"job_type":"representative","behavior":"complete"}'
        status, payload, _ = _request_json(
            server, "POST", "/api/jobs", body=body, headers={"Content-Type": "text/plain"}
        )
        assert status == 415 and payload["error"] == "unsupported_media_type"
        _, created, _ = _create_http(server)
        status, payload, _ = _request_json(
            server, "POST", f"/api/jobs/{created['job_id']}/cancel", body=b"x"
        )
        assert status == 400 and payload["error"] == "invalid_request"
    finally:
        shutdown_server(server, thread)


def test_http_unknown_and_malformed_ids_do_not_echo(state_root: Path):
    server, thread = _start_server(state_root)
    sentinels = ["not-an-id-C:/Users/private", "f" * 32]
    try:
        for sentinel in sentinels:
            safe_path = sentinel.replace("/", "%2F").replace(":", "%3A")
            status, payload, _ = _request_json(server, "GET", f"/api/jobs/{safe_path}")
            assert status == 404 and payload["error"] == "job_not_found"
            assert sentinel not in json.dumps(payload)
            status, payload, _ = _request_json(
                server, "POST", f"/api/jobs/{safe_path}/cancel", body=b""
            )
            assert status == 404 and payload["error"] == "job_not_found"
            assert sentinel not in json.dumps(payload)
    finally:
        shutdown_server(server, thread)


def test_http_limit_start_failure_and_shutdown_refusal(state_root: Path):
    captured: list[threading.Thread] = []
    registry = JobRegistry(_thread_start=captured.append)
    server, thread = _start_server(state_root, registry)
    try:
        for _ in range(MAX_ACTIVE_JOBS):
            assert _create_http(server)[0] == 202
        status, payload, _ = _create_http(server)
        assert status == 409 and payload["error"] == "job_limit_reached"
        assert len(captured) == MAX_ACTIVE_JOBS
        registry.begin_shutdown()
        status, payload, _ = _create_http(server)
        assert status == 409 and payload["error"] == "server_shutting_down"
        for worker in captured:
            worker.start()
    finally:
        shutdown_server(server, thread)

    def fail_start(worker):
        _ = worker
        raise RuntimeError("C:/Users/sentinel/start.txt")

    failed_registry = JobRegistry(_thread_start=fail_start)
    failed_server, failed_thread = _start_server(state_root, failed_registry)
    try:
        status, payload, _ = _create_http(failed_server)
        assert status == 202 and payload["state"] == "failed"
        assert payload["error"]["code"] == "job_start_failure"
        assert "sentinel" not in json.dumps(payload)
    finally:
        shutdown_server(failed_server, failed_thread)


def test_handler_exception_is_fixed_500_without_traceback(state_root: Path, capsys, monkeypatch):
    registry = JobRegistry()
    sentinel = "C:/Users/sentinel/handler-private.txt"

    def raise_internal(*, behavior):
        _ = behavior
        raise RuntimeError(sentinel)

    monkeypatch.setattr(registry, "create_representative", raise_internal)
    server, thread = _start_server(state_root, registry)
    try:
        status, payload, _ = _create_http(server)
        assert status == 500
        assert payload == {"error": "internal_error", "message": "Internal server error."}
    finally:
        shutdown_server(server, thread)
    captured = capsys.readouterr()
    assert sentinel not in captured.err
    assert "Traceback" not in captured.err


def test_shutdown_cancels_joins_and_refuses_late_create(state_root: Path):
    entered = threading.Event()

    def cancellable(registry, job_id, behavior, cancel_event):
        _ = behavior
        entered.set()
        cancel_event.wait(2)
        registry._acknowledge_cancel(job_id)

    registry = JobRegistry(_runner=cancellable)
    server, thread = _start_server(state_root, registry)
    _, created, _ = _create_http(server)
    assert entered.wait(1)
    shutdown_server(server, thread, join_timeout=3)
    assert registry.get_snapshot(created["job_id"])["state"] == "cancelled"
    assert not registry.has_live_workers()
    with pytest.raises(JobRegistryError) as exc_info:
        registry.create_representative(behavior="complete")
    assert exc_info.value.code == "server_shutting_down"
    assert list(state_root.iterdir()) == []


def test_safe_server_handle_error_emits_fixed_notice_only(state_root: Path, capsys):
    server = make_server(build_state_config(state_root), port=0)
    server.handle_error(object(), ("sentinel-client", 1))
    captured = capsys.readouterr()
    assert captured.err.strip() == "HTTP request handling failed safely."
    assert "sentinel-client" not in captured.err
    assert "Traceback" not in captured.err
    server.server_close()
