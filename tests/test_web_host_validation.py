"""WEB5.5 Host-header and request-target validation tests."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from pathlib import Path

import pytest

from reconstruction_web.jobs import JobRegistry
from reconstruction_web.server import HOST, make_server, shutdown_server
from reconstruction_web.state import build_state_config


FORBIDDEN_BODY = b'{"error": "forbidden_host", "message": "Host not allowed."}'
BAD_TARGET_BODY = b'{"error": "bad_request", "message": "Unsupported request target."}'
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
JOB_CREATE_BODY = json.dumps(
    {"job_type": "representative", "behavior": "complete"}
).encode("utf-8")


@pytest.fixture
def host_server(tmp_path: Path):
    state_root = tmp_path / "web_state"
    state_root.mkdir()
    registry = JobRegistry(_step_wait=1)
    server = make_server(build_state_config(state_root), port=0, job_registry=registry)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        shutdown_server(server, thread, join_timeout=3)


def _request(
    server,
    method: str,
    target: str,
    host_header: str,
    *,
    headers: tuple[tuple[str, str], ...] = (),
    body: bytes | None = None,
) -> tuple[int, list[tuple[str, str]], bytes]:
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.putrequest(method, target, skip_host=True)
        connection.putheader("Host", host_header)
        for name, value in headers:
            connection.putheader(name, value)
        if body is not None and not any(
            name.lower() == "content-length" for name, _ in headers
        ):
            connection.putheader("Content-Length", str(len(body)))
        connection.endheaders(body)
        response = connection.getresponse()
        response_headers = [
            (name.lower(), value) for name, value in response.getheaders()
        ]
        return response.status, response_headers, response.read()
    finally:
        connection.close()


def _raw_request(
    server, request: bytes
) -> tuple[int, list[tuple[str, str]], bytes, bytes]:
    host, port = server.server_address
    with socket.create_connection((host, port), timeout=5) as client:
        client.settimeout(5)
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    raw_response = b"".join(chunks)
    raw_headers, body = raw_response.split(b"\r\n\r\n", 1)
    header_lines = raw_headers.decode("iso-8859-1").split("\r\n")
    status = int(header_lines[0].split()[1])
    headers = [
        (name.strip().lower(), value.strip())
        for line in header_lines[1:]
        for name, value in [line.split(":", 1)]
    ]
    return status, headers, body, raw_response


def _header(headers: list[tuple[str, str]], name: str) -> str | None:
    return dict(headers).get(name.lower())


def _assert_forbidden(
    response: tuple[int, list[tuple[str, str]], bytes]
) -> None:
    status, headers, body = response
    assert status == 403
    assert _header(headers, "content-type") == JSON_CONTENT_TYPE
    assert body == FORBIDDEN_BODY


def test_allowed_loopback_hosts_reach_normal_routes(host_server):
    port = host_server.server_address[1]
    host_headers = (
        f"127.0.0.1:{port}",
        f"localhost:{port}",
        f"LOCALHOST:{port}",
        f"  localhost:{port} ",
    )
    routes = ("/", "/api/health", "/api/version", "/api/roots")

    for host_header in host_headers:
        for route in routes:
            status, headers, body = _request(host_server, "GET", route, host_header)
            assert status == 200, (host_header, route, body)
            if route == "/":
                assert _header(headers, "content-type") == "text/html; charset=utf-8"
                assert body
            else:
                assert _header(headers, "content-type") == JSON_CONTENT_TYPE
                assert isinstance(json.loads(body), dict)


def test_spoofed_host_rejected_before_every_dispatch_branch(host_server):
    port = host_server.server_address[1]
    spoofed_host = f"evil.example:{port}"
    requests = (
        ("GET", "/", (), None),
        ("GET", "/api/health", (), None),
        ("GET", "/api/version", (), None),
        ("GET", "/api/roots", (), None),
        ("GET", "/api/files/list", (), None),
        ("GET", "/api/files/stat", (), None),
        ("GET", "/api/jobs", (), None),
        ("GET", f"/api/jobs/{'0' * 32}", (), None),
        (
            "POST",
            "/api/jobs",
            (("Content-Type", "application/json"),),
            JOB_CREATE_BODY,
        ),
        ("POST", f"/api/jobs/{'0' * 32}/cancel", (), b""),
    )

    for method, target, headers, body in requests:
        _assert_forbidden(
            _request(
                host_server,
                method,
                target,
                spoofed_host,
                headers=headers,
                body=body,
            )
        )


def test_spoofed_cancel_does_not_mutate_job(host_server):
    port = host_server.server_address[1]
    allowed_host = f"127.0.0.1:{port}"
    status, _, body = _request(
        host_server,
        "POST",
        "/api/jobs",
        allowed_host,
        headers=(("Content-Type", "application/json"),),
        body=JOB_CREATE_BODY,
    )
    assert status == 202
    job_id = json.loads(body)["job_id"]

    deadline = time.monotonic() + 2
    before = None
    while time.monotonic() < deadline:
        status, _, body = _request(
            host_server, "GET", f"/api/jobs/{job_id}", allowed_host
        )
        assert status == 200
        before = json.loads(body)
        if before["state"] == "running":
            break
        threading.Event().wait(0.005)
    assert before is not None and before["state"] == "running"

    _assert_forbidden(
        _request(
            host_server,
            "POST",
            f"/api/jobs/{job_id}/cancel",
            f"evil.example:{port}",
            body=b"",
        )
    )
    status, _, body = _request(
        host_server, "GET", f"/api/jobs/{job_id}", allowed_host
    )
    assert status == 200
    after = json.loads(body)
    assert after["state"] == before["state"] == "running"


def test_host_matching_is_exact(host_server):
    port = host_server.server_address[1]
    rejected_hosts = (
        "localhost",
        f"user@localhost:{port}",
        f"localhost.:{port}",
        f"localhost:{port}0",
        f"xlocalhost:{port}",
        "127.0.0.1",
        "127.0.0.1:1",
        f"[::1]:{port}",
        f"127.0.0.1:0{port}",
        f"127.0.0.1:{port}garbage",
    )

    for host_header in rejected_hosts:
        _assert_forbidden(
            _request(host_server, "GET", "/api/health", host_header)
        )


def test_duplicate_host_headers_rejected_in_both_orders(host_server):
    port = host_server.server_address[1]
    allowed = f"localhost:{port}"
    spoofed = f"evil.example:{port}"

    for first, second in ((allowed, spoofed), (spoofed, allowed)):
        request = (
            f"GET /api/roots HTTP/1.1\r\nHost: {first}\r\nHost: {second}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        status, headers, body, _ = _raw_request(host_server, request)
        _assert_forbidden((status, headers, body))


def test_missing_host_header_rejected_for_http_1_0(host_server):
    status, headers, body, _ = _raw_request(
        host_server, b"GET / HTTP/1.0\r\n\r\n"
    )
    _assert_forbidden((status, headers, body))


def test_absolute_form_targets_rejected_before_routes(host_server):
    port = host_server.server_address[1]
    allowed_host = f"localhost:{port}"
    get_request = (
        f"GET http://evil.example:{port}/api/roots HTTP/1.1\r\n"
        f"Host: {allowed_host}\r\nConnection: close\r\n\r\n"
    ).encode("ascii")
    status, headers, body, _ = _raw_request(host_server, get_request)
    assert status == 400
    assert _header(headers, "content-type") == JSON_CONTENT_TYPE
    assert body == BAD_TARGET_BODY

    post_request = (
        f"POST http://evil.example:{port}/api/jobs HTTP/1.1\r\n"
        f"Host: {allowed_host}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(JOB_CREATE_BODY)}\r\nConnection: close\r\n\r\n"
    ).encode("ascii") + JOB_CREATE_BODY
    status, headers, body, _ = _raw_request(host_server, post_request)
    assert status == 400
    assert _header(headers, "content-type") == JSON_CONTENT_TYPE
    assert body == BAD_TARGET_BODY

    status, _, body = _request(host_server, "GET", "/api/jobs", allowed_host)
    assert status == 200
    assert json.loads(body) == {"jobs": []}


def test_x_forwarded_host_does_not_rescue_spoofed_host(host_server):
    port = host_server.server_address[1]
    _assert_forbidden(
        _request(
            host_server,
            "GET",
            "/api/roots",
            f"evil.example:{port}",
            headers=(("X-Forwarded-Host", f"localhost:{port}"),),
        )
    )


def test_forbidden_response_does_not_echo_received_host(host_server):
    port = host_server.server_address[1]
    raw_host = f"evil.example:{port}"
    request = (
        f"GET /api/roots HTTP/1.1\r\nHost: {raw_host}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii")
    status, headers, body, raw_response = _raw_request(host_server, request)
    _assert_forbidden((status, headers, body))
    lowered_response = raw_response.lower()
    assert raw_host.encode("ascii").lower() not in lowered_response
    assert b"evil.example" not in lowered_response


def test_server_binding_remains_ipv4_loopback(host_server):
    assert HOST == "127.0.0.1"
    assert host_server.server_address[0] == "127.0.0.1"


def test_host_guard_precedes_unsupported_method_dispatch(host_server):
    port = host_server.server_address[1]
    _assert_forbidden(
        _request(host_server, "OPTIONS", "/", f"evil.example:{port}")
    )
    status, _, _ = _request(host_server, "OPTIONS", "/", f"localhost:{port}")
    assert status == 501
