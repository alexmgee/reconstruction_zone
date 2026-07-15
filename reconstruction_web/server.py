"""Localhost-only HTTP server skeleton for Reconstruction Zone web."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from reconstruction_web import __version__
from reconstruction_web.file_access import FileAccessError, FolderTokenRegistry
from reconstruction_web.jobs import JobRegistry, JobRegistryError
from reconstruction_web.shell import SHELL_HTML
from reconstruction_web.state import WebStateConfig, WebStateConfigError, build_state_config

__all__ = ["HOST", "make_server", "main", "parse_root_argument", "shutdown_server"]

HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def parse_root_argument(value: str) -> tuple[str, str]:
    """Parse ``label=path`` root registration argument."""
    if "=" not in value:
        raise ValueError("Root argument must use label=path form.")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError("Root argument must use label=path form.")
    return label, path


def build_root_registry(root_specs: list[str]) -> FolderTokenRegistry:
    registry = FolderTokenRegistry()
    for spec in root_specs:
        try:
            label, path = parse_root_argument(spec)
        except ValueError as exc:
            raise WebStateConfigError(str(exc)) from exc
        try:
            registry.register_root(path, label=label)
        except FileAccessError as exc:
            raise WebStateConfigError(exc.message) from exc
    return registry


def make_server(
    state_config: WebStateConfig,
    *,
    port: int = DEFAULT_PORT,
    root_registry: FolderTokenRegistry | None = None,
    job_registry: JobRegistry | None = None,
) -> ThreadingHTTPServer:
    _ = state_config  # validated by caller; reserved for future route context
    registry = root_registry or FolderTokenRegistry()
    jobs = job_registry or JobRegistry()
    handler = _build_handler(registry, jobs)
    server = _SafeThreadingHTTPServer((HOST, port), handler)
    server.job_registry = jobs
    return server


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.state_root is None or str(args.state_root).strip() == "":
        print("Missing explicit non-production state root.", file=sys.stderr)
        return 1
    try:
        state_config = build_state_config(args.state_root)
        root_registry = build_root_registry(args.root)
    except WebStateConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    server = make_server(state_config, port=args.port, root_registry=root_registry)
    bind_host, bind_port = server.server_address
    print(f"reconstruction_web listening on http://{bind_host}:{bind_port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
    finally:
        shutdown_server(server)
    return 0


def shutdown_server(
    server: ThreadingHTTPServer,
    server_thread: threading.Thread | None = None,
    *,
    join_timeout: float = 5.0,
) -> None:
    """Stop HTTP acceptance, close the server, and join representative workers."""
    jobs = getattr(server, "job_registry", None)
    if jobs is not None:
        jobs.begin_shutdown()
    if server_thread is not None and server_thread.is_alive():
        server.shutdown()
    server.server_close()
    if server_thread is not None:
        server_thread.join(join_timeout)
        if server_thread.is_alive():
            raise RuntimeError("HTTP server did not stop during shutdown.")
    if jobs is not None:
        jobs.shutdown(join_timeout=join_timeout)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="reconstruction_web")
    parser.add_argument(
        "--state-root",
        default=None,
        help="Existing non-production directory for web-track state.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="label=path",
        help="Register a read-only content root (repeatable, label=path).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Localhost port (default: {DEFAULT_PORT}; use 0 for an OS-assigned port).",
    )
    return parser.parse_args(argv)


def _build_handler(
    registry: FolderTokenRegistry,
    jobs: JobRegistry,
) -> type[BaseHTTPRequestHandler]:
    health_payload = {
        "ok": True,
        "service": "reconstruction_web",
        "mode": "local",
        "local_only": True,
        "state_configured": True,
    }
    version_payload = {
        "service": "reconstruction_web",
        "version": __version__,
        "version_source": "reconstruction_web",
    }

    class ReconstructionWebHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def parse_request(self) -> bool:
            if not super().parse_request():
                return False
            if not self.path.startswith("/"):
                self._send_error_json(400, "bad_request", "Unsupported request target.")
                return False
            hosts = self.headers.get_all("Host") or []
            port = self.server.server_address[1]
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if len(hosts) != 1 or hosts[0].strip().lower() not in allowed:
                self._send_error_json(403, "forbidden_host", "Host not allowed.")
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                route = parsed.path
                query = parse_qs(parsed.query, keep_blank_values=True)

                if route == "/api/health":
                    self._send_json(200, health_payload)
                    return
                if route == "/api/version":
                    self._send_json(200, version_payload)
                    return
                if route == "/api/roots":
                    self._send_json(200, {"roots": registry.roots_for_api()})
                    return
                if route == "/api/files/list":
                    self._handle_files_list(query)
                    return
                if route == "/api/files/stat":
                    self._handle_files_stat(query)
                    return
                if route == "/api/jobs":
                    self._send_json(200, {"jobs": jobs.list_summaries()})
                    return
                if route.startswith("/api/jobs/") and route.count("/") == 3:
                    self._handle_job_detail(route.removeprefix("/api/jobs/"))
                    return
                if self.path == "/":
                    self._send_html(SHELL_HTML)
                    return

                self._send_error_json(404, "not_found", "Route not found.")
            except Exception:
                self._send_error_json(500, "internal_error", "Internal server error.")

        def do_POST(self) -> None:  # noqa: N802
            try:
                route = urlparse(self.path).path
                if route == "/api/jobs":
                    self._handle_job_create()
                    return
                match = re.fullmatch(r"/api/jobs/([^/]+)/cancel", route)
                if match is not None:
                    self._handle_job_cancel(match.group(1))
                    return
                self._send_error_json(404, "not_found", "Route not found.")
            except Exception:
                self._send_error_json(500, "internal_error", "Internal server error.")

        def _handle_job_create(self) -> None:
            body = self._read_bounded_body(require_empty=False)
            if body is None:
                return
            media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if media_type != "application/json":
                self._send_error_json(
                    415, "unsupported_media_type", "Content type must be application/json."
                )
                return
            try:
                request = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_error_json(400, "invalid_request", "Invalid request.")
                return
            if (
                not isinstance(request, dict)
                or set(request) != {"job_type", "behavior"}
                or type(request.get("job_type")) is not str
                or type(request.get("behavior")) is not str
                or request["job_type"] != "representative"
                or request["behavior"] not in {"complete", "fail"}
            ):
                self._send_error_json(400, "invalid_request", "Invalid request.")
                return
            try:
                snapshot = jobs.create_representative(behavior=request["behavior"])
            except JobRegistryError as exc:
                self._send_error_json(exc.status, exc.code, exc.message)
                return
            self._send_json(202, snapshot)

        def _handle_job_detail(self, job_id: str) -> None:
            try:
                snapshot = jobs.get_snapshot(job_id)
            except JobRegistryError as exc:
                self._send_error_json(exc.status, exc.code, exc.message)
                return
            self._send_json(200, snapshot)

        def _handle_job_cancel(self, job_id: str) -> None:
            body = self._read_bounded_body(require_empty=True)
            if body is None:
                return
            try:
                accepted, snapshot = jobs.request_cancel(job_id)
            except JobRegistryError as exc:
                self._send_error_json(exc.status, exc.code, exc.message)
                return
            self._send_json(
                202 if accepted else 200,
                {"cancel_accepted": accepted, "job": snapshot},
            )

        def _read_bounded_body(self, *, require_empty: bool) -> bytes | None:
            if self.headers.get("Transfer-Encoding") is not None:
                self._send_error_json(400, "invalid_request", "Invalid request.")
                return None
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length, 10) if raw_length is not None else None
            except ValueError:
                length = None
            if length is None or not 0 <= length <= 4096 or (require_empty and length != 0):
                self._send_error_json(400, "invalid_request", "Invalid request.")
                return None
            body = self.rfile.read(length)
            if len(body) != length:
                self._send_error_json(400, "invalid_request", "Invalid request.")
                return None
            return body

        def _handle_files_list(self, query: dict[str, list[str]]) -> None:
            token = _first_query_value(query, "root")
            rel_path = _first_query_value(query, "path")
            try:
                payload = registry.list_dir(token, rel_path)
            except FileAccessError as exc:
                self._send_json(exc.status, {"error": exc.code, "message": exc.message})
                return
            self._send_json(200, payload)

        def _handle_files_stat(self, query: dict[str, list[str]]) -> None:
            token = _first_query_value(query, "root")
            rel_path = _first_query_value(query, "path")
            try:
                payload = registry.stat(token, rel_path)
            except FileAccessError as exc:
                self._send_json(exc.status, {"error": exc.code, "message": exc.message})
                return
            self._send_json(200, payload)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                "connect-src 'self'; img-src data:; font-src 'none'; object-src 'none'; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status: int, code: str, message: str) -> None:
            self._send_json(status, {"error": code, "message": message})

    return ReconstructionWebHandler


class _SafeThreadingHTTPServer(ThreadingHTTPServer):
    job_registry: JobRegistry

    def handle_error(self, request: object, client_address: object) -> None:
        _ = request, client_address
        print("HTTP request handling failed safely.", file=sys.stderr)


def _first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]
