"""Localhost-only HTTP server skeleton for Reconstruction Zone web."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from reconstruction_web import __version__
from reconstruction_web.file_access import FileAccessError, FolderTokenRegistry
from reconstruction_web.state import WebStateConfig, WebStateConfigError, build_state_config

__all__ = ["HOST", "make_server", "main", "parse_root_argument"]

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
) -> ThreadingHTTPServer:
    _ = state_config  # validated by caller; reserved for future route context
    registry = root_registry or FolderTokenRegistry()
    handler = _build_handler(registry)
    return ThreadingHTTPServer((HOST, port), handler)


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
        server.server_close()
    return 0


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


def _build_handler(registry: FolderTokenRegistry) -> type[BaseHTTPRequestHandler]:
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

        def do_GET(self) -> None:  # noqa: N802
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

            self._send_json(404, {"error": "not_found"})

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

    return ReconstructionWebHandler


def _first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]
