"""Localhost-only HTTP server skeleton for Reconstruction Zone web."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from reconstruction_web import __version__
from reconstruction_web.state import WebStateConfig, WebStateConfigError, build_state_config

__all__ = ["HOST", "make_server", "main"]

HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def make_server(state_config: WebStateConfig, *, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    _ = state_config  # validated by caller; reserved for future route context
    handler = _build_handler()
    return ThreadingHTTPServer((HOST, port), handler)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.state_root is None or str(args.state_root).strip() == "":
        print("Missing explicit non-production state root.", file=sys.stderr)
        return 1
    try:
        state_config = build_state_config(args.state_root)
    except WebStateConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    server = make_server(state_config, port=args.port)
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
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Localhost port (default: {DEFAULT_PORT}; use 0 for an OS-assigned port).",
    )
    return parser.parse_args(argv)


def _build_handler() -> type[BaseHTTPRequestHandler]:
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

            if route == "/api/health":
                self._send_json(200, health_payload)
                return
            if route == "/api/version":
                self._send_json(200, version_payload)
                return

            self._send_json(404, {"error": "not_found"})

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ReconstructionWebHandler
