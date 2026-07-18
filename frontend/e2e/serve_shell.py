"""Serve the compiled browser shell for the Playwright end-to-end test."""

from __future__ import annotations

import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Any

from reconstruction_web.jobs import JobRegistry
from reconstruction_web.server import HOST, make_server, shutdown_server
from reconstruction_web.state import build_state_config


DEFAULT_PORT = 18765


def _stop_on_signal(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _arm_windows_delete_on_close(directory: Path) -> tuple[Any, int] | None:
    """Let Windows remove the empty state root even after forced termination."""
    if os.name != "nt":
        return None

    import ctypes
    from ctypes import wintypes

    delete_access = 0x00010000
    share_all = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    directory_delete_on_close = 0x02000000 | 0x04000000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateFileW(
        str(directory),
        delete_access,
        share_all,
        None,
        open_existing,
        directory_delete_on_close,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel32, handle


def main() -> None:
    port = int(os.environ.get("E2E_PORT", str(DEFAULT_PORT)))
    temp_root = Path(tempfile.gettempdir()).resolve()
    state_root = Path(
        tempfile.mkdtemp(prefix="reconstruction-zone-web-e2e-", dir=temp_root)
    ).resolve()
    server = None
    delete_on_close = None
    try:
        registry = JobRegistry(_step_wait=0)
        server = make_server(
            build_state_config(state_root),
            port=port,
            job_registry=registry,
        )
        delete_on_close = _arm_windows_delete_on_close(state_root)
        signal.signal(signal.SIGTERM, _stop_on_signal)
        print(f"E2E shell ready at http://{HOST}:{port}/", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if server is not None:
                shutdown_server(server)
        finally:
            if delete_on_close is not None:
                kernel32, handle = delete_on_close
                kernel32.CloseHandle(handle)
            if state_root.exists():
                shutil.rmtree(state_root)


if __name__ == "__main__":
    main()
