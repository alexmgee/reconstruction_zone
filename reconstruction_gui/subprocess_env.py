"""
Subprocess environment isolation for external binary calls.

External binaries (COLMAP, SphereSfM, ffmpeg, ffprobe, exiftool) must not
inherit the user's full PATH — Miniconda/Conda directories can inject
incompatible DLLs (e.g., miniconda's tiff.dll breaking COLMAP).

This module provides a single helper that builds a sanitized subprocess
environment for any external binary call.

Usage:
    env_kwargs = isolated_subprocess_kwargs(Path("C:/COLMAP/bin/colmap.exe"))
    subprocess.run(["colmap", "help"], **env_kwargs)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict

# Directories containing these substrings are demoted from PATH.
# They are not removed entirely (some tools may need them) — they are
# moved to the end of PATH so the binary's own directory wins.
_DEMOTE_PATH_PATTERNS = (
    "miniconda",
    "anaconda",
    "conda",
    "miniforge",
    "mambaforge",
)


def _sanitize_path(binary_dir: str, original_path: str) -> str:
    """Build a PATH with binary_dir first and conda dirs demoted to end."""
    dirs = original_path.split(os.pathsep)
    promoted = [binary_dir]
    demoted = []

    for d in dirs:
        # Skip if it's already the binary dir (avoid duplicates)
        if os.path.normcase(os.path.normpath(d)) == os.path.normcase(os.path.normpath(binary_dir)):
            continue
        d_lower = d.lower()
        if any(pattern in d_lower for pattern in _DEMOTE_PATH_PATTERNS):
            demoted.append(d)
        else:
            promoted.append(d)

    return os.pathsep.join(promoted + demoted)


def isolated_subprocess_kwargs(
    binary_path: Path,
    *,
    inherit_env: bool = True,
) -> Dict[str, Any]:
    """Build subprocess kwargs that isolate an external binary from PATH contamination.

    Sets:
        cwd     — binary's parent directory (so it finds its own DLLs)
        env     — copy of os.environ with binary dir first on PATH, conda dirs demoted

    On Windows, also adds CREATE_NO_WINDOW and STARTF_USESHOWWINDOW to hide
    console windows from spawned processes.

    Args:
        binary_path: Path to the external binary (e.g., colmap.exe, ffmpeg.exe)
        inherit_env: If True, start from os.environ. If False, start with empty env.

    Returns:
        Dict of kwargs to pass to subprocess.run() or subprocess.Popen().
    """
    binary_dir = str(binary_path.resolve().parent)
    env = dict(os.environ) if inherit_env else {}
    env["PATH"] = _sanitize_path(binary_dir, env.get("PATH", ""))

    kwargs: Dict[str, Any] = {
        "cwd": binary_dir,
        "env": env,
    }

    # Windows: hide console windows
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        kwargs["startupinfo"] = startupinfo

    return kwargs
