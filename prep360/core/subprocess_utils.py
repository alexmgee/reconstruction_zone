"""
Shared subprocess utilities for prep360 core modules.

Provides isolated subprocess kwargs that prevent DLL contamination
from Miniconda/Conda directories on PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

# Directories containing these substrings are demoted to end of PATH.
_DEMOTE_PATH_PATTERNS = (
    "miniconda",
    "anaconda",
    "conda",
    "miniforge",
    "mambaforge",
)


def subprocess_kwargs_for_binary(binary_name_or_path: str) -> Dict[str, Any]:
    """Build subprocess kwargs with PATH isolation for an external binary.

    Resolves the binary path via shutil.which() if needed, then builds
    kwargs with cwd set to the binary's directory and conda dirs demoted
    on PATH.

    Falls back to basic CREATE_NO_WINDOW flags if the binary can't be found.
    """
    # Resolve to absolute path
    resolved = shutil.which(binary_name_or_path)
    if resolved is None:
        # Can't find it — return basic flags, let subprocess fail naturally
        if os.name == "nt":
            return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        return {}

    binary_dir = str(Path(resolved).resolve().parent)

    # Build sanitized PATH
    original_path = os.environ.get("PATH", "")
    dirs = original_path.split(os.pathsep)
    promoted = [binary_dir]
    demoted = []
    for d in dirs:
        norm = os.path.normcase(os.path.normpath(d))
        if norm == os.path.normcase(os.path.normpath(binary_dir)):
            continue
        if any(p in d.lower() for p in _DEMOTE_PATH_PATTERNS):
            demoted.append(d)
        else:
            promoted.append(d)

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(promoted + demoted)

    kwargs: Dict[str, Any] = {
        "cwd": binary_dir,
        "env": env,
    }

    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        kwargs["startupinfo"] = startupinfo

    return kwargs
