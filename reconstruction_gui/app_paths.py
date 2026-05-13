"""Application storage paths for Reconstruction Zone.

All user-writable runtime state should resolve through this module so packaged
builds, normal installs, and isolated new-install tests use the same rules.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_HOME_ENV = "RECONSTRUCTION_ZONE_APP_HOME"
MODEL_DIR_ENV = "RECONSTRUCTION_ZONE_MODEL_DIR"


def app_home(create: bool = False) -> Path:
    """Return the root directory for user-writable app state."""
    override = os.environ.get(APP_HOME_ENV)
    if override:
        path = Path(override)
    elif os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            path = Path(local_app) / "ReconstructionZone"
        else:
            path = Path.home() / ".reconstruction_zone"
    else:
        path = Path.home() / ".reconstruction_zone"

    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir(create: bool = False) -> Path:
    """Return the directory for model weight files."""
    override = os.environ.get(MODEL_DIR_ENV)
    path = Path(override) if override else app_home(create=create) / "models"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def log_dir(create: bool = False) -> Path:
    """Return the directory for runtime logs."""
    path = app_home(create=create) / "logs"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def crash_log_file(create: bool = False) -> Path:
    """Return the crash log path."""
    return log_dir(create=create) / "crash.log"


def prefs_dir(create: bool = False) -> Path:
    """Return the directory for GUI preferences."""
    path = app_home(create=create) / "prefs"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def prefs_file(create: bool = False) -> Path:
    """Return the GUI preferences file path."""
    return prefs_dir(create=create) / "studio_prefs.json"


def project_store_file(create: bool = False) -> Path:
    """Return the default project tracker file path."""
    path = app_home(create=create) / "projects" / "tracker.json"
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def activity_store_file(create: bool = False) -> Path:
    """Return the default project activity log file path."""
    path = app_home(create=create) / "projects" / "activity_log.json"
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def executable_dir() -> Path:
    """Return the packaged executable directory, or this package's directory."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundled_model_dirs() -> list[Path]:
    """Return intentional packaged model locations, in priority order."""
    base = executable_dir()
    return [base / "models", base]
