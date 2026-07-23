"""Fail-closed, read-only access to an explicitly configured project store."""

from __future__ import annotations

import io
import json
import ntpath
import os
import string
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path, PureWindowsPath
from typing import Any

from reconstruction_gui.project_store import ProjectStore
from reconstruction_web.state import (
    WebStateConfigError,
    reject_if_under_forbidden_home_children,
    resolve_state_root,
)

PROJECT_STORE_UNAVAILABLE = {
    "error": "project_store_unavailable",
    "message": "A valid non-production project store is required.",
}
PROJECT_NOT_FOUND = {
    "error": "project_not_found",
    "message": "Project not found.",
}


class ProjectStoreConfigError(ValueError):
    """Raised when the explicit project-store path is absent or unsafe."""


class ProjectReader:
    """Fixed-schema read facade over one startup-loaded project store."""

    __slots__ = ("_store",)

    def __init__(self, store: ProjectStore) -> None:
        self._store = store

    def list_payload(self) -> dict[str, Any]:
        return {
            "projects": [
                {
                    "id": project.id,
                    "title": project.title,
                    "tags": list(project.tags),
                    "source_count": len(project.sources),
                    "work_dir_count": len(project.work_dirs),
                    "created_at": project.created_at,
                    "updated_at": project.updated_at,
                }
                for project in self._store.list_projects()
            ]
        }

    def detail_payload(self, project_id: str) -> dict[str, Any] | None:
        project = self._store.get_project(project_id)
        if project is None:
            return None

        stored_paths = {item.path for item in (*project.sources, *project.work_dirs)}
        existence = {path: _local_path_exists(path) for path in stored_paths}
        return {
            "id": project.id,
            "title": project.title,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "sources": [
                {
                    "label": source.label,
                    "path": source.path,
                    "media_type": source.media_type,
                    "file_count": source.file_count,
                    "notes": source.notes,
                    "exists": existence[source.path],
                }
                for source in project.sources
            ],
            "work_dirs": [
                {
                    "label": work_dir.label,
                    "path": work_dir.path,
                    "stage": work_dir.stage,
                    "file_count": work_dir.file_count,
                    "derived_from": work_dir.derived_from,
                    "exists": existence[work_dir.path],
                }
                for work_dir in project.work_dirs
            ],
            "notes": project.notes,
            "tags": list(project.tags),
            "root_dir": project.root_dir,
            "static_masks_dir": project.static_masks_dir,
        }


def resolve_project_store_path(explicit_store: str | Path | None) -> Path:
    """Resolve one explicit existing store file without a production fallback."""
    if explicit_store is None or str(explicit_store).strip() == "":
        raise ProjectStoreConfigError("Project store is not configured.")

    text = os.fspath(explicit_store)
    windows_path = PureWindowsPath(text)
    if text.startswith(("\\\\", "//")) or windows_path.drive.startswith("\\\\"):
        raise ProjectStoreConfigError("Project store path is not allowed.")
    if len(text) >= 2 and text[1] == ":" and (
        len(text) == 2 or text[2] not in ("\\", "/")
    ):
        raise ProjectStoreConfigError("Drive-relative project store paths are not allowed.")

    raw = Path(text)
    if _is_production_store_path(raw):
        raise ProjectStoreConfigError("Project store path is not allowed.")
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProjectStoreConfigError("Project store does not exist.") from exc
    if not resolved.is_file():
        raise ProjectStoreConfigError("Project store must be an existing file.")

    resolved_text = os.fspath(resolved)
    resolved_windows = PureWindowsPath(resolved_text)
    if resolved_text.startswith(("\\\\", "//")) or resolved_windows.drive.startswith("\\\\"):
        raise ProjectStoreConfigError("Project store path is not allowed.")
    try:
        resolve_state_root(resolved.parent)
        reject_if_under_forbidden_home_children(resolved)
    except WebStateConfigError as exc:
        raise ProjectStoreConfigError("Project store path is not allowed.") from exc
    if _is_production_store_path(resolved):
        raise ProjectStoreConfigError("Project store path is not allowed.")
    return resolved


def build_project_reader(explicit_store: str | Path | None) -> ProjectReader | None:
    """Load one validated store, returning an unavailable sentinel on any failure."""
    try:
        resolved = resolve_project_store_path(explicit_store)
        envelope = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(envelope, dict) or not isinstance(envelope.get("projects"), list):
            raise ProjectStoreConfigError("Invalid project store envelope.")

        captured_output = io.StringIO()
        with redirect_stdout(captured_output), redirect_stderr(captured_output):
            store = ProjectStore(store_path=str(resolved))
        if captured_output.getvalue():
            raise ProjectStoreConfigError("Project store failed to load.")
    except Exception:
        return None
    return ProjectReader(store)


def _is_production_store_path(path: Path) -> bool:
    candidate = ntpath.normcase(ntpath.normpath(os.fspath(path)))
    store_name = "tracker" + ".json"
    for drive_letter in string.ascii_uppercase:
        denied = Path(f"{drive_letter}:{os.sep}") / store_name
        if candidate == ntpath.normcase(ntpath.normpath(os.fspath(denied))):
            return True
    return False


def _local_path_exists(stored_path: str) -> bool:
    text = os.fspath(stored_path)
    windows_path = PureWindowsPath(text)
    if text.startswith(("\\\\", "//")) or windows_path.drive.startswith("\\\\"):
        return False
    try:
        return Path(text).exists()
    except (OSError, ValueError):
        return False
