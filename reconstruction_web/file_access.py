"""Folder-token file access boundary for the local web server."""

from __future__ import annotations

import os
import re
import secrets
import stat as _stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from reconstruction_web.state import (
    WebStateConfigError,
    reject_if_under_forbidden_home_children,
    resolve_state_root,
)

__all__ = [
    "FileAccessError",
    "FolderTokenRegistry",
    "RegisteredRoot",
    "is_same_or_descendant",
    "MAX_RELATIVE_PATH_LENGTH",
]

MAX_RELATIVE_PATH_LENGTH = 4096

_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

_DRIVE_LEAK_PATTERN = re.compile(r"[A-Za-z]:[\\/]")


class FileAccessError(ValueError):
    """Raised when a file-access operation is refused."""

    def __init__(self, code: str, message: str, *, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass(frozen=True)
class RegisteredRoot:
    token: str
    label: str
    root: Path


class FolderTokenRegistry:
    """In-memory registry of approved read-only content roots."""

    def __init__(self) -> None:
        self._roots_by_token: dict[str, RegisteredRoot] = {}
        self._token_by_label: dict[str, str] = {}

    def register_root(self, root: str | Path, *, label: str) -> RegisteredRoot:
        _validate_label(label)
        if label in self._token_by_label:
            raise FileAccessError(
                "duplicate_label",
                "Root label is already registered.",
                status=400,
            )
        try:
            resolved = resolve_state_root(root)
            reject_if_under_forbidden_home_children(resolved)
        except WebStateConfigError as exc:
            raise FileAccessError(
                "invalid_root",
                "Registered root is not allowed.",
                status=400,
            ) from exc

        token = secrets.token_urlsafe(32)
        registered = RegisteredRoot(token=token, label=label, root=resolved)
        self._roots_by_token[token] = registered
        self._token_by_label[label] = token
        return registered

    def roots_for_api(self) -> list[dict[str, object]]:
        items = [
            {"token": entry.token, "label": entry.label, "read_only": True}
            for entry in sorted(self._roots_by_token.values(), key=lambda item: item.label.lower())
        ]
        return items

    def resolve(self, token: str, relative_path: str | None = None) -> Path:
        registered = self._get_registered_root(token)
        normalized = _normalize_relative_path(relative_path)
        _validate_relative_path(normalized)
        candidate = _join_under_root(registered.root, normalized)
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileAccessError(
                "path_not_found",
                "Path was not found.",
                status=404,
            ) from exc
        except OSError as exc:
            raise FileAccessError(
                "path_not_found",
                "Path was not found.",
                status=404,
            ) from exc

        if not is_same_or_descendant(resolved, registered.root):
            raise FileAccessError(
                "outside_root",
                "Path is outside the registered root.",
                status=403,
            )
        return resolved

    def stat(self, token: str, relative_path: str | None = None) -> dict[str, object]:
        registered = self._get_registered_root(token)
        normalized = _normalize_relative_path(relative_path)
        resolved = self.resolve(token, normalized)
        rel = _relative_path_for_response(registered.root, resolved, normalized)
        entry = _entry_metadata(resolved, registered.root, use_lstat=False)
        return {"root": token, "path": rel, "entry": entry}

    def list_dir(self, token: str, relative_path: str | None = None) -> dict[str, object]:
        registered = self._get_registered_root(token)
        normalized = _normalize_relative_path(relative_path)
        resolved = self.resolve(token, normalized)
        if not resolved.is_dir():
            raise FileAccessError(
                "not_a_directory",
                "Path is not a directory.",
                status=400,
            )
        rel = _relative_path_for_response(registered.root, resolved, normalized)
        entries = [_entry_metadata(child, registered.root, use_lstat=True) for child in resolved.iterdir()]
        entries.sort(key=_entry_sort_key)
        return {"root": token, "path": rel, "entries": entries}

    def _get_registered_root(self, token: str) -> RegisteredRoot:
        if token is None or str(token).strip() == "":
            raise FileAccessError(
                "missing_root",
                "Root token is required.",
                status=400,
            )
        registered = self._roots_by_token.get(token)
        if registered is None:
            raise FileAccessError(
                "unknown_root",
                "Root token is not recognized.",
                status=404,
            )
        return registered


def is_same_or_descendant(candidate: Path, root: Path) -> bool:
    """Return True if candidate is the same path as root or a descendant of root."""
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    if os.name == "nt":
        root_key = os.path.normcase(os.fspath(root_resolved))
        candidate_key = os.path.normcase(os.fspath(candidate_resolved))
        if candidate_key == root_key:
            return True
        try:
            Path(candidate_key).relative_to(Path(root_key))
        except ValueError:
            return False
        return True
    if candidate_resolved == root_resolved:
        return True
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        return False
    return True


def _validate_label(label: str) -> None:
    if label is None or str(label).strip() == "":
        raise FileAccessError(
            "invalid_label",
            "Root label is not allowed.",
            status=400,
        )
    if any(ord(ch) < 32 for ch in label):
        raise FileAccessError(
            "invalid_label",
            "Root label is not allowed.",
            status=400,
        )
    if any(sep in label for sep in ("/", "\\", ":")):
        raise FileAccessError(
            "invalid_label",
            "Root label is not allowed.",
            status=400,
        )
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.")
    if any(ch not in allowed for ch in label):
        raise FileAccessError(
            "invalid_label",
            "Root label is not allowed.",
            status=400,
        )


def _normalize_relative_path(relative_path: str | None) -> str:
    if relative_path is None:
        return ""
    return relative_path


def _validate_relative_path(relative_path: str) -> None:
    if relative_path in ("", "."):
        return

    if len(relative_path) > MAX_RELATIVE_PATH_LENGTH:
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )

    if "\x00" in relative_path or any(ord(ch) < 32 for ch in relative_path):
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )

    if ":" in relative_path:
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )

    if relative_path.startswith(("/", "\\")):
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )

    if relative_path.startswith("~"):
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )

    if _has_malformed_separator_segments(relative_path):
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )

    lowered = relative_path.lower()
    if lowered.startswith("file://") or lowered.startswith("\\\\?\\") or lowered.startswith("\\\\.\\"):
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )

    if relative_path.startswith("\\\\"):
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )

    windows_path = PureWindowsPath(relative_path)
    if windows_path.drive:
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )
    if windows_path.is_absolute():
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )

    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute():
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )

    parts = [part for part in re.split(r"[\\/]+", relative_path) if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise FileAccessError(
            "invalid_relative_path",
            "Relative path is not allowed.",
            status=400,
        )

    for part in parts:
        if _is_reserved_device_component(part):
            raise FileAccessError(
                "invalid_relative_path",
                "Relative path is not allowed.",
                status=400,
            )


def _is_reserved_device_component(component: str) -> bool:
    stem = component.split(".")[0].rstrip(" .").upper()
    return stem in _RESERVED_DEVICE_NAMES


def _has_malformed_separator_segments(relative_path: str) -> bool:
    return any(segment == "" for segment in re.split(r"[\\/]", relative_path))


def _join_under_root(root: Path, relative_path: str) -> Path:
    if relative_path in ("", "."):
        return root
    parts = [part for part in re.split(r"[\\/]+", relative_path) if part not in ("", ".")]
    candidate = root
    for part in parts:
        candidate = candidate / part
    return candidate


def _relative_path_for_response(root: Path, resolved: Path, normalized: str) -> str:
    if normalized in ("", "."):
        return ""
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        return normalized.replace("\\", "/")
    return rel.as_posix()


def _is_reparse_point(path: Path) -> bool:
    """True for Windows junctions/symlinks/reparse points (not just POSIX symlinks)."""
    try:
        attrs = path.lstat().st_file_attributes
    except (OSError, AttributeError):
        return False
    return bool(attrs & _stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _entry_metadata(path: Path, root: Path, *, use_lstat: bool) -> dict[str, Any]:
    name = path.name
    if use_lstat and (path.is_symlink() or _is_reparse_point(path)):
        try:
            target = path.resolve()
            if is_same_or_descendant(target, root):
                kind = "directory" if target.is_dir() else "file"
                stat_result = target.stat()
                entry: dict[str, Any] = {
                    "name": name,
                    "kind": kind,
                    "modified_time_utc": _format_mtime(stat_result.st_mtime),
                }
                if kind == "file" and target.is_file():
                    entry["size_bytes"] = stat_result.st_size
                return entry
        except OSError:
            pass
        return {"name": name, "kind": "other"}

    if path.is_dir():
        entry = {"name": name, "kind": "directory"}
        try:
            entry["modified_time_utc"] = _format_mtime(path.stat().st_mtime)
        except OSError:
            pass
        return entry

    if path.is_file():
        stat_result = path.stat()
        return {
            "name": name,
            "kind": "file",
            "size_bytes": stat_result.st_size,
            "modified_time_utc": _format_mtime(stat_result.st_mtime),
        }

    return {"name": name, "kind": "other"}


def _entry_sort_key(entry: dict[str, Any]) -> tuple[int, str, str]:
    kind_order = 0 if entry.get("kind") == "directory" else 1
    name = str(entry.get("name", ""))
    return (kind_order, name.casefold(), name)


def _format_mtime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def response_has_path_leak(body_text: str, *, root_path: Path | None = None) -> bool:
    """Test helper: detect absolute path leaks in JSON response bodies."""
    if _DRIVE_LEAK_PATTERN.search(body_text):
        return True
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if home and home in body_text:
        return True
    if root_path is not None and os.fspath(root_path) in body_text:
        return True
    return False
