"""Fail-closed state-root resolution for the local web server."""

from __future__ import annotations

import os
import string
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

__all__ = [
    "WebStateConfig",
    "WebStateConfigError",
    "build_state_config",
    "reject_if_under_forbidden_home_children",
    "resolve_state_root",
]

DOT_RECONSTRUCTION_ZONE = ".reconstruction_zone"
DOT_PREP360_MODELS = ".prep360_models"
DOT_PREP360_THUMB_CACHE = ".prep360_thumb_cache"


class WebStateConfigError(ValueError):
    """Raised when explicit web state configuration is missing or unsafe."""


@dataclass(frozen=True)
class WebStateConfig:
    state_root: Path


def resolve_state_root(explicit_state_root: str | Path | None = None) -> Path:
    """Resolve and validate an explicit non-production state root."""
    if explicit_state_root is None or str(explicit_state_root).strip() == "":
        raise WebStateConfigError("Missing explicit non-production state root.")

    raw = Path(explicit_state_root)
    _reject_drive_relative(raw)
    _reject_unc(raw)

    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise WebStateConfigError("State root does not exist.") from exc

    if not resolved.is_dir():
        raise WebStateConfigError("State root must be an existing directory.")

    _reject_unc(resolved)

    _reject_unsafe_root(resolved)
    return resolved


def build_state_config(explicit_state_root: str | Path | None = None) -> WebStateConfig:
    return WebStateConfig(state_root=resolve_state_root(explicit_state_root))


def reject_if_under_forbidden_home_children(resolved: Path) -> None:
    """Reject paths inside production home child directories (exact or descendant)."""
    home = _user_home()
    if home is None:
        return
    for forbidden_child in _forbidden_home_children(home):
        if _is_same_or_descendant(resolved, forbidden_child):
            raise WebStateConfigError("State root is not allowed.")


def _reject_unc(path: Path) -> None:
    """Reject UNC / device-prefixed roots structurally, before any existence check.

    Windows-guarded to match _reject_drive_relative and avoid rejecting legitimate
    POSIX filenames that may contain backslashes.
    """
    if os.name != "nt":
        return
    text = os.fspath(path)
    if text.startswith("\\\\") or text.startswith("//"):
        raise WebStateConfigError("State root is not allowed.")
    if PureWindowsPath(text).drive.startswith("\\\\"):
        raise WebStateConfigError("State root is not allowed.")


def _reject_drive_relative(path: Path) -> None:
    if os.name != "nt":
        return
    text = os.fspath(path)
    if len(text) >= 2 and text[1] == ":":
        if len(text) == 2 or text[2] not in ("\\", "/"):
            raise WebStateConfigError("Drive-relative paths are not allowed.")


def _reject_unsafe_root(resolved: Path) -> None:
    if _is_filesystem_root(resolved):
        raise WebStateConfigError("State root is too broad.")

    for forbidden in _forbidden_exact_paths():
        if resolved == forbidden:
            raise WebStateConfigError("State root is not allowed.")

    for anchor in _forbidden_workspace_anchors():
        if _is_same_or_descendant(resolved, anchor):
            raise WebStateConfigError("State root is not allowed.")

    home = _user_home()
    if home is not None:
        if resolved == home:
            raise WebStateConfigError("State root is not allowed.")
        for forbidden_child in _forbidden_home_children(home):
            if resolved == forbidden_child:
                raise WebStateConfigError("State root is not allowed.")


def _is_same_or_descendant(path: Path, anchor: Path) -> bool:
    try:
        anchor_resolved = anchor.resolve()
    except OSError:
        anchor_resolved = anchor
    if path == anchor_resolved:
        return True
    try:
        path.relative_to(anchor_resolved)
    except ValueError:
        return False
    return True


def _forbidden_workspace_anchors() -> tuple[Path, ...]:
    """Source-code workspace roots whose descendants must not be used as state roots."""
    anchors: list[Path] = []
    for drive_letter in string.ascii_uppercase:
        projects = Path(f"{drive_letter}:{os.sep}") / "Projects"
        anchors.append(projects / "reconstruction-zone")
        anchors.append(projects / "reconstruction-zone-web")
    return tuple(anchors)


def _is_filesystem_root(path: Path) -> bool:
    if path.parent == path:
        return True
    anchor = path.anchor
    if not anchor:
        return False
    try:
        relative = path.relative_to(anchor)
    except ValueError:
        return False
    return len(relative.parts) == 0


def _user_home() -> Path | None:
    profile = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not profile:
        return None
    return Path(profile).resolve()


def _forbidden_home_children(home: Path) -> tuple[Path, ...]:
    return (
        home / DOT_RECONSTRUCTION_ZONE,
        home / DOT_PREP360_MODELS,
        home / DOT_PREP360_THUMB_CACHE,
    )


def _forbidden_exact_paths() -> frozenset[Path]:
    forbidden: set[Path] = set()
    for drive_letter in string.ascii_uppercase:
        drive_root = Path(f"{drive_letter}:{os.sep}")
        if not drive_root.exists() and os.name == "nt":
            # Still reject even if the drive is absent.
            forbidden.add(drive_root.resolve())
        elif drive_root.exists():
            drive_root = drive_root.resolve()
            forbidden.add(drive_root)
            forbidden.add(drive_root / "Projects")
            forbidden.add(drive_root / "Capture")

    return frozenset(forbidden)
