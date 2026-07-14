"""
Shared model-path resolver for Reconstruction Zone.

Single source of truth for where model weight files live and where
runtime code looks for them. Used by both the setup wizard (to check
readiness and download weights) and the masking pipeline (to load
models at inference time).

Model files are stored in a user-writable app directory:
    Windows:  %LOCALAPPDATA%\\ReconstructionZone\\models
    Fallback: ~/.reconstruction_zone/models

Override with RECONSTRUCTION_ZONE_MODEL_DIR environment variable.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from app_paths import bundled_model_dirs, model_dir
except ImportError:
    from reconstruction_gui.app_paths import bundled_model_dirs, model_dir

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Weight file names — must match what each library's runtime constructor expects
# ═══════════════════════════════════════════════════════════════════════════════

YOLO26_WEIGHT_NAMES: Dict[str, str] = {
    "n": "yolo26n-seg.pt",
    "s": "yolo26s-seg.pt",
    "m": "yolo26m-seg.pt",
    "l": "yolo26l-seg.pt",
    "x": "yolo26x-seg.pt",
}

RFDETR_SEG_WEIGHT_NAMES: Dict[str, str] = {
    "nano": "rf-detr-seg-nano.pt",
    "small": "rf-detr-seg-small.pt",
    "medium": "rf-detr-seg-medium.pt",
    "large": "rf-detr-seg-large.pt",
    "xlarge": "rf-detr-seg-xlarge.pt",
    "2xlarge": "rf-detr-seg-xxlarge.pt",
}

RFDETR_SEG_URLS: Dict[str, str] = {
    "nano": "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth",
    "small": "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
    "medium": "https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
    "large": "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth",
    "xlarge": "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth",
    "2xlarge": "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth",
}

SAM3_MODEL_ID = "facebook/sam3"
SAM3_SENTINEL_FILE = "config.json"


# ═══════════════════════════════════════════════════════════════════════════════
# App model directory
# ═══════════════════════════════════════════════════════════════════════════════

def app_model_dir(create: bool = False) -> Path:
    """Return the app's model storage directory.

    Priority is delegated to app_paths.model_dir().
    """
    return model_dir(create=create)


def candidate_model_dirs() -> List[Path]:
    """Return all directories to search for model files, in priority order."""
    dirs = [app_model_dir()]
    strict_model_dirs = os.environ.get("RECONSTRUCTION_ZONE_STRICT_MODEL_DIRS") == "1"

    # Intentional packaged locations: a bundled "models" directory first, then
    # the executable directory for local testing.
    if getattr(sys, 'frozen', False):
        dirs.extend(bundled_model_dirs())

    # Development fallbacks are useful for source checkouts, but release/new-
    # install tests must be able to disable them so repo-root weights cannot
    # mask missing app-home models.
    if not strict_model_dirs and not getattr(sys, 'frozen', False):
        dirs.append(Path.cwd())
        try:
            project_root = Path(__file__).resolve().parent.parent
            dirs.append(project_root)
            dirs.append(project_root / "models")
        except Exception:
            pass

    # Ultralytics cache (legacy YOLO downloads)
    ul_cache = Path(os.path.expanduser("~/.cache/ultralytics"))
    if ul_cache.exists():
        dirs.append(ul_cache)

    return dirs


# ═══════════════════════════════════════════════════════════════════════════════
# File resolution
# ═══════════════════════════════════════════════════════════════════════════════

def find_model_file(names: Iterable[str]) -> Optional[Path]:
    """Search candidate directories for any file matching the given names.

    Returns the first match found, or None.
    """
    dirs = candidate_model_dirs()
    for name in names:
        for d in dirs:
            candidate = d / name
            if candidate.is_file():
                return candidate
    return None


def find_hf_cached_file(repo_id: str, filename: str) -> Optional[Path]:
    """Check if a file exists in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(repo_id, filename)
        if result is not None and isinstance(result, str):
            return Path(result)
    except Exception:
        pass
    return None


def resolve_yolo26_weights(size: str = "n") -> Optional[Path]:
    """Find YOLO26 weights for the given size.

    Search order:
        1. App model dir and other candidate dirs (by runtime filename)
        2. HuggingFace cache (openvision/yolo26-{size}-seg)
    """
    name = YOLO26_WEIGHT_NAMES.get(size)
    if not name:
        return None

    # Check candidate directories
    found = find_model_file([name])
    if found:
        return found

    # Check HF cache
    hf_repo = f"openvision/yolo26-{size}-seg"
    return find_hf_cached_file(hf_repo, "model.pt")


def resolve_rfdetr_seg_weights(size: str = "small") -> Optional[Path]:
    """Find RF-DETR-Seg weights for the given size.

    Search order:
        1. App model dir and other candidate dirs (by runtime filename)
        2. rfdetr's own download name (same filename, different location)
    """
    name = RFDETR_SEG_WEIGHT_NAMES.get(size)
    if not name:
        return None

    return find_model_file([name])


def resolve_sam3_weights() -> Optional[Path]:
    """Check if SAM3 weights exist in the HuggingFace cache."""
    return find_hf_cached_file(SAM3_MODEL_ID, SAM3_SENTINEL_FILE)
