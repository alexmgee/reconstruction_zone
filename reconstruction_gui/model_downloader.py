"""
Compatibility stub for the removed first-launch model downloader.

The active first-launch flow lives in setup_wizard.py and uses model_paths.py
for all model readiness and download paths. This module intentionally imports
no model libraries, OpenCV, rfdetr, or ultralytics so stale packaging inputs do
not pull obsolete downloader behavior into Nuitka/PyInstaller analysis.
"""
from __future__ import annotations

from typing import List, Tuple


def check_missing_models() -> List[Tuple[str, str, int]]:
    """Deprecated compatibility API.

    The setup wizard owns model checks now. Returning an empty list keeps old
    callers from triggering obsolete download behavior.
    """
    return []


class ModelDownloadDialog:
    """Deprecated compatibility placeholder.

    Kept only so stale imports fail softly. New code must use
    setup_wizard.run_setup_wizard_if_needed().
    """

    success = False

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "model_downloader.py is deprecated; use setup_wizard.py instead"
        )
