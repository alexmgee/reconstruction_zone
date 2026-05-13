"""
Minimal GUI import smoke test for Nuitka packaging.

This intentionally imports the app module without constructing the CTk root
window. It verifies that Nuitka can analyze the GUI import graph before we
spend time on a full executable build.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = ROOT / "reconstruction_gui"

for path in (ROOT, GUI_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import reconstruction_gui.reconstruction_zone as reconstruction_zone  # noqa: E402

print(reconstruction_zone.__version__)
print("GUI_IMPORT_OK")
