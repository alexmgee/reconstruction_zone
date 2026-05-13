"""
Model-path resolver smoke test.

Verifies that the shared model-path resolver finds all model weights
in the expected locations.

Usage (PowerShell):
    & 'C:\\Python314\\python.exe' scripts/smoke_model_paths.py
"""
import sys
from pathlib import Path

# Ensure reconstruction_gui is importable
_this = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_this / "reconstruction_gui"))
sys.path.insert(0, str(_this))

from reconstruction_gui.model_paths import (
    app_model_dir,
    resolve_rfdetr_seg_weights,
    resolve_yolo26_weights,
    resolve_sam3_weights,
    candidate_model_dirs,
)

print(f"model_dir: {app_model_dir(create=False)}")
print(f"candidates: {[str(d) for d in candidate_model_dirs()]}")
print(f"rfdetr: {resolve_rfdetr_seg_weights('small')}")
print(f"yolo26: {resolve_yolo26_weights('n')}")
print(f"sam3: {resolve_sam3_weights()}")
print("OK")
