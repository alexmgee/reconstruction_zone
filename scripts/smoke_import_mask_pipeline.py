"""Import smoke for the packaged Mask-tab pipeline path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "reconstruction_gui"

if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reconstruction_pipeline as rp  # noqa: E402

print("MASK_PIPELINE_IMPORT_OK")
print(
    json.dumps(
        {
            "HAS_SAM3": rp.HAS_SAM3,
            "HAS_FASTSAM": rp.HAS_FASTSAM,
            "HAS_YOLO": rp.HAS_YOLO,
            "HAS_RFDETR": rp.HAS_RFDETR,
        },
        sort_keys=True,
    )
)
