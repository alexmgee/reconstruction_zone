r"""
Minimal RF-DETR import smoke test for Phase 6B packaging decisions.

This deliberately stops short of constructing a model instance, because model
construction can download weights or touch a much broader runtime surface. The
purpose here is narrower:

1. Verify the installed ``rfdetr`` package imports.
2. Verify the four RF-DETR segmentation classes used by the app are present.
3. Print the eager ``transformers.models.*`` footprint so packaged runs can be
   compared against source runs.

Usage (PowerShell, source run):
    & 'C:\Python314\python.exe' scripts\smoke_import_rfdetr.py

Usage (PowerShell, bounded Nuitka packaging experiment):
    & 'C:\Python314\python.exe' -m nuitka `
      --standalone `
      --assume-yes-for-downloads `
      --windows-console-mode=attach `
      --include-package=rfdetr `
      --include-package=transformers `
      --include-package=torch `
      --include-package=torchvision `
      --user-package-configuration-file=scripts/nuitka-torch-fix.yml `
      --module-parameter=torch-disable-jit=yes `
      --output-dir=dist_test `
      scripts\smoke_import_rfdetr.py

Then run:
    & .\dist_test\smoke_import_rfdetr.dist\smoke_import_rfdetr.exe
"""
from __future__ import annotations

import importlib.metadata as metadata
import sys

import rfdetr


EXPECTED_SEG_CLASSES = (
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
)


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "UNKNOWN"


def main() -> int:
    missing = [name for name in EXPECTED_SEG_CLASSES if not hasattr(rfdetr, name)]
    if missing:
        print("RFDETR_CLASS_MISSING", ",".join(missing))
        return 1

    transformer_model_modules = sorted(
        name for name in sys.modules if name.startswith("transformers.models.")
    )

    print("RFDETR_VERSION", _distribution_version("rfdetr"))
    print("TRANSFORMERS_VERSION", _distribution_version("transformers"))
    print("RFDETR_CLASSES_OK", ",".join(EXPECTED_SEG_CLASSES))
    print("TRANSFORMERS_MODEL_MODULE_COUNT", len(transformer_model_modules))
    for module_name in transformer_model_modules:
        print("TRANSFORMERS_MODEL_MODULE", module_name)
    print("RFDETR_IMPORT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
