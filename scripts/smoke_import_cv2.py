"""
Minimal OpenCV smoke test for Nuitka packaging.

Verifies that cv2 loads without the recursive import error
that occurs when Nuitka bundles duplicate cv2.pyd files.

Usage (PowerShell):
    & 'C:\Python314\python.exe' -m nuitka `
      --standalone --assume-yes-for-downloads `
      --include-package=cv2 `
      --output-dir=dist_test `
      scripts/smoke_import_cv2.py

    & .\\dist_test\\smoke_import_cv2.dist\\smoke_import_cv2.exe
"""
import cv2

print(cv2.__file__)
print(cv2.__version__)
print("OK")
