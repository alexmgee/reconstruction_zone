"""
Verify a Gumroad build contains no AGPL-licensed code.

Scans the PyInstaller dist/ output for ultralytics/YOLO references
that should have been stripped by the _is_gumroad() gate.

Usage:
    python scripts/verify_gumroad_build.py [dist_path]
"""

import sys
from pathlib import Path

AGPL_MARKERS = [
    b"ultralytics",
    b"from ultralytics",
    b"import ultralytics",
    b"AGPL",
]

# Files that legitimately mention ultralytics in comments/strings (not imports)
ALLOWLIST = {
    "reconstruction_pipeline.pyc",  # Has "YOLO excluded from Gumroad build" string
}


def scan_directory(dist_path: Path) -> list[tuple[Path, str]]:
    """Scan all .pyc and .pyd files for AGPL markers."""
    violations = []

    for f in dist_path.rglob("*"):
        if f.suffix not in (".pyc", ".pyd", ".py"):
            continue
        if f.name in ALLOWLIST:
            continue

        try:
            content = f.read_bytes()
        except (PermissionError, OSError):
            continue

        for marker in AGPL_MARKERS:
            if marker in content:
                violations.append((f.relative_to(dist_path), marker.decode()))
                break

    return violations


def main():
    root = Path(__file__).resolve().parent.parent
    default_dist = root / "dist" / "ReconstructionZone"

    dist_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dist

    if not dist_path.exists():
        print(f"ERROR: dist path not found: {dist_path}")
        return 1

    print(f"Scanning: {dist_path}")
    violations = scan_directory(dist_path)

    if violations:
        print(f"\nFAILED — {len(violations)} AGPL violation(s) found:\n")
        for path, marker in violations:
            print(f"  {path}  (matched: {marker!r})")
        return 1
    else:
        print("PASSED — no AGPL markers found in build output.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
