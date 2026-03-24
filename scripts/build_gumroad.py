"""
Build script for Gumroad distribution of Reconstruction Zone.

Temporarily sets DISTRIBUTION = "gumroad" in prep360/distribution.py,
runs PyInstaller with the project spec file, then restores the original
distribution flag regardless of build outcome.

Usage:
    python build/build_gumroad.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    dist_file = ROOT / "prep360" / "distribution.py"

    if not dist_file.exists():
        print(f"ERROR: {dist_file} not found")
        return 1

    original = dist_file.read_text(encoding="utf-8")

    if 'DISTRIBUTION = "github"' not in original:
        print(f"WARNING: Expected DISTRIBUTION = \"github\" in {dist_file}")
        print("         File may already be set to gumroad or has unexpected content.")
        print("         Proceeding anyway...")

    try:
        # Swap to gumroad distribution
        modified = original.replace(
            'DISTRIBUTION = "github"',
            'DISTRIBUTION = "gumroad"',
        )
        dist_file.write_text(modified, encoding="utf-8")
        print("Set DISTRIBUTION = \"gumroad\"")

        # Run PyInstaller
        spec_file = ROOT / "reconstruction_zone.spec"
        if not spec_file.exists():
            print(f"ERROR: {spec_file} not found")
            return 1

        print(f"Running PyInstaller with {spec_file.name}...")
        result = subprocess.run(
            [
                sys.executable, "-m", "PyInstaller",
                str(spec_file),
                "--noconfirm",
            ],
            cwd=str(ROOT),
        )

        if result.returncode == 0:
            print("\nBuild succeeded!")
            print(f"Output: {ROOT / 'dist' / 'ReconstructionZone'}")
        else:
            print(f"\nBuild failed with exit code {result.returncode}")

        return result.returncode

    finally:
        # Always restore original distribution flag
        dist_file.write_text(original, encoding="utf-8")
        print("Restored DISTRIBUTION = \"github\"")


if __name__ == "__main__":
    sys.exit(main())
