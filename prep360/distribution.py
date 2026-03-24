"""
Build distribution flag.

Set DISTRIBUTION to "gumroad" in the build script before PyInstaller.
Default is "github" (open source, all models available).
"""

DISTRIBUTION = "github"  # "gumroad" or "github"


def is_gumroad() -> bool:
    """True when running the paid Gumroad build."""
    return DISTRIBUTION == "gumroad"
