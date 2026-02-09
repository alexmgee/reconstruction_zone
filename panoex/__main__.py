"""
Allow running panoex as a module: python -m panoex
"""
from .cli import main

if __name__ == "__main__":
    exit(main())
