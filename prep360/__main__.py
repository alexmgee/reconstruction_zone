"""
Allow running prep360 as a module: python -m prep360
"""
from .cli import main

if __name__ == "__main__":
    exit(main())
