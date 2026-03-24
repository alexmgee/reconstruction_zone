import sys

if __name__ == "__main__":
    from prep360.distribution import is_gumroad
    if is_gumroad():
        print("CLI is not available in this build. Use the GUI instead.", file=sys.stderr)
        sys.exit(1)
    from .cli import main
    sys.exit(main())
