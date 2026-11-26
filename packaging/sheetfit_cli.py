"""Entry point for the packaged command-line tool (PyInstaller)."""
import sys

from sheetfit.cli import main

if __name__ == "__main__":
    sys.exit(main())
