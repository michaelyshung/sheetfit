"""Entry point for the packaged app (PyInstaller)."""
import sys

from sheetfit.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
