"""The sheetfit desktop app (pip install "sheetfit[gui]", Python 3.12+)."""
import sys

if sys.version_info < (3, 12):
    raise ImportError("the sheetfit app needs Python 3.12 or newer "
                      "(the command-line tool works on 3.10+)")
