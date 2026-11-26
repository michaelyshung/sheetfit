#!/usr/bin/env bash
# Wrap dist/sheetfit.app (from pyinstaller) into a drag-to-Applications .dmg.
# Usage: packaging/build_dmg.sh   (run from the repo root, on macOS)
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=$(python3 -c "import re;print(re.search(r'__version__ = \"(.+?)\"', open('src/sheetfit/__init__.py').read())[1])")
ARCH=$(uname -m)                      # arm64 or x86_64
STAGE=build/dmg
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -R dist/sheetfit.app "$STAGE/"
ln -s /Applications "$STAGE/Applications"
# not notarized: ad-hoc sign the whole bundle so it runs on Apple silicon
codesign --force --deep --sign - "$STAGE/sheetfit.app"
hdiutil create -volname "sheetfit" -srcfolder "$STAGE" -ov -format UDZO \
    "dist/sheetfit-${VERSION}-macos-${ARCH}.dmg"
