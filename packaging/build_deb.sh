#!/usr/bin/env bash
# Wrap the PyInstaller folder (dist/sheetfit) into a .deb:
#   /opt/sheetfit/...            the app and the command-line tool
#   /usr/bin/sheetfit(-app)      symlinks
#   menu entry + icon
# Usage: packaging/build_deb.sh  (run from the repo root after pyinstaller)
set -euo pipefail
umask 022
cd "$(dirname "$0")/.."
VERSION=$(python3 -c "import re;print(re.search(r'__version__ = \"(.+?)\"', open('src/sheetfit/__init__.py').read())[1])")
ARCH=$(dpkg --print-architecture)
PKG=build/deb/sheetfit_${VERSION}_${ARCH}
rm -rf "$PKG"
mkdir -p "$PKG/DEBIAN" "$PKG/opt" "$PKG/usr/bin" "$PKG/usr/share/applications" \
         "$PKG/usr/share/icons/hicolor/256x256/apps" "$PKG/usr/share/icons/hicolor/512x512/apps" \
         "$PKG/usr/share/doc/sheetfit"
cp -a dist/sheetfit "$PKG/opt/sheetfit"
ln -s /opt/sheetfit/sheetfit "$PKG/usr/bin/sheetfit"
ln -s /opt/sheetfit/sheetfit-app "$PKG/usr/bin/sheetfit-app"
python3 - "$PKG" <<'PY'
import sys
from PIL import Image
pkg = sys.argv[1]
im = Image.open("packaging/icon.png")
for n in (256, 512):
    im.resize((n, n), Image.LANCZOS).save(f"{pkg}/usr/share/icons/hicolor/{n}x{n}/apps/sheetfit.png")
PY
cat > "$PKG/usr/share/applications/sheetfit.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=sheetfit
GenericName=Sheet music cleanup
Comment=Straighten, flatten and clean photos of sheet music for printing
Exec=sheetfit-app %F
Icon=sheetfit
Terminal=false
Categories=Graphics;
MimeType=image/jpeg;image/png;image/tiff;image/webp;image/heic;image/heif;
StartupWMClass=sheetfit
DESK
cp LICENSE "$PKG/usr/share/doc/sheetfit/copyright"
SIZE=$(du -sk "$PKG/opt" | cut -f1)
cat > "$PKG/DEBIAN/control" <<CTRL
Package: sheetfit
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: michaelyshung <noreply@github.com>
Installed-Size: ${SIZE}
Depends: libc6 (>= 2.35), libegl1, libgl1, libfontconfig1, libxcb1, libxcb-cursor0, libxkbcommon-x11-0, libwayland-client0, libwayland-cursor0, libwayland-egl1
Section: graphics
Priority: optional
Homepage: https://github.com/michaelyshung/sheetfit
Description: straighten, flatten and clean photos of sheet music for printing
 sheetfit turns phone photos of sheet music (including pages photographed
 out of a bound book) into straight, flat, high-contrast pages fitted to
 Letter or A4. Includes the desktop app (sheetfit-app) and the
 command-line tool (sheetfit).
CTRL
chmod -R u=rwX,go=rX "$PKG"
mkdir -p dist
dpkg-deb --root-owner-group -Zxz --build "$PKG" "dist/sheetfit_${VERSION}_${ARCH}.deb"
