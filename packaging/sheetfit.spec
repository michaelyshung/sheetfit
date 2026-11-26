# PyInstaller spec: one folder holding both the app (sheetfit-app) and the
# command-line tool (sheetfit); on macOS it is wrapped as sheetfit.app.
#
#     pyinstaller --noconfirm packaging/sheetfit.spec
import sys
from pathlib import Path

HERE = Path(SPECPATH)
ICON = {"win32": "icon.ico", "darwin": "icon.icns"}.get(sys.platform)  # Linux: .desktop file
EXCLUDES = ["tkinter", "unittest", "pydoc", "pytest", "IPython", "matplotlib", "scipy"]

sys.path.insert(0, str(HERE.parent / "src"))
from sheetfit import __version__  # noqa: E402

app_a = Analysis([str(HERE / "sheetfit_app.py")], pathex=[str(HERE.parent / "src")],
                 datas=[(str(HERE.parent / "src/sheetfit/gui/icon.png"), "sheetfit/gui")],
                 excludes=EXCLUDES)
cli_a = Analysis([str(HERE / "sheetfit_cli.py")], pathex=[str(HERE.parent / "src")],
                 excludes=EXCLUDES + ["PySide6", "shiboken6"])

# Qt plugins the app doesn't use, and libraries only they need (saves ~90 MB).
DROP = (
    "platforminputcontexts/libqtvirtualkeyboard", "VirtualKeyboard", "Qt6Qml", "Qt6Quick",
    "imageformats/libqpdf", "imageformats/qpdf", "Qt6Pdf",
    "platformthemes/libqgtk3", "libgtk-3", "libgdk-3", "libatk", "libcairo", "libpango",
    "libepoxy", "libpixman", "libfribidi", "libdatrie", "libthai", "libgdk_pixbuf",
    "egldeviceintegrations", "Qt6EglFS", "Qt6EglFsKms", "Qt/translations/", "platforms/libqeglfs", "platforms/libqlinuxfb",
    "platforms/libqvnc", "platforms/libqvkkhrdisplay", "platforms/libqminimalegl",
    "networkinformation", "/generic/", "wayland-graphics-integration-client/libvulkan",
)


def keep(toc):
    return [e for e in toc if not any(d in e[0].replace("\\", "/") for d in DROP)]


app_a.binaries = keep(app_a.binaries)
app_a.datas = keep(app_a.datas)

app_exe = EXE(PYZ(app_a.pure), app_a.scripts, [], exclude_binaries=True,
              name="sheetfit-app", console=False, icon=ICON and str(HERE / ICON))
cli_exe = EXE(PYZ(cli_a.pure), cli_a.scripts, [], exclude_binaries=True,
              name="sheetfit", console=True, icon=ICON and str(HERE / ICON))

coll = COLLECT(app_exe, app_a.binaries, app_a.datas,
               cli_exe, cli_a.binaries, cli_a.datas,
               name="sheetfit")

if sys.platform == "darwin":
    app = BUNDLE(coll, name="sheetfit.app", icon=str(HERE / "icon.icns"),
                 bundle_identifier="io.github.michaelyshung.sheetfit",
                 version=__version__,
                 info_plist={
                     "CFBundleName": "sheetfit",
                     "CFBundleDisplayName": "sheetfit",
                     "CFBundleShortVersionString": __version__,
                     "NSHighResolutionCapable": True,
                     "LSMinimumSystemVersion": "11.0",
                     "CFBundleDocumentTypes": [{
                         "CFBundleTypeName": "Image",
                         "CFBundleTypeRole": "Viewer",
                         "LSHandlerRank": "Alternate",
                         "LSItemContentTypes": ["public.image"],
                     }],
                 })
