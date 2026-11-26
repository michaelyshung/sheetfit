"""sheetfit desktop app: one editor per pipeline step.

    sheetfit-app [project.sheetfit | photos...]
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPainter
from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..core import IMAGE_EXTS
from .canvas import to_qimage
from .editors import EDITORS
from .model import Document

STATUS_TEXT = {"pending": "Waiting", "working": "Working…", "ready": "Ready", "photo": "Photo",
               "skipped": "Skipped", "check": "Check", "error": "Error"}


class PageStrip(QListWidget):
    """Thumbnails of every page along the bottom; drag to reorder."""

    THUMB = QSize(72, 94)

    def __init__(self, doc: Document, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.setViewMode(QListView.IconMode)
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(False)
        self.setIconSize(self.THUMB)
        self.setGridSize(QSize(100, 128))
        self.setMovement(QListView.Snap)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setFixedHeight(150)
        self.setHorizontalScrollMode(QListView.ScrollPerPixel)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.model().rowsMoved.connect(self._moved)
        self._rebuilding = False

    def rebuild(self):
        self._rebuilding = True
        cur = self.currentRow()
        self.clear()
        for i in range(len(self.doc.pages)):
            self.addItem(QListWidgetItem())
            self.update_page(i)
        if self.count():
            self.setCurrentRow(min(max(cur, 0), self.count() - 1))
        self._rebuilding = False

    def update_page(self, i: int):
        if i >= self.count():
            return
        st = self.doc.pages[i]
        it = self.item(i)
        mark = " •" if st.recipe.edited else ""
        it.setText(f"{i + 1} · {STATUS_TEXT.get(st.status, st.status)}{mark}")
        it.setToolTip(f"{st.name}\n{st.error}" if st.error else st.name)
        img = None
        if st.preview is not None:
            try:
                img = self.doc.stage(i, "final") if st.recipe.include else st.preview.rgb
            except Exception:
                img = st.preview.rgb
        if img is not None:
            t = img.copy()
            t.thumbnail((self.THUMB.width(), self.THUMB.height()))
            pad = Image.new("L" if t.mode == "L" else "RGB", (self.THUMB.width(), self.THUMB.height()), 255)
            pad.paste(t, ((pad.width - t.width) // 2, (pad.height - t.height) // 2))
            from PySide6.QtGui import QPixmap
            it.setIcon(QIcon(QPixmap.fromImage(to_qimage(pad))))

    def _moved(self, parent, start, end, dest, row):
        if self._rebuilding:
            return
        j = row if row < start else row - 1
        self.doc.move(start, j)
        self.setCurrentRow(j)

    def _menu(self, pos):
        it = self.itemAt(pos)
        if it is None:
            return
        i = self.row(it)
        st = self.doc.pages[i]
        m = QMenu(self)
        inc = m.addAction("Leave out of the PDF" if st.recipe.include else "Include in the PDF")
        redo = m.addAction("Redo all steps automatically")
        left = m.addAction("Move left")
        right = m.addAction("Move right")
        m.addSeparator()
        rm = m.addAction("Remove page")
        act = m.exec(self.mapToGlobal(pos))
        if act is inc:
            self.doc.update(i, "output", include=not st.recipe.include)
            st.status = "skipped" if not st.recipe.include else (
                "photo" if st.recipe.kind == "photo" else "ready")
            self.update_page(i)
        elif act is redo:
            self.doc.reanalyze(i)
        elif act is left and i > 0:
            self.doc.move(i, i - 1)
            self.setCurrentRow(i - 1)
        elif act is right and i + 1 < len(self.doc.pages):
            self.doc.move(i, i + 1)
            self.setCurrentRow(i + 1)
        elif act is rm:
            self.doc.remove(i)


class EmptyState(QWidget):
    def __init__(self, on_add):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        t = QLabel("Add photos of sheet music")
        t.setStyleSheet("font-size: 18px; font-weight: 500;")
        t.setAlignment(Qt.AlignCenter)
        b = QLabel("Drop photos or a folder anywhere in this window, or choose them.\n"
                   "Pages are straightened, flattened and cleaned automatically; "
                   "every step can then be adjusted.")
        b.setAlignment(Qt.AlignCenter)
        b.setStyleSheet("color: palette(placeholder-text);")
        btn = QPushButton("Choose photos…")
        btn.clicked.connect(on_add)
        lay.addWidget(t)
        lay.addWidget(b)
        lay.addSpacing(12)
        lay.addWidget(btn, alignment=Qt.AlignCenter)


class MainWindow(QMainWindow):
    def __init__(self, doc: Document | None = None):
        super().__init__()
        self.doc = doc or Document()
        self.setWindowTitle("sheetfit")
        self.resize(1280, 860)
        self.setAcceptDrops(True)

        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # top bar: file actions + step bar
        top = QWidget()
        th = QHBoxLayout(top)
        th.setContentsMargins(10, 8, 10, 8)
        self.step_group = QButtonGroup(self)
        for k, E in enumerate(EDITORS):
            b = QPushButton(f"{k + 1}  {E.title}")
            b.setCheckable(True)
            b.setFlat(True)
            self.step_group.addButton(b, k)
            th.addWidget(b)
        th.addStretch(1)
        for text, fn in (("Add photos…", self.add_photos), ("Open…", self.open_project),
                         ("Save", self.save_project)):
            th.addWidget(self._btn(text, fn))
        v.addWidget(top)
        v.addWidget(_hline())

        self.stack = QStackedWidget()
        self.empty = EmptyState(self.add_photos)
        self.stack.addWidget(self.empty)
        self.editors = [E(self.doc) for E in EDITORS]
        for e in self.editors:
            self.stack.addWidget(e)
        v.addWidget(self.stack, 1)
        v.addWidget(_hline())
        self.strip = PageStrip(self.doc)
        v.addWidget(self.strip)
        self.setCentralWidget(central)

        self.busy = QLabel()
        self.statusBar().addPermanentWidget(self.busy)

        self.step_group.idClicked.connect(self.show_step)
        self.strip.currentRowChanged.connect(self._page_selected)
        self.doc.pages_changed.connect(self._pages_changed)
        self.doc.page_changed.connect(self._page_changed)
        self.doc.busy.connect(self.busy.setText)

        for text, key, fn in (("Add photos", QKeySequence.Open, self.add_photos),
                              ("Save project", QKeySequence.Save, self.save_project),
                              ("Save project as", QKeySequence.SaveAs, lambda: self.save_project(True)),
                              ("Print", QKeySequence.Print, self.print_pages),
                              ("Export PDF", QKeySequence("Ctrl+E"), self.export_pdf)):
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(fn)
            self.addAction(a)
        self.step = 1
        self.step_group.button(self.step).setChecked(True)
        self._pages_changed()

    @staticmethod
    def _btn(text, fn):
        b = QPushButton(text)
        b.clicked.connect(fn)
        return b

    # ---- navigation
    def show_step(self, k: int):
        self.step = k
        self.step_group.button(k).setChecked(True)
        if not self.doc.pages:
            self.stack.setCurrentWidget(self.empty)
            return
        ed = self.editors[k]
        self.stack.setCurrentWidget(ed)
        ed.set_page(max(0, self.strip.currentRow()))

    def _page_selected(self, i: int):
        if i >= 0 and self.doc.pages:
            ed = self.stack.currentWidget()
            if ed in self.editors:
                ed.set_page(i)

    def _pages_changed(self):
        self.strip.rebuild()
        self.show_step(self.step)
        self._title()

    def _page_changed(self, i: int):
        self.strip.update_page(i)
        self._title()

    def _title(self):
        name = self.doc.path.name if self.doc.path else "Untitled"
        self.setWindowTitle(f"{name}{' *' if self.doc.dirty else ''} — sheetfit")

    # ---- files
    def add_photos(self):
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, "Add photos", "", f"Images ({exts})")
        if files:
            self.add_paths(files)

    def add_paths(self, paths: list[str]):
        files = []
        for p in map(Path, paths):
            if p.is_dir():
                files += sorted(str(q) for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
            elif p.suffix.lower() == ".sheetfit":
                return self.doc.open_project(str(p))
            elif p.suffix.lower() in IMAGE_EXTS:
                files.append(str(p))
        if files:
            first = not self.doc.pages
            self.doc.add_files(sorted(files) if first else files)

    def open_project(self):
        if not self._discard_ok():
            return
        f, _ = QFileDialog.getOpenFileName(self, "Open project", "", "sheetfit projects (*.sheetfit)")
        if f:
            self.doc.open_project(f)

    def save_project(self, ask: bool = False) -> bool:
        if not self.doc.pages:
            return False
        if ask or self.doc.path is None:
            f, _ = QFileDialog.getSaveFileName(self, "Save project", "untitled.sheetfit",
                                               "sheetfit projects (*.sheetfit)")
            if not f:
                return False
            if not f.endswith(".sheetfit"):
                f += ".sheetfit"
            self.doc.save_project(f)
        else:
            self.doc.save_project()
        self._title()
        self.statusBar().showMessage(f"Saved {self.doc.path.name}", 4000)
        return True

    def _discard_ok(self) -> bool:
        if not self.doc.dirty or not self.doc.pages:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save your changes to this project first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save_project()
        return r == QMessageBox.Discard

    def closeEvent(self, e):
        e.accept() if self._discard_ok() else e.ignore()

    # ---- output
    def _render_then(self, then):
        if not self.doc.included():
            QMessageBox.information(self, "Nothing to save", "Add some photos first.")
            return
        self.statusBar().showMessage("Rendering pages at full resolution…")
        self.doc.export(then, lambda msg: QMessageBox.warning(self, "Couldn't render", msg))

    def export_pdf(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save PDF", "sheets.pdf", "PDF (*.pdf)")
        if not f:
            return
        if not f.lower().endswith(".pdf"):
            f += ".pdf"
        dpi = self.doc.settings.dpi

        def save(pages):
            pages[0].save(f, "PDF", resolution=dpi, save_all=True, append_images=pages[1:])
            self.statusBar().showMessage(f"Saved {Path(f).name} ({len(pages)} pages)", 6000)

        self._render_then(save)

    def print_pages(self):
        printer = QPrinter(QPrinter.HighResolution)
        dlg = QPrintDialog(printer, self)
        if dlg.exec() != QPrintDialog.Accepted:
            return

        def send(pages):
            p = QPainter(printer)
            for k, page in enumerate(pages):
                if k:
                    printer.newPage()
                img = to_qimage(page)
                r = p.viewport()
                sz = img.size()
                sz.scale(r.size(), Qt.KeepAspectRatio)
                p.setViewport(r.x(), r.y(), sz.width(), sz.height())
                p.setWindow(img.rect())
                p.drawImage(0, 0, img)
                p.setViewport(r)
            p.end()
            self.statusBar().showMessage(f"Sent {len(pages)} pages to the printer", 6000)

        self._render_then(send)

    # ---- drag and drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color: palette(mid);")
    return f


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("sheetfit")
    app.setApplicationVersion(__version__)
    win = MainWindow()
    if len(argv) > 1:
        QTimer.singleShot(0, lambda: win.add_paths(argv[1:]))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
