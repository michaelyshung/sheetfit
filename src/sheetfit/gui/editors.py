"""One editor per pipeline step. Each shows the page at that step on a
canvas, with handles for the step's settings, and a side panel."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QKeySequence, QPainterPath, QPainterPathStroker, QPen, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..pipeline import box_to_abs
from ..recipe import Guide
from ..warp import _fit_cubic, bezier_sample, ensure_bezier, resolve, split_bezier
from .canvas import ACCENT, CORAL, PURPLE, TEAL, Canvas, Handle, dashed, solid, to_pixmap
from .model import Document, new_guide_at

SIDE_W = 260


# ---------------------------------------------------------------- side-panel bits
def heading(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setStyleSheet("color: palette(placeholder-text); font-size: 11px; margin-top: 10px;")
    return lb


def note(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setStyleSheet("color: palette(placeholder-text); font-size: 11px; margin-top: 8px;")
    return lb


class Slider(QWidget):
    """Float slider with a value readout. Applies live: on_change(v) and
    then on_commit(v) run at every increment while dragging (undo groups
    a whole drag into one step)."""

    def __init__(self, lo: float, hi: float, step: float, fmt: str, on_change=None, on_commit=None):
        super().__init__()
        self.step, self.fmt = step, fmt
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.s = QSlider(Qt.Horizontal)
        self.s.setRange(round(lo / step), round(hi / step))
        self.val = QLabel()
        self.val.setMinimumWidth(52)
        self.val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.s, 1)
        lay.addWidget(self.val)
        self._quiet = False
        self.on_change, self.on_commit = on_change, on_commit
        self.s.valueChanged.connect(self._changed)

    def value(self) -> float:
        return self.s.value() * self.step

    def set(self, v: float) -> None:
        self._quiet = True
        self.s.setValue(round(v / self.step))
        self.val.setText(self.fmt.format(v))
        self._quiet = False

    def _changed(self, _):
        v = self.value()
        self.val.setText(self.fmt.format(v))
        if self._quiet:
            return
        if self.on_change:
            self.on_change(v)
        if self.on_commit:
            self.on_commit(v)


class Segmented(QWidget):
    """A row of mutually exclusive buttons."""

    def __init__(self, labels: list[str], on_pick=None, current: int = 0):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        for k, t in enumerate(labels):
            b = QPushButton(t)
            b.setCheckable(True)
            b.setChecked(k == current)
            self.group.addButton(b, k)
            lay.addWidget(b)
        if on_pick:
            self.group.idClicked.connect(on_pick)

    def current(self) -> int:
        return self.group.checkedId()

    def set(self, k: int) -> None:
        self.group.button(k).setChecked(True)


def button(text: str, fn) -> QPushButton:
    b = QPushButton(text)
    b.clicked.connect(fn)
    return b


def row(*widgets) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x)
    return w


# ---------------------------------------------------------------- base
class Editor(QWidget):
    step = ""
    title = ""
    has_auto = True
    has_preview = True
    PREVIEW_DPI = 60

    def __init__(self, doc: Document):
        super().__init__()
        self.doc = doc
        self.i = -1
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.canvas = Canvas()
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        self.split = split
        self._sized = False
        split.addWidget(self.canvas)
        if self.has_preview:
            pv = QWidget()
            pl = QVBoxLayout(pv)
            pl.setContentsMargins(0, 6, 0, 0)
            cap = QLabel("  Result on paper")
            cap.setStyleSheet("color: palette(placeholder-text); font-size: 11px;")
            pl.addWidget(cap)
            self.result_view = Canvas()
            pl.addWidget(self.result_view, 1)
            split.addWidget(pv)
            pv.setMinimumWidth(180)
            split.setStretchFactor(0, 3)
            split.setStretchFactor(1, 2)
            self._ptimer = QTimer(self)
            self._ptimer.setSingleShot(True)
            self._ptimer.setInterval(150)
            self._ptimer.timeout.connect(self._render_preview)
        lay.addWidget(split, 1)
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("color: palette(mid);")
        lay.addWidget(line)
        side = QWidget()
        side.setFixedWidth(SIDE_W)
        self.form = QVBoxLayout(side)
        self.form.setContentsMargins(14, 10, 14, 14)
        self.form.setAlignment(Qt.AlignTop)
        self.status = QLabel()
        self.status.setStyleSheet("font-weight: 500;")
        self.form.addWidget(self.status)
        lay.addWidget(side)
        self.build()
        if self.has_auto:
            self.form.addSpacing(10)
            self.form.addWidget(button("Auto (undo my changes)", self.auto))
        self.form.addStretch(1)
        doc.page_changed.connect(self._page_changed)
        doc.settings_changed.connect(lambda: self.isVisible() and self.refresh())
        if self.has_preview:
            # other pages' edits can change this page's size (same staff size)
            doc.page_changed.connect(lambda _: self.isVisible() and self._ptimer.start())

    def showEvent(self, e):
        super().showEvent(e)
        if self.has_preview and not self._sized and self.split.width() > 0:
            w = self.split.width()
            self.split.setSizes([int(w * 0.6), w - int(w * 0.6)])
            self._sized = True

    # subclasses
    def build(self) -> None: ...
    def draw(self, fit: bool) -> None: ...
    def sync_side(self) -> None: ...

    # common
    @property
    def st(self):
        return self.doc.pages[self.i] if 0 <= self.i < len(self.doc.pages) else None

    @property
    def ready(self) -> bool:
        st = self.st
        return st is not None and st.preview is not None

    def set_page(self, i: int) -> None:
        self.i = i
        self.refresh(fit=True)

    def _page_changed(self, i: int) -> None:
        if i == self.i and self.isVisible():
            self.refresh()

    def refresh(self, fit: bool = False) -> None:
        self.canvas.clear_overlay()
        st = self.st
        if st is None:
            self.canvas.set_image(None)
            self.status.setText("No page selected")
            return
        if st.preview is None:
            self.canvas.set_image(None)
            self.status.setText("Error: " + st.error if st.status == "error" else "Working on this page…")
            return
        edited = self.step in st.recipe.edited
        self.status.setText(f"Page {self.i + 1} of {len(self.doc.pages)}"
                            + ("  ·  edited" if edited else "  ·  auto"))
        self.sync_side()
        self.draw(fit)
        if self.has_preview:
            self._ptimer.start()

    def _render_preview(self) -> None:
        """The finished page on paper, beside every editor."""
        st = self.st
        if st is None or st.preview is None:
            self.result_view.set_image(None)
            return
        lay = self.doc.paper_layout(self.PREVIEW_DPI)
        if self.i not in lay:
            self.result_view.set_image(None)
            return
        img, pw, ph, x, y, nw, nh = lay[self.i]
        page = Image.new("L", (pw, ph), 255)
        page.paste(img.resize((nw, nh), Image.LANCZOS), (x, y))
        s = self.doc.settings
        if s.bw and st.recipe.kind == "music":
            page = page.point(lambda v: 255 if v > s.bw_threshold else 0)
        self.result_view.set_image(page)
        self.result_view.clear_overlay()
        self.result_view.scene().addItem(_rect(QRectF(0, 0, pw, ph), solid(QColor(0, 0, 0, 60), 1)))

    def auto(self) -> None:
        if self.ready:
            self.doc.reset_step(self.i, self.step)

    # frame helpers: recipe (full-res, centre-relative) <-> scene (preview px)
    def to_scene(self, img: Image.Image, x: float, y: float) -> QPointF:
        f = self.st.f
        return QPointF(x * f + img.width / 2, y * f + img.height / 2)

    def from_scene(self, img: Image.Image, p: QPointF) -> tuple[float, float]:
        f = self.st.f
        return (p.x() - img.width / 2) / f, (p.y() - img.height / 2) / f


# ---------------------------------------------------------------- 1 clean
class CleanEditor(Editor):
    step, title, has_auto = "clean", "Clean", False

    def build(self):
        self.form.addWidget(heading("Show"))
        self.view = Segmented(["Original", "Cleaned"], lambda _: self.refresh(), current=1)
        self.form.addWidget(self.view)
        self.form.addWidget(heading("For all pages"))
        self.keep = QCheckBox("Keep coloured markings")
        self.light = QCheckBox("Even out lighting")
        self.edges = QCheckBox("Remove page edges and shadows")
        for cb, key in ((self.keep, "keep_color_marks"), (self.light, "flatten_lighting"),
                        (self.edges, "clean_edges")):
            cb.toggled.connect(lambda v, key=key: self._set(key, v))
            self.form.addWidget(cb)
        self.form.addWidget(note("Colour is normally removed so tinted paper, highlighter and "
                                 "coloured banners print white. Keep coloured markings keeps "
                                 "pencil or pen annotations (as grey) instead."))

    def _set(self, key, v):
        if self._quiet or getattr(self.doc.settings, key) == v:
            return
        self.doc.set_settings(reclean=True, **{key: v})

    _quiet = False

    def sync_side(self):
        self._quiet = True
        s = self.doc.settings
        self.keep.setChecked(s.keep_color_marks)
        self.light.setChecked(s.flatten_lighting)
        self.edges.setChecked(s.clean_edges)
        self._quiet = False

    def draw(self, fit):
        p = self.st.preview
        self.canvas.set_image(p.rgb if self.view.current() == 0 else p.clean, keep_view=not fit)


# ---------------------------------------------------------------- 2 straighten
class StraightenEditor(Editor):
    step, title = "straighten", "Straighten"

    def build(self):
        self.form.addWidget(heading("Angle"))
        self.angle = Slider(-10, 10, 0.01, "{:+.2f}°", on_commit=self._set_angle)
        self.form.addWidget(self.angle)
        self.form.addWidget(row(button("↺ 90°", lambda: self._quarter(-1)),
                                button("↻ 90°", lambda: self._quarter(1))))
        self.grid = QCheckBox("Grid overlay")
        self.grid.setChecked(True)
        self.grid.toggled.connect(lambda _: self.refresh())
        self.form.addWidget(self.grid)
        self.form.addWidget(note("Drag the ends of the blue guide onto any staff line. "
                                 "The page turns to make that line level."))
        self._guide_y = 0.5

    def _set_angle(self, v):
        if self.ready and abs(v - self.st.recipe.angle) > 1e-6:
            self.doc.update(self.i, self.step, angle=round(v, 2))

    def _quarter(self, d):
        if self.st:
            self.doc.reanalyze(self.i, (self.st.recipe.rotate90 + d) % 4)
            self.refresh()

    def sync_side(self):
        self.angle.set(self.st.recipe.angle)

    def draw(self, fit):
        img = self.doc.stage(self.i, "straight")
        self.canvas.set_image(img, keep_view=not fit)
        w, h = img.size
        sc = self.canvas.scene()
        if self.grid.isChecked():
            pen = solid(QColor(55, 138, 221, 70), 1)
            for k in range(1, 16):
                sc.addLine(0, h * k / 16, w, h * k / 16, pen)
        y = h * self._guide_y
        self.line = sc.addLine(w * 0.12, y, w * 0.88, y, dashed(ACCENT, 2))
        self.a = Handle(w * 0.12, y, moved=self._moved, released=self._released)
        self.b = Handle(w * 0.88, y, moved=self._moved, released=self._released)
        sc.addItem(self.a)
        sc.addItem(self.b)

    def _moved(self, *_):
        self.line.setLine(QLineF(self.a.pos(), self.b.pos()))

    def _released(self, _):
        pa, pb = self.a.pos(), self.b.pos()
        if pb.x() < pa.x():
            pa, pb = pb, pa
        dx, dy = pb.x() - pa.x(), pb.y() - pa.y()
        if abs(dx) < 5:
            return
        theta = math.degrees(math.atan2(dy, dx))  # screen y points down
        img = self.doc.stage(self.i, "straight")
        self._guide_y = min(0.95, max(0.05, (pa.y() + pb.y()) / 2 / img.height))
        if abs(theta) > 0.01:
            self.doc.update(self.i, self.step, angle=round(self.st.recipe.angle + theta, 2))


# ---------------------------------------------------------------- 3 flatten
class _GuidePath(QGraphicsPathItem):
    """A guide's curve. Click to select the guide, double-click to add an
    anchor there. Its hit area is wider than the drawn line."""

    def __init__(self, editor, gi: int, hit: float):
        super().__init__()
        self.editor, self.gi, self.hit = editor, gi, hit
        self.setCursor(Qt.PointingHandCursor)
        self.setZValue(5)

    def shape(self):
        st = QPainterPathStroker()
        st.setWidth(self.hit)
        return st.createStroke(self.path())

    def mousePressEvent(self, e):
        self.editor.select(self.gi, None)
        e.accept()

    def mouseDoubleClickEvent(self, e):
        self.editor.split_at(self.gi, e.scenePos())
        e.accept()


class FlattenEditor(Editor):
    step, title = "flatten", "Flatten"

    def build(self):
        self.form.addWidget(heading("Overall strength"))
        self.overall = Slider(0, 150, 5, "{:.0f}%", on_commit=lambda v: self._commit(strength=v / 100))
        self.form.addWidget(self.overall)
        self.form.addWidget(heading("Selected anchor strength"))
        self.point = Slider(0, 150, 5, "{:.0f}%", on_change=self._anchor_strength,
                            on_commit=lambda v: self._commit())
        self.point.setEnabled(False)
        self.form.addWidget(self.point)
        self.add = QPushButton("+ Line")
        self.add.setCheckable(True)
        self.form.addWidget(row(self.add, button("Delete", self._delete)))
        self.result = QCheckBox("Show result")
        self.result.toggled.connect(lambda _: self.refresh())
        self.form.addWidget(self.result)
        self.form.addWidget(note(
            "Each orange line follows a staff. Click a line to edit it: drag the "
            "square anchors, and the round handles to bend it. Double-click the "
            "line to add an anchor; Delete removes the selected anchor, or the "
            "whole line. + Line, then click where a staff isn't covered."))
        self.canvas.clicked_empty.connect(self._clicked)
        for key in (QKeySequence.Delete, QKeySequence(Qt.Key_Backspace)):
            QShortcut(key, self, self._delete, context=Qt.WidgetWithChildrenShortcut)
        self.bz: list[np.ndarray] = []      # per guide: control points, scene coords
        self.astr: list[list[float]] = []   # per guide: anchor strengths
        self.sg: int | None = None          # selected guide
        self.sa: int | None = None          # selected anchor (index into anchors)

    def sync_side(self):
        self.overall.set(self.st.recipe.flatten_strength * 100)
        self.point.setEnabled(self.sg is not None and self.sa is not None)
        if self.sg is not None and self.sa is not None and self.sg < len(self.astr):
            self.point.set(self.astr[self.sg][self.sa] * 100)

    # ---- drawing
    def draw(self, fit):
        if self.result.isChecked():
            self.canvas.set_image(self.doc.stage(self.i, "flat"), keep_view=not fit)
            return
        img = self.doc.stage(self.i, "straight")
        self.img = img
        self.canvas.set_image(img, keep_view=not fit)
        self.bz, self.astr = [], []
        for g in self.st.recipe.guides:
            ensure_bezier(g)
            pts = [self.to_scene(img, x, y) for x, y in g.bezier]
            self.bz.append(np.array([[p.x(), p.y()] for p in pts]))
            self.astr.append(list(g.anchor_strengths))
        if self.sg is not None and self.sg >= len(self.bz):
            self.sg = self.sa = None
        self.paths = []
        hit = 0.012 * img.height
        for gi in range(len(self.bz)):
            p = _GuidePath(self, gi, hit)
            self.canvas.scene().addItem(p)
            self.paths.append(p)
            self._redraw_path(gi)
        self._draw_handles()
        self.sync_side()

    def _redraw_path(self, gi):
        pts = bezier_sample(self.bz[gi], 24)
        path = QPainterPath(QPointF(*pts[0]))
        for x, y in pts[1:]:
            path.lineTo(x, y)
        sel = gi == self.sg
        self.paths[gi].setPen(solid(CORAL, 2.5 if sel else 1.2))
        self.paths[gi].setPath(path)

    def _draw_handles(self):
        self.handles = []   # (Handle, ctrl index)
        self.extras = []    # handle lines and strength bars
        if self.sg is None:
            return
        sc = self.canvas.scene()
        c = self.bz[self.sg]
        bar = 0.04 * self.img.height
        for k in range(len(c)):
            if k % 3 == 0:
                s = self.astr[self.sg][k // 3]
                b = QGraphicsLineItem(c[k][0], c[k][1], c[k][0], c[k][1] - bar * s)
                b.setPen(solid(PURPLE, 3))
                b.setZValue(6)
                sc.addItem(b)
                self.extras.append(("bar", k, b))
                h = Handle(c[k][0], c[k][1], CORAL, r=6, square=True,
                           moved=lambda hd, pos, k=k: self._moved(k, pos),
                           released=lambda hd: self._commit())
                h.picked = lambda k=k: self.select(self.sg, k // 3)
            else:
                a = k - 1 if k % 3 == 1 else k + 1
                ln = QGraphicsLineItem(c[a][0], c[a][1], c[k][0], c[k][1])
                ln.setPen(solid(QColor(120, 120, 120), 1))
                ln.setZValue(6)
                sc.addItem(ln)
                self.extras.append(("line", k, ln))
                h = Handle(c[k][0], c[k][1], QColor(110, 110, 110), r=4,
                           moved=lambda hd, pos, k=k: self._moved(k, pos),
                           released=lambda hd: self._commit())
            sc.addItem(h)
            self.handles.append((h, k))
            if k % 3 == 0 and self.sa == k // 3:
                h.setSelected(True)

    def _sync_handles(self):
        """Move handles, handle lines and bars to match self.bz (no callbacks)."""
        c = self.bz[self.sg]
        bar = 0.04 * self.img.height
        for h, k in self.handles:
            if h.pos() != QPointF(*c[k]):
                h._moved, m = None, h._moved
                h.setPos(*c[k])
                h._moved = m
        for kind, k, it in self.extras:
            if kind == "bar":
                s = self.astr[self.sg][k // 3]
                it.setLine(c[k][0], c[k][1], c[k][0], c[k][1] - bar * s)
            else:
                a = k - 1 if k % 3 == 1 else k + 1
                it.setLine(c[a][0], c[a][1], c[k][0], c[k][1])

    # ---- editing
    def select(self, gi, anchor):
        if (gi, anchor) == (self.sg, self.sa):
            return
        same_guide = gi == self.sg
        self.sg, self.sa = gi, anchor
        if same_guide:  # just a different anchor: update the panel, no redraw
            self.sync_side()
        else:           # redraw after the current mouse event has finished
            QTimer.singleShot(0, self.refresh)

    def _moved(self, k, pos):
        c = self.bz[self.sg]
        new = np.array([pos.x(), pos.y()])
        if k % 3 == 0:  # anchor: carry its handles along
            d = new - c[k]
            for j in (k - 1, k, k + 1):
                if 0 <= j < len(c):
                    c[j] = c[j] + d
            self.sa = k // 3
        else:           # handle: mirror its partner round the anchor (smooth)
            c[k] = new
            a = k - 1 if k % 3 == 1 else k + 1
            other = a - 1 if k % 3 == 1 else a + 1
            if 0 <= other < len(c) and 0 < a < len(c) - 1:
                v = c[a] - c[k]
                n = np.hypot(*v)
                if n > 0:
                    c[other] = c[a] + v / n * np.hypot(*(c[other] - c[a]))
        self._redraw_path(self.sg)
        self._sync_handles()

    def _anchor_strength(self, v):
        if self.sg is None or self.sa is None:
            return
        self.astr[self.sg][self.sa] = v / 100
        self._sync_handles()

    def _guides(self) -> list[Guide]:
        img = self.doc.stage(self.i, "straight")
        out = []
        for c, s in zip(self.bz, self.astr):
            frame = [list(self.from_scene(img, QPointF(x, y))) for x, y in c]
            out.append(resolve(Guide(points=[], bezier=frame, anchor_strengths=list(s))))
        return out

    def _commit(self, strength: float | None = None):
        if not self.ready or self.result.isChecked():
            return
        ch = {"guides": self._guides()}
        if strength is not None:
            ch["flatten_strength"] = strength
        self.doc.update(self.i, self.step, **ch)

    def split_at(self, gi, pos: QPointF):
        """Add an anchor on guide gi at the point nearest `pos`."""
        if gi != self.sg:
            return self.select(gi, None)
        c = self.bz[gi]
        best = (1e18, 0, 0.5)
        for seg in range(len(c) // 3):
            ts = np.linspace(0.02, 0.98, 97)
            pts = np.array([_cubic_pt(c[3 * seg:3 * seg + 4], t) for t in ts])
            d = np.hypot(pts[:, 0] - pos.x(), pts[:, 1] - pos.y())
            j = int(np.argmin(d))
            if d[j] < best[0]:
                best = (d[j], seg, float(ts[j]))
        _, seg, t = best
        self.bz[gi] = np.array(split_bezier(c.tolist(), seg, t))
        s = self.astr[gi]
        self.astr[gi] = s[:seg + 1] + [(s[seg] + s[seg + 1]) / 2] + s[seg + 1:]
        self.sa = seg + 1
        self._commit()

    def _delete(self):
        if self.sg is None:
            return
        c, s = self.bz[self.sg], self.astr[self.sg]
        n_anchors = len(s)
        if self.sa is not None and 0 < self.sa < n_anchors - 1:
            # remove an interior anchor: refit that stretch as one segment
            a = self.sa
            pts = bezier_sample(c[3 * (a - 1):3 * (a + 1) + 1], 24)
            seg, _ = _fit_cubic(pts)
            self.bz[self.sg] = np.vstack([c[:3 * (a - 1)], np.asarray(seg), c[3 * (a + 1) + 1:]])
            del s[a]
            self.sa = None
        else:
            del self.bz[self.sg], self.astr[self.sg]
            self.sg = self.sa = None
        self._commit()

    def _clicked(self, pos: QPointF):
        if self.result.isChecked() or not self.ready:
            return
        if not self.add.isChecked():
            if self.sg is not None:
                self.select(None, None)
            return
        self.add.setChecked(False)
        img = self.doc.stage(self.i, "straight")
        r = self.st.recipe
        if r.crop:
            x0, _, x1, _ = box_to_abs([v * self.st.f for v in r.crop], img.size)
        else:
            x0, x1 = 0.1 * img.width, 0.9 * img.width
        y = pos.y()
        self.bz.append(np.array([[x0, y], [x0 + (x1 - x0) / 3, y], [x0 + 2 * (x1 - x0) / 3, y], [x1, y]]))
        self.astr.append([1.0, 1.0])
        self.sg, self.sa = len(self.bz) - 1, None
        self._commit()


def _cubic_pt(c, t):
    u = 1 - t
    return u ** 3 * c[0] + 3 * u * u * t * c[1] + 3 * u * t * t * c[2] + t ** 3 * c[3]


# ---------------------------------------------------------------- 4 crop
class CropEditor(Editor):
    step, title = "crop", "Crop"

    def build(self):
        self.form.addWidget(note("Drag the corners. Everything outside the box is left off the page."))
        self.form.addWidget(button("Whole page", self._whole))

    def _whole(self):
        if self.ready:
            img = self.doc.stage(self.i, "flat")
            f = self.st.f
            self.doc.update(self.i, self.step, crop=[-img.width / 2 / f, -img.height / 2 / f,
                                                     img.width / 2 / f, img.height / 2 / f])

    def draw(self, fit):
        img = self.doc.stage(self.i, "flat")
        self.img = img
        self.canvas.set_image(img, keep_view=not fit)
        r = self.st.recipe
        if r.crop:
            x0, y0, x1, y1 = box_to_abs([v * self.st.f for v in r.crop], img.size)
        else:
            x0, y0, x1, y1 = 0, 0, img.width, img.height
        sc = self.canvas.scene()
        self.shade = QGraphicsPathItem()
        self.shade.setBrush(QBrush(QColor(0, 0, 0, 90)))
        self.shade.setPen(Qt.NoPen)
        sc.addItem(self.shade)
        self.box_item = QGraphicsRectItem()
        self.box_item.setPen(solid(ACCENT, 2))
        sc.addItem(self.box_item)
        self.h = [Handle(x, y, moved=self._moved, released=self._released)
                  for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        for h in self.h:
            h.constrain = lambda p: QPointF(min(max(p.x(), 0), img.width), min(max(p.y(), 0), img.height))
            sc.addItem(h)
        self._box = [x0, y0, x1, y1]
        self._update()

    def _moved(self, h, pos):
        k = self.h.index(h)
        b = self._box
        if k in (0, 3):
            b[0] = pos.x()
        else:
            b[2] = pos.x()
        if k in (0, 1):
            b[1] = pos.y()
        else:
            b[3] = pos.y()
        self._update()

    def _update(self):
        x0, y0, x1, y1 = self._box
        rect = QRectF(QPointF(min(x0, x1), min(y0, y1)), QPointF(max(x0, x1), max(y0, y1)))
        self.box_item.setRect(rect)
        path = QPainterPath()
        path.addRect(QRectF(0, 0, self.img.width, self.img.height))
        path.addRect(rect)
        path.setFillRule(Qt.OddEvenFill)
        self.shade.setPath(path)
        pts = ((rect.left(), rect.top()), (rect.right(), rect.top()),
               (rect.right(), rect.bottom()), (rect.left(), rect.bottom()))
        for h, (x, y) in zip(self.h, pts):
            if h.pos() != QPointF(x, y) and not h.isUnderMouse():
                h._moved, m = None, h._moved  # move without re-triggering _moved
                h.setPos(x, y)
                h._moved = m

    def _released(self, _):
        x0, y0, x1, y1 = self._box
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return self.refresh()
        a = self.from_scene(self.img, QPointF(x0, y0))
        b = self.from_scene(self.img, QPointF(x1, y1))
        self.doc.update(self.i, self.step, crop=[a[0], a[1], b[0], b[1]])


# ---------------------------------------------------------------- 5 size and position
class _Movable(QGraphicsPixmapItem):
    def __init__(self, pix, on_move, on_release):
        super().__init__(pix)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setCursor(Qt.SizeAllCursor)
        self.setTransformationMode(Qt.SmoothTransformation)
        self.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)  # grab anywhere, not just ink
        self.on_move, self.on_release = on_move, on_release

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.on_move:
            return self.on_move(value)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.on_release()


class SizeEditor(Editor):
    step, title = "size", "Size and position"
    DPI = 72

    def build(self):
        self.form.addWidget(heading("Ghost: pages to show behind this one"))
        self.ghost_list = QListWidget()
        self.ghost_list.setFixedHeight(130)
        self.ghost_list.itemChanged.connect(self._ghost_toggled)
        self.form.addWidget(self.ghost_list)
        self.form.addWidget(row(button("Previous", lambda: self._ghost_preset("prev")),
                                button("All music", lambda: self._ghost_preset("all")),
                                button("None", lambda: self._ghost_preset("none"))))
        self.ghost_sel = None   # set of PageState; None = previous music page
        self.form.addWidget(heading("Ghost opacity"))
        self.opacity = Slider(0, 100, 5, "{:.0f}%", on_commit=lambda _: self.refresh())
        self.opacity.set(40)
        self.form.addWidget(self.opacity)
        self.form.addWidget(heading("Size"))
        self.size_slider = Slider(50, 150, 1, "{:.0f}%", on_commit=self._set_size)
        self.form.addWidget(self.size_slider)
        self.form.addWidget(row(button("Match others", self.auto), button("Apply to all", self._all)))
        self.form.addWidget(note("Drag the page to move it, drag the corner to resize. It snaps "
                                 "to the page centre and to the ghost's top edge. Size 100% keeps "
                                 "staff lines the same size as on every other page."))

    has_auto = False

    def sync_side(self):
        self.size_slider.set(self.st.recipe.scale_adjust * 100)
        sel = self._ghost_set()
        self.ghost_list.blockSignals(True)
        self.ghost_list.clear()
        for k, p in enumerate(self.doc.pages):
            if k == self.i:
                continue
            label = f"{k + 1} · {p.name}" + ("  (photo)" if p.recipe.kind == "photo" else "")
            it = QListWidgetItem(label)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if p in sel else Qt.Unchecked)
            it.setData(Qt.UserRole, k)
            self.ghost_list.addItem(it)
        self.ghost_list.blockSignals(False)

    def _ghost_set(self) -> set:
        if self.ghost_sel is not None:
            return {p for p in self.ghost_sel if p in self.doc.pages}
        return self._preset("prev")

    def _preset(self, which: str) -> set:
        music = [k for k, p in enumerate(self.doc.pages)
                 if k != self.i and p.recipe.kind == "music" and p.recipe.include]
        if which == "all":
            return {self.doc.pages[k] for k in music}
        if which == "prev":
            before = [k for k in music if k < self.i]
            pick = before[-1:] or music[:1]
            return {self.doc.pages[k] for k in pick}
        return set()

    def _ghost_preset(self, which: str):
        self.ghost_sel = self._preset(which)
        self.refresh()

    def _ghost_toggled(self, it: QListWidgetItem):
        sel = self._ghost_set()
        p = self.doc.pages[it.data(Qt.UserRole)]
        (sel.add if it.checkState() == Qt.Checked else sel.discard)(p)
        self.ghost_sel = sel
        self.refresh()

    def _set_size(self, v):
        if self.ready:
            self._resize_to(v / 100 / self.st.recipe.scale_adjust)

    def _all(self):
        r = self.st.recipe
        with self.doc.batch("Apply size to all pages"):  # one undo step
            for i, p in enumerate(self.doc.pages):
                if i != self.i and p.recipe.kind == "music":
                    self.doc.update(i, self.step, scale_adjust=r.scale_adjust,
                                    offset_in=list(r.offset_in))

    def draw(self, fit):
        lay = self.doc.paper_layout(self.DPI)
        if self.i not in lay:
            self.canvas.set_image(None)
            self.status.setText(self.status.text() + "  ·  not included")
            return
        img, pw, ph, x, y, nw, nh = lay[self.i]
        self.geom = (pw, ph, x, y, nw, nh)
        self.canvas.set_image(Image.new("L", (pw, ph), 255), keep_view=not fit)
        sc = self.canvas.scene()
        s = self.doc.settings
        m = round(s.margin * self.DPI)
        sc.addItem(_rect(QRectF(0, 0, pw, ph), solid(QColor(0, 0, 0, 60), 1)))
        sc.addItem(_rect(QRectF(m, m, pw - 2 * m, ph - 2 * m), dashed(QColor(0, 0, 0, 50), 1)))
        # ghosts
        sel = self._ghost_set()
        others = [k for k in lay if k != self.i and self.doc.pages[k] in sel]
        self.ghost_tops = []
        for k in others:
            gimg, _, _, gx, gy, gw, gh = lay[k]
            g = gimg.resize((gw, gh), Image.BILINEAR)
            a = np.asarray(g, dtype=np.uint8)
            rgba = np.zeros((gh, gw, 4), dtype=np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = 29, 158, 117  # teal ink
            rgba[..., 3] = (255 - a) * (self.opacity.value() / 100)
            it = QGraphicsPixmapItem(to_pixmap(Image.fromarray(rgba, "RGBA")))
            it.setPos(gx, gy)
            it.setZValue(2)
            sc.addItem(it)
            self.ghost_tops.append(gy)
        # this page
        cur = img.resize((nw, nh), Image.LANCZOS)
        rgba = np.zeros((nh, nw, 4), dtype=np.uint8)
        a = np.asarray(cur, dtype=np.uint8)
        rgba[..., 3] = 255 - a
        self._k = 1.0
        self.outline = _rect(QRectF(x, y, nw, nh), solid(ACCENT, 1.5))
        self.outline.setZValue(4)
        sc.addItem(self.outline)
        self.corner = Handle(x + nw, y + nh, moved=self._corner, released=self._corner_done)
        sc.addItem(self.corner)
        self.item = _Movable(to_pixmap(Image.fromarray(rgba, "RGBA")), None, self._moved_done)
        self.item.setPos(x, y)
        self.item.on_move = self._snap  # only after the initial placement
        self.item.setZValue(3)
        sc.addItem(self.item)

    def _snap(self, p: QPointF) -> QPointF:
        pw, ph, x, y, nw, nh = self.geom
        tol = 8
        nx, ny = p.x(), p.y()
        if abs(nx + nw / 2 - pw / 2) < tol:
            nx = pw / 2 - nw / 2
        for gt in self.ghost_tops:
            if abs(ny - gt) < tol:
                ny = gt
        self.outline.setRect(QRectF(nx, ny, nw * self._k, nh * self._k))
        self.corner._moved, mv = None, self.corner._moved
        self.corner.setPos(nx + nw * self._k, ny + nh * self._k)
        self.corner._moved = mv
        return QPointF(nx, ny)

    def _moved_done(self):
        pw, ph, x, y, nw, nh = self.geom
        p = self.item.pos()
        dx, dy = (p.x() - x) / self.DPI, (p.y() - y) / self.DPI
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return
        o = self.st.recipe.offset_in
        self.doc.update(self.i, self.step, offset_in=[round(o[0] + dx, 3), round(o[1] + dy, 3)])

    def _corner(self, h, pos):
        pw, ph, x, y, nw, nh = self.geom
        p = self.item.pos()
        k = max(0.2, (pos.x() - p.x()) / nw)
        self._k = k
        self.item.setScale(k)
        self.outline.setRect(QRectF(p.x(), p.y(), nw * k, nh * k))

    def _corner_done(self, _):
        self._resize_to(self._k)

    def _resize_to(self, k: float):
        """Scale this page by k, keeping its top-left corner where it is."""
        if abs(k - 1) < 1e-3:
            return
        pw, ph, x, y, nw, nh = self.geom
        p = self.item.pos()
        s = self.doc.settings
        m = round(s.margin * self.DPI)
        nw2, nh2 = round(nw * k), round(nh * k)
        xdef = (pw - nw2) // 2
        ydef = m if s.align == "top" else (ph - nh2) // 2
        r = self.st.recipe
        self.doc.update(self.i, self.step, scale_adjust=round(r.scale_adjust * k, 4),
                        offset_in=[round((p.x() - xdef) / self.DPI, 3),
                                   round((p.y() - ydef) / self.DPI, 3)])


def _rect(r: QRectF, pen: QPen) -> QGraphicsRectItem:
    it = QGraphicsRectItem(r)
    it.setPen(pen)
    return it


# ---------------------------------------------------------------- 6 contrast
class ContrastEditor(Editor):
    step, title, has_auto = "contrast", "Contrast", False

    def build(self):
        self.form.addWidget(heading("Clean background"))
        self.white = Slider(0, 10, 1, "{:.0f}", on_commit=lambda v: self._set(white_point=round(250 - v * 7)))
        self.form.addWidget(self.white)
        self.form.addWidget(heading("Darken lines"))
        self.dark = Slider(0, 10, 1, "{:.0f}", on_commit=lambda v: self._set(darken=round(1 + v * 0.2, 2)))
        self.form.addWidget(self.dark)
        self.bw = QCheckBox("Pure black and white")
        self.bw.toggled.connect(lambda v: self._set(bw=v))
        self.form.addWidget(self.bw)
        self.form.addWidget(note("Applies to every page. Zoom in (scroll) to judge faint lines; "
                                 "double-click to fit the page again."))
        self.form.addWidget(button("Reset to defaults", lambda: self._set(white_point=215, darken=1.8, bw=False)))
        self._quiet = False

    def _set(self, **ch):
        if self._quiet:
            return
        s = self.doc.settings
        if all(getattr(s, k) == v for k, v in ch.items()):
            return
        self.doc.set_settings(**ch)

    def sync_side(self):
        self._quiet = True
        s = self.doc.settings
        self.white.set(round((250 - s.white_point) / 7))
        self.dark.set(round((s.darken - 1) / 0.2))
        self.bw.setChecked(s.bw)
        self._quiet = False

    def draw(self, fit):
        img = self.doc.stage(self.i, "final")
        s = self.doc.settings
        if s.bw and self.st.recipe.kind == "music":
            img = img.point(lambda v: 255 if v > s.bw_threshold else 0)
        self.canvas.set_image(img, keep_view=not fit)


# ---------------------------------------------------------------- 7 output
class OutputEditor(Editor):
    step, title, has_auto, has_preview = "output", "Output", False, False
    DPI = 90

    def build(self):
        self.form.addWidget(heading("Paper"))
        self.paper = QComboBox()
        self.paper.addItems(["Letter", "A4", "Legal"])
        self.paper.currentTextChanged.connect(lambda t: self._set(paper=t.lower()))
        self.form.addWidget(self.paper)
        self.form.addWidget(heading("Margins (inches)"))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0, 2)
        self.margin.setSingleStep(0.1)
        self.margin.valueChanged.connect(lambda v: self._set(margin=round(v, 2)))
        self.form.addWidget(self.margin)
        self.form.addWidget(heading("Orientation"))
        self.orient = QComboBox()
        self.orient.addItems(["Portrait", "Landscape", "Auto"])
        self.orient.currentTextChanged.connect(lambda t: self._set(orientation=t.lower()))
        self.form.addWidget(self.orient)
        self.form.addWidget(heading("Place music"))
        self.align = QComboBox()
        self.align.addItems(["Top", "Center"])
        self.align.currentTextChanged.connect(lambda t: self._set(align=t.lower()))
        self.form.addWidget(self.align)
        self.same = QCheckBox("Same staff size on every page")
        self.same.toggled.connect(lambda v: self._set(uniform_scale=v))
        self.form.addWidget(self.same)
        self.form.addSpacing(12)
        save = button("Save PDF…", lambda: self.window().export_pdf())
        save.setDefault(True)
        self.form.addWidget(save)
        self.form.addWidget(button("Print…", lambda: self.window().print_pages()))
        self._quiet = False

    def _set(self, **ch):
        if not self._quiet:
            self.doc.set_settings(**ch)

    def sync_side(self):
        self._quiet = True
        s = self.doc.settings
        self.paper.setCurrentText({"letter": "Letter", "a4": "A4", "legal": "Legal"}[s.paper])
        self.margin.setValue(s.margin)
        self.orient.setCurrentText(s.orientation.capitalize())
        self.align.setCurrentText("Center" if s.align == "center" else "Top")
        self.same.setChecked(s.uniform_scale)
        self._quiet = False

    def draw(self, fit):
        lay = self.doc.paper_layout(self.DPI)
        if self.i not in lay:
            self.canvas.set_image(None)
            return
        img, pw, ph, x, y, nw, nh = lay[self.i]
        page = Image.new("L", (pw, ph), 255)
        page.paste(img.resize((nw, nh), Image.LANCZOS), (x, y))
        if self.doc.settings.bw and self.st.recipe.kind == "music":
            t = self.doc.settings.bw_threshold
            page = page.point(lambda v: 255 if v > t else 0)
        self.canvas.set_image(page, keep_view=not fit)
        self.canvas.scene().addItem(_rect(QRectF(0, 0, pw, ph), solid(QColor(0, 0, 0, 60), 1)))


EDITORS = [CleanEditor, StraightenEditor, FlattenEditor, CropEditor, SizeEditor,
           ContrastEditor, OutputEditor]
