"""Zoomable image canvas with an overlay layer for editor handles."""
from __future__ import annotations

from PIL import Image
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

ACCENT = QColor("#378ADD")
CORAL = QColor("#D85A30")
PURPLE = QColor("#7F77DD")
TEAL = QColor("#1D9E75")


def to_qimage(img: Image.Image) -> QImage:
    """PIL -> QImage (copied, so the PIL buffer can go away)."""
    if img.mode == "1":
        img = img.convert("L")
    if img.mode == "L":
        data = img.tobytes()
        return QImage(data, img.width, img.height, img.width, QImage.Format_Grayscale8).copy()
    if img.mode == "RGBA":
        data = img.tobytes()
        return QImage(data, img.width, img.height, img.width * 4, QImage.Format_RGBA8888).copy()
    img = img.convert("RGB")
    data = img.tobytes()
    return QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888).copy()


def to_pixmap(img: Image.Image) -> QPixmap:
    return QPixmap.fromImage(to_qimage(img))


class Handle(QGraphicsPathItem):
    """A draggable dot (or square). Calls `moved(handle, pos)` while
    dragging, `picked()` when pressed and `released(handle)` when let go.
    Stays the same size on screen."""

    def __init__(self, x: float, y: float, color: QColor = ACCENT, r: float = 6,
                 moved=None, released=None, square: bool = False):
        super().__init__()
        path = QPainterPath()
        if square:
            path.addRect(-r, -r, 2 * r, 2 * r)
        else:
            path.addEllipse(-r, -r, 2 * r, 2 * r)
        self.setPath(path)
        self.picked = None
        self.setPos(x, y)
        self.setBrush(QBrush(QColor("white")))
        self.setPen(QPen(color, 2))
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable
                      | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemIgnoresTransformations)
        self.setCursor(Qt.SizeAllCursor)
        self.setZValue(10)
        self._moved, self._released = moved, released
        self.constrain = None  # optional fn(QPointF) -> QPointF

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.constrain:
            return self.constrain(value)
        if change == QGraphicsItem.ItemPositionHasChanged and self._moved:
            self._moved(self, self.pos())
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.setBrush(QBrush(self.pen().color() if value else QColor("white")))
        return super().itemChange(change, value)

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        if self.picked:
            self.picked()

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if self._released:
            # after this event returns: the callback may redraw (and delete us)
            cb = self._released
            QTimer.singleShot(0, lambda: cb(self))


class Canvas(QGraphicsView):
    """Shows one image; editors add overlay items to `scene()`.

    Scroll wheel zooms; dragging anywhere that isn't a handle pans, and so
    does Space + drag or middle-button drag anywhere; double-click fits. `clicked_empty` fires
    with scene coordinates on a click (no drag) off any handle.
    """
    clicked_empty = Signal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#ECEAE4"))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.pix = QGraphicsPixmapItem()
        self.pix.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.pix)
        self._fit = True
        self._pan = None
        self._press = None
        self._space = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.viewport().setCursor(Qt.OpenHandCursor)  # items set their own cursors

    def set_image(self, img: Image.Image | None, keep_view: bool = False) -> None:
        if img is None:
            self.pix.setPixmap(QPixmap())
            return
        self.pix.setPixmap(to_pixmap(img))
        r = self.pix.boundingRect()
        pad = 0.06 * max(r.width(), r.height())
        self.scene().setSceneRect(r.adjusted(-pad, -pad, pad, pad))
        if not keep_view or self._fit:
            self.fit()

    def fit(self) -> None:
        self._fit = True
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def clear_overlay(self) -> None:
        for it in list(self.scene().items()):
            if it is not self.pix and it.parentItem() is None:
                self.scene().removeItem(it)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit()

    def wheelEvent(self, e):
        self._fit = False
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    DRAG_PX = 4

    @staticmethod
    def _takes_mouse(item) -> bool:
        while item is not None:
            if getattr(item, "interactive", False) or item.flags() & (
                    QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable):
                return True
            item = item.parentItem()
        return False

    def _interactive(self, at) -> bool:
        """Is there a handle, guide line or movable page under view point `at`?
        The image and decorations (grid, shading, outlines, handle lines)
        don't take the mouse, so dragging on them pans."""
        if not isinstance(at, QPoint):
            at = QPoint(int(at[0]), int(at[1])) if isinstance(at, tuple) else None
        if at is None:
            return False
        return any(self._takes_mouse(it) for it in self.items(at))

    def _start_pan(self, pos: QPointF) -> None:
        # pan by the total distance from here, so no rounding drift
        self._pan = pos
        self._pan_from = (self.horizontalScrollBar().value(), self.verticalScrollBar().value())
        self._fit = False
        self.viewport().setCursor(Qt.ClosedHandCursor)

    # Space + drag pans anywhere (even over handles or the movable page)
    def enterEvent(self, e):
        super().enterEvent(e)
        self.setFocus(Qt.MouseFocusReason)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            return
        super().keyReleaseEvent(e)

    def mousePressEvent(self, e):
        pos = e.position()
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self._space):
            self._start_pan(pos)
            return
        if e.button() == Qt.LeftButton and not self._interactive(pos.toPoint()):
            self._press = pos  # a click or a pan: decided when the mouse moves
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._press is not None and self._pan is None and \
                (pos - self._press).manhattanLength() > self.DRAG_PX:
            self._start_pan(self._press)
        if self._pan is not None:
            d = pos - self._pan
            self.horizontalScrollBar().setValue(self._pan_from[0] - round(d.x()))
            self.verticalScrollBar().setValue(self._pan_from[1] - round(d.y()))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pan is not None or self._press is not None:
            click = self._pan is None and e.button() == Qt.LeftButton
            press = self._press
            self._pan = self._press = None
            self.viewport().setCursor(Qt.OpenHandCursor)
            if click:
                self.scene().clearSelection()
                self.clicked_empty.emit(self.mapToScene(press.toPoint()))
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        # double-click a guide line adds an anchor; anywhere else fits the page
        if self._interactive(e.position().toPoint()):
            super().mouseDoubleClickEvent(e)
        else:
            self.fit()


def dashed(color: QColor, width: float = 1.5) -> QPen:
    p = QPen(color, width, Qt.DashLine)
    p.setCosmetic(True)
    return p


def solid(color: QColor, width: float = 1.5) -> QPen:
    p = QPen(color, width)
    p.setCosmetic(True)
    return p


def rect_item(rect: QRectF, pen: QPen) -> QGraphicsRectItem:
    it = QGraphicsRectItem(rect)
    it.setPen(pen)
    return it
