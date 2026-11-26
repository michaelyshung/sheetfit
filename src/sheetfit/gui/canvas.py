"""Zoomable image canvas with an overlay layer for editor handles."""
from __future__ import annotations

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
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

    Scroll wheel zooms, middle-drag (or space+drag) pans. `clicked_empty`
    fires with scene coordinates when the user clicks off any item.
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

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._pan = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        item = self.itemAt(e.position().toPoint())
        if e.button() == Qt.LeftButton and (item is None or item is self.pix):
            self.scene().clearSelection()
            self.clicked_empty.emit(self.mapToScene(e.position().toPoint()))
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self._fit = False
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self._pan = None
            self.unsetCursor()
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
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
