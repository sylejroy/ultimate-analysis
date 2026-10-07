"""A frame with boxes that can be drawn, moved, resized, and deleted with the mouse."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QWidget

from ...utils.label_files import CLASS_NAMES, LabelBox
from ..widgets.images import frame_to_pixmap

CLASS_COLORS = [QColor(255, 220, 0), QColor(0, 220, 90)]  # disc, player
HANDLE_RADIUS = 5  # Size of the grips on the selected box, in screen pixels
GRIP_REACH = 8  # How close the mouse must be to a grip to take it
MIN_BOX_SIZE = 3  # A drawn box smaller than this (frame pixels) was a stray click
CLICK_SLACK = 4  # The mouse may move this far (screen pixels) and it is still a click
MAX_ZOOM = 16.0

# Grip -> the edges of the box it moves
GRIPS: Dict[str, str] = {
    "top-left": "lt",
    "top": "t",
    "top-right": "rt",
    "right": "r",
    "bottom-right": "rb",
    "bottom": "b",
    "bottom-left": "lb",
    "left": "l",
}

# What a drag on free space draws, by mouse button: a disc (class 0) or a player (class 1)
BUTTON_CLASS = {Qt.LeftButton: 0, Qt.RightButton: 1}


class BoxCanvas(QWidget):
    """Shows a frame and lets the user edit the boxes on it.

    - Drag: draw a new box, a disc with the left button and a player with the right,
      also across existing boxes (a disc in front of a player lies inside the player's box)
    - Click a box: select it (the smallest one under the mouse)
    - Drag the selected box: move it; drag one of its grips: resize it
    - Click on free space or Escape: select nothing
    - Delete or Backspace: remove the selected box
    - Mouse wheel: zoom at the mouse; middle button drag: move the view
    """

    boxes_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.boxes: List[LabelBox] = []
        self.selected: Optional[int] = None
        # The classes boxes may have here; a disc-only dataset takes no player boxes
        self.classes = set(BUTTON_CLASS.values())
        self._button = Qt.LeftButton  # The button the drag under way began with
        self.unconfirmed = False  # Boxes are a suggestion that has not been saved

        self._pixmap: Optional[QPixmap] = None
        self._frame_size = (1, 1)  # width, height
        self._zoom = 1.0
        self._view_center: Optional[Tuple[float, float]] = None  # Frame point in the middle
        self._drag: Optional[Tuple[str, QPointF, Optional[LabelBox]]] = None
        self._pan_from: Optional[QPointF] = None

        self.setMinimumSize(640, 360)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    # ------------------------------------------------------------------ content

    def set_frame(self, frame: np.ndarray, boxes: List[LabelBox], unconfirmed: bool) -> None:
        """Show a frame with its boxes; the zoom stays, so stepping through frames is calm."""
        size = (frame.shape[1], frame.shape[0])
        if size != self._frame_size:
            self._frame_size = size
            self.reset_view()
        self._pixmap = frame_to_pixmap(frame)
        self.boxes = boxes
        self.unconfirmed = unconfirmed
        self.selected = None
        self._drag = None
        self.update()

    def reset_view(self) -> None:
        """Show the whole frame."""
        self._zoom = 1.0
        self._view_center = None
        self.update()

    def set_class(self, class_id: int) -> None:
        """Change the selected box to another class."""
        if self.selected is not None and class_id in self.classes:
            self.boxes[self.selected].class_id = class_id
            self._changed()

    def delete_selected(self) -> None:
        if self.selected is not None:
            del self.boxes[self.selected]
            self.selected = None
            self._changed()

    def _changed(self) -> None:
        self.update()
        self.boxes_changed.emit()

    # ------------------------------------------------------------------ view

    def _scale(self) -> float:
        """Screen pixels per frame pixel."""
        width, height = self._frame_size
        return min(self.width() / width, self.height() / height) * self._zoom

    def _origin(self) -> Tuple[float, float]:
        """Screen position of the frame's top-left corner."""
        width, height = self._frame_size
        center_x, center_y = self._view_center or (width / 2, height / 2)
        scale = self._scale()
        return self.width() / 2 - center_x * scale, self.height() / 2 - center_y * scale

    def to_frame(self, position: QPointF) -> Tuple[float, float]:
        origin_x, origin_y = self._origin()
        scale = self._scale()
        return (position.x() - origin_x) / scale, (position.y() - origin_y) / scale

    def to_screen(self, x: float, y: float) -> QPointF:
        origin_x, origin_y = self._origin()
        scale = self._scale()
        return QPointF(origin_x + x * scale, origin_y + y * scale)

    def wheelEvent(self, event):
        before = self.to_frame(event.pos())
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom = min(MAX_ZOOM, max(1.0, self._zoom * factor))
        if self._zoom == 1.0:
            self._view_center = None
        else:
            # Keep the frame point under the mouse where it is
            scale = self._scale()
            self._view_center = (
                before[0] - (event.pos().x() - self.width() / 2) / scale,
                before[1] - (event.pos().y() - self.height() / 2) / scale,
            )
        self.update()

    # ------------------------------------------------------------------ mouse

    def _grip_positions(self, box: LabelBox) -> Dict[str, QPointF]:
        middle_x, middle_y = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        points = {
            "top-left": (box.x1, box.y1),
            "top": (middle_x, box.y1),
            "top-right": (box.x2, box.y1),
            "right": (box.x2, middle_y),
            "bottom-right": (box.x2, box.y2),
            "bottom": (middle_x, box.y2),
            "bottom-left": (box.x1, box.y2),
            "left": (box.x1, middle_y),
        }
        return {name: self.to_screen(x, y) for name, (x, y) in points.items()}

    def _grip_at(self, position: QPointF) -> Optional[str]:
        if self.selected is None:
            return None
        for name, point in self._grip_positions(self.boxes[self.selected]).items():
            if (point - position).manhattanLength() <= GRIP_REACH:
                return name
        return None

    def _box_at(self, x: float, y: float) -> Optional[int]:
        """The smallest box that contains a frame point."""
        inside = [
            (abs((box.x2 - box.x1) * (box.y2 - box.y1)), index)
            for index, box in enumerate(self.boxes)
            if min(box.x1, box.x2) <= x <= max(box.x1, box.x2)
            and min(box.y1, box.y2) <= y <= max(box.y1, box.y2)
        ]
        return min(inside)[1] if inside else None

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MiddleButton:
            self._pan_from = QPointF(event.pos())
            return
        if event.button() not in BUTTON_CLASS or self._pixmap is None or self._drag is not None:
            return
        self._button = event.button()

        position = QPointF(event.pos())
        x, y = self.to_frame(position)
        grip = self._grip_at(position)
        if grip is not None:
            self._drag = (GRIPS[grip], position, None)
            return

        selected = self.boxes[self.selected] if self.selected is not None else None
        if (
            selected is not None
            and selected.x1 <= x <= selected.x2
            and selected.y1 <= y <= selected.y2
        ):
            copy = LabelBox(selected.class_id, selected.x1, selected.y1, selected.x2, selected.y2)
            self._drag = ("move", position, copy)
        else:
            # A click or the start of a new box: the first movement tells
            self._drag = ("undecided", position, None)

    def mouseMoveEvent(self, event):
        position = QPointF(event.pos())
        if self._pan_from is not None:
            scale = self._scale()
            width, height = self._frame_size
            center_x, center_y = self._view_center or (width / 2, height / 2)
            delta = position - self._pan_from
            self._view_center = (center_x - delta.x() / scale, center_y - delta.y() / scale)
            self._pan_from = position
            self.update()
            return
        if self._drag is None:
            return

        mode, start, original = self._drag
        if mode == "undecided":
            if (position - start).manhattanLength() <= CLICK_SLACK:
                return
            # Dragging draws a new box from where the button went down: a disc with the
            # left button, a player with the right
            new_class = BUTTON_CLASS[self._button]
            if new_class not in self.classes:
                self._drag = None
                return
            start_x, start_y = self.to_frame(start)
            self.boxes.append(LabelBox(new_class, start_x, start_y, start_x, start_y))
            self.selected = len(self.boxes) - 1
            mode = "rb"
            self._drag = (mode, start, None)
        if self.selected is None:
            return
        box = self.boxes[self.selected]
        width, height = self._frame_size
        x, y = self.to_frame(position)
        if mode == "move":
            start_x, start_y = self.to_frame(start)
            # The box keeps its size at the frame border
            shift_x = min(max(x - start_x, -original.x1), width - original.x2)
            shift_y = min(max(y - start_y, -original.y1), height - original.y2)
            box.x1, box.x2 = original.x1 + shift_x, original.x2 + shift_x
            box.y1, box.y2 = original.y1 + shift_y, original.y2 + shift_y
        else:
            x, y = min(max(x, 0), width), min(max(y, 0), height)
            if "l" in mode:
                box.x1 = x
            if "r" in mode:
                box.x2 = x
            if "t" in mode:
                box.y1 = y
            if "b" in mode:
                box.y2 = y
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_from = None
            return
        if self._drag is None or event.button() != self._button:
            return
        if self._drag[0] == "undecided":
            # A click: select the box under the mouse, or nothing on free space
            self._drag = None
            self.selected = self._box_at(*self.to_frame(QPointF(event.pos())))
            self.update()
            return
        if self.selected is None:
            self._drag = None
            return
        moved = (QPointF(event.pos()) - self._drag[1]).manhattanLength() > 0
        self._drag = None

        # Dragging a grip past the opposite edge turns the box inside out
        box = self.boxes[self.selected]
        box.x1, box.x2 = sorted((box.x1, box.x2))
        box.y1, box.y2 = sorted((box.y1, box.y2))
        if box.x2 - box.x1 < MIN_BOX_SIZE or box.y2 - box.y1 < MIN_BOX_SIZE:
            del self.boxes[self.selected]
            self.selected = None
            moved = True
        if moved:
            self._changed()
        else:
            self.update()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif event.key() == Qt.Key_Escape:
            self.selected = None
            self.update()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------ drawing

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(26, 26, 26))
        if self._pixmap is None:
            return

        width, height = self._frame_size
        scale = self._scale()
        origin_x, origin_y = self._origin()
        # Single pixels stay sharp when zoomed in: a disc is only a few of them
        painter.setRenderHint(QPainter.SmoothPixmapTransform, scale < 2)
        target = QRectF(origin_x, origin_y, width * scale, height * scale)
        painter.drawPixmap(target, self._pixmap, QRectF(0, 0, width, height))

        for index, box in enumerate(self.boxes):
            color = CLASS_COLORS[box.class_id % len(CLASS_COLORS)]
            pen = QPen(color, 3 if index == self.selected else 2)
            if self.unconfirmed:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            rectangle = QRectF(self.to_screen(box.x1, box.y1), self.to_screen(box.x2, box.y2))
            painter.drawRect(rectangle.normalized())
            if index == self.selected:
                painter.drawText(
                    rectangle.normalized().topLeft() + QPointF(0, -4), CLASS_NAMES[box.class_id]
                )
                painter.setBrush(color)
                for point in self._grip_positions(box).values():
                    painter.drawRect(
                        QRectF(
                            point.x() - HANDLE_RADIUS / 2,
                            point.y() - HANDLE_RADIUS / 2,
                            HANDLE_RADIUS,
                            HANDLE_RADIUS,
                        )
                    )
                painter.setBrush(Qt.NoBrush)
