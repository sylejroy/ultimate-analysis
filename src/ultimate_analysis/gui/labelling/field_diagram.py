"""A drawing of the field on which the element to label next is picked."""

from typing import Dict, Optional, Set, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from ...utils.field_template import FieldTemplate

BACKGROUND = QColor(26, 26, 26)
GRASS = QColor(34, 70, 44)
END_ZONE = QColor(44, 88, 58)
NOT_LABELLED = QColor(150, 150, 150)
LABELLED = QColor(0, 220, 90)
CURRENT = QColor(255, 220, 0)
REACH = 9  # How close a click must be to an element, in screen pixels
MARGIN = 14


def title(name: str) -> str:
    """An element's name as shown to the user."""
    return name.replace("_", " ").capitalize()


class FieldDiagram(QWidget):
    """The field seen from above, far end at the top, as the camera looks down it.

    Shows which line of the field is under the mouse in the frame (yellow). A click on a
    line or a mark is reported, for a use that wants one.
    """

    element_selected = pyqtSignal(str)

    def __init__(self, template: FieldTemplate):
        super().__init__()
        self.template = template
        self.labelled: Set[str] = set()
        self.current: Optional[str] = None
        self.setMinimumSize(150, 300)

    def set_state(self, labelled: Set[str], current: Optional[str]) -> None:
        self.labelled = set(labelled)
        self.current = current
        self.update()

    def set_template(self, template: FieldTemplate) -> None:
        self.template = template
        self.update()

    # ------------------------------------------------------------------ geometry

    def _scale_and_origin(self) -> Tuple[float, float, float]:
        """Screen pixels per field unit, and the screen position of the near left corner."""
        scale = min(
            (self.width() - 2 * MARGIN) / self.template.width,
            (self.height() - 2 * MARGIN) / self.template.length,
        )
        left = (self.width() - self.template.width * scale) / 2
        bottom = (self.height() + self.template.length * scale) / 2
        return scale, left, bottom

    def _to_screen(self, x: float, y: float) -> QPointF:
        scale, left, bottom = self._scale_and_origin()
        return QPointF(left + x * scale, bottom - y * scale)  # The far end is up

    def _screen_lines(self) -> Dict[str, Tuple[QPointF, QPointF]]:
        return {
            name: (self._to_screen(*start), self._to_screen(*end))
            for name, (start, end) in self.template.lines.items()
        }

    def _element_at(self, position: QPointF) -> Optional[str]:
        # Marks first: the corners lie on the lines
        best: Tuple[float, Optional[str]] = (REACH, None)
        for name, place in self.template.points.items():
            distance = (self._to_screen(*place) - position).manhattanLength()
            if distance < best[0]:
                best = (distance, name)
        if best[1] is not None:
            return best[1]
        for name, (start, end) in self._screen_lines().items():
            along = end - start
            length_sq = along.x() ** 2 + along.y() ** 2
            offset = position - start
            share = max(
                0.0, min(1.0, (offset.x() * along.x() + offset.y() * along.y()) / length_sq)
            )
            nearest = start + along * share
            distance = (nearest - position).manhattanLength()
            if distance < best[0]:
                best = (distance, name)
        return best[1]

    # ------------------------------------------------------------------ events

    def mousePressEvent(self, event):
        name = self._element_at(QPointF(event.pos()))
        if name is not None:
            self.element_selected.emit(name)

    def _colour(self, name: str) -> QColor:
        if name == self.current:
            return CURRENT
        return LABELLED if name in self.labelled else NOT_LABELLED

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), BACKGROUND)

        field = self.template
        painter.fillRect(
            QRectF(self._to_screen(0, field.length), self._to_screen(field.width, 0)), GRASS
        )
        for low, high in ((0.0, field.end_zone), (field.length - field.end_zone, field.length)):
            painter.fillRect(
                QRectF(self._to_screen(0, high), self._to_screen(field.width, low)), END_ZONE
            )

        for name, (start, end) in self._screen_lines().items():
            painter.setPen(QPen(self._colour(name), 4 if name == self.current else 2))
            painter.drawLine(start, end)
        for name, place in field.points.items():
            colour = self._colour(name)
            painter.setPen(QPen(colour, 2))
            painter.setBrush(colour if name in self.labelled or name == self.current else GRASS)
            radius = 6 if name == self.current else 4
            painter.drawEllipse(self._to_screen(*place), radius, radius)
        painter.setBrush(Qt.NoBrush)

        painter.setPen(QColor(200, 200, 200))
        painter.drawText(QRectF(0, 0, self.width(), MARGIN), Qt.AlignCenter, "far")
        painter.drawText(
            QRectF(0, self.height() - MARGIN, self.width(), MARGIN), Qt.AlignCenter, "near (camera)"
        )
