"""What a simulated player who chooses where to run (or where to throw) is weighing now,
drawn, not said: the choice taken first, then the likeliest others, one row each, with
its number as on the field beside. Along a row: which way and how far it is, as an arrow
from the player (their team attacking upwards) ending in what it is (a place, the disc,
a teammate, an opponent; two bars: holding on); how likely it is; who gets there first
(green to the right: the player or the receiver, red to the left: an opponent, by up to
three seconds); how much room there is (a disc as wide as the nearest opponent is far,
up to the ring: twenty yards); and whether a throw gets in (green, amber, red)."""

from typing import List, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from .choices import DISC, HOLD, PLACE, TEAMMATE

GROUND = QColor(22, 24, 30)
TEXT = QColor(226, 229, 234)
FAINT = QColor(110, 116, 128)
TAKEN = QColor(255, 214, 64)
OTHER = QColor(120, 150, 200)
TRACK = QColor(46, 50, 60)
GOOD, TIGHT, BAD = QColor(90, 220, 120), QColor(240, 180, 70), QColor(235, 90, 80)
ROW_MOST = 96.0  # Pixels a choice takes at most
REACH = 30.0  # Yards an arrow as long as its dial is wide stands for


def _between(low: QColor, high: QColor, share: float) -> QColor:
    share = min(1.0, max(0.0, share))
    return QColor(
        int(low.red() + (high.red() - low.red()) * share),
        int(low.green() + (high.green() - low.green()) * share),
        int(low.blue() + (high.blue() - low.blue()) * share),
    )


class ChoicesView(QWidget):
    """The choices of one player in one step, the one taken first."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # A choice: its chance, and choices.spot_glyph's: way (2,), what, first, room, lane
        self.rows: List[dict] = []
        self.colours = (QColor(240, 240, 240), QColor(220, 60, 60))  # Teammates, opponents
        self.setToolTip(
            "Each row a choice, numbered as on the field, the one taken first. From the left:\n"
            "which way and how far (their team attacks upwards), ending in what it is: a\n"
            "place, the disc, a teammate, an opponent (two bars: holding on); how likely;\n"
            "who gets there first (green: them or the receiver, red: an opponent, up to 3 s);\n"
            "room around it (the nearest opponent, up to the ring: 20 yd); whether a throw\n"
            "gets in (green, amber, red)"
        )

    def show_choices(self, rows: Sequence[dict], colours: Optional[tuple] = None) -> None:
        self.rows = list(rows)
        if colours is not None:
            self.colours = colours
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), GROUND)
        if not self.rows:
            return
        width, height = float(self.width()), float(self.height())
        row = min(ROW_MOST, (height - 12.0) / len(self.rows))
        dial = row - 14.0
        most = max(r["chance"] for r in self.rows) or 1.0
        # Columns: the number, the dial, the chance, who is first, the room, the lane
        x_dial = 46.0
        x_chance = x_dial + dial + 22.0
        fixed = 194.0 + 2 * dial  # The race, the room and the lane, right of the chance
        long = max(80.0, width - x_chance - fixed - 70.0)
        x_race = x_chance + long + 70.0
        x_room = x_race + 150.0
        x_lane = x_room + dial + 20.0
        bold = QFont(self.font())
        bold.setBold(True)
        for n, r in enumerate(self.rows):
            top = 6.0 + n * row
            middle = top + row / 2
            colour = TAKEN if n == 0 else OTHER
            if n == 0:  # The one taken: set apart
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 214, 64, 22))
                painter.drawRoundedRect(QRectF(4, top, width - 8, row - 4), 6, 6)
            painter.setPen(Qt.NoPen)
            painter.setBrush(colour)
            painter.drawEllipse(QPointF(24.0, middle - 2), 12.0, 12.0)
            painter.setFont(bold)
            painter.setPen(GROUND)
            painter.drawText(QRectF(12, middle - 14, 24, 24), Qt.AlignCenter, str(n + 1))
            self._dial(painter, QPointF(x_dial + dial / 2, middle - 2), dial / 2, r, colour)
            # How likely
            painter.setPen(Qt.NoPen)
            painter.setBrush(TRACK)
            painter.drawRoundedRect(QRectF(x_chance, middle - 7, long, 10), 5, 5)
            painter.setBrush(colour)
            painter.drawRoundedRect(
                QRectF(x_chance, middle - 7, max(6.0, long * r["chance"] / most), 10), 5, 5
            )
            painter.setPen(TAKEN if n == 0 else TEXT)
            said = f"{r['chance']:.0%}" if r["chance"] >= 0.005 else "<1%"
            painter.drawText(
                QRectF(x_chance + long + 6, middle - 13, 56, 22),
                Qt.AlignLeft | Qt.AlignVCenter,
                said,
            )
            if r["what"] == HOLD:
                continue
            self._race(painter, x_race, middle - 2, 130.0, r["first"])
            self._room(painter, QPointF(x_room + dial / 2, middle - 2), dial / 2 - 2, r["room"])
            if r["lane"] is not None:
                self._lane(painter, x_lane, middle - 2, dial, r["lane"])

    def _dial(self, painter: QPainter, centre: QPointF, radius: float, r: dict, colour) -> None:
        """Which way and how far, from the player in the middle; what is at the end."""
        painter.setPen(QPen(TRACK, 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, radius, radius)
        if r["what"] == HOLD:  # Two bars: holding on
            painter.setPen(Qt.NoPen)
            painter.setBrush(colour)
            for side in (-1, 1):
                painter.drawRoundedRect(
                    QRectF(centre.x() + side * 6 - 3, centre.y() - radius / 2, 6, radius), 2, 2
                )
            return
        way = np.asarray(r["way"], dtype=float)
        far = float(np.hypot(*way))
        reach = min(1.0, far / REACH) * (radius - 5.0)
        step = way / far * reach if far > 1e-6 else np.zeros(2)
        end = QPointF(centre.x() + step[0], centre.y() - step[1])  # Up the field: up
        painter.setPen(QPen(colour, 2.5, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(centre, end)
        painter.setPen(Qt.NoPen)
        painter.setBrush(TEXT)
        painter.drawEllipse(centre, 2.5, 2.5)
        if r["what"] == PLACE:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(colour, 2))
            painter.drawEllipse(end, 4.0, 4.0)
        elif r["what"] == DISC:
            painter.setPen(QPen(TAKEN, 2.5))
            for a, b in ((-4, -4), (-4, 4)):
                painter.drawLine(
                    QPointF(end.x() + a, end.y() + b), QPointF(end.x() - a, end.y() - b)
                )
        else:  # A player: in their team's colour
            painter.setPen(QPen(GROUND, 1))
            painter.setBrush(self.colours[0 if r["what"] == TEAMMATE else 1])
            painter.drawEllipse(end, 5.0, 5.0)

    def _race(self, painter: QPainter, x: float, y: float, long: float, first: float) -> None:
        """Who gets there first: from the middle, green to the right or red to the left."""
        painter.setPen(Qt.NoPen)
        painter.setBrush(TRACK)
        painter.drawRoundedRect(QRectF(x, y - 5, long, 10), 5, 5)
        half = long / 2
        share = min(1.0, abs(first))
        painter.setBrush(GOOD if first >= 0 else BAD)
        if first >= 0:
            painter.drawRoundedRect(QRectF(x + half, y - 5, max(2.0, half * share), 10), 3, 3)
        else:
            painter.drawRoundedRect(QRectF(x + half - half * share, y - 5, half * share, 10), 3, 3)
        painter.setPen(QPen(TEXT, 1.5))
        painter.drawLine(QPointF(x + half, y - 9), QPointF(x + half, y + 9))

    def _room(self, painter: QPainter, centre: QPointF, radius: float, room: float) -> None:
        """Room around it: a disc as wide as the nearest opponent is far, in a ring."""
        painter.setPen(Qt.NoPen)
        painter.setBrush(_between(BAD, GOOD, room / 0.5))
        painter.drawEllipse(centre, max(2.0, radius * room), max(2.0, radius * room))
        painter.setPen(QPen(FAINT, 1.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, radius, radius)

    def _lane(self, painter: QPainter, x: float, y: float, long: float, lane: float) -> None:
        """Whether a throw gets in: a line from the disc to the place, by how clear."""
        colour = BAD if lane < 0.2 else (TIGHT if lane < 0.5 else GOOD)
        painter.setPen(QPen(colour, 3.0, Qt.SolidLine if lane >= 0.2 else Qt.DotLine, Qt.RoundCap))
        painter.drawLine(QPointF(x + 8, y), QPointF(x + long - 8, y))
        painter.setPen(Qt.NoPen)
        painter.setBrush(TAKEN)
        painter.drawEllipse(QPointF(x + 5, y), 4.5, 4.5)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(colour, 2))
        painter.drawEllipse(QPointF(x + long - 5, y), 4.0, 4.0)
