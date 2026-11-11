"""A simulated player's brain, as it works in the present step: the network's inputs
grouped by what they say (one node a group, labelled, with what it says now), its two
hidden layers and its output, each node as bright as it is active, and the connections
that carry the most of the signal now, green where they push up and red where they push
down. Connections a brain has cut are not there to draw. A network that scores choices
is shown at work on the one taken, its output that choice, drawn on the field beside."""

from typing import List, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from ...simulation.learned import INPUT_GROUPS, OUTPUT_NAMES

STRONGEST = 40  # Connections drawn between two layers
GROUND = QColor(22, 24, 30)
TEXT = QColor(200, 204, 210)
FAINT = QColor(130, 136, 146)
CHOSEN = QColor(255, 214, 64)
UP, DOWN = QColor(90, 220, 120), QColor(235, 90, 80)
LABELS = 340.0  # Pixels kept left of the inputs for their names and what they say
NOTES = 130.0  # ... of which for what they say
OUTPUT_LABELS = 210.0  # ... and right of the outputs


class NetworkView(QWidget):
    """The network of one player in one step."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layers: List[np.ndarray] = []  # Activations: input, hidden, hidden, output
        self.weights: List[np.ndarray] = []  # (outputs, inputs) per layer
        self.masks: Optional[List[np.ndarray]] = None  # Which connections there are
        self.groups = INPUT_GROUPS  # (name, how many) of the inputs, in order
        self.notes: Sequence[str] = ()  # What each group of inputs says now
        self.outputs = OUTPUT_NAMES
        # Choices scored by the network: (label, chance), the one taken (its inputs shown)
        # first; None: the outputs are the network's own
        self.options: Optional[Sequence[tuple]] = None
        self.message = "Click a player to look at their brain"

    def show_brain(
        self,
        layers: Sequence[np.ndarray],
        weights: Sequence[np.ndarray],
        masks: Optional[Sequence[np.ndarray]] = None,
        groups=INPUT_GROUPS,
        outputs=OUTPUT_NAMES,
        notes: Sequence[str] = (),
        options: Optional[Sequence[tuple]] = None,
    ) -> None:
        self.layers, self.weights = list(layers), list(weights)
        self.groups, self.outputs, self.notes, self.options = groups, outputs, notes, options
        self.masks = None if masks is None else list(masks)
        self.message = ""
        self.update()

    def clear(self, message: str) -> None:
        self.layers, self.weights, self.message = [], [], message
        self.update()

    def _grouped(self) -> tuple:
        """The inputs one value a group (how much is in it), and what each group carries
        into the first hidden layer now: (activity (groups,), carried (hidden, groups))."""
        inputs, weights = self.layers[0], self.weights[0]
        carried = weights * inputs[None, :]
        if self.masks is not None:
            carried = carried * self.masks[0]
        bounds = np.cumsum([0] + [count for _, count in self.groups])
        activity = np.array([np.abs(inputs[a:b]).sum() for a, b in zip(bounds, bounds[1:])])
        summed = np.stack([carried[:, a:b].sum(axis=1) for a, b in zip(bounds, bounds[1:])], 1)
        return activity, summed

    def _places(self, counts: Sequence[int]) -> List[np.ndarray]:
        """Where each node of each layer is drawn: (n, 2) per layer."""
        width, height = self.width(), self.height()
        right_room = OUTPUT_LABELS if self.options is not None else 190.0
        left, right, top, bottom = LABELS, width - right_room, 34.0, height - 14.0
        columns = np.linspace(left, right, len(counts))
        places = []
        for k, n in enumerate(counts):
            if n > 1:
                margin = 6.0 if k == 0 else 12.0
                ys = np.linspace(top + margin, bottom - margin, n)
            else:
                ys = np.array([(top + bottom) / 2])
            places.append(np.column_stack([np.full(n, columns[k]), ys]))
        return places

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), GROUND)
        if not self.layers:
            painter.setPen(TEXT)
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.message)
            return
        activity, first = self._grouped()
        shown = [activity] + list(self.layers[1:-1])
        outputs = np.array([self.options[0][1]]) if self.options is not None else self.layers[-1]
        places = self._places([len(a) for a in shown] + [len(outputs)])
        # The connections that carry the most now: weight times what flows into them
        flows = [first]
        for k in range(1, len(self.weights)):
            carried = self.weights[k] * self.layers[k][None, :]
            if self.masks is not None:
                carried = carried * self.masks[k]
            flows.append(carried)
        for k, carried in enumerate(flows):
            flat = np.abs(carried).ravel()
            if not flat.any():
                continue
            chosen = np.argsort(-flat)[:STRONGEST]
            top = flat[chosen[0]]
            for index in chosen[::-1]:
                j, i = divmod(int(index), carried.shape[1])
                strength = flat[index] / top
                colour = QColor(UP if carried[j, i] > 0 else DOWN)
                colour.setAlpha(int(30 + 200 * strength))
                painter.setPen(QPen(colour, 0.5 + 2.5 * strength))
                a, b = places[k][i], places[k + 1][j]
                painter.drawLine(QPointF(*a), QPointF(*b))
        # The nodes, as bright as they are active
        painter.setPen(Qt.NoPen)
        for k, (layer, place) in enumerate(zip(shown, places)):
            top = float(np.abs(layer).max()) or 1.0
            radius = 5.0 if k == 0 else (3.0 if len(layer) > 20 else 6.0)
            for value, (x, y) in zip(layer, place):
                share = min(1.0, abs(float(value)) / top)
                colour = QColor(255, 214, 64)
                colour.setAlpha(int(50 + 205 * share))
                painter.setBrush(colour)
                painter.drawEllipse(QPointF(x, y), radius, radius)
        self._draw_outputs(painter, outputs, places[-1])
        # What the inputs are, and what they say now
        small = QFont(painter.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1.0))
        painter.setFont(small)
        for g, ((name, _), (x, y)) in enumerate(zip(self.groups, places[0])):
            note = self.notes[g] if g < len(self.notes) else ""
            painter.setPen(TEXT)
            painter.drawText(
                QRectF(4, y - 8, LABELS - NOTES - 18, 16), Qt.AlignRight | Qt.AlignVCenter, name
            )
            if note:
                painter.setPen(FAINT)
                painter.drawText(
                    QRectF(LABELS - NOTES - 6, y - 8, NOTES - 6, 16),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    note,
                )
        painter.setFont(QFont(self.font()))
        painter.setPen(TEXT)
        headers = [
            "What it sees",
            "Hidden",
            "Hidden",
            "Its choice" if self.options else "Outputs",
        ]
        for k, place in enumerate(places[: len(headers)]):
            name = headers[k] if k < len(places) - 1 else headers[-1]
            painter.drawText(QRectF(place[0, 0] - 90, 6, 180, 18), Qt.AlignCenter, name)

    def _draw_outputs(self, painter: QPainter, values: np.ndarray, place: np.ndarray) -> None:
        if self.options is None:
            for name, value, (x, y) in zip(self.outputs, values, place):
                colour = QColor(UP if value >= 0 else DOWN)
                painter.setPen(Qt.NoPen)
                painter.setBrush(colour)
                painter.drawEllipse(QPointF(x, y), 6.0, 6.0)
                painter.setPen(TEXT)
                painter.drawText(
                    QRectF(x + 10, y - 8, 180, 16),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    f"{name} {value:+.2f}",
                )
            return
        label, chance = self.options[0]
        x, y = place[0]
        painter.setPen(Qt.NoPen)
        painter.setBrush(CHOSEN)
        painter.drawEllipse(QPointF(x, y), 8.0, 8.0)
        painter.setPen(QPen(CHOSEN, 2))
        end = self.width() - 6.0
        painter.drawLine(QPointF(x + 12, y), QPointF(end, y))
        painter.drawLine(QPointF(end, y), QPointF(end - 8, y - 5))
        painter.drawLine(QPointF(end, y), QPointF(end - 8, y + 5))
        font = QFont(painter.font())
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QRectF(x + 12, y - 44, OUTPUT_LABELS - 16, 40),
            Qt.AlignLeft | Qt.AlignBottom | Qt.TextWordWrap,
            label,
        )
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(
            QRectF(x + 12, y + 4, OUTPUT_LABELS - 16, 18),
            Qt.AlignLeft | Qt.AlignTop,
            f"{chance:.0%} likely",
        )
