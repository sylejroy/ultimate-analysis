"""A simulated player's brain, as it works in the present step: the network's inputs
grouped by kind, its two hidden layers and its outputs, each node as bright as it is
active, and the connections that carry the most of the signal now, green where they push
up and red where they push down. Connections a brain has cut are not there to draw."""

from typing import List, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from ...simulation.learned import INPUT_GROUPS, OUTPUT_NAMES

STRONGEST = 40  # Connections drawn between two layers
GROUND = QColor(22, 24, 30)
TEXT = QColor(200, 204, 210)
UP, DOWN = QColor(90, 220, 120), QColor(235, 90, 80)


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
        self.outputs = OUTPUT_NAMES
        self.message = "Click a player to look at their brain"

    def show_brain(
        self,
        layers: Sequence[np.ndarray],
        weights: Sequence[np.ndarray],
        masks: Optional[Sequence[np.ndarray]] = None,
        groups=INPUT_GROUPS,
        outputs=OUTPUT_NAMES,
    ) -> None:
        self.layers, self.weights = list(layers), list(weights)
        self.groups, self.outputs = groups, outputs
        self.masks = None if masks is None else list(masks)
        self.message = ""
        self.update()

    def clear(self, message: str) -> None:
        self.layers, self.weights, self.message = [], [], message
        self.update()

    def _places(self) -> List[np.ndarray]:
        """Where each node of each layer is drawn: (n, 2) per layer."""
        width, height = self.width(), self.height()
        left, right, top, bottom = 150.0, width - 90.0, 14.0, height - 14.0
        columns = np.linspace(left, right, len(self.layers))
        places = []
        for k, layer in enumerate(self.layers):
            n = len(layer)
            if k == 0:
                # The inputs by kind, a gap between kinds
                gaps = len(self.groups) - 1
                step = (bottom - top) / (n + 3 * gaps)
                ys, y = [], top
                for _, count in self.groups:
                    for _ in range(count):
                        ys.append(y)
                        y += step
                    y += 3 * step
                ys = np.array(ys[:n])
            else:
                ys = (
                    np.linspace(top + 20, bottom - 20, n)
                    if n > 1
                    else np.array([(top + bottom) / 2])
                )
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
        places = self._places()
        # The connections that carry the most now: weight times what flows into them
        for k, weights in enumerate(self.weights):
            if k + 1 >= len(places):
                break
            carried = weights * self.layers[k][None, :]
            if self.masks is not None:
                carried = carried * self.masks[k]
            flat = np.abs(carried).ravel()
            if not flat.any():
                continue
            chosen = np.argsort(-flat)[:STRONGEST]
            top = flat[chosen[0]]
            for index in chosen:
                j, i = divmod(int(index), carried.shape[1])
                strength = flat[index] / top
                colour = QColor(UP if carried[j, i] > 0 else DOWN)
                colour.setAlpha(int(40 + 200 * strength))
                painter.setPen(QPen(colour, 0.5 + 2.5 * strength))
                a, b = places[k][i], places[k + 1][j]
                painter.drawLine(QPointF(*a), QPointF(*b))
        # The nodes, as bright as they are active
        for k, (layer, place) in enumerate(zip(self.layers, places)):
            top = float(np.abs(layer).max()) or 1.0
            radius = 3.0 if len(layer) > 20 else 6.0
            for value, (x, y) in zip(layer, place):
                share = min(1.0, abs(float(value)) / top)
                colour = (
                    QColor(UP if value >= 0 else DOWN)
                    if k == len(self.layers) - 1
                    else QColor(255, 214, 64)
                )
                colour.setAlpha(int(50 + 205 * share))
                painter.setPen(Qt.NoPen)
                painter.setBrush(colour)
                painter.drawEllipse(QPointF(x, y), radius, radius)
        # What the inputs and outputs are
        painter.setPen(TEXT)
        start = 0
        for name, count in self.groups:
            ys = places[0][start : start + count, 1]
            if len(ys):
                painter.drawText(
                    QRectF(4, ys.mean() - 8, 140, 16), Qt.AlignRight | Qt.AlignVCenter, name
                )
            start += count
        for name, value, (x, y) in zip(self.outputs, self.layers[-1], places[-1]):
            painter.drawText(
                QRectF(x + 10, y - 8, 90, 16),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"{name} {value:+.2f}",
            )
        headers = ["Inputs", "Hidden", "Hidden", "Outputs"]
        for k, place in enumerate(places[: len(headers)]):
            painter.drawText(QRectF(place[0, 0] - 40, 0, 80, 14), Qt.AlignCenter, headers[k])
