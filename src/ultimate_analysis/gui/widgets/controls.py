"""Controls drawn by the app itself: an on/off switch, and the stage timings as bars."""

from collections import deque
from typing import Deque, Dict, List

from PyQt5.QtCore import QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QPainter
from PyQt5.QtWidgets import QCheckBox, QSizePolicy, QWidget

from ..theme import ACCENT, ACCENT_TEXT, CONTROL, EDGE, FAINT_TEXT, TEXT


class ToggleSwitch(QCheckBox):
    """A tick box drawn as a switch: its text on the left, the switch on the right."""

    TRACK = QSize(34, 18)

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(super().sizeHint().width() + self.TRACK.width(), 28)

    def hitButton(self, _position) -> bool:
        return True  # The whole row switches, not only the switch

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(TEXT if self.isEnabled() else FAINT_TEXT))
        painter.drawText(self.rect(), Qt.AlignVCenter | Qt.AlignLeft, self.text())

        track = QRectF(
            self.width() - self.TRACK.width(),
            (self.height() - self.TRACK.height()) / 2,
            self.TRACK.width(),
            self.TRACK.height(),
        )
        painter.setPen(QColor(ACCENT if self.isChecked() or self.underMouse() else EDGE))
        painter.setBrush(QColor(ACCENT if self.isChecked() else CONTROL))
        painter.drawRoundedRect(track, track.height() / 2, track.height() / 2)
        knob = track.height() - 6
        left = track.right() - knob - 3 if self.isChecked() else track.left() + 3
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(ACCENT_TEXT if self.isChecked() else FAINT_TEXT))
        painter.drawEllipse(QRectF(left, track.top() + 3, knob, knob))


class StageBars(QWidget):
    """What each stage of the analysis costs per frame, as bars, and the frame rate.

    Takes the measurements the way the pipeline names them and adds them up to the few
    stages a viewer thinks in. A stage that does not run on every frame (the field model
    runs on every fifth) shows its cost spread over the frames.
    """

    # Stage shown -> the names its measurements come under
    STAGES: Dict[str, tuple] = {
        "Decoding": ("Frame I/O",),
        "Detection": ("Inference",),
        "Camera": ("Camera Motion",),
        "Tracking": ("Tracking",),
        "Possession": ("Possession",),
        "Field": ("Field Segmentation", "Mask Unification", "Line Extraction", "Field Estimate"),
        "Numbers": ("Player ID",),
        "Drawing": ("Visualization",),
        "Top-down": ("Homography", "Top-down"),
        "Display": ("UI Display",),
    }
    FRAMES = 90  # The mean is taken over this many frames
    ROW = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: Dict[str, Deque[float]] = {
            stage: deque(maxlen=self.FRAMES) for stage in [*self.STAGES, "Other"]
        }
        self._frame: Dict[str, float] = {}
        self._totals: Deque[float] = deque(maxlen=self.FRAMES)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip(
            "Milliseconds per frame for each stage, averaged over the last frames, and the\n"
            "frames per second the analysis and display manage together."
        )

    def sizeHint(self) -> QSize:
        return QSize(220, 34 + self.ROW * len(self._shown()))

    def _shown(self) -> List[str]:
        return [stage for stage, values in self._history.items() if values and max(values) > 0.05]

    def begin_frame(self) -> None:
        """A new frame: what was measured for the last one is taken into the means."""
        if self._frame:
            for stage, values in self._history.items():
                values.append(self._frame.get(stage, 0.0))
            self._totals.append(sum(self._frame.values()))
            self._frame = {}
            self.updateGeometry()
            self.update()

    def add_processing_measurement(self, name: str, duration_ms: float) -> None:
        """Add a measurement under the name the pipeline gives it."""
        if name == "Total Runtime":
            return  # The sum of the others, measured once more
        stage = next(
            (
                stage
                for stage, names in self.STAGES.items()
                if any(name.startswith(prefix) for prefix in names)
            ),
            "Other",
        )
        self._frame[stage] = self._frame.get(stage, 0.0) + duration_ms

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        total = sum(self._totals) / len(self._totals) if self._totals else 0.0

        big = QFont(self.font())
        big.setPointSizeF(self.font().pointSizeF() * 1.5)
        big.setBold(True)
        painter.setFont(big)
        painter.setPen(QColor(TEXT))
        rate = f"{1000.0 / total:.0f}" if total > 0 else "–"
        rate_width = painter.fontMetrics().horizontalAdvance(rate)
        painter.drawText(QRectF(0, 0, width, 26), Qt.AlignVCenter | Qt.AlignLeft, rate)
        painter.setFont(self.font())
        painter.setPen(QColor(FAINT_TEXT))
        painter.drawText(
            QRectF(rate_width + 6, 0, width, 28), Qt.AlignVCenter | Qt.AlignLeft, "frames/s"
        )
        painter.drawText(
            QRectF(0, 0, width, 28), Qt.AlignVCenter | Qt.AlignRight, f"{total:.0f} ms a frame"
        )

        means = {
            stage: sum(self._history[stage]) / len(self._history[stage]) for stage in self._shown()
        }
        longest = max(means.values(), default=1.0)
        name_width, value_width = 84, 52
        bar_left, bar_width = name_width, max(10, width - name_width - value_width - 8)
        for row, (stage, mean) in enumerate(means.items()):
            top = 32 + row * self.ROW
            painter.setPen(QColor(TEXT))
            painter.drawText(QRectF(0, top, name_width, self.ROW), Qt.AlignVCenter, stage)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(CONTROL))
            painter.drawRoundedRect(QRectF(bar_left, top + 7, bar_width, 6), 3, 3)
            painter.setBrush(QColor(ACCENT))
            painter.drawRoundedRect(
                QRectF(bar_left, top + 7, max(3.0, bar_width * mean / longest), 6), 3, 3
            )
            painter.setPen(QColor(FAINT_TEXT))
            painter.drawText(
                QRectF(width - value_width, top, value_width, self.ROW),
                Qt.AlignVCenter | Qt.AlignRight,
                f"{mean:.1f} ms",
            )
