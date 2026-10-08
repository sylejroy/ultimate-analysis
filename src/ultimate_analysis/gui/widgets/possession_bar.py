"""A bar that shows which team had the disc over the last seconds of playback."""

from collections import deque
from typing import Optional, Tuple

from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QSizePolicy, QWidget

SECONDS = 30.0  # How far back the bar reaches; now is at its right end
BACKGROUND = QColor(42, 42, 42)
UNKNOWN_TEAM = QColor(175, 175, 175)  # Someone has the disc, of a team not yet known
NO_TEAM = QColor(110, 110, 110)  # The disc is in the air or on the ground, and so far nobody had it
TEXT = QColor(150, 150, 150)
DARK_TEXT, LIGHT_TEXT = QColor(20, 20, 20), QColor(245, 245, 245)
# The share of the bar's height that is filled: a held disc fills it, a disc in the air is
# a band in the middle, and one on the ground a line along the bottom
HEIGHTS = {"held": (0.0, 1.0), "air": (0.35, 0.3), "ground": (0.8, 0.2)}


class PossessionBar(QWidget):
    """Who had the disc, frame by frame: the team's colour, full height while a player
    holds it, a band while it flies, and a line along the bottom while it lies on the
    ground."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip(
            "Which team had the disc over the last 30 seconds, in the team's colour.\n"
            "Full height: a player holds it. Band: in the air. Line at the bottom: on the "
            "ground."
        )
        self._frame_rate = 30.0
        # Runs of frames that look the same: [first frame, last frame, colour, state, holder]
        self._runs: deque = deque()
        self._numbers: dict = {}  # Holder (track ID) -> jersey number, as far as read

    def clear(self) -> None:
        self._runs.clear()
        self._numbers.clear()
        self.update()

    def set_frame_rate(self, frames_per_second: float) -> None:
        if frames_per_second and frames_per_second > 0:
            self._frame_rate = float(frames_per_second)

    def add(
        self,
        frame_index: int,
        colour: Optional[Tuple[int, int, int]],
        state: str,
        since: Optional[int] = None,
        holder: Optional[int] = None,
        number: str = "",
    ) -> None:
        """Take in a frame: the colour (BGR) of the team in possession, or None if it is
        not known, where the disc is ("held", "air", "ground"), who holds it (track ID)
        and their jersey number if it has been read; the number is written on the bar.

        A change is confirmed a moment after it happened. `since` is the frame from
        which what is said holds: the bar is put right back to there.
        """
        if holder is not None and number:
            self._numbers[holder] = number  # Read later, it shows on the earlier frames too
        now = [colour, state, holder]
        if self._runs:
            last = self._runs[-1][1]
            if frame_index == last:
                return  # The same frame shown again
            if frame_index < last or frame_index - last > self._frame_rate:
                self._runs.clear()  # A seek: what was before does not lead up to this
        if since is not None and self._runs and self._runs[-1][2:] != now:
            # What was shown from there on was the state before the change
            since = max(since, self._runs[0][0])
            while self._runs and self._runs[-1][0] >= since:
                self._runs.pop()
            if self._runs:
                self._runs[-1][1] = min(self._runs[-1][1], since - 1)
            if since < frame_index:
                self._runs.append([since, frame_index - 1, *now])
        if self._runs and self._runs[-1][2:] == now:
            self._runs[-1][1] = frame_index
        else:
            self._runs.append([frame_index, frame_index, *now])
        oldest = frame_index - SECONDS * self._frame_rate
        while self._runs and self._runs[0][1] < oldest:
            self._runs.popleft()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        if not self._runs:
            painter.setPen(TEXT)
            painter.drawText(self.rect().adjusted(8, 0, 0, 0), 0x0081, "Possession")
            return
        width, height = self.width(), self.height()
        newest = self._runs[-1][1]
        per_frame = width / (SECONDS * self._frame_rate)
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(int(height * 0.62))
        painter.setFont(font)
        for first, last, colour, state, holder in self._runs:
            if colour is not None:
                blue, green, red = colour
                fill = QColor(red, green, blue)
            else:
                fill = UNKNOWN_TEAM if state == "held" else NO_TEAM
            top, share = HEIGHTS.get(state, HEIGHTS["air"])
            left = max(0.0, width - (newest - first + 1) * per_frame)
            right = width - (newest - last) * per_frame
            painter.fillRect(QRectF(left, top * height, right - left, share * height), fill)
            number = self._numbers.get(holder, "")
            if state == "held" and number:
                text = f"#{number}"
                if painter.fontMetrics().horizontalAdvance(text) + 6 <= right - left:
                    # Dark on a light team colour, light on a dark one
                    painter.setPen(DARK_TEXT if fill.lightness() > 140 else LIGHT_TEXT)
                    painter.drawText(QRectF(left, 0, right - left, height), 0x0084, text)
