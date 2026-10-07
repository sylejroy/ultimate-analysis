"""Field lines that stay on the field from frame to frame.

The lines are fitted to the field's outline each time the field is segmented, every few
frames. In between the camera keeps moving, and each new fit differs a little from the
last because the outline does. Shown as they come, the lines stand still while the
picture pans and then jump. Here they are moved with the camera between fits, and a new
fit is blended into the lines shown rather than replacing them.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting

# A new line is the same as a shown one if it runs within this angle of it and its middle
# is this close to it (pixels)
MAX_ANGLE_DEGREES = 12.0
MAX_OFFSET = 60.0
# How much of the way to a new fit a line moves at once; the rest at the next fits
NEW_FIT_WEIGHT = 0.4
# A line the new fit does not have is kept for this many fits (the outline may miss it once)
KEEP_FITS = 1
# A line that was not shown before is shown once this many fits in a row have it; a line
# found by one fit only is mostly a dent in the outline. With no lines shown at all, the
# first fit's are shown at once.
CONFIRM_FITS = 2

Line = np.ndarray  # (2, 2): the two end points, in picture pixels


def _angle(line: Line) -> float:
    """Direction of a line in degrees, between 0 and 180."""
    dx, dy = line[1] - line[0]
    return float(np.degrees(np.arctan2(dy, dx)) % 180.0)


def _offset(point: np.ndarray, line: Line) -> float:
    """Distance of a point from the (endless) line through a segment."""
    direction = line[1] - line[0]
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return float(np.linalg.norm(point - line[0]))
    cross = direction[0] * (point[1] - line[0][1]) - direction[1] * (point[0] - line[0][0])
    return abs(float(cross)) / length


def _difference(old: Line, new: Line) -> Optional[float]:
    """How different two lines are (0 = the same), or None if they are different lines."""
    angle = abs(_angle(old) - _angle(new))
    angle = min(angle, 180.0 - angle)
    offset = _offset(new.mean(axis=0), old)
    if angle > MAX_ANGLE_DEGREES or offset > MAX_OFFSET:
        return None
    return angle / MAX_ANGLE_DEGREES + offset / MAX_OFFSET


class FieldLineFilter:
    """The field lines to show, smoothed over the fits and moved with the camera."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget the lines (another video, a seek)."""
        self._lines: List[Line] = []
        self._confidences: List[float] = []
        self._missed: List[int] = []
        self._candidates: List[Tuple[Line, int]] = []  # New lines not shown yet: (line, fits seen)

    def move(self, camera_motion: np.ndarray) -> None:
        """Move the lines with the picture (homography from the last frame to this one)."""

        def moved(lines: List[Line]) -> List[Line]:
            if not lines:
                return lines
            points = np.array(lines, dtype=np.float32).reshape(-1, 1, 2)
            result = cv2.perspectiveTransform(points, camera_motion).reshape(-1, 2, 2)
            return [line.astype(np.float64) for line in result]

        self._lines = moved(self._lines)
        self._candidates = list(
            zip(moved([line for line, _ in self._candidates]), [n for _, n in self._candidates])
        )

    def update(self, lines: List[Line], confidences: List[float]) -> None:
        """Take in the lines of a new fit."""
        new = [np.asarray(line, dtype=np.float64).reshape(2, 2) for line in lines]
        if not get_setting("models.segmentation.contour.ransac.smooth_lines", True):
            self._lines, self._confidences = new, list(confidences)
            self._missed = [0] * len(new)
            return

        # Pair new lines with shown ones, most alike first
        pairs = sorted(
            (difference, old_index, new_index)
            for old_index, old in enumerate(self._lines)
            for new_index, line in enumerate(new)
            if (difference := _difference(old, line)) is not None
        )
        old_taken, new_taken = set(), set()
        lines_out, confidences_out, missed_out = [], [], []
        for _, old_index, new_index in pairs:
            if old_index in old_taken or new_index in new_taken:
                continue
            old_taken.add(old_index)
            new_taken.add(new_index)
            old, line = self._lines[old_index], new[new_index]
            if np.dot(old[1] - old[0], line[1] - line[0]) < 0:
                line = line[::-1]  # The same end first
            lines_out.append((1 - NEW_FIT_WEIGHT) * old + NEW_FIT_WEIGHT * line)
            confidences_out.append(confidences[new_index])
            missed_out.append(0)

        # Lines the new fit has not got are kept a little; new lines are shown at once
        for index, old in enumerate(self._lines):
            if index not in old_taken and self._missed[index] < KEEP_FITS:
                lines_out.append(old)
                confidences_out.append(self._confidences[index])
                missed_out.append(self._missed[index] + 1)
        candidates_out = []
        show_at_once = not self._lines
        for index, line in enumerate(new):
            if index in new_taken:
                continue
            # Seen by the fits before as well?
            seen = 1 + max(
                (n for old, n in self._candidates if _difference(old, line) is not None),
                default=0,
            )
            if show_at_once or seen >= CONFIRM_FITS:
                lines_out.append(line)
                confidences_out.append(confidences[index])
                missed_out.append(0)
            else:
                candidates_out.append((line, seen))
        self._candidates = candidates_out

        self._lines, self._confidences, self._missed = lines_out, confidences_out, missed_out

    def current(self) -> Tuple[List[Line], List[float]]:
        """The lines to show now, and their confidences."""
        return [line.copy() for line in self._lines], list(self._confidences)
