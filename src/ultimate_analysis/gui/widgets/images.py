"""Converting OpenCV frames for display in Qt."""

import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    """A QPixmap copy of a BGR frame."""
    frame = np.ascontiguousarray(frame)
    height, width = frame.shape[:2]
    # BGR888 is OpenCV's channel order, so no swapped copy is needed. The QImage only
    # wraps the array; the pixmap made from it holds its own pixels.
    image = QImage(frame.data, width, height, 3 * width, QImage.Format_BGR888)
    return QPixmap.fromImage(image)
