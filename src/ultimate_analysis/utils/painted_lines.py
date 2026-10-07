"""The painted lines of a field, found in a frame by their look.

A painted line is a thin streak that is whiter than the grass on both sides of it. Each
pixel gets a strength for being the middle of such a streak (at three widths: lines far
from the camera are a pixel or two wide, near ones ten), held against the strength the
grass around it gives, since grass in the sun is streaky too. What is left is kept where
it forms long thin pieces: lines, and not shirts, tents, or lettering.

Faint lines far away are found in parts or not at all, and the edge of a road or the sky
line above the trees comes out as well. It is a guide for the eye, not a measurement.
"""

import cv2
import numpy as np

# Widths looked for, as the blur (in pixels) at which a line of that width stands out most
LINE_SCALES = (1.2, 2.5, 4.5)
# A streak counts if it is this many times stronger than the average around it
MIN_CONTRAST = 3.5
# The size of "around it", in pixels
SURROUNDINGS = 61
# Pieces shorter than this are not lines
MIN_LENGTH = 70


def _ridge_strength(white: np.ndarray, scale: float) -> np.ndarray:
    """How much each pixel is the middle of a bright streak of the scale's width."""
    smooth = cv2.GaussianBlur(white, (0, 0), scale)
    xx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
    yy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
    xy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
    # The lower curvature of the two: strongly negative across a bright streak
    lower = (xx + yy) / 2 - np.sqrt(((xx - yy) / 2) ** 2 + xy**2)
    return np.maximum(-lower, 0) * scale**2


def painted_line_mask(frame: np.ndarray) -> np.ndarray:
    """Where a frame (BGR) shows painted lines: a mask of the frame's size, 255 on a line."""
    # White is high in all three colours; grass is not
    white = frame.min(axis=2).astype(np.float32)
    found = np.zeros(frame.shape[:2], dtype=np.uint8)
    for scale in LINE_SCALES:
        strength = _ridge_strength(white, scale)
        around = cv2.blur(strength, (SURROUNDINGS, SURROUNDINGS)) + 1.0
        streaks = (strength > MIN_CONTRAST * around).astype(np.uint8)
        _, pieces, stats, _ = cv2.connectedComponentsWithStats(streaks, 8)
        width, height, area = stats[:, 2], stats[:, 3], stats[:, 4]
        longer = np.maximum(width, height)
        # Long, and thin: little of the box it spans, or just a strip along it
        is_line = (longer >= MIN_LENGTH) & (area <= 0.3 * width * height + (2 + 2 * scale) * longer)
        is_line[0] = False  # The background
        found[is_line[pieces]] = 255
    return found
