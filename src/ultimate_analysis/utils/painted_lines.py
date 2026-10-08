"""The painted lines of a field, found in a frame by their look.

A painted line is a thin streak that is whiter than the grass on both sides of it. Each
pixel gets a strength for being the middle of such a streak (at three widths: lines far
from the camera are a pixel or two wide, near ones ten), held against the strength the
grass around it gives, since grass in the sun is streaky too. What is left is kept where
it forms long thin pieces: lines, and not shirts, tents, or lettering. A player's arm,
leg or shirt stripe is long and thin as well, so a piece that lies mostly on players is
left out when the players' boxes are given.

Faint lines far away are found in parts or not at all, and the edge of a road or the sky
line above the trees comes out as well. It is a guide for the eye, not a measurement.
"""

from typing import Optional, Sequence

import cv2
import numpy as np

# Widths looked for, as the blur (in pixels) at which a line of that width stands out most
LINE_SCALES = (1.2, 2.5, 4.5)
# A streak counts if it is this many times stronger than the average around it. Lower
# finds more of the faint far lines and more that is no line: on the labelled field
# frames, 3.5 found 40% of the field's lines, 3.0 finds 47% and marks 1.7 times as much
# elsewhere; 2.4 with shorter pieces finds 64% and marks five times as much.
MIN_CONTRAST = 3.0
# The size of "around it", in pixels
SURROUNDINGS = 61
# Pieces shorter than this are not lines
MIN_LENGTH = 70
# A piece with more than this share of it on players is part of a player. Their boxes are
# widened by a share of their height: an arm or a foot reaches out of the box.
MAX_ON_PLAYERS = 0.5
PLAYER_PAD = 0.1
# A line drawn through two points snaps onto a painted line only if that many of its
# pixels lie on painted ones, and that share of what the frame shows of it: a long line
# that is found along much of its length, not a few stray streaks
SNAP_MIN_OVERLAP = 150
SNAP_MIN_SHARE = 0.25
SNAP_STEP = 0.25  # Pixels between the positions tried


def _ridge_strength(white: np.ndarray, scale: float) -> np.ndarray:
    """How much each pixel is the middle of a bright streak of the scale's width."""
    smooth = cv2.GaussianBlur(white, (0, 0), scale)
    xx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
    yy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
    xy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
    # The lower curvature of the two: strongly negative across a bright streak
    lower = (xx + yy) / 2 - np.sqrt(((xx - yy) / 2) ** 2 + xy**2)
    return np.maximum(-lower, 0) * scale**2


def painted_line_mask(
    frame: np.ndarray, players: Optional[Sequence[Sequence[float]]] = None
) -> np.ndarray:
    """Where a frame (BGR) shows painted lines: a mask of the frame's size, 255 on a line.

    Args:
        frame: The picture
        players: Boxes (x1, y1, x2, y2) of the players in it, if known. A piece that lies
            mostly within them is part of a player and no line. A line that runs past a
            player's feet lies mostly outside and stays, with a gap where the player is:
            a leg or a stripe that touches the line there would count as part of it.
    """
    on_players = np.zeros(frame.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in players or ():
        pad = PLAYER_PAD * (y2 - y1)
        cv2.rectangle(
            on_players,
            (int(x1 - pad), int(y1 - pad)),
            (int(x2 + pad), int(y2 + pad)),
            1,
            -1,
        )
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
        if players:
            inside = np.bincount(pieces.ravel(), weights=on_players.ravel(), minlength=len(area))
            is_line &= inside <= MAX_ON_PLAYERS * area
        is_line[0] = False  # The background
        found[is_line[pieces]] = 255
    found[on_players > 0] = 0
    return found


def snap_onto_line(
    mask: np.ndarray, anchor: Sequence[float], point: Sequence[float], reach: float
) -> Optional[np.ndarray]:
    """Move a point sideways so that the line through it and an anchor lies on a painted line.

    The anchor stays; the point keeps how far along the line it is and moves across it,
    by at most `reach` pixels, to where the line covers the most painted pixels.

    Args:
        mask: Where the painted lines are (painted_line_mask)
        anchor: A pixel the line goes through, which stays
        point: The pixel that is being placed
        reach: How far the point may be moved, in pixels

    Returns:
        The moved point, or None if no long painted line lies within reach
    """
    anchor, point = np.asarray(anchor, dtype=np.float64), np.asarray(point, dtype=np.float64)
    length = float(np.linalg.norm(point - anchor))
    if length < 5.0 or reach <= 0:
        return None
    along = (point - anchor) / length
    across = np.array([-along[1], along[0]])
    offsets = np.arange(-reach, reach + 1e-9, SNAP_STEP)
    targets = point + offsets[:, None] * across
    directions = (targets - anchor) / np.linalg.norm(targets - anchor, axis=1, keepdims=True)

    # A pixel either side of a painted line still counts as on it
    near_lines = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8)) > 0
    height, width = mask.shape
    steps = np.arange(-float(np.hypot(width, height)), float(np.hypot(width, height)), 1.0)
    pixels = np.rint(anchor + steps[None, :, None] * directions[:, None, :]).astype(np.int64)
    inside = (
        (pixels[..., 0] >= 0)
        & (pixels[..., 0] < width)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < height)
    )
    on_lines = np.zeros(inside.shape, dtype=bool)
    on_lines[inside] = near_lines[pixels[..., 1][inside], pixels[..., 0][inside]]
    overlap = on_lines.sum(axis=1)
    shown = np.maximum(inside.sum(axis=1), 1)

    # The most covered; of equally covered ones the nearest to where the point was put
    best = int(np.argmax(overlap - 0.01 * np.abs(offsets)))
    if overlap[best] < SNAP_MIN_OVERLAP or overlap[best] / shown[best] < SNAP_MIN_SHARE:
        return None
    return targets[best]
