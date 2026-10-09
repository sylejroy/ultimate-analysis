"""What a player looks like, reduced to the colour of their kit.

Used to keep a returning player from being taken for one of the other team. It cannot
tell teammates apart, and nothing here tries to: measured on game footage, not even a
network trained to recognise people (OSNet) could (the right player in one case out of
four, among sixteen), because the players are about 90 pixels tall and teammates wear the
same kit. Kit colour separates the two teams better than that network did, and costs
almost nothing.
"""

from typing import List, Optional, Sequence

import cv2
import numpy as np

# A box is reduced to this size (width, height); shirt and shorts are the middle columns
# of these rows
SIGNATURE_SIZE = (16, 32)
SHIRT_ROWS = slice(6, 16)
SHORTS_ROWS = slice(16, 22)
MIDDLE_COLUMNS = slice(4, 12)
MIN_BOX_SIZE = (8, 16)  # Smaller boxes show too little of a player


def encode(frame: np.ndarray, boxes: Sequence[Sequence[float]]) -> List[Optional[np.ndarray]]:
    """Kit colour of the player in each box.

    Args:
        frame: Video frame (BGR)
        boxes: Boxes (x1, y1, x2, y2) in frame pixels

    Returns:
        Per box the mean colour of shirt and of shorts in the Lab colour space (six
        numbers; equal distances there are about equally visible differences), or None
        for a box that is too small.
    """
    frame_h, frame_w = frame.shape[:2]
    signatures: List[Optional[np.ndarray]] = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame_w, int(x2)), min(frame_h, int(y2))
        if x2 - x1 < MIN_BOX_SIZE[0] or y2 - y1 < MIN_BOX_SIZE[1]:
            signatures.append(None)
            continue
        small = cv2.resize(frame[y1:y2, x1:x2], SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        shirt = lab[SHIRT_ROWS, MIDDLE_COLUMNS].reshape(-1, 3).mean(axis=0)
        shorts = lab[SHORTS_ROWS, MIDDLE_COLUMNS].reshape(-1, 3).mean(axis=0)
        signatures.append(np.concatenate((shirt, shorts)))
    return signatures


def shirt_colour(frame: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    """The colour of the shirt in a box (BGR): the middle value of what the shirt's part
    of the box shows, for showing a team by its colour.

    The mean of that part is dulled by what else is in it: grass beside a slim player,
    the number, a sleeve. The middle value is the shirt as long as the shirt is most of
    it, and it does not have to tell grass from a green shirt, which taking the grass
    out by its colour had to and could not (a team in green was shown grey, or blue).
    None for a box that is too small.
    """
    frame_h, frame_w = frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(frame_w, int(box[2])), min(frame_h, int(box[3]))
    if x2 - x1 < MIN_BOX_SIZE[0] or y2 - y1 < MIN_BOX_SIZE[1]:
        return None
    small = cv2.resize(frame[y1:y2, x1:x2], SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)
    return np.median(small[SHIRT_ROWS, MIDDLE_COLUMNS].reshape(-1, 3).astype(np.float64), axis=0)


# A shirt with less colour than this (distance from grey in Lab, where 128 is none) is
# white, grey or black, and what tint it has comes from the light: the sky, the grass.
# Of the shirts of ten games, the white, grey and black ones measured up to 17, the
# coloured ones (blue, red, orange) from 21. Below this a tint must not be made much of:
# shown vivid, a white team was lime green and a black one yellow or pink.
COLOURLESS_CHROMA = 20.0
# A shirt without colour is a light one from this lightness (Lab, 0 to 255), a dark one below
LIGHT_FROM = 110.0


def shirt_kind(colour: Sequence[float]) -> str:
    """Whether a shirt colour (BGR) is "light", "dark" or "coloured"."""
    pixel = np.clip(np.asarray(colour, dtype=np.float64), 0, 255).astype(np.uint8)
    lightness, a, b = (
        float(v) for v in cv2.cvtColor(pixel.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0]
    )
    if np.hypot(a - 128.0, b - 128.0) >= COLOURLESS_CHROMA:
        return "coloured"
    return "light" if lightness >= LIGHT_FROM else "dark"


def told_apart(colours: dict) -> dict:
    """Two teams' shirt colours {0: BGR, 1: BGR} such that they are not shown alike: two
    teams without colour, both light or both dark (white against grey, say), are given
    as white and black, the lighter and the darker."""
    if len(colours) != 2:
        return colours
    kinds = [shirt_kind(colours[team]) for team in (0, 1)]
    if kinds[0] == kinds[1] != "coloured":
        lighter = int(sum(colours[1]) > sum(colours[0]))
        colours = dict(colours)
        colours[lighter], colours[1 - lighter] = (235, 235, 235), (45, 45, 45)
    return colours


def box_overlaps(boxes: np.ndarray) -> np.ndarray:
    """IoU of every box (x1, y1, x2, y2) with every other: (N, N), zero on the diagonal."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    shared = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    overlap = shared / (area[:, None] + area[None, :] - shared + 1e-9)
    np.fill_diagonal(overlap, 0.0)
    return overlap
