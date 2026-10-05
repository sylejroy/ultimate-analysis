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
