import cv2
import numpy as np

def draw_player_id(vis_frame, bbox, digit_str, digits=None, color=(0, 255, 255), ocr_boxes=None):
    """
    Draws the player ID (jersey number) and optionally digit bounding boxes on the frame.

    Args:
        vis_frame: The frame to draw on (BGR, np.ndarray).
        bbox: [x, y, w, h] of the player in the frame.
        digit_str: The detected player ID string.
        digits: List of (box, class) tuples for each digit (optional, for YOLO).
        color: Color for text and bbox (default: yellow).
        ocr_boxes: List of bounding boxes from easyocr (relative to the crop), optional.
    """
    x, y, w, h = [int(v) for v in bbox]
    # Draw the player's bounding box
    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
    # Draw the player ID string to the right of the bbox
    if digit_str:
        cv2.putText(
            vis_frame,
            digit_str,
            (x + w + 10, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )
    # Optionally, draw digit bounding boxes if using YOLO
    if digits:
        for db, dc in digits:
            dx1, dy1, dx2, dy2 = db
            cv2.rectangle(
                vis_frame,
                (x + dx1, y + dy1),
                (x + dx2, y + dy2),
                (0, 255, 0),
                1
            )
    # Draw EasyOCR digit bounding boxes if provided
    if ocr_boxes:
        for box in ocr_boxes:
            # Each box is a list of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            pts = np.array(box, dtype=np.int32)
            pts[:, 0] += x  # shift x by bbox x
            pts[:, 1] += y  # shift y by bbox y
            cv2.polylines(vis_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    return vis_frame