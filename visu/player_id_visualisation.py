import cv2

def draw_player_id(vis_frame, bbox, digit_str, digits=None, color=(0, 255, 255)):
    """
    Draws the player ID (jersey number) and optionally digit bounding boxes on the frame.

    Args:
        vis_frame: The frame to draw on (BGR, np.ndarray).
        bbox: [x, y, w, h] of the player in the frame.
        digit_str: The detected player ID string.
        digits: List of (box, class) tuples for each digit (optional, for YOLO).
        color: Color for text and bbox (default: yellow).
    """
    x, y, w, h = bbox
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
            # Offset digit box to the correct position in the full frame
            cv2.rectangle(
                vis_frame,
                (x + dx1, y + dy1),
                (x + dx2, y + dy2),
                (0, 255, 0),
                1
            )
    return vis_frame