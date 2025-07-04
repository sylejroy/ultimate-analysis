def get_pitch_projection_qimage(tracks, frame, label_font_size=10):
    """
    Returns a QImage of the pitch projection, rotated 180deg, with upright labels for each track.
    - tracks: list of track objects
    - frame: current video frame (numpy array)
    - label_font_size: font size for track labels
    """
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor
    import cv2
    pitch_img = draw_pitch_projection(tracks, frame.shape)
    h2, w2, ch2 = pitch_img.shape
    qpix = QPixmap.fromImage(QImage(pitch_img.data, w2, h2, ch2 * w2, QImage.Format_BGR888))
    painter = QPainter(qpix)
    font = QFont()
    font.setPointSize(label_font_size)
    painter.setFont(font)
    frame_h, frame_w = frame.shape[:2]
    img_w, img_h = w2, h2
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx = (x1 + x2) // 2
        cy = y2
        # Direct mapping: x (frame) -> x (pitch), y (frame) -> y (pitch)
        px = int(cx / frame_w * img_w)
        py = int(cy / frame_h * img_h)
        color = get_color(int(track_id))
        painter.setPen(QColor(*color))
        painter.setBrush(QColor(*color))
        painter.drawText(px + 8, py - 8, str(track_id))
    painter.end()
    return qpix.toImage()
def draw_pitch_projection(tracks, frame_shape, pitch_size=(100, 37), history_length=300, image_size=(600, 222)):
    """
    Projects player tracks onto a 2D pitch and returns an image with the pitch and tracks drawn.
    - tracks: list of track objects (must have .to_ltrb() and .track_id)
    - frame_shape: shape of the video frame (h, w, c)
    - pitch_size: (length, width) in meters
    - history_length: how many points to show per track
    - image_size: (width, height) of the pitch image in pixels
    """
    import cv2
    import numpy as np
    # Set pitch image so that length is vertical (long axis = vertical)
    pitch_length, pitch_width = pitch_size
    # Default image_size is (600, 222) (w, h), but for vertical pitch, swap to (width, length)
    img_h = 600  # vertical (length)
    img_w = 222  # horizontal (width)
    pitch_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    # Draw pitch outline
    cv2.rectangle(pitch_img, (0, 0), (img_w-1, img_h-1), (255, 255, 255), 2)
    # Center line (horizontal across the middle)
    cv2.line(pitch_img, (0, img_h//2), (img_w-1, img_h//2), (200, 200, 200), 1)
    # End zones (18m each side for ultimate, vertical)
    endzone_py = int(18 / pitch_length * img_h)
    cv2.rectangle(pitch_img, (0, 0), (img_w-1, endzone_py), (100, 100, 255), 1)
    cv2.rectangle(pitch_img, (0, img_h-endzone_py), (img_w-1, img_h-1), (100, 100, 255), 1)
    # Map from image (frame) to pitch: x (frame) -> x (pitch), y (frame) -> y (pitch)
    frame_h, frame_w = frame_shape[:2]
    def img_to_pitch_coords(x, y):
        px = int(x / frame_w * img_w)
        py = int(y / frame_h * img_h)
        return px, py
    # Draw tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        # Use bottom middle of bbox
        cx = (x1 + x2) // 2
        cy = y2
        # Get history if available
        history = []
        if hasattr(draw_track_history, "track_histories"):
            track_histories = draw_track_history.track_histories
            if track_id in track_histories:
                history = track_histories[track_id][-history_length:]
        else:
            history = [(cx, cy)]
        color = get_color(int(track_id))
        # Draw history
        for i in range(len(history) - 1):
            pt1 = img_to_pitch_coords(*history[i])
            pt2 = img_to_pitch_coords(*history[i+1])
            cv2.line(pitch_img, pt1, pt2, color, 2)
        # Draw current position
        px, py = img_to_pitch_coords(cx, cy)
        cv2.circle(pitch_img, (px, py), 6, color, -1)
        # Do NOT draw text label here; label will be drawn after in get_pitch_projection_qimage
    return pitch_img
import cv2
import numpy as np
from processing.inference import get_class_names

def draw_track_history(frame, tracks, detections, history_length=300):
    class_names = get_class_names()
    gray = (180, 180, 180)
    for det in detections:
        # Defensive: handle both tuple and dict detection formats, and restore class/confidence
        if isinstance(det, dict):
            bbox = det.get('bbox', None)
            if bbox is not None and len(bbox) == 4:
                dx, dy, dw, dh = bbox
            else:
                continue
        else:
            try:
                (dx, dy, dw, dh), _, _ = det
            except Exception:
                continue
        try:
            # Only draw bbox in gray, no label
            cv2.rectangle(frame, (int(dx), int(dy)), (int(dx + dw), int(dy + dh)), gray, 1)
        except Exception:
            continue
    # --- Track history logic (no optical flow compensation) ---
    if not hasattr(draw_track_history, "track_histories"):
        draw_track_history.track_histories = {}
    track_histories = draw_track_history.track_histories
    # Now update with new detections
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        color = get_color(int(track_id))
        cx = (x1 + x2) // 2
        cy = y2
        if track_id not in track_histories:
            track_histories[track_id] = []
        track_histories[track_id].append((cx, cy))
        # Keep only the last 'history_length' points
        history = track_histories[track_id][-history_length:]
        for i in range(len(history) - 1):
            pt1 = (int(history[i][0]), int(history[i][1]))
            pt2 = (int(history[i + 1][0]), int(history[i + 1][1]))
            cv2.line(frame, pt1, pt2, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Always draw class name below the bbox, in the same color as the bbox
        class_name = None
        # Try all possible class attributes for both DeepSort and HistogramTracker tracks
        if hasattr(track, 'cls') and track.cls is not None:
            class_idx = int(track.cls)
            class_name = class_names[class_idx] if class_idx < len(class_names) else str(track.cls)
        elif hasattr(track, 'class_id') and track.class_id is not None:
            class_idx = int(track.class_id)
            class_name = class_names[class_idx] if class_idx < len(class_names) else str(track.class_id)
        elif hasattr(track, 'det_class') and track.det_class is not None:
            class_idx = int(track.det_class)
            class_name = class_names[class_idx] if class_idx < len(class_names) else str(track.det_class)
        if class_name is not None:
            label = f"{class_name}"
            if hasattr(track, 'conf') and track.conf is not None:
                label += f" {track.conf:.2f}"
            cv2.putText(frame, label, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            # Fallback: show track id if no class
            cv2.putText(frame, str(track_id), (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame

def get_color(idx):
    np.random.seed(idx)
    color = tuple(int(x) for x in np.random.randint(0, 255, 3))
    return color

def reset_track_histories():
    if hasattr(draw_track_history, "track_histories"):
        draw_track_history.track_histories = {}