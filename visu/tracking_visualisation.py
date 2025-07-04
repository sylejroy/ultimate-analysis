def get_pitch_projection_qimage(tracks, frame, label_font_size=10):
    """
    Returns a QImage with two pitch projections stacked vertically:
    - Top: Camera-model-based pitch projection
    - Bottom: Naive top-down pitch projection (assume video is top-down)
    Each with upright labels for each track.
    """
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor
    import cv2
    import numpy as np
    # Draw both projections
    pitch_img_camera = draw_pitch_projection(tracks, frame.shape)
    pitch_img_naive = draw_pitch_projection_naive(tracks, frame.shape)
    # Stack vertically
    h, w, ch = pitch_img_camera.shape
    stacked = np.zeros((h*2, w, ch), dtype=np.uint8)
    stacked[:h] = pitch_img_camera
    stacked[h:] = pitch_img_naive
    # Convert to QImage/QPixmap for label drawing
    qpix = QPixmap.fromImage(QImage(stacked.data, w, h*2, ch * w, QImage.Format_BGR888))
    painter = QPainter(qpix)
    font = QFont()
    font.setPointSize(label_font_size)
    painter.setFont(font)
    frame_h, frame_w = frame.shape[:2]
    img_w, img_h = w, h
    # Draw labels for both projections
    for proj_idx, y_offset in enumerate([0, h]):
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cx = (x1 + x2) // 2
            cy = y2
            if proj_idx == 0:
                # Camera model projection
                px, py = get_pitch_coords_camera(cx, cy, frame.shape, img_w, img_h)
            else:
                # Naive top-down projection
                px, py = get_pitch_coords_naive(cx, cy, frame.shape, img_w, img_h)
            color = get_color(int(track_id))
            painter.setPen(QColor(*color))
            painter.setBrush(QColor(*color))
            painter.drawText(px + 8, py + y_offset - 8, str(track_id))
    # Draw labels for each projection
    painter.setPen(QColor(255,255,255))
    painter.setFont(QFont(font.family(), label_font_size+2, QFont.Bold))
    painter.drawText(10, 30, "Camera Model Projection")
    painter.drawText(10, h + 30, "Naive Top-Down Projection")
    painter.end()
    return qpix.toImage()

def get_pitch_coords_camera(x, y, frame_shape, img_w, img_h):
    # Use the same camera model as draw_pitch_projection, but with 45° tilt
    import numpy as np
    frame_h, frame_w = frame_shape[:2]
    CAMERA_HEIGHT = 7.0
    TILT_DEG = 45.0  # <-- 45 degree camera angle
    TILT_RAD = np.deg2rad(TILT_DEG)
    f = frame_w
    cx, cy = frame_w / 2, frame_h / 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    R = np.array([
        [1, 0, 0],
        [0, np.cos(TILT_RAD), -np.sin(TILT_RAD)],
        [0, np.sin(TILT_RAD),  np.cos(TILT_RAD)]
    ])
    t = np.array([[0], [0], [CAMERA_HEIGHT]])
    def img_to_ground_coords(u, v):
        uv1 = np.array([u, v, 1.0])
        ray = np.linalg.inv(K) @ uv1
        ray = R @ ray
        cam_center = -R.T @ t
        s = -cam_center[2] / ray[2]
        ground_point = cam_center.flatten() + s * ray
        return ground_point[0], ground_point[1]
    # Compute ground bounds
    corners_img = np.array([
        [0, 0],
        [frame_w-1, 0],
        [0, frame_h-1],
        [frame_w-1, frame_h-1]
    ])
    ground_corners = np.array([img_to_ground_coords(x, y) for x, y in corners_img])
    min_xg, max_xg = ground_corners[:,0].min(), ground_corners[:,0].max()
    min_yg, max_yg = ground_corners[:,1].min(), ground_corners[:,1].max()
    xg, yg = img_to_ground_coords(x, y)
    px = int((xg - min_xg) / (max_xg - min_xg) * (img_w-1))
    py = int((yg - min_yg) / (max_yg - min_yg) * (img_h-1))
    px = np.clip(px, 0, img_w-1)
    py = np.clip(py, 0, img_h-1)
    return px, py

def get_pitch_coords_naive(x, y, frame_shape, img_w, img_h):
    # Naive mapping: x (frame) -> x (pitch), y (frame) -> y (pitch)
    frame_h, frame_w = frame_shape[:2]
    px = int(x / frame_w * img_w)
    py = int(y / frame_h * img_h)
    px = np.clip(px, 0, img_w-1)
    py = np.clip(py, 0, img_h-1)
    return px, py

def draw_pitch_projection_naive(tracks, frame_shape, pitch_size=(100, 37), history_length=300, image_size=(600, 222)):
    """
    Projects player tracks onto a 2D pitch using a naive top-down mapping (no camera model).
    """
    import cv2
    import numpy as np
    pitch_length, pitch_width = pitch_size
    img_h = 600
    img_w = 222
    pitch_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    cv2.rectangle(pitch_img, (0, 0), (img_w-1, img_h-1), (255, 255, 255), 2)
    cv2.line(pitch_img, (0, img_h//2), (img_w-1, img_h//2), (200, 200, 200), 1)
    endzone_py = int(18 / pitch_length * img_h)
    cv2.rectangle(pitch_img, (0, 0), (img_w-1, endzone_py), (100, 100, 255), 1)
    cv2.rectangle(pitch_img, (0, img_h-endzone_py), (img_w-1, img_h-1), (100, 100, 255), 1)
    frame_h, frame_w = frame_shape[:2]
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
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
        for i in range(len(history) - 1):
            pt1 = get_pitch_coords_naive(*history[i], frame_shape, img_w, img_h)
            pt2 = get_pitch_coords_naive(*history[i+1], frame_shape, img_w, img_h)
            cv2.line(pitch_img, pt1, pt2, color, 2)
        px, py = get_pitch_coords_naive(cx, cy, frame_shape, img_w, img_h)
        cv2.circle(pitch_img, (px, py), 6, color, -1)
    return pitch_img
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
    # Camera parameters for drone: 7m above ground, 20 deg tilt
    CAMERA_HEIGHT = 7.0  # meters
    TILT_DEG = 20.0
    TILT_RAD = np.deg2rad(TILT_DEG)
    # Estimate intrinsics (assume 1920x1080, focal length ~image width)
    f = frame_w  # or tune as needed
    cx, cy = frame_w / 2, frame_h / 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    # Rotation: tilt about x axis
    R = np.array([
        [1, 0, 0],
        [0, np.cos(TILT_RAD), -np.sin(TILT_RAD)],
        [0, np.sin(TILT_RAD),  np.cos(TILT_RAD)]
    ])
    t = np.array([[0], [0], [CAMERA_HEIGHT]])

    def img_to_ground_coords(u, v):
        uv1 = np.array([u, v, 1.0])
        ray = np.linalg.inv(K) @ uv1
        ray = R @ ray
        cam_center = -R.T @ t
        s = -cam_center[2] / ray[2]
        ground_point = cam_center.flatten() + s * ray
        return ground_point[0], ground_point[1]  # x, y on ground

    # Compute ground coordinates for image corners to get bounds
    corners_img = np.array([
        [0, 0],
        [frame_w-1, 0],
        [0, frame_h-1],
        [frame_w-1, frame_h-1]
    ])
    ground_corners = np.array([img_to_ground_coords(x, y) for x, y in corners_img])
    min_xg, max_xg = ground_corners[:,0].min(), ground_corners[:,0].max()
    min_yg, max_yg = ground_corners[:,1].min(), ground_corners[:,1].max()

    def ground_to_pitch_coords(xg, yg):
        # Normalize ground coordinates to pitch image
        px = int((xg - min_xg) / (max_xg - min_xg) * (img_w-1))
        py = int((yg - min_yg) / (max_yg - min_yg) * (img_h-1))
        # Clamp to image
        px = np.clip(px, 0, img_w-1)
        py = np.clip(py, 0, img_h-1)
        return px, py

    def img_to_pitch_coords(x, y):
        # Use camera model to map image to ground, then to pitch
        xg, yg = img_to_ground_coords(x, y)
        return ground_to_pitch_coords(xg, yg)
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