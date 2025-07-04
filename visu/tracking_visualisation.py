import logging
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor
import cv2
import numpy as np

logger = logging.getLogger("ultimate_analysis.tracking_visualisation")

def get_pitch_projection_qimage(tracks, frame, label_font_size=10):
    """
    Returns a QImage with a single pitch projection (naive top-down mapping).
    Only visualization logic is present here. Optimized version.
    """
    # Cache the pitch image if tracks haven't changed significantly
    if not hasattr(get_pitch_projection_qimage, '_cache'):
        get_pitch_projection_qimage._cache = {}
    
    # Create cache key based on track positions
    cache_key = hash(tuple(
        (getattr(track, 'track_id', 0), tuple(track.to_ltrb() if hasattr(track, 'to_ltrb') else [0,0,0,0]))
        for track in tracks if hasattr(track, 'is_confirmed') and track.is_confirmed()
    ))
    
    # Check if we can use cached result
    if cache_key in get_pitch_projection_qimage._cache:
        return get_pitch_projection_qimage._cache[cache_key]
    
    pitch_img_naive = draw_pitch_projection_naive(tracks, frame.shape)
    h, w, ch = pitch_img_naive.shape
    qpix = QPixmap.fromImage(QImage(pitch_img_naive.data, w, h, ch * w, QImage.Format_BGR888))
    painter = QPainter(qpix)
    font = QFont()
    font.setPointSize(label_font_size)
    painter.setFont(font)
    frame_h, frame_w = frame.shape[:2]
    img_w, img_h = w, h
    
    # Draw labels for each track
    for track in tracks:
        if not hasattr(track, 'is_confirmed') or not track.is_confirmed():
            continue
        track_id = getattr(track, 'track_id', None)
        ltrb = track.to_ltrb() if hasattr(track, 'to_ltrb') else None
        if ltrb is None or track_id is None:
            continue
        x1, y1, x2, y2 = map(int, ltrb)
        cx = (x1 + x2) // 2
        cy = y2
        px, py = get_pitch_coords_naive(cx, cy, frame.shape, img_w, img_h)
        color = get_color(int(track_id)) if callable(get_color) else (255,255,255)
        painter.setPen(QColor(*color))
        painter.setBrush(QColor(*color))
        painter.drawText(px + 8, py - 8, str(track_id))
    
    # Draw label for projection
    painter.setPen(QColor(255,255,255))
    painter.setFont(QFont(font.family(), label_font_size+2, QFont.Bold))
    painter.drawText(10, 30, "Top-Down Projection (No Camera Model)")
    painter.end()
    
    result = qpix.toImage()
    
    # Cache the result (limit cache size to prevent memory issues)
    if len(get_pitch_projection_qimage._cache) >= 3:  # Reduced cache size
        # Remove oldest entry
        oldest_key = next(iter(get_pitch_projection_qimage._cache))
        del get_pitch_projection_qimage._cache[oldest_key]
    
    get_pitch_projection_qimage._cache[cache_key] = result
    return result



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

import cv2
import numpy as np
from processing.inference import get_class_names

def draw_track_history(frame, tracks, detections, history_length=300):
    """
    Optimized version of draw_track_history with reduced memory allocations
    and improved performance.
    """
    # Get class names once
    class_names = get_class_names()
    gray = (180, 180, 180)
    
    # Early exit if no tracks or detections
    if not tracks and not detections:
        return frame
    
    # Optimize detection drawing - batch process rectangles
    if detections:
        detection_rects = []
        for det in detections:
            # Defensive: handle both tuple and dict detection formats
            if isinstance(det, dict):
                bbox = det.get('bbox', None)
                if bbox is not None and len(bbox) == 4:
                    dx, dy, dw, dh = bbox
                    detection_rects.append((int(dx), int(dy), int(dx + dw), int(dy + dh)))
            else:
                try:
                    (dx, dy, dw, dh), _, _ = det
                    detection_rects.append((int(dx), int(dy), int(dx + dw), int(dy + dh)))
                except Exception:
                    continue
        
        # Draw all detection rectangles
        for x1, y1, x2, y2 in detection_rects:
            try:
                cv2.rectangle(frame, (x1, y1), (x2, y2), gray, 1)
            except Exception:
                continue
    
    # Early exit if no tracks
    if not tracks:
        return frame
    
    # --- Track history logic (optimized) ---
    if not hasattr(draw_track_history, "track_histories"):
        draw_track_history.track_histories = {}
    track_histories = draw_track_history.track_histories
    
    # Batch process tracks
    confirmed_tracks = [track for track in tracks if track.is_confirmed()]
    
    # Early exit if no confirmed tracks
    if not confirmed_tracks:
        return frame
    
    # Pre-compute colors for all tracks
    track_colors = {}
    for track in confirmed_tracks:
        track_id = track.track_id
        if track_id not in track_colors:
            track_colors[track_id] = get_color(int(track_id))
    
    # Update track histories and draw
    for track in confirmed_tracks:
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        color = track_colors[track_id]
        cx = (x1 + x2) // 2
        cy = y2
        
        # Update history
        if track_id not in track_histories:
            track_histories[track_id] = []
        track_histories[track_id].append((cx, cy))
        
        # Keep only the last 'history_length' points
        history = track_histories[track_id][-history_length:]
        track_histories[track_id] = history  # Update the stored history
        
        # Draw history lines (optimized)
        if len(history) > 1:
            # Convert to numpy array for faster processing
            history_array = np.array(history, dtype=np.int32)
            for i in range(len(history_array) - 1):
                pt1 = tuple(history_array[i])
                pt2 = tuple(history_array[i + 1])
                cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Get class name and draw label
        class_name = None
        # Try all possible class attributes for both DeepSort and HistogramTracker tracks
        if hasattr(track, 'cls') and track.cls is not None:
            class_idx = int(track.cls)
            class_name = class_names.get(class_idx, str(track.cls))
        elif hasattr(track, 'class_id') and track.class_id is not None:
            class_idx = int(track.class_id)
            class_name = class_names.get(class_idx, str(track.class_id))
        elif hasattr(track, 'det_class') and track.det_class is not None:
            class_idx = int(track.det_class)
            class_name = class_names.get(class_idx, str(track.det_class))
        
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

def clear_pitch_projection_cache():
    """Clear the pitch projection cache to free memory"""
    if hasattr(get_pitch_projection_qimage, '_cache'):
        get_pitch_projection_qimage._cache.clear()