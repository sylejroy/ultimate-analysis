"""
Visualization utilities for drawing detections, tracks, and other overlays in the GUI.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any


# --- Color map for unique track colors ---
def get_color(track_id: int) -> tuple:
    """Generate a visually distinct color for a track ID using a color map."""
    try:
        idx = int(track_id) if track_id is not None else 0
    except Exception:
        idx = 0
    # Use tab20 colormap for up to 20, fallback to hsv for more
    if idx < 20:
        cmap = plt.get_cmap('tab20')
        color = cmap(idx % 20)[:3]
    else:
        cmap = plt.get_cmap('hsv')
        color = cmap((idx % 256) / 256.0)[:3]
    # Convert to 0-255 BGR
    color = tuple(int(255 * c) for c in color[::-1])
    return color


# --- Subtle detection visualization ---
def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw subtle detection bounding boxes on the frame."""
    overlay = frame.copy()
    for detection in detections:
        (x, y, w, h), conf, cls = detection
        # Subtle color: light green with alpha
        box_color = (180, 255, 180)  # BGR
        alpha = 0.3
        cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)
        # Draw label background
        label = f"{cls}:{conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(overlay, (x, y - label_size[1] - 5), (x + label_size[0], y), box_color, -1)
        cv2.putText(overlay, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    # Blend overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


# --- Draw tracks with history ---
# Global dictionary to store history for DeepSort tracks
_deepsort_track_history = {}
MAX_HISTORY = 300


def draw_tracks_deepsort(frame: np.ndarray, tracks: list) -> np.ndarray:
    """
    Draw DeepSort tracking results on the frame.
    
    Args:
        frame: Input frame
        tracks: List of DeepSort track objects
    
    Returns:
        Frame with tracking visualization
    """
    global _deepsort_track_history
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        # Get consistent color for track
        color = get_color(track_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID
        label = f"ID:{track_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        # --- Track history for DeepSort ---
        # Use (center_x, bottom_y) for history
        pt = (center_x, y2)
        if track_id not in _deepsort_track_history:
            _deepsort_track_history[track_id] = []
        _deepsort_track_history[track_id].append(pt)
        if len(_deepsort_track_history[track_id]) > MAX_HISTORY:
            _deepsort_track_history[track_id] = _deepsort_track_history[track_id][-MAX_HISTORY:]
        pts = _deepsort_track_history[track_id]
        if len(pts) > 1:
            for i in range(1, len(pts)):
                alpha = 0.2 + 0.6 * (i / len(pts))
                overlay = frame.copy()
                cv2.line(overlay, pts[i-1], pts[i], color, 2)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame


def draw_tracks_histogram(frame: np.ndarray, tracks: list) -> np.ndarray:
    """
    Draw histogram-based tracking results on the frame.
    
    Args:
        frame: Input frame
        tracks: List of histogram tracker Track objects
    
    Returns:
        Frame with tracking visualization
    """
    for track in tracks:
        if not track.confirmed:
            continue
            
        track_id = track.track_id
        x1, y1, x2, y2 = track.bbox
        
        # Get consistent color for track
        color = get_color(track_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID
        label = f"ID:{track_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        # Draw track history if available
        if hasattr(track, 'history') and len(track.history) > 1:
            pts = [((b[0]+b[2])//2, b[3]) for b in track.history]  # (center_x, bottom_y)
            for i in range(1, len(pts)):
                alpha = 0.2 + 0.6 * (i / len(pts))
                overlay = frame.copy()
                cv2.line(overlay, pts[i-1], pts[i], color, 2)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame


def draw_tracks(frame: np.ndarray, tracks: list, tracker_type: str = "deepsort") -> np.ndarray:
    """
    Draw tracking results on the frame based on tracker type.
    
    Args:
        frame: Input frame
        tracks: List of track objects
        tracker_type: Type of tracker ("deepsort" or "histogram")
    
    Returns:
        Frame with tracking visualization
    """
    if tracker_type == "deepsort":
        return draw_tracks_deepsort(frame, tracks)
    elif tracker_type == "histogram":
        return draw_tracks_histogram(frame, tracks)
    else:
        return frame
