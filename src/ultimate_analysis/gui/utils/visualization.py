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
    N = 200  # Number of history points to show
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        color = get_color(track_id)
        # Draw bounding box (fully opaque)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Draw class name under the bounding box if available
        class_name = getattr(track, 'det_class', None)
        if class_name is not None:
            label = str(class_name)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_x = x1
            label_y = y2 + label_size[1] + 5
            cv2.rectangle(frame, (label_x, y2 + 5), (label_x + label_size[0], label_y), color, -1)
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        # --- Track history for DeepSort ---
        pt = (center_x, y2)
        if track_id not in _deepsort_track_history:
            _deepsort_track_history[track_id] = []
        _deepsort_track_history[track_id].append(pt)
        if len(_deepsort_track_history[track_id]) > MAX_HISTORY:
            _deepsort_track_history[track_id] = _deepsort_track_history[track_id][-MAX_HISTORY:]
        pts = _deepsort_track_history[track_id][-N:]  # Always use last N points
        if len(pts) > 1:
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 2)
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
    N = 200
    for track in tracks:
        if not track.confirmed:
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = track.bbox
        color = get_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Draw class name under the bounding box if available
        class_name = getattr(track, 'det_class', None)
        if class_name is not None:
            label = str(class_name)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_x = x1
            label_y = y2 + label_size[1] + 5
            cv2.rectangle(frame, (label_x, y2 + 5), (label_x + label_size[0], label_y), color, -1)
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        if hasattr(track, 'history') and len(track.history) > 1:
            pts = [((b[0]+b[2])//2, b[3]) for b in track.history][-N:]
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i-1], pts[i], color, 2)
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


def clear_track_history():
    """Clear the global track history for DeepSort tracks."""
    global _deepsort_track_history
    _deepsort_track_history.clear()


def overlay_segmentation_mask(frame: np.ndarray, masks: list, color: tuple = (0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    """
    Overlay segmentation masks on the frame for visualization.
    Args:
        frame: Input frame (np.ndarray)
        masks: List of masks (np.ndarray, values 0/1 or 0-255)
        color: BGR color for the mask overlay
        alpha: Transparency for the overlay
    Returns:
        Frame with mask overlay
    """
    overlay = frame.copy()
    for mask in masks:
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        colored_mask = np.zeros_like(frame)
        for i in range(3):
            colored_mask[:, :, i] = mask_resized * (color[i] / 255.0)
        cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay


def draw_player_id_results(frame: np.ndarray, digits: list, digit_str: str = "", color: tuple = (0, 255, 255)) -> np.ndarray:
    """
    Draws the player ID (jersey number) and digit bounding boxes on the frame.
    Args:
        frame: The frame to draw on (BGR, np.ndarray).
        digits: List of (box, class) tuples for each digit (from YOLO).
        digit_str: The detected player ID string.
        color: Color for text and bbox (default: yellow).
    Returns:
        Frame with player ID visualizations.
    """
    vis_frame = frame.copy()
    # Draw bounding boxes for each digit
    if digits:
        for box, digit in digits:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, str(digit), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    # Always show the player ID string or a fallback label
    label = f"ID: {digit_str}" if digit_str else "No ID detected"
    cv2.putText(vis_frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return vis_frame
