"""
Visualization utilities for drawing detections, tracks, and other overlays.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def get_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a consistent color for a track ID."""
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (255, 255, 128), # Light Yellow
        (128, 255, 128), # Light Green
    ]
    return colors[track_id % len(colors)]


def draw_detections(frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
    """
    Draw detection bounding boxes on the frame.
    
    Args:
        frame: Input frame
        detections: List of detections in format [([x, y, w, h], conf, cls), ...]
    
    Returns:
        Frame with detection boxes drawn
    """
    for detection in detections:
        (x, y, w, h), conf, cls = detection
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw confidence and class
        label = f"Class {cls}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x, y - label_size[1] - 5), (x + label_size[0], y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame


def draw_tracks_deepsort(frame: np.ndarray, tracks: List[Any]) -> np.ndarray:
    """
    Draw DeepSort tracking results on the frame.
    
    Args:
        frame: Input frame
        tracks: List of DeepSort track objects
    
    Returns:
        Frame with tracking visualization
    """
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
        label = f"ID: {track_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
    
    return frame


def draw_tracks_histogram(frame: np.ndarray, tracks: List[Any]) -> np.ndarray:
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
        label = f"ID: {track_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        # Draw track history if available
        if hasattr(track, 'history') and len(track.history) > 1:
            for i in range(1, len(track.history)):
                prev_x1, prev_y1, prev_x2, prev_y2 = track.history[i-1]
                curr_x1, curr_y1, curr_x2, curr_y2 = track.history[i]
                
                prev_center = ((prev_x1 + prev_x2) // 2, (prev_y1 + prev_y2) // 2)
                curr_center = ((curr_x1 + curr_x2) // 2, (curr_y1 + curr_y2) // 2)
                
                # Draw line with decreasing alpha
                alpha = 0.3 + 0.7 * (i / len(track.history))
                overlay = frame.copy()
                cv2.line(overlay, prev_center, curr_center, color, 2)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame


def draw_tracks(frame: np.ndarray, tracks: List[Any], tracker_type: str = "deepsort") -> np.ndarray:
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
