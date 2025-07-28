"""Object tracking module - DeepSORT and histogram-based tracking.

This module handles tracking of detected objects across video frames.
Maintains consistent identities for players and discs throughout the game.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..config.settings import get_setting
from ..constants import MAX_TRACKS_ACTIVE, TRACK_HISTORY_MAX_LENGTH, FALLBACK_DEFAULTS


# Global tracking state
_tracker_type = "deepsort"
_tracker_instance = None
_track_histories = defaultdict(list)
_next_track_id = 1


class Track:
    """Represents a tracked object with consistent identity."""
    
    def __init__(self, track_id: int, bbox: List[float], class_id: int, confidence: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.confidence = confidence
        self.det_class = class_id  # For compatibility with existing code
        
    def to_ltrb(self) -> List[float]:
        """Return bounding box in [x1, y1, x2, y2] format."""
        return self.bbox
        
    def to_tlwh(self) -> List[float]:
        """Return bounding box in [x, y, width, height] format."""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]


def run_tracking(frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Track]:
    """Run object tracking on detected objects.
    
    Args:
        frame: Input video frame as numpy array (H, W, C) in BGR format
        detections: List of detection dictionaries from inference
        
    Returns:
        List of Track objects with consistent IDs across frames
        
    Example:
        tracks = run_tracking(frame, detections)
        for track in tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()
    """
    global _next_track_id
    
    print(f"[TRACKING] Processing {len(detections)} detections with {_tracker_type} tracker")
    
    if not detections:
        return []
    
    # TODO: Implement actual tracking algorithms
    # For now, create simple tracks from detections
    tracks = []
    
    for i, detection in enumerate(detections):
        # Simple placeholder: assign consecutive track IDs
        track = Track(
            track_id=_next_track_id + i,
            bbox=detection['bbox'],
            class_id=detection['class_id'],
            confidence=detection['confidence']
        )
        tracks.append(track)
        
        # Update track history for visualization
        _update_track_history(track.track_id, track.bbox)
    
    _next_track_id += len(detections)
    
    # Limit number of active tracks
    max_tracks = get_setting("models.tracking.max_tracks", MAX_TRACKS_ACTIVE)
    if len(tracks) > max_tracks:
        tracks = tracks[:max_tracks]
    
    print(f"[TRACKING] Returning {len(tracks)} tracks")
    return tracks


def set_tracker_type(tracker_type: str) -> bool:
    """Set the type of tracker to use.
    
    Args:
        tracker_type: Type of tracker ("deepsort" or "histogram")
        
    Returns:
        True if tracker type set successfully, False otherwise
        
    Example:
        set_tracker_type("deepsort")
    """
    global _tracker_type, _tracker_instance
    
    tracker_type = tracker_type.lower()
    
    if tracker_type not in ["deepsort", "histogram"]:
        print(f"[TRACKING] Unsupported tracker type: {tracker_type}")
        return False
    
    print(f"[TRACKING] Setting tracker type to: {tracker_type}")
    _tracker_type = tracker_type
    
    # Reset tracker instance to force reinitialization
    _tracker_instance = None
    reset_tracker()
    
    return True


def reset_tracker() -> None:
    """Reset the tracker state and clear all tracks.
    
    This should be called when switching videos or when tracking quality degrades.
    """
    global _tracker_instance, _track_histories, _next_track_id
    
    print("[TRACKING] Resetting tracker state")
    
    _tracker_instance = None
    _track_histories.clear()
    _next_track_id = 1
    
    print("[TRACKING] Tracker reset complete")


def get_tracker_type() -> str:
    """Get the current tracker type.
    
    Returns:
        Current tracker type string
    """
    return _tracker_type


def get_track_histories() -> Dict[int, List[Tuple[float, float]]]:
    """Get track history for all tracked objects.
    
    Returns:
        Dictionary mapping track_id to list of (center_x, center_y) positions
    """
    return dict(_track_histories)


def _update_track_history(track_id: int, bbox: List[float]) -> None:
    """Update the position history for a track.
    
    Args:
        track_id: Unique track identifier
        bbox: Bounding box [x1, y1, x2, y2]
    """
    # Calculate center point
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Add to history
    _track_histories[track_id].append((center_x, center_y))
    
    # Limit history length
    max_length = get_setting("models.tracking.track_history_length", TRACK_HISTORY_MAX_LENGTH)
    if len(_track_histories[track_id]) > max_length:
        _track_histories[track_id] = _track_histories[track_id][-max_length:]


def _initialize_tracker() -> None:
    """Initialize the tracker instance based on current type."""
    global _tracker_instance
    
    if _tracker_instance is not None:
        return
    
    print(f"[TRACKING] Initializing {_tracker_type} tracker")
    
    try:
        if _tracker_type == "deepsort":
            # TODO: Initialize DeepSORT tracker
            # from deep_sort_realtime import DeepSort
            # _tracker_instance = DeepSort()
            _tracker_instance = "deepsort_placeholder"
            
        elif _tracker_type == "histogram":
            # TODO: Initialize histogram-based tracker
            _tracker_instance = "histogram_placeholder"
            
        print(f"[TRACKING] {_tracker_type} tracker initialized successfully")
        
    except Exception as e:
        print(f"[TRACKING] Failed to initialize {_tracker_type} tracker: {e}")
        _tracker_instance = None


# Initialize default tracker when module is imported
def _load_default_tracker():
    """Load the default tracker type."""
    default_tracker = get_setting(
        "models.tracking.default_tracker",
        FALLBACK_DEFAULTS['tracker_type']
    )
    set_tracker_type(default_tracker)


_load_default_tracker()
