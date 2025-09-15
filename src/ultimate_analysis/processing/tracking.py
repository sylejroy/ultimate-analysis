"""Object tracking module - DeepSORT tracking.

This module handles tracking of detected objects across video frames.
Maintains consistent identities for players and discs throughout the game.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from ..config.settings import get_setting
from ..constants import TRACK_HISTORY_MAX_LENGTH

# Try to import DeepSORT
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort

    DEEPSORT_AVAILABLE = True
except ImportError:
    print("[TRACKING] DeepSORT not available, install with: pip install deep-sort-realtime")
    DEEPSORT_AVAILABLE = False
    DeepSort = None


# Global tracking state
_tracker_type = "deepsort"
_deepsort_tracker = None
_track_histories = defaultdict(list)
_frame_count = 0


class Track:
    """Represents a tracked object with consistent identity."""

    def __init__(
        self,
        track_id: int,
        bbox: List[float],
        class_id: int,
        confidence: float,
        class_name: str = "unknown",
        model_type: str = "unknown",
    ):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.confidence = confidence
        self.class_name = class_name
        self.model_type = model_type  # Track which model detected this
        self.det_class = class_name  # For compatibility with existing code

    def to_ltrb(self) -> List[float]:
        """Return bounding box in [x1, y1, x2, y2] format."""
        return self.bbox

    def to_tlwh(self) -> List[float]:
        """Return bounding box in [x, y, width, height] format."""
        x1, y1, x2, y2 = self.bbox
        return [x1, y1, x2 - x1, y2 - y1]


def _initialize_deepsort_tracker():
    """Initialize DeepSORT tracker with optimal settings."""
    global _deepsort_tracker

    if not DEEPSORT_AVAILABLE:
        print("[TRACKING] Cannot initialize DeepSORT - not available")
        return False

    if _deepsort_tracker is not None:
        return True

    try:
        # DeepSORT configuration optimized for Ultimate Frisbee
        _deepsort_tracker = DeepSort(
            max_age=get_setting("models.tracking.max_age", 50),  # Frames to keep lost tracks
            n_init=get_setting("models.tracking.n_init", 3),  # Frames needed to confirm track
            nms_max_overlap=get_setting("models.tracking.nms_overlap", 0.7),  # Non-max suppression
            max_cosine_distance=get_setting(
                "models.tracking.max_cosine_distance", 0.7
            ),  # Feature similarity
            nn_budget=get_setting("models.tracking.nn_budget", 100),  # Feature budget per class
            override_track_class=None,  # Don't override class predictions
            embedder="mobilenet",  # Feature extractor model
            half=True,  # Use half precision for speed
            bgr=True,  # Input is BGR format
            embedder_gpu=True,  # Use GPU for feature extraction if available
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,  # Don't use polygon tracking
            today=None,
        )

        print("[TRACKING] DeepSORT tracker initialized successfully")
        return True

    except Exception as e:
        print(f"[TRACKING] Failed to initialize DeepSORT: {e}")
        _deepsort_tracker = None
        return False


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
    global _frame_count
    _frame_count += 1

    print(
        f"[TRACKING] Processing {len(detections)} detections with DeepSORT tracker (frame {_frame_count})"
    )

    if not detections:
        return []

    return _run_deepsort_tracking(frame, detections)


def _run_deepsort_tracking(frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Track]:
    """Run DeepSORT tracking on detections.

    Args:
        frame: Input video frame
        detections: List of detection dictionaries

    Returns:
        List of Track objects with consistent IDs
    """
    if not _initialize_deepsort_tracker():
        print("[TRACKING] DeepSORT not available, falling back to simple tracking")
        return _run_simple_tracking(detections)

    try:
        # Convert detections to DeepSORT format: [([x1, y1, x2, y2], confidence, class_id), ...]
        deepsort_detections = []

        print(f"[TRACKING] Processing {len(detections)} detections for DeepSORT")

        for i, det in enumerate(detections):
            print(f"[TRACKING] Detection {i}: {det}")

            bbox = det.get("bbox")
            confidence = det.get("confidence")
            class_id = det.get("class_id")

            print(f"[TRACKING] bbox: {bbox} (type: {type(bbox)})")
            print(f"[TRACKING] confidence: {confidence} (type: {type(confidence)})")
            print(f"[TRACKING] class_id: {class_id} (type: {type(class_id)})")

            # Ensure bbox exists and has 4 values
            if bbox is None:
                print("[TRACKING] Warning: bbox is None, skipping detection")
                continue

            if not hasattr(bbox, "__len__") or len(bbox) != 4:
                print(
                    f"[TRACKING] Warning: Invalid bbox format or length {bbox}, skipping detection"
                )
                continue

            try:
                # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height] for DeepSORT
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

                # DeepSORT expects TLWH format: [x, y, width, height]
                x = x1
                y = y1
                width = x2 - x1
                height = y2 - y1

                conf = float(confidence)
                cls = int(class_id)

                # DeepSORT expects ([x, y, width, height], confidence, class_id) format
                deepsort_det = ([x, y, width, height], conf, cls)
                deepsort_detections.append(deepsort_det)

                print(
                    f"[TRACKING] Formatted detection: LTRB {[x1, y1, x2, y2]} -> TLWH {[x, y, width, height]}"
                )

            except (ValueError, TypeError) as e:
                print(f"[TRACKING] Warning: Invalid detection values, skipping: {e}")
                continue

        if not deepsort_detections:
            print("[TRACKING] No valid detections for DeepSORT")
            return []

        print(f"[TRACKING] Formatted {len(deepsort_detections)} detections for DeepSORT")

        # Update tracker with current frame and detections
        tracks_deepsort = _deepsort_tracker.update_tracks(deepsort_detections, frame=frame)

        # Convert DeepSORT tracks to our Track format
        tracks = []
        for track in tracks_deepsort:
            if not track.is_confirmed():
                continue  # Skip unconfirmed tracks

            # Get track bounding box
            ltrb = track.to_ltrb()

            # Get class info (use the most recent detection class)
            class_id = int(track.get_det_class()) if track.get_det_class() is not None else 0
            confidence = float(track.get_det_conf()) if track.get_det_conf() is not None else 0.5

            # Map class_id to class_name
            class_name = _get_class_name_from_id(class_id)
            
            # Determine model type from class name (this works since each model specializes in its class)
            model_type = "player_model" if class_name == "player" else "disc_model"

            # Create our Track object
            our_track = Track(
                track_id=int(track.track_id),
                bbox=[float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])],
                class_id=class_id,
                confidence=confidence,
                class_name=class_name,
                model_type=model_type,
            )

            tracks.append(our_track)

            # Update track history for visualization (at player's feet - bottom center)
            foot_x = (ltrb[0] + ltrb[2]) / 2  # Center X
            foot_y = ltrb[3]  # Bottom Y (feet level)
            _update_track_history(our_track.track_id, (int(foot_x), int(foot_y)))

        print(f"[TRACKING] DeepSORT returned {len(tracks)} confirmed tracks")
        return tracks

    except Exception as e:
        print(f"[TRACKING] Error in DeepSORT tracking: {e}")
        import traceback

        traceback.print_exc()
        return _run_simple_tracking(detections)


def _run_simple_tracking(detections: List[Dict[str, Any]]) -> List[Track]:
    """Simple tracking fallback that assigns new IDs to each detection.

    Args:
        detections: List of detection dictionaries

    Returns:
        List of Track objects with new IDs
    """
    tracks = []

    for i, detection in enumerate(detections):
        # Create a simple track with frame-based ID
        track_id = _frame_count * 1000 + i  # Simple ID generation

        track = Track(
            track_id=track_id,
            bbox=detection["bbox"],
            class_id=detection["class_id"],
            confidence=detection["confidence"],
            class_name=detection.get("class_name", "unknown"),
            model_type=detection.get("model_type", "unknown"),
        )
        tracks.append(track)

        # Update track history (at player's feet - bottom center)
        foot_x = (detection["bbox"][0] + detection["bbox"][2]) / 2  # Center X
        foot_y = detection["bbox"][3]  # Bottom Y (feet level)
        _update_track_history(track_id, (int(foot_x), int(foot_y)))

    return tracks


def _get_class_name_from_id(class_id: int) -> str:
    """Convert class ID to class name.

    Args:
        class_id: Numeric class identifier

    Returns:
        String class name
    """
    # Map based on our model's class structure
    class_mapping = {0: "disc", 1: "player"}

    return class_mapping.get(class_id, "unknown")


def set_tracker_type(tracker_type: str) -> bool:
    """Set the type of tracker to use.

    Args:
        tracker_type: Type of tracker (only "deepsort" is supported)

    Returns:
        True if tracker type set successfully, False otherwise

    Example:
        set_tracker_type("deepsort")
    """
    global _tracker_type, _deepsort_tracker

    tracker_type = tracker_type.lower()

    if tracker_type != "deepsort":
        print(f"[TRACKING] Unsupported tracker type: {tracker_type}. Only 'deepsort' is supported.")
        return False

    print(f"[TRACKING] Setting tracker type to: {tracker_type}")
    _tracker_type = tracker_type

    # Reset tracker instance to force reinitialization
    _deepsort_tracker = None

    reset_tracker()

    return True


def reset_tracker() -> None:
    """Reset the tracker state and clear all tracks.

    This should be called when switching videos or when tracking quality degrades.
    """
    global _deepsort_tracker, _track_histories, _frame_count

    print("[TRACKING] Resetting tracker state")

    # Reset DeepSORT tracker
    if _deepsort_tracker is not None:
        _deepsort_tracker = None

    # Clear track histories and reset frame count
    _track_histories.clear()
    _frame_count = 0

    # Reset jersey tracking as well
    try:
        from .jersey_tracker import reset_jersey_tracker

        reset_jersey_tracker()
    except ImportError:
        print("[TRACKING] Jersey tracker not available for reset")

    print("[TRACKING] Tracker reset complete")


def get_tracker_type() -> str:
    """Get the current tracker type.

    Returns:
        Current tracker type string
    """
    return _tracker_type


def get_track_histories() -> Dict[int, List[Tuple[int, int]]]:
    """Get track history for all tracked objects.

    Returns:
        Dictionary mapping track_id to list of (center_x, center_y) positions
    """
    return dict(_track_histories)


def _update_track_history(track_id: int, center_point: Tuple[int, int]) -> None:
    """Update the position history for a track.

    Args:
        track_id: Unique track identifier
        center_point: Center point (x, y) of the tracked object
    """
    # Add center point to history
    _track_histories[track_id].append(center_point)

    # Limit history length
    max_length = get_setting("models.tracking.track_history_length", TRACK_HISTORY_MAX_LENGTH)
    if len(_track_histories[track_id]) > max_length:
        _track_histories[track_id] = _track_histories[track_id][-max_length:]


# Remove the old initialization functions and add proper initialization
def _load_default_tracker():
    """Load the default tracker type."""
    default_tracker = get_setting("models.tracking.default_tracker", "deepsort")
    set_tracker_type(default_tracker)


# Initialize default tracker when module is imported
_load_default_tracker()
