"""Object tracking module - DeepSORT tracking.

This module handles tracking of detected objects across video frames.
Maintains consistent identities for players and discs throughout the game.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..constants import TRACK_HISTORY_MAX_LENGTH
from ..utils.logger import get_logger

# Try to import DeepSORT
try:
    import torch
    from deep_sort_realtime.deepsort_tracker import DeepSort
    from deep_sort_realtime.embedder.embedder_pytorch import INPUT_WIDTH, MobileNetv2_Embedder

    DEEPSORT_AVAILABLE = True
except ImportError:
    logger = get_logger("TRACKING")
    logger.warning("DeepSORT not available, install with: pip install deep-sort-realtime")
    DEEPSORT_AVAILABLE = False
    DeepSort = None


# Global tracking state
_deepsort_tracker = None
# (embedder, network) - the embedder's network compiled for inference, or the network
# itself when compiling is not possible
_compiled_embedder: Tuple[Any, Any] = (None, None)
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


def _initialize_deepsort_tracker():
    """Initialize DeepSORT tracker with optimal settings."""
    global _deepsort_tracker
    logger = get_logger("TRACKING")

    if not DEEPSORT_AVAILABLE:
        logger.warning("Cannot initialize DeepSORT - not available")
        return False

    if _deepsort_tracker is not None:
        return True

    try:
        # DeepSORT configuration optimized for Ultimate Frisbee
        _deepsort_tracker = DeepSort(
            max_age=get_setting(
                "models.tracking.max_age", 30
            ),  # Reduced from 50 for faster cleanup
            n_init=get_setting("models.tracking.n_init", 3),  # Frames needed to confirm track
            nms_max_overlap=get_setting("models.tracking.nms_overlap", 0.7),  # Non-max suppression
            max_cosine_distance=get_setting(
                "models.tracking.max_cosine_distance", 0.5
            ),  # Tighter for faster matching (was 0.7)
            nn_budget=get_setting("models.tracking.nn_budget", 50),  # Reduced from 100 for speed
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

        logger.info("DeepSORT tracker initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize DeepSORT: {e}")
        _deepsort_tracker = None
        return False


def _get_embedder_network(embedder: Any) -> Any:
    """The embedder's network, traced once so it runs without Python overhead.

    Tracing keeps the arithmetic identical. Freezing the trace would be faster still, but
    it folds batch normalization into half-precision weights and shifts the embeddings.

    The compiled network is only used when it reproduces the original's embeddings; any
    failure or mismatch keeps the original network.
    """
    global _compiled_embedder
    if _compiled_embedder[0] is embedder:
        return _compiled_embedder[1]

    network = embedder.model
    try:
        dtype = torch.half if embedder.half else torch.float
        shape = (embedder.max_batch_size, 3, INPUT_WIDTH, INPUT_WIDTH)
        example = torch.rand(shape, device="cuda", dtype=dtype)
        with torch.inference_mode():
            compiled = torch.jit.trace(network, example, check_trace=False)
            # A different batch size than the traced one, as during tracking
            check = torch.rand((3, *shape[1:]), device="cuda", dtype=dtype)
            expected = network(check).float()
            actual = compiled(check).float()
            similarity = torch.nn.functional.cosine_similarity(expected, actual).min().item()
        if actual.shape == expected.shape and similarity > 0.99999:
            network = compiled
        else:
            get_logger("TRACKING").warning("Compiled embedder differs; using the original")
    except Exception as e:
        get_logger("TRACKING").warning(f"Could not compile embedder, using the original: {e}")

    _compiled_embedder = (embedder, network)
    return network


def _embed_detections(frame: np.ndarray, deepsort_detections: List[tuple]) -> Optional[list]:
    """Appearance embeddings for the detections, as DeepSORT's embedder computes them.

    The library converts and normalizes every crop separately on the CPU and runs the
    network with gradient tracking. This does the same arithmetic for the whole batch on
    the GPU without gradients. Returns None to let the library compute them instead.
    """
    embedder = _deepsort_tracker.embedder
    if not isinstance(embedder, MobileNetv2_Embedder) or not embedder.gpu or not embedder.bgr:
        return None

    crops, _ = _deepsort_tracker.crop_bb(frame, deepsort_detections)
    if any(crop.size == 0 for crop in crops):
        return None

    resized = np.stack([cv2.resize(crop[..., ::-1], (INPUT_WIDTH, INPUT_WIDTH)) for crop in crops])

    network = _get_embedder_network(embedder)
    embeds = []
    with torch.inference_mode():
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        for start in range(0, len(resized), embedder.max_batch_size):
            batch = torch.from_numpy(resized[start : start + embedder.max_batch_size]).cuda()
            batch = (batch.permute(0, 3, 1, 2).float().div_(255.0) - mean) / std
            if embedder.half:
                batch = batch.half()
            embeds.extend(network(batch).cpu().numpy())
    return embeds


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
    logger = get_logger("TRACKING")

    logger.debug(
        f"Processing {len(detections)} detections with DeepSORT tracker (frame {_frame_count})"
    )

    if not detections and _deepsort_tracker is None:
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
    logger = get_logger("TRACKING")

    if not _initialize_deepsort_tracker():
        logger.warning("DeepSORT not available, falling back to simple tracking")
        return _run_simple_tracking(detections)

    try:
        # Convert detections to DeepSORT format: [([x1, y1, x2, y2], confidence, class_id), ...]
        deepsort_detections = []

        logger.debug(f"Processing {len(detections)} detections for DeepSORT")

        for i, det in enumerate(detections):
            logger.debug(f"Detection {i}: {det}")

            bbox = det.get("bbox")
            confidence = det.get("confidence")
            class_id = det.get("class_id")

            logger.debug(f"bbox: {bbox} (type: {type(bbox)})")
            logger.debug(f"confidence: {confidence} (type: {type(confidence)})")
            logger.debug(f"class_id: {class_id} (type: {type(class_id)})")

            # Ensure bbox exists and has 4 values
            if bbox is None:
                logger.warning("bbox is None, skipping detection")
                continue

            if not hasattr(bbox, "__len__") or len(bbox) != 4:
                logger.warning(f"Invalid bbox format or length {bbox}, skipping detection")
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
                # IDs are local to each YOLO model: both may use class zero.
                class_name = det.get("class_name")
                cls = {"disc": 0, "player": 1}.get(class_name)
                if cls is None:
                    cls = int(class_id)

                if width <= 0 or height <= 0:
                    continue

                # DeepSORT expects ([x, y, width, height], confidence, class_id) format
                deepsort_det = ([x, y, width, height], conf, cls)
                deepsort_detections.append(deepsort_det)

                logger.debug(
                    f"Formatted detection: LTRB {[x1, y1, x2, y2]} -> TLWH {[x, y, width, height]}"
                )

            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid detection values, skipping: {e}")
                continue

        logger.debug(f"Formatted {len(deepsort_detections)} detections for DeepSORT")

        # Update tracker with current frame and detections
        embeds = _embed_detections(frame, deepsort_detections) if deepsort_detections else None
        with torch.inference_mode():
            tracks_deepsort = _deepsort_tracker.update_tracks(
                deepsort_detections, embeds=embeds, frame=frame
            )

        # Retired tracks no longer need trajectory storage.
        active_ids = {int(track.track_id) for track in tracks_deepsort}
        for track_id in list(_track_histories):
            if track_id not in active_ids:
                del _track_histories[track_id]

        # Convert DeepSORT tracks to our Track format
        tracks = []
        for track in tracks_deepsort:
            if not track.is_confirmed() or track.time_since_update > 0:
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

        logger.debug(f"DeepSORT returned {len(tracks)} confirmed tracks")
        return tracks

    except Exception as e:
        logger.error(f"Error in DeepSORT tracking: {e}")
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
    # Fallback IDs are new each frame, so old histories can never be reused.
    _track_histories.clear()

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
    global _deepsort_tracker
    logger = get_logger("TRACKING")

    tracker_type = tracker_type.lower()

    if tracker_type != "deepsort":
        logger.warning(f"Unsupported tracker type: {tracker_type}. Only 'deepsort' is supported.")
        return False

    logger.info(f"Setting tracker type to: {tracker_type}")

    # Reset tracker instance to force reinitialization
    _deepsort_tracker = None

    reset_tracker()

    return True


def reset_tracker() -> None:
    """Reset the tracker state and clear all tracks.

    This should be called when switching videos or when tracking quality degrades.
    """
    global _deepsort_tracker, _track_histories, _frame_count
    logger = get_logger("TRACKING")

    logger.info("Resetting tracker state")

    # Reset DeepSORT tracker
    if _deepsort_tracker is not None:
        # Keep the loaded appearance model, but discard identities and old embeddings.
        _deepsort_tracker.delete_all_tracks()
        _deepsort_tracker.tracker.metric.samples.clear()

    # Clear track histories and reset frame count
    _track_histories.clear()
    _frame_count = 0

    # Reset jersey tracking as well
    try:
        from .jersey_tracker import reset_jersey_tracker

        reset_jersey_tracker()
    except ImportError:
        logger.debug("Jersey tracker not available for reset")

    logger.info("Tracker reset complete")


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


def _load_default_tracker():
    """Load the default tracker type."""
    default_tracker = get_setting("models.tracking.default_tracker", "deepsort")
    set_tracker_type(default_tracker)


# Initialize default tracker when module is imported
_load_default_tracker()
