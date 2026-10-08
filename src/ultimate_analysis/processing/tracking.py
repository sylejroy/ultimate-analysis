"""Tracking of players and discs across video frames.

Two trackers can do the work (models.tracking.backend):

- "bytetrack": follows boxes by their motion and keeps each player track within its
  team (team_tracker.py). The default.
- "deepsort": matches boxes by what they look like. Between two players who cover each
  other it decides by appearance alone, and often wrongly.

On top of either, the identity layer keeps a player's ID when the tracker loses them
and picks them up again as a new track.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..constants import TRACK_HISTORY_MAX_LENGTH
from ..utils.logger import get_logger
from . import appearance, health
from .player_identity import Observation, PlayerIdentities
from .team_tracker import BYTETracker, DetectionBoxes, TeamTracker, tracker_settings

logger = get_logger("TRACKING")

# Try to import DeepSORT
try:
    import torch
    from deep_sort_realtime.deepsort_tracker import DeepSort
    from deep_sort_realtime.embedder.embedder_pytorch import INPUT_WIDTH, MobileNetv2_Embedder

    DEEPSORT_AVAILABLE = True
except ImportError:
    logger.warning("DeepSORT not available, install with: pip install deep-sort-realtime")
    DEEPSORT_AVAILABLE = False
    DeepSort = None


# Global tracking state
_deepsort_tracker = None
_player_tracker: Optional[TeamTracker] = None  # The "bytetrack" backend: players,
_disc_tracker: Optional[BYTETracker] = None  # and discs, which have no team
# (embedder, network) - the embedder's network compiled for inference, or the network
# itself when compiling is not possible
_compiled_embedder: Tuple[Any, Any] = (None, None)
# Track ID -> the trail of its feet, an array (N, 2) of picture positions. Kept as
# fractions of a pixel: the camera motion moves every stored point every frame, and
# rounding each time would let a trail drift.
_track_histories: Dict[int, np.ndarray] = {}
# Frames per second of the video; how long a lost track is kept is set in seconds
_frame_rate = 30.0

# Players keep their identity when the tracker loses them and picks them up again as a
# new track. The IDs handed out below are those of the players, not of the tracks.
_identities = PlayerIdentities()
_history_last_frame: Dict[int, int] = {}
# The video frame analysed last, and how many video frames lie between two analysed ones.
# Live playback skips frames to keep up, so a trail of a fixed number of points would
# reach back the further the slower the analysis runs.
_video_frame: Optional[int] = None
_frames_per_step = 1
# Discs are not players; their track IDs are moved out of the way of the player IDs
DISC_ID_OFFSET = 100000
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
        # The player's team (0 or 1) and its average shirt colour (BGR), once the tracker
        # knows them
        self.team: Optional[int] = None
        self.team_colour: Optional[Tuple[int, int, int]] = None

    def to_ltrb(self) -> List[float]:
        """Return bounding box in [x1, y1, x2, y2] format."""
        return self.bbox


def _initialize_deepsort_tracker():
    """Initialize DeepSORT tracker with optimal settings."""
    global _deepsort_tracker

    if not DEEPSORT_AVAILABLE:
        logger.warning("Cannot initialize DeepSORT - not available")
        return False

    if _deepsort_tracker is not None:
        return True

    try:
        # DeepSORT configuration optimized for Ultimate Frisbee
        _deepsort_tracker = DeepSort(
            max_age=_max_age_frames(),
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


def _max_age_frames() -> int:
    """Frames a track is kept without being detected.

    Half a second, the tracker's own default at 60 frames per second, loses a player who
    runs behind another one; the track then returns as somebody new.
    """
    seconds = float(get_setting("models.tracking.max_age_seconds", 3.0))
    return max(1, round(seconds * _frame_rate))


def set_frame_rate(frames_per_second: float) -> None:
    """Tell the tracker the frame rate of the video its frames come from."""
    global _frame_rate
    if frames_per_second and frames_per_second > 0:
        _frame_rate = float(frames_per_second)
    if _deepsort_tracker is not None:
        _deepsort_tracker.tracker.max_age = _max_age_frames()
    for tracker in (_player_tracker, _disc_tracker):
        if tracker is not None:
            tracker.max_frames_lost = tracker.args.track_buffer = _max_age_frames()


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
            logger.warning("Compiled embedder differs; using the original")
    except Exception as e:
        logger.warning(f"Could not compile embedder, using the original: {e}")

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


def run_tracking(
    frame: np.ndarray, detections: List[Dict[str, Any]], frame_index: Optional[int] = None
) -> List[Track]:
    """Run object tracking on detected objects.

    Args:
        frame: Input video frame as numpy array (H, W, C) in BGR format
        detections: List of detection dictionaries from inference
        frame_index: The frame's number in the video, if known: frames may be skipped
            between calls, and the trails are kept for a time, not a number of calls

    Returns:
        List of Track objects with consistent IDs across frames

    Example:
        tracks = run_tracking(frame, detections)
        for track in tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()
    """
    global _frame_count, _video_frame, _frames_per_step
    _frame_count += 1
    if frame_index is not None:
        skipped = frame_index - _video_frame if _video_frame is not None else 1
        # Anything else is a seek, after which the caller resets the tracker anyway
        _frames_per_step = skipped if 0 < skipped <= _frame_rate else 1
        _video_frame = frame_index

    if _uses_bytetrack():
        try:
            return _run_bytetrack_tracking(frame, detections)
        except Exception as e:
            logger.exception(f"Error in ByteTrack tracking: {e}")
            health.report("Tracking", f"failed ({type(e).__name__}); players are not followed")
            return _run_simple_tracking(detections)

    if not detections and _deepsort_tracker is None:
        return []

    return _run_deepsort_tracking(frame, detections)


def _uses_bytetrack() -> bool:
    return get_setting("models.tracking.backend", "bytetrack") == "bytetrack"


def _run_bytetrack_tracking(frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Track]:
    """Track players within their teams, and discs, by the motion of their boxes."""
    global _player_tracker, _disc_tracker
    if _player_tracker is None or _disc_tracker is None:
        _player_tracker = TeamTracker(tracker_settings(_max_age_frames()))
        _disc_tracker = BYTETracker(tracker_settings(_max_age_frames()))

    tracks = []
    for class_name, class_id, tracker in (
        ("player", 1, _player_tracker),
        ("disc", 0, _disc_tracker),
    ):
        found = [
            detection
            for detection in detections
            if detection.get("class_name") == class_name and detection.get("bbox") is not None
        ]
        boxes = DetectionBoxes(
            [detection["bbox"] for detection in found],
            [detection["confidence"] for detection in found],
        )
        # A row is x1, y1, x2, y2, track ID, confidence, class, index of the detection
        rows = tracker.update(boxes, frame)
        # Observers and others in neither team's colours are followed but not shown
        hidden = (
            tracker.outsiders()
            if tracker is _player_tracker and get_setting("models.tracking.hide_non_players", True)
            else ()
        )
        teams = tracker.teams_of_tracks() if tracker is _player_tracker else {}
        team_colours = tracker.shirt_colours() if tracker is _player_tracker else {}
        for row in rows:
            if int(row[4]) in hidden:
                continue
            tracks.append(
                Track(
                    track_id=int(row[4]),
                    bbox=[float(value) for value in row[:4]],
                    class_id=class_id,
                    confidence=float(row[5]),
                    class_name=class_name,
                    model_type=f"{class_name}_model",
                )
            )
            tracks[-1].team = teams.get(int(row[4]))
            tracks[-1].team_colour = team_colours.get(tracks[-1].team)
    _finish_tracks(frame, tracks)
    return tracks


def team_shirt_colours() -> Dict[int, Tuple[int, int, int]]:
    """{team (0 or 1): its average shirt colour (BGR)}; empty until the teams are known."""
    return _player_tracker.shirt_colours() if _player_tracker is not None else {}


def _finish_tracks(frame: np.ndarray, tracks: List[Track]) -> None:
    """Give the tracks of a frame their player IDs and add them to the trails."""
    _assign_player_identities(frame, tracks)

    # The trail follows the player's feet (bottom centre of the box)
    for track in tracks:
        x1, _, x2, y2 = track.bbox
        _update_track_history(track.track_id, (int((x1 + x2) / 2), int(y2)))
        _history_last_frame[track.track_id] = _frame_count

    # Trails of tracks that are gone for good no longer need to be stored
    oldest = _frame_count - _max_age_frames()
    for track_id in [
        t for t in _track_histories if _history_last_frame.get(t, float("-inf")) < oldest
    ]:
        del _track_histories[track_id]
        _history_last_frame.pop(track_id, None)


def _followed_tracks() -> Tuple[List[Any], float]:
    """Player tracks the tracker still follows, found in this frame or not, and how long
    a track exists before the tracker reports it (seconds)."""
    if _uses_bytetrack() and _player_tracker is not None:
        followed = _player_tracker.tracked_stracks + _player_tracker.lost_stracks
        return followed, 1.0 / _frame_rate  # Reported from its second frame on
    if _deepsort_tracker is not None:
        # Reported once it has been detected in n_init frames in a row
        age = float(get_setting("models.tracking.n_init", 3)) / _frame_rate
        return list(_deepsort_tracker.tracker.tracks), age
    return [], 0.0


def _run_deepsort_tracking(frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Track]:
    """Run DeepSORT tracking on detections.

    Args:
        frame: Input video frame
        detections: List of detection dictionaries

    Returns:
        List of Track objects with consistent IDs
    """

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

        _finish_tracks(frame, tracks)

        logger.debug(f"DeepSORT returned {len(tracks)} confirmed tracks")
        return tracks

    except Exception as e:
        logger.exception(f"Error in DeepSORT tracking: {e}")
        return _run_simple_tracking(detections)


def _assign_player_identities(frame: np.ndarray, tracks: List[Track]) -> None:
    """Replace the track IDs of this frame's tracks by the IDs of the players they are.

    The kit colour of a player is taken every few frames; a track that is new gets it at
    once, since that is when it is compared with the missing players.
    """
    players = [track for track in tracks if track.class_name == "player"]
    for track in tracks:
        if track.class_name != "player":
            track.track_id += DISC_ID_OFFSET
    if not players:
        return

    if not get_setting("models.tracking.identity.enabled", True):
        return
    interval = max(1, int(get_setting("models.tracking.identity.appearance_interval", 10)))
    due = [
        index
        for index, track in enumerate(players)
        if not _identities.knows_track(track.track_id)
        or (_frame_count + track.track_id) % interval == 0
    ]
    features: List[Optional[np.ndarray]] = [None] * len(players)
    for index, vector in zip(due, appearance.encode(frame, [players[i].bbox for i in due])):
        features[index] = vector

    observations = [
        Observation(
            track_id=track.track_id,
            position=((track.bbox[0] + track.bbox[2]) / 2, track.bbox[3]),
            height=track.bbox[3] - track.bbox[1],
            feature=feature,
        )
        for track, feature in zip(players, features)
    ]
    followed, track_age = _followed_tracks()
    alive = {int(track.track_id) for track in followed}
    player_of_track = _identities.assign(_frame_count / _frame_rate, observations, alive, track_age)
    for track in players:
        track.track_id = player_of_track[track.track_id]


def merge_players(player_id: int, into_player_id: int) -> None:
    """Declare a player to be an earlier one who went missing (e.g. same jersey number)."""
    _identities.merge(player_id, into_player_id)
    if player_id in _track_histories:
        _track_histories[into_player_id] = _track_histories.pop(player_id)
        _history_last_frame[into_player_id] = _history_last_frame.pop(player_id, _frame_count)


def missing_players(present: set) -> List[int]:
    """Players that are remembered but not among the given ones."""
    return _identities.missing_players(present)


def kit_distance(first_player: int, second_player: int) -> Optional[float]:
    """How different the kits of two players are (0 = the same), or None if not known."""
    return _identities.kit_distance(first_player, second_player)


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


def reset_tracker() -> None:
    """Reset the tracker state and clear all tracks.

    This should be called when switching videos or when tracking quality degrades.
    """
    global _deepsort_tracker, _track_histories, _frame_count, _video_frame, _frames_per_step

    logger.info("Resetting tracker state")

    # Reset DeepSORT tracker
    if _deepsort_tracker is not None:
        # Keep the loaded appearance model, but discard identities and old embeddings.
        _deepsort_tracker.delete_all_tracks()
        _deepsort_tracker.tracker.metric.samples.clear()
    for tracker in (_player_tracker, _disc_tracker):
        if tracker is not None:
            tracker.reset()

    # Clear track histories and reset frame count
    _track_histories.clear()
    _history_last_frame.clear()
    _identities.reset()
    _frame_count = 0
    _video_frame, _frames_per_step = None, 1

    # Reset jersey tracking as well
    try:
        from .jersey_tracker import reset_jersey_tracker

        reset_jersey_tracker()
    except ImportError:
        logger.debug("Jersey tracker not available for reset")

    logger.info("Tracker reset complete")


def get_track_histories() -> Dict[int, np.ndarray]:
    """The trail of every tracked object.

    Returns:
        Track ID -> positions of the feet in the picture, oldest first, as an integer
        array of shape (N, 2)
    """
    return {
        track_id: np.rint(history).astype(np.int32)
        for track_id, history in _track_histories.items()
    }


def apply_camera_motion(camera_motion: np.ndarray) -> None:
    """Move everything the tracker remembers along with the picture.

    Positions from earlier frames are pixel positions. When the camera pans or zooms, the
    spot on the field a player stood on is somewhere else in the picture.

    - The trails are moved there, so a trail stays on the ground the player ran over.
    - So is where each track expects its player next. Otherwise a pan looks to the tracker
      as if every player had jumped, and it matches them worse or loses them.

    Call this before run_tracking for the frame.

    Args:
        camera_motion: Homography from the previous frame to the current one
    """
    _move_track_states(camera_motion)
    _identities.apply_camera_motion(camera_motion)
    # All tracks in one call; there are thousands of stored positions
    track_ids = list(_track_histories)
    if not track_ids:
        return
    lengths = [len(_track_histories[track_id]) for track_id in track_ids]
    stacked = np.concatenate([_track_histories[track_id] for track_id in track_ids])
    moved = cv2.perspectiveTransform(stacked.reshape(-1, 1, 2), camera_motion).reshape(-1, 2)
    start = 0
    for track_id, length in zip(track_ids, lengths):
        _track_histories[track_id] = moved[start : start + length]
        start += length


def _move_track_states(camera_motion: np.ndarray) -> None:
    """Apply the camera motion to the tracker's motion model of every track.

    A track's state is its box centre, aspect ratio, and height, and the speed of each
    (x, y, a, h, vx, vy, va, vh). The centre moves with the picture; near a point the
    motion is a small linear map, which turns the speed and scales the height.
    """
    if _uses_bytetrack():
        states = [
            track
            for tracker in (_player_tracker, _disc_tracker)
            if tracker is not None
            for track in tracker.tracked_stracks + tracker.lost_stracks
            if track.mean is not None
        ]
    else:
        states = [] if _deepsort_tracker is None else list(_deepsort_tracker.tracker.tracks)
    for track in states:
        x, y = float(track.mean[0]), float(track.mean[1])
        moved = camera_motion @ np.array([x, y, 1.0])
        if abs(moved[2]) < 1e-9:
            continue
        new_x, new_y = moved[0] / moved[2], moved[1] / moved[2]
        # How the picture stretches and turns around this point
        step = np.array([[x + 1.0, y, 1.0], [x, y + 1.0, 1.0]]) @ camera_motion.T
        local = (step[:, :2] / step[:, 2:3] - [new_x, new_y]).T
        scale = float(np.sqrt(abs(np.linalg.det(local))))

        track.mean[0], track.mean[1] = new_x, new_y
        track.mean[3] *= scale
        track.mean[4:6] = local @ track.mean[4:6]
        track.mean[7] *= scale


def _update_track_history(track_id: int, center_point: Tuple[int, int]) -> None:
    """Update the position history for a track.

    Args:
        track_id: Unique track identifier
        center_point: Center point (x, y) of the tracked object
    """
    max_length = get_setting("models.tracking.track_history_length", TRACK_HISTORY_MAX_LENGTH)
    # As many points as cover the trail's time at the rate frames are analysed at
    seconds = float(get_setting("models.tracking.trail_seconds", 4.0))
    max_length = max(2, min(max_length, round(seconds * _frame_rate / _frames_per_step)))
    point = np.array([center_point], dtype=np.float32)
    history = _track_histories.get(track_id)
    if history is None:
        _track_histories[track_id] = point
    else:
        _track_histories[track_id] = np.concatenate((history[-(max_length - 1) :], point))
