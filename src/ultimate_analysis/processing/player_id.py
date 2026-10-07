"""Player identification module - OCR and jersey number detection.

This module identifies players by their jersey numbers. The numbers are read by EasyOCR
or by one of the readers in jersey_readers.py (models.player_id.method), and combined
over time by probabilistic tracking.
"""

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

from ..config.settings import get_setting
from ..constants import JERSEY_NUMBER_MAX, JERSEY_NUMBER_MIN
from ..utils.logger import get_logger
from . import facing
from .jersey_crops import (
    JerseyCropSelector,
    best_number,
    crop_quality,
    crop_top_fraction,
    easyocr_readtext_parameters,
    preprocess_crop,
)
from .jersey_readers import READER_LABELS, get_reader
from .jersey_tracker import (
    add_jersey_measurement,
    get_best_jersey_number,
    get_jersey_probabilities,
    get_jersey_tracker,
)
from .model_lock import GPU_SETUP_LOCK

logger = get_logger("PLAYER_ID")

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available; jersey numbers will not be read")

# Global player ID state
_easyocr_config_checked = 0.0  # When the settings file was last looked at (monotonic s)
_easyocr_reader = None
# Reader chosen in the GUI; None means the configured one (models.player_id.method)
_method_override: Optional[str] = None

# Parsed easyocr_params.yaml, reloaded only when the file changes on disk
_easyocr_config_path: Optional[Path] = None
_easyocr_config_mtime_ns: Optional[int] = None
_easyocr_config_cache: Dict[str, Any] = {}


def get_player_id_method() -> str:
    """Name of the jersey reader in use: easyocr, parseq, florence, or yolo_digits."""
    method = _method_override or get_setting("models.player_id.method", "easyocr")
    return method if method in READER_LABELS else "easyocr"


def set_player_id_method(method: str) -> None:
    """Choose the jersey reader for the following frames."""
    global _method_override
    if method not in READER_LABELS:
        raise ValueError(f"Unknown player ID method: {method}")
    _method_override = method


def _get_text_detector() -> Any:
    """The EasyOCR reader, whose text detector the PARSeq reader uses."""
    _initialize_easyocr()
    return _easyocr_reader


def _get_active_reader() -> Optional[Any]:
    """The reader replacing EasyOCR's recognition, or None to use EasyOCR."""
    method = get_player_id_method()
    if method == "easyocr":
        return None
    return get_reader(method, _get_text_detector)


def run_player_id_on_tracks(
    frame: np.ndarray,
    tracks: List[Any],
    frame_index: int = 0,
    finalized_tracks: Optional[Set[int]] = None,
    crop_selector: Optional[JerseyCropSelector] = None,
    background: bool = False,
) -> Tuple[Dict[int, Tuple[str, Any]], Dict[str, float], Set[int]]:
    """Run player identification on tracked objects using batch EasyOCR with probabilistic tracking.

    Args:
        frame: Current video frame
        tracks: List of track objects from tracking system

    Args:
        frame_index: Current global frame index (for interval/stagger logic)
        finalized_tracks: Set of track_ids whose jersey number is finalized (probability >= threshold)
        crop_selector: Pipeline-owned recent crop cache; None keeps fixed-frame sampling
        background: Read the numbers on another thread instead of waiting for them here.
            The readings of a call are then recorded, and returned, by a later call.
            Needs the crop selector.

    Returns:
        Tuple of (player_identifications, timing_info, finalized_tracks)
        player_identifications: Dictionary mapping track_id -> (jersey_number, detection_details)
        timing_info: Dictionary with 'preprocessing_ms', 'ocr_ms', and 'filtering_ms' totals

    Detection details now include both single-frame and historical tracking results:
        - 'single_frame': Single-frame EasyOCR result
        - 'tracking_history': Top 3 probabilities from historical tracking
        - 'best_tracked': Most probable jersey number from tracking

        Performance optimizations:
                - OCR interval (player_id.ocr_frame_interval): each track is processed only every N frames
                - Staggering: optional (player_id.ocr_frame_interval_stagger) uses (frame_index + track_id) % N
                    so not all tracks run OCR on the same frame
                - Finalization: once best tracked probability exceeds
                    player_id.finalized_certainty_threshold the track is skipped for future OCR calls

        Example:
        results, timing = run_player_id_on_tracks(frame, current_tracks)
        for track_id, (number, details) in results.items():
            logger.debug(f"Track {track_id}: Player #{number}")
            if 'tracking_history' in details:
                for jersey, prob, count in details['tracking_history']:
                    logger.debug(f"  {jersey}: {prob:.1%} ({count} measurements)")
    """

    player_identifications = {}
    if finalized_tracks is None:
        finalized_tracks = set()

    # Optimization settings
    ocr_frame_interval = max(1, get_setting("models.player_id.ocr_frame_interval", 1))
    stagger_enabled = get_setting("models.player_id.ocr_frame_interval_stagger", True)
    finalized_threshold = get_setting("models.player_id.finalized_certainty_threshold", 0.999)
    verbose_debug = get_setting("models.player_id.verbose_debug", False)
    selection_enabled = crop_selector is not None and get_setting(
        "models.player_id.crop_selection.enabled", False
    )
    selection_start = time.perf_counter()
    crop_config = _load_easyocr_config().get("preprocessing", {}) if selection_enabled else {}
    total_timing = {"preprocessing_ms": 0.0, "ocr_ms": 0.0, "filtering_ms": 0.0}
    batch_timing: Dict[str, float] = {"preprocessing_ms": 0.0, "ocr_ms": 0.0, "filtering_ms": 0.0}

    if crop_selector is not None:
        active_ids = {
            getattr(track, "track_id", getattr(track, "id", None)) for track in tracks
        } - finalized_tracks
        crop_selector.begin_frame(frame_index, active_ids)
    background = background and selection_enabled

    def record(metadata: Dict[str, Any], jersey_number: str, details: Optional[Dict]) -> tuple:
        return _record_reading(
            metadata,
            jersey_number,
            details,
            frame_index,
            crop_selector if selection_enabled else None,
            ocr_frame_interval,
            finalized_tracks,
            finalized_threshold,
            verbose_debug,
        )

    reader_idle = True
    if background:
        # What the reader finished since the last call
        finished, reader_idle = _background_reader.collect()
        for read_metadata, read_results in finished:
            for metadata, (jersey_number, details) in zip(read_metadata, read_results):
                player_identifications[metadata["track_id"]] = record(
                    metadata, jersey_number, details
                )

    if not tracks:
        return player_identifications, total_timing, finalized_tracks

    # Initialize EasyOCR if needed
    if _easyocr_reader is None:
        _initialize_easyocr()

    # Prepare player crops for batch processing
    player_crops = []
    track_metadata = []

    tracks_for_ocr: List[Any] = []  # subset that will have OCR this frame

    skipped_interval = 0
    skipped_finalized = 0
    for track in tracks:
        try:
            # Extract track information
            if hasattr(track, "track_id"):
                track_id = track.track_id
            elif hasattr(track, "id"):
                track_id = track.id
            else:
                continue

            # Get bounding box
            if hasattr(track, "to_tlbr"):
                # DeepSORT format
                bbox = track.to_tlbr().astype(int)
                x1, y1, x2, y2 = bbox
            elif hasattr(track, "bbox"):
                # Generic bbox format
                x1, y1, x2, y2 = map(int, track.bbox)
            else:
                continue

            # Skip disc tracks - only process players for jersey number detection
            if hasattr(track, "class_id") and track.class_id == 0:  # 0 = disc
                continue
            elif hasattr(track, "class_name") and track.class_name.lower() == "disc":
                continue

            # Skip if track already finalized
            if track_id in finalized_tracks:
                skipped_finalized += 1
                continue

            # Collect every observed crop, but keep the existing staggered OCR cadence.
            due = True
            if ocr_frame_interval > 1:
                if stagger_enabled:
                    # Stagger by track_id so load is distributed; each track processed every N frames
                    if (frame_index + track_id) % ocr_frame_interval != 0:
                        due = False
                else:
                    if frame_index % ocr_frame_interval != 0:
                        due = False
            if background:
                # The reader takes the next crops when it is done with the last ones; each
                # player's own rhythm is kept by the crop selector
                due = reader_idle and len(player_crops) < MAX_BACKGROUND_CROPS
            if not selection_enabled and not due:
                skipped_interval += 1
                continue
            if selection_enabled and getattr(track, "time_since_update", 0) > 0:
                continue

            # Ensure bbox is within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Crop the tracked object
            crop = frame[y1:y2, x1:x2]

            if selection_enabled:
                top_fraction = crop_config.get("crop_top_fraction", 0.33)
                torso_y2 = y1 + max(1, int((y2 - y1) * top_fraction)) if top_fraction > 0 else y2
                torso_area = (x2 - x1) * (torso_y2 - y1)
                overlap = 0.0
                for other in tracks:
                    if other is track or getattr(other, "time_since_update", 0) > 0:
                        continue
                    if getattr(other, "class_name", "player").lower() == "disc":
                        continue
                    other_box = (
                        other.to_tlbr()
                        if hasattr(other, "to_tlbr")
                        else getattr(other, "bbox", None)
                    )
                    if other_box is None:
                        continue
                    ox1, oy1, ox2, oy2 = other_box
                    intersection = max(0, min(x2, ox2) - max(x1, ox1)) * max(
                        0, min(torso_y2, oy2) - max(y1, oy1)
                    )
                    overlap = max(overlap, intersection / torso_area)
                score = 0.0
                if crop.shape[1] >= crop_config.get("min_crop_width", 20) and crop.shape[
                    0
                ] >= crop_config.get("min_crop_height", 30):
                    score = crop_quality(
                        crop,
                        top_fraction,
                        overlap,
                        float(get_setting("models.player_id.crop_selection.min_sharpness", 5.0)),
                    )
                crop_selector.observe(track_id, crop, frame_index, ocr_frame_interval, score)
                if not due:
                    skipped_interval += 1
                    continue
                crop = crop_selector.take(track_id, frame_index)
                if crop is None:
                    continue

            if crop.size > 0:
                player_crops.append(crop)
                track_metadata.append(
                    {"track_id": track_id, "bbox": (x1, y1, x2, y2), "crop_width": crop.shape[1]}
                )
                tracks_for_ocr.append(track_id)
            else:
                player_identifications[track_id] = ("Unknown", None)

        except Exception as e:
            logger.error(f"Error preparing track: {e}")
            continue

    selection_ms = (time.perf_counter() - selection_start) * 1000
    if player_crops and background:
        # The reader works on these while the next frames are analysed; what it finds is
        # recorded at the start of a later call
        _background_reader.submit(player_crops, track_metadata)
    elif player_crops:
        batch_results, batch_timing = _read_jersey_numbers(player_crops)
        for metadata, (jersey_number, details) in zip(track_metadata, batch_results):
            player_identifications[metadata["track_id"]] = record(metadata, jersey_number, details)

    # Add batch timing to totals
    total_timing["preprocessing_ms"] += batch_timing.get("preprocessing_ms", 0.0)
    if selection_enabled:
        # Include crop scoring and collection, excluding the batch's separately timed work.
        total_timing["preprocessing_ms"] += selection_ms
    total_timing["ocr_ms"] += batch_timing.get("ocr_ms", 0.0)
    total_timing["filtering_ms"] += batch_timing.get("filtering_ms", 0.0)

    # Attach simple counters for caller visibility (in timing dict to avoid signature change)
    total_timing["tracks_total"] = len(tracks)
    total_timing["tracks_ocr"] = len(tracks_for_ocr)
    total_timing["tracks_skipped_interval"] = skipped_interval
    total_timing["tracks_skipped_finalized"] = skipped_finalized
    return player_identifications, total_timing, finalized_tracks


def _record_reading(
    metadata: Dict[str, Any],
    jersey_number: str,
    details: Optional[Dict],
    frame_index: int,
    crop_selector: Optional[JerseyCropSelector],
    ocr_frame_interval: int,
    finalized_tracks: Set[int],
    finalized_threshold: float,
    verbose_debug: bool,
) -> Tuple[str, Dict[str, Any]]:
    """Take one reading of a player into account; returns (number to show, details).

    The reading is added to the player's votes, the crop selector learns whether the crop
    was readable, and the player is finalized once the votes are certain enough.
    """
    track_id = metadata["track_id"]
    if crop_selector is not None:
        crop_selector.record_read(
            track_id,
            frame_index,
            jersey_number != "Unknown",
            ocr_frame_interval,
            int(get_setting("models.player_id.crop_selection.max_backoff", 4)),
        )

    if jersey_number and jersey_number != "Unknown" and details:
        # Where across the crop the number was read (0 = left edge, 1 = right edge)
        centres = []
        for reading in details.get("ocr_results") or []:
            box = reading[0] if len(reading) >= 2 else None
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            if isinstance(box[0], (list, tuple)):  # Corner points
                centre = sum(point[0] for point in box) / len(box)
            else:  # x1, y1, x2, y2
                centre = (box[0] + box[2]) / 2
            centres.append(centre / metadata["crop_width"])
        position = max(0.0, min(1.0, sum(centres) / len(centres))) if centres else 0.5
        add_jersey_measurement(track_id, jersey_number, details.get("confidence", 0.0), position)

    tracking_history = get_jersey_probabilities(track_id, top_k=3)
    best_number, best_probability = get_best_jersey_number(track_id)
    is_finalized = bool(best_number) and best_probability >= finalized_threshold
    if is_finalized:
        finalized_tracks.add(track_id)

    enhanced_details = details.copy() if details else {}
    enhanced_details.update(
        {
            "single_frame": {
                "jersey_number": jersey_number,
                "confidence": details.get("confidence", 0.0) if details else 0.0,
            },
            "tracking_history": tracking_history,
            "best_tracked": {"jersey_number": best_number, "probability": best_probability},
            "finalized": is_finalized,
            # Only attach tracker object for debugging (it is large)
            **({"jersey_tracker": get_jersey_tracker()} if verbose_debug else {}),
        }
    )
    if verbose_debug:
        logger.debug(
            f"Track {track_id}: single='{jersey_number}' tracked='{best_number}' "
            f"({best_probability:.3f})"
        )
    # A number is shown once the readings so far add up to one; a single reading is too
    # often wrong to be shown on its own
    return best_number or "Unknown", enhanced_details


class _BackgroundReader:
    """Reads jersey numbers on a thread of its own, one batch of crops at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job: Optional[tuple] = None  # (generation, crops, metadata) being read
        self._finished: List[tuple] = []  # (metadata, results) not yet collected
        self._generation = 0
        self._thread: Optional[threading.Thread] = None

    def collect(self) -> Tuple[List[tuple], bool]:
        """What was read since the last call, and whether the reader is free for more."""
        with self._lock:
            finished, self._finished = self._finished, []
            return finished, self._job is None

    def submit(self, crops: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        """Hand over crops to read. Only call when collect() said the reader is free."""
        with self._lock:
            self._job = (self._generation, crops, metadata)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="jersey-reader", daemon=True)
                self._thread.start()
        self._wake.set()

    def discard(self) -> None:
        """Forget what is being read and what was read (another video, a seek)."""
        with self._lock:
            self._generation += 1
            self._finished = []

    def wait(self, timeout: float = 10.0) -> None:
        """Block until the reader is free (for tests and benchmarks)."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._job is None:
                    return
            time.sleep(0.001)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                job = self._job
            if job is None:
                continue
            generation, crops, metadata = job
            try:
                results, _ = _read_jersey_numbers(crops)
            except Exception as e:
                logger.exception(f"Error reading jersey numbers in the background: {e}")
                results = None
            with self._lock:
                if results is not None and generation == self._generation:
                    self._finished.append((metadata, results))
                self._job = None


_background_reader = _BackgroundReader()
# A batch for the background reader is kept small: while it is read, the GPU is shared
# with the detection of the frames that go on
MAX_BACKGROUND_CROPS = 4


def discard_pending_readings() -> None:
    """Drop the readings the background reader has not delivered yet (seek, new video)."""
    _background_reader.discard()


MIN_READING_CONFIDENCE = 0.5  # Readings the reader is less sure of are dropped


def _read_jersey_numbers(
    crop_images: List[np.ndarray],
) -> Tuple[List[Tuple[str, Optional[Dict]]], Dict[str, float]]:
    """Read the jersey number on each player crop.

    Args:
        crop_images: Player crops as cut from the frame

    Returns:
        (results, timing). `results` holds (jersey number or "Unknown", details or None)
        for each crop; the details carry the confidence, the readings, and the sizes of
        the crop before and after preprocessing. `timing` gives the milliseconds spent
        on preprocessing, reading ("ocr_ms") and picking the number ("filtering_ms").

    Crops are read one after another: they all share a single GPU model, so a
    thread pool measured no faster than this loop and gave identical results.
    """
    timing = {"preprocessing_ms": 0.0, "ocr_ms": 0.0, "filtering_ms": 0.0}
    unread: List[Tuple[str, Optional[Dict]]] = [("Unknown", None) for _ in crop_images]
    if not crop_images:
        return unread, timing

    # One read at a time: the reader models are shared with the background reader. And
    # never while a GPU engine is being set up on another thread.
    with GPU_SETUP_LOCK:
        return _read_crops(crop_images, unread, timing)


def _read_crops(
    crop_images: List[np.ndarray],
    unread: List[Tuple[str, Optional[Dict]]],
    timing: Dict[str, float],
) -> Tuple[List[Tuple[str, Optional[Dict]]], Dict[str, float]]:
    if _easyocr_reader is None:
        _initialize_easyocr()
    if not EASYOCR_AVAILABLE or _easyocr_reader is None:
        logger.debug("EasyOCR not available for batch processing")
        return unread, timing

    try:
        user_config = _load_easyocr_config()
        crop_config = user_config.get("preprocessing", {})
        verbose = get_setting("models.player_id.verbose_debug", False)
        # A reader other than EasyOCR takes the upper-body crop as it is; the contrast
        # and scaling steps are tuned for EasyOCR.
        reader = _get_active_reader()

        # Upper body of each crop, prepared for the reader; None for crops left out
        start = time.perf_counter()
        # The number is on the back: a player seen from the front or the side is not read
        seen_from = [facing.BACK] * len(crop_images)
        if get_setting("models.player_id.only_from_behind", False) and facing.available():
            seen_from = facing.facings(crop_images)
        prepared: List[Optional[np.ndarray]] = []
        metadata: List[Optional[Dict]] = []
        for i, crop_image in enumerate(crop_images):
            try:
                if seen_from[i] != facing.BACK:
                    raise ValueError(f"seen from the {seen_from[i] or 'unknown side'}")
                crop_height, crop_width = crop_image.shape[:2]
                if crop_width < crop_config.get(
                    "min_crop_width", 20
                ) or crop_height < crop_config.get("min_crop_height", 30):
                    if verbose:
                        logger.debug(f"Crop {i} too small ({crop_width}x{crop_height}), skipping")
                    raise ValueError("crop too small")
                upper_body = crop_top_fraction(crop_image, crop_config)
                final = preprocess_crop(upper_body, crop_config) if reader is None else upper_body
                prepared.append(final)
                metadata.append(
                    {
                        "original_width": crop_width,
                        "original_height": crop_height,
                        "crop_width": upper_body.shape[1],
                        "crop_height": upper_body.shape[0],
                        "final_width": final.shape[1],
                        "final_height": final.shape[0],
                        "crop_fraction": crop_config.get("crop_top_fraction", 0.33),
                    }
                )
            except Exception as e:
                if verbose:
                    logger.debug(f"Crop {i} not prepared: {e}")
                prepared.append(None)
                metadata.append(None)
        timing["preprocessing_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        valid_crops = [crop for crop in prepared if crop is not None]
        readings: List[list] = []
        if valid_crops and reader is not None:
            try:
                readings = reader.read(valid_crops)
            except Exception as e:
                logger.error(f"Error reading jersey numbers with {get_player_id_method()}: {e}")
                readings = [[] for _ in valid_crops]
        elif valid_crops:
            readtext_params = easyocr_readtext_parameters(user_config.get("easyocr", {}))
            for crop_index, crop in enumerate(valid_crops):
                try:
                    readings.append(_easyocr_reader.readtext(crop, **readtext_params))
                except Exception as e:
                    logger.error(f"Error processing crop {crop_index}: {e}")
                    readings.append([])
        timing["ocr_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        results: List[Tuple[str, Optional[Dict]]] = []
        readings_left = iter(readings)
        for details in metadata:
            crop_readings = next(readings_left, None) if details is not None else None
            if crop_readings is None:
                results.append(("Unknown", None))
                continue
            confident = [
                (bbox, text, confidence)
                for bbox, text, confidence in crop_readings
                if confidence >= MIN_READING_CONFIDENCE
            ]
            best_text, best_confidence = best_number(confident)
            readable = bool(best_text) and _validate_jersey_number(best_text)
            results.append(
                (
                    best_text if readable else "Unknown",
                    {
                        "confidence": best_confidence if readable else 0.0,
                        "ocr_results": confident,
                        "best_text": best_text if readable else None,
                        **details,
                    },
                )
            )
        timing["filtering_ms"] = (time.perf_counter() - start) * 1000
        return results, timing

    except Exception as e:
        logger.exception(f"Error in batch OCR processing: {e}")
        return unread, timing


def _load_easyocr_config() -> Dict[str, Any]:
    """Load EasyOCR configuration from easyocr_params.yaml file.

    This runs for every OCR batch, so the parsed file is cached and only re-read
    when its modification time changes (e.g. after saving from the tuning tab).
    """
    global _easyocr_config_path, _easyocr_config_mtime_ns, _easyocr_config_cache

    try:
        if _easyocr_config_path is None:
            # Find project root by looking for configs directory
            current_path = Path(__file__).parent
            project_root = None

            for parent in [current_path] + list(current_path.parents):
                if (parent / "configs").exists():
                    project_root = parent
                    break

            if project_root is None:
                if get_setting("models.player_id.verbose_debug", False):
                    logger.error("Could not find configs directory")
                return {}

            _easyocr_config_path = project_root / "configs" / "easyocr_params.yaml"

        config_path = _easyocr_config_path

        # Asking the file system costs half a millisecond; once a second is enough to
        # notice a save from the tuning tab
        global _easyocr_config_checked
        now = time.monotonic()
        if _easyocr_config_mtime_ns is not None and now - _easyocr_config_checked < 1.0:
            return _easyocr_config_cache
        _easyocr_config_checked = now

        try:
            mtime_ns = config_path.stat().st_mtime_ns
        except FileNotFoundError:
            if get_setting("models.player_id.verbose_debug", False):
                logger.warning(f"Config file not found: {config_path}")
            return {}

        if mtime_ns != _easyocr_config_mtime_ns:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            _easyocr_config_cache = config.get("player_id", {})
            _easyocr_config_mtime_ns = mtime_ns

        return _easyocr_config_cache

    except Exception as e:
        if get_setting("models.player_id.verbose_debug", False):
            logger.error(f"Error loading config: {e}")
        return {}


def _initialize_easyocr() -> None:
    """Initialize EasyOCR reader for text detection."""
    global _easyocr_reader

    if _easyocr_reader is not None:
        return

    if get_setting("models.player_id.verbose_debug", False):
        logger.debug("Initializing EasyOCR reader")

    try:
        if EASYOCR_AVAILABLE:
            # Load user configuration for language settings
            user_config = _load_easyocr_config()
            easyocr_config = user_config.get("easyocr", {})

            # Use English for jersey numbers
            languages = ["en"]
            gpu = easyocr_config.get("gpu", True)

            _easyocr_reader = easyocr.Reader(languages, gpu=gpu)
            if get_setting("models.player_id.verbose_debug", False):
                logger.debug("EasyOCR reader initialized successfully")
        else:
            logger.warning("EasyOCR not available, using mock reader")
            _easyocr_reader = None

    except Exception as e:
        logger.error(f"Failed to initialize EasyOCR: {e}")
        _easyocr_reader = None


def _validate_jersey_number(number_str: str) -> bool:
    """Validate that a detected jersey number is reasonable.

    Args:
        number_str: Detected number string

    Returns:
        True if number is valid jersey number, False otherwise
    """
    try:
        number = int(number_str)
        return JERSEY_NUMBER_MIN <= number <= JERSEY_NUMBER_MAX
    except ValueError:
        return False


def initialize_player_id_system() -> None:
    """Pre-initialize EasyOCR to avoid first-frame delay.

    This function should be called during application startup or video load
    to avoid the 2-5 second initialization delay that would otherwise occur
    during the first frame with player ID enabled.

    Safe to call multiple times - initialization only happens once.
    """
    logger.info("Pre-initializing player ID system...")
    _initialize_easyocr()
    _get_active_reader()
    if EASYOCR_AVAILABLE and _easyocr_reader is not None:
        logger.info("Player ID system pre-initialized successfully")
    else:
        logger.warning("Player ID system initialization skipped (EasyOCR not available)")
