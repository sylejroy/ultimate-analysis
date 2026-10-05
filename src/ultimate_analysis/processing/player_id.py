"""Player identification module - OCR and jersey number detection.

This module identifies players by their jersey numbers. The numbers are read by EasyOCR
or by one of the readers in jersey_readers.py (models.player_id.method), and combined
over time by probabilistic tracking.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

from ..config.settings import get_setting
from ..constants import JERSEY_NUMBER_MAX, JERSEY_NUMBER_MIN
from ..utils.logger import get_logger
from .jersey_crops import (
    best_number,
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

logger = get_logger("PLAYER_ID")

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available; jersey numbers will not be read")

# Global player ID state
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
) -> Tuple[Dict[int, Tuple[str, Any]], Dict[str, float], Set[int]]:
    """Run player identification on tracked objects using batch EasyOCR with probabilistic tracking.

    Args:
        frame: Current video frame
        tracks: List of track objects from tracking system

    Args:
        frame_index: Current global frame index (for interval/stagger logic)
        finalized_tracks: Set of track_ids whose jersey number is finalized (probability >= threshold)

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
    total_timing = {"preprocessing_ms": 0.0, "ocr_ms": 0.0, "filtering_ms": 0.0}
    batch_timing: Dict[str, float] = {"preprocessing_ms": 0.0, "ocr_ms": 0.0, "filtering_ms": 0.0}

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

            # Interval / stagger decision: only process a subset of tracks this frame
            if ocr_frame_interval > 1:
                if stagger_enabled:
                    # Stagger by track_id so load is distributed; each track processed every N frames
                    if (frame_index + track_id) % ocr_frame_interval != 0:
                        skipped_interval += 1
                        continue
                else:
                    if frame_index % ocr_frame_interval != 0:
                        skipped_interval += 1
                        continue

            # Ensure bbox is within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Crop the tracked object
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                player_crops.append(crop)
                track_metadata.append(
                    {"track_id": track_id, "bbox": (x1, y1, x2, y2), "crop_width": x2 - x1}
                )
                tracks_for_ocr.append(track_id)
            else:
                player_identifications[track_id] = ("Unknown", None)

        except Exception as e:
            logger.error(f"Error preparing track: {e}")
            continue

    # Run batch OCR processing if we have crops
    if player_crops:
        batch_results, batch_timing = _read_jersey_numbers(player_crops)

        # Process batch results
        for i, metadata in enumerate(track_metadata):
            if i < len(batch_results):
                track_id = metadata["track_id"]
                crop_width = metadata["crop_width"]
                jersey_number, details, timing = batch_results[i]

                # Add single-frame result to tracking history if valid
                if jersey_number and jersey_number != "Unknown" and details:
                    # Try to get OCR detection position, otherwise use bbox center
                    bbox_center_x = 0.5  # Default to center
                    if details and "ocr_results" in details and details["ocr_results"]:
                        # Calculate average x position of detected text
                        total_x = 0
                        count = 0
                        for ocr_result in details["ocr_results"]:
                            if len(ocr_result) >= 2:  # [bbox, text, confidence]
                                ocr_bbox = ocr_result[0]
                                if isinstance(ocr_bbox, (list, tuple)) and len(ocr_bbox) >= 4:
                                    # Calculate center x of OCR detection
                                    if isinstance(ocr_bbox[0], (list, tuple)):
                                        # Polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                        xs = [point[0] for point in ocr_bbox]
                                        center_x = sum(xs) / len(xs)
                                    else:
                                        # Box format: [x1, y1, x2, y2]
                                        center_x = (ocr_bbox[0] + ocr_bbox[2]) / 2

                                    # Normalize to 0-1 within crop
                                    bbox_center_x = center_x / crop_width
                                    total_x += bbox_center_x
                                    count += 1

                        if count > 0:
                            bbox_center_x = total_x / count
                            # Clamp to [0, 1]
                            bbox_center_x = max(0.0, min(1.0, bbox_center_x))

                    # Add measurement to tracker
                    confidence = details.get("confidence", 0.0)
                    ocr_results = details.get("ocr_results", [])
                    add_jersey_measurement(
                        track_id, jersey_number, confidence, bbox_center_x, ocr_results
                    )

                # Get tracking history and best tracked result
                tracking_history = get_jersey_probabilities(track_id, top_k=3)
                best_tracked_number, best_tracked_prob = get_best_jersey_number(track_id)

                # Finalization check
                is_finalized = False
                if best_tracked_number and best_tracked_prob >= finalized_threshold:
                    finalized_tracks.add(track_id)
                    is_finalized = True

                # Prepare enhanced details
                enhanced_details = details.copy() if details else {}
                enhanced_details.update(
                    {
                        "single_frame": {
                            "jersey_number": jersey_number,
                            "confidence": details.get("confidence", 0.0) if details else 0.0,
                        },
                        "tracking_history": tracking_history,
                        "best_tracked": {
                            "jersey_number": best_tracked_number,
                            "probability": best_tracked_prob,
                        },
                        "finalized": is_finalized,
                        # Only attach tracker object for debugging (it is large)
                        **({"jersey_tracker": get_jersey_tracker()} if verbose_debug else {}),
                    }
                )

                # Decide which result to return as primary
                if best_tracked_number and best_tracked_prob > 0.5:
                    # Use tracked result if high confidence
                    primary_result = best_tracked_number
                else:
                    # Fall back to single-frame detection
                    primary_result = jersey_number

                player_identifications[track_id] = (primary_result, enhanced_details)

                if verbose_debug:
                    logger.debug(
                        f"Track {track_id}: single='{jersey_number}' tracked='{best_tracked_number}' ({best_tracked_prob:.2%}) primary='{primary_result}' finalized={'yes' if track_id in finalized_tracks else 'no'}"
                    )

    # Add batch timing to totals
    total_timing["preprocessing_ms"] += batch_timing.get("preprocessing_ms", 0.0)
    total_timing["ocr_ms"] += batch_timing.get("ocr_ms", 0.0)
    total_timing["filtering_ms"] += batch_timing.get("filtering_ms", 0.0)

    # Attach simple counters for caller visibility (in timing dict to avoid signature change)
    total_timing["tracks_total"] = len(tracks)
    total_timing["tracks_ocr"] = len(tracks_for_ocr)
    total_timing["tracks_skipped_interval"] = skipped_interval
    total_timing["tracks_skipped_finalized"] = skipped_finalized
    return player_identifications, total_timing, finalized_tracks


def _read_jersey_numbers(
    crop_images: List[np.ndarray],
) -> Tuple[List[Tuple[str, Optional[Dict], Dict[str, float]]], Dict[str, float]]:
    """Run EasyOCR detection on a batch of cropped player images.

    Args:
        crop_images: List of cropped player images

    Returns:
        Tuple of (batch_results, total_timing)
        - batch_results: List of (jersey_number, result_details, individual_timing) for each crop
        - total_timing: Dict with 'preprocessing_ms' and 'ocr_ms' totals for the batch

    Performance Features:
        - Batch preprocessing to reduce setup overhead
        - Shared EasyOCR parameters across all crops

    Crops are read one after another: they all share a single GPU model, so a
    thread pool measured no faster than this loop and gave identical results.
    """
    batch_timing = {"preprocessing_ms": 0.0, "ocr_ms": 0.0, "filtering_ms": 0.0}
    batch_results = []

    if not crop_images:
        return batch_results, batch_timing

    if _easyocr_reader is None:
        _initialize_easyocr()

    if not EASYOCR_AVAILABLE or _easyocr_reader is None:
        logger.debug("EasyOCR not available for batch processing")
        # Return empty results for all crops
        for _ in crop_images:
            individual_timing = {"preprocessing_ms": 0.0, "ocr_ms": 0.0, "filtering_ms": 0.0}
            batch_results.append(("Unknown", None, individual_timing))
        return batch_results, batch_timing

    try:
        logger.debug(f"Starting batch OCR processing for {len(crop_images)} crops")

        # Start preprocessing timer for batch
        prep_start_time = time.perf_counter()

        # Load user configuration (same as individual processing)
        user_config = _load_easyocr_config()

        # A reader other than EasyOCR takes the upper-body crop as it is; the contrast
        # and scaling steps below are tuned for EasyOCR.
        reader = _get_active_reader()

        # Preprocess all crops in batch
        processed_crops = []
        crop_metadata = []  # Store metadata for each crop

        for i, crop_image in enumerate(crop_images):
            try:
                # Check minimum crop size
                crop_config = user_config.get("preprocessing", {})
                min_crop_width = crop_config.get("min_crop_width", 20)
                min_crop_height = crop_config.get("min_crop_height", 30)

                crop_height, crop_width = crop_image.shape[:2]
                if crop_width < min_crop_width or crop_height < min_crop_height:
                    if get_setting("models.player_id.verbose_debug", False):
                        logger.debug(
                            f"Batch crop {i} too small ({crop_width}x{crop_height}), skipping"
                        )
                    processed_crops.append(None)  # Placeholder for skipped crop
                    crop_metadata.append(None)
                    continue

                # Apply preprocessing pipeline
                processed_crop = crop_top_fraction(crop_image, crop_config)
                final_processed_crop = (
                    preprocess_crop(processed_crop, crop_config)
                    if reader is None
                    else processed_crop
                )

                processed_crops.append(final_processed_crop)
                crop_metadata.append(
                    {
                        "original_width": crop_width,
                        "original_height": crop_height,
                        "crop_width": processed_crop.shape[1],
                        "crop_height": processed_crop.shape[0],
                        "final_width": final_processed_crop.shape[1],
                        "final_height": final_processed_crop.shape[0],
                        "crop_fraction": crop_config.get("crop_top_fraction", 0.33),
                    }
                )

            except Exception as e:
                if get_setting("models.player_id.verbose_debug", False):
                    logger.debug(f"Error preprocessing batch crop {i}: {e}")
                processed_crops.append(None)
                crop_metadata.append(None)

        # End preprocessing timer
        batch_timing["preprocessing_ms"] = (time.perf_counter() - prep_start_time) * 1000

        # Start OCR timer for batch
        ocr_start_time = time.perf_counter()

        readtext_params = easyocr_readtext_parameters(user_config.get("easyocr", {}))

        # Process valid crops
        valid_crops = [crop for crop in processed_crops if crop is not None]
        batch_ocr_results = []

        if valid_crops and reader is not None:
            try:
                batch_ocr_results = reader.read(valid_crops)
            except Exception as e:
                logger.error(f"Error reading jersey numbers with {get_player_id_method()}: {e}")
                batch_ocr_results = [[] for _ in valid_crops]
        elif valid_crops:
            logger.debug(f"Running batch OCR on {len(valid_crops)} valid crops")
            for crop_index, crop in enumerate(valid_crops):
                try:
                    batch_ocr_results.append(_easyocr_reader.readtext(crop, **readtext_params))
                except Exception as e:
                    logger.error(f"Error processing crop {crop_index}: {e}")
                    batch_ocr_results.append([])

        # End OCR timer
        batch_timing["ocr_ms"] = (time.perf_counter() - ocr_start_time) * 1000

        # Process results for each original crop
        valid_crop_index = 0
        for i, crop in enumerate(processed_crops):
            individual_timing = {
                "preprocessing_ms": batch_timing["preprocessing_ms"]
                / len(crop_images),  # Distribute batch time
                "ocr_ms": batch_timing["ocr_ms"] / max(1, len(valid_crops)) if valid_crops else 0.0,
                "filtering_ms": 0.0,  # Will set per-crop below
            }

            if crop is None or crop_metadata[i] is None:
                # Skipped crop
                batch_results.append(("Unknown", None, individual_timing))
                continue

            # Get OCR results for this crop
            if valid_crop_index < len(batch_ocr_results):
                ocr_results = batch_ocr_results[valid_crop_index]
                valid_crop_index += 1

                # Filter low confidence detections (same as individual processing)
                per_crop_filter_start = time.perf_counter()
                min_confidence = 0.5
                filtered_ocr_results = []
                for bbox, text, confidence in ocr_results:
                    if confidence >= min_confidence:
                        filtered_ocr_results.append((bbox, text, confidence))

                best_text, best_confidence = best_number(filtered_ocr_results)

                # Prepare result
                if best_text and _validate_jersey_number(best_text):
                    jersey_number = best_text
                    result_details = {
                        "confidence": best_confidence,
                        "ocr_results": filtered_ocr_results,
                        "best_text": best_text,
                        **crop_metadata[i],  # Include preprocessing metadata
                    }
                else:
                    jersey_number = "Unknown"
                    result_details = {
                        "confidence": 0.0,
                        "ocr_results": filtered_ocr_results,
                        "best_text": None,
                        **crop_metadata[i],  # Include preprocessing metadata
                    }

                # End per-crop filtering timer and accumulate
                filtering_ms = (time.perf_counter() - per_crop_filter_start) * 1000
                individual_timing["filtering_ms"] = filtering_ms
                batch_timing["filtering_ms"] += filtering_ms

                batch_results.append((jersey_number, result_details, individual_timing))
            else:
                # No OCR results available
                batch_results.append(("Unknown", None, individual_timing))

        logger.debug(f"Batch OCR processing complete: {len(batch_results)} results")
        logger.debug(
            f"Batch timing - Preprocessing: {batch_timing['preprocessing_ms']:.1f}ms, OCR: {batch_timing['ocr_ms']:.1f}ms, Filtering: {batch_timing['filtering_ms']:.1f}ms"
        )

        return batch_results, batch_timing

    except Exception as e:
        logger.exception(f"Error in batch OCR processing: {e}")

        # Return empty results for all crops on error
        for _ in crop_images:
            individual_timing = {"preprocessing_ms": 0.0, "ocr_ms": 0.0, "filtering_ms": 0.0}
            batch_results.append(("Unknown", None, individual_timing))

        return batch_results, batch_timing


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
