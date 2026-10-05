"""Inference processing module - YOLO object detection.

This module handles running YOLO models for object detection on video frames.
Detects players, discs, and other relevant objects in Ultimate Frisbee games.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..config.settings import get_setting
from ..utils.logger import get_logger
from ..utils.model_files import default_model_path, get_training_image_size
from .tensorrt_engines import get_engine

logger = get_logger("INFERENCE")

try:
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    YOLO_AVAILABLE = True
    # Newer Ultralytics replaced `half` with `quantize` and warns on every call otherwise
    FP16_KWARGS = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
except ImportError:
    logger.warning("ultralytics not available, inference will be disabled")
    YOLO_AVAILABLE = False
    FP16_KWARGS = {}


# Global model cache - separate models for players and discs
_player_model = None
_player_model_path = None
_player_model_imgsz = None

_disc_model = None
_disc_model_path = None
_disc_model_imgsz = None

# Performance optimization: disc detection skipping
_frames_since_last_disc = 0

# Following the disc: where it was last seen, how it moved, and for how long it is missing
_disc_position: Optional[Tuple[float, float]] = None
_disc_velocity = (0.0, 0.0)  # Pixels per frame
_disc_frames_missing = 0
_frames_since_full_search = 0


def reset_inference_state() -> None:
    """Resume disc detection when the video or playback position changes."""
    global _frames_since_last_disc, _disc_position, _disc_velocity
    global _disc_frames_missing, _frames_since_full_search
    _frames_since_last_disc = 0
    _disc_position = None
    _disc_velocity = (0.0, 0.0)
    _disc_frames_missing = 0
    _frames_since_full_search = 0


def _disc_search_window(frame_shape: Tuple[int, ...]) -> Optional[Tuple[int, int, int, int]]:
    """Part of the frame (x1, y1, x2, y2) to look for the disc in, or None for all of it.

    While the disc is being followed it can only be near where it was, so a small window
    around its expected position is searched: several times faster than the whole frame,
    and a look-alike elsewhere in the picture cannot be mistaken for it. The whole frame
    is searched when the disc has not been seen lately, and every so often regardless, so
    that a wrong lock does not last.
    """
    size = int(get_setting("models.disc_detection.follow_window", 0))
    frame_h, frame_w = frame_shape[:2]
    if (
        size <= 0
        or size >= min(frame_h, frame_w)
        or _disc_position is None
        or _disc_frames_missing > int(get_setting("models.disc_detection.follow_max_missing", 5))
        or _frames_since_full_search
        >= int(get_setting("models.disc_detection.full_search_interval", 15))
    ):
        return None

    frames_ahead = _disc_frames_missing + 1
    center_x = _disc_position[0] + _disc_velocity[0] * frames_ahead
    center_y = _disc_position[1] + _disc_velocity[1] * frames_ahead
    x1 = int(min(max(center_x - size / 2, 0), frame_w - size))
    y1 = int(min(max(center_y - size / 2, 0), frame_h - size))
    return x1, y1, x1 + size, y1 + size


def disc_window_image_size(frame_shape: Tuple[int, ...], model_imgsz: Optional[int]) -> int:
    """Image size the disc model runs a search window at (0 if there is no window).

    The window is shown to the model at the scale it sees whole frames at. A TensorRT
    engine has to be built for this size as well (scripts/export_tensorrt.py), otherwise
    the window is searched with PyTorch, which is slower than the whole frame with an
    engine.
    """
    window = int(get_setting("models.disc_detection.follow_window", 0))
    if window <= 0 or window >= min(frame_shape[:2]):
        return 0
    scale = (model_imgsz or 640) / max(frame_shape[:2])
    return max(32, int(round(window * scale / 32)) * 32)


def _detect_disc(frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run the disc model, on a window around the followed disc when there is one."""
    global _disc_position, _disc_velocity, _disc_frames_missing, _frames_since_full_search

    window = _disc_search_window(frame.shape)
    if window is None:
        _frames_since_full_search = 0
        detections, timing = _run_single_model_inference(
            frame, _disc_model, _disc_model_imgsz, "models.disc_detection", "disc"
        )
    else:
        _frames_since_full_search += 1
        x1, y1, x2, y2 = window
        window_imgsz = disc_window_image_size(frame.shape, _disc_model_imgsz)
        detections, timing = _run_single_model_inference(
            np.ascontiguousarray(frame[y1:y2, x1:x2]),
            _disc_model,
            window_imgsz,
            "models.disc_detection",
            "disc",
        )
        for detection in detections:
            bx1, by1, bx2, by2 = detection["bbox"]
            detection["bbox"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]

    if detections:
        best = max(detections, key=lambda detection: detection["confidence"])
        position = (
            (best["bbox"][0] + best["bbox"][2]) / 2,
            (best["bbox"][1] + best["bbox"][3]) / 2,
        )
        if _disc_position is not None and _disc_frames_missing <= 5:
            frames = _disc_frames_missing + 1
            _disc_velocity = (
                (position[0] - _disc_position[0]) / frames,
                (position[1] - _disc_position[1]) / frames,
            )
        else:
            _disc_velocity = (0.0, 0.0)
        _disc_position = position
        _disc_frames_missing = 0
    else:
        _disc_frames_missing += 1
    return detections, timing


# Detected class -> (settings prefix, model type shown by the visualization)
_MODEL_ROLES = {
    "player": ("models.player_detection", "player_model"),
    "disc": ("models.disc_detection", "disc_model"),
}


def _run_single_model_inference(
    frame: np.ndarray, model: Any, model_imgsz: Optional[int], config_prefix: str, target_class: str
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run inference on a single model and return normalized detections with detailed timing.

    Args:
        frame: Input video frame
        model: Loaded YOLO model
        model_imgsz: Model's training image size
        config_prefix: Configuration prefix for thresholds (e.g., "models.player_detection")
        target_class: Class this model is used for ("player" or "disc")

    Returns:
        Tuple of (List of detection dictionaries, timing_breakdown dict)
    """
    model_type = _MODEL_ROLES.get(target_class, (config_prefix, "disc_model"))[1]
    return _predict_detections(
        frame, model, model_imgsz, {target_class: (config_prefix, model_type)}
    )


def _predict_detections(
    frame: np.ndarray,
    model: Any,
    model_imgsz: Optional[int],
    roles: Dict[str, Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run one model and return the detections of the requested classes.

    Args:
        frame: Input video frame
        model: Loaded YOLO model
        model_imgsz: Model's training image size
        roles: Class name -> (settings prefix for its thresholds, model type label)

    Returns:
        Tuple of (List of detection dictionaries, timing_breakdown dict)
    """
    detections: List[Dict[str, Any]] = []

    # Timing breakdown structure
    timing = {"preprocessing": 0.0, "inference": 0.0, "postprocessing": 0.0, "total": 0.0}

    total_start = time.perf_counter()

    try:
        # === PREPROCESSING PHASE ===
        preprocess_start = time.perf_counter()

        thresholds = {
            name: get_setting(f"{prefix}.confidence_threshold", 0.5)
            for name, (prefix, _) in roles.items()
        }
        first_prefix = next(iter(roles.values()))[0]
        nms_threshold = get_setting(f"{first_prefix}.nms_threshold", 0.45)

        # Only the classes this model is used for. A model trained on players and discs
        # would otherwise report its discs as players. Models that do not name their
        # classes this way (single role only) keep every detection, labelled with the role.
        model_names = getattr(model, "names", None)
        model_names = dict(model_names) if isinstance(model_names, dict) else {}
        class_ids = [cls for cls, name in model_names.items() if name in roles]
        fallback_class = None if class_ids else next(iter(roles))

        # Determine image size to use - prefer model's training size
        imgsz = model_imgsz if model_imgsz else 640

        # FP16 for ~2x speedup on compatible GPUs
        precision = FP16_KWARGS if get_setting("models.inference.half_precision", False) else {}

        # A TensorRT engine built for this model and frame size replaces the PyTorch model
        runtime_model, runtime_kwargs = model, {"imgsz": imgsz, **precision}
        engine = get_engine(model, frame.shape, imgsz, half=bool(precision))
        if engine is not None:
            runtime_model, runtime_kwargs = engine[0], {"imgsz": engine[1]}

        timing["preprocessing"] = time.perf_counter() - preprocess_start

        # === INFERENCE PHASE ===
        inference_start = time.perf_counter()
        results = runtime_model.predict(
            frame,
            conf=min(thresholds.values()),
            iou=nms_threshold,
            classes=class_ids or None,
            verbose=False,
            save=False,
            show=False,
            **runtime_kwargs,
        )
        timing["inference"] = time.perf_counter() - inference_start

        # === POSTPROCESSING PHASE ===
        postprocess_start = time.perf_counter()

        # Process results
        for result in results:
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = float(confidences[i])
                    cls = int(classes[i])
                    class_name = fallback_class or model_names[cls]

                    # Skip detections below this class's confidence threshold
                    if conf < thresholds[class_name]:
                        continue

                    detections.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                            "class_id": cls,
                            "class_name": class_name,
                            "model_type": roles[class_name][1],  # For visualization
                        }
                    )

        timing["postprocessing"] = time.perf_counter() - postprocess_start

    except Exception as e:
        logger.exception(f"Error during {'/'.join(roles)} model inference: {e}")

    timing["total"] = time.perf_counter() - total_start

    return detections, timing


def load_detection_model(model_path: str) -> Optional[Tuple[Any, int]]:
    """Load detection weights for use outside the live pipeline.

    The pipeline's own player and disc models are set with set_player_model and
    set_disc_model; a tool that analyses single frames loads its own copy here, so
    choosing a model in the tool does not change what the pipeline runs.

    Returns:
        (model, image size it runs at), or None if the weights cannot be loaded
    """
    model_file_path = _resolve_model_path(model_path)
    if model_file_path is None or not YOLO_AVAILABLE:
        return None
    try:
        return YOLO(model_file_path), get_training_image_size(model_file_path)
    except Exception as e:
        logger.error(f"Failed to load detection model {model_path}: {e}")
        return None


def detect_players(frame: np.ndarray, model: Any, model_imgsz: int) -> List[Dict[str, Any]]:
    """Players a model finds in a frame, with the pipeline's player detection settings."""
    detections, _ = _run_single_model_inference(
        frame, model, model_imgsz, "models.player_detection", "player"
    )
    return detections


def detect_discs(frame: np.ndarray, model: Any, model_imgsz: int) -> List[Dict[str, Any]]:
    """Discs a model finds in a whole frame, with the pipeline's disc detection settings."""
    detections, _ = _run_single_model_inference(
        frame, model, model_imgsz, "models.disc_detection", "disc"
    )
    return detections


def _resolve_model_path(model_path: str) -> Optional[str]:
    """Resolve a model path to an absolute file path.

    Args:
        model_path: Model path (absolute, relative, or just filename)

    Returns:
        Absolute path to model file or None if not found
    """
    # Handle different model path formats
    model_file_path = None

    # If it's an absolute path or contains path separators, use it directly
    if Path(model_path).is_absolute() or "/" in model_path or "\\" in model_path:
        if Path(model_path).exists():
            model_file_path = model_path
        else:
            logger.warning(f"Absolute path does not exist: {model_path}")
            return None
    else:
        # If it's just a filename, try to find it in the models directory
        models_path = Path(get_setting("models.base_path", "data/models"))

        # Try pretrained models first
        pretrained_path = models_path / "pretrained" / model_path
        if pretrained_path.exists():
            model_file_path = str(pretrained_path)
        else:
            # Try detection models
            detection_path = models_path / "detection" / model_path
            if detection_path.exists():
                model_file_path = str(detection_path)

    # Validate model path exists
    if model_file_path is None or not Path(model_file_path).exists():
        logger.warning(f"Model file not found: {model_path}")
        return None

    return model_file_path


def warmup_models() -> None:
    """Warmup YOLO models with a dummy inference to avoid cold-start penalty.

    Should be called after loading models and before first actual inference.
    This reduces first-frame latency by pre-allocating GPU memory and
    initializing CUDA kernels.
    """

    if not YOLO_AVAILABLE:
        return

    # Ensure models are loaded
    if _player_model is None or _disc_model is None:
        _load_default_models()

    try:
        # Create dummy frame matching model input size
        player_size = _player_model_imgsz if _player_model_imgsz else 640
        disc_size = _disc_model_imgsz if _disc_model_imgsz else 640

        logger.info("Warming up inference models...")

        # Warmup player model
        if _player_model is not None:
            dummy_frame = np.zeros((player_size, player_size, 3), dtype=np.uint8)
            _ = _player_model.predict(dummy_frame, verbose=False, imgsz=player_size)
            logger.info("Player model warmed up")

        # Warmup disc model
        if _disc_model is not None:
            dummy_frame = np.zeros((disc_size, disc_size, 3), dtype=np.uint8)
            _ = _disc_model.predict(dummy_frame, verbose=False, imgsz=disc_size)
            logger.info("Disc model warmed up")

        logger.info("Model warmup complete")

    except Exception as e:
        logger.warning(f"Model warmup failed (non-critical): {e}")


def _load_default_models() -> None:
    """Load the default player and disc detection models if none are loaded."""
    global _player_model, _disc_model

    if not YOLO_AVAILABLE:
        return

    # Load default player model
    if _player_model is None:
        default_player_model = default_model_path("player_detection")
        logger.debug(f"Loading default player model: {default_player_model}")
        set_player_model(default_player_model)

    # Load default disc model
    if _disc_model is None:
        default_disc_model = default_model_path("disc_detection")
        logger.debug(f"Loading default disc model: {default_disc_model}")
        set_disc_model(default_disc_model)


def set_player_model(model_path: str) -> bool:
    """Set the player detection model to use for inference.

    Args:
        model_path: Path to the YOLO model file (.pt) or model name

    Returns:
        True if model loaded successfully, False otherwise
    """
    global _player_model, _player_model_path, _player_model_imgsz

    if _player_model is not None and model_path == _player_model_path:
        return True

    if not YOLO_AVAILABLE:
        logger.error("YOLO not available, cannot load player model")
        return False

    logger.info(f"Setting player detection model: {model_path}")

    model_file_path = _resolve_model_path(model_path)
    if model_file_path is None:
        return False

    try:
        logger.debug(f"Loading player YOLO model from: {model_file_path}")
        # The same weights in both roles are loaded once and run in a single pass
        _player_model = _disc_model if model_path == _disc_model_path else YOLO(model_file_path)
        _player_model_path = model_path

        _player_model_imgsz = get_training_image_size(model_file_path)

        logger.info(f"Player model loaded successfully: {model_path}")
        logger.debug(f"Player model image size: {_player_model_imgsz}")
        if hasattr(_player_model, "names"):
            logger.debug(f"Player model classes: {dict(_player_model.names)}")

        return True

    except Exception as e:
        logger.exception(f"Failed to load player model {model_path}: {e}")
        return False


def set_disc_model(model_path: str) -> bool:
    """Set the disc detection model to use for inference.

    Args:
        model_path: Path to the YOLO model file (.pt) or model name

    Returns:
        True if model loaded successfully, False otherwise
    """
    global _disc_model, _disc_model_path, _disc_model_imgsz

    if _disc_model is not None and model_path == _disc_model_path:
        return True

    if not YOLO_AVAILABLE:
        logger.error("YOLO not available, cannot load disc model")
        return False

    logger.info(f"Setting disc detection model: {model_path}")

    model_file_path = _resolve_model_path(model_path)
    if model_file_path is None:
        return False

    try:
        logger.debug(f"Loading disc YOLO model from: {model_file_path}")
        # The same weights in both roles are loaded once and run in a single pass
        _disc_model = _player_model if model_path == _player_model_path else YOLO(model_file_path)
        _disc_model_path = model_path
        reset_inference_state()

        _disc_model_imgsz = get_training_image_size(model_file_path)

        logger.info(f"Disc model loaded successfully: {model_path}")
        logger.debug(f"Disc model image size: {_disc_model_imgsz}")
        if hasattr(_disc_model, "names"):
            logger.debug(f"Disc model classes: {dict(_disc_model.names)}")

        return True

    except Exception as e:
        logger.exception(f"Failed to load disc model {model_path}: {e}")
        return False


def run_inference(
    frame: np.ndarray, return_timing: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, float]]]:
    """Run YOLO inference on a video frame using the player and disc models.

    When both roles use the same model it runs once and its detections are split by class.

    Args:
        frame: Input video frame as numpy array (H, W, C) in BGR format
        return_timing: If True, return tuple of (detections, timing_info)

    Returns:
        If return_timing=False:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - confidence: Detection confidence score
            - class_id: Integer class ID (local to each model)
            - class_name: String class name ('player' or 'disc')
            - model_type: String model type ('player_model' or 'disc_model')

        If return_timing=True:
            Tuple of (detections_list, timing_dict) where timing_dict contains:
            - player_timing: Player model timing breakdown in seconds
            - disc_timing: Disc model timing breakdown in seconds
            - total_time: Total inference time in seconds
            - player_count: Number of player detections
            - disc_count: Number of disc detections

    Example:
        detections = run_inference(frame)
        # Or with timing:
        detections, timing = run_inference(frame, return_timing=True)
        print(f"Inference took {timing['total_time']*1000:.1f}ms")
    """

    logger.debug(f"Processing frame with shape {frame.shape}")

    if not YOLO_AVAILABLE:
        logger.warning("YOLO not available, returning empty detections")
        if return_timing:
            return [], {
                "player_timing": {},
                "disc_timing": {},
                "total_time": 0.0,
                "player_count": 0,
                "disc_count": 0,
            }
        return []

    # Ensure we have models (lazy loading optimization)
    if _player_model is None or _disc_model is None:
        logger.debug("Loading default models on first use (lazy loading)")
        _load_default_models()

    # Collect detections from both models
    all_detections: List[Dict[str, Any]] = []
    total_inference_start = time.perf_counter()

    player_timing = {}
    disc_timing = {}
    player_count = 0
    disc_count = 0

    # One model in both roles: a single pass finds players and discs
    shared_model = _player_model is not None and _player_model is _disc_model

    # Run player detection
    if shared_model:
        all_detections, player_timing = _predict_detections(
            frame, _player_model, _player_model_imgsz, dict(_MODEL_ROLES)
        )
        disc_count = sum(1 for det in all_detections if det["class_name"] == "disc")
        player_count = len(all_detections) - disc_count
    elif _player_model is not None:
        logger.debug("┌─ Running player model inference...")
        player_detections, player_timing = _run_single_model_inference(
            frame, _player_model, _player_model_imgsz, "models.player_detection", "player"
        )
        player_count = len(player_detections)
        all_detections.extend(player_detections)
    else:
        logger.warning("Player model not loaded")

    # Run disc detection with adaptive skipping
    # Skip disc model if no disc detected in recent frames (optimization)
    global _frames_since_last_disc
    skip_disc = (
        get_setting("models.disc_detection.adaptive_skip", True)
        and _frames_since_last_disc > get_setting("models.disc_detection.skip_threshold", 30)
        # Periodically probe so a disc entering the frame can be detected again.
        and _frames_since_last_disc
        % max(1, int(get_setting("models.disc_detection.retry_interval", 30)))
        != 0
    )

    if shared_model:
        disc_timing = {"preprocessing": 0.0, "inference": 0.0, "postprocessing": 0.0, "total": 0.0}
    elif _disc_model is not None and not skip_disc:
        logger.debug("┌─ Running disc model inference...")
        disc_detections, disc_timing = _detect_disc(frame)
        disc_count = len(disc_detections)
        all_detections.extend(disc_detections)

        # Update disc detection history
        if disc_count > 0:
            _frames_since_last_disc = 0
        else:
            _frames_since_last_disc += 1
    elif skip_disc:
        logger.debug(f"Skipping disc model (no disc for {_frames_since_last_disc} frames)")
        _frames_since_last_disc += 1
        disc_timing = {"preprocessing": 0.0, "inference": 0.0, "postprocessing": 0.0, "total": 0.0}
    else:
        logger.warning("Disc model not loaded")

    total_inference_time = time.perf_counter() - total_inference_start

    if _player_model is None and _disc_model is None:
        logger.warning("No detection models available")

    logger.debug(f"Found {len(all_detections)} total detections")

    if return_timing:
        timing_info = {
            "player_timing": player_timing,
            "disc_timing": disc_timing,
            "total_time": total_inference_time,
            "player_count": player_count,
            "disc_count": disc_count,
        }
        return all_detections, timing_info

    return all_detections
