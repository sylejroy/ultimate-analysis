"""Inference processing module - YOLO object detection.

This module handles running YOLO models for object detection on video frames.
Detects players, discs, and other relevant objects in Ultimate Frisbee games.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..config.settings import get_setting
from ..constants import FALLBACK_DEFAULTS
from ..utils.logger import get_logger
from ._inference_helpers import get_model_training_params, resolve_model_path

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print("[INFERENCE] Warning: ultralytics not available, inference will be disabled")
    YOLO_AVAILABLE = False


# Global model cache - separate models for players and discs
_player_model = None
_player_model_path = None
_player_model_imgsz = None

_disc_model = None
_disc_model_path = None
_disc_model_imgsz = None

# Backward compatibility
_detection_model = None
_current_model_path = None
_model_imgsz = None


# training params & path resolution moved to _inference_helpers.py


def _run_single_model_inference(
    frame: np.ndarray, model: Any, model_imgsz: Optional[int], config_prefix: str, target_class: str
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run inference on a single model and return normalized detections with detailed timing.

    Args:
        frame: Input video frame
        model: Loaded YOLO model
        model_imgsz: Model's training image size
        config_prefix: Configuration prefix for thresholds (e.g., "models.player_detection")
        target_class: Target class name to assign to all detections

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

        # Get inference parameters from config
        confidence_threshold = get_setting(f"{config_prefix}.confidence_threshold", 0.5)
        nms_threshold = get_setting(f"{config_prefix}.nms_threshold", 0.45)

        # Determine image size to use - prefer model's training size
        imgsz = model_imgsz if model_imgsz else 640

        # Determine model type from config prefix
        model_type = "player_model" if "player_detection" in config_prefix else "disc_model"

        timing["preprocessing"] = time.perf_counter() - preprocess_start

        # === INFERENCE PHASE ===
        inference_start = time.perf_counter()
        results = model.predict(
            frame,
            conf=confidence_threshold,
            iou=nms_threshold,
            imgsz=imgsz,
            verbose=False,
            save=False,
            show=False,
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

                    # Skip detections below confidence threshold
                    if conf < confidence_threshold:
                        continue

                    detections.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                            "class_id": cls,
                            "class_name": target_class,  # Explicitly set target class
                            "model_type": model_type,  # Add model type for visualization differentiation
                        }
                    )

        timing["postprocessing"] = time.perf_counter() - postprocess_start

    except Exception as e:
        print(f"[INFERENCE] Error during {target_class} model inference: {e}")
        import traceback

        traceback.print_exc()

    timing["total"] = time.perf_counter() - total_start

    # Enhanced logging with subcategories (debug level to avoid spamming)
    logger = get_logger("INFERENCE")
    logger.debug(f"{target_class.title()} model breakdown:")
    logger.debug(f"  ├─ Preprocessing: {timing['preprocessing']*1000:5.1f}ms")
    logger.debug(f"  ├─ Inference:     {timing['inference']*1000:5.1f}ms")
    logger.debug(f"  ├─ Postprocessing:{timing['postprocessing']*1000:5.1f}ms")
    logger.debug(
        f"  └─ Total:         {timing['total']*1000:5.1f}ms ({len(detections)} detections)"
    )

    return detections, timing


_resolve_model_path = resolve_model_path  # backward alias


def _load_default_models() -> None:
    """Load the default player and disc detection models if none are loaded."""
    global _player_model, _disc_model

    if not YOLO_AVAILABLE:
        return

    # Load default player model
    if _player_model is None:
        default_player_model = get_setting(
            "models.player_detection.default_model", FALLBACK_DEFAULTS["model_player_detection"]
        )
        print(f"[INFERENCE] Loading default player model: {default_player_model}")
        set_player_model(default_player_model)

    # Load default disc model
    if _disc_model is None:
        default_disc_model = get_setting(
            "models.disc_detection.default_model", FALLBACK_DEFAULTS["model_disc_detection"]
        )
        print(f"[INFERENCE] Loading default disc model: {default_disc_model}")
        set_disc_model(default_disc_model)


def set_player_model(model_path: str) -> bool:
    """Set the player detection model to use for inference.

    Args:
        model_path: Path to the YOLO model file (.pt) or model name

    Returns:
        True if model loaded successfully, False otherwise
    """
    global _player_model, _player_model_path, _player_model_imgsz

    if not YOLO_AVAILABLE:
        print("[INFERENCE] YOLO not available, cannot load player model")
        return False

    print(f"[INFERENCE] Setting player detection model: {model_path}")

    model_file_path = _resolve_model_path(model_path)
    if model_file_path is None:
        return False

    try:
        print(f"[INFERENCE] Loading player YOLO model from: {model_file_path}")
        _player_model = YOLO(model_file_path)
        _player_model_path = model_path

        # Load training parameters to get the image size used during training
        training_params = get_model_training_params(model_file_path)
        _player_model_imgsz = training_params.get("imgsz", 640)

        print(f"[INFERENCE] Player model loaded successfully: {model_path}")
        print(f"[INFERENCE] Player model image size: {_player_model_imgsz}")
        if hasattr(_player_model, "names"):
            print(f"[INFERENCE] Player model classes: {dict(_player_model.names)}")

        return True

    except Exception as e:
        print(f"[INFERENCE] Failed to load player model {model_path}: {e}")
        import traceback

        traceback.print_exc()
        return False


def set_disc_model(model_path: str) -> bool:
    """Set the disc detection model to use for inference.

    Args:
        model_path: Path to the YOLO model file (.pt) or model name

    Returns:
        True if model loaded successfully, False otherwise
    """
    global _disc_model, _disc_model_path, _disc_model_imgsz

    if not YOLO_AVAILABLE:
        print("[INFERENCE] YOLO not available, cannot load disc model")
        return False

    print(f"[INFERENCE] Setting disc detection model: {model_path}")

    model_file_path = _resolve_model_path(model_path)
    if model_file_path is None:
        return False

    try:
        print(f"[INFERENCE] Loading disc YOLO model from: {model_file_path}")
        _disc_model = YOLO(model_file_path)
        _disc_model_path = model_path

        # Load training parameters to get the image size used during training
        training_params = get_model_training_params(model_file_path)
        _disc_model_imgsz = training_params.get("imgsz", 640)

        print(f"[INFERENCE] Disc model loaded successfully: {model_path}")
        print(f"[INFERENCE] Disc model image size: {_disc_model_imgsz}")
        if hasattr(_disc_model, "names"):
            print(f"[INFERENCE] Disc model classes: {dict(_disc_model.names)}")

        return True

    except Exception as e:
        print(f"[INFERENCE] Failed to load disc model {model_path}: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_inference(
    frame: np.ndarray, model_name: Optional[str] = None, return_timing: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, float]]]:
    """Run YOLO inference on a video frame using separate player and disc models.

    Args:
        frame: Input video frame as numpy array (H, W, C) in BGR format
        model_name: Optional model name for backward compatibility (will set as player model)
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
            - player_time: Player model inference time in seconds
            - disc_time: Disc model inference time in seconds
            - total_time: Total inference time in seconds
            - player_count: Number of player detections
            - disc_count: Number of disc detections

    Example:
        detections = run_inference(frame)
        # Or with timing:
        detections, timing = run_inference(frame, return_timing=True)
        print(f"Player model took {timing['player_time']*1000:.1f}ms")
    """
    global _player_model, _disc_model, _detection_model, _current_model_path
    logger = get_logger("INFERENCE")

    logger.debug(f"Processing frame with shape {frame.shape}")

    if not YOLO_AVAILABLE:
        logger.warning("YOLO not available, returning empty detections")
        return []

    # Handle backward compatibility
    if model_name and model_name != _current_model_path:
        print(
            "[INFERENCE] Warning: set_detection_model is deprecated, use set_player_model and set_disc_model"
        )
        if not set_player_model(model_name):
            print(f"[INFERENCE] Failed to load model {model_name} as player model")

    # Ensure we have models (lazy loading optimization)
    if _player_model is None or _disc_model is None:
        print("[INFERENCE] Loading default models on first use (lazy loading)")
        _load_default_models()

    # Collect detections from both models
    all_detections: List[Dict[str, Any]] = []
    total_inference_start = time.perf_counter()

    player_timing = {}
    disc_timing = {}
    player_count = 0
    disc_count = 0

    # Run player detection
    if _player_model is not None:
        logger.debug(f"[INFERENCE] ┌─ Running player model inference...")
        player_detections, player_timing = _run_single_model_inference(
            frame, _player_model, _player_model_imgsz, "models.player_detection", "player"
        )
        player_count = len(player_detections)
        all_detections.extend(player_detections)
    else:
        print("[INFERENCE] Player model not loaded")

    # Run disc detection
    if _disc_model is not None:
        logger.debug(f"[INFERENCE] ┌─ Running disc model inference...")
        disc_detections, disc_timing = _run_single_model_inference(
            frame, _disc_model, _disc_model_imgsz, "models.disc_detection", "disc"
        )
        disc_count = len(disc_detections)
        all_detections.extend(disc_detections)
    else:
        print("[INFERENCE] Disc model not loaded")

    total_inference_time = time.perf_counter() - total_inference_start

    # Hierarchical timing summary (debug level to avoid spamming)
    logger = get_logger("INFERENCE")
    logger.debug("═══ INFERENCE TIMING SUMMARY ═══")

    if player_timing:
        logger.debug(f"Player Model ({player_count} detections):")
        logger.debug(f"  ├─ Preprocessing: {player_timing.get('preprocessing', 0)*1000:5.1f}ms")
        logger.debug(f"  ├─ Inference:     {player_timing.get('inference', 0)*1000:5.1f}ms")
        logger.debug(f"  ├─ Postprocessing:{player_timing.get('postprocessing', 0)*1000:5.1f}ms")
        logger.debug(f"  └─ Subtotal:      {player_timing.get('total', 0)*1000:5.1f}ms")
    else:
        logger.debug("Player Model: Not loaded")

    if disc_timing:
        logger.debug(f"Disc Model ({disc_count} detections):")
        logger.debug(f"  ├─ Preprocessing: {disc_timing.get('preprocessing', 0)*1000:5.1f}ms")
        logger.debug(f"  ├─ Inference:     {disc_timing.get('inference', 0)*1000:5.1f}ms")
        logger.debug(f"  ├─ Postprocessing:{disc_timing.get('postprocessing', 0)*1000:5.1f}ms")
        logger.debug(f"  └─ Subtotal:      {disc_timing.get('total', 0)*1000:5.1f}ms")
    else:
        logger.debug("Disc Model: Not loaded")

    logger.debug("─────────────────────────────────")
    logger.debug(f"TOTAL TIME:      {total_inference_time*1000:5.1f}ms")
    logger.debug(f"TOTAL DETECTIONS: {len(all_detections)}")

    # Performance comparison
    player_total = player_timing.get("total", 0) if player_timing else 0
    disc_total = disc_timing.get("total", 0) if disc_timing else 0

    if player_total > 0 and disc_total > 0:
        logger.debug(f"MODEL RATIO:     {player_total/disc_total:.2f}x (Player/Disc)")

        # Show which phases take the most time
        player_inference_time = player_timing.get("inference", 0)
        disc_inference_time = disc_timing.get("inference", 0)
        total_inference_only = player_inference_time + disc_inference_time

        if total_inference_only > 0:
            inference_percentage = (total_inference_only / total_inference_time) * 100
            logger.debug(f"INFERENCE %:     {inference_percentage:.1f}% of total time")

    logger.debug("═══════════════════════════════════")

    # If no models are available, return debug detections
    if _player_model is None and _disc_model is None:
        print("[INFERENCE] No detection models available")
        debug_detections = []
        if get_setting("app.debug", False):
            h, w = frame.shape[:2]
            debug_detections = [
                {
                    "bbox": [w // 4, h // 4, 3 * w // 4, 3 * h // 4],
                    "confidence": 0.85,
                    "class_id": 0,
                    "class_name": "player",
                    "model_type": "player_model",
                },
                {
                    "bbox": [w // 2 - 20, h // 2 - 20, w // 2 + 20, h // 2 + 20],
                    "confidence": 0.75,
                    "class_id": 0,
                    "class_name": "disc",
                    "model_type": "disc_model",
                },
            ]

        if return_timing:
            timing_info = {
                "player_time": 0.0,
                "disc_time": 0.0,
                "total_time": 0.0,
                "player_count": 0,
                "disc_count": 0,
            }
            return debug_detections, timing_info
        return debug_detections

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


def set_detection_model(model_path: str) -> bool:
    """Set the detection model to use for inference (DEPRECATED).

    Args:
        model_path: Path to the YOLO model file (.pt) or model name

    Returns:
        True if model loaded successfully, False otherwise

    Note:
        This function is deprecated. Use set_player_model() and set_disc_model() instead.
        For backward compatibility, this will set the player model.
    """
    print(
        "[INFERENCE] Warning: set_detection_model is deprecated. Use set_player_model and set_disc_model instead."
    )
    return set_player_model(model_path)


def get_current_model_path() -> Optional[str]:
    """Get the path of the currently loaded detection model (DEPRECATED).

    Returns:
        Path to current model or None if no model loaded

    Note:
        This function is deprecated. Use get_current_model_paths() for both models.
    """
    return _player_model_path


def get_current_model_paths() -> Dict[str, Optional[str]]:
    """Get the paths of the currently loaded models.

    Returns:
        Dictionary with 'player' and 'disc' model paths
    """
    return {"player": _player_model_path, "disc": _disc_model_path}


def get_model_info() -> Dict[str, Any]:
    """Get information about the current detection models.

    Returns:
        Dictionary with combined model information:
        - player_model: Player model information
        - disc_model: Disc model information
        - loaded: Whether any models are loaded
    """
    player_classes = []
    disc_classes = []

    if _player_model is not None and hasattr(_player_model, "names"):
        player_classes = list(_player_model.names.values())

    if _disc_model is not None and hasattr(_disc_model, "names"):
        disc_classes = list(_disc_model.names.values())

    return {
        "player_model": {
            "path": _player_model_path,
            "classes": player_classes,
            "input_size": [_player_model_imgsz or 640, _player_model_imgsz or 640],
            "loaded": _player_model is not None,
        },
        "disc_model": {
            "path": _disc_model_path,
            "classes": disc_classes,
            "input_size": [_disc_model_imgsz or 640, _disc_model_imgsz or 640],
            "loaded": _disc_model is not None,
        },
        "loaded": _player_model is not None or _disc_model is not None,
    }


def _load_default_model() -> None:
    """Load the default detection model if none is loaded (DEPRECATED)."""
    print(
        "[INFERENCE] Warning: _load_default_model is deprecated. Use _load_default_models instead."
    )
    _load_default_models()


# Initialize with default model when first needed (lazy loading)
# This optimization prevents slow startup by deferring model loading until actually used
# _load_default_model()  # Commented out for performance optimization
