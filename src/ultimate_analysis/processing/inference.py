"""Inference processing module - YOLO object detection.

This module handles running YOLO models for object detection on video frames.
Detects players, discs, and other relevant objects in Ultimate Frisbee games.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from ..config.settings import get_setting
from ..constants import FALLBACK_DEFAULTS
from ..utils.logger import get_logger
from .tensorrt_engines import get_engine

try:
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    YOLO_AVAILABLE = True
    # Newer Ultralytics replaced `half` with `quantize` and warns on every call otherwise
    FP16_KWARGS = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
except ImportError:
    print("[INFERENCE] Warning: ultralytics not available, inference will be disabled")
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


def reset_inference_state() -> None:
    """Resume disc detection when the video or playback position changes."""
    global _frames_since_last_disc
    _frames_since_last_disc = 0


def _get_model_training_params(model_path: str) -> Dict[str, Any]:
    """Extract training parameters from model's args.yaml file.

    Args:
        model_path: Path to the model file (.pt)

    Returns:
        Dictionary of training parameters, or empty dict if not found
    """
    try:
        model_path = Path(model_path)

        # Ultralytics writes args.yaml to the run folder, one level above weights/
        args_yaml_path = model_path.parent.parent / "args.yaml"
        if not args_yaml_path.exists():
            args_yaml_path = model_path.parent / "args.yaml"

        if args_yaml_path.exists():
            with open(args_yaml_path, "r") as f:
                args = yaml.safe_load(f)
                print(f"[INFERENCE] Loaded training parameters from {args_yaml_path}")
                return args if args else {}
        else:
            print(f"[INFERENCE] No args.yaml found at {args_yaml_path}")

    except Exception as e:
        print(f"[INFERENCE] Error reading model training parameters: {e}")

    return {}


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
        print(f"[INFERENCE] Error during {'/'.join(roles)} model inference: {e}")
        import traceback

        traceback.print_exc()

    timing["total"] = time.perf_counter() - total_start

    return detections, timing


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
            print(f"[INFERENCE] Absolute path does not exist: {model_path}")
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
        print(f"[INFERENCE] Model file not found: {model_path}")
        return None

    return model_file_path


def warmup_models() -> None:
    """Warmup YOLO models with a dummy inference to avoid cold-start penalty.

    Should be called after loading models and before first actual inference.
    This reduces first-frame latency by pre-allocating GPU memory and
    initializing CUDA kernels.
    """
    logger = get_logger("INFERENCE")

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

    if _player_model is not None and model_path == _player_model_path:
        return True

    if not YOLO_AVAILABLE:
        print("[INFERENCE] YOLO not available, cannot load player model")
        return False

    print(f"[INFERENCE] Setting player detection model: {model_path}")

    model_file_path = _resolve_model_path(model_path)
    if model_file_path is None:
        return False

    try:
        print(f"[INFERENCE] Loading player YOLO model from: {model_file_path}")
        # The same weights in both roles are loaded once and run in a single pass
        _player_model = _disc_model if model_path == _disc_model_path else YOLO(model_file_path)
        _player_model_path = model_path

        # Load training parameters to get the image size used during training
        training_params = _get_model_training_params(model_file_path)
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

    if _disc_model is not None and model_path == _disc_model_path:
        return True

    if not YOLO_AVAILABLE:
        print("[INFERENCE] YOLO not available, cannot load disc model")
        return False

    print(f"[INFERENCE] Setting disc detection model: {model_path}")

    model_file_path = _resolve_model_path(model_path)
    if model_file_path is None:
        return False

    try:
        print(f"[INFERENCE] Loading disc YOLO model from: {model_file_path}")
        # The same weights in both roles are loaded once and run in a single pass
        _disc_model = _player_model if model_path == _player_model_path else YOLO(model_file_path)
        _disc_model_path = model_path
        reset_inference_state()

        # Load training parameters to get the image size used during training
        training_params = _get_model_training_params(model_file_path)
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
    logger = get_logger("INFERENCE")

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
        print("[INFERENCE] Loading default models on first use (lazy loading)")
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
        logger.debug("[INFERENCE] ┌─ Running player model inference...")
        player_detections, player_timing = _run_single_model_inference(
            frame, _player_model, _player_model_imgsz, "models.player_detection", "player"
        )
        player_count = len(player_detections)
        all_detections.extend(player_detections)
    else:
        print("[INFERENCE] Player model not loaded")

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
        logger.debug("[INFERENCE] ┌─ Running disc model inference...")
        disc_detections, disc_timing = _run_single_model_inference(
            frame, _disc_model, _disc_model_imgsz, "models.disc_detection", "disc"
        )
        disc_count = len(disc_detections)
        all_detections.extend(disc_detections)

        # Update disc detection history
        if disc_count > 0:
            _frames_since_last_disc = 0
        else:
            _frames_since_last_disc += 1
    elif skip_disc:
        logger.debug(
            f"[INFERENCE] Skipping disc model (no disc for {_frames_since_last_disc} frames)"
        )
        _frames_since_last_disc += 1
        disc_timing = {"preprocessing": 0.0, "inference": 0.0, "postprocessing": 0.0, "total": 0.0}
    else:
        print("[INFERENCE] Disc model not loaded")

    total_inference_time = time.perf_counter() - total_inference_start

    if _player_model is None and _disc_model is None:
        print("[INFERENCE] No detection models available")

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
