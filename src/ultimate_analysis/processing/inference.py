"""Inference processing module - YOLO object detection.

This module handles running YOLO models for object detection on video frames.
Detects players, discs, and other relevant objects in Ultimate Frisbee games.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from ..config.settings import get_setting
from ..constants import FALLBACK_DEFAULTS

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


def _get_model_training_params(model_path: str) -> Dict[str, Any]:
    """Extract training parameters from model's args.yaml file.

    Args:
        model_path: Path to the model file (.pt)

    Returns:
        Dictionary of training parameters, or empty dict if not found
    """
    try:
        model_path = Path(model_path)

        # Look for args.yaml in the same directory as the model
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


def _run_single_model_inference(
    frame: np.ndarray, model: Any, model_imgsz: Optional[int], config_prefix: str, target_class: str
) -> List[Dict[str, Any]]:
    """Run inference on a single model and return normalized detections.

    Args:
        frame: Input video frame
        model: Loaded YOLO model
        model_imgsz: Model's training image size
        config_prefix: Configuration prefix for thresholds (e.g., "models.player_detection")
        target_class: Target class name to assign to all detections

    Returns:
        List of detection dictionaries
    """
    detections: List[Dict[str, Any]] = []

    try:
        # Get inference parameters from config
        confidence_threshold = get_setting(f"{config_prefix}.confidence_threshold", 0.5)
        nms_threshold = get_setting(f"{config_prefix}.nms_threshold", 0.45)

        # Determine image size to use - prefer model's training size
        imgsz = model_imgsz if model_imgsz else 640

        # Run YOLO inference
        results = model.predict(
            frame,
            conf=confidence_threshold,
            iou=nms_threshold,
            imgsz=imgsz,
            verbose=False,
            save=False,
            show=False,
        )

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
                        }
                    )

    except Exception as e:
        print(f"[INFERENCE] Error during {target_class} model inference: {e}")
        import traceback

        traceback.print_exc()

    print(f"[INFERENCE] Found {len(detections)} {target_class} detections")
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


def run_inference(frame: np.ndarray, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run YOLO inference on a video frame using separate player and disc models.

    Args:
        frame: Input video frame as numpy array (H, W, C) in BGR format
        model_name: Optional model name for backward compatibility (will set as player model)

    Returns:
        List of detection dictionaries with keys:
        - bbox: [x1, y1, x2, y2] bounding box coordinates
        - confidence: Detection confidence score
        - class_id: Integer class ID (local to each model)
        - class_name: String class name ('player' or 'disc')

    Example:
        detections = run_inference(frame)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
    """
    global _player_model, _disc_model, _detection_model, _current_model_path

    print(f"[INFERENCE] Processing frame with shape {frame.shape}")

    if not YOLO_AVAILABLE:
        print("[INFERENCE] YOLO not available, returning empty detections")
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

    # Run player detection
    if _player_model is not None:
        player_detections = _run_single_model_inference(
            frame, _player_model, _player_model_imgsz, "models.player_detection", "player"
        )
        all_detections.extend(player_detections)

    # Run disc detection
    if _disc_model is not None:
        disc_detections = _run_single_model_inference(
            frame, _disc_model, _disc_model_imgsz, "models.disc_detection", "disc"
        )
        all_detections.extend(disc_detections)

    # If no models are available, return debug detections
    if _player_model is None and _disc_model is None:
        print("[INFERENCE] No detection models available")
        if get_setting("app.debug", False):
            h, w = frame.shape[:2]
            return [
                {
                    "bbox": [w // 4, h // 4, 3 * w // 4, 3 * h // 4],
                    "confidence": 0.85,
                    "class_id": 0,
                    "class_name": "player",
                },
                {
                    "bbox": [w // 2 - 20, h // 2 - 20, w // 2 + 20, h // 2 + 20],
                    "confidence": 0.75,
                    "class_id": 0,
                    "class_name": "disc",
                },
            ]
        return []

    print(f"[INFERENCE] Found {len(all_detections)} total detections")
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
