"""Field segmentation module - YOLO-based field boundary detection.

This module handles segmenting the Ultimate Frisbee field boundaries and
identifying important field features like end zones and sidelines.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[FIELD_SEG] Warning: ultralytics not available, using mock results")

from ..config.settings import get_setting
from .tensorrt_engines import get_engine

# Global field segmentation state
_field_model = None
_model_imgsz = None  # Store the model's training image size
_last_segmentation_frame = -1  # Track when we last ran segmentation
_last_segmentation_results = []  # Cache last segmentation results
_last_segmentation_shape = None


def reset_segmentation_cache() -> None:
    """Discard results after a model, video, or playback-position change."""
    global _last_segmentation_frame, _last_segmentation_results, _last_segmentation_shape
    _last_segmentation_frame = -1
    _last_segmentation_results = []
    _last_segmentation_shape = None


def _preprocess_frame_for_segmentation(
    frame: np.ndarray, target_size: int = 640
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Preprocess frame for segmentation by creating square input without stretching.

    Args:
        frame: Input frame (H, W, C)
        target_size: Target square size for the model

    Returns:
        Tuple of (preprocessed_frame, transform_info)
        - preprocessed_frame: Square frame ready for segmentation model
        - transform_info: Information needed to transform results back to original frame
    """
    original_h, original_w = frame.shape[:2]

    # Scale the frame to the model size first, then pad it to a square (letterboxing).
    # Padding the full-resolution frame first would resize mostly black pixels.
    scale_factor = target_size / max(original_h, original_w)
    content_h = max(1, round(original_h * scale_factor))
    content_w = max(1, round(original_w * scale_factor))
    pad_h = (target_size - content_h) // 2
    pad_w = (target_size - content_w) // 2

    square_frame = np.zeros((target_size, target_size, 3), dtype=frame.dtype)
    content = frame
    if (content_h, content_w) != (original_h, original_w):
        content = cv2.resize(frame, (content_w, content_h))
    square_frame[pad_h : pad_h + content_h, pad_w : pad_w + content_w] = content

    # Where the frame sits inside the square model input
    transform_info = {
        "target_size": target_size,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "content_h": content_h,
        "content_w": content_w,
    }

    return square_frame, transform_info


def _postprocess_segmentation_results(
    results: List[Any], transform_info: Dict[str, Any]
) -> List[Any]:
    """Remove the letterbox padding from the segmentation masks.

    The masks stay at model resolution; they are scaled to the frame once, after being
    combined, when the unified field mask is built.

    Args:
        results: Segmentation results from YOLO model
        transform_info: Transform information from preprocessing

    Returns:
        Segmentation results whose masks cover exactly the original frame
    """
    if not results:
        return results

    try:
        target_size = transform_info["target_size"]

        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                # Get mask data
                masks_data = result.masks.data
                if hasattr(masks_data, "cpu"):
                    masks_data = masks_data.cpu().numpy()
                elif hasattr(masks_data, "numpy"):
                    masks_data = masks_data.numpy()

                # The masks may be at a different resolution than the model input
                scale_y = masks_data.shape[1] / target_size
                scale_x = masks_data.shape[2] / target_size
                top = round(transform_info["pad_h"] * scale_y)
                left = round(transform_info["pad_w"] * scale_x)
                bottom = top + round(transform_info["content_h"] * scale_y)
                right = left + round(transform_info["content_w"] * scale_x)

                result.masks.data = np.ascontiguousarray(masks_data[:, top:bottom, left:right])

        return results

    except Exception as e:
        print(f"[FIELD_SEG] Error in postprocessing segmentation results: {e}")
        return results


def run_field_segmentation(frame: np.ndarray, frame_index: int = 0) -> List[Any]:
    """Run field segmentation on a single frame with frame interval optimization.

    Args:
        frame: Input video frame as numpy array (H, W, C) in BGR format
        frame_index: Current frame index for interval logic (optional)

    Returns:
        List of segmentation results with masks and field boundaries
    """
    global _field_model, _model_imgsz
    global _last_segmentation_frame, _last_segmentation_results
    global _last_segmentation_shape

    if not ULTRALYTICS_AVAILABLE:
        print("[FIELD_SEG] YOLO not available, returning mock results")
        return _create_mock_results(frame)

    # Check frame interval optimization
    frame_interval = get_setting("models.segmentation.frame_interval", 5)
    if frame_interval > 1 and frame_index > 0:
        frames_since_last = frame_index - _last_segmentation_frame
        if (
            _last_segmentation_frame >= 0
            and 0 <= frames_since_last < frame_interval
            and _last_segmentation_shape == frame.shape
        ):
            # Empty results are also valid; do not re-run an empty scene every frame.
            return _last_segmentation_results

    # Load default model if none is loaded (lazy loading optimization)
    if _field_model is None:
        print("[FIELD_SEG] Loading default model on first use (lazy loading)")
        _load_default_model()

    if _field_model is None:
        print("[FIELD_SEG] No field segmentation model available")
        return _create_mock_results(frame)

    try:
        # Get segmentation parameters from config
        confidence_threshold = get_setting("models.segmentation.confidence_threshold", 0.25)
        iou_threshold = get_setting("models.segmentation.iou_threshold", 0.7)

        # Determine image size to use - prefer model's training size
        imgsz = _model_imgsz if _model_imgsz else 640

        # Preprocess frame to square format without stretching
        preprocessed_frame, transform_info = _preprocess_frame_for_segmentation(frame, imgsz)

        # A TensorRT engine built for this model replaces the PyTorch model
        runtime_model, runtime_imgsz = _field_model, imgsz
        engine = get_engine(_field_model, preprocessed_frame.shape, imgsz, half=False)
        if engine is not None:
            runtime_model, runtime_imgsz = engine

        # Run YOLO segmentation on square image
        results = runtime_model.predict(
            preprocessed_frame,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=runtime_imgsz,
            verbose=False,
            save=False,
            show=False,
        )

        # Transform results back to original frame coordinates
        results = _postprocess_segmentation_results(list(results), transform_info)

        # Cache results for frame interval optimization
        _last_segmentation_results = results
        _last_segmentation_frame = frame_index
        _last_segmentation_shape = frame.shape

        return results

    except Exception as e:
        print(f"[FIELD_SEG] Error during field segmentation: {e}")
        import traceback

        traceback.print_exc()
        return _create_mock_results(frame)


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
                print(f"[FIELD_SEG] Loaded training parameters from {args_yaml_path}")
                return args if args else {}
        else:
            print(f"[FIELD_SEG] No args.yaml found at {args_yaml_path}")

    except Exception as e:
        print(f"[FIELD_SEG] Error reading model training parameters: {e}")

    return {}


def _create_mock_results(frame: np.ndarray) -> List[Any]:
    """Create mock segmentation results for testing when YOLO is unavailable."""

    # Create a mock result object for testing
    class MockResult:
        def __init__(self):
            self.masks = MockMasks()
            self.boxes = None
            self.classes = None

    class MockMasks:
        def __init__(self):
            h, w = frame.shape[:2]
            # Create a simple field mask (rectangular field area)
            mask = np.zeros((h, w), dtype=np.uint8)
            # Field area is roughly center 60% of frame
            field_h = int(h * 0.6)
            field_w = int(w * 0.8)
            start_y = (h - field_h) // 2
            start_x = (w - field_w) // 2
            mask[start_y : start_y + field_h, start_x : start_x + field_w] = 1

            self.data = np.array([mask])  # Shape: (1, H, W)

    return [MockResult()]


def set_field_model(model_path: str) -> bool:
    """Set the field segmentation model to use.

    Args:
        model_path: Path to the YOLO segmentation model file (.pt)

    Returns:
        True if model loaded successfully, False otherwise

    Example:
        success = set_field_model("data/models/segmentation/field_finder_best.pt")
    """
    global _field_model, _model_imgsz

    print(f"[FIELD_SEG] Setting field segmentation model: {model_path}")

    # Validate model path
    if not Path(model_path).exists():
        print(f"[FIELD_SEG] Model file not found: {model_path}")
        return False

    try:
        if ULTRALYTICS_AVAILABLE:
            # Load actual YOLO segmentation model
            _field_model = YOLO(model_path)
            print(f"[FIELD_SEG] YOLO model loaded successfully: {model_path}")

            # Load training parameters to get the image size used during training
            training_params = _get_model_training_params(model_path)
            _model_imgsz = training_params.get("imgsz", 640)
            print(f"[FIELD_SEG] Using model training image size: {_model_imgsz}")
        else:
            print(
                f"[FIELD_SEG] Ultralytics not available, model path stored for mock mode: {model_path}"
            )

        reset_segmentation_cache()
        return True

    except Exception as e:
        print(f"[FIELD_SEG] Failed to load field model {model_path}: {e}")
        return False


def _load_default_model() -> None:
    """Load the default field segmentation model if none is loaded."""
    if _field_model is None:
        # Get the default model path from configuration
        default_model = get_setting(
            "models.segmentation.default_model",
            "data/models/segmentation/20250826_1_segmentation_yolo11s-seg_field finder.v8i.yolov8/finetune_20250826_092226/weights/best.pt",
        )

        print(f"[FIELD_SEG] Loading default segmentation model: {default_model}")

        # Try the configured model path first
        if Path(default_model).exists():
            set_field_model(default_model)
            return

        # Fallback to other segmentation models
        models_base = Path(get_setting("models.base_path", "data/models"))
        fallback_paths = [
            models_base
            / "segmentation/20250826_1_segmentation_yolo11s-seg_field finder.v8i.yolov8/finetune_20250826_092226/weights/best.pt",
            models_base
            / "segmentation/field_finder_yolo11m-seg/segmentation_finetune/weights/best.pt",
            models_base / "segmentation/field_finder_yolo11m-seg/finetune/weights/best.pt",
            models_base
            / "segmentation/field_finder_yolo11n-seg/segmentation_finetune/weights/best.pt",
            models_base / "pretrained/yolo11m-seg.pt",
            models_base / "pretrained/yolo11n-seg.pt",
        ]

        for fallback_path in fallback_paths:
            if fallback_path.exists():
                print(f"[FIELD_SEG] Loading fallback segmentation model: {fallback_path}")
                set_field_model(str(fallback_path))
                return

        print("[FIELD_SEG] No field segmentation models found, will use mock results")

