"""Field segmentation module - YOLO-based field boundary detection.

This module handles segmenting the Ultimate Frisbee field boundaries and
identifying important field features like end zones and sidelines.
"""

from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..utils.logger import get_logger
from ..utils.model_files import default_model_path, get_training_image_size
from .tensorrt_engines import get_engine

logger = get_logger("FIELD_SEG")

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not available, field segmentation is disabled")

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


def _preprocess_frame_for_segmentation(frame: np.ndarray, target_size: int = 640) -> np.ndarray:
    """Stretch a frame to the square the segmentation model takes.

    The training images are 16:9 frames stretched to a square, so the model knows the field
    with exactly this distortion. Padding the frame to a square instead keeps the proportions
    but moves the predicted field outline several times further from the true one.
    """
    if frame.shape[:2] == (target_size, target_size):
        return frame
    return cv2.resize(frame, (target_size, target_size))


def _masks_to_numpy(results: List[Any]) -> List[Any]:
    """Move the masks off the GPU once; several stages read them.

    The masks stay at model resolution and cover the whole (stretched) frame. They are
    scaled back to the frame's shape once, after being combined, when the unified field
    mask is built.
    """
    for result in results:
        if getattr(result, "masks", None) is not None:
            masks_data = result.masks.data
            if hasattr(masks_data, "cpu"):
                result.masks.data = np.ascontiguousarray(masks_data.cpu().numpy())
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
        return []

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
        logger.debug("Loading default model on first use (lazy loading)")
        _load_default_model()

    if _field_model is None:
        logger.warning("No field segmentation model available")
        return []

    try:
        # Get segmentation parameters from config
        confidence_threshold = get_setting("models.segmentation.confidence_threshold", 0.25)
        iou_threshold = get_setting("models.segmentation.iou_threshold", 0.7)

        # Determine image size to use - prefer model's training size
        imgsz = _model_imgsz if _model_imgsz else 640

        preprocessed_frame = _preprocess_frame_for_segmentation(frame, imgsz)

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

        results = _masks_to_numpy(list(results))

        # Cache results for frame interval optimization
        _last_segmentation_results = results
        _last_segmentation_frame = frame_index
        _last_segmentation_shape = frame.shape

        return results

    except Exception as e:
        logger.exception(f"Error during field segmentation: {e}")
        return []


def warmup_field_model(frame_shape: Tuple[int, int, int] = (1080, 1920, 3)) -> None:
    """Load the field model and its engine now, on an empty frame of the video's size.

    Not only to keep the first frame from being the slow one: in the app, an engine
    that is first loaded after frames have been analysed crashes the process (an access
    violation inside TensorRT). Loaded with the video, it is there before any frame is.
    """
    run_field_segmentation(np.zeros(frame_shape, dtype=np.uint8), 0)
    reset_segmentation_cache()


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

    logger.info(f"Setting field segmentation model: {model_path}")

    # Validate model path
    if not Path(model_path).exists():
        logger.warning(f"Model file not found: {model_path}")
        return False

    if not ULTRALYTICS_AVAILABLE:
        return False

    try:
        _field_model = YOLO(model_path)
        _model_imgsz = get_training_image_size(model_path)
        logger.info(f"Field segmentation model loaded: {model_path} (image size {_model_imgsz})")

        reset_segmentation_cache()
        return True

    except Exception as e:
        logger.error(f"Failed to load field model {model_path}: {e}")
        return False


def _load_default_model() -> None:
    """Load the default field segmentation model if none is loaded."""
    if _field_model is None:
        # Get the default model path from configuration
        default_model = default_model_path("segmentation")

        logger.debug(f"Loading default segmentation model: {default_model}")

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
                logger.debug(f"Loading fallback segmentation model: {fallback_path}")
                set_field_model(str(fallback_path))
                return

        logger.debug("No field segmentation models found")
