"""Field segmentation module - YOLO-based field boundary detection.

This module handles segmenting the Ultimate Frisbee field boundaries and 
identifying important field features like end zones and sidelines.
"""

import cv2
import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[FIELD_SEG] Warning: ultralytics not available, using mock results")

from ..config.settings import get_setting
from ..constants import FIELD_DIMENSIONS


# Global field segmentation state
_field_model = None
_current_model_path = None
_model_imgsz = None  # Store the model's training image size


def _preprocess_frame_for_segmentation(frame: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    
    # Create square canvas by padding to the larger dimension
    max_dim = max(original_h, original_w)
    
    # Calculate padding needed
    pad_h = (max_dim - original_h) // 2
    pad_w = (max_dim - original_w) // 2
    
    # Create square frame with padding (letterboxing)
    square_frame = np.zeros((max_dim, max_dim, 3), dtype=frame.dtype)
    square_frame[pad_h:pad_h + original_h, pad_w:pad_w + original_w] = frame
    
    # Resize to target size if needed
    if max_dim != target_size:
        square_frame = cv2.resize(square_frame, (target_size, target_size))
        scale_factor = target_size / max_dim
    else:
        scale_factor = 1.0
    
    # Store transform information for converting results back
    transform_info = {
        'original_h': original_h,
        'original_w': original_w,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'max_dim': max_dim,
        'scale_factor': scale_factor,
        'target_size': target_size
    }
    
    return square_frame, transform_info


def _postprocess_segmentation_results(results: List[Any], transform_info: Dict[str, Any]) -> List[Any]:
    """Transform segmentation results back to original frame coordinates.
    
    Args:
        results: Segmentation results from YOLO model
        transform_info: Transform information from preprocessing
        
    Returns:
        Segmentation results adjusted to original frame coordinates
    """
    if not results:
        return results
    
    try:
        original_h = transform_info['original_h']
        original_w = transform_info['original_w']
        pad_h = transform_info['pad_h']
        pad_w = transform_info['pad_w']
        scale_factor = transform_info['scale_factor']
        
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                # Get mask data
                masks_data = result.masks.data
                if hasattr(masks_data, 'cpu'):
                    masks_data = masks_data.cpu().numpy()
                elif hasattr(masks_data, 'numpy'):
                    masks_data = masks_data.numpy()
                
                # Transform each mask back to original coordinates
                transformed_masks = []
                for mask in masks_data:
                    # Scale back from target size to max_dim
                    if scale_factor != 1.0:
                        scaled_size = int(transform_info['target_size'] / scale_factor)
                        mask_scaled = cv2.resize(mask.astype(np.float32), (scaled_size, scaled_size))
                    else:
                        mask_scaled = mask
                    
                    # Remove padding to get back to original frame size
                    mask_original = mask_scaled[pad_h:pad_h + original_h, pad_w:pad_w + original_w]
                    
                    # Ensure correct size
                    if mask_original.shape != (original_h, original_w):
                        mask_original = cv2.resize(mask_original, (original_w, original_h))
                    
                    transformed_masks.append(mask_original)
                
                # Update the result with transformed masks
                result.masks.data = np.array(transformed_masks)
            
            # Transform bounding boxes if present
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes_data = result.boxes.xyxy
                if hasattr(boxes_data, 'cpu'):
                    boxes_data = boxes_data.cpu().numpy()
                elif hasattr(boxes_data, 'numpy'):
                    boxes_data = boxes_data.numpy()
                
                # Scale and translate boxes back to original coordinates
                transformed_boxes = []
                for box in boxes_data:
                    # Scale back from target size
                    box = box / scale_factor
                    # Remove padding offset
                    box[0] -= pad_w  # x1
                    box[1] -= pad_h  # y1
                    box[2] -= pad_w  # x2
                    box[3] -= pad_h  # y2
                    # Clamp to original frame bounds
                    box[0] = max(0, min(box[0], original_w))
                    box[1] = max(0, min(box[1], original_h))
                    box[2] = max(0, min(box[2], original_w))
                    box[3] = max(0, min(box[3], original_h))
                    transformed_boxes.append(box)
                
                # Note: We don't modify the original boxes data as it may be read-only
                # The calling code should handle coordinate transformation if needed
        
        return results
        
    except Exception as e:
        print(f"[FIELD_SEG] Error in postprocessing segmentation results: {e}")
        return results


def run_field_segmentation(frame: np.ndarray) -> List[Any]:
    """Run field segmentation on a single frame.
    
    Args:
        frame: Input video frame as numpy array (H, W, C) in BGR format
        
    Returns:
        List of segmentation results with masks and field boundaries
    """
    global _field_model, _current_model_path, _model_imgsz
    
    if not ULTRALYTICS_AVAILABLE:
        print("[FIELD_SEG] YOLO not available, returning mock results")
        return _create_mock_results(frame)
    
    # Load default model if none is loaded
    if _field_model is None:
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
        
        # Run YOLO segmentation on square image
        results = _field_model.predict(
            preprocessed_frame,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
            save=False,
            show=False
        )
        
        # Transform results back to original frame coordinates
        results = _postprocess_segmentation_results(list(results), transform_info)
        
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
        
        # Look for args.yaml in the same directory as the model
        args_yaml_path = model_path.parent / "args.yaml"
        
        if args_yaml_path.exists():
            with open(args_yaml_path, 'r') as f:
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
            mask[start_y:start_y+field_h, start_x:start_x+field_w] = 1
            
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
    global _field_model, _current_model_path, _model_imgsz
    
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
            _model_imgsz = training_params.get('imgsz', 640)
            print(f"[FIELD_SEG] Using model training image size: {_model_imgsz}")
        else:
            print(f"[FIELD_SEG] Ultralytics not available, model path stored for mock mode: {model_path}")
        
        _current_model_path = model_path
        return True
        
    except Exception as e:
        print(f"[FIELD_SEG] Failed to load field model {model_path}: {e}")
        return False


def get_current_field_model_path() -> Optional[str]:
    """Get the path of the currently loaded field segmentation model.
    
    Returns:
        Path to current model or None if no model loaded
    """
    return _current_model_path


def _load_default_model() -> None:
    """Load the default field segmentation model if none is loaded."""
    if _field_model is None:
        # Use the specified default model path
        default_model_path = "data/models/segmentation/20250826_1_segmentation_yolo11s-seg_field finder.v8i.yolov8/finetune_20250826_092226/weights/best.pt"
        
        # Try the specified model first
        models_base = Path(get_setting("models.base_path", "data/models"))
        full_path = models_base / "segmentation/20250826_1_segmentation_yolo11s-seg_field finder.v8i.yolov8/finetune_20250826_092226/weights/best.pt"
        
        if full_path.exists():
            print(f"[FIELD_SEG] Loading default field segmentation model: {full_path}")
            set_field_model(str(full_path))
            return
        
        # Fallback to other segmentation models
        fallback_paths = [
            models_base / "segmentation/field_finder_yolo11m-seg/segmentation_finetune/weights/best.pt",
            models_base / "segmentation/field_finder_yolo11m-seg/finetune/weights/best.pt",
            models_base / "segmentation/field_finder_yolo11n-seg/segmentation_finetune/weights/best.pt", 
            models_base / "pretrained/yolo11m-seg.pt",
            models_base / "pretrained/yolo11n-seg.pt"
        ]
        
        for fallback_path in fallback_paths:
            if fallback_path.exists():
                print(f"[FIELD_SEG] Loading fallback segmentation model: {fallback_path}")
                set_field_model(str(fallback_path))
                return
        
        print("[FIELD_SEG] No field segmentation models found, will use mock results")


# Initialize with default model when module is imported
_load_default_model()


def visualize_segmentation(frame: np.ndarray, results: List[Any], alpha: float = 0.5) -> np.ndarray:
    """Visualize field segmentation results on a frame.
    
    Args:
        frame: Original frame to overlay segmentation on
        results: Segmentation results from run_field_segmentation()
        alpha: Transparency for overlay (0.0 = transparent, 1.0 = opaque)
        
    Returns:
        Frame with segmentation overlay applied
    """
    if not results:
        return frame
    
    viz_frame = frame.copy()
    
    # Define colors for different field regions  
    colors = [
        (0, 255, 0),    # Green for field
        (255, 0, 0),    # Red for end zone
        (0, 0, 255),    # Blue for sideline
        (255, 255, 0),  # Yellow for goal line
    ]
    
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            # Get masks data
            masks = result.masks.data.cpu().numpy() if hasattr(result.masks.data, 'cpu') else result.masks.data
            
            for idx, mask in enumerate(masks):
                # Ensure mask is right shape
                if mask.shape != frame.shape[:2]:
                    mask_resized = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]))
                else:
                    mask_resized = mask
                
                # Create colored overlay
                color = colors[idx % len(colors)]
                overlay = np.zeros_like(frame)
                overlay[mask_resized > 0.5] = color
                
                # Blend with original frame
                viz_frame = cv2.addWeighted(viz_frame, 1.0, overlay, alpha, 0)
                
                # Draw contours for better visibility
                contours, _ = cv2.findContours(
                    (mask_resized > 0.5).astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(viz_frame, contours, -1, color, 2)
    
    return viz_frame
