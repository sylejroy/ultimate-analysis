"""Field segmentation module - YOLO-based field boundary detection.

This module handles segmenting the Ultimate Frisbee field boundaries and 
identifying important field features like end zones and sidelines.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[FIELD_SEG] Warning: ultralytics not available, using mock results")

from ..config.settings import get_setting
from ..constants import FIELD_DIMENSIONS, FALLBACK_DEFAULTS


# Global field segmentation state
_field_model = None
_current_model_path = None


def run_field_segmentation(frame: np.ndarray) -> List[Any]:
    """Run field segmentation on a video frame.
    
    Args:
        frame: Input video frame as numpy array (H, W, C) in BGR format
        
    Returns:
        List of segmentation results with masks and field boundaries
        Each result contains:
        - masks: Segmentation masks for different field regions
        - boxes: Bounding boxes for field elements
        - classes: Class IDs for different field regions
        
    Example:
        results = run_field_segmentation(frame)
        for result in results:
            if hasattr(result, 'masks') and result.masks:
                # Process field masks
                pass
    """
    print(f"[FIELD_SEG] Processing frame with shape {frame.shape}")
    
    if _field_model is None:
        _load_default_model()
    
    try:
        if ULTRALYTICS_AVAILABLE and _field_model is not None:
            # Run actual YOLO segmentation
            results = _field_model(frame, verbose=False)
            print(f"[FIELD_SEG] YOLO segmentation complete: {len(results)} results")
            return results
        else:
            # Fall back to mock results for testing
            results = _create_mock_results(frame)
            print(f"[FIELD_SEG] Using mock segmentation: {len(results)} results")
            return results
            
    except Exception as e:
        print(f"[FIELD_SEG] Error during segmentation: {e}")
        # Return mock results as fallback
        return _create_mock_results(frame)


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
    global _field_model, _current_model_path
    
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


def get_field_info() -> Dict[str, Any]:
    """Get information about field segmentation capabilities.
    
    Returns:
        Dictionary with field segmentation information:
        - model_path: Current model file path
        - field_dimensions: Ultimate Frisbee field dimensions
        - classes: Field region class names
        - loaded: Whether model is loaded
    """
    return {
        'model_path': _current_model_path,
        'field_dimensions': FIELD_DIMENSIONS,
        'classes': ['field', 'end_zone', 'sideline', 'goal_line'],
        'loaded': _field_model is not None
    }


def detect_field_boundaries(mask: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
    """Extract field boundary lines from segmentation mask.
    
    Args:
        mask: Binary field segmentation mask
        
    Returns:
        Dictionary with field boundary coordinates:
        - sidelines: List of (x, y) points for left and right sidelines  
        - goal_lines: List of (x, y) points for goal lines
        - end_zone_lines: List of (x, y) points for end zone boundaries
    """
    boundaries = {
        'sidelines': [],
        'goal_lines': [],
        'end_zone_lines': []
    }
    
    # TODO: Implement actual boundary detection from mask
    # This would involve edge detection and line fitting
    
    print(f"[FIELD_SEG] Extracted field boundaries")
    return boundaries


def calculate_field_homography(boundaries: Dict[str, List[Tuple[int, int]]]) -> Optional[np.ndarray]:
    """Calculate homography matrix to transform field coordinates.
    
    Args:
        boundaries: Field boundary coordinates from detect_field_boundaries()
        
    Returns:
        3x3 homography matrix for perspective transformation, or None if failed
    """
    # TODO: Implement homography calculation
    # This would map field coordinates to real-world Ultimate field dimensions
    
    print("[FIELD_SEG] Calculating field homography")
    return None


def transform_to_field_coordinates(points: List[Tuple[int, int]], 
                                 homography: np.ndarray) -> List[Tuple[float, float]]:
    """Transform pixel coordinates to field coordinates using homography.
    
    Args:
        points: List of (x, y) pixel coordinates
        homography: 3x3 homography matrix
        
    Returns:
        List of (x, y) field coordinates in meters
    """
    # TODO: Implement coordinate transformation
    field_coords = []
    
    for x, y in points:
        # Apply homography transformation
        # field_x, field_y = apply_homography(x, y, homography)
        field_coords.append((0.0, 0.0))  # Placeholder
    
    return field_coords


def _load_default_model() -> None:
    """Load the default field segmentation model if none is loaded."""
    if _field_model is None:
        # Use the specified default model path
        default_model_path = "data/models/segmentation/field_finder_yolo11m-seg/segmentation_finetune/weights/best.pt"
        
        # Try the specified model first
        models_base = Path(get_setting("models.base_path", "data/models"))
        full_path = models_base / "segmentation/field_finder_yolo11m-seg/segmentation_finetune/weights/best.pt"
        
        if full_path.exists():
            print(f"[FIELD_SEG] Loading default field segmentation model: {full_path}")
            set_field_model(str(full_path))
            return
        
        # Fallback to other segmentation models
        fallback_paths = [
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


def get_field_mask(results: List[Any], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Extract field mask from segmentation results.
    
    Args:
        results: Segmentation results from run_field_segmentation()
        frame_shape: Shape of the frame (height, width)
        
    Returns:
        Binary mask of the field region, or None if no field detected
    """
    if not results:
        return None
    
    field_mask = np.zeros(frame_shape, dtype=np.uint8)
    
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy() if hasattr(result.masks.data, 'cpu') else result.masks.data
            
            for mask in masks:
                # Resize mask if necessary
                if mask.shape != frame_shape:
                    mask_resized = cv2.resize(mask.astype(np.float32), (frame_shape[1], frame_shape[0]))
                else:
                    mask_resized = mask
                
                # Add to field mask
                field_mask = np.logical_or(field_mask, mask_resized > 0.5).astype(np.uint8)
    
    return field_mask
