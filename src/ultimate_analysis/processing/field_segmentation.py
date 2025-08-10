"""Field segmentation module - YOLO-based field boundary detection.

This module handles segmenting the Ultimate Frisbee field boundaries and 
identifying important field features like end zones and sidelines.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def run_batch_field_segmentation(frames: List[np.ndarray], use_parallel: bool = True) -> List[List[Any]]:
    """Run field segmentation on a batch of video frames with optional parallel processing.
    
    Args:
        frames: List of input video frames as numpy arrays (H, W, C) in BGR format
        use_parallel: Whether to use parallel processing for post-processing
        
    Returns:
        List of segmentation results for each frame, with same format as run_field_segmentation()
        
    Performance Benefits:
        - Batch YOLO segmentation reduces model overhead
        - Parallel post-processing of segmentation masks
        - Optimal for processing video sequences
    """
    if not frames:
        return []
    
    print(f"[FIELD_SEG] Processing batch of {len(frames)} frames")
    
    if _field_model is None:
        print("[FIELD_SEG] No field segmentation model loaded")
        # Return mock results for all frames
        return [_create_mock_results(frame) for frame in frames]
    
    if not ULTRALYTICS_AVAILABLE:
        print("[FIELD_SEG] YOLO not available, returning mock results")
        return [_create_mock_results(frame) for frame in frames]
    
    try:
        # Get segmentation parameters from config
        confidence_threshold = get_setting("models.segmentation.confidence_threshold", 0.25)
        iou_threshold = get_setting("models.segmentation.iou_threshold", 0.7)
        
        # Run batch YOLO segmentation
        print(f"[FIELD_SEG] Running batch YOLO segmentation on {len(frames)} frames")
        batch_results = _field_model.predict(
            frames,  # YOLO can process multiple frames at once
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False,
            save=False,
            show=False
        )
        
        # Process results - can be parallelized for large batches
        if use_parallel and len(frames) > 2:
            # Parallel post-processing for large batches
            def process_single_segmentation(frame_data):
                frame_idx, result = frame_data
                return frame_idx, _process_segmentation_result(result)
            
            # Determine optimal number of workers
            max_workers = min(len(frames), 4)  # Cap at 4 workers
            batch_segmentations = [[] for _ in frames]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                print(f"[FIELD_SEG] Using {max_workers} parallel workers for segmentation post-processing")
                
                # Submit all tasks
                indexed_results = [(i, result) for i, result in enumerate(batch_results)]
                future_to_index = {
                    executor.submit(process_single_segmentation, frame_data): frame_data[0] 
                    for frame_data in indexed_results
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    try:
                        frame_idx, segmentations = future.result()
                        batch_segmentations[frame_idx] = segmentations
                    except Exception as e:
                        frame_idx = future_to_index[future]
                        print(f"[FIELD_SEG] Parallel segmentation processing failed for frame {frame_idx}: {e}")
                        batch_segmentations[frame_idx] = []
        else:
            # Sequential post-processing for small batches
            print(f"[FIELD_SEG] Using sequential segmentation processing for {len(frames)} frames")
            batch_segmentations = []
            for result in batch_results:
                segmentations = _process_segmentation_result(result)
                batch_segmentations.append(segmentations)
        
        print(f"[FIELD_SEG] Batch segmentation complete: {[len(segs) for segs in batch_segmentations]} results per frame")
        return batch_segmentations
        
    except Exception as e:
        print(f"[FIELD_SEG] Error during batch field segmentation: {e}")
        import traceback
        traceback.print_exc()
        # Return mock results for all frames on error
        return [_create_mock_results(frame) for frame in frames]


def _process_segmentation_result(result) -> List[Any]:
    """Process a single YOLO segmentation result.
    
    Args:
        result: Single YOLO segmentation prediction result
        
    Returns:
        List of segmentation results (usually just one result per frame)
    """
    try:
        # For field segmentation, we typically return the raw result
        # The calling code will extract masks, boxes, and classes as needed
        return [result] if result is not None else []
    except Exception as e:
        print(f"[FIELD_SEG] Error processing segmentation result: {e}")
        return []


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
        # Get the default model path from configuration
        default_model_path = get_setting("models.segmentation.default_model", "yolo11l-seg.pt")
        models_base = Path(get_setting("models.base_path", "data/models"))
        
        # If it's a relative path, make it absolute
        if not Path(default_model_path).is_absolute():
            full_path = models_base / default_model_path
        else:
            full_path = Path(default_model_path)
        
        if full_path.exists():
            print(f"[FIELD_SEG] Loading default field segmentation model: {full_path}")
            set_field_model(str(full_path))
            return
        
        # Fallback to other segmentation models
        fallback_paths = [
            models_base / "segmentation/field_finder_yolo11x-seg/segmentation_finetune4/weights/best.pt",
            models_base / "segmentation/field_finder_yolo11m-seg/segmentation_finetune/weights/best.pt",
            models_base / "segmentation/field_finder_yolo11m-seg/finetune/weights/best.pt",
            models_base / "segmentation/field_finder_yolo11n-seg/segmentation_finetune/weights/best.pt", 
            models_base / "pretrained/yolo11x-seg.pt",
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
