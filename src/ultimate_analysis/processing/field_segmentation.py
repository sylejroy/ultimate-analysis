from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# Global model and path
field_model = None
field_model_path = None
_field_model_cache = {}  # Cache for loaded field models

def load_field_model(path):
    global field_model, field_model_path, _field_model_cache
    
    # Convert to absolute path if relative
    if isinstance(path, str):
        path = Path(path)
    
    # If path is relative, resolve it relative to the project root
    if not path.is_absolute():
        # Get project root (assuming this file is in src/ultimate_analysis/processing/)
        project_root = Path(__file__).parent.parent.parent.parent
        path = project_root / path
    
    # Convert back to string for consistency
    path_str = str(path)
    
    # Check if model is already cached
    if path_str in _field_model_cache:
        field_model = _field_model_cache[path_str]
        field_model_path = path_str
        return
    
    # Check if model file exists
    if not path.exists():
        raise FileNotFoundError(f"Field model weights not found: {path_str}")
    
    field_model_path = path_str
    field_model = YOLO(field_model_path)
    
    # Check GPU availability and set device
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    try:
        field_model.to(device)
    except Exception as e:
        field_model.to('cpu')
        device = 'cpu'
    
    # Store device info with model
    field_model._device = device
    
    # Cache the model (limit cache size to prevent memory issues)
    if len(_field_model_cache) >= 3:  # Keep max 3 models in cache
        # Remove oldest model from cache
        oldest_key = next(iter(_field_model_cache))
        del _field_model_cache[oldest_key]
    
    _field_model_cache[path_str] = field_model

# Field segmentation model is loaded on demand via set_field_model()
# No default model loading at startup - models should be configured via settings

def set_field_model(path):
    """Set and reload the field segmentation model at runtime."""
    load_field_model(path)

def run_field_segmentation(frame):
    """
    Run field segmentation on the given frame.
    
    Args:
        frame: Input frame (numpy array)
        
    Returns:
        Segmentation results
    """
    if field_model is None:
        raise RuntimeError("Field segmentation model is not loaded.")
    
    # Run inference
    results = field_model(frame, verbose=False)
    
    # Extract masks if available
    masks = []
    if hasattr(results[0], 'masks') and results[0].masks is not None:
        for mask in results[0].masks.data:
            masks.append(mask.cpu().numpy())
    
    return masks

def extract_field_contours(masks, frame_shape):
    """
    Extract field contours from segmentation masks.
    
    Args:
        masks: List of segmentation masks
        frame_shape: Shape of the input frame
        
    Returns:
        List of field contours
    """
    contours = []
    
    for mask in masks:
        # Resize mask to frame size
        mask_resized = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
        
        # Convert to binary mask
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
        
        # Find contours
        mask_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (keep only significant ones)
        min_area = frame_shape[0] * frame_shape[1] * 0.01  # At least 1% of frame
        for contour in mask_contours:
            if cv2.contourArea(contour) > min_area:
                contours.append(contour)
    
    return contours

def get_field_boundary_points(contours, frame_shape):
    """
    Extract field boundary points from contours.
    
    Args:
        contours: List of field contours
        frame_shape: Shape of the input frame
        
    Returns:
        List of boundary points
    """
    if not contours:
        return []
    
    # Get the largest contour (assumed to be the field)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to get key points
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Extract points
    boundary_points = []
    for point in approx:
        x, y = point[0]
        boundary_points.append((int(x), int(y)))
    
    return boundary_points