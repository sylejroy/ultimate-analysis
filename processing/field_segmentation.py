from ultralytics import YOLO
import cv2
import numpy as np

# Global model and path
field_model = None
field_model_path = None
_field_model_cache = {}  # Cache for loaded field models

def load_field_model(path):
    global field_model, field_model_path, _field_model_cache
    
    # Check if model is already cached
    if path in _field_model_cache:
        print(f"[DEBUG] Using cached field segmentation model: {path}")
        field_model = _field_model_cache[path]
        field_model_path = path
        return
    
    print(f"[DEBUG] Loading field segmentation model from: {path}")
    field_model_path = path
    field_model = YOLO(field_model_path)
    
    # Check GPU availability and set device
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"[DEBUG] CUDA available for field model, using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print(f"[DEBUG] CUDA not available for field model, using CPU")
    
    try:
        field_model.to(device)
        print(f"[DEBUG] Field model moved to {device} successfully")
    except Exception as e:
        print(f"[DEBUG] Failed to move field model to {device}, falling back to CPU: {e}")
        field_model.to('cpu')
        device = 'cpu'
    
    # Store device info with model
    field_model._device = device
    
    # Cache the model (limit cache size to prevent memory issues)
    if len(_field_model_cache) >= 2:  # Keep max 2 field models in cache
        # Remove oldest model from cache
        oldest_key = next(iter(_field_model_cache))
        del _field_model_cache[oldest_key]
    
    _field_model_cache[path] = field_model
    print(f"[DEBUG] Field model loaded and cached successfully on {device}")

# Load a default field segmentation model at startup
load_field_model("finetune/field_finder_yolo11x-seg/segmentation_finetune4/weights/best.pt")

def set_field_model(path):
    """Set and reload the field segmentation model at runtime."""
    print(f"[DEBUG] set_field_model called with: {path}")
    load_field_model(path)

def run_field_segmentation(frame):
    """
    Runs YOLO segmentation on the given frame and returns the results.
    Optimized version with reduced memory allocation.
    """
    if field_model is None:
        raise RuntimeError("Field segmentation model is not loaded.")
    
    # Optimize: Use smaller input size if frame is very large
    original_shape = frame.shape[:2]
    input_size = 960
    
    # Resize frame if too large to speed up inference
    if max(original_shape) > input_size * 1.5:
        scale_factor = input_size / max(original_shape)
        new_width = int(frame.shape[1] * scale_factor)
        new_height = int(frame.shape[0] * scale_factor)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    else:
        resized_frame = frame
        scale_factor = 1.0
    
    # Run inference with proper device detection
    device = getattr(field_model, '_device', 'cpu')
    use_half = device == 'cuda'  # Only use half precision on GPU
    
    print(f"[DEBUG] Running field model.predict with device='{device}', half={use_half}, input_size={input_size}")
    
    try:
        results = field_model.predict(
            resized_frame, 
            verbose=False, 
            imgsz=input_size, 
            half=use_half, 
            device=device
        )
        print(f"[DEBUG] Field model.predict completed successfully on {device}")
    except Exception as e:
        print(f"[DEBUG] Field inference failed on {device}, trying CPU fallback: {e}")
        # Fallback to CPU if GPU fails
        try:
            field_model.to('cpu')
            field_model._device = 'cpu'
            results = field_model.predict(
                resized_frame, 
                verbose=False, 
                imgsz=input_size, 
                half=False, 
                device='cpu'
            )
            print(f"[DEBUG] Field model.predict completed successfully on CPU fallback")
        except Exception as e2:
            print(f"[ERROR] Field inference failed on both GPU and CPU: {e2}")
            raise e2
    
    # Scale masks back to original size if needed
    if scale_factor != 1.0 and results and len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        if masks.shape[0] > 0:
            # Resize masks back to original frame size
            original_h, original_w = original_shape
            resized_masks = []
            for mask in masks:
                resized_mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                resized_masks.append(resized_mask)
            results[0].masks.data = np.array(resized_masks)
    
    return results  # Return the results object

def clear_field_model_cache():
    """Clear the field model cache to free memory"""
    global _field_model_cache
    _field_model_cache.clear()