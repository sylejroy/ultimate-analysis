from ultralytics import YOLO
import cv2
import numpy as np

# Global model and path
model = None
weights_path = None
_model_cache = {}  # Cache for loaded models

def load_model(path):
    global model, weights_path, _model_cache
    
    # Check if model is already cached
    if path in _model_cache:
        print(f"[DEBUG] Using cached detection model: {path}")
        model = _model_cache[path]
        weights_path = path
        return
    
    print(f"[DEBUG] Loading detection model from: {path}")
    weights_path = path
    model = YOLO(weights_path)
    
    # Check GPU availability and set device
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"[DEBUG] CUDA available, using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print(f"[DEBUG] CUDA not available, using CPU")
    
    try:
        model.to(device)
        print(f"[DEBUG] Model moved to {device} successfully")
    except Exception as e:
        print(f"[DEBUG] Failed to move model to {device}, falling back to CPU: {e}")
        model.to('cpu')
        device = 'cpu'
    
    # Store device info with model
    model._device = device
    
    # Cache the model (limit cache size to prevent memory issues)
    if len(_model_cache) >= 3:  # Keep max 3 models in cache
        # Remove oldest model from cache
        oldest_key = next(iter(_model_cache))
        del _model_cache[oldest_key]
    
    _model_cache[path] = model
    print(f"[DEBUG] Model loaded and cached successfully on {device}")

# Load default model at startup
load_model("finetune/object_detection_yolo11l/finetune3/weights/best.pt")

def set_detection_model(path):
    """Set and reload the detection model at runtime."""
    print(f"[DEBUG] set_detection_model called with: {path}")
    load_model(path)

def run_inference(frame):
    """
    Runs YOLO inference on the given frame and returns detections in the format:
    [ ([x, y, w, h], conf, cls), ... ]
    Optimized version with reduced memory allocation.
    """
    if model is None:
        raise RuntimeError("YOLO model is not loaded.")
    
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
    device = getattr(model, '_device', 'cpu')
    use_half = device == 'cuda'  # Only use half precision on GPU
    
    print(f"[DEBUG] Running model.predict with device='{device}', half={use_half}, input_size={input_size}")
    
    try:
        results = model.predict(
            resized_frame, 
            verbose=False, 
            imgsz=input_size, 
            half=use_half, 
            device=device
        )[0]
        print(f"[DEBUG] Model.predict completed successfully on {device}")
    except Exception as e:
        print(f"[DEBUG] Inference failed on {device}, trying CPU fallback: {e}")
        # Fallback to CPU if GPU fails
        try:
            model.to('cpu')
            model._device = 'cpu'
            results = model.predict(
                resized_frame, 
                verbose=False, 
                imgsz=input_size, 
                half=False, 
                device='cpu'
            )[0]
            print(f"[DEBUG] Model.predict completed successfully on CPU fallback")
        except Exception as e2:
            print(f"[ERROR] Inference failed on both GPU and CPU: {e2}")
            raise e2
    
    detections = []
    if results.boxes is not None:
        # Vectorized processing for better performance
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        
        # Scale back to original frame size if needed
        if scale_factor != 1.0:
            boxes = boxes / scale_factor
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = float(confidences[i])
            cls = int(classes[i])
            
            # Convert to [x, y, w, h] format
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            detections.append(([x, y, w, h], conf, cls))
    
    return detections

def get_class_names():
    return model.names if model is not None and hasattr(model, 'names') else {}

def clear_model_cache():
    """Clear the model cache to free memory"""
    global _model_cache
    _model_cache.clear()