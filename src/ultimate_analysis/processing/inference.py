"""Inference processing module - YOLO object detection.

This module handles running YOLO models for object detection on video frames.
Detects players, discs, and other relevant objects in Ultimate Frisbee games.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..config.settings import get_setting
from ..constants import YOLO_CLASSES, FALLBACK_DEFAULTS

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[INFERENCE] Warning: ultralytics not available, inference will be disabled")
    YOLO_AVAILABLE = False


# Global model cache
_detection_model = None
_current_model_path = None


def run_inference(frame: np.ndarray, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run YOLO inference on a video frame.
    
    Args:
        frame: Input video frame as numpy array (H, W, C) in BGR format
        model_name: Optional model name to use for inference
        
    Returns:
        List of detection dictionaries with keys:
        - bbox: [x1, y1, x2, y2] bounding box coordinates
        - confidence: Detection confidence score
        - class_id: Integer class ID (0=person, 1=player, 2=disc)
        - class_name: String class name
        
    Example:
        detections = run_inference(frame)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
    """
    global _detection_model, _current_model_path
    
    print(f"[INFERENCE] Processing frame with shape {frame.shape}")
    
    if not YOLO_AVAILABLE:
        print("[INFERENCE] YOLO not available, returning empty detections")
        return []
    
    # Load model if needed
    if model_name and model_name != _current_model_path:
        if not set_detection_model(model_name):
            print(f"[INFERENCE] Failed to load model {model_name}, using current model")
    
    # Ensure we have a model
    if _detection_model is None:
        _load_default_model()
    
    if _detection_model is None:
        print("[INFERENCE] No detection model available")
        # Return debug detections if enabled
        if get_setting("app.debug", False):
            h, w = frame.shape[:2]
            return [{
                'bbox': [w//4, h//4, 3*w//4, 3*h//4],
                'confidence': 0.85,
                'class_id': YOLO_CLASSES['PLAYER'],
                'class_name': 'player'
            }]
        return []
    
    detections: List[Dict[str, Any]] = []
    
    try:
        # Get inference parameters from config
        confidence_threshold = get_setting("models.detection.confidence_threshold", 0.5)
        nms_threshold = get_setting("models.detection.nms_threshold", 0.45)
        
        # Run YOLO inference
        results = _detection_model.predict(
            frame,
            conf=confidence_threshold,
            iou=nms_threshold,
            verbose=False,
            save=False,
            show=False
        )
        
        # Process results
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = float(confidences[i])
                    cls = int(classes[i])
                    
                    # Map class ID to class name
                    class_name = "unknown"
                    for name, class_id in YOLO_CLASSES.items():
                        if class_id == cls:
                            class_name = name.lower()
                            break
                    
                    # Skip detections below confidence threshold
                    if conf < confidence_threshold:
                        continue
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': class_name
                    })
        
    except Exception as e:
        print(f"[INFERENCE] Error during YOLO inference: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"[INFERENCE] Found {len(detections)} detections")
    return detections


def set_detection_model(model_path: str) -> bool:
    """Set the detection model to use for inference.
    
    Args:
        model_path: Path to the YOLO model file (.pt) or model name
        
    Returns:
        True if model loaded successfully, False otherwise
        
    Example:
        success = set_detection_model("data/models/detection/best.pt")
        success = set_detection_model("yolo11l.pt")
    """
    global _detection_model, _current_model_path
    
    if not YOLO_AVAILABLE:
        print("[INFERENCE] YOLO not available, cannot load model")
        return False
    
    print(f"[INFERENCE] Setting detection model: {model_path}")
    
    # Handle different model path formats
    model_file_path = None
    
    # If it's an absolute path or contains path separators, use it directly
    if Path(model_path).is_absolute() or "/" in model_path or "\\" in model_path:
        if Path(model_path).exists():
            model_file_path = model_path
        else:
            print(f"[INFERENCE] Absolute path does not exist: {model_path}")
            return False
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
        return False
    
    try:
        print(f"[INFERENCE] Loading YOLO model from: {model_file_path}")
        _detection_model = YOLO(model_file_path)
        _current_model_path = model_path
        
        print(f"[INFERENCE] Model loaded successfully: {model_path}")
        print(f"[INFERENCE] Model classes: {list(_detection_model.names.values()) if hasattr(_detection_model, 'names') else 'Unknown'}")
        
        return True
        
    except Exception as e:
        print(f"[INFERENCE] Failed to load model {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_current_model_path() -> Optional[str]:
    """Get the path of the currently loaded detection model.
    
    Returns:
        Path to current model or None if no model loaded
    """
    return _current_model_path


def get_model_info() -> Dict[str, Any]:
    """Get information about the current detection model.
    
    Returns:
        Dictionary with model information:
        - path: Model file path
        - classes: List of class names
        - input_size: Model input size
        - loaded: Whether model is loaded
    """
    return {
        'path': _current_model_path,
        'classes': list(YOLO_CLASSES.keys()),
        'input_size': [640, 640],  # Standard YOLO input size
        'loaded': _detection_model is not None
    }


def _load_default_model() -> None:
    """Load the default detection model if none is loaded."""
    if _detection_model is None and YOLO_AVAILABLE:
        default_model = get_setting(
            "models.detection.default_model", 
            FALLBACK_DEFAULTS['model_detection']
        )
        
        print(f"[INFERENCE] Loading default model: {default_model}")
        
        # Try the configured model path first
        if Path(default_model).exists():
            set_detection_model(default_model)
        else:
            # Try to find the finetuned model
            finetuned_path = Path("data/models/detection/object_detection_yolo11l/finetune3/weights/best.pt")
            if finetuned_path.exists():
                print(f"[INFERENCE] Using finetuned model: {finetuned_path}")
                set_detection_model(str(finetuned_path))
            else:
                # Fallback to pretrained models
                models_path = Path(get_setting("models.base_path", "data/models"))
                pretrained_path = models_path / "pretrained" / "yolo11l.pt"
                
                if pretrained_path.exists():
                    print(f"[INFERENCE] Fallback to pretrained model: {pretrained_path}")
                    set_detection_model(str(pretrained_path))
                else:
                    print(f"[INFERENCE] No models found, inference will be disabled")
                    print(f"[INFERENCE] Searched paths:")
                    print(f"  - {default_model}")
                    print(f"  - {finetuned_path}")
                    print(f"  - {pretrained_path}")


# Initialize with default model when module is imported
_load_default_model()
