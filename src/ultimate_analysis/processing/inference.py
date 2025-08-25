"""Inference processing module - YOLO object detection.

This module handles running YOLO models for object detection on video frames.
Detects players, discs, and other relevant objects in Ultimate Frisbee games.
"""

import numpy as np
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
_model_imgsz = None  # Store the model's training image size


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
                print(f"[INFERENCE] Loaded training parameters from {args_yaml_path}")
                return args if args else {}
        else:
            print(f"[INFERENCE] No args.yaml found at {args_yaml_path}")
            
    except Exception as e:
        print(f"[INFERENCE] Error reading model training parameters: {e}")
    
    return {}


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
        
        # Determine image size to use - prefer model's training size
        imgsz = _model_imgsz if _model_imgsz else 640
        
        # Run YOLO inference
        results = _detection_model.predict(
            frame,
            conf=confidence_threshold,
            iou=nms_threshold,
            imgsz=imgsz,  # Use model's training image size
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
                    
                    # Get class name from the model itself (important for finetuned models)
                    class_name = "unknown"
                    if hasattr(_detection_model, 'names') and cls in _detection_model.names:
                        class_name = _detection_model.names[cls]
                    else:
                        # Fallback to our predefined mapping
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


def run_batch_inference(frames: List[np.ndarray], model_name: Optional[str] = None, 
                       use_parallel: bool = True) -> List[List[Dict[str, Any]]]:
    """Run YOLO inference on a batch of video frames with optional parallel processing.
    
    Args:
        frames: List of input video frames as numpy arrays (H, W, C) in BGR format
        model_name: Optional model name to use for inference
        use_parallel: Whether to use parallel processing for post-processing
        
    Returns:
        List of detection dictionaries for each frame, with same format as run_inference()
        
    Performance Benefits:
        - Batch YOLO prediction reduces model overhead
        - Parallel post-processing of results
        - Optimal for processing multiple frames or video sequences
    """
    if not frames:
        return []
    
    global _detection_model, _current_model_path
    
    print(f"[INFERENCE] Processing batch of {len(frames)} frames")
    
    if not YOLO_AVAILABLE:
        print("[INFERENCE] YOLO not available, returning empty detections")
        return [[] for _ in frames]
    
    # Load model if needed
    if model_name and model_name != _current_model_path:
        if not set_detection_model(model_name):
            print(f"[INFERENCE] Failed to load model {model_name}, using current model")
    
    # Ensure we have a model
    if _detection_model is None:
        _load_default_model()
    
    if _detection_model is None:
        print("[INFERENCE] No detection model available")
        return [[] for _ in frames]
    
    try:
        # Get inference parameters from config
        confidence_threshold = get_setting("models.detection.confidence_threshold", 0.5)
        nms_threshold = get_setting("models.detection.nms_threshold", 0.45)
        
        # Determine image size to use - prefer model's training size
        imgsz = _model_imgsz if _model_imgsz else 640
        
        # Run batch YOLO inference - this is where the real performance gain comes from
        print(f"[INFERENCE] Running batch YOLO prediction on {len(frames)} frames")
        batch_results = _detection_model.predict(
            frames,  # YOLO can process multiple frames at once
            conf=confidence_threshold,
            iou=nms_threshold,
            imgsz=imgsz,  # Use model's training image size
            verbose=False,
            save=False,
            show=False
        )
        
        # Process results - can be parallelized for large batches
        if use_parallel and len(frames) > 2:
            # Parallel post-processing for large batches
            def process_single_result(frame_data):
                frame_idx, result = frame_data
                return frame_idx, _process_yolo_result(result)
            
            # Determine optimal number of workers
            max_workers = min(len(frames), 4)  # Cap at 4 workers
            batch_detections = [[] for _ in frames]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                print(f"[INFERENCE] Using {max_workers} parallel workers for post-processing")
                
                # Submit all tasks
                indexed_results = [(i, result) for i, result in enumerate(batch_results)]
                future_to_index = {
                    executor.submit(process_single_result, frame_data): frame_data[0] 
                    for frame_data in indexed_results
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    try:
                        frame_idx, detections = future.result()
                        batch_detections[frame_idx] = detections
                    except Exception as e:
                        frame_idx = future_to_index[future]
                        print(f"[INFERENCE] Parallel post-processing failed for frame {frame_idx}: {e}")
                        batch_detections[frame_idx] = []
        else:
            # Sequential post-processing for small batches
            print(f"[INFERENCE] Using sequential post-processing for {len(frames)} frames")
            batch_detections = []
            for result in batch_results:
                detections = _process_yolo_result(result)
                batch_detections.append(detections)
        
        print(f"[INFERENCE] Batch processing complete: {[len(dets) for dets in batch_detections]} detections per frame")
        return batch_detections
        
    except Exception as e:
        print(f"[INFERENCE] Error during batch YOLO inference: {e}")
        import traceback
        traceback.print_exc()
        return [[] for _ in frames]


def _process_yolo_result(result) -> List[Dict[str, Any]]:
    """Process a single YOLO result into detection dictionaries.
    
    Args:
        result: Single YOLO prediction result
        
    Returns:
        List of detection dictionaries
    """
    detections = []
    
    try:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            # Get confidence threshold for filtering
            confidence_threshold = get_setting("models.detection.confidence_threshold", 0.5)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = float(confidences[i])
                cls = int(classes[i])
                
                # Skip detections below confidence threshold (double-check)
                if conf < confidence_threshold:
                    continue
                
                # Get class name from the model itself (important for finetuned models)
                class_name = "unknown"
                if hasattr(_detection_model, 'names') and cls in _detection_model.names:
                    class_name = _detection_model.names[cls]
                else:
                    # Fallback to our predefined mapping
                    for name, class_id in YOLO_CLASSES.items():
                        if class_id == cls:
                            class_name = name.lower()
                            break
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': class_name
                })
    
    except Exception as e:
        print(f"[INFERENCE] Error processing YOLO result: {e}")
    
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
    global _detection_model, _current_model_path, _model_imgsz
    
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
        
        # Load training parameters to get the image size used during training
        training_params = _get_model_training_params(model_file_path)
        _model_imgsz = training_params.get('imgsz', 640)
        print(f"[INFERENCE] Using model training image size: {_model_imgsz}")
        
        # Log model information
        print(f"[INFERENCE] Model loaded successfully: {model_path}")
        if hasattr(_detection_model, 'names'):
            model_classes = _detection_model.names
            print(f"[INFERENCE] Model classes: {dict(model_classes)}")
        else:
            print(f"[INFERENCE] Model classes: Unknown (no names attribute)")
        
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
        - classes: List of class names from the model
        - input_size: Model input size
        - loaded: Whether model is loaded
    """
    model_classes = []
    
    if _detection_model is not None and hasattr(_detection_model, 'names'):
        # Get actual class names from the loaded model
        model_classes = list(_detection_model.names.values())
    else:
        # Fallback to our predefined classes
        model_classes = list(YOLO_CLASSES.keys())
    
    return {
        'path': _current_model_path,
        'classes': model_classes,
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
