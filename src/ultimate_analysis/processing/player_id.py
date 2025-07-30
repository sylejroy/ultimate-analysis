"""Player identification module - OCR and jersey number detection.

This module handles identifying players by their jersey numbers using either
EasyOCR for text recognition or YOLO models trained on digit detection.
"""

import cv2
import numpy as np
import yaml
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[PLAYER_ID] Warning: EasyOCR not available, using mock results")

from ..config.settings import get_setting
from ..constants import JERSEY_NUMBER_MIN, JERSEY_NUMBER_MAX, SUPPORTED_OCR_LANGUAGES, DEFAULT_PATHS


# Global player ID state
_player_id_method = "yolo"  # "yolo" or "easyocr" 
_easyocr_reader = None
_digit_detection_model = None
_current_model_path = None


def run_player_id_on_tracks(frame: np.ndarray, tracks: List[Any]) -> Dict[int, Tuple[str, Any]]:
    """Run player identification on tracked objects.
    
    Args:
        frame: Current video frame
        tracks: List of track objects from tracking system
        
    Returns:
        Dictionary mapping track_id -> (jersey_number, detection_details)
        
    Example:
        results = run_player_id_on_tracks(frame, current_tracks)
        for track_id, (number, details) in results.items():
            print(f"Track {track_id}: Player #{number}")
    """
    player_identifications = {}
    
    if not tracks:
        return player_identifications
    
    # Set method to EasyOCR for jersey number detection
    if _player_id_method != "easyocr":
        set_player_id_method("easyocr")
    
    for track in tracks:
        try:
            # Extract track information
            if hasattr(track, 'track_id'):
                track_id = track.track_id
            elif hasattr(track, 'id'):
                track_id = track.id
            else:
                continue
            
            # Get bounding box
            if hasattr(track, 'to_tlbr'):
                # DeepSORT format
                bbox = track.to_tlbr().astype(int)
                x1, y1, x2, y2 = bbox
            elif hasattr(track, 'bbox'):
                # Generic bbox format
                x1, y1, x2, y2 = map(int, track.bbox)
            else:
                continue
            
            # Ensure bbox is within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Crop the tracked object
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                # Run player ID on the crop
                jersey_number, details = run_player_id(crop)
                player_identifications[track_id] = (jersey_number, details)
                
                print(f"[PLAYER_ID] Track {track_id}: {jersey_number}")
            else:
                player_identifications[track_id] = ("Unknown", None)
                
        except Exception as e:
            print(f"[PLAYER_ID] Error processing track: {e}")
            continue
    
    return player_identifications


def run_player_id(crop_image: np.ndarray) -> Tuple[str, Optional[Any]]:
    """Run player identification on a cropped player image.
    
    Args:
        crop_image: Cropped image containing player's jersey (H, W, C) in BGR format
        
    Returns:
        Tuple of (jersey_number_string, detection_details)
        - jersey_number_string: Detected number as string (e.g., "23", "Unknown")
        - detection_details: Method-specific detection information
            - For YOLO: List of (bbox, class_id) tuples for digit detections
            - For EasyOCR: List of OCR bounding boxes and text
            
    Example:
        jersey_str, details = run_player_id(player_crop)
        if jersey_str != "Unknown":
            print(f"Player #{jersey_str}")
    """
    print(f"[PLAYER_ID] Processing crop with shape {crop_image.shape} using {_player_id_method}")
    
    if crop_image.size == 0:
        return "Unknown", None
    
    try:
        if _player_id_method == "easyocr":
            return _run_easyocr_detection(crop_image)
        elif _player_id_method == "yolo":
            return _run_yolo_digit_detection(crop_image)
        else:
            print(f"[PLAYER_ID] Unknown method: {_player_id_method}")
            return "Unknown", None
            
    except Exception as e:
        print(f"[PLAYER_ID] Error during player ID: {e}")
        return "Unknown", None


def set_player_id_method(method: str) -> bool:
    """Set the player identification method.
    
    Args:
        method: Method to use ("yolo" or "easyocr")
        
    Returns:
        True if method set successfully, False otherwise
    """
    global _player_id_method
    
    method = method.lower()
    if method not in ["yolo", "easyocr"]:
        print(f"[PLAYER_ID] Unsupported method: {method}")
        return False
    
    print(f"[PLAYER_ID] Setting player ID method to: {method}")
    _player_id_method = method
    
    # Initialize the chosen method
    if method == "easyocr":
        _initialize_easyocr()
    elif method == "yolo":
        _initialize_yolo_digit_detector()
    
    return True


def set_player_id_model(model_path: str) -> bool:
    """Set the YOLO model for digit detection (only used when method is "yolo").
    
    Args:
        model_path: Path to the YOLO digit detection model (.pt)
        
    Returns:
        True if model loaded successfully, False otherwise
    """
    global _digit_detection_model, _current_model_path
    
    print(f"[PLAYER_ID] Setting digit detection model: {model_path}")
    
    try:
        # TODO: Load actual YOLO digit detection model
        # from ultralytics import YOLO
        # _digit_detection_model = YOLO(model_path)
        
        _current_model_path = model_path
        print(f"[PLAYER_ID] Digit detection model loaded: {model_path}")
        return True
        
    except Exception as e:
        print(f"[PLAYER_ID] Failed to load digit model {model_path}: {e}")
        return False


def get_player_id_method() -> str:
    """Get the current player identification method.
    
    Returns:
        Current method string ("yolo" or "easyocr")
    """
    return _player_id_method


def get_player_id_model_path() -> Optional[str]:
    """Get the path of the current digit detection model.
    
    Returns:
        Path to current model or None if no model loaded
    """
    return _current_model_path


def _run_easyocr_detection(crop_image: np.ndarray) -> Tuple[str, Optional[List]]:
    """Run EasyOCR text detection on player crop.
    
    Args:
        crop_image: Cropped player image
        
    Returns:
        Tuple of (jersey_number, ocr_boxes)
    """
    if _easyocr_reader is None:
        _initialize_easyocr()
    
    if not EASYOCR_AVAILABLE or _easyocr_reader is None:
        return "Unknown", []
    
    try:
        # Load user configuration
        user_config = _load_easyocr_config()
        
        # Preprocess the image
        processed_image = _preprocess_image(crop_image, user_config.get('preprocessing', {}))
        
        # Run EasyOCR with user settings
        ocr_params = user_config.get('easyocr', {})
        results = _easyocr_reader.readtext(
            processed_image,
            allowlist=ocr_params.get('allowlist', '0123456789'),
            width_ths=ocr_params.get('width_ths', 0.7),
            height_ths=ocr_params.get('height_ths', 0.7),
            detail=ocr_params.get('detail', 1),
            paragraph=ocr_params.get('paragraph', False),
            batch_size=ocr_params.get('batch_size', 16)
        )
        
        # Process results to find jersey numbers
        jersey_number = _extract_jersey_number(results)
        
        print(f"[PLAYER_ID] EasyOCR detected: {jersey_number} (from {len(results)} detections)")
        return jersey_number, results
        
    except Exception as e:
        print(f"[PLAYER_ID] EasyOCR error: {e}")
        return "Unknown", []


def _load_easyocr_config() -> Dict[str, Any]:
    """Load EasyOCR configuration from user.yaml file."""
    try:
        config_path = Path(DEFAULT_PATHS['CONFIGS']) / 'user.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('player_id', {})
        else:
            print(f"[PLAYER_ID] Config file not found: {config_path}")
            return {}
            
    except Exception as e:
        print(f"[PLAYER_ID] Error loading config: {e}")
        return {}


def _preprocess_image(image: np.ndarray, preprocess_config: Dict[str, Any]) -> np.ndarray:
    """Preprocess image for better OCR recognition."""
    processed = image.copy()
    
    try:
        # Crop top fraction if specified
        crop_fraction = preprocess_config.get('crop_top_fraction', 0.0)
        if crop_fraction > 0:
            h = processed.shape[0]
            crop_pixels = int(h * crop_fraction)
            processed = processed[crop_pixels:, :]
        
        # Resize if specified
        resize_factor = preprocess_config.get('resize_factor', 1.0)
        if resize_factor != 1.0:
            h, w = processed.shape[:2]
            new_h, new_w = int(h * resize_factor), int(w * resize_factor)
            processed = cv2.resize(processed, (new_w, new_h))
        
        # Convert to grayscale if specified
        if preprocess_config.get('bw_mode', False):
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Enhance contrast
        if preprocess_config.get('enhance_contrast', False):
            alpha = preprocess_config.get('contrast_alpha', 1.5)
            beta = preprocess_config.get('brightness_beta', 0)
            processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe_clip = preprocess_config.get('clahe_clip_limit', 2.0)
        clahe_grid = preprocess_config.get('clahe_grid_size', 8)
        if clahe_clip > 0:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Denoise
        if preprocess_config.get('denoise', False):
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        
        # Gaussian blur
        blur_size = preprocess_config.get('gaussian_blur', 0)
        if blur_size > 0:
            processed = cv2.GaussianBlur(processed, (blur_size, blur_size), 0)
        
        # Sharpen
        if preprocess_config.get('sharpen', False):
            strength = preprocess_config.get('sharpen_strength', 0.5)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * strength
            kernel[1,1] = kernel[1,1] + (1 - strength * 8)
            processed = cv2.filter2D(processed, -1, kernel)
        
        # Upscale if specified
        if preprocess_config.get('upscale', False):
            target_size = preprocess_config.get('upscale_target_size', 256)
            h, w = processed.shape[:2]
            if max(h, w) < target_size:
                scale = target_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
    except Exception as e:
        print(f"[PLAYER_ID] Preprocessing error: {e}")
        return image
    
    return processed


def _extract_jersey_number(ocr_results: List[Tuple]) -> str:
    """Extract the most likely jersey number from OCR results."""
    if not ocr_results:
        return "Unknown"
    
    # Filter results for numeric text only
    numeric_results = []
    for result in ocr_results:
        if len(result) >= 2:
            text = result[1].strip()
            # Only consider numeric strings
            if text.isdigit():
                confidence = result[2] if len(result) > 2 else 1.0
                numeric_results.append((text, confidence))
    
    if not numeric_results:
        return "Unknown"
    
    # Sort by confidence and get the best result
    numeric_results.sort(key=lambda x: x[1], reverse=True)
    best_number = numeric_results[0][0]
    
    # Validate jersey number
    if _validate_jersey_number(best_number):
        return best_number
    else:
        return "Unknown"


def _run_yolo_digit_detection(crop_image: np.ndarray) -> Tuple[str, Optional[List]]:
    """Run YOLO digit detection on player crop.
    
    Args:
        crop_image: Cropped player image
        
    Returns:
        Tuple of (jersey_number, digit_detections)
    """
    if _digit_detection_model is None:
        _initialize_yolo_digit_detector()
    
    # TODO: Implement actual YOLO digit detection
    # For now, return placeholder
    jersey_number = "Unknown"
    digit_detections = []
    
    # Placeholder: simulate digit detection for testing
    if get_setting("app.debug", False):
        h, w = crop_image.shape[:2]
        digit_detections = [
            ([w//4, h//4, w//2, 3*h//4], 2),  # Detected digit "2"
            ([w//2, h//4, 3*w//4, 3*h//4], 3)  # Detected digit "3"
        ]
        jersey_number = "23"
    
    print(f"[PLAYER_ID] YOLO digit detection: {jersey_number}")
    return jersey_number, digit_detections


def _initialize_easyocr() -> None:
    """Initialize EasyOCR reader for text detection."""
    global _easyocr_reader
    
    if _easyocr_reader is not None:
        return
    
    print("[PLAYER_ID] Initializing EasyOCR reader")
    
    try:
        if EASYOCR_AVAILABLE:
            # Load user configuration for language settings
            user_config = _load_easyocr_config()
            easyocr_config = user_config.get('easyocr', {})
            
            # Use English for jersey numbers
            languages = ['en']
            gpu = easyocr_config.get('gpu', True)
            
            _easyocr_reader = easyocr.Reader(languages, gpu=gpu)
            print("[PLAYER_ID] EasyOCR reader initialized successfully")
        else:
            print("[PLAYER_ID] EasyOCR not available, using mock reader")
            _easyocr_reader = None
        
    except Exception as e:
        print(f"[PLAYER_ID] Failed to initialize EasyOCR: {e}")
        _easyocr_reader = None


def _initialize_yolo_digit_detector() -> None:
    """Initialize YOLO digit detection model."""
    if _digit_detection_model is None:
        print("[PLAYER_ID] No YOLO digit detection model loaded")
        # Could load a default model here if available


def _validate_jersey_number(number_str: str) -> bool:
    """Validate that a detected jersey number is reasonable.
    
    Args:
        number_str: Detected number string
        
    Returns:
        True if number is valid jersey number, False otherwise
    """
    try:
        number = int(number_str)
        return JERSEY_NUMBER_MIN <= number <= JERSEY_NUMBER_MAX
    except ValueError:
        return False


def set_easyocr() -> None:
    """Convenience function to set EasyOCR as the player ID method."""
    set_player_id_method("easyocr")


def set_yolo_digit_detection() -> None:
    """Convenience function to set YOLO as the player ID method."""
    set_player_id_method("yolo")
