"""Player identification module - OCR and jersey number detection.

This module handles identifying players by their jersey numbers using either
EasyOCR for text recognition or YOLO models trained on digit detection.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from ..config.settings import get_setting
from ..constants import JERSEY_NUMBER_MIN, JERSEY_NUMBER_MAX, SUPPORTED_OCR_LANGUAGES


# Global player ID state
_player_id_method = "yolo"  # "yolo" or "easyocr" 
_easyocr_reader = None
_digit_detection_model = None
_current_model_path = None


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
    
    # TODO: Implement actual EasyOCR detection
    # For now, return placeholder
    jersey_number = "Unknown"
    ocr_boxes = []
    
    print(f"[PLAYER_ID] EasyOCR detected: {jersey_number}")
    return jersey_number, ocr_boxes


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
        # TODO: Initialize actual EasyOCR
        # import easyocr
        # languages = get_setting("models.player_id.ocr_languages", SUPPORTED_OCR_LANGUAGES[:1])
        # _easyocr_reader = easyocr.Reader(languages)
        
        _easyocr_reader = "easyocr_placeholder"
        print("[PLAYER_ID] EasyOCR reader initialized")
        
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
