import os
import json
import logging
from typing import Any, Dict, Optional, Tuple, List, Union
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# --- Public getters for GUI access ---
def get_player_id_method() -> str:
    """Return the current player ID method ('easyocr' or 'yolo')."""
    return _player_id_method

def get_player_id_model_path() -> Optional[str]:
    """Return the current player ID model path, if any."""
    return _player_id_model_path

# ---- CONFIGURABLE DEFAULT ----
DEFAULT_PLAYER_ID_METHOD: str = "easyocr"  # Change to "yolo" to use YOLO by default
# ------------------------------

# Set up logging
logger = logging.getLogger("ultimate_analysis.player_id")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Global state (minimized, but kept for model/reader reuse)
_player_id_model: Optional[YOLO] = None
_player_id_model_path: Optional[str] = None
_player_id_method: str = DEFAULT_PLAYER_ID_METHOD
_easyocr_reader: Optional[easyocr.Reader] = None
_easyocr_params: Optional[Dict[str, Any]] = None

def load_easyocr_params(json_path: str = "easyocr_params.json") -> Dict[str, Any]:
    """
    Load EasyOCR and preprocessing parameters from a JSON file. Uses a cached version if already loaded.
    """
    global _easyocr_params
    if _easyocr_params is not None:
        return _easyocr_params
    if not os.path.exists(json_path):
        logger.warning("easyocr_params.json not found, using defaults.")
        _easyocr_params = {}
        return _easyocr_params
    with open(json_path, "r") as f:
        _easyocr_params = json.load(f)
    return _easyocr_params

def preprocess_for_easyocr(
    crop: np.ndarray,
    preproc_params: Dict[str, Any],
    preproc_enables: Dict[str, bool],
    colour_mode: bool = False,
    bw_mode: bool = False,
    upscale_to_size: bool = False,
    upscale_target_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a crop for EasyOCR using the given parameters.
    Returns (upscaled_rgb, proc_gray)
    """
    enabled = preproc_enables
    proc = crop.copy() if colour_mode else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if not colour_mode:
        # CLAHE
        if enabled.get("clahe_clip", False) or enabled.get("clahe_grid", False):
            proc = cv2.createCLAHE(
                clipLimit=preproc_params.get("clahe_clip", 3.0),
                tileGridSize=(preproc_params.get("clahe_grid", 8), preproc_params.get("clahe_grid", 8))
            ).apply(proc)
        if bw_mode:
            proc = cv2.bitwise_not(proc)
    # Sharpening
    if enabled.get("sharpen", False) and preproc_params.get("sharpen", 0) > 0:
        sharpen_strength = preproc_params.get("sharpen", 1.0)
        kernel = np.array([[0, -1, 0], [-1, 5 + sharpen_strength, -1], [0, -1, 0]])
        proc = cv2.filter2D(proc, -1, kernel)
    # Upscale
    if enabled.get("upscale", False):
        if upscale_to_size:
            h, w = proc.shape[:2]
            max_dim = max(h, w)
            target = upscale_target_size
            if max_dim != target:
                scale = target / max_dim
                proc = cv2.resize(proc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        elif preproc_params.get("upscale", 1.0) > 1.0:
            scale = preproc_params.get("upscale", 1.0)
            proc = cv2.resize(proc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Blur
    if enabled.get("blur_ksize", False) and preproc_params.get("blur_ksize", 1) > 1:
        ksize = preproc_params.get("blur_ksize", 1)
        if ksize % 2 == 0:
            ksize += 1
        proc = cv2.GaussianBlur(proc, (ksize, ksize), 0)
    # Convert to 3-channel RGB for easyocr if not already
    if len(proc.shape) == 2:
        upscaled_rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
    else:
        upscaled_rgb = proc
    return upscaled_rgb, proc

def set_player_id_method(method: str) -> None:
    """
    Set the player ID method ("easyocr" or "yolo").
    """
    global _player_id_method
    _player_id_method = method.lower()
    logger.info(f"Set player_id_method: {_player_id_method}")

def load_player_id_model(path: str) -> None:
    """
    Load the YOLO model for player ID detection.
    """
    global _player_id_model, _player_id_model_path
    logger.info(f"Loading player ID model from: {path}")
    _player_id_model_path = path
    _player_id_model = YOLO(_player_id_model_path)

def set_player_id_model(path: str) -> None:
    """
    Set the player ID method to YOLO and load the model.
    """
    set_player_id_method("yolo")
    load_player_id_model(path)


def set_easyocr() -> None:
    """
    Set the player ID method to EasyOCR and initialize the reader if needed.
    Optimized version with better error handling and CPU preference.
    """
    global _easyocr_reader
    set_player_id_method("easyocr")
    if _easyocr_reader is None:
        logger.info("Initializing EasyOCR reader")
        try:
            # Try GPU first, fallback to CPU for stability
            _easyocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            logger.info("EasyOCR reader initialized with GPU support")
        except Exception as e:
            logger.warning(f"GPU initialization failed, falling back to CPU: {e}")
            try:
                _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                logger.info("EasyOCR reader initialized with CPU")
            except Exception as e2:
                logger.error(f"Failed to initialize EasyOCR reader: {e2}")
                _easyocr_reader = None


def run_player_id(frame: np.ndarray) -> Tuple[str, Any]:
    """
    Run player ID detection on a frame using the selected method (YOLO or EasyOCR).
    Returns a tuple of (digit_str, details), where details are method-specific.
    """
    global _easyocr_reader
    if _player_id_method == "yolo":
        if _player_id_model is None:
            raise RuntimeError("Player ID YOLO model not loaded.")
        results = _player_id_model(frame, verbose=False, imgsz=640, conf=0.5)
        digits = []
        for result in results:
            d_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            d_classes = result.boxes.cls.cpu().numpy().astype(int)
            # Sort digits left-to-right
            digits = sorted(zip(d_boxes, d_classes), key=lambda x: x[0][0])
        digit_str = ''.join(str(d[1]) for d in digits)
        return digit_str, digits
    elif _player_id_method == "easyocr":
        if _easyocr_reader is None:
            set_easyocr()
        params = load_easyocr_params()
        preproc_params = params.get('preproc', {})
        preproc_enables = params.get('preproc_enables', {})
        colour_mode = params.get('colour_mode', False)
        bw_mode = params.get('bw_mode', False)
        upscale_to_size = params.get('upscale_to_size', False)
        upscale_target_size = params.get('upscale_target_size', 64)
        ocr_params = params.get('ocr', {})
        # Preprocess
        preproc_rgb, _ = preprocess_for_easyocr(
            frame, preproc_params, preproc_enables,
            colour_mode=colour_mode, bw_mode=bw_mode,
            upscale_to_size=upscale_to_size, upscale_target_size=upscale_target_size
        )
        # Run OCR
        ocr_results = _easyocr_reader.readtext(preproc_rgb, **ocr_params)
        ocr_boxes = [res[0] for res in ocr_results]
        digit_str = ''.join([res[1] for res in ocr_results])
        return digit_str, ocr_boxes
    else:
        raise RuntimeError("Unknown player ID method.")

# Initialize the default method at startup
if DEFAULT_PLAYER_ID_METHOD == "easyocr":
    try:
        set_easyocr()
    except Exception as e:
        logger.warning(f"Could not initialize EasyOCR at startup: {e}")