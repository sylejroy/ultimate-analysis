"""Player identification module - OCR and jersey number detection.

This module handles identifying players by their jersey numbers using either
EasyOCR for text recognition or YOLO models trained on digit detection.
Includes probabilistic tracking for reliable jersey number identification.
"""

import cv2
import numpy as np
import yaml
import time
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
from .jersey_tracker import add_jersey_measurement, get_jersey_probabilities, get_best_jersey_number, get_jersey_tracker


# Global player ID state - only EasyOCR is supported
_easyocr_reader = None


def run_player_id_on_tracks(frame: np.ndarray, tracks: List[Any]) -> Tuple[Dict[int, Tuple[str, Any]], Dict[str, float]]:
    """Run player identification on tracked objects using EasyOCR with probabilistic tracking.
    
    Args:
        frame: Current video frame
        tracks: List of track objects from tracking system
        
    Returns:
        Tuple of (player_identifications, timing_info)
        player_identifications: Dictionary mapping track_id -> (jersey_number, detection_details)
        timing_info: Dictionary with 'preprocessing_ms' and 'ocr_ms' totals
        
    Detection details now include both single-frame and historical tracking results:
        - 'single_frame': Single-frame EasyOCR result
        - 'tracking_history': Top 3 probabilities from historical tracking
        - 'best_tracked': Most probable jersey number from tracking
        
    Example:
        results, timing = run_player_id_on_tracks(frame, current_tracks)
        for track_id, (number, details) in results.items():
            print(f"Track {track_id}: Player #{number}")
            if 'tracking_history' in details:
                for jersey, prob, count in details['tracking_history']:
                    print(f"  {jersey}: {prob:.1%} ({count} measurements)")
    """
    player_identifications = {}
    total_timing = {'preprocessing_ms': 0.0, 'ocr_ms': 0.0}

    if not tracks:
        return player_identifications, total_timing

    # Initialize EasyOCR if needed
    if _easyocr_reader is None:
        _initialize_easyocr()

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

            # Skip disc tracks - only process players for jersey number detection
            if hasattr(track, 'class_id') and track.class_id == 0:  # 0 = disc
                continue
            elif hasattr(track, 'class_name') and track.class_name.lower() == 'disc':
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
                # Run EasyOCR player ID on the crop
                jersey_number, details, timing = _run_easyocr_detection(crop)
                
                # Add single-frame result to tracking history if valid
                if jersey_number and jersey_number != "Unknown" and details:
                    # Calculate bbox center position for spatial weighting
                    crop_width = x2 - x1
                    
                    # Try to get OCR detection position, otherwise use bbox center
                    bbox_center_x = 0.5  # Default to center
                    if details and 'ocr_results' in details and details['ocr_results']:
                        # Calculate average x position of detected text
                        total_x = 0
                        count = 0
                        for ocr_result in details['ocr_results']:
                            if len(ocr_result) >= 2:  # [bbox, text, confidence]
                                ocr_bbox = ocr_result[0]
                                if isinstance(ocr_bbox, (list, tuple)) and len(ocr_bbox) >= 4:
                                    # Calculate center x of OCR detection
                                    if isinstance(ocr_bbox[0], (list, tuple)):
                                        # Polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                        xs = [point[0] for point in ocr_bbox]
                                        center_x = sum(xs) / len(xs)
                                    else:
                                        # Box format: [x1, y1, x2, y2]
                                        center_x = (ocr_bbox[0] + ocr_bbox[2]) / 2
                                    
                                    # Normalize to 0-1 within crop
                                    bbox_center_x = center_x / crop_width
                                    total_x += bbox_center_x
                                    count += 1
                        
                        if count > 0:
                            bbox_center_x = total_x / count
                            # Clamp to [0, 1]
                            bbox_center_x = max(0.0, min(1.0, bbox_center_x))
                    
                    # Add measurement to tracker
                    confidence = details.get('confidence', 0.0)
                    ocr_results = details.get('ocr_results', [])
                    add_jersey_measurement(track_id, jersey_number, confidence, bbox_center_x, ocr_results)

                # Get tracking history and best tracked result
                tracking_history = get_jersey_probabilities(track_id, top_k=3)
                best_tracked_number, best_tracked_prob = get_best_jersey_number(track_id)

                # Prepare enhanced details
                enhanced_details = details.copy() if details else {}
                enhanced_details.update({
                    'single_frame': {
                        'jersey_number': jersey_number,
                        'confidence': details.get('confidence', 0.0) if details else 0.0
                    },
                    'tracking_history': tracking_history,
                    'best_tracked': {
                        'jersey_number': best_tracked_number,
                        'probability': best_tracked_prob
                    },
                    'jersey_tracker': get_jersey_tracker()  # Add tracker instance for top probabilities
                })

                # Decide which result to return as primary
                if best_tracked_number and best_tracked_prob > 0.5:
                    # Use tracked result if high confidence
                    primary_result = best_tracked_number
                else:
                    # Fall back to single-frame detection
                    primary_result = jersey_number

                player_identifications[track_id] = (primary_result, enhanced_details)

                # Add timing to totals
                total_timing['preprocessing_ms'] += timing['preprocessing_ms']
                total_timing['ocr_ms'] += timing['ocr_ms']
                
                print(f"[PLAYER_ID] Track {track_id}: Single-frame='{jersey_number}', Tracked='{best_tracked_number}' ({best_tracked_prob:.2%}), Primary='{primary_result}'")
            else:
                player_identifications[track_id] = ("Unknown", None)
                
        except Exception as e:
            print(f"[PLAYER_ID] Error processing track: {e}")
            continue

    return player_identifications, total_timing
def _run_easyocr_detection(crop_image: np.ndarray) -> Tuple[str, Optional[List], Dict[str, float]]:
    """Run EasyOCR text detection on player crop using the same algorithm as tuning tab.
    
    Args:
        crop_image: Cropped player image
        
    Returns:
        Tuple of (jersey_number, ocr_results, timing_info)
        timing_info contains 'preprocessing_ms' and 'ocr_ms'
    """
    timing_info = {'preprocessing_ms': 0.0, 'ocr_ms': 0.0}
    
    if _easyocr_reader is None:
        _initialize_easyocr()

    if not EASYOCR_AVAILABLE or _easyocr_reader is None:
        print("[PLAYER_ID] EasyOCR not available, returning Unknown")
        return "Unknown", [], timing_info

    try:
        # Start preprocessing timer
        prep_start_time = time.time()
        
        # Load user configuration (same as tuning tab)
        user_config = _load_easyocr_config()
        
        # Check minimum crop size before processing
        crop_config = user_config.get('preprocessing', {})
        min_crop_width = crop_config.get('min_crop_width', 20)  # Default 20 pixels
        min_crop_height = crop_config.get('min_crop_height', 30)  # Default 30 pixels
        
        crop_height, crop_width = crop_image.shape[:2]
        if crop_width < min_crop_width or crop_height < min_crop_height:
            timing_info['preprocessing_ms'] = (time.time() - prep_start_time) * 1000
            timing_info['ocr_ms'] = 0.0
            print(f"[PLAYER_ID] Crop too small ({crop_width}x{crop_height}), skipping OCR (min: {min_crop_width}x{min_crop_height})")
            return "Unknown", [], timing_info
        
        # Apply top crop fraction like tuning tab
        crop_config = user_config.get('preprocessing', {})
        processed_crop = _apply_crop_fraction(crop_image, crop_config)
        
        # Store original and processed dimensions for coordinate mapping
        original_height, original_width = crop_image.shape[:2]
        cropped_height, cropped_width = processed_crop.shape[:2]
        
        # Apply full preprocessing like tuning tab
        final_processed_crop = _preprocess_crop(processed_crop, crop_config)
        final_height, final_width = final_processed_crop.shape[:2]
        
        # Get EasyOCR parameters from config
        ocr_params = user_config.get('easyocr', {})        # Prepare parameters for readtext (same as tuning tab)
        readtext_params = {
            'text_threshold': ocr_params.get('text_threshold', 0.7),
            'low_text': ocr_params.get('low_text', 0.6),
            'link_threshold': ocr_params.get('link_threshold', 0.4),
            'width_ths': ocr_params.get('width_ths', 0.4),
            'height_ths': ocr_params.get('height_ths', 0.7),
            'canvas_size': ocr_params.get('canvas_size', 2560),
            'mag_ratio': ocr_params.get('mag_ratio', 2.0),
            'slope_ths': ocr_params.get('slope_ths', 0.1),
            'ycenter_ths': ocr_params.get('ycenter_ths', 0.5),
            'y_ths': ocr_params.get('y_ths', 0.5),
            'x_ths': ocr_params.get('x_ths', 1.0),
            'paragraph': ocr_params.get('paragraph', False),
            'adjust_contrast': ocr_params.get('adjust_contrast', 0.5),
            'filter_ths': ocr_params.get('filter_ths', 0.003),
            'batch_size': ocr_params.get('batch_size', 1),
            'workers': ocr_params.get('workers', 0),
            'decoder': ocr_params.get('decoder', 'greedy'),
            'beamWidth': ocr_params.get('beamWidth', 5),
            'detail': ocr_params.get('detail', 1)
        }
        
        # Add character filtering if specified
        if ocr_params.get('allowlist'):
            readtext_params['allowlist'] = ocr_params['allowlist']
        
        # End preprocessing timer
        timing_info['preprocessing_ms'] = (time.time() - prep_start_time) * 1000
        
        # Start OCR timer
        ocr_start_time = time.time()
        
        # Run EasyOCR with user settings (same as tuning tab)
        ocr_results = _easyocr_reader.readtext(final_processed_crop, **readtext_params)
        
        # Filter out low confidence detections (below 0.5)
        min_confidence = 0.5
        filtered_ocr_results = []
        for bbox, text, confidence in ocr_results:
            if confidence >= min_confidence:
                filtered_ocr_results.append((bbox, text, confidence))
            else:
                print(f"[PLAYER_ID] Filtered out low confidence detection: '{text}' ({confidence:.3f} < {min_confidence})")
        
        print(f"[PLAYER_ID] OCR results: {len(ocr_results)} total, {len(filtered_ocr_results)} after confidence filter")
        
        # End OCR timer
        timing_info['ocr_ms'] = (time.time() - ocr_start_time) * 1000
        
        # Process filtered results - find best numeric text (same as tuning tab)
        best_text = ""
        best_confidence = 0.0
        
        for bbox, text, confidence in filtered_ocr_results:
            # Clean text and check if it's a valid jersey number
            clean_text = ''.join(filter(str.isdigit, text))
            if clean_text and confidence > best_confidence:
                best_text = clean_text
                best_confidence = confidence
        
        # Validate jersey number and prepare result with coordinate mapping info
        if best_text and _validate_jersey_number(best_text):
            jersey_number = best_text
            result_details = {
                'confidence': best_confidence,
                'ocr_results': filtered_ocr_results,  # Use filtered results
                'best_text': best_text,
                'original_width': original_width,
                'original_height': original_height,
                'crop_width': cropped_width,
                'crop_height': cropped_height,
                'final_width': final_width,
                'final_height': final_height,
                'crop_fraction': crop_config.get('crop_top_fraction', 0.33)
            }
        else:
            jersey_number = "Unknown"
            result_details = {
                'confidence': 0.0,
                'ocr_results': filtered_ocr_results,  # Use filtered results
                'best_text': None,
                'original_width': original_width,
                'original_height': original_height,
                'crop_width': cropped_width,
                'crop_height': cropped_height,
                'final_width': final_width,
                'final_height': final_height,
                'crop_fraction': crop_config.get('crop_top_fraction', 0.33)
            }
        
        print(f"[PLAYER_ID] EasyOCR detected: {jersey_number} (confidence: {best_confidence:.3f})")
        return jersey_number, result_details, timing_info
        
    except Exception as e:
        print(f"[PLAYER_ID] EasyOCR error: {e}")
        return "Unknown", [], timing_info


def _load_easyocr_config() -> Dict[str, Any]:
    """Load EasyOCR configuration from user.yaml file."""
    try:
        # Find project root by looking for configs directory
        current_path = Path(__file__).parent
        project_root = None
        
        for parent in [current_path] + list(current_path.parents):
            if (parent / "configs").exists():
                project_root = parent
                break
        
        if project_root is None:
            print("[PLAYER_ID] Could not find configs directory")
            return {}
        
        config_path = project_root / "configs" / "user.yaml"
        
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


def _apply_crop_fraction(image: np.ndarray, preprocess_config: Dict[str, Any]) -> np.ndarray:
    """Apply crop fraction to image (same as tuning tab)."""
    crop_fraction = preprocess_config.get('crop_top_fraction', 0.33)
    if crop_fraction > 0:
        h = image.shape[0]
        crop_pixels = int(h * crop_fraction)
        return image[:crop_pixels, :]
    return image


def _preprocess_crop(crop: np.ndarray, preprocess_params: Dict[str, Any]) -> np.ndarray:
    """Apply preprocessing to a crop (copied from tuning tab)."""
    processed = crop.copy()
    
    # Resize (absolute takes priority over factor)
    abs_width = preprocess_params.get('resize_absolute_width', 0)
    abs_height = preprocess_params.get('resize_absolute_height', 0)
    resize_factor = preprocess_params.get('resize_factor', 1.0)
    
    if abs_width > 0 and abs_height > 0:
        # Absolute resize
        processed = cv2.resize(processed, (abs_width, abs_height))
    elif abs_width > 0:
        # Absolute width, maintain aspect ratio
        current_height, current_width = processed.shape[:2]
        new_height = int(current_height * abs_width / current_width)
        processed = cv2.resize(processed, (abs_width, new_height))
    elif abs_height > 0:
        # Absolute height, maintain aspect ratio
        current_height, current_width = processed.shape[:2]
        new_width = int(current_width * abs_height / current_height)
        processed = cv2.resize(processed, (new_width, abs_height))
    elif resize_factor != 1.0:
        # Factor-based resize
        new_height = int(processed.shape[0] * resize_factor)
        new_width = int(processed.shape[1] * resize_factor)
        processed = cv2.resize(processed, (new_width, new_height))
    
    # Color mode conversion
    if preprocess_params.get('bw_mode', True):
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    elif not preprocess_params.get('colour_mode', False):
        # Default grayscale processing
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # Denoising
    if preprocess_params.get('denoise', False):
        if len(processed.shape) == 3:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        else:
            processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
    
    # Contrast and brightness
    alpha = preprocess_params.get('contrast_alpha', 1.0)
    beta = preprocess_params.get('brightness_beta', 0)
    if alpha != 1.0 or beta != 0:
        processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
    
    # Gaussian blur
    blur_kernel = preprocess_params.get('gaussian_blur', 13)
    if blur_kernel > 0:
        # Ensure kernel size is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)
    
    # CLAHE enhancement
    if preprocess_params.get('enhance_contrast', False):
        clip_limit = preprocess_params.get('clahe_clip_limit', 3.0)
        grid_size = preprocess_params.get('clahe_grid_size', 8)
        
        if len(processed.shape) == 3:
            # Convert to LAB, apply CLAHE to L channel
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            processed = clahe.apply(processed)
    
    # Sharpening
    if preprocess_params.get('sharpen', True):
        strength = preprocess_params.get('sharpen_strength', 0.05)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * strength
        kernel[1,1] = 1 + (8 * strength)  # Adjust center to maintain brightness
        processed = cv2.filter2D(processed, -1, kernel)
    
    # Upscaling
    if preprocess_params.get('upscale', True):
        if preprocess_params.get('upscale_to_size', True):
            # Upscale to fixed size
            target_size = preprocess_params.get('upscale_target_size', 256)
            processed = cv2.resize(processed, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        else:
            # Upscale by factor
            factor = preprocess_params.get('upscale_factor', 3.0)
            new_height = int(processed.shape[0] * factor)
            new_width = int(processed.shape[1] * factor)
            processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return processed





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
