"""Image preprocessing utilities for OCR optimization.

This module provides centralized image preprocessing functionality used by both
the EasyOCR tuning tab and player ID detection. It includes adaptive preprocessing
that adjusts parameters based on image conditions for improved OCR accuracy.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..config.settings import get_setting


def preprocess_crop(crop: np.ndarray, preprocess_params: Optional[Dict[str, Any]] = None, 
                   adaptive: bool = True) -> np.ndarray:
    """Apply preprocessing to a crop with optional adaptive enhancements.
    
    Args:
        crop: Input image crop as numpy array
        preprocess_params: Dictionary of preprocessing parameters. If None, uses default settings.
        adaptive: Whether to apply adaptive preprocessing based on image conditions
        
    Returns:
        Preprocessed image crop
    """
    if preprocess_params is None:
        preprocess_params = get_default_preprocessing_params()
    
    processed = crop.copy()
    
    # Apply adaptive adjustments if enabled
    if adaptive:
        preprocess_params = _apply_adaptive_adjustments(processed, preprocess_params)
    
    # Resize (absolute takes priority over factor)
    processed = _apply_resize(processed, preprocess_params)
    
    # Color mode conversion
    processed = _apply_color_conversion(processed, preprocess_params)
    
    # Denoising
    processed = _apply_denoising(processed, preprocess_params)
    
    # Contrast and brightness
    processed = _apply_contrast_brightness(processed, preprocess_params)
    
    # Gaussian blur
    processed = _apply_gaussian_blur(processed, preprocess_params)
    
    # CLAHE enhancement
    processed = _apply_clahe_enhancement(processed, preprocess_params)
    
    # Sharpening
    processed = _apply_sharpening(processed, preprocess_params)
    
    # Upscaling
    processed = _apply_upscaling(processed, preprocess_params)
    
    return processed


def get_default_preprocessing_params() -> Dict[str, Any]:
    """Get default preprocessing parameters from configuration.
    
    Returns:
        Dictionary of default preprocessing parameters
    """
    return {
        # Resize parameters
        'resize_absolute_width': get_setting("processing.ocr.preprocessing.resize_absolute_width", 0),
        'resize_absolute_height': get_setting("processing.ocr.preprocessing.resize_absolute_height", 0),
        'resize_factor': get_setting("processing.ocr.preprocessing.resize_factor", 1.0),
        
        # Color mode parameters
        'bw_mode': get_setting("processing.ocr.preprocessing.bw_mode", True),
        'colour_mode': get_setting("processing.ocr.preprocessing.colour_mode", False),
        
        # Enhancement parameters
        'denoise': get_setting("processing.ocr.preprocessing.denoise", False),
        'contrast_alpha': get_setting("processing.ocr.preprocessing.contrast_alpha", 1.0),
        'brightness_beta': get_setting("processing.ocr.preprocessing.brightness_beta", 0),
        'gaussian_blur': get_setting("processing.ocr.preprocessing.gaussian_blur", 13),
        
        # Advanced enhancement
        'enhance_contrast': get_setting("processing.ocr.preprocessing.enhance_contrast", False),
        'clahe_clip_limit': get_setting("processing.ocr.preprocessing.clahe_clip_limit", 3.0),
        'clahe_grid_size': get_setting("processing.ocr.preprocessing.clahe_grid_size", 8),
        'sharpen': get_setting("processing.ocr.preprocessing.sharpen", True),
        'sharpen_strength': get_setting("processing.ocr.preprocessing.sharpen_strength", 0.05),
        
        # Upscaling parameters
        'upscale': get_setting("processing.ocr.preprocessing.upscale", True),
        'upscale_to_size': get_setting("processing.ocr.preprocessing.upscale_to_size", True),
        'upscale_target_size': get_setting("processing.ocr.preprocessing.upscale_target_size", 256),
        'upscale_factor': get_setting("processing.ocr.preprocessing.upscale_factor", 3.0),
    }


def _apply_adaptive_adjustments(image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply adaptive adjustments to preprocessing parameters based on image conditions.
    
    Args:
        image: Input image to analyze
        params: Original preprocessing parameters
        
    Returns:
        Adjusted preprocessing parameters
    """
    adapted_params = params.copy()
    
    # Analyze image properties
    height, width = image.shape[:2]
    total_pixels = height * width
    
    # Convert to grayscale for analysis if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate image statistics
    mean_brightness = np.mean(gray)
    brightness_std = np.std(gray)
    contrast_ratio = brightness_std / (mean_brightness + 1e-8)
    
    # Adaptive brightness adjustment
    if mean_brightness < 80:  # Dark image
        adapted_params['brightness_beta'] = max(adapted_params.get('brightness_beta', 0), 20)
        adapted_params['contrast_alpha'] = min(adapted_params.get('contrast_alpha', 1.0) * 1.2, 2.0)
    elif mean_brightness > 180:  # Bright image
        adapted_params['brightness_beta'] = min(adapted_params.get('brightness_beta', 0), -10)
        adapted_params['contrast_alpha'] = max(adapted_params.get('contrast_alpha', 1.0) * 0.9, 0.5)
    
    # Adaptive contrast enhancement
    if contrast_ratio < 0.3:  # Low contrast
        adapted_params['enhance_contrast'] = True
        adapted_params['clahe_clip_limit'] = min(adapted_params.get('clahe_clip_limit', 3.0) * 1.5, 6.0)
    elif contrast_ratio > 1.0:  # High contrast
        adapted_params['enhance_contrast'] = False
        adapted_params['gaussian_blur'] = max(adapted_params.get('gaussian_blur', 0), 3)
    
    # Adaptive denoising for small images
    if total_pixels < 1000:  # Very small image
        adapted_params['denoise'] = True
        adapted_params['upscale_factor'] = max(adapted_params.get('upscale_factor', 1.0), 4.0)
    
    # Adaptive sharpening
    if brightness_std < 20:  # Low detail/blurry image
        adapted_params['sharpen'] = True
        adapted_params['sharpen_strength'] = min(adapted_params.get('sharpen_strength', 0.05) * 1.5, 0.2)
    
    return adapted_params


def _apply_resize(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply resize transformation."""
    abs_width = params.get('resize_absolute_width', 0)
    abs_height = params.get('resize_absolute_height', 0)
    resize_factor = params.get('resize_factor', 1.0)
    
    if abs_width > 0 and abs_height > 0:
        # Absolute resize
        return cv2.resize(image, (abs_width, abs_height))
    elif abs_width > 0:
        # Absolute width, maintain aspect ratio
        current_height, current_width = image.shape[:2]
        new_height = int(current_height * abs_width / current_width)
        return cv2.resize(image, (abs_width, new_height))
    elif abs_height > 0:
        # Absolute height, maintain aspect ratio
        current_height, current_width = image.shape[:2]
        new_width = int(current_width * abs_height / current_height)
        return cv2.resize(image, (new_width, abs_height))
    elif resize_factor != 1.0:
        # Factor-based resize
        new_height = int(image.shape[0] * resize_factor)
        new_width = int(image.shape[1] * resize_factor)
        return cv2.resize(image, (new_width, new_height))
    
    return image


def _apply_color_conversion(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply color mode conversion."""
    if params.get('bw_mode', True):
        if len(image.shape) == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    elif not params.get('colour_mode', False):
        # Default grayscale processing
        if len(image.shape) == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    return image


def _apply_denoising(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply denoising."""
    if params.get('denoise', False):
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    return image


def _apply_contrast_brightness(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply contrast and brightness adjustment."""
    alpha = params.get('contrast_alpha', 1.0)
    beta = params.get('brightness_beta', 0)
    
    if alpha != 1.0 or beta != 0:
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image


def _apply_gaussian_blur(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply Gaussian blur."""
    blur_kernel = params.get('gaussian_blur', 0)
    
    if blur_kernel > 0:
        # Ensure kernel size is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        return cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    
    return image


def _apply_clahe_enhancement(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply CLAHE enhancement."""
    if params.get('enhance_contrast', False):
        clip_limit = params.get('clahe_clip_limit', 3.0)
        grid_size = params.get('clahe_grid_size', 8)
        
        if len(image.shape) == 3:
            # Convert to LAB, apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            return clahe.apply(image)
    
    return image


def _apply_sharpening(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply sharpening."""
    if params.get('sharpen', True):
        strength = params.get('sharpen_strength', 0.05)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * strength
        kernel[1,1] = 1 + (8 * strength)  # Adjust center to maintain brightness
        return cv2.filter2D(image, -1, kernel)
    
    return image


def _apply_upscaling(image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply upscaling."""
    if params.get('upscale', True):
        if params.get('upscale_to_size', True):
            # Upscale to fixed size
            target_size = params.get('upscale_target_size', 256)
            return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        else:
            # Upscale by factor
            factor = params.get('upscale_factor', 3.0)
            new_height = int(image.shape[0] * factor)
            new_width = int(image.shape[1] * factor)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return image


def analyze_image_quality(image: np.ndarray) -> Dict[str, float]:
    """Analyze image quality metrics for adaptive preprocessing.
    
    Args:
        image: Input image to analyze
        
    Returns:
        Dictionary containing quality metrics
    """
    # Convert to grayscale for analysis if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    height, width = gray.shape
    total_pixels = height * width
    
    # Basic statistics
    mean_brightness = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    contrast_ratio = brightness_std / (mean_brightness + 1e-8)
    
    # Sharpness measure using Laplacian variance
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Noise estimation using high-frequency content
    noise_estimate = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1))))
    
    return {
        'width': float(width),
        'height': float(height),
        'total_pixels': float(total_pixels),
        'mean_brightness': mean_brightness,
        'brightness_std': brightness_std,
        'contrast_ratio': contrast_ratio,
        'sharpness': laplacian_var,
        'noise_estimate': noise_estimate,
    }
