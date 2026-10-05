"""Preparing player crops for EasyOCR and picking the jersey number from its output.

Shared by live player identification and the EasyOCR tuning tab, so a setting tuned in
the tab behaves the same during analysis.
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def crop_top_fraction(image: np.ndarray, preprocess_config: Dict[str, Any]) -> np.ndarray:
    """The upper part of a player crop, where the jersey number is."""
    crop_fraction = preprocess_config.get("crop_top_fraction", 0.33)
    if crop_fraction > 0:
        h = image.shape[0]
        crop_pixels = int(h * crop_fraction)
        return image[:crop_pixels, :]
    return image


def preprocess_crop(crop: np.ndarray, preprocess_params: Dict[str, Any]) -> np.ndarray:
    """Apply the configured resizing, colour, contrast, and sharpening steps to a crop."""
    processed = crop.copy()

    # Resize (absolute takes priority over factor)
    abs_width = preprocess_params.get("resize_absolute_width", 0)
    abs_height = preprocess_params.get("resize_absolute_height", 0)
    resize_factor = preprocess_params.get("resize_factor", 1.0)

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
    if preprocess_params.get("bw_mode", True):
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    elif not preprocess_params.get("colour_mode", False):
        # Default grayscale processing
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # Denoising
    if preprocess_params.get("denoise", False):
        if len(processed.shape) == 3:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        else:
            processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)

    # Contrast and brightness
    alpha = preprocess_params.get("contrast_alpha", 1.0)
    beta = preprocess_params.get("brightness_beta", 0)
    if alpha != 1.0 or beta != 0:
        processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)

    # Gaussian blur
    blur_kernel = preprocess_params.get("gaussian_blur", 13)
    if blur_kernel > 0:
        # Ensure kernel size is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)

    # CLAHE enhancement
    if preprocess_params.get("enhance_contrast", False):
        clip_limit = preprocess_params.get("clahe_clip_limit", 3.0)
        grid_size = preprocess_params.get("clahe_grid_size", 8)

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
    if preprocess_params.get("sharpen", True):
        strength = preprocess_params.get("sharpen_strength", 0.05)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * strength
        kernel[1, 1] = 1 + (8 * strength)  # Adjust center to maintain brightness
        processed = cv2.filter2D(processed, -1, kernel)

    # Upscaling
    if preprocess_params.get("upscale", True):
        if preprocess_params.get("upscale_to_size", True):
            # Upscale to fixed size
            target_size = preprocess_params.get("upscale_target_size", 256)
            processed = cv2.resize(
                processed, (target_size, target_size), interpolation=cv2.INTER_CUBIC
            )
        else:
            # Upscale by factor
            factor = preprocess_params.get("upscale_factor", 3.0)
            new_height = int(processed.shape[0] * factor)
            new_width = int(processed.shape[1] * factor)
            processed = cv2.resize(
                processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )

    return processed


def easyocr_readtext_parameters(ocr_params: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword arguments for easyocr.Reader.readtext from the stored OCR settings."""
    parameters = {
        "text_threshold": ocr_params.get("text_threshold", 0.7),
        "low_text": ocr_params.get("low_text", 0.6),
        "link_threshold": ocr_params.get("link_threshold", 0.4),
        "width_ths": ocr_params.get("width_ths", 0.4),
        "height_ths": ocr_params.get("height_ths", 0.7),
        "canvas_size": ocr_params.get("canvas_size", 2560),
        "mag_ratio": ocr_params.get("mag_ratio", 2.0),
        "slope_ths": ocr_params.get("slope_ths", 0.1),
        "ycenter_ths": ocr_params.get("ycenter_ths", 0.5),
        "y_ths": ocr_params.get("y_ths", 0.5),
        "x_ths": ocr_params.get("x_ths", 1.0),
        "paragraph": ocr_params.get("paragraph", False),
        "adjust_contrast": ocr_params.get("adjust_contrast", 0.5),
        "filter_ths": ocr_params.get("filter_ths", 0.003),
        "batch_size": ocr_params.get("batch_size", 1),
        "workers": ocr_params.get("workers", 0),
        "decoder": ocr_params.get("decoder", "greedy"),
        "beamWidth": ocr_params.get("beamWidth", 5),
        "detail": ocr_params.get("detail", 1),
    }
    # Restrict the recognizer to these characters, if any are given
    if ocr_params.get("allowlist"):
        parameters["allowlist"] = ocr_params["allowlist"]
    return parameters


def best_number(ocr_results: List[Tuple[Any, str, float]]) -> Tuple[str, float]:
    """The most confident reading that contains digits, as (digits, confidence).

    Returns ("", 0.0) when nothing numeric was read.
    """
    best_text, best_confidence = "", 0.0
    for _, text, confidence in ocr_results:
        digits = "".join(filter(str.isdigit, text))
        if digits and confidence > best_confidence:
            best_text, best_confidence = digits, confidence
    return best_text, best_confidence
