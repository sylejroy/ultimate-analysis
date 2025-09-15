"""Segmentation utility functions for Ultimate Analysis.

This module contains common segmentation-related functions used across GUI tabs.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..constants import DEFAULT_PATHS
from ..utils.logger import get_logger
from ..gui.visualization import (
    calculate_field_contour,
    create_unified_field_mask,
    draw_field_contour,
    draw_unified_field_mask,
    get_primary_field_color,
)


def transform_contour_points(contour: np.ndarray, h_matrix: np.ndarray) -> Optional[np.ndarray]:
    """Transform contour points using homography matrix.

    Args:
        contour: Contour points as numpy array of shape (N, 1, 2)
        h_matrix: 3x3 homography transformation matrix

    Returns:
        Transformed contour points in same format, or None if transformation fails
    """
    if contour is None or len(contour) == 0:
        return None

    try:
        # Reshape contour points for perspective transformation
        # contour is (N, 1, 2), we need (N, 2) for cv2.perspectiveTransform
        points = contour.reshape(-1, 1, 2).astype(np.float32)

        # Apply perspective transformation
        transformed_points = cv2.perspectiveTransform(points, h_matrix)

        # Reshape back to contour format (N, 1, 2)
        transformed_contour = transformed_points.reshape(-1, 1, 2).astype(np.int32)

        return transformed_contour

    except Exception as e:
        print(f"[SEGMENTATION_UTILS] Error transforming contour points: {e}")
        return None


def apply_segmentation_to_warped_frame(
    warped_frame: np.ndarray,
    segmentation_results: List[Any],
    homography_matrix: np.ndarray,
    original_frame_shape: Tuple[int, int],
    tab_name: str = "MAIN_TAB",
) -> np.ndarray:
    """Apply segmentation overlay to warped frame by transforming contour points from original image.

    Args:
        warped_frame: The homography-transformed frame
        segmentation_results: List of segmentation results from processing
        homography_matrix: 3x3 homography transformation matrix
        original_frame_shape: Shape of original frame (height, width)
        tab_name: Name of calling tab for logging

    Returns:
        Warped frame with segmentation overlay applied
    """
    if not segmentation_results:
        return warped_frame

    try:
        # Create unified mask from segmentation results on original frame
        unified_mask = create_unified_field_mask(segmentation_results, original_frame_shape)

        if unified_mask is None:
            logger = get_logger("SEGMENTATION_UTILS")
            logger.debug(f"[{tab_name}] No unified mask could be created")
            return warped_frame

        logger = get_logger("SEGMENTATION_UTILS")
        logger.debug(
            f"[{tab_name}] Created unified mask with shape {unified_mask.shape}, {np.sum(unified_mask)} pixels"
        )

        # Calculate contour on the original image
        original_contour = calculate_field_contour(unified_mask)

        if original_contour is None or len(original_contour) == 0:
            print(f"[{tab_name}] No contour found in original unified mask")
            return warped_frame

        # Transform contour points using homography matrix
        transformed_contour = transform_contour_points(original_contour, homography_matrix)

        if transformed_contour is None:
            print(f"[{tab_name}] Failed to transform contour points")
            return warped_frame

        # Create mask from transformed contour points
        warped_mask = np.zeros((warped_frame.shape[0], warped_frame.shape[1]), dtype=np.uint8)
        cv2.fillPoly(warped_mask, [transformed_contour], 1)

        # Apply overlay and draw contour on warped frame - contour only for consistency
        field_color = get_primary_field_color()  # Bright cyan (BGR) - same as segmentation
        result_frame, _, _ = draw_unified_field_mask(
            warped_frame, warped_mask, field_color, alpha=0.4, draw_contour=False, fill_mask=False
        )

        # Draw the transformed contour directly
        result_frame = draw_field_contour(result_frame, transformed_contour)

        logger = get_logger("SEGMENTATION_UTILS")
        logger.debug(
            f"[{tab_name}] Applied transformed contour to warped frame: {len(transformed_contour)} points"
        )
        return result_frame

    except Exception as e:
        print(f"[{tab_name}] Error applying transformed segmentation to warped frame: {e}")
        import traceback

        traceback.print_exc()
        return warped_frame


def load_segmentation_models() -> List[str]:
    """Load available field segmentation models.

    Returns:
        List of full paths to segmentation model files
    """
    available_models = []

    # Look for models in the models directory
    models_path = Path(get_setting("models.base_path", DEFAULT_PATHS["MODELS"]))

    if not models_path.exists():
        print(f"[SEGMENTATION_UTILS] Models directory not found: {models_path}")
        return available_models

    # Search for segmentation model files
    model_files = []
    for model_dir in models_path.rglob("*"):
        if model_dir.is_file() and model_dir.suffix == ".pt":
            # Skip last.pt files - we only want best.pt from finetuned models
            if model_dir.name == "last.pt":
                continue

            # Check if this is a segmentation model
            if "segmentation" in str(model_dir).lower():
                model_files.append(str(model_dir))

    # Add pretrained segmentation models
    pretrained_path = models_path / "pretrained"
    if pretrained_path.exists():
        for model_file in pretrained_path.glob("*seg*.pt"):
            model_files.append(str(model_file))

    # Store the full paths
    available_models = model_files

    print(f"[SEGMENTATION_UTILS] Loaded {len(available_models)} segmentation models")
    return available_models


def create_model_display_name(model_path: str) -> str:
    """Create a display name from a model path.

    Args:
        model_path: Full path to the model file

    Returns:
        User-friendly display name for the model
    """
    if "pretrained" in model_path:
        return f"Pretrained: {Path(model_path).stem}"
    else:
        # Extract model name from path
        path_parts = Path(model_path).parts
        if "segmentation" in path_parts:
            seg_index = path_parts.index("segmentation")
            if seg_index + 1 < len(path_parts):
                return path_parts[seg_index + 1]
            else:
                return Path(model_path).parent.parent.name
        else:
            return Path(model_path).parent.parent.name


def populate_segmentation_model_combo(
    combo_box, available_models: List[str], default_model_path: str = None
) -> None:
    """Populate a combo box with segmentation models and select default.

    Args:
        combo_box: QComboBox to populate
        available_models: List of model paths
        default_model_path: Optional path to auto-select as default
    """
    combo_box.clear()

    for model_path in available_models:
        display_name = create_model_display_name(model_path)
        combo_box.addItem(display_name, model_path)

    # Auto-select the default model if specified
    if default_model_path and combo_box.count() > 0:
        for i in range(combo_box.count()):
            item_path = combo_box.itemData(i)
            if item_path and default_model_path in str(item_path):
                combo_box.setCurrentIndex(i)
                print(
                    f"[SEGMENTATION_UTILS] Auto-selected default segmentation model: {combo_box.itemText(i)}"
                )
                break
