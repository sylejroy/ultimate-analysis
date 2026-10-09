"""Drawing the field segmentation: masks, outline, and simplified contour."""

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..processing.field_analysis import calculate_field_contour, fit_field_lines_ransac
from ..utils.logger import get_logger
from .field_lines import draw_field_lines_ransac_with_outliers

logger = get_logger("RENDERING")


# The main and top-down views alternate. A single entry evicts the unchanged main mask
# every frame; two entries keep it while the warped mask changes with camera motion.
_mask_outline_cache: Deque[Tuple[np.ndarray, tuple]] = deque(maxlen=2)


def _get_mask_outline(unified_mask: np.ndarray) -> tuple:
    """External contours of a field mask, computed once per mask object."""
    for index, (mask, contours) in enumerate(_mask_outline_cache):
        if mask is unified_mask:
            del _mask_outline_cache[index]
            _mask_outline_cache.append((mask, contours))
            return contours
    contours, _ = cv2.findContours(unified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _mask_outline_cache.append((unified_mask, contours))
    return contours


def get_segmentation_colors() -> Dict[int, Tuple[int, int, int]]:
    """Get the standard segmentation colors used throughout the application.

    Returns:
        Dictionary mapping class indices to BGR color tuples
    """
    return {
        0: (0, 255, 255),  # Central Field: bright cyan (BGR)
        1: (255, 0, 255),  # Endzone: bright magenta (BGR)
    }


def get_primary_field_color() -> Tuple[int, int, int]:
    """Get the primary field color (central field) for consistent visualization.

    Returns:
        BGR color tuple for the primary field area
    """
    colors = get_segmentation_colors()
    return colors[0]  # Return central field color (cyan)


def draw_field_segmentation(frame: np.ndarray, segmentation_results: List[Any]) -> np.ndarray:
    """Draw field segmentation masks and boundaries.

    Args:
        frame: Input frame to draw on
        segmentation_results: List of segmentation result objects

    Returns:
        Frame with field segmentation overlays
    """
    if not segmentation_results:
        return frame

    vis_frame = frame.copy()

    for result in segmentation_results:
        if not hasattr(result, "masks") or result.masks is None:
            continue

        try:
            # Get mask data
            if hasattr(result.masks, "data"):
                mask_data = result.masks.data
            else:
                continue

            # Convert to numpy if needed
            if hasattr(mask_data, "cpu"):
                mask = mask_data.cpu().numpy()
            else:
                mask = mask_data.numpy() if hasattr(mask_data, "numpy") else mask_data

            # Draw field segmentation overlay
            vis_frame = _draw_segmentation_masks(vis_frame, mask)

        except Exception as e:
            logger.error(f"Error drawing field segmentation: {e}")

    return vis_frame


def _draw_segmentation_masks(frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Draw segmentation masks with color overlays.

    Args:
        frame: Input frame
        masks: Mask array with shape (N, H, W)

    Returns:
        Frame with mask overlays
    """
    if masks.size == 0:
        return frame

    overlay = frame.copy()
    color_mask = np.zeros_like(frame)

    # Use centralized segmentation colors for consistency
    color_dict = get_segmentation_colors()

    name_dict = {0: "Central Field", 1: "Endzone"}

    n_classes = min(masks.shape[0], 2)  # Only process class 0 and 1
    frame_h, frame_w = frame.shape[:2]

    for cls in range(n_classes):
        # Resize each class mask to match frame size
        class_mask = masks[cls]

        # Skip if mask is empty
        if np.sum(class_mask) == 0:
            continue

        # Masks arrive at model resolution; scale smoothly so the edge is not blocky
        class_mask_resized = cv2.resize(
            class_mask.astype(np.float32), (frame_w, frame_h), interpolation=cv2.INTER_LINEAR
        )
        mask_bool = class_mask_resized > 0.5
        class_mask_resized = mask_bool.view(np.uint8)

        # Skip if no pixels in mask
        if not np.any(mask_bool):
            continue

        color = color_dict.get(cls, (200, 200, 200))
        color_mask[mask_bool] = color

        # Draw border for the mask
        contours, _ = cv2.findContours(
            class_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            border_color = tuple(int(c * 0.7) for c in color)
            cv2.drawContours(overlay, contours, -1, border_color, 2)

            # Find center of mask for label
            ys, xs = np.where(mask_bool)
            if len(xs) > 0 and len(ys) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                label = name_dict.get(cls, str(cls))

                cv2.putText(
                    overlay,
                    label,
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    border_color,
                    2,
                    cv2.LINE_AA,
                )

    # Blend with higher alpha for maximum visibility
    cv2.addWeighted(color_mask, 0.4, overlay, 0.6, 0, overlay)
    return overlay


def draw_field_contour(
    frame: np.ndarray,
    contour: np.ndarray,
    contour_color: Tuple[int, int, int] = None,
    point_color: Tuple[int, int, int] = None,
    line_thickness: int = None,
    point_radius: int = None,
    draw_points: bool = None,
    in_place: bool = False,
) -> np.ndarray:
    """Draw field contour lines and points on the frame.

    Args:
        frame: Input frame to draw on
        contour: Contour points as numpy array of shape (N, 1, 2)
        contour_color: BGR color for contour lines
        point_color: BGR color for contour points
        line_thickness: Thickness of contour lines
        point_radius: Radius of contour points
        draw_points: Whether to draw individual contour points
        in_place: Draw directly on frame instead of a copy (caller must own the frame)

    Returns:
        Frame with contour overlay
    """
    if contour is None or len(contour) == 0:
        return frame

    # Use config values if parameters not provided
    if contour_color is None:
        # Get color from config as list and convert to tuple
        color_list = get_setting("models.segmentation.contour.line_color", [255, 255, 0])
        contour_color = tuple(color_list) if isinstance(color_list, list) else (255, 255, 0)
    if point_color is None:
        color_list = get_setting("models.segmentation.contour.point_color", [0, 255, 255])
        point_color = tuple(color_list) if isinstance(color_list, list) else (0, 255, 255)
    if line_thickness is None:
        line_thickness = get_setting("models.segmentation.contour.line_thickness", 3)
    if point_radius is None:
        point_radius = get_setting("models.segmentation.contour.point_radius", 5)
    if draw_points is None:
        draw_points = get_setting("models.segmentation.contour.draw_points", True)

    result = frame if in_place else frame.copy()

    try:
        # Draw contour lines
        cv2.drawContours(result, [contour], -1, contour_color, line_thickness)

        # Draw contour points if enabled
        if draw_points:
            for point in contour:
                center = tuple(point[0])  # point is shape (1, 2), so point[0] is (x, y)
                cv2.circle(result, center, point_radius, point_color, -1)
                # Add small white border for better visibility
                cv2.circle(result, center, point_radius + 1, (255, 255, 255), 1)

        logger.debug(f"Drew contour with {len(contour)} points")

    except Exception as e:
        logger.error(f"Error drawing field contour: {e}")

    return result


def _draw_points(frame: np.ndarray, points: Any, radius: int, color: Tuple[int, ...]) -> None:
    """Dots with a white rim, for the points a line fit used or left out."""
    height, width = frame.shape[:2]
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), radius + 1, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), radius, color, -1)


def draw_unified_field_mask(
    frame: np.ndarray,
    unified_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
    draw_contour: bool = True,
    fill_mask: bool = False,
    ransac_fit: Optional[tuple] = None,
    field_contour: Optional[np.ndarray] = None,
    in_place: bool = False,
) -> np.ndarray:
    """Draw the field: its outline, and the fitted lines or the simplified outline.

    Args:
        frame: Input frame to draw on
        unified_mask: Binary mask (H, W) where 1 indicates field area
        color: BGR color tuple for the overlay
        alpha: Transparency of the fill (0.0 = transparent, 1.0 = opaque)
        draw_contour: Also draw the fitted lines (or, without them, the simplified outline)
        fill_mask: Whether to fill the mask area (False = outline only)
        ransac_fit: fit_field_lines_ransac result for this mask. The lines are only fitted
            here when this is None, so per-frame callers should pass a cached fit.
        field_contour: calculate_field_contour result for this mask, if already known
        in_place: Draw directly on frame instead of a copy (caller must own the frame)

    Returns:
        The frame with the field drawn
    """
    # A mask without an outline is empty; the outline is kept per mask, where looking
    # through the mask for a set pixel would be done again in every frame
    if unified_mask is None or not _get_mask_outline(unified_mask):
        return frame

    result = frame if in_place else frame.copy()

    if fill_mask:
        overlay = frame.copy()
        overlay[unified_mask == 1] = color
        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, dst=result)

    contours = _get_mask_outline(unified_mask)
    if contours:
        border_color = tuple(int(c * 0.7) for c in color)
        cv2.drawContours(result, contours, -1, border_color, 2)

    if not draw_contour:
        return result

    def setting(name: str, default: Any) -> Any:
        return get_setting(f"models.segmentation.contour.ransac.{name}", default)

    contour = field_contour
    if setting("enabled", False):
        if ransac_fit is None:
            if contour is None:
                contour = calculate_field_contour(unified_mask)
            if contour is None:
                return result
            ransac_fit = fit_field_lines_ransac(
                contour,
                frame.shape,
                num_lines=setting("num_lines", 4),
                distance_threshold=setting("distance_threshold", 10.0),
                min_samples=setting("min_samples", 2),
                max_trials=setting("max_trials", 1000),
            )
        lines, outliers, inliers, edge_points = ransac_fit
        if lines:
            result = draw_field_lines_ransac_with_outliers(
                result,
                lines,
                outliers,
                line_color=tuple(setting("line_color", [0, 255, 0])),
                in_place=True,
            )
            if (
                setting("edge_filtering.enabled", False)
                and setting("edge_filtering.show_edge_points", True)
                and len(edge_points) > 0
            ):
                _draw_points(
                    result,
                    edge_points,
                    setting("edge_filtering.edge_point_radius", 2),
                    tuple(setting("edge_filtering.edge_point_color", [0, 0, 255])),
                )
            if setting("show_inliers", True):
                inlier_color = tuple(setting("inlier_color", [0, 255, 0]))
                for on_line in inliers:
                    _draw_points(result, on_line, setting("inlier_radius", 2), inlier_color)
            return result
        logger.debug("No field line found, drawing the simplified outline instead")

    if contour is None:
        contour = calculate_field_contour(unified_mask)
    if contour is not None:
        result = draw_field_contour(result, contour, in_place=True)
    return result
