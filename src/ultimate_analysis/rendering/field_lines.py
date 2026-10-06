"""Drawing fitted field boundary lines, in the camera view or the top-down view."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..utils.logger import get_logger

logger = get_logger("RENDERING")


def draw_field_lines_ransac(
    frame: np.ndarray,
    fitted_lines: List[Tuple[np.ndarray, np.ndarray]],
    line_color: Tuple[int, int, int] = None,
    line_thickness: int = None,
    in_place: bool = False,
) -> np.ndarray:
    """Draw RANSAC-fitted field lines on the frame.

    Args:
        frame: Input frame to draw on
        fitted_lines: List of (start_point, end_point) tuples for each line
        line_color: BGR color for the lines
        line_thickness: Thickness of the lines
        in_place: Draw directly on frame instead of a copy (caller must own the frame)

    Returns:
        Frame with fitted lines drawn
    """
    if not fitted_lines:
        return frame

    # Use config values if parameters not provided
    if line_color is None:
        color_list = get_setting("models.segmentation.contour.line_color", [0, 255, 0])
        line_color = tuple(color_list) if isinstance(color_list, list) else (0, 255, 0)
    if line_thickness is None:
        line_thickness = get_setting("models.segmentation.contour.line_thickness", 3)

    result = frame if in_place else frame.copy()

    try:
        for i, (start_point, end_point) in enumerate(fitted_lines):
            # Convert points to integers for drawing
            start = tuple(start_point.astype(np.int32))
            end = tuple(end_point.astype(np.int32))

            # Draw the line
            cv2.line(result, start, end, line_color, line_thickness)

            # Draw endpoint markers
            cv2.circle(result, start, line_thickness + 2, line_color, -1)
            cv2.circle(result, end, line_thickness + 2, line_color, -1)

        logger.debug(f"Drew {len(fitted_lines)} RANSAC-fitted field lines")

    except Exception as e:
        logger.error(f"Error drawing RANSAC lines: {e}")

    return result


def draw_field_lines_ransac_with_outliers(
    frame: np.ndarray,
    fitted_lines: List[Tuple[np.ndarray, np.ndarray]],
    outlier_points: List[np.ndarray],
    line_color: Tuple[int, int, int] = None,
    line_thickness: int = None,
    outlier_color: Tuple[int, int, int] = None,
    outlier_radius: int = None,
    show_outliers: bool = None,
    in_place: bool = False,
) -> np.ndarray:
    """Draw RANSAC-fitted field lines and outlier points on the frame.

    Args:
        frame: Input frame to draw on
        fitted_lines: List of (start_point, end_point) tuples for each line
        outlier_points: List of outlier point arrays for each segment
        line_color: BGR color for the lines
        line_thickness: Thickness of the lines
        outlier_color: BGR color for outlier points
        outlier_radius: Radius for outlier points
        show_outliers: Whether to draw outlier points
        in_place: Draw directly on frame instead of a copy (caller must own the frame)

    Returns:
        Frame with fitted lines and outliers drawn
    """
    if not fitted_lines and not outlier_points:
        return frame

    result = frame if in_place else frame.copy()

    try:
        # Draw RANSAC lines first
        if fitted_lines:
            result = draw_field_lines_ransac(
                result, fitted_lines, line_color, line_thickness, in_place=True
            )

        # Draw outlier points if enabled
        if outlier_points and (
            show_outliers
            if show_outliers is not None
            else get_setting("models.segmentation.contour.ransac.show_outliers", True)
        ):
            # Get default values from config
            outlier_color = outlier_color or tuple(
                get_setting("models.segmentation.contour.ransac.outlier_color", [0, 0, 255])
            )
            outlier_radius = (
                outlier_radius
                if outlier_radius is not None
                else get_setting("models.segmentation.contour.ransac.outlier_radius", 3)
            )

            outlier_count = 0
            for segment_outliers in outlier_points:
                if segment_outliers is not None and len(segment_outliers) > 0:
                    for point in segment_outliers:
                        center = (int(point[0]), int(point[1]))
                        # Draw white border for better visibility
                        cv2.circle(result, center, outlier_radius + 1, (255, 255, 255), -1)
                        # Draw colored point on top
                        cv2.circle(result, center, outlier_radius, outlier_color, -1)
                        outlier_count += 1

            if outlier_count > 0:
                logger.debug(f"Drew {outlier_count} RANSAC outlier points")

    except Exception as e:
        logger.error(f"Error drawing RANSAC lines and outliers: {e}")

    return result


def draw_ransac_field_lines(
    frame: np.ndarray,
    ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
    confidences: List[float],
    transformation_matrix: Optional[np.ndarray] = None,
    scale_factor: float = 1.0,
    in_place: bool = False,
    show_confidence: Optional[bool] = None,
) -> np.ndarray:
    """Draw RANSAC-calculated field lines with clean visualization.

    Args:
        frame: Frame to draw on
        ransac_lines: List of (start_point, end_point) tuples from RANSAC
        confidences: List of confidence scores for each line
        transformation_matrix: Optional homography matrix to transform lines to warped view
        scale_factor: Scale factor for text and line thickness (useful for top-down view)
        in_place: Draw directly on frame instead of a copy (caller must own the frame)
        show_confidence: Label each line with its confidence (default: only at scale >= 1.5)

    Returns:
        Frame with RANSAC lines drawn
    """
    if not ransac_lines:
        return frame

    result = frame if in_place else frame.copy()
    if show_confidence is None:
        show_confidence = scale_factor >= 1.5  # Only show text at larger scales

    try:
        # Color scheme based on confidence
        line_colors = {
            "excellent": (64, 255, 64),  # Bright green - excellent confidence (>0.8)
            "good": (0, 200, 255),  # Orange-yellow - good confidence (0.6-0.8)
            "fair": (0, 165, 255),  # Orange - fair confidence (0.4-0.6)
            "low": (0, 100, 255),  # Red-orange - low confidence (<0.4)
        }

        # Line thickness based on scale factor
        base_thickness = max(1, int(2 * scale_factor))

        for i, ((start_point, end_point), confidence) in enumerate(zip(ransac_lines, confidences)):
            # Transform line if homography matrix provided
            if transformation_matrix is not None:
                # Convert points to homogeneous coordinates
                start_homo = np.array([start_point[0], start_point[1], 1.0])
                end_homo = np.array([end_point[0], end_point[1], 1.0])

                # Apply transformation
                start_transformed = transformation_matrix @ start_homo
                end_transformed = transformation_matrix @ end_homo

                # Convert back to 2D coordinates
                if start_transformed[2] != 0 and end_transformed[2] != 0:
                    start_2d = (start_transformed[:2] / start_transformed[2]).astype(int)
                    end_2d = (end_transformed[:2] / end_transformed[2]).astype(int)
                else:
                    continue  # Skip if transformation fails
            else:
                start_2d = start_point.astype(int)
                end_2d = end_point.astype(int)

            # Choose color based on confidence
            if confidence > 0.8:
                color = line_colors["excellent"]
            elif confidence > 0.6:
                color = line_colors["good"]
            elif confidence > 0.4:
                color = line_colors["fair"]
            else:
                color = line_colors["low"]

            # Draw the line
            cv2.line(result, tuple(start_2d), tuple(end_2d), color, base_thickness)

            # Optionally add confidence text near the line (for debugging)
            if show_confidence:
                mid_point = ((start_2d + end_2d) // 2).astype(int)
                font_scale = 0.4 * scale_factor
                cv2.putText(
                    result,
                    f"{confidence:.2f}",
                    tuple(mid_point),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    1,
                )

        # Add summary info if there are lines
        if len(ransac_lines) > 0:
            _draw_ransac_summary(result, ransac_lines, confidences, scale_factor)

        return result

    except Exception as e:
        logger.error(f"Error drawing RANSAC lines: {e}")
        return frame


def _draw_ransac_summary(
    frame: np.ndarray,
    ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
    confidences: List[float],
    scale_factor: float,
):
    """Draw summary information about RANSAC lines."""
    total_lines = len(ransac_lines)
    excellent_count = len([c for c in confidences if c > 0.8])
    good_count = len([c for c in confidences if 0.6 < c <= 0.8])
    avg_confidence = np.mean(confidences)

    # Position for text overlay
    text_y_start = 30
    font_scale = 0.5 * scale_factor
    text_color = (255, 255, 255)  # White

    # Draw summary
    summary_lines = [
        f"RANSAC Lines: {total_lines}",
        f"Excellent: {excellent_count}, Good: {good_count}",
        f"Avg Confidence: {avg_confidence:.2f}",
    ]

    for i, line in enumerate(summary_lines):
        y_pos = text_y_start + int(i * 20 * scale_factor)
        cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
