"""Visualization functions for Ultimate Analysis GUI.

This module provides functions for drawing detection boxes, tracking overlays,
player IDs, and field segmentation on video frames.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..constants import VISUALIZATION_COLORS
from ..utils.logger import get_logger

# Initialize logger
logger = get_logger("VISUALIZATION")


# Outline of the most recently drawn field mask: (mask, contours). The mask only changes
# when segmentation runs again, so the frames in between reuse its outline.
_mask_outline_cache: Tuple[Optional[np.ndarray], tuple] = (None, ())


def _get_mask_outline(unified_mask: np.ndarray) -> tuple:
    """External contours of a field mask, computed once per mask object."""
    global _mask_outline_cache
    if _mask_outline_cache[0] is not unified_mask:
        contours, _ = cv2.findContours(unified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _mask_outline_cache = (unified_mask, contours)
    return _mask_outline_cache[1]


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


def draw_detections(
    frame: np.ndarray, detections: List[Dict[str, Any]], in_place: bool = False
) -> np.ndarray:
    """Draw detection bounding boxes and labels on frame.

    Args:
        frame: Input frame to draw on
        detections: List of detection dictionaries
        in_place: Draw directly on frame instead of a copy (caller must own the frame)

    Returns:
        Frame with detection overlays
    """
    if not detections:
        return frame

    # Create a copy to avoid modifying original
    vis_frame = frame if in_place else frame.copy()

    for detection in detections:
        bbox = detection.get("bbox", [])
        confidence = detection.get("confidence", 0.0)
        class_name = detection.get("class_name", "unknown")
        model_type = detection.get("model_type", "unknown")

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Choose color based on model type to differentiate between the two detection models
        color = VISUALIZATION_COLORS["DETECTION_BOX"]  # Default fallback (green)

        # Primary color selection based on model type
        if model_type == "player_model":
            color = VISUALIZATION_COLORS["PLAYER_MODEL"]  # Bright green for player model
        elif model_type == "disc_model":
            color = VISUALIZATION_COLORS["DISC_MODEL"]  # Bright orange for disc model
        else:
            # Fallback to class-based coloring for backward compatibility
            if class_name and isinstance(class_name, str):
                class_name_lower = class_name.lower().strip()
                if class_name_lower == "disc" or "disc" in class_name_lower:
                    color = VISUALIZATION_COLORS["DISC"]  # Bright cyan for disc
                elif class_name_lower == "player" or "player" in class_name_lower:
                    color = VISUALIZATION_COLORS["PLAYER"]  # Subtle gray for player

        # Draw bounding box with thicker line for better visibility
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)

        # Create enhanced label showing both class and model
        model_label = ""
        if model_type == "player_model":
            model_label = " [PM]"  # Player Model
        elif model_type == "disc_model":
            model_label = " [DM]"  # Disc Model

        label = f"{class_name}: {confidence:.2f}{model_label}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # Draw label background
        cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)

        # Draw label text
        cv2.putText(
            vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return vis_frame


def draw_tracks_with_player_ids(
    frame: np.ndarray,
    tracks: List[Any],
    track_histories: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    player_ids: Optional[Dict[int, Tuple[str, Any]]] = None,
    in_place: bool = False,
) -> np.ndarray:
    """Draw tracking bounding boxes with player jersey numbers and confidence.

    Args:
        frame: Input frame to draw on
        tracks: List of track objects
        track_histories: Optional dictionary of track histories
        player_ids: Optional dictionary mapping track_id -> (jersey_number, details)
        in_place: Draw directly on frame instead of a copy (caller must own the frame)

    Returns:
        Frame with tracking and player ID overlays
    """
    if not tracks:
        return frame

    vis_frame = frame if in_place else frame.copy()

    for track in tracks:
        # Get track properties
        track_id = getattr(track, "track_id", None)
        if track_id is None:
            continue

        # Get bounding box
        bbox = None
        if hasattr(track, "to_ltrb"):
            bbox = track.to_ltrb()
        elif hasattr(track, "to_tlbr"):
            bbox = track.to_tlbr()
        elif hasattr(track, "bbox"):
            bbox = track.bbox

        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Generate unique color for each track ID
        color = _get_track_color(track_id)

        # Draw thinner bounding box (thickness 1 instead of 3)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 1)

        # Handle player ID display
        jersey_number = "Unknown"
        details = None

        if player_ids and track_id in player_ids:
            jersey_number, details = player_ids[track_id]

        # Always show track ID and jersey number when using player ID mode
        if jersey_number != "Unknown":
            # Create simple jersey number label without confidence
            jersey_label = f"#{jersey_number}"
        else:
            # Show compact "?" for unknown tracks
            jersey_label = f"{track_id}:?"

        # Calculate label size and position using smaller font
        label_size = cv2.getTextSize(jersey_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Draw label background
        cv2.rectangle(
            vis_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0] + 10, y1), color, -1
        )

        # Draw jersey number text using smaller, slightly bold font
        cv2.putText(
            vis_frame,
            jersey_label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Draw detection regions if OCR results are available for known players AND not finalized
        if (
            jersey_number != "Unknown"
            and isinstance(details, dict)
            and "ocr_results" in details
            and not details.get("finalized", False)
        ):
            ocr_results = details["ocr_results"]
            if ocr_results:
                # Get transformation information
                original_width = details.get("original_width", x2 - x1)
                original_height = details.get("original_height", y2 - y1)
                crop_width = details.get("crop_width", original_width)
                crop_height = details.get("crop_height", original_height)
                final_width = details.get("final_width", crop_width)
                final_height = details.get("final_height", crop_height)
                crop_fraction = details.get("crop_fraction", 0.33)

                # Calculate the actual dimensions of the jersey area in the track
                track_width = x2 - x1
                track_height = y2 - y1
                jersey_area_height = int(track_height * crop_fraction)

                # Draw bounding boxes for each OCR detection
                for bbox_ocr, text, conf in ocr_results:
                    if isinstance(bbox_ocr, list) and len(bbox_ocr) == 4:
                        # EasyOCR bbox format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                        ocr_points = np.array(bbox_ocr, dtype=np.float32)
                        ocr_x1 = int(np.min(ocr_points[:, 0]))
                        ocr_y1 = int(np.min(ocr_points[:, 1]))
                        ocr_x2 = int(np.max(ocr_points[:, 0]))
                        ocr_y2 = int(np.max(ocr_points[:, 1]))

                        # Correct coordinate transformation accounting for all processing steps:
                        # 1. Original track -> Cropped jersey area (crop_fraction)
                        # 2. Cropped area -> Resized for processing (128x64)
                        # 3. OCR results are in the final processed image coordinates

                        if (
                            final_width > 0
                            and final_height > 0
                            and crop_width > 0
                            and crop_height > 0
                        ):
                            # Step 1: Scale from final processed coordinates back to cropped coordinates
                            # The final processed image maintains aspect ratio, so we need to account for padding
                            crop_to_final_scale_x = crop_width / final_width
                            crop_to_final_scale_y = crop_height / final_height

                            # Scale OCR coordinates back to cropped image space
                            crop_x1 = ocr_x1 * crop_to_final_scale_x
                            crop_y1 = ocr_y1 * crop_to_final_scale_y
                            crop_x2 = ocr_x2 * crop_to_final_scale_x
                            crop_y2 = ocr_y2 * crop_to_final_scale_y

                            # Step 2: Scale from cropped coordinates to jersey area coordinates
                            # The cropped area is the top crop_fraction of the track
                            jersey_scale_x = track_width / crop_width
                            jersey_scale_y = jersey_area_height / crop_height

                            # Scale and position within the jersey area of the track
                            final_x1 = x1 + int(crop_x1 * jersey_scale_x)
                            final_y1 = y1 + int(crop_y1 * jersey_scale_y)
                            final_x2 = x1 + int(crop_x2 * jersey_scale_x)
                            final_y2 = y1 + int(crop_y2 * jersey_scale_y)

                            # Ensure minimum box size
                            if final_x2 - final_x1 < 3:
                                final_x2 = final_x1 + 3
                            if final_y2 - final_y1 < 3:
                                final_y2 = final_y1 + 3

                            # Clamp to jersey area bounds
                            final_x1 = max(x1, min(final_x1, x2 - 3))
                            final_y1 = max(y1, min(final_y1, y1 + jersey_area_height - 3))
                            final_x2 = max(final_x1 + 3, min(final_x2, x2))
                            final_y2 = max(final_y1 + 3, min(final_y2, y1 + jersey_area_height))

                            # Color based on confidence: Red (low) -> Orange -> Green (high)
                            if conf >= 0.7:
                                bbox_color = (0, 255, 0)  # Green for high confidence
                            elif conf >= 0.4:
                                bbox_color = (0, 165, 255)  # Orange for medium confidence
                            else:
                                bbox_color = (0, 0, 255)  # Red for low confidence

                            # Draw the OCR bounding box
                            cv2.rectangle(
                                vis_frame, (final_x1, final_y1), (final_x2, final_y2), bbox_color, 2
                            )

                            # Draw detected text and confidence
                            if text and text.strip():
                                text_label = f"'{text}' ({conf:.2f})"
                                # Draw text above the box
                                cv2.putText(
                                    vis_frame,
                                    text_label,
                                    (final_x1, final_y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    bbox_color,
                                    1,
                                )

        # Draw trajectory history if available
        if track_histories and track_id in track_histories:
            history = track_histories[track_id]
            if len(history) > 1:
                # Draw trajectory line
                points = np.array(history, dtype=np.int32)
                cv2.polylines(vis_frame, [points], False, color, 2)

                # Draw trajectory points
                for point in history[-10:]:  # Show last 10 points
                    cv2.circle(vis_frame, tuple(point), 3, color, -1)

    return vis_frame


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Any],
    track_histories: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    in_place: bool = False,
) -> np.ndarray:
    """Draw tracking bounding boxes, IDs, and history trails.

    Args:
        frame: Input frame to draw on
        tracks: List of track objects
        track_histories: Optional dictionary of track histories
        in_place: Draw directly on frame instead of a copy (caller must own the frame)

    Returns:
        Frame with tracking overlays
    """
    if not tracks:
        return frame

    vis_frame = frame if in_place else frame.copy()

    for track in tracks:
        # Get track properties
        track_id = getattr(track, "track_id", None)
        if track_id is None:
            continue

        # Get bounding box
        bbox = None
        if hasattr(track, "to_ltrb"):
            bbox = track.to_ltrb()
        elif hasattr(track, "bbox"):
            bbox = track.bbox

        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Generate unique color for each track ID (better for tracking visualization)
        color = _get_track_color(track_id)

        # Draw bounding box with unique track color
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)  # Thicker line for better visibility

        # Add model type indicator in corner if available
        model_type = getattr(track, "model_type", None)
        if model_type:
            # Add colored corner marker to indicate model type
            corner_size = 15
            if model_type == "player_model":
                corner_color = VISUALIZATION_COLORS["PLAYER_MODEL"]  # Bright green
                # Top-left corner marker
                cv2.rectangle(
                    vis_frame, (x1, y1), (x1 + corner_size, y1 + corner_size), corner_color, -1
                )
                # Add "P" for player model
                cv2.putText(
                    vis_frame,
                    "P",
                    (x1 + 2, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
            elif model_type == "disc_model":
                corner_color = VISUALIZATION_COLORS["DISC_MODEL"]  # Bright orange
                # Top-right corner marker
                cv2.rectangle(
                    vis_frame, (x2 - corner_size, y1), (x2, y1 + corner_size), corner_color, -1
                )
                # Add "D" for disc model
                cv2.putText(
                    vis_frame,
                    "D",
                    (x2 - 12, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        # Draw track ID with background for better visibility
        track_label = f"ID:{track_id}"
        label_size = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # Draw label background
        cv2.rectangle(
            vis_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0] + 4, y1), color, -1
        )

        # Draw track ID text
        cv2.putText(
            vis_frame,
            track_label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # White text for contrast
            2,
        )

        # Draw class information if available
        class_info = ""
        if hasattr(track, "class_name") and track.class_name:
            class_info = track.class_name
        elif hasattr(track, "det_class") and track.det_class is not None:
            class_info = str(track.det_class)
        elif hasattr(track, "class_id") and track.class_id is not None:
            class_info = f"C{track.class_id}"

        if class_info:
            cv2.putText(
                vis_frame, class_info, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Draw track history if available (tracks are at foot level)
        if track_histories and track_id in track_histories:
            history = track_histories[track_id]
            if len(history) > 1:
                # Draw trajectory lines with decreasing opacity for older points
                for i in range(1, len(history)):
                    pt1 = history[i - 1]
                    pt2 = history[i]

                    # Calculate line thickness and opacity based on recency
                    alpha = min(1.0, (i / len(history)) + 0.3)  # Newer points more visible
                    thickness = max(1, int(3 * alpha))

                    # Draw trajectory line (foot-level tracking)
                    cv2.line(vis_frame, pt1, pt2, color, thickness)

                # Draw small circles at trajectory points (representing foot positions)
                for i, point in enumerate(history[-10:]):  # Only last 10 points
                    alpha = (i + 1) / min(10, len(history))
                    radius = max(2, int(4 * alpha))  # Slightly larger for foot positions
                    cv2.circle(vis_frame, point, radius, color, -1)

                    # Add small ground indicator for most recent position
                    if i == len(history[-10:]) - 1:  # Most recent point
                        # Draw small line below the point to indicate ground level
                        cv2.line(
                            vis_frame, (point[0] - 5, point[1]), (point[0] + 5, point[1]), color, 2
                        )

    return vis_frame


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
            print(f"[VISUALIZATION] Error drawing field segmentation: {e}")

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


def calculate_field_contour(
    unified_mask: np.ndarray, simplify_epsilon: float = None, min_contour_area: int = None
) -> Optional[np.ndarray]:
    """Calculate and simplify the contour of the field mask.

    This function now delegates to the processing module for consistent algorithm.

    Args:
        unified_mask: Binary mask (H, W) where 1 indicates field area
        simplify_epsilon: Epsilon parameter for contour simplification (as fraction of perimeter)
        min_contour_area: Minimum area threshold for contours

    Returns:
        Simplified contour points as numpy array of shape (N, 1, 2), or None if no contour found
    """
    # Import here to avoid circular imports
    from ..processing.field_analysis import calculate_field_contour_processing

    return calculate_field_contour_processing(unified_mask, simplify_epsilon, min_contour_area)


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

    # Import here to avoid circular imports
    from ..config.settings import get_setting

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

        logger.debug(f"[VISUALIZATION] Drew contour with {len(contour)} points")

    except Exception as e:
        print(f"[VISUALIZATION] Error drawing field contour: {e}")

    return result


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

    # Import here to avoid circular imports
    from ..config.settings import get_setting

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

        logger.debug(f"[VISUALIZATION] Drew {len(fitted_lines)} RANSAC-fitted field lines")

    except Exception as e:
        print(f"[VISUALIZATION] Error drawing RANSAC lines: {e}")

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
                logger.debug(f"[VISUALIZATION] Drew {outlier_count} RANSAC outlier points")

    except Exception as e:
        print(f"[VISUALIZATION] Error drawing RANSAC lines and outliers: {e}")

    return result


def create_unified_field_mask(
    segmentation_results: List[Any], frame_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Create a unified mask combining all segmentation classes into one binary mask.

    This function now delegates to the processing module for consistent algorithm.

    Args:
        segmentation_results: List of segmentation result objects
        frame_shape: (height, width) of the target frame

    Returns:
        Unified binary mask (H, W) where 1 indicates field area, or None if no results
    """
    # Import here to avoid circular imports
    from ..processing.field_analysis import create_unified_field_mask_processing

    return create_unified_field_mask_processing(segmentation_results, frame_shape)


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
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, float, bool]]]:
    """Draw a unified field mask with optional fill and contour.

    Args:
        frame: Input frame to draw on
        unified_mask: Binary mask (H, W) where 1 indicates field area
        color: BGR color tuple for the overlay
        alpha: Transparency for overlay (0.0 = transparent, 1.0 = opaque)
        draw_contour: Whether to calculate and draw simplified contours
        fill_mask: Whether to fill the mask area (False = contour only)
        ransac_fit: Precomputed fit_field_lines_ransac result for this mask. RANSAC only
            runs here when this is None, so per-frame callers should pass a cached fit.
        field_contour: Precomputed calculate_field_contour result for this mask
        in_place: Draw directly on frame instead of a copy (caller must own the frame)

    Returns:
        Tuple of (frame with unified mask overlay and optional contour, empty dictionary, all_lines_for_display dictionary)
    """
    if unified_mask is None or not np.any(unified_mask):
        return frame, {}, {}

    # Import here to avoid circular imports
    from ..config.settings import get_setting

    result = frame if in_place else frame.copy()
    classified_lines = {}  # Always empty since classification removed
    all_lines_for_display = {}  # Initialize empty all lines dictionary

    # Only fill mask if explicitly requested (disabled by default for better runtime)
    if fill_mask:
        overlay = frame.copy()
        overlay[unified_mask == 1] = color
        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, dst=result)

    # Always draw contour for field boundary visibility
    contours = _get_mask_outline(unified_mask)
    if contours:
        border_color = tuple(int(c * 0.7) for c in color)
        cv2.drawContours(result, contours, -1, border_color, 2)

    # Draw simplified contour or RANSAC lines if requested
    if draw_contour:
        # Check if RANSAC line fitting is enabled
        ransac_enabled = get_setting("models.segmentation.contour.ransac.enabled", False)

        if ransac_enabled:
            # Use RANSAC line fitting approach
            simplified_contour = field_contour
            if ransac_fit is None and simplified_contour is None:
                simplified_contour = calculate_field_contour(unified_mask)
            if ransac_fit is None and simplified_contour is not None:
                # Fit lines using RANSAC
                num_lines = get_setting("models.segmentation.contour.ransac.num_lines", 4)
                distance_threshold = get_setting(
                    "models.segmentation.contour.ransac.distance_threshold", 10.0
                )
                min_samples = get_setting("models.segmentation.contour.ransac.min_samples", 2)
                max_trials = get_setting("models.segmentation.contour.ransac.max_trials", 1000)

                # Import processing function directly
                from ..processing.field_analysis import fit_field_lines_ransac

                ransac_fit = fit_field_lines_ransac(
                    simplified_contour,
                    frame,
                    num_lines=num_lines,
                    distance_threshold=distance_threshold,
                    min_samples=min_samples,
                    max_trials=max_trials,
                )

            if ransac_fit is not None:
                (
                    fitted_lines,
                    outlier_points,
                    inlier_points,
                    edge_filtered_points,
                    classified_lines,
                    all_lines_for_display,
                ) = ransac_fit

                if fitted_lines:
                    # Draw RANSAC-fitted lines and outliers
                    line_color_list = get_setting(
                        "models.segmentation.contour.ransac.line_color", [0, 255, 0]
                    )
                    line_color = (
                        tuple(line_color_list) if isinstance(line_color_list, list) else (0, 255, 0)
                    )
                    result = draw_field_lines_ransac_with_outliers(
                        result, fitted_lines, outlier_points, line_color=line_color, in_place=True
                    )

                    # Draw edge-filtered points if enabled
                    edge_filtering_enabled = get_setting(
                        "models.segmentation.contour.ransac.edge_filtering.enabled", False
                    )
                    show_edge_points = get_setting(
                        "models.segmentation.contour.ransac.edge_filtering.show_edge_points", True
                    )
                    if (
                        edge_filtering_enabled
                        and show_edge_points
                        and len(edge_filtered_points) > 0
                    ):
                        edge_color_list = get_setting(
                            "models.segmentation.contour.ransac.edge_filtering.edge_point_color",
                            [0, 0, 255],
                        )
                        edge_color = (
                            tuple(edge_color_list)
                            if isinstance(edge_color_list, list)
                            else (0, 0, 255)
                        )
                        edge_radius = get_setting(
                            "models.segmentation.contour.ransac.edge_filtering.edge_point_radius", 2
                        )

                        for point in edge_filtered_points:
                            x, y = int(point[0]), int(point[1])
                            if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                                # Draw white border for better visibility
                                cv2.circle(result, (x, y), edge_radius + 1, (255, 255, 255), -1)
                                # Draw colored point on top
                                cv2.circle(result, (x, y), edge_radius, edge_color, -1)

                        logger.debug(
                            f"[VISUALIZATION] Drew {len(edge_filtered_points)} edge-filtered points"
                        )

                    # Draw inlier points if enabled
                    show_inliers = get_setting(
                        "models.segmentation.contour.ransac.show_inliers", True
                    )
                    if show_inliers and inlier_points and len(inlier_points) > 0:
                        inlier_color_list = get_setting(
                            "models.segmentation.contour.ransac.inlier_color", [0, 255, 0]
                        )
                        inlier_color = (
                            tuple(inlier_color_list)
                            if isinstance(inlier_color_list, list)
                            else (0, 255, 0)
                        )
                        inlier_radius = get_setting(
                            "models.segmentation.contour.ransac.inlier_radius", 2
                        )

                        # Draw inliers for each segment
                        total_inliers = 0
                        for inlier_segment in inlier_points:
                            if len(inlier_segment) > 0:
                                for point in inlier_segment:
                                    x, y = int(point[0]), int(point[1])
                                    if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                                        # Draw white border for better visibility
                                        cv2.circle(
                                            result, (x, y), inlier_radius + 1, (255, 255, 255), -1
                                        )
                                        # Draw colored point on top
                                        cv2.circle(result, (x, y), inlier_radius, inlier_color, -1)
                                        total_inliers += 1

                        logger.debug(f"[VISUALIZATION] Drew {total_inliers} inlier points")
                else:
                    print("[VISUALIZATION] RANSAC line fitting failed, falling back to contour")
                    if simplified_contour is None:
                        simplified_contour = calculate_field_contour(unified_mask)
                    result = draw_field_contour(result, simplified_contour, in_place=True)
            else:
                print("[VISUALIZATION] No contour found for RANSAC line fitting")
        else:
            # Use traditional contour approach
            simplified_contour = field_contour
            if simplified_contour is None:
                simplified_contour = calculate_field_contour(unified_mask)
            if simplified_contour is not None:
                result = draw_field_contour(result, simplified_contour, in_place=True)

    return result, classified_lines, all_lines_for_display


def draw_all_field_lines(
    frame: np.ndarray,
    all_lines_for_display: Dict[str, Tuple[np.ndarray, float, bool]],
    transformation_matrix: Optional[np.ndarray] = None,
    scale_factor: float = 1.0,
    draw_raw_lines_only: bool = False,
    in_place: bool = False,
) -> np.ndarray:
    """Draw all field lines (both classified and unclassified) with appropriate coloring based on confidence.

    Args:
        frame: Frame to draw on
        all_lines_for_display: Dictionary mapping line types to (line_coordinates, confidence, is_classified) tuples
        transformation_matrix: Optional homography matrix to transform lines to warped view
        scale_factor: Scale factor for text and line thickness (useful for top-down view)
        draw_raw_lines_only: If True, only draw simple white lines without any special colors/labels
        in_place: Draw directly on frame instead of a copy (caller must own the frame)

    Returns:
        Frame with all lines drawn
    """
    if not all_lines_for_display:
        return frame

    result = frame if in_place else frame.copy()

    # Define fallback colors for different line types (kept for compatibility)
    line_colors = {
        "left_sideline": (255, 0, 0),  # Blue
        "right_sideline": (255, 0, 255),  # Magenta
        "far_endzone_back": (0, 255, 255),  # Yellow
        "far_endzone_front": (0, 165, 255),  # Orange
        "near_endzone_front": (0, 255, 0),  # Green
        "near_endzone_back": (255, 255, 0),  # Cyan
    }

    line_thickness = max(1, int(3 * scale_factor))  # Scale line thickness

    for line_type, line_data in all_lines_for_display.items():
        line_coords, confidence, is_classified = line_data

        if line_coords is None or len(line_coords) != 2:
            continue

        # Determine color based on mode
        if draw_raw_lines_only:
            # Use a single color for all raw RANSAC lines
            color = (0, 255, 0)  # Green for all raw lines
        else:
            # Use simple coloring (no classification)
            if is_classified:
                base_color = line_colors.get(line_type, (255, 255, 255))  # Default white
            else:
                base_color = (128, 128, 128)  # Grey for unclassified

            # Further grey out lines with very low confidence (< 0.5)
            confidence_threshold = 0.5
            if confidence < confidence_threshold:
                grey_intensity = int(sum(base_color) / 3 * 0.4)  # Even darker grey
                color = (grey_intensity, grey_intensity, grey_intensity)
            else:
                color = base_color

        start_point = line_coords[0].copy()
        end_point = line_coords[1].copy()

        # Transform points if transformation matrix is provided
        if transformation_matrix is not None:
            # Convert to homogeneous coordinates
            start_homo = np.array([start_point[0], start_point[1], 1.0])
            end_homo = np.array([end_point[0], end_point[1], 1.0])

            # Apply transformation
            start_transformed = transformation_matrix @ start_homo
            end_transformed = transformation_matrix @ end_homo

            # Convert back to 2D coordinates
            if start_transformed[2] != 0:
                start_point = start_transformed[:2] / start_transformed[2]
            if end_transformed[2] != 0:
                end_point = end_transformed[:2] / end_transformed[2]

        # Draw the line
        start_int = (int(start_point[0]), int(start_point[1]))
        end_int = (int(end_point[0]), int(end_point[1]))

        # Check if points are within frame bounds
        h, w = frame.shape[:2]
        if (
            0 <= start_int[0] < w
            and 0 <= start_int[1] < h
            and 0 <= end_int[0] < w
            and 0 <= end_int[1] < h
        ):
            cv2.line(result, start_int, end_int, color, line_thickness)

            # Add text labels only for classified lines in top-down view (not raw mode)
            if (
                not draw_raw_lines_only and is_classified and scale_factor > 1.0
            ):  # Only in top-down view
                mid_point = ((start_int[0] + end_int[0]) // 2, (start_int[1] + end_int[1]) // 2)

                # Calculate offset perpendicular to the line
                line_vec = (end_int[0] - start_int[0], end_int[1] - start_int[1])
                line_length = max(
                    1, (line_vec[0] ** 2 + line_vec[1] ** 2) ** 0.5
                )  # Avoid division by zero

                # Perpendicular vector (rotate 90 degrees)
                perp_vec = (-line_vec[1], line_vec[0])

                # Normalize and scale for offset distance
                offset_distance = max(10, int(15 * scale_factor))  # Smaller offset for simpler text
                offset_x = int((perp_vec[0] / line_length) * offset_distance)
                offset_y = int((perp_vec[1] / line_length) * offset_distance)

                # Apply offset to text position
                text_pos = (mid_point[0] + offset_x, mid_point[1] + offset_y)

                # Ensure text position is within frame bounds
                text_pos = (max(10, min(w - 10, text_pos[0])), max(20, min(h - 10, text_pos[1])))

                # Create simple label - just the line type name
                clean_name = line_type.replace("_", " ").title()
                simple_label = clean_name.replace(" ", "")  # Remove spaces for compactness

                # Draw simple text with smaller font for better readability in top-down view
                font_scale = max(0.4, 0.6 * scale_factor)  # Smaller, simpler font
                font_thickness = max(1, int(1 * scale_factor))  # Thinner text
                cv2.putText(
                    result,
                    simple_label,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    font_thickness,
                )

            logger.debug(f"[VISUALIZATION] Drew {line_type} line from {start_int} to {end_int}")

    return result


def _get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a consistent, distinct color for a track ID.

    Args:
        track_id: Unique track identifier

    Returns:
        BGR color tuple
    """
    # Predefined distinct colors for better visual separation
    distinct_colors = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 165, 255),  # Orange
        (128, 0, 128),  # Purple
        (255, 20, 147),  # Deep Pink
        (0, 255, 127),  # Spring Green
        (255, 69, 0),  # Red Orange
        (30, 144, 255),  # Dodger Blue
        (255, 215, 0),  # Gold
        (50, 205, 50),  # Lime Green
        (255, 105, 180),  # Hot Pink
        (0, 206, 209),  # Dark Turquoise
        (255, 140, 0),  # Dark Orange
    ]

    # Use modulo to cycle through distinct colors
    color_index = track_id % len(distinct_colors)
    base_color = distinct_colors[color_index]

    # Add slight variation based on track_id for uniqueness when cycling
    if track_id >= len(distinct_colors):
        variation = (track_id // len(distinct_colors)) * 30
        r, g, b = base_color
        # Apply variation while keeping colors bright
        r = max(50, min(255, r + (variation % 100)))
        g = max(50, min(255, g + ((variation * 2) % 100)))
        b = max(50, min(255, b + ((variation * 3) % 100)))
        return (int(b), int(g), int(r))  # Return as BGR

    return base_color
