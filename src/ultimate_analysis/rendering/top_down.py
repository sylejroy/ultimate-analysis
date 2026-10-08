"""Drawing on the top-down (homography-warped) view."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..processing.field_analysis import calculate_field_contour, create_unified_field_mask
from ..processing.tracking import DISC_ID_OFFSET
from ..utils.logger import get_logger
from .field import draw_field_contour, draw_unified_field_mask, get_primary_field_color
from .tracks import get_track_color, possession_colour

logger = get_logger("RENDERING")


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
        logger.error(f"Error transforming contour points: {e}")
        return None


def apply_segmentation_to_warped_frame(
    warped_frame: np.ndarray,
    segmentation_results: List[Any],
    homography_matrix: np.ndarray,
    original_frame_shape: Tuple[int, int],
    tab_name: str = "MAIN_TAB",
    field_contour: Optional[np.ndarray] = None,
    draw_scale: float = 1.0,
    in_place: bool = False,
) -> np.ndarray:
    """Apply segmentation overlay to warped frame by transforming contour points from original image.

    Args:
        warped_frame: The homography-transformed frame
        segmentation_results: List of segmentation results from processing
        homography_matrix: 3x3 homography transformation matrix
        original_frame_shape: Shape of original frame (height, width)
        tab_name: Name of calling tab for logging
        field_contour: Precomputed field contour in original image coordinates. When given,
            the mask and contour are not rebuilt from segmentation_results.
        draw_scale: Scale of warped_frame relative to the full-size canvas, applied to the
            contour line and point sizes so they keep their apparent size
        in_place: Draw directly on warped_frame instead of a copy (caller must own the frame)

    Returns:
        Warped frame with segmentation overlay applied
    """
    if not segmentation_results:
        return warped_frame

    try:
        original_contour = field_contour
        if original_contour is None:
            # Create unified mask from segmentation results on original frame
            unified_mask = create_unified_field_mask(segmentation_results, original_frame_shape)

            if unified_mask is None:
                logger.debug(f"[{tab_name}] No unified mask could be created")
                return warped_frame

            logger.debug(
                f"[{tab_name}] Created unified mask with shape {unified_mask.shape}, {np.sum(unified_mask)} pixels"
            )

            # Calculate contour on the original image
            original_contour = calculate_field_contour(unified_mask)

        if original_contour is None or len(original_contour) == 0:
            logger.debug(f"[{tab_name}] No contour found in original unified mask")
            return warped_frame

        # Transform contour points using homography matrix
        transformed_contour = transform_contour_points(original_contour, homography_matrix)

        if transformed_contour is None:
            logger.debug(f"[{tab_name}] Failed to transform contour points")
            return warped_frame

        # Create mask from transformed contour points
        warped_mask = np.zeros((warped_frame.shape[0], warped_frame.shape[1]), dtype=np.uint8)
        cv2.fillPoly(warped_mask, [transformed_contour], 1)

        # Apply overlay and draw contour on warped frame - contour only for consistency
        field_color = get_primary_field_color()  # Bright cyan (BGR) - same as segmentation
        result_frame = draw_unified_field_mask(
            warped_frame,
            warped_mask,
            field_color,
            alpha=0.4,
            draw_contour=False,
            fill_mask=False,
            in_place=in_place,
        )

        # Draw the transformed contour directly; when the mask was empty the frame
        # above came back untouched, so it is only private if the caller said so.
        contour_sizes = {}
        if draw_scale != 1.0:
            contour_sizes = {
                "line_thickness": max(
                    1,
                    round(
                        get_setting("models.segmentation.contour.line_thickness", 3) * draw_scale
                    ),
                ),
                "point_radius": max(
                    1,
                    round(get_setting("models.segmentation.contour.point_radius", 5) * draw_scale),
                ),
            }
        result_frame = draw_field_contour(
            result_frame,
            transformed_contour,
            in_place=in_place or result_frame is not warped_frame,
            **contour_sizes,
        )

        logger.debug(
            f"[{tab_name}] Applied transformed contour to warped frame: {len(transformed_contour)} points"
        )
        return result_frame

    except Exception as e:
        logger.exception(f"[{tab_name}] Error applying segmentation to warped frame: {e}")
        return warped_frame


def draw_tracks_top_down(
    warped_frame: np.ndarray,
    matrix: np.ndarray,
    tracks: List[Any],
    player_ids: Dict[int, Tuple[str, Any]],
    track_histories: Dict[int, list],
    scale: float = 1.0,
    holder_id: Optional[int] = None,
    disc_position: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Map tracked objects to the top-down view using their foot positions.

    Args:
        warped_frame: The homography-transformed frame (drawn on in place)
        matrix: Homography that produced warped_frame (including any display scaling)
        tracks: Tracked objects in camera-view coordinates
        player_ids: Track ID -> (jersey number, details)
        track_histories: Track ID -> recent foot positions, for the direction arrows
        scale: Display scale of warped_frame, applied to marker and label sizes
        holder_id: Track ID of the player holding the disc, marked with a ring
        disc_position: Where in the view a flying disc is, if that has been worked out:
            taken for a spot on the ground like the players' feet, a disc in the air
            would be drawn yards behind where it is

    Returns:
        Frame with tracked objects mapped to top-down view
    """
    if not tracks or matrix is None:
        return warped_frame

    result_frame = warped_frame

    def px(value: float) -> int:
        return max(1, int(round(value * scale)))

    # The trails first, under the markers: where each player has run, seen from above
    for track in tracks:
        track_id = getattr(track, "track_id", None)
        history = (track_histories or {}).get(track_id)
        if history is None or len(history) < 2 or track_id >= DISC_ID_OFFSET:
            continue
        places = np.column_stack([np.asarray(history, dtype=np.float64), np.ones(len(history))])
        mapped = places @ np.asarray(matrix, dtype=np.float64).T
        # A point of the picture above the horizon has no place on the ground
        ahead = mapped[:, 2] * mapped[-1, 2] > 0
        if ahead.sum() < 2:
            continue
        trail = np.rint(mapped[ahead, :2] / mapped[ahead, 2:3]).astype(np.int32)
        cv2.polylines(result_frame, [trail], False, get_track_color(track_id), px(3), cv2.LINE_AA)

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

        # Calculate foot position (bottom center of bounding box)
        foot_x = (x1 + x2) / 2
        foot_y = y2  # Bottom of bounding box represents feet

        # Transform foot position using homography matrix
        foot_point = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        try:
            transformed_foot = cv2.perspectiveTransform(foot_point, matrix)
            transformed_x = int(transformed_foot[0][0][0])
            transformed_y = int(transformed_foot[0][0][1])
            if disc_position is not None and track_id >= DISC_ID_OFFSET:
                transformed_x, transformed_y = (int(round(value)) for value in disc_position)

            # Check if transformed position is within frame bounds
            frame_h, frame_w = warped_frame.shape[:2]
            if 0 <= transformed_x < frame_w and 0 <= transformed_y < frame_h:
                # Generate unique color for each track ID
                color = get_track_color(track_id)

                # Draw foot position as a circle (larger for top-down view)
                cv2.circle(result_frame, (transformed_x, transformed_y), px(12), color, -1)
                if track_id == holder_id:
                    cv2.circle(
                        result_frame,
                        (transformed_x, transformed_y),
                        px(20),
                        possession_colour(track),
                        px(4),
                    )

                # Draw track ID label with larger font for top-down view
                label_text = "disc" if track_id >= DISC_ID_OFFSET else f"ID:{track_id}"

                # Add player jersey number if available
                if track_id in player_ids:
                    jersey_number, _ = player_ids[track_id]
                    if jersey_number != "Unknown":
                        label_text = f"#{jersey_number}"

                # Draw label background for better visibility (larger for top-down view)
                font_scale = 1.0 * scale  # Sized for visibility in top-down view
                font_thickness = px(3)
                label_size = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )[0]
                label_bg_x1 = transformed_x - label_size[0] // 2 - px(5)
                label_bg_y1 = transformed_y - px(35)
                label_bg_x2 = transformed_x + label_size[0] // 2 + px(5)
                label_bg_y2 = transformed_y - px(5)

                cv2.rectangle(
                    result_frame,
                    (label_bg_x1, label_bg_y1),
                    (label_bg_x2, label_bg_y2),
                    color,
                    -1,
                )

                # Draw label text with larger font
                cv2.putText(
                    result_frame,
                    label_text,
                    (transformed_x - label_size[0] // 2, transformed_y - px(15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                )

                # Draw direction indicator if track has history
                if track_histories and track_id in track_histories:
                    history = track_histories[track_id]
                    if len(history) >= 2:
                        # Get last two foot positions and transform them
                        prev_pos = history[-2] if len(history) > 1 else history[-1]

                        # Transform previous position
                        prev_point = np.array([[[prev_pos[0], prev_pos[1]]]], dtype=np.float32)
                        try:
                            transformed_prev = cv2.perspectiveTransform(prev_point, matrix)
                            prev_x = int(transformed_prev[0][0][0])
                            prev_y = int(transformed_prev[0][0][1])

                            # Draw direction arrow
                            if 0 <= prev_x < frame_w and 0 <= prev_y < frame_h:
                                # Calculate direction vector
                                dx = transformed_x - prev_x
                                dy = transformed_y - prev_y
                                length = (dx * dx + dy * dy) ** 0.5

                                if length > 5 * scale:  # Only draw if significant movement
                                    # Normalize and scale
                                    dx = int(dx / length * 15 * scale)
                                    dy = int(dy / length * 15 * scale)

                                    # Draw arrow
                                    arrow_end_x = transformed_x + dx
                                    arrow_end_y = transformed_y + dy
                                    cv2.arrowedLine(
                                        result_frame,
                                        (transformed_x, transformed_y),
                                        (arrow_end_x, arrow_end_y),
                                        color,
                                        px(2),
                                        tipLength=0.3,
                                    )
                        except Exception:
                            pass  # Skip if transformation fails

        except Exception as e:
            logger.error(f"Error transforming track {track_id} position: {e}")
            continue

    return result_frame


def hide_behind_camera(view: np.ndarray, frame_to_view: np.ndarray) -> None:
    """Black out the part of a warped view that lies behind the camera.

    Warping fills it with a mirror image of what is in front. In place.
    """
    view_to_frame = np.linalg.inv(frame_to_view)
    height, width = view.shape[:2]
    # What the frame shows is in front: the sign of its pixels says which side that is
    in_front = np.sign(np.linalg.det(view_to_frame))
    a, b, c = view_to_frame[2] * in_front  # Behind: a*x + b*y + c <= 0, one side of a line

    def depth(point: Tuple[float, float]) -> float:
        return a * point[0] + b * point[1] + c

    # That side of the line within the view is a polygon: the view's rectangle cut along
    # the line. Filling it costs nothing next to testing every pixel.
    corners = [(0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)]
    behind = []
    for start, end in zip(corners, corners[1:] + corners[:1]):
        from_depth, to_depth = depth(start), depth(end)
        if from_depth <= 0:
            behind.append(start)
        if (from_depth <= 0) != (to_depth <= 0):
            along = from_depth / (from_depth - to_depth)
            behind.append(
                (start[0] + along * (end[0] - start[0]), start[1] + along * (end[1] - start[1]))
            )
    if len(behind) >= 3:
        # Rounded outwards by the pixel the line passes through
        polygon = np.array(behind)
        outwards = -np.sign([a, b]) * 0.5
        cv2.fillConvexPoly(view, np.int32(np.rint(polygon + outwards)), (0, 0, 0))
    elif len(behind) > 0:
        for x, y in behind:
            view[int(round(y)), int(round(x))] = 0


def draw_field_template(
    view: np.ndarray,
    lines: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    field_to_view: np.ndarray,
    colour: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw the lines of the field (in field units) on a top-down view. In place."""
    for start, end in lines:
        ends = np.array([[*start, 1.0], [*end, 1.0]]) @ field_to_view.T
        (x1, y1), (x2, y2) = np.rint(ends[:, :2] / ends[:, 2:3]).astype(int)
        cv2.line(view, (int(x1), int(y1)), (int(x2), int(y2)), colour, 1, cv2.LINE_AA)
