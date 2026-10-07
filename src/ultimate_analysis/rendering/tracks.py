"""Drawing detections, tracks, trails, and jersey numbers on a frame."""

import colorsys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..constants import VISUALIZATION_COLORS
from ..processing.tracking import DISC_ID_OFFSET, team_of_player, team_shirt_colour
from ..utils.logger import get_logger

logger = get_logger("RENDERING")

POSSESSION_COLOR = VISUALIZATION_COLORS["POSSESSION"]


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
        color = get_track_color(track_id)

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
                points = np.asarray(history, dtype=np.int32)
                cv2.polylines(vis_frame, [points], False, color, 2)

                # Draw trajectory points
                for point in points[-10:].tolist():  # Show last 10 points
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
        color = get_track_color(track_id)

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
            history = [
                tuple(point)
                for point in np.asarray(track_histories[track_id], dtype=np.int32).tolist()
            ]
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


def draw_possession(frame: np.ndarray, tracks: List[Any], holder_id: Optional[int]) -> None:
    """Mark the player holding the disc with a thick box and a caption (drawn in place)."""
    for track in tracks:
        if holder_id is None or getattr(track, "track_id", None) != holder_id:
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), POSSESSION_COLOR, 3)

        label = "DISC"
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            frame, (x1 - 3, y2 + 3), (x1 + width + 7, y2 + height + 13), POSSESSION_COLOR, -1
        )
        cv2.putText(
            frame, label, (x1 + 2, y2 + height + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )


# The colours used while the shirts of the teams are not known: blue and pink, as hues in
# degrees. Apart from each other, from the grass, and from the yellow that marks who has
# the disc.
TEAM_HUES = (210.0, 330.0)
# How far a player's colour may lie from the team's, as a hue in degrees
TEAM_HUE_SPREAD = 20.0
# Teammates differ in how strong and how bright their colour is, as shares of the team's
SATURATIONS = (1.0, 0.7, 0.85)
BRIGHTNESSES = (1.0, 0.78, 0.9)
GOLDEN_STEP = 0.6180339887
# A shirt less colourful than this (white, grey, black) has no hue to vary: its players
# get faint tints of all hues instead
MIN_SHIRT_SATURATION = 0.25
TINT = (0.12, 0.32)  # Saturation of those tints, from and to
# Colours are kept at least this bright and this strong to stand out on the grass
MIN_BRIGHTNESS = 0.55
MIN_SATURATION = 0.5
# A dark shirt without colour is drawn in greys no brighter than this, a light one no darker
DARK_SHIRT = 0.5


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """The colour a player is drawn in: the colour of their team's shirts, varied a little.

    Teammates share the colour their shirts have and differ in shade and tint, so the
    teams are told apart at a glance and a player still keeps a colour of their own.
    Dark shirts are drawn brighter than they are, to show on the grass. Until the shirts
    are known a team is blue or pink; a player whose team is not known (yet) is grey, a
    disc white.

    Args:
        track_id: The player's ID (or a disc's)

    Returns:
        BGR colour
    """
    if track_id >= DISC_ID_OFFSET:
        return (255, 255, 255)
    step = track_id * GOLDEN_STEP % 1.0  # Spread evenly however many IDs there are
    weaker = SATURATIONS[track_id % len(SATURATIONS)]
    darker = BRIGHTNESSES[(track_id // len(SATURATIONS)) % len(BRIGHTNESSES)]
    team = team_of_player(track_id)
    if team is None:
        grey = int(255 * (0.55 + 0.4 * step))
        return (grey, grey, grey)

    shirt = team_shirt_colour(team)
    if shirt is None:
        hue, saturation, brightness = TEAM_HUES[team] / 360.0, 0.95, 1.0
    else:
        hue, saturation, brightness = colorsys.rgb_to_hsv(*(value / 255.0 for value in shirt[::-1]))
    if shirt is not None and saturation < MIN_SHIRT_SATURATION:
        # White, grey or black: every hue as a faint tint, light or dark as the shirt is
        hue = step
        saturation = TINT[0] + (TINT[1] - TINT[0]) * (1.0 - weaker) / (1.0 - min(SATURATIONS))
        brightness = (
            (0.35 + (DARK_SHIRT - 0.35) * darker)
            if brightness < DARK_SHIRT
            else max(MIN_BRIGHTNESS + 0.15, brightness) * darker
        )
    else:
        hue = (hue + (step - 0.5) * 2.0 * TEAM_HUE_SPREAD / 360.0) % 1.0
        saturation = max(MIN_SATURATION, saturation) * weaker
        brightness = min(1.0, max(MIN_BRIGHTNESS / min(BRIGHTNESSES), brightness)) * darker
    red, green, blue = colorsys.hsv_to_rgb(hue, min(1.0, saturation), min(1.0, brightness))
    return (int(blue * 255), int(green * 255), int(red * 255))
