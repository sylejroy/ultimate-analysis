"""Informational overlays on the main view: processing rate and jersey number table."""

import cv2
import numpy as np

from ..processing.jersey_tracker import get_jersey_tracker
from ..utils.logger import get_logger

logger = get_logger("RENDERING")


def draw_fps_overlay(frame: np.ndarray, fps: float) -> None:
    """Draw FPS overlay on the top right of the frame.

    Args:
        frame: OpenCV frame to draw on (modified in place)
        fps: Processing rate to show; nothing is drawn until it is known
    """
    if fps <= 0:
        return

    # Format FPS text
    fps_text = f"Processing: {fps:.1f} FPS"

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)  # Green color
    thickness = 2

    # Get text size for positioning
    (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)

    # Position in top right with some padding, moved down slightly
    x = width - text_width - 15
    y = text_height + 45  # Increased from 15 to 45 to lower the position

    # Draw background rectangle for better visibility
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )  # Black background

    # Draw the FPS text
    cv2.putText(frame, fps_text, (x, y), font, font_scale, color, thickness)


def draw_jersey_table(frame: np.ndarray) -> None:
    """Draw the jersey numbers read so far, per track, as a table on the frame (in place)."""
    try:
        tracker = get_jersey_tracker()

        # Get best and second-best jersey numbers for each track
        tracked_data = []
        for track_id in tracker.tracked_ids():
            top_probs = tracker.get_top_probabilities(track_id, top_k=2)
            if top_probs:
                best_jersey, best_prob = top_probs[0][0], top_probs[0][1]
                second_jersey, second_prob = None, 0.0
                if len(top_probs) > 1:
                    second_jersey, second_prob = top_probs[1][0], top_probs[1][1]

                tracked_data.append(
                    {
                        "track_id": track_id,
                        "best_jersey": best_jersey,
                        "best_prob": best_prob,
                        "second_jersey": second_jersey,
                        "second_prob": second_prob,
                    }
                )

        if not tracked_data:
            return

        # Sort by track ID
        tracked_data.sort(key=lambda x: x["track_id"])

        # Overlay position (top-left corner with margin, lowered slightly)
        start_x = 20
        start_y = 80  # Lowered from 30 to 80
        line_height = 30  # Increased to accommodate two lines per track

        # Draw semi-transparent background
        table_height = len(tracked_data) * line_height + 40
        table_width = 250  # Increased width for second jersey

        # Darken only the table area; blending a copy of the whole frame costs far more
        background = frame[
            start_y - 20 : start_y + table_height - 19, start_x - 10 : start_x + table_width + 1
        ]
        cv2.convertScaleAbs(background, dst=background, alpha=0.7)

        # Draw header
        cv2.putText(
            frame,
            "Jersey Tracking",
            (start_x, start_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Draw table entries
        y_offset = start_y + 25
        for data in tracked_data:
            # Determine color based on best confidence
            if data["best_prob"] >= 0.7:
                color = (0, 255, 0)  # Green
            elif data["best_prob"] >= 0.4:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red

            # Draw best jersey number (primary line)
            best_text = (
                f"Track {data['track_id']}: #{data['best_jersey']} ({data['best_prob']:.2f})"
            )
            cv2.putText(
                frame, best_text, (start_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

            # Draw second jersey number (secondary line) if available
            if (
                data["second_jersey"] and data["second_prob"] > 0.1
            ):  # Only show if reasonable confidence
                second_color = (128, 128, 128)  # Gray for secondary
                second_text = f"   Alt: #{data['second_jersey']} ({data['second_prob']:.2f})"
                cv2.putText(
                    frame,
                    second_text,
                    (start_x, y_offset + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    second_color,
                    1,
                )

            y_offset += line_height

    except Exception as e:
        logger.error(f"Error drawing jersey overlay: {e}")
