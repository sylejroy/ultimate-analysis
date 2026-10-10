"""Informational overlays on the main view: processing rate and jersey number table."""

import cv2
import numpy as np

from ..processing.jersey_tracker import get_jersey_tracker
from ..utils.logger import get_logger

logger = get_logger("RENDERING")


# What is written on the picture stands on a dark ground the picture shows through
LABEL_GROUND = 20
LABEL_SHOWS_THROUGH = 0.45
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_label(
    frame: np.ndarray,
    text: str,
    left: int,
    bottom: int,
    scale: float = 0.45,
    colour: tuple = (255, 255, 255),
    bar: tuple = None,
) -> None:
    """Write a small label on the frame (in place): thin text on a dark ground that the
    picture shows through, so that it hides little of what goes on behind it.

    Args:
        frame: The frame to write on
        text: What to say
        left, bottom: The label's lower left corner
        scale: Size of the text
        colour: Of the text (BGR)
        bar: A colour to mark the label with at its left edge (a track's), if any
    """
    (width, height), _ = cv2.getTextSize(text, LABEL_FONT, scale, 1)
    pad, bar_width = 3, 3 if bar is not None else 0
    x1, y1 = max(0, left), max(0, bottom - height - 2 * pad)
    x2 = min(frame.shape[1], x1 + bar_width + width + 2 * pad)
    y2 = min(frame.shape[0], y1 + height + 2 * pad)
    if x2 <= x1 or y2 <= y1:
        return
    ground = frame[y1:y2, x1:x2]
    ground[:] = cv2.addWeighted(
        ground,
        LABEL_SHOWS_THROUGH,
        np.full_like(ground, LABEL_GROUND),
        1.0 - LABEL_SHOWS_THROUGH,
        0,
    )
    if bar is not None:
        frame[y1:y2, x1 : x1 + bar_width] = bar
    cv2.putText(
        frame, text, (x1 + bar_width + pad, y2 - pad), LABEL_FONT, scale, colour, 1, cv2.LINE_AA
    )


def draw_notice(frame: np.ndarray, text: str, line: int = 0, alarm: bool = False) -> None:
    """Write a short notice at the top left of the frame (in place).

    Args:
        frame: The frame to write on
        text: What to say
        line: Which line from the top, for several notices
        alarm: On red, for something that went wrong
    """
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
    top = 12 + line * (height + 26)
    background = (30, 30, 190) if alarm else (0, 0, 0)
    cv2.rectangle(frame, (12, top), (12 + width + 20, top + height + 20), background, -1)
    cv2.putText(frame, text, (22, top + 10 + height), font, scale, (255, 255, 255), thickness)


def draw_fps_overlay(frame: np.ndarray, fps: float) -> None:
    """Draw FPS overlay on the top right of the frame.

    Args:
        frame: OpenCV frame to draw on (modified in place)
        fps: Processing rate to show; nothing is drawn until it is known
    """
    if fps <= 0:
        return

    fps_text = f"{fps:.1f} FPS"
    (text_width, _), _ = cv2.getTextSize(fps_text, LABEL_FONT, 0.5, 1)
    # In the top right corner, small: it is there to be glanced at
    draw_label(frame, fps_text, frame.shape[1] - text_width - 18, 34, 0.5, (210, 210, 210))


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
