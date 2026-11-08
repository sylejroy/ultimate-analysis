"""Maps over the field, drawn on the top-down view and on the camera view: where the next
throw is likely to go (processing/throw_targets.py) or the dangerous space
(processing/open_space.py). A map is one number per square of the field, rows along it
from the near back line; it is drawn as shading or as lines of equal height, and what is
opening as thin lines over it."""

from typing import Any, Callable, Dict, Optional, Sequence

import cv2
import numpy as np

# How opaque the shading is at the map's top: on the camera view less, so that the
# players under it stay to be seen
FROM_ABOVE_OPACITY = 0.8
CAMERA_OPACITY = 0.6
CAMERA_RASTER = 4  # Pixels per field unit of a map before it is laid on the frame
LEAST = 0.02  # Of the map's top: less is not drawn
LINE_LEVELS = (0.25, 0.5, 0.75)  # Of the map's top, drawn as lines
OPENING_LEVEL = 0.3  # Of the map's top: space that became this much more dangerous
OPENING_COLOUR = (255, 230, 120)  # Light blue (BGR)
GIVES_UP_SHOWN = 0.2  # A defender who gives up this share of the danger is marked
GIVES_UP_COLOUR = (60, 60, 235)  # Red (BGR)


def _colour_of(level: float) -> tuple:
    """The shading's colour at a share of the map's top."""
    pixel = cv2.applyColorMap(np.uint8([[255 * level]]), cv2.COLORMAP_INFERNO)[0, 0]
    return tuple(int(c) for c in pixel)


def _paint(part: np.ndarray, share: np.ndarray, opacity: float, lines: bool) -> None:
    """Draw a map in place on a picture of the same size, by a share from 0 (nothing) to
    1 (the map's top): shading from dark red to bright yellow, or lines of equal height."""
    if lines:
        thickness = max(1, round(min(part.shape[:2]) / 250))
        for level in LINE_LEVELS:
            # The shading's darkest colours are lost on grass: the lines start brighter
            _outline(part, share >= level, _colour_of(0.4 + 0.6 * level), thickness)
        return
    colours = cv2.applyColorMap(np.uint8(255 * np.clip(share, 0.0, 1.0)), cv2.COLORMAP_INFERNO)
    weight = (opacity * share)[..., None]
    part[:] = (part * (1.0 - weight) + colours * weight).astype(np.uint8)


def _outline(part: np.ndarray, inside: np.ndarray, colour: tuple, thickness: int) -> None:
    """Draw the border of a region in place."""
    contours, _ = cv2.findContours(np.uint8(inside), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(part, contours, -1, colour, thickness, cv2.LINE_AA)


def _shares(grid: np.ndarray, opening: Optional[np.ndarray]) -> tuple:
    """A map and what is opening in it, as shares of the map's top (None if it is flat)."""
    top = float(grid.max())
    if top <= 0:
        return None, None
    return grid / top, None if opening is None else opening / top


def shade_from_above(
    view: np.ndarray,
    grid: np.ndarray,
    in_view: Callable[[np.ndarray], np.ndarray],
    width: float,
    length: float,
    lines: bool = False,
    opening: Optional[np.ndarray] = None,
) -> None:
    """Draw a map over the field of the top-down view, in place.

    Args:
        view: The top-down view (BGR)
        grid: (rows, columns) over the field
        in_view: Places on the field (N, 2) -> pixels of the view (N, 2)
        width, length: The field's sizes in its unit
        lines: Lines of equal height instead of shading
        opening: (rows, columns): how much more the map is than a moment ago
    """
    share, rising = _shares(grid, opening)
    if share is None:
        return
    (left, top), (right, bottom) = in_view(np.array([[0.0, length], [width, 0.0]]))
    left, top = max(0, left), max(0, top)
    right, bottom = min(view.shape[1], right), min(view.shape[0], bottom)
    if right <= left or bottom <= top:
        return
    size = (right - left, bottom - top)
    part = view[top:bottom, left:right]
    # The far end of the field is at the top of the picture
    flat = cv2.resize(share[::-1].astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    _paint(part, flat, FROM_ABOVE_OPACITY, lines)
    if rising is not None:
        rising = cv2.resize(rising[::-1].astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        _outline(part, rising >= OPENING_LEVEL, OPENING_COLOUR, 2)


def shade_camera_view(
    frame: np.ndarray,
    grid: np.ndarray,
    image_to_field: np.ndarray,
    in_camera: Optional[np.ndarray],
    width: float,
    length: float,
    lines: bool = False,
    opening: Optional[np.ndarray] = None,
) -> None:
    """Draw a map over the field as the camera sees it, in place.

    Args:
        frame: The camera view (BGR)
        grid: (rows, columns) over the field
        image_to_field: 3x3, pixel of the frame -> place on the field
        in_camera: The outline on the field of what the frame shows (seen_by_camera):
            only the ground inside it is drawn on, which lies in front of the camera
        width, length: The field's sizes in its unit
        lines: Lines of equal height instead of shading
        opening: (rows, columns): how much more the map is than a moment ago
    """
    share, rising = _shares(grid, opening)
    if share is None or in_camera is None or len(in_camera) < 3:
        return
    try:
        field_to_image = np.linalg.inv(image_to_field)
    except np.linalg.LinAlgError:
        return
    size = (int(width * CAMERA_RASTER), int(length * CAMERA_RASTER))
    seen = np.zeros((size[1], size[0]), dtype=np.float32)
    cv2.fillPoly(seen, [np.rint(np.asarray(in_camera) * CAMERA_RASTER).astype(np.int32)], 1.0)
    raster_to_image = field_to_image @ np.diag([1.0 / CAMERA_RASTER, 1.0 / CAMERA_RASTER, 1.0])
    height, frame_width = frame.shape[:2]

    def on_frame(values: np.ndarray) -> np.ndarray:
        raster = cv2.resize(values.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        return cv2.warpPerspective(
            raster * seen, raster_to_image, (frame_width, height), flags=cv2.INTER_LINEAR
        )

    laid = on_frame(share)
    rows, columns = np.nonzero(laid > LEAST)
    if not len(rows):
        return
    top, bottom, left, right = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
    part = frame[top:bottom, left:right]
    _paint(part, laid[top:bottom, left:right], CAMERA_OPACITY, lines)
    if rising is not None:
        rising = on_frame(rising)[top:bottom, left:right]
        _outline(part, rising >= OPENING_LEVEL, OPENING_COLOUR, 2)


def mark_defenders(frame: np.ndarray, tracks: Sequence[Any], gives_up: Dict[int, float]) -> None:
    """Mark on the camera view, in place, the defenders who give up much of the danger:
    a red ring at their feet and their share."""
    for track in tracks:
        share = gives_up.get(track.track_id, 0.0)
        if share < GIVES_UP_SHOWN:
            continue
        x1, _, x2, y2 = (int(v) for v in track.to_ltrb())
        middle, half = ((x1 + x2) // 2, y2), max(8, (x2 - x1) // 2 + 6)
        cv2.ellipse(frame, middle, (half, half // 3), 0, 0, 360, GIVES_UP_COLOUR, 3, cv2.LINE_AA)
        text = f"{share:.0%}"
        origin = (middle[0] - 14, y2 + half // 3 + 18)
        cv2.putText(
            frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 4, cv2.LINE_AA
        )
        cv2.putText(
            frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, GIVES_UP_COLOUR, 2, cv2.LINE_AA
        )
