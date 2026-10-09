"""The field as seen by a real camera.

A mapping between field and picture has eight free numbers. A camera has fewer: where it
is (three), which way it looks (three), and its focal length, which a drone that does not
zoom keeps for a whole video. Lines at the far end of the field alone leave a free mapping
open on how far the field reaches towards the camera; a camera of known focal length they
pin down. With labelled corners two pixels off, the near goal line then lands some 20
pixels off instead of 70 to 170.

The camera is taken to have square pixels and to look through the middle of the picture.
The field is flat at height zero: x across, y along, z up, in the units of the ruleset.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from .field_template import FieldTemplate, Point, _line_through, fit_field

# A camera this high above the field, in field units, is believed; drones and poles both
MIN_HEIGHT, MAX_HEIGHT = 2.0, 40.0
# Where a camera is tried from when nothing better is known: behind the near end, above
# the near half, above the far half; distances along the field as shares of its length
START_SHARES = (-0.12, 0.1, 0.35, 0.55)
START_HEIGHT_SHARE = 0.09
START_FOCAL_SHARE = 0.8  # Of the picture's width
# How strongly a camera that is moved to follow some pixels stays as it was, where the
# pixels leave it free: small against the pixels' own pull of one
STAY = 0.001
# A mark this near its pixel has been moved onto it
MOVED_ONTO_PIXELS = 0.5


@dataclass
class CameraFit:
    """A camera placed so that the field's elements lie on their labelled pixels."""

    field_to_image: np.ndarray  # 3x3; places in front of the camera get a positive third number
    image_to_field: np.ndarray  # 3x3
    focal: float  # In pixels
    position: np.ndarray  # x, y, z of the camera
    error: float  # Root mean square distance of the pixels from their elements, in pixels


def camera_mapping(
    focal: float, rotation: Sequence[float], position: Sequence[float], size: Tuple[int, int]
) -> np.ndarray:
    """Field place -> pixel for a camera.

    Args:
        focal: Focal length in pixels
        rotation: Rotation vector (Rodrigues) from field to camera axes
        position: Where the camera is
        size: (width, height) of the picture
    """
    turn = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0]
    inner = np.array([[focal, 0.0, size[0] / 2.0], [0.0, focal, size[1] / 2.0], [0.0, 0.0, 1.0]])
    return inner @ np.column_stack([turn[:, 0], turn[:, 1], -turn @ np.asarray(position)])


def focal_of(field_to_image: np.ndarray, size: Tuple[int, int]) -> Optional[float]:
    """The focal length a mapping implies, or None if it is not the view of such a camera.

    From the two directions of the field being equally long seen from the camera, which
    holds up better with imprecise lines than their being at right angles.
    """
    centred = (
        np.array([[1.0, 0.0, -size[0] / 2.0], [0.0, 1.0, -size[1] / 2.0], [0.0, 0.0, 1.0]])
        @ field_to_image
    )
    a, b = centred[:, 0], centred[:, 1]
    below = a[2] ** 2 - b[2] ** 2
    if abs(below) < 1e-18:
        return None
    squared = (b[0] ** 2 + b[1] ** 2 - a[0] ** 2 - a[1] ** 2) / below
    return float(np.sqrt(squared)) if squared > 0 else None


def camera_of(
    field_to_image: np.ndarray, size: Tuple[int, int], focal: Optional[float] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """(focal, rotation vector, position) of the camera nearest to a mapping."""
    if focal is None:
        focal = focal_of(field_to_image, size) or START_FOCAL_SHARE * size[0]
    centred = (
        np.diag([1.0 / focal, 1.0 / focal, 1.0])
        @ np.array([[1.0, 0.0, -size[0] / 2.0], [0.0, 1.0, -size[1] / 2.0], [0.0, 0.0, 1.0]])
        @ field_to_image
    )
    scale = 2.0 / (np.linalg.norm(centred[:, 0]) + np.linalg.norm(centred[:, 1]))
    first, second, shift = centred[:, 0] * scale, centred[:, 1] * scale, centred[:, 2] * scale
    u, _, vt = np.linalg.svd(np.column_stack([first, second, np.cross(first, second)]))
    turn = u @ vt
    return focal, cv2.Rodrigues(turn)[0].ravel(), -turn.T @ shift


# How often a fit may ask for the misfits, and how exactly it is to be solved
MAX_FIT_EVALUATIONS = 400
FIT_TOLERANCE = 1e-4


def _misfit_function(
    template: FieldTemplate, lines: Dict[str, Sequence[Point]], points: Dict[str, Point]
) -> Callable[[np.ndarray], np.ndarray]:
    """A function: field-to-image mapping -> distance in pixels of each labelled pixel from
    where the mapping puts its element."""
    of_lines = np.array(
        [_line_through(*template.lines[name]) for name, pixels in lines.items() for _ in pixels]
    ).reshape(-1, 3)
    on_lines = np.array(
        [[u, v, 1.0] for pixels in lines.values() for u, v in pixels], dtype=np.float64
    ).reshape(-1, 3)
    places = np.array([[*template.points[name], 1.0] for name in points], dtype=np.float64).reshape(
        -1, 3
    )
    marks = np.array(list(points.values()), dtype=np.float64).reshape(-1, 2)

    def misfits(field_to_image: np.ndarray) -> np.ndarray:
        # A line of the field as a line of the picture: l' = H^-T l
        in_image = of_lines @ np.linalg.inv(field_to_image)
        in_image = in_image / np.maximum(np.hypot(in_image[:, 0], in_image[:, 1]), 1e-12)[:, None]
        from_lines = np.sum(in_image * on_lines, axis=1)
        mapped = places @ field_to_image.T
        depth = np.where(np.abs(mapped[:, 2:3]) < 1e-12, 1e-12, mapped[:, 2:3])
        return np.concatenate([from_lines, (mapped[:, :2] / depth - marks).ravel()])

    return misfits


def _focal(logarithm: float) -> float:
    """The focal length from the number the fit works with, kept within what a lens can be."""
    return float(np.exp(np.clip(logarithm, 4.0, 11.0)))


def _starts(
    template: FieldTemplate, size: Tuple[int, int], focal: Optional[float]
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Cameras to try from: looking down the field from above its middle line."""
    focal = focal or START_FOCAL_SHARE * size[0]
    found = []
    for share in START_SHARES:
        position = np.array(
            [template.width / 2.0, share * template.length, START_HEIGHT_SHARE * template.length]
        )
        # Tilted down so that the far back line is a quarter of the way down the picture
        to_far = np.arctan2(position[2], template.length - position[1])
        pitch = to_far + np.arctan2(0.25 * size[1], focal)
        turn = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -np.sin(pitch), -np.cos(pitch)],
                [0.0, np.cos(pitch), -np.sin(pitch)],
            ]
        )
        found.append((focal, cv2.Rodrigues(turn)[0].ravel(), position))
    return found


def fit_camera(
    template: FieldTemplate,
    lines: Dict[str, Sequence[Point]],
    points: Dict[str, Point],
    size: Tuple[int, int],
    focal: Optional[float] = None,
) -> Optional[CameraFit]:
    """Place a camera so that the labelled elements of the field lie on their pixels.

    Args:
        template: The field
        lines: Line name -> two (or more) pixels on that line
        points: Mark name -> its pixel
        size: (width, height) of the picture
        focal: The camera's focal length in pixels if known; found as well otherwise

    Returns:
        The best camera, or None if the elements do not fix one or none is believable
    """
    lines = {name: pixels for name, pixels in lines.items() if name in template.lines}
    points = {name: pixel for name, pixel in points.items() if name in template.points}
    statements = sum(len(pixels) for pixels in lines.values()) + 2 * len(points)
    free = 6 if focal is not None else 7
    if statements < free:
        return None

    starts = _starts(template, size, focal)
    free_fit = fit_field(template, lines, points)
    if free_fit is not None:
        starts.insert(0, camera_of(free_fit.field_to_image * free_fit.front_sign, size, focal))
    # Starting from the camera of the frame before instead was tried: the search takes
    # as many steps from there as from the free mapping's camera, and ends in the same place

    def mapping_of(values: np.ndarray) -> np.ndarray:
        if focal is not None:
            return camera_mapping(focal, values[0:3], values[3:6], size)
        return camera_mapping(_focal(values[0]), values[1:4], values[4:7], size)

    misfits = _misfit_function(template, lines, points)

    best: Optional[CameraFit] = None
    for start_focal, rotation, position in starts:
        first = (
            [*rotation, *position]
            if focal is not None
            else [np.log(start_focal), *rotation, *position]
        )
        try:
            # Lines that fix no camera send the search round in circles: left to itself it
            # asks for the misfits up to 5,600 times. With these two limits every estimate
            # of 480 frames of six clips is still found, in the same place to a hundredth
            # of a yard, in 15 ms instead of 37 (docs/MEASUREMENTS.md). Tighter limits
            # lose estimates: without a known focal length some good fits come slowly.
            solved = least_squares(
                lambda values: misfits(mapping_of(values)),
                first,
                method="lm",
                max_nfev=MAX_FIT_EVALUATIONS,
                ftol=FIT_TOLERANCE,
                xtol=FIT_TOLERANCE,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        position = solved.x[-3:]
        mapping = mapping_of(solved.x)
        if not MIN_HEIGHT <= position[2] <= MAX_HEIGHT or abs(np.linalg.det(mapping)) < 1e-12:
            continue
        # The labelled pixels are seen, so their places must be in front of the camera
        seen = [pixel for pixels in lines.values() for pixel in pixels] + list(points.values())
        places = np.column_stack([seen, np.ones(len(seen))]) @ np.linalg.inv(mapping).T
        if np.any((places / places[:, 2:3]) @ mapping[2] <= 0):
            continue
        error = float(np.sqrt(np.mean(solved.fun**2)))
        if best is None or error < best.error:
            best = CameraFit(
                field_to_image=mapping,
                image_to_field=np.linalg.inv(mapping),
                focal=focal if focal is not None else _focal(solved.x[0]),
                position=position.copy(),
                error=error,
            )
        if best is not None:
            # The first start is the free mapping's camera where there is one, which is
            # nearly there already; the others are for when it leads nowhere
            break
    return best


def move_camera(
    template: FieldTemplate,
    field_to_image: np.ndarray,
    points: Dict[str, Point],
    size: Tuple[int, int],
    focal: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Move the camera of a view so that some marks of the field come to lie on given pixels.

    Fewer than three marks leave a camera free; it then changes as little as it can, so
    one mark dragged takes the whole field along and a second one turns and stretches it.

    Args:
        template: The field
        field_to_image: The view as it is
        points: Mark name -> the pixel it shall lie on
        size: (width, height) of the picture
        focal: The camera's focal length in pixels; that of the view as it is if not given

    Returns:
        The new mapping from field to picture, or None if no camera does it
    """
    points = {name: pixel for name, pixel in points.items() if name in template.points}
    if not points:
        return None
    focal, rotation, position = camera_of(field_to_image, size, focal)
    before = np.array([*rotation, *position])
    # A step of a pixel in the picture: this much turn, this much way
    weights = STAY * np.array([focal] * 3 + [focal / (0.3 * template.length)] * 3)
    misfits = _misfit_function(template, {}, points)

    def residuals(values: np.ndarray) -> np.ndarray:
        mapping = camera_mapping(focal, values[:3], values[3:], size)
        return np.concatenate([misfits(mapping), weights * (values - before)])

    # From where the camera is; if that leaves the marks off their pixels (the fit can get
    # stuck), from other places too
    places = np.array([[*template.points[name], 1.0] for name in points])
    best: Optional[Tuple[float, np.ndarray]] = None
    for start in [before, *([*turn, *place] for _, turn, place in _starts(template, size, focal))]:
        try:
            solved = least_squares(residuals, start, method="lm", ftol=1e-12, xtol=1e-12)
        except (ValueError, np.linalg.LinAlgError):
            continue
        mapping = camera_mapping(focal, solved.x[:3], solved.x[3:], size)
        if (
            not MIN_HEIGHT <= solved.x[5] <= MAX_HEIGHT
            or abs(np.linalg.det(mapping)) < 1e-12
            or np.any(places @ mapping[2] <= 0)
        ):
            continue
        off = float(np.abs(misfits(mapping)).max())
        if best is None or off < best[0]:
            best = (off, mapping)
        if best[0] < MOVED_ONTO_PIXELS:
            break
    return best[1] if best is not None else None
