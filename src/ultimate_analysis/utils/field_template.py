"""The Ultimate field as a drawing of known size, and how a picture maps onto it.

Field coordinates are in the units of the ruleset (yards or metres): x runs across the
field from the left sideline, y along it from the near back line, both as the camera
sees them. "Near" and "far", "left" and "right" are therefore the camera's, not the
teams'; the field is symmetric, so any view can be labelled this way.

A picture is tied to the field by labelled elements:

- a line, by any two points on it (its ends are usually outside the picture)
- a point, such as a brick mark or a corner cone

Every labelled point on a line says "this pixel lies on that line of the field", every
labelled point "this pixel is that spot of the field". Eight such statements fix the
mapping (a homography), for example four lines; more make it more exact.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass(frozen=True)
class FieldTemplate:
    """Sizes of a field. End zones are at both ends of the length."""

    name: str
    unit: str
    width: float
    length: float  # Back line to back line
    end_zone: float  # Depth of each end zone
    brick: float  # Distance of a brick mark from its goal line

    @property
    def lines(self) -> Dict[str, Tuple[Point, Point]]:
        """Name -> the two ends of each line of the field."""
        w, far_back = self.width, self.length
        near_goal, far_goal = self.end_zone, self.length - self.end_zone
        return {
            "left_sideline": ((0.0, 0.0), (0.0, far_back)),
            "right_sideline": ((w, 0.0), (w, far_back)),
            "far_back_line": ((0.0, far_back), (w, far_back)),
            "far_goal_line": ((0.0, far_goal), (w, far_goal)),
            "near_goal_line": ((0.0, near_goal), (w, near_goal)),
            "near_back_line": ((0.0, 0.0), (w, 0.0)),
        }

    @property
    def points(self) -> Dict[str, Point]:
        """Name -> place of each mark of the field: brick marks, midfield, the corners."""
        w, far_back = self.width, self.length
        near_goal, far_goal = self.end_zone, self.length - self.end_zone
        return {
            "far_brick": (w / 2, far_goal - self.brick),
            "midfield": (w / 2, self.length / 2),
            "near_brick": (w / 2, near_goal + self.brick),
            "far_back_left": (0.0, far_back),
            "far_back_right": (w, far_back),
            "far_goal_left": (0.0, far_goal),
            "far_goal_right": (w, far_goal),
            "near_goal_left": (0.0, near_goal),
            "near_goal_right": (w, near_goal),
            "near_back_left": (0.0, 0.0),
            "near_back_right": (w, 0.0),
        }


TEMPLATES: Dict[str, FieldTemplate] = {
    # USA Ultimate: 110 x 40 yards, end zones 20 yards, brick marks 20 yards from the goal lines
    "usau": FieldTemplate("usau", "yd", 40.0, 110.0, 20.0, 20.0),
    # WFDF: 100 x 37 metres, end zones 18 metres, brick marks 18 metres from the goal lines
    "wfdf": FieldTemplate("wfdf", "m", 37.0, 100.0, 18.0, 18.0),
}
DEFAULT_RULESET = "usau"
MIN_STATEMENTS = 8  # What a homography needs


@dataclass
class FieldFit:
    """A picture mapped onto the field."""

    image_to_field: np.ndarray  # 3x3
    field_to_image: np.ndarray  # 3x3
    statements: int  # How many "this pixel lies on / is ..." went in
    error: (
        float  # Root mean square distance of the labelled pixels from their elements, field units
    )
    worst: Tuple[str, float]  # The element that fits worst, and by how much
    # Sign of the third number of field_to_image @ (x, y, 1) for places in front of the camera
    front_sign: float = 1.0


def _line_through(first: Point, second: Point) -> np.ndarray:
    """Line a*x + b*y + c = 0 through two points, with a^2 + b^2 = 1."""
    line = np.cross([first[0], first[1], 1.0], [second[0], second[1], 1.0])
    return line / np.hypot(line[0], line[1])


def fit_field(
    template: FieldTemplate,
    lines: Dict[str, Sequence[Point]],
    points: Dict[str, Point],
) -> Optional[FieldFit]:
    """The mapping from picture to field that the labelled elements give.

    Args:
        template: The field
        lines: Line name -> two (or more) pixels on that line
        points: Mark name -> its pixel

    Returns:
        The fit, or None if the elements do not fix a mapping (too few, or all saying the
        same, like lines that are all parallel)
    """
    # Both sides are brought to numbers around one, which keeps the solution exact
    image_scale = 1.0 / max(
        1.0,
        max(
            (abs(value) for pixels in lines.values() for pixel in pixels for value in pixel),
            default=1.0,
        ),
        max((abs(value) for pixel in points.values() for value in pixel), default=1.0),
    )
    field_scale = 1.0 / template.length
    from_small_field = np.diag([1.0 / field_scale, 1.0 / field_scale, 1.0])
    to_small_image = np.diag([image_scale, image_scale, 1.0])

    rows: List[np.ndarray] = []
    for name, pixels in lines.items():
        if name not in template.lines:
            continue
        # A line of the field in the small field: l' = T^-T l
        line = from_small_field.T @ _line_through(*template.lines[name])
        for u, v in pixels:
            pixel = np.array([u * image_scale, v * image_scale, 1.0])
            rows.append(np.kron(line, pixel))  # line . (H pixel) = 0
    for name, (u, v) in points.items():
        if name not in template.points:
            continue
        pixel = np.array([u * image_scale, v * image_scale, 1.0])
        x, y = np.array(template.points[name]) * field_scale
        zero = np.zeros(3)
        rows.append(np.concatenate([zero, -pixel, y * pixel]))
        rows.append(np.concatenate([pixel, zero, -x * pixel]))

    if len(rows) < MIN_STATEMENTS:
        return None
    system = np.array(rows)
    _, singular, vt = np.linalg.svd(system)
    # One direction must be free (the scale of H) and no second one
    if singular[-2] < 1e-9 * singular[0]:
        return None
    small = vt[-1].reshape(3, 3)
    image_to_field = from_small_field @ small @ to_small_image
    if abs(np.linalg.det(image_to_field)) < 1e-12:
        return None
    if abs(image_to_field[2, 2]) > 1e-12:
        image_to_field = image_to_field / image_to_field[2, 2]

    # The solution above makes an algebraic misfit small, which weighs the elements
    # unevenly. Refined here to make the distances in the picture small, which is what
    # the person labelling sees and what tells which element is off.
    def misfits(image_to_field: np.ndarray) -> List[Tuple[str, float]]:
        """(element, distance in pixels of each labelled pixel from where the element is drawn)."""
        field_to_image = np.linalg.inv(image_to_field)
        found: List[Tuple[str, float]] = []
        for name, pixels in lines.items():
            if name not in template.lines:
                continue
            # The field's line as a line of the picture
            line = image_to_field.T @ _line_through(*template.lines[name])
            line = line / max(np.hypot(line[0], line[1]), 1e-12)
            found += [(name, float(line @ [u, v, 1.0])) for u, v in pixels]
        for name, (u, v) in points.items():
            if name not in template.points:
                continue
            mapped = field_to_image @ [*template.points[name], 1.0]
            if abs(mapped[2]) < 1e-12:
                found += [(name, 1e6), (name, 1e6)]
                continue
            found += [
                (name, float(mapped[0] / mapped[2] - u)),
                (name, float(mapped[1] / mapped[2] - v)),
            ]
        return found

    def unpack(values: np.ndarray) -> np.ndarray:
        return np.append(values, 1.0).reshape(3, 3)

    if abs(image_to_field[2, 2]) > 1e-12 and len(rows) > MIN_STATEMENTS:
        from scipy.optimize import least_squares

        try:
            refined = least_squares(
                lambda values: [distance for _, distance in misfits(unpack(values))],
                image_to_field.ravel()[:8],
                x_scale="jac",
                max_nfev=50,
            )
            candidate = unpack(refined.x)
            if abs(np.linalg.det(candidate)) > 1e-12:
                image_to_field = candidate
        except (ValueError, np.linalg.LinAlgError):
            pass  # The first solution stands

    distances = misfits(image_to_field)
    values = np.array([distance for _, distance in distances])
    field_to_image = np.linalg.inv(image_to_field)
    # The labelled pixels are in the picture, so their places on the field are in front
    labelled = [pixel for pixels in lines.values() for pixel in pixels] + list(points.values())
    places = to_field(image_to_field, labelled)
    if places is None:
        return None
    depth = np.column_stack([places, np.ones(len(places))]) @ field_to_image[2]
    worst = max(distances, key=lambda item: abs(item[1]))
    return FieldFit(
        front_sign=1.0 if np.median(depth) >= 0 else -1.0,
        image_to_field=image_to_field,
        field_to_image=field_to_image,
        statements=len(rows),
        error=float(np.sqrt(np.mean(values**2))),
        worst=(worst[0], abs(worst[1])),
    )


def to_field(image_to_field: np.ndarray, pixels: Sequence[Point]) -> Optional[np.ndarray]:
    """Pixels -> field positions (N, 2); None if one of them maps to no place on the ground."""
    homogeneous = np.column_stack([np.asarray(pixels, dtype=np.float64), np.ones(len(pixels))])
    mapped = homogeneous @ image_to_field.T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        return None
    return mapped[:, :2] / mapped[:, 2:3]


def field_segment_in_image(
    fit: FieldFit, start: Point, end: Point, pieces: int = 64
) -> List[np.ndarray]:
    """A stretch of the field as it appears in the picture: polylines of pixels.

    A line that runs past the camera has a part behind it, which has no place in the
    picture; the stretch is then cut there, into the parts in front.
    """
    steps = np.linspace(0.0, 1.0, pieces + 1)[:, None]
    along = np.asarray(start) + steps * (np.asarray(end) - np.asarray(start))
    mapped = np.column_stack([along, np.ones(len(along))]) @ fit.field_to_image.T
    front = mapped[:, 2] * fit.front_sign > 1e-9
    polylines: List[np.ndarray] = []
    current: List[np.ndarray] = []
    for visible, row in zip(front, mapped):
        if visible:
            current.append(row[:2] / row[2])
        elif current:
            polylines.append(np.array(current))
            current = []
    if current:
        polylines.append(np.array(current))
    return [line for line in polylines if len(line) >= 2]
