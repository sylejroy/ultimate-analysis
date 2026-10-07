"""Where the field lies in a frame, estimated from what the field model sees.

The field model marks the central field and the end zones. Their outline gives the two
sidelines and the far back line; where an end zone meets the central field is a goal line.
A camera is placed so that the field's lines lie on these (utils/field_camera.py).

The estimate is as good as the masks: a few pixels at the far end, where all the lines are,
and less certain towards the camera when no near goal line is in view. Knowing the focal
length of the video's camera makes up for much of that.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils.field_camera import CameraFit, fit_camera
from ..utils.field_template import FieldTemplate
from ..utils.logger import get_logger
from .field_analysis import create_unified_field_mask, fit_lines_from_mask

logger = get_logger("FIELD_REGISTRATION")

Line = np.ndarray  # Two pixels, shape (2, 2)

CENTRAL_FIELD, END_ZONE = 0, 1  # Classes of the field model
# A line counts as running across the field if it is this flat in the picture
ACROSS_MAX_DEGREES = 20.0
# Rows between the two pixels compared to find where one area ends and the next begins,
# in pixels of the masks
BOUNDARY_STEP = 3
# A goal line must be found over at least this share of the picture's width
MIN_GOAL_LINE_SHARE = 0.12
# Pixels of a boundary count towards a line up to this far from it, in mask pixels
GOAL_LINE_REACH = 3.0
# Lines that agree on one camera this well, in pixels, are all taken as found
GOOD_MISFIT_PIXELS = 4.0
# Lines that agree worse than this give no estimate
MAX_MISFIT_PIXELS = 8.0
# A video's focal length is taken as known once this many frames have given one. After
# that only every so many frames are asked, up to the last number: frames one after
# the other show the same view and say the same
MIN_FOCAL_SAMPLES = 15
FOCAL_SAMPLE_EVERY = 6
MAX_FOCAL_SAMPLES = 300
# How much of a new estimate goes into the field as followed; the rest is where the field
# was, moved with the camera. Evens out the estimates' jitter.
NEW_ESTIMATE_WEIGHT = 0.4
# An estimate further than this from where the field was followed to, in field units,
# replaces it outright: a cut, or the following has drifted
JUMP_DISTANCE = 8.0
# The top-down view shows the field with this much room around it, as a share of the view
CANVAS_MARGIN = 0.06


@dataclass
class FieldEstimate:
    """The field as estimated in one frame."""

    field_to_image: np.ndarray  # 3x3; places in front of the camera get a positive third number
    image_to_field: np.ndarray  # 3x3
    lines: Dict[str, Line]  # The lines it was made from, in frame pixels
    misfit: float  # How well the lines agree on one camera, in pixels
    focal: float  # Focal length of the camera, in pixels
    position: np.ndarray  # Where the camera is: x, y, z in field units


class FocalLength:
    """The focal length of a video's camera, learned from the frames seen so far.

    A camera that does not zoom keeps its focal length, and each frame's lines give it
    only roughly; the middle of many frames' values is steady.
    """

    def __init__(self) -> None:
        self._samples: List[float] = []
        self._asked = 0

    def add(self, focal: float) -> None:
        self._samples.append(float(focal))

    def wants_one(self) -> bool:
        """Whether the current frame should give a focal length."""
        self._asked += 1
        if len(self._samples) < MIN_FOCAL_SAMPLES:
            return True
        return len(self._samples) < MAX_FOCAL_SAMPLES and self._asked % FOCAL_SAMPLE_EVERY == 0

    @property
    def value(self) -> Optional[float]:
        """The focal length in pixels, or None while too few frames have given one."""
        if len(self._samples) < MIN_FOCAL_SAMPLES:
            return None
        return float(np.median(self._samples))


class FieldFollower:
    """Where the field lies from frame to frame.

    Estimated whenever the field model has run, moved with the camera in between, and
    evened out over the estimates. The focal length of the video's camera is learned on
    the way and then held.
    """

    def __init__(self, template: FieldTemplate):
        self.template = template
        self.image_to_field: Optional[np.ndarray] = None  # None until the field was found
        self._focal = FocalLength()

    def new_video(self) -> None:
        """Another camera: forget its focal length too."""
        self._focal = FocalLength()
        self.reset()

    def reset(self) -> None:
        """Forget where the field was (after a seek or a cut)."""
        self.image_to_field = None

    def move(self, motion: np.ndarray) -> None:
        """Follow the camera: `motion` takes pixels of the frame before to this frame."""
        if self.image_to_field is not None:
            self.image_to_field = self.image_to_field @ np.linalg.inv(motion)

    def update(
        self,
        segmentation_results: List[Any],
        frame_shape: Tuple[int, int],
        outline_lines: Optional[Sequence[Line]] = None,
    ) -> None:
        """Take in what the field model saw in the current frame."""
        lines = found_lines(segmentation_results, frame_shape, outline_lines)
        if self._focal.wants_one():
            free = estimate_field(segmentation_results, frame_shape, self.template, lines=lines)
            if free is not None and free.misfit <= GOOD_MISFIT_PIXELS:
                self._focal.add(free.focal)
        estimate = estimate_field(
            segmentation_results, frame_shape, self.template, self._focal.value, lines=lines
        )
        if estimate is None:
            return
        self.image_to_field = self._evened_out(estimate.image_to_field, frame_shape)

    def _evened_out(self, estimated: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """A new estimate blended with where the field was followed to."""
        if self.image_to_field is None:
            return estimated
        height, width = frame_shape
        # Four pixels in the lower part of the frame, where the field is
        pixels = np.array(
            [
                [0.2 * width, 0.45 * height],
                [0.8 * width, 0.45 * height],
                [0.8 * width, 0.95 * height],
                [0.2 * width, 0.95 * height],
            ]
        )
        ones = np.column_stack([pixels, np.ones(4)])
        followed, new = ones @ self.image_to_field.T, ones @ estimated.T
        if np.any(np.abs(followed[:, 2]) < 1e-12) or np.any(np.abs(new[:, 2]) < 1e-12):
            return estimated
        followed, new = followed[:, :2] / followed[:, 2:3], new[:, :2] / new[:, 2:3]
        if np.linalg.norm(followed - new, axis=1).max() > JUMP_DISTANCE:
            return estimated
        places = (1.0 - NEW_ESTIMATE_WEIGHT) * followed + NEW_ESTIMATE_WEIGHT * new
        return cv2.getPerspectiveTransform(np.float32(pixels), np.float32(places)).astype(
            np.float64
        )


def field_to_canvas(template: FieldTemplate, canvas_size: Tuple[int, int]) -> np.ndarray:
    """Field place -> pixel of a top-down view (width, height) with the far end at the top."""
    width, height = canvas_size
    scale = (1.0 - 2.0 * CANVAS_MARGIN) * min(width / template.width, height / template.length)
    left = (width - scale * template.width) / 2.0
    top = (height - scale * template.length) / 2.0
    return np.array(
        [[scale, 0.0, left], [0.0, -scale, top + scale * template.length], [0.0, 0.0, 1.0]]
    )


def estimate_field(
    segmentation_results: List[Any],
    frame_shape: Tuple[int, int],
    template: FieldTemplate,
    focal: Optional[float] = None,
    outline_lines: Optional[Sequence[Line]] = None,
    lines: Optional[Tuple[Dict[str, Line], List[Line]]] = None,
) -> Optional[FieldEstimate]:
    """Estimate where the field lies from the field model's results for a frame.

    Args:
        segmentation_results: What run_field_segmentation returned for the frame
        frame_shape: (height, width) of the frame
        template: The field
        focal: Focal length of the camera in pixels, if known for this video
        outline_lines: Lines already fitted to the field's outline, if the caller has them
        lines: What found_lines returned for the frame, if the caller has that

    Returns:
        The estimate, or None if the masks do not show enough of the field
    """
    lines, unsure = lines or found_lines(segmentation_results, frame_shape, outline_lines)
    if not {"left_sideline", "right_sideline", "far_back_line"} <= set(lines):
        return None
    size = (frame_shape[1], frame_shape[0])

    def camera_for(tried: Dict[str, Line]) -> Optional[CameraFit]:
        if len(tried) < 4:
            return None
        if (
            "near_goal_line" in tried
            and "far_goal_line" in tried
            and tried["near_goal_line"][:, 1].mean() <= tried["far_goal_line"][:, 1].mean()
        ):
            return None
        by_name = {name: [tuple(pixel) for pixel in line] for name, line in tried.items()}
        return fit_camera(template, by_name, {}, size, focal)

    # A goal line the classes do not name is tried as either; the camera decides
    best: Optional[Tuple[CameraFit, Dict[str, Line]]] = None
    for names in product(("far_goal_line", "near_goal_line"), repeat=len(unsure)):
        tried = dict(lines)
        for name, line in zip(names, unsure):
            tried.setdefault(name, line)
        fit = camera_for(tried)
        if fit is not None and (best is None or fit.error < best[0].error):
            best = (fit, tried)
    if best is not None and best[0].error > GOOD_MISFIT_PIXELS:
        # The lines do not agree on one camera: one of the goal lines may be none (an end
        # zone seen where there is a caption, say), so each is left out in turn
        for left_out in ("near_goal_line", "far_goal_line"):
            tried = {name: line for name, line in best[1].items() if name != left_out}
            fit = camera_for(tried) if len(tried) < len(best[1]) else None
            if fit is not None and fit.error < best[0].error:
                best = (fit, tried)
    if best is None or best[0].error > MAX_MISFIT_PIXELS:
        return None
    fit, lines = best
    return FieldEstimate(
        field_to_image=fit.field_to_image,
        image_to_field=fit.image_to_field,
        lines=lines,
        misfit=fit.error,
        focal=fit.focal,
        position=fit.position,
    )


def found_lines(
    segmentation_results: List[Any],
    frame_shape: Tuple[int, int],
    outline_lines: Optional[Sequence[Line]] = None,
) -> Tuple[Dict[str, Line], List[Line]]:
    """The lines of the field that the masks show, in frame pixels.

    Returns:
        (lines by name, goal lines of which the masks do not tell whether far or near)
    """
    if outline_lines is None:
        mask = create_unified_field_mask(segmentation_results, frame_shape)
        outline_lines = fit_lines_from_mask(mask)[0] if mask is not None else []

    across, along = [], []
    for line in outline_lines:
        line = np.asarray(line, dtype=np.float64).reshape(2, 2)
        (across if _is_across(line) else along).append(line)
    lines: Dict[str, Line] = {}
    if across:
        lines["far_back_line"] = min(across, key=lambda line: line[:, 1].sum())
    if len(along) >= 2:
        by_place = sorted(along, key=lambda line: line[:, 0].sum())
        lines["left_sideline"], lines["right_sideline"] = by_place[0], by_place[-1]
    unsure: List[Line] = []
    if "far_back_line" in lines:
        named, unsure = _goal_lines(segmentation_results, frame_shape, lines["far_back_line"])
        lines.update(named)
    return lines, unsure


def _is_across(line: Line) -> bool:
    (x1, y1), (x2, y2) = line
    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
    return min(angle, 180.0 - angle) <= ACROSS_MAX_DEGREES


def _areas(segmentation_results: List[Any]) -> Tuple[Optional[np.ndarray], List[int]]:
    """The masks as one picture of area numbers (0: no field), and the class of each area.

    Where masks overlap, the more certain one counts.
    """
    for result in segmentation_results:
        if getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
            continue
        masks = np.asarray(result.masks.data)
        if masks.ndim == 4:
            masks = masks[:, 0]
        if len(masks) == 0:
            continue
        classes = np.asarray(_to_numpy(result.boxes.cls)).astype(int)
        certainty = np.asarray(_to_numpy(result.boxes.conf), dtype=np.float64)
        areas = np.zeros(masks.shape[1:], dtype=np.int32)
        for number in np.argsort(certainty):
            areas[masks[number] > 0.5] = number + 1
        return areas, [0, *classes.tolist()]
    return None, []


def _to_numpy(values: Any) -> Any:
    return values.cpu().numpy() if hasattr(values, "cpu") else values


def _goal_lines(
    segmentation_results: List[Any], frame_shape: Tuple[int, int], far_back_line: Line
) -> Tuple[Dict[str, Line], List[Line]]:
    """The goal lines: where an end zone and the central field, or two areas, meet.

    Returns:
        (goal lines by name, those of which the classes do not tell whether far or near)
    """
    areas, classes = _areas(segmentation_results)
    if areas is None:
        return {}, []
    height, width = frame_shape
    to_frame = np.array([width / areas.shape[1], height / areas.shape[0]])

    above, below = areas[:-BOUNDARY_STEP], areas[BOUNDARY_STEP:]
    rows, columns = np.nonzero((above != below) & (above > 0) & (below > 0))
    found: List[Tuple[float, Line, int, int]] = []  # (length, line, class above, class below)
    pairs = above[rows, columns].astype(np.int64) * (len(classes) + 1) + below[rows, columns]
    for pair in np.unique(pairs):
        chosen = pairs == pair
        points = np.column_stack([columns[chosen], rows[chosen] + BOUNDARY_STEP / 2.0])
        line = _fit_line(points)
        if line is None:
            continue
        line = line * to_frame
        length = float(np.linalg.norm(line[1] - line[0]))
        if length < MIN_GOAL_LINE_SHARE * width or not _is_across(line):
            continue
        upper, lower = divmod(int(pair), len(classes) + 1)
        found.append((length, line, classes[upper], classes[lower]))

    far_height = float(far_back_line[:, 1].mean())
    lines: Dict[str, Line] = {}
    unsure: List[Line] = []
    for _, line, upper, lower in sorted(found, key=lambda item: -item[0]):  # The longest first
        if float(line[:, 1].mean()) <= far_height:
            continue  # Beyond the field: the neighbouring one
        if upper == END_ZONE and lower == CENTRAL_FIELD:
            lines.setdefault("far_goal_line", line)
        elif upper == CENTRAL_FIELD and lower == END_ZONE:
            lines.setdefault("near_goal_line", line)
        elif len(unsure) < 2:
            # The model took both sides for the same kind
            unsure.append(line)
    return lines, unsure


def _fit_line(points: np.ndarray) -> Optional[Line]:
    """A straight line through most of the points, as its two ends; None if there is none."""
    if len(points) < 10:
        return None
    points = points.astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    direction, start = np.array([vx, vy]), np.array([x0, y0])
    offset = points - start
    away = np.abs(offset[:, 0] * direction[1] - offset[:, 1] * direction[0])
    on_line = away <= GOAL_LINE_REACH
    if on_line.sum() < 10:
        return None
    along = offset[on_line] @ direction
    return np.array([start + along.min() * direction, start + along.max() * direction])
