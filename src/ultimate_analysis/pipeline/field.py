"""Where the field is, frame by frame.

The field model runs every few frames, on another thread unless that is switched off
(models.segmentation.in_background), and what it finds is taken in a frame or two
later, moved by how the camera moved since. From its masks follow the field's outline,
the lines of the outline, and where the field lies in the frame (FieldFollower), which
the top-down view and every place on the field go by.
"""

import time
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config.settings import get_setting
from ..constants import DEFAULT_PATHS
from ..processing import health
from ..processing.field_analysis import create_unified_field_mask, fit_lines_from_mask
from ..processing.field_registration import FieldFollower
from ..processing.field_segmentation import (
    prepare_for_segmentation,
    run_field_segmentation,
    segment_prepared,
)
from ..utils.field_label_files import video_focal
from ..utils.field_template import DEFAULT_RULESET, TEMPLATES, FieldTemplate
from ..utils.logger import get_logger

logger = get_logger("PIPELINE")

# The dataset of field labels that a video's focal length is taken from
FIELD_LABELS = "hand_field"


@dataclass
class _FieldGeometry:
    """What follows from one segmentation result at one frame size.

    Segmentation is reused for several frames, so this is worked out once per result.
    """

    source: Optional[List[Any]] = None  # The segmentation result it was made from
    frame_shape: Optional[Tuple[int, int]] = None
    mask: Optional[np.ndarray] = None


def _feet_of(detections: List[Dict[str, Any]]) -> np.ndarray:
    """(N, 2) the middle of the lower edge of each detected player's box."""
    return np.array(
        [
            [(d["bbox"][0] + d["bbox"][2]) / 2.0, d["bbox"][3]]
            for d in detections
            if d["class_name"] == "player"
        ]
    ).reshape(-1, 2)


class FieldStage:
    """The field model's masks of the frames, and the field's place followed over them."""

    def __init__(self, threads: Executor, record: Callable[[str, float], float]):
        """
        Args:
            threads: Where the field model's work beside the frames is done
            record: Notes how long a part took: record(stage, perf_counter at its start)
        """
        self.follower = FieldFollower(
            TEMPLATES.get(
                get_setting("homography.ruleset", DEFAULT_RULESET), TEMPLATES[DEFAULT_RULESET]
            )
        )
        self._threads = threads
        self._record = record
        # The field model's masks of the frame being analysed (reused between runs)
        self.results: List[Any] = []
        # The field model's work beside the frames: the job under way, the frame and
        # frame size the last one was begun at, and the masks last taken in
        self._job: Optional[dict] = None
        self._begun: Optional[Tuple[int, Tuple[int, int]]] = None
        self._seen: List[Any] = []
        self._geometry = _FieldGeometry()

    @property
    def template(self) -> FieldTemplate:
        return self.follower.template

    @property
    def mask(self) -> Optional[np.ndarray]:
        """The field as the model sees it in the frame (H, W, 1 = field), if it was found."""
        return self._geometry.mask

    def new_video(self, video_path: Optional[str]) -> None:
        """Another camera: its focal length is taken from the video's field labels, or
        learned from the frames and kept for the next time."""
        focal = kept_in = None
        if video_path:
            dataset = Path(DEFAULT_PATHS["TRAINING_DATA"]) / FIELD_LABELS
            focal = video_focal(dataset, video_path, self.template)
            folder = get_setting("homography.focal_folder", "data/cache/focal")
            kept_in = Path(folder) / f"{Path(video_path).stem}.json" if folder else None
        self.follower.new_video(focal, kept_in)

    def reset(self) -> None:
        """Forget where the field was, and what is being worked out of an earlier view."""
        self.follower.reset()
        self.forget_job()
        self.invalidate()
        self.results = []

    def forget_job(self) -> None:
        """What the other thread is working out is of the view before."""
        self._job, self._begun, self._seen = None, None, []

    def invalidate(self) -> None:
        """Work the mask out again for the next frame (a redraw with other options)."""
        self._geometry = _FieldGeometry()

    def camera_moved(self, motion: Optional[np.ndarray]) -> None:
        """How the camera moved to this frame (None: not known)."""
        if self._job is not None:
            # What is being worked out of the field is of the frame it was begun in
            before = self._job["since"]
            self._job["since"] = None if motion is None or before is None else motion @ before
        if motion is not None:
            self.follower.move(motion)

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        detections: List[Dict[str, Any]],
        follow: bool,
    ) -> None:
        """Run the field model for a frame where it is due, and take in what it found.

        Args:
            frame: The frame (BGR)
            frame_index: Its position in the video
            detections: Its detections: the players' feet tell a wrong estimate
            follow: Whether the field's place is followed (and not only its masks)
        """
        start = time.perf_counter()
        in_background = bool(get_setting("models.segmentation.in_background", True))
        if in_background:
            self._beside_the_frames(frame, frame_index, detections, follow)
        else:
            self.results = run_field_segmentation(frame, frame_index)
        self._record("Field Segmentation", start)
        if self.results and not in_background:
            self._update_geometry(frame.shape[:2], detections, follow)

    def _beside_the_frames(
        self,
        frame: np.ndarray,
        frame_index: int,
        detections: List[Dict[str, Any]],
        follow: bool,
    ) -> None:
        """The field model and what follows from it, on another thread: begun every
        few frames, and taken in when it is done, a frame or two later. Finding the
        field takes several frames' worth of time, and the field does not move."""
        job = self._job
        if job is not None and job["work"].done():
            self._job = None
            self._take(job)
        interval = max(1, int(get_setting("models.segmentation.frame_interval", 5)))
        shape = frame.shape[:2]
        due = (
            self._begun is None
            or not 0 <= frame_index - self._begun[0] < interval
            or self._begun[1] != shape
        )
        if due and self._job is None:
            prepared = prepare_for_segmentation(frame)
            if prepared is not None:
                self._begun = (frame_index, shape)
                self._job = {
                    "work": self._threads.submit(self._look, prepared, shape, follow),
                    "shape": shape,
                    "feet": _feet_of(detections),
                    "follow": follow,
                    "since": np.eye(3),
                }
        self.results = self._seen

    def _look(self, prepared: np.ndarray, shape: Tuple[int, int], follow: bool) -> dict:
        """On another thread: the field model's masks for a frame, the lines of their
        outline, and where the field lies by them."""
        results = segment_prepared(prepared)
        mask = create_unified_field_mask(results, shape) if results else None
        lines = fit_lines_from_mask(mask)[0] if mask is not None else []
        estimate = self.follower.estimate(results, shape, lines) if follow else None
        return {"results": results, "mask": mask, "estimate": estimate}

    def _take(self, job: dict) -> None:
        """Take in what was worked out of the field beside the frames."""
        try:
            seen = job["work"].result()
        except Exception as error:
            logger.exception(f"Error finding the field: {error}")
            health.report("Field", f"failed ({type(error).__name__}); see the log")
            return
        self._seen = seen["results"]
        self._geometry = _FieldGeometry(seen["results"], job["shape"], seen["mask"])
        since = job["since"]
        if since is None:
            return  # The camera's motion was lost meanwhile: where that frame lay is not known
        if job["follow"]:
            self.follower.take(seen["estimate"], job["shape"], seen["mask"], job["feet"], since)

    def _update_geometry(
        self, frame_shape: Tuple[int, int], detections: List[Dict[str, Any]], follow: bool
    ) -> None:
        """Work out the field's mask and lines for the current segmentation result, and
        where the field lies by them.

        Only when the segmentation result or the frame size has changed, so the frames
        between two segmentation runs reuse one mask and one line fit.
        """
        if self.results is self._geometry.source and frame_shape == self._geometry.frame_shape:
            return
        start = time.perf_counter()
        mask = create_unified_field_mask(self.results, frame_shape)
        self._record("Mask Unification", start)
        lines: List[Any] = []
        if mask is not None:
            start = time.perf_counter()
            lines = fit_lines_from_mask(mask)[0]
            self._record("Line Extraction", start)
        self._geometry = _FieldGeometry(self.results, frame_shape, mask)
        if follow:
            start = time.perf_counter()
            self.follower.update(self.results, frame_shape, lines, mask, _feet_of(detections))
            self._record("Field Estimate", start)
