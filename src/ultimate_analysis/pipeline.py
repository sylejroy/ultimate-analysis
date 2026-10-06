"""The per-frame analysis pipeline.

A frame goes through detection, tracking, possession, jersey number reading, and field
segmentation.
The results are drawn onto the camera view and onto the top-down view. There is no Qt in
this module: the GUI owns one AnalysisPipeline and runs it on a worker thread.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .config.settings import get_setting
from .processing.camera_motion import CameraMotionEstimator
from .processing.camera_motion import is_enabled as camera_motion_enabled
from .processing.field_analysis import create_unified_field_mask, fit_lines_from_mask
from .processing.field_segmentation import reset_segmentation_cache, run_field_segmentation
from .processing.homography import output_canvas_size
from .processing.inference import reset_inference_state, run_inference
from .processing.jersey_crops import JerseyCropSelector
from .processing.jersey_tracker import (
    get_best_jersey_number,
    merge_jersey_readings,
    reset_jersey_tracker,
)
from .processing.player_id import run_player_id_on_tracks
from .processing.possession import PossessionTracker
from .processing.tracking import (
    apply_camera_motion,
    get_track_histories,
    kit_distance,
    merge_players,
    missing_players,
    reset_tracker,
    run_tracking,
    set_frame_rate,
)
from .rendering.field import (
    draw_field_segmentation,
    draw_unified_field_mask,
    get_primary_field_color,
)
from .rendering.field_lines import draw_ransac_field_lines
from .rendering.overlays import draw_fps_overlay, draw_jersey_table
from .rendering.top_down import apply_segmentation_to_warped_frame, draw_tracks_top_down
from .rendering.tracks import (
    draw_detections,
    draw_possession,
    draw_tracks,
    draw_tracks_with_player_ids,
)
from .utils.logger import get_logger

logger = get_logger("PIPELINE")

Line = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class PipelineOptions:
    """Which stages run for a frame."""

    detection: bool = True
    tracking: bool = True
    player_id: bool = True
    field_segmentation: bool = True
    top_down_view: bool = True

    @property
    def analysis_key(self) -> Tuple[bool, bool, bool, bool]:
        """The options that change the analysis results (the top-down view only draws)."""
        return (self.detection, self.tracking, self.player_id, self.field_segmentation)


@dataclass
class FrameResult:
    """Everything the pipeline produced for one frame."""

    frame_index: int
    main_view: np.ndarray  # Camera view with overlays (BGR)
    top_down_view: Optional[np.ndarray]  # Warped view with overlays, None if unavailable
    top_down_message: str  # Why there is no top-down view
    detections: List[Dict[str, Any]]
    tracks: List[Any]
    player_ids: Dict[int, Tuple[str, Any]]
    holder_id: Optional[int]  # Track ID of the player holding the disc, if any
    timings: Dict[str, float]  # Milliseconds per stage, in the order they ran


class AnalysisPipeline:
    """Runs the enabled analysis stages on a frame and renders the results.

    Tracking, possession, and jersey numbers build on earlier frames, so frames must arrive in playback
    order; call reset() after a seek, a video change, or a model change.
    """

    FPS_WINDOW = 30  # Frames the displayed processing rate is averaged over

    def __init__(self):
        self._jersey_crop_selector = JerseyCropSelector()
        # Camera-to-top-down homography as calibrated; without one there is no top-down
        # view. The camera has moved since the frame it was calibrated on (taken to be the
        # first frame after a reset), which is undone before it is applied.
        self._homography_matrix: Optional[np.ndarray] = None
        self._camera_since_calibration = np.eye(3)

        # Results for the current frame
        self.detections: List[Dict[str, Any]] = []
        self.tracks: List[Any] = []
        self.field_results: List[Any] = []
        self.ransac_lines: List[Line] = []
        self.ransac_confidences: List[float] = []

        # Jersey numbers persist across frames: OCR only runs for some tracks per frame
        self.player_ids: Dict[int, Tuple[str, Any]] = {}
        self._player_id_last_seen: Dict[int, int] = {}
        self._finalized_player_ids: Set[int] = set()

        self._possession = PossessionTracker()
        self._camera_motion = CameraMotionEstimator()

        # Redrawing the frame that was just analysed (pause, toggling an overlay) must not
        # advance the tracker again, so its results are reused.
        self._analysed_key: Optional[tuple] = None

        # Field geometry is derived only from a segmentation result and the frame size.
        # Segmentation is reused for several frames, so its mask, contour, and line fit
        # are computed once per result.
        self._geometry_source: Optional[List[Any]] = None
        self._geometry_frame_shape: Optional[Tuple[int, int]] = None
        self._field_mask: Optional[np.ndarray] = None
        self._field_contour: Optional[np.ndarray] = None
        self._ransac_fit: Optional[tuple] = None
        self._geometry_lines: List[Line] = []
        self._geometry_confidences: List[float] = []

        self._frame_times_ms: List[float] = []
        self.fps = 0.0
        self._timings: Dict[str, float] = {}

    # ------------------------------------------------------------------ state

    @property
    def homography_matrix(self) -> Optional[np.ndarray]:
        """Calibrated camera-to-top-down homography."""
        return self._homography_matrix

    @homography_matrix.setter
    def homography_matrix(self, matrix: Optional[np.ndarray]) -> None:
        self._homography_matrix = matrix
        self._camera_since_calibration = np.eye(3)

    def _top_down_matrix(self) -> Optional[np.ndarray]:
        """Homography from the current frame to the top-down view."""
        if self._homography_matrix is None:
            return None
        return self._homography_matrix @ np.linalg.inv(self._camera_since_calibration)

    def reset(self) -> None:
        """Forget everything derived from earlier frames."""
        self._jersey_crop_selector.reset()
        reset_tracker()
        reset_inference_state()
        reset_segmentation_cache()
        self._possession.reset()
        self._camera_motion.reset()
        self._camera_since_calibration = np.eye(3)
        self.invalidate()
        self.detections = []
        self.tracks = []
        self.field_results = []
        self.player_ids.clear()
        self._player_id_last_seen.clear()
        self._finalized_player_ids.clear()

    def set_frame_rate(self, frames_per_second: float) -> None:
        """Tell the pipeline the frame rate of the video; durations are set in seconds."""
        set_frame_rate(frames_per_second)

    def reset_player_ids(self) -> None:
        """Forget the jersey numbers read so far (e.g. after switching the reader)."""
        self._jersey_crop_selector.reset()
        reset_jersey_tracker()
        self.player_ids.clear()
        self._player_id_last_seen.clear()
        self._finalized_player_ids.clear()
        self._analysed_key = None

    def reset_fps(self) -> None:
        """Start the processing rate average again (new video)."""
        self._frame_times_ms.clear()
        self.fps = 0.0

    def invalidate(self) -> None:
        """Analyse the current frame again on the next call instead of reusing results."""
        self._analysed_key = None
        self._geometry_source = None
        self._geometry_frame_shape = None
        self._field_mask = None
        self._field_contour = None
        self._ransac_fit = None
        self._geometry_lines = []
        self._geometry_confidences = []

    # ------------------------------------------------------------------ per frame

    def process(self, frame: np.ndarray, frame_index: int, options: PipelineOptions) -> FrameResult:
        """Analyse a frame and render the camera and top-down views.

        Args:
            frame: Video frame (BGR); it is not modified
            frame_index: Position of the frame in the video
            options: Stages to run

        Returns:
            The rendered views, the analysis results, and the time each stage took
        """
        start = time.perf_counter()
        self._timings = {}

        analysis_key = (frame_index, *options.analysis_key)
        if analysis_key != self._analysed_key:
            self._analyse(frame, frame_index, options)
            self._analysed_key = analysis_key

        main_view = self._draw_main_view(frame.copy(), options)

        top_down_view, top_down_message = None, "Homography view disabled"
        if options.top_down_view:
            top_down_view, top_down_message = self._draw_top_down_view(frame, options)

        total_ms = (time.perf_counter() - start) * 1000
        self._update_fps(total_ms)
        return FrameResult(
            frame_index=frame_index,
            main_view=main_view,
            top_down_view=top_down_view,
            top_down_message=top_down_message,
            detections=self.detections,
            tracks=self.tracks,
            player_ids=dict(self.player_ids),
            holder_id=self._possession.holder_id,
            timings=self._timings,
        )

    def _record(self, stage: str, start: float) -> float:
        """Record the time since `start` (perf_counter) for a stage; returns it in ms."""
        duration_ms = (time.perf_counter() - start) * 1000
        self._timings[stage] = self._timings.get(stage, 0.0) + duration_ms
        return duration_ms

    def _update_fps(self, frame_time_ms: float) -> None:
        self._frame_times_ms.append(frame_time_ms)
        if len(self._frame_times_ms) > self.FPS_WINDOW:
            self._frame_times_ms.pop(0)
        average_ms = sum(self._frame_times_ms) / len(self._frame_times_ms)
        self.fps = 1000.0 / average_ms if average_ms > 0 else 0.0

    # ------------------------------------------------------------------ analysis

    def _analyse(self, frame: np.ndarray, frame_index: int, options: PipelineOptions) -> None:
        """Run the enabled stages and store their results."""
        self.detections = []
        self.tracks = []
        self.field_results = []

        if options.detection:
            start = time.perf_counter()
            self.detections = run_inference(frame)
            self._record("Inference", start)

        if (options.tracking or options.top_down_view) and camera_motion_enabled():
            # Before tracking adds this frame's positions: the trails from earlier frames
            # follow the picture when the camera moves, and so does the top-down calibration
            start = time.perf_counter()
            boxes = [detection["bbox"] for detection in self.detections]
            motion = self._camera_motion.update(frame, boxes)
            if motion is not None:
                apply_camera_motion(motion)
                self._camera_since_calibration = motion @ self._camera_since_calibration
            self._record("Camera Motion", start)

        if options.tracking:
            start = time.perf_counter()
            self.tracks = run_tracking(frame, self.detections)
            self._possession.update(self.detections, self.tracks)
            self._record("Tracking", start)
        else:
            self._possession.reset()

        if options.field_segmentation:
            start = time.perf_counter()
            self.field_results = run_field_segmentation(frame, frame_index)
            self._record("Field Segmentation", start)
        else:
            self.ransac_lines = []
            self.ransac_confidences = []

        if options.player_id and self.tracks:
            self._read_player_ids(frame, frame_index)
        elif not options.player_id:
            self.player_ids = {}
            self._player_id_last_seen.clear()

    def _read_player_ids(self, frame: np.ndarray, frame_index: int) -> None:
        """Read jersey numbers for the tracks due this frame and merge them into player_ids."""
        new_ids, timing, self._finalized_player_ids = run_player_id_on_tracks(
            frame,
            self.tracks,
            frame_index=frame_index,
            finalized_tracks=self._finalized_player_ids,
            crop_selector=self._jersey_crop_selector,
        )
        if timing["preprocessing_ms"] > 0 or timing["ocr_ms"] > 0:
            self._timings["Player ID - Preprocessing"] = timing["preprocessing_ms"]
            self._timings["Player ID - Reading"] = timing["ocr_ms"]
        if timing.get("filtering_ms", 0) > 0:
            self._timings["Player ID - Jersey Number Filtering"] = timing["filtering_ms"]

        for track_id, value in new_ids.items():
            self.player_ids[track_id] = value
            self._player_id_last_seen[track_id] = frame_index

        # Tracks that were not read this frame, or read as unknown, take the best number
        # the jersey tracker has accumulated for them.
        for track in self.tracks:
            track_id = getattr(track, "track_id", getattr(track, "id", None))
            if track_id is None:
                continue
            current = self.player_ids.get(track_id)
            if current is None or current[0] in ("Unknown", None, ""):
                best_number, best_probability = get_best_jersey_number(track_id)
                if best_number and best_probability > 0.0:
                    details = (current[1] if current else None) or {}
                    details["best_tracked"] = {
                        "jersey_number": best_number,
                        "probability": best_probability,
                    }
                    self.player_ids[track_id] = (best_number, details)
                    self._player_id_last_seen[track_id] = frame_index

        self._merge_players_by_number()

        # Drop the numbers of tracks that are gone
        current_ids = {getattr(t, "track_id", getattr(t, "id", -1)) for t in self.tracks}
        for track_id in [tid for tid in self.player_ids if tid not in current_ids]:
            self.player_ids.pop(track_id, None)
            self._player_id_last_seen.pop(track_id, None)

    def _merge_players_by_number(self) -> None:
        """A player whose number is that of a missing player is that player.

        Place and looks could not decide who a new track was when it appeared; the jersey
        number can, once it has been read often enough. Both teams may have the same
        number, so the two must also wear the same kit.
        """
        certainty_needed = float(get_setting("models.tracking.identity.number_certainty", 0.6))
        max_distance = float(get_setting("models.tracking.identity.max_kit_distance", 30.0))
        present = {track.track_id for track in self.tracks if track.class_name == "player"}
        missing = {}
        for player_id in missing_players(present):
            number, certainty = get_best_jersey_number(player_id)
            if number and certainty >= certainty_needed:
                missing.setdefault(number, player_id)

        for track in self.tracks:
            player_id = track.track_id
            if track.class_name != "player" or not missing:
                continue
            number, certainty = get_best_jersey_number(player_id)
            earlier = missing.get(number) if number and certainty >= certainty_needed else None
            if earlier is None:
                continue
            distance = kit_distance(player_id, earlier)
            if distance is None or distance > max_distance:
                continue

            merge_players(player_id, earlier)
            merge_jersey_readings(player_id, earlier)
            track.track_id = earlier
            if player_id in self.player_ids:
                self.player_ids[earlier] = self.player_ids.pop(player_id)
                self._player_id_last_seen[earlier] = self._player_id_last_seen.pop(player_id, 0)
            if player_id in self._finalized_player_ids:
                self._finalized_player_ids.discard(player_id)
                self._finalized_player_ids.add(earlier)
            self._possession.rename(player_id, earlier)
            del missing[number]
            logger.info(f"Player {player_id} is player {earlier} again (number {number})")

    def _field_geometry(self, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Field mask for the current segmentation result; also updates contour and lines.

        Recomputed only when the segmentation result or the frame size changes, so the
        frames between two segmentation runs reuse one mask and one randomized line fit.
        """
        if (
            self.field_results is not self._geometry_source
            or frame_shape != self._geometry_frame_shape
        ):
            start = time.perf_counter()
            mask = create_unified_field_mask(self.field_results, frame_shape)
            self._record("Mask Unification", start)

            lines: List[Line] = []
            confidences: List[float] = []
            contour: Optional[np.ndarray] = None
            ransac_fit: Optional[tuple] = None
            if mask is not None:
                start = time.perf_counter()
                lines, confidences, contour, ransac_fit = fit_lines_from_mask(mask)
                self._record("Line Extraction", start)

            self._geometry_source = self.field_results
            self._geometry_frame_shape = frame_shape
            self._field_mask = mask
            self._field_contour = contour
            self._ransac_fit = ransac_fit
            self._geometry_lines = lines
            self._geometry_confidences = confidences

        return self._field_mask

    # ------------------------------------------------------------------ rendering

    def _draw_main_view(self, frame: np.ndarray, options: PipelineOptions) -> np.ndarray:
        """Draw the overlays for the current results on a frame the caller owns."""
        if not (options.field_segmentation or self.detections or self.tracks):
            draw_fps_overlay(frame, self.fps)
            return frame

        start = time.perf_counter()
        geometry_ms = 0.0

        # Field first, as background for the players
        if self.field_results and options.field_segmentation:
            if get_setting("models.segmentation.show_raw_masks", True):
                frame = draw_field_segmentation(frame, self.field_results)

            before = dict(self._timings)
            mask = self._field_geometry(frame.shape[:2])
            geometry_ms = sum(
                self._timings.get(stage, 0.0) - before.get(stage, 0.0)
                for stage in ("Mask Unification", "Line Extraction")
            )

            if mask is not None:
                self.ransac_lines = self._geometry_lines
                self.ransac_confidences = self._geometry_confidences

                frame = self._draw_field_overlay(frame, mask)
            else:
                self.ransac_lines = []
                self.ransac_confidences = []

        # Plain detections only without tracking; tracks carry the same boxes
        if self.detections and not options.tracking:
            frame = draw_detections(frame, self.detections, in_place=True)
        elif self.tracks and options.tracking:
            track_histories = get_track_histories()
            if options.player_id:
                frame = draw_tracks_with_player_ids(
                    frame, self.tracks, track_histories, self.player_ids, in_place=True
                )
            else:
                frame = draw_tracks(frame, self.tracks, track_histories, in_place=True)
            draw_possession(frame, self.tracks, self._possession.holder_id)

        draw_fps_overlay(frame, self.fps)
        if options.player_id and self.player_ids:
            draw_jersey_table(frame)

        # Geometry is reported under its own stages
        total_ms = (time.perf_counter() - start) * 1000
        self._timings["Visualization"] = max(0.0, total_ms - geometry_ms)
        return frame

    def _draw_field_overlay(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """The field outline and lines, drawn before the players."""
        frame = draw_unified_field_mask(
            frame,
            mask,
            get_primary_field_color(),
            alpha=0.3,
            fill_mask=False,
            ransac_fit=self._ransac_fit,
            field_contour=self._field_contour,
            in_place=True,
        )
        if self.ransac_lines:
            frame = draw_ransac_field_lines(
                frame,
                self.ransac_lines,
                self.ransac_confidences,
                transformation_matrix=None,
                scale_factor=1.0,
                in_place=True,
            )
        return frame

    def _draw_top_down_view(
        self, frame: np.ndarray, options: PipelineOptions
    ) -> Tuple[Optional[np.ndarray], str]:
        """Warp the frame to the top-down view and draw field, players, and lines on it.

        Returns:
            (view, "") or (None, reason there is no view)
        """
        top_down_matrix = self._top_down_matrix()
        if top_down_matrix is None:
            return None, "Homography matrix not available"

        try:
            start = time.perf_counter()
            height, width = frame.shape[:2]
            output_width, output_height = output_canvas_size(width, height)

            # The panel is far smaller than the full canvas, so render it at a reduced
            # scale; warp cost is proportional to the output pixel count.
            scale = float(get_setting("homography.display_scale", 0.5))
            scale = min(1.0, max(0.1, scale))
            matrix = top_down_matrix
            if scale != 1.0:
                output_width = max(1, int(output_width * scale))
                output_height = max(1, int(output_height * scale))
                matrix = np.diag([scale, scale, 1.0]) @ top_down_matrix

            view = cv2.warpPerspective(frame, matrix, (output_width, output_height))
            warp_ms = self._record("Homography Calculation", start)

            if self.field_results and options.field_segmentation:
                try:
                    # Normally already computed for the main view
                    self._field_geometry(frame.shape[:2])
                    with_field = apply_segmentation_to_warped_frame(
                        view,
                        self.field_results,
                        matrix,
                        frame.shape[:2],
                        "PIPELINE",
                        field_contour=self._field_contour,
                        draw_scale=scale,
                        in_place=True,
                    )
                    if with_field is not None:
                        view = with_field
                except Exception as e:
                    logger.error(f"Error applying segmentation to top-down view: {e}")

            if options.tracking and self.tracks:
                view = draw_tracks_top_down(
                    view,
                    matrix,
                    self.tracks,
                    self.player_ids,
                    get_track_histories(),
                    scale,
                    holder_id=self._possession.holder_id,
                )

            if self.ransac_lines:
                view = draw_ransac_field_lines(
                    view,
                    self.ransac_lines,
                    self.ransac_confidences,
                    matrix,
                    scale_factor=2.0 * scale,
                    show_confidence=True,
                    in_place=True,
                )

            overlays_ms = (time.perf_counter() - start) * 1000 - warp_ms
            if overlays_ms > 1.0:
                self._timings["Homography Other"] = overlays_ms
            return np.ascontiguousarray(view), ""

        except Exception as e:
            logger.error(f"Error rendering top-down view: {e}")
            return None, f"Error: {e}"
