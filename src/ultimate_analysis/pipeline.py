"""The per-frame analysis pipeline.

A frame goes through detection, tracking, possession, jersey number reading, and field
segmentation.
The results are drawn onto the camera view and onto the top-down view. There is no Qt in
this module: the GUI owns one AnalysisPipeline and runs it on a worker thread.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config.settings import get_setting
from .constants import DEFAULT_PATHS
from .processing import health
from .processing.camera_motion import CameraMotionEstimator
from .processing.camera_motion import is_enabled as camera_motion_enabled
from .processing.disc_flight import DiscPath, place_under_disc
from .processing.field_analysis import create_unified_field_mask, fit_lines_from_mask
from .processing.field_line_filter import FieldLineFilter
from .processing.field_registration import FieldFollower, OffFieldWatcher, field_to_canvas
from .processing.field_segmentation import reset_segmentation_cache, run_field_segmentation
from .processing.game_state import GameStateTracker, Player
from .processing.homography import output_canvas_size
from .processing.inference import reset_inference_state, run_inference
from .processing.jersey_tracker import get_best_jersey_number
from .processing.player_looks import PlayerLooks
from .processing.player_numbers import PlayerNumbers
from .processing.possession import PossessionTracker
from .processing.shot_type import ShotWatcher, players_are_elsewhere
from .processing.tracking import (
    apply_camera_motion,
    get_track_histories,
    reset_tracker,
    run_tracking,
    set_frame_rate,
    team_shirt_colours,
)
from .rendering.field import (
    draw_field_segmentation,
    draw_unified_field_mask,
    get_primary_field_color,
)
from .rendering.field_lines import draw_ransac_field_lines
from .rendering.overlays import draw_fps_overlay, draw_jersey_table, draw_notice
from .rendering.top_down import (
    apply_segmentation_to_warped_frame,
    draw_field_diagram,
    draw_field_template,
    draw_tracks_top_down,
    hide_behind_camera,
)
from .rendering.tracks import (
    draw_detections,
    draw_possession,
    draw_tracks,
    draw_tracks_with_player_ids,
    team_display_colour,
)
from .utils.field_camera import camera_of
from .utils.field_label_files import video_focal
from .utils.field_template import DEFAULT_RULESET, TEMPLATES
from .utils.logger import get_logger

logger = get_logger("PIPELINE")

# Two team colours (BGR) nearer than this are one team's, seen before and after a cut
SAME_TEAM_COLOUR = 90.0
# The table of the players' statistics is handed out with every so many frames
STATS_EVERY_FRAMES = 30

Line = Tuple[np.ndarray, np.ndarray]
# The dataset of field labels that a video's focal length is taken from
FIELD_LABELS = "labelled_field_v1"


@dataclass(frozen=True)
class PipelineOptions:
    """Which stages run for a frame."""

    detection: bool = True
    tracking: bool = True
    player_id: bool = True
    field_segmentation: bool = True
    top_down_view: bool = True
    # What the top-down view is made from: "calibration" (the mapping set by hand in the
    # Homography tab, moved with the camera) or "field" (where the field model sees the
    # field in each frame)
    top_down_source: str = "field"

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
    # Track -> jersey number of the player a track without a number looks like
    numbers_by_looks: Dict[int, str]
    # A row per player of the video (AnalysisPipeline.player_stats); None in most frames
    player_stats: Optional[List[dict]]
    holder_id: Optional[int]  # Track ID of the player holding the disc, if any
    timings: Dict[str, float]  # Milliseconds per stage, in the order they ran
    wide_shot: bool = True  # False on a close-up or title card, where nothing is analysed
    problems: Tuple[str, ...] = ()  # Stages that failed on this frame, for showing
    # The colour (BGR) of the team in possession, None while it is not known, and where
    # the disc is: "held", "air" (also: not seen) or "ground"
    possession_colour: Optional[Tuple[int, int, int]] = None
    disc_state: str = "air"
    # The frame from which that is so: a change is confirmed some frames after it happened
    possession_since: int = 0
    # How long the disc has been flying, in seconds; None if it is not seen to fly
    flight_seconds: Optional[float] = None
    # For whoever keeps a record: the team in possession (0 or 1), where the field lies
    # in this frame (pixel -> field units, 3x3; None if not known), and over which spot
    # of the field a flying disc is
    possession_team: Optional[int] = None
    image_to_field: Optional[np.ndarray] = None
    disc_place: Optional[Tuple[float, float]] = None
    # The phase of the game: "unknown", "between points", "lined up", "pull", "live" or
    # "score" (processing/game_state.py)
    game_state: str = "unknown"
    # Points begun so far (pulls seen), how long the present one has been played (None
    # between points), and the scores seen: (a team's colour (BGR), its points) each
    points_begun: int = 0
    point_seconds: Optional[float] = None
    score: Tuple[Tuple[Tuple[int, int, int], int], ...] = ()


@dataclass
class _FieldGeometry:
    """What follows from one segmentation result at one frame size.

    Segmentation is reused for several frames, so this is worked out once per result.
    """

    source: Optional[List[Any]] = None  # The segmentation result it was made from
    frame_shape: Optional[Tuple[int, int]] = None
    mask: Optional[np.ndarray] = None
    contour: Optional[np.ndarray] = None
    ransac_fit: Optional[tuple] = None


class AnalysisPipeline:
    """Runs the enabled analysis stages on a frame and renders the results.

    Tracking, possession, and jersey numbers build on earlier frames, so frames must arrive in playback
    order; call reset() after a seek, a video change, or a model change.
    """

    FPS_WINDOW = 30  # Frames the displayed processing rate is averaged over

    def __init__(self):
        # Camera-to-top-down homography as calibrated; without one there is no top-down
        # view. The camera has moved since the frame it was calibrated on (taken to be the
        # first frame after a reset), which is undone before it is applied.
        self._homography_matrix: Optional[np.ndarray] = None
        self._camera_since_calibration = np.eye(3)
        # Where the field model sees the field, for a top-down view without calibration
        self._field_follower = FieldFollower(
            TEMPLATES.get(
                get_setting("homography.ruleset", DEFAULT_RULESET), TEMPLATES[DEFAULT_RULESET]
            )
        )
        self._off_field = OffFieldWatcher(self._field_follower.template)
        template = self._field_follower.template
        self._game = GameStateTracker(template.length, template.end_zone)
        self._score: List[list] = []  # [a team's colour (BGR), its points]
        self._disc_path = DiscPath(float(get_setting("homography.disc_path_seconds", 8.0)))
        self._looks = PlayerLooks()
        self.numbers_by_looks: Dict[int, str] = {}
        self._frame_size: Optional[Tuple[int, int]] = None
        self._trails_now: Optional[Dict[int, np.ndarray]] = None
        self._events_counted = 0
        # Over which spot of the field (field units) a flying disc is, if that is known
        self.disc_place: Optional[Tuple[float, float]] = None
        # For telling a cut: whether the camera's motion was known at the frame before,
        # and where the players stood in it
        self._motion_known = False
        self._players_before = np.zeros((0, 4))
        self._follow_field = False

        # Results for the current frame
        self.detections: List[Dict[str, Any]] = []
        self.tracks: List[Any] = []
        self.field_results: List[Any] = []
        self.ransac_lines: List[Line] = []
        self.ransac_confidences: List[float] = []

        # Jersey numbers persist across frames: only some players are read per frame
        self._numbers = PlayerNumbers()

        self._possession = PossessionTracker()
        self._camera_motion = CameraMotionEstimator()
        # Close-ups and title cards of an edited game are not analysed
        self._shot = ShotWatcher()
        # The field lines shown: moved with the camera between fits, smoothed over fits
        self._line_filter = FieldLineFilter()

        # Redrawing the frame that was just analysed (pause, toggling an overlay) must not
        # advance the tracker again, so its results are reused.
        self._analysed_key: Optional[tuple] = None

        # Field geometry is derived only from a segmentation result and the frame size.
        # Segmentation is reused for several frames, so its mask, contour, and line fit
        # are computed once per result.
        self._geometry = _FieldGeometry()

        self._frame_times_ms: List[float] = []
        self.fps = 0.0
        self._timings: Dict[str, float] = {}

    # ------------------------------------------------------------------ state

    @property
    def player_ids(self) -> Dict[int, Tuple[str, Any]]:
        """Player ID -> (jersey number or "Unknown", details of the reading)."""
        return self._numbers.numbers

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

    def new_video(self, video_path: Optional[str] = None) -> None:
        """Start on another video: nothing of the last one holds, its camera included.

        Args:
            video_path: The video, if it is one; its field labels then give the camera's
                focal length, which makes the field's place in a frame much more certain
        """
        self.reset()
        self._game.reset()
        self._score, self._events_counted = [], 0
        self.reset_fps()
        focal = None
        if video_path:
            dataset = Path(DEFAULT_PATHS["TRAINING_DATA"]) / FIELD_LABELS
            focal = video_focal(dataset, video_path, self._field_follower.template)
        self._field_follower.new_video(focal)
        self._looks.new_video(video_path)

    def reset(self) -> None:
        """Forget everything derived from earlier frames."""
        self._field_follower.reset()
        self._off_field.reset()
        self._game.cut()
        self._looks.cut()
        self.disc_place = None
        self._disc_path.reset()
        self._numbers.reset()
        self._shot.reset()
        reset_tracker()
        reset_inference_state()
        reset_segmentation_cache()
        self._possession.reset()
        self._camera_motion.reset()
        self._motion_known = False
        self._line_filter.reset()
        self._camera_since_calibration = np.eye(3)
        self.invalidate()
        self.detections = []
        self.tracks = []
        self.field_results = []

    def set_frame_rate(self, frames_per_second: float) -> None:
        """Tell the pipeline the frame rate of the video; durations are set in seconds."""
        set_frame_rate(frames_per_second)
        if frames_per_second and frames_per_second > 0:
            self._possession.frame_rate = float(frames_per_second)
            self._looks.frame_rate = float(frames_per_second)

    def reset_player_ids(self) -> None:
        """Forget the jersey numbers read so far (e.g. after switching the reader)."""
        self._numbers.reset(readings_too=True)
        self._analysed_key = None

    def reset_fps(self) -> None:
        """Start the processing rate average again (new video)."""
        self._frame_times_ms.clear()
        self.fps = 0.0

    def invalidate(self) -> None:
        """Analyse the current frame again on the next call instead of reusing results."""
        self._analysed_key = None
        self._geometry = _FieldGeometry()
        self._line_filter.reset()

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
        self._trails_now = None
        # Also without the top-down view shown: who stands beside the field, and whether
        # the fitted lines are drawn, should not depend on a view being switched on
        self._follow_field = options.field_segmentation and options.top_down_source == "field"
        self._frame_size = (frame.shape[1], frame.shape[0])

        analysis_key = (frame_index, *options.analysis_key)
        if analysis_key != self._analysed_key:
            health.start_frame()
            self._analyse(frame, frame_index, options)
            self._analysed_key = analysis_key

        # The copy of the frame that is drawn on belongs to the drawing
        copy_start = time.perf_counter()
        drawn_on = frame.copy()
        self._record("Visualization", copy_start)
        main_view = self._draw_main_view(drawn_on, options)
        problems = health.problems()
        for line, problem in enumerate(problems):
            draw_notice(main_view, problem, line=line + 1, alarm=True)

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
            numbers_by_looks=dict(self.numbers_by_looks),
            # A table of every player is not made for every frame
            player_stats=self.player_stats() if frame_index % STATS_EVERY_FRAMES == 0 else None,
            holder_id=self._possession.holder_id,
            possession_colour=self._possession_colour(),
            disc_state=self._possession.disc_state,
            possession_since=self._possession.since,
            flight_seconds=self._possession.flight_seconds,
            possession_team=self._possession.team,
            game_state=self._game.state,
            points_begun=sum(1 for event in self._game.events if event.kind == "pull"),
            point_seconds=self._point_seconds(frame_index),
            score=tuple((tuple(colour), points) for colour, points in self._score),
            image_to_field=self._field_follower.image_to_field if self._follow_field else None,
            disc_place=self.disc_place,
            timings=self._timings,
            wide_shot=self._shot.wide,
            problems=problems,
        )

    def _record(self, stage: str, start: float) -> float:
        """Record the time since `start` (perf_counter) for a stage; returns it in ms."""
        duration_ms = (time.perf_counter() - start) * 1000
        self._timings[stage] = self._timings.get(stage, 0.0) + duration_ms
        return duration_ms

    def _follow_flight(self, frame_shape: Tuple[int, int], frame_index: int) -> None:
        """Work out over which spot of the field a flying disc is (`self.disc_place`).

        Needs to know where the field lies; without that the disc stays where its pixel
        would be on the ground.
        """
        self.disc_place = None
        to_field = self._field_follower.image_to_field
        if to_field is None:
            return
        state = self._possession.disc_state
        ground = raised = holder = None
        seen = []
        for detection in self.detections:
            if detection["class_name"] == "disc":
                x1, y1, x2, y2 = detection["bbox"]
                over = to_field @ [(x1 + x2) / 2, (y1 + y2) / 2, 1.0]
                if abs(over[2]) > 1e-12:
                    seen.append((detection.get("confidence", 0.0), over[:2] / over[2]))
        if seen:
            # Of several things taken for a disc, the one nearest to where the disc was;
            # with nothing to go by, the surest
            last = self._disc_path.last_place
            if last is None:
                ground = max(seen, key=lambda sighting: sighting[0])[1]
            else:
                ground = min(seen, key=lambda sighting: np.linalg.norm(sighting[1] - last))[1]
            try:
                _, _, camera = camera_of(
                    np.linalg.inv(to_field),
                    (frame_shape[1], frame_shape[0]),
                    self._field_follower.focal,
                )
                raised = place_under_disc(ground, camera)
            except (np.linalg.LinAlgError, cv2.error):
                pass
        if state == "air":
            self.disc_place = raised
        for track in self.tracks:
            if state == "held" and track.track_id == self._possession.holder_id:
                x1, _, x2, y2 = track.to_ltrb()
                feet = to_field @ [(x1 + x2) / 2, y2, 1.0]
                if abs(feet[2]) > 1e-12:
                    holder = feet[:2] / feet[2]
        # The path of the disc, put right once it is known what the disc was doing
        rate = self._possession.frame_rate
        self._disc_path.add(
            frame_index / rate, state, ground, raised, holder, self._possession.since / rate
        )

    def _follow_game(self, frame_index: int) -> None:
        """Work out the phase of the game from where the players stand on the field."""
        to_field = self._field_follower.image_to_field
        players = []
        if to_field is not None:
            for track in self.tracks:
                if track.class_name != "player":
                    continue
                x1, _, x2, y2 = track.to_ltrb()
                place = to_field @ [(x1 + x2) / 2, y2, 1.0]
                if abs(place[2]) > 1e-12:
                    players.append(
                        Player(track.track_id, track.team, place[0] / place[2], place[1] / place[2])
                    )
        self._game.update(
            frame_index / self._possession.frame_rate,
            players,
            self._possession.disc_state,
            self._possession.holder_id,
            self._possession.flight_seconds,
            known=self._shot.wide and to_field is not None,
        )
        # A score is counted for the colour of the team that made it: the tracker numbers
        # its teams anew after every cut, the colours stay
        for event in self._game.events[self._events_counted :]:
            shirt = team_shirt_colours().get(event.team) if event.kind == "score" else None
            if shirt is not None:
                colour = team_display_colour(shirt)
                for entry in self._score:
                    if np.linalg.norm(np.subtract(entry[0], colour)) < SAME_TEAM_COLOUR:
                        entry[1] += 1
                        break
                else:
                    self._score.append([colour, 1])
        self._events_counted = len(self._game.events)

    def _point_seconds(self, frame_index: int) -> Optional[float]:
        """How long the point under way has been played, None if none is."""
        pulls = [event for event in self._game.events if event.kind == "pull"]
        if not pulls or self._game.state not in ("pull", "live"):
            return None
        return max(0.0, frame_index / self._possession.frame_rate - pulls[-1].seconds)

    def _draw_field_diagram(
        self, on_canvas: np.ndarray, size: Tuple[int, int], options: PipelineOptions
    ) -> np.ndarray:
        """The top-down view as a drawing: the field, and the players where they stand."""
        to_field = self._field_follower.image_to_field
        template = self._field_follower.template
        # A drawing needs no more pixels than the panel it is shown in
        height = int(get_setting("homography.diagram_height", 900))
        size = (int(height * 1.25 * template.width / template.length), height)
        to_view = field_to_canvas(template, size)

        def on_field(pixels: np.ndarray) -> np.ndarray:
            mapped = pixels @ to_field[:, :2].T + to_field[:, 2]
            ahead = mapped[:, 2] * mapped[-1, 2] > 0  # Above the horizon is no ground
            return mapped[ahead, :2] / mapped[ahead, 2:3]

        players, disc = [], None
        histories = self._trails() if options.tracking else {}
        shown_numbers = self._shown_numbers()
        for track in self.tracks if options.tracking else ():
            x1, y1, x2, y2 = track.to_ltrb()
            if track.class_name != "player":
                if self._possession.holder_id is None:
                    disc = self.disc_place or tuple(
                        on_field(np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]))[0]
                    )
                continue
            shirt = getattr(track, "team_colour", None)
            number = shown_numbers.get(track.track_id, ("", None))[0]
            trail = histories.get(track.track_id)
            players.append(
                {
                    "place": tuple(on_field(np.array([[(x1 + x2) / 2, y2]]))[0]),
                    "colour": None if shirt is None else team_display_colour(shirt),
                    # A number with a tilde is known by the player's looks, not read
                    "label": f"#{number}"
                    if str(number).isdigit()
                    else (str(number) if str(number).startswith("~") else ""),
                    "holder": track.track_id == self._possession.holder_id,
                    "trail": on_field(np.asarray(trail, dtype=float)) if trail is not None else (),
                }
            )
        return draw_field_diagram(
            size,
            template.width,
            template.length,
            template.end_zone,
            to_view,
            players,
            disc,
            self._seen_by_camera(to_field),
            self._disc_path.stretches() if options.tracking else (),
        )

    def _seen_by_camera(self, to_field: np.ndarray) -> Optional[np.ndarray]:
        """The outline on the field of what the frame shows: its border, as far as that
        lies on the ground (the part above the horizon does not)."""
        if self._frame_size is None:
            return None
        width, height = self._frame_size
        along = np.linspace(0.0, 1.0, 24, endpoint=False)
        border = np.concatenate(
            [
                np.column_stack([along * width, np.zeros_like(along)]),
                np.column_stack([np.full_like(along, width), along * height]),
                np.column_stack([(1 - along) * width, np.full_like(along, height)]),
                np.column_stack([np.zeros_like(along), (1 - along) * height]),
            ]
        )
        mapped = np.column_stack([border, np.ones(len(border))]) @ to_field.T
        # The middle of the frame's lower edge is ground in front of the camera
        ground = to_field @ [width / 2.0, height - 1.0, 1.0]
        ahead = mapped[:, 2] * ground[2] > 1e-9 * abs(ground[2])
        if ahead.sum() < 3:
            return None
        places = mapped[ahead, :2] / mapped[ahead, 2:3]
        # Near the horizon the ground runs off to infinity: kept within sight of the field
        reach = 3.0 * self._field_follower.template.length
        return np.clip(places, -reach, reach)

    def _trails(self) -> Dict[int, np.ndarray]:
        """The trails of the tracks in this frame: both views draw them."""
        if self._trails_now is None:
            self._trails_now = get_track_histories()
        return self._trails_now

    def _possession_colour(self) -> Optional[Tuple[int, int, int]]:
        """The colour that stands for the team in possession, None while it is not known."""
        shirt = team_shirt_colours().get(self._possession.team)
        return None if shirt is None else team_display_colour(shirt)

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
            if get_setting("models.shot_type.enabled", True) and not self._watch_shot():
                return  # A close-up or a title card: nothing to follow

        if (options.tracking or options.top_down_view) and camera_motion_enabled():
            # Before tracking adds this frame's positions: the trails from earlier frames
            # follow the picture when the camera moves, and so does the top-down calibration
            start = time.perf_counter()
            boxes = [detection["bbox"] for detection in self.detections]
            motion = self._camera_motion.update(frame, boxes)
            players = np.array(
                [d["bbox"] for d in self.detections if d["class_name"] == "player"]
            ).reshape(-1, 4)
            if (
                motion is None
                and self._motion_known
                and players_are_elsewhere(self._players_before, players)
            ):
                # A cut to another view: every track would break and be matched with
                # whoever stands near. What was followed belongs to the view before.
                logger.info("Cut to another view: starting again")
                detections = self.detections
                self.reset()
                self.detections = detections
            self._motion_known, self._players_before = motion is not None, players
            if motion is not None:
                apply_camera_motion(motion)
                self._possession.move(motion)
                self._line_filter.move(motion)
                self._camera_since_calibration = motion @ self._camera_since_calibration
                self._field_follower.move(motion)
            self._record("Camera Motion", start)

        if options.tracking:
            start = time.perf_counter()
            self.tracks = run_tracking(frame, self.detections, frame_index)
            if self._follow_field and get_setting("models.tracking.hide_off_field", True):
                # Those standing along the sidelines are followed but not shown or counted
                off_field = self._off_field.update(self._field_follower.image_to_field, self.tracks)
                self.tracks = [track for track in self.tracks if track.track_id not in off_field]
            self._record("Tracking", start)
            # What is worked out from the tracks: who has the disc, where it flies, and
            # the phase of the game
            start = time.perf_counter()
            self._possession.update(self.detections, self.tracks, frame_index)
            self._follow_flight(frame.shape[:2], frame_index)
            self._follow_game(frame_index)
            self._record("Possession", start)
        else:
            self._possession.reset()

        self.ransac_lines, self.ransac_confidences = [], []
        if options.field_segmentation:
            start = time.perf_counter()
            self.field_results = run_field_segmentation(frame, frame_index)
            self._record("Field Segmentation", start)
            if self.field_results:
                self._update_field_geometry(frame.shape[:2])
                # The lines fitted to the outline are shown where they are what the view
                # is made from; with the field model's own estimate they are only clutter
                if self._geometry.mask is not None and not self._follow_field:
                    self.ransac_lines, self.ransac_confidences = self._line_filter.current()

        if options.player_id and self.tracks:
            timing, renamed = self._numbers.update(frame, self.tracks, frame_index)
            if timing["preprocessing_ms"] > 0 or timing["ocr_ms"] > 0:
                self._timings["Player ID - Preprocessing"] = timing["preprocessing_ms"]
                self._timings["Player ID - Reading"] = timing["ocr_ms"]
            if timing.get("filtering_ms", 0) > 0:
                self._timings["Player ID - Jersey Number Filtering"] = timing["filtering_ms"]
            for was, now in renamed:
                self._possession.rename(was, now)
                self._off_field.rename(was, now)
                self._looks.rename(was, now)
        elif not options.player_id:
            self._numbers.numbers.clear()

        self.numbers_by_looks = {}
        if options.tracking and self.tracks and get_setting("models.reid.enabled", True):
            start = time.perf_counter()
            self.numbers_by_looks = self._looks.update(
                frame,
                self.tracks,
                frame_index,
                self._certain_numbers(),
                self._possession.holder_id,
                self._places_on_field(),
            )
            self._record("Player ID - Looks", start)

    def _places_on_field(self) -> Dict[int, np.ndarray]:
        """Track -> where the player stands on the field, if the field's place is known."""
        to_field = self._field_follower.image_to_field if self._follow_field else None
        players = [track for track in self.tracks if track.class_name == "player"]
        if to_field is None or not players:
            return {}
        boxes = np.array([track.to_ltrb() for track in players], dtype=np.float64)
        feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]])
        mapped = feet @ to_field[:, :2].T + to_field[:, 2]
        return {
            int(track.track_id): place[:2] / place[2]
            for track, place in zip(players, mapped)
            if abs(place[2]) > 1e-12
        }

    def player_stats(self) -> List[dict]:
        """What is known of each player of the video and what they did: a row each
        (PlayerRoster.table), with the colour their team is drawn in."""
        rows = self._looks.roster.table()
        for row in rows:
            shirt = row.pop("team_colour")
            row["colour"] = None if shirt is None else team_display_colour(shirt)
        return rows

    def _certain_numbers(self) -> Dict[int, str]:
        """Track -> jersey number, for the players whose number is read often enough to
        name them by."""
        needed = float(get_setting("models.tracking.identity.number_certainty", 0.6))
        certain = {}
        for track in self.tracks:
            if track.class_name == "player" and track.track_id in self.player_ids:
                number, certainty = get_best_jersey_number(track.track_id)
                if number and certainty >= needed:
                    certain[track.track_id] = str(number)
        return certain

    def _shown_numbers(self) -> Dict[int, Tuple[str, Any]]:
        """The numbers to draw: those read, and for the others the number of the player
        they look like, marked with a tilde."""
        shown = dict(self.player_ids)
        for track_id, number in self.numbers_by_looks.items():
            if not str(shown.get(track_id, ("", None))[0]).isdigit():
                shown[track_id] = (f"~{number}", None)
        return shown

    def _watch_shot(self) -> bool:
        """Whether this frame is drone footage; what was followed is dropped when it stops.

        After a cut away, the players are others or elsewhere when the drone is back, and
        the camera has moved in between.
        """
        if health.failed("Detection"):
            return self._shot.wide  # No detections because of a failure say nothing
        was_wide = self._shot.wide
        players = sum(1 for detection in self.detections if detection["class_name"] == "player")
        wide = self._shot.update(players)
        if was_wide and not wide:
            detections = self.detections
            self.reset()
            self.detections = detections
            self._shot.pause()
        return wide

    def _update_field_geometry(self, frame_shape: Tuple[int, int]) -> None:
        """Work out the field's mask, outline, and lines for the current segmentation result.

        Only when the segmentation result or the frame size has changed, so the frames
        between two segmentation runs reuse one mask and one randomized line fit.
        """
        if (
            self.field_results is not self._geometry.source
            or frame_shape != self._geometry.frame_shape
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

            self._geometry = _FieldGeometry(
                self.field_results, frame_shape, mask, contour, ransac_fit
            )
            self._line_filter.update(lines, confidences)
            if self._follow_field:
                start = time.perf_counter()
                feet = np.array(
                    [
                        [(d["bbox"][0] + d["bbox"][2]) / 2.0, d["bbox"][3]]
                        for d in self.detections
                        if d["class_name"] == "player"
                    ]
                ).reshape(-1, 2)
                self._field_follower.update(self.field_results, frame_shape, lines, mask, feet)
                self._record("Field Estimate", start)

    # ------------------------------------------------------------------ rendering

    def _draw_main_view(self, frame: np.ndarray, options: PipelineOptions) -> np.ndarray:
        """Draw the overlays for the current results on a frame the caller owns."""
        if not self._shot.wide:
            draw_notice(frame, "Close-up: analysis paused")
            draw_fps_overlay(frame, self.fps)
            return frame
        if not (options.field_segmentation or self.detections or self.tracks):
            draw_fps_overlay(frame, self.fps)
            return frame

        start = time.perf_counter()

        # Field first, as background for the players
        if self.field_results and options.field_segmentation:
            if get_setting("models.segmentation.show_raw_masks", True):
                frame = draw_field_segmentation(frame, self.field_results)
            if self._geometry.mask is not None:
                frame = self._draw_field_overlay(frame, self._geometry.mask)

        # Plain detections only without tracking; tracks carry the same boxes
        if self.detections and not options.tracking:
            frame = draw_detections(frame, self.detections, in_place=True)
        elif self.tracks and options.tracking:
            track_histories = self._trails()
            if options.player_id:
                frame = draw_tracks_with_player_ids(
                    frame, self.tracks, track_histories, self._shown_numbers(), in_place=True
                )
            else:
                frame = draw_tracks(frame, self.tracks, track_histories, in_place=True)
            draw_possession(
                frame,
                self.tracks,
                self._possession.holder_id,
                self._possession.holder_box,
                self._possession_colour(),
            )

        draw_fps_overlay(frame, self.fps)
        if (
            options.player_id
            and self.player_ids
            and get_setting("models.player_id.show_reading_table", False)
        ):
            draw_jersey_table(frame)

        self._record("Visualization", start)
        return frame

    def _draw_field_overlay(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """The field outline and lines, drawn before the players."""
        frame = draw_unified_field_mask(
            frame,
            mask,
            get_primary_field_color(),
            alpha=0.3,
            draw_contour=not self._follow_field,
            fill_mask=False,
            # The fitted lines as shown: filtered
            ransac_fit=(
                (self.ransac_lines, *self._geometry.ransac_fit[1:])
                if self._geometry.ransac_fit is not None
                else None
            ),
            field_contour=self._geometry.contour,
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
        if not self._shot.wide:
            return None, "Close-up: no top-down view"
        height, width = frame.shape[:2]
        output_width, output_height = output_canvas_size(width, height)
        from_field = options.top_down_source == "field"
        if from_field:
            if not options.field_segmentation:
                return None, "The top-down view from the field needs field segmentation"
            if self._field_follower.image_to_field is None:
                return None, "The field has not been found in this view yet"
            on_canvas = field_to_canvas(
                self._field_follower.template, (output_width, output_height)
            )
            top_down_matrix = on_canvas @ self._field_follower.image_to_field
        else:
            top_down_matrix = self._top_down_matrix()
            if top_down_matrix is None:
                return None, "Homography matrix not available"

        if from_field and get_setting("homography.field_diagram", True):
            start = time.perf_counter()
            view = self._draw_field_diagram(on_canvas, (output_width, output_height), options)
            self._record("Top-down view", start)
            return view, ""

        try:
            start = time.perf_counter()

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
            if from_field:
                hide_behind_camera(view, matrix)
                draw_field_template(
                    view,
                    list(self._field_follower.template.lines.values()),
                    np.diag([scale, scale, 1.0]) @ on_canvas,
                )
            warp_ms = self._record("Homography Calculation", start)

            if self.field_results and options.field_segmentation:
                try:
                    with_field = apply_segmentation_to_warped_frame(
                        view,
                        self.field_results,
                        matrix,
                        frame.shape[:2],
                        "PIPELINE",
                        field_contour=self._geometry.contour,
                        draw_scale=scale,
                        in_place=True,
                    )
                    if with_field is not None:
                        view = with_field
                except Exception as e:
                    logger.error(f"Error applying segmentation to top-down view: {e}")

            if options.tracking and self.tracks:
                disc_in_view = None
                if from_field and self.disc_place is not None:
                    placed = np.diag([scale, scale, 1.0]) @ on_canvas @ [*self.disc_place, 1.0]
                    disc_in_view = (placed[0] / placed[2], placed[1] / placed[2])
                view = draw_tracks_top_down(
                    view,
                    matrix,
                    self.tracks,
                    self.player_ids,
                    self._trails(),
                    scale,
                    holder_id=self._possession.holder_id,
                    disc_position=disc_in_view,
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
