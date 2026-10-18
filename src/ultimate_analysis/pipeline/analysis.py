"""The per-frame analysis pipeline.

A frame goes through detection, tracking, possession, jersey number reading, and field
segmentation.
The results are drawn onto the camera view and onto the top-down view. There is no Qt in
this module: the GUI owns one AnalysisPipeline and runs it on a worker thread.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..processing import health
from ..processing.camera_motion import CameraMotionEstimator
from ..processing.camera_motion import is_enabled as camera_motion_enabled
from ..processing.detector_process import detect as detect_players_and_discs
from ..processing.disc_flight import DiscPath, FlyingDisc, place_under_disc
from ..processing.disc_fusion import fuse_discs, weak_discs_from
from ..processing.disc_fusion import is_enabled as fusion_enabled
from ..processing.disc_heatmap import peaks_of
from ..processing.field_registration import OffFieldWatcher
from ..processing.field_segmentation import reset_segmentation_cache
from ..processing.game_state import GameStateTracker, Player
from ..processing.inference import reset_inference_state
from ..processing.model_lock import GPU_SETUP_LOCK
from ..processing.possession import PossessionTracker, players_at_disc
from ..processing.shot_type import ShotWatcher, players_are_elsewhere
from ..processing.steady_places import SteadyPlaces
from ..processing.tracking import PlayerTracker
from ..rendering.overlays import draw_notice
from ..rendering.tracks import team_display_colour
from ..utils.field_camera import camera_of, raised_mapping
from ..utils.logger import get_logger
from .field import FieldStage
from .identity import IdentityStage
from .views import Scene, draw_camera_view, draw_top_down_view, seen_by_camera

logger = get_logger("PIPELINE")

# Two team colours (BGR) nearer than this are one team's, seen before and after a cut
SAME_TEAM_COLOUR = 90.0


def _exchange_names(rename: Callable[[int, int], None], first: int, second: int) -> None:
    """Exchange two names with a `rename(was, now)` that takes one at a time."""
    between = -1
    for was, now in ((first, between), (second, first), (between, second)):
        rename(was, now)


# The field's place is followed by the camera's motion alone for this long at most,
# however little the camera moves
FIELD_UNSEEN_SECONDS = 20.0
# The table of the players' statistics is handed out with every so many frames
STATS_EVERY_FRAMES = 30
# A heat map that is only shown is worked out about this often (a second of video)
HEAT_SHOWN_A_SECOND = 30.0

Line = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class PipelineOptions:
    """Which stages run for a frame."""

    detection: bool = True
    tracking: bool = True
    player_id: bool = True
    field_segmentation: bool = True
    top_down_view: bool = True
    # Whether the views are drawn. Without, the camera view of a result is the frame as
    # it came and there is no top-down view: for a run that only reads the results
    draw: bool = True

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
    # Track -> where the player stands on the field (x across, y along), held steady
    # over the frames; empty where the field's place is not known
    places: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    # The same for the observers, who are followed but are not among the tracks
    observers: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    # The outline on the field of what the frame shows, (N, 2) in field units; None
    # where the field's place is not known
    seen_by_camera: Optional[np.ndarray] = None
    # The phase of the game: "unknown", "between points", "lined up", "pull", "live" or
    # "score" (processing/game_state.py)
    game_state: str = "unknown"
    # Points begun so far (pulls seen), how long the present one has been played (None
    # between points), and the scores seen: (a team's colour (BGR), its points) each
    points_begun: int = 0
    point_seconds: Optional[float] = None
    score: Tuple[Tuple[Tuple[int, int, int], int], ...] = ()


class AnalysisPipeline:
    """Runs the enabled analysis stages on a frame and renders the results.

    Tracking, possession, and jersey numbers build on earlier frames, so frames must arrive in playback
    order; call reset() after a seek, a video change, or a model change.
    """

    FPS_WINDOW = 30  # Frames the displayed processing rate is averaged over

    def __init__(self):
        # For what can be done beside the main work of a frame
        self._threads = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline")
        self._timings: Dict[str, float] = {}
        # Where the field model sees the field: the top-down view and every place on it
        self._field = FieldStage(self._threads, self._record)
        self._off_field = OffFieldWatcher(self._field.template)
        self._steady = SteadyPlaces()
        self.places: Dict[int, Tuple[float, float]] = {}
        self.observers: Dict[int, Tuple[float, float]] = {}
        template = self._field.template
        self._game = GameStateTracker(template.length, template.end_zone)
        self._score: List[list] = []  # [a team's colour (BGR), its points]
        self._disc_path = DiscPath(float(get_setting("homography.disc_path_seconds", 8.0)))
        # Whether the disc is seen to fly: the holder network is not asked meanwhile
        self._flying = FlyingDisc()
        self.disc_seen_flying = False
        self._ahead: Optional[Any] = None  # The detectors' work on the next frame
        # ... and with it how the camera moved to that frame, looked at but not taken in
        self._motion_ahead: Optional[tuple] = None
        # Discs the detector is not sure of, and what the discs of this frame were made
        # of (processing/disc_fusion.py; None where they are the detector's alone)
        self._weak_discs: List[Dict[str, Any]] = []
        self._disc_sensors: Optional[Dict[str, Any]] = None
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
        # The frame analysed before, and how many video frames lie between it and this
        # one: playback passes over frames to keep up
        self._last_index: Optional[int] = None
        self._frames_passed = 1

        # Results for the current frame
        self.detections: List[Dict[str, Any]] = []
        self.tracks: List[Any] = []

        self._possession = PossessionTracker()
        # The players and the disc followed from frame to frame, and who the players are
        self._tracker = PlayerTracker()
        self._identity = IdentityStage(self._tracker, self._rename_boxes, self._exchange_boxes)

        # A network that tells from a picture of a player whether they hold the disc:
        # its weights (None: not used), and the network once it is loaded
        weights = str(get_setting("models.holder.weights") or "")
        self._holder_weights: Optional[str] = weights or None
        self._holder_net: Any = None
        # Whether the network's guess is drawn on the camera view (it then looks at
        # every player in every frame), and what it said of the players of this frame
        self.show_holder_guess = bool(get_setting("models.holder.show_guess", True))
        self.holder_scores: Optional[Dict[int, float]] = None
        # What the network says of every player, for showing: worked out beside the
        # frames (track IDs asked about, the work), and so of the frame before
        self.holder_guess: Optional[Dict[int, float]] = None
        self._guess_job: Optional[tuple] = None
        # Whether a heat map of where a disc is likely is drawn on the camera view
        # (processing/disc_heatmap.py; a network of its own, run only while shown), the
        # network once it has been asked for (False: it does not load), and the map of
        # this frame
        self.show_disc_heat = bool(get_setting("models.disc_heatmap.show", False))
        self._disc_heatmap: Any = None
        self.disc_heat: Optional[np.ndarray] = None
        # The map that is drawn: this frame's, or the last one worked out
        self.disc_heat_shown: Optional[np.ndarray] = None
        self._heat_frames = 0
        self._holder_loading = threading.Lock()
        self._camera_motion = CameraMotionEstimator()
        # Close-ups and title cards of an edited game are not analysed
        self._shot = ShotWatcher()

        # Redrawing the frame that was just analysed (pause, toggling an overlay) must not
        # advance the tracker again, so its results are reused.
        self._analysed_key: Optional[tuple] = None

        self._frame_times_ms: List[float] = []
        self.fps = 0.0

    # ------------------------------------------------------------------ state

    @property
    def player_ids(self) -> Dict[int, Tuple[str, Any]]:
        """Player ID -> (jersey number or "Unknown", details of the reading)."""
        return self._identity.player_ids

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
        self._field.new_video(video_path)
        self._identity.new_video(video_path)

    def reset(self) -> None:
        """Forget everything derived from earlier frames."""
        self._field.reset()
        self._off_field.reset()
        self._steady.reset()
        self.places, self.observers = {}, {}
        self.holder_guess, self._guess_job = None, None  # Of tracks that are no more
        self._game.cut()
        self._forget_side_work()
        self.disc_place = None
        self._disc_path.reset()
        self._flying.reset()
        self._identity.reset()
        self._shot.reset()
        self._tracker.reset()
        reset_inference_state()
        reset_segmentation_cache()
        self._possession.reset()
        if self._disc_heatmap:
            self._disc_heatmap.reset()
        self.disc_heat = self.disc_heat_shown = None
        self._camera_motion.reset()
        self._motion_known = False
        self._last_index = None  # After a seek no frames were passed over
        self.invalidate()
        self.detections = []
        self.tracks = []

    def _forget_side_work(self) -> None:
        """What the other threads were working out is of the view before."""
        if self._ahead is not None:
            self._ahead.result()  # Waited for: its frame must not be drawn on meanwhile
            self._ahead = None

    def set_frame_rate(self, frames_per_second: float) -> None:
        """Tell the pipeline the frame rate of the video; durations are set in seconds."""
        self._tracker.set_frame_rate(frames_per_second)
        if frames_per_second and frames_per_second > 0:
            self._possession.frame_rate = float(frames_per_second)
            self._identity.looks.frame_rate = float(frames_per_second)
            self._identity.frame_rate = float(frames_per_second)
            self._field.follower.max_frames_unseen = round(
                FIELD_UNSEEN_SECONDS * float(frames_per_second)
            )

    def set_holder_model(self, weights: Optional[str]) -> None:
        """Tell who holds a disc that is in two players' boxes with a network that looks
        at the players (None: by where the disc is alone)."""
        if weights != self._holder_weights:
            self._holder_weights, self._holder_net = weights or None, None
            self.holder_guess, self._guess_job = None, None
            self._analysed_key = None

    def _holder_scores(self, frame: np.ndarray, frame_index: int) -> Optional[Dict[int, float]]:
        """How sure the holder network is that each player holds the disc, for possession:
        of the players at a disc that is in the boxes of two or more, and, where the
        network stands in for a disc that is not found (models.holder.sure), of everybody
        in a frame without one. None where it has nothing to say, or there is none, and
        while the disc is seen to fly: the network looks at one player at a time and
        cannot know that nobody holds the disc.

        The guess that is shown for every player (`holder_guess`) is worked out beside
        the frames and is of the frame before: nothing waits for what is only drawn."""
        players = [t for t in self.tracks if t.class_name == "player" and hasattr(t, "to_ltrb")]
        discs = [d for d in self.detections if d["class_name"] == "disc"]
        self.disc_seen_flying = self._flying.update(
            frame_index / self._possession.frame_rate, discs, [t.to_ltrb() for t in players]
        )
        if self._holder_weights is None or (
            self.disc_seen_flying and get_setting("models.holder.not_while_flying", True)
        ):
            self.holder_guess, self._guess_job = None, None
            return None
        margin = float(get_setting("models.possession.box_margin", 0.15))
        asked: Dict[int, Any] = {}
        for detection in discs:
            at = set(players_at_disc(detection["bbox"], self.tracks, margin))
            if len(at) > 1:
                asked.update({t.track_id: t for t in players if t.track_id in at})
        # Without a disc, or with one that may be a thing left lying: everybody
        # (or in every frame, for a recording that is to be played back with other
        # settings: scripts/benchmark_possession.py)
        if get_setting("models.holder.ask_everybody", False) or (
            float(get_setting("models.holder.sure", 0.0)) > 0
            and (not discs or self._possession.strays)
        ):
            asked.update({t.track_id: t for t in players})
        try:
            from ..processing.holder import BOX_ALONE, crop_around

            scores = None
            if asked:
                crops = [crop_around(frame, t.to_ltrb(), BOX_ALONE)[0] for t in asked.values()]
                scores = dict(zip(asked, self._holding(crops)))
            if self.show_holder_guess:
                job = self._guess_job
                if job is not None and job[1].done():
                    self.holder_guess, self._guess_job = dict(zip(job[0], job[1].result())), None
                if self._guess_job is None and players:
                    # The frame is drawn on before the network gets to it: its own copies
                    crops = [crop_around(frame, t.to_ltrb(), BOX_ALONE)[0].copy() for t in players]
                    self._guess_job = (
                        [t.track_id for t in players],
                        self._threads.submit(self._holding, crops),
                    )
                if scores and self.holder_guess is not None:
                    self.holder_guess.update(scores)  # What possession went by, at once
            else:
                self.holder_guess, self._guess_job = None, None
        except Exception as error:  # No PyTorch, or weights that do not load
            logger.exception(f"The holder network failed: {error}")
            health.report("Possession", "the holder network failed; going by the disc alone")
            self._holder_weights, self.holder_guess, self._guess_job = None, None, None
            return None
        return scores

    def _holding(self, crops: List[np.ndarray]) -> List[float]:
        """What the holder network says of some crops; also run beside the frames."""
        from ..processing.holder import holding, load_holder_net

        with GPU_SETUP_LOCK.work():
            with self._holder_loading:
                if self._holder_net is None:
                    self._holder_net = load_holder_net(Path(self._holder_weights), compiled=True)
                network = self._holder_net
            return [float(score) for score in holding(network, crops)]

    def set_looks_model(self, weights: Optional[str]) -> None:
        """Know the players by their looks with another network (None: the one in the
        settings). The tracks go on; what they look like is learned anew."""
        if self._identity.set_looks_model(weights):
            self._analysed_key = None

    def reset_player_ids(self) -> None:
        """Forget the jersey numbers read so far (e.g. after switching the reader)."""
        self._identity.numbers.reset(readings_too=True)
        self._analysed_key = None

    def reset_fps(self) -> None:
        """Start the processing rate average again (new video)."""
        self._frame_times_ms.clear()
        self.fps = 0.0

    def invalidate(self) -> None:
        """Analyse the current frame again on the next call instead of reusing results."""
        self._analysed_key = None
        self._field.invalidate()

    # ------------------------------------------------------------------ per frame

    def process(
        self,
        frame: np.ndarray,
        frame_index: int,
        options: PipelineOptions,
        frame_is_ours: bool = False,
        next_frame: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ) -> FrameResult:
        """Analyse a frame and render the camera and top-down views.

        Args:
            frame: Video frame (BGR); it is not modified, unless `frame_is_ours`
            frame_index: Position of the frame in the video
            options: Stages to run
            frame_is_ours: The caller has no further use for the frame: the camera view
                may be drawn on it, which saves a copy of every frame
            next_frame: Gives the frame that will be processed next, if the caller knows
                it: its players and disc are then looked for while this frame is being
                followed and drawn. The very array has to be passed as `frame` next.

        Returns:
            The rendered views, the analysis results, and the time each stage took
        """
        start = time.perf_counter()
        self._timings = {}
        self._trails_now = None
        # Also without the top-down view shown: who stands beside the field should not
        # depend on a view being switched on
        self._follow_field = options.field_segmentation
        self._frame_size = (frame.shape[1], frame.shape[0])

        analysis_key = (frame_index, *options.analysis_key)
        if analysis_key != self._analysed_key:
            health.start_frame()
            self._analyse(frame, frame_index, options, next_frame)
            self._analysed_key = analysis_key

        # The two views are drawn side by side, from what this frame's analysis left
        scene = self._scene(options) if options.draw else None
        side_by_side = bool(get_setting("app.draw_views_side_by_side", True))
        top_down = None
        if options.top_down_view and side_by_side and options.draw:
            top_down = self._threads.submit(draw_top_down_view, scene, options, self._record)

        # The copy of the frame that is drawn on belongs to the drawing. None is needed
        # of a frame that is ours.
        main_view = frame
        if options.draw:
            copy_start = time.perf_counter()
            drawn_on = frame if frame_is_ours else frame.copy()
            self._record("Visualization", copy_start)
            main_view = draw_camera_view(drawn_on, scene, options, self._record)
        problems = health.problems()
        for line, problem in enumerate(problems if options.draw else ()):
            draw_notice(main_view, problem, line=line + 1, alarm=True)

        top_down_view, top_down_message = None, "Homography view disabled"
        if top_down is not None:
            top_down_view, top_down_message = top_down.result()
        elif options.top_down_view and options.draw:
            top_down_view, top_down_message = draw_top_down_view(scene, options, self._record)

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
            numbers_by_looks=dict(self._identity.numbers_by_looks),
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
            image_to_field=self._field.follower.image_to_field if self._follow_field else None,
            disc_place=self.disc_place,
            places=self.places,
            observers=self.observers,
            seen_by_camera=seen_by_camera(
                self._field.follower.image_to_field, self._frame_size, self._field.template
            )
            if self._follow_field and self._field.follower.image_to_field is not None
            else None,
            timings=self._timings,
            wide_shot=self._shot.wide,
            problems=problems,
        )

    def _scene(self, options: PipelineOptions) -> Scene:
        """What the views of this frame are drawn from, taken here on the frame's own
        thread: the top-down view is drawn beside it on another."""
        tracking = options.tracking and bool(self.tracks)
        return Scene(
            template=self._field.template,
            frame_size=self._frame_size,
            wide=self._shot.wide,
            fps=self.fps,
            detections=list(self.detections),
            tracks=list(self.tracks),
            trails=self._trails() if tracking else {},
            numbers=self._identity.shown_numbers(),
            read_numbers=bool(self.player_ids),
            holder_id=self._possession.holder_id,
            holder_box=self._possession.holder_box,
            possession_colour=self._possession_colour(),
            holder_guess=dict(self.holder_guess) if self.holder_guess else self.holder_guess,
            disc_heat=self.disc_heat_shown,
            field_results=self._field.results,
            field_mask=self._field.mask,
            image_to_field=self._field.follower.image_to_field,
            places=dict(self.places),
            disc_place=self.disc_place,
            disc_path=list(self._disc_path.stretches()) if options.tracking else [],
            show_holder_guess=self.show_holder_guess,
            show_disc_heat=self.show_disc_heat,
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
        to_field = self._field.follower.image_to_field
        if to_field is None:
            return
        state = self._possession.disc_state
        ground = raised = holder = camera = None
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
                    self._field.follower.focal,
                )
                raised = place_under_disc(ground, camera)
            except (np.linalg.LinAlgError, cv2.error):
                pass
        if state == "air":
            self.disc_place = raised
        for track in self.tracks:
            if state == "held" and track.track_id == self._possession.holder_id:
                # Where the player is shown to stand, if that is known: held steady,
                # and right for one whose legs are covered
                if track.track_id in self.places:
                    holder = np.array(self.places[track.track_id])
                    continue
                x1, _, x2, y2 = track.to_ltrb()
                feet = to_field @ [(x1 + x2) / 2, y2, 1.0]
                if abs(feet[2]) > 1e-12:
                    holder = feet[:2] / feet[2]
        # The path of the disc, put right once it is known what the disc was doing
        rate = self._possession.frame_rate
        self._disc_path.add(
            frame_index / rate,
            state,
            ground,
            raised,
            holder,
            self._possession.since / rate,
            camera,
        )

    def _follow_game(self, frame_index: int) -> None:
        """Work out the phase of the game from where the players stand on the field."""
        to_field = self._field.follower.image_to_field
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
            shirt = self._identity.team_colours().get(event.team) if event.kind == "score" else None
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

    def _trails(self) -> Dict[int, np.ndarray]:
        """The trails of the tracks in this frame: both views draw them."""
        if self._trails_now is None:
            self._trails_now = self._tracker.get_track_histories()
        return self._trails_now

    def _possession_colour(self) -> Optional[Tuple[int, int, int]]:
        """The colour that stands for the team in possession, None while it is not known."""
        shirt = self._identity.team_colours().get(self._possession.team)
        return None if shirt is None else team_display_colour(shirt)

    def _update_fps(self, frame_time_ms: float) -> None:
        self._frame_times_ms.append(frame_time_ms)
        if len(self._frame_times_ms) > self.FPS_WINDOW:
            self._frame_times_ms.pop(0)
        average_ms = sum(self._frame_times_ms) / len(self._frame_times_ms)
        self.fps = 1000.0 / average_ms if average_ms > 0 else 0.0

    # ------------------------------------------------------------------ analysis

    def _detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """The players and discs of a frame: found ahead, if it was this frame that was
        looked at ahead, or found now."""
        ahead, self._ahead = self._ahead, None
        self._motion_ahead = None
        if ahead is not None:
            # Waited for also when it is of another frame (after a seek, say)
            looked_at, found, motion = ahead.result()
            if looked_at is frame:
                self._motion_ahead = motion
                return found
        return detect_players_and_discs(
            frame,
            weak_discs_from=weak_discs_from(),
            frames_passed=self._frames_passed,
        )

    def _detect_ahead(
        self, next_frame: Callable[[], Optional[np.ndarray]], with_motion: bool
    ) -> tuple:
        frame = next_frame()
        if frame is None:
            return None, [], None
        # The next frame lies as far on as this one did from the one before
        found = detect_players_and_discs(
            frame,
            weak_discs_from=weak_discs_from(),
            frames_passed=self._frames_passed,
        )
        motion = None
        if with_motion:
            # How the camera moved to that frame needs the frame and its boxes and
            # nothing else: looked at here, taken in when it is the frame's turn
            boxes = [detection["bbox"] for detection in found if not detection.get("weak")]
            try:
                motion = self._camera_motion.look(frame, boxes)
            except cv2.error:
                motion = None  # Worked out again, and failing again, on the frame's turn
        return frame, found, motion

    def _begin_ahead(
        self, next_frame: Optional[Callable[[], Optional[np.ndarray]]], options: PipelineOptions
    ) -> None:
        """Put the next frame before the detectors while this one is followed and
        drawn: by the time it is that frame's turn, they are done."""
        if next_frame is not None and get_setting("models.inference.detect_ahead", True):
            self._ahead = self._threads.submit(
                self._detect_ahead, next_frame, self._follows_camera(options)
            )

    @staticmethod
    def _follows_camera(options: PipelineOptions) -> bool:
        return (options.tracking or options.top_down_view) and camera_motion_enabled()

    def _look_for_disc_heat(self, frame: np.ndarray, motion: Optional[np.ndarray]) -> None:
        """Work out the heat map of where a disc is likely (`self.disc_heat`), if it is
        shown or the discs are found with it. `motion`: how the camera moved to this
        frame, None if not known."""
        self.disc_heat, before = None, self.disc_heat_shown
        self.disc_heat_shown = None
        wanted = self.show_disc_heat or fusion_enabled()
        if not wanted or self._disc_heatmap is False:
            if self._disc_heatmap:
                self._disc_heatmap.reset()  # Not fed meanwhile: its frames are old
            return
        start = time.perf_counter()
        try:
            self.disc_heat_shown = before
            if self._disc_heatmap is None:
                from ..processing.disc_heatmap import load_disc_heatmap

                self._disc_heatmap = load_disc_heatmap(get_setting("models.disc_heatmap.weights"))
                if self._disc_heatmap is None:
                    self._disc_heatmap = False
                    health.report("Disc heat map", "no network to show; see the log")
                    return
            # For finding the disc it is asked only where it can change which disc is
            # gone by: not with one disc the detector is sure of, half the frames. On
            # the 447 labelled frames that costs 2 of the 16 it wins. For showing, it
            # is asked about thirty times a second, and the map of the frame before
            # stays up in between.
            sure = sum(1 for d in self.detections if d["class_name"] == "disc")
            needed = fusion_enabled() and sure != 1
            every = max(1, round(self._possession.frame_rate / HEAT_SHOWN_A_SECOND))
            self._heat_frames += 1
            shown = self.show_disc_heat and self._heat_frames % every == 0
            before = self.disc_heat_shown
            self.disc_heat = self._disc_heatmap.update(
                frame, motion, self._possession.frame_rate, needed or shown
            )
            if self.show_disc_heat:
                self.disc_heat_shown = self.disc_heat if needed or shown else before
        except Exception as error:
            logger.exception(f"The disc heat map failed: {error}")
            health.report("Disc heat map", "failed; see the log")
            self._disc_heatmap = False
        self._record("Disc Heat Map", start)

    def _put_discs_together(self, frame: np.ndarray) -> None:
        """The discs of the frame from the detector and the heat map together, in place
        of the detector's alone (processing/disc_fusion.py), if that is switched on."""
        if not fusion_enabled():
            return
        discs = [d for d in self.detections if d["class_name"] == "disc"]
        players = [d for d in self.detections if d["class_name"] == "player"]
        heights = [d["bbox"][3] - d["bbox"][1] for d in players]
        self._disc_sensors = {
            "discs": discs,
            "weak": self._weak_discs,
            "peaks": peaks_of(self.disc_heat, (frame.shape[1], frame.shape[0])),
            "height": float(np.median(heights)) if heights else 80.0,
        }
        others = [d for d in self.detections if d["class_name"] != "disc"]
        self.detections = others + fuse_discs(**self._disc_sensors)

    def _camera_motion_of(self, frame: np.ndarray, boxes: List[Any]) -> Optional[np.ndarray]:
        """How the camera moved to this frame: as looked at ahead, or worked out now."""
        looked, self._motion_ahead = self._motion_ahead, None
        if looked is not None and self._camera_motion.take(looked[1]):
            return looked[0]
        return self._camera_motion.update(frame, boxes)

    def _analyse(
        self,
        frame: np.ndarray,
        frame_index: int,
        options: PipelineOptions,
        next_frame: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ) -> None:
        """Run the enabled stages and store their results."""
        self.detections = []
        self._weak_discs, self._disc_sensors = [], None
        self.tracks = []
        self._field.results = []
        passed = frame_index - self._last_index if self._last_index is not None else 1
        # Anything else is a seek or the same frame again
        self._frames_passed = passed if 0 < passed <= self._possession.frame_rate else 1
        self._last_index = frame_index

        if options.detection:
            start = time.perf_counter()
            found = self._detect(frame)
            # Discs the detector is not sure of are for putting the discs together alone
            self.detections = [detection for detection in found if not detection.get("weak")]
            self._weak_discs = [d for d in found if d.get("weak") and d["class_name"] == "disc"]
            self._record("Inference", start)
            if get_setting("models.shot_type.enabled", True) and not self._watch_shot():
                self._begin_ahead(next_frame, options)
                return  # A close-up or a title card: nothing to follow
            if not self._follows_camera(options):
                self._begin_ahead(next_frame, options)

        if self._follows_camera(options):
            # Before tracking adds this frame's positions: the trails from earlier frames
            # follow the picture when the camera moves, and so does the top-down calibration
            start = time.perf_counter()
            boxes = [detection["bbox"] for detection in self.detections]
            motion = self._camera_motion_of(frame, boxes)
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
            if options.detection:
                # The next frame goes before the detectors only now: what they work out
                # of the camera's motion starts from this frame's, and a cut forgets it
                self._begin_ahead(next_frame, options)
            self._field.camera_moved(motion)
            if motion is not None:
                self._tracker.apply_camera_motion(motion)
                self._possession.move(motion)
                self._flying.move(motion)
            self._record("Camera Motion", start)
            self._look_for_disc_heat(frame, motion)
        else:
            self._look_for_disc_heat(frame, np.eye(3))  # As if the camera stood still
        self._put_discs_together(frame)

        if options.tracking:
            start = time.perf_counter()
            self._tracker.set_field(
                self._field.follower.image_to_field if self._follow_field else None
            )
            self.tracks = self._tracker.run_tracking(frame, self.detections, frame_index)
            self._identity.teams_by_looks(frame, self.tracks)
            if self._follow_field and get_setting("models.tracking.hide_off_field", True):
                # Those standing along the sidelines are followed but not shown or counted
                # ... judged only by a field seen lately: followed blind through a swing
                # of the camera, it puts players on the field beside it
                off_field = self._off_field.update(
                    self._field.follower.image_to_field
                    if self._field.follower.seen_lately
                    else None,
                    self.tracks,
                    bool(get_setting("models.tracking.hide_newcomers_off_field", False)),
                )
                self.tracks = [track for track in self.tracks if track.track_id not in off_field]
            # Players and observers alike: their numbers are the one tracker's
            watching = self._tracker.observers_in_view()
            feet = {
                track.track_id: ((track.bbox[0] + track.bbox[2]) / 2.0, track.bbox[3])
                for track in self.tracks
                if track.class_name == "player"
            }
            feet.update({who: ((box[0] + box[2]) / 2.0, box[3]) for who, box in watching.items()})
            to_field = self._field.follower.image_to_field if self._follow_field else None
            boxes = camera = None
            if to_field is not None and get_setting("models.tracking.covered_feet", True):
                boxes = {t.track_id: t.bbox for t in self.tracks if t.class_name == "player"}
                boxes.update(watching)
                try:
                    camera = raised_mapping(
                        np.linalg.inv(to_field),
                        (frame.shape[1], frame.shape[0]),
                        self._field.follower.focal,
                    )
                except (np.linalg.LinAlgError, cv2.error):
                    pass  # By the players' own boxes alone
            steady = self._steady.update(
                frame_index / self._possession.frame_rate, feet, to_field, boxes, camera
            )
            self.observers = {who: place for who, place in steady.items() if who in watching}
            self.places = {who: place for who, place in steady.items() if who not in watching}
            self._record("Tracking", start)
            # What is worked out from the tracks: who has the disc, where it flies, and
            # the phase of the game
            start = time.perf_counter()
            self.holder_scores = self._holder_scores(frame, frame_index)
            self._record("Holder network", start)
            start = time.perf_counter()
            self._possession.update(
                self.detections, self.tracks, frame_index, self.holder_scores, self._disc_sensors
            )
            self._follow_flight(frame.shape[:2], frame_index)
            self._follow_game(frame_index)
            self._record("Possession", start)
        else:
            self._possession.reset()

        if options.field_segmentation:
            self._field.update(frame, frame_index, self.detections, self._follow_field)

        if options.player_id and self.tracks:
            timing = self._identity.read_numbers(frame, self.tracks, frame_index)
            if timing["preprocessing_ms"] > 0 or timing["ocr_ms"] > 0:
                self._timings["Player ID - Preprocessing"] = timing["preprocessing_ms"]
                self._timings["Player ID - Reading"] = timing["ocr_ms"]
            if timing.get("filtering_ms", 0) > 0:
                self._timings["Player ID - Jersey Number Filtering"] = timing["filtering_ms"]
        elif not options.player_id:
            self._identity.player_ids.clear()

        self._identity.numbers_by_looks = {}
        if options.tracking and self.tracks and get_setting("models.reid.enabled", True):
            start = time.perf_counter()
            self._identity.look(
                frame,
                self.tracks,
                frame_index,
                self._possession.holder_id,
                self._places_on_field(),
                self.places,
            )
            self._record("Player ID - Looks", start)

    def _with_the_box(self) -> Tuple[Any, ...]:
        """What is kept per track and belongs to the box, not to the player's name: who
        has the disc, who stands beside the field, where each stands on it. Each has a
        rename(was, now)."""
        return (self._possession, self._off_field, self._steady)

    def _rename_boxes(self, was: int, now: int) -> None:
        """A track is known under another ID from now on."""
        for watcher in self._with_the_box():
            watcher.rename(was, now)

    def _exchange_boxes(self, first: int, second: int) -> None:
        """Two tracks exchanged their IDs in a crossing."""
        for watcher in self._with_the_box():
            _exchange_names(watcher.rename, first, second)

    def _places_on_field(self) -> Dict[int, np.ndarray]:
        """Track -> where the player stands on the field, if the field's place is known."""
        to_field = self._field.follower.image_to_field if self._follow_field else None
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

    def players_of_tracks(self) -> Dict[int, int]:
        """Track -> player, looking back over the tracks since the last cut."""
        return self._identity.players_of_tracks()

    def player_stats(self) -> List[dict]:
        """What is known of each player of the video and what they did: a row each."""
        return self._identity.player_stats()

    def _watch_shot(self) -> bool:
        """Whether this frame is drone footage; what was followed is dropped when it stops.

        After a cut away, the players are others or elsewhere when the drone is back, and
        the camera has moved in between.
        """
        if health.failed("Detection"):
            return self._shot.wide  # No detections because of a failure say nothing
        was_wide = self._shot.wide
        players = sum(1 for detection in self.detections if detection["class_name"] == "player")
        # Of the frame before: this frame's camera motion is not worked out yet
        wide = self._shot.update(players, continuous=self._motion_known, frames=self._frames_passed)
        if was_wide and not wide:
            detections = self.detections
            self.reset()
            self.detections = detections
            self._shot.pause()
        return wide
