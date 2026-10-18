"""Tracking of players and discs across video frames.

ByteTrack follows boxes by their motion and keeps each player track within its team
(team_tracker.py); with models.tracking.backend "ground" a lost player is also found
again by their place on the field (ground_tracker.py). On top of it, the identity layer
keeps a player's ID when the tracker loses them and picks them up again as a new track.
All of it is held by a `PlayerTracker`, one per pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config.settings import get_setting
from ..constants import TRACK_HISTORY_MAX_LENGTH
from ..utils.logger import get_logger
from . import appearance, health
from .ground_tracker import GroundTeamTracker
from .player_identity import Observation, PlayerIdentities
from .team_tracker import BYTETracker, DetectionBoxes, TeamTracker, tracker_settings

logger = get_logger("TRACKING")


# A trail's point further from the picture's corner than this is no longer in view
TRAIL_REACH_PIXELS = 20000.0
# Discs are not players; their track IDs are moved out of the way of the player IDs
DISC_ID_OFFSET = 100000


class Track:
    """Represents a tracked object with consistent identity."""

    def __init__(
        self,
        track_id: int,
        bbox: List[float],
        class_id: int,
        confidence: float,
        class_name: str = "unknown",
        model_type: str = "unknown",
    ):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.confidence = confidence
        self.class_name = class_name
        self.model_type = model_type  # Track which model detected this
        self.det_class = class_name  # For compatibility with existing code
        # The player's team (0 or 1) and its average shirt colour (BGR), once the tracker
        # knows them
        self.team: Optional[int] = None
        self.team_colour: Optional[Tuple[int, int, int]] = None
        # The team a young track's few sightings lean to, before the tracker settles it
        self.team_leaning: Optional[int] = None

    def to_ltrb(self) -> List[float]:
        """Return bounding box in [x1, y1, x2, y2] format."""
        return self.bbox


def evened_out(trail: np.ndarray, reach: int) -> np.ndarray:
    """A trail without the bobbing of a runner's stride.

    The trail follows the bottom of a player's box, and that goes up and down with every
    step: a foot on the ground, both in the air. Each point is replaced by the mean of
    the points up to `reach` before and after it, over a whole stride, so the bobbing
    cancels and the course stays. Towards either end as many points are taken as there
    are on both sides, so the newest point is left where the player's feet are.

    Args:
        trail: Positions (N, 2), oldest first
        reach: How many points before and after a point go into its mean
    """
    count = len(trail)
    if count < 3 or reach < 1:
        return trail
    sums = np.vstack([np.zeros((1, 2)), np.cumsum(trail.astype(np.float64), axis=0)])
    index = np.arange(count)
    half = np.minimum(reach, np.minimum(index, count - 1 - index))
    return (sums[index + half + 1] - sums[index - half]) / (2 * half + 1)[:, None]


class PlayerTracker:
    """Follows the players and the disc from frame to frame: the trackers, the trails,
    the identity layer, and what is known of each player's team. One per pipeline;
    frames go in in playback order, and `reset` after a seek or a cut."""

    def __init__(self) -> None:
        self._image_to_field: Optional[np.ndarray] = None  # Of the frame about to be tracked
        self._player_tracker: Optional[TeamTracker] = None  # Players, each within a team,
        self._disc_tracker: Optional[BYTETracker] = None  # and discs, which have no team
        # Track ID -> the trail of its feet, an array (N, 2) of picture positions. Kept as
        # fractions of a pixel: the camera motion moves every stored point every frame,
        # and rounding each time would let a trail drift.
        self._track_histories: Dict[int, np.ndarray] = {}
        self._history_last_frame: Dict[int, int] = {}
        # Frames per second of the video; how long a lost track is kept is set in seconds
        self._frame_rate = 30.0
        # Players keep their identity when the tracker loses them and picks them up again
        # as a new track. The IDs handed out are those of the players, not of the tracks.
        self._identities = PlayerIdentities()
        # Player ID -> the team (0 or 1) the tracker last gave them, also once missing
        self._player_teams: Dict[int, int] = {}
        # Track -> box of those followed in the last frame who are no players: observers
        # and others in neither team's colours, who are not among the tracks handed out
        self._observers: Dict[int, List[float]] = {}
        # The video frame analysed last, and how many video frames lie between two
        # analysed ones: live playback skips frames to keep up
        self._video_frame: Optional[int] = None
        self._frames_per_step = 1
        # The tracker's clock: video frames since the reset, however many were analysed;
        # and the calls since the reset (frames analysed)
        self._frame_count = 0
        self._calls = 0

    def _max_age_frames(self) -> int:
        """Video frames a track is kept without being detected.

        Half a second, the tracker's own default at 60 frames per second, loses a player who
        runs behind another one; the track then returns as somebody new.
        """
        seconds = float(get_setting("models.tracking.max_age_seconds", 3.0))
        return max(1, round(seconds * self._frame_rate))

    def _keep_lost(self, tracker: Any, frames: int) -> None:
        """Have a ByteTrack tracker keep a lost track for so many video frames. It counts its
        calls, not frames, so this is turned into calls at the present step."""
        tracker.keep_frames = frames
        tracker.max_frames_lost = tracker.args.track_buffer = max(
            1, round(frames / self._frames_per_step)
        )

    def _step_changed(self, before: int) -> None:
        """Every `_frames_per_step` video frames are analysed now, every `before` until now.

        ByteTrack counts calls: the time a lost track is kept is put right, and so is the
        speed each track is expected to move by from one call to the next (its Kalman state
        is per call), with its doubt.
        """
        ratio = self._frames_per_step / before
        for tracker in (self._player_tracker, self._disc_tracker):
            if tracker is None:
                continue
            self._keep_lost(tracker, getattr(tracker, "keep_frames", self._max_age_frames()))
            for track in tracker.tracked_stracks + tracker.lost_stracks:
                if track.mean is None:
                    continue
                track.mean[4:] *= ratio
                if track.covariance is not None:
                    track.covariance[4:, :] *= ratio
                    track.covariance[:, 4:] *= ratio

    def set_frame_rate(self, frames_per_second: float) -> None:
        """Tell the tracker the frame rate of the video its frames come from."""
        if frames_per_second and frames_per_second > 0:
            self._frame_rate = float(frames_per_second)
        for tracker in (self._player_tracker, self._disc_tracker):
            if tracker is not None:
                self._keep_lost(tracker, self._max_age_frames())

    def run_tracking(
        self, frame: np.ndarray, detections: List[Dict[str, Any]], frame_index: Optional[int] = None
    ) -> List[Track]:
        """Run object tracking on detected objects.

        Args:
            frame: Input video frame as numpy array (H, W, C) in BGR format
            detections: List of detection dictionaries from inference
            frame_index: The frame's number in the video, if known: frames may be skipped
                between calls, and the trails are kept for a time, not a number of calls

        Returns:
            List of Track objects with consistent IDs across frames

        Example:
            tracks = tracker.run_tracking(frame, detections)
            for track in tracks:
                track_id = track.track_id
                x1, y1, x2, y2 = track.to_ltrb()
        """
        self._calls += 1
        before = self._frames_per_step
        if frame_index is not None:
            skipped = frame_index - self._video_frame if self._video_frame is not None else 1
            # Anything else is a seek, after which the caller resets the tracker anyway
            self._frames_per_step = skipped if 0 < skipped <= self._frame_rate else 1
            self._video_frame = frame_index
        self._frame_count += self._frames_per_step
        if self._frames_per_step != before:
            self._step_changed(before)

        try:
            return self._run_bytetrack_tracking(frame, detections)
        except Exception as e:
            logger.exception(f"Error in ByteTrack tracking: {e}")
            health.report("Tracking", f"failed ({type(e).__name__}); players are not followed")
            return self._run_simple_tracking(detections)

    def _uses_the_ground(self) -> bool:
        return get_setting("models.tracking.backend", "bytetrack") == "ground"

    def set_field(self, image_to_field: Optional[np.ndarray]) -> None:
        """Where the field is in the frame about to be tracked (image -> field, yards), or
        None if that is not known. Only backend "ground" makes use of it."""
        self._image_to_field = image_to_field

    def _run_bytetrack_tracking(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> List[Track]:
        """Track players within their teams, and discs, by the motion of their boxes."""
        on_the_ground = self._uses_the_ground()
        if (
            self._player_tracker is None
            or self._disc_tracker is None
            or isinstance(self._player_tracker, GroundTeamTracker) != on_the_ground
        ):
            keep = self._max_age_frames()
            if on_the_ground:
                # Kept longer: where the field is known, a lost player is found by their place
                seconds = float(get_setting("models.tracking.ground_keep_seconds", 6.0))
                keep = max(keep, round(seconds * self._frame_rate))
                self._player_tracker = GroundTeamTracker(
                    tracker_settings(keep), self._max_age_frames() / self._frame_rate
                )
            else:
                self._player_tracker = TeamTracker(tracker_settings(keep))
            self._disc_tracker = BYTETracker(tracker_settings(self._max_age_frames()))
            self._keep_lost(self._player_tracker, keep)
            self._keep_lost(self._disc_tracker, self._max_age_frames())
        if on_the_ground:
            self._player_tracker.look(
                self._image_to_field, self._frames_per_step / self._frame_rate
            )

        tracks = []
        for class_name, class_id, tracker in (
            ("player", 1, self._player_tracker),
            ("disc", 0, self._disc_tracker),
        ):
            found = [
                detection
                for detection in detections
                if detection.get("class_name") == class_name and detection.get("bbox") is not None
            ]
            boxes = DetectionBoxes(
                [detection["bbox"] for detection in found],
                [detection["confidence"] for detection in found],
            )
            # A row is x1, y1, x2, y2, track ID, confidence, class, index of the detection
            rows = tracker.update(boxes, frame)
            # Observers and others in neither team's colours are followed but not shown
            hidden = (
                tracker.outsiders()
                if tracker is self._player_tracker
                and get_setting("models.tracking.hide_non_players", True)
                else ()
            )
            teams = tracker.teams_of_tracks() if tracker is self._player_tracker else {}
            leanings = tracker.leanings_of_tracks() if tracker is self._player_tracker else {}
            team_colours = tracker.shirt_colours() if tracker is self._player_tracker else {}
            if tracker is self._player_tracker:
                self._observers.clear()
            for row in rows:
                if int(row[4]) in hidden:
                    self._observers[int(row[4])] = [float(value) for value in row[:4]]
                    continue
                tracks.append(
                    Track(
                        track_id=int(row[4]),
                        bbox=[float(value) for value in row[:4]],
                        class_id=class_id,
                        confidence=float(row[5]),
                        class_name=class_name,
                        model_type=f"{class_name}_model",
                    )
                )
                tracks[-1].team = teams.get(int(row[4]))
                tracks[-1].team_leaning = leanings.get(int(row[4]))
                tracks[-1].team_colour = team_colours.get(tracks[-1].team)
        self._finish_tracks(frame, tracks)
        return tracks

    def observers_in_view(self) -> Dict[int, List[float]]:
        """Track -> box (x1, y1, x2, y2) of the observers followed in the last frame."""
        return dict(self._observers)

    def team_shirt_colours(self) -> Dict[int, Tuple[int, int, int]]:
        """{team (0 or 1): its average shirt colour (BGR)}; empty until the teams are known."""
        return self._player_tracker.shirt_colours() if self._player_tracker is not None else {}

    def swap_players(self, first: int, second: int) -> bool:
        """Exchange the players of two tracks: the tracker went on with the wrong one of two
        players who crossed. Returns whether it was done (both are still followed)."""
        if self._player_tracker is None:
            return False
        track_of = {}
        for track in self._player_tracker.tracked_stracks + self._player_tracker.lost_stracks:
            player = self._identities.player_of(int(track.track_id))
            if player in (first, second):
                track_of[player] = track
        if len(track_of) != 2:
            return False
        track_of[first].exchange_player(track_of[second])
        for player in (first, second):
            self._identities.forget_course(player)
        return True

    def set_team_of_player(self, player_id: int, team: int) -> None:
        """Say which team a player is in, by other knowledge than their shirt colour."""
        self._player_teams[player_id] = team

    def team_of_player(self, player_id: int) -> Optional[int]:
        """The team (0 or 1) of a player, present or missing; None if it is not known."""
        return self._player_teams.get(player_id)

    def _finish_tracks(self, frame: np.ndarray, tracks: List[Track]) -> None:
        """Give the tracks of a frame their player IDs and add them to the trails."""
        self._assign_player_identities(frame, tracks)
        for track in tracks:
            if track.team is not None:
                self._player_teams[track.track_id] = track.team

        # The trail follows the player's feet (bottom centre of the box)
        for track in tracks:
            x1, _, x2, y2 = track.bbox
            self._update_track_history(track.track_id, (int((x1 + x2) / 2), int(y2)))
            self._history_last_frame[track.track_id] = self._frame_count

        # Trails of tracks that are gone for good no longer need to be stored
        oldest = self._frame_count - self._max_age_frames()
        for track_id in [
            t
            for t in self._track_histories
            if self._history_last_frame.get(t, float("-inf")) < oldest
        ]:
            del self._track_histories[track_id]
            self._history_last_frame.pop(track_id, None)

    def _followed_tracks(self) -> Tuple[List[Any], float]:
        """Player tracks the tracker still follows, found in this frame or not, and how long
        a track exists before the tracker reports it (seconds)."""
        if self._player_tracker is None:
            return [], 0.0
        followed = self._player_tracker.tracked_stracks + self._player_tracker.lost_stracks
        return followed, 1.0 / self._frame_rate  # Reported from its second frame on

    def _assign_player_identities(self, frame: np.ndarray, tracks: List[Track]) -> None:
        """Replace the track IDs of this frame's tracks by the IDs of the players they are.

        The kit colour of a player is taken every few frames; a track that is new gets it at
        once, since that is when it is compared with the missing players.
        """
        players = [track for track in tracks if track.class_name == "player"]
        for track in tracks:
            if track.class_name != "player":
                track.track_id += DISC_ID_OFFSET
        if not players:
            return

        if not get_setting("models.tracking.identity.enabled", True):
            return
        interval = max(1, int(get_setting("models.tracking.identity.appearance_interval", 10)))
        due = [
            index
            for index, track in enumerate(players)
            if not self._identities.knows_track(track.track_id)
            or (self._calls + track.track_id) % interval == 0
        ]
        features: List[Optional[np.ndarray]] = [None] * len(players)
        for index, vector in zip(due, appearance.encode(frame, [players[i].bbox for i in due])):
            features[index] = vector

        observations = [
            Observation(
                track_id=track.track_id,
                position=((track.bbox[0] + track.bbox[2]) / 2, track.bbox[3]),
                height=track.bbox[3] - track.bbox[1],
                feature=feature,
                team=track.team if track.team is not None else track.team_leaning,
            )
            for track, feature in zip(players, features)
        ]
        followed, track_age = self._followed_tracks()
        alive = {int(track.track_id) for track in followed}
        player_of_track = self._identities.assign(
            self._frame_count / self._frame_rate, observations, alive, track_age
        )
        for track in players:
            track.track_id = player_of_track[track.track_id]

    def merge_players(self, player_id: int, into_player_id: int) -> None:
        """Declare a player to be an earlier one who went missing (e.g. same jersey number)."""
        self._identities.merge(player_id, into_player_id)
        self._player_teams.pop(player_id, None)
        if player_id in self._track_histories:
            self._track_histories[into_player_id] = self._track_histories.pop(player_id)
            self._history_last_frame[into_player_id] = self._history_last_frame.pop(
                player_id, self._frame_count
            )

    def missing_players(self, present: set) -> List[int]:
        """Players that are remembered but not among the given ones."""
        return self._identities.missing_players(present)

    def kit_distance(self, first_player: int, second_player: int) -> Optional[float]:
        """How different the kits of two players are (0 = the same), or None if not known."""
        return self._identities.kit_distance(first_player, second_player)

    def _run_simple_tracking(self, detections: List[Dict[str, Any]]) -> List[Track]:
        """Simple tracking fallback that assigns new IDs to each detection.

        Args:
            detections: List of detection dictionaries

        Returns:
            List of Track objects with new IDs
        """
        tracks = []
        # Fallback IDs are new each frame, so old histories can never be reused.
        self._track_histories.clear()

        for i, detection in enumerate(detections):
            # Create a simple track with frame-based ID
            track_id = self._frame_count * 1000 + i  # Simple ID generation

            track = Track(
                track_id=track_id,
                bbox=detection["bbox"],
                class_id=detection["class_id"],
                confidence=detection["confidence"],
                class_name=detection.get("class_name", "unknown"),
                model_type=detection.get("model_type", "unknown"),
            )
            tracks.append(track)

            # Update track history (at player's feet - bottom center)
            foot_x = (detection["bbox"][0] + detection["bbox"][2]) / 2  # Center X
            foot_y = detection["bbox"][3]  # Bottom Y (feet level)
            self._update_track_history(track_id, (int(foot_x), int(foot_y)))

        return tracks

    def reset(self) -> None:
        """Forget all tracks: after a seek, a cut, or another video."""
        logger.info("Resetting tracker state")
        self._observers.clear()
        for tracker in (self._player_tracker, self._disc_tracker):
            if tracker is not None:
                tracker.reset()

        # Clear track histories and reset frame count
        self._track_histories.clear()
        self._history_last_frame.clear()
        self._player_teams.clear()
        self._identities.reset()
        self._frame_count = self._calls = 0
        self._video_frame, self._frames_per_step = None, 1

        # Reset jersey tracking as well
        try:
            from .jersey_tracker import reset_jersey_tracker

            reset_jersey_tracker()
        except ImportError:
            logger.debug("Jersey tracker not available for reset")

        logger.info("Tracker reset complete")

    def get_track_histories(self) -> Dict[int, np.ndarray]:
        """The trail of every tracked object, without the bobbing of the stride.

        Returns:
            Track ID -> positions of the feet in the picture, oldest first, as an integer
            array of shape (N, 2)
        """
        seconds = float(get_setting("models.tracking.trail_smoothing_seconds", 0.4))
        reach = int(round(seconds / 2 * self._frame_rate / self._frames_per_step))
        return {
            track_id: np.rint(evened_out(history, reach)).astype(np.int32)
            for track_id, history in self._track_histories.items()
        }

    def apply_camera_motion(self, camera_motion: np.ndarray) -> None:
        """Move everything the tracker remembers along with the picture.

        Positions from earlier frames are pixel positions. When the camera pans or zooms, the
        spot on the field a player stood on is somewhere else in the picture.

        - The trails are moved there, so a trail stays on the ground the player ran over.
        - So is where each track expects its player next. Otherwise a pan looks to the tracker
          as if every player had jumped, and it matches them worse or loses them.

        Call this before run_tracking for the frame.

        Args:
            camera_motion: Homography from the previous frame to the current one
        """
        self._move_track_states(camera_motion)
        self._identities.apply_camera_motion(camera_motion)
        # All tracks in one call; there are thousands of stored positions
        track_ids = list(self._track_histories)
        if not track_ids:
            return
        lengths = [len(self._track_histories[track_id]) for track_id in track_ids]
        stacked = np.concatenate([self._track_histories[track_id] for track_id in track_ids])
        mapped = np.column_stack([stacked, np.ones(len(stacked))]) @ np.asarray(camera_motion).T
        # Ground the camera has flown over has no place in the picture any more: its points
        # land far outside, or behind the camera, where the depth changes sign. A trail ends
        # there, or it is drawn as a streak right across the frame.
        depth = mapped[:, 2] * np.sign(camera_motion[2, 2] or 1.0)
        in_front = depth > 1e-9
        moved = (mapped[:, :2] / np.where(in_front, depth, 1.0)[:, None]).astype(np.float32)
        in_reach = in_front & (np.abs(moved) < TRAIL_REACH_PIXELS).all(axis=1)
        start = 0
        for track_id, length in zip(track_ids, lengths):
            trail, kept = moved[start : start + length], in_reach[start : start + length]
            if not kept.all():
                # What follows the last point that was lost
                trail = trail[length - int(np.argmin(kept[::-1])) :]
            self._track_histories[track_id] = trail
            start += length

    def _move_track_states(self, camera_motion: np.ndarray) -> None:
        """Apply the camera motion to the tracker's motion model of every track.

        A track's state is its box centre, aspect ratio, and height, and the speed of each
        (x, y, a, h, vx, vy, va, vh). The centre moves with the picture; near a point the
        motion is a small linear map, which turns the speed and scales the height.
        """
        states = [
            track
            for tracker in (self._player_tracker, self._disc_tracker)
            if tracker is not None
            for track in tracker.tracked_stracks + tracker.lost_stracks
            if track.mean is not None
        ]
        if not states:
            return
        # All tracks at once: each centre, and a step to its right and one below it, which
        # say how the picture stretches and turns around it
        centres = np.array([[track.mean[0], track.mean[1]] for track in states], dtype=np.float64)
        points = np.concatenate([centres, centres + [1.0, 0.0], centres + [0.0, 1.0]])
        mapped = points @ camera_motion[:, :2].T + camera_motion[:, 2]
        count = len(states)
        depth = mapped[:, 2].reshape(3, count)
        usable = np.abs(depth[0]) >= 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            moved = (mapped[:, :2] / mapped[:, 2:3]).reshape(3, count, 2)
        for index, track in enumerate(states):
            if not usable[index]:
                continue
            new = moved[0, index]
            local = (moved[1:, index] - new).T
            scale = float(np.sqrt(abs(local[0, 0] * local[1, 1] - local[0, 1] * local[1, 0])))

            track.mean[0], track.mean[1] = new
            track.mean[3] *= scale
            track.mean[4:6] = local @ track.mean[4:6]
            track.mean[7] *= scale

    def _update_track_history(self, track_id: int, center_point: Tuple[int, int]) -> None:
        """Update the position history for a track.

        Args:
            track_id: Unique track identifier
            center_point: Center point (x, y) of the tracked object
        """
        max_length = get_setting("models.tracking.track_history_length", TRACK_HISTORY_MAX_LENGTH)
        # As many points as cover the trail's time at the rate frames are analysed at
        seconds = float(get_setting("models.tracking.trail_seconds", 4.0))
        max_length = max(
            2, min(max_length, round(seconds * self._frame_rate / self._frames_per_step))
        )
        point = np.array([center_point], dtype=np.float32)
        history = self._track_histories.get(track_id)
        if history is None:
            self._track_histories[track_id] = point
        else:
            self._track_histories[track_id] = np.concatenate((history[-(max_length - 1) :], point))
