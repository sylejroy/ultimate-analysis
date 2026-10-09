"""The team tracker with the players followed on the ground as well as in the picture.

In the picture a player who is not detected for a second is hard to find again: the box
the tracker expects drifts with every pan and every stride that was misjudged, and the
player returns as somebody new. On the field a player is where they were, give or take
the few yards they can have run. Where the field's place in the frame is known, every
track therefore also keeps the place of its player's feet on the field (yards) and how
fast they move, and

- a track that lost its player, and whose box no longer overlaps any detection, is
  matched with a detection of its team's colours where the player can have got to;
- with that to go by, a lost track is kept longer than the picture alone allows.

The place only ever adds a match the picture did not have. The field's place is too
rough, above all far from the camera, to forbid one the boxes agree on.

Where the field's place is not known, everything is as in the team tracker.

The field's place is itself an estimate that shifts a little from frame to frame. The
players followed in a frame show by how much (their places all shift together), and the
places of the lost ones are shifted along.
"""

from typing import Any, List, Optional, Sequence

import numpy as np

from .team_tracker import TeamTrack, TeamTracker

# A player's top speed in field units (yards) a second
TOP_SPEED = 11.0
# How far a lost player's detection may lie from where they are expected at once, and
# how much further for every second they have been lost; never more than the last
REACH = 2.0
REACH_PER_SECOND = 7.0
MAX_REACH = 14.0
# A lost player is expected to slow down: their speed falls to a third in this time
CARRY_ON_SECONDS = 0.7
# How much of the difference between where a player was expected and where they are
# found goes into their place and into their speed
PLACE_GAIN = 0.5
SPEED_GAIN = 0.15
# The cost of matching a lost track with a detection right where its player is expected
# and at the edge of their reach. Above what a followed player's own box costs, so a
# lost track never takes a detection from the track that is on it; below the tracker's
# threshold (0.8).
LOST_COST = (0.6, 0.78)
# Followed players from whom the frame's shift of the field's place is taken
MIN_FOR_SHIFT = 5


def feet_on_field(image_to_field: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Where the feet (bottom centres) of boxes (x1, y1, x2, y2) are on the field: (N, 2),
    NaN where a box has no place on it."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3], np.ones(len(boxes))])
    mapped = feet @ np.asarray(image_to_field, dtype=np.float64).T
    depth = mapped[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        places = mapped[:, :2] / depth
    places[np.abs(depth[:, 0]) < 1e-12] = np.nan
    return places


def reach_after(seconds: float) -> float:
    """How far from where they are expected a player lost for some time may be found."""
    return min(REACH + REACH_PER_SECOND * max(seconds, 0.0), MAX_REACH)


class GroundTrack(TeamTrack):
    """A team track that also knows where its player is on the field."""

    place: Optional[np.ndarray] = None  # A detection: where its feet are on the field
    ground: Optional[np.ndarray] = None  # A track: where its player is expected
    speed: Optional[np.ndarray] = None  # ... and how fast they move, yards a second
    unseen: float = 0.0  # Seconds since the player was last detected
    shift: Optional[np.ndarray] = None  # By how much the player was off where expected

    def carry_on(self, seconds: float) -> None:
        """Move the player on by a step in which they were not (yet) detected."""
        self.unseen += seconds
        self.shift = None
        if self.ground is not None and self.speed is not None:
            self.ground = self.ground + self.speed * seconds
            self.speed = self.speed * (1.0 / 3.0) ** (seconds / CARRY_ON_SECONDS)

    def _found(self, detection: "GroundTrack", seconds: float) -> None:
        """Take in where the player was detected."""
        was_unseen, self.unseen = self.unseen, 0.0
        place = getattr(detection, "place", None)
        if place is None or not np.all(np.isfinite(place)):
            return  # The field's place is not known: the last known one is kept
        if self.ground is None or was_unseen > 2 * seconds:
            # New, or back from being lost: where they went meanwhile says little
            # about how they move now
            self.ground, self.speed = place.copy(), np.zeros(2)
            return
        off = place - self.ground
        self.shift = off
        self.ground = self.ground + PLACE_GAIN * off
        speed = self.speed + SPEED_GAIN * off / max(seconds, 1e-6)
        fast = float(np.linalg.norm(speed))
        self.speed = speed if fast <= TOP_SPEED else speed * (TOP_SPEED / fast)

    def activate(self, kalman_filter: Any, frame_id: int) -> None:
        super().activate(kalman_filter, frame_id)
        self._found(self, GroundTeamTracker.seconds_per_step)

    def update(self, new_track: "GroundTrack", frame_id: int) -> None:
        super().update(new_track, frame_id)
        self._found(new_track, GroundTeamTracker.seconds_per_step)

    def re_activate(self, new_track: "GroundTrack", frame_id: int, new_id: bool = False) -> None:
        super().re_activate(new_track, frame_id, new_id)
        self._found(new_track, GroundTeamTracker.seconds_per_step)

    def exchange_player(self, other: "GroundTrack") -> None:
        super().exchange_player(other)
        # The boxes stay with the tracks, and so does where they are on the field


class GroundTeamTracker(TeamTracker):
    """The team tracker, with lost players found again by their place on the field."""

    track_class = GroundTrack
    seconds_per_step = 1.0 / 30.0  # Set for the frame by `look`; the tracks read it

    def __init__(self, args: Any, picture_seconds: float):
        """
        Args:
            args: The tracker's settings; `track_buffer` is how long a lost track is kept
                where the field's place is known
            picture_seconds: How long a lost track can be found again by the picture
                alone, where the field's place is not known
        """
        super().__init__(args)
        self.picture_seconds = float(picture_seconds)
        self.image_to_field: Optional[np.ndarray] = None

    def look(self, image_to_field: Optional[np.ndarray], seconds_per_step: float) -> None:
        """Before a frame: where the field is in it, if known, and the time since the
        frame before."""
        self.image_to_field = None if image_to_field is None else np.asarray(image_to_field)
        GroundTeamTracker.seconds_per_step = float(seconds_per_step)

    def update(self, results: Any, img: Optional[np.ndarray] = None, *args, **kwargs):
        for track in self.tracked_stracks + self.lost_stracks:
            track.carry_on(self.seconds_per_step)
        rows = super().update(results, img, *args, **kwargs)
        self._follow_the_field()
        return rows

    def _follow_the_field(self) -> None:
        """Shift the lost players' places by what the followed ones' all shifted by."""
        shifts = [
            track.shift
            for track in self.tracked_stracks
            if getattr(track, "shift", None) is not None
        ]
        if len(shifts) < MIN_FOR_SHIFT:
            return
        shift = np.median(np.array(shifts), axis=0)
        for track in self.lost_stracks:
            if track.ground is not None:
                track.ground = track.ground + shift

    def init_track(self, results: Any, img: Optional[np.ndarray] = None) -> List[Any]:
        detections = super().init_track(results, img)
        if self.image_to_field is not None and detections:
            places = feet_on_field(self.image_to_field, results.xyxy)
            for detection, place in zip(detections, places):
                detection.place = place if np.all(np.isfinite(place)) else None
        return detections

    @staticmethod
    def _in_the_colours_of(track: Any, detections: Sequence[Any]) -> np.ndarray:
        """Which detections' shirts are of the track's team, as far as both are known."""
        team = getattr(track, "team", None)
        shirts = [getattr(detection, "shirt_team", None) for detection in detections]
        return np.array([team is None or shirt is None or shirt == team for shirt in shirts])

    def get_dists(self, tracks: Sequence[Any], detections: Sequence[Any]) -> np.ndarray:
        distances = self.picture_distances(tracks, detections)
        if not len(tracks) or not len(detections):
            return distances
        from ultralytics.trackers.basetrack import TrackState

        places = np.array(
            [
                getattr(d, "place", None)
                if getattr(d, "place", None) is not None
                else (np.nan,) * 2
                for d in detections
            ],
            dtype=np.float64,
        )
        for row, track in enumerate(tracks):
            if track.state != TrackState.Lost:
                continue
            ground = getattr(track, "ground", None)
            if ground is None or self.image_to_field is None:
                if track.unseen > self.picture_seconds:
                    distances[row] = 1.0  # Too long ago to be found by the picture
                continue
            too_long_ago = track.unseen > self.picture_seconds
            if too_long_ago:
                distances[row] = 1.0  # Only their place can still tell
            apart = np.linalg.norm(places - ground, axis=1) / reach_after(track.unseen)
            # Found by the place alone, a player must at least wear the track's colours:
            # whoever else has run to where they were lost is not them. And only where
            # the picture has nothing to say: the field's place is too rough a guide to
            # overrule boxes that overlap.
            by_place_alone = (
                np.isfinite(apart)
                & (apart <= 1.0)
                & self._in_the_colours_of(track, detections)
                & (distances[row] > self.args.match_thresh)
            )
            cost = LOST_COST[0] + (LOST_COST[1] - LOST_COST[0]) * apart
            distances[row, by_place_alone] = cost[by_place_alone]
        distances[self.other_team(tracks, detections)] = 1.0
        return distances
