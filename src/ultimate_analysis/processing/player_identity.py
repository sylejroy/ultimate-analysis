"""Player identities that outlive the tracker's tracks.

The tracker follows a player from frame to frame and sometimes starts a new track for a
player it already had: after losing them behind somebody else, or after giving up on
them. Here every player has a profile that is kept for much longer than a track. A new
track is compared with the players that are currently missing and takes over the identity
of the one it is.

Two things can tell, and each has its range:

- Shortly after a player was lost, physics. A runner cannot stop or turn on the spot, so
  they are expected ahead of where they were, the way they were going, and can only be as
  far from there as their legs can accelerate them. A new track at that place is that
  player. After a couple of seconds a player can be almost anywhere and this says nothing.
- After any length of time, the jersey number. Once the new player's number has been read
  and it is that of a missing player in the same kit, the two are joined (see the
  pipeline).

Looks decide nothing at the moment a track appears: one look at a kit is thrown off by
shadow or by a second player in the box, and nothing in the picture tells two teammates
apart (see processing/appearance.py). The kit colour is averaged over time and used only
to keep the two teams' equal numbers apart.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config.settings import get_setting

NO_MATCH = 1e6


@dataclass
class Observation:
    """A tracked player in the current frame."""

    track_id: int
    position: Tuple[float, float]  # Feet, in picture pixels
    height: float  # Of the box, in pixels
    feature: Optional[np.ndarray] = None  # Kit colour, if taken for this frame


@dataclass
class Profile:
    """What is known about a player."""

    feature: Optional[np.ndarray]  # Kit colour, averaged over time
    position: Tuple[float, float]
    height: float
    last_seen: float  # Seconds
    samples: int = 0  # Observations averaged into the feature
    velocity: Tuple[float, float] = (0.0, 0.0)  # Picture pixels per second


class PlayerIdentities:
    """Gives every track the identity of the player it belongs to."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Forget all players (new video, seek)."""
        self._profiles: Dict[int, Profile] = {}
        self._player_of_track: Dict[int, int] = {}
        self._next_player_id = 1

    # ------------------------------------------------------------------ settings

    @staticmethod
    def _setting(name: str, default: float) -> float:
        return float(get_setting(f"models.tracking.identity.{name}", default))

    # ------------------------------------------------------------------ per frame

    def assign(
        self,
        time_s: float,
        observations: List[Observation],
        alive_tracks: Optional[set] = None,
        new_track_age_s: float = 0.0,
    ) -> Dict[int, int]:
        """Identities of the tracks in a frame.

        Args:
            time_s: Position of the frame in the video in seconds
            observations: The tracked players of the frame
            alive_tracks: IDs of all tracks the tracker still follows, including those it
                did not find in this frame. A track keeps its player while it is alive;
                without this, only the observed tracks count as alive.
            new_track_age_s: How long a track exists before the tracker reports it. A
                player who was still seen during that time is not the new track.

        Returns:
            Track ID -> player ID
        """
        self._forget_long_gone(time_s)
        known = [obs for obs in observations if obs.track_id in self._player_of_track]
        new = [obs for obs in observations if obs.track_id not in self._player_of_track]
        present = {self._player_of_track[obs.track_id] for obs in known}

        if new:
            matches = self._match(time_s, new, present, new_track_age_s)
            for observation, player_id in matches.items():
                # The track that had this player until now did not find them again
                for track_id in [t for t, p in self._player_of_track.items() if p == player_id]:
                    del self._player_of_track[track_id]
                self._player_of_track[new[observation].track_id] = player_id
            for observation in new:
                if observation.track_id not in self._player_of_track:
                    player_id = self._next_player_id
                    self._next_player_id += 1
                    self._player_of_track[observation.track_id] = player_id
                    self._profiles[player_id] = Profile(None, observation.position, 0.0, time_s)

        for observation in observations:
            self._update(
                self._profiles[self._player_of_track[observation.track_id]], observation, time_s
            )

        observed = {obs.track_id for obs in observations}
        self._forget(observed if alive_tracks is None else observed | set(alive_tracks))
        return {obs.track_id: self._player_of_track[obs.track_id] for obs in observations}

    def _match(
        self, time_s: float, new: List[Observation], present: set, new_track_age_s: float
    ) -> Dict[int, int]:
        """Index of a new observation -> missing player it clearly is."""
        longest_gap = self._setting("position_match_seconds", 0.0)
        missing = [
            player_id
            for player_id, profile in self._profiles.items()
            if player_id not in present
            # Seen while the new track already existed: two different players
            and new_track_age_s < time_s - profile.last_seen <= longest_gap
        ]
        candidates = list(range(len(new)))
        if not missing:
            return {}

        margin = self._setting("min_margin", 0.25)
        kit_limit = self._setting("max_kit_distance", 60.0)

        # Cost of a match: how far the new track is from where the player was expected,
        # as a share of how far from there they can have got
        cost = np.full((len(candidates), len(missing)), NO_MATCH)
        for row, index in enumerate(candidates):
            observation = new[index]
            for column, player_id in enumerate(missing):
                profile = self._profiles[player_id]
                expected, reach = self._reachable(profile, time_s - profile.last_seen)
                off = np.hypot(
                    observation.position[0] - expected[0], observation.position[1] - expected[1]
                )
                if off <= reach and not self._other_kit(profile, observation, kit_limit):
                    cost[row, column] = off / reach

        matches = {}
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            if cost[row, column] >= NO_MATCH:
                continue
            # Another missing player could be this track nearly as well, or another new
            # track could be this player: not clear enough to decide
            rivals = np.concatenate((np.delete(cost[row], column), np.delete(cost[:, column], row)))
            if len(rivals) and rivals.min() - cost[row, column] < margin:
                continue
            matches[candidates[row]] = missing[column]
        return matches

    @staticmethod
    def _other_kit(profile: Profile, observation: Observation, kit_limit: float) -> bool:
        """Whether a track wears another kit than a player: then it is not that player.

        Two players of different teams who come apart after covering each other are both
        within reach of both new tracks; the kit tells which is which.
        """
        if profile.feature is None or observation.feature is None:
            return False
        difference = np.asarray(observation.feature, dtype=np.float32) - profile.feature
        return float(np.linalg.norm(difference)) > kit_limit

    def _reachable(self, profile: Profile, gone_s: float) -> Tuple[Tuple[float, float], float]:
        """Where a missing player is expected, and how far from there they can be (pixels).

        For a short while they carry on as they were going, and can only have left that
        course as far as a person can accelerate. After that anything is possible that
        their top speed allows.
        """
        acceleration = self._setting("max_acceleration_heights_per_second2", 4.0)
        speed = self._setting("max_speed_heights_per_second", 5.0)
        carry_on = self._setting("carry_on_seconds", 1.0)

        coasting = min(gone_s, carry_on)
        expected = (
            profile.position[0] + profile.velocity[0] * coasting,
            profile.position[1] + profile.velocity[1] * coasting,
        )
        # One body height of slack for the box jittering around the player
        heights = 0.5 * acceleration * coasting**2 + speed * (gone_s - coasting) + 1.0
        return expected, heights * max(profile.height, 1.0)

    def _update(self, profile: Profile, observation: Observation, time_s: float) -> None:
        elapsed = time_s - profile.last_seen
        if profile.height > 0 and 0 < elapsed <= 0.5:
            # Smoothed: the feet of a box wobble from frame to frame
            keep = 0.7
            profile.velocity = (
                keep * profile.velocity[0]
                + (1 - keep) * (observation.position[0] - profile.position[0]) / elapsed,
                keep * profile.velocity[1]
                + (1 - keep) * (observation.position[1] - profile.position[1]) / elapsed,
            )
        elif elapsed > 0.5:
            profile.velocity = (0.0, 0.0)  # Back after a gap: the old course is history
        profile.position = observation.position
        profile.height = observation.height
        profile.last_seen = time_s
        if observation.feature is not None:
            feature = np.asarray(observation.feature, dtype=np.float32)
            # A plain average at first, then one that slowly follows changes in lighting
            weight = max(1.0 / (profile.samples + 1), self._setting("feature_update", 0.05))
            previous = profile.feature if profile.feature is not None else feature
            profile.feature = (1 - weight) * previous + weight * feature
            profile.samples += 1

    def _forget(self, live_tracks: set) -> None:
        # A track the tracker has given up does not come back under the same track ID
        for track_id in [t for t in self._player_of_track if t not in live_tracks]:
            del self._player_of_track[track_id]

    def _forget_long_gone(self, time_s: float) -> None:
        memory = self._setting("memory_seconds", 30.0)
        followed = set(self._player_of_track.values())
        for player_id in [p for p in self._profiles if p not in followed]:
            if time_s - self._profiles[player_id].last_seen > memory:
                del self._profiles[player_id]

    # ------------------------------------------------------------------ other knowledge

    def apply_camera_motion(self, camera_motion: np.ndarray) -> None:
        """Move the last known places and courses of all players along with the picture."""

        def move(x: float, y: float) -> Tuple[float, float]:
            moved = camera_motion @ np.array([x, y, 1.0])
            return (moved[0] / moved[2], moved[1] / moved[2]) if abs(moved[2]) > 1e-9 else (x, y)

        for profile in self._profiles.values():
            x, y = profile.position
            # The course turns and stretches with the picture: move the point a second ahead
            ahead = move(x + profile.velocity[0], y + profile.velocity[1])
            profile.position = move(x, y)
            profile.velocity = (ahead[0] - profile.position[0], ahead[1] - profile.position[1])

    def knows_track(self, track_id: int) -> bool:
        """Whether a track has been given a player's identity already."""
        return track_id in self._player_of_track

    def missing_players(self, live_players: set) -> List[int]:
        """Players that are remembered but not on screen."""
        return [player_id for player_id in self._profiles if player_id not in live_players]

    def kit_distance(self, first: int, second: int) -> Optional[float]:
        """How different the kits of two players are (0 = the same), or None if not known."""
        a, b = self._profiles.get(first), self._profiles.get(second)
        if a is None or b is None or a.feature is None or b.feature is None:
            return None
        return float(np.linalg.norm(a.feature - b.feature))

    def merge(self, player_id: int, into_player_id: int) -> None:
        """Declare a player to be an earlier, missing one, e.g. after reading their number."""
        profile = self._profiles.pop(player_id, None)
        if profile is None or into_player_id not in self._profiles:
            return
        self._profiles[into_player_id] = profile
        for track_id, owner in self._player_of_track.items():
            if owner == player_id:
                self._player_of_track[track_id] = into_player_id
