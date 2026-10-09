"""Knowing the players of a video by their looks, frame by frame.

The stage between the tracker and the roster (player_roster.py): it looks at every
tracked player a few times a second, turns the crop into a feature vector (reid.py),
and hands vectors, team colours and the jersey numbers read with certainty to the
roster. What comes back is, for the tracks whose number was never read, the number of
the player the roster takes them for. The roster of a video is kept in a file, so what
is known of a game, and what its players did, is there again the next time.

Without the network's weights, or without PyTorch, the stage does nothing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config.settings import get_setting
from ..utils.logger import get_logger
from .player_roster import PlayerRoster

logger = get_logger("PLAYER_LOOKS")

# A crop smaller than this shows no player to speak of (pixels: width, height)
MIN_CROP = (8, 16)
# Of what a track has shown so far, the most blurred share is not looked at: a player
# smeared by motion looks like any other. So many crops are remembered per track.
SKIP_BLURRED = 0.3
SHARPNESS_MEMORY = 50
# A track not seen for this long is over; the roster keeps its player
TRACK_ENDED_SECONDS = 3.0
# A gap longer than this between two frames is not time a player was seen for
MAX_FRAME_SECONDS = 0.5
# How far a player ran is added up from where they stand every so often, not frame by
# frame: the foot of a box jitters by a foot or two, which would count as running
RUN_STEP_SECONDS = 0.5
# Faster than this (field units a second) nobody runs: the field's place jumped
MAX_RUN_SPEED = 12.0
# A disc that changes hands within this time and distance, in one team, was passed
MAX_PASS_SECONDS = 10.0
MAX_PASS_DISTANCE = 100.0


class PlayerLooks:
    """Looks at the tracked players of a video and keeps its roster."""

    def __init__(self) -> None:
        self.roster = PlayerRoster()
        self.frame_rate = 30.0
        self._embedder: Any = None
        self._tried_to_load = False
        self._roster_path: Optional[Path] = None
        self._looked_at: Optional[int] = None  # The frame the players were last looked at
        self._seen_at: Dict[int, int] = {}  # Track -> frame it was last seen in
        self._sharpness: Dict[int, List[float]] = {}
        self._frame: Optional[int] = None
        # Track -> (frame, place on the field) its running was last counted from
        self._ran_from: Dict[int, Tuple[int, np.ndarray]] = {}
        # Who held the disc last: track, team, place on the field (if known), frame
        self._last_held: Optional[Tuple[int, Optional[int], Optional[np.ndarray], int]] = None

    # ------------------------------------------------------------------ the video

    def new_video(self, video_path: Optional[str]) -> None:
        """Another video: its roster is read if there is one, the last one's written."""
        self.save()
        self.roster = PlayerRoster()
        self._roster_path = None
        self._forget_tracks()
        folder = get_setting("models.reid.roster_folder", "data/cache/rosters")
        if video_path and folder:
            self._roster_path = Path(folder) / f"{Path(video_path).stem}.json"
            if self._roster_path.exists():
                try:
                    self.roster = PlayerRoster.load(self._roster_path)
                except (OSError, ValueError, KeyError) as error:
                    logger.warning(f"Roster not read, starting a new one: {error}")

    def cut(self) -> None:
        """The tracker starts again (a cut, a seek): its tracks are over, the players stay."""
        self.roster.tracks_ended()
        self._forget_tracks()
        self.save()

    def save(self) -> None:
        """Write the roster of the video, if it has one and anybody is in it."""
        if self._roster_path is not None and self.roster.entries:
            try:
                self.roster.save(self._roster_path)
            except OSError as error:
                logger.warning(f"Roster not written: {error}")

    def rename(self, track_id: int, new_track_id: int) -> None:
        """A track turned out to be a player the tracker knew before."""
        self.roster.rename(track_id, new_track_id)
        for kept in (self._seen_at, self._sharpness, self._ran_from):
            if track_id in kept:
                kept[new_track_id] = kept.pop(track_id)
        if self._last_held is not None and self._last_held[0] == track_id:
            self._last_held = (new_track_id, *self._last_held[1:])

    def _forget_tracks(self) -> None:
        self._looked_at = None
        self._seen_at.clear()
        self._sharpness.clear()
        self._ran_from.clear()
        self._frame = self._last_held = None

    # ------------------------------------------------------------------ a frame

    def _network(self) -> Any:
        if not self._tried_to_load:
            self._tried_to_load = True
            weights = Path(str(get_setting("models.reid.weights", "")))
            try:
                from .reid import load_embedder

                if weights.is_file():
                    self._embedder = load_embedder(weights)
                else:
                    logger.info(f"No network to know players by their looks at {weights}")
            except Exception as error:  # No PyTorch, or weights of another build
                logger.warning(f"Players are not told apart by their looks: {error}")
        return self._embedder

    def update(
        self,
        frame: np.ndarray,
        tracks: Sequence[Any],
        frame_index: int,
        numbers: Dict[int, str],
        holder_id: Optional[int] = None,
        places: Optional[Dict[int, Sequence[float]]] = None,
    ) -> Dict[int, str]:
        """Take in a frame.

        Args:
            frame: The video frame (BGR)
            tracks: The frame's tracks (track_id, class_name, to_ltrb(), team_colour)
            frame_index: Position of the frame in the video
            numbers: Track -> jersey number, for the tracks whose number is read with
                certainty
            holder_id: The track holding the disc, if one does
            places: Track -> where the player stands on the field, as far as the field's
                place is known in this frame

        Returns:
            Track -> jersey number, for the tracks without a number of their own whose
            player the roster knows the number of
        """
        network = self._network()
        if network is None:
            return {}
        from .reid import embed, sharpness

        players = [track for track in tracks if getattr(track, "class_name", None) == "player"]
        every = max(1, int(get_setting("models.reid.look_every_frames", 6)))
        height, width = frame.shape[:2]
        crops, looked = [], []
        for track in players:
            self._seen_at[int(track.track_id)] = frame_index
        # All players in one frame, and none in the frames between: the network takes
        # a crop or twenty in about the same time
        look = self._looked_at is None or not 0 <= frame_index - self._looked_at < every
        if look:
            self._looked_at = frame_index
        for track in players if look else ():
            track_id = int(track.track_id)
            x1, y1, x2, y2 = (int(round(value)) for value in track.to_ltrb())
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 - x1 < MIN_CROP[0] or y2 - y1 < MIN_CROP[1]:
                continue
            crop = frame[y1:y2, x1:x2]
            history = self._sharpness.setdefault(track_id, [])
            history.append(sharpness(crop))
            del history[:-SHARPNESS_MEMORY]
            if len(history) >= 5 and history[-1] < np.quantile(history, SKIP_BLURRED):
                continue
            crops.append(crop)
            looked.append(track)
        if crops:
            for track, vector in zip(looked, embed(network, crops)):
                self.roster.look(int(track.track_id), vector, getattr(track, "team_colour", None))
        for track_id, number in numbers.items():
            self.roster.number_read(int(track_id), number)
        present = [int(track.track_id) for track in players]
        if look:
            self.roster.match(present)

        # What the players do is counted on their entries
        if self._frame is not None and 0 < frame_index - self._frame:
            seconds = min((frame_index - self._frame) / self.frame_rate, MAX_FRAME_SECONDS)
            for track_id in present:
                self.roster.count(track_id, "seconds_seen", seconds)
            if holder_id in present:
                self.roster.count(holder_id, "seconds_with_disc", seconds)
        self._frame = frame_index
        places = {track: np.asarray(place, dtype=float) for track, place in (places or {}).items()}
        self._count_running(frame_index, places)
        self._count_passes(frame_index, holder_id, players, places)

        ended = [
            track_id
            for track_id, seen in self._seen_at.items()
            if frame_index - seen > TRACK_ENDED_SECONDS * self.frame_rate
        ]
        if ended:
            self.roster.tracks_ended(ended)
            for track_id in ended:
                for kept in (self._seen_at, self._sharpness, self._ran_from):
                    kept.pop(track_id, None)

        by_looks = {}
        for track_id in present:
            if track_id not in numbers:
                number = self.roster.number_of(track_id)
                if number:
                    by_looks[track_id] = number
        return by_looks

    def _count_running(self, frame_index: int, places: Dict[int, np.ndarray]) -> None:
        """Add to how far each player has run."""
        for track_id, place in places.items():
            since = self._ran_from.get(track_id)
            if since is None or frame_index < since[0]:
                self._ran_from[track_id] = (frame_index, place)
                continue
            seconds = (frame_index - since[0]) / self.frame_rate
            if seconds < RUN_STEP_SECONDS:
                continue
            distance = float(np.linalg.norm(place - since[1]))
            # A player not seen for a while ran somewhere in between: counted as the
            # straight way, if that can be run in the time
            if distance <= MAX_RUN_SPEED * seconds:
                self.roster.count(track_id, "distance_run", distance)
            self._ran_from[track_id] = (frame_index, place)

    def _count_passes(
        self,
        frame_index: int,
        holder_id: Optional[int],
        players: Sequence[Any],
        places: Dict[int, np.ndarray],
    ) -> None:
        """Count a disc taken, and a pass from the teammate who held it before: how far
        it went, for the thrower and for the receiver."""
        if holder_id is None:
            return
        team = next(
            (getattr(t, "team", None) for t in players if int(t.track_id) == holder_id), None
        )
        place = places.get(holder_id)
        before = self._last_held
        if before is None or before[0] != holder_id:
            self.roster.count(holder_id, "times_with_disc", 1.0)
            if (
                before is not None
                and team is not None
                and before[1] == team
                and (frame_index - before[3]) <= MAX_PASS_SECONDS * self.frame_rate
            ):
                self.roster.count(before[0], "passes_thrown", 1.0)
                self.roster.count(holder_id, "catches", 1.0)
                if place is not None and before[2] is not None:
                    distance = float(np.linalg.norm(place - before[2]))
                    if distance <= MAX_PASS_DISTANCE:
                        self.roster.count(before[0], "distance_thrown", distance)
                        self.roster.count(holder_id, "distance_received", distance)
        elif place is None:
            place = before[2]  # Where they were last known to stand with it
        self._last_held = (holder_id, team, place, frame_index)
