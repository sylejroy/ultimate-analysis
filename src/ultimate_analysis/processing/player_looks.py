"""Knowing the players of a video by their looks, frame by frame.

The stage between the tracker and the roster (player_roster.py): it looks at every
tracked player a few times a second, turns the crop into a feature vector (reid.py),
and hands vectors, team colours and the jersey numbers read with certainty to the
roster. What comes back is, for the tracks whose number was never read, the number of
the player the roster takes them for. The roster of a video is kept in a file, so what
is known of a game, and what its players did, is there again the next time.

The looks serve the tracker too: which team a player is in (`teams`), whether two
tracks changed players while their boxes overlapped (`swapped`), and whether a new
track is a player lost a moment ago (`returned`).

Without the network's weights, or without PyTorch, the stage does nothing.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config.settings import get_setting
from ..utils.logger import get_logger
from .appearance import box_overlaps
from .model_lock import GPU_SETUP_LOCK
from .player_roster import PlayerRoster
from .tracklet_linking import TrackletBook

logger = get_logger("PLAYER_LOOKS")

# A crop smaller than this shows no player to speak of (pixels: width, height)
MIN_CROP = (8, 16)
# Of what a track has shown so far, the most blurred share is not looked at: a player
# smeared by motion looks like any other. So many crops are remembered per track.
SKIP_BLURRED = 0.3
SHARPNESS_MEMORY = 50
# A track not seen for this long is over; the roster keeps its player
TRACK_ENDED_SECONDS = 3.0
# What a track looked like is kept this long after it was last seen
PROFILE_SECONDS = 10.0
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
# A new track is a player the tracker lost if it has been looked at this often, is this
# much like them (cosine), and that much more than like any other lost teammate. Of
# tracks cut in two, the second part found its first among eight teammates in 67 to 74%
# of cases with these, and 98.7% of those rightly (docs/MEASUREMENTS.md).
REJOIN_LOOKS = 5
REJOIN_LIKENESS = 0.7
REJOIN_LEAD = 0.1
# The teams are told from the players' looks once this many tracks have been looked at
# this often each, if they fall into two groups of at least so many whose members are
# this much more alike among themselves than across (cosine). The colour of the shirt
# area, which the tracker goes by, puts a dark shirt with white panels with the white
# team, and red shirts on either side of a split between white and black.
TEAM_LOOKS = 12
TEAM_MIN_TRACKS = 6
TEAM_MIN_GROUP = 2
TEAM_MIN_GAP = 0.15
# Two players whose boxes overlap by this much (share of the two boxes together) are
# crossing: a box then shows both, and which track goes on with whom is where a tracker
# errs. Neither is looked at meanwhile. Once they are apart and each has been looked at
# this often, the looks say whether the tracks changed players: if each is this much
# more like the other's looks from before than like its own (the sum over both, cosine).
# A crossing that is not over after this many seconds is given up.
CROSSING_OVERLAP = 0.1
CROSSING_LOOKS = 2
CROSSING_SWAP_MARGIN = 0.3
CROSSING_SECONDS = 4.0


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


@dataclass
class _Profile:
    """What a track looks like and when it was there. Kept for a while after the track
    is gone, to know the player when they return."""

    first: int  # The first and the last frame the track was seen in
    last: int
    total: Optional[np.ndarray] = None  # The sum of the vectors of its looks
    looks: int = 0  # ... and how many
    team: Optional[int] = None
    ended: bool = False  # Told to the roster as over
    # Looks since the track began to cross another: not known yet whose they are
    held: List[Tuple[np.ndarray, Any]] = field(default_factory=list)

    @property
    def mean(self) -> np.ndarray:
        """What the track looks like: the mean of its looks, of length 1."""
        return _unit(self.total)

    def learn(self, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float64)
        self.total = vector if self.total is None else self.total + vector
        self.looks += 1

    def take_over(self, other: "_Profile") -> None:
        """What was seen of another track turns out to be of this one too."""
        if other.total is not None:
            self.total = other.total if self.total is None else self.total + other.total
        self.looks += other.looks
        self.first, self.last = min(self.first, other.first), max(self.last, other.last)
        self.team = other.team if other.team is not None else self.team
        self.held += other.held


def _embed_beside(embed: Any, network: Any, crops: List[np.ndarray]) -> np.ndarray:
    """Run the network, as one of the threads that use the GPU."""
    with GPU_SETUP_LOCK.work():
        return embed(network, crops)


class PlayerLooks:
    """Looks at the tracked players of a video and keeps its roster."""

    def __init__(self) -> None:
        self.roster = PlayerRoster()
        self.frame_rate = 30.0
        self._embedder: Any = None
        self._tried_to_load = False
        # The network chosen in place of the one in the settings, and the video
        self._weights: Optional[Path] = None
        self._video_path: Optional[str] = None
        self._roster_path: Optional[Path] = None
        self._thread: Optional[ThreadPoolExecutor] = None
        self._looked_at: Optional[int] = None  # The frame the players were last looked at
        self._frame: Optional[int] = None  # The frame taken in last
        self._profiles: Dict[int, _Profile] = {}
        self.book = TrackletBook()  # The tracks since the last cut, to join looking back
        self._sharpness: Dict[int, List[float]] = {}  # Track -> sharpness of its last crops
        # The look under way on the second thread: its vectors when they are ready, and
        # whose they are (track, team colour). Tracks renamed meanwhile: old -> new.
        self._looking: Optional[Tuple[Future, List[Tuple[int, Any]]]] = None
        self._renamed: Dict[int, int] = {}
        # Crossings not settled yet: (track, track) -> the last frame the two overlapped
        self._crossings: Dict[Tuple[int, int], int] = {}
        self._overlapping: set = set()  # The tracks whose box overlaps another's now
        # Track -> (frame, place on the field) its running was last counted from
        self._ran_from: Dict[int, Tuple[int, np.ndarray]] = {}
        # Who held the disc last: track, team, place on the field (if known), frame
        self._last_held: Optional[Tuple[int, Optional[int], Optional[np.ndarray], int]] = None

    # ------------------------------------------------------------------ the video

    def could_be(self, track_id: int, other_id: int) -> bool:
        """Whether two tracks may be one player, by when and where they were seen and
        what they look like."""
        return self.book.could_be(track_id, other_id, self.frame_rate)

    def players_of_tracks(self) -> Dict[int, int]:
        """Track -> player, looking back over the tracks since the last cut: the tracks
        that turn out to be one player carry the name of the first of them."""
        return self.book.players(self.frame_rate)

    def new_video(self, video_path: Optional[str]) -> None:
        """Another video: its roster is read if there is one, the last one's written."""
        self.save()
        self._video_path = video_path
        self.roster = PlayerRoster()
        self._roster_path = None
        self._forget_tracks()
        folder = get_setting("models.reid.roster_folder", "data/cache/rosters")
        if video_path and folder:
            # A roster holds what the players look like to one network, which says
            # nothing to another: one chosen in place of the settings' keeps its own
            folder = (
                Path(folder) if self._weights is None else Path(folder) / self._weights.parent.name
            )
            self._roster_path = folder / f"{Path(video_path).stem}.json"
            if self._roster_path.exists():
                try:
                    self.roster = PlayerRoster.load(self._roster_path)
                except (OSError, ValueError, KeyError) as error:
                    logger.warning(f"Roster not read, starting a new one: {error}")

    def set_weights(self, weights: Optional[str]) -> bool:
        """Look at the players with another network (None: the one in the settings).
        What was known of the players by the last one does not hold for this one: the
        tracks are looked at anew, and the video's roster is that of this network.
        Returns whether anything changed."""
        chosen = None if weights is None else Path(weights)
        if chosen is not None and chosen == Path(str(get_setting("models.reid.weights", ""))):
            chosen = None
        if chosen == self._weights:
            return False
        self.roster.tracks_ended()
        self._weights = chosen
        self._embedder, self._tried_to_load = None, False
        self.new_video(self._video_path)
        return True

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
        self.book.rename(track_id, new_track_id)
        if self._looking is not None:
            self._renamed[track_id] = new_track_id
        for kept in (self._sharpness, self._ran_from):
            if track_id in kept:
                kept[new_track_id] = kept.pop(track_id)
        if self._last_held is not None and self._last_held[0] == track_id:
            self._last_held = (new_track_id, *self._last_held[1:])
        profile = self._profiles.pop(track_id, None)
        if profile is not None:
            known = self._profiles.get(new_track_id)
            if known is None:
                self._profiles[new_track_id] = profile
            else:
                known.take_over(profile)

    def swap(self, first: int, second: int) -> None:
        """Two tracks exchanged names (see `swapped`): what belongs to the box goes with
        the box. What a player looks like belongs to the name and stays."""
        for kept in (self._sharpness, self._ran_from):
            mine, theirs = kept.pop(first, None), kept.pop(second, None)
            for track_id, value in ((first, theirs), (second, mine)):
                if value is not None:
                    kept[track_id] = value
        renamed = {first: second, second: first}
        self._crossings = {
            tuple(sorted(renamed.get(t, t) for t in pair)): until
            for pair, until in self._crossings.items()
        }
        self._overlapping = {renamed.get(t, t) for t in self._overlapping}

    def _forget_tracks(self) -> None:
        self.book.clear()
        self._looked_at = None
        self._frame = self._last_held = None
        self._profiles.clear()
        self._sharpness.clear()
        self._ran_from.clear()
        self._looking = None  # What it finds is of tracks that are over
        self._renamed.clear()
        self._crossings.clear()
        self._overlapping = set()

    # ------------------------------------------------------------------ a frame

    def _network(self) -> Any:
        if not self._tried_to_load:
            self._tried_to_load = True
            weights = self._weights or Path(str(get_setting("models.reid.weights", "")))
            try:
                from .reid import load_embedder

                if weights.is_file():
                    # Putting a network on the GPU is work there like running one
                    with GPU_SETUP_LOCK.work():
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
        players = [track for track in tracks if getattr(track, "class_name", None) == "player"]
        present = [int(track.track_id) for track in players]
        places = {track: np.asarray(place, dtype=float) for track, place in (places or {}).items()}
        # Whether or not the network is there yet to look at them
        for track in players:
            track_id = int(track.track_id)
            self.book.see(
                frame_index,
                track_id,
                getattr(track, "team", None),
                numbers.get(track_id, ""),
                places.get(track_id),
            )
        network = self._network()
        if network is None:
            return {}
        for track in players:
            profile = self._profiles.setdefault(
                int(track.track_id), _Profile(first=frame_index, last=frame_index)
            )
            profile.last = frame_index
            if getattr(track, "team", None) is not None:
                profile.team = track.team
        self._watch_crossings(players, frame_index)
        looked = self._look(network, frame, players, frame_index)
        for track_id, number in numbers.items():
            self.roster.number_read(int(track_id), number)
        if looked:
            self.roster.match(present)

        self._count_time(present, holder_id, frame_index)
        self._count_running(frame_index, places)
        self._count_passes(frame_index, holder_id, players, places)
        self._forget_old(frame_index)

        return {
            track_id: number
            for track_id in present
            if track_id not in numbers and (number := self.roster.number_of(track_id))
        }

    def _look(
        self, network: Any, frame: np.ndarray, players: Sequence[Any], frame_index: int
    ) -> bool:
        """Look at the players of a frame, every few frames, on the second thread unless
        told otherwise; returns whether the roster learned anything in this frame."""
        from .reid import embed, sharpness

        # All players in one frame, and none in the frames between: the network takes
        # a crop or twenty in about the same time
        in_background = bool(get_setting("models.reid.in_background", True))
        every = max(1, int(get_setting("models.reid.look_every_frames", 6)))
        # A look under way is taken in when it is done; until then no other is begun
        learned = self._take_in_look(wait=not in_background)
        due = self._looked_at is None or not 0 <= frame_index - self._looked_at < every
        if not due or self._looking is not None:
            return learned
        self._looked_at = frame_index

        height, width = frame.shape[:2]
        crops, whose = [], []
        for track in players:
            track_id = int(track.track_id)
            if track_id in self._overlapping:
                continue  # The box shows two players: whose looks would these be
            x1, y1, x2, y2 = (int(round(value)) for value in track.to_ltrb())
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 - x1 < MIN_CROP[0] or y2 - y1 < MIN_CROP[1]:
                continue
            # A copy: the frame may be drawn on before the network has seen the crop
            crop = frame[y1:y2, x1:x2].copy() if in_background else frame[y1:y2, x1:x2]
            history = self._sharpness.setdefault(track_id, [])
            history.append(sharpness(crop))
            del history[:-SHARPNESS_MEMORY]
            if len(history) >= 5 and history[-1] < np.quantile(history, SKIP_BLURRED):
                continue
            crops.append(crop)
            whose.append((track_id, getattr(track, "team_colour", None)))
        if not crops:
            return learned
        if in_background:
            if self._thread is None:
                self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="looks")
            self._looking = (self._thread.submit(_embed_beside, embed, network, crops), whose)
            return learned
        done: Future = Future()
        done.set_result(_embed_beside(embed, network, crops))
        self._looking = (done, whose)
        return self._take_in_look(wait=True)

    def _take_in_look(self, wait: bool) -> bool:
        """Give the roster the vectors of the look under way, if it is done (or, with
        `wait`, when it is); returns whether there were any."""
        if self._looking is None:
            return False
        future, whose = self._looking
        if not wait and not future.done():
            return False
        self._looking = None
        renamed, self._renamed = self._renamed, {}
        try:
            vectors = future.result()
        except Exception as error:  # The network failed on this batch: the next will tell
            logger.warning(f"Players could not be looked at: {error}")
            return False
        for (track_id, team_colour), vector in zip(whose, vectors):
            track_id = renamed.get(track_id, track_id)
            if self._crossing(track_id):
                # Not known yet whose looks these are: kept until the crossing is settled
                self._profiles[track_id].held.append((np.asarray(vector), team_colour))
            else:
                self._learn(track_id, vector, team_colour)
        return True

    def _learn(self, track_id: int, vector: np.ndarray, team_colour: Any) -> None:
        """Take a look at a track into what is known of its player."""
        self.roster.look(track_id, vector, team_colour)
        self.book.learn(track_id, vector)
        profile = self._profiles.get(track_id)
        if profile is not None:
            profile.learn(vector)

    def _count_time(
        self, present: Sequence[int], holder_id: Optional[int], frame_index: int
    ) -> None:
        """Count the time the players were seen, and the holder had the disc, for."""
        if self._frame is not None and 0 < frame_index - self._frame:
            seconds = min((frame_index - self._frame) / self.frame_rate, MAX_FRAME_SECONDS)
            for track_id in present:
                self.roster.count(track_id, "seconds_seen", seconds)
            if holder_id in present:
                self.roster.count(holder_id, "seconds_with_disc", seconds)
        self._frame = frame_index

    def _forget_old(self, frame_index: int) -> None:
        """Tell the roster of the tracks that are over, and drop what is kept of tracks
        gone for good."""
        ended = [
            track_id
            for track_id, profile in self._profiles.items()
            if not profile.ended
            and frame_index - profile.last > TRACK_ENDED_SECONDS * self.frame_rate
        ]
        if ended:
            self.roster.tracks_ended(ended)
            for track_id in ended:
                self._profiles[track_id].ended = True
                self._sharpness.pop(track_id, None)
                self._ran_from.pop(track_id, None)
        remembered = self.frame_rate * max(
            PROFILE_SECONDS, float(get_setting("models.reid.rejoin_seconds", 0.0))
        )
        for track_id in [t for t, p in self._profiles.items() if frame_index - p.last > remembered]:
            del self._profiles[track_id]

    # ------------------------------------------------------------------ crossings

    def _crossing(self, track_id: int) -> bool:
        return any(track_id in pair for pair in self._crossings)

    def _watch_crossings(self, players: Sequence[Any], frame_index: int) -> None:
        """Note which players' boxes overlap in this frame."""
        self._overlapping = set()
        if len(players) < 2 or not get_setting("models.reid.check_crossings", True):
            return
        overlap = box_overlaps(np.array([track.to_ltrb() for track in players], dtype=np.float64))
        for first, second in zip(*np.nonzero(np.triu(overlap >= CROSSING_OVERLAP, 1))):
            pair = tuple(sorted((int(players[first].track_id), int(players[second].track_id))))
            self._crossings[pair] = frame_index
            self._overlapping.update(pair)

    def swapped(self, present: Sequence[int], frame_index: int) -> List[Tuple[int, int]]:
        """Which pairs of tracks have changed players while their boxes overlapped.

        For every crossing that is over: each of the two as it looks now, against what
        both looked like before. Asked for once per frame; a pair is given once, and the
        caller then exchanges the two tracks' players (nothing here is renamed: what was
        seen since the crossing is put with the player it is of).

        Args:
            present: The tracks in the frame
            frame_index: Position of the frame in the video
        """
        found = []
        for pair, until in list(self._crossings.items()):
            before = [self._profiles.get(track_id) for track_id in pair]
            gone_for = (frame_index - until) / self.frame_rate
            given_up = (
                gone_for > CROSSING_SECONDS
                or (any(t not in present for t in pair) and gone_for > TRACK_ENDED_SECONDS)
                or any(p is None or p.looks < REJOIN_LOOKS for p in before)
            )
            settled = (
                frame_index > until
                and not given_up
                and all(t in present for t in pair)  # Both there to be exchanged
                and all(len(p.held) >= CROSSING_LOOKS for p in before)
            )
            if not (given_up or settled):
                continue
            changed = False
            if settled:
                now = [_unit(np.mean([vector for vector, _ in p.held], axis=0)) for p in before]
                was = [p.mean for p in before]
                straight = float(now[0] @ was[0] + now[1] @ was[1])
                crossed = float(now[0] @ was[1] + now[1] @ was[0])
                changed = crossed - straight >= CROSSING_SWAP_MARGIN
            del self._crossings[pair]
            # What was held is of the player the track is with now; unless the track is
            # in another crossing still, which will say
            for track_id, other in (pair, pair[::-1]):
                profile = self._profiles.get(track_id)
                if profile is None or self._crossing(track_id):
                    continue
                held, profile.held = profile.held, []
                for vector, team_colour in held:
                    self._learn(other if changed else track_id, vector, team_colour)
            if changed:
                found.append(pair)
        return found

    # ------------------------------------------------------------------ what players do

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

    # ------------------------------------------------------------------ for the tracker

    def returned(
        self, present: Sequence[int], missing: Sequence[int], frame_index: int
    ) -> List[Tuple[int, int]]:
        """Which of the tracks in the frame are players the tracker lost a while ago.

        The tracker gives a player back their track if it finds them within a moment
        where they can have got to. After that it starts a new track, and only the
        player's looks say who it is: among the lost players of the same team who were
        gone before the track began, the one it is clearly most like.

        Args:
            present: The tracks in the frame
            missing: The players the tracker still remembers and does not see
            frame_index: Position of the frame in the video

        Returns:
            (track, lost player it is) for each track that is one; the caller joins the
            two, and tells this stage with `rename`
        """
        longest = float(get_setting("models.reid.rejoin_seconds", 0.0)) * self.frame_rate
        if longest <= 0:
            return []
        lost = [
            (track_id, self._profiles[track_id])
            for track_id in missing
            if track_id in self._profiles
            and self._profiles[track_id].looks >= REJOIN_LOOKS
            and frame_index - self._profiles[track_id].last <= longest
        ]
        offers = []
        for track_id in present:
            profile = self._profiles.get(track_id)
            if profile is None or profile.looks < REJOIN_LOOKS or profile.team is None:
                continue
            likeness = sorted(
                (
                    (float(profile.mean @ other.mean), other_id)
                    for other_id, other in lost
                    # Gone before this track began: otherwise they are two players
                    if other.last < profile.first and other.team == profile.team
                ),
                reverse=True,
            )
            if not likeness or likeness[0][0] < REJOIN_LIKENESS:
                continue
            if len(likeness) > 1 and likeness[0][0] - likeness[1][0] < REJOIN_LEAD:
                continue
            offers.append((likeness[0][0], track_id, likeness[0][1]))
        joined, taken = [], set()
        for _, track_id, earlier in sorted(offers, reverse=True):
            if earlier not in taken:
                taken.add(earlier)
                joined.append((track_id, earlier))
        return joined

    def teams(self, tracker_teams: Dict[int, Optional[int]]) -> Dict[int, int]:
        """The team of each track by its looks: the tracks looked at often enough, split
        in two by what they look like.

        Args:
            tracker_teams: Track -> the team (0 or 1) the tracker gives it by its shirt
                colour, or None. The split begins from these.

        Returns:
            Track -> team, for the tracks looked at often enough; empty while there are
            too few, or they do not fall into two groups, or the tracker has no teams yet
        """
        known = {
            track_id: profile.mean
            for track_id, profile in self._profiles.items()
            if profile.looks >= TEAM_LOOKS
        }
        if len(known) < TEAM_MIN_TRACKS:
            return {}
        ids = sorted(known)
        looks = np.array([known[track_id] for track_id in ids])
        likeness = looks @ looks.T
        # Begun from the tracker's two teams, and each player then put with the group
        # they are more like, until nobody moves: the split by shirt colour, put right.
        # Begun anywhere else, the looks may part the players by something else than
        # their kit: the two who stand far off, say.
        given = np.array([tracker_teams.get(track_id) for track_id in ids], dtype=object)
        if not all(np.count_nonzero(given == team) for team in (0, 1)):
            return {}
        centres = np.array([looks[given == team].mean(axis=0) for team in (0, 1)])
        group = np.full(len(ids), -1)
        for _ in range(10):
            centres /= np.linalg.norm(centres, axis=1, keepdims=True)
            now = np.argmax(looks @ centres.T, axis=1)
            if min(np.count_nonzero(now == 0), np.count_nonzero(now == 1)) < TEAM_MIN_GROUP:
                return {}
            if np.array_equal(now, group):
                break
            group = now
            centres = np.array([looks[group == g].mean(axis=0) for g in (0, 1)])
        same = group[:, None] == group[None, :]
        apart = ~np.eye(len(ids), dtype=bool)
        if likeness[same & apart].mean() - likeness[~same].mean() < TEAM_MIN_GAP:
            return {}
        return {track_id: int(g) for track_id, g in zip(ids, group)}
