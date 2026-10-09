"""Which tracks are one player: the tracks of a stretch of play joined after the fact.

The tracker follows a player for as long as it sees them and starts a new track when it
finds them again. Looking back over a whole stretch, more can be said than in the frame
a track began: two tracks are one player only if they never show at the same time, are
of one team, do not carry different numbers, and lie no further apart than a player
runs in the time between them. Among the tracks that may be one player by all that,
the ones that look alike are joined, most alike first, and only where no other track
that would exclude the partner looks nearly as much alike. Where the field's place is
not known, so that nothing says how far apart two tracks lie, only numbers join them.

`TrackletBook` is told of every track in every frame and of every look taken at it;
`players()` gives the track -> player naming at any time.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Looks from which a track's looks are trusted to tell whose it is
MIN_LOOKS = 3
# How alike (cosine) two tracks must look to be joined, and by how much more than any
# rival: a track that could be the partner's player instead, but not together with it
MIN_LIKENESS = 0.5
MIN_LEAD = 0.1
# Two tracks that look less alike than this are not one player, whatever else says so
LEAST_LIKENESS = 0.2
# A player's top speed in field units (yards) a second, and what the place on the field
# may be off by
TOP_SPEED = 11.0
REACH_SLACK = 4.0
# The share of a track's sightings in one team from which it is of that team
TEAM_SHARE = 0.75


@dataclass
class Tracklet:
    """What is known of one track of the tracker."""

    track_id: int
    first: int
    last: int
    number: str = ""
    looks: int = 0
    total: Optional[np.ndarray] = None
    first_place: Optional[np.ndarray] = None
    last_place: Optional[np.ndarray] = None
    team_sightings: List[int] = field(default_factory=lambda: [0, 0])

    @property
    def team(self) -> Optional[int]:
        seen = sum(self.team_sightings)
        if not seen:
            return None
        team = int(self.team_sightings[1] > self.team_sightings[0])
        return team if self.team_sightings[team] >= TEAM_SHARE * seen else None

    def learn(self, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float64)
        self.total = vector.copy() if self.total is None else self.total + vector
        self.looks += 1

    def take_over(self, other: "Tracklet") -> None:
        """Another track turned out to be this one's player all along."""
        if other.first < self.first:
            self.first, self.first_place = other.first, other.first_place
        if other.last > self.last:
            self.last, self.last_place = other.last, other.last_place
        self.number = self.number or other.number
        if other.total is not None:
            self.total = other.total.copy() if self.total is None else self.total + other.total
        self.looks += other.looks
        self.team_sightings = [a + b for a, b in zip(self.team_sightings, other.team_sightings)]


def may_be_one_player(first: Tracklet, second: Tracklet, frame_rate: float) -> bool:
    """Whether nothing known of two tracks excludes that they are one player."""
    earlier, later = (first, second) if first.first <= second.first else (second, first)
    if later.first <= earlier.last:
        return False  # Seen at the same time
    if None not in (first.team, second.team) and first.team != second.team:
        return False
    if first.number and second.number and first.number != second.number:
        return False
    if earlier.last_place is not None and later.first_place is not None:
        seconds = (later.first - earlier.last) / frame_rate
        gap = float(np.linalg.norm(later.first_place - earlier.last_place))
        if gap > TOP_SPEED * seconds + REACH_SLACK:
            return False
    return True


def reach_known(first: Tracklet, second: Tracklet) -> bool:
    """Whether it is known where the earlier of two tracks ended and the later began."""
    earlier, later = (first, second) if first.first <= second.first else (second, first)
    return earlier.last_place is not None and later.first_place is not None


def _mean(group: Sequence[Tracklet]) -> Optional[np.ndarray]:
    """What a group of tracks looks like, or None if too little was seen of it."""
    known = [t for t in group if t.total is not None]
    if sum(t.looks for t in known) < MIN_LOOKS:
        return None
    total = np.sum([t.total for t in known], axis=0)
    return total / max(float(np.linalg.norm(total)), 1e-12)


def connect(
    tracklets: Iterable[Tracklet],
    frame_rate: float,
    min_likeness: float = MIN_LIKENESS,
    min_lead: float = MIN_LEAD,
) -> List[List[Tracklet]]:
    """Group tracks into players.

    Args:
        tracklets: The tracks of a stretch of play
        frame_rate: Frames per second of the video
        min_likeness: How alike two groups must look to be joined
        min_lead: By how much more alike than any rival

    Returns:
        The groups, each the tracks of one player
    """
    groups: List[List[Tracklet]] = [[t] for t in tracklets]

    def compatible(one: List[Tracklet], other: List[Tracklet]) -> bool:
        return all(may_be_one_player(a, b, frame_rate) for a in one for b in other)

    def placed(one: List[Tracklet], other: List[Tracklet]) -> bool:
        return all(reach_known(a, b) for a in one for b in other)

    def numbered_alike(one: List[Tracklet], other: List[Tracklet]) -> bool:
        numbers = {t.number for t in one if t.number}, {t.number for t in other if t.number}
        teams = {t.team for t in one + other if t.team is not None}
        return bool(numbers[0] & numbers[1]) and len(teams) == 1

    while len(groups) > 1:
        count = len(groups)
        allowed = np.zeros((count, count), dtype=bool)
        for i in range(count):
            for j in range(i + 1, count):
                allowed[i, j] = allowed[j, i] = compatible(groups[i], groups[j])
        # The same number in the same team is the same player, whatever they look like
        pair = next(
            (
                (i, j)
                for i in range(count)
                for j in range(i + 1, count)
                if allowed[i, j] and numbered_alike(groups[i], groups[j])
            ),
            None,
        )
        if pair is None:
            means = [_mean(group) for group in groups]
            likeness = np.full((count, count), -1.0)
            for i in range(count):
                for j in range(i + 1, count):
                    if not allowed[i, j] or means[i] is None or means[j] is None:
                        continue
                    # Looks alone do not tell teammates apart well enough: only tracks
                    # known to lie within a player's reach of each other are joined by them
                    if placed(groups[i], groups[j]):
                        likeness[i, j] = likeness[j, i] = float(means[i] @ means[j])
            best = 0.0
            for i in range(count):
                for j in range(i + 1, count):
                    if likeness[i, j] < max(min_likeness, best):
                        continue
                    # Rivals: who else either could be, but not together with the other
                    rivals = [
                        likeness[a, c]
                        for a, b in ((i, j), (j, i))
                        for c in range(count)
                        if c not in (i, j) and not allowed[b, c]
                    ]
                    if likeness[i, j] - max(rivals, default=-1.0) >= min_lead:
                        best, pair = likeness[i, j], (i, j)
        if pair is None:
            break
        i, j = pair
        groups[i] = groups[i] + groups[j]
        del groups[j]
    return groups


class TrackletBook:
    """The tracks of the stretch of play under way, and whose they are."""

    def __init__(self) -> None:
        self._tracklets: Dict[int, Tracklet] = {}
        self._renamed: Dict[int, int] = {}

    def clear(self) -> None:
        self._tracklets.clear()
        self._renamed.clear()

    def see(
        self,
        frame_index: int,
        track_id: int,
        team: Optional[int] = None,
        number: str = "",
        place: Optional[Sequence[float]] = None,
    ) -> None:
        """A track is in a frame."""
        tracklet = self._tracklets.get(track_id)
        if tracklet is None:
            tracklet = self._tracklets[track_id] = Tracklet(track_id, frame_index, frame_index)
        tracklet.last = max(tracklet.last, frame_index)
        if team in (0, 1):
            tracklet.team_sightings[team] += 1
        if number:
            tracklet.number = number
        if place is not None:
            place = np.asarray(place, dtype=np.float64)
            if tracklet.first_place is None:
                tracklet.first_place = place
            tracklet.last_place = place

    def learn(self, track_id: int, vector: np.ndarray) -> None:
        """A look was taken at a track."""
        tracklet = self._tracklets.get(track_id)
        if tracklet is not None:
            tracklet.learn(vector)

    def rename(self, track_id: int, new_track_id: int) -> None:
        """The tracker itself found a track to be a player it knew."""
        self._renamed[track_id] = new_track_id
        tracklet = self._tracklets.pop(track_id, None)
        if tracklet is None:
            return
        known = self._tracklets.get(new_track_id)
        if known is None:
            tracklet.track_id = new_track_id
            self._tracklets[new_track_id] = tracklet
        else:
            known.take_over(tracklet)

    def could_be(self, track_id: int, other_id: int, frame_rate: float) -> bool:
        """Whether two tracks may be one player by when and where they were seen and what
        they look like. True where too little is known of either to say."""
        one, other = self._tracklets.get(track_id), self._tracklets.get(other_id)
        if one is None or other is None:
            return True
        if not may_be_one_player(one, other, frame_rate):
            return False
        looks, other_looks = _mean([one]), _mean([other])
        if looks is None or other_looks is None:
            return True
        return float(looks @ other_looks) >= LEAST_LIKENESS

    def players(self, frame_rate: float) -> Dict[int, int]:
        """Track -> player, the player named by the first of their tracks. Tracks the
        tracker renamed are in it under their old names, too."""
        naming = {}
        for group in connect(self._tracklets.values(), frame_rate):
            player = min(group, key=lambda t: (t.first, t.track_id)).track_id
            for tracklet in group:
                naming[tracklet.track_id] = player
        for was in self._renamed:
            now, seen = was, set()
            while now in self._renamed and now not in seen:
                seen.add(now)
                now = self._renamed[now]
            if now in naming:
                naming[was] = naming[now]
        return naming
