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

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config.settings import get_setting

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


def _pairs(tracklets: Sequence[Tracklet], frame_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """`may_be_one_player` and `reach_known` of every two tracks at once, as two
    matrices: a call per pair takes seconds for the hundreds of tracks of a stretch."""
    first = np.array([t.first for t in tracklets], dtype=np.float64)
    last = np.array([t.last for t in tracklets], dtype=np.float64)
    # Of each pair the one in the row is the earlier (also when they begin together)
    row_earlier = first[:, None] <= first[None, :]
    earlier_last = np.where(row_earlier, last[:, None], last[None, :])
    later_first = np.maximum(first[:, None], first[None, :])
    same_time = later_first <= earlier_last
    team = np.array([-1 if t.team is None else t.team for t in tracklets])
    known = team >= 0
    other_team = known[:, None] & known[None, :] & (team[:, None] != team[None, :])
    codes = {number: code for code, number in enumerate(sorted({t.number for t in tracklets}))}
    code = np.array([codes[t.number] if t.number else -1 for t in tracklets])
    read = code >= 0
    other_number = read[:, None] & read[None, :] & (code[:, None] != code[None, :])
    # From where the earlier ended to where the later began, where both are known
    nowhere = np.full(
        max((len(t.first_place) for t in tracklets if t.first_place is not None), default=1),
        np.nan,
    )
    starts = np.array([nowhere if t.first_place is None else t.first_place for t in tracklets])
    ends = np.array([nowhere if t.last_place is None else t.last_place for t in tracklets])
    gap = np.linalg.norm(ends[:, None, :] - starts[None, :, :], axis=2)
    gap = np.where(row_earlier, gap, gap.T)
    placed = ~np.isnan(gap)
    seconds = (later_first - earlier_last) / frame_rate
    too_far = placed & (gap > TOP_SPEED * seconds + REACH_SLACK)
    allowed = ~(same_time | other_team | other_number | too_far)
    np.fill_diagonal(allowed, False)
    np.fill_diagonal(placed, False)
    return allowed, placed


def connect(
    tracklets: Iterable[Tracklet],
    frame_rate: float,
    min_likeness: float = MIN_LIKENESS,
    min_lead: float = MIN_LEAD,
    needs_place: bool = True,
) -> List[List[Tracklet]]:
    """Group tracks into players.

    Args:
        tracklets: The tracks of a stretch of play
        frame_rate: Frames per second of the video
        min_likeness: How alike two groups must look to be joined
        min_lead: By how much more alike than any rival
        needs_place: Join by looks only tracks known to lie within a player's reach of
            each other on the field (False: also where the field's place is not known)

    Returns:
        The groups, each the tracks of one player
    """
    groups: List[List[Tracklet]] = [[t] for t in tracklets]
    count = len(groups)
    if count < 2:
        return groups
    # Of every two groups, whether they may be one player and whether it is known how
    # far apart they lie: found once, over the tracks. A group made of two is to a third
    # what both were, so on a join only the joined group's row is done again; the group
    # joined in stays as a dead row that nothing is alike to. The tracks of a group all
    # carry one number and are of one team (others are never joined), and what the
    # group looks like is the mean over its tracks.
    allowed, placed = _pairs([group[0] for group in groups], frame_rate)
    number = np.array([group[0].number for group in groups], dtype=object)
    team = np.array([-1 if group[0].team is None else group[0].team for group in groups])
    dims = max((len(t.total) for t in tracklets if t.total is not None), default=1)
    # The sum of each group's looks, added up track by track as `_mean` does, and
    # how many; the mean where there are enough
    totals, looks = np.zeros((count, dims)), [0] * count
    means, has_mean = np.zeros((count, dims)), np.zeros(count, dtype=bool)
    alive = [True] * count
    version = [0] * count  # Counted up each time a group's row is done again

    def numbered_alike(i: int) -> np.ndarray:
        """Which groups carry the number of group i, in its team."""
        if not number[i]:
            return np.zeros(count, dtype=bool)
        one_team = ((team == team[i]) & (team[i] >= 0)) | ((team >= 0) != (team[i] >= 0))
        return (number == number[i]) & one_team

    def add_looks(i: int, tracklets: Sequence[Tracklet]) -> None:
        for tracklet in tracklets:
            if tracklet.total is not None:
                totals[i] += tracklet.total
                looks[i] += tracklet.looks
        has_mean[i] = looks[i] >= MIN_LOOKS
        if has_mean[i]:
            means[i] = totals[i] / max(float(np.linalg.norm(totals[i])), 1e-12)

    def looks_like(i: int) -> np.ndarray:
        """How alike group i looks to each, -1 where the looks are not to be gone by."""
        # Looks alone do not tell teammates apart well enough: only tracks known to
        # lie within a player's reach of each other are joined by them
        told = allowed[i] & has_mean & has_mean[i] & (placed[i] | (not needs_place))
        return np.where(told, means @ means[i], -1.0)

    for i in range(count):
        add_looks(i, groups[i])
    numbered = np.array([numbered_alike(i) for i in range(count)])
    likeness = np.array([looks_like(i) for i in range(count)])

    # The pairs that may be joined are kept in two heaps, so that a round looks at a
    # few of them and not at every pair: the pairs with one number, first in order,
    # and the pairs alike enough, the most alike first (of equals the later one). An
    # entry is of the row as it was when it was put in; one of a row done since, or
    # of a dead group, is stale and dropped when it comes up.
    floor = max(min_likeness, 0.0)

    def by_number(i: int, others: np.ndarray) -> List[Tuple[int, int, int]]:
        low, high = np.minimum(i, others), np.maximum(i, others)
        return list(zip((low * count + high).tolist(), low.tolist(), high.tolist()))

    def by_looks(i: int, others: np.ndarray) -> List[Tuple[float, int, int, int, int, int]]:
        # Plain Python numbers: the heap compares entries far more often than it is given
        low, high = np.minimum(i, others), np.maximum(i, others)
        place = -(low * count + high)
        low, high = low.tolist(), high.tolist()
        return [
            (alike, where, a, b, version[a], version[b])
            for alike, where, a, b in zip(
                (-likeness[i, others]).tolist(), place.tolist(), low, high
            )
        ]

    numbered_pairs = [
        pair
        for i in range(count)
        for pair in by_number(i, np.flatnonzero(allowed[i, i + 1 :] & numbered[i, i + 1 :]) + i + 1)
    ]
    alike_pairs = [
        pair
        for i in range(count)
        for pair in by_looks(i, np.flatnonzero(likeness[i, i + 1 :] >= floor) + i + 1)
    ]
    heapq.heapify(numbered_pairs)
    heapq.heapify(alike_pairs)
    while sum(alive) > 1:
        # The same number in the same team is the same player, whatever they look like
        pair = None
        while numbered_pairs and pair is None:
            _, i, j = numbered_pairs[0]
            if alive[i] and alive[j] and allowed[i, j] and numbered[i, j]:
                pair = (i, j)
            else:
                heapq.heappop(numbered_pairs)
        # Else the most alike pair that is more alike than any rival by the lead: who
        # else either could be, but not together with the other. A pair without the
        # lead now may have it once a rival is joined away: it goes back in.
        without_lead = []
        while alike_pairs and pair is None:
            entry = heapq.heappop(alike_pairs)
            alike, _, i, j, seen_i, seen_j = entry
            if version[i] != seen_i or version[j] != seen_j:
                continue
            rivals = -1.0
            for a, b in ((i, j), (j, i)):
                rival_of = ~allowed[b]
                rival_of[b] = False
                rivals = max(rivals, likeness[a, rival_of].max(initial=-1.0))
            if -alike - rivals >= min_lead:
                pair = (i, j)
            else:
                without_lead.append(entry)
        for entry in without_lead:
            heapq.heappush(alike_pairs, entry)
        if pair is None:
            break
        i, j = pair
        groups[i] = groups[i] + groups[j]
        alive[j] = False
        version[i] += 1
        version[j] += 1
        for matrix in (allowed, placed):
            matrix[i] &= matrix[j]
            matrix[:, i] = matrix[i]
            matrix[j, :] = matrix[:, j] = False
        number[i], team[i] = number[i] or number[j], max(team[i], team[j])
        numbered[i] = numbered[:, i] = numbered_alike(i)
        numbered[j, :] = numbered[:, j] = False
        add_looks(i, groups[j])
        likeness[i] = likeness[:, i] = looks_like(i)
        likeness[j, :] = likeness[:, j] = -1.0
        for entry in by_number(i, np.flatnonzero(allowed[i] & numbered[i])):
            heapq.heappush(numbered_pairs, entry)
        for entry in by_looks(i, np.flatnonzero(likeness[i] >= floor)):
            heapq.heappush(alike_pairs, entry)
    return [group for group, live in zip(groups, alive) if live]


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
        groups = connect(
            self._tracklets.values(),
            frame_rate,
            float(get_setting("models.reid.join_min_likeness", MIN_LIKENESS)),
            float(get_setting("models.reid.join_min_lead", MIN_LEAD)),
            bool(get_setting("models.reid.join_needs_place", True)),
        )
        for group in groups:
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
