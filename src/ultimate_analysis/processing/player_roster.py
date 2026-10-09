"""The players of a game, known by their looks across its points.

The tracker follows a player for as long as it can; at a cut, a close-up or a new point
it starts again and knows nobody. A jersey number says who a player is, but it is read
for some players some of the time. The roster keeps, for a whole video, one entry per
player: what they look like (a feature vector, see reid.py), which team they are in,
which number was read on them, and what they did. A track is matched to the entry it
looks like; from then on what is known of the entry is known of the track.

- A number read on one track names its entry, and with it every other track of that
  player, also where the number was never in view.
- What a player does is counted on the entry from the first moment, before anyone knows
  their number. When the number is read, it is the name of what was counted.
- Two tracks that are on the field at the same moment are two players, however alike.

The roster can be written to a file and read again, so a game can be worked on in parts.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# A track is matched once it has this many looks at its player: one crop is one pose
MIN_LOOKS = 15
# How alike (cosine, 1 = the same) a track and an entry must be to be one player, and by
# how much the best entry must beat the next: two teammates may both be somewhat alike
# (chosen on one game and confirmed on another: docs/MEASUREMENTS.md)
MIN_LIKENESS = 0.8
MIN_LEAD = 0.05
# A number counts as an entry's once it was the reading of this many of its tracks
MIN_NUMBER_VOTES = 1
# Two team colours (BGR) nearer than this are one team
SAME_TEAM_COLOUR = 80.0


@dataclass
class Entry:
    """A player of the game."""

    entry: int
    team: int  # 0 or 1: the roster's own count, kept over the whole video
    vector: np.ndarray  # What they look like: the mean over their looks, length 1
    looks: int = 0
    numbers: Counter = field(default_factory=Counter)  # Number -> tracks it was read on
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def number(self) -> Optional[str]:
        """The jersey number, if one was read."""
        if not self.numbers:
            return None
        number, votes = self.numbers.most_common(1)[0]
        return number if votes >= MIN_NUMBER_VOTES else None


@dataclass
class _Track:
    """A track the tracker follows at the moment."""

    team: Optional[int] = None
    total: Optional[np.ndarray] = None  # Sum of the vectors of its looks
    looks: int = 0
    entry: Optional[int] = None
    number: Optional[str] = None


class PlayerRoster:
    """One entry per player of a video, and which entry each live track is."""

    def __init__(self) -> None:
        self.entries: Dict[int, Entry] = {}
        self._tracks: Dict[int, _Track] = {}
        self._team_colours: List[np.ndarray] = []  # Of the roster's teams 0 and 1
        self._next_entry = 1

    # ------------------------------------------------------------------ what is seen

    def team_of(self, colour: Optional[Sequence[float]]) -> Optional[int]:
        """The roster's team for a shirt colour (BGR): the tracker counts its teams
        anew after every cut, the colours stay."""
        if colour is None:
            return None
        colour = np.asarray(colour, dtype=np.float64)
        distances = [float(np.linalg.norm(colour - known)) for known in self._team_colours]
        if distances and (min(distances) <= SAME_TEAM_COLOUR or len(self._team_colours) >= 2):
            return int(np.argmin(distances))
        self._team_colours.append(colour)
        return len(self._team_colours) - 1

    def look(
        self, track_id: int, vector: np.ndarray, team_colour: Optional[Sequence[float]] = None
    ) -> None:
        """Take in the vector of one crop of a track."""
        track = self._tracks.setdefault(track_id, _Track())
        vector = np.asarray(vector, dtype=np.float64)
        track.total = vector.copy() if track.total is None else track.total + vector
        track.looks += 1
        team = self.team_of(team_colour)
        if team is not None:
            track.team = team
        if track.entry is not None:
            self._add_look(self.entries[track.entry], vector)

    def number_read(self, track_id: int, number: str) -> None:
        """A jersey number was read on a track with certainty."""
        track = self._tracks.setdefault(track_id, _Track())
        if track.number == number:
            return
        if track.entry is not None and track.number is not None:
            self.entries[track.entry].numbers[track.number] -= 1
        track.number = number
        if track.entry is not None:
            self._name(track.entry, number)

    def count(self, track_id: int, what: str, amount: float = 1.0) -> None:
        """Add to what a track's player did (for a track without an entry yet: nothing;
        a few frames of a track that is never matched are no player's)."""
        track = self._tracks.get(track_id)
        if track is not None and track.entry is not None:
            stats = self.entries[track.entry].stats
            stats[what] = stats.get(what, 0.0) + amount

    def rename(self, track_id: int, new_track_id: int) -> None:
        """A track turned out to be one the tracker knew before, and goes on under that
        name: what was seen of it is the other's."""
        track = self._tracks.pop(track_id, None)
        if track is None:
            return
        known = self._tracks.get(new_track_id)
        if known is None:
            self._tracks[new_track_id] = track
            return
        if track.total is not None:
            known.total = track.total if known.total is None else known.total + track.total
            known.looks += track.looks
        known.team = known.team if known.team is not None else track.team
        known.number = known.number or track.number
        known.entry = known.entry if known.entry is not None else track.entry

    def tracks_ended(self, track_ids: Optional[Sequence[int]] = None) -> None:
        """The tracker has dropped these tracks, or all of them (a cut): their numbers
        will be given to other players. The entries stay."""
        for track_id in list(self._tracks) if track_ids is None else track_ids:
            self._tracks.pop(track_id, None)

    # ------------------------------------------------------------------ matching

    def match(self, present: Sequence[int]) -> Dict[int, int]:
        """Match the tracks that have enough looks; returns track -> entry for the tracks
        present that have one.

        Args:
            present: The tracks in the frame: none of them is another one's player
        """
        taken = {
            self._tracks[t].entry for t in present if t in self._tracks and self._tracks[t].entry
        }
        waiting = [
            t
            for t in present
            if t in self._tracks
            and self._tracks[t].entry is None
            and self._tracks[t].looks >= MIN_LOOKS
            and self._tracks[t].team is not None
        ]
        # The surest first: a track takes the entry it is most like, and the next track
        # cannot have it
        offers = []
        for track_id in waiting:
            track = self._tracks[track_id]
            mean = track.total / max(float(np.linalg.norm(track.total)), 1e-12)
            likeness = sorted(
                (
                    (float(entry.vector @ mean), entry.entry)
                    for entry in self.entries.values()
                    if entry.team == track.team
                ),
                reverse=True,
            )
            offers.append((likeness[0][0] if likeness else -1.0, track_id, likeness))
        for _, track_id, likeness in sorted(offers, reverse=True):
            track = self._tracks[track_id]
            # An entry offered may be gone by now: naming a track matched before this
            # one can show two entries to be one player
            free = [
                (value, entry)
                for value, entry in likeness
                if entry not in taken and entry in self.entries
            ]
            found = None
            if free and free[0][0] >= MIN_LIKENESS:
                lead = free[0][0] - (free[1][0] if len(free) > 1 else -1.0)
                found = free[0][1] if lead >= MIN_LEAD else None
                if found is None:
                    continue  # Two entries are about as alike: wait for more looks
            if found is None:
                found = self._new_entry(track)
            else:
                entry = self.entries[found]
                weight = track.looks / (entry.looks + track.looks)
                mean = track.total / max(float(np.linalg.norm(track.total)), 1e-12)
                entry.vector = _unit((1 - weight) * entry.vector + weight * mean)
                entry.looks += track.looks
            track.entry = found
            taken.add(found)
            if track.number is not None:
                self._name(found, track.number)
        return {
            t: self._tracks[t].entry
            for t in present
            if t in self._tracks and self._tracks[t].entry is not None
        }

    def number_of(self, track_id: int) -> Optional[str]:
        """The jersey number of a track's player as far as the roster knows it."""
        track = self._tracks.get(track_id)
        if track is None or track.entry is None:
            return None
        return self.entries[track.entry].number

    def _new_entry(self, track: _Track) -> int:
        entry = Entry(self._next_entry, track.team, _unit(track.total), track.looks)
        self.entries[entry.entry] = entry
        self._next_entry += 1
        return entry.entry

    @staticmethod
    def _add_look(entry: Entry, vector: np.ndarray) -> None:
        entry.vector = _unit(entry.vector * entry.looks + vector)
        entry.looks += 1

    def _name(self, entry_id: int, number: str) -> None:
        """Give an entry a number reading; if another entry of the team has that number
        and is not on the field now, the two are one player."""
        entry = self.entries[entry_id]
        entry.numbers[number] += 1
        if entry.number != number:
            return
        live = {track.entry for track in self._tracks.values()}
        for other in list(self.entries.values()):
            if (
                other.entry != entry_id
                and other.team == entry.team
                and other.number == number
                and other.entry not in live
            ):
                self._merge(other, entry)

    def _merge(self, gone: Entry, into: Entry) -> None:
        weight = gone.looks / max(1, gone.looks + into.looks)
        into.vector = _unit((1 - weight) * into.vector + weight * gone.vector)
        into.looks += gone.looks
        into.numbers.update(gone.numbers)
        for what, amount in gone.stats.items():
            into.stats[what] = into.stats.get(what, 0.0) + amount
        del self.entries[gone.entry]

    # ------------------------------------------------------------------ on disk

    def table(self) -> List[dict]:
        """The roster as rows: entry, team and its shirt colour (BGR), number, looks,
        stats (without the vectors)."""
        colours = [tuple(int(round(value)) for value in colour) for colour in self._team_colours]
        return [
            {
                "entry": entry.entry,
                "team": entry.team,
                "team_colour": colours[entry.team] if entry.team < len(colours) else None,
                "number": entry.number or "",
                "looks": entry.looks,
                **{what: round(amount, 2) for what, amount in sorted(entry.stats.items())},
            }
            for entry in sorted(self.entries.values(), key=lambda e: (e.team, e.entry))
        ]

    def save(self, path: Path) -> None:
        """Write the roster, vectors included, to a file."""
        content = {
            "team_colours": [colour.tolist() for colour in self._team_colours],
            "next_entry": self._next_entry,
            "entries": [
                {
                    "entry": entry.entry,
                    "team": entry.team,
                    "looks": entry.looks,
                    "numbers": dict(entry.numbers),
                    "stats": entry.stats,
                    "vector": [round(float(value), 5) for value in entry.vector],
                }
                for entry in self.entries.values()
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(content))

    @classmethod
    def load(cls, path: Path) -> "PlayerRoster":
        """A roster written by `save`; no track is live in it."""
        content = json.loads(Path(path).read_text())
        roster = cls()
        roster._team_colours = [np.array(colour) for colour in content["team_colours"]]
        roster._next_entry = content["next_entry"]
        for saved in content["entries"]:
            roster.entries[saved["entry"]] = Entry(
                saved["entry"],
                saved["team"],
                np.array(saved["vector"]),
                saved["looks"],
                Counter(saved["numbers"]),
                dict(saved["stats"]),
            )
        return roster


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)
