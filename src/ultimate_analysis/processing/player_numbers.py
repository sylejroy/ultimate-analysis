"""The jersey numbers of the players on screen, kept from frame to frame.

Only some players are read in a frame (processing/player_id.py), and a reading may say
nothing. This keeps for every player on screen the number last read, falls back to the
number most of their readings agreed on (processing/jersey_tracker.py), and uses a number
that is read often enough to tell that a "new" player is one who had gone missing.

A number shown stays until another clearly leads it, and a team has one player of a
number: where two show it, the tracker has swapped them, and the votes went along.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from ..config.settings import get_setting
from ..utils.logger import get_logger
from .jersey_crops import JerseyCropSelector
from .jersey_tracker import (
    get_best_jersey_number,
    get_jersey_last_read,
    get_jersey_probabilities,
    get_jersey_read_count,
    merge_jersey_readings,
    reset_jersey_tracker,
)
from .player_id import discard_pending_readings, run_player_id_on_tracks
from .tracking import PlayerTracker

logger = get_logger("PLAYER_NUMBERS")

# Times a number must have been read on a player for them to be joined with another by
# it: a number misread twice on one box is not rare, and a join cannot be undone
MIN_READS_TO_JOIN = 3

# Share of the votes by which another number must lead the one a player shows to take its
# place. A digit that turns out to be part of a two-digit number needs no lead.
KEEP_LEAD = 0.2

NO_NUMBER = ("Unknown", None, "")


class PlayerNumbers:
    """Jersey numbers by player ID, for the players on screen."""

    def __init__(self, tracker: PlayerTracker) -> None:
        # The tracker whose players these are: joined by their numbers there
        self._tracker = tracker
        # Player ID -> (number or "Unknown", details of the reading)
        self.numbers: Dict[int, Tuple[str, Any]] = {}
        self._shown: Dict[int, str] = {}  # Player ID -> the number they went by last
        self.crops = JerseyCropSelector()
        self._finalized: Set[int] = set()  # Players whose number is settled
        # Asked before two players are joined by their number: whether anything else
        # known of them (when and where they were seen, what they look like) excludes it
        self.could_be: Optional[Callable[[int, int], bool]] = None

    def reset(self, readings_too: bool = False) -> None:
        """Forget the numbers (after a seek); with `readings_too` also every reading so far
        (after switching the reader)."""
        self.crops.reset()
        discard_pending_readings()
        if readings_too:
            reset_jersey_tracker()
        self.numbers.clear()
        self._shown.clear()
        self._finalized.clear()

    def update(
        self, frame: np.ndarray, tracks: List[Any], frame_index: int
    ) -> Tuple[Dict[str, float], List[Tuple[int, int]]]:
        """Read the players due this frame and bring the numbers up to date.

        A track that turns out to be a player who was missing is given that player's ID.

        Returns:
            (milliseconds the reading took by step, [(ID a track had, ID it has now)])
        """
        read, timing, self._finalized = run_player_id_on_tracks(
            frame,
            tracks,
            frame_index=frame_index,
            finalized_tracks=self._finalized,
            crop_selector=self.crops,
            background=bool(get_setting("models.player_id.background_reading", True)),
        )
        for player_id, (number, details) in read.items():
            self.numbers[player_id] = (self._to_show(player_id) or number, details)

        # Players not read this frame, or read as unknown, take the number most of their
        # readings so far agreed on
        for track in tracks:
            current = self.numbers.get(track.track_id)
            if current is None or current[0] in NO_NUMBER:
                number = self._to_show(track.track_id)
                if number:
                    details = (current[1] if current else None) or {}
                    details["best_tracked"] = {
                        "jersey_number": number,
                        "probability": get_best_jersey_number(track.track_id)[1],
                    }
                    self.numbers[track.track_id] = (number, details)

        renamed = self._merge_by_number(tracks)

        # The numbers of players who are gone are dropped
        on_screen = {track.track_id for track in tracks}
        for player_id in [player for player in self.numbers if player not in on_screen]:
            del self.numbers[player_id]
        self._one_of_a_number_per_team(tracks)
        return timing, renamed

    def _to_show(self, player_id: int) -> Optional[str]:
        """The number a player goes by, if any: the one the votes lead with, unless the
        one they go by already is not clearly led."""
        best, certainty = get_best_jersey_number(player_id)
        shown = self._shown.get(player_id)
        if not best:
            return shown
        if shown and best != shown and not (len(best) > len(shown) and shown in best):
            ranked = get_jersey_probabilities(player_id, top_k=10)
            held = next((sure for number, sure, _ in ranked if number == shown), 0.0)
            if certainty - held < KEEP_LEAD:
                return shown
        self._shown[player_id] = best
        return best

    def _one_of_a_number_per_team(self, tracks: List[Any]) -> None:
        """Where two players of a team show one number, only the one it was read on last
        shows it: the other was left with the votes of whoever the tracker swapped them with.
        A settled player is read no more and can lose it to a teammate. Putting them first
        was worse: 1,231 wrong frames against 1,061, since a swap leaves the wrong one settled."""
        wearers: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        for track in tracks:
            number = self.numbers.get(track.track_id, NO_NUMBER)[0]
            team = getattr(track, "team", None)
            if track.class_name == "player" and team is not None and str(number).isdigit():
                wearers[(team, str(number))].append(track.track_id)
        for (_, number), players in wearers.items():
            if len(players) > 1:
                keeper = max(players, key=lambda player: get_jersey_last_read(player, number))
                for player in players:
                    if player != keeper:
                        self.numbers[player] = ("Unknown", self.numbers[player][1])

    def join(self, track: Any, earlier: int) -> None:
        """A track is a player known before and missing since: it goes on as that player,
        with what was read on both."""
        player_id = track.track_id
        self._tracker.merge_players(player_id, earlier)
        merge_jersey_readings(player_id, earlier)
        track.track_id = earlier
        if player_id in self.numbers:
            self.numbers[earlier] = self.numbers.pop(player_id)
        # The player goes by the votes of both, not by what the newcomer went by
        self._shown.pop(player_id, None)
        self._shown.pop(earlier, None)
        shown = self._to_show(earlier)
        if shown and earlier in self.numbers:
            self.numbers[earlier] = (shown, self.numbers[earlier][1])
        if player_id in self._finalized:
            self._finalized.discard(player_id)
            self._finalized.add(earlier)

    def _merge_by_number(self, tracks: List[Any]) -> List[Tuple[int, int]]:
        """A player whose number is that of a missing player is that player.

        Place and looks could not decide who a new track was when it appeared; the jersey
        number can, once it has been read often enough. Both teams may have the same
        number, so the two must be of the same team and wear the same kit. While the
        tracker does not know the team of one of them, they are left apart: the number
        is read again later, and two players joined wrongly cannot be parted.
        """
        certainty_needed = float(get_setting("models.tracking.identity.number_certainty", 0.6))
        max_distance = float(get_setting("models.tracking.identity.max_kit_distance", 60.0))
        present = {track.track_id for track in tracks if track.class_name == "player"}
        missing = {}
        for player_id in self._tracker.missing_players(present):
            number, certainty = get_best_jersey_number(player_id)
            if number and certainty >= certainty_needed and _read_often(player_id, number):
                missing.setdefault(number, player_id)

        renamed = []
        for track in tracks:
            player_id = track.track_id
            if track.class_name != "player" or not missing:
                continue
            number, certainty = get_best_jersey_number(player_id)
            earlier = missing.get(number) if number and certainty >= certainty_needed else None
            if earlier is None or not _read_often(player_id, number):
                continue
            if self.could_be is not None and not self.could_be(player_id, earlier):
                continue  # Seen elsewhere at the time, or looking like somebody else
            teams = self._tracker.team_of_player(player_id), self._tracker.team_of_player(earlier)
            if None in teams or teams[0] != teams[1]:
                continue
            distance = self._tracker.kit_distance(player_id, earlier)
            if distance is None or distance > max_distance:
                continue

            self.join(track, earlier)
            renamed.append((player_id, earlier))
            del missing[number]
            logger.info(f"Player {player_id} is player {earlier} again (number {number})")
        return renamed


def _read_often(player_id: int, number: str) -> bool:
    """Whether a number was read on a player often enough to join two players by."""
    return get_jersey_read_count(player_id, number) >= MIN_READS_TO_JOIN
