"""The jersey numbers of the players on screen, kept from frame to frame.

Only some players are read in a frame (processing/player_id.py), and a reading may say
nothing. This keeps for every player on screen the number last read, falls back to the
number most of their readings agreed on (processing/jersey_tracker.py), and uses a number
that is read often enough to tell that a "new" player is one who had gone missing.
"""

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from ..config.settings import get_setting
from ..utils.logger import get_logger
from .jersey_crops import JerseyCropSelector
from .jersey_tracker import get_best_jersey_number, merge_jersey_readings, reset_jersey_tracker
from .player_id import discard_pending_readings, run_player_id_on_tracks
from .tracking import kit_distance, merge_players, missing_players, team_of_player

logger = get_logger("PLAYER_NUMBERS")

NO_NUMBER = ("Unknown", None, "")


class PlayerNumbers:
    """Jersey numbers by player ID, for the players on screen."""

    def __init__(self) -> None:
        # Player ID -> (number or "Unknown", details of the reading)
        self.numbers: Dict[int, Tuple[str, Any]] = {}
        self.crops = JerseyCropSelector()
        self._finalized: Set[int] = set()  # Players whose number is settled

    def reset(self, readings_too: bool = False) -> None:
        """Forget the numbers (after a seek); with `readings_too` also every reading so far
        (after switching the reader)."""
        self.crops.reset()
        discard_pending_readings()
        if readings_too:
            reset_jersey_tracker()
        self.numbers.clear()
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
            background=bool(get_setting("models.player_id.background_reading", False)),
        )
        self.numbers.update(read)

        # Players not read this frame, or read as unknown, take the number most of their
        # readings so far agreed on
        for track in tracks:
            current = self.numbers.get(track.track_id)
            if current is None or current[0] in NO_NUMBER:
                best_number, best_probability = get_best_jersey_number(track.track_id)
                if best_number and best_probability > 0.0:
                    details = (current[1] if current else None) or {}
                    details["best_tracked"] = {
                        "jersey_number": best_number,
                        "probability": best_probability,
                    }
                    self.numbers[track.track_id] = (best_number, details)

        renamed = self._merge_by_number(tracks)

        # The numbers of players who are gone are dropped
        on_screen = {track.track_id for track in tracks}
        for player_id in [player for player in self.numbers if player not in on_screen]:
            del self.numbers[player_id]
        return timing, renamed

    def _merge_by_number(self, tracks: List[Any]) -> List[Tuple[int, int]]:
        """A player whose number is that of a missing player is that player.

        Place and looks could not decide who a new track was when it appeared; the jersey
        number can, once it has been read often enough. Both teams may have the same
        number, so the two must be of the same team where the tracker knows their teams,
        and wear the same kit.
        """
        certainty_needed = float(get_setting("models.tracking.identity.number_certainty", 0.6))
        max_distance = float(get_setting("models.tracking.identity.max_kit_distance", 60.0))
        present = {track.track_id for track in tracks if track.class_name == "player"}
        missing = {}
        for player_id in missing_players(present):
            number, certainty = get_best_jersey_number(player_id)
            if number and certainty >= certainty_needed:
                missing.setdefault(number, player_id)

        renamed = []
        for track in tracks:
            player_id = track.track_id
            if track.class_name != "player" or not missing:
                continue
            number, certainty = get_best_jersey_number(player_id)
            earlier = missing.get(number) if number and certainty >= certainty_needed else None
            if earlier is None:
                continue
            teams = team_of_player(player_id), team_of_player(earlier)
            if None not in teams and teams[0] != teams[1]:
                continue
            distance = kit_distance(player_id, earlier)
            if distance is None or distance > max_distance:
                continue

            merge_players(player_id, earlier)
            merge_jersey_readings(player_id, earlier)
            track.track_id = earlier
            if player_id in self.numbers:
                self.numbers[earlier] = self.numbers.pop(player_id)
            if player_id in self._finalized:
                self._finalized.discard(player_id)
                self._finalized.add(earlier)
            renamed.append((player_id, earlier))
            del missing[number]
            logger.info(f"Player {player_id} is player {earlier} again (number {number})")
        return renamed
