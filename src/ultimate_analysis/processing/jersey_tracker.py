"""Jersey numbers per player, decided from many noisy readings.

A single reading is often wrong or partial ("7" off a "17"). Every reading is a vote,
weighted by the reader's confidence and by how central the digits sit on the player. A
number is only reported once it has been read more than once, and its certainty is its
share of the votes with one vote's worth always held back for "unknown": one reading can
never make a number certain, twenty agreeing ones nearly do.

Votes do not fade with time. A jersey number does not change during a game, and a player
turned away from the camera for a minute still wears the same one.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from ..config.settings import get_setting
from ..utils.logger import get_logger

logger = get_logger("JERSEY_TRACKER")

# A one-digit reading may be half of a two-digit number that is partly hidden. It backs the
# two-digit numbers it could be part of with this share of its vote.
PARTIAL_READING_SUPPORT = 0.5


@dataclass
class JerseyReading:
    """One reading of a jersey number."""

    jersey_number: str
    weight: float


class JerseyNumberTracker:
    """Collects the readings of each player and reports the number they add up to."""

    def __init__(self):
        self._readings: Dict[int, Deque[JerseyReading]] = defaultdict(deque)
        self.max_readings = int(get_setting("models.player_id.tracking.max_history_length", 40))
        self.min_readings = int(get_setting("models.player_id.tracking.min_readings", 2))
        self.unknown_weight = float(get_setting("models.player_id.tracking.unknown_weight", 1.0))
        self.center_bonus = float(
            get_setting("models.player_id.tracking.spatial_weight_center_bonus", 0.3)
        )
        self.center_region_width = float(
            get_setting("models.player_id.tracking.center_region_width", 0.4)
        )

    def _spatial_weight(self, center_x: float) -> float:
        """Weight of a reading by where the digits sit across the player's box (0 to 1).

        A number in the middle belongs to this player; one at the edge may be a neighbour's.
        Between 0.5 at the very edge and 1 + bonus in the middle.
        """
        distance = abs(center_x - 0.5)
        half_center = self.center_region_width / 2
        if distance <= half_center:
            return 1.0 + self.center_bonus * (1.0 - distance / half_center)
        return max(0.5, 1.0 - 0.5 * (distance - half_center) / (0.5 - half_center))

    def add_measurement(
        self, track_id: int, jersey_number: str, confidence: float, bbox_center_x: float = 0.5
    ) -> None:
        """Add a reading for a player.

        Args:
            track_id: The player
            jersey_number: Number that was read
            confidence: Confidence of the reader (0-1)
            bbox_center_x: Where the digits sit across the player's box (0-1)
        """
        if not jersey_number or jersey_number == "Unknown" or confidence <= 0:
            return
        readings = self._readings[track_id]
        weight = float(confidence) * self._spatial_weight(bbox_center_x)
        readings.append(JerseyReading(str(jersey_number), weight))
        while len(readings) > self.max_readings:
            readings.popleft()

    def get_top_probabilities(self, track_id: int, top_k: int = 3) -> List[Tuple[str, float, int]]:
        """The most likely numbers of a player.

        Returns:
            [(number, certainty, times read)], most certain first. Numbers read fewer
            times than `min_readings` are left out.
        """
        readings = self._readings.get(track_id)
        if not readings:
            return []

        votes: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for reading in readings:
            votes[reading.jersey_number] += reading.weight
            counts[reading.jersey_number] += 1

        # One-digit readings also back the two-digit numbers that contain the digit
        support = dict(votes)
        support_counts = dict(counts)
        for number in votes:
            if len(number) == 2:
                for digit in set(number):
                    support[number] += PARTIAL_READING_SUPPORT * votes.get(digit, 0.0)
                    support_counts[number] += counts.get(digit, 0)

        total = sum(votes.values()) + self.unknown_weight
        ranked = sorted(support, key=lambda number: support[number], reverse=True)
        return [
            (number, min(1.0, support[number] / total), counts[number])
            for number in ranked
            if support_counts[number] >= self.min_readings
        ][:top_k]

    def get_best_jersey_number(self, track_id: int) -> Tuple[Optional[str], float]:
        """(number, certainty) of a player, or (None, 0.0) while there is too little to go on."""
        top = self.get_top_probabilities(track_id, top_k=1)
        return (top[0][0], top[0][1]) if top else (None, 0.0)

    def tracked_ids(self) -> List[int]:
        """Players with at least one reading."""
        return [track_id for track_id, readings in self._readings.items() if readings]

    def merge(self, from_id: int, into_id: int) -> None:
        """Hand the readings of one player over to another; the two were the same person."""
        if from_id == into_id or from_id not in self._readings:
            return
        merged = self._readings[into_id]
        merged.extend(self._readings.pop(from_id))
        while len(merged) > self.max_readings:
            merged.popleft()


_jersey_tracker: Optional[JerseyNumberTracker] = None


def get_jersey_tracker() -> JerseyNumberTracker:
    """The jersey number tracker of the current video."""
    global _jersey_tracker
    if _jersey_tracker is None:
        _jersey_tracker = JerseyNumberTracker()
    return _jersey_tracker


def add_jersey_measurement(
    track_id: int, jersey_number: str, confidence: float, bbox_center_x: float = 0.5
) -> None:
    """Add a reading for a player (see JerseyNumberTracker.add_measurement)."""
    get_jersey_tracker().add_measurement(track_id, jersey_number, confidence, bbox_center_x)


def get_jersey_probabilities(track_id: int, top_k: int = 3) -> List[Tuple[str, float, int]]:
    """The most likely numbers of a player: [(number, certainty, times read)]."""
    return get_jersey_tracker().get_top_probabilities(track_id, top_k)


def get_best_jersey_number(track_id: int) -> Tuple[Optional[str], float]:
    """(number, certainty) of a player, or (None, 0.0)."""
    return get_jersey_tracker().get_best_jersey_number(track_id)


def merge_jersey_readings(from_id: int, into_id: int) -> None:
    """Hand the readings of one player over to another; the two were the same person."""
    get_jersey_tracker().merge(from_id, into_id)


def reset_jersey_tracker() -> None:
    """Forget all readings (new video, tracker reset)."""
    global _jersey_tracker
    _jersey_tracker = None
