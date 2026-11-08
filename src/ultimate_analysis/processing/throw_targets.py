"""Where on the field the next throw is likely to go.

A small network (scripts/train_throw_targets.py) gives each square of two yards a score
from what is known while a player holds the disc: where the square lies from the thrower
and on the field, and how near the players of each team will be to it when the disc gets
there, moved on along their way. The scores of all squares are made into chances that
add up to one.

The network works on the field turned so that the thrower's team attacks the far end
zone. Which end a team attacks comes from the last line-up; where none was seen since
the last cut, from the way the team's throws went since then.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("ultimate_analysis.throw_targets")

WIDTH, LENGTH, END_ZONE = 40.0, 110.0, 20.0  # The USAU field, in yards
SQUARE = 2.0  # Yards
DISC_YARDS_PER_SECOND = 18.0  # Roughly: how long the disc takes to a square
LONGEST_LOOK_AHEAD = 1.0  # Seconds a player is moved on along their way, at most
NEAR = 5.0  # Yards: players this near a square are counted
MARK_REACH = 3.0  # Yards: the defender nearest the thrower, if this near, is the mark
SPEED_SECONDS = 0.5  # How far back a player's speed is measured over

PLACE_FEATURES = ["across", "downfield", "distance", "along_field", "from_sideline", "end_zone"]
OFFENCE_FEATURES = ["nearest_offence", "offence_near"]
PLAYER_FEATURES = [
    "nearest_offence",
    "nearest_defence",
    "second_defence",
    "margin",
    "offence_near",
    "defence_near",
    "towards_mark",
]

COLUMNS, ROWS = int(WIDTH / SQUARE), int(LENGTH / SQUARE)
# The middles of the squares, row by row along the field from the near back line
CENTRES = np.stack(
    np.meshgrid(
        (np.arange(COLUMNS) + 0.5) * SQUARE, (np.arange(ROWS) + 0.5) * SQUARE, indexing="xy"
    ),
    axis=-1,
).reshape(-1, 2)


@dataclass
class Moment:
    """What the network looks at, turned so that the thrower's team attacks the far end."""

    thrower: Tuple[float, float]
    offence: np.ndarray  # (n, 4): x, y, vx, vy; the thrower not among them
    defence: np.ndarray


def turned(x: float, y: float, attack: int) -> Tuple[float, float]:
    """A place on the field as seen by a team attacking in `attack` (+1: the far end)."""
    return (x, y) if attack > 0 else (WIDTH - x, LENGTH - y)


def square_of(place: Tuple[float, float]) -> int:
    """The index into CENTRES of the square a place lies in."""
    column = min(COLUMNS - 1, max(0, int(place[0] / SQUARE)))
    row = min(ROWS - 1, max(0, int(place[1] / SQUARE)))
    return row * COLUMNS + column


def distances(players: np.ndarray, flight: np.ndarray) -> np.ndarray:
    """(squares, players): how far each player will be from each square when the disc
    gets there, moved on along their way for as long as it flies (at most a second)."""
    if len(players) == 0:
        return np.full((len(CENTRES), 0), np.inf, dtype=np.float32)
    ahead = np.minimum(flight, LONGEST_LOOK_AHEAD)[:, None]
    x = players[None, :, 0] + players[None, :, 2] * ahead
    y = players[None, :, 1] + players[None, :, 3] * ahead
    return np.hypot(x - CENTRES[:, :1], y - CENTRES[:, 1:]).astype(np.float32)


def nth(near: np.ndarray, n: int, far: float = 60.0) -> np.ndarray:
    """The n-th smallest distance per square (0: the nearest), `far` where there is none."""
    if near.shape[1] <= n:
        return np.full(len(near), far, dtype=np.float32)
    return np.minimum(np.partition(near, n, axis=1)[:, n], far)


def features(moment: Moment, names: Sequence[str]) -> np.ndarray:
    """(squares, features) for one moment."""
    tx, ty = moment.thrower
    across, downfield = CENTRES[:, 0] - tx, CENTRES[:, 1] - ty
    distance = np.hypot(across, downfield)
    flight = 0.3 + distance / DISC_YARDS_PER_SECOND
    values = {
        "across": np.abs(across),
        "downfield": downfield,
        "distance": distance,
        "along_field": CENTRES[:, 1],
        "from_sideline": np.minimum(CENTRES[:, 0], WIDTH - CENTRES[:, 0]),
        "end_zone": (CENTRES[:, 1] >= LENGTH - END_ZONE).astype(np.float32),
    }
    if any(name in PLAYER_FEATURES for name in names):  # Those of the offence among them
        offence, defence = distances(moment.offence, flight), distances(moment.defence, flight)
        values["nearest_offence"] = nth(offence, 0)
        values["nearest_defence"] = nth(defence, 0)
        values["second_defence"] = nth(defence, 1)
        values["margin"] = np.clip(values["nearest_defence"] - values["nearest_offence"], -20, 20)
        values["offence_near"] = (offence < NEAR).sum(axis=1).astype(np.float32)
        values["defence_near"] = (defence < NEAR).sum(axis=1).astype(np.float32)
        # The mark stands between the thrower and one side: squares that way are harder
        towards = np.zeros(len(CENTRES), dtype=np.float32)
        if len(moment.defence):
            offsets = moment.defence[:, :2] - np.array(moment.thrower)
            mark = offsets[np.argmin(np.hypot(*offsets.T))]
            reach = float(np.hypot(*mark))
            if reach < MARK_REACH:
                unit = mark / max(reach, 1e-6)
                towards = (across * unit[0] + downfield * unit[1]) / np.maximum(distance, 1e-6)
        values["towards_mark"] = towards
    return np.stack([values[name] for name in names], axis=1).astype(np.float32)


def square_scorer(inputs: int, hidden: int = 64) -> Any:
    """The network: a score per square from its features, the same for every square."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


class ThrowTargets:
    """A trained network, ready to say where a throw goes."""

    def __init__(self, path: Path) -> None:
        import torch

        kept = torch.load(path, map_location="cpu", weights_only=True)
        self.names = list(kept["names"])
        self.mean = np.asarray(kept["mean"], dtype=np.float32)
        self.std = np.asarray(kept["std"], dtype=np.float32)
        self.network = square_scorer(len(self.names))
        self.network.load_state_dict(kept["state"])
        self.network.eval()

    def chances(self, moment: Moment) -> np.ndarray:
        """(ROWS, COLUMNS): the chance of each square, as turned in the moment."""
        import torch

        x = (features(moment, self.names) - self.mean) / self.std
        with torch.no_grad():
            scores = self.network(torch.from_numpy(x)).squeeze(-1)
            return torch.softmax(scores, dim=0).numpy().reshape(ROWS, COLUMNS)


def load_throw_targets(weights: Any) -> Optional[ThrowTargets]:
    """The network of a file, or None if there is no such file or it does not load
    (said in the log)."""
    path = Path(str(weights or ""))
    if not str(weights or "") or not path.is_file():
        logger.warning(f"No throw target network at '{weights}'")
        return None
    try:
        return ThrowTargets(path)
    except Exception as error:
        logger.exception(f"The throw target network did not load: {error}")
        return None


class ThrowReader:
    """Follows what the network needs from frame to frame, and asks it when wanted."""

    def __init__(self) -> None:
        self._trails: Dict[int, Deque[Tuple[float, float, float]]] = defaultdict(deque)
        # Team -> yards its completed throws went along the field since the last reset
        self._went: Dict[int, float] = defaultdict(float)
        self._last: Optional[Tuple[int, int, float]] = None  # Holder, team, place along
        # The moment of this frame, the attack it was turned by, and the tracks of its
        # defenders in order
        self._moment: Optional[Tuple[Moment, int, List[int]]] = None

    def reset(self) -> None:
        """Start again: after a cut or a seek, nothing seen before holds."""
        self.__init__()

    def update(
        self,
        seconds: float,
        players: Sequence[Tuple[int, Optional[int], float, float]],
        holder: Optional[int],
        team: Optional[int],
        attacks: Dict[int, int],
    ) -> None:
        """Take in a frame.

        Args:
            seconds: Its time in the video
            players: (track, team or None, x, y on the field) of each player placed
            holder: The track holding the disc, if anyone does
            team: The team in possession, if known
            attacks: Team -> +1 if it attacks the far end zone, -1 the near one
        """
        speeds = {}
        for track, _, x, y in players:
            trail = self._trails[track]
            trail.append((seconds, x, y))
            while seconds - trail[0][0] > SPEED_SECONDS:
                trail.popleft()
            then, earlier_x, earlier_y = trail[0]
            speeds[track] = (
                ((x - earlier_x) / (seconds - then), (y - earlier_y) / (seconds - then))
                if seconds > then
                else (0.0, 0.0)
            )
        for gone in [t for t, trail in self._trails.items() if seconds - trail[-1][0] > 2.0]:
            del self._trails[gone]

        self._moment = None
        placed = {track: (side, x, y) for track, side, x, y in players}
        if holder is None or team is None or holder not in placed:
            return
        along = placed[holder][2]
        if self._last is not None and self._last[0] != holder and self._last[1] == team:
            self._went[team] += along - self._last[2]  # A completed throw
        self._last = (holder, team, along)
        attack = attacks.get(team) or (int(np.sign(self._went[team])) or None)
        if attack is None:
            return
        offence, defence, defenders = [], [], []
        for track, (side, x, y) in placed.items():
            if track == holder or side is None:
                continue
            vx, vy = speeds.get(track, (0.0, 0.0))
            row = (*turned(x, y, attack), attack * vx, attack * vy)
            if side == team:
                offence.append(row)
            else:
                defence.append(row)
                defenders.append(track)
        moment = Moment(
            turned(placed[holder][1], along, attack),
            np.array(offence, dtype=np.float32).reshape(-1, 4),
            np.array(defence, dtype=np.float32).reshape(-1, 4),
        )
        self._moment = (moment, attack, defenders)

    def moment(self) -> Optional[Tuple[Moment, int, List[int]]]:
        """What is known while someone holds the disc, turned so that their team attacks
        the far end; the attack it was turned by; the tracks of the defence in order.
        None while nobody holds the disc or which end is attacked is not known."""
        return self._moment

    def chances(self, network: ThrowTargets) -> Optional[np.ndarray]:
        """(ROWS, COLUMNS) over the field as it lies (row 0 at the near back line): the
        chance that the next throw goes to each square; None while nobody holds the disc
        or which end their team attacks is not known."""
        if self._moment is None:
            return None
        moment, attack, _ = self._moment
        grid = network.chances(moment)
        return grid if attack > 0 else grid[::-1, ::-1]
