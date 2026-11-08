"""Dangerous space: where on the field the offence would get the disc, and what that is
worth, looking one throw ahead.

Nothing here is learned from where throwers chose to throw. For each square of the field
(processing/throw_targets.py lays them out):

- Who gets there first. Each player keeps running their way for a moment, then runs
  straight to the square at full speed; the disc takes its time to get there. The
  offence is likely to have the disc there if one of them is there in time and before
  any defender. The thrower's mark makes the squares on their side harder.
- What it is worth: how likely a possession is to score from a catch there (a curve
  along the field, fitted to recorded games by scripts/fit_field_value.py; a rough one
  without it), and how much better the next throw from there would be than the best the
  thrower has now. Looking that throw ahead, everyone has moved on and the new thrower
  is unmarked until a defender gets to them: a dump that opens the other side of the
  field is worth what it opens, though it goes backwards.
- Dangerous is open and worth more than where the disc is now.

Each defender is answerable for the squares they would get to first of their team: the
danger of those is what they give up. And space that is opening is that which became
more dangerous over the last half second.
"""

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import numpy as np

from .throw_targets import CENTRES, COLUMNS, END_ZONE, LENGTH, ROWS, Moment

# How players and the disc move, as measured on the recorded games; how sharply who
# gets there first decides it was tuned on their throws, which cannot judge the timing
# against the disc or the mark (MEASUREMENTS.md, "Dangerous space")
RUN_SPEED = 8.0  # Yards a second at full speed: between the 95th and 99th percentile
REACTION = 0.2  # Seconds a player keeps going their way before turning to a square
# The disc's time from the last frame held to the first caught, fitted to completions
DISC_YARDS_PER_SECOND = 17.7
RELEASE_SECONDS = 0.24
CATCH_SLACK = 0.4  # Seconds a receiver may arrive after the disc: it floats, they dive
SPREAD = 0.1  # Seconds: how sharply who gets there first decides it
MARK_REACH = 3.0  # Yards: a defender this near the thrower marks them
MARK_BLOCKS = 0.5  # Of a throw straight past the mark's side, taken away (chosen)
# How much a better next throw counts, against a better place on the field: the curve
# of the field's worth already holds what an average next throw is worth
OPTIONS_WEIGHT = 0.5
# The next throw is judged on every other square each way: enough for a second throw
NEXT_SQUARES = (
    (np.arange(ROWS)[:, None] % 2 == 0) & (np.arange(COLUMNS)[None, :] % 2 == 0)
).ravel()
SQUARES = CENTRES.astype(np.float32)
NOBODY = 60.0  # Seconds: when a team with nobody there gets there
OPENING_SECONDS = 0.5  # Space is opening if it became more dangerous over this long
# Chance of scoring from a catch along the field, as the logistic of a + b * y, the
# field turned so that the far end zone is attacked; a catch in it is a score
ROUGH_VALUE = (-0.63, 0.033)


class FieldValue:
    """How likely a possession is to score from a catch at each place along the field."""

    def __init__(self, a: float = ROUGH_VALUE[0], b: float = ROUGH_VALUE[1]) -> None:
        self.a, self.b = float(a), float(b)

    def __call__(self, along: np.ndarray) -> np.ndarray:
        along = np.asarray(along, dtype=np.float32)
        chance = 1.0 / (1.0 + np.exp(-(self.a + self.b * along)))
        return np.where(along >= LENGTH - END_ZONE, 1.0, chance).astype(np.float32)


def load_field_value(path: Any) -> FieldValue:
    """The curve fitted to recorded games (a JSON file with a and b), or the rough one."""
    if path and Path(str(path)).is_file():
        try:
            kept = json.loads(Path(str(path)).read_text())
            return FieldValue(kept["a"], kept["b"])
        except (OSError, ValueError, KeyError):
            pass
    return FieldValue()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def arrival(players: np.ndarray, squares: np.ndarray) -> np.ndarray:
    """(..., squares, players): seconds each player needs to get to each square, for
    players (..., n, 4) as x, y, vx, vy."""
    if players.shape[-2] == 0:
        return np.full((*players.shape[:-2], len(squares), 0), NOBODY, dtype=np.float32)
    going = players[..., :2] + players[..., 2:4] * REACTION
    run = np.hypot(
        going[..., None, :, 0] - squares[:, None, 0], going[..., None, :, 1] - squares[:, None, 1]
    )
    return (REACTION + run / RUN_SPEED).astype(np.float32)


def open_to(
    thrower: np.ndarray,
    offence: np.ndarray,
    defence: np.ndarray,
    squares: np.ndarray,
) -> np.ndarray:
    """(..., squares): how likely the offence is to have the disc at each square if it is
    thrown there now, from throwers (..., 2), with the offence (..., n, 4) without the
    thrower and the defence (..., m, 4). A defender within reach of the thrower marks
    them."""
    thrower = np.asarray(thrower, dtype=np.float32)
    across = squares[:, 0] - thrower[..., None, 0]
    along = squares[:, 1] - thrower[..., None, 1]
    distance = np.hypot(across, along)
    flight = RELEASE_SECONDS + distance / DISC_YARDS_PER_SECOND
    # Nobody of a team: as if they would get there in a minute
    first_offence = arrival(offence, squares).min(axis=-1, initial=NOBODY)
    first_defence = arrival(defence, squares).min(axis=-1, initial=NOBODY)
    chance = _sigmoid((flight + CATCH_SLACK - first_offence) / SPREAD) * _sigmoid(
        (first_defence - first_offence) / SPREAD
    )
    if defence.shape[-2]:
        offsets = defence[..., :2] - thrower[..., None, :]
        reach = np.hypot(offsets[..., 0], offsets[..., 1])
        nearest = np.argmin(reach, axis=-1)[..., None]
        mark = np.take_along_axis(offsets, nearest[..., None], axis=-2)[..., 0, :]
        near = np.take_along_axis(reach, nearest, axis=-1)
        side = mark / np.maximum(near, 1e-6)
        towards = (across * side[..., :1] + along * side[..., 1:]) / np.maximum(distance, 1e-6)
        blocked = MARK_BLOCKS * np.clip(towards, 0.0, 1.0) * (near < MARK_REACH)
        chance = chance * (1.0 - blocked)
    return chance.astype(np.float32)


@dataclass
class Danger:
    """The dangerous space of a moment, over all squares (as turned in the moment)."""

    open: np.ndarray  # How likely the offence is to have the disc there
    worth: np.ndarray  # How likely a possession is to score from a catch there
    options: np.ndarray  # The best next throw from there: open, times what it is worth
    danger: np.ndarray  # Open, times what it is worth beyond where the disc is now
    given_up: np.ndarray  # Per defender of the moment: the danger nearest to them


def dangerous_space(moment: Moment, value: FieldValue) -> Danger:
    """The dangerous space of a moment, looking one throw ahead."""
    squares = SQUARES
    chance = open_to(np.array(moment.thrower), moment.offence, moment.defence, squares)
    worth = value(squares[:, 1])
    next_squares, next_worth = squares[NEXT_SQUARES], worth[NEXT_SQUARES]

    # One throw on, from where each receiver is likeliest to catch it, for all the squares
    # they would get to first: everyone has run on, the receiver stands there with the
    # disc and the old thrower stays where they threw from. A defender who is not at the
    # new thrower yet does not mark them.
    best_now = float(np.max(chance[NEXT_SQUARES] * next_worth))
    options = np.full(len(squares), best_now, dtype=np.float32)
    if len(moment.offence):
        owners = np.argmin(arrival(moment.offence, squares), axis=1)
        receivers, catches = [], []
        for player in range(len(moment.offence)):
            theirs = np.flatnonzero(owners == player)
            if len(theirs) and chance[theirs].max() >= 0.2:
                receivers.append(player)
                catches.append(theirs[np.argmax(chance[theirs])])
        if receivers:
            there = squares[catches]
            flight = RELEASE_SECONDS + np.hypot(*(there - moment.thrower).T) / DISC_YARDS_PER_SECOND
            moved = np.minimum(flight, 1.0)[:, None, None]
            offence = np.repeat(moment.offence[None], len(there), axis=0)
            offence[..., :2] += offence[..., 2:4] * moved
            offence[np.arange(len(there)), receivers] = [*moment.thrower, 0.0, 0.0]
            defence = np.repeat(moment.defence[None], len(there), axis=0)
            defence[..., :2] += defence[..., 2:4] * moved
            then = open_to(there, offence, defence, next_squares)
            for player, best in zip(receivers, np.max(then * next_worth, axis=-1)):
                options[owners == player] = best
    now = float(value(np.array([moment.thrower[1]]))[0])
    gain = worth - now + OPTIONS_WEIGHT * (options - best_now)
    danger = chance * np.clip(gain, 0.0, None)

    given_up = np.zeros(len(moment.defence), dtype=np.float32)
    if len(moment.defence):
        nearest = np.argmin(arrival(moment.defence, squares), axis=1)
        np.add.at(given_up, nearest, danger)
    return Danger(chance, worth, options, danger.astype(np.float32), given_up)


@dataclass
class SpaceMap:
    """What is drawn of the dangerous space, over the field as it lies (rows along it
    from the near back line, as the throw targets)."""

    danger: np.ndarray  # (ROWS, COLUMNS)
    opening: Optional[np.ndarray]  # (ROWS, COLUMNS): more dangerous than a moment ago
    given_up: Dict[int, float] = field(default_factory=dict)  # Defender's track -> share


class SpaceWatcher:
    """The dangerous space from frame to frame: what is opening needs the moment before."""

    def __init__(self) -> None:
        self._before: Deque[Tuple[float, np.ndarray]] = deque()

    def reset(self) -> None:
        self._before.clear()

    def update(
        self,
        seconds: float,
        moment: Moment,
        attack: int,
        defenders: Sequence[int],
        value: FieldValue,
    ) -> SpaceMap:
        """The map of a moment (turned by `attack`), its defenders' tracks in order."""
        found = dangerous_space(moment, value)
        grid = found.danger.reshape(ROWS, COLUMNS)
        if attack < 0:
            grid = grid[::-1, ::-1]
        opening = None
        while self._before and seconds - self._before[0][0] > OPENING_SECONDS:
            then, earlier = self._before.popleft()
            if seconds - then <= 2 * OPENING_SECONDS:
                opening = np.clip(grid - earlier, 0.0, None)
        self._before.append((seconds, grid))
        total = float(found.given_up.sum())
        shares = {
            int(track): float(part) / total
            for track, part in zip(defenders, found.given_up)
            if total > 0
        }
        return SpaceMap(grid, opening, shares)
