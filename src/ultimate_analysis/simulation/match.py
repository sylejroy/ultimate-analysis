"""Playing points between two teams, and what came of them."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .game import GameState, line_up, step
from .perception import Senses

TURNOVERS = ("block", "mark block", "drop", "out", "stall")


@dataclass
class PointSummary:
    """What happened in a point."""

    scored: Optional[int]  # The team that scored, None if nobody did in time
    seconds: float
    throws: int
    completions: int
    turnovers: int


class Match:
    """A point being played, step by step: two teams (anything with decide(views) ->
    Decision, views being what each of its players knows) and what they know."""

    def __init__(
        self,
        teams: Sequence[Any],
        rng: np.random.Generator,
        start: Optional[GameState] = None,
    ) -> None:
        self.teams = teams
        self.rng = rng
        self.state = start if start is not None else line_up(rng)
        self.senses = (Senses(0, rng), Senses(1, rng))
        self.decisions: tuple = ()
        self.views: list = [[], []]  # What each team's players knew at the last step

    def step(self) -> GameState:
        views = [senses.look(self.state) for senses in self.senses]
        for team in self.teams:  # A team that learns sees the game as it is, for its values
            if hasattr(team, "watch_game"):
                team.watch_game(self.state)
        self.views = views
        self.decisions = tuple(team.decide(v) for team, v in zip(self.teams, views))
        for senses, decision in zip(self.senses, self.decisions):
            if decision.messages is not None:
                senses.hear(decision.messages)
        return step(self.state, self.decisions, self.rng)


def play_point(
    teams: Sequence[Any],
    rng: np.random.Generator,
    start: Optional[GameState] = None,
    watch: Optional[Callable[[GameState], None]] = None,
    limit: Optional[float] = None,
) -> GameState:
    """Play a point between two teams: team 0 starts with the disc as after a pull,
    unless another start is given; stopped unscored at `limit` seconds if given."""
    match = Match(teams, rng, start)
    while not match.state.over and (limit is None or match.state.seconds < limit):
        match.step()
        if watch is not None:
            watch(match.state)
    return match.state


def summary(state: GameState) -> PointSummary:
    kinds = [event.kind for event in state.events]
    return PointSummary(
        state.scored,
        state.seconds,
        kinds.count("throw"),
        kinds.count("catch"),
        sum(kinds.count(k) for k in TURNOVERS),
    )
