"""Teams that learn where to go and where to throw, from nothing (scripts/train_spots.py).

Players do not steer: each chooses a spot about once a second (and at once when the
disc changes hands, is thrown or comes down), and runs there as the game lets them
(game.py: top speed, backpedalling, turning). The spots: a grid around the disc; spots
a step from any player, which follow that player; and the disc itself. The holder
chooses every few tenths of a second between holding on and throwing to a spot of a
grid over the field.

Every choice is scored by a small network from what is around the spot, as the player
knows it: where it is; the players nearest it, of both teams (where, how they run, how
they speed up or turn, which way they face); the disc; the wind; the stall; the
player's role. Nothing else: no notion of open space, of a stack or of a matchup. At
first every choice is as likely as any other (the holder holds on most of the time,
or a team would throw at once every time). How the throw is shaped (curve, height,
angle) is as scripted.py has it, and players go for a disc in the air or on the ground
by reflex (learned.reflexes), as before.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import spot_features as sf
from .game import PLAYERS, Decision, GameState
from .learned import reflexes
from .perception import MESSAGE_SIZE, PlayerView
from .ppo import CRITIC_INPUTS, _mlp, critic_inputs
from .scripted import STYLE, throw_shape, towards
from .spot_features import (
    DISC,
    FIELD,
    OPPONENT,
    SPOT_FEATURES,
    TEAMMATE,
    THROW_CHOICES,
    THROW_FEATURES,
    THROW_TARGETS,
)

DECIDE_EVERY = 1.0  # Seconds between a player's choices of spot
THROW_EVERY = 0.3  # Seconds between the holder's choices (hold on, or throw where)
SHORTEST_HOLD = 0.3  # Seconds before a holder may throw
HOLD_AT_FIRST = 0.9  # The chance of holding on, at first, each time the holder chooses


def _disc_place(view: PlayerView) -> np.ndarray:
    if view.holder is not None:
        return (view.ours if view.attacking else view.theirs)[view.holder, :2]
    if view.disc is not None:
        return np.asarray(view.disc, dtype=float)
    return view.ours[view.me, :2]


# The holder holds on most of the time at first: holding on gets this much more weight
# than all the targets together
HOLD_PRIOR = float(np.log(HOLD_AT_FIRST / (1.0 - HOLD_AT_FIRST)))
THROW_PRIOR = np.r_[HOLD_PRIOR, np.full(THROW_CHOICES - 1, -np.log(THROW_CHOICES - 1))].astype(
    np.float32
)


# ---------------------------------------------------------------------- the networks


class SpotBrains:
    """A team's networks: where offence and defence players go, where the holder throws,
    and the values (on the game as it is, ppo.critic_inputs) of each side."""

    def __init__(self) -> None:
        import torch

        self.spots = {side: _mlp(SPOT_FEATURES, 1, 64) for side in ("offence", "defence")}
        self.throws = _mlp(THROW_FEATURES, 1, 64)
        self.values = {side: _mlp(CRITIC_INPUTS, 1) for side in ("offence", "defence")}
        with torch.no_grad():  # At first every choice as likely as any other
            for network in [*self.spots.values(), self.throws]:
                network[-1].weight.zero_()
                network[-1].bias.zero_()

    def parameters(self, side: str) -> List[Any]:
        found = list(self.spots[side].parameters())
        return found + list(self.throws.parameters()) if side == "offence" else found

    def networks(self) -> List[Any]:
        return [*self.spots.values(), self.throws, *self.values.values()]

    def to(self, device: str) -> "SpotBrains":
        for network in self.networks():
            network.to(device)
        return self

    def state(self) -> Dict[str, Any]:
        """The weights, on the processor (to send to the players' processes, or keep)."""

        def kept(network):
            return {k: v.detach().cpu() for k, v in network.state_dict().items()}

        return {
            "spots": {side: kept(n) for side, n in self.spots.items()},
            "throws": kept(self.throws),
            "values": {side: kept(n) for side, n in self.values.items()},
        }

    def load(self, state: Dict[str, Any]) -> None:
        for side in ("offence", "defence"):
            self.spots[side].load_state_dict(state["spots"][side])
            self.values[side].load_state_dict(state["values"][side])
        self.throws.load_state_dict(state["throws"])


def save_spot_brains(brains: SpotBrains, folder: Path, **more: Any) -> None:
    import torch

    Path(folder).mkdir(parents=True, exist_ok=True)
    torch.save({**brains.state(), **more}, Path(folder) / "spots.pt")


def load_spot_brains(folder: Path) -> SpotBrains:
    import torch

    brains = SpotBrains()
    brains.load(torch.load(Path(folder) / "spots.pt", map_location="cpu", weights_only=True))
    return brains


# ---------------------------------------------------------------------- playing


@dataclass
class Choice:
    """A choice one player made, for learning."""

    team: int
    player: int
    step: int
    side: str  # "offence" or "defence": whose network chose (a throw: "throw")
    features: np.ndarray  # (choices, features), float16
    prior: Optional[np.ndarray]  # Added to the scores (throws), or None
    chosen: int
    log_chance: float
    value: float
    value_inputs: np.ndarray


@dataclass
class _Plan:
    choice: Optional[tuple] = None
    decided_at: float = -np.inf
    seen: tuple = field(default_factory=tuple)  # What the disc was doing when decided


class SpotTeam:
    """A team played by SpotBrains: drawing its choices (or the likeliest, to watch, but
    when to throw is always drawn), keeping them when asked to."""

    def __init__(
        self,
        brains: SpotBrains,
        team: int,
        rng: np.random.Generator,
        record: bool = True,
        greedy: bool = False,
    ) -> None:
        self.brains = brains
        self.team = team
        self.rng = rng
        self.record = record
        self.greedy = greedy
        self.choices: List[Choice] = []
        self.plans = [_Plan() for _ in range(PLAYERS)]
        self.thrown_at = -np.inf
        self.step = 0
        self.game: Optional[GameState] = None

    def watch_game(self, state: GameState) -> None:
        """The game as it is, before the team decides (match.Match): for the values."""
        self.game = state

    def _values(self, side: str) -> tuple:
        import torch

        seen = (
            critic_inputs(self.game, self.team)
            if self.game is not None
            else np.zeros((PLAYERS, CRITIC_INPUTS), dtype=np.float32)
        )
        with torch.no_grad():
            value = self.brains.values[side](torch.from_numpy(seen))[:, 0].numpy()
        return seen, value

    def _draw(self, network, features: np.ndarray, prior: Optional[np.ndarray], always_draw=False):
        import torch

        with torch.no_grad():
            scores = network(torch.from_numpy(features))[:, 0].numpy()
        if prior is not None:
            scores = scores + prior
        scores = scores - scores.max()
        chances = np.exp(scores) / np.exp(scores).sum()
        if self.greedy and not always_draw:
            chosen = int(np.argmax(chances))
        else:
            chosen = int(self.rng.choice(len(chances), p=chances))
        return chosen, float(np.log(max(chances[chosen], 1e-12)))

    def _keep(self, view, side, features, prior, chosen, log_chance, seen, value) -> None:
        if self.record:
            self.choices.append(
                Choice(
                    self.team,
                    view.me,
                    self.step,
                    side,
                    features.astype(np.float16),
                    prior,
                    chosen,
                    log_chance,
                    float(value[view.me]),
                    seen[view.me],
                )
            )

    def decide(self, views: List[PlayerView]) -> Decision:
        import torch

        attacking = views[0].attacking
        side = "offence" if attacking else "defence"
        velocities = np.zeros((PLAYERS, 2))
        throw_to = None
        shape = (0.0, None, None)
        seen = value = None
        disc_now = (views[0].holder, views[0].lands_in is not None, views[0].disc is None)
        reflex = reflexes(views) if views[0].disc is not None and views[0].holder is None else {}
        deciding = []
        for k, view in enumerate(views):
            if attacking and view.holder == k:
                if view.held >= SHORTEST_HOLD and view.seconds - self.thrown_at >= THROW_EVERY:
                    self.thrown_at = view.seconds
                    if seen is None:
                        seen, value = self._values("offence")
                    with torch.no_grad():
                        features = sf.throw_choices(sf.views_of([view]))[0].numpy()
                    chosen, log_chance = self._draw(
                        self.brains.throws, features, THROW_PRIOR, always_draw=True
                    )
                    self._keep(
                        view, "throw", features, THROW_PRIOR, chosen, log_chance, seen, value
                    )
                    if chosen > 0:
                        target = THROW_TARGETS[chosen - 1].astype(float)
                        shape = throw_shape(view, target, STYLE)
                        throw_to = target
                continue
            if k in reflex:
                continue
            plan = self.plans[k]
            due = view.seconds - plan.decided_at >= DECIDE_EVERY - 1e-9
            if due or plan.seen != disc_now or plan.choice is None:
                deciding.append(k)
        if deciding:
            if seen is None:
                seen, value = self._values(side)
            with torch.no_grad():
                features, kinds, players, params = sf.spot_choices(
                    sf.views_of(
                        [views[k] for k in deciding], [self.plans[k].choice for k in deciding]
                    )
                )
            features = features.numpy()
            for row, k in enumerate(deciding):
                chosen, log_chance = self._draw(self.brains.spots[side], features[row], None)
                self._keep(views[k], side, features[row], None, chosen, log_chance, seen, value)
                kind = int(kinds[row, chosen])
                plan = self.plans[k]
                plan.choice = (
                    kind,
                    int(players[row, chosen]),
                    None if kind == DISC else params[row, chosen].numpy().astype(float),
                )
                # Staggered, so that a team does not all choose at once
                plan.decided_at = views[k].seconds + 0.1 * self.rng.integers(-2, 3)
                plan.seen = disc_now
        for k, view in enumerate(views):
            if attacking and view.holder == k:
                continue
            if k in reflex:
                velocities[k] = reflex[k]
            else:
                velocities[k] = towards(view.ours[k, :2], _spot_of(view, self.plans[k].choice))
        self.step += 1
        return Decision(velocities, throw_to, np.zeros((PLAYERS, MESSAGE_SIZE)), *shape)


def _spot_of(view: PlayerView, choice: tuple) -> np.ndarray:
    """Where a choice is now: a place, a step from a player as they are now, the disc."""
    kind, player, where = choice
    if kind == FIELD:
        return np.asarray(where, dtype=float)
    if kind == TEAMMATE:
        return view.ours[player, :2] + where
    if kind == OPPONENT:
        return view.theirs[player, :2] + where
    return _disc_place(view) if view.disc is None else np.asarray(view.disc, dtype=float)
