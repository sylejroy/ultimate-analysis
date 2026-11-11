"""Teams that learn by reinforcement (PPO): a policy for how each player runs and what
they say, and for the holder whether to throw now and where to, each with a value of
the situation; rewards from what the team does (scripts/train_ppo.py trains them).

Every player of a side uses the same networks on what they know (learned.py's 111
inputs). How they run starts as copied from the recorded players (learned.py), with a
spread around it to try other ways. The holder decides each step whether to throw now
(a chance from how long they have held it, the stall and how dangerous the best space
is), and on throwing where to: every square of the field and each teammate led to
where they will be when the disc gets there (as the scripted thrower leads them), each
scored from what the dangerous space says of it (processing/open_space.py: how open,
what it is worth, what it opens next), where it lies, and for a teammate how open they
are there and how clear of a defender. It starts at throwing into open, dangerous
space. How the throw is shaped (curve, height, angle, the wind) is as learned.py has it.

Rewards for the team with the disc, the other team getting the opposite: a completed
pass, and the yards it gained (from where it was thrown to where it was caught, to
the goal line at most); losing the disc; scoring. The defence also gains a little for every
second its mark counts the stall (not taken from the offence: it would throw sooner).
Besides what the team gets, each player gets credit for what they did themselves
(player_rewards): a throw caught, a catch, a score thrown or caught, a block; and loses
for a disc they lost (a throw not caught, a drop, a catch out of bounds, a stall).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..processing.open_space import dangerous_space
from ..processing.throw_targets import CENTRES, LENGTH, WIDTH, Moment
from .disc import DISC_YARDS_PER_SECOND, RELEASE_SECONDS, SLOWEST, WIND_ALONG
from .game import END_ZONE, PLAYERS, RUN_SPEED, Decision, GameState, attack_of, turn
from .learned import (
    KNOWN_UNSURE,
    Minds,
    load_minds,
    reflexes,
    save_minds,
    view_features,
    with_inputs,
)
from .perception import PlayerView
from .scripted import CALL, aim_off, throw_options, throw_shape, to_the_open_side

# Rewards, to the team with the disc (the other team gets the opposite): losing the
# disc costs as much as a score is worth
COMPLETION, PER_YARD, TURNOVER, SCORE = 0.5, 0.1, 5.0, 5.0
GOAL_LINE = LENGTH - END_ZONE  # The attacked one, as the team attacks
TURNOVERS = ("block", "mark block", "drop", "out", "stall")
STALL_REWARD = 0.1  # To the defence per second the stall is counted
# To the player alone: a throw caught, a catch, a score thrown or caught, a block, a
# disc they lost; a pass caught by the attacker a defender is matched with (game.match_up)
THROWN, CAUGHT, SCORED, BLOCKED, LOST = 0.3, 0.3, 1.0, 1.0, -2.0
THEIRS_CAUGHT = -2.0  # As much as an attacker losing the disc
SQUARE_FEATURES = 18
# What the values see: the game as it is (they are used only in learning), as the team
# attacks: every player's place and velocity (the team's own first, by number), the
# disc (held: where; in the air: where it comes down and how soon; on the ground:
# where), the stall, how long it is held, the wind, whether the team has the disc,
# and which player it is for
CRITIC_INPUTS = 4 * 2 * PLAYERS + 10 + 4 + 1 + PLAYERS
FAR = 10.0  # Yards: a defender further than this from a throw counts as this far
THROW_FEATURES = 12
FIRST_SPREAD = (0.1, 0.1, 0.3, 0.3, 0.3, 0.3)  # Of the outputs, at first: running, words
THROW_AT_FIRST = -3.5  # The logit of throwing in a step, at first (about every 3 s)


def _mlp(inputs: int, outputs: int, hidden: int = 128) -> Any:
    from torch import nn

    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, outputs),
    )


class Policy:
    """A side's networks (offence or defence): how players run and talk, with their
    spread, and the value of what a player sees; for the offence also the holder's."""

    def __init__(self, movement: Any, offence: bool) -> None:
        import torch

        self.movement = movement
        self.log_spread = torch.nn.Parameter(
            torch.log(torch.tensor(FIRST_SPREAD, dtype=torch.float32))
        )
        self.value = _mlp(CRITIC_INPUTS, 1)
        self.offence = offence
        self.squares = self.throw_now = None
        if offence:
            self.squares = _mlp(SQUARE_FEATURES, 1, 64)
            self.throw_now = _mlp(THROW_FEATURES, 1, 64)
            with torch.no_grad():  # At first: the prior alone, and throwing now and then
                self.squares[-1].weight.zero_()
                self.squares[-1].bias.zero_()
                self.throw_now[-1].weight.zero_()
                self.throw_now[-1].bias.fill_(THROW_AT_FIRST)

    def parameters(self) -> List[Any]:
        found = list(self.movement.parameters()) + [self.log_spread] + list(self.value.parameters())
        if self.offence:
            found += list(self.squares.parameters()) + list(self.throw_now.parameters())
        return found

    def state(self) -> Dict[str, Any]:
        return {
            "movement": self.movement.state_dict(),
            "log_spread": self.log_spread.detach().clone(),
            "value": self.value.state_dict(),
            **(
                {"squares": self.squares.state_dict(), "throw_now": self.throw_now.state_dict()}
                if self.offence
                else {}
            ),
        }

    def load(self, state: Dict[str, Any]) -> None:
        import torch

        self.movement.load_state_dict(with_inputs(state["movement"]))
        with torch.no_grad():
            self.log_spread.copy_(state["log_spread"])
        try:  # Values of an earlier kind (seeing a player's view) start afresh
            self.value.load_state_dict(state["value"])
        except RuntimeError:
            pass
        if self.offence:
            # The holder's networks of an earlier shape (fewer inputs) start afresh
            for name in ("squares", "throw_now"):
                try:
                    getattr(self, name).load_state_dict(state[name])
                except RuntimeError:
                    pass


@dataclass
class Brains:
    """A team that learns by reinforcement: the policy of each side, and what of the
    copied minds it keeps (set plays are not called; how throws are shaped)."""

    offence: Policy
    defence: Policy
    minds: Minds

    @classmethod
    def from_minds(cls, minds: Minds) -> "Brains":
        import copy

        return cls(
            Policy(copy.deepcopy(minds.offence), True),
            Policy(copy.deepcopy(minds.defence), False),
            minds,
        )

    def state(self) -> Dict[str, Any]:
        return {"offence": self.offence.state(), "defence": self.defence.state()}

    def load(self, state: Dict[str, Any]) -> None:
        self.offence.load(state["offence"])
        self.defence.load(state["defence"])


def save_brains(brains: Brains, folder: Path, **more: Any) -> None:
    """Kept as ppo.pt, with the movement networks as minds.pt so that a team of them can
    also be played as a LearnedTeam."""
    import torch

    Path(folder).mkdir(parents=True, exist_ok=True)
    torch.save({**brains.state(), **more}, Path(folder) / "ppo.pt")
    minds = brains.minds
    minds.offence, minds.defence = brains.offence.movement, brains.defence.movement
    save_minds(minds, folder)


def load_brains(folder: Path) -> Brains:
    import torch

    brains = Brains.from_minds(load_minds(folder))
    brains.load(torch.load(Path(folder) / "ppo.pt", map_location="cpu", weights_only=True))
    return brains


# ---------------------------------------------------------------------- the holder


def holder_inputs(view: PlayerView) -> tuple:
    """(choices, prior, throw, targets): what the holder's choices see. Per choice (each
    square of the field, then each teammate led): how open and dangerous it is and where
    it lies, and for a teammate how open they are where led and how clear of a defender;
    the prior over them; for throwing now, the stall, how long they have held it, how
    dangerous the best of the field is and how open the best teammate; and where each
    choice is on the field."""
    thrower = view.ours[view.holder, :2]
    mates = np.delete(view.ours, view.holder, axis=0)
    known = np.delete(view.ours_unsure, view.holder) < KNOWN_UNSURE
    moment = Moment(
        tuple(thrower),
        mates[known].astype(np.float32),
        view.theirs[view.theirs_unsure < KNOWN_UNSURE].astype(np.float32),
    )
    found = dangerous_space(moment, _value())
    led = throw_options(view)  # (6, 5): x, y, how open, gain, how clear
    targets = np.vstack([CENTRES, led[:, :2]])
    # A teammate's choice takes the dangerous space of the square they are led to
    square_of = np.argmin(np.hypot(*(CENTRES[None] - led[:, None, :2]).transpose(2, 0, 1)), axis=1)
    at = np.concatenate([np.arange(len(CENTRES)), square_of])
    teammate = np.r_[np.zeros(len(CENTRES)), np.ones(len(led))]
    led_open = np.r_[np.zeros(len(CENTRES)), led[:, 2]]
    clear = np.r_[np.zeros(len(CENTRES)), np.minimum(led[:, 4], 10.0)]
    across, along = (targets - thrower).T
    choices = np.stack(
        [
            found.open[at],
            found.worth[at],
            found.options[at],
            found.danger[at] * 10.0,
            across / 20.0,
            along / 20.0,
            np.hypot(across, along) / 30.0,
            targets[:, 1] / LENGTH,
            np.minimum(targets[:, 0], WIDTH - targets[:, 0]) / 20.0,
            np.full(len(targets), view.stall / 10.0),
            teammate,
            led_open,
            clear / 5.0,
            *_throw_risks(view, thrower, targets),
        ],
        axis=1,
    ).astype(np.float32)
    # Where the disc could go at all, and the prior: open and dangerous (for a teammate,
    # how open they are where led)
    opened = np.where(teammate > 0, led_open, found.open[at])
    prior = 4.0 * np.log(np.clip(opened, 1e-3, 1.0)) + 10.0 * found.danger[at]
    prior[np.hypot(across, along) > 70.0] = -1e4
    best = np.argsort(-found.danger)[: max(1, len(found.danger) // 20)]
    wind = np.zeros(2) if view.wind is None else np.asarray(view.wind) / 10.0
    forward = led[:, 3] > 0
    throw = np.array(
        [
            view.stall / 10.0,
            view.held / 5.0,
            found.danger.max() * 10.0,
            found.danger[best].mean() * 10.0,
            float(np.max(found.open * found.worth)),
            float(found.open[int(np.argmax(found.danger))]),
            *wind,
            float(np.max(led[:, 2])),
            float(np.max(led[:, 2] * forward)),
            float(np.max(np.minimum(led[:, 4], 10.0) * (led[:, 2] > 0.8))) / 5.0,
            float(np.max(led[:, 3] * led[:, 2])) * 10.0,
        ],
        dtype=np.float32,
    )
    return choices, prior.astype(np.float32), throw, targets.astype(np.float32)


def _throw_risks(view: PlayerView, thrower: np.ndarray, targets: np.ndarray) -> list:
    """Per choice, what makes a throw there risky besides how open it is: the wind along
    it and across it, how near a defender is to its path (where they could knock it down)
    and to where it comes down, and how long it is in the air."""
    wind = np.zeros(2) if view.wind is None else np.asarray(view.wind, dtype=float)
    way = targets - thrower
    length = np.maximum(np.hypot(*way.T), 1e-6)
    unit = way / length[:, None]
    along_wind = unit @ wind
    across_wind = unit[:, 0] * wind[1] - unit[:, 1] * wind[0]
    speed = np.maximum(DISC_YARDS_PER_SECOND + WIND_ALONG * along_wind, SLOWEST)
    airborne = RELEASE_SECONDS + length / speed
    known = view.theirs[view.theirs_unsure < KNOWN_UNSURE, :2]
    if len(known):
        # The nearest point of the middle of the path (the ends are the mark's and the
        # catch's) to each defender
        offsets = known[None] - thrower  # (1, m, 2)
        share = np.clip((offsets @ way.T).transpose(0, 2, 1)[0] / length[:, None] ** 2, 0.15, 0.85)
        nearest = thrower + share[..., None] * way[:, None]  # (n, m, 2)
        path = np.hypot(*(known[None] - nearest).transpose(2, 0, 1)).min(axis=1)
        spot = np.hypot(*(known[None] - targets[:, None]).transpose(2, 0, 1)).min(axis=1)
    else:
        path = spot = np.full(len(targets), FAR)
    return [
        along_wind / 10.0,
        across_wind / 10.0,
        np.minimum(path, FAR) / 5.0,
        np.minimum(spot, FAR) / 5.0,
        airborne / 3.0,
    ]


_VALUE = None


def _value():
    global _VALUE
    if _VALUE is None:
        from ..processing.open_space import FieldValue

        _VALUE = FieldValue()
    return _VALUE


# ---------------------------------------------------------------------- playing


@dataclass
class Sample:
    """What one player saw and did in a step, for learning."""

    team: int
    player: int
    step: int
    side: str  # "offence" or "defence": the policy that decided
    inputs: np.ndarray
    action: np.ndarray  # How they ran and talked, as drawn
    log_chance: float  # Of the action drawn, under the policy that drew it
    moved: bool  # Whether the action was carried out (not a reflex, not the holder)
    value: float
    value_inputs: Optional[np.ndarray] = None  # What the value saw (critic_inputs)
    throw_inputs: Optional[np.ndarray] = None  # The holder's: whether to throw now
    threw: Optional[bool] = None
    throw_log_chance: float = 0.0
    square_inputs: Optional[np.ndarray] = None  # On throwing: per square, and the prior
    prior: Optional[np.ndarray] = None
    square: int = -1


class PolicyTeam:
    """A team played by Brains: drawing actions to try things (or their most likely, to
    watch), and keeping what it saw and did when asked to."""

    def __init__(
        self,
        brains: Brains,
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
        self.samples: List[Sample] = []
        self.step = 0
        self.game: Optional[GameState] = None  # The game as it is, for the values

    def watch_game(self, state: GameState) -> None:
        """The game as it is, before the team decides (match.Match): for the values."""
        self.game = state

    def decide(self, views: List[PlayerView]) -> Decision:
        import torch

        attacking = views[0].attacking
        policy = self.brains.offence if attacking else self.brains.defence
        x = torch.from_numpy(np.stack([view_features(v) for v in views]))
        with torch.no_grad():
            mean = policy.movement(x)
            spread = torch.exp(policy.log_spread)
            action = mean if self.greedy else mean + spread * torch.randn_like(mean)
            log_chance = _log_chance(action, mean, spread).numpy()
            seen = (
                critic_inputs(self.game, self.team)
                if self.game is not None
                else np.zeros((PLAYERS, CRITIC_INPUTS), dtype=np.float32)
            )
            value = policy.value(torch.from_numpy(seen))[:, 0].numpy()
        action = action.numpy()
        velocities = (x[:, 2:4].numpy() + action[:, :2]) * RUN_SPEED
        messages = np.tanh(action[:, 2:])
        messages[:, CALL] = 0.0
        moved = np.ones(PLAYERS, dtype=bool)
        throw_to, shape, holder_sample = None, (0.0, None, None), None
        if attacking and views[0].holder is not None:
            holder = views[0].holder
            velocities[holder] = 0.0
            moved[holder] = False
            throw_to, holder_sample = self._throw(views[holder], policy)
            if throw_to is not None:
                throw_to = to_the_open_side(views[holder], throw_to)
                shape = throw_shape(views[holder], throw_to, self.brains.minds.style)
                throw_to = aim_off(views[holder], throw_to, shape, self.brains.minds.style)
        elif views[0].disc is not None:
            for k, velocity in reflexes(views).items():
                velocities[k] = velocity
                moved[k] = False
        if self.record:
            for k in range(PLAYERS):
                sample = Sample(
                    self.team,
                    k,
                    self.step,
                    "offence" if attacking else "defence",
                    x[k].numpy(),
                    action[k],
                    float(log_chance[k]),
                    bool(moved[k]),
                    float(value[k]),
                    seen[k],
                )
                if holder_sample is not None and k == views[0].holder:
                    sample.__dict__.update(holder_sample)
                self.samples.append(sample)
        self.step += 1
        return Decision(velocities, throw_to, messages, *shape)

    def _throw(self, view: PlayerView, policy: Policy) -> tuple:
        """Whether the holder throws now and where to; and what of it is kept."""
        import torch

        if view.held < 0.3:
            return None, None
        squares, prior, throw, targets = holder_inputs(view)
        with torch.no_grad():
            logit = float(policy.throw_now(torch.from_numpy(throw))[0])
        chance = 1.0 / (1.0 + np.exp(-logit))
        # When throwing is drawn by chance even to watch: a chance per step, never above
        # a half, would otherwise never throw
        threw = self.rng.random() < chance
        kept = {
            "throw_inputs": throw,
            "threw": bool(threw),
            "throw_log_chance": float(np.log(max(chance if threw else 1.0 - chance, 1e-9))),
        }
        if not threw:
            return None, kept
        with torch.no_grad():
            logits = policy.squares(torch.from_numpy(squares))[:, 0].numpy() + prior
        logits = logits - logits.max()
        chances = np.exp(logits) / np.exp(logits).sum()
        square = (
            int(np.argmax(chances))
            if self.greedy
            else int(self.rng.choice(len(chances), p=chances))
        )
        kept.update(
            square_inputs=squares.astype(np.float16),
            prior=prior,
            square=square,
            throw_log_chance=kept["throw_log_chance"] + float(np.log(max(chances[square], 1e-12))),
        )
        return targets[square].astype(float), kept


def critic_inputs(state: GameState, team: int) -> np.ndarray:
    """(7, CRITIC_INPUTS): what the values see, for each of a team's players."""
    attack = attack_of(team)
    places = turn(state.places, attack)
    going = state.velocities * attack
    order = np.r_[np.arange(PLAYERS), np.arange(PLAYERS, 2 * PLAYERS)]
    if team == 1:
        order = np.r_[np.arange(PLAYERS, 2 * PLAYERS), np.arange(PLAYERS)]
    players = np.column_stack([places[order] / [WIDTH, LENGTH], going[order] / RUN_SPEED]).ravel()
    disc = np.zeros(10)
    if state.holder is not None:
        disc[0:3] = [1.0, *(places[state.holder] / [WIDTH, LENGTH])]
    elif state.throw is not None:
        lands = turn(state.throw.target, attack)
        disc[3:7] = [1.0, *(lands / [WIDTH, LENGTH]), (state.throw.lands - state.seconds) / 3.0]
    elif state.disc is not None:
        disc[7:10] = [1.0, *(turn(state.disc, attack) / [WIDTH, LENGTH])]
    held = state.seconds - state.held_since if state.holder is not None else 0.0
    shared = np.r_[
        players,
        disc,
        state.stall / 10.0,
        held / 5.0,
        np.asarray(state.wind) * attack / 10.0,
        float(state.offence == team),
    ]
    return np.hstack([np.tile(shared, (PLAYERS, 1)), np.eye(PLAYERS)]).astype(np.float32)


def _log_chance(action, mean, spread):
    """Log density of actions under independent normals, summed over their numbers."""
    import torch

    return (
        -0.5 * (((action - mean) / spread) ** 2) - torch.log(spread) - 0.5 * np.log(2 * np.pi)
    ).sum(-1)


# ---------------------------------------------------------------------- rewards


def stall_watch() -> tuple:
    """(counted, watch): a watch for match.play_point that keeps, per step, the team
    whose mark counted the stall then (-1 for none)."""
    counted: List[int] = []
    last = [0.0]

    def watch(state: GameState) -> None:
        rose = state.holder is not None and state.stall > last[0] + 1e-9
        counted.append(1 - state.offence if rose else -1)
        last[0] = state.stall if state.holder is not None else 0.0

    return counted, watch


def player_rewards(state: GameState, steps: int, step_seconds: float) -> np.ndarray:
    """(14, steps): each player's own reward at each step, for what they did (the players
    numbered as in the game: 0 to 6 team 0, 7 to 13 team 1)."""
    found = np.zeros((2 * PLAYERS, steps))

    def give(player: Optional[int], seconds: float, amount: float) -> None:
        if player is not None:
            found[player, min(steps - 1, max(0, int(round(seconds / step_seconds)) - 1))] += amount

    thrower = None
    for event in state.events:
        if event.kind == "throw":
            thrower = event.player
        elif event.kind == "catch":
            give(event.player, event.seconds, CAUGHT)
            give(thrower, event.seconds, THROWN)
            give(event.by, event.seconds, THEIRS_CAUGHT)
        elif event.kind == "score":
            give(event.player, event.seconds, SCORED)
            give(thrower, event.seconds, SCORED)
        elif event.kind in TURNOVERS:
            give(event.player, event.seconds, LOST)
            give(event.by, event.seconds, BLOCKED)
    return found


def rewards_of(
    state: GameState,
    steps: int,
    step_seconds: float,
    counted: Optional[List[int]] = None,
    stall_reward: float = STALL_REWARD,
    stalled: float = 0.0,
) -> np.ndarray:
    """(2, steps): each team's reward at each step of a point played. `counted`: per
    step, the team counting the stall (stall_watch), for it to gain `stall_reward` a
    second then."""
    found = np.zeros((2, steps))
    for i, team in enumerate((counted or [])[:steps]):
        if team >= 0:
            found[team, i] += stall_reward * step_seconds

    def at(seconds: float) -> int:
        return min(steps - 1, max(0, int(round(seconds / step_seconds)) - 1))

    def give(team: int, seconds: float, amount: float) -> None:
        found[team, at(seconds)] += amount
        found[1 - team, at(seconds)] -= amount

    def along(event) -> float:
        # As far as the attacked goal line: a score is a score wherever it is caught
        return min(float(turn(np.array(event.place), attack_of(event.team))[1]), GOAL_LINE)

    thrown = None  # How far up the field the disc in the air was thrown from
    for event in state.events:
        if event.kind == "throw":
            thrown = along(event)
        if event.kind == "catch":
            # The pass, and the yards it gained (none for a pull caught)
            gained = 0.0 if thrown is None else along(event) - thrown
            give(event.team, event.seconds, COMPLETION + PER_YARD * gained)
        if event.kind in TURNOVERS:
            give(event.team, event.seconds, -TURNOVER)
        if event.kind == "stall":  # Holding on to the end costs more than losing it
            give(event.team, event.seconds, -stalled)
        if event.kind == "score":
            give(event.team, event.seconds, SCORE)
    return found
