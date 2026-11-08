"""Players whose minds are small networks: learned first from how the recorded players
moved (scripts/train_movement.py), then changed by playing (scripts/evolve_teams.py).

A player sees the game as the recorded players were seen from the drone, turned so that
their team attacks the far end: themselves, the holder, and the others nearest first,
each known or not. In the recorded games a player out of the picture is not known; in
the simulator one remembered too vaguely is not. What teammates said is heard too: in
the recorded games nothing, so that learning by playing can find a use for it.

The network says how the player changes how they run, and what they say to their teammates. One
network moves the offence and one the defence; the holder decides apart: whether to
throw now, as often as the recorded throwers did after holding so long, and where to,
by the network of where the next throw goes (processing/throw_targets.py).
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..processing.open_space import SQUARES, open_to
from ..processing.throw_targets import CENTRES, Moment, ThrowTargets, load_throw_targets
from .game import PLAYERS, RUN_SPEED, Decision
from .perception import MESSAGE_SIZE, PlayerView
from .plays import PLAYS, Play, role_of
from .scripted import CALL, STYLE, aim_off, goes_for_it, throw_shape, to_the_open_side, towards

KNOWN_UNSURE = 3.0  # Yards: a player believed less surely than this is known
NEAREST = (PLAYERS - 1, PLAYERS)  # Teammates and opponents seen, nearest first
OTHER = 5  # Numbers per other player: dx, dy, dvx, dvy, known
FLIGHT = 7  # In the air, where it is, how high, where it comes down, how soon
WIND = 2
ROLE = PLAYERS  # Which place counted from the holder (one of 7)
ACCELERATIONS = 2 * (1 + sum(NEAREST))  # Their own, then the others' as ordered
GROUND = 3  # A disc lying on the field: whether, where
FACINGS = 2 * (1 + sum(NEAREST))  # The way each faces: their own, then the others'
HEARD_BY_SLOT = MESSAGE_SIZE * NEAREST[0]  # What each teammate said, as they are ordered
INPUTS = (
    4
    + 4
    + 1
    + OTHER * sum(NEAREST)
    + PLAYERS * MESSAGE_SIZE
    + FLIGHT
    + WIND
    + ROLE
    + 1
    + ACCELERATIONS
    + GROUND
    + FACINGS
    + HEARD_BY_SLOT
)
# The inputs by kind, in order: (name, how many)
INPUT_GROUPS = (
    ("Themselves", 4),
    ("The holder", 4),
    ("Stall", 1),
    ("Teammates", OTHER * NEAREST[0]),
    ("Opponents", OTHER * NEAREST[1]),
    ("Heard", PLAYERS * MESSAGE_SIZE),
    ("Disc in the air", FLIGHT),
    ("Wind", WIND),
    ("Role", ROLE),
    ("Holding for", 1),
    ("Accelerations", ACCELERATIONS),
    ("Disc on the ground", GROUND),
    ("Facing", FACINGS),
    ("Heard, by teammate", HEARD_BY_SLOT),
)
OUTPUT_NAMES = ("Run across", "Run along", "Say 1", "Say 2", "Play call", "Say 4")
OUTPUTS = 2 + MESSAGE_SIZE
HIDDEN = 128
SPREAD, SPEED = 20.0, RUN_SPEED  # Yards and yards a second the inputs are divided by
ACCELERATION_SCALE = 7.0  # Yards a second per second: the most a player speeds up


def features(
    me: np.ndarray,
    ours: np.ndarray,
    theirs: np.ndarray,
    ours_known: np.ndarray,
    theirs_known: np.ndarray,
    holder: Optional[np.ndarray],
    stall: float,
    heard: Optional[np.ndarray] = None,
    flight: Optional[np.ndarray] = None,
    wind: Optional[np.ndarray] = None,
    role: Optional[int] = None,
    held: float = 0.0,
    accelerations: Optional[tuple] = None,
    ground: Optional[np.ndarray] = None,
    first: Optional[int] = None,
    facings: Optional[tuple] = None,
    mate_numbers: Optional[List[int]] = None,
) -> np.ndarray:
    """What a player's network sees.

    Args:
        me: x, y, vx, vy of the player
        ours: (n, 4) their teammates, without them; theirs: (m, 4) the other team
        ours_known, theirs_known: Whether each is known
        holder: Where the holder stands, if anyone holds the disc (the player too)
        stall: Seconds of the stall count
        heard: (7, MESSAGE_SIZE) what teammates said; nothing if not given
        flight: A disc in the air: x, y, height, x, y where it comes down, seconds
            until it does; nothing if not given (or none in the air)
        wind: Yards a second, as the team attacks; none if not given
        role: The player's place counted from the holder (plays.role_of; on defence, of
            the player of the same number), if anyone holds the disc
        held: Seconds the holder has had the disc
        accelerations: (theirs (2,), teammates' (n, 2), the other team's (m, 2)) in
            yards a second per second, as seen; nothing if not given
        ground: Where a disc lies on the field, if one does
        first: Which of the other team (an index into `theirs`) comes first, before the
            rest nearest first: their matchup (the player of their number)
        facings: (theirs (2,), teammates' (n, 2), the other team's (m, 2)) unit vectors
            as the team attacks, 0 for those not seen; nothing if not given
        mate_numbers: The teammates' numbers (rows of `heard`) in the order of `ours`,
            for what each said to come with where they are
    """
    rows = [
        [me[0] / 40.0, me[1] / 110.0, me[2] / SPEED, me[3] / SPEED],
        [0.0, 0.0, 0.0, 0.0]
        if holder is None
        else [
            (holder[0] - me[0]) / SPREAD,
            (holder[1] - me[1]) / SPREAD,
            float(np.hypot(*(holder - me[:2])) < 0.5),
            1.0,
        ],
        [stall / 10.0],
    ]
    orders = []
    for others, known, slots in (
        (ours, ours_known, NEAREST[0]),
        (theirs, theirs_known, NEAREST[1]),
    ):
        block = np.zeros((slots, OTHER))
        order = np.zeros(0, dtype=int)
        if len(others):
            relative = others - me
            order = np.argsort(np.hypot(*relative[:, :2].T))
            if others is theirs and first is not None:  # Their matchup first
                order = np.r_[first, order[order != first]]
            order = order[:slots]
            block[: len(order), 0:2] = relative[order, 0:2] / SPREAD
            block[: len(order), 2:4] = relative[order, 2:4] / SPEED
            block[: len(order), 4] = known[order]
            block[: len(order)] *= block[: len(order), 4:5]  # Nothing of those not known
        rows.append(block.ravel())
        orders.append((order, known, slots))
    rows.append(np.zeros(PLAYERS * MESSAGE_SIZE) if heard is None else np.ravel(heard))
    in_air = np.zeros(FLIGHT)
    if flight is not None:
        x, y, height, to_x, to_y, soon = flight
        in_air = [
            1.0,
            (x - me[0]) / SPREAD,
            (y - me[1]) / SPREAD,
            height / 10.0,
            (to_x - me[0]) / SPREAD,
            (to_y - me[1]) / SPREAD,
            soon / 3.0,
        ]
    rows.append(in_air)
    rows.append(np.zeros(WIND) if wind is None else np.asarray(wind) / 10.0)
    rows.append(np.eye(ROLE)[role % ROLE] if role is not None else np.zeros(ROLE))
    rows.append([held / 5.0])
    seen = np.zeros(ACCELERATIONS)
    if accelerations is not None:
        own, *others_accel = accelerations
        seen[:2] = np.asarray(own) / ACCELERATION_SCALE
        at = 2
        for accel, (order, known, slots) in zip(others_accel, orders):
            block = np.zeros((slots, 2))
            if len(order):
                block[: len(order)] = np.asarray(accel)[order] * np.asarray(known)[order, None]
            seen[at : at + 2 * slots] = block.ravel() / ACCELERATION_SCALE
            at += 2 * slots
    rows.append(seen)
    rows.append(
        np.zeros(GROUND)
        if ground is None
        else [1.0, (ground[0] - me[0]) / SPREAD, (ground[1] - me[1]) / SPREAD]
    )
    faced = np.zeros(FACINGS)
    if facings is not None:
        own, *others_facing = facings
        faced[:2] = own
        at = 2
        for facing, (order, known, slots) in zip(others_facing, orders):
            block = np.zeros((slots, 2))
            if len(order):
                block[: len(order)] = np.asarray(facing)[order]
            faced[at : at + 2 * slots] = block.ravel()
            at += 2 * slots
    rows.append(faced)
    by_slot = np.zeros((NEAREST[0], MESSAGE_SIZE))
    if heard is not None and len(orders[0][0]):
        mates_heard = (
            np.asarray(heard)[np.asarray(mate_numbers)] if mate_numbers is not None else None
        )
        if mates_heard is not None:
            by_slot[: len(orders[0][0])] = mates_heard[orders[0][0]]
    rows.append(by_slot.ravel())
    return np.concatenate([np.ravel(np.asarray(r, dtype=np.float32)) for r in rows])


def view_features(view: PlayerView) -> np.ndarray:
    """What a simulated player's network sees, from what they know."""
    me = view.ours[view.me]
    mates = [k for k in range(PLAYERS) if k != view.me]
    holder = None
    if view.holder is not None:
        holder = (view.ours if view.attacking else view.theirs)[view.holder, :2]
    return features(
        me,
        view.ours[mates],
        view.theirs,
        view.ours_unsure[mates] < KNOWN_UNSURE,
        view.theirs_unsure < KNOWN_UNSURE,
        holder,
        view.stall,
        view.heard,
        None
        if view.disc_now is None or view.lands_in is None
        else np.array([*view.disc_now, view.disc_height, *view.disc, view.lands_in]),
        view.wind,
        None if view.holder is None else role_of(view.me, view.holder),
        view.held,
        None
        if view.ours_accel is None
        else (view.ours_accel[view.me], view.ours_accel[mates], view.theirs_accel),
        view.disc
        if view.disc is not None and view.lands_in is None and view.holder is None
        else None,
        view.me,
        None
        if view.ours_facing is None
        else (view.ours_facing[view.me], view.ours_facing[mates], view.theirs_facing),
        mates,
    )


def activations(network: Any, x: np.ndarray) -> List[np.ndarray]:
    """What each layer of a movement network gives for one input: the input, both
    hidden layers (after the ReLU), the outputs."""
    import torch

    found = [np.asarray(x, dtype=np.float32)]
    value = torch.from_numpy(found[0])
    with torch.no_grad():
        for layer in network:
            value = layer(value)
            if isinstance(layer, torch.nn.ReLU) or layer is network[-1]:
                found.append(value.numpy().copy())
    return found


# Rewiring: the weakest share of a network's connections is cut, and new ones grow at
# random with small weights; while more than DENSITY of the connections there could
# be are wired, fewer grow than are cut
CUT_SHARE = 0.05
DENSITY = 0.5
NEW_WEIGHT = 0.01
GROW_WHILE_DENSE = 0.5  # Of the number cut, grown back while too densely wired


def wiring_of(network: Any) -> List[np.ndarray]:
    """Which connections of a network are there: its non-zero weights, per layer."""
    return [w != 0 for w in weights_of(network)]


def rewire(
    network: Any, rng: np.random.Generator, share: float = CUT_SHARE, density: float = DENSITY
) -> List[np.ndarray]:
    """Cut the weakest `share` of each layer's connections and grow new ones at random
    where there are none, with small weights (fewer while the layer is more densely
    wired than `density`). The wiring after."""
    import torch

    for layer in network:
        if not isinstance(layer, torch.nn.Linear):
            continue
        weight = layer.weight.detach().numpy().copy()
        wired = np.flatnonzero(weight.ravel() != 0)
        cut = int(share * len(wired))
        if cut == 0:
            continue
        weakest = wired[np.argsort(np.abs(weight.ravel()[wired]))[:cut]]
        flat = weight.ravel()
        flat[weakest] = 0.0
        dense = len(wired) / flat.size > density
        grow = int(cut * (GROW_WHILE_DENSE if dense else 1.0))
        empty = np.flatnonzero(flat == 0)
        empty = empty[~np.isin(empty, weakest)]  # Not back where just cut
        grown = rng.choice(empty, size=min(grow, len(empty)), replace=False)
        flat[grown] = rng.normal(0.0, NEW_WEIGHT, len(grown))
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(flat.reshape(weight.shape)))
    return wiring_of(network)


def keep_wiring(network: Any, wiring: List[np.ndarray]) -> None:
    """Hold cut connections at nothing (after a learning step has moved them)."""
    import torch

    layers = [m for m in network if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for layer, wired in zip(layers, wiring):
            layer.weight.mul_(torch.from_numpy(wired.astype(np.float32)))


def weights_of(network: Any) -> List[np.ndarray]:
    """The weights of a movement network's layers, (outputs, inputs) each."""
    import torch

    return [m.weight.detach().numpy() for m in network if isinstance(m, torch.nn.Linear)]


def movement_network() -> Any:
    """The network that moves a player: what they see in, velocity and words out."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(INPUTS, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, OUTPUTS),
    )


@dataclass
class Minds:
    """A team's networks, how its holder decides when to throw, and its set plays."""

    offence: Any
    defence: Any
    # Chance of throwing in a tenth of a second, per half second held so far
    throw_hazard: np.ndarray
    throw_temperature: float = 1.0  # Over the chances of where the throw goes
    # The team's set plays (plays.Play), and how much a holder is for calling each:
    # the first for calling none
    plays: List[Play] = field(default_factory=lambda: copy.deepcopy(PLAYS))
    play_leaning: np.ndarray = field(default_factory=lambda: np.array([1.0] + [0.0] * len(PLAYS)))
    # How its holders throw (scripted.STYLE: curve, loft, flatness, wind, pace)
    style: np.ndarray = field(default_factory=lambda: STYLE.copy())


def with_inputs(weights: dict) -> dict:
    """A network's weights made before players saw all they see now: the inputs added
    since (at the end) start unconnected, so it plays as it did."""
    import torch

    first = weights["0.weight"]
    if first.shape[1] < INPUTS:
        weights = dict(weights)
        added = torch.zeros(first.shape[0], INPUTS - first.shape[1], dtype=first.dtype)
        weights["0.weight"] = torch.cat([first, added], dim=1)
    return weights


def load_minds(folder: Path) -> Minds:
    """The networks of a run of train_movement.py or evolve_teams.py."""
    import torch

    kept = torch.load(Path(folder) / "minds.pt", map_location="cpu", weights_only=True)
    offence, defence = movement_network(), movement_network()
    offence.load_state_dict(with_inputs(kept["offence"]))
    defence.load_state_dict(with_inputs(kept["defence"]))
    offence.eval()
    defence.eval()
    minds = Minds(
        offence,
        defence,
        np.asarray(kept["throw_hazard"], dtype=float),
        float(kept.get("throw_temperature", 1.0)),
    )
    if "plays" in kept:
        minds.plays = [
            Play(name, {int(role): np.asarray(route) for role, route in routes.items()})
            for name, routes in kept["plays"]
        ]
        minds.play_leaning = np.asarray(kept["play_leaning"], dtype=float)
    if "style" in kept:
        minds.style = np.asarray(kept["style"], dtype=float)
    return minds


def save_minds(minds: Minds, folder: Path, **more: Any) -> None:
    import torch

    Path(folder).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "offence": minds.offence.state_dict(),
            "defence": minds.defence.state_dict(),
            "throw_hazard": [float(v) for v in minds.throw_hazard],
            "throw_temperature": float(minds.throw_temperature),
            "plays": [
                (play.name, {str(role): route.tolist() for role, route in play.routes.items()})
                for play in minds.plays
            ],
            "play_leaning": [float(v) for v in minds.play_leaning],
            "style": [float(v) for v in minds.style],
            **more,
        },
        Path(folder) / "minds.pt",
    )


def reflexes(views: List[PlayerView]) -> dict:
    """Who goes for a disc in the air or on the ground, and how: the offence players who
    can get there (scripted.goes_for_it); a defender who has seen it thrown, if it goes
    to the player nearest them, they would clearly be there first, or they are one of
    the few who would get there in time (scripted.peels)."""
    from .scripted import INTERCEPT_MARGIN, SEEN_THROWN, peels

    going = {}
    for view in views:
        place = view.ours[view.me, :2]
        if view.attacking:
            if goes_for_it(view):
                going[view.me] = towards(place, view.disc)
        elif view.lands_in is not None and view.flown >= SEEN_THROWN:
            receiver = int(np.argmin(np.hypot(*(view.theirs[:, :2] - view.disc).T)))
            nearest = int(np.argmin(np.hypot(*(view.theirs[:, :2] - place).T)))
            mine = np.hypot(*(place - view.disc)) / RUN_SPEED
            theirs = np.hypot(*(view.theirs[receiver, :2] - view.disc)) / RUN_SPEED
            first = mine < min(view.lands_in, theirs) - INTERCEPT_MARGIN
            if receiver == nearest or first or peels(view):
                going[view.me] = towards(place, view.disc)
    return going


class LearnedTeam:
    """A team played by networks."""

    def __init__(self, minds: Minds, throw_weights: Any = None, seed: Optional[int] = None) -> None:
        self.minds = minds
        self.rng = np.random.default_rng(seed)
        self.throw_targets = (
            throw_weights
            if isinstance(throw_weights, ThrowTargets)
            else load_throw_targets(throw_weights)
        )
        # What each player keeps: the play they heard called, when, and from where; and
        # as holder, when they last got the disc and decided on a play
        self.called: List[Optional[tuple]] = [None] * PLAYERS
        self.decided: dict = {}

    def decide(self, views: List[PlayerView]) -> Decision:
        import torch

        attacking = views[0].attacking
        network = self.minds.offence if attacking else self.minds.defence
        x = torch.from_numpy(np.stack([view_features(v) for v in views]))
        with torch.no_grad():
            out = network(x).numpy()
        # The network says how a player changes how they are running
        velocities = (x[:, 2:4].numpy() + out[:, :2]) * RUN_SPEED
        messages = np.tanh(out[:, 2:])
        messages[:, CALL] = 0.0  # Kept for the holder's call of a set play
        throw_to, curve, height, tilt = None, 0.0, None, None
        if attacking and views[0].holder is not None:
            holder = views[0].holder
            velocities[holder] = 0.0
            messages[holder, CALL] = self._call(views[holder])
            throw_to = self._throw(views[holder])
            if throw_to is not None:
                throw_to = to_the_open_side(views[holder], throw_to)
                style = self.minds.style
                curve, height, tilt = throw_shape(views[holder], throw_to, style)
                throw_to = aim_off(views[holder], throw_to, (curve, height, tilt), style)
            for k, view in enumerate(views):
                spot = self._route(view) if k != holder else None
                if spot is not None:
                    velocities[k] = towards(view.ours[k, :2], spot)
        elif views[0].disc is not None:
            # The recorded players were learned from while someone held the disc: going
            # for it is a reflex, as the scripted players have it
            for k, velocity in reflexes(views).items():
                velocities[k] = velocity
        return Decision(velocities, throw_to, messages, curve, height, tilt)

    def _call(self, view: PlayerView) -> float:
        """The play the holder calls on getting the disc (its number plus one, 0 for
        none), and keeps calling while it runs."""
        me = view.me
        got_it = round(view.seconds - view.held, 3)
        if self.decided.get(me) != got_it:  # Once each time they get the disc
            self.decided[me] = got_it
            leaning = np.exp(self.minds.play_leaning - self.minds.play_leaning.max())
            pick = int(self.rng.choice(len(leaning), p=leaning / leaning.sum()))
            self.called[me] = (pick - 1, view.seconds, view.ours[me, :2].copy()) if pick else None
        return 0.0 if self.called[me] is None else float(self.called[me][0] + 1)

    def _route(self, view: PlayerView) -> Optional[np.ndarray]:
        """Where a set play the holder called has this player run, if one does."""
        me, holder = view.me, view.holder
        heard = int(round(view.heard[holder, CALL]))
        if not 0 < heard <= len(self.minds.plays):
            self.called[me] = None
            return None
        if self.called[me] is None or self.called[me][0] != heard - 1:
            self.called[me] = (heard - 1, view.seconds, view.ours[holder, :2].copy())
        play, when, origin = self.called[me]
        return self.minds.plays[play].spot(role_of(me, holder), view.seconds - when, origin)

    def _throw(self, view: PlayerView) -> Optional[np.ndarray]:
        """Whether the holder throws now, and where to."""
        hazard = self.minds.throw_hazard
        pace = max(float(self.minds.style[4]), 0.0)  # Sooner or later than recorded
        chance = hazard[min(int(view.held / 0.5), len(hazard) - 1)] * pace
        if view.held < 0.3 or self.rng.random() >= chance or self.throw_targets is None:
            return None
        others = np.delete(view.ours, view.holder, axis=0)
        known = np.delete(view.ours_unsure, view.holder) < KNOWN_UNSURE
        moment = Moment(
            tuple(view.ours[view.holder, :2]),
            others[known].astype(np.float32),
            view.theirs[view.theirs_unsure < KNOWN_UNSURE].astype(np.float32),
        )
        # Where throws like this go, as recorded, and only where a teammate can get to
        # in time before a defender
        chances = self.throw_targets.chances(moment).ravel()
        reachable = open_to(np.array(moment.thrower), moment.offence, moment.defence, SQUARES)
        weights = chances ** (1.0 / max(self.minds.throw_temperature, 1e-3)) * reachable**2
        if weights.sum() <= 0:
            return None
        weights /= weights.sum()
        return CENTRES[self.rng.choice(len(weights), p=weights)]
