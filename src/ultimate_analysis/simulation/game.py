"""The rules and the physics of the simulated game.

Seven against seven on the USAU field, in steps of a tenth of a second. Each team says
for each of its players the velocity they want; players get there as fast as people do:
no faster than RUN_SPEED, and speeding up, slowing down or turning by at most
ACCELERATION. The player holding the disc stands (they may pivot, not run) and says
whether to throw, and where to.

The disc flies as simulation/disc.py has it: curved or straight, flat or high, slower the
higher; on its way a defender who gets a hand to it, where it passes low enough, knocks
it down. When it gets there the nearest player within reach has it: an offence player catches it, with a
defender within reach as well the window is tight and the catch is a toss-up weighed by
who is nearer; a defender alone blocks it; nobody within reach and it falls. The mark
can block a throw straight past them as it leaves the hand. A disc caught or fallen
outside the field is brought in at the sideline. The defence picks up a fallen disc by
running to it. Ten seconds of holding with a defender within reach is a stall: the disc
goes over where it is held. A catch in the end zone attacked is a score and ends the
point.

Team 0 attacks the far end zone (y grows), team 1 the near one. Teams see the game
turned so that they attack the far end (`view`).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .disc import (  # noqa: F401 (DISC_YARDS_PER_SECOND, RELEASE_SECONDS: for the teams)
    BLADE_DROPPED,
    DISC_YARDS_PER_SECOND,
    FLAT,
    JUMP_REACH,
    PATH_REACH,
    RELEASE_SECONDS,
    Flight,
    flight_of,
)

WIDTH, LENGTH, END_ZONE = 40.0, 110.0, 20.0
PLAYERS = 7
STEP = 0.1  # Seconds
# How players move, measured on the recorded games (docs/experiments/player_physics.py)
RUN_SPEED = 8.0  # Yards a second
ACCELERATION = 7.0  # Yards a second per second, any way: speeding up, braking, turning
CATCH_REACH = (
    2.0  # Yards: who is this near where the disc comes down can have it: an arm and a lunge
)
DEFENDER_REACH = 2.5  # Yards: a defender this near contests the catch
TIGHT_WINDOW = 0.75  # Yards: how much nearer a receiver must be to be sure of a contested catch
POSITION_SECONDS = 0.3  # ... or how much sooner they must have got there
# Getting there sooner counts up to this many seconds: a receiver waiting under a long
# throw does not keep a defender coming in from behind off it
POSITION_MOST = 0.3
DROPPED = 0.02  # Of catches nobody contests
# How far a throw goes from where it was aimed, as a typical error: a little, more for a
# long throw, and more again with the mark close
AIM_OFF, AIM_OFF_PER_YARD, MARKED_AIM = 0.4, 0.03, 1.5
# ... and more again, by the square of the length: a long huck is much harder to place
AIM_OFF_PER_SQUARE_YARD = 0.0005
MARKED_WITHIN = 2.0  # Yards from the thrower
AIM_OFF_PER_CURVE, AIM_OFF_PER_HEIGHT = 0.02, 0.02  # More off, per yard of curve or height
CAUGHT_ON_THE_WAY = 0.3  # Of discs a defender gets a hand to on their way: the rest fall
GUSTS = 0.05  # A throw goes off more by this share per yard a second of wind
STRONGEST_WIND = 7.0  # Yards a second (about 14 miles an hour)
MARK_REACH = 1.0  # Yards from the line of a throw: the mark can block it as it leaves
MARK_ARM = 3.0  # Yards: how far from the thrower along the throw the mark can reach
MARK_BLOCKS = 0.15  # The chance they do, straight at them, less further off: a break goes past
STALL = 10.0  # Seconds
STALL_REACH = 3.0  # Yards: a defender this near the thrower counts the stall
PICK_UP_REACH = 1.0  # Yards: a fallen disc is picked up by a player this near
BODY = 0.6  # Yards: no two players stand nearer than this
PULL_HEIGHT = 15.0  # Yards: the top of the pull's arc
# Players face the disc and run fastest the way they face: sideways slower, backwards
# (backpedalling) slower still. Running away from the disc faster than they can
# backpedal, they turn and run, and so lose sight of it; turning takes time
BACKPEDAL_SPEED = 5.0  # Yards a second
FORWARD_WITHIN, BACKWARD_FROM = np.radians(60.0), np.radians(120.0)  # Of the facing
TURN_RATE = np.radians(360.0)  # A second: half a second to turn round, standing
# Turning costs: the faster they run the slower they turn (at top speed by TURN_SLOWER of
# the rate); a change of direction beyond TURN_FREE loses speed first, up to TURN_LOSS of
# it for a turn right round; and sideways, across the way they run, they speed up at
# most SIDE_ACCELERATION, not ACCELERATION
TURN_SLOWER, TURN_FREE, TURN_LOSS = 0.5, np.radians(60.0), 0.3
SIDE_ACCELERATION = 4.0  # Yards a second per second
# Crowds: players in the way slow a runner (picks, bumps), each by a share of top
# speed, down to at most the slowest share; each defender within reach of where a
# disc comes down beyond the nearest makes a contested catch harder, by CROWD_WEIGHT of
# the measure it is weighed by (who is nearer, who got there first): a receiver well
# first still mostly catches it, one level with them less with every defender more; a
# defender near a low disc's path, beyond where they knock it down for sure, may tip it,
# by a chance each tenth of a second
TRAFFIC_REACH, TRAFFIC_SLOW, TRAFFIC_SLOWEST = 1.5, 0.2, 0.4
CROWD_WEIGHT = 0.5
TIP_REACH, TIP_CHANCE = 2.0, 0.1
# A disc that hangs gives a defender near where it comes down time to make a play on it:
# from LONG_FLIGHT seconds on, a contested catch is harder, up to HANG_WEIGHT (of the
# measure a catch is weighed by) at HANGING seconds longer
LONG_FLIGHT, HANGING, HANG_WEIGHT = 2.0, 2.0, 1.0
POINT_SECONDS = 120.0  # A point not scored by then ends without a score


@dataclass
class Event:
    """Something the game records: a throw, a catch, a turnover, a score."""

    seconds: float
    # "throw", "catch", "block", "mark block", "drop", "out", "stall", "pick up", "score"
    kind: str
    team: int  # The team that had the disc
    place: Tuple[float, float]
    # Who did it: the thrower of a throw, the receiver of a catch or a score, the player
    # whose throw, catch or stall lost the disc; and the defender who blocked it (of a
    # catch: the defender matched with the receiver)
    player: Optional[int] = None
    by: Optional[int] = None


@dataclass
class GameState:
    """Where everything is. Players 0 to 6 are team 0, 7 to 13 team 1."""

    places: np.ndarray  # (14, 2) on the field
    velocities: np.ndarray  # (14, 2)
    offence: int  # The team with the disc, or to pick it up
    holder: Optional[int] = None  # The player holding the disc
    disc: Optional[np.ndarray] = None  # Where the disc lies, if nobody holds it or throws it
    throw: Optional[Flight] = None
    seconds: float = 0.0
    held_since: float = 0.0
    stall: float = 0.0
    scored: Optional[int] = None  # The team that scored the point
    wind: np.ndarray = field(default_factory=lambda: np.zeros(2))  # Yards a second
    events: List[Event] = field(default_factory=list)
    facing: Optional[np.ndarray] = None  # (14, 2) the way each faces (unit); None: the disc
    # Who is matched with whom this possession (match_up): each player's opponent (0 to
    # 13), set when the offence first holds the disc; None before
    pairs: Optional[np.ndarray] = None

    @property
    def over(self) -> bool:
        return self.scored is not None or self.seconds >= POINT_SECONDS


def match_up(places: np.ndarray, offence: int) -> np.ndarray:
    """(14,): each player's opponent for the possession, as teams pick up: the nearest
    pair of attacker and defender first, then the nearest of the rest, and so on."""
    attackers, defenders = _team_players(offence), _team_players(1 - offence)
    gaps = np.hypot(*(places[attackers][:, None] - places[defenders][None]).transpose(2, 0, 1))
    pairs = np.full(2 * PLAYERS, -1)
    for _ in range(PLAYERS):
        a, d = np.unravel_index(int(np.argmin(gaps)), gaps.shape)
        pairs[attackers[a]], pairs[defenders[d]] = defenders[d], attackers[a]
        gaps[a, :], gaps[:, d] = np.inf, np.inf
    return pairs


def team_of(player: int) -> int:
    return 0 if player < PLAYERS else 1


def attack_of(team: int) -> int:
    """+1 if the team attacks the far end zone, -1 the near one."""
    return 1 if team == 0 else -1


def turn(places: np.ndarray, attack: int) -> np.ndarray:
    """Places (..., 2) as seen by a team attacking in `attack`, and back (it is its own
    inverse); velocities turn by the sign alone."""
    places = np.asarray(places, dtype=float)
    return places if attack > 0 else np.array([WIDTH, LENGTH]) - places


@dataclass
class View:
    """The game as a team sees it: turned so that it attacks the far end zone, its own
    players first."""

    team: int
    ours: np.ndarray  # (7, 4): x, y, vx, vy
    theirs: np.ndarray  # (7, 4)
    attacking: bool  # Whether this team has the disc or is to pick it up
    holder: Optional[int]  # Index among `ours` (attacking) or `theirs` of the holder
    disc: Optional[np.ndarray]  # Where the disc lies or comes down, if nobody holds it
    stall: float
    seconds: float
    lands_in: Optional[float] = None  # Seconds until a disc in the air comes down
    held: float = 0.0  # Seconds the holder has had the disc


def view(state: GameState, team: int) -> View:
    """The game as `team` sees it."""
    attack = attack_of(team)
    places = turn(state.places, attack)
    velocities = state.velocities * attack
    rows = np.hstack([places, velocities])
    ours = rows[:PLAYERS] if team == 0 else rows[PLAYERS:]
    theirs = rows[PLAYERS:] if team == 0 else rows[:PLAYERS]
    holder = None if state.holder is None else state.holder % PLAYERS
    disc, lands_in = None, None
    if state.throw is not None:
        disc = turn(state.throw.target, attack)
        lands_in = state.throw.lands - state.seconds
    elif state.disc is not None:
        disc = turn(state.disc, attack)
    return View(
        team,
        ours,
        theirs,
        state.offence == team,
        holder,
        disc,
        state.stall,
        state.seconds,
        lands_in,
        state.seconds - state.held_since if state.holder is not None else 0.0,
    )


@dataclass
class Decision:
    """What a team's players want, each as they see the game turned their team's way: a
    velocity each (7, 2); for a holder of theirs, where to throw (None: hold on); and
    what each says to the others (7, perception.MESSAGE_SIZE), heard the next step."""

    velocities: np.ndarray
    throw_to: Optional[np.ndarray] = None
    messages: Optional[np.ndarray] = None
    throw_curve: float = 0.0  # Yards to the left (as thrown) the path bends, at its middle
    throw_height: Optional[float] = None  # Yards: the top of its arc; None: as usual
    throw_tilt: Optional[float] = None  # The disc's angle, 0 flat to 1 a blade; None: as curved


def a_wind(rng: np.random.Generator, strongest: float = STRONGEST_WIND) -> np.ndarray:
    """A wind for a point, on the field: any direction, up to `strongest`."""
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return rng.uniform(0.0, strongest) * np.array([np.cos(angle), np.sin(angle)])


def pull_start(
    rng: Optional[np.random.Generator] = None, receiving: int = 0, wind: Optional[np.ndarray] = None
) -> GameState:
    """The start of a point: both teams on their goal lines, the other team pulling to
    `receiving`: a long, high throw from the middle of its line to the receiving half,
    as far as pulls go; the receiving team catches it or picks it up where it lands.
    Nobody plays it on its way, and it is neither a pass nor a turnover."""
    rng = rng or np.random.default_rng()
    attack = attack_of(receiving)
    across = np.linspace(4.0, WIDTH - 4.0, PLAYERS)
    receivers = np.column_stack([across, np.full(PLAYERS, END_ZONE)])
    pullers = np.column_stack([across, np.full(PLAYERS, LENGTH - END_ZONE)])
    receivers = receivers + rng.normal(0, 0.5, receivers.shape)
    receivers, pullers = turn(receivers, attack), turn(pullers, attack)
    places = np.vstack([receivers, pullers] if receiving == 0 else [pullers, receivers])
    state = GameState(places, np.zeros_like(places), offence=receiving, holder=None)
    if wind is not None:
        state.wind = np.asarray(wind, dtype=float)
    puller = (PLAYERS if receiving == 0 else 0) + PLAYERS // 2
    target = turn(np.array([rng.uniform(8.0, WIDTH - 8.0), rng.uniform(15.0, 35.0)]), attack)
    state.throw = flight_of(
        state.places[puller], target, 0.0, puller, rng.normal(0, 3.0), PULL_HEIGHT, state.wind, 0.0
    )
    state.throw.pull = True
    return state


def line_up(rng: Optional[np.random.Generator] = None, offence: int = 0) -> GameState:
    """The start of a point as after a pull caught: the team `offence` has the disc on
    its own brick mark, its players in a vertical stack ahead of it; the other team
    stands goal side of them, each a step towards the end zone they defend."""
    rng = rng or np.random.default_rng()
    attack = attack_of(offence)
    thrower = np.array([WIDTH / 2, END_ZONE + 10.0])
    stack = [thrower + [0.0, 10.0 + 5.0 * i] for i in range(PLAYERS - 1)]
    attackers = np.array([thrower, *stack]) + rng.normal(0, 0.5, (PLAYERS, 2))
    defenders = attackers + [1.5, 1.5]  # Goal side and to the force side
    attackers, defenders = turn(attackers, attack), turn(defenders, attack)
    places = np.vstack([attackers, defenders] if offence == 0 else [defenders, attackers])
    holder = 0 if offence == 0 else PLAYERS
    return GameState(places, np.zeros_like(places), offence=offence, holder=holder)


def _move(state: GameState, wanted: np.ndarray) -> None:
    """Move the players towards the velocities wanted, as fast as people can: fastest
    the way they face (see BACKPEDAL_SPEED)."""
    wanted = np.asarray(wanted, dtype=float).copy()
    if state.holder is not None:
        wanted[state.holder] = 0.0  # The thrower pivots and does not run
    _turn(state, wanted)
    speed = np.hypot(*wanted.T)
    facing_angle = np.arccos(
        np.clip(np.sum(wanted * state.facing, axis=1) / np.maximum(speed, 1e-9), -1.0, 1.0)
    )
    share = np.clip((facing_angle - FORWARD_WITHIN) / (BACKWARD_FROM - FORWARD_WITHIN), 0.0, 1.0)
    fastest = RUN_SPEED + (BACKPEDAL_SPEED - RUN_SPEED) * share
    fastest = fastest * _traffic(state.places)
    too_fast = speed > fastest
    wanted[too_fast] *= (fastest[too_fast] / speed[too_fast])[:, None]
    velocities = _after_turning(state.velocities, wanted)
    change = wanted - velocities
    change = _limit_sideways(velocities, change)
    size = np.hypot(*change.T)
    most = ACCELERATION * STEP
    over = size > most
    change[over] *= (most / size[over])[:, None]
    state.velocities = velocities + change
    state.places = state.places + state.velocities * STEP
    _make_room(state)
    # Players may stand out of bounds but not far from the field
    state.places[:, 0] = np.clip(state.places[:, 0], -5.0, WIDTH + 5.0)
    state.places[:, 1] = np.clip(state.places[:, 1], -5.0, LENGTH + 5.0)


def _after_turning(velocities: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """The velocities after the speed lost to a sharp change of direction."""
    now, goal = np.hypot(*velocities.T), np.hypot(*wanted.T)
    cosine = np.sum(velocities * wanted, axis=1) / np.maximum(now * goal, 1e-9)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    sharp = (now > 1.0) & (goal > 1.0)
    lost = TURN_LOSS * np.clip((angle - TURN_FREE) / (np.pi - TURN_FREE), 0.0, 1.0)
    return velocities * (1.0 - np.where(sharp, lost, 0.0))[:, None]


def _limit_sideways(velocities: np.ndarray, change: np.ndarray) -> np.ndarray:
    """A change of velocity with what is across the way they run held to
    SIDE_ACCELERATION."""
    speed = np.hypot(*velocities.T)
    way = velocities / np.maximum(speed, 1e-9)[:, None]
    along = np.sum(change * way, axis=1)[:, None] * way
    across = change - along
    size = np.hypot(*across.T)
    most = SIDE_ACCELERATION * STEP
    across = np.where(
        ((speed > 1.0) & (size > most))[:, None],
        across * (most / np.maximum(size, 1e-9))[:, None],
        across,
    )
    return np.where((speed > 1.0)[:, None], along + across, change)


def _traffic(places: np.ndarray) -> np.ndarray:
    """The share of top speed each player keeps for the others in their way."""
    apart = np.hypot(*(places[:, None] - places[None]).transpose(2, 0, 1))
    np.fill_diagonal(apart, np.inf)
    crowd = (apart < TRAFFIC_REACH).sum(axis=1)
    return np.clip(1.0 - TRAFFIC_SLOW * crowd, TRAFFIC_SLOWEST, 1.0)


def _turn(state: GameState, wanted: np.ndarray) -> None:
    """Each player turns towards the way they mean to face, at TURN_RATE: the disc (the
    holder, the field ahead), or the way they run if they run from the disc faster than
    they can backpedal."""
    if state.throw is not None:
        disc = state.throw.at(state.seconds)[0]
    elif state.holder is not None:
        disc = state.places[state.holder]
    else:
        disc = state.disc
    towards = np.zeros_like(state.places) if disc is None else np.asarray(disc) - state.places
    distance = np.hypot(*towards.T)
    towards = towards / np.maximum(distance, 1e-9)[:, None]
    if state.holder is not None:
        towards[state.holder] = [0.0, float(attack_of(state.offence))]
    speed = np.hypot(*wanted.T)
    away = (np.sum(wanted * towards, axis=1) < 0.0) & (speed > BACKPEDAL_SPEED)
    target = np.where(away[:, None], wanted / np.maximum(speed, 1e-9)[:, None], towards)
    close = distance < 1e-6
    if state.facing is None:  # At first everyone faces the disc
        state.facing = np.where(close[:, None], [0.0, 1.0], towards)
    target[close & ~away] = state.facing[close & ~away]
    now = np.arctan2(state.facing[:, 1], state.facing[:, 0])
    goal = np.arctan2(target[:, 1], target[:, 0])
    change = (goal - now + np.pi) % (2.0 * np.pi) - np.pi
    rate = TURN_RATE * (
        1.0 - TURN_SLOWER * np.minimum(np.hypot(*state.velocities.T) / RUN_SPEED, 1.0)
    )
    now = now + np.clip(change, -rate * STEP, rate * STEP)
    state.facing = np.column_stack([np.cos(now), np.sin(now)])


def _make_room(state: GameState) -> None:
    """Players take up room: two nearer than BODY are pushed apart, the faster one
    (who ran into the other) further. The holder stands their ground."""
    places = state.places
    apart = places[:, None, :] - places[None, :, :]
    distance = np.hypot(apart[..., 0], apart[..., 1])
    np.fill_diagonal(distance, np.inf)
    for i, j in zip(*np.nonzero(np.triu(distance < BODY))):
        gap = BODY - distance[i, j]
        way = apart[i, j] / max(distance[i, j], 1e-6)
        speeds = np.hypot(*state.velocities[[i, j]].T) + 1e-6
        share_i = speeds[0] / speeds.sum()
        if state.holder == i:
            share_i = 0.0
        elif state.holder == j:
            share_i = 1.0
        places[i] += way * gap * share_i
        places[j] -= way * gap * (1.0 - share_i)


def _in_bounds(place: np.ndarray) -> np.ndarray:
    return np.array([np.clip(place[0], 0.0, WIDTH), np.clip(place[1], 0.0, LENGTH)])


def _turnover(
    state: GameState,
    kind: str,
    place: np.ndarray,
    holder: Optional[int],
    fault: Optional[int] = None,
    by: Optional[int] = None,
) -> None:
    """The disc goes over to the other team, held by `holder` or lying at `place`; lost by
    `fault`, blocked by `by`."""
    state.events.append(Event(state.seconds, kind, state.offence, tuple(place), fault, by))
    state.offence = 1 - state.offence
    state.pairs = None  # Matched afresh when the other team holds the disc
    state.throw = None
    state.holder = holder
    # A disc that falls in the end zone the team now attacks is brought to its goal line
    state.disc = None if holder is not None else _short_of_the_end_zone(state, _in_bounds(place))
    state.held_since, state.stall = state.seconds, 0.0
    if holder is not None:
        state.velocities[holder] = 0.0
        attack = attack_of(state.offence)
        if turn(state.places[holder], attack)[1] >= LENGTH - END_ZONE:
            # Caught by a defender in the end zone they attack: a score (a Callahan)
            state.scored = state.offence
            state.events.append(
                Event(state.seconds, "score", state.offence, tuple(state.places[holder]), holder)
            )


def _short_of_the_end_zone(state: GameState, place: np.ndarray) -> np.ndarray:
    """A place in the end zone the team with the disc attacks, brought to its goal line
    (a disc picked up there is not a score); other places as they are."""
    attack = attack_of(state.offence)
    across, along = turn(place, attack)
    if along <= LENGTH - END_ZONE:
        return place
    return turn(np.array([across, LENGTH - END_ZONE]), attack)


def _start_throw(
    state: GameState,
    target: np.ndarray,
    rng: np.random.Generator,
    curve: float = 0.0,
    height: Optional[float] = None,
    tilt: Optional[float] = None,
) -> None:
    """The holder throws to `target` (field places), with a curve, a height and an angle."""
    start = state.places[state.holder].copy()
    defenders = _team_players(1 - state.offence)
    marked = (np.hypot(*(state.places[defenders] - start).T) < MARKED_WITHIN).any()
    aimed = float(np.hypot(*(np.asarray(target) - start)))
    off = (AIM_OFF + AIM_OFF_PER_YARD * aimed + AIM_OFF_PER_SQUARE_YARD * aimed**2) * (
        MARKED_AIM if marked else 1.0
    )
    # A curve and height are harder to put just right
    off *= (
        1.0 + AIM_OFF_PER_CURVE * abs(curve) + AIM_OFF_PER_HEIGHT * max(0.0, (height or 0) - FLAT)
    )
    off *= 1.0 + GUSTS * float(np.hypot(*state.wind))  # A strong wind gusts
    target = np.asarray(target, dtype=float) + rng.normal(0.0, off, 2)
    state.events.append(Event(state.seconds, "throw", state.offence, tuple(start), state.holder))
    state.throw = flight_of(
        start, target, state.seconds, state.holder, curve, height, state.wind, tilt
    )
    state.holder = None
    # The mark: a defender the disc leaves the hand towards, within reach (a throw that
    # leaves away from the mark's side goes free, curved round if need be)
    away = state.places[defenders] - start
    way = state.throw.leaving()
    along = away @ way
    beside = np.abs(away @ np.array([-way[1], way[0]]))
    reached = (along > 0.0) & (along < MARK_ARM) & (beside < MARK_REACH)
    if reached.any():
        nearest = int(np.flatnonzero(reached)[np.argmin(beside[reached])])
        blocker = defenders[nearest]
        if rng.random() < MARK_BLOCKS * (1.0 - beside[nearest] / MARK_REACH):
            _turnover(
                state,
                "mark block",
                state.places[blocker].copy(),
                blocker,
                state.throw.thrower,
                blocker,
            )


def _in_the_way(state: GameState, rng: np.random.Generator) -> bool:
    """A defender gets a hand to the disc on its way, where it passes low enough near
    them: knocked down (now and then caught). Whether it happened."""
    flight = state.throw
    if flight.pull:
        return False
    share = flight.share(state.seconds)
    if not 0.1 < share < 0.85:  # Leaving the hand is the mark's; coming down, a contest
        return False
    place, height = flight.at(state.seconds)
    if height > JUMP_REACH:
        return False
    defenders = _team_players(1 - state.offence)
    near = np.hypot(*(state.places[defenders] - place).T)
    if near.min() > PATH_REACH:
        # Further off, each defender near its path may get a fingertip to it
        tippers = (near > PATH_REACH) & (near <= TIP_REACH)
        if not tippers.any() or rng.random() >= 1.0 - (1.0 - TIP_CHANCE) ** int(tippers.sum()):
            return False
        tipper = defenders[int(np.argmin(np.where(tippers, near, np.inf)))]
        _turnover(state, "block", place, None, flight.thrower, tipper)
        return True
    blocker = defenders[int(np.argmin(near))]
    if rng.random() < CAUGHT_ON_THE_WAY:
        _turnover(state, "block", state.places[blocker].copy(), blocker, flight.thrower, blocker)
    else:
        _turnover(state, "block", place, None, flight.thrower, blocker)
    return True


def _unit(v: np.ndarray) -> np.ndarray:
    size = float(np.hypot(*v))
    return v / size if size > 1e-9 else np.zeros(2)


def _team_players(team: int) -> np.ndarray:
    return np.arange(PLAYERS) + (0 if team == 0 else PLAYERS)


def _land(state: GameState, rng: np.random.Generator) -> None:
    """The disc comes down: caught, blocked or fallen."""
    throw = state.throw
    distance = np.hypot(*(state.places - throw.target).T)
    if throw.pull:
        # The receiving team catches the pull or picks it up where it lands
        receiving = _team_players(state.offence)
        nearest = receiving[np.argmin(distance[receiving])]
        state.throw = None
        if distance[nearest] <= CATCH_REACH and rng.random() >= DROPPED:
            state.holder = nearest
            state.velocities[nearest] = 0.0
            state.held_since, state.stall = state.seconds, 0.0
            state.events.append(
                Event(state.seconds, "pull caught", state.offence, tuple(state.places[nearest]))
            )
        else:
            state.disc = _in_bounds(throw.target)
            state.events.append(
                Event(state.seconds, "pull lands", state.offence, tuple(state.disc))
            )
        return
    offence, defence = _team_players(state.offence), _team_players(1 - state.offence)
    near_offence = offence[np.argmin(distance[offence])]
    near_defence = defence[np.argmin(distance[defence])]
    d_off, d_def = distance[near_offence], distance[near_defence]
    if d_off <= CATCH_REACH:
        dropped = DROPPED + BLADE_DROPPED * throw.tilt  # A blade comes in on edge
        if d_def > DEFENDER_REACH:
            caught = rng.random() >= dropped
        else:
            # A defender near: the window is tight, weighed by who is nearer and who
            # got there first
            position = throw.in_reach_since.get(
                near_defence, state.seconds
            ) - throw.in_reach_since.get(near_offence, state.seconds)
            position = float(np.clip(position, -POSITION_MOST, POSITION_MOST))
            hang = np.clip((throw.lands - throw.released - LONG_FLIGHT) / HANGING, 0.0, 1.0)
            measure = (d_def - d_off) / TIGHT_WINDOW + position / POSITION_SECONDS
            # Every other defender there makes it harder still
            crowd = max(int((distance[defence] <= DEFENDER_REACH).sum()) - 1, 0)
            measure -= CROWD_WEIGHT * crowd
            chance = 0.5 + 0.5 * np.tanh(measure - HANG_WEIGHT * hang)
            caught = rng.random() < (1.0 - dropped) * chance
        if caught:
            _catch(state, near_offence)
        elif d_def <= CATCH_REACH:
            _turnover(
                state,
                "block",
                state.places[near_defence].copy(),
                near_defence,
                throw.thrower,
                near_defence,
            )
        elif d_def <= DEFENDER_REACH:  # Knocked down
            _turnover(state, "block", throw.target, None, throw.thrower, near_defence)
        else:  # Dropped by the receiver
            _turnover(state, "drop", throw.target, None, near_offence)
        return
    if d_def <= CATCH_REACH:
        _turnover(
            state,
            "block",
            state.places[near_defence].copy(),
            near_defence,
            throw.thrower,
            near_defence,
        )
        return
    _turnover(state, "drop", throw.target, None, throw.thrower)  # Nobody there


def _catch(state: GameState, player: int) -> None:
    place = state.places[player].copy()
    if not (0.0 <= place[0] <= WIDTH and 0.0 <= place[1] <= LENGTH):
        # Caught out of bounds: the other team brings it in where it went out
        _turnover(state, "out", _in_bounds(place), None, player)
        return
    state.throw = None
    attack = attack_of(state.offence)
    # The defender matched with the receiver, whose player got it
    guard = None if state.pairs is None else int(state.pairs[player])
    state.events.append(Event(state.seconds, "catch", state.offence, tuple(place), player, guard))
    along = turn(place, attack)[1]
    if along >= LENGTH - END_ZONE:
        state.scored = state.offence
        state.events.append(Event(state.seconds, "score", state.offence, tuple(place), player))
        return
    state.holder = player
    state.velocities[player] = 0.0
    state.held_since, state.stall = state.seconds, 0.0


def step(
    state: GameState, decisions: Tuple[Decision, Decision], rng: np.random.Generator
) -> GameState:
    """Play a tenth of a second, in place; the decisions are each team's, as it sees the
    game."""
    if state.over:
        return state
    wanted = np.zeros_like(state.velocities)
    for team, decision in enumerate(decisions):
        attack = attack_of(team)
        wanted[_team_players(team)] = np.asarray(decision.velocities, dtype=float) * attack
    if state.holder is not None:
        decision = decisions[state.offence]
        if decision.throw_to is not None:
            target = turn(decision.throw_to, attack_of(state.offence))
            _start_throw(
                state, target, rng, decision.throw_curve, decision.throw_height, decision.throw_tilt
            )
    _move(state, wanted)
    state.seconds = round(state.seconds + STEP, 6)

    if state.throw is not None:
        near = np.hypot(*(state.places - state.throw.target).T) <= CATCH_REACH
        for player in range(len(near)):
            if not near[player]:
                state.throw.in_reach_since.pop(player, None)
            else:
                state.throw.in_reach_since.setdefault(player, state.seconds)
    if state.throw is not None and state.seconds < state.throw.lands:
        _in_the_way(state, rng)
    if state.throw is not None and state.seconds >= state.throw.lands:
        _land(state, rng)
    elif state.holder is not None:
        defenders = _team_players(1 - state.offence)
        marked = (
            np.hypot(*(state.places[defenders] - state.places[state.holder]).T) < STALL_REACH
        ).any()
        state.stall = state.stall + STEP if marked else state.stall
        if state.stall >= STALL:
            _turnover(state, "stall", state.places[state.holder].copy(), None, state.holder)
    elif state.disc is not None:
        # The offence picks up a fallen disc by getting to it
        offence = _team_players(state.offence)
        distance = np.hypot(*(state.places[offence] - state.disc).T)
        if distance.min() <= PICK_UP_REACH:
            state.holder = offence[int(np.argmin(distance))]
            state.disc = None
            state.velocities[state.holder] = 0.0
            state.held_since, state.stall = state.seconds, 0.0
            state.events.append(
                Event(state.seconds, "pick up", state.offence, tuple(state.places[state.holder]))
            )
    if state.holder is not None and state.pairs is None:
        state.pairs = match_up(state.places, state.offence)
    return state
