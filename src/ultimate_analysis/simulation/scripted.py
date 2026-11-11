"""Teams that play by simple rules: a baseline for the learned ones, and something to
test the simulator with. Each player decides from what they know (perception.py): what
they see, what they remember, what their teammates said.

On offence roles count from the holder (plays.role_of): role 1 is the dump, behind the
thrower on the side the mark leaves open; the others stand in a vertical stack ahead and
take turns to cut to the most open spot a cut can reach, then clear back. On getting the
disc the holder may call a set play, which everybody hears and runs until it is over.
The thrower throws to an open teammate, leading them to where they will be when the disc
gets there, if the throw gains or the stall is late; whoever is nearest to where the
disc comes down runs there.

On defence each player guards a player of the other team, a step goal side and to the
side they force, the mark beside the thrower. Two defenders whose players are nearer to
each other's defender by enough each call to switch; when both have called it, they
switch. Whoever can get to where a throw comes down in time goes for it, once they have
seen it thrown.

ScriptedTeam2 is the same team set as the recorded teams play, a benchmark: a wider
stack, cuts that start with a jab the other way, a thrower who waits for a cutter clear
of their defender, and a deep defender on whoever is furthest downfield.
"""

from typing import List, Optional

import numpy as np

from ..processing.open_space import FieldValue, open_to
from ..processing.throw_targets import CENTRES
from .disc import JUMP_REACH, drift_of, usual_height
from .game import (
    DISC_YARDS_PER_SECOND,
    LENGTH,
    PLAYERS,
    RELEASE_SECONDS,
    RUN_SPEED,
    WIDTH,
    Decision,
)
from .perception import MESSAGE_SIZE, PlayerView
from .plays import PLAYS, role_of

REACT = 0.3  # Seconds: how quickly a player closes on where they want to be
SEEN_THROWN = 0.3  # Seconds before a defender reacts to a throw
STACK_GAP = 5.0  # Yards between players in the stack
STACK_FROM = 12.0  # Yards ahead of the thrower that the stack starts
CUT_EVERY = 2.5  # Seconds between cuts
CUT_SECONDS = 2.0  # How long a cut lasts before the cutter clears
DUMP_BEHIND, DUMP_ACROSS = 6.0, 6.0  # Yards from the thrower
SHORTEST_HOLD = 0.6  # Seconds before a thrower is ready
OPEN_ENOUGH = 0.8  # A teammate this open is thrown to
CLEAR_OF = 2.5  # Yards from the nearest defender a teammate must be to be thrown to
LATE_IN_THE_STALL = 6.0  # Seconds: then any throw that is open enough, even backwards
LAST_RESORT = 8.5  # Seconds: then the most open teammate, however open
CALLS_A_PLAY = 0.4  # The chance a holder calls a set play on getting the disc
GOAL_SIDE, FORCE_SIDE = 1.5, 1.0  # Yards a defender stands from their player
MARK_OFFSET = np.array([1.2, -0.6])  # From the thrower, as the defence sees it
SWITCH_GAIN = 4.0  # Yards both would run less: worth switching
INTERCEPT_MARGIN = 0.5  # Seconds a defender must be first by to go for another's throw
# Throws: curved round a mark this near and this far off the line, by a share of the
# throw's length; lofted over players this far from its path
MARK_NEAR, MARK_LANE = 3.0, 1.5
CURVE_SHARE, LARGEST_CURVE = 0.15, 6.0
UNDER_PATH = 2.0
ALLOWED_FOR = 0.8  # Of the wind's drift a thrower aims off for
OPEN_SIDE, OPEN_SIDE_FROM = 1.0, 3.0  # Yards a throw is put away from a defender this near
CHASERS = 2  # Receivers who go for the pull
PEEL = 2  # Defenders besides the receiver's who go for a throw they can get to in time
PEEL_SLACK = 0.3  # Seconds after it comes down they may still arrive to contest it
LOFT = 1.5  # Yards over jumping reach a throw goes over players under its path
# How a thrower throws, as numbers learning may change: how hard they curve round a mark
# (share of the length), how far over jumping reach they loft, how flat a lofted throw
# is (1: flat, 0: as the curve asks), how much of the wind's drift they aim off for, and
# how much sooner than the recorded throwers they throw (1: as soon)
STYLE = np.array([CURVE_SHARE, LOFT, 1.0, ALLOWED_FOR, 1.0])
FIELD_VALUE = FieldValue()
# What each number of a message means: whom a defender guards, whom they call to switch
# with (each a player's number plus one, 0 for none), the play the holder called
GUARDING, SWITCH, CALL = 0, 1, 2


def towards(place: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """The velocity that takes a player to where they want to be (at full speed if far)."""
    velocity = (np.asarray(wanted) - place) / REACT
    speed = float(np.hypot(*velocity))
    return velocity * (RUN_SPEED / speed) if speed > RUN_SPEED else velocity


def throw_shape(view: PlayerView, target: np.ndarray, style: np.ndarray = STYLE) -> tuple:
    """How a holder throws to `target`: curved away from a mark in the way the disc
    would leave the hand (on a disc as steep as that takes), flat and high over players
    standing under its path. (curve, height, tilt): yards to the left at the middle, the
    top of the arc and the disc's angle (None: as usual). The wind they feel they allow
    for in `aim_off`."""
    thrower = view.ours[view.holder, :2]
    line = np.asarray(target) - thrower
    distance = float(np.hypot(*line))
    if distance < 1e-6:
        return 0.0, None, None
    way = line / distance
    left = np.array([-way[1], way[0]])
    curve = 0.0
    away = view.theirs[:, :2] - thrower
    mark = away[np.argmin(np.hypot(*away.T))]
    if np.hypot(*mark) < MARK_NEAR and mark @ way > 0 and abs(mark @ left) < MARK_LANE:
        curve = -np.sign(mark @ left or 1.0) * min(style[0] * distance, LARGEST_CURVE)
    height = None
    others = np.vstack([np.delete(view.ours[:, :2], view.holder, axis=0), view.theirs[:, :2]])
    along = (others - thrower) @ way
    beside = np.abs((others - thrower) @ left)
    under = (along > 0.2 * distance) & (along < 0.8 * distance) & (beside < UNDER_PATH)
    tilt = None  # As steep as the curve asks
    if under.any():
        height = max(usual_height(distance), JUMP_REACH + style[1] + 0.05 * distance)
        if curve == 0.0:
            tilt = float(np.clip(1.0 - style[2], 0.0, 1.0))  # Flat: a floating huck
    return float(curve), height, tilt


def to_the_open_side(view: PlayerView, target: np.ndarray) -> np.ndarray:
    """Where to put a throw for the teammate nearest `target`: up to OPEN_SIDE yards
    away from the defender nearest to it, as long as the teammate stays the nearer;
    as throwers put the disc where the receiver is and the defender is not."""
    target = np.asarray(target, dtype=float)
    defenders = view.theirs[:, :2]
    away = target - defenders
    distance = np.hypot(*away.T)
    nearest = int(np.argmin(distance))
    if distance[nearest] >= OPEN_SIDE_FROM:
        return target
    way = away[nearest] / max(distance[nearest], 1e-6)
    return target + way * min(OPEN_SIDE, OPEN_SIDE_FROM - distance[nearest])


def aim_off(
    view: PlayerView, target: np.ndarray, shape: tuple, style: np.ndarray = STYLE
) -> np.ndarray:
    """Where to aim so that the wind brings the disc to `target`: most of the way."""
    if view.wind is None or not np.any(view.wind):
        return target
    curve, height, tilt = shape
    drift = drift_of(view.ours[view.holder, :2], target, curve, height, view.wind, tilt)
    return target - style[3] * drift


def throw_options(view: PlayerView) -> np.ndarray:
    """(6, 5): for each teammate of the holder, in order: where to lead them (x, y: where
    they will be when the disc gets there), how open they are there (open_space.open_to),
    what a catch there gains (in the chance of scoring), and yards to the nearest
    defender."""
    thrower = view.ours[view.holder, :2]
    found = []
    for k in range(PLAYERS):
        if k == view.holder:
            continue
        lead = view.ours[k, :2].copy()
        for _ in range(2):  # Where they will be when the disc gets there
            flight = RELEASE_SECONDS + np.hypot(*(lead - thrower)) / DISC_YARDS_PER_SECOND
            lead = view.ours[k, :2] + view.ours[k, 2:] * min(flight, 1.5)
        lead = np.clip(lead, [1.0, 1.0], [WIDTH - 1.0, LENGTH - 1.0])
        receiver = view.ours[[k]].copy()
        chance = float(open_to(thrower, receiver, view.theirs, lead[None].astype(np.float32))[0])
        gain = float(FIELD_VALUE(lead[1:2])[0] - FIELD_VALUE(thrower[1:2])[0])
        clear = float(np.min(np.hypot(*(view.theirs[:, :2] - view.ours[k, :2]).T)))
        found.append([lead[0], lead[1], chance, gain, clear])
    return np.array(found)


def peels(view: PlayerView) -> bool:
    """Whether a defender goes for a throw in the air besides the receiver's own: one of
    the PEEL of the team who would get to where they judge it comes down soonest, if
    they would be there by then or soon enough after to contest it."""
    times = np.hypot(*(view.ours[:, :2] - view.disc).T) / RUN_SPEED
    soonest = np.argsort(times)[:PEEL]
    return bool(view.me in soonest and times[view.me] <= view.lands_in + PEEL_SLACK)


def goes_for_it(view: PlayerView) -> bool:
    """Whether an offence player goes for the disc in the air or on the ground: whoever
    believes they are nearest, and whoever believes they can get to a throw in time (the
    one it is meant for may not be the nearest); of the pull, which everyone could reach,
    only the CHASERS nearest. Others are seen late, so two may go; that is better than
    none."""
    distances = np.hypot(*(view.ours[:, :2] - view.disc).T)
    nearest = np.argsort(distances)
    if view.lands_in is not None and (view.me in nearest[:CHASERS] or not view.pulled):
        if distances[view.me] / RUN_SPEED <= view.lands_in + 0.3:
            return True
    return int(nearest[0]) == view.me


class ScriptedTeam:
    """A team of players who play by the rules above, either way round. What each keeps
    from step to step is theirs alone: whom they guard, when they heard a play called."""

    # How the team plays (see the constants above); ScriptedTeam2 sets them otherwise
    stack_gap, stack_from, stack_across = STACK_GAP, STACK_FROM, 0.0
    cut_every, cut_seconds, jab = CUT_EVERY, CUT_SECONDS, 0.0
    shortest_hold, open_enough, clear_of = SHORTEST_HOLD, OPEN_ENOUGH, CLEAR_OF
    huck_from, huck_open = np.inf, 1.0  # Throws this long only to a teammate this open
    length_cost = 0.0  # Per yard: a longer throw is liked less by exp(-this * length)
    goal_side, force_side, deep_goal_side = GOAL_SIDE, FORCE_SIDE, 0.0

    def __init__(self, seed: Optional[int] = None, plays: bool = True) -> None:
        self.rng = np.random.default_rng(seed)
        self.plays = plays
        self.guarding = list(range(PLAYERS))  # Defender -> the player they guard
        self.called: List[Optional[tuple]] = [None] * PLAYERS  # (play, when, origin)
        self.cutting: dict = {}  # Player -> (when their cut began, where it goes)

    def decide(self, views: List[PlayerView]) -> Decision:
        velocities = np.zeros((PLAYERS, 2))
        messages = np.zeros((PLAYERS, MESSAGE_SIZE))
        throw_to, curve, height, tilt = None, 0.0, None, None
        for k, view in enumerate(views):
            if view.attacking:
                velocities[k], messages[k], throw = self._attack(view)
                if throw is not None:
                    throw = to_the_open_side(view, throw)
                    curve, height, tilt = throw_shape(view, throw)
                    throw_to = aim_off(view, throw, (curve, height, tilt))
            else:
                velocities[k], messages[k] = self._defend(view)
        return Decision(velocities, throw_to, messages, curve, height, tilt)

    # ------------------------------------------------------------------ offence

    def _attack(self, view: PlayerView):
        me = view.me
        place = view.ours[me, :2]
        said = np.zeros(MESSAGE_SIZE)
        self.guarding[me] = me  # On defence again, each starts with their own player
        if view.holder is None:
            self.called[me] = None
            wanted = self._default_spot(view, me) if view.disc is None else place
            if view.disc is not None:
                wanted = view.disc if goes_for_it(view) else self._default_spot(view, me)
            return towards(place, wanted), said, None
        holder = view.holder
        if me == holder:
            if view.held < 0.15 and self.plays and self.rng.random() < CALLS_A_PLAY:
                self.called[me] = (int(self.rng.integers(len(PLAYS))), view.seconds, place.copy())
            if self.called[me] is not None:
                said[CALL] = self.called[me][0] + 1
            return np.zeros(2), said, self._throw(view)
        # A play called by the holder is heard and run until it is over
        heard = int(round(view.heard[holder, CALL]))
        if not 0 <= heard - 1 < len(PLAYS):
            heard = 0
        if heard and self.called[me] is None:
            self.called[me] = (heard - 1, view.seconds, view.ours[holder, :2].copy())
        if not heard:
            self.called[me] = None
        if self.called[me] is not None:
            play, when, origin = self.called[me]
            spot = PLAYS[play].spot(role_of(me, holder), view.seconds - when, origin)
            if spot is not None:
                return towards(place, spot), said, None
        return towards(place, self._default_spot(view, me)), said, None

    def _default_spot(self, view: PlayerView, me: int) -> np.ndarray:
        """Where a player stands or cuts to without a play. With the disc in the air or on
        the ground, the stack forms on where it comes down, around whoever is nearest it."""
        holder, thrower = view.holder, None
        if holder is None:
            if view.disc is None:
                holder = me
            else:
                holder = int(np.argmin(np.hypot(*(view.ours[:, :2] - view.disc).T)))
                thrower = np.asarray(view.disc, dtype=float)
        if thrower is None:
            thrower = view.ours[holder, :2]
        role = role_of(me, holder)
        if role == 1:
            # The dump: behind the thrower, on the side the mark leaves open
            marks = view.theirs[:, :2]
            mark = marks[np.argmin(np.hypot(*(marks - thrower).T))]
            side = -1.0 if mark[0] > thrower[0] else 1.0
            return thrower + [side * DUMP_ACROSS, -DUMP_BEHIND]
        k = role - 2
        ahead = thrower[1] + self.stack_from + self.stack_gap * k
        # Zig-zag across the field by stack_across, every other one
        across = WIDTH / 2 + self.stack_across * (1 if k % 2 else -1)
        spot = np.array([across, min(ahead, LENGTH - 2.0)])
        # Cutters take turns from when the holder got the disc
        every = self.cut_every
        turn = int(view.held // every) % (PLAYERS - 2)
        if k == turn and view.held % every < self.cut_seconds and view.holder is not None:
            # A cut commits to its spot, chosen as it starts
            began = round(view.seconds - view.held % every, 3)
            if self.cutting.get(me, (None,))[0] != began:
                self.cutting[me] = (began, self._cut_to(view, me))
            target = self.cutting[me][1]
            if view.seconds - began < self.jab:
                # First a jab: a few steps the other way, to take the defender with them
                place = view.ours[me, :2]
                away = place - target
                return place + 3.0 * away / max(float(np.hypot(*away)), 1e-6)
            return target
        return spot

    def _cut_to(self, view: PlayerView, me: int) -> np.ndarray:
        """The most open spot a cut can reach in a second or two, worth the most."""
        place = view.ours[me, :2]
        reach = np.hypot(*(CENTRES - place).T)
        near = CENTRES[(reach > 4.0) & (reach < 16.0)].astype(np.float32)
        chance = open_to(view.ours[view.holder, :2], view.ours[[me]], view.theirs, near)
        return near[int(np.argmax(chance * FIELD_VALUE(near[:, 1])))]

    def _throw(self, view: PlayerView) -> Optional[np.ndarray]:
        """Where the thrower throws now, if they do: to an open teammate, led."""
        if view.held < self.shortest_hold:
            return None
        best, best_score, best_open = None, -np.inf, 0.0
        # Late in the count; or held as long, unmarked (nobody counts, but nobody waits for
        # ever)
        late = max(view.stall, view.held)
        thrower = view.ours[view.holder, :2]
        for lead_x, lead_y, chance, gain, clear in throw_options(view):
            lead = np.array([lead_x, lead_y])
            if np.hypot(*(lead - thrower)) >= self.huck_from and chance < self.huck_open:
                continue  # A long throw only to someone well clear
            forward = gain > 0 or late >= LATE_IN_THE_STALL
            score = chance * (1.0 + 4.0 * gain) if forward else -np.inf
            score *= np.exp(-self.length_cost * np.hypot(*(lead - thrower)))
            # Open: a step clear of the nearest defender, unless the stall is late
            covered = clear < self.clear_of and late < LATE_IN_THE_STALL
            if chance >= self.open_enough and score > best_score and not covered:
                best, best_score = lead, score
            if chance > best_open:
                best_open, last_resort = chance, lead
        if best is None and late >= LAST_RESORT and best_open > 0:
            best = last_resort
        if best is None:
            return None
        return best  # The game puts it off by as much as throws go off

    # ------------------------------------------------------------------ defence

    def _defend(self, view: PlayerView):
        me = view.me
        self.called[me] = None
        place = view.ours[me, :2]
        mine = self.guarding[me]
        # Switch: when this defender and another would both run less by enough, each
        # calls it; when both have called it, both switch. Whom the others guard is
        # known only from what they said
        said = np.zeros(MESSAGE_SIZE)
        theirs = view.theirs[:, :2]
        best, best_gain = None, SWITCH_GAIN
        for other in range(PLAYERS):
            their_man = int(round(view.heard[other, GUARDING])) - 1
            if other == me or their_man < 0 or their_man == mine:
                continue
            here, there = place, view.ours[other, :2]
            now = np.hypot(*(here - theirs[mine])) + np.hypot(*(there - theirs[their_man]))
            after = np.hypot(*(here - theirs[their_man])) + np.hypot(*(there - theirs[mine]))
            if now - after > best_gain:
                best, best_gain = other, now - after
        if best is not None:
            said[SWITCH] = best + 1
            if int(round(view.heard[best, SWITCH])) == me + 1:
                mine = self.guarding[me] = int(round(view.heard[best, GUARDING])) - 1
                said[SWITCH] = 0
        said[GUARDING] = mine + 1
        # Seen by the defence, the offence attacks the near end: goal side is -y; all
        # stand on the side the mark stands, which they deny
        man = view.theirs[mine]
        # The defender of whoever is furthest downfield (the least y, as the defence
        # sees it) stands further goal side: the deep defender
        deep = self.deep_goal_side if mine == int(np.argmin(view.theirs[:, 1])) else 0.0
        wanted = man[:2] + man[2:] * REACT + [self.force_side, -(self.goal_side + deep)]
        if view.holder is not None and mine == view.holder:
            wanted = man[:2] + MARK_OFFSET
        if view.disc is not None and view.lands_in is not None and view.flown >= SEEN_THROWN:
            # A throw to their own player they play; another they go for only if they
            # would clearly be there first
            receiver = int(np.argmin(np.hypot(*(view.theirs[:, :2] - view.disc).T)))
            mine_time = np.hypot(*(place - view.disc)) / RUN_SPEED
            theirs_time = np.hypot(*(view.theirs[receiver, :2] - view.disc)) / RUN_SPEED
            first = mine_time < min(view.lands_in, theirs_time) - INTERCEPT_MARGIN
            if receiver == mine or first or peels(view):
                wanted = view.disc
        return towards(place, wanted), said


class ScriptedTeam2(ScriptedTeam):
    """The scripted team set as the recorded teams play (a benchmark, not to learn
    from): spaced as wide as they are (nearest teammate about 8 yd), cuts that start
    with a jab, a thrower who holds the disc about as long as recorded throwers do and
    throws to a cutter clear of their defender, and a deep defender."""

    stack_gap, stack_from, stack_across = 8.0, 12.0, 8.0
    cut_every, cut_seconds, jab = 3.0, 2.5, 0.5
    shortest_hold, open_enough, clear_of = 1.8, 0.85, 3.0
    huck_from, huck_open = 30.0, 0.97
    length_cost = 0.07
    goal_side, force_side, deep_goal_side = 1.5, 1.0, 4.0
