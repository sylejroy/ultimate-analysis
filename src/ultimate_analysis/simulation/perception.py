"""What each simulated player knows: what they see, what they remember, what they hear.

A player looks at the disc: at whoever holds it, at it in the air, or where it lies.
Others within VIEW_ANGLE of that, or close enough to be noticed anyway, are seen a little
off, and as they were REACTION ago: a player knows at once what they themselves do, the
others' moves they see late. A cutter who plants and goes has that start on their
defender. Of the rest they know only where they last saw them, carried on
the way they were running then, and how sure that is: less with every second unseen.
Everybody knows who holds the disc, sees the disc in the air (where it is and how high)
and feels the wind.

Each player also says something every step: a few numbers their teammates hear the
step after. Teams may use it to switch, to call a play, or however they learn to; the
holder's call of a set play is heard by all of the team.

A team's players see the game turned so that their team attacks the far end zone.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .game import PLAYERS, STEP, GameState, attack_of, turn

MESSAGE_SIZE = 4  # Numbers a player says each step
VIEW_ANGLE = np.radians(110.0)  # Either side of where a player looks
NOTICED = 3.0  # Yards: a player this near is noticed wherever they are
SEEN_OFF = 0.3  # Yards: how far off a place seen is, as a typical error
DRIFT = 1.0  # Yards a second: how fast a remembered place becomes unsure
CARRY_ON = 1.0  # Seconds: how long someone out of sight is believed to keep running
REACTION = 0.25  # Seconds: how late the other team's moves are seen
TEAM_REACTION = 0.1  # ... and teammates': a team knows its patterns and reads its cutters
ACCELERATION_OVER = 0.2  # Seconds: how a change of pace or way is seen
LANDING_GUESS = 4.0  # Yards: how far off the defence judges where a throw comes down, at first


@dataclass
class PlayerView:
    """The game as one player knows it, turned so that their team attacks the far end."""

    me: int  # Their index among `ours`
    attacking: bool  # Whether their team has the disc or is to pick it up
    ours: np.ndarray  # (7, 4): x, y, vx, vy as they believe; their own row as it is
    theirs: np.ndarray  # (7, 4)
    ours_unsure: np.ndarray  # (7,): yards each belief may be off
    theirs_unsure: np.ndarray
    holder: Optional[int]  # Index of the holder among `ours` (attacking) or `theirs`
    disc: Optional[np.ndarray]  # Where the disc lies or comes down, if nobody holds it
    lands_in: Optional[float]  # Seconds until a disc in the air comes down
    flown: float  # Seconds since it was thrown
    stall: float
    held: float  # Seconds the holder has had the disc
    seconds: float
    heard: np.ndarray  # (7, MESSAGE_SIZE): what each teammate said last step
    disc_now: Optional[np.ndarray] = None  # Where a disc in the air is now, over the field
    disc_height: float = 0.0  # ... and how high
    wind: np.ndarray = None  # Yards a second, as their team attacks
    pulled: bool = False  # Whether the disc in the air is the pull
    ours_accel: np.ndarray = None  # (7, 2): how each is seen speeding up or turning
    theirs_accel: np.ndarray = None  # (0 for those not seen now)
    ours_facing: np.ndarray = None  # (7, 2): the way each faces, as seen (0: not seen)
    theirs_facing: np.ndarray = None


class Senses:
    """What one team's players know, kept from step to step."""

    def __init__(self, team: int, rng: Optional[np.random.Generator] = None) -> None:
        self.team = team
        self.rng = rng or np.random.default_rng()
        # Per observer and per player: where last seen (field places), how fast, when
        self._seen_at = np.zeros((PLAYERS, 2 * PLAYERS, 2))
        self._seen_going = np.zeros((PLAYERS, 2 * PLAYERS, 2))
        self._seen_when = np.full((PLAYERS, 2 * PLAYERS), -np.inf)
        self._seen_accel = np.zeros((PLAYERS, 2 * PLAYERS, 2))
        self._seen_facing = np.zeros((PLAYERS, 2 * PLAYERS, 2))
        self.heard = np.zeros((PLAYERS, MESSAGE_SIZE))
        self._before: deque = deque(maxlen=int(round((REACTION + ACCELERATION_OVER) / STEP)) + 2)
        self._guesses: dict = {}  # (throw, player) -> how far off they judge it

    def hear(self, messages: np.ndarray) -> None:
        """What the team's players said this step, heard by the team the next."""
        self.heard = np.asarray(messages, dtype=float).reshape(PLAYERS, MESSAGE_SIZE)

    def look(self, state: GameState) -> List[PlayerView]:
        """Each of the team's players looks; what each then knows."""
        first = 0 if self.team == 0 else PLAYERS
        focus = _disc_now(state)
        # Others are seen as they were REACTION ago
        self._before.append((state.seconds, state.places.copy(), state.velocities.copy()))
        _, seen_places, seen_velocities = self._as_of(state.seconds - REACTION)
        _, team_places, team_velocities = self._as_of(state.seconds - TEAM_REACTION)
        mine = slice(first, first + PLAYERS)
        seen_places, seen_velocities = seen_places.copy(), seen_velocities.copy()
        seen_places[mine], seen_velocities[mine] = team_places[mine], team_velocities[mine]
        # How each was speeding up or turning, seen as late
        before = self._as_of(state.seconds - REACTION - ACCELERATION_OVER)[2].copy()
        before[mine] = self._as_of(state.seconds - TEAM_REACTION - ACCELERATION_OVER)[2][mine]
        seen_accel = (seen_velocities - before) / ACCELERATION_OVER
        own_before = self._as_of(state.seconds - ACCELERATION_OVER)[2]
        # All the team's players at once: k the player, me their number in the game
        ks = np.arange(PLAYERS)
        mes = first + ks
        offsets = state.places[None] - state.places[mes][:, None]  # (7, 14, 2)
        distance = np.hypot(offsets[..., 0], offsets[..., 1])
        # They see the way they face (game._turn): the disc, unless they have turned to run
        looking = (focus - state.places[mes]) if state.facing is None else state.facing[mes].copy()
        size = np.hypot(looking[:, 0], looking[:, 1])
        flat = size < 1e-6  # The holder looks downfield
        looking[flat] = [0.0, float(attack_of(self.team))]
        size[flat] = 1.0
        cosine = np.einsum("kpd,kd->kp", offsets, looking) / np.maximum(
            distance * size[:, None], 1e-9
        )
        seen = (cosine >= np.cos(VIEW_ANGLE)) | (distance <= NOTICED)
        seen[ks, mes] = True
        if state.holder is not None and first <= state.holder < first + PLAYERS:
            seen[state.holder - first] = True  # The holder pivots and looks round
        seen[~np.isfinite(self._seen_when).any(axis=1)] = True  # At the line-up, everybody
        noise = self.rng.normal(0.0, SEEN_OFF, (PLAYERS, 2 * PLAYERS, 2))
        noise[ks, mes] = 0.0
        places = np.repeat(seen_places[None], PLAYERS, axis=0)
        velocities = np.repeat(seen_velocities[None], PLAYERS, axis=0)
        places[ks, mes], velocities[ks, mes] = state.places[mes], state.velocities[mes]
        wide = seen[..., None]
        self._seen_at = np.where(wide, places + noise, self._seen_at)
        self._seen_going = np.where(wide, velocities, self._seen_going)
        self._seen_when = np.where(seen, state.seconds, self._seen_when)
        accel = np.repeat(seen_accel[None], PLAYERS, axis=0)
        accel[ks, mes] = (state.velocities[mes] - own_before[mes]) / ACCELERATION_OVER
        self._seen_accel = np.where(wide, accel, 0.0)
        facing = np.zeros((2 * PLAYERS, 2)) if state.facing is None else state.facing
        self._seen_facing = np.where(wide, facing[None], 0.0)
        return self._views_of(state)

    def _as_of(self, seconds: float) -> tuple:
        """The game as it was at a moment just past (the oldest kept if before them)."""
        late = [s for s in self._before if s[0] <= seconds + 1e-6]
        return late[-1] if late else self._before[0]

    def _views_of(self, state: GameState) -> List[PlayerView]:
        attack = attack_of(self.team)
        first = 0 if self.team == 0 else PLAYERS
        since = np.minimum(state.seconds - self._seen_when, 60.0)  # (7, 14)
        # What was seen late is carried on the way they were running: people expect a
        # runner to keep running (not the player themselves, known as they are)
        lag = np.full(2 * PLAYERS, REACTION)
        lag[first : first + PLAYERS] = TEAM_REACTION
        # Carried on as they ran, for a moment: then believed where they would be by then
        ahead = np.minimum(since, CARRY_ON) + lag
        ahead[np.arange(PLAYERS), first + np.arange(PLAYERS)] = 0.0
        believed = self._seen_at + self._seen_going * ahead[..., None]
        unsure = SEEN_OFF + DRIFT * since
        rows = np.concatenate([turn(believed, attack), self._seen_going * attack], axis=2)
        accel = self._seen_accel * attack
        facing = self._seen_facing * attack
        return [
            self._view_of(state, k, rows[k], unsure[k], accel[k], facing[k]) for k in range(PLAYERS)
        ]

    def _view_of(self, state, k, rows, unsure, accel, facing) -> PlayerView:
        attack = attack_of(self.team)
        first = 0 if self.team == 0 else PLAYERS
        mine = slice(first, first + PLAYERS)
        other = slice(PLAYERS - first, 2 * PLAYERS - first)
        holder = None if state.holder is None else state.holder % PLAYERS
        disc, lands_in, flown, now, height, pulled = None, None, 0.0, None, 0.0, False
        if state.throw is not None:
            throw = state.throw
            target = throw.target
            if state.offence != self.team:
                # The defence reads where it comes down from its flight: off at first,
                # surer the longer it flies
                # A throw is known by when and by whom it was thrown (an object's id may
                # come back for a later one)
                which = (throw.released, throw.thrower)
                key = (which, k)
                if key not in self._guesses:
                    if any(seen != which for seen, _ in self._guesses):
                        self._guesses = {}  # Those of an earlier throw
                    self._guesses[key] = self.rng.normal(0.0, LANDING_GUESS, 2)
                left = (throw.lands - state.seconds) / max(throw.lands - throw.released, 1e-6)
                target = target + self._guesses[key] * float(np.clip(left, 0.0, 1.0))
            disc = turn(target, attack)
            now, height = throw.at(state.seconds)
            now = turn(now, attack)
            lands_in = throw.lands - state.seconds
            flown = state.seconds - throw.released
            pulled = throw.pull
        elif state.disc is not None:
            disc = turn(state.disc, attack)
        return PlayerView(
            k,
            state.offence == self.team,
            rows[mine],
            rows[other],
            unsure[mine],
            unsure[other],
            holder,
            disc,
            lands_in,
            flown,
            state.stall,
            state.seconds - state.held_since if state.holder is not None else 0.0,
            state.seconds,
            self.heard.copy(),
            now,
            float(height),
            np.asarray(state.wind, dtype=float) * attack,
            pulled,
            accel[mine],
            accel[other],
            facing[mine],
            facing[other],
        )


def _disc_now(state: GameState) -> np.ndarray:
    """Where the disc is now, on the field."""
    if state.holder is not None:
        return state.places[state.holder]
    if state.throw is not None:
        return state.throw.at(state.seconds)[0]
    if state.disc is not None:
        return state.disc
    return np.array([20.0, 55.0])
