"""Many points at once, on the graphics card: the game of game.py, what players see of
it (perception.py), the reflexes (learned.reflexes) and the rewards (ppo.py), as
tensors with a first dimension for the points, for teams that choose spots (spots.py)
to learn by playing themselves (scripts/train_spots_gpu.py).

The rules are game.py's, step by step; what differs: players who stand too near are
pushed apart all at once, not pair by pair; a point that ends starts again at once with
a new pull. Players are 0 to 6 (team 0) and 7 to 13 (team 1), places on the field.
"""

import math
from typing import Dict

import numpy as np
import torch

from . import perception
from . import spot_features as sf
from .disc import (
    BLADE_ANGLE,
    BLADE_CURVE,
    BLADE_DROPPED,
    BLADE_SPEED,
    DISC_YARDS_PER_SECOND,
    DRIFT_PER_HEIGHT,
    FLAT,
    FLAT_ANGLE,
    FLAT_CURVE,
    HAND_HEIGHT,
    HANG,
    HIGHEST,
    JUMP_REACH,
    LONGEST_THROW,
    PATH_REACH,
    RELEASE_SECONDS,
    SLOWEST,
    WIND_ALONG,
    WIND_DRIFT,
)
from .game import (
    ACCELERATION,
    AIM_OFF,
    AIM_OFF_PER_CURVE,
    AIM_OFF_PER_HEIGHT,
    AIM_OFF_PER_SQUARE_YARD,
    AIM_OFF_PER_YARD,
    BACKPEDAL_SPEED,
    BACKWARD_FROM,
    BODY,
    CATCH_REACH,
    CAUGHT_ON_THE_WAY,
    CROWD_WEIGHT,
    DEFENDER_REACH,
    DROPPED,
    END_ZONE,
    FORWARD_WITHIN,
    GUSTS,
    HANG_WEIGHT,
    HANGING,
    LENGTH,
    LONG_FLIGHT,
    MARK_ARM,
    MARK_BLOCKS,
    MARK_REACH,
    MARKED_AIM,
    MARKED_WITHIN,
    PICK_UP_REACH,
    PLAYERS,
    POSITION_MOST,
    POSITION_SECONDS,
    PULL_HEIGHT,
    RUN_SPEED,
    SIDE_ACCELERATION,
    STALL,
    STALL_REACH,
    STEP,
    STRONGEST_WIND,
    TIGHT_WINDOW,
    TIP_CHANCE,
    TIP_REACH,
    TRAFFIC_REACH,
    TRAFFIC_SLOW,
    TRAFFIC_SLOWEST,
    TURN_FREE,
    TURN_LOSS,
    TURN_RATE,
    TURN_SLOWER,
    WIDTH,
)
from .perception import (
    ACCELERATION_OVER,
    CARRY_ON,
    DRIFT,
    LANDING_GUESS,
    NOTICED,
    REACTION,
    RECEIVER_GUESS,
    RECEIVER_SURE,
    RECEIVER_UNSURE,
    SEEN_OFF,
    TEAM_REACTION,
    VIEW_ANGLE,
)
from .ppo import (
    BLOCKED,
    CAUGHT,
    COMPLETION,
    CRITIC_INPUTS,
    GOAL_LINE,
    LOST,
    PER_YARD,
    SCORE,
    SCORED,
    STALL_REWARD,
    THEIRS_CAUGHT,
    THROWN,
    TURNOVER,
)
from .scripted import (
    CHASERS,
    INTERCEPT_MARGIN,
    LARGEST_CURVE,
    MARK_LANE,
    MARK_NEAR,
    PEEL,
    PEEL_SLACK,
    REACT,
    SEEN_THROWN,
    STYLE,
    UNDER_PATH,
)
from .spots import (
    COMMIT,
    DECIDE_EVERY,
    REACT_STEPS,
    SHORTEST_HOLD,
    THROW_EVERY,
    THROW_PRIOR,
)

P = 2 * PLAYERS
SIZE = (WIDTH, LENGTH)
HISTORY = 9  # Snapshots kept: now and 8 steps back (REACTION and ACCELERATION_OVER)
STEPS_BACK = {  # Steps back: the latest snapshot at least so long ago
    "opponents": math.ceil(REACTION / STEP - 1e-4),
    "team": math.ceil(TEAM_REACTION / STEP - 1e-4),
    "opponents before": math.ceil((REACTION + ACCELERATION_OVER) / STEP - 1e-4),
    "team before": math.ceil((TEAM_REACTION + ACCELERATION_OVER) / STEP - 1e-4),
    "own before": math.ceil(ACCELERATION_OVER / STEP - 1e-4),
}


def _norm(v):
    return torch.linalg.norm(v, dim=-1)


def _turn(places, attack):
    """Places (..., 2) as a team attacking `attack` (..., +1 or -1) sees them."""
    size = torch.tensor(SIZE, device=places.device)
    return torch.where((attack > 0)[..., None], places, size - places)


def _towards(place, wanted):
    """scripted.towards, for many: to where they want to be, at full speed if far."""
    velocity = (wanted - place) / REACT
    speed = _norm(velocity)[..., None]
    return torch.where(speed > RUN_SPEED, velocity * (RUN_SPEED / speed.clamp(min=1e-9)), velocity)


class Flights:
    """The discs in the air, one slot per point (see disc.Flight)."""

    def __init__(self, B, device):
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.start, self.aimed, self.drift = z(B, 2), z(B, 2), z(B, 2)
        self.released, self.lands, self.curve = z(B), z(B), z(B)
        self.height, self.tilt = z(B), z(B)
        self.thrower = torch.zeros(B, dtype=torch.long, device=device)
        self.pull = torch.zeros(B, dtype=torch.bool, device=device)

    @property
    def target(self):
        return self.aimed + self.drift

    def bend(self, idx=slice(None)):
        line = self.aimed[idx] - self.start[idx]
        size = _norm(line).clamp(min=1e-9)[..., None]
        left = torch.stack([-line[..., 1], line[..., 0]], dim=-1) / size
        return (self.start[idx] + self.aimed[idx]) / 2.0 + left * 2.0 * self.curve[idx][..., None]

    def point(self, s, idx=slice(None), drift=True):
        s2 = s[..., None]
        place = (1 - s2) ** 2 * self.start[idx] + 2 * s2 * (1 - s2) * self.bend(idx)
        place = place + s2**2 * self.aimed[idx]
        if drift:
            place = place + self.drift[idx] * s2**2
        height = HAND_HEIGHT + (self.height[idx] - HAND_HEIGHT) * 4.0 * s * (1.0 - s)
        return place, height

    def share(self, seconds, idx=slice(None)):
        span = (self.lands[idx] - self.released[idx]).clamp(min=1e-9)
        return ((seconds - self.released[idx]) / span).clamp(0.0, 1.0)

    def at(self, seconds, idx=slice(None)):
        return self.point(self.share(seconds, idx), idx)

    def leaving(self, idx=slice(None)):
        way = self.bend(idx) - self.start[idx]
        return way / _norm(way).clamp(min=1e-9)[..., None]

    def throw(self, idx, start, target, released, thrower, curve, height, wind, tilt, pull=False):
        """disc.flight_of for the points `idx` (height and tilt NaN: as usual)."""
        offset = target - start
        distance = _norm(offset)
        far = distance > LONGEST_THROW
        target = torch.where(
            far[:, None],
            start + offset * (LONGEST_THROW / distance.clamp(min=1e-9))[:, None],
            target,
        )
        distance = distance.clamp(max=LONGEST_THROW)
        usual = (FLAT + 0.08 * distance).clamp(FLAT, 8.0)
        height = torch.where(torch.isnan(height), usual, height)
        highest = torch.clamp(HAND_HEIGHT + 0.4 * distance + 1.0, max=HIGHEST)
        height = torch.maximum(torch.minimum(height, highest), torch.full_like(height, HAND_HEIGHT))
        steep = (curve.abs() / distance.clamp(min=1e-9) - FLAT_CURVE) / (BLADE_CURVE - FLAT_CURVE)
        tilt = torch.where(torch.isnan(tilt), steep, tilt).clamp(0.0, 1.0)
        most = (FLAT_CURVE + (BLADE_CURVE - FLAT_CURVE) * tilt) * distance
        curve = torch.maximum(torch.minimum(curve, most), -most)
        self.start[idx], self.aimed[idx], self.released[idx] = start, target, released
        self.thrower[idx], self.curve[idx], self.height[idx], self.tilt[idx] = (
            thrower,
            curve,
            height,
            tilt,
        )
        self.pull[idx] = pull
        self.drift[idx] = 0.0
        # Its length over the ground, from 9 points along it (as disc.Flight.length)
        s = torch.linspace(0, 1, 9, device=start.device)
        points = torch.stack([self.point(s[i].expand(len(idx)), idx, False)[0] for i in range(9)])
        length = _norm(points[1:] - points[:-1]).sum(0)
        way = offset / distance.clamp(min=1e-9)[:, None]
        behind = (wind * way).sum(-1)
        speed = DISC_YARDS_PER_SECOND / (1.0 + HANG * (height - FLAT).clamp(min=0.0))
        speed = (speed * (1.0 + BLADE_SPEED * tilt) + WIND_ALONG * behind).clamp(min=SLOWEST)
        flying = RELEASE_SECONDS + length / speed
        self.lands[idx] = released + flying
        across = wind - behind[:, None] * way
        higher = 1.0 + DRIFT_PER_HEIGHT * (height - FLAT).clamp(min=0.0)
        angle = FLAT_ANGLE + (BLADE_ANGLE - FLAT_ANGLE) * tilt.clamp(0.0, 1.0)
        face = torch.sin(angle) / math.sin(FLAT_ANGLE)
        self.drift[idx] = across * (WIND_DRIFT * flying * higher * face)[:, None]


class BatchGame:
    """B points at once: the game, what each team sees of it, and the rewards."""

    def __init__(
        self, B, device="cuda", seed=0, point_seconds=70.0, stalled=3.0, stall_reward=STALL_REWARD
    ):
        self.B, self.device = B, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.point_seconds, self.stalled, self.stall_reward = point_seconds, stalled, stall_reward
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        zl = lambda *s: torch.zeros(*s, dtype=torch.long, device=device)  # noqa: E731
        zb = lambda *s: torch.zeros(*s, dtype=torch.bool, device=device)  # noqa: E731
        self.places, self.velocities, self.facing = z(B, P, 2), z(B, P, 2), z(B, P, 2)
        self.offence, self.holder = zl(B), zl(B) - 1
        self.loose, self.disc = zb(B), z(B, 2)
        self.air = zb(B)
        self.flight = Flights(B, device)
        self.reach_since = torch.full((B, P), float("inf"), device=device)
        self.seconds, self.held_since, self.stall = z(B), z(B), z(B)
        self.scored = zl(B) - 1
        self.wind = z(B, 2)
        self.guess = z(B, PLAYERS, 2)  # The defence's misjudgement of where a throw comes down
        self.guess_off = z(B, PLAYERS, 2)  # ... and the offence's (perception.landing_guess)
        # What each team's players know: per team, per player of it, of every player
        self.seen_at, self.seen_going = z(B, 2, PLAYERS, P, 2), z(B, 2, PLAYERS, P, 2)
        self.seen_when = torch.full((B, 2, PLAYERS, P), -float("inf"), device=device)
        self.seen_accel, self.seen_facing = z(B, 2, PLAYERS, P, 2), z(B, 2, PLAYERS, P, 2)
        self.past_places, self.past_velocities = z(B, HISTORY, P, 2), z(B, HISTORY, P, 2)
        # Who is matched with whom this possession (game.match_up): -1 before
        self.pairs = torch.full((B, P), -1, dtype=torch.long, device=device)
        # Who threw the disc last, and from how far up the field as their team attacks
        self.last_thrower, self.thrown_along = zl(B) - 1, z(B)
        self.step_count = 0  # Steps of the present rollout
        self.tally = {}  # What happened, counted: throws, catches, kinds of turnover, scores
        self.rewards = None  # (T, B, 2) the teams', (T, B, P) the players' own
        self.own = None
        self.reset(torch.ones(B, dtype=torch.bool, device=device))

    # ------------------------------------------------------------------ helpers

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _normal(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def attack_of(self, team):
        return 1.0 - 2.0 * team.float()  # +1 for team 0, -1 for team 1

    def team_rows(self, team):
        """(n, 7) the players of `team` (n,)."""
        return team[:, None] * PLAYERS + torch.arange(PLAYERS, device=self.device)

    # ------------------------------------------------------------------ points

    def reset(self, mask):
        """New points (a pull) where `mask`."""
        idx = torch.nonzero(mask)[:, 0]
        n = len(idx)
        if n == 0:
            return
        self.pairs[idx] = -1
        receiving = (self._rand(n) < 0.5).long()
        attack = self.attack_of(receiving)
        across = torch.linspace(4.0, WIDTH - 4.0, PLAYERS, device=self.device)
        receivers = torch.stack(
            [across.expand(n, -1), torch.full((n, PLAYERS), END_ZONE, device=self.device)], -1
        )
        receivers = receivers + 0.5 * self._normal(n, PLAYERS, 2)
        pullers = torch.stack(
            [across.expand(n, -1), torch.full((n, PLAYERS), LENGTH - END_ZONE, device=self.device)],
            -1,
        )
        receivers = _turn(receivers, attack[:, None])
        pullers = _turn(pullers, attack[:, None])
        first = receiving[:, None, None] == 0
        places = torch.where(
            first, torch.cat([receivers, pullers], 1), torch.cat([pullers, receivers], 1)
        )
        self.places[idx] = places
        self.velocities[idx] = 0.0
        self.offence[idx], self.holder[idx] = receiving, -1
        self.loose[idx], self.air[idx] = False, True
        self.seconds[idx], self.held_since[idx], self.stall[idx] = 0.0, 0.0, 0.0
        self.scored[idx] = -1
        angle = self._rand(n) * 2 * math.pi
        strength = self._rand(n) * STRONGEST_WIND
        self.wind[idx] = torch.stack([angle.cos(), angle.sin()], -1) * strength[:, None]
        puller = torch.where(receiving == 0, PLAYERS, 0) + PLAYERS // 2
        start = places[torch.arange(n, device=self.device), puller]
        target = torch.stack(
            [8.0 + self._rand(n) * (WIDTH - 16.0), 15.0 + self._rand(n) * 20.0], -1
        )
        target = _turn(target, attack)
        self.flight.throw(
            idx,
            start,
            target,
            torch.zeros(n, device=self.device),
            puller,
            3.0 * self._normal(n),
            torch.full((n,), PULL_HEIGHT, device=self.device),
            self.wind[idx],
            torch.zeros(n, device=self.device),
            pull=True,
        )
        self.reach_since[idx] = float("inf")
        self.guess[idx] = LANDING_GUESS * self._normal(n, PLAYERS, 2)
        self.guess_off[idx] = 0.0  # The pull: as before, only the defence (kicking) misjudges
        towards = start[:, None] - places
        size = _norm(towards)[..., None]
        self.facing[idx] = torch.where(
            size > 1e-6,
            towards / size.clamp(min=1e-9),
            torch.tensor([0.0, 1.0], device=self.device),
        )
        self.seen_when[idx] = -float("inf")
        self.seen_at[idx], self.seen_going[idx] = 0.0, 0.0
        self.seen_accel[idx], self.seen_facing[idx] = 0.0, 0.0
        self.past_places[idx] = places[:, None]
        self.past_velocities[idx] = 0.0
        self.last_thrower[idx] = -1

    # ------------------------------------------------------------------ seeing

    def look(self, team: int) -> Dict:
        """What each player of `team` knows (perception.Senses.look for many): views with
        dimensions (B, 7, ...), as spot_features wants them."""
        B, dev = self.B, self.device
        b = torch.arange(B, device=dev)
        first = team * PLAYERS
        mine = slice(first, first + PLAYERS)
        other = slice(PLAYERS - first, 2 * PLAYERS - first)
        attack = 1.0 if team == 0 else -1.0
        back = STEPS_BACK
        seen_places = self.past_places[:, back["opponents"]].clone()
        seen_velocities = self.past_velocities[:, back["opponents"]].clone()
        seen_places[:, mine] = self.past_places[:, back["team"], mine]
        seen_velocities[:, mine] = self.past_velocities[:, back["team"], mine]
        before = self.past_velocities[:, back["opponents before"]].clone()
        before[:, mine] = self.past_velocities[:, back["team before"], mine]
        seen_accel = (seen_velocities - before) / ACCELERATION_OVER
        own_before = self.past_velocities[:, back["own before"]]
        ks = torch.arange(PLAYERS, device=dev)
        mes = first + ks
        offsets = self.places[:, None] - self.places[:, mes, None]  # (B, 7, 14, 2)
        distance = _norm(offsets)
        looking = self.facing[:, mes].clone()
        size = _norm(looking)
        flat = size < 1e-6
        looking[flat] = torch.tensor([0.0, attack], device=dev)
        size = torch.where(flat, torch.ones_like(size), size)
        cosine = (offsets * looking[:, :, None]).sum(-1) / (distance * size[..., None]).clamp(
            min=1e-9
        )
        seen = (cosine >= math.cos(VIEW_ANGLE)) | (distance <= NOTICED)
        seen[:, ks, mes] = True
        holds = (self.holder >= first) & (self.holder < first + PLAYERS)
        seen[b[holds], (self.holder[holds] - first)] = True  # The holder looks round
        fresh = ~torch.isfinite(self.seen_when[:, team]).any(-1)  # (B, 7)
        seen = seen | fresh[..., None]
        if perception.PERFECT:
            seen = torch.ones_like(seen)
        noise = (0.0 if perception.PERFECT else SEEN_OFF) * self._normal(B, PLAYERS, P, 2)
        noise[:, ks, mes] = 0.0
        places = seen_places[:, None].repeat(1, PLAYERS, 1, 1)
        velocities = seen_velocities[:, None].repeat(1, PLAYERS, 1, 1)
        places[:, ks, mes] = self.places[:, mes]
        velocities[:, ks, mes] = self.velocities[:, mes]
        wide = seen[..., None]
        self.seen_at[:, team] = torch.where(wide, places + noise, self.seen_at[:, team])
        self.seen_going[:, team] = torch.where(wide, velocities, self.seen_going[:, team])
        self.seen_when[:, team] = torch.where(
            seen, self.seconds[:, None, None], self.seen_when[:, team]
        )
        accel = seen_accel[:, None].repeat(1, PLAYERS, 1, 1)
        accel[:, ks, mes] = (self.velocities[:, mes] - own_before[:, mes]) / ACCELERATION_OVER
        self.seen_accel[:, team] = torch.where(wide, accel, torch.zeros_like(accel))
        self.seen_facing[:, team] = torch.where(
            wide, self.facing[:, None].expand(-1, PLAYERS, -1, -1), torch.zeros_like(accel)
        )
        # What they believe
        since = (self.seconds[:, None, None] - self.seen_when[:, team]).clamp(max=60.0)
        lag = torch.full((P,), REACTION, device=dev)
        lag[mine] = TEAM_REACTION
        ahead = since.clamp(max=CARRY_ON) + lag
        ahead[:, ks, mes] = 0.0
        believed = self.seen_at[:, team] + self.seen_going[:, team] * ahead[..., None]
        unsure = (0.0 if perception.PERFECT else SEEN_OFF) + DRIFT * since
        attack_t = torch.full((B, PLAYERS, P), attack, device=dev)
        rows = torch.cat([_turn(believed, attack_t), self.seen_going[:, team] * attack], -1)
        known = unsure < sf_known()
        partner = self.pairs[:, mine]  # (B, 7): each of ours, matched with whom
        partner = torch.where(partner >= 0, partner - other.start, torch.full_like(partner, -1))
        v = {
            "partner": partner[:, None].expand(-1, PLAYERS, -1),
            "ours": rows[:, :, mine],
            "theirs": rows[:, :, other],
            "ours_known": known[:, :, mine],
            "theirs_known": known[:, :, other],
            "ours_accel": self.seen_accel[:, team][:, :, mine] * attack,
            "theirs_accel": self.seen_accel[:, team][:, :, other] * attack,
            "ours_facing": self.seen_facing[:, team][:, :, mine] * attack,
            "theirs_facing": self.seen_facing[:, team][:, :, other] * attack,
            "me": ks.expand(B, -1),
        }
        holder = torch.where(self.holder >= 0, self.holder % PLAYERS, -1)
        v["holder"] = holder[:, None].expand(-1, PLAYERS)
        v["attacking"] = (self.offence == team)[:, None].expand(-1, PLAYERS)
        mode = torch.full((B,), sf.NO_DISC, dtype=torch.long, device=dev)
        mode[self.loose] = sf.ON_GROUND
        mode[self.air] = sf.IN_AIR
        mode[self.holder >= 0] = sf.HELD
        v["disc_mode"] = mode[:, None].expand(-1, PLAYERS)
        attack_b = torch.full((B,), attack, device=dev)
        now, height = self.flight.at(self.seconds)
        v["disc_now"] = _turn(now, attack_b)[:, None].expand(-1, PLAYERS, -1)
        # Where it comes down: the defence misjudges it at first, surer as it flies
        target = self.flight.target[:, None].expand(-1, PLAYERS, -1)
        left = (
            (self.flight.lands - self.seconds)
            / (self.flight.lands - self.flight.released).clamp(min=1e-6)
        ).clamp(0.0, 1.0)
        defending = (self.offence != team)[:, None, None]
        target = target + torch.where(
            defending, self.guess * left[:, None, None], self.guess_off * left[:, None, None]
        )
        lies = self.disc[:, None].expand(-1, PLAYERS, -1)
        disc = torch.where(self.air[:, None, None], target, lies)
        v["disc"] = _turn(disc, attack_b[:, None].expand(-1, PLAYERS))
        v["lands_in"] = torch.where(self.air, self.flight.lands - self.seconds, 0.0)[
            :, None
        ].expand(-1, PLAYERS)
        v["height"] = torch.where(self.air, height, 0.0)[:, None].expand(-1, PLAYERS)
        v["wind"] = (self.wind * attack)[:, None].expand(-1, PLAYERS, -1)
        v["stall"] = self.stall[:, None].expand(-1, PLAYERS)
        held = torch.where(self.holder >= 0, self.seconds - self.held_since, 0.0)
        v["held"] = held[:, None].expand(-1, PLAYERS)
        v["flown"] = (self.seconds - self.flight.released)[:, None].expand(-1, PLAYERS)
        v["pulled"] = (self.air & self.flight.pull)[:, None].expand(-1, PLAYERS)
        return v

    # ------------------------------------------------------------------ reflexes

    def reflexes(self, v) -> tuple:
        """learned.reflexes for a team's views (B, 7, ...): who goes for the disc in the air
        or on the ground, and how fast (as the team sees it)."""
        _B, dev = self.B, self.device
        ks = torch.arange(PLAYERS, device=dev)
        mode = v["disc_mode"]
        loose = (mode == sf.IN_AIR) | (mode == sf.ON_GROUND)
        disc = v["disc"]  # (B, 7, 2): as each judges it
        ours = v["ours"][..., :2]  # (B, 7 observers, 7, 2)
        place = ours[:, ks, ks]  # Their own place
        distances = _norm(ours - disc[:, :, None])  # (B, 7, 7)
        rank = distances.argsort(-1).argsort(-1)[:, ks, ks]  # Where they come among the team
        mine = distances[:, ks, ks]
        in_air = mode == sf.IN_AIR
        chase = (
            in_air & ((rank < CHASERS) | ~v["pulled"]) & (mine / RUN_SPEED <= v["lands_in"] + 0.3)
        )
        offence = loose & v["attacking"] & (chase | (rank == 0))
        theirs = v["theirs"][..., :2]
        to_disc = _norm(theirs - disc[:, :, None])  # (B, 7, 7)
        receiver = to_disc.argmin(-1)
        nearest = _norm(theirs - place[:, :, None]).argmin(-1)
        mine_t = mine / RUN_SPEED
        theirs_t = to_disc.gather(-1, receiver[..., None])[..., 0] / RUN_SPEED
        first = mine_t < torch.minimum(v["lands_in"], theirs_t) - INTERCEPT_MARGIN
        peel = (rank < PEEL) & (mine_t <= v["lands_in"] + PEEL_SLACK)
        defence = (
            in_air
            & ~v["attacking"]
            & (v["flown"] >= SEEN_THROWN)
            & ((receiver == nearest) | first | peel)
        )
        going = offence | defence
        return going, _towards(place, disc)

    # ------------------------------------------------------------------ the throw's shape

    def throw_shape(self, v, target) -> tuple:
        """scripted.throw_shape for holders' views (n, ...) and targets (n, 2), as the team
        sees it: (curve, height, tilt), NaN for as usual."""
        n = len(target)
        rows = torch.arange(n, device=self.device)
        thrower = v["ours"][rows, v["holder"], :2]
        line = target - thrower
        distance = _norm(line).clamp(min=1e-6)
        way = line / distance[:, None]
        left = torch.stack([-way[:, 1], way[:, 0]], -1)
        away = v["theirs"][..., :2] - thrower[:, None]
        mark = away[rows, _norm(away).argmin(-1)]
        along_m, beside_m = (mark * way).sum(-1), (mark * left).sum(-1)
        curved = (_norm(mark) < MARK_NEAR) & (along_m > 0) & (beside_m.abs() < MARK_LANE)
        sign = torch.where(beside_m == 0, torch.ones_like(beside_m), beside_m.sign())
        curve = torch.where(
            curved, -sign * torch.clamp(STYLE[0] * distance, max=LARGEST_CURVE), 0.0
        )
        others = torch.cat([v["ours"][..., :2], v["theirs"][..., :2]], 1) - thrower[:, None]
        along = (others * way[:, None]).sum(-1)
        beside = (others * left[:, None]).sum(-1).abs()
        under = (
            (along > 0.2 * distance[:, None])
            & (along < 0.8 * distance[:, None])
            & (beside < UNDER_PATH)
        )
        under[rows, v["holder"]] = False
        any_under = under.any(-1)
        usual = (FLAT + 0.08 * distance).clamp(FLAT, 8.0)
        lofted = torch.maximum(usual, JUMP_REACH + STYLE[1] + 0.05 * distance)
        nan = torch.full_like(distance, float("nan"))
        height = torch.where(any_under, lofted, nan)
        flat_tilt = float(np.clip(1.0 - STYLE[2], 0.0, 1.0))
        tilt = torch.where(any_under & (curve == 0.0), torch.full_like(distance, flat_tilt), nan)
        return curve, height, tilt

    # ------------------------------------------------------------------ playing a step

    def begin_rollout(self, steps):
        self.rewards = torch.zeros(steps, self.B, 2, device=self.device)
        self.own = torch.zeros(steps, self.B, P, device=self.device)
        self.done = torch.zeros(steps, self.B, dtype=torch.bool, device=self.device)
        self.stall_counted = torch.zeros(steps, self.B, dtype=torch.long, device=self.device) - 1
        self.step_count = 0

    def _count(self, what, n):
        self.tally[what] = self.tally.get(what, 0) + int(n)

    def _give_team(self, idx, team, amount, step=None):
        """`amount` to `team` and its opposite to the other, at a step of the rollout."""
        step = self.step_count if step is None else step
        sign = torch.where(team == 0, 1.0, -1.0) * amount
        self.rewards[step, idx, 0] += sign
        self.rewards[step, idx, 1] -= sign

    def _give_own(self, idx, player, amount):
        ok = player >= 0
        self.own[self.step_count, idx[ok], player[ok]] += amount

    def _caught(self, idx, player):
        """A pass caught at `idx` by `player`: to the team the pass and the yards it gained
        (to the goal line at most; none for a pull caught), to the two their catch and
        their throw."""
        attack = self.attack_of(self.offence[idx])
        along = _turn(self.places[idx, player], attack)[:, 1].clamp(max=GOAL_LINE)
        gained = torch.where(self.last_thrower[idx] >= 0, along - self.thrown_along[idx], 0.0)
        self._give_team(idx, self.offence[idx], COMPLETION + PER_YARD * gained)
        self._give_own(idx, player, CAUGHT)
        self._give_own(idx, self.last_thrower[idx], THROWN)

    def _turnover(self, idx, place, holder, fault, by, stalled=False, kind="turnover"):
        """game._turnover for the points `idx` (holder -1: the disc lies at `place`)."""
        if len(idx) == 0:
            return
        self._count(kind, len(idx))
        self._give_team(idx, self.offence[idx], -TURNOVER)
        if stalled:
            self._give_team(idx, self.offence[idx], -self.stalled)
        self._give_own(idx, fault, LOST)
        self._give_own(idx, by, BLOCKED)
        self.offence[idx] = 1 - self.offence[idx]
        self.pairs[idx] = -1  # Matched afresh when the other team holds the disc
        self.air[idx] = False
        self.holder[idx] = holder
        lies = holder < 0
        bounded = torch.stack([place[:, 0].clamp(0.0, WIDTH), place[:, 1].clamp(0.0, LENGTH)], -1)
        attack = self.attack_of(self.offence[idx])
        turned = _turn(bounded, attack)
        turned[:, 1] = turned[:, 1].clamp(max=LENGTH - END_ZONE)  # Brought to the goal line
        self.disc[idx] = _turn(turned, attack)
        self.loose[idx] = lies
        self.held_since[idx] = self.seconds[idx]
        self.stall[idx] = 0.0
        has = ~lies
        if has.any():
            i, h = idx[has], holder[has]
            self.velocities[i, h] = 0.0
            along = _turn(self.places[i, h], self.attack_of(self.offence[i]))[:, 1]
            callahan = along >= LENGTH - END_ZONE
            if callahan.any():
                j, hj = i[callahan], h[callahan]
                self._count("score", len(j))
                self.scored[j] = self.offence[j]
                self._give_team(j, self.offence[j], SCORE)
                self._give_own(j, hj, SCORED)

    def _catch(self, idx, player):
        """game._catch for the points `idx`."""
        place = self.places[idx, player]
        out = (place[:, 0] < 0) | (place[:, 0] > WIDTH) | (place[:, 1] < 0) | (place[:, 1] > LENGTH)
        if out.any():
            self._count("catch out", int(out.sum()))
            self._turnover(
                idx[out],
                place[out],
                torch.full_like(player[out], -1),
                player[out],
                torch.full_like(player[out], -1),
                kind="out",
            )
        idx, player, place = idx[~out], player[~out], place[~out]
        if len(idx) == 0:
            return
        # The defender matched with the receiver, whose player got it (ppo.THEIRS_CAUGHT)
        self._give_own(idx, self.pairs[idx, player], THEIRS_CAUGHT)
        self.air[idx] = False
        attack = self.attack_of(self.offence[idx])
        along = _turn(place, attack)[:, 1]
        score = along >= LENGTH - END_ZONE
        self._count("catch", len(idx))
        if score.any():
            i, p = idx[score], player[score]
            self._count("score", len(i))
            self._caught(i, p)
            self.scored[i] = self.offence[i]
            self._give_team(i, self.offence[i], SCORE)
            self._give_own(i, p, SCORED)
            self._give_own(i, self.last_thrower[i], SCORED)
        rest = ~score
        if rest.any():
            i, p = idx[rest], player[rest]
            self._caught(i, p)
            self.holder[i] = p
            self.velocities[i, p] = 0.0
            self.held_since[i] = self.seconds[i]
            self.stall[i] = 0.0

    def _start_throws(self, idx, target, curve, height, tilt):
        """game._start_throw for the points `idx` (field places)."""
        n = len(idx)
        if n == 0:
            return
        rows = torch.arange(n, device=self.device)
        thrower = self.holder[idx]
        start = self.places[idx, thrower]
        defenders = self.team_rows(1 - self.offence[idx])  # (n, 7)
        their_places = self.places[idx[:, None], defenders]
        marked = (_norm(their_places - start[:, None]) < MARKED_WITHIN).any(-1)
        aimed = _norm(target - start)
        off = (
            AIM_OFF + AIM_OFF_PER_YARD * aimed + AIM_OFF_PER_SQUARE_YARD * aimed**2
        ) * torch.where(marked, MARKED_AIM, 1.0)
        given = torch.nan_to_num(height, nan=0.0)
        off = off * (
            1.0
            + AIM_OFF_PER_CURVE * curve.abs()
            + AIM_OFF_PER_HEIGHT * (given - FLAT).clamp(min=0.0)
        )
        off = off * (1.0 + GUSTS * _norm(self.wind[idx]))
        target = target + off[:, None] * self._normal(n, 2)
        self._count("throw", n)
        self.last_thrower[idx] = thrower
        along = _turn(start, self.attack_of(self.offence[idx]))[:, 1]
        self.thrown_along[idx] = along.clamp(max=GOAL_LINE)
        self.flight.throw(
            idx, start, target, self.seconds[idx], thrower, curve, height, self.wind[idx], tilt
        )
        self.air[idx] = True
        self.holder[idx] = -1
        self.reach_since[idx] = float("inf")
        self.guess[idx] = LANDING_GUESS * self._normal(n, PLAYERS, 2)
        flight = self.flight.lands[idx] - self.flight.released[idx]
        unsure = ((flight - RECEIVER_SURE) / RECEIVER_UNSURE).clamp(0.0, 1.0) * RECEIVER_GUESS
        self.guess_off[idx] = unsure[:, None, None] * self._normal(n, PLAYERS, 2)
        # The mark
        away = their_places - start[:, None]
        way = self.flight.leaving(idx)
        along = (away * way[:, None]).sum(-1)
        beside = (away * torch.stack([-way[:, 1], way[:, 0]], -1)[:, None]).sum(-1).abs()
        reached = (along > 0.0) & (along < MARK_ARM) & (beside < MARK_REACH)
        beside_r = torch.where(reached, beside, torch.full_like(beside, float("inf")))
        nearest = beside_r.argmin(-1)
        chance = MARK_BLOCKS * (1.0 - beside_r[rows, nearest] / MARK_REACH)
        blocked = reached.any(-1) & (self._rand(n) < chance)
        if blocked.any():
            i = idx[blocked]
            blocker = defenders[blocked, nearest[blocked]]
            self._turnover(
                i, self.places[i, blocker], blocker, thrower[blocked], blocker, kind="mark block"
            )

    def _move(self, wanted):
        """game._move and _turn for every point."""
        B, dev = self.B, self.device
        b = torch.arange(B, device=dev)
        holds = self.holder >= 0
        wanted = wanted.clone()
        wanted[b[holds], self.holder[holds]] = 0.0
        # Turning: towards the disc, or the way they run from it flat out
        disc = self.flight.at(self.seconds)[0]
        disc = torch.where(holds[:, None], self.places[b, self.holder.clamp(min=0)], disc)
        disc = torch.where((self.loose & ~self.air)[:, None], self.disc, disc)
        nothing = ~self.air & ~holds & ~self.loose
        towards = disc[:, None] - self.places
        distance = _norm(towards)
        towards = towards / distance.clamp(min=1e-9)[..., None]
        towards = torch.where(nothing[:, None, None], torch.zeros_like(towards), towards)
        ahead = torch.stack([torch.zeros(B, device=dev), self.attack_of(self.offence)], -1)
        towards[b[holds], self.holder[holds]] = ahead[holds]
        speed = _norm(wanted)
        away = ((wanted * towards).sum(-1) < 0.0) & (speed > BACKPEDAL_SPEED)
        target = torch.where(away[..., None], wanted / speed.clamp(min=1e-9)[..., None], towards)
        close = (distance < 1e-6) | nothing[:, None]
        keep = close & ~away
        target = torch.where(keep[..., None], self.facing, target)
        now = torch.atan2(self.facing[..., 1], self.facing[..., 0])
        goal = torch.atan2(target[..., 1], target[..., 0])
        change = torch.remainder(goal - now + math.pi, 2 * math.pi) - math.pi
        rate = TURN_RATE * (1.0 - TURN_SLOWER * (_norm(self.velocities) / RUN_SPEED).clamp(max=1.0))
        now = now + torch.maximum(torch.minimum(change, rate * STEP), -rate * STEP)
        self.facing = torch.stack([now.cos(), now.sin()], -1)
        # Fastest the way they face
        angle = torch.arccos(
            ((wanted * self.facing).sum(-1) / speed.clamp(min=1e-9)).clamp(-1.0, 1.0)
        )
        share = ((angle - FORWARD_WITHIN) / (BACKWARD_FROM - FORWARD_WITHIN)).clamp(0.0, 1.0)
        fastest = RUN_SPEED + (BACKPEDAL_SPEED - RUN_SPEED) * share
        # Others in the way slow them (game._traffic)
        apart = _norm(self.places[:, :, None] - self.places[:, None])
        apart = apart + torch.eye(P, device=dev) * 1e9
        crowd = (apart < TRAFFIC_REACH).sum(-1).float()
        fastest = fastest * (1.0 - TRAFFIC_SLOW * crowd).clamp(TRAFFIC_SLOWEST, 1.0)
        wanted = torch.where(
            (speed > fastest)[..., None],
            wanted * (fastest / speed.clamp(min=1e-9))[..., None],
            wanted,
        )
        # Turning costs (game._after_turning and _limit_sideways)
        now_speed, goal_speed = _norm(self.velocities), _norm(wanted)
        cosine = (self.velocities * wanted).sum(-1) / (now_speed * goal_speed).clamp(min=1e-9)
        turned = torch.arccos(cosine.clamp(-1.0, 1.0))
        lost = TURN_LOSS * ((turned - TURN_FREE) / (math.pi - TURN_FREE)).clamp(0.0, 1.0)
        sharp = (now_speed > 1.0) & (goal_speed > 1.0)
        velocities = (
            self.velocities * (1.0 - torch.where(sharp, lost, torch.zeros_like(lost)))[..., None]
        )
        change = wanted - velocities
        speed_now = _norm(velocities)
        way = velocities / speed_now.clamp(min=1e-9)[..., None]
        along = (change * way).sum(-1, keepdim=True) * way
        across = change - along
        across_size = _norm(across)
        side_most = SIDE_ACCELERATION * STEP
        across = across * (side_most / across_size.clamp(min=1e-9)).clamp(max=1.0)[..., None]
        change = torch.where((speed_now > 1.0)[..., None], along + across, change)
        size = _norm(change)
        most = ACCELERATION * STEP
        change = torch.where(
            (size > most)[..., None], change * (most / size.clamp(min=1e-9))[..., None], change
        )
        self.velocities = velocities + change
        self.places = self.places + self.velocities * STEP
        # Room: pairs nearer than BODY pushed apart, all at once
        apart = self.places[:, :, None] - self.places[:, None]
        gap_d = _norm(apart)
        eye = torch.eye(P, dtype=torch.bool, device=dev)
        near = (gap_d < BODY) & ~eye
        if near.any():
            speeds = _norm(self.velocities) + 1e-6
            share_i = speeds[:, :, None] / (speeds[:, :, None] + speeds[:, None, :])
            is_holder = torch.zeros(B, P, dtype=torch.bool, device=dev)
            is_holder[b[holds], self.holder[holds]] = True
            share_i = torch.where(is_holder[:, :, None], torch.zeros_like(share_i), share_i)
            share_i = torch.where(
                is_holder[:, None, :] & ~is_holder[:, :, None], torch.ones_like(share_i), share_i
            )
            push = (
                apart / gap_d.clamp(min=1e-6)[..., None] * (BODY - gap_d).clamp(min=0.0)[..., None]
            )
            push = torch.where(near[..., None], push * share_i[..., None], torch.zeros_like(push))
            self.places = self.places + push.sum(2)
        self.places[..., 0] = self.places[..., 0].clamp(-5.0, WIDTH + 5.0)
        self.places[..., 1] = self.places[..., 1].clamp(-5.0, LENGTH + 5.0)

    def step(self, wanted, throwing, target, curve, height, tilt):
        """game.step for every point: `wanted` (B, 14, 2) velocities on the field; where
        `throwing` (B,), the holder throws to `target` (field) with the shape given. Points
        that end are recorded in `done` and start again."""
        B, dev = self.B, self.device
        torch.arange(B, device=dev)
        t = self.step_count
        idx = torch.nonzero(throwing & (self.holder >= 0))[:, 0]
        self._start_throws(idx, target[idx], curve[idx], height[idx], tilt[idx])
        self._move(wanted)
        self.seconds = self.seconds + STEP
        # Who is within reach of where it comes down, since when
        near = _norm(self.places - self.flight.target[:, None]) <= CATCH_REACH
        self.reach_since = torch.where(
            near & self.air[:, None],
            torch.minimum(self.reach_since, self.seconds[:, None].expand(-1, P)),
            torch.full_like(self.reach_since, float("inf")),
        )
        # Knocked down on its way
        flying = self.air & ~self.flight.pull & (self.seconds < self.flight.lands)
        if flying.any():
            i = torch.nonzero(flying)[:, 0]
            share = self.flight.share(self.seconds[i], i)
            place, height_now = self.flight.point(share, i)
            defenders = self.team_rows(1 - self.offence[i])
            gaps = _norm(self.places[i[:, None], defenders] - place[:, None])
            nearest = gaps.argmin(-1)
            hit = (
                (share > 0.1)
                & (share < 0.85)
                & (height_now <= JUMP_REACH)
                & (gaps.min(-1).values <= PATH_REACH)
            )
            if hit.any():
                j = i[hit]
                blocker = defenders[hit, nearest[hit]]
                caught = self._rand(len(j)) < CAUGHT_ON_THE_WAY
                holder = torch.where(caught, blocker, -1)
                where = torch.where(caught[:, None], self.places[j, blocker], place[hit])
                self._turnover(j, where, holder, self.flight.thrower[j], blocker, kind="block")
            # Further off, each defender near its path may get a fingertip to it
            low = (share > 0.1) & (share < 0.85) & (height_now <= JUMP_REACH) & ~hit
            tippers = (gaps > PATH_REACH) & (gaps <= TIP_REACH)
            count = tippers.sum(-1).float()
            chance = 1.0 - (1.0 - TIP_CHANCE) ** count
            tipped = low & (count > 0) & (self._rand(len(i)) < chance)
            if tipped.any():
                j = i[tipped]
                masked = torch.where(tippers, gaps, torch.full_like(gaps, float("inf")))
                tipper = defenders[tipped, masked[tipped].argmin(-1)]
                minus = torch.full_like(tipper, -1)
                self._turnover(
                    j, place[tipped], minus, self.flight.thrower[j], tipper, kind="block"
                )
        # Coming down
        landing = self.air & (self.seconds >= self.flight.lands)
        if landing.any():
            self._land(torch.nonzero(landing)[:, 0])
        # The stall (not in a step the disc came down in, as game.step)
        holding = (self.holder >= 0) & ~self.air & (self.scored < 0) & ~landing
        if holding.any():
            i = torch.nonzero(holding)[:, 0]
            defenders = self.team_rows(1 - self.offence[i])
            marked = (
                _norm(self.places[i[:, None], defenders] - self.places[i, self.holder[i]][:, None])
                < STALL_REACH
            ).any(-1)
            self.stall[i] = self.stall[i] + torch.where(marked, STEP, 0.0)
            counting = i[marked]
            self.stall_counted[t, counting] = 1 - self.offence[counting]
            self.rewards[t, counting, 0] += (
                torch.where(self.offence[counting] == 1, 1.0, 0.0) * self.stall_reward * STEP
            )
            self.rewards[t, counting, 1] += (
                torch.where(self.offence[counting] == 0, 1.0, 0.0) * self.stall_reward * STEP
            )
            over = self.stall[i] >= STALL
            if over.any():
                j = i[over]
                h = self.holder[j]
                self._turnover(
                    j,
                    self.places[j, h],
                    torch.full_like(h, -1),
                    h,
                    torch.full_like(h, -1),
                    stalled=True,
                    kind="stall",
                )
        # Picking up a disc on the ground
        lying = self.loose & ~self.air & (self.holder < 0) & ~landing & (self.scored < 0)
        if lying.any():
            i = torch.nonzero(lying)[:, 0]
            players = self.team_rows(self.offence[i])
            gaps = _norm(self.places[i[:, None], players] - self.disc[i][:, None])
            got = gaps.min(-1).values <= PICK_UP_REACH
            if got.any():
                j = i[got]
                who = players[got, gaps[got].argmin(-1)]
                self.holder[j] = who
                self.loose[j] = False
                self.velocities[j, who] = 0.0
                self.held_since[j] = self.seconds[j]
                self.stall[j] = 0.0
        # A possession's match-ups, as the offence first holds the disc (game.match_up)
        unmatched = (self.holder >= 0) & (self.pairs[:, 0] < 0)
        if unmatched.any():
            self._match_up(torch.nonzero(unmatched)[:, 0])
        # Snapshots for what is seen late
        self.past_places = torch.cat([self.places[:, None], self.past_places[:, :-1]], 1)
        self.past_velocities = torch.cat(
            [self.velocities[:, None], self.past_velocities[:, :-1]], 1
        )
        # Points over: start again
        over = (self.scored >= 0) | (self.seconds >= self.point_seconds - 1e-6)
        self.done[t] = over
        self.scores_this_step = torch.where(over, self.scored, -2)
        self.step_count += 1
        self.reset(over)

    def _match_up(self, idx):
        """game.match_up for the points `idx`: the nearest attacker and defender first."""
        n = len(idx)
        r = torch.arange(n, device=self.device)
        attackers = self.team_rows(self.offence[idx])  # (n, 7)
        defenders = self.team_rows(1 - self.offence[idx])
        gaps = _norm(
            self.places[idx[:, None], attackers][:, :, None]
            - self.places[idx[:, None], defenders][:, None]
        )  # (n, 7, 7)
        for _ in range(PLAYERS):
            flat = gaps.reshape(n, -1).argmin(-1)
            a, d = flat // PLAYERS, flat % PLAYERS
            ga, gd = attackers[r, a], defenders[r, d]
            self.pairs[idx, ga], self.pairs[idx, gd] = gd, ga
            gaps[r, a, :] = float("inf")
            gaps[r, :, d] = float("inf")

    def _land(self, idx):
        """game._land for the points `idx`."""
        target = self.flight.target[idx]
        distance = _norm(self.places[idx] - target[:, None])  # (n, 14)
        pull = self.flight.pull[idx]
        if pull.any():
            i, d = idx[pull], distance[pull]
            receiving = self.team_rows(self.offence[i])
            dr = d.gather(1, receiving)
            k = dr.argmin(-1)
            who = receiving[torch.arange(len(i), device=self.device), k]
            caught = (dr.min(-1).values <= CATCH_REACH) & (self._rand(len(i)) >= DROPPED)
            self.air[i] = False
            j = i[caught]
            self.holder[j] = who[caught]
            self.velocities[j, who[caught]] = 0.0
            self.held_since[j] = self.seconds[j]
            self.stall[j] = 0.0
            m = i[~caught]
            place = target[pull][~caught]
            self.disc[m] = torch.stack(
                [place[:, 0].clamp(0.0, WIDTH), place[:, 1].clamp(0.0, LENGTH)], -1
            )
            self.loose[m] = True
        rest = ~pull
        if not rest.any():
            return
        i, d = idx[rest], distance[rest]
        n = len(i)
        rows = torch.arange(n, device=self.device)
        offence = self.team_rows(self.offence[i])
        defence = self.team_rows(1 - self.offence[i])
        d_off_all = d.gather(1, offence)
        d_def_all = d.gather(1, defence)
        near_off = offence[rows, d_off_all.argmin(-1)]
        near_def = defence[rows, d_def_all.argmin(-1)]
        d_off, d_def = d_off_all.min(-1).values, d_def_all.min(-1).values
        dropped = DROPPED + BLADE_DROPPED * self.flight.tilt[i]
        since_def = self.reach_since[i, near_def]
        since_off = self.reach_since[i, near_off]
        now = self.seconds[i]
        position = torch.where(torch.isfinite(since_def), since_def, now) - torch.where(
            torch.isfinite(since_off), since_off, now
        )
        position = position.clamp(-POSITION_MOST, POSITION_MOST)
        hang = ((self.flight.lands[i] - self.flight.released[i] - LONG_FLIGHT) / HANGING).clamp(
            0.0, 1.0
        )
        measure = (d_def - d_off) / TIGHT_WINDOW + position / POSITION_SECONDS
        crowd = ((d_def_all <= DEFENDER_REACH).sum(-1) - 1).clamp(min=0).float()
        measure = measure - CROWD_WEIGHT * crowd  # Every other defender there (game._land)
        chance = 0.5 + 0.5 * torch.tanh(measure - HANG_WEIGHT * hang)
        contested = (1.0 - dropped) * chance
        draw = self._rand(n)
        caught = torch.where(d_def > DEFENDER_REACH, draw >= dropped, draw < contested)
        there = d_off <= CATCH_REACH
        thrower = self.flight.thrower[i]
        minus = torch.full_like(thrower, -1)
        # Caught
        c = there & caught
        # Not caught, a defender there to catch it or knock it down, or dropped
        intercepted = (there & ~caught & (d_def <= CATCH_REACH)) | (~there & (d_def <= CATCH_REACH))
        knocked = there & ~caught & (d_def > CATCH_REACH) & (d_def <= DEFENDER_REACH)
        dropped_r = there & ~caught & (d_def > DEFENDER_REACH)
        nobody = ~there & (d_def > CATCH_REACH)
        if intercepted.any():
            s = intercepted
            self._turnover(
                i[s],
                self.places[i[s], near_def[s]],
                near_def[s],
                thrower[s],
                near_def[s],
                kind="block",
            )
        if knocked.any():
            s = knocked
            self._turnover(i[s], target[rest][s], minus[s], thrower[s], near_def[s], kind="block")
        if dropped_r.any():
            s = dropped_r
            self._turnover(i[s], target[rest][s], minus[s], near_off[s], minus[s], kind="drop")
        if nobody.any():
            s = nobody
            self._turnover(
                i[s], target[rest][s], minus[s], thrower[s], minus[s], kind="nobody there"
            )
        if c.any():
            self._catch(i[c], near_off[c])


def sf_known():
    from .learned import KNOWN_UNSURE

    return KNOWN_UNSURE


def critic_inputs(game: BatchGame, team: int) -> torch.Tensor:
    """ppo.critic_inputs for every point: (B, 7, CRITIC_INPUTS) for `team`'s players."""
    B, dev = game.B, game.device
    attack = 1.0 if team == 0 else -1.0
    attack_b = torch.full((B, P), attack, device=dev)
    places = _turn(game.places, attack_b)
    going = game.velocities * attack
    order = (
        torch.cat([torch.arange(PLAYERS, 2 * PLAYERS), torch.arange(PLAYERS)])
        if team == 1
        else torch.arange(P)
    )
    order = order.to(dev)
    scale = torch.tensor(SIZE, device=dev)
    players = torch.cat([places[:, order] / scale, going[:, order] / RUN_SPEED], -1).reshape(B, -1)
    disc = torch.zeros(B, 10, device=dev)
    b = torch.arange(B, device=dev)
    holds = game.holder >= 0
    disc[holds, 0] = 1.0
    disc[holds, 1:3] = places[b[holds], game.holder[holds]] / scale
    air = game.air
    attack_1 = torch.full((B,), attack, device=dev)
    lands = _turn(game.flight.target, attack_1)
    disc[air, 3] = 1.0
    disc[air, 4:6] = lands[air] / scale
    disc[air, 6] = (game.flight.lands[air] - game.seconds[air]) / 3.0
    ground = game.loose & ~air & ~holds
    disc[ground, 7] = 1.0
    disc[ground, 8:10] = _turn(game.disc, attack_1)[ground] / scale
    held = torch.where(holds, game.seconds - game.held_since, 0.0)
    shared = torch.cat(
        [
            players,
            disc,
            game.stall[:, None] / 10.0,
            held[:, None] / 5.0,
            game.wind * attack / 10.0,
            (game.offence == team).float()[:, None],
        ],
        -1,
    )
    eye = torch.eye(PLAYERS, device=dev)
    found = torch.cat([shared[:, None].expand(-1, PLAYERS, -1), eye.expand(B, -1, -1)], -1)
    assert found.shape[-1] == CRITIC_INPUTS
    return found


THROW_PRIOR_T = None


def throw_prior(device):
    global THROW_PRIOR_T
    if THROW_PRIOR_T is None or THROW_PRIOR_T.device.type != torch.device(device).type:
        THROW_PRIOR_T = torch.as_tensor(THROW_PRIOR, device=device)
    return THROW_PRIOR_T


# A defence by rules, to learn to attack against (BatchPlayers' `marking`), as the
# scripted teams defend (scripted.ScriptedTeam2): each defender marks the attacker of
# their own number where the attacker will be MARK_AHEAD seconds on, GOAL_SIDE towards
# their own end and FORCE_SIDE across; the defender of whoever is deepest DEEP_SIDE
# further back; the holder's marker MARK_GAP from them, between them and the end they
# attack, to the force side
MARK_GAP, MARK_AHEAD = 1.5, 0.3
GOAL_SIDE, FORCE_SIDE, DEEP_SIDE = 1.5, 1.0, 4.0
FORCE = (0.6, -1.0)  # As the defence sees the field: across, and towards the end attacked
# An offence by rules, for the defence to learn to mark (BatchPlayers' `spread`), spread as
# the recorded teams are: a stack zig-zagging STACK_ACROSS either side of the middle,
# STACK_GAP apart from STACK_FROM ahead of the holder; a dump DUMP behind and across; one
# cutter at a time, every CUT_EVERY seconds, a jab of JAB seconds the other way and then
# to CUT_TO ahead of the holder; the holder throws, from HOLD_FOR on, to the teammate
# most clear of defenders (OPEN_BY) within THROW_WITHIN, leading them; late in the
# stall (LATE) to the most open whoever
STACK_ACROSS, STACK_GAP, STACK_FROM = 6.0, 9.0, 16.0
DUMP = (6.0, -6.0)
CUT_EVERY, JAB, CUT_TO = 3.0, 0.5, (13.0, 8.0)
HOLD_FOR, OPEN_BY, THROW_WITHIN, LATE = 1.5, 3.0, 30.0, 7.0


class BatchPlayers:
    """Both teams of every point, played by one set of spots.SpotBrains (it plays itself):
    each player's choice of spot, the holder's of a throw, as spots.SpotTeam makes them,
    for every point at once. What was chosen is kept for learning: the view it was chosen
    from (spot_features.VIEW_FIELDS), the choice, its chance, the value.

    In a share `marking` of the points one team (drawn each point) defends by rules
    instead, man to man (MARK_GAP): the other team's offence learns to beat a defender
    first. That team's networks still play its offence. In a share `league` of the
    others one team is played by an earlier team (`rivals`, set with set_rivals: frozen,
    its choices not kept), so that the team does not only learn to beat itself."""

    def __init__(
        self,
        game: BatchGame,
        brains,
        marking: float = 0.0,
        league: float = 0.0,
        spread: float = 0.0,
    ):
        self.game, self.brains, self.marking, self.league = game, brains, marking, league
        self.spread = spread
        B, dev = game.B, game.device
        self.rivals: list = []
        self.marker = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.passer = torch.full((B,), -1, dtype=torch.long, device=dev)  # Offence by rules
        self.rival = torch.full((B,), -1, dtype=torch.long, device=dev)  # Which rival
        self.rival_team = torch.full((B,), -1, dtype=torch.long, device=dev)
        self._draw_markers(torch.ones(B, dtype=torch.bool, device=dev))
        # Where each team's players saw the opponents in the last steps, oldest first
        self.before = ([], [])
        self.kind = torch.full((B, P), -1, dtype=torch.long, device=dev)
        self.player = torch.full((B, P), -1, dtype=torch.long, device=dev)
        self.param = torch.zeros(B, P, 2, device=dev)
        self.decided_at = torch.full((B, P), -float("inf"), device=dev)
        self.key = torch.full((B, P, 3), -9, dtype=torch.long, device=dev)
        self.thrown_at = torch.full((B, 2), -float("inf"), device=dev)
        self.kept = []

    def set_rivals(self, rivals: list) -> None:
        """The earlier teams (spots.SpotBrains on the game's device) points may be played
        against from the next new point on."""
        self.rivals = list(rivals)
        self.rival[:] = -1
        self.rival_team[:] = -1

    def _draw_markers(self, mask):
        """Which team defends by rules, and which is an earlier team, in the points where
        `mask` (-1: neither)."""
        n = int(mask.sum())
        if not n:
            return
        dev = self.game.device
        draw = torch.rand(n, generator=self.game.gen, device=dev)
        ruled = draw < self.marking
        team = torch.randint(0, 2, (n,), generator=self.game.gen, device=dev)
        none = torch.full_like(team, -1)
        self.marker[mask] = torch.where(ruled, team, none)
        # An offence by rules: in a share `spread` of the points
        passing = ~ruled & (draw < self.marking + self.spread)
        self.passer[mask] = torch.where(passing, team, none)
        # Against an earlier team: in a share `league` of the points not played by rules
        taken = self.marking + self.spread
        rivalled = ~ruled & ~passing & (draw < taken + self.league) & bool(self.rivals)
        which = torch.randint(
            0, max(len(self.rivals), 1), (n,), generator=self.game.gen, device=dev
        )
        self.rival_team[mask] = torch.where(rivalled, 1 - team, none)
        self.rival[mask] = torch.where(rivalled, which, none)

    def _passing(self, v, disc_at):
        """An offence by rules (STACK_GAP...), as the team sees the field, set up around
        `disc_at` (B, 2: the holder, or where the disc lies or comes down): each player's
        velocity (B, 7, 2), where the holder throws (B, 2) and whether they throw now."""
        dev = self.game.device
        B = v["held"].shape[0]
        b = torch.arange(B, device=dev)
        ks = torch.arange(PLAYERS, device=dev)
        rows, them = v["ours"][:, 0], v["theirs"][:, 0]
        place, going = rows[..., :2], rows[..., 2:4]
        h = v["holder"][:, 0].clamp(min=0)
        at = disc_at
        role = (ks[None] - h[:, None]) % PLAYERS  # 0 the holder, 1 the dump
        r = (role - 2).float()
        across = WIDTH / 2 + STACK_ACROSS * torch.where(role % 2 == 0, -1.0, 1.0)
        along = (at[:, 1:2] + STACK_FROM + STACK_GAP * r).clamp(2.0, LENGTH - 2.0)
        spot = torch.stack([across.expand(B, -1), along], -1)
        side = torch.where(at[:, 0] < WIDTH / 2, 1.0, -1.0)
        dump = at + torch.stack([DUMP[0] * side, torch.full_like(side, DUMP[1])], -1)
        dump = torch.stack([dump[:, 0].clamp(1.0, WIDTH - 1.0), dump[:, 1].clamp(1.0, LENGTH)], -1)
        spot = torch.where((role == 1)[..., None], dump[:, None], spot)
        # One cutter at a time: a jab, then the cut
        held = v["held"][:, 0]
        turn = (held // CUT_EVERY).long() % (PLAYERS - 2)
        into = held % CUT_EVERY
        way = torch.where(turn % 2 == 0, -1.0, 1.0)
        cut = at + torch.stack([CUT_TO[0] * way, torch.full_like(way, CUT_TO[1])], -1)
        cut = torch.stack([cut[:, 0].clamp(1.0, WIDTH - 1.0), cut[:, 1].clamp(1.0, LENGTH)], -1)
        cutter = (role == (turn + 2)[:, None]) & (into < CUT_EVERY - 0.5)[:, None]
        away = place - cut[:, None]
        jab = place + 3.0 * away / _norm(away).clamp(min=1e-6)[..., None]
        going_to = torch.where((into < JAB)[:, None, None], jab, cut[:, None])
        spot = torch.where(cutter[..., None], going_to, spot)
        velocity = _towards(place, spot)
        # The throw: to whoever is most clear, forward liked a little more
        clear = _norm(place[:, :, None] - them[:, None, :, :2]).min(-1).values
        far = _norm(place - at[:, None])
        able = (role > 0) & (clear >= OPEN_BY) & (far <= THROW_WITHIN)
        score = torch.where(able, clear + 0.2 * (place[..., 1] - at[:, 1:2]), -1e9)
        best = score.argmax(-1)
        fallback = torch.where(role > 0, clear, -1e9).argmax(-1)
        late = v["stall"][:, 0] >= LATE
        to = torch.where(able.any(-1), best, fallback)
        now = (held >= HOLD_FOR) & (able.any(-1) | late)
        flight = RELEASE_SECONDS + far[b, to] / DISC_YARDS_PER_SECOND
        lead = place[b, to] + going[b, to] * flight.clamp(max=1.5)[:, None]
        lead = torch.stack(
            [lead[:, 0].clamp(1.0, WIDTH - 1.0), lead[:, 1].clamp(1.0, LENGTH - 1.0)], -1
        )
        return velocity, lead, now

    def _rival_scores(self, scores, gi, team, network_of):
        """Scores of the rows (games `gi`) whose team here is an earlier team's, by that
        team's network (network_of(brains, rows)); and which rows those are."""
        rivalled = self.rival_team[gi] == team
        for r in self.rival[gi][rivalled].unique().tolist():
            rows = rivalled & (self.rival[gi] == r)
            scores[rows] = network_of(self.rivals[r], rows)
        return rivalled

    def _marks(self, v, disc_spot):
        """Where each defender marking by rules stands (B, 7, 2), as the team sees it."""
        dev = self.game.device
        ks = torch.arange(PLAYERS, device=dev)
        attackers = v["theirs"][:, 0]  # (B, 7, 4): the same for every player of the team
        place, going = attackers[..., :2], attackers[..., 2:4]
        # The offence attacks the near end, as the defence sees it: goal side is -y
        deepest = place[..., 1].argmin(-1, keepdim=True)
        back = GOAL_SIDE + DEEP_SIDE * (ks[None] == deepest).float()
        side = torch.full_like(back, FORCE_SIDE)
        marks = place + MARK_AHEAD * going + torch.stack([side, -back], -1)
        force = torch.tensor(FORCE, device=dev)
        on_holder = place + MARK_GAP * force / _norm(force)
        holds = (ks[None] == v["holder"][:, :1]) & (v["disc_mode"][:, :1] == sf.HELD)
        return torch.where(holds[..., None], on_holder, marks)

    def _spots(self, v, rows):
        """(B, 7, 2): where the spot each player of a team chose last is now, as the team
        sees the field (a spot beside a player follows them, an opponent a moment late); their own place without
        one."""
        ks = torch.arange(PLAYERS, device=self.game.device)
        place = v["ours"][:, ks, ks, :2]
        kind = self.kind[:, rows]
        who = self.player[:, rows].clamp(min=0)
        at = who[..., None, None].expand(-1, -1, 1, 2)
        # An opponent is followed REACT_STEPS late (as they are now in a point just begun)
        theirs = v["theirs"][..., :2]
        before = self.before[rows.start // PLAYERS]
        if len(before) > REACT_STEPS:
            theirs = torch.where(before[0].isnan(), theirs, before[0])
        opponent = theirs.gather(2, at)[:, :, 0]
        mate = v["ours"][..., :2].gather(2, at)[:, :, 0]
        param = self.param[:, rows]
        return torch.where(
            (kind == sf.OPPONENT)[..., None],
            opponent + param,
            torch.where(
                (kind == sf.TEAMMATE)[..., None],
                mate + param,
                torch.where((kind == sf.FIELD)[..., None], param, place),
            ),
        )

    def forget(self, mask):
        """New points where `mask`: no plans."""
        self._draw_markers(mask)
        for before in self.before:
            for seen in before:
                seen[mask] = float("nan")
        self.kind[mask] = -1
        self.decided_at[mask] = -float("inf")
        self.key[mask] = -9
        self.thrown_at[mask] = -float("inf")

    def _draw(self, scores):
        chances = torch.softmax(scores, -1)
        chosen = torch.multinomial(chances, 1, generator=self.game.gen)[:, 0]
        return chosen, torch.log(chances.gather(1, chosen[:, None])[:, 0].clamp(min=1e-12))

    def decide(self, record=True):
        """Every player's velocity (B, 14, 2) on the field, and the throws: (throwing (B,),
        target (B, 2) on the field, curve, height, tilt)."""
        game = self.game
        B, dev = game.B, game.device
        b = torch.arange(B, device=dev)
        wanted = torch.zeros(B, P, 2, device=dev)
        throwing = torch.zeros(B, dtype=torch.bool, device=dev)
        target = torch.zeros(B, 2, device=dev)
        curve = torch.zeros(B, device=dev)
        height = torch.full((B,), float("nan"), device=dev)
        tilt = torch.full((B,), float("nan"), device=dev)
        ks = torch.arange(PLAYERS, device=dev)
        for team in (0, 1):
            first = team * PLAYERS
            rows = slice(first, first + PLAYERS)
            v = game.look(team)
            self.before[team].append(v["theirs"][..., :2].clone())
            del self.before[team][: -(REACT_STEPS + 1)]
            v["plan_kind"], v["plan_player"], v["plan_param"] = (
                self.kind[:, rows],
                self.player[:, rows],
                self.param[:, rows],
            )
            age = (game.seconds[:, None] - self.decided_at[:, rows]).clamp(0.0, 5.0)
            v["plan_age"] = torch.where(self.kind[:, rows] >= 0, age, torch.zeros_like(age))
            attack = 1.0 if team == 0 else -1.0
            attacking = game.offence == team
            holder_here = attacking & (game.holder >= first) & (game.holder < first + PLAYERS)
            is_holder = torch.zeros(B, PLAYERS, dtype=torch.bool, device=dev)
            is_holder[b[holder_here], game.holder[holder_here] - first] = True
            going, reflex_velocity = game.reflexes(v)
            reflex_on = ((v["disc_mode"] == sf.IN_AIR) | (v["disc_mode"] == sf.ON_GROUND)) & going
            key = torch.stack(
                [
                    v["holder"],
                    (v["disc_mode"] == sf.IN_AIR).long(),
                    ((v["disc_mode"] == sf.HELD) | (v["disc_mode"] == sf.NO_DISC)).long(),
                ],
                -1,
            )
            # An attacker on their way chooses again every COMMIT seconds only (they may
            # break the run off); a defender chooses again every time
            elapsed = game.seconds[:, None] - self.decided_at[:, rows]
            there = _norm(self._spots(v, rows) - v["ours"][:, ks, ks, :2]) <= sf.REACHED
            run_out = ~attacking[:, None] | there | (elapsed >= COMMIT - 1e-9)
            due = (elapsed >= DECIDE_EVERY - 1e-9) & run_out
            changed = (key != self.key[:, rows]).any(-1)
            ruled = ((self.marker == team) & ~attacking)[:, None] & ~reflex_on
            passing = (self.passer == team) & attacking
            ruled_attack = passing[:, None] & ~reflex_on
            deciding = ~is_holder & ~reflex_on & ~ruled & ~ruled_attack
            deciding &= due | changed | (self.kind[:, rows] < 0)
            if deciding.any():
                gi, ki = torch.nonzero(deciding, as_tuple=True)
                view = {name: v[name][gi, ki] for name in sf.VIEW_FIELDS}
                with torch.no_grad():
                    parts = sf.spot_parts(view)
                    kinds, players, params = parts["kinds"], parts["players"], parts["params"]
                    allowed = parts["allowed"]
                    offence = view["attacking"]

                    def spots_by(brains, rows):
                        found = torch.empty(int(rows.sum()), sf.SPOT_CHOICES, device=dev)
                        on = offence[rows]
                        for side, of in (("offence", on), ("defence", ~on)):
                            if of.any():
                                some = sf.take_parts(sf.take_parts(parts, rows), of)
                                found[of] = sf.spot_scores(brains.spots[side], some)
                        return found

                    scores = spots_by(self.brains, torch.ones_like(offence))
                    rival_spots = spots_by

                    rivalled = self._rival_scores(scores, gi, team, rival_spots)
                    chosen, log_chance = self._draw(scores + sf.spot_leans(view, allowed))
                    seen = critic_inputs(game, team)[gi, ki]
                    value = torch.empty(len(gi), device=dev)
                    if offence.any():
                        value[offence] = self.brains.values["offence"](seen[offence])[:, 0]
                    if (~offence).any():
                        value[~offence] = self.brains.values["defence"](seen[~offence])[:, 0]
                r = torch.arange(len(gi), device=dev)
                self.kind[gi, first + ki] = kinds[r, chosen]
                self.player[gi, first + ki] = players[r, chosen]
                self.param[gi, first + ki] = params[r, chosen]
                jitter = (
                    0.1 * torch.randint(-2, 3, (len(gi),), generator=game.gen, device=dev).float()
                )
                self.decided_at[gi, first + ki] = game.seconds[gi] + jitter
                self.key[gi, first + ki] = key[gi, ki]
                if record and (~rivalled).any():
                    own = ~rivalled  # An earlier team's choices are not learned from
                    self.kept.append(
                        {
                            "view": {name: view[name][own] for name in view},
                            "chosen": chosen[own],
                            "log_chance": log_chance[own],
                            "value": value[own],
                            "seen": seen[own],
                            "game": gi[own],
                            "player": (first + ki)[own],
                            "step": torch.full_like(gi[own], game.step_count),
                            "kind": torch.where(offence[own], 0, 1),  # Offence, defence spots
                        }
                    )
            # Where each goes: the reflex, or the spot chosen (as it is now)
            place = v["ours"][:, ks, ks, :2]
            holder_rows = torch.where(v["attacking"][..., None, None], v["ours"], v["theirs"])[
                ..., :2
            ]
            disc_spot = torch.where(
                ((v["disc_mode"] == sf.IN_AIR) | (v["disc_mode"] == sf.ON_GROUND))[..., None],
                v["disc"],
                torch.where(
                    (v["disc_mode"] == sf.HELD)[..., None],
                    holder_rows.gather(
                        2, v["holder"].clamp(min=0)[..., None, None].expand(-1, -1, 1, 2)
                    )[:, :, 0],
                    place,
                ),
            )
            spot = self._spots(v, rows)
            velocity = torch.where(reflex_on[..., None], reflex_velocity, _towards(place, spot))
            if ruled.any():
                marks = _towards(place, self._marks(v, disc_spot[:, :1].expand(-1, PLAYERS, -1)))
                velocity = torch.where(ruled[..., None], marks, velocity)
            rule_throw = torch.zeros(B, dtype=torch.bool, device=dev)
            if ruled_attack.any():
                moves, rule_aim, rule_now = self._passing(v, disc_spot[:, 0])
                velocity = torch.where(ruled_attack[..., None], moves, velocity)
                rule_throw = passing & holder_here & rule_now
            velocity = torch.where(is_holder[..., None], torch.zeros_like(velocity), velocity)
            wanted[:, rows] = velocity * attack
            # The holder: hold on, or throw where
            if rule_throw.any():
                gt = torch.nonzero(rule_throw)[:, 0]
                kt = game.holder[gt] - first
                holder_view = {name: v[name][gt, kt] for name in sf.VIEW_FIELDS}
                aim = rule_aim[gt]
                c, h, t = game.throw_shape(holder_view, aim)
                throwing[gt] = True
                target[gt] = _turn(aim, torch.full((len(gt),), attack, device=dev))
                curve[gt], height[gt], tilt[gt] = c, h, t
            ready = (
                holder_here
                & ~passing
                & (v["held"][:, 0] >= SHORTEST_HOLD)
                & (game.seconds - self.thrown_at[:, team] >= THROW_EVERY)
            )
            if ready.any():
                gi = torch.nonzero(ready)[:, 0]
                ki = game.holder[gi] - first
                self.thrown_at[gi, team] = game.seconds[gi]
                view = {name: v[name][gi, ki] for name in sf.VIEW_FIELDS}
                with torch.no_grad():
                    features = sf.throw_choices(view)
                    scores = self.brains.throws(features)[..., 0] + throw_prior(dev)
                    rivalled = self._rival_scores(
                        scores,
                        gi,
                        team,
                        lambda brains, rows: brains.throws(features[rows])[..., 0]
                        + throw_prior(dev),
                    )
                    chosen, log_chance = self._draw(scores)
                    seen = critic_inputs(game, team)[gi, ki]
                    value = self.brains.values["offence"](seen)[:, 0]
                if record and (~rivalled).any():
                    own = ~rivalled
                    self.kept.append(
                        {
                            "view": {name: view[name][own] for name in view},
                            "chosen": chosen[own],
                            "log_chance": log_chance[own],
                            "value": value[own],
                            "seen": seen[own],
                            "game": gi[own],
                            "player": (first + ki)[own],
                            "step": torch.full_like(gi[own], game.step_count),
                            "kind": torch.full_like(gi[own], 2),  # A throw
                        }
                    )
                throws = chosen > 0
                if throws.any():
                    gt = gi[throws]
                    holder_view = {name: view[name][throws] for name in sf.VIEW_FIELDS}
                    aim = sf.throw_aims(holder_view)[
                        torch.arange(len(gt), device=dev), chosen[throws] - 1
                    ]
                    c, h, t = game.throw_shape(holder_view, aim)
                    throwing[gt] = True
                    target[gt] = _turn(aim, torch.full((len(gt),), attack, device=dev))
                    curve[gt], height[gt], tilt[gt] = c * attack * attack, h, t
        return wanted, throwing, target, curve, height, tilt
