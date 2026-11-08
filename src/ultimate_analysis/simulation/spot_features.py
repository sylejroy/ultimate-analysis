"""What a player choosing a spot, or a holder choosing a throw, sees of each choice
(spots.py), for many players at once: one description of the game shared by the game
played on the processor (spots.SpotTeam) and the one played many points at a time
(batch.py), so that the two cannot drift apart.

A player's view is a set of tensors with a first dimension for the players (as a team
sees the game, attacking the far end): see `views_of`.
"""

from typing import Dict, List, Optional

import numpy as np

from .disc import DISC_YARDS_PER_SECOND, RELEASE_SECONDS, SLOWEST, WIND_ALONG
from .game import LENGTH, PLAYERS, RUN_SPEED, WIDTH
from .learned import KNOWN_UNSURE
from .perception import PlayerView

NEAREST = 3  # Players of each team described around a spot
AROUND = 9  # Per player: where (from the spot), velocity, acceleration, facing, known
SPOT_ACROSS = np.arange(2.0, WIDTH, 4.0)  # The grid around the disc: across the field
SPOT_ALONG = np.arange(-15.0, 46.0, 4.0)  # ... and from behind the disc to ahead of it
GRID = np.array([[x, y] for y in SPOT_ALONG for x in SPOT_ACROSS])
# Spots beside a player, as their team attacks: behind, ahead, left, right of them
BESIDE = np.array([[0.0, -1.5], [0.0, 1.5], [-1.5, 0.0], [1.5, 0.0]])
THROW_ACROSS = np.arange(1.5, WIDTH, 3.0)  # Throw targets over the whole field
THROW_ALONG = np.arange(1.5, LENGTH, 3.0)
THROW_TARGETS = np.array([[x, y] for y in THROW_ALONG for x in THROW_ACROSS])
FIELD, TEAMMATE, OPPONENT, DISC = range(4)  # Kinds of spot
SPOT_CHOICES = len(GRID) + (2 * PLAYERS - 1) * len(BESIDE) + 1
THROW_CHOICES = 1 + len(THROW_TARGETS)  # Holding on, then the targets
SPOT_FEATURES = 2 + 3 + 4 + 1 + 2 * NEAREST * AROUND + 9 + 2 + 2 + PLAYERS + 1 + 2
THROW_FEATURES = 1 + 2 + 3 + 3 + 2 * NEAREST * AROUND + 2
NO_DISC, HELD, IN_AIR, ON_GROUND = range(4)  # What the disc is doing, in a view
# What a spot's or a throw's numbers are, by kind, in order (for drawing the networks)
SPOT_GROUPS = (
    ("Where the spot is", 5),
    ("What kind of spot", 4),
    ("Same as last choice", 1),
    ("3 teammates nearest it", NEAREST * AROUND),
    ("3 opponents nearest it", NEAREST * AROUND),
    ("The disc, from the spot", 9),
    ("Wind, stall, role, speed", 2 + 2 + PLAYERS + 1 + 2),
)
THROW_GROUPS = (
    ("Hold on instead", 1),
    ("Where the target is", 5),
    ("Flight time, wind", 3),
    ("3 teammates nearest it", NEAREST * AROUND),
    ("3 opponents nearest it", NEAREST * AROUND),
    ("Stall, time held", 2),
)

# A view: name -> (shape per player, kind)
VIEW_FIELDS = {
    "ours": ((PLAYERS, 4), "f"),  # x, y, vx, vy as believed (their own row as it is)
    "theirs": ((PLAYERS, 4), "f"),
    "ours_known": ((PLAYERS,), "b"),
    "theirs_known": ((PLAYERS,), "b"),
    "ours_accel": ((PLAYERS, 2), "f"),
    "theirs_accel": ((PLAYERS, 2), "f"),
    "ours_facing": ((PLAYERS, 2), "f"),
    "theirs_facing": ((PLAYERS, 2), "f"),
    "me": ((), "i"),
    "holder": ((), "i"),  # Among ours (attacking) or theirs; -1 for none
    "attacking": ((), "b"),
    "disc_mode": ((), "i"),  # NO_DISC, HELD, IN_AIR, ON_GROUND
    "disc_now": ((2,), "f"),  # In the air: where it is
    "disc": ((2,), "f"),  # In the air: where it comes down (as judged); on the ground: where
    "lands_in": ((), "f"),
    "height": ((), "f"),
    "wind": ((2,), "f"),
    "stall": ((), "f"),
    "held": ((), "f"),
    "plan_kind": ((), "i"),  # The spot chosen last (-1: none)
    "plan_player": ((), "i"),
    "plan_param": ((2,), "f"),
}


def views_of(views: List[PlayerView], plans: Optional[List[Optional[tuple]]] = None, device="cpu"):
    """The tensors of a list of PlayerView (spots.py's plans: (kind, player, param) or
    None per view)."""
    import torch

    found = {name: [] for name in VIEW_FIELDS}
    for i, view in enumerate(views):
        found["ours"].append(view.ours)
        found["theirs"].append(view.theirs)
        found["ours_known"].append(view.ours_unsure < KNOWN_UNSURE)
        found["theirs_known"].append(view.theirs_unsure < KNOWN_UNSURE)
        zeros = np.zeros((PLAYERS, 2))
        found["ours_accel"].append(zeros if view.ours_accel is None else view.ours_accel)
        found["theirs_accel"].append(zeros if view.theirs_accel is None else view.theirs_accel)
        found["ours_facing"].append(zeros if view.ours_facing is None else view.ours_facing)
        found["theirs_facing"].append(zeros if view.theirs_facing is None else view.theirs_facing)
        found["me"].append(view.me)
        found["holder"].append(-1 if view.holder is None else view.holder)
        found["attacking"].append(view.attacking)
        if view.holder is not None:
            mode = HELD
        elif view.lands_in is not None and view.disc_now is not None:
            mode = IN_AIR
        elif view.disc is not None:
            mode = ON_GROUND
        else:
            mode = NO_DISC
        found["disc_mode"].append(mode)
        found["disc_now"].append(np.zeros(2) if view.disc_now is None else view.disc_now)
        found["disc"].append(np.zeros(2) if view.disc is None else view.disc)
        found["lands_in"].append(0.0 if view.lands_in is None else view.lands_in)
        found["height"].append(view.disc_height)
        found["wind"].append(np.zeros(2) if view.wind is None else view.wind)
        found["stall"].append(view.stall)
        found["held"].append(view.held)
        plan = None if plans is None else plans[i]
        found["plan_kind"].append(-1 if plan is None else plan[0])
        found["plan_player"].append(-1 if plan is None else plan[1])
        found["plan_param"].append(
            np.zeros(2) if plan is None or plan[2] is None else np.asarray(plan[2])
        )
    kinds = {"f": torch.float32, "b": torch.bool, "i": torch.long}
    return {
        name: torch.as_tensor(np.array(found[name]), dtype=kinds[kind], device=device)
        for name, (_, kind) in VIEW_FIELDS.items()
    }


def _rows(v, side):
    return v["ours"] if side == "ours" else v["theirs"]


def disc_place(v):
    """(B, 2): where the disc is, for choosing spots: the holder; where it comes down or
    lies; else the player."""
    import torch

    b = torch.arange(len(v["me"]), device=v["me"].device)
    me = v["ours"][b, v["me"], :2]
    holder_rows = torch.where(v["attacking"][:, None, None], v["ours"], v["theirs"])
    holder = holder_rows[b, v["holder"].clamp(min=0), :2]
    mode = v["disc_mode"][:, None]
    return torch.where(
        mode == HELD, holder, torch.where((mode == IN_AIR) | (mode == ON_GROUND), v["disc"], me)
    )


def around(spots, rows, known, accel, facing, skip=None):
    """(B, n, NEAREST * AROUND): the players of one team nearest each spot, as known."""
    import torch

    offsets = rows[:, None, :, :2] - spots[:, :, None, :]  # (B, n, 7, 2)
    distance = torch.linalg.norm(offsets, dim=-1)
    valid = known.clone()
    if skip is not None:
        valid[torch.arange(len(skip), device=skip.device), skip.clamp(min=0)] &= skip < 0
    distance = torch.where(valid[:, None, :], distance, torch.full_like(distance, float("inf")))
    nearest, order = torch.topk(distance, NEAREST, dim=2, largest=False)  # (B, n, 3)
    present = torch.isfinite(nearest)[..., None].float()

    def pick(values):  # (B, 7, d) -> (B, n, 3, d)
        expanded = values[:, None].expand(-1, spots.shape[1], -1, -1)
        return torch.gather(expanded, 2, order[..., None].expand(-1, -1, -1, values.shape[-1]))

    where = torch.gather(offsets, 2, order[..., None].expand(-1, -1, -1, 2))
    found = torch.cat(
        [
            where / 20.0,
            pick(rows[..., 2:4]) / RUN_SPEED,
            pick(accel) / 7.0,
            pick(facing),
            torch.ones_like(nearest[..., None]),
        ],
        dim=-1,
    )
    return (found * present).reshape(spots.shape[0], spots.shape[1], -1)


def disc_around(v, spots):
    """(B, n, 9): the disc from each spot: in the air, held, on the ground; where it is;
    where it comes down and how soon; how high."""
    import torch

    B, n = spots.shape[:2]
    mode = v["disc_mode"][:, None, None]
    holder = disc_place(v)[:, None]
    held = (mode == HELD).float()
    air = (mode == IN_AIR).float()
    ground = (mode == ON_GROUND).float()
    now = held * (holder - spots) + air * (v["disc_now"][:, None] - spots)
    now = now + ground * (v["disc"][:, None] - spots)
    lands = air * (v["disc"][:, None] - spots)
    return torch.cat(
        [
            air.expand(B, n, 1),
            held.expand(B, n, 1),
            ground.expand(B, n, 1),
            now / 20.0,
            lands / 20.0,
            (air * v["lands_in"][:, None, None] / 3.0).expand(B, n, 1),
            (air * v["height"][:, None, None] / 10.0).expand(B, n, 1),
        ],
        dim=-1,
    )


def spot_choices(v):
    """(features (B, SPOT_CHOICES, SPOT_FEATURES), kinds, players, params (B, SPOT_CHOICES
    [, 2])): every spot each player may choose, and what is around it. A spot beside a
    player follows them; a spot on the grid stays where it is."""
    import torch

    device = v["me"].device
    B = len(v["me"])
    b = torch.arange(B, device=device)
    me = v["ours"][b, v["me"]]
    disc = disc_place(v)
    grid = torch.as_tensor(GRID, dtype=torch.float32, device=device)[None].repeat(B, 1, 1)
    grid[..., 1] = (grid[..., 1] + disc[:, None, 1]).clamp(1.0, LENGTH - 1.0)
    # Beside a teammate (every one but themselves) or a player of the other team
    six = torch.arange(PLAYERS - 1, device=device)[None].expand(B, -1)
    mates = six + (six >= v["me"][:, None]).long()  # (B, 6)
    beside = torch.as_tensor(BESIDE, dtype=torch.float32, device=device)
    mate_spots = v["ours"][b[:, None], mates, :2][:, :, None] + beside  # (B, 6, 4, 2)
    their_spots = v["theirs"][..., :2][:, :, None] + beside  # (B, 7, 4, 2)
    lying = torch.where(
        ((v["disc_mode"] == IN_AIR) | (v["disc_mode"] == ON_GROUND))[:, None], v["disc"], disc
    )
    spots = torch.cat(
        [grid, mate_spots.reshape(B, -1, 2), their_spots.reshape(B, -1, 2), lying[:, None]], dim=1
    )
    n_grid = len(GRID)
    kinds = torch.cat(
        [
            torch.full((B, n_grid), FIELD, device=device),
            torch.full((B, (PLAYERS - 1) * 4), TEAMMATE, device=device),
            torch.full((B, PLAYERS * 4), OPPONENT, device=device),
            torch.full((B, 1), DISC, device=device),
        ],
        dim=1,
    )
    players = torch.cat(
        [
            torch.full((B, n_grid), -1, device=device),
            mates.repeat_interleave(4, dim=1),
            torch.arange(PLAYERS, device=device).repeat_interleave(4)[None].expand(B, -1),
            torch.full((B, 1), -1, device=device),
        ],
        dim=1,
    )
    params = torch.cat(
        [
            grid,
            beside.repeat(PLAYERS - 1, 1)[None].expand(B, -1, -1),
            beside.repeat(PLAYERS, 1)[None].expand(B, -1, -1),
            torch.zeros((B, 1, 2), device=device),
        ],
        dim=1,
    )
    current = (kinds == v["plan_kind"][:, None]) & (players == v["plan_player"][:, None])
    current &= (kinds == DISC) | ((params - v["plan_param"][:, None]).abs().amax(-1) < 1e-6)
    offset = spots - me[:, None, :2]
    role = torch.zeros((B, PLAYERS), device=device)
    has = v["holder"] >= 0
    role[b[has], (v["me"][has] - v["holder"][has]) % PLAYERS] = 1.0
    shared = torch.cat(
        [
            v["wind"] / 10.0,
            v["stall"][:, None] / 10.0,
            v["held"][:, None] / 5.0,
            role,
            v["attacking"][:, None].float(),
            me[:, 2:4] / RUN_SPEED,
        ],
        dim=1,
    )
    scale = torch.tensor([WIDTH, LENGTH], device=device)
    features = torch.cat(
        [
            spots / scale,
            offset / 20.0,
            torch.linalg.norm(offset, dim=-1, keepdim=True) / 30.0,
            torch.nn.functional.one_hot(kinds, 4).float(),
            current[..., None].float(),
            around(spots, v["ours"], v["ours_known"], v["ours_accel"], v["ours_facing"], v["me"]),
            around(spots, v["theirs"], v["theirs_known"], v["theirs_accel"], v["theirs_facing"]),
            disc_around(v, spots),
            shared[:, None].expand(-1, spots.shape[1], -1),
        ],
        dim=-1,
    )
    return features, kinds, players, params


def throw_choices(v):
    """(B, THROW_CHOICES, THROW_FEATURES): holding on, then throwing to each target, and
    what is around each: where it is, how far and which way from the thrower, how long
    the disc would take (the wind with it or against it), the wind along and across, the
    players nearest it; the stall and how long it has been held. For views of holders."""
    import torch

    device = v["me"].device
    B = len(v["me"])
    b = torch.arange(B, device=device)
    thrower = v["ours"][b, v["holder"].clamp(min=0), :2]
    targets = torch.as_tensor(THROW_TARGETS, dtype=torch.float32, device=device)[None].expand(
        B, -1, -1
    )
    way = targets - thrower[:, None]
    length = torch.linalg.norm(way, dim=-1).clamp(min=1e-6)
    unit = way / length[..., None]
    wind = v["wind"][:, None]
    along_wind = (unit * wind).sum(-1)
    across_wind = unit[..., 0] * wind[..., 1] - unit[..., 1] * wind[..., 0]
    speed = (DISC_YARDS_PER_SECOND + WIND_ALONG * along_wind).clamp(min=SLOWEST)
    flying = RELEASE_SECONDS + length / speed
    timing = torch.stack([v["stall"] / 10.0, v["held"] / 5.0], dim=1)
    n = targets.shape[1]
    scale = torch.tensor([WIDTH, LENGTH], device=device)
    throws = torch.cat(
        [
            torch.zeros((B, n, 1), device=device),
            targets / scale,
            way / 20.0,
            length[..., None] / 30.0,
            flying[..., None] / 3.0,
            along_wind[..., None] / 10.0,
            across_wind[..., None] / 10.0,
            around(
                targets, v["ours"], v["ours_known"], v["ours_accel"], v["ours_facing"], v["holder"]
            ),
            around(targets, v["theirs"], v["theirs_known"], v["theirs_accel"], v["theirs_facing"]),
            timing[:, None].expand(-1, n, -1),
        ],
        dim=-1,
    )
    hold = torch.zeros((B, 1, THROW_FEATURES), device=device)
    hold[:, 0, 0] = 1.0
    hold[:, 0, -2:] = timing
    return torch.cat([hold, throws], dim=1)


def stack_views(parts: List[Dict]) -> Dict:
    import torch

    return {name: torch.cat([p[name] for p in parts]) for name in VIEW_FIELDS}


def take(v: Dict, index) -> Dict:
    return {name: value[index] for name, value in v.items()}
