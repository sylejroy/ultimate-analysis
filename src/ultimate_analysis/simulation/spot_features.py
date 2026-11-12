"""What a player choosing where to run, or a holder choosing a throw, sees of each choice
(spots.py), for many players at once: one description of the game shared by the game
played on the processor (spots.SpotTeam) and the one played many points at a time
(batch.py), so that the two cannot drift apart.

A player's view is a set of tensors with a first dimension for the players (as a team
sees the game, attacking the far end): see `views_of`.
"""

from typing import Dict, List, Optional

import numpy as np

from .disc import DISC_YARDS_PER_SECOND, RELEASE_SECONDS, SLOWEST, WIND_ALONG
from .game import END_ZONE, LENGTH, PLAYERS, RUN_SPEED, WIDTH
from .learned import KNOWN_UNSURE
from .perception import PlayerView

# Per player, every one, each team nearest the player choosing first: where from the
# spot, where from the player choosing, velocity, acceleration, facing, whether they hold
# the disc, whether matched with the player choosing (game.match_up: the attacker they
# guard, or the defender guarding them), whether known
PER_PLAYER = 13
MATES, OPPONENTS = PLAYERS - 1, PLAYERS
# Where a player may run: a cut from where they are, each way of CUT_WAYS round them and
# each length (kept EDGE inside the field) ...
CUT_LENGTHS = (5.0, 12.0, 25.0)
CUT_WAYS = 8
EDGE = 1.0
CUTS = np.array(
    [
        [length * np.sin(turn), length * np.cos(turn)]
        for length in CUT_LENGTHS
        for turn in np.arange(CUT_WAYS) * 2.0 * np.pi / CUT_WAYS
    ]
)
# ... places around the disc (a grid, and a ring close to it: a cup, a reset, a zone)
AROUND = np.array(
    [[x, y] for y in np.arange(-24.0, 25.0, 8.0) for x in np.arange(-16.0, 17.0, 8.0)]
    + [
        [4.0 * np.sin(turn), 4.0 * np.cos(turn)]
        for turn in np.arange(CUT_WAYS) * 2.0 * np.pi / CUT_WAYS
    ]
)
# ... and beside any other player, of either team, as the team attacks: behind, ahead,
# left, right of them (a spot that follows them)
BESIDE = np.array([[0.0, -1.5], [0.0, 1.5], [-1.5, 0.0], [1.5, 0.0]])
THROW_ACROSS = np.arange(1.5, WIDTH, 3.0)  # Throw targets over the whole field
THROW_ALONG = np.arange(1.5, LENGTH, 3.0)
THROW_TARGETS = np.array([[x, y] for y in THROW_ALONG for x in THROW_ACROSS])
# ... and a throw to each teammate of the holder, in order, led to where they will be when
# the disc gets there (as scripted.throw_options): at most LEAD_MOST seconds of their
# running, LEAD_MARGIN yards in from the edges
LEAD_MOST, LEAD_MARGIN = 1.5, 1.0
MATE_SHARE = 0.5  # At first: the teammates' share of the throws (the targets the rest)
# Kinds of spot run to: a place on the field; a step from an opponent or a teammate,
# following them
FIELD, OPPONENT, TEAMMATE = range(3)
# The choices, in order: staying; going on to the spot chosen last; the cuts; the
# places around the disc; the spots beside each teammate, then each opponent
STAY, KEEP, FIRST_CUT = 0, 1, 2
FIRST_AROUND = FIRST_CUT + len(CUTS)
FIRST_TEAMMATE = FIRST_AROUND + len(AROUND)
FIRST_OPPONENT = FIRST_TEAMMATE + PLAYERS * len(BESIDE)
SPOT_CHOICES = FIRST_OPPONENT + PLAYERS * len(BESIDE)
# Staying, going on, a cut, around the disc, by a teammate, by an opponent; and whether
# that opponent is the attacker guarded
CHOICE_KINDS = 7
THROW_CHOICES = 1 + len(THROW_TARGETS) + PLAYERS - 1  # Holding on, the targets, the teammates
AHEAD = 0.5  # Seconds a player is taken to run on as they do, in how soon they get to a spot
FAR, LATE, LANE = 20.0, 3.0, 10.0  # Yards, seconds, yards: what counts as far, late, clear
LANE_FROM = 3.0  # Yards from the disc: a defender nearer (the mark) is not counted in a lane
NEVER = 99.0  # Seconds: no one known gets there
UNAVAILABLE = -1e9  # The score of a choice that may not be drawn
REACHED = 1.0  # Yards: a player this near the spot they chose is there
# An attacker choosing again before they are there, and a defender with somewhere to go
# on to, goes on this often at first (or no long cut would ever be run out, and no
# attacker stayed with, before a team has learned anything)
GO_ON_AT_FIRST = 0.8
# A defender choosing afresh takes a spot beside the attacker they guard this often at
# first: a team that has learned nothing defends man to man, and learns from there
MARK_AT_FIRST = 0.8
EVERYONE = (MATES + OPPONENTS) * PER_PLAYER
SPOT_FEATURES = 5 + CHOICE_KINDS + EVERYONE + 9 + 3 + 3 + 7 + 7 + 5
THROW_FEATURES = 1 + 5 + 3 + 3 + EVERYONE + 3 + 2 + 1


def _spot_columns() -> tuple:
    """Which of a spot's numbers are its own, which are the same for every spot a player
    weighs, and which are a place seen from the spot (spot_parts): three lists of
    columns, in the order of SPOT_GROUPS."""
    own, shared, seen = list(range(5 + CHOICE_KINDS)), [], []
    at = 5 + CHOICE_KINDS
    for _ in range(MATES + OPPONENTS):  # From the spot; then the player as they are
        seen += [at, at + 1]
        shared += range(at + 2, at + PER_PLAYER)
        at += PER_PLAYER
    shared += [at, at + 1, at + 2]  # The disc: flying, held, down
    seen += range(at + 3, at + 7)  # ... where it is and lands, from the spot
    shared += [at + 7, at + 8]  # ... how soon, how high
    at += 9
    own += range(at, at + 6)  # A throw to it, the field there
    shared += range(at + 6, at + 20)  # The wind, the stall, the player themselves
    own += range(at + 20, at + 25)  # How open it is
    assert at + 25 == SPOT_FEATURES
    return np.array(own), np.array(shared), np.array(seen)


SPOT_COLUMNS = _spot_columns()
# Log chances of the throws (not holding on) at first: the targets alike, the teammates alike
THROW_SHARES = np.r_[
    np.full(len(THROW_TARGETS), np.log((1.0 - MATE_SHARE) / len(THROW_TARGETS))),
    np.full(PLAYERS - 1, np.log(MATE_SHARE / (PLAYERS - 1))),
]
NO_DISC, HELD, IN_AIR, ON_GROUND = range(4)  # What the disc is doing, in a view
# What a spot's or a throw's numbers are, by kind, in order (for drawing the networks)
_ORDINALS = ("", "2nd ", "3rd ", "4th ", "5th ", "6th ", "7th ")
_TEAMMATES = tuple((f"{n}teammate nearest me".capitalize(), PER_PLAYER) for n in _ORDINALS[:MATES])
_OPPONENTS = tuple(
    (f"{n}opponent nearest me".capitalize(), PER_PLAYER) for n in _ORDINALS[:OPPONENTS]
)
_FIELD = (("In the end zone attacked", 1), ("Out of bounds", 1), ("To the goal line", 1))
SPOT_GROUPS = (
    ("Spot: on the field", 2),
    ("Spot: from me", 2),
    ("Spot: how far", 1),
    ("Stay, go on, cut, by the disc, a teammate, an opponent", CHOICE_KINDS - 1),
    ("By the attacker I guard", 1),
    *_TEAMMATES,
    *_OPPONENTS,
    ("Disc: flying, held, down", 3),
    ("Disc: where now", 2),
    ("Disc: where it lands", 2),
    ("Disc: how soon, how high", 2),
    ("Throw to it: flight time", 1),
    ("Throw to it: wind along, across", 2),
    *_FIELD,
    ("Wind", 2),
    ("Stall count", 1),
    ("Time held", 1),
    ("On offence", 1),
    ("My velocity", 2),
    ("My facing", 2),
    ("My acceleration", 2),
    ("Running to: from me", 2),
    ("Running to it for", 1),
    ("Nearest opponent, teammate to it", 2),
    ("How soon I get there, and before them", 2),
    ("Lane from the disc", 1),
)
THROW_GROUPS = (
    ("Hold on instead", 1),
    ("Target: on the field", 2),
    ("Target: from me", 2),
    ("Target: how far", 1),
    ("Flight time", 1),
    ("Wind along, across", 2),
    *_FIELD,
    *_TEAMMATES,
    *_OPPONENTS,
    ("Receiver there before a defender, the disc", 2),
    ("Lane to it", 1),
    ("Stall count", 1),
    ("Time held", 1),
    ("To a teammate", 1),
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
    "plan_age": ((), "f"),  # Seconds since it was chosen
    "partner": ((PLAYERS,), "i"),  # Each of ours: the opponent matched with them (-1: none)
}


def views_of(
    views: List[PlayerView],
    plans: Optional[List[Optional[tuple]]] = None,
    device="cpu",
    ages: Optional[List[float]] = None,
):
    """The tensors of a list of PlayerView (spots.py's plans: (kind, player, param) or
    None per view, and how many seconds ago each was chosen)."""
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
        found["partner"].append(
            np.full(PLAYERS, -1) if view.partner is None else np.asarray(view.partner)
        )
        age = 0.0 if plan is None or ages is None else ages[i]
        found["plan_age"].append(float(np.clip(age, 0.0, 5.0)))
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


def _known_players(rows, known, accel, facing, holds, mine, centre, skip=None):
    """Every player of one team as known (but `skip`), the nearest `centre` first: where
    they are (B, slots, 2), whether there is one (B, slots), and each as they are, from
    the centre (B, slots, PER_PLAYER - 2)."""
    import torch

    B = len(rows)
    gap = torch.linalg.norm(rows[..., :2] - centre[:, None], dim=-1)  # (B, 7)
    valid = known.clone()
    slots = PLAYERS
    if skip is not None:
        valid[torch.arange(B, device=skip.device), skip.clamp(min=0)] &= skip < 0
        slots = PLAYERS - 1
    gap = torch.where(valid, gap, torch.full_like(gap, float("inf")))
    ordered, order = torch.sort(gap, dim=1)
    order, present = order[:, :slots], torch.isfinite(ordered[:, :slots]).float()

    def pick(values):  # (B, 7, d) -> (B, slots, d)
        return torch.gather(values, 1, order[..., None].expand(-1, -1, values.shape[-1]))

    places = pick(rows[..., :2])
    own = (
        torch.cat(
            [
                (places - centre[:, None]) / 20.0,
                pick(rows[..., 2:4]) / RUN_SPEED,
                pick(accel) / 7.0,
                pick(facing),
                pick(holds[..., None].float()),
                pick(mine[..., None].float()),
                torch.ones_like(present[..., None]),
            ],
            dim=-1,
        )
        * present[..., None]
    )
    return places, present, own


def everyone(spots, rows, known, accel, facing, holds, mine, centre, skip=None):
    """(B, n, slots * PER_PLAYER): every player of one team as known (but `skip`), the
    nearest `centre` first, from each spot and from the centre."""
    import torch

    B, n = spots.shape[:2]
    places, present, own = _known_players(rows, known, accel, facing, holds, mine, centre, skip)
    from_spot = (places[:, None] - spots[:, :, None]) / 20.0 * present[:, None, :, None]
    found = torch.cat([from_spot, own[:, None].expand(-1, n, -1, -1)], dim=-1)
    return found.reshape(B, n, -1)


def _holds(v):
    """(B, 7) each: which of ours, and of theirs, holds the disc."""
    import torch

    index = torch.arange(PLAYERS, device=v["me"].device)[None]
    holds = (index == v["holder"][:, None]) & (v["holder"][:, None] >= 0)
    return holds & v["attacking"][:, None], holds & ~v["attacking"][:, None]


def _teams(v, skip):
    """What everyone() takes of each team, ours then theirs: the players, which are
    known, how they speed up and face, who holds the disc, and which is the opponent
    matched with the player choosing (`skip`: themselves, or the thrower)."""
    import torch

    ours_hold, theirs_hold = _holds(v)
    b = torch.arange(len(skip), device=skip.device)
    partner = v["partner"][b, skip.clamp(min=0)]
    index = torch.arange(PLAYERS, device=skip.device)[None]
    theirs_mine = (index == partner[:, None]) & (partner[:, None] >= 0)
    nobody = torch.zeros_like(theirs_mine)
    return (
        (v["ours"], v["ours_known"], v["ours_accel"], v["ours_facing"], ours_hold, nobody),
        (
            v["theirs"],
            v["theirs_known"],
            v["theirs_accel"],
            v["theirs_facing"],
            theirs_hold,
            theirs_mine,
        ),
    )


def _players(v, spots, centre, skip):
    """Every teammate (but `skip`) and every opponent, as everyone() describes them."""
    import torch

    ours, theirs = _teams(v, skip)
    return torch.cat(
        [everyone(spots, *ours, centre, skip), everyone(spots, *theirs, centre)], dim=-1
    )


def _flight(start, ends, wind):
    """(B, n) each: how long a throw from `start` (B, 2) to each of `ends` (B, n, 2) flies,
    and the wind along and across it."""
    import torch

    way = ends - start[:, None]
    length = torch.linalg.norm(way, dim=-1).clamp(min=1e-6)
    unit = way / length[..., None]
    along = (unit * wind[:, None]).sum(-1)
    across = unit[..., 0] * wind[:, None, 1] - unit[..., 1] * wind[:, None, 0]
    speed = (DISC_YARDS_PER_SECOND + WIND_ALONG * along).clamp(min=SLOWEST)
    return RELEASE_SECONDS + length / speed, along, across


def _field(places):
    """(B, n, 3): whether in the end zone attacked, whether out of bounds, and how far to
    the goal line attacked (as a share of the field's length, below 0 past it)."""
    import torch

    x, y = places[..., 0], places[..., 1]
    goal = LENGTH - END_ZONE
    inside = (x >= 0) & (x <= WIDTH) & (y >= 0) & (y <= LENGTH)
    return torch.stack(
        [((y > goal) & inside).float(), (~inside).float(), (goal - y) / LENGTH], dim=-1
    )


def _seconds_to(spots, rows, valid):
    """(B, n, 7): how soon each player of `rows` (B, 7, 4) gets to each of `spots`
    (B, n, 2), running on as they do for AHEAD and then straight there at top speed;
    NEVER for those not `valid` (B, 7)."""
    import torch

    ahead = rows[..., :2] + AHEAD * rows[..., 2:4]
    seconds = AHEAD + torch.linalg.norm(spots[:, :, None] - ahead[:, None], dim=-1) / RUN_SPEED
    return torch.where(valid[:, None], seconds, torch.full_like(seconds, NEVER))


def _nearest(spots, rows, valid):
    """(B, n): how far the nearest `valid` player of `rows` is from each spot, as a share
    of FAR (1: that far or further, or no one)."""
    import torch

    gaps = torch.linalg.norm(spots[:, :, None] - rows[:, None, :, :2], dim=-1)
    gaps = torch.where(valid[:, None], gaps, torch.full_like(gaps, FAR))
    return gaps.min(-1).values.clamp(max=FAR) / FAR


def _lane(start, ends, rows, valid):
    """(B, n): how clear the straight line from `start` (B, 2) to each of `ends` (B, n, 2)
    is of the `valid` players of `rows`: how near the nearest comes to it, as a share of
    LANE (1: that clear or clearer). A player within LANE_FROM of the start is not
    counted (the mark: the players' own numbers say where they stand)."""
    import torch

    places = rows[..., :2]
    way = ends - start[:, None]  # (B, n, 2)
    squared = (way**2).sum(-1).clamp(min=1e-6)
    off = places[:, None] - start[:, None, None]  # (B, 1, 7, 2)
    share = ((off * way[:, :, None]).sum(-1) / squared[..., None]).clamp(0.0, 1.0)
    gaps = torch.linalg.norm(off - share[..., None] * way[:, :, None], dim=-1)  # (B, n, 7)
    counted = valid & (torch.linalg.norm(places - start[:, None], dim=-1) >= LANE_FROM)
    gaps = torch.where(counted[:, None], gaps, torch.full_like(gaps, LANE))
    return gaps.min(-1).values.clamp(max=LANE) / LANE


def plan_spots(v):
    """(B, 2) where each player's spot chosen last is now (a spot beside a player follows
    them; their own place without one), and (B,) whether they may go on to it (not to a
    spot beside a player no longer known)."""
    import torch

    b = torch.arange(len(v["me"]), device=v["me"].device)
    place = v["ours"][b, v["me"], :2]
    who = v["plan_player"].clamp(min=0)
    by_them, by_mate = v["plan_kind"] == OPPONENT, v["plan_kind"] == TEAMMATE
    theirs = v["theirs"][b, who, :2] + v["plan_param"]
    ours = v["ours"][b, who, :2] + v["plan_param"]
    may = (
        (v["plan_kind"] == FIELD)
        | (by_them & v["theirs_known"][b, who])
        | (by_mate & v["ours_known"][b, who] & (who != v["me"]))
    )
    spot = torch.where(
        by_them[:, None], theirs, torch.where(by_mate[:, None], ours, v["plan_param"])
    )
    return torch.where(may[:, None], spot, place), may


def spot_leans(v, allowed):
    """(B, SPOT_CHOICES), to add to the scores of spot_choices: how likely each choice is
    while the network scores them all alike (their log chances). A choice not `allowed`
    is never drawn. An attacker still on their way, and a defender with somewhere to go
    on to, goes on GO_ON_AT_FIRST of the time; a defender takes one of the spots beside
    the attacker they guard MARK_AT_FIRST of the rest; all else is as likely as each
    other. The network learns from there when to break a cut off, leave an attacker,
    stand elsewhere."""
    import torch

    device = allowed.device
    b = torch.arange(len(v["me"]), device=device)
    going_to, may_go_on = plan_spots(v)
    away = torch.linalg.norm(going_to - v["ours"][b, v["me"], :2], dim=-1) > REACHED
    goes_on = may_go_on & (away | ~v["attacking"])
    # The spots beside the attacker this defender guards
    partner = torch.where(v["attacking"], -1, v["partner"][b, v["me"]])
    choice = torch.arange(allowed.shape[1], device=device)[None]
    slot = torch.div(choice - FIRST_OPPONENT, len(BESIDE), rounding_mode="floor")
    guarded = allowed & (choice >= FIRST_OPPONENT) & (slot == partner[:, None])
    keeps = (choice == KEEP) & goes_on[:, None]
    others = allowed & ~guarded & ~keeps
    kept = GO_ON_AT_FIRST * goes_on.float()
    marks = MARK_AT_FIRST * guarded.any(-1).float()
    rest = 1.0 - kept
    chances = torch.where(
        guarded,
        (rest * marks / guarded.sum(-1).clamp(min=1))[:, None],
        (rest * (1.0 - marks) / others.sum(-1).clamp(min=1))[:, None] * others,
    )
    chances = torch.where(keeps, kept[:, None], chances)
    return torch.log(chances.clamp(min=1e-30)).masked_fill(~allowed, UNAVAILABLE)


def spot_parts(v) -> dict:
    """Where each player may run, and what is around each place, in parts (spot_choices
    has what they are): "kinds", "players", "params" and "allowed" of the choices; and
    their numbers apart by how they depend on the place (SPOT_COLUMNS): "own" (B,
    SPOT_CHOICES, .), each place's own; "shared" (B, .), the same for every place a
    player weighs; and the places seen from each place ("anchors" (B, ., 2), with
    "weights" (B, .): 0 for one not there), which are (anchor - weight x place) / 20 for
    the place ("spots" (B, SPOT_CHOICES, 2)). Most numbers are shared or seen, so
    spot_scores scores the places without writing them out for each."""
    import torch

    device = v["me"].device
    B = len(v["me"])
    b = torch.arange(B, device=device)
    me = v["ours"][b, v["me"]]
    place = me[:, :2]
    disc = disc_place(v)
    low = torch.tensor([EDGE, EDGE], device=device)
    high = torch.tensor([WIDTH - EDGE, LENGTH - EDGE], device=device)
    cuts = place[:, None] + torch.as_tensor(CUTS, dtype=torch.float32, device=device)
    cuts = torch.maximum(torch.minimum(cuts, high), low)
    around = disc[:, None] + torch.as_tensor(AROUND, dtype=torch.float32, device=device)
    inside = ((around >= low) & (around <= high)).all(-1)
    around = torch.maximum(torch.minimum(around, high), low)
    going_to, may_go_on = plan_spots(v)
    beside = torch.as_tensor(BESIDE, dtype=torch.float32, device=device)
    by_mates = (v["ours"][:, :, None, :2] + beside).reshape(B, -1, 2)
    by_them = (v["theirs"][:, :, None, :2] + beside).reshape(B, -1, 2)
    index = torch.arange(PLAYERS, device=device)[None]
    mates = v["ours_known"] & (index != v["me"][:, None])
    spots = torch.cat([place[:, None], going_to[:, None], cuts, around, by_mates, by_them], dim=1)
    n, n_field, n_by = spots.shape[1], len(CUTS) + len(AROUND), PLAYERS * len(BESIDE)

    def full(count, value):
        return torch.full((B, count), value, dtype=torch.long, device=device)

    slots = index.repeat_interleave(len(BESIDE), dim=1).expand(B, -1)
    gone_on = torch.where(may_go_on, v["plan_kind"], full(1, FIELD)[:, 0])
    kinds = torch.cat(
        [
            full(1, FIELD),
            gone_on[:, None],
            full(n_field, FIELD),
            full(n_by, TEAMMATE),
            full(n_by, OPPONENT),
        ],
        dim=1,
    )
    followed = torch.where(may_go_on & (v["plan_kind"] != FIELD), v["plan_player"], -1)
    players = torch.cat([full(1, -1), followed[:, None], full(n_field, -1), slots, slots], dim=1)
    steps = beside.repeat(PLAYERS, 1)[None].expand(B, -1, -1)
    params = torch.cat(
        [
            place[:, None],
            torch.where(may_go_on[:, None], v["plan_param"], place)[:, None],
            cuts,
            around,
            steps,
            steps,
        ],
        dim=1,
    )
    yes = torch.ones((B, 1), dtype=torch.bool, device=device)
    allowed = torch.cat(
        [
            yes,
            may_go_on[:, None],
            yes.expand(-1, len(CUTS)),
            inside,
            mates.repeat_interleave(len(BESIDE), dim=1),
            v["theirs_known"].repeat_interleave(len(BESIDE), dim=1),
        ],
        dim=1,
    )
    choice = torch.cat(
        [
            full(1, 0),
            full(1, 1),
            full(len(CUTS), 2),
            full(len(AROUND), 3),
            full(n_by, 4),
            full(n_by, 5),
        ],
        dim=1,
    )
    # Whether a spot is beside the attacker this defender guards
    partner = torch.where(v["attacking"], -1, v["partner"][b, v["me"]])
    guarded = torch.cat(
        [torch.zeros((B, n - n_by), dtype=torch.bool, device=device), slots == partner[:, None]],
        dim=1,
    )
    offset = spots - place[:, None]
    going = may_go_on.float()[:, None]
    shared = torch.cat(
        [
            v["wind"] / 10.0,
            v["stall"][:, None] / 10.0,
            v["held"][:, None] / 5.0,
            v["attacking"][:, None].float(),
            me[:, 2:4] / RUN_SPEED,
            v["ours_facing"][b, v["me"]],
            v["ours_accel"][b, v["me"]] / 7.0,
            (going_to - place) * going / 20.0,
            v["plan_age"][:, None] * going / 2.0,
        ],
        dim=1,
    )
    # A throw from the holder to each spot, if someone holds the disc
    held = (v["disc_mode"] == HELD).float()[:, None]
    flying, along, across = _flight(disc, spots, v["wind"])
    throw = torch.stack([flying / 3.0, along / 10.0, across / 10.0], dim=-1) * held[..., None]
    # How open each spot is: who is near it, who gets there first, and whether a throw
    # from the disc gets through (past the defence: the other team, or their own)
    mine = _seconds_to(spots, me[:, None], torch.ones((B, 1), dtype=torch.bool, device=device))
    theirs = _seconds_to(spots, v["theirs"], v["theirs_known"]).min(-1).values
    defence = torch.where(v["attacking"][:, None, None], v["theirs"], v["ours"])
    defending = torch.where(v["attacking"][:, None], v["theirs_known"], mates)
    openness = torch.stack(
        [
            _nearest(spots, v["theirs"], v["theirs_known"]),
            _nearest(spots, v["ours"], mates),
            mine[..., 0].clamp(max=2.0 * LATE) / LATE,
            (theirs - mine[..., 0]).clamp(-LATE, LATE) / LATE,
            _lane(disc, spots, defence, defending) * held,
        ],
        dim=-1,
    )
    # Everyone as they are (shared), and where they are (seen from each place)
    ours, theirs = _teams(v, v["me"])
    our_places, our_present, our_own = _known_players(*ours, place, v["me"])
    their_places, their_present, their_own = _known_players(*theirs, place)
    present = torch.cat([our_present, their_present], dim=1)
    people = torch.cat([our_places, their_places], dim=1) * present[..., None]
    # The disc: what it does (shared), where it is and comes down (seen from each place)
    mode = v["disc_mode"][:, None]
    in_hand, air = (mode == HELD).float(), (mode == IN_AIR).float()
    ground = (mode == ON_GROUND).float()
    now = in_hand * disc + air * v["disc_now"] + ground * v["disc"]
    lands = air * v["disc"]
    scale = torch.tensor([WIDTH, LENGTH], device=device)
    own = torch.cat(
        [
            spots / scale,
            offset / 20.0,
            torch.linalg.norm(offset, dim=-1, keepdim=True) / 30.0,
            torch.nn.functional.one_hot(choice, CHOICE_KINDS - 1).float(),
            guarded[..., None].float(),
            throw,
            _field(spots),
            openness,
        ],
        dim=-1,
    )
    same = torch.cat(
        [
            torch.cat([our_own, their_own], dim=1).reshape(B, -1),
            air,
            in_hand,
            ground,
            air * v["lands_in"][:, None] / 3.0,
            air * v["height"][:, None] / 10.0,
            shared,
        ],
        dim=1,
    )
    return {
        "own": own,
        "shared": same,
        "anchors": torch.cat([people, now[:, None], lands[:, None]], dim=1),
        "weights": torch.cat([present, in_hand + air + ground, air], dim=1),
        "spots": spots,
        "kinds": kinds,
        "players": players,
        "params": params,
        "allowed": allowed,
    }


PART_NUMBERS = ("own", "shared", "anchors", "weights", "spots")


def take_parts(parts: dict, rows) -> dict:
    """The numbers of spot_parts of some players only, for spot_scores."""
    return {name: parts[name][rows] for name in PART_NUMBERS}


def spot_choices(v):
    """(features (B, SPOT_CHOICES, SPOT_FEATURES), kinds, players, params (B, SPOT_CHOICES
    [, 2]), allowed (B, SPOT_CHOICES)): where each player may run, and what is around
    each place: staying; going on to the spot chosen last; a cut each way and length from
    where they are; a place around the disc; beside any teammate or opponent (a spot that
    follows them). The choices not `allowed` are not to be drawn: going on with nowhere
    to go on to, a place off the field, a spot beside themselves or a player not known."""
    import torch

    parts = spot_parts(v)
    own, shared, seen = SPOT_COLUMNS
    B, n = parts["own"].shape[:2]
    features = torch.zeros((B, n, SPOT_FEATURES), device=parts["own"].device)
    features[..., own] = parts["own"]
    features[..., shared] = parts["shared"][:, None]
    from_spot = (
        parts["anchors"][:, None] - parts["weights"][:, None, :, None] * parts["spots"][:, :, None]
    )
    features[..., seen] = (from_spot / 20.0).reshape(B, n, -1)
    return features, parts["kinds"], parts["players"], parts["params"], parts["allowed"]


def spot_scores(network, parts: dict):
    """(B, SPOT_CHOICES): what `network` scores each place of spot_parts, as it would
    from spot_choices' numbers written out, with its first layer taken apart: what is
    shared goes through once a player, and what is seen from a place is a line in the
    place."""
    import torch

    own, shared, seen = (
        torch.as_tensor(columns, device=parts["own"].device) for columns in SPOT_COLUMNS
    )
    weight, bias = network[0].weight, network[0].bias
    towards = weight[:, seen].reshape(len(weight), -1, 2)  # (hidden, anchors, 2)
    base = parts["shared"] @ weight[:, shared].T + bias
    base = base + torch.einsum("hjd,bjd->bh", towards, parts["anchors"]) / 20.0
    slope = torch.einsum("hjd,bj->bhd", towards, parts["weights"]) / 20.0
    hidden = parts["own"] @ weight[:, own].T + base[:, None]
    hidden = hidden - parts["spots"] @ slope.transpose(1, 2)
    return network[1:](hidden)[..., 0]


def throw_aims(v):
    """(B, THROW_CHOICES - 1, 2): where each throw goes, as the holder's team sees the
    field: the targets, then each teammate of the holder, in order, where they will be when
    the disc gets there. For views of holders."""
    import torch

    device = v["me"].device
    B = len(v["me"])
    b = torch.arange(B, device=device)
    holder = v["holder"].clamp(min=0).long()
    thrower = v["ours"][b, holder, :2]
    slot = torch.arange(MATES, device=device)[None]
    rows = v["ours"][b[:, None], slot + (slot >= holder[:, None]).long()]  # (B, 6, 4)
    lead = rows[..., :2]
    for _ in range(2):  # Where they will be when the disc gets there
        gap = torch.linalg.norm(lead - thrower[:, None], dim=-1)
        flying = RELEASE_SECONDS + gap / DISC_YARDS_PER_SECOND
        lead = rows[..., :2] + rows[..., 2:] * flying.clamp(max=LEAD_MOST)[..., None]
    low = torch.tensor([LEAD_MARGIN, LEAD_MARGIN], device=device)
    high = torch.tensor([WIDTH - LEAD_MARGIN, LENGTH - LEAD_MARGIN], device=device)
    lead = torch.maximum(torch.minimum(lead, high), low)
    targets = torch.as_tensor(THROW_TARGETS, dtype=torch.float32, device=device)
    return torch.cat([targets[None].expand(B, -1, -1), lead], dim=1)


def throw_choices(v):
    """(B, THROW_CHOICES, THROW_FEATURES): holding on, then throwing to each target and
    to each teammate (throw_aims), and what is around each: where it is, how far and which
    way from the thrower, how long the disc would take (the wind with it or against it),
    the wind along and across, the players nearest it; how much sooner a teammate gets
    there than a defender and than the disc, and how clear the way there is of defenders;
    the stall and how long it has been held; whether it is to a teammate. For views of
    holders."""
    import torch

    device = v["me"].device
    B = len(v["me"])
    b = torch.arange(B, device=device)
    thrower = v["ours"][b, v["holder"].clamp(min=0), :2]
    targets = throw_aims(v)
    way = targets - thrower[:, None]
    length = torch.linalg.norm(way, dim=-1).clamp(min=1e-6)
    flying, along_wind, across_wind = _flight(thrower, targets, v["wind"])
    timing = torch.stack([v["stall"] / 10.0, v["held"] / 5.0], dim=1)
    n = targets.shape[1]
    scale = torch.tensor([WIDTH, LENGTH], device=device)
    index = torch.arange(PLAYERS, device=device)[None]
    receivers = v["ours_known"] & (index != v["holder"][:, None])
    receiver = _seconds_to(targets, v["ours"], receivers).min(-1).values
    defender = _seconds_to(targets, v["theirs"], v["theirs_known"]).min(-1).values
    openness = torch.stack(
        [
            (defender - receiver).clamp(-LATE, LATE) / LATE,
            (flying - receiver).clamp(-LATE, LATE) / LATE,
            _lane(thrower, targets, v["theirs"], v["theirs_known"]),
        ],
        dim=-1,
    )
    throws = torch.cat(
        [
            torch.zeros((B, n, 1), device=device),
            targets / scale,
            way / 20.0,
            length[..., None] / 30.0,
            flying[..., None] / 3.0,
            along_wind[..., None] / 10.0,
            across_wind[..., None] / 10.0,
            _field(targets),
            _players(v, targets, thrower, v["holder"]),
            openness,
            timing[:, None].expand(-1, n, -1),
            (torch.arange(n, device=device) >= len(THROW_TARGETS))
            .float()[None, :, None]
            .expand(B, -1, -1),
        ],
        dim=-1,
    )
    hold = torch.zeros((B, 1, THROW_FEATURES), device=device)
    hold[:, 0, 0] = 1.0
    hold[:, 0, -3:-1] = timing
    return torch.cat([hold, throws], dim=1)


def stack_views(parts: List[Dict]) -> Dict:
    import torch

    return {name: torch.cat([p[name] for p in parts]) for name in VIEW_FIELDS}


def take(v: Dict, index) -> Dict:
    return {name: value[index] for name, value in v.items()}
