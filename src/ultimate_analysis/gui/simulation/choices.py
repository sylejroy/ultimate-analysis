"""Words for what a spot-choosing player weighs (simulation/spots.py), for the brain panel:
each choice said plainly, and what each group of a choice's inputs says now
(spot_features.SPOT_GROUPS and THROW_GROUPS, in the same order)."""

from typing import List

import numpy as np

from ...simulation import spot_features as sf
from ...simulation.game import LENGTH, PLAYERS, RUN_SPEED, WIDTH

BESIDE_NAMES = ("behind", "ahead of", "left of", "right of")  # spot_features.BESIDE
SHOWN = 5  # Choices listed


def _way(offset: np.ndarray) -> str:
    """An offset on the field as a player attacking upwards says it."""
    across, along = float(offset[0]), float(offset[1])
    parts = []
    if abs(along) >= 1.0:
        parts.append(f"{abs(along):.0f} yd {'up' if along > 0 else 'back'}")
    if abs(across) >= 1.0:
        parts.append(f"{abs(across):.0f} {'right' if across > 0 else 'left'}")
    return ", ".join(parts) or "here"


def spot_label(choice: int, player: int, param: np.ndarray, offset: np.ndarray) -> str:
    """A choice of where to run (spot_features.spot_choices), in words. `offset`: the
    place from the player."""
    if choice == sf.STAY:
        return "Stay here"
    if choice == sf.KEEP:
        return f"Go on, {_way(offset)}"
    if choice < sf.FIRST_AROUND:
        return f"Cut {_way(offset)}"
    if choice < sf.FIRST_TEAMMATE:
        return f"By the disc, {_way(offset)}"
    side = BESIDE_NAMES[int(np.argmin(np.abs(sf.BESIDE - param).sum(1)))]
    who = "teammate" if choice < sf.FIRST_OPPONENT else "opponent"
    return f"Just {side} {who} {player + 1}"


def throw_label(view, choice: int) -> str:
    """Holding on, or a throw to a target, in words, with the teammate nearest it."""
    if choice == 0:
        return "Hold on"
    if choice > len(sf.THROW_TARGETS):
        slot = choice - 1 - len(sf.THROW_TARGETS)
        return f"Throw to teammate {slot + (slot >= view.holder) + 1}, leading them"
    target = sf.THROW_TARGETS[choice - 1]
    me = view.ours[view.me, :2]
    way = target - me
    near = ""
    mates = [k for k in range(PLAYERS) if k != view.me]
    if mates:
        gaps = np.hypot(*(view.ours[mates, :2] - target).T)
        k = int(np.argmin(gaps))
        if gaps[k] <= 10.0:
            near = f", to teammate {mates[k] + 1}"
    return f"Throw {_way(way)}{near}"


def ranked(chances: np.ndarray, taken: int) -> List[int]:
    """The choices to list: the one taken first, then the likeliest others."""
    rest = [int(i) for i in np.argsort(-chances) if int(i) != taken][: SHOWN - 1]
    return [taken] + rest


def _player(block: np.ndarray, what: str) -> str:
    """A player as spot_features.everyone() describes them."""
    if block[12] < 0.5:
        return f"no {what} known"
    from_spot = np.hypot(*block[0:2]) * 20.0
    from_me = np.hypot(*block[2:4]) * 20.0
    speed = np.hypot(*block[4:6]) * RUN_SPEED
    holds = ", has the disc" if block[10] > 0.5 else ""
    mine = ", mine" if block[11] > 0.5 else ""
    return f"{from_me:.0f} yd from me, {from_spot:.0f} from it, {speed:.0f} yd/s{holds}{mine}"


def _yes(value: float) -> str:
    return "yes" if value > 0.5 else "no"


def _parts(f: np.ndarray, groups) -> dict:
    """The inputs of each group, by its name."""
    found, start = {}, 0
    for name, count in groups:
        found[name] = f[start : start + count]
        start += count
    return found


def _people(part: dict) -> dict:
    return {
        name: _player(values, "teammate" if "teammate" in name else "opponent")
        for name, values in part.items()
        if "nearest me" in name
    }


def _field_notes(part: dict) -> dict:
    return {
        "In the end zone attacked": _yes(part["In the end zone attacked"][0]),
        "Out of bounds": _yes(part["Out of bounds"][0]),
        "To the goal line": f"{part['To the goal line'][0] * LENGTH:.0f} yd",
    }


def spot_notes(f: np.ndarray) -> List[str]:
    """What each group of a spot's inputs says (SPOT_GROUPS)."""
    part = _parts(f, sf.SPOT_GROUPS)
    kinds = ("staying", "going on", "a cut", "by the disc", "by a teammate", "by an opponent")
    kind = "Stay, go on, cut, by the disc, a teammate, an opponent"
    modes = part["Disc: flying, held, down"]
    mode = int(np.argmax(modes)) if modes.any() else -1
    held = mode == 1
    flying = mode == 0
    said = {
        "Spot: on the field": "{:.0f}, {:.0f} yd".format(
            *(part["Spot: on the field"] * (WIDTH, LENGTH))
        ),
        "Spot: from me": _way(part["Spot: from me"] * 20.0),
        "Spot: how far": f"{part['Spot: how far'][0] * 30.0:.0f} yd",
        kind: kinds[int(np.argmax(part[kind]))],
        "By the attacker I guard": _yes(part["By the attacker I guard"][0]),
        **_people(part),
        "Disc: flying, held, down": ("flying", "held", "on the ground")[mode]
        if mode >= 0
        else "not seen",
        "Disc: where now": f"{np.hypot(*part['Disc: where now']) * 20.0:.0f} yd from it"
        if mode >= 0
        else "",
        "Disc: where it lands": f"{np.hypot(*part['Disc: where it lands']) * 20.0:.0f} yd from it"
        if flying
        else "",
        "Disc: how soon, how high": "in {:.1f} s, {:.0f} yd up".format(
            *(part["Disc: how soon, how high"] * (3.0, 10.0))
        )
        if flying
        else "",
        "Throw to it: flight time": f"{part['Throw to it: flight time'][0] * 3.0:.1f} s"
        if held
        else "",
        "Throw to it: wind along, across": "{:+.0f}, {:+.0f} yd/s".format(
            *(part["Throw to it: wind along, across"] * 10.0)
        )
        if held
        else "",
        **_field_notes(part),
        "Wind": f"{np.hypot(*part['Wind']) * 10.0:.0f} yd/s",
        "Stall count": f"{part['Stall count'][0] * 10.0:.0f} s",
        "Time held": f"{part['Time held'][0] * 5.0:.1f} s",
        "On offence": _yes(part["On offence"][0]),
        "My velocity": f"{np.hypot(*part['My velocity']) * RUN_SPEED:.0f} yd/s",
        "My facing": _way(part["My facing"] * 10.0).replace(" yd", "").replace(", ", " and "),
        "My acceleration": f"{np.hypot(*part['My acceleration']) * 7.0:.0f} yd/s²",
        "Running to: from me": _way(part["Running to: from me"] * 20.0),
        "Running to it for": f"{part['Running to it for'][0] * 2.0:.1f} s",
        "Nearest opponent, teammate to it": "{:.0f}, {:.0f} yd".format(
            *(part["Nearest opponent, teammate to it"] * sf.FAR)
        ),
        "How soon I get there, and before them": "in {:.1f} s, {:+.1f} s".format(
            *(part["How soon I get there, and before them"] * sf.LATE)
        ),
        "Lane from the disc": f"{part['Lane from the disc'][0] * sf.LANE:.0f} yd clear"
        if held
        else "",
    }
    return [said.get(name, "") for name, _ in sf.SPOT_GROUPS]


def throw_notes(f: np.ndarray) -> List[str]:
    """What each group of a throw's inputs says (THROW_GROUPS)."""
    part = _parts(f, sf.THROW_GROUPS)
    timing = {
        "Stall count": f"{part['Stall count'][0] * 10.0:.0f} s",
        "Time held": f"{part['Time held'][0] * 5.0:.1f} s",
    }
    if part["Hold on instead"][0] > 0.5:  # Holding on: only the stall and the time held go in
        said = {"Hold on instead": "yes", **timing}
    else:
        said = {
            "Hold on instead": "no",
            "Target: on the field": "{:.0f}, {:.0f} yd".format(
                *(part["Target: on the field"] * (WIDTH, LENGTH))
            ),
            "Target: from me": _way(part["Target: from me"] * 20.0),
            "Target: how far": f"{part['Target: how far'][0] * 30.0:.0f} yd",
            "Flight time": f"{part['Flight time'][0] * 3.0:.1f} s",
            "Wind along, across": "{:+.0f}, {:+.0f} yd/s".format(
                *(part["Wind along, across"] * 10.0)
            ),
            **_field_notes(part),
            **_people(part),
            "Receiver there before a defender, the disc": "{:+.1f}, {:+.1f} s".format(
                *(part["Receiver there before a defender, the disc"] * sf.LATE)
            ),
            "Lane to it": f"{part['Lane to it'][0] * sf.LANE:.0f} yd clear",
            **timing,
            "To a teammate": "yes" if part["To a teammate"][0] > 0.5 else "no",
        }
    return [said.get(name, "") for name, _ in sf.THROW_GROUPS]
