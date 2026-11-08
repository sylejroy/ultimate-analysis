#!/usr/bin/env python3
"""Make a table of the throws of a game from what export_analysis.py wrote of it.

A throw is the disc going from one holder to the next within one stretch of the camera
(a restart begins the players' IDs anew). Two tables are written next to the export:

    <export>/throws.csv       throw, restart, release_frame, release_seconds,
                              catch_frame, catch_seconds, between_seconds, flew,
                              thrower, receiver, team, outcome, game_state, from_x,
                              from_y, to_x, to_y, attack, gain
    <export>/at_release.csv   throw, player, side, x, y, vx, vy

- restart is that of the export: thrower, receiver and team are named within it.
- release is the last frame the thrower holds the disc, catch the first the next holder
  does; between_seconds the time between, flew whether the disc was seen in the air
  then (a holder that changes without it is often a box that changed players).
- outcome is "complete" when the next holder is a teammate, "turnover" when they are of
  the other team: then to_x and to_y are where the other team took the disc.
- attack is +1 where the thrower's team attacks the far end zone (y grows), -1 the near
  one, empty where no line-up has shown it; gain is the yards the throw went towards
  that end zone (negative backwards), empty without attack.
- at_release has a row per player in the release frame: side is "thrower", "offence",
  "defence" or "unknown" (team not known); x, y their place on the field, vx, vy how fast
  they ran there over the half second before, in yards a second.

Places are the field model's estimate: docs/MEASUREMENTS.md says how far off it is.

Usage:
    python scripts/build_throw_table.py data/exports/whole_game [more exports]
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

THROW_COLUMNS = (
    "throw restart release_frame release_seconds catch_frame catch_seconds between_seconds flew "
    "thrower receiver team outcome game_state from_x from_y to_x to_y attack gain"
)
RELEASE_COLUMNS = "throw player side x y vx vy"
LONGEST_THROW_SECONDS = 6.0  # Holders further apart than this: the disc was lost from view
SPEED_SECONDS = 0.5  # How far back a player's speed at the release is measured over
PLACE_SEARCH_SECONDS = 0.5  # How far a missing place of thrower or receiver is looked for


@dataclass
class Hold:
    """A stretch of frames in which one player holds the disc."""

    holder: int
    team: Optional[int]
    first: int  # Rows of disc.csv
    last: int


def number(text: str) -> Optional[float]:
    """A cell as a number, None if empty."""
    return float(text) if text != "" else None


def read(folder: Path) -> Tuple[List[dict], Dict[int, Dict[int, dict]]]:
    """The disc rows in order, and frame -> player -> row of the players."""
    with open(folder / "disc.csv", newline="") as file:
        disc = list(csv.DictReader(file))
    players: Dict[int, Dict[int, dict]] = defaultdict(dict)
    with open(folder / "players.csv", newline="") as file:
        for row in csv.DictReader(file):
            players[int(row["frame"])][int(row["player"])] = row
    return disc, players


def holds(disc: List[dict]) -> List[List[Hold]]:
    """The holds of each stretch of the camera, in order. Frames in the air between two
    holds of one player belong to the hold: a pump fake, or the disc lost from view."""
    stretches: Dict[str, List[Hold]] = {}
    for row_index, row in enumerate(disc):
        if row["state"] != "held" or row["holder"] == "":
            continue
        holder = int(row["holder"])
        team = None if row["team_in_possession"] == "" else int(row["team_in_possession"])
        stretch = stretches.setdefault(row["restart"], [])
        if stretch and stretch[-1].holder == holder:
            stretch[-1].last = row_index
        else:
            stretch.append(Hold(holder, team, row_index, row_index))
    return list(stretches.values())


def place(row: Optional[dict]) -> Optional[Tuple[float, float]]:
    """A player's place on the field, None if not known."""
    if row is None or row["place_x"] == "":
        return None
    return float(row["place_x"]), float(row["place_y"])


def place_near(
    players: Dict[int, Dict[int, dict]], frames: List[int], at: int, player: int, step: int
) -> Optional[Tuple[float, float]]:
    """A player's place at frames[at], or the nearest one within PLACE_SEARCH_SECONDS
    looking backwards (step -1) or forwards (+1)."""
    seconds = None
    index = at
    while 0 <= index < len(frames):
        row = players[frames[index]].get(player)
        if row is not None:
            seconds = seconds if seconds is not None else float(row["seconds"])
            if abs(float(row["seconds"]) - seconds) > PLACE_SEARCH_SECONDS:
                return None
            found = place(row)
            if found is not None:
                return found
        index += step
    return None


def at_release(
    players: Dict[int, Dict[int, dict]], frames: List[int], at: int, thrower: int, team: int
) -> List[list]:
    """A row per player in the release frame: side, place, and speed."""
    now = players[frames[at]]
    seconds = float(next(iter(now.values()))["seconds"]) if now else 0.0
    earlier_at = at
    while earlier_at > 0:
        row = next(iter(players[frames[earlier_at - 1]].values()), None)
        if row is None or seconds - float(row["seconds"]) > SPEED_SECONDS:
            break
        earlier_at -= 1
    rows = []
    for player, row in now.items():
        here = place(row)
        if here is None:
            continue
        side = (
            "thrower"
            if player == thrower
            else "unknown"
            if row["team"] == ""
            else "offence"
            if int(row["team"]) == team
            else "defence"
        )
        vx = vy = ""
        before = players[frames[earlier_at]].get(player)
        there = place(before)
        if there is not None and earlier_at < at:
            elapsed = seconds - float(before["seconds"])
            vx = round((here[0] - there[0]) / elapsed, 2)
            vy = round((here[1] - there[1]) / elapsed, 2)
        rows.append([player, side, round(here[0], 2), round(here[1], 2), vx, vy])
    return rows


def build(folder: Path) -> Tuple[List[list], List[list]]:
    """The rows of throws.csv and at_release.csv of one export."""
    disc, players = read(folder)
    frames = [int(row["frame"]) for row in disc]
    throws, released = [], []
    for stretch in holds(disc):
        for one, other in zip(stretch, stretch[1:]):
            release, catch = disc[one.last], disc[other.first]
            between = float(catch["seconds"]) - float(release["seconds"])
            if between > LONGEST_THROW_SECONDS or one.team is None or other.team is None:
                continue
            flew = any(disc[i]["state"] == "air" for i in range(one.last + 1, other.first))
            start = place_near(players, frames, one.last, one.holder, -1)
            end = place_near(players, frames, other.first, other.holder, +1)
            attack = number(release[f"attack_{one.team}"])
            gain = ""
            if attack is not None and start is not None and end is not None:
                gain = round((end[1] - start[1]) * attack, 2)
            throw = len(throws)
            throws.append(
                [
                    throw,
                    release["restart"],
                    release["frame"],
                    release["seconds"],
                    catch["frame"],
                    catch["seconds"],
                    round(between, 3),
                    int(flew),
                    one.holder,
                    other.holder,
                    one.team,
                    "complete" if other.team == one.team else "turnover",
                    release["game_state"],
                    *(("", "") if start is None else (round(start[0], 2), round(start[1], 2))),
                    *(("", "") if end is None else (round(end[0], 2), round(end[1], 2))),
                    "" if attack is None else int(attack),
                    gain,
                ]
            )
            for row in at_release(players, frames, one.last, one.holder, one.team):
                released.append([throw, *row])
    return throws, released


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("exports", type=Path, nargs="+", help="Folders export_analysis.py wrote")
    args = parser.parse_args()

    for folder in args.exports:
        if (folder / "throws.csv").exists() or (folder / "at_release.csv").exists():
            sys.exit(f"Refusing to overwrite the throw tables in {folder}")
        throws, released = build(folder)
        with open(folder / "throws.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(THROW_COLUMNS.split())
            writer.writerows(throws)
        with open(folder / "at_release.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(RELEASE_COLUMNS.split())
            writer.writerows(released)
        complete = sum(1 for throw in throws if throw[11] == "complete")
        print(f"{folder}: {len(throws)} throws, {complete} complete")


if __name__ == "__main__":
    main()
