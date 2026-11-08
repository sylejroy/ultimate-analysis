#!/usr/bin/env python3
"""Make a dataset of how the recorded players moved, for the simulated players to learn
from (scripts/train_movement.py).

From exports of scripts/export_analysis.py with their throw tables
(scripts/build_throw_table.py): every SAMPLE seconds of live play in which someone holds
the disc, for every other player whose place and team are known, what they saw as a
simulated player sees it (simulation/learned.py: themselves, the holder, the others
nearest first, turned so that their team attacks the far end) and how they ran over the
next AHEAD seconds. Offence and defence apart; which end a team attacks as in
scripts/train_throw_targets.py. One file per game:

    <output>/<game>.npz   x_offence, y_offence, x_defence, y_defence, and the holds:
                          seconds held before each throw

    python scripts/build_movement_dataset.py data/exports/throws_20261107 \\
        --output data/raw/training_data/generated_movement_20261108
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import train_throw_targets as training  # noqa: E402

from ultimate_analysis.simulation.game import RUN_SPEED  # noqa: E402
from ultimate_analysis.simulation.learned import features  # noqa: E402

SAMPLE = 0.2  # Seconds between the moments taken
AHEAD = 1.0  # Seconds over which where a player went is measured
BEFORE = 0.5  # ... and how they were running before
FIT = 0.05  # Seconds a place may be off the time it is looked for
LIVE_STATES = {"live", "unknown"}


def turned(x, y, attack):
    return (x, y) if attack > 0 else (40.0 - x, 110.0 - y)


def build(folder: Path) -> dict:
    with open(folder / "throws.csv", newline="") as file:
        throws = list(csv.DictReader(file))
    guessed = training.attack_by_throws(throws)
    with open(folder / "disc.csv", newline="") as file:
        disc = list(csv.DictReader(file))
    frames = defaultdict(list)
    tracks = defaultdict(list)  # (restart, player) -> (seconds, x, y)
    restart_of = {row["frame"]: row["restart"] for row in disc}
    with open(folder / "players.csv", newline="") as file:
        for row in csv.DictReader(file):
            if row["place_x"] == "":
                continue
            frames[row["frame"]].append(row)
            key = (restart_of.get(row["frame"]), row["player"])
            tracks[key].append(
                (float(row["seconds"]), float(row["place_x"]), float(row["place_y"]))
            )
    tracks = {key: np.array(rows) for key, rows in tracks.items()}

    def place_at(key, seconds):
        track = tracks.get(key)
        if track is None:
            return None
        i = int(np.searchsorted(track[:, 0], seconds))
        for j in (i - 1, i):
            if 0 <= j < len(track) and abs(track[j, 0] - seconds) <= FIT:
                return track[j, 1:]
        return None

    found = {"x_offence": [], "y_offence": [], "x_defence": [], "y_defence": []}
    held_since, last_holder, next_sample = None, None, -1.0
    for row in disc:
        seconds = float(row["seconds"])
        holder = row["holder"] if row["state"] == "held" else ""
        if holder != last_holder:
            held_since, last_holder = seconds, holder
        if seconds < next_sample or not holder or row["team_in_possession"] == "":
            continue
        if row["game_state"] not in LIVE_STATES:
            continue
        team = row["team_in_possession"]
        attack = row[f"attack_{team}"] or guessed.get((row["restart"], team))
        if attack is None or attack == "":
            continue
        next_sample = seconds + SAMPLE
        attack = int(attack)
        here = {}
        for player in frames[row["frame"]]:
            if player["team"] == "":
                continue
            key = (row["restart"], player["player"])
            now, before = place_at(key, seconds), place_at(key, seconds - BEFORE)
            if now is None or before is None:
                continue
            here[player["player"]] = (player["team"], now, (now - before) / BEFORE)
        if holder not in here:
            continue
        for name, (side, now, going) in here.items():
            if name == holder:
                continue
            later = place_at((row["restart"], name), seconds + AHEAD)
            if later is None:
                continue
            mine = attack if side == team else -attack

            def state_of(place, velocity):
                return [*turned(place[0], place[1], mine), velocity[0] * mine, velocity[1] * mine]

            me = np.array(state_of(now, going))
            ours = np.array(
                [state_of(p, v) for n, (s, p, v) in here.items() if s == side and n != name]
            )
            theirs = np.array([state_of(p, v) for n, (s, p, v) in here.items() if s != side])
            holder_place = np.array(turned(*here[holder][1], mine))
            x = features(
                me,
                ours.reshape(-1, 4),
                theirs.reshape(-1, 4),
                np.ones(len(ours)),
                np.ones(len(theirs)),
                holder_place,
                min(seconds - held_since, 10.0),
            )
            ran = (later - now) / AHEAD * mine / RUN_SPEED
            which = "offence" if side == team else "defence"
            found[f"x_{which}"].append(x)
            found[f"y_{which}"].append(ran)

    holds = []
    caught_at = {}
    for throw in throws:
        if throw["flew"] != "1":
            continue
        key = (throw["restart"], throw["thrower"])
        if key in caught_at:
            holds.append(float(throw["release_seconds"]) - caught_at[key])
        caught_at[(throw["restart"], throw["receiver"])] = float(throw["catch_seconds"])
    return {
        **{k: np.array(v, dtype=np.float32) for k, v in found.items()},
        "holds": np.array(holds, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("exports", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="A new folder")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in sorted(p for p in args.exports.iterdir() if (p / "throws.csv").exists()):
        target = args.output / f"{folder.name}.npz"
        if target.exists():
            print(f"{folder.name}: already made")
            continue
        found = build(folder)
        np.savez_compressed(target, **found)
        print(
            f"{folder.name}: {len(found['x_offence'])} offence, {len(found['x_defence'])} "
            f"defence, {len(found['holds'])} holds"
        )


if __name__ == "__main__":
    main()
