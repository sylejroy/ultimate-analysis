#!/usr/bin/env python3
"""Fit how likely a possession is to score from a catch at each place along the field.

From the throw tables of scripts/build_throw_table.py: the throws of one team in a row,
within one stretch of the camera, are a possession. It ends in a score when one of them
is caught in the end zone the team attacks, and without one at a turnover. A possession
cut off by the camera says nothing and is left out. Each completion outside the end
zones is then a catch from which the possession went on to score, or not; the chance is
fitted as the logistic of a + b * y, the field turned so that the far end zone is
attacked (processing/open_space.py). Which end a team attacks is taken as in
scripts/train_throw_targets.py.

A table of the catches per ten yards, scored and as fitted, is printed; the curve is
written to a new file:

    python scripts/fit_field_value.py data/exports/throws_20261107
        --output data/models/field_value/20261108.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

LENGTH, END_ZONE = 110.0, 20.0
LIVE_STATES = {"live", "unknown", "score"}


def attack_by_throws(throws: List[dict]) -> Dict[Tuple[str, str], int]:
    """(restart, team) -> +1 or -1, the way that team's completed throws went in that
    stretch of the camera."""
    went: Dict[Tuple[str, str], float] = defaultdict(float)
    for throw in throws:
        if throw["outcome"] == "complete" and throw["from_y"] != "" and throw["to_y"] != "":
            went[throw["restart"], throw["team"]] += float(throw["to_y"]) - float(throw["from_y"])
    return {key: 1 if total > 0 else -1 for key, total in went.items() if total != 0}


def catches_of(folder: Path) -> List[Tuple[float, int]]:
    """(place along the field as attacked, whether the possession scored) per catch."""
    with open(folder / "throws.csv", newline="") as file:
        throws = [t for t in csv.DictReader(file) if t["flew"] == "1"]
    guessed = attack_by_throws(throws)
    stretches: Dict[str, List[dict]] = defaultdict(list)
    for throw in throws:
        stretches[throw["restart"]].append(throw)
    found = []
    for stretch in stretches.values():
        stretch.sort(key=lambda t: int(t["release_frame"]))
        possession: List[float] = []
        team = None
        for throw in stretch:
            if throw["team"] != team:  # Another team has it: what was before is cut off
                possession, team = [], throw["team"]
            attack = throw["attack"] or guessed.get((throw["restart"], throw["team"]))
            if throw["to_y"] == "" or attack is None or throw["game_state"] not in LIVE_STATES:
                possession = []  # Not to be followed: left out
                continue
            along = float(throw["to_y"]) if int(attack) > 0 else LENGTH - float(throw["to_y"])
            if throw["outcome"] == "turnover":
                found += [(y, 0) for y in possession]
                possession = []
            elif along >= LENGTH - END_ZONE:
                found += [(y, 1) for y in possession]
                possession = []
            elif along > END_ZONE:
                possession.append(along)
    return found


def fit(along: np.ndarray, scored: np.ndarray, steps: int = 200) -> Tuple[float, float]:
    """a and b of the logistic of a + b * y, by Newton's method."""
    x = np.column_stack([np.ones_like(along), along / LENGTH])
    weights = np.zeros(2)
    for _ in range(steps):
        chance = 1.0 / (1.0 + np.exp(-x @ weights))
        gradient = x.T @ (scored - chance)
        hessian = (x * (chance * (1 - chance))[:, None]).T @ x + 1e-6 * np.eye(2)
        weights += np.linalg.solve(hessian, gradient)
    return float(weights[0]), float(weights[1] / LENGTH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("exports", type=Path, help="Folder of the games' exports")
    parser.add_argument("--output", type=Path, required=True, help="The curve, a new JSON file")
    args = parser.parse_args()
    if args.output.exists():
        sys.exit(f"Refusing to overwrite {args.output}")

    catches = []
    for folder in sorted(p for p in args.exports.iterdir() if (p / "throws.csv").exists()):
        theirs = catches_of(folder)
        print(f"{folder.name}: {len(theirs)} catches, {sum(s for _, s in theirs)} scored after")
        catches += theirs
    along = np.array([y for y, _ in catches])
    scored = np.array([s for _, s in catches], dtype=float)
    a, b = fit(along, scored)
    print(f"\n{'yards':>10} {'catches':>8} {'scored':>7} {'fitted':>7}")
    for start in range(int(END_ZONE), int(LENGTH - END_ZONE), 10):
        here = (along >= start) & (along < start + 10)
        if here.any():
            fitted = 1.0 / (1.0 + np.exp(-(a + b * (start + 5))))
            print(
                f"{start:>4} - {start + 10:<3} {here.sum():>8} {scored[here].mean():>7.0%} {fitted:>7.0%}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"a": a, "b": b, "catches": len(catches)}, indent=2))
    print(f"\na {a:.3f}, b {b:.4f} ({len(catches)} catches) written to {args.output}")


if __name__ == "__main__":
    main()
