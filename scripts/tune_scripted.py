#!/usr/bin/env python3
"""Search the scripted team's settings for those that do best against a team of learned
players: how a benchmark stronger than Scripted 2 is found (scripted.ScriptedTeam3 has
the settings this came to).

Each generation draws --population sets of settings around the best so far (the first
around Scripted 2's), plays each --points points against the team in --against, the same
points for all, and moves towards the --elite that did best (scored less conceded). The
best of the last generation are then played --final points each, on points of their own,
with Scripted 2 beside them to compare. Every generation is printed, and all of it kept
in <output>/<time>_scripted/search.json.

    python scripts/tune_scripted.py --against data/models/movement/<run>/iter_0440
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402

# What is searched: the setting, the least and the most it may be
SETTINGS = (
    ("stack_gap", 4.0, 12.0),
    ("stack_from", 6.0, 20.0),
    ("stack_across", 0.0, 14.0),
    ("cut_every", 1.5, 4.0),
    ("cut_seconds", 1.0, 3.5),
    ("jab", 0.0, 0.8),
    ("shortest_hold", 0.3, 2.5),
    ("open_enough", 0.6, 0.97),
    ("clear_of", 1.0, 5.0),
    ("huck_from", 15.0, 45.0),
    ("huck_open", 0.85, 0.99),
    ("length_cost", 0.0, 0.12),
    ("goal_side", 0.5, 3.0),
    ("force_side", 0.0, 2.5),
    ("deep_goal_side", 0.0, 8.0),
    ("late_from", 3.0, 8.0),
    ("last_resort", 5.0, 9.5),
)
NAMES = [name for name, _, _ in SETTINGS]
LOW = np.array([low for _, low, _ in SETTINGS])
HIGH = np.array([high for _, _, high in SETTINGS])
POINT_SECONDS = 120.0


def settings_of(shares: np.ndarray) -> dict:
    """Settings from their shares of the way between the least and the most."""
    found = dict(zip(NAMES, (LOW + np.clip(shares, 0.0, 1.0) * (HIGH - LOW)).tolist()))
    found["cut_seconds"] = min(found["cut_seconds"], found["cut_every"])
    found["last_resort"] = max(found["last_resort"], found["late_from"] + 0.5)
    return found


def play(job) -> tuple:
    """(scored, conceded) by a scripted team of these settings over the points of `seeds`."""
    settings, plays, against, seeds = job
    import torch

    from ultimate_analysis.simulation import game
    from ultimate_analysis.simulation.match import play_point
    from ultimate_analysis.simulation.scripted import ScriptedTeam2
    from ultimate_analysis.simulation.spots import SpotTeam, load_spot_brains

    torch.set_num_threads(1)
    brains = load_spot_brains(against)
    scored = conceded = 0
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        side = k % 2
        ours = ScriptedTeam2(seed, plays=plays)
        for name, value in settings.items():
            setattr(ours, name, value)
        theirs = SpotTeam(brains, 1 - side, rng, record=False)
        teams = (ours, theirs) if side == 0 else (theirs, ours)
        start = game.pull_start(rng, int(rng.integers(2)), game.a_wind(rng))
        point = play_point(teams, rng, start, limit=POINT_SECONDS)
        scored += int(point.scored == side)
        conceded += int(point.scored == 1 - side)
    return scored, conceded


def results(pool, candidates, against, seeds, parts) -> list:
    """(scored, conceded) of each candidate ((settings, plays)) over the same points."""
    chunks = np.array_split(np.array(seeds), parts)
    jobs = [
        (settings, plays, against, [int(s) for s in chunk])
        for settings, plays in candidates
        for chunk in chunks
    ]
    found = pool.map(play, jobs)
    return [
        tuple(int(sum(part[i] for part in found[n * parts : (n + 1) * parts])) for i in (0, 1))
        for n in range(len(candidates))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--against", type=Path, required=True, help="A folder with a spots.pt")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--points", type=int, default=96, help="A set of settings plays")
    parser.add_argument("--final", type=int, default=480, help="Points for the best at the end")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/models/scripted"))
    args = parser.parse_args()

    from ultimate_analysis.simulation.scripted import ScriptedTeam2

    rng = np.random.default_rng(args.seed)
    scripted2 = {name: float(getattr(ScriptedTeam2, name)) for name in NAMES}
    mean = (np.array([scripted2[name] for name in NAMES]) - LOW) / (HIGH - LOW)
    spread = np.full(len(NAMES), 0.25)
    plays_chance = 0.5
    run = args.output / f"{datetime.now():%Y%m%d_%H%M%S}_scripted"
    run.mkdir(parents=True)
    print(f"Written to {run}", flush=True)
    kept = {"against": str(args.against), "scripted 2": scripted2, "generations": []}
    parts = max(1, min(args.workers, args.points // 24))
    with Pool(args.workers) as pool:
        for generation in range(1, args.generations + 1):
            began = time.perf_counter()
            shares = np.clip(rng.normal(mean, spread, (args.population, len(NAMES))), 0.0, 1.0)
            plays = rng.random(args.population) < plays_chance
            if generation == 1:  # Scripted 2 itself among the first, with and without plays
                shares[0], plays[0] = mean, False
                shares[1], plays[1] = mean, True
            candidates = [(settings_of(s), bool(p)) for s, p in zip(shares, plays)]
            seeds = rng.integers(0, 2**31, args.points)
            found = results(pool, candidates, str(args.against), seeds, parts)
            margins = np.array([scored - conceded for scored, conceded in found])
            order = np.argsort(-margins)
            elite = order[: args.elite]
            mean = shares[elite].mean(0)
            spread = np.maximum(shares[elite].std(0), 0.05)
            plays_chance = float(np.clip(plays[elite].mean(), 0.1, 0.9))
            best = int(order[0])
            print(
                f"generation {generation}: the best scored {found[best][0]}, conceded "
                f"{found[best][1]} of {args.points} (plays {'on' if plays[best] else 'off'}); "
                f"the middle one {int(np.median(margins)):+d}; {time.perf_counter() - began:.0f} s",
                flush=True,
            )
            kept["generations"].append(
                [
                    {"settings": c[0], "plays": c[1], "scored": f[0], "conceded": f[1]}
                    for c, f in zip(candidates, found)
                ]
            )
            (run / "search.json").write_text(json.dumps(kept, indent=2))
        # The best of the last generation and what the search settled on, on points of
        # their own, with Scripted 2 to compare
        finalists = [candidates[int(i)] for i in elite]
        finalists.append((settings_of(mean), plays_chance >= 0.5))
        finalists.append((scripted2, False))
        seeds = rng.integers(0, 2**31, args.final)
        found = results(pool, finalists, str(args.against), seeds, args.workers)
    kept["final"] = [
        {"settings": c[0], "plays": c[1], "scored": f[0], "conceded": f[1], "points": args.final}
        for c, f in zip(finalists, found)
    ]
    (run / "search.json").write_text(json.dumps(kept, indent=2))
    names = [f"best {n + 1}" for n in range(args.elite)] + ["settled on", "scripted 2"]
    print(f"\nOver {args.final} points each:")
    for name, (scored, conceded) in zip(names, found):
        print(f"  {name}: scored {scored}, conceded {conceded}", flush=True)
    best = int(np.argmax([s - c for s, c in found[:-1]]))
    print(f"\nThe best ({names[best]}), plays {'on' if finalists[best][1] else 'off'}:")
    for name, value in finalists[best][0].items():
        print(f"  {name} = {value:.3g}")


if __name__ == "__main__":
    main()
