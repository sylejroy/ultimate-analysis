#!/usr/bin/env python3
"""Train the networks that move the simulated players, on how the recorded players moved.

From the dataset of scripts/build_movement_dataset.py: one network for the offence and
one for the defence (simulation/learned.py), each learning from what a player saw how
they ran over the next second, as a change to how they were running. One game chooses the epoch, another is left out
altogether; the networks learn from the rest. Progress is printed per epoch: the mean
error of the velocity, in yards a second, on the games learned from and on the one
choosing the epoch, against keeping on as one was running and against standing still.

How long the recorded throwers held the disc before throwing gives the holder's chance
of throwing at each moment (simulation/learned.py: throw_hazard).

    python scripts/train_movement.py data/raw/training_data/generated_movement_20261108 \\
        --validate portland_vs_san_francisco_2024 --test chicago_vs_new_york_2024
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ultimate_analysis.simulation.game import RUN_SPEED, STEP  # noqa: E402
from ultimate_analysis.simulation.learned import (  # noqa: E402
    Minds,
    movement_network,
    save_minds,
)

EPOCHS = 30
BATCH = 512
HOLD_BINS = 20  # Half seconds


def ran(network, x: torch.Tensor) -> torch.Tensor:
    """How the network has a player run: how they were running, changed."""
    return x[:, 2:4] + network(x)[:, :2]


def errors(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean error of the velocity, in yards a second."""
    return float(np.mean(np.hypot(*((predicted - actual) * RUN_SPEED).T)))


def train_side(name: str, sets: dict, seed: int):
    """A network for one side; its error per set at the epoch chosen."""
    torch.manual_seed(seed)
    x, y = (torch.from_numpy(sets["learn"][i]) for i in (0, 1))
    check_x = torch.from_numpy(sets["validate"][0])
    network = movement_network()
    optimiser = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, best_epoch = np.inf, None, 0
    for epoch in range(1, EPOCHS + 1):
        network.train()
        order = torch.randperm(len(x))
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            loss = ((ran(network, x[batch]) - y[batch]) ** 2).sum(1).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        network.eval()
        with torch.no_grad():
            learned = errors(ran(network, x).numpy(), sets["learn"][1])
            checked = errors(ran(network, check_x).numpy(), sets["validate"][1])
        if checked < best:
            best, best_epoch = checked, epoch
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
        print(
            f"  {name} epoch {epoch:2d}: {learned:.2f} yd/s learned, {checked:.2f} held back",
            flush=True,
        )
    network.load_state_dict(best_state)
    network.eval()
    found = {"epoch": best_epoch}
    with torch.no_grad():
        for part, (px, py) in sets.items():
            found[part] = {
                "network": errors(ran(network, torch.from_numpy(px)).numpy(), py),
                "keep running": errors(px[:, 2:4], py),
                "stand still": errors(np.zeros_like(py), py),
                "samples": len(px),
            }
    return network, found


def throw_hazard(holds: np.ndarray) -> np.ndarray:
    """Chance of throwing within a step, per half second held so far."""
    hazard = np.zeros(HOLD_BINS)
    for b in range(HOLD_BINS):
        start = 0.5 * b
        still = holds >= start
        if still.sum() < 5:  # Too few held this long to say: as the half second before
            hazard[b] = hazard[b - 1] if b else 0.1
            continue
        per_half = (still & (holds < start + 0.5)).sum() / still.sum()
        hazard[b] = 1.0 - (1.0 - per_half) ** (STEP / 0.5)
    return hazard


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--validate", required=True, help="Game that chooses the epoch")
    parser.add_argument("--test", required=True, help="Game left out")
    parser.add_argument("--output", type=Path, default=Path("data/models/movement"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    games = {p.stem: dict(np.load(p)) for p in sorted(args.dataset.glob("*.npz"))}
    parts = {
        "learn": [g for g in games if g not in (args.validate, args.test)],
        "validate": [args.validate],
        "test": [args.test],
    }
    print({part: len(names) for part, names in parts.items()}, "games")
    began = time.perf_counter()
    networks, results = {}, {}
    for side in ("offence", "defence"):
        sets = {
            part: tuple(
                np.concatenate([games[g][f"{k}_{side}"] for g in names]) for k in ("x", "y")
            )
            for part, names in parts.items()
        }
        networks[side], results[side] = train_side(side, sets, args.seed)
        print(f"{side}: {json.dumps(results[side])} ({time.perf_counter() - began:.0f} s)")
    holds = np.concatenate([games[g]["holds"] for g in parts["learn"]])
    hazard = throw_hazard(holds[(holds > 0) & (holds < 15)])
    print("chance of throwing per step, by half second held:", np.round(hazard, 3).tolist())

    run = args.output / f"{datetime.now():%Y%m%d_%H%M%S}_imitation"
    save_minds(Minds(networks["offence"], networks["defence"], hazard), run, parts=parts)
    (run / "results.json").write_text(json.dumps({"parts": parts, **results}, indent=2))
    print(f"Written to {run}")


if __name__ == "__main__":
    main()
