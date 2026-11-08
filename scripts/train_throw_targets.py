#!/usr/bin/env python3
"""Train the network that says where on the field the next throw is likely to go.

It learns from the throw tables of scripts/build_throw_table.py. For each throw the field
is cut into squares of two yards, and the network gives each square a score from what is
known at the release: where the square lies from the thrower and on the field, and how
near the players of each team are to it, as they will be when the disc gets there. The
scores of all squares are made into chances that add up to one; the network learns to
give the square the throw was caught in a high one.

The field is turned so that the thrower's team attacks the far end zone. Where no
line-up showed which end a team attacks, the direction its completed throws went in
over that stretch of the camera says it.

Each game is left out in turn and the network trained on the others; it is measured on
the game left out, against two others: every square alike, and the same network knowing
only where the thrower stands (--features place), and knowing where the thrower's team
stands but not the other (--features offence). Then it is trained on all games but
the last, which chooses the epoch, and kept. Progress is printed per epoch:

- loss of the games learned from and of the one held back to choose the epoch by
- of the throws of that game: how often the catch lies in the likeliest twentieth of
  the field, and the chance given to the square it lies in, against every square alike

    python scripts/train_throw_targets.py data/exports/throws_20261107

Pictures of throws of the games left out are written with the network: the chances
shaded, the thrower's team in blue, the other in red, where the disc was caught as a
white cross.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ultimate_analysis.processing.throw_targets import (  # noqa: E402
    CENTRES,
    COLUMNS,
    END_ZONE,
    LENGTH,
    OFFENCE_FEATURES,
    PLACE_FEATURES,
    PLAYER_FEATURES,
    ROWS,
    SQUARE,
    WIDTH,
    Moment,
    features,
    square_of,
    square_scorer,
    turned,
)

LIVE_STATES = {"live", "unknown", "score"}  # A throw at a score is the one that scored
SHORTEST_THROW_SECONDS = 0.2  # A holder that changes sooner: a box that changed players
EPOCHS = 40
TOP_SHARE = 0.05  # The likeliest twentieth of the field
PICTURES = 12
PIXELS_PER_YARD = 6


@dataclass
class Throw(Moment):
    """A throw, turned so that the thrower's team attacks the far end zone."""

    game: str
    frame: int
    caught: Tuple[float, float]
    gain: float
    completed: bool = True
    attack: int = 1  # The way it was turned by
    defenders: Tuple[int, ...] = ()  # The tracks of the defence, in order


def number(text: str) -> Optional[float]:
    return float(text) if text != "" else None


def attack_by_throws(throws: List[dict]) -> Dict[Tuple[str, str], int]:
    """(restart, team) -> +1 or -1, the way that team's completed throws went in that
    stretch of the camera."""
    went: Dict[Tuple[str, str], float] = defaultdict(float)
    for throw in throws:
        if throw["outcome"] == "complete" and throw["from_y"] != "" and throw["to_y"] != "":
            went[throw["restart"], throw["team"]] += float(throw["to_y"]) - float(throw["from_y"])
    return {key: 1 if total > 0 else -1 for key, total in went.items() if total != 0}


def load_game(folder: Path, report: Dict[str, int]) -> List[Throw]:
    """The throws of one export that can be learned from."""
    with open(folder / "throws.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    players: Dict[str, List[dict]] = defaultdict(list)
    with open(folder / "at_release.csv", newline="") as file:
        for row in csv.DictReader(file):
            players[row["throw"]].append(row)
    by_throws = attack_by_throws(rows)
    throws = []
    for row in rows:
        report["throws"] += 1
        if (
            row["flew"] != "1"
            or float(row["between_seconds"]) < SHORTEST_THROW_SECONDS
            or row["game_state"] not in LIVE_STATES
            or row["from_x"] == ""
            or row["to_x"] == ""
        ):
            continue
        guessed = by_throws.get((row["restart"], row["team"]))
        attack = number(row["attack"])
        if attack is not None and guessed is not None:
            report["attack both known"] += 1
            report["attack throws agree"] += int(attack == guessed)
        if attack is None:
            attack = guessed
            report["attack from throws"] += int(guessed is not None)
        if attack is None:
            continue
        attack = int(attack)
        sides: Dict[str, list] = {"offence": [], "defence": []}
        defenders = []
        for player in players[row["throw"]]:
            if player["side"] in sides:
                x, y = turned(float(player["x"]), float(player["y"]), attack)
                vx = attack * (number(player["vx"]) or 0.0)
                vy = attack * (number(player["vy"]) or 0.0)
                sides[player["side"]].append((x, y, vx, vy))
                if player["side"] == "defence":
                    defenders.append(int(player["player"]))
        thrower = turned(float(row["from_x"]), float(row["from_y"]), attack)
        caught = turned(float(row["to_x"]), float(row["to_y"]), attack)
        if not (0 <= caught[0] < WIDTH and 0 <= caught[1] < LENGTH):
            report["caught off the field"] += 1
            continue
        report["learned from"] += 1
        throws.append(
            Throw(
                thrower,
                np.array(sides["offence"], dtype=np.float32).reshape(-1, 4),
                np.array(sides["defence"], dtype=np.float32).reshape(-1, 4),
                folder.name,
                int(row["release_frame"]),
                caught,
                caught[1] - thrower[1],
                row["outcome"] == "complete",
                attack,
                tuple(defenders),
            )
        )
    return throws


def soft_targets(throws: Sequence[Throw], spread: float = SQUARE) -> np.ndarray:
    """(throws, squares): where each throw was caught, spread over the squares around it
    as far as a place on the field is off."""
    targets = []
    for throw in throws:
        near = np.hypot(*(CENTRES - np.array(throw.caught)).T)
        weights = np.exp(-0.5 * (near / spread) ** 2)
        targets.append(weights / weights.sum())
    return np.array(targets, dtype=np.float32)


@dataclass
class Scores:
    loss: float
    in_top: float  # Share of catches in the likeliest twentieth
    log_chance: float  # Mean log chance of the square caught in, against every square alike
    gain_error: float  # Mean of |expected gain - gain|


def measure(model: torch.nn.Module, x: torch.Tensor, targets: torch.Tensor, throws) -> Scores:
    with torch.no_grad():
        logits = model(x).squeeze(-1)
        log_p = torch.log_softmax(logits, dim=1)
        loss = float(-(targets * log_p).sum(dim=1).mean())
        caught = torch.tensor([square_of(t.caught) for t in throws])
        chance = log_p[torch.arange(len(throws)), caught]
        rank = (log_p > chance[:, None]).sum(dim=1).float() / log_p.shape[1]
        downfield = torch.tensor(CENTRES[:, 1])
        expected = (log_p.exp() * downfield[None, :]).sum(dim=1)
        gains = torch.tensor([t.caught[1] for t in throws])
    return Scores(
        loss,
        float((rank < TOP_SHARE).float().mean()),
        float(chance.mean()) + math.log(len(CENTRES)),
        float((expected - gains).abs().mean()),
    )


def tensors(throws, names, mean=None, std=None):
    x = np.stack([features(throw, names) for throw in throws])
    if mean is None:
        mean, std = x.reshape(-1, x.shape[-1]).mean(0), x.reshape(-1, x.shape[-1]).std(0) + 1e-6
    x = (x - mean) / std
    return torch.tensor(x), torch.tensor(soft_targets(throws)), mean, std


def train(
    learn: List[Throw], check: List[Throw], names: Sequence[str], seed: int, say: bool
) -> Tuple[torch.nn.Module, np.ndarray, np.ndarray, int]:
    """A network trained on `learn`, its epoch chosen by the loss on `check`."""
    torch.manual_seed(seed)
    random.seed(seed)
    x, targets, mean, std = tensors(learn, names)
    check_x, check_targets, _, _ = tensors(check, names, mean, std)
    model = square_scorer(len(names))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, best_epoch = math.inf, None, 0
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(x))
        model.train()
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            loss = (
                -(targets[batch] * torch.log_softmax(model(x[batch]).squeeze(-1), dim=1))
                .sum(1)
                .mean()
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        learned, checked = (
            measure(model, x, targets, learn),
            measure(model, check_x, check_targets, check),
        )
        if checked.loss < best:
            best, best_epoch = checked.loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if say:
            print(
                f"  epoch {epoch:2d}: loss {learned.loss:.3f}, held back {checked.loss:.3f}; "
                f"caught in the likeliest twentieth {checked.in_top:.0%}, "
                f"chance {math.exp(checked.log_chance):.1f} times every square alike",
                flush=True,
            )
    model.load_state_dict(best_state)
    return model, mean, std, best_epoch


def picture(throw: Throw, chances: np.ndarray) -> np.ndarray:
    """The field from above, the far end zone at the top: chances shaded, the thrower's
    team in blue (the thrower ringed), the other in red, the catch a white cross."""
    s = PIXELS_PER_YARD
    height, width = int(LENGTH * s), int(WIDTH * s)

    def at(x: float, y: float) -> Tuple[int, int]:
        return int(x * s), int((LENGTH - y) * s)

    grid = chances.reshape(ROWS, COLUMNS)[::-1]
    shade = (255 * grid / max(grid.max(), 1e-9)).astype(np.uint8)
    shade = cv2.resize(shade, (width, height), interpolation=cv2.INTER_NEAREST)
    image = cv2.applyColorMap(shade, cv2.COLORMAP_INFERNO)
    image = cv2.addWeighted(image, 0.8, np.full_like(image, (40, 90, 40)), 0.2, 0)
    for y in (END_ZONE, LENGTH - END_ZONE):
        cv2.line(image, at(0, y), at(WIDTH, y), (255, 255, 255), 1)
    for players, colour in ((throw.offence, (255, 160, 60)), (throw.defence, (60, 60, 255))):
        for x, y, vx, vy in players:
            cv2.circle(image, at(x, y), 5, colour, -1)
            cv2.line(image, at(x, y), at(x + vx, y + vy), colour, 1)
    cv2.circle(image, at(*throw.thrower), 9, (255, 160, 60), 2)
    cx, cy = at(*throw.caught)
    cv2.drawMarker(image, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("exports", type=Path, help="Folder of the games' exports")
    parser.add_argument("--features", choices=["players", "offence", "place"], default="players")
    parser.add_argument("--output", type=Path, default=Path("data/models/throw_targets"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report: Dict[str, int] = defaultdict(int)
    games = {}
    for folder in sorted(p for p in args.exports.iterdir() if (p / "throws.csv").exists()):
        games[folder.name] = load_game(folder, report)
    print(", ".join(f"{key} {value}" for key, value in report.items()))
    for game, throws in games.items():
        print(f"  {game}: {len(throws)}")
    if len(games) < 3:
        sys.exit("Needs at least three games: one left out, one to choose the epoch by")

    run = args.output / f"{datetime.now():%Y%m%d_%H%M%S}_{args.features}"
    run.mkdir(parents=True)
    names = (
        PLACE_FEATURES
        + {
            "players": PLAYER_FEATURES,
            "offence": OFFENCE_FEATURES,
            "place": [],
        }[args.features]
    )
    order = list(games)
    results: Dict[str, dict] = {}
    began = time.perf_counter()
    for index, left_out in enumerate(order):
        check = order[(index + 1) % len(order)]
        learn = [t for g in order if g not in (left_out, check) for t in games[g]]
        model, mean, std, epoch = train(learn, games[check], names, args.seed, say=False)
        x, targets, _, _ = tensors(games[left_out], names, mean, std)
        scores = measure(model, x, targets, games[left_out])
        results[left_out] = {"throws": len(games[left_out]), "epoch": epoch, **scores.__dict__}
        print(
            f"{left_out}: {len(games[left_out])} throws, epoch {epoch}; caught in the likeliest "
            f"twentieth {scores.in_top:.0%} (alike {TOP_SHARE:.0%}), chance "
            f"{math.exp(scores.log_chance):.1f} times alike, where along the field "
            f"{scores.gain_error:.1f} yd off ({time.perf_counter() - began:.0f} s)",
            flush=True,
        )
        with torch.no_grad():
            chances = torch.softmax(model(x).squeeze(-1), dim=1).numpy()
        each = math.ceil(PICTURES / len(order))
        for i in np.linspace(0, len(games[left_out]) - 1, each).astype(int):
            throw = games[left_out][i]
            cv2.imwrite(str(run / f"{left_out}_{throw.frame}.png"), picture(throw, chances[i]))

    total = sum(r["throws"] for r in results.values())
    summary = {
        key: sum(r[key] * r["throws"] for r in results.values()) / total
        for key in ("in_top", "log_chance", "gain_error")
    }
    print(
        f"All games, each left out: caught in the likeliest twentieth {summary['in_top']:.0%}, "
        f"chance {math.exp(summary['log_chance']):.1f} times alike, along the field "
        f"{summary['gain_error']:.1f} yd off"
    )

    print("On all games:")
    check = order[-1]
    model, mean, std, epoch = train(
        [t for g in order[:-1] for t in games[g]], games[check], names, args.seed, say=True
    )
    torch.save(
        {
            "state": model.state_dict(),
            "names": names,
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "epoch": epoch,
        },
        run / "model.pt",
    )
    (run / "results.json").write_text(
        json.dumps(
            {"features": names, "report": report, "games": results, "all": summary}, indent=2
        )
    )
    print(f"Written to {run}")


if __name__ == "__main__":
    main()
