#!/usr/bin/env python3
"""Train the network that tells players apart by their looks (a prototype).

Reads a dataset made by `build_reid_dataset.py`. Who is on a crop: the same jersey number
in the same kind of shirt in one video is one player across its stretches; without a
number, a track within its stretch is.

Each step takes some players and several crops of each. The network is to put the crops
of one player nearer to each other than to any other player's (the hardest pair of each
kind in the step counts), and to say which of the known players a crop shows. Players
of one stretch are taken together: telling a player from the teammates around them is
the task, and those are the ones that look alike.

The games named with --validate are not trained on; they are scored after every few
epochs as `benchmark_reid.py` scores them, and the epoch that does best on them is kept.
Games named with --test are left out altogether, for measuring the kept model on
afterwards. Which network is trained is chosen with --architecture.

    python scripts/train_reid.py reid_players_v1 --validate chicago_vs_san_francisco \
        --test san_francisco_vs_colorado --architecture resnet18_s3

The model goes to data/models/reid/<date>_reid_<architecture>_<dataset>/best.pt. This is
not a YOLO model, so it cannot be trained from the app's Model Training tab.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from benchmark_reid import load_index, picture_of, score, sharp_enough  # noqa: E402
from torch import nn  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.reid import (  # noqa: E402
    ARCHITECTURES,
    CROP_SIZE,
    DEFAULT_ARCHITECTURE,
    MEAN,
    SPREAD,
    PlayerEmbedder,
    fit,
    save_embedder,
)

MARGIN = 0.3  # How much nearer a crop of the same player must be than one of another


def identity_of(row: dict) -> tuple:
    """Who a crop shows, as far as that is known."""
    if row["number"]:
        return (row["video"], "number", row["light_shirt"], row["number"])
    return (row["video"], row["stretch"], row["player"])


def augmented(
    crop: np.ndarray, rng: random.Random, size: tuple = CROP_SIZE, keep_proportions: bool = False
) -> np.ndarray:
    """A crop as the network sees it in training: cut, mirrored, lit and covered a little
    differently each time. The colours are left as they are: they are what tells."""
    height, width = crop.shape[:2]
    cut_x, cut_y = int(0.1 * width), int(0.08 * height)
    left, top = rng.randint(0, cut_x), rng.randint(0, cut_y)
    right, bottom = width - rng.randint(0, cut_x), height - rng.randint(0, cut_y)
    crop = crop[top:bottom, left:right]
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    picture = fit(np.ascontiguousarray(crop), size, keep_proportions).astype(np.float32)
    picture = picture * rng.uniform(0.8, 1.2) + rng.uniform(-12, 12)  # Sun and shadow
    if rng.random() < 0.4:
        # Something in front: another player's arm, the disc
        box_w = rng.randint(round(0.15 * size[0]), round(0.47 * size[0]))
        box_h = rng.randint(round(0.12 * size[1]), round(0.35 * size[1]))
        x, y = rng.randint(0, size[0] - box_w), rng.randint(0, size[1] - box_h)
        picture[y : y + box_h, x : x + box_w] = rng.uniform(60, 160)
    rgb = np.clip(picture, 0, 255)[:, :, ::-1] / 255.0
    return ((rgb - MEAN) / SPREAD).transpose(2, 0, 1).astype(np.float32)


def hardest_triplets(vectors: torch.Tensor, who: torch.Tensor) -> torch.Tensor:
    """For each crop: its furthest crop of the same player must be nearer than its
    nearest crop of another, by a margin."""
    distance = torch.cdist(vectors, vectors)
    same = who[:, None] == who[None, :]
    furthest_same = (distance - 1e6 * (~same)).max(dim=1).values
    nearest_other = (distance + 1e6 * same).min(dim=1).values
    return torch.relu(furthest_same - nearest_other + MARGIN).mean()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="Folder in data/raw/training_data")
    parser.add_argument("--validate", nargs="+", required=True, help="Videos to choose by")
    parser.add_argument("--test", nargs="+", default=[], help="Videos left out altogether")
    parser.add_argument(
        "--architecture", default=DEFAULT_ARCHITECTURE, choices=sorted(ARCHITECTURES)
    )
    parser.add_argument("--parts", type=int, default=2, help="Bands from head to foot")
    parser.add_argument("--from-nothing", action="store_true", help="No ImageNet weights")
    parser.add_argument(
        "--size", type=int, nargs=2, default=list(CROP_SIZE), help="Width and height of a crop"
    )
    parser.add_argument(
        "--keep-proportions",
        action="store_true",
        help="Scale crops to fit and pad them, where they are otherwise stretched",
    )
    parser.add_argument(
        "--skip-blurred",
        type=float,
        default=0.0,
        help="Share of each track's crops, the most blurred, that is not used",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=150, help="Steps per epoch")
    parser.add_argument("--players", type=int, default=16, help="Players per step")
    parser.add_argument("--crops", type=int, default=4, help="Crops per player and step")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.dataset
    rows = load_index(dataset)
    held_out = [row for row in rows if any(part in row["video"] for part in args.validate)]
    training = [
        row for row in rows if not any(part in row["video"] for part in args.validate + args.test)
    ]
    if not held_out or not training:
        sys.exit("Both the training and the held-out part need crops")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    # Player -> stretch -> crops. A player known by number is in several stretches, and
    # the crops of a step are taken from as many of them as there are: matching a
    # player from one point to another is what this is for, not from one frame to the next
    crops_of = defaultdict(lambda: defaultdict(list))
    for row, sharp in zip(training, sharp_enough(training, args.skip_blurred)):
        if sharp:
            crops_of[identity_of(row)][row["stretch"]].append(row)
    crops_of = {
        who: dict(stretches)
        for who, stretches in crops_of.items()
        if sum(len(paths) for paths in stretches.values()) >= args.crops
    }
    identities = sorted(crops_of, key=str)
    number_of = {who: index for index, who in enumerate(identities)}
    # Players that are seen together: those of one stretch, and a numbered player with
    # every stretch they are in
    together = defaultdict(set)
    for row in training:
        who = identity_of(row)
        if who in crops_of:
            together[(row["video"], row["stretch"])].add(who)
    groups = [sorted(group, key=str) for group in together.values() if len(group) >= 4]
    print(
        f"{len(training)} crops of {len(identities)} players to train on "
        f"({sum(1 for who in identities if who[1] == 'number')} known by number, "
        f"{sum(1 for who in identities if len(crops_of[who]) > 1)} of them in several stretches), "
        f"{len(held_out)} crops held out",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PlayerEmbedder(
        args.architecture,
        not args.from_nothing,
        len(identities),
        args.parts,
        tuple(args.size),
        args.keep_proportions,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, total_steps=args.epochs * args.steps
    )
    which_player = nn.CrossEntropyLoss(label_smoothing=0.1)
    name = args.architecture + (f"_{args.parts}parts" if args.parts != 2 else "")
    name += "_from_nothing" if args.from_nothing and args.architecture != "small_cnn" else ""
    name += f"_{args.size[0]}x{args.size[1]}" if tuple(args.size) != CROP_SIZE else ""
    name += "_proportions" if args.keep_proportions else ""
    name += f"_sharp{round(100 * (1 - args.skip_blurred))}" if args.skip_blurred else ""
    output = (
        REPO / "data" / "models" / "reid" / f"{time.strftime('%Y%m%d')}_reid_{name}_{args.dataset}"
    )
    output.mkdir(parents=True, exist_ok=True)
    best, best_line = -1.0, {}
    history = []
    began = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.steps):
            chosen = []
            while len(chosen) < args.players:
                group = rng.choice(groups)
                chosen += rng.sample(group, min(len(group), args.players - len(chosen), 8))
            batch, who = [], []
            for player in chosen:
                stretches = list(crops_of[player])
                rng.shuffle(stretches)
                for index in range(args.crops):
                    row = rng.choice(crops_of[player][stretches[index % len(stretches)]])
                    batch.append(
                        augmented(
                            picture_of(row, args.keep_proportions),
                            rng,
                            tuple(args.size),
                            args.keep_proportions,
                        )
                    )
                    who.append(number_of[player])
            batch = torch.from_numpy(np.stack(batch)).to(device)
            who = torch.tensor(who, device=device)
            vectors = model(batch)
            # The vectors have length 1: scaled up before a player is read from them
            loss = hardest_triplets(vectors, who) + which_player(model.who(vectors) * 16.0, who)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            losses.append(float(loss))

        line = {"epoch": epoch, "loss": round(float(np.mean(losses)), 3)}
        if epoch % 3 == 0 or epoch == args.epochs:
            model.eval()
            results = score(model, held_out, device, args.skip_blurred)
            line.update({key: round(value, 3) for key, value in results.items()})
            # What is aimed at: the right teammate from one point to another. Within a
            # point only decides between epochs that are level on that.
            aim = (
                results.get("across_points_teammates", 0.0) + 0.01 * results["same_point_teammates"]
            )
            if aim > best:
                best, best_line = aim, line
                save_embedder(model, output / "best.pt")
        history.append(line)
        print(json.dumps(line), flush=True)

    (output / "training.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "architecture": args.architecture,
                "parts": args.parts,
                "pretrained": not args.from_nothing,
                "size": args.size,
                "keep_proportions": args.keep_proportions,
                "skip_blurred": args.skip_blurred,
                "validated_on": args.validate,
                "left_out": args.test,
                "parameters": sum(p.numel() for n, p in model.named_parameters() if "who" not in n),
                "minutes": round((time.perf_counter() - began) / 60, 1),
                "best": best_line,
                "history": history,
            },
            indent=2,
        )
    )
    with open(output / "players.csv", "w", newline="") as file:
        csv.writer(file).writerows([[index, *who] for who, index in number_of.items()])
    print(f"Best model: {output / 'best.pt'}")


if __name__ == "__main__":
    main()
