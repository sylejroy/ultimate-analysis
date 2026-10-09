#!/usr/bin/env python3
"""Train the network that says from a crop of a player whether they hold the disc.

The crops and their marks come from scripts/build_holder_dataset.py. One game is used to
choose the best epoch by and another is left out altogether; the network learns from the
rest. Progress is printed per epoch:

- average precision over all crops
- of the frames in which somebody holds the disc: how often the player the network is
  surest of is the holder (among the players of that frame in the dataset: the holder,
  the one nearest to them, and two others)
- the same for the frames in which no disc was detected on the holder: what the
  network is for

    python scripts/train_holder.py holder_players_v1 --validate chicago_vs_san_francisco \\
        --test san_francisco_vs_colorado
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.holder import (  # noqa: E402
    CROP_SIZE,
    DEFAULT_ARCHITECTURE,
    HolderNet,
    holding,
    save_holder_net,
)
from ultimate_analysis.processing.reid import MEAN, SPREAD, fit  # noqa: E402

# At this speed over the ground (field units a second) a player is half as likely to be
# the holder as one who stands, going by speed alone
STANDING_SPEED = 3.0


def load_index(dataset: Path) -> List[dict]:
    rows = []
    with open(dataset / "index.csv", newline="") as file:
        for row in csv.DictReader(file):
            for key in ("stretch", "frame", "holding", "disc_seen", "nearest"):
                row[key] = int(row[key])
            # Field units a second, or None where the field's place was not known
            row["speed"] = float(row["speed"]) if row.get("speed") else None
            # A stretch is a folder of two digits, a labelled frame one of six
            name = f"{row['player']}_{row['frame']}.jpg"
            folders = [dataset / "crops" / row["video"] / f"{row['stretch']:0{n}d}" for n in (2, 6)]
            row["path"] = next(
                (f / name for f in folders if (f / name).exists()), folders[0] / name
            )
            rows.append(row)
    return rows


def pictures_of(rows: List[dict]) -> List[np.ndarray]:
    """The crops of some rows, read once and kept at about the size they are used at."""
    pictures = []
    for row in rows:
        crop = cv2.imread(str(row["path"]))
        if crop is None:
            raise FileNotFoundError(f"No crop at {row['path']}")
        pictures.append(cv2.resize(crop, (CROP_SIZE[0] + 12, CROP_SIZE[1] + 16)))
    return pictures


def augmented(crop: np.ndarray, rng: random.Random, size: tuple) -> np.ndarray:
    """A crop as the network sees it in training: cut, mirrored and lit a little
    differently each time. Nothing is painted over: the disc is small."""
    height, width = crop.shape[:2]
    cut_x, cut_y = int(0.1 * width), int(0.1 * height)
    left, top = rng.randint(0, cut_x), rng.randint(0, cut_y)
    right, bottom = width - rng.randint(0, cut_x), height - rng.randint(0, cut_y)
    crop = crop[top:bottom, left:right]
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    picture = fit(np.ascontiguousarray(crop), size).astype(np.float32)
    picture = picture * rng.uniform(0.8, 1.2) + rng.uniform(-12, 12)
    rgb = np.clip(picture, 0, 255)[:, :, ::-1] / 255.0
    return ((rgb - MEAN) / SPREAD).transpose(2, 0, 1).astype(np.float32)


def average_precision(sure: np.ndarray, truth: np.ndarray) -> float:
    order = np.argsort(-sure)
    hits = truth[order]
    if not hits.sum():
        return 0.0
    precision = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((precision * hits).sum() / hits.sum())


def standing(speed: float) -> float:
    """How well a speed over the ground (field units a second) goes with holding the
    disc, 0 to 1: a holder stands or pivots; 1 where the speed is not known."""
    return 1.0 if speed is None else 1.0 / (1.0 + (speed / STANDING_SPEED) ** 2)


def score(model: HolderNet, rows: List[dict], pictures: List[np.ndarray]) -> Dict[str, float]:
    """How well a network tells the holders on some crops: by their looks, by their
    looks and whether they stand, and by whether they stand alone."""
    by_looks = holding(model, pictures)
    by_speed = np.array([standing(row["speed"]) for row in rows])
    result = {}
    for title, sure in (
        ("", by_looks),
        ("_with_speed", by_looks * by_speed),
        ("_speed_alone", by_speed),
    ):
        for key, value in _score(sure, rows).items():
            result[key + title] = value
    return result


def _score(sure: np.ndarray, rows: List[dict]) -> Dict[str, float]:
    truth = np.array([row["holding"] for row in rows])
    by_frame = defaultdict(list)
    for index, row in enumerate(rows):
        by_frame[(row["video"], row["stretch"], row["frame"])].append(index)
    picked = {"all": [], "disc not seen": [], "disc seen": []}
    for members in by_frame.values():
        holders = [i for i in members if truth[i]]
        if not holders or len(members) < 2:
            continue
        right = float(int(np.argmax(sure[members])) == members.index(holders[0]))
        picked["all"].append(right)
        picked["disc seen" if rows[holders[0]]["disc_seen"] else "disc not seen"].append(right)
    hidden = np.array([bool(row["holding"]) and not row["disc_seen"] for row in rows])
    others = truth == 0
    return {
        "average_precision": average_precision(sure, truth),
        "holder_picked": float(np.mean(picked["all"])) if picked["all"] else 0.0,
        "holder_picked_frames": len(picked["all"]),
        "holder_picked_disc_not_seen": (
            float(np.mean(picked["disc not seen"])) if picked["disc not seen"] else 0.0
        ),
        "disc_not_seen_frames": len(picked["disc not seen"]),
        "holder_picked_disc_seen": (
            float(np.mean(picked["disc seen"])) if picked["disc seen"] else 0.0
        ),
        # Of the crops over a half: holders found, and others taken for holders
        "holders_found": float((sure[truth == 1] >= 0.5).mean()) if truth.any() else 0.0,
        "hidden_holders_found": float((sure[hidden] >= 0.5).mean()) if hidden.any() else 0.0,
        "others_taken_for_holders": float((sure[others] >= 0.5).mean()) if others.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="Folder in data/raw/training_data")
    parser.add_argument("--validate", nargs="+", required=True, help="Videos to choose by")
    parser.add_argument("--test", nargs="+", default=[], help="Videos left out altogether")
    parser.add_argument("--architecture", default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--from-nothing", action="store_true", help="No ImageNet weights")
    parser.add_argument(
        "--max-holder-speed",
        type=float,
        default=5.0,
        help="Crops marked as holding of a player faster than this are not learned from",
    )
    parser.add_argument("--also", nargs="*", default=[], help="More datasets to learn from")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.dataset
    rows = load_index(dataset)
    for name in args.also:
        rows += load_index(REPO / DEFAULT_PATHS["TRAINING_DATA"] / name)

    def of(videos: List[str]) -> List[dict]:
        return [row for row in rows if any(part in row["video"] for part in videos)]

    held_out = {id(row) for row in of(args.validate) + of(args.test)}
    training = [
        row
        for row in rows
        if id(row) not in held_out
        # A player at a run marked as holding the disc is a mistake of the marking
        and not (row["holding"] and (row["speed"] or 0.0) > args.max_holder_speed)
    ]
    validation, testing = of(args.validate), of(args.test)
    print(
        f"{len(training)} crops to learn from ({sum(r['holding'] for r in training)} holding), "
        f"{len(validation)} to choose by, {len(testing)} left out",
        flush=True,
    )
    train_pictures = pictures_of(training)
    validation_pictures, test_pictures = pictures_of(validation), pictures_of(testing)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HolderNet(args.architecture, pretrained=not args.from_nothing).to(device)
    positives = [i for i, row in enumerate(training) if row["holding"]]
    negatives = [i for i, row in enumerate(training) if not row["holding"]]
    steps = len(training) // args.batch
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=3e-4, total_steps=args.epochs * steps
    )
    loss_of = nn.BCEWithLogitsLoss()

    name = f"{datetime.now():%Y%m%d}_holder_{args.architecture}_{args.dataset}"
    name += "_from_nothing" if args.from_nothing else ""
    run = REPO / DEFAULT_PATHS["MODELS"] / "holder" / name
    run.mkdir(parents=True, exist_ok=True)
    best, history = None, []
    began = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps):
            # Half of a batch holds the disc: one player in fourteen does
            chosen = rng.choices(positives, k=args.batch // 2) + rng.choices(
                negatives, k=args.batch - args.batch // 2
            )
            batch = torch.from_numpy(
                np.stack([augmented(train_pictures[i], rng, model.size) for i in chosen])
            ).to(device)
            truth = torch.tensor(
                [float(training[i]["holding"]) for i in chosen], dtype=torch.float32, device=device
            )
            loss = loss_of(model(batch), truth)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss)
        model.eval()
        result = {"epoch": epoch, "loss": round(total / steps, 4)}
        result.update(score(model, validation, validation_pictures))
        history.append(result)
        shown = ("epoch", "loss", "average_precision", "holder_picked")
        shown += ("holder_picked_with_speed", "holder_picked_disc_not_seen")
        print(
            json.dumps({key: round(result[key], 3) for key in shown}),
            flush=True,
        )
        if best is None or result["holder_picked"] > best["holder_picked"]:
            best = result
            save_holder_net(model, run / "best.pt")

    from ultimate_analysis.processing.holder import load_holder_net  # noqa: E402

    chosen_model = load_holder_net(run / "best.pt")
    test = score(chosen_model, testing, test_pictures) if testing else {}
    (run / "training.json").write_text(
        json.dumps(
            {
                "architecture": args.architecture,
                "dataset": args.dataset,
                "validate": args.validate,
                "test": args.test,
                "parameters": sum(p.numel() for p in model.parameters()),
                "minutes": round((time.perf_counter() - began) / 60, 1),
                "best": best,
                "left_out": test,
                "history": history,
            },
            indent=2,
        )
    )
    print(
        "Left out:",
        json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in test.items()}),
    )
    print(f"Best model: {run / 'best.pt'}")


if __name__ == "__main__":
    main()
