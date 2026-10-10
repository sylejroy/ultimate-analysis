#!/usr/bin/env python3
"""Find the frames of a holder dataset in which the disc is seen to fly.

A holder dataset (scripts/build_holder_sighted_dataset.py) marks a frame as "the disc
flies" where the disc model saw something away from every player twice within a moment.
Of 120 such frames looked at on 10 Oct 2026, 36% showed a disc moving away from the
players; in the others it lay still (on the ground, or a thing that is no disc), was at
a player, or was not found again. To ask somebody about frames in which nobody holds
the disc, those have to be told apart.

Here each such frame is looked at with the frames a moment before and after: the disc
flies if it is found in all three, away from every player in the middle one, and has
moved on both times, the same way. What each frame shows is written down as it is
found, so a run can be stopped and what there is can be used. The games are looked at
in turn, a frame of each.

Reads:  data/raw/training_data/<source>/index.csv, data/raw/videos
Writes: data/cache/flying_frames/<source>.csv (video, stretch, frame, kind, x, y)

    python scripts/find_flying_disc_frames.py holder_sighted_v1 --frames 5000
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.inference import (  # noqa: E402
    detect_discs,
    detect_players,
    load_detection_model,
)
from ultimate_analysis.utils.judgement_files import FLYING, FLYING_COLUMNS  # noqa: E402
from ultimate_analysis.utils.model_files import default_model_path  # noqa: E402

APART = 4  # Frames between the three looked at
MOVED = 0.2  # The disc flies if it moved this share of a player's height each time
ON_PLAYER = 0.15  # A disc is at a player if it is in their box widened by this share


def kind_of(frames: dict, number: int, players: tuple, discs: tuple) -> tuple:
    """What a frame shows of the disc, and where the disc is in it (None if not found)."""
    boxes = [d["bbox"] for d in detect_players(frames[number], *players)]
    height = float(np.median([b[3] - b[1] for b in boxes])) if boxes else 80.0
    seen = {}
    for at, frame in frames.items():
        found = max(detect_discs(frame, *discs), key=lambda d: d["confidence"], default=None)
        if found is not None:
            box = found["bbox"]
            seen[at] = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    if number not in seen:
        return "no disc found", None
    x, y = seen[number]
    for x1, y1, x2, y2 in boxes:
        pad_x, pad_y = ON_PLAYER * (x2 - x1), ON_PLAYER * (y2 - y1)
        if x1 - pad_x <= x <= x2 + pad_x and y1 - pad_y <= y <= y2 + pad_y:
            return "at a player", (x, y)
    if len(seen) < 3:
        return "not found a moment before or after", (x, y)
    before, after = seen[number] - seen[number - APART], seen[number + APART] - seen[number]
    if min(np.linalg.norm(before), np.linalg.norm(after)) < MOVED * height:
        return "lying still", (x, y)
    return (FLYING if float(before @ after) > 0 else "jumping about"), (x, y)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="The holder dataset, in data/raw/training_data")
    parser.add_argument("--frames", type=int, default=5000, help="Frames to look at, at most")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = defaultdict(list)
    index = REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.source / "index.csv"
    with open(index, newline="") as file:
        for row in csv.DictReader(file):
            rows[row["video"], row["stretch"], row["frame"]].append(row["holding"])
    asked = sorted(key for key, holding in rows.items() if "1" not in holding)
    out = REPO / "data" / "cache" / "flying_frames" / f"{args.source}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, newline="") as file:
            done = {(r["video"], r["stretch"], r["frame"]) for r in csv.DictReader(file)}
    else:
        with open(out, "w", newline="") as file:
            csv.DictWriter(file, FLYING_COLUMNS).writeheader()
    left = [key for key in asked if key not in done]
    random.Random(args.seed).shuffle(left)
    # A frame of each game in turn: what there is when a run is stopped, or used while
    # it goes on, is of all games alike
    turn = defaultdict(int)
    order = []
    for key in left:
        order.append((turn[key[0]], key[0], key))
        turn[key[0]] += 1
    left = [key for _, _, key in sorted(order)][: args.frames]
    print(f"{len(asked)} frames without a holder, {len(done)} looked at before", flush=True)

    players = load_detection_model(default_model_path("player_detection"))
    discs = load_detection_model(default_model_path("disc_detection"))
    if players is None or discs is None:
        sys.exit("The player and the disc model are needed")
    videos, kinds = {}, Counter()
    for count, key in enumerate(left):
        video, _, number = key[0], key[1], int(key[2])
        if video not in videos:
            videos[video] = cv2.VideoCapture(
                str(REPO / DEFAULT_PATHS["RAW_VIDEOS"] / f"{video}.mp4")
            )
        videos[video].set(cv2.CAP_PROP_POS_FRAMES, max(0, number - APART))
        frames = {}
        for at in range(max(0, number - APART), number + APART + 1):
            ok, frame = videos[video].read()
            if ok and at in (number - APART, number, number + APART):
                frames[at] = frame
        if number not in frames:
            continue
        kind, place = kind_of(frames, number, players, discs)
        kinds[kind] += 1
        with open(out, "a", newline="") as file:
            csv.DictWriter(file, FLYING_COLUMNS).writerow(
                {
                    "video": key[0],
                    "stretch": key[1],
                    "frame": key[2],
                    "kind": kind,
                    "x": "" if place is None else round(float(place[0]), 1),
                    "y": "" if place is None else round(float(place[1]), 1),
                }
            )
        if count % 100 == 99:
            print(f"{count + 1} of {len(left)} frames: {dict(kinds)}", flush=True)
    print(f"Written to {out}: {dict(kinds)}")


if __name__ == "__main__":
    main()
