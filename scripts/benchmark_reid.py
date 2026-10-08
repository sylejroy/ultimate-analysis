#!/usr/bin/env python3
"""Measure how well players are told apart by their looks.

On the crops of a dataset made by `build_reid_dataset.py`, for the videos named. A track
is given one vector (the mean over its crops) and matched with the nearest track of a
set to choose from:

- same point: each track of a stretch is cut in two in time. The later half is matched
  with the earlier halves of all tracks of the stretch. Who is who is known from the
  tracker. This is the easy case: the same light, minutes apart at most.
- across points: a player whose jersey number was read in two stretches. Their track in
  one is matched with all tracks of the other. Who is who is known from the number, so
  there are few of these, and a misread number counts as a miss.

Each is given for all tracks to choose from, and for those in the same kind of shirt
only (teammates): telling the teams apart needs no network.

The players matched across points are those whose number was read, so a network could do
well by recognising the number and nothing else. --cover-number paints over the part of
every crop where a number would be; what is left of the score is what the rest of the
player tells.

Next to the network, two ways without one: the kit colour the tracker already uses
(shirt and shorts, six numbers), and a histogram of the colours of the upper and the
lower half of the crop.

    python scripts/benchmark_reid.py reid_players_v1 --videos san_francisco_vs_colorado \\
        --weights data/models/reid/<run>/best.pt
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing import appearance  # noqa: E402

MIN_CROPS = 8  # A track with fewer crops is not matched: half of them make a vector
NUMBER_CERTAINTY = 0.6  # Of a jersey number, from which it says who a player is
# Crops are kept at this size (width, height): a little larger than the network takes
# them, so that training can cut them a little differently each time
STORED_SIZE = (72, 144)


def stored_pictures(dataset: Path, rows: List[dict]) -> np.ndarray:
    """All crops of a dataset at one size, in the order of its index.

    Reading a hundred thousand small files takes many minutes each time. They are read
    once, brought to one size and kept in one file beside the index. A crop that cannot
    be read is kept as a black picture and its row marked (`readable`).
    """
    store = dataset / f"pictures_{STORED_SIZE[0]}x{STORED_SIZE[1]}.npy"
    readable_file = store.with_name(store.stem + "_readable.npy")
    shape = (len(rows), STORED_SIZE[1], STORED_SIZE[0], 3)
    # The list of what could be read is written last: with it, the store is complete
    if store.exists() and readable_file.exists():
        pictures = np.load(store, mmap_mode="r")
        if pictures.shape == shape:
            for row, readable in zip(rows, np.load(readable_file)):
                row["readable"] = bool(readable)
            return pictures
    print(f"Reading {len(rows)} crops once, into {store.name} ...", flush=True)
    pictures = np.lib.format.open_memmap(store, mode="w+", dtype=np.uint8, shape=shape)
    readable = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        crop = cv2.imread(str(row["path"]))
        if crop is None or crop.size == 0:
            continue
        pictures[index] = cv2.resize(crop, STORED_SIZE, interpolation=cv2.INTER_LINEAR)
        readable[index] = True
    pictures.flush()
    del pictures
    np.save(readable_file, readable)
    print(f"{int((~readable).sum())} crops could not be read and are left out", flush=True)
    for row, is_readable in zip(rows, readable):
        row["readable"] = bool(is_readable)
    return np.load(store, mmap_mode="r")


# With --cover-number: the part of a crop where a number on the shirt would be, as shares
# of its width and height, is painted over in the shirt's own average colour
NUMBER_AREA = (0.2, 0.15, 0.8, 0.5)
cover_numbers = False


def picture_of(row: dict, true_proportions: bool = False) -> np.ndarray:
    """The crop of a row of the index (BGR).

    At the stored size, to which every crop was stretched; or, with `true_proportions`,
    as wide for its height as the player's box was.
    """
    picture = row["pictures"][row["position"]]
    if true_proportions:
        ratio = (float(row["x2"]) - float(row["x1"])) / max(
            float(row["y2"]) - float(row["y1"]), 1.0
        )
        width = int(np.clip(round(picture.shape[0] * ratio), 8, 2 * picture.shape[0]))
        picture = cv2.resize(picture, (width, picture.shape[0]), interpolation=cv2.INTER_LINEAR)
    if cover_numbers:
        picture = picture.copy()
        height, width = picture.shape[:2]
        x1, y1 = int(NUMBER_AREA[0] * width), int(NUMBER_AREA[1] * height)
        x2, y2 = int(NUMBER_AREA[2] * width), int(NUMBER_AREA[3] * height)
        picture[y1:y2, x1:x2] = picture[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)
    return picture


def load_index(dataset: Path) -> List[dict]:
    """The crops of a dataset: the rows of its index, each with where its picture is."""
    rows = []
    with open(dataset / "index.csv", newline="") as file:
        for row in csv.DictReader(file):
            row["stretch"], row["frame"] = int(row["stretch"]), int(row["frame"])
            # A number counts only where the reader was sure of it
            if float(row.get("certainty") or 1.0) < NUMBER_CERTAINTY:
                row["number"] = ""
            row["path"] = (
                dataset
                / "crops"
                / row["video"]
                / f"{row['stretch']:02d}"
                / f"{row['player']}_{row['frame']}.jpg"
            )
            row["position"] = len(rows)
            rows.append(row)
    pictures = stored_pictures(dataset, rows)
    for row in rows:
        row["pictures"] = pictures
    return [row for row in rows if row["readable"]]


def kit_colour(crop: np.ndarray) -> np.ndarray:
    """What the tracker goes by: the mean colour of shirt and of shorts."""
    height, width = crop.shape[:2]
    kit = appearance.encode(crop, [[0, 0, width, height]])[0]
    return np.zeros(6, dtype=np.float32) if kit is None else kit / 100.0


def colour_histogram(crop: np.ndarray) -> np.ndarray:
    """How much of each colour the upper and the lower half of a crop show."""
    hsv = cv2.cvtColor(cv2.resize(crop, (32, 64)), cv2.COLOR_BGR2HSV)
    parts = []
    for half in (hsv[:32], hsv[32:]):
        histogram = cv2.calcHist([half], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
        parts.append(histogram.ravel() / max(float(histogram.sum()), 1.0))
    return np.concatenate(parts).astype(np.float32)


def sharp_enough(rows: List[dict], skip_blurred: float) -> np.ndarray:
    """For each crop whether it counts: within each track, the most blurred share of the
    crops (a player smeared by motion) is left out."""
    from ultimate_analysis.processing.reid import sharpness

    counts = np.ones(len(rows), dtype=bool)
    if skip_blurred <= 0:
        return counts
    tracks = defaultdict(list)
    for index, row in enumerate(rows):
        tracks[(row["video"], row["stretch"], row["player"])].append(index)
    for crops in tracks.values():
        sharp = np.array([sharpness(picture_of(rows[index])) for index in crops])
        counts[np.array(crops)[sharp < np.quantile(sharp, skip_blurred)]] = False
    return counts


def matches(
    vectors: np.ndarray, rows: List[dict], counts: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """The share of tracks matched with the right one, for one vector per crop.

    Args:
        vectors: One vector per crop
        rows: The crops' rows of the index
        counts: For each crop whether it goes into its track's vector (see
            sharp_enough); a track is matched all the same

    Returns:
        same_point_all, same_point_teammates, across_points_all, across_points_teammates
        (shares, the last two missing without a player numbered in two stretches), and
        for each the number of tracks matched (..._count) and how often picking at
        random would be right (..._chance)
    """
    tracks = defaultdict(list)  # (video, stretch, player) -> indices of its crops, in time
    for index, row in sorted(enumerate(rows), key=lambda item: item[1]["frame"]):
        tracks[(row["video"], row["stretch"], row["player"])].append(index)
    tracks = {key: crops for key, crops in tracks.items() if len(crops) >= MIN_CROPS}

    def _mean_vector(crops: List[int]) -> np.ndarray:
        chosen = [index for index in crops if counts is None or counts[index]] or crops
        mean = vectors[chosen].mean(axis=0)
        return mean / max(float(np.linalg.norm(mean)), 1e-12)

    light = {key: rows[crops[0]]["light_shirt"] for key, crops in tracks.items()}
    number = {key: rows[crops[0]]["number"] for key, crops in tracks.items()}
    by_stretch = defaultdict(list)
    for key in tracks:
        by_stretch[key[:2]].append(key)

    hits = defaultdict(list)
    chance = defaultdict(list)

    def match(name: str, query: np.ndarray, gallery: Dict[tuple, np.ndarray], right) -> None:
        for teammates_only, kind in ((False, "all"), (True, "teammates")):
            chosen = {
                key: vector
                for key, vector in gallery.items()
                if not teammates_only or light[key] == right[0]
            }
            if not any(right[1](key) for key in chosen) or len(chosen) < 2:
                continue
            keys = list(chosen)
            distance = np.linalg.norm(np.stack([chosen[key] for key in keys]) - query, axis=1)
            hits[f"{name}_{kind}"].append(float(right[1](keys[int(np.argmin(distance))])))
            chance[f"{name}_{kind}"].append(sum(right[1](key) for key in keys) / len(keys))

    # Same point: the later half of each track against the earlier halves of all
    for keys in by_stretch.values():
        earlier = {key: _mean_vector(tracks[key][: len(tracks[key]) // 2]) for key in keys}
        for key in keys:
            later = _mean_vector(tracks[key][len(tracks[key]) // 2 :])
            match("same_point", later, earlier, (light[key], lambda other, key=key: other == key))

    # Across points: a numbered player's track in one stretch against the tracks of another
    whole = {key: _mean_vector(crops) for key, crops in tracks.items()}
    for key in tracks:
        if not number[key]:
            continue
        for stretch, keys in by_stretch.items():
            if stretch[0] != key[0] or stretch == key[:2]:
                continue

            def same_player(other: tuple, key: tuple = key) -> bool:
                return number[other] == number[key] and light[other] == light[key]

            match(
                "across_points",
                whole[key],
                {other: whole[other] for other in keys},
                (light[key], same_player),
            )

    results = {}
    for name, values in hits.items():
        results[name] = float(np.mean(values))
        results[f"{name}_count"] = len(values)
        results[f"{name}_chance"] = float(np.mean(chance[name]))
    return results


def vectors_by(function: Callable[[np.ndarray], np.ndarray], rows: List[dict]) -> np.ndarray:
    return np.stack([function(picture_of(row)) for row in rows])


def score(
    model, rows: List[dict], device: Optional[str] = None, skip_blurred: float = 0.0
) -> Dict[str, float]:
    """The matches of a network's vectors on some crops (see `matches`)."""
    from ultimate_analysis.processing.reid import embed

    pictures = [picture_of(row, model.keep_proportions) for row in rows]
    return matches(embed(model, pictures), rows, sharp_enough(rows, skip_blurred))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="Folder in data/raw/training_data")
    parser.add_argument("--videos", nargs="+", required=True, help="Videos (parts of names)")
    parser.add_argument("--weights", type=Path, help="A trained network to measure as well")
    parser.add_argument(
        "--cover-number",
        action="store_true",
        help="Paint over where a jersey number would be: how much is told without it?",
    )
    parser.add_argument(
        "--skip-blurred",
        type=float,
        default=0.0,
        help="Share of each track's crops, the most blurred, left out of its vector",
    )
    args = parser.parse_args()
    global cover_numbers
    cover_numbers = args.cover_number

    dataset = REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.dataset
    rows = [row for row in load_index(dataset) if any(part in row["video"] for part in args.videos)]
    if not rows:
        sys.exit("No crops of these videos")
    ways = {
        "kit colour (as the tracker)": matches(vectors_by(kit_colour, rows), rows),
        "colour histogram": matches(vectors_by(colour_histogram, rows), rows),
    }
    if args.weights:
        from ultimate_analysis.processing.reid import load_embedder

        ways["network"] = score(load_embedder(args.weights), rows, None, args.skip_blurred)

    print(f"{len(rows)} crops of {', '.join(sorted({row['video'] for row in rows}))}")
    kinds = (
        ("same_point_all", "Same point, all players"),
        ("same_point_teammates", "Same point, teammates only"),
        ("across_points_all", "Across points, all players"),
        ("across_points_teammates", "Across points, teammates only"),
    )
    first = next(iter(ways.values()))
    print(f"{'':<32}" + "".join(f"{name:>30}" for name in ways) + f"{'at random':>12}{'tracks':>8}")
    for key, title in kinds:
        if key not in first:
            print(f"{title:<32} no player numbered in two stretches")
            continue
        print(
            f"{title:<32}"
            + "".join(f"{result[key]:>30.0%}" for result in ways.values())
            + f"{first[key + '_chance']:>12.0%}{first[key + '_count']:>8}"
        )


if __name__ == "__main__":
    main()
