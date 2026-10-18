#!/usr/bin/env python3
"""Which frames a benchmark is scored on lie near frames a network learned from.

A frame a network was trained on, or one a few seconds from it in the same game, says
little about how the network does on a game it has not seen: the same players, light and
places. This lists, for the frames of the benchmark datasets, the nearest frame of the
same video in the training part of the training datasets, and counts how many lie
within a given time of one.

Frames are told by their file names (`<video>_frame_<number>`, with or without a prefix
such as `labelled_`). The training part of a dataset is its `train` folder, or the
images listed in its `train.txt`; with --valid-too the validation part counts as well
(it chooses which epoch's weights are kept).

    python scripts/check_splits.py --benchmark hand_players_discs hand_discs \\
        --train train_discs train_disc_frames train_disc_frames_full

Reads the datasets in data/raw/training_data and the frame rates of data/raw/videos;
writes nothing unless --write is given (a CSV of every benchmark frame).
"""

import argparse
import csv
import os
import re
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402

DATA = REPO / DEFAULT_PATHS["TRAINING_DATA"]
VIDEOS = REPO / "data" / "raw" / "videos"
DEFAULT_RATE = 60.0
NEAR_SECONDS = 10.0

Frame = Tuple[str, int]


def frame_of(name: str) -> Optional[Frame]:
    """(video as named in the file, frame number), or None for a name of another kind.

    A frame of a clip cut from a game (`<game>_snippet_<n>_<first frame>`) is given as
    the game's frame, the clip taken to begin at the frame its name ends in."""
    stem = Path(name).stem
    video, _, number = stem.rpartition("_frame_")
    if not video or not number.isdigit():
        return None
    clip = re.match(r"(.+)_snippet_\d+_(\d+)$", video)
    if clip:
        return clip.group(1), int(clip.group(2)) + int(number)
    return video, int(number)


def _names(folder: Path) -> Iterable[str]:
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_file():
                yield entry.name


def frames_of(dataset: Path, parts: Tuple[str, ...]) -> List[Frame]:
    """The frames of the given parts ("train", "valid", "test") of a dataset; all of
    them for a dataset that is not split (a hand-labelled one)."""
    found: List[Frame] = []
    split_folders = [dataset / part for part in ("train", "valid", "val", "test")]
    if any(folder.is_dir() for folder in split_folders):
        for part in parts:
            for name in ("valid", "val") if part == "valid" else (part,):
                labels = dataset / name / "labels"
                folder = labels if labels.is_dir() else dataset / name / "images"
                if folder.is_dir():
                    found += [f for f in map(frame_of, _names(folder)) if f]
    elif any((dataset / f"{part}.txt").exists() for part in ("train", "val", "test")):
        for part in parts:
            for name in ("val", "valid") if part == "valid" else (part,):
                listing = dataset / f"{name}.txt"
                if listing.exists():
                    lines = listing.read_text(encoding="utf-8").splitlines()
                    found += [f for f in map(frame_of, lines) if f]
    else:
        folder = dataset / "labels" if (dataset / "labels").is_dir() else dataset / "images"
        found += [f for f in map(frame_of, _names(folder)) if f]
    return found


def frame_rate(video: str) -> float:
    """Frames a second of a video in data/raw/videos (60 if it is not there)."""
    path = VIDEOS / f"{video}.mp4"
    if not path.exists():
        return DEFAULT_RATE
    import cv2

    capture = cv2.VideoCapture(str(path))
    rate = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_RATE
    capture.release()
    return float(rate)


def nearest_training(
    frames: Iterable[Frame], training: Iterable[Frame]
) -> Dict[Frame, Optional[int]]:
    """Benchmark frame -> how many frames away the nearest training frame of the same
    video is (None: none of that video). A training file named with a prefix before the
    video's name (`labelled_<video>`) is of that video."""
    frames = list(frames)
    videos = {video for video, _ in frames}
    by_video: Dict[str, List[int]] = defaultdict(list)
    for video, number in training:
        names = [video] if video in videos else [v for v in videos if video.endswith("_" + v)]
        for name in names:
            by_video[name].append(number)
    for numbers in by_video.values():
        numbers.sort()
    away: Dict[Frame, Optional[int]] = {}
    for video, number in frames:
        numbers = by_video.get(video)
        if not numbers:
            away[(video, number)] = None
            continue
        at = bisect_left(numbers, number)
        near = [abs(numbers[i] - number) for i in (at - 1, at) if 0 <= i < len(numbers)]
        away[(video, number)] = min(near)
    return away


def seen_frames(
    frames: Iterable[Frame],
    train_datasets: Iterable[str],
    seconds: float = NEAR_SECONDS,
    valid_too: bool = False,
) -> Set[Frame]:
    """The benchmark frames that lie within `seconds` of a training frame of their video."""
    parts = ("train", "valid") if valid_too else ("train",)
    training = [f for name in train_datasets for f in frames_of(DATA / name, parts)]
    away = nearest_training(frames, training)
    rates: Dict[str, float] = {}
    seen = set()
    for (video, number), frames_away in away.items():
        if frames_away is None:
            continue
        if video not in rates:
            rates[video] = frame_rate(video)
        if frames_away <= seconds * rates[video]:
            seen.add((video, number))
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--benchmark", nargs="+", required=True, help="Datasets scored on")
    parser.add_argument("--train", nargs="+", required=True, help="Datasets learned from")
    parser.add_argument("--seconds", type=float, default=NEAR_SECONDS)
    parser.add_argument("--valid-too", action="store_true", help="Validation parts count too")
    parser.add_argument("--write", type=Path, help="CSV of every benchmark frame")
    parser.add_argument(
        "--test-only", action="store_true", help="Only the test part of the benchmark datasets"
    )
    args = parser.parse_args()

    scored = ("test",) if args.test_only else ("train", "valid", "test")
    frames = sorted({f for name in args.benchmark for f in frames_of(DATA / name, scored)})
    parts = ("train", "valid") if args.valid_too else ("train",)
    training = [f for name in args.train for f in frames_of(DATA / name, parts)]
    away = nearest_training(frames, training)
    rates = {video: frame_rate(video) for video in {v for v, _ in frames}}

    same = sum(1 for d in away.values() if d == 0)
    near = sum(1 for (v, _), d in away.items() if d is not None and d <= args.seconds * rates[v])
    print(f"{len(frames)} benchmark frames, {len(training)} training frames")
    print(f"  the very frame trained on: {same}")
    print(f"  a training frame within {args.seconds:g} s: {near}")
    print(f"  none within {args.seconds:g} s (unseen): {len(frames) - near}")
    games = defaultdict(lambda: [0, 0])
    for (video, number), d in away.items():
        games[video][0] += 1
        games[video][1] += d is not None and d <= args.seconds * rates[video]
    print("By game (frames, of which near a training frame):")
    for video, (count, close) in sorted(games.items()):
        print(f"  {video[:48]:48} {count:5d} {close:5d}")
    if args.write:
        with open(args.write, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["video", "frame", "frames_to_nearest_training", "near"])
            for (video, number), d in sorted(away.items()):
                close = d is not None and d <= args.seconds * rates[video]
                writer.writerow([video, number, "" if d is None else d, int(close)])


if __name__ == "__main__":
    main()
