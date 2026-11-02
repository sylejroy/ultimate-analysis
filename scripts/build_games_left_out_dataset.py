#!/usr/bin/env python3
"""Build a disc dataset that leaves whole games out of training, to be scored on them.

The frames of `train_discs` were split one by one, so every validation and test frame
comes from a game the network also learned from: the same players, light and field. A
network trained on this dataset is scored on games it has never seen, which is what the
app meets with a new video.

    python scripts/build_games_left_out_dataset.py train_discs train_discs_games_out

Every image goes to the split of its game, whichever split it had. The Roboflow export
of Portland v San Francisco is named `s1_frame_...`; `train_disc_frames/index.json` found
all of its frames in that video. The source is read-only; the result is written to a new
dataset directory, with summary.txt saying what went where.
"""

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402

DATA = REPO / DEFAULT_PATHS["TRAINING_DATA"]
VIDEOS = REPO / "data" / "raw" / "videos"
SPLITS = ("train", "valid", "test")
PREFIXES = {"s1_": "portland_vs_san_francisco_2024"}
# Footage of both kinds in the test games, and in the validation games the same; the
# games with the most frames stay in training. Teams do come back in other games.
VALID_GAMES = ("Ring_of_Fire_VS_Sub_Zero", "Truck_Stop_VS_Chain_Lightning")
TEST_GAMES = (
    "chicago_vs_san_francisco_2024",
    "Pacmen_VS_Chain_Lightning",
    "Truck_Stop_VS_Revolver",
)


def game_of(name: str, games: Sequence[str]) -> Optional[str]:
    """The game an image comes from, by its file name; None if the name does not say."""
    for prefix, game in PREFIXES.items():
        if name.startswith(prefix):
            return game
    return next((game for game in games if game in name), None)


def split_of(game: str, valid: Sequence[str], test: Sequence[str]) -> str:
    """The split a game's images go to."""
    if any(game.startswith(part) for part in test):
        return "test"
    if any(game.startswith(part) for part in valid):
        return "valid"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="Dataset in data/raw/training_data")
    parser.add_argument("output", help="New dataset folder")
    parser.add_argument("--valid", nargs="+", default=VALID_GAMES, help="Start of the game names")
    parser.add_argument("--test", nargs="+", default=TEST_GAMES, help="Start of the game names")
    args = parser.parse_args()

    source, output = DATA / args.source, DATA / args.output
    if output.exists():
        sys.exit(f"{output} exists; datasets are written to new folders")
    games = sorted(path.stem for path in VIDEOS.glob("*.mp4"))
    unknown = [
        image.name
        for split in SPLITS
        for image in (source / split / "images").iterdir()
        if game_of(image.stem, games) is None
    ]
    if unknown:
        sys.exit(f"{len(unknown)} images of no known game, such as {unknown[0]}")

    data = yaml.safe_load((source / "data.yaml").read_text())
    data["path"] = str(output)

    for split in SPLITS:
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)
    images: Dict[str, Counter] = defaultdict(Counter)
    discs: Dict[str, Counter] = defaultdict(Counter)
    for split in SPLITS:
        for image in sorted((source / split / "images").iterdir()):
            game = game_of(image.stem, games)
            target = split_of(game, args.valid, args.test)
            shutil.copy2(image, output / target / "images" / image.name)
            images[game][target] += 1
            label = source / split / "labels" / f"{image.stem}.txt"
            if label.exists():
                shutil.copy2(label, output / target / "labels" / label.name)
                discs[game][target] += len(label.read_text().strip().splitlines())

    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    lines = [f"From {args.source}: whole games to validation and test", ""]
    for game in sorted(images):
        target = next(iter(images[game]))
        lines.append(
            f"{target:6s}{images[game][target]:6d} images{discs[game][target]:6d} discs  {game}"
        )
    lines.append("")
    for split in SPLITS:
        count = sum(images[game][split] for game in images)
        found = sum(discs[game][split] for game in images)
        lines.append(f"{split:6s}{count:6d} images{found:6d} discs")
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
