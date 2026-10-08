#!/usr/bin/env python3
"""Build a dataset of player crops for telling players apart by their looks.

Jersey numbers are read for some players some of the time. To know who is who across the
points of a game, a player must be recognised by what else sets them apart: cleats,
hat, socks, skin, build. This collects pictures to learn that from.

Stretches spread over each video go through the analysis pipeline. Every player the
tracker follows gives a crop a few times a second, when no other player overlaps them.
Who is on a crop is known in two ways:

- within a stretch, by the tracker: the crops of one track show one player (as far as
  the tracker is right)
- across stretches, by the jersey number: the same number in the same kind of shirt
  (the lighter or the darker of the two teams) is the same player. This is what the
  dataset is for: matching a player from one point of a game to another. So the
  stretches should be many and long enough for numbers to be read (a number and how
  certain it is are written for every player who has one; what counts as certain is
  decided when the data is used).

The videos are read-only; the result is written to a new folder:

    <dataset>/crops/<video>/<stretch>/<player>_<frame>.jpg
    <dataset>/index.csv    video, stretch, frame, player, light_shirt, number, certainty,
                           x1, y1, x2, y2

A player is named <part>-<ID>: the part of the stretch between two cuts, and the
tracker's ID in it.

    python scripts/build_reid_dataset.py reid_players_v2 --videos chicago portland
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.jersey_tracker import get_best_jersey_number  # noqa: E402
from ultimate_analysis.processing.player_id import discard_pending_readings  # noqa: E402

MIN_HEIGHT = 50  # Pixels: smaller players show nothing to tell them by
MAX_OVERLAP = 0.05  # IoU with any other player above which a crop shows two people
PAD = 0.08  # Share of the box added around it: feet and a raised arm are often cut off
MIN_CROPS = 6  # A track with fewer crops is left out
NUMBER_CERTAINTY = 0.6


def overlaps(boxes: np.ndarray) -> np.ndarray:
    """For each box the largest IoU with another."""
    if len(boxes) < 2:
        return np.zeros(len(boxes))
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    shared = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = shared / (area[:, None] + area[None, :] - shared)
    np.fill_diagonal(iou, 0.0)
    return iou.max(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="Name of the new folder in data/raw/training_data")
    parser.add_argument("--videos", nargs="+", required=True, help="Videos (parts of names)")
    parser.add_argument("--stretches", type=int, default=20, help="Stretches per video")
    parser.add_argument("--seconds", type=float, default=45.0, help="Length of a stretch")
    parser.add_argument("--every", type=float, default=0.25, help="Seconds between crops")
    args = parser.parse_args()

    output = REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.name
    if output.exists():
        sys.exit(f"Refusing to overwrite existing folder: {output}")
    videos = [
        path
        for path in sorted((REPO / DEFAULT_PATHS["RAW_VIDEOS"]).glob("*.mp4"))
        if any(part in path.name for part in args.videos)
    ]
    if not videos:
        sys.exit("No video is called like that")
    (output / "crops").mkdir(parents=True)

    # Written stretch by stretch: a build takes hours, and what is done should be usable
    index_file = open(output / "index.csv", "w", newline="")
    index_writer = csv.writer(index_file)
    index_writer.writerow(
        "video stretch frame player light_shirt number certainty x1 y1 x2 y2".split()
    )
    written = 0
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        rate = capture.get(cv2.CAP_PROP_FPS) or 30.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Every frame of a 30 fps video, every second one of a 60 fps one
        step = max(1, round(rate / 30.0))
        length = int(args.seconds * rate)
        starts = np.linspace(0.08 * count, 0.92 * count - length, args.stretches).astype(int)
        for stretch, first in enumerate(starts):
            pipeline = AnalysisPipeline()
            pipeline.new_video(str(video))
            pipeline.set_frame_rate(rate)
            options = PipelineOptions(top_down_view=False)
            folder = output / "crops" / video.stem / f"{stretch:02d}"
            folder.mkdir(parents=True)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(first))
            taken = defaultdict(list)  # Player -> rows, without the number yet
            last_crop = {}
            shirts = defaultdict(list)
            numbers = {}
            # The pipeline starts again at a cut or after a close-up, and then counts its
            # players from one again: a player is known by the part of the stretch too
            part = [0]
            start_again = pipeline.reset

            def note_numbers() -> None:
                for player in taken:
                    if player.startswith(f"{part[0]}-"):
                        number, certainty = get_best_jersey_number(int(player.split("-")[1]))
                        if number:
                            numbers[player] = (number, round(float(certainty), 2))

            def counted_reset() -> None:
                note_numbers()
                part[0] += 1
                start_again()

            pipeline.reset = counted_reset
            for index in range(int(first), int(first) + length):
                ok, frame = capture.read()
                if not ok:
                    break
                if (index - first) % step:
                    continue
                result = pipeline.process(frame, index, options)
                players = [t for t in result.tracks if t.class_name == "player"]
                if not result.wide_shot or not players:
                    continue
                boxes = np.array([t.to_ltrb() for t in players], dtype=np.float64)
                for track, box, overlap in zip(players, boxes, overlaps(boxes)):
                    x1, y1, x2, y2 = box
                    if y2 - y1 < MIN_HEIGHT or overlap > MAX_OVERLAP:
                        continue
                    player = f"{part[0]}-{track.track_id}"
                    if index - last_crop.get(player, -1e9) < args.every * rate:
                        continue
                    pad_x, pad_y = PAD * (x2 - x1), PAD * (y2 - y1)
                    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
                    right = min(frame.shape[1], int(x2 + pad_x))
                    bottom = min(frame.shape[0], int(y2 + pad_y))
                    crop = frame[top:bottom, left:right]
                    name = f"{player}_{index}.jpg"
                    cv2.imwrite(str(folder / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    last_crop[player] = index
                    taken[player].append([index, *(round(float(v), 1) for v in box)])
                    # How light the shirt is: the upper middle of the box
                    shirt = frame[
                        int(y1 + 0.2 * (y2 - y1)) : int(y1 + 0.5 * (y2 - y1)),
                        int(x1 + 0.3 * (x2 - x1)) : int(x1 + 0.7 * (x2 - x1)),
                    ]
                    if shirt.size:
                        shirts[player].append(float(shirt.mean()))
            # The stretch is over: now it is known what number each player has, and
            # which of the two kinds of shirt they wear
            note_numbers()
            lightness = {p: float(np.median(v)) for p, v in shirts.items() if v}
            middle = float(np.median(list(lightness.values()))) if lightness else 0.0
            kept = sure = 0
            for player, crops in taken.items():
                if len(crops) < MIN_CROPS:
                    for index, *_ in crops:
                        (folder / f"{player}_{index}.jpg").unlink(missing_ok=True)
                    continue
                number, certainty = numbers.get(player, ("", ""))
                light = int(lightness.get(player, middle) > middle)
                found = [
                    [video.stem, stretch, index, player, light, number, certainty, *box]
                    for index, *box in crops
                ]
                index_writer.writerows(found)
                written += len(found)
                kept += 1
                sure += number != "" and certainty >= NUMBER_CERTAINTY
            index_file.flush()
            discard_pending_readings()
            print(
                f"{video.stem[:28]} stretch {stretch}: {kept} players, {sure} with a number, "
                f"{sum(len(c) for c in taken.values() if len(c) >= MIN_CROPS)} crops",
                flush=True,
            )
        capture.release()

    index_file.close()
    print(f"Written to {output}: {written} crops")


if __name__ == "__main__":
    main()
