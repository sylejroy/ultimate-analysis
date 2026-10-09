#!/usr/bin/env python3
"""Build a dataset of player crops marked as holding the disc or not.

Who has the disc is worked out from where the disc is seen, and the disc is often not
seen: the thrower's body or the mark is in front of it. But a player with the disc
stands and moves like one: pivoting, the arm drawn back, standing while the others run.
This collects pictures to learn that from.

Stretches of play go through the analysis pipeline, which says who holds the disc from
when to when, also over the frames in which no disc is found on them: a player who
caught it and has not thrown it has it. Every player gives a crop a few times a second,
somewhat wider than their box (a held disc is often just outside it), marked:

- holding: the pipeline has them as the holder, and had and will have for a moment
  before and after, so that catches and throws are left out
- not holding: any other player at such a time, or any player while the disc flies.
  The player nearest to the holder, most often the mark, is always among them.

The videos are read-only; the result is written to a new folder:

    <dataset>/crops/<video>/<stretch>/<player>_<frame>.jpg
    <dataset>/index.csv    video, stretch, frame, player, holding, disc_seen (whether a
                           disc was detected on the crop), nearest (the one nearest to
                           the holder), x1, y1, x2, y2

    python scripts/build_holder_dataset.py holder_players_v1 --from-dataset reid_players_v2
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ultimate_analysis.config.settings import get_config  # noqa: E402
from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.holder import (  # noqa: E402
    crop_around,
    disc_on,
    place_on_field,
    speed_of,
)

MIN_HEIGHT = 40  # Pixels: of a smaller player nothing is seen of how they stand
SETTLED_SECONDS = 0.25  # A possession counts this long after it began and before it ends
OTHERS_PER_FRAME = 3  # Players not holding the disc taken from a frame
COLUMNS = ("video", "stretch", "frame", "player", "holding", "disc_seen", "nearest", "speed")


def speed_text(speed: Optional[float]) -> str:
    """A speed for the index: two decimals, or nothing where it cannot be said."""
    return "" if speed is None else f"{speed:.2f}"


def stretches_of(dataset: Path, every: int) -> list:
    """(video, stretch, first frame) of the stretches of another dataset's index."""
    first = {}
    with open(dataset / "index.csv", newline="") as file:
        for row in csv.DictReader(file):
            key = (row["video"], int(row["stretch"]))
            first[key] = min(first.get(key, 10**12), int(row["frame"]))
    return [(v, s, f) for (v, s), f in sorted(first.items()) if s % every == 0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="New folder in data/raw/training_data")
    parser.add_argument("--from-dataset", required=True, help="Take the stretches of this one")
    parser.add_argument("--every", type=int, default=2, help="Every so many of its stretches")
    parser.add_argument("--seconds", type=float, default=30.0, help="Length of a stretch")
    parser.add_argument("--step", type=int, default=2, help="Analyse every so many frames")
    parser.add_argument("--crop-every", type=int, default=3, help="Crops from every so many")
    args = parser.parse_args()

    data = REPO / DEFAULT_PATHS["TRAINING_DATA"]
    out = data / args.dataset
    if out.exists():
        sys.exit(f"{out} exists: datasets are not overwritten, choose another name")
    out.mkdir(parents=True)
    # Nothing of this needs the players' numbers or looks
    config = get_config()
    config["models"]["reid"]["enabled"] = False
    options = PipelineOptions(player_id=False, top_down_view=False)
    rng = random.Random(0)

    with open(out / "index.csv", "w", newline="") as file:
        index = csv.writer(file)
        index.writerow((*COLUMNS, "x1", "y1", "x2", "y2"))
        for video, stretch, start in stretches_of(data / args.from_dataset, args.every):
            capture = cv2.VideoCapture(str(REPO / "data" / "raw" / "videos" / f"{video}.mp4"))
            rate = capture.get(cv2.CAP_PROP_FPS)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            pipeline = AnalysisPipeline()
            pipeline.reset()
            pipeline.set_frame_rate(rate)

            holder_at = {}  # Frame -> who held the disc (None: nobody), as known afterwards
            places = {}  # Frame -> {player: where they stand on the field}
            seen = {}  # Frame -> [(player, box, crop as JPEG, disc seen on it)]
            frames = []
            for step in range(int(args.seconds * rate)):
                ok, frame = capture.read()
                if not ok:
                    break
                if step % args.step:
                    continue
                number = start + step
                result = pipeline.process(frame, number, options)
                if not result.wide_shot:
                    continue
                frames.append(number)
                # Who has the disc is known late and holds from earlier on
                for earlier in frames[::-1]:
                    if earlier < result.possession_since:
                        break
                    holder_at[earlier] = result.holder_id
                places[number] = {
                    track.track_id: place
                    for track in result.tracks
                    if track.class_name == "player"
                    and (place := place_on_field(result.image_to_field, track.to_ltrb()))
                }
                if (step // args.step) % args.crop_every:
                    continue
                discs = [d["bbox"] for d in result.detections if d["class_name"] == "disc"]
                players = []
                for track in result.tracks:
                    if track.class_name != "player":
                        continue
                    x1, y1, x2, y2 = track.to_ltrb()
                    if y2 - y1 < MIN_HEIGHT:
                        continue
                    picture, box = crop_around(frame, (x1, y1, x2, y2))
                    disc_seen = disc_on(box, discs)
                    crop = cv2.imencode(".jpg", picture)[1]
                    players.append((track.track_id, (x1, y1, x2, y2), crop, disc_seen))
                seen[number] = players

            settled = int(SETTLED_SECONDS * rate)
            folder = out / "crops" / video / f"{stretch:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            written = [0, 0]
            for number, players in seen.items():
                around = [f for f in frames if abs(f - number) <= settled]
                holders = {holder_at.get(f, "unknown") for f in around}
                if len(holders) != 1 or "unknown" in holders or number - settled < frames[0]:
                    continue  # The disc changes hands about now
                holder = holders.pop()
                others = [p for p in players if p[0] != holder]
                mine = [p for p in players if p[0] == holder]
                if holder is not None and not mine:
                    continue  # The holder is not seen: who is nearest cannot be said
                nearest = None
                if mine and others:
                    hx, hy = (mine[0][1][0] + mine[0][1][2]) / 2, mine[0][1][3]
                    nearest = min(
                        others,
                        key=lambda p: np.hypot((p[1][0] + p[1][2]) / 2 - hx, p[1][3] - hy),
                    )[0]
                chosen = [p for p in others if p[0] == nearest]
                rest = [p for p in others if p[0] != nearest]
                chosen += rng.sample(rest, min(len(rest), OTHERS_PER_FRAME - len(chosen)))
                for player, box, crop, disc_seen in mine + chosen:
                    holding = int(player == holder)
                    (folder / f"{player}_{number}.jpg").write_bytes(crop.tobytes())
                    index.writerow(
                        (
                            video,
                            stretch,
                            number,
                            player,
                            holding,
                            int(disc_seen),
                            int(player == nearest),
                            speed_text(speed_of(places, player, number, rate)),
                            *(round(float(value), 1) for value in box),
                        )
                    )
                    written[holding] += 1
            file.flush()
            print(
                f"{video} stretch {stretch}: {written[1]} crops holding, {written[0]} not",
                flush=True,
            )


if __name__ == "__main__":
    main()
