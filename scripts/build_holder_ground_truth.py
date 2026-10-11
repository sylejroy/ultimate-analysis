#!/usr/bin/env python3
"""Build crops of players holding the disc or not from the frames with a hand-labelled disc.

In the datasets made in the Labelling tab somebody has marked where the disc is. That
says who holds it in that frame without the disc model or the possession tracking having
a part in it: the player the labelled disc is on, or nobody if it is away from them all.

Each such frame is an anchor. The video is analysed from two seconds before it to a
second and a half after, and:

- at the anchor, every player gives a crop marked by the label ("anchor")
- if the pipeline has the same player holding the disc there, the frames around it in
  which it goes on saying so are taken as well ("possession"): the same player, known
  to hold the disc. Among them are the frames this is for: the disc model finds no
  disc on the player, and the player has it all the same.

How often the pipeline agrees with the labels at the anchors is printed at the end: a
measure of the possession tracking itself.

The result goes to a new folder, laid out as by build_holder_dataset.py, with three
more columns in index.csv: kind ("anchor" or "possession"), anchor (the labelled frame)
and agrees (whether the pipeline had the labelled holder at the anchor).

    python scripts/build_holder_ground_truth.py generated_holders_anchored \\
        --labels hand_players_discs hand_discs
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from build_holder_dataset import COLUMNS, MIN_HEIGHT, SETTLED_SECONDS, speed_text  # noqa: E402

from ultimate_analysis.config.settings import get_config  # noqa: E402
from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.holder import (  # noqa: E402
    CROP_PAD,
    crop_around,
    disc_on,
    place_on_field,
    speed_of,
)

BEFORE_SECONDS, AFTER_SECONDS = 2.0, 1.5
STEP = 2  # Every so many frames are analysed
CROP_EVERY = 3  # Of those, every so many give crops of a possession
ON_PLAYER = 0.15  # A disc is on a player if it is in their box widened by this share
AT_THE_FEET = 0.85  # ... and not below this share of its height: that disc lies there


def labelled_discs(dataset: Path) -> list:
    """(video, frame, disc box as shares of the picture) for each frame with one disc."""
    names = yaml.safe_load((dataset / "data.yaml").read_text())["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    disc = next(key for key, name in names.items() if name == "disc")
    found = []
    for path in sorted((dataset / "labels").rglob("*.txt")):
        video, _, frame = path.stem.rpartition("_frame_")
        boxes = [
            [float(v) for v in line.split()[1:5]]
            for line in path.read_text().splitlines()
            if line.strip() and int(line.split()[0]) == disc
        ]
        if video and frame.isdigit() and len(boxes) == 1:
            found.append((video, int(frame), boxes[0]))
    return found


def holder_by_label(disc: tuple, players: list) -> tuple:
    """(player the labelled disc is on or None, whether that can be said)."""
    on = []
    for player, (x1, y1, x2, y2) in players:
        pad_x, pad_y = ON_PLAYER * (x2 - x1), ON_PLAYER * (y2 - y1)
        if x1 - pad_x <= disc[0] <= x2 + pad_x and y1 - pad_y <= disc[1] <= y2 + pad_y:
            at_feet = disc[1] > y1 + AT_THE_FEET * (y2 - y1)
            centre = np.hypot(disc[0] - (x1 + x2) / 2, disc[1] - (y1 + y2) / 2)
            on.append((centre, player, at_feet))
    if not on:
        return None, True
    # On two players at once (the thrower and the mark), or at somebody's feet: whose
    # it is cannot be said from the label
    if len(on) > 1 or on[0][2]:
        return None, False
    return on[0][1], True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="New folder in data/raw/training_data")
    parser.add_argument("--labels", nargs="+", required=True, help="Datasets with disc labels")
    args = parser.parse_args()

    data = REPO / DEFAULT_PATHS["TRAINING_DATA"]
    out = data / args.dataset
    if out.exists():
        sys.exit(f"{out} exists: datasets are not overwritten, choose another name")
    anchors = sorted(
        {a[:2]: a for name in args.labels for a in labelled_discs(data / name)}.values()
    )
    out.mkdir(parents=True)
    get_config()["models"]["reid"]["enabled"] = False
    options = PipelineOptions(player_id=False, top_down_view=False)
    said = Counter()

    with open(out / "index.csv", "w", newline="") as file:
        index = csv.writer(file)
        index.writerow((*COLUMNS, "x1", "y1", "x2", "y2", "kind", "anchor", "agrees"))
        for count, (video, anchor, disc_share) in enumerate(anchors):
            path = REPO / "data" / "raw" / "videos" / f"{video}.mp4"
            if not path.exists():
                said["video not found"] += 1
                continue
            capture = cv2.VideoCapture(str(path))
            rate = capture.get(cv2.CAP_PROP_FPS)
            before = int(BEFORE_SECONDS * rate) // STEP * STEP
            start = max(anchor % STEP, anchor - before)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            pipeline = AnalysisPipeline()
            pipeline.reset()
            pipeline.set_frame_rate(rate)

            holder_at, seen, frames, places = {}, {}, [], {}
            for number in range(start, anchor + int(AFTER_SECONDS * rate) + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                if (number - start) % STEP:
                    continue
                result = pipeline.process(frame, number, options)
                frames.append(number)
                for earlier in frames[::-1]:
                    if earlier < result.possession_since:
                        break
                    holder_at[earlier] = result.holder_id if result.wide_shot else "unknown"
                height, width = frame.shape[:2]
                places[number] = {
                    track.track_id: place
                    for track in result.tracks
                    if track.class_name == "player"
                    and (place := place_on_field(result.image_to_field, track.to_ltrb()))
                }
                discs = [d["bbox"] for d in result.detections if d["class_name"] == "disc"]
                players = []
                for track in result.tracks:
                    if track.class_name != "player":
                        continue
                    x1, y1, x2, y2 = track.to_ltrb()
                    picture, box = crop_around(frame, (x1, y1, x2, y2))
                    disc_seen = disc_on(box, discs)
                    crop = cv2.imencode(".jpg", picture)[1]
                    players.append((track.track_id, (x1, y1, x2, y2), crop, disc_seen, box))
                seen[number] = players
                if number == anchor:
                    disc = (disc_share[0] * width, disc_share[1] * height)

            if anchor not in seen or not seen[anchor]:
                said["no players followed at the labelled frame"] += 1
                continue
            labelled, clear = holder_by_label(disc, [(p[0], p[1]) for p in seen[anchor]])
            if not clear:
                said["the label does not say whose the disc is"] += 1
                continue
            pipeline_says = holder_at.get(anchor, "unknown")
            agrees = pipeline_says == labelled
            if labelled is None:
                said[
                    "nobody holds it: the pipeline agrees"
                    if agrees
                    else "nobody holds it: the pipeline names a holder"
                ] += 1
            elif agrees:
                said["a player holds it: the pipeline names them"] += 1
            elif pipeline_says is None:
                said["a player holds it: the pipeline names nobody"] += 1
            else:
                said["a player holds it: the pipeline names another"] += 1

            folder = out / "crops" / video / f"{anchor:06d}"
            folder.mkdir(parents=True, exist_ok=True)

            def write(number: int, player: tuple, holding: bool, nearest: bool, kind: str) -> None:
                track_id, _, crop, disc_seen, box = player
                if box[3] - box[1] < MIN_HEIGHT * (1 + 2 * CROP_PAD):
                    return
                (folder / f"{track_id}_{number}.jpg").write_bytes(crop.tobytes())
                index.writerow(
                    (video, anchor, number, track_id, int(holding), int(disc_seen), int(nearest))
                    + (speed_text(speed_of(places, track_id, number, rate)),)
                    + tuple(box)
                    + (kind, anchor, int(agrees))
                )

            def nearest_to(holder: int, players: list) -> int:
                mine = next(p for p in players if p[0] == holder)
                hx, hy = (mine[1][0] + mine[1][2]) / 2, mine[1][3]
                others = [p for p in players if p[0] != holder]
                if not others:
                    return -1
                return min(
                    others, key=lambda p: np.hypot((p[1][0] + p[1][2]) / 2 - hx, p[1][3] - hy)
                )[0]

            # The anchor: everybody, marked by the label
            near = nearest_to(labelled, seen[anchor]) if labelled is not None else -1
            for player in seen[anchor]:
                write(anchor, player, player[0] == labelled, player[0] == near, "anchor")
            # The possession around it, where the pipeline has the labelled holder
            if labelled is not None and agrees:
                settled = int(SETTLED_SECONDS * rate)
                for position, number in enumerate(frames):
                    if number == anchor or position % CROP_EVERY:
                        continue
                    around = {holder_at.get(f) for f in frames if abs(f - number) <= settled}
                    between = {
                        holder_at.get(f)
                        for f in frames
                        if min(number, anchor) <= f <= max(number, anchor)
                    }
                    players = seen[number]
                    if around != {labelled} or between != {labelled}:
                        continue
                    if not any(p[0] == labelled for p in players):
                        continue
                    near = nearest_to(labelled, players)
                    for player in players:
                        if player[0] == labelled or player[0] == near:
                            write(
                                number,
                                player,
                                player[0] == labelled,
                                player[0] == near,
                                "possession",
                            )
            file.flush()
            if count % 20 == 19:
                print(f"{count + 1} of {len(anchors)} labelled frames: {dict(said)}", flush=True)

    print("At the labelled frames:")
    for what, times in sorted(said.items()):
        print(f"  {times:4d}  {what}")


if __name__ == "__main__":
    main()
