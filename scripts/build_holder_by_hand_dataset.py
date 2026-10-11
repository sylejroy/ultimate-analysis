#!/usr/bin/env python3
"""A holder dataset of the players somebody answered for, and of nobody else.

The holder datasets made by a rule (scripts/build_holder_sighted_dataset.py) mark a
player as the holder wherever the disc was seen in their box, also in the frames between
two sightings, and a quarter of those marks are wrong by eye. Here only what was said
by hand on the labelling pages is kept:

- a player said to hold the disc, or said not to
- a player of a frame in which another was said to hold it: one disc is in play, so
  they do not

A box said to show two players is left out: it is neither. The crops are copied, so
the new dataset stands by itself; the source and the answers are only read.

With --everybody, what follows from the answers for the players the source does not
have (it has four of a frame at most) is added, cut from the videos:

- in a frame in which somebody was said to hold the disc, every other player the
  player model finds does not: one disc is in play. Left out are boxes that touch the
  holder's, which may show the holder as well.
- a holder clicked on the "Click the holder" page who is none of the source's players
  (clicked_holders.csv) holds the disc.
- in a frame of which it was said that nobody holds the disc (the N key of that page,
  which shows the whole frame), nobody the player model finds does. These are what
  teach a network that there may be no holder at all: the disc in the air, or hidden.

Nothing is guessed in this: no frame nobody looked at is given a holder.

Reads:  data/raw/training_data/<source>/index.csv and crops/
        data/raw/training_data/<answers>/holders.csv and two_in_box.csv
Writes: data/raw/training_data/<name>/index.csv (the source's columns, `holding` as
        answered, and `how` it is known) and crops/, in a new folder

    python scripts/build_holder_by_hand_dataset.py train_holders_v2 \\
        --source generated_holders --answers hand_holders
"""

import argparse
import csv
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.utils.judgement_files import (  # noqa: E402
    SAID,
    boxes_of_two,
    said_of_the_frame,
    stretch_folder,
)

FOUND = "another holds it in the frame; found by the player model"
CLICKED = "said, by a click on a player the source does not have"
NOBODY = "nobody holds the disc in the frame; found by the player model"
SAME_PLAYER = 0.3  # A box found is one the source has from this overlap (IoU)


def overlap(first: tuple, second: tuple) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    area = (first[2] - first[0]) * (first[3] - first[1])
    return shared / (area + (second[2] - second[0]) * (second[3] - second[1]) - shared)


def everybody_else(source: Path, answers: Path, kept: list, out: Path) -> list:
    """The rows that follow from the answers for players the source does not have,
    their crops written as they are found: (row, kind) each."""
    import cv2

    from ultimate_analysis.processing.inference import detect_players, load_detection_model
    from ultimate_analysis.utils.model_files import default_model_path

    def box_of(row: dict) -> tuple:
        return tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))

    # Frame -> the boxes the source has of it, and the holder's
    known, holder = defaultdict(list), {}
    for row, _, _ in kept:
        key = (row["video"], row["stretch"], row["frame"])
        known[key].append(box_of(row))
        if row["holding"]:
            holder[key] = box_of(row)
    clicked = {}
    path = answers / "clicked_holders.csv"
    if path.exists():
        with open(path, newline="") as file:
            for line in csv.DictReader(file):
                if line["two"] == "0":
                    key = (line["video"], line["stretch"], line["frame"])
                    clicked[key] = holder[key] = box_of(line)

    # Frames of which it was said that nobody holds the disc: no holder's box in them
    for key in said_of_the_frame(answers / "holders.csv"):
        holder.setdefault(key, None)

    detector = load_detection_model(default_model_path("player_detection"))
    if detector is None:
        sys.exit("The player model is not there: --everybody needs it")
    videos, rows = {}, []
    for count, (key, held) in enumerate(sorted(holder.items(), key=lambda item: item[0])):
        video, stretch, frame_number = key
        if video not in videos:
            videos[video] = cv2.VideoCapture(
                str(REPO / DEFAULT_PATHS["RAW_VIDEOS"] / f"{video}.mp4")
            )
        videos[video].set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = videos[video].read()
        if not ok:
            continue
        folder = out / "crops" / video / stretch_folder(source / "crops", video, stretch).name
        folder.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]

        def add(box: tuple, name: str, holding: int, how: str, kind: str) -> None:
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
            if x2 - x1 < 4 or y2 - y1 < 8:
                return
            cv2.imwrite(str(folder / f"{name}_{frame_number}.jpg"), frame[y1:y2, x1:x2])
            row = {"video": video, "stretch": stretch, "frame": frame_number, "player": name}
            row.update(holding=holding, disc_seen=0, nearest=0, kind="found", how=how)
            row.update(zip(("x1", "y1", "x2", "y2"), (round(float(v), 1) for v in box)))
            rows.append((row, kind))

        if key in clicked:
            add(held, "clicked", 1, CLICKED, "clicked as the holder, not in the source")
        found = [tuple(float(v) for v in d["bbox"]) for d in detect_players(frame, *detector)]
        for number, box in enumerate(found):
            # Not one the source has, and not touching the holder's box
            if any(overlap(box, b) >= SAME_PLAYER for b in known[key]):
                continue
            if held is None:
                add(box, f"found{number}", 0, NOBODY, "nobody holds it; found by the player model")
            elif overlap(box, held) == 0:
                add(box, f"found{number}", 0, FOUND, "another holds it; found by the player model")
        if count % 200 == 199:
            print(f"{count + 1} of {len(holder)} frames with a holder looked at", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="New folder in data/raw/training_data")
    parser.add_argument("--source", required=True, help="The dataset the answers are about")
    parser.add_argument("--answers", required=True, help="The folder of the answers")
    parser.add_argument(
        "--everybody",
        action="store_true",
        help="Add the other players of a frame with a holder, and the holders clicked",
    )
    args = parser.parse_args()

    data = REPO / DEFAULT_PATHS["TRAINING_DATA"]
    source, answers, out = data / args.source, data / args.answers, data / args.name
    if out.exists():
        sys.exit(f"{out} exists: datasets are not overwritten, choose another name")

    # The last answer for a player is the one that counts
    said = {}
    with open(answers / "holders.csv", newline="") as file:
        for line in csv.DictReader(file):
            key = (line["video"], line["stretch"], line["frame"], line["player"])
            said[key] = (int(line["holding"]), line["how"])
    two = boxes_of_two(answers / "holders.csv")

    kept, kinds, missing = [], Counter(), 0
    with open(source / "index.csv", newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames) + ["how"]
        for row in reader:
            key = (row["video"], row["stretch"], row["frame"], row["player"])
            if key not in said or key in two:
                continue
            folder = stretch_folder(source / "crops", row["video"], row["stretch"])
            crop = folder / f"{row['player']}_{row['frame']}.jpg"
            if not crop.exists():
                missing += 1
                continue
            holding, how = said[key]
            kinds[
                "said to hold the disc"
                if holding
                else "said not to hold it"
                if how == SAID
                else "another holds it in the frame"
            ] += 1
            kept.append(({**row, "holding": holding, "how": how}, crop, folder.name))

    out.mkdir(parents=True)
    for row, crop, folder in kept:
        to = out / "crops" / row["video"] / folder
        to.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(crop, to / crop.name)
    more = everybody_else(source, answers, kept, out) if args.everybody else []
    for _, kind in more:
        kinds[kind] += 1
    with open(out / "index.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, columns, restval="")
        writer.writeheader()
        writer.writerows(row for row, _, _ in kept)
        writer.writerows(row for row, _ in more)

    games = Counter(row["video"] for row, _, _ in kept if row["holding"])
    games.update(row["video"] for row, _ in more if row["holding"])
    lines = [
        f"{len(kept) + len(more)} players, {len(kept)} of them answered by hand, "
        f"from {args.source} and {args.answers}:",
        *(f"  {count:5d}  {kind}" for kind, count in sorted(kinds.items())),
        f"Left out: {len(two)} boxes of two players, {missing} without a crop",
        "Holders by game:",
        *(f"  {count:5d}  {video}" for video, count in sorted(games.items())),
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
