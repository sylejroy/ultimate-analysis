#!/usr/bin/env python3
"""The tracker against what was said by hand on the "Same player?" page.

The page shows pictures of tracks of a dataset of player crops and takes answers. Of the
tracker that made the dataset they say:

- a picture taken out of its track: the tracker went on with somebody else there
- two tracks of one stretch said to be one player: the tracker lost the player and began
  again under a new number
- two tracks of one stretch said to be different players, and the pictures left in a
  track: what the tracker had right

Here the stretches are analysed again as the app analyses them now, and every picture
that was looked at is found among the tracks of its frame by its box. Then, per answer,
whether the tracker of today does what the answer asks for:

- followed: the pictures that are in a track at all
- one number: the tracks whose pictures left in all carry one track number
- told apart: the pictures taken out that carry another number than the rest of their
  track
- joined: the pairs said to be one player that carry one number
- kept apart: the pairs said to be different players that carry two

The tracker that made the dataset has, by what the answers are, every track at one
number, none of the pictures taken out told apart and none of the pairs joined.

    python scripts/benchmark_tracking_answers.py tracking_answers \\
        --answers hand_pairs --dataset generated_reid

Pairs of tracks from different stretches are not looked at: they are for telling players
apart by their looks, not for the tracker.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402

from ultimate_analysis.config.settings import get_config  # noqa: E402
from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.utils import judgement_files  # noqa: E402

SAME_BOX = 0.5  # A track is at a picture if their boxes overlap by this much (IoU)
SHOWN = 8  # Pictures of a track the page shows (where it did not write them down)
COLUMNS = ("video", "stretch", "player", "frame", "of_the_player", "track_now")


def overlap(first: tuple, second: tuple) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    area = (first[2] - first[0]) * (first[3] - first[1])
    return shared / (area + (second[2] - second[0]) * (second[3] - second[1]) - shared)


def looked_at(answers: Path, dataset: Path) -> tuple:
    """What the answers say of the tracks of one stretch: the pictures looked at, per
    track (frame -> whether it is of the track's player), and the pairs said to be one
    player and to be different players."""
    said = {}
    for row in judgement_files._read(answers / "pairs.csv"):
        first = (row["video"], row["first_stretch"], row["first_player"])
        second = (row["video"], row["second_stretch"], row["second_player"])
        said[judgement_files.pair_of(first, second)] = (first, second, row["answer"])
    pictures = defaultdict(dict)
    for row in judgement_files._read(answers / "pictures.csv"):
        track = (row["video"], row["stretch"], row["player"])
        pictures[track][int(row["frame"])] = row["of_the_player"] == "1"
    # The first answers were given before the pictures shown were written down
    tracks = judgement_files.load_tracks(dataset)
    for first, second, answer in said.values():
        for track in (first, second):
            if track not in pictures and track in tracks and answer in ("same", "different"):
                pictures[track] = {frame: True for frame, _ in tracks[track].chosen(SHOWN)}
    # A track said to show several people has no player of its own to keep a number for
    for first, second, answer in said.values():
        if answer.endswith("_mixed"):
            pictures.pop(first if answer == "first_mixed" else second, None)
    pairs = {
        answer: [
            (first, second)
            for first, second, given in said.values()
            if given == answer and first[1] == second[1]
        ]
        for answer in ("same", "different")
    }
    return dict(pictures), pairs


def boxes_of(dataset: Path, wanted: set) -> tuple:
    """The box of every picture wanted (video, stretch, player, frame), and the first
    frame of every stretch in the dataset."""
    boxes, begins = {}, {}
    with open(dataset / "index.csv", newline="") as file:
        for row in csv.DictReader(file):
            stretch, frame = (row["video"], row["stretch"]), int(row["frame"])
            begins[stretch] = min(begins.get(stretch, frame), frame)
            key = (*stretch, row["player"], frame)
            if key in wanted:
                boxes[key] = tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))
    return boxes, begins


def tracks_now(video: str, first: int, frames: dict, options: PipelineOptions) -> dict:
    """Analyse a stretch from its first frame; the track at each box of `frames`
    (frame -> {key: box}) as key -> track number, None where no track is there."""
    capture = cv2.VideoCapture(str(REPO / DEFAULT_PATHS["RAW_VIDEOS"] / f"{video}.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    pipeline = AnalysisPipeline()
    pipeline.reset()
    pipeline.set_frame_rate(capture.get(cv2.CAP_PROP_FPS))
    found = {}
    for number in range(first, max(frames) + 1):
        ok, frame = capture.read()
        if not ok:
            break
        result = pipeline.process(frame, number, options)
        for key, box in frames.get(number, {}).items():
            players = [t for t in result.tracks if getattr(t, "class_name", None) == "player"]
            best = max(players, key=lambda t: overlap(box, t.to_ltrb()), default=None)
            at_it = best is not None and overlap(box, best.to_ltrb()) >= SAME_BOX
            found[key] = best.track_id if at_it else None
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="New folder in data/benchmarks")
    parser.add_argument("--answers", required=True, help="Folder of answers in training_data")
    parser.add_argument("--dataset", required=True, help="The dataset the answers are about")
    parser.add_argument("--videos", nargs="+", default=[], help="Only these (parts of names)")
    parser.add_argument("--stretches", type=int, default=0, help="This many at most (a trial)")
    parser.add_argument("--set", action="append", default=[], help="setting=value, as JSON")
    args = parser.parse_args()
    for item in args.set:
        key, value = item.split("=", 1)
        node = get_config()
        *path, last = key.split(".")
        for part in path:
            node = node[part]
        node[last] = json.loads(value)

    out = REPO / "data" / "benchmarks" / args.name
    if out.exists():
        sys.exit(f"{out} exists: results are not overwritten, choose another name")
    data = REPO / DEFAULT_PATHS["TRAINING_DATA"]
    pictures, pairs = looked_at(data / args.answers, data / args.dataset)
    if args.videos:
        pictures = {
            track: frames
            for track, frames in pictures.items()
            if any(part in track[0] for part in args.videos)
        }
    wanted = {(*track, frame) for track, frames in pictures.items() for frame in frames}
    boxes, begins = boxes_of(data / args.dataset, wanted)
    by_stretch = defaultdict(lambda: defaultdict(dict))
    for key, box in boxes.items():
        by_stretch[key[:2]][key[3]][key] = box
    stretches = sorted(by_stretch)[: args.stretches or None]
    out.mkdir(parents=True)

    options = PipelineOptions(player_id=False, top_down_view=False, draw=False)
    now = {}
    for count, stretch in enumerate(stretches):
        now.update(tracks_now(stretch[0], begins[stretch], by_stretch[stretch], options))
        print(f"{count + 1} of {len(stretches)} stretches analysed", flush=True)

    with open(out / "pictures.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        for key in sorted(now):
            of_the_player = pictures[key[:3]][key[3]]
            writer.writerow([*key, int(of_the_player), "" if now[key] is None else now[key]])

    # Per track: the numbers its pictures carry now, those left in and those taken out
    left_in, taken_out = defaultdict(list), defaultdict(list)
    for key in sorted(now):
        (left_in if pictures[key[:3]][key[3]] else taken_out)[key[:3]].append(now[key])
    counts = Counter()
    number_of = {}
    for track, numbers in left_in.items():
        followed = [number for number in numbers if number is not None]
        counts["pictures left in"] += len(numbers)
        counts["pictures left in, followed"] += len(followed)
        if followed:
            number_of[track] = Counter(followed).most_common(1)[0][0]
        if len(followed) >= 2:
            counts["tracks"] += 1
            counts["tracks at one number"] += len(set(followed)) == 1
            counts["changes of number"] += sum(a != b for a, b in zip(followed, followed[1:]))
    for track, numbers in taken_out.items():
        for number in numbers:
            counts["pictures taken out"] += 1
            if number is None or track not in number_of:
                counts["pictures taken out, not followed"] += 1
            else:
                counts["pictures taken out, told apart"] += number != number_of[track]
    for answer, title in (("same", "one player"), ("different", "different players")):
        for first, second in pairs[answer]:
            if first in number_of and second in number_of:
                counts[f"pairs said to be {title}"] += 1
                counts[f"pairs said to be {title}, at one number"] += (
                    number_of[first] == number_of[second]
                )

    def share(part: str, whole: str) -> str:
        total = counts[whole]
        return f"{counts[part]} of {total}" + (f" ({counts[part] / total:.0%})" if total else "")

    apart = (
        counts["pairs said to be different players"]
        - counts["pairs said to be different players, at one number"]
    )
    counts["pairs kept apart"] = apart
    lines = [
        f"{len(stretches)} stretches of {args.dataset}, answers of {args.answers}",
        "",
        f"Followed: {share('pictures left in, followed', 'pictures left in')} pictures left "
        "in their track are in a track now",
        f"One number: {share('tracks at one number', 'tracks')} tracks carry one number in "
        f"all their pictures ({counts['changes of number']} changes of number among them)",
        f"Told apart: {share('pictures taken out, told apart', 'pictures taken out')} pictures "
        "taken out carry another number than the rest of their track "
        f"({counts['pictures taken out, not followed']} are in no track now)",
        "Joined: "
        + share("pairs said to be one player, at one number", "pairs said to be one player")
        + " pairs of one stretch said to be one player carry one number",
        f"Kept apart: {share('pairs kept apart', 'pairs said to be different players')} pairs "
        "of one stretch said to be different players carry two numbers",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
