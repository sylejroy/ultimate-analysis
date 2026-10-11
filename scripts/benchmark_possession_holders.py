#!/usr/bin/env python3
"""Who has the disc, against the holders named by hand on the "Holding the disc?" page.

scripts/benchmark_possession.py takes its truth from the discs labelled by hand and must
leave out every frame in which the disc touches two players' boxes: the label does not
say whose it is. Those are the frames in which the mark stands in front of the thrower.
Here the truth is the player somebody named as the holder, so those frames count.

The video is analysed around each such frame as the app analyses it. The holder the
pipeline has at the frame (dated back as it dates a change) is right if their box is the
named holder's. Otherwise, the first that holds:

- nobody: the pipeline names no holder
- not followed: no track is at the named holder (covered, or lost)
- the mark: the pipeline's holder is of the other team and stands right by the named one
- a teammate / an opponent: somebody else of the same team, or of the other
- other: the teams are not known

    python scripts/benchmark_possession_holders.py possession_holders_recording --record \\
        --answers hand_holders --dataset generated_holders
    python scripts/benchmark_possession_holders.py possession_holders_try \\
        --replay possession_holders_recording --set models.possession.confirm_seconds=0.2

With --record what the possession tracking was told is kept, and a change to its logic or
settings is scored on that in a moment with --replay, as in benchmark_possession.py.
"""

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Union

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

import cv2  # noqa: E402
from build_holder_ground_truth import AFTER_SECONDS, BEFORE_SECONDS, STEP  # noqa: E402

from ultimate_analysis.config.settings import get_config  # noqa: E402
from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.possession_replay import (  # noqa: E402
    RecordingPossessionTracker,
    replay,
)
from ultimate_analysis.utils.judgement_files import holders_said  # noqa: E402

SAME_BOX = 0.5  # A track is the named holder if their boxes overlap by this much (IoU)
RIGHT = "right"
KINDS = (RIGHT, "nobody", "not followed", "the mark", "a teammate", "an opponent", "other")
COLUMNS = ("video", "frame", "kind", "holder_track", "pipeline_holder", "live_holder")


def iou(first: tuple, second: tuple) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    area = (first[2] - first[0]) * (first[3] - first[1])
    return shared / (area + (second[2] - second[0]) * (second[3] - second[1]) - shared)


def named_holders(answers: Path, dataset: Path) -> list:
    """(video, frame, box) of every player named as the holder by hand."""
    said = {key for key, holding in holders_said(answers / "holders.csv").items() if holding}
    found = []
    with open(dataset / "index.csv", newline="") as file:
        for row in csv.DictReader(file):
            if (row["video"], row["stretch"], row["frame"], row["player"]) in said:
                box = tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))
                found.append((row["video"], int(row["frame"]), box))
    return sorted(found)


def watch(anchor_of: tuple, options: PipelineOptions) -> Union[dict, str]:
    """Analyse the video around a frame with a named holder; what was seen, or why the
    frame is left out."""
    video, anchor, box = anchor_of
    path = REPO / DEFAULT_PATHS["RAW_VIDEOS"] / f"{video}.mp4"
    if not path.exists():
        return "video not found"
    capture = cv2.VideoCapture(str(path))
    rate = capture.get(cv2.CAP_PROP_FPS)
    start = max(anchor % STEP, anchor - int(BEFORE_SECONDS * rate) // STEP * STEP)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    pipeline = AnalysisPipeline()
    pipeline._possession = RecordingPossessionTracker()
    pipeline.reset()
    pipeline.set_frame_rate(rate)
    said, players, wide = [], {}, True
    for number in range(start, anchor + int(AFTER_SECONDS * rate) + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if (number - start) % STEP:
            continue
        result = pipeline.process(frame, number, options)
        pipeline._possession.frame_done(number)
        said.append((number, result.holder_id, result.possession_since, result.possession_team))
        if number == anchor:
            wide = bool(result.wide_shot)
            players = {
                track.track_id: (
                    tuple(float(v) for v in track.to_ltrb()),
                    getattr(track, "team", None),
                )
                for track in result.tracks
                if track.class_name == "player"
            }
    if not wide:
        return "a close-up"
    if not players:
        return "no players followed at the frame"
    return {
        "video": video,
        "anchor": anchor,
        "box": box,
        "players": players,
        "said": said,
        "calls": pipeline._possession.calls,
    }


def judged(case: dict, said: list) -> dict:
    """The row of a frame: the pipeline's holder against the one named by hand."""
    anchor, players = case["anchor"], case["players"]
    holder_at, frames = {}, []
    for number, holder, since, _ in said:
        frames.append(number)
        for earlier in frames[::-1]:
            if earlier < since:
                break
            holder_at[earlier] = holder
    says = holder_at.get(anchor)
    live = {number: holder for number, holder, _, _ in said}.get(anchor)
    # The track at the named holder, if any
    at_holder = max(players, key=lambda player: iou(players[player][0], case["box"]))
    if iou(players[at_holder][0], case["box"]) < SAME_BOX:
        at_holder = None
    if says is not None and says == at_holder:
        kind = RIGHT
    elif says is None:
        kind = "nobody"
    elif at_holder is None:
        kind = "not followed"
    elif says not in players:
        kind = "other"
    else:
        (box, team), (truly, their_team) = players[says], players[at_holder]
        beside = box[0] < truly[2] and truly[0] < box[2] and box[1] < truly[3] and truly[1] < box[3]
        if team is None or their_team is None:
            kind = "other"
        elif team != their_team:
            kind = "the mark" if beside else "an opponent"
        else:
            kind = "a teammate"
    return {
        "video": case["video"],
        "frame": anchor,
        "kind": kind,
        "holder_track": "" if at_holder is None else at_holder,
        "pipeline_holder": "" if says is None else says,
        "live_holder": "" if live is None else live,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="New folder in data/benchmarks")
    parser.add_argument("--answers", default="", help="Folder of answers in training_data")
    parser.add_argument("--dataset", default="", help="The dataset the answers are about")
    parser.add_argument("--record", action="store_true", help="Keep what possession was told")
    parser.add_argument("--replay", help="Folder in data/benchmarks of a run with --record")
    parser.add_argument("--set", action="append", default=[], help="setting=value, as JSON")
    parser.add_argument("--frames", type=int, default=0, help="This many at most (a trial)")
    parser.add_argument("--videos", nargs="+", default=[], help="Only these (parts of names)")
    args = parser.parse_args()
    if bool(args.answers) == bool(args.replay):
        parser.error("give either --answers and --dataset, for a run over the videos, or --replay")

    benchmarks = REPO / "data" / "benchmarks"
    out = benchmarks / args.name
    if out.exists():
        sys.exit(f"{out} exists: results are not overwritten, choose another name")
    config = get_config()
    for item in args.set:
        key, value = item.split("=", 1)
        node = config
        *path, last = key.split(".")
        for part in path:
            node = node[part]
        node[last] = json.loads(value)

    left_out, cases = Counter(), []
    if args.replay:
        with open(benchmarks / args.replay / "recording.pkl", "rb") as file:
            recording = pickle.load(file)
        left_out.update(recording["left_out"])
        cases = recording["cases"]
        out.mkdir(parents=True)
        rows = [judged(case, replay(case["calls"])) for case in cases]
    else:
        data = REPO / DEFAULT_PATHS["TRAINING_DATA"]
        anchors = named_holders(data / args.answers, data / args.dataset)
        if args.videos:
            anchors = [a for a in anchors if any(part in a[0] for part in args.videos)]
        anchors = anchors[: args.frames or None]
        out.mkdir(parents=True)
        options = PipelineOptions(player_id=False, top_down_view=False, draw=False)
        for count, anchor_of in enumerate(anchors):
            case = watch(anchor_of, options)
            if isinstance(case, str):
                left_out[case] += 1
            else:
                cases.append(case)
            if count % 20 == 19:
                print(f"{count + 1} of {len(anchors)} frames analysed", flush=True)
        rows = [judged(case, case["said"]) for case in cases]
        if args.record:
            with open(out / "recording.pkl", "wb") as file:
                pickle.dump({"cases": cases, "left_out": left_out}, file)

    with open(out / "frames.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    kinds = Counter(row["kind"] for row in rows)
    followed = len(rows) - kinds["not followed"]
    live = sum(
        row["kind"] == RIGHT and row["live_holder"] == row["pipeline_holder"] for row in rows
    )
    lines = [f"Settings changed: {', '.join(args.set)}"] if args.set else []
    lines += [
        f"{len(rows)} frames with a holder named by hand",
        *(f"  left out: {count} {why}" for why, count in left_out.items()),
        "",
        *(f"  {kind:<14}{kinds[kind]:>5}  {kinds[kind] / max(len(rows), 1):.0%}" for kind in KINDS),
        "",
        f"Right where the named holder is followed: {kinds[RIGHT]} of {followed} "
        f"({kinds[RIGHT] / max(followed, 1):.0%})",
        f"Right, and shown so in that very frame: {live} of {len(rows)}",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
