#!/usr/bin/env python3
"""Say why the pipeline has the wrong holder at the frames with a hand-labelled disc.

The frames and the holder by the label are those of scripts/build_holder_ground_truth.py
(the video is analysed the same way: from two seconds before the labelled frame to a
second and a half after). Each frame at which the pipeline does not say what the label
says is given one cause, the first of these that holds:

1. the disc is not detected: no detection near the labelled disc in that frame
2. the disc is detected and put at the wrong place: at another player (the mark, a box
   that overlaps the holder's), at nobody, or a surer detection elsewhere is taken
3. the disc is put right in that frame, but a track was lost or begun again in the
   second before it: the labelled holder's, the one the pipeline has as holder, or a player
   detected at the disc who has no track
4. the holder is right and the team in possession is not the holder's
5. other: the disc is put right in that frame and the tracks are steady (the holder not
   changed yet, or not taken back), or the frame is not taken for drone footage

The counts are printed overall, by what the label says and by game, with a few frames
of each cause. Every labelled frame is written to frames.csv in a new folder.

    python scripts/benchmark_possession.py possession_breakdown_v1 \\
        --labels labelled_players_discs_v1 labelled_discs_v1
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from build_holder_ground_truth import (  # noqa: E402
    AFTER_SECONDS,
    BEFORE_SECONDS,
    ON_PLAYER,
    STEP,
    holder_by_label,
    labelled_discs,
)

from ultimate_analysis.config.settings import get_config, get_setting  # noqa: E402
from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.possession import player_at_disc  # noqa: E402

# A detection is of the labelled disc if it is within this share of a player's height of it
NEAR_LABEL = 0.3
AROUND_SECONDS = 1.0  # A track lost or begun this long before the labelled frame counts
EXAMPLES = 5

RIGHT = "right"
CAUSES = (
    "1 disc not detected",
    "2 disc put at the wrong place",
    "3 track lost or begun again",
    "4 team in possession wrong",
    "5 other",
)
COLUMNS = (
    "video",
    "frame",
    "label",
    "labelled_holder",
    "pipeline_holder",
    "holder_in_that_frame",
    "cause",
    "note",
    "disc_detected",
    "disc_put_at",
    "holder_team",
    "possession_team",
    "disc_at_label_last_second",
)


def overlap(first: tuple, second: tuple) -> bool:
    return (
        first[0] < second[2]
        and second[0] < first[2]
        and first[1] < second[3]
        and second[1] < first[3]
    )


def on_box(point: tuple, box: tuple) -> bool:
    """Whether a point is in a box widened as the label's holder is found by."""
    pad_x, pad_y = ON_PLAYER * (box[2] - box[0]), ON_PLAYER * (box[3] - box[1])
    return (
        box[0] - pad_x <= point[0] <= box[2] + pad_x
        and box[1] - pad_y <= point[1] <= box[3] + pad_y
    )


def unsteady(track_id: int, boxes: dict, anchor: int, rate: float) -> str:
    """How the track was lost or begun in the second before the labelled frame ("": not)."""
    around = [f for f in sorted(boxes) if 0 <= anchor - f <= AROUND_SECONDS * rate]
    there = [f for f in around if track_id in boxes[f]]
    if anchor not in there:
        return f"track {track_id} is not followed at this frame"
    if there[0] > around[0]:
        return f"track {track_id} began {(anchor - there[0]) / rate:.1f} s before"
    if len(there) < len(around):
        return f"track {track_id} was lost for {len(around) - len(there)} frames in the second"
    return ""


def cause_of(at: dict, boxes: dict, anchor: int, rate: float) -> tuple:
    """(cause, note) for a labelled frame at which the pipeline has the wrong holder."""
    labelled, says = at["labelled"], at["says"]
    if says == "unknown":
        return CAUSES[4], "not taken for drone footage"
    if not at["detected"]:
        kept = "" if says is None else f", {says} kept as the holder"
        return CAUSES[0], f"no disc detected at the label{kept}"
    if not at["used_is_labelled"]:
        return CAUSES[1], "a surer disc detection elsewhere is the one taken"
    put_at = at["put_at"]
    if put_at != labelled:
        if put_at is None:
            return CAUSES[1], "the detected disc is in no player's box"
        if labelled is None:
            return CAUSES[1], f"the detected disc is in the box of {put_at}"
        beside = overlap(boxes[anchor][put_at], boxes[anchor][labelled])
        whose = "whose box overlaps the holder's" if beside else "away from the holder"
        return CAUSES[1], f"the detected disc is nearer to {put_at}, {whose}"
    # The disc is put right in this frame: what was followed before decides
    if labelled is None and at["untracked_at_disc"]:
        return CAUSES[2], "a player is detected at the disc but not followed"
    for track_id in (labelled, says):
        if track_id is not None and (how := unsteady(track_id, boxes, anchor, rate)):
            return CAUSES[2], how
    seen = at["label_seen_lately"]
    if labelled is None:
        return CAUSES[4], f"disc at nobody in {seen} frames of the last second, {says} kept"
    named = "nobody" if says is None else str(says)
    return CAUSES[4], f"disc at the holder in {seen} frames of the last second, {named} named"


def table(rows: list, key, title: str) -> list:
    """Counts of right and of each cause for the rows grouped by a key, as lines."""
    width = max([len(title)] + [len(str(key(row))) for row in rows])
    heads = ("frames", RIGHT, *(cause.split()[0] for cause in CAUSES))
    lines = [f"{title:<{width}}" + "".join(f"{head:>8}" for head in heads)]
    groups = sorted({key(row) for row in rows}, key=str)
    for group in [*groups, None]:
        chosen = rows if group is None else [row for row in rows if key(row) == group]
        counts = Counter(row["cause"] for row in chosen)
        name = "all" if group is None else str(group)
        lines.append(
            f"{name:<{width}}{len(chosen):>8}"
            + "".join(f"{counts[what]:>8}" for what in (RIGHT, *CAUSES))
        )
    return lines


def kinds_of(rows: list) -> Counter:
    """The notes of the rows counted, with the track numbers and times left out."""
    return Counter(
        " ".join(w for w in row["note"].split() if not w.strip(",").replace(".", "").isdigit())
        for row in rows
    )


def a_few(rows: list) -> list:
    """Some of the rows to show, from as many games as there are."""
    by_game = defaultdict(list)
    for row in rows:
        by_game[row["video"]].append(row)
    turns = [
        [game[turn] for game in by_game.values() if len(game) > turn] for turn in range(EXAMPLES)
    ]
    return [row for turn in turns for row in turn][:EXAMPLES]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="New folder in data/benchmarks")
    parser.add_argument("--labels", nargs="+", required=True, help="Datasets with disc labels")
    parser.add_argument("--videos", nargs="+", default=[], help="Only these (parts of names)")
    args = parser.parse_args()

    data = REPO / DEFAULT_PATHS["TRAINING_DATA"]
    out = REPO / "data" / "benchmarks" / args.name
    if out.exists():
        sys.exit(f"{out} exists: results are not overwritten, choose another name")
    anchors = sorted(
        {a[:2]: a for name in args.labels for a in labelled_discs(data / name)}.values()
    )
    if args.videos:
        anchors = [a for a in anchors if any(part in a[0] for part in args.videos)]
    out.mkdir(parents=True)
    get_config()["models"]["reid"]["enabled"] = False
    options = PipelineOptions(player_id=False, top_down_view=False)
    margin = float(get_setting("models.possession.box_margin", 0.15))
    left_out = Counter()
    rows = []

    with open(out / "frames.csv", "w", newline="") as file:
        index = csv.DictWriter(file, COLUMNS, extrasaction="ignore")
        index.writeheader()
        for count, (video, anchor, disc_share) in enumerate(anchors):
            path = REPO / "data" / "raw" / "videos" / f"{video}.mp4"
            if not path.exists():
                left_out["video not found"] += 1
                continue
            capture = cv2.VideoCapture(str(path))
            rate = capture.get(cv2.CAP_PROP_FPS)
            before = int(BEFORE_SECONDS * rate) // STEP * STEP
            start = max(anchor % STEP, anchor - before)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            pipeline = AnalysisPipeline()
            pipeline.reset()
            pipeline.set_frame_rate(rate)

            # Per analysed frame: the holder and the team in possession (dated back as
            # the pipeline dates them), the player boxes by track, and the discs detected
            holder_at, team_at, boxes, discs_at, frames = {}, {}, {}, {}, []
            at: Optional[dict] = None
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
                    team_at[earlier] = result.possession_team
                players = [track for track in result.tracks if track.class_name == "player"]
                boxes[number] = {
                    track.track_id: tuple(float(v) for v in track.to_ltrb()) for track in players
                }
                discs = [d for d in result.detections if d["class_name"] == "disc"]
                discs_at[number] = [
                    ((d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2)
                    for d in discs
                ]
                if number != anchor or not players:
                    continue
                height, width = frame.shape[:2]
                disc = (disc_share[0] * width, disc_share[1] * height)
                labelled, clear = holder_by_label(disc, list(boxes[number].items()))
                near = NEAR_LABEL * float(np.median([b[3] - b[1] for b in boxes[number].values()]))
                used = max(discs, key=lambda d: d.get("confidence", 0.0), default=None)
                teams = {track.track_id: getattr(track, "team", None) for track in players}
                at = {
                    "disc": disc,
                    "near": near,
                    "labelled": labelled,
                    "clear": clear,
                    "live": result.holder_id if result.wide_shot else "unknown",
                    "detected": any(
                        np.hypot(x - disc[0], y - disc[1]) <= near for x, y in discs_at[number]
                    ),
                    "used_is_labelled": used is not None
                    and np.hypot(
                        (used["bbox"][0] + used["bbox"][2]) / 2 - disc[0],
                        (used["bbox"][1] + used["bbox"][3]) / 2 - disc[1],
                    )
                    <= near,
                    "put_at": None
                    if used is None
                    else player_at_disc(used["bbox"], players, margin),
                    "untracked_at_disc": any(
                        d["class_name"] == "player"
                        and on_box(disc, d["bbox"])
                        and not any(overlap(d["bbox"], box) for box in boxes[number].values())
                        for d in result.detections
                    ),
                    "holder_team": teams.get(labelled),
                }

            if at is None:
                left_out["no players followed at the labelled frame"] += 1
                continue
            if not at["clear"]:
                left_out["the label does not say whose the disc is"] += 1
                continue
            labelled = at["labelled"]
            at["says"] = holder_at.get(anchor, "unknown")
            at["possession_team"] = team_at.get(anchor)
            # In how many frames of the second before the disc was detected where the
            # label has it: on the labelled holder, or at the labelled place
            lately = [f for f in frames if 0 <= anchor - f <= AROUND_SECONDS * rate]
            at["label_seen_lately"] = sum(
                any(
                    on_box(d, boxes[f][labelled])
                    if labelled is not None
                    else np.hypot(d[0] - at["disc"][0], d[1] - at["disc"][1]) <= at["near"]
                    for d in discs_at[f]
                )
                for f in lately
                if labelled is None or labelled in boxes[f]
            )
            if at["says"] != labelled:
                cause, note = cause_of(at, boxes, anchor, rate)
            elif labelled is not None and at["holder_team"] not in (None, at["possession_team"]):
                known = at["possession_team"] is not None
                cause = CAUSES[3]
                note = "the other team" if known else "no team in possession"
            else:
                cause, note = RIGHT, ""
            row = {
                "video": video,
                "frame": anchor,
                "label": "nobody" if labelled is None else "held",
                "labelled_holder": "" if labelled is None else labelled,
                "pipeline_holder": "" if at["says"] is None else at["says"],
                "holder_in_that_frame": "" if at["live"] is None else at["live"],
                "cause": cause,
                "note": note,
                "disc_detected": int(at["detected"]),
                "disc_put_at": "" if at["put_at"] is None else at["put_at"],
                "holder_team": "" if at["holder_team"] is None else at["holder_team"],
                "possession_team": "" if at["possession_team"] is None else at["possession_team"],
                "disc_at_label_last_second": f"{at['label_seen_lately']}/{len(lately)}",
            }
            rows.append(row)
            index.writerow(row)
            file.flush()
            if count % 20 == 19:
                done = dict(Counter(r["cause"].split()[0] for r in rows))
                print(f"{count + 1} of {len(anchors)} labelled frames: {done}", flush=True)

    lines = ["The holder against the label (the team aside):"]
    for label in ("held", "nobody"):
        chosen = [row for row in rows if row["label"] == label]
        same = sum(row["cause"] in (RIGHT, CAUSES[3]) for row in chosen)
        lines.append(
            f"  {label:<7}{len(chosen):>5} frames, {same} the same ({same / len(chosen):.0%})"
        )
    lines += [f"  left out: {times} {what}" for what, times in sorted(left_out.items())]
    lines += ["", "Causes: " + "; ".join(CAUSES), ""]
    lines += table(rows, lambda row: row["label"], "By the label") + [""]
    lines += table(rows, lambda row: row["video"][:32], "By game") + [""]
    for label in ("held", "nobody"):
        chosen = [row for row in rows if row["label"] == label]
        lines += table(chosen, lambda row: row["video"][:32], f"By game, {label}") + [""]
    for cause in CAUSES:
        chosen = [row for row in rows if row["cause"] == cause]
        lines.append(f"{cause}: {len(chosen)} frames")
        lines += [f"    {times:3d}  {note}" for note, times in kinds_of(chosen).most_common()]
        lines += [
            f"  {row['video'][:32]} frame {row['frame']}: label "
            f"{row['labelled_holder'] or 'nobody'}, pipeline "
            f"{row['pipeline_holder'] or 'nobody'}; {row['note']}"
            for row in a_few(chosen)
        ]
        lines.append("")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
