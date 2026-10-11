#!/usr/bin/env python3
"""Score the tracker against stretches labelled by hand (scripts/label_tracks.py or the
sheets answered in chat).

The truth is which boxes belong to which player: the boxes of the labelled run with
their labels, "x" for people who are not players of the point. A tracker is scored by
running the pipeline over the same stretch and matching its player boxes to the truth
frame by frame (Hungarian on IoU, at least 0.5):

- DetA: of the truth boxes, how many have a matching box, and of the tracker's boxes,
  how many match a truth box (boxes on an "x" person count for neither)
- AssA: for each matched pair, how much of the truth player's life and of the tracker
  ID's life they spend matched to each other (Jaccard), averaged
- HOTA at 0.5: the geometric mean of the two (the full HOTA averages over IoU
  thresholds; this is the one threshold)
- IDF1 and ID switches, as usual

Since the truth boxes come from a run of the same detector, detection is flattered for
that run; the association numbers are what one tracker is compared with another by.

    python scripts/benchmark_tracking.py hand_tracks
    python scripts/benchmark_tracking.py hand_tracks --set models.tracking.weak_detections_from=0.25
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Tuple

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from ultimate_analysis.config.settings import get_config  # noqa: E402
from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.appearance import box_overlaps  # noqa: E402
from ultimate_analysis.processing.player_id import initialize_player_id_system  # noqa: E402

MATCH_IOU = 0.5


def truth_of(folder: Path) -> dict:
    """Frame -> [(player label, box)], with the unlabelled left out and "x" kept apart."""
    data = json.loads((folder / "tracks.json").read_text())
    labels = {int(k): v for k, v in json.loads((folder / "labels.json").read_text()).items()}

    def label_of(track_id: int, frame: int) -> str:
        label = ""
        for since, name in sorted(labels.get(track_id, [])):
            if since <= frame:
                label = name
        return label

    truth = {}
    for frame, boxes in data["boxes"].items():
        frame = int(frame)
        truth[frame] = [
            (label_of(box[0], frame), box[1:5]) for box in boxes if label_of(box[0], frame)
        ]
    return {
        "video": data["video"],
        "start": data["start"],
        "frames": data["frames"],
        "truth": truth,
    }


def run_tracker(video: str, start: int, count: int) -> Tuple[dict, dict]:
    """Frame -> [(tracker ID, box)] of the pipeline as it is now, and track -> player
    as the tracks are joined looking back over the stretch."""
    path = REPO / "data" / "raw" / "videos" / f"{video}.mp4"
    capture = cv2.VideoCapture(str(path))
    rate = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    pipeline = AnalysisPipeline()
    pipeline.new_video(str(path))
    pipeline.set_frame_rate(rate)
    options = PipelineOptions()
    found = {}
    for step in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        result = pipeline.process(frame, start + step, options)
        found[step] = [
            (int(t.track_id), [float(v) for v in t.to_ltrb()])
            for t in result.tracks
            if t.class_name == "player"
        ]
    return found, pipeline.players_of_tracks()


def score(truth: dict, found: dict) -> dict:
    """The measures of one stretch."""
    matched = []  # (frame, player, tracker ID)
    truth_boxes = tracker_boxes = false_alarms = misses = 0
    for frame, players in truth.items():
        real = [(label, box) for label, box in players if label != "x"]
        ignored = [box for label, box in players if label == "x"]
        boxes = found.get(frame, [])
        truth_boxes += len(real)
        pairs = []
        if real and boxes:
            overlap = box_overlaps([box for _, box in real] + [box for _, box in boxes])
            overlap = overlap[: len(real), len(real) :]
            rows, columns = linear_sum_assignment(-overlap)
            pairs = [(r, c) for r, c in zip(rows, columns) if overlap[r, c] >= MATCH_IOU]
        taken = {c for _, c in pairs}
        for r, c in pairs:
            matched.append((frame, real[r][0], boxes[c][0]))
        misses += len(real) - len(pairs)
        for c, (_, box) in enumerate(boxes):
            if c in taken:
                continue
            on_ignored = (
                ignored
                and max(box_overlaps([box] + ignored)[0, 1:].tolist(), default=0.0) >= MATCH_IOU
            )
            if not on_ignored:
                false_alarms += 1
                tracker_boxes += 1
        tracker_boxes += len(pairs)

    # Association: how much of each other's life a player and an ID share
    life_player = Counter(
        label for players in truth.values() for label, _ in players if label != "x"
    )
    life_id = Counter(track_id for boxes in found.values() for track_id, _ in boxes)
    together = Counter((player, track_id) for _, player, track_id in matched)
    association = [
        together[(p, t)] / (life_player[p] + life_id[t] - together[(p, t)]) for _, p, t in matched
    ]
    hits = len(matched)
    det_a = hits / (hits + misses + false_alarms) if hits + misses + false_alarms else 0.0
    ass_a = float(np.mean(association)) if association else 0.0

    # IDF1: each player to the ID it spent most frames with, one to one
    players, ids = sorted(life_player), sorted(life_id)
    if players and ids:
        shared = np.array([[together[(p, t)] for t in ids] for p in players])
        rows, columns = linear_sum_assignment(-shared)
        idtp = int(shared[rows, columns].sum())
    else:
        idtp = 0
    idf1 = 2 * idtp / (truth_boxes + sum(life_id.values())) if truth_boxes else 0.0
    # ID switches: a player matched to another ID than in their last matched frame
    last = {}
    switches = 0
    for frame, player, track_id in sorted(matched):
        if player in last and last[player] != track_id:
            switches += 1
        last[player] = track_id
    by_player = defaultdict(set)
    for _, player, track_id in matched:
        by_player[player].add(track_id)
    return {
        "players": len(life_player),
        "truth_boxes": truth_boxes,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "DetA": det_a,
        "AssA": ass_a,
        "HOTA@0.5": float(np.sqrt(det_a * ass_a)),
        "IDF1": idf1,
        "ID switches": switches,
        "IDs per player": float(np.mean([len(v) for v in by_player.values()]))
        if by_player
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="Folder in data/raw/training_data with labelled stretches")
    parser.add_argument("--set", action="append", default=[], help="setting=value, as JSON")
    parser.add_argument("--only", help="Only stretches whose name contains this")
    args = parser.parse_args()
    config = get_config()
    config["models"]["reid"]["roster_folder"] = ""
    config["models"]["player_id"]["background_reading"] = False
    for item in args.set:
        key, value = item.split("=", 1)
        node = config
        *path, last = key.split(".")
        for part in path:
            node = node[part]
        node[last] = json.loads(value)
    initialize_player_id_system()
    root = REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.dataset
    folders = sorted(
        p
        for p in root.iterdir()
        if (p / "labels.json").exists() and (not args.only or args.only in p.name)
    )
    totals = {"as tracked": defaultdict(list), "joined after": defaultdict(list)}
    for folder in folders:
        labelled = truth_of(folder)
        found, naming = run_tracker(labelled["video"], labelled["start"], labelled["frames"])
        joined = {
            frame: [(naming.get(track_id, track_id), box) for track_id, box in boxes]
            for frame, boxes in found.items()
        }
        provisional = " (provisional truth)" if (folder / "PROVISIONAL.txt").exists() else ""
        print(f"{folder.name}{provisional}", flush=True)
        for kind, tracks in (("as tracked", found), ("joined after", joined)):
            result = score(labelled["truth"], tracks)
            for key, value in result.items():
                totals[kind][key].append(value)
            print(line(kind, result), flush=True)
    if len(folders) > 1:
        print("all")
        for kind, sums in totals.items():
            mean = {key: float(np.mean(values)) for key, values in sums.items()}
            mean["players"] = sum(sums["players"])
            mean["ID switches"] = sum(sums["ID switches"])
            print(line(kind, mean))


def line(kind: str, result: dict) -> str:
    return (
        f"  {kind:13} players {result['players']:3d}  HOTA@0.5 {result['HOTA@0.5']:.3f}  "
        f"DetA {result['DetA']:.3f}  AssA {result['AssA']:.3f}  IDF1 {result['IDF1']:.3f}  "
        f"switches {result['ID switches']:3d}  IDs/player {result['IDs per player']:.2f}"
    )


if __name__ == "__main__":
    main()
