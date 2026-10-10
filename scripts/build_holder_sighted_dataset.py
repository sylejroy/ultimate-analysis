#!/usr/bin/env python3
"""Build crops of players holding the disc or not from where the disc model sees it.

build_holder_dataset.py marks the holder by the possession tracking, which goes on
naming a holder for as long as it has not seen the disc elsewhere: through a sprint, a
lost track, a throw it missed. Here a player holds the disc only between two sightings
of it on them:

- the disc is seen on a player when its centre is in their box and in nobody else's
  (the player's box takes in the disc they hold, for the most part)
- seen on the same player twice, at most --max-gap seconds apart, and not seen anywhere
  else in between: the frames in between are frames in which they hold it, whether it
  is seen there or not. Of these, --per-gap frames evenly spread give crops.
- seen away from every player twice in a short time (--flight-gap): it flies, and
  nobody holds it in between

In a frame in which somebody holds the disc the holder gives a crop, and so do the
player nearest to them (most often the mark) and --others more. In a frame in which it
flies, --others + 1 players do. The crops are of the boxes themselves, without padding.

With --labels, the frames with a hand-labelled disc (as build_holder_ground_truth.py)
are analysed too, a few seconds around each, and at the end it is printed how often
these marks agree with the hand labels: a measure of this ground truth itself.

Stretches are taken from every video in data/raw/videos, or those given with --videos:
--seconds long, one every --every seconds. The result goes to a new folder, laid out as
by build_holder_dataset.py (train_holder.py reads it), with three more columns in
index.csv: kind ("held" or "flight"), gap (seconds between the two sightings) and
labelled (1 for the stretches around a hand-labelled frame).

    python scripts/build_holder_sighted_dataset.py holder_sighted_v1 \\
        --labels labelled_players_discs_v1 labelled_discs_v1
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from build_holder_dataset import COLUMNS, MIN_HEIGHT, speed_text  # noqa: E402
from build_holder_ground_truth import holder_by_label, labelled_discs  # noqa: E402

from ultimate_analysis.config.settings import get_config  # noqa: E402
from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.holder import (  # noqa: E402
    crop_around,
    place_on_field,
    speed_of,
)

VIDEOS = REPO / "data" / "raw" / "videos"
AROUND_LABEL = 3.0  # Seconds analysed before and after a hand-labelled frame


def sightings_of(discs: list, players: list) -> list:
    """Where each detected disc is: ("on", player), ("between", players) when it is in
    the boxes of several, or ("away", None)."""
    found = []
    for x1, y1, x2, y2 in discs:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        on = [p[0] for p in players if p[1][0] <= cx <= p[1][2] and p[1][1] <= cy <= p[1][3]]
        if len(on) == 1:
            found.append(("on", on[0]))
        elif on:
            found.append(("between", tuple(on)))
        else:
            found.append(("away", None))
    return found


def marks_of(
    frames: List[int], sighted: dict, rate: float, max_gap: float, flight_gap: float
) -> dict:
    """Frame -> who holds the disc (a player, or None while it flies), for the frames
    that can be said from the sightings; other frames are left out."""
    marked = {}
    seen = [f for f in frames if sighted.get(f)]
    for first, second in zip(seen, seen[1:]):
        a, b = sighted[first], sighted[second]
        gap = (second - first) / rate
        between = [f for f in frames if first <= f <= second]
        holders_a = {s[1] for s in a if s[0] == "on"}
        holders_b = {s[1] for s in b if s[0] == "on"}
        # A single sighting on one player at both ends, nothing else seen at either end
        if len(a) == 1 and len(b) == 1 and holders_a and holders_a == holders_b:
            if gap <= max_gap:
                holder = next(iter(holders_a))
                for f in between:
                    marked[f] = (holder, gap)
        elif all(s[0] == "away" for s in a + b) and gap <= flight_gap:
            for f in between:
                marked.setdefault(f, (None, gap))
    return marked


def spread(frames: List[int], count: int) -> List[int]:
    """Up to `count` of some frames, evenly spread and away from both ends."""
    if len(frames) <= count:
        return frames
    return [frames[int((i + 0.5) * len(frames) / count)] for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="New folder in data/raw/training_data")
    parser.add_argument("--videos", nargs="*", default=None, help="Video names (all if none)")
    parser.add_argument("--labels", nargs="*", default=[], help="Datasets with disc labels")
    parser.add_argument("--seconds", type=float, default=30.0, help="Length of a stretch")
    parser.add_argument("--every", type=float, default=180.0, help="A stretch every so long")
    parser.add_argument("--step", type=int, default=2, help="Analyse every so many frames")
    parser.add_argument("--max-gap", type=float, default=2.0, help="Seconds between sightings")
    parser.add_argument("--flight-gap", type=float, default=0.5, help="The same in flight")
    parser.add_argument("--per-gap", type=int, default=3, help="Frames taken between two")
    parser.add_argument("--others", type=int, default=2, help="Players besides the nearest")
    parser.add_argument("--pad", type=float, default=0.0, help="Share added around a box")
    args = parser.parse_args()

    data = REPO / DEFAULT_PATHS["TRAINING_DATA"]
    out = data / args.dataset
    if out.exists():
        sys.exit(f"{out} exists: datasets are not overwritten, choose another name")
    names = args.videos or sorted(p.stem for p in VIDEOS.glob("*.mp4"))

    # (video, first frame, last frame, hand-labelled frame or None)
    stretches = []
    for video in names:
        capture = cv2.VideoCapture(str(VIDEOS / f"{video}.mp4"))
        rate, total = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        start = args.every / 2
        while (start + args.seconds) * rate < total:
            stretches.append((video, int(start * rate), int((start + args.seconds) * rate), None))
            start += args.every
    anchors = {a[:2]: a for name in args.labels for a in labelled_discs(data / name)}
    for video, frame, disc in sorted(anchors.values()):
        if (VIDEOS / f"{video}.mp4").exists():
            stretches.append((video, frame, frame, disc))

    out.mkdir(parents=True)
    get_config()["models"]["reid"]["enabled"] = False
    options = PipelineOptions(player_id=False, top_down_view=False)
    rng = random.Random(0)
    agreement = Counter()

    with open(out / "index.csv", "w", newline="") as file:
        index = csv.writer(file)
        index.writerow((*COLUMNS, "x1", "y1", "x2", "y2", "kind", "gap", "labelled"))
        for number_of, (video, first, last, disc_share) in enumerate(stretches):
            capture = cv2.VideoCapture(str(VIDEOS / f"{video}.mp4"))
            rate = capture.get(cv2.CAP_PROP_FPS)
            if disc_share is not None:
                anchor = first
                first = max(0, anchor - int(AROUND_LABEL * rate))
                first += (anchor - first) % args.step  # So the labelled frame is analysed
                last = anchor + int(AROUND_LABEL * rate)
            capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            pipeline = AnalysisPipeline()
            pipeline.reset()
            pipeline.set_frame_rate(rate)

            frames, sighted, seen, places, size = [], {}, {}, {}, None
            for number in range(first, last + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                if (number - first) % args.step:
                    continue
                result = pipeline.process(frame, number, options)
                size = frame.shape[:2]
                if not result.wide_shot:
                    # A close-up or a cut: sightings on either side say nothing together
                    frames.append(number)
                    sighted[number] = [("cut", None), ("cut", None)]
                    continue
                frames.append(number)
                places[number] = {
                    track.track_id: place
                    for track in result.tracks
                    if track.class_name == "player"
                    and (place := place_on_field(result.image_to_field, track.to_ltrb()))
                }
                players = []
                for track in result.tracks:
                    if track.class_name != "player":
                        continue
                    box = tuple(float(v) for v in track.to_ltrb())
                    picture, widened = crop_around(frame, box, pad=args.pad)
                    if picture.size == 0:
                        continue
                    crop = cv2.imencode(".jpg", picture)[1]
                    players.append((track.track_id, box, crop, widened))
                discs = [d["bbox"] for d in result.detections if d["class_name"] == "disc"]
                sighted[number] = sightings_of(discs, players)
                seen[number] = players
            capture.release()

            marked = marks_of(frames, sighted, rate, args.max_gap, args.flight_gap)
            labelled = int(disc_share is not None)
            if labelled:
                anchor_players = seen.get(anchor, [])
                if not anchor_players or size is None:
                    agreement["no players at the labelled frame"] += 1
                else:
                    disc = (disc_share[0] * size[1], disc_share[1] * size[0])
                    truth, clear = holder_by_label(disc, [(p[0], p[1]) for p in anchor_players])
                    if not clear:
                        agreement["the label does not say whose it is"] += 1
                    elif anchor not in marked:
                        agreement["no mark here (not between two sightings)"] += 1
                    elif marked[anchor][0] == truth:
                        agreement["marked, and the label agrees"] += 1
                    else:
                        agreement["marked, and the label says otherwise"] += 1

            stretch = first
            folder = out / "crops" / video / f"{stretch:06d}"
            folder.mkdir(parents=True, exist_ok=True)
            # Group the marked frames by the run of sightings they lie between
            runs, current, last_mark = [], [], None
            for number in frames:
                mark = marked.get(number)
                if mark is None or number not in seen:
                    if current:
                        runs.append(current)
                    current, last_mark = [], None
                    continue
                if last_mark is not None and mark != last_mark and current:
                    runs.append(current)
                    current = []
                current.append(number)
                last_mark = mark
            if current:
                runs.append(current)

            written = Counter()
            for run in runs:
                for number in spread(run, args.per_gap):
                    holder, gap = marked[number]
                    players = seen[number]
                    mine = [p for p in players if p[0] == holder]
                    others = [p for p in players if p[0] != holder]
                    if holder is not None and not mine:
                        continue
                    nearest = None
                    if mine and others:
                        hx, hy = (mine[0][1][0] + mine[0][1][2]) / 2, mine[0][1][3]
                        nearest = min(
                            others,
                            key=lambda p: np.hypot((p[1][0] + p[1][2]) / 2 - hx, p[1][3] - hy),
                        )[0]
                    chosen = [p for p in others if p[0] == nearest]
                    rest = [p for p in others if p[0] != nearest]
                    wanted = args.others + (1 if holder is None else len(chosen))
                    chosen += rng.sample(rest, min(len(rest), wanted - len(chosen)))
                    discs_on = {s[1] for s in sighted.get(number, []) if s[0] == "on"}
                    for track_id, box, crop, widened in mine + chosen:
                        if box[3] - box[1] < MIN_HEIGHT:
                            continue
                        is_holder = int(track_id == holder)
                        (folder / f"{track_id}_{number}.jpg").write_bytes(crop.tobytes())
                        index.writerow(
                            (video, stretch, number, track_id, is_holder)
                            + (int(track_id in discs_on), int(track_id == nearest))
                            + (speed_text(speed_of(places, track_id, number, rate)),)
                            + tuple(widened)
                            + ("flight" if holder is None else "held", f"{gap:.2f}", labelled)
                        )
                        written[is_holder] += 1
            file.flush()
            print(
                f"{number_of + 1}/{len(stretches)} {video} {first}: "
                f"{written[1]} crops holding, {written[0]} not",
                flush=True,
            )

    if agreement:
        print("At the hand-labelled frames:")
        for what, times in sorted(agreement.items()):
            print(f"  {times:4d}  {what}")


if __name__ == "__main__":
    main()
