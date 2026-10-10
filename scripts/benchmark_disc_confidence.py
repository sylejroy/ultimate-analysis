#!/usr/bin/env python3
"""What a lower confidence of the disc model would find, and what it would cost.

At half of the labelled frames with the wrong holder the disc is not detected
(scripts/benchmark_possession.py). The disc model is asked for the discs it is far less
sure of than the app's threshold, in every frame analysed around the labelled frames of
a recorded run of that benchmark, and counted per threshold:

- found: of the frames at which the pipeline had no disc at the label, those with a
  detection within a disc's width of the labelled disc, and those with one anywhere on
  the labelled holder (their box widened by a quarter)
- false: detections on a player that are not the labelled disc: on any player where
  nobody holds it, on the players other than the holder where somebody does
- what the possession tracking would say with them: the recorded run is played back
  with the discs of the threshold in place of those it was given, the less sure ones
  only where they are on a player. "As recorded" and the app's threshold in every frame
  differ by the frames in which the pipeline does not run the disc model at all.

Measures only; nothing in the pipeline uses the lower thresholds.

    python scripts/benchmark_disc_confidence.py disc_confidence_v1 \\
        --recording possession_recorded_v1 --labels labelled_players_discs_v1 labelled_discs_v1
"""

import argparse
import os
import pickle
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
from benchmark_possession import CAUSES, RIGHT, judged  # noqa: E402
from build_holder_ground_truth import labelled_discs  # noqa: E402

from ultimate_analysis.config.settings import get_config, get_setting  # noqa: E402
from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.inference import (  # noqa: E402
    detect_discs,
    load_detection_model,
)
from ultimate_analysis.processing.possession_replay import replay  # noqa: E402

LOWEST = 0.05
THRESHOLDS = (0.05, 0.1, 0.15, 0.2)
ON_PLAYER = 0.25  # A detection is on a player if it is in their box widened by this share


def on(box: tuple, disc: tuple) -> bool:
    pad_x, pad_y = ON_PLAYER * (box[2] - box[0]), ON_PLAYER * (box[3] - box[1])
    x, y = (disc[0] + disc[2]) / 2, (disc[1] + disc[3]) / 2
    return box[0] - pad_x <= x <= box[2] + pad_x and box[1] - pad_y <= y <= box[3] + pad_y


def detect(cases: list, model: tuple) -> dict:
    """(video, frame) -> [(x1, y1, x2, y2, confidence)] for every analysed frame."""
    found = {}
    for count, case in enumerate(cases):
        frames = [number for number, *_ in case["said"]]
        capture = cv2.VideoCapture(str(REPO / "data" / "raw" / "videos" / f"{case['video']}.mp4"))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        wanted = set(frames)
        for number in range(frames[0], frames[-1] + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if number in wanted and (case["video"], number) not in found:
                found[case["video"], number] = [
                    (*(float(v) for v in d["bbox"]), float(d["confidence"]))
                    for d in detect_discs(frame, *model)
                ]
        if count % 20 == 19:
            print(f"{count + 1} of {len(cases)} labelled frames searched", flush=True)
    return found


def with_discs(case: dict, found: dict, threshold: float, sure: float) -> list:
    """What the possession tracking was told, with the discs of a threshold in place of
    those it was given: the sure ones anywhere, the others where they are on a player."""
    calls = []
    for call in case["calls"]:
        if call[0] == "update":
            _, _, seen, number = call
            players = [box for _, box, _, kind in seen if kind == "player"]
            discs = [
                {"class_name": "disc", "confidence": d[4], "bbox": list(d[:4])}
                for d in found.get((case["video"], number), [])
                if d[4] >= sure or (d[4] >= threshold and any(on(box, d) for box in players))
            ]
            call = ("update", discs, seen, number)
        calls.append(call)
    return calls


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="New folder in data/benchmarks")
    parser.add_argument("--recording", required=True, help="A run of benchmark_possession.py")
    parser.add_argument("--labels", nargs="+", required=True, help="Datasets with disc labels")
    args = parser.parse_args()

    benchmarks = REPO / "data" / "benchmarks"
    out = benchmarks / args.name
    if out.exists():
        sys.exit(f"{out} exists: results are not overwritten, choose another name")
    with open(benchmarks / args.recording / "recording.pkl", "rb") as file:
        cases = pickle.load(file)["cases"]
    data = REPO / DEFAULT_PATHS["TRAINING_DATA"]
    labels = {a[:2]: a[2] for name in args.labels for a in labelled_discs(data / name)}
    sure = float(get_setting("models.disc_detection.confidence_threshold", 0.3))
    thresholds = (*THRESHOLDS, sure)

    out.mkdir(parents=True)
    get_config()["models"]["disc_detection"]["confidence_threshold"] = LOWEST
    model = load_detection_model(get_setting("models.disc_detection.default_model"))
    if model is None:
        sys.exit("The disc model cannot be loaded")
    found = detect(cases, model)
    with open(out / "discs.pkl", "wb") as file:
        pickle.dump(found, file)

    # At the labelled frames: the labelled disc found, and detections on players that
    # are not it
    counts = Counter()
    for case in cases:
        video, anchor, at = case["video"], case["anchor"], case["at"]
        recorded = judged(case, case["said"])
        missed = recorded["cause"] == CAUSES[0]
        # The label is in shares of the picture, the disc at it in pixels
        share = labels[video, anchor]
        diameter = max(share[2] * at["disc"][0] / share[0], share[3] * at["disc"][1] / share[1])
        boxes, holder = case["boxes"][anchor], at["labelled"]
        counts["frames", recorded["label"]] += 1
        counts["missed", recorded["label"]] += missed
        for threshold in thresholds:
            discs = [d for d in found.get((video, anchor), []) if d[4] >= threshold]
            at_label = [
                d
                for d in discs
                if np.hypot((d[0] + d[2]) / 2 - at["disc"][0], (d[1] + d[3]) / 2 - at["disc"][1])
                <= diameter
            ]
            counts["at label", recorded["label"], threshold] += bool(at_label)
            if missed:
                counts["found", recorded["label"], threshold] += bool(at_label)
                counts["on holder", recorded["label"], threshold] += holder is not None and any(
                    on(boxes[holder], d) for d in discs
                )
            false = [
                d
                for d in discs
                if d not in at_label
                and any(on(box, d) for player, box in boxes.items() if player != holder)
                and not (holder is not None and on(boxes[holder], d))
            ]
            counts["false", recorded["label"], threshold] += len(false)
            counts["frames with false", recorded["label"], threshold] += bool(false)

    lines = [f"Disc model at confidence {LOWEST} and up; the app's threshold is {sure}", ""]
    for label in ("held", "nobody"):
        frames, missed = counts["frames", label], counts["missed", label]
        lines.append(
            f"Labelled frames, {label}: {frames}, of them {missed} with no disc at the label "
            "in the recorded run"
        )
        lines.append(
            f"  {'from':>6}{'disc at label':>15}{'missed: found':>15}{'on the holder':>15}"
            f"{'false on players':>18}{'frames with one':>17}"
        )
        for threshold in thresholds:
            lines.append(
                f"  {threshold:>6}{counts['at label', label, threshold]:>15}"
                f"{counts['found', label, threshold]:>15}"
                f"{counts['on holder', label, threshold]:>15}"
                f"{counts['false', label, threshold]:>18}"
                f"{counts['frames with false', label, threshold]:>17}"
            )
        lines.append("")

    # What the possession tracking would make of them
    lines.append("The holder against the label, the possession tracking played back with:")
    lines.append(f"  {'discs':<34}{'held right':>12}{'nobody right':>14}")

    def right(rows: list, label: str) -> str:
        chosen = [row for row in rows if row["label"] == label]
        same = sum(row["cause"] in (RIGHT, CAUSES[3]) for row in chosen)
        return f"{same} of {len(chosen)}"

    played = [("as recorded", [judged(case, replay(case["calls"])) for case in cases])]
    for threshold in (sure, *THRESHOLDS[::-1]):
        title = f"from {threshold}, every frame" + ("" if threshold == sure else ", on players")
        rows = [judged(c, replay(with_discs(c, found, threshold, sure))) for c in cases]
        played.append((title, rows))
    for title, rows in played:
        lines.append(f"  {title:<34}{right(rows, 'held'):>12}{right(rows, 'nobody'):>14}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
