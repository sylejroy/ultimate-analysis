#!/usr/bin/env python3
"""Prepare stretches of video for labelling the tracks by hand (scripts/label_tracks.py).

The tracker's own output is the starting point: every frame's player boxes with the
tracker's IDs and, where read, the jersey number. Labelling then means naming each
track and marking where a track changes player, not drawing boxes.

Writes, to a new folder in data/raw/training_data:

    <dataset>/<video>_<start>/tracks.json   {"video": ..., "start": ..., "frames": N,
                                              "rate": fps, "boxes": {frame: [[id, x1, y1,
                                              x2, y2, number, team], ...]}}

    python scripts/prepare_track_labels.py hand_tracks \\
        chicago_vs_new_york_2024:77933 chicago_vs_san_francisco_2024:40283 \\
        portland_vs_san_francisco_2024:17744 san_francisco_vs_colorado_2024:42071
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402

from ultimate_analysis.config.settings import get_config  # noqa: E402
from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.player_id import initialize_player_id_system  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="New folder in data/raw/training_data")
    parser.add_argument("stretches", nargs="+", help="<video name>:<first frame>")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    out = REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    config = get_config()
    config["models"]["player_id"]["background_reading"] = False  # Numbers in the frame read
    config["models"]["reid"]["roster_folder"] = ""
    initialize_player_id_system()
    options = PipelineOptions(top_down_view=False)
    for stretch in args.stretches:
        video, start = stretch.rsplit(":", 1)
        start = int(start)
        folder = out / f"{video}_{start}"
        if (folder / "tracks.json").exists():
            print(f"{folder.name}: there already", flush=True)
            continue
        path = REPO / "data" / "raw" / "videos" / f"{video}.mp4"
        capture = cv2.VideoCapture(str(path))
        rate = capture.get(cv2.CAP_PROP_FPS)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        pipeline = AnalysisPipeline()
        pipeline.new_video(str(path))
        pipeline.set_frame_rate(rate)
        boxes = {}
        count = int(args.seconds * rate)
        for step in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            result = pipeline.process(frame, start + step, options)
            boxes[step] = [
                [
                    int(track.track_id),
                    *(round(float(v), 1) for v in track.to_ltrb()),
                    str(result.player_ids.get(track.track_id, ("", None))[0] or ""),
                    track.team if track.team in (0, 1) else -1,
                ]
                for track in result.tracks
                if track.class_name == "player"
            ]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "tracks.json").write_text(
            json.dumps(
                {"video": video, "start": start, "frames": len(boxes), "rate": rate, "boxes": boxes}
            )
        )
        ids = {box[0] for frame in boxes.values() for box in frame}
        print(f"{folder.name}: {len(boxes)} frames, {len(ids)} tracker IDs", flush=True)


if __name__ == "__main__":
    main()
