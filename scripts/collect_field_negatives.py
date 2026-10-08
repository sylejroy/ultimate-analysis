#!/usr/bin/env python3
"""Collect pictures that show no field from above, for training the field model.

A field model that has only ever seen drone footage of fields learns that a large area
of grass is a field, and marks the grass in a close-up from ground level. It needs
pictures with grass and no field in them, labelled as showing nothing.

Edited games cut to such close-ups. They are found here as frames in which the player
model, which was trained on drone footage, finds nobody at all. That is not proof: a
drone shot of an empty field has nobody in it either, and a title card may lie over
drone footage. So the result has to be looked through by hand. Frames that do show a
field from above are listed in `excluded.txt` of the result (one name per line) and are
then left out by `build_field_mask_dataset.py --negatives`.

    python scripts/collect_field_negatives.py field_negatives_v1 --videos Machine Ring_of_Fire

The videos are read-only; the result is written to a new folder:

    <dataset>/images/<video>_frame_<number>.jpg
    <dataset>/excluded.txt      names to leave out, filled in by hand
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.inference import (  # noqa: E402
    detect_players,
    load_detection_model,
)
from ultimate_analysis.utils.label_files import JPEG_QUALITY, frame_name  # noqa: E402
from ultimate_analysis.utils.model_files import default_model_path  # noqa: E402

MIN_BRIGHTNESS = 12.0  # A black frame says nothing
MIN_APART = 120  # Frames between two pictures of one video: not the same shot twice over


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="Name of the new folder in data/raw/training_data")
    parser.add_argument("--videos", nargs="+", required=True, help="Videos (parts of names)")
    parser.add_argument("--per-video", type=int, default=60, help="Pictures to take at most")
    parser.add_argument("--tries", type=int, default=900, help="Frames to look at per video")
    parser.add_argument(
        "--skip-start", type=int, default=1800, help="Frames at the start left out (title cards)"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.name
    if output.exists():
        sys.exit(f"Refusing to overwrite existing folder: {output}")
    videos = [
        path
        for path in sorted((REPO / DEFAULT_PATHS["RAW_VIDEOS"]).glob("*.mp4"))
        if any(part in path.name for part in args.videos)
    ]
    if not videos:
        sys.exit("No video is called like that")
    (output / "images").mkdir(parents=True)
    (output / "excluded.txt").write_text("")

    players = load_detection_model(default_model_path("player_detection"))
    random = np.random.default_rng(args.seed)
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        taken: list = []
        for index in random.permutation(np.arange(args.skip_start, count))[: args.tries]:
            index = int(index)
            if any(abs(index - other) < MIN_APART for other in taken):
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame.mean() < MIN_BRIGHTNESS or detect_players(frame, *players):
                continue
            cv2.imwrite(
                str(output / "images" / f"{frame_name(str(video), index)}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            taken.append(index)
            if len(taken) >= args.per_video:
                break
        capture.release()
        print(f"{video.name}: {len(taken)} pictures", flush=True)
    print(f"Written to {output}. Look through them and list in excluded.txt what shows a field.")


if __name__ == "__main__":
    main()
