#!/usr/bin/env python3
"""Build a dataset for the disc heat map (scripts/train_disc_heatmap.py).

Every labelled frame of a disc dataset that is found again in its video, with the two
frames a thirtieth and two thirtieths of a second before it, each laid onto the
labelled frame by how the camera moved: what stands still lies on itself in all three,
and what moves does not. Kept with the places of the discs, in the splits of the source.

    python scripts/build_disc_frames.py train_discs train_disc_frames_full --full

writes a new folder in data/raw/training_data: <split>/<number>_0.jpg, _1.jpg and _2.jpg
(the labelled frame), index.json and summary.txt. Without --full the frames are kept at
960 x 540, with it at 1920 x 1080: a disc is a few pixels, and half the size loses many.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.camera_motion import CameraMotionEstimator  # noqa: E402
from ultimate_analysis.processing.disc_heatmap import FRAME_GAP_SECONDS  # noqa: E402

# A dataset image is the video's frame if they differ by less than this on average (of
# 255): some are copies turned or cut by the dataset's maker, which no video has
SAME_PICTURE = 9.0
# The video of the images of the first datasets, whose names do not say it
FIRST_VIDEO = "portland_vs_san_francisco_2024"


def where(stem: str) -> Tuple[Optional[str], int]:
    """(video, frame) of a dataset image, by its name; no video for a snippet's."""
    own = re.match(r"labelled_(.+?)(?:_snippet_\d+_\d+)?_frame_(\d+)$", stem)
    if own:
        return (None if "_snippet_" in stem else own.group(1)), int(own.group(2))
    old = re.match(r"s\d_(?:(.+?)_)?frame_(\d+)_", stem)
    if old:
        return old.group(1) or FIRST_VIDEO, int(old.group(2))
    return None, 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="Disc dataset in data/raw/training_data")
    parser.add_argument("name", help="New folder in data/raw/training_data")
    parser.add_argument("--full", action="store_true", help="Frames at 1920 x 1080")
    args = parser.parse_args()

    data = REPO / DEFAULT_PATHS["TRAINING_DATA"]
    source, out = data / args.source, data / args.name
    if out.exists():
        sys.exit(f"{out} exists: nothing is overwritten")
    size = (1920, 1080) if args.full else (960, 540)
    videos: dict = {}

    def video_of(name: str) -> Optional[cv2.VideoCapture]:
        if name not in videos:
            path = REPO / "data" / "raw" / "videos" / f"{name}.mp4"
            videos[name] = cv2.VideoCapture(str(path)) if path.exists() else None
        return videos[name]

    index = []
    left_out = {"no video": 0, "not the video's frame": 0, "camera motion not found": 0}
    for split in ("train", "valid", "test"):
        (out / split).mkdir(parents=True)
        for image_path in sorted((source / split / "images").iterdir()):
            video, frame_index = where(image_path.stem)
            capture = video_of(video) if video else None
            if capture is None:
                left_out["no video"] += 1
                continue
            step = max(1, round(capture.get(cv2.CAP_PROP_FPS) * FRAME_GAP_SECONDS))
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index - 2 * step))
            frames = []
            for offset in range(2 * step + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                if offset % step == 0:
                    frames.append(frame)
            if len(frames) != 3:
                left_out["no video"] += 1
                continue
            labelled = cv2.imread(str(image_path))
            same = cv2.resize(
                frames[2], (labelled.shape[1], labelled.shape[0]), interpolation=cv2.INTER_AREA
            )
            if np.abs(same.astype(np.int16) - labelled.astype(np.int16)).mean() > SAME_PICTURE:
                left_out["not the video's frame"] += 1
                continue
            # How the camera moved from each of the two earlier frames to the labelled one
            estimator = CameraMotionEstimator()
            estimator.update(frames[0], [])
            first = estimator.update(frames[1], [])
            second = estimator.update(frames[2], [])
            if first is None or second is None:
                left_out["camera motion not found"] += 1
                continue
            height, width = frames[2].shape[:2]
            laid = [
                cv2.warpPerspective(frames[0], second @ first, (width, height)),
                cv2.warpPerspective(frames[1], second, (width, height)),
                frames[2],
            ]
            discs = []
            label_path = source / split / "labels" / (image_path.stem + ".txt")
            if label_path.exists():
                for line in label_path.read_text().split("\n"):
                    parts = line.split()
                    if len(parts) == 5:
                        cx, cy, w, h = (float(value) for value in parts[1:])
                        discs.append([cx * size[0], cy * size[1], w * size[0], h * size[1]])
            name = f"{len(index):05d}"
            for number, picture in enumerate(laid):
                small = cv2.resize(picture, size, interpolation=cv2.INTER_AREA)
                path = out / split / f"{name}_{number}.jpg"
                cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
            index.append(
                {
                    "name": name,
                    "split": split,
                    "from": image_path.stem,
                    "video": video,
                    "frame": frame_index,
                    "discs": discs,
                }
            )
        rows = [row for row in index if row["split"] == split]
        print(
            f"{split}: {len(rows)} frames, {sum(len(row['discs']) for row in rows)} discs; "
            f"left out so far: {left_out}",
            flush=True,
        )
    (out / "index.json").write_text(json.dumps(index))
    (out / "summary.txt").write_text(
        f"Frames of {source.name} found again in their videos, each with the two frames "
        f"before it ({FRAME_GAP_SECONDS:.3f} s apart) laid onto it by the camera's motion, at "
        f"{size[0]} x {size[1]}. {len(index)} frames, "
        f"{sum(len(row['discs']) for row in index)} discs. Left out: {left_out}.\n"
    )


if __name__ == "__main__":
    main()
