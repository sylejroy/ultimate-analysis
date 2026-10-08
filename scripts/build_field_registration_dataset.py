#!/usr/bin/env python3
"""Build a dataset for a model that finds the field: hand labels carried along the video.

A field label (Labelling tab, "Field lines") says where the field lies in one frame. The
field does not move, so the same label holds for the frames around it once the camera's
motion is taken into account. This script follows that motion from every labelled frame,
forwards and backwards, and writes a frame every so often with the field's place in it.

The motion is followed twice, from every frame and from every second frame. Where the two
disagree at the labelled corners by more than a few pixels, or the footage cuts, the
label is carried no further: one hand label gives up to a few dozen frames, fewer where
the camera moves fast.

Every frame is stored with the game it is from. Frames of one game look alike, so a model
must be scored on games it has not seen: leave one game out, train on the others.

    python scripts/build_field_registration_dataset.py propagated_field_v1
    python scripts/build_field_registration_dataset.py propagated_field_v1 --reach 480 --step 60

The sources are read-only; the result is written to a new dataset directory:

    <dataset>/images/<video>_frame_<number>.jpg
    <dataset>/labels/<video>_frame_<number>.json   field_to_image, focal, game, the hand
                                                   label it comes from and how far away
    <dataset>/dataset.json                         ruleset, games, counts
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.camera_motion import CameraMotionEstimator  # noqa: E402
from ultimate_analysis.utils import field_label_files, field_template  # noqa: E402
from ultimate_analysis.utils.field_camera import fit_camera  # noqa: E402
from ultimate_analysis.utils.label_files import JPEG_QUALITY, frame_name  # noqa: E402

TRAINING_DATA = REPO / DEFAULT_PATHS["TRAINING_DATA"]
VIDEO_FOLDERS = (
    REPO / DEFAULT_PATHS["RAW_VIDEOS"],
    REPO / "data" / "processed" / "benchmark_clips",
)
# A label needs this many corners inside the frame to be carried: fewer do not show drift
MIN_CORNERS = 3


def game_of(video: str) -> str:
    """The game a video is from: its name without a clip's numbering."""
    return re.sub(r"_snippet_\d+_\d+$", "", video)


def reference_mappings(dataset: Path, template) -> Dict[str, dict]:
    """Frame name -> the hand label as a camera of its game's focal length."""
    labels = {}
    for name in field_label_files.labelled_frames(dataset):
        label = field_label_files.load_label(dataset, name)
        stored = json.loads((dataset / "labels" / f"{name}.json").read_text())
        size = tuple(stored["image_size"])
        inside = {
            key: pixel
            for key, pixel in label.points.items()
            if 0 <= pixel[0] < size[0] and 0 <= pixel[1] < size[1]
        }
        free = fit_camera(template, {}, label.points, size)
        if free is None or len(inside) < MIN_CORNERS:
            print(f"  left out, too few corners in the frame: {name}")
            continue
        labels[name] = {"size": size, "corners": inside, "free_focal": free.focal}

    # A drone does not zoom: one focal length per game, the middle of what its labels give
    for name, entry in labels.items():
        game = game_of(name.rsplit("_frame_", 1)[0])
        focal = float(
            np.median(
                [
                    other["free_focal"]
                    for other_name, other in labels.items()
                    if game_of(other_name.rsplit("_frame_", 1)[0]) == game
                ]
            )
        )
        fit = fit_camera(template, {}, entry["corners"], entry["size"], focal=focal)
        entry.update(game=game, focal=focal, field_to_image=fit.field_to_image if fit else None)
    return {name: entry for name, entry in labels.items() if entry["field_to_image"] is not None}


def carried(
    capture: cv2.VideoCapture,
    index: int,
    corners: np.ndarray,
    reach: int,
    step: int,
    max_drift: float,
) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
    """Follow the camera around a labelled frame.

    Args:
        capture: The video
        index: The labelled frame
        corners: The labelled corners in that frame, (n, 2) pixels
        reach: How many frames to go each way at most (even)
        step: A frame is taken every so many frames (even)
        max_drift: The two ways of following may disagree by this many pixels at most

    Returns:
        [(frame index, the frame, motion from the labelled frame to it, drift in pixels)],
        the labelled frame itself included
    """
    first = max(0, index - reach)
    first += (index - first) % 2  # The labelled frame is one of every second frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    every, second = CameraMotionEstimator(), CameraMotionEstimator()
    since_a, since_b = np.eye(3), np.eye(3)
    stretch_a = stretch_b = 0  # Counts up at every cut: motions across a cut mean nothing
    seen: Dict[int, tuple] = {}
    for position in range(first, index + reach + 1):
        ok, frame = capture.read()
        if not ok:
            break
        motion = every.update(frame, [])
        if position > first:
            if motion is None:
                stretch_a, since_a = stretch_a + 1, np.eye(3)
            else:
                since_a = motion @ since_a
        if (position - first) % 2 == 0:
            motion = second.update(frame, [])
            if position > first:
                if motion is None:
                    stretch_b, since_b = stretch_b + 1, np.eye(3)
                else:
                    since_b = motion @ since_b
        if (position - index) % step == 0:
            seen[position] = (frame, since_a.copy(), stretch_a, since_b.copy(), stretch_b)
    if index not in seen:
        return []

    _, at_label_a, label_stretch_a, at_label_b, label_stretch_b = seen[index]
    points = corners.reshape(-1, 1, 2).astype(np.float64)
    found = []
    for direction in (-1, 1):
        position = index if direction == 1 else index - step
        while position in seen:
            frame, a, a_stretch, b, b_stretch = seen[position]
            if a_stretch != label_stretch_a or b_stretch != label_stretch_b:
                break  # Beyond a cut
            motion_a = a @ np.linalg.inv(at_label_a)
            motion_b = b @ np.linalg.inv(at_label_b)
            drift = float(
                np.linalg.norm(
                    cv2.perspectiveTransform(points, motion_a)
                    - cv2.perspectiveTransform(points, motion_b),
                    axis=2,
                ).max()
            )
            if drift > max_drift:
                break  # And no further: what lies beyond was reached through here
            found.append((position, frame, motion_a, drift))
            position += direction * step
    return sorted(found, key=lambda item: item[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="Name of the new dataset folder in data/raw/training_data")
    parser.add_argument("--source", default="labelled_field_v1", help="The hand labels")
    parser.add_argument("--reach", type=int, default=480, help="Frames to go each way at most")
    parser.add_argument("--step", type=int, default=60, help="A frame every so many frames")
    parser.add_argument("--max-drift", type=float, default=3.0, help="Pixels at the corners")
    args = parser.parse_args()
    if args.reach % 2 or args.step % 2:
        parser.error("--reach and --step must be even")

    source = TRAINING_DATA / args.source
    output = TRAINING_DATA / args.name
    if output.exists():
        sys.exit(f"Refusing to overwrite existing dataset: {output}")
    ruleset = field_label_files.dataset_ruleset(source)
    template = field_template.TEMPLATES[ruleset]
    videos = {
        path.stem: path for folder in VIDEO_FOLDERS if folder.exists() for path in folder.glob("*")
    }

    labels = reference_mappings(source, template)
    (output / "images").mkdir(parents=True)
    (output / "labels").mkdir()
    written: Dict[str, Tuple[int, dict]] = {}  # Frame name -> (frames from its label, label file)
    per_game: Dict[str, List[int]] = {}
    for name, entry in sorted(labels.items()):
        video, index = name.rsplit("_frame_", 1)
        index = int(index)
        if video not in videos:
            print(f"  video not found, hand label only: {name}")
            frames = []
        else:
            capture = cv2.VideoCapture(str(videos[video]))
            corners = np.array(list(entry["corners"].values()), dtype=np.float64)
            frames = carried(capture, index, corners, args.reach, args.step, args.max_drift)
            capture.release()
        if not frames:
            image = cv2.imread(str(source / "images" / f"{name}.jpg"))
            frames = [(index, image, np.eye(3), 0.0)]

        kept = 0
        for position, frame, motion, drift in frames:
            target = frame_name(str(videos.get(video, video)), position)
            away = abs(position - index)
            # Two hand labels may reach the same frame: the nearer one is the better word
            if target in written and written[target][0] <= away:
                continue
            mapping = motion @ entry["field_to_image"]
            stored = {
                "ruleset": ruleset,
                "image_size": [int(frame.shape[1]), int(frame.shape[0])],
                "field_to_image": mapping.tolist(),
                "focal": entry["focal"],
                "game": entry["game"],
                "hand_label": name,
                "frames_from_hand_label": position - index,
                "drift_pixels": round(drift, 3),
            }
            cv2.imwrite(
                str(output / "images" / f"{target}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            (output / "labels" / f"{target}.json").write_text(json.dumps(stored, indent=2))
            written[target] = (away, stored)
            kept += 1
        per_game.setdefault(entry["game"], []).append(kept)
        span = (frames[0][0] - index, frames[-1][0] - index)
        print(f"  {name}: {kept} frames, from {span[0]:+d} to {span[1]:+d}", flush=True)

    games = {
        game: {
            "hand_labels": len(counts),
            "frames": sum(1 for _, stored in written.values() if stored["game"] == game),
        }
        for game, counts in sorted(per_game.items())
    }
    summary = {
        "ruleset": ruleset,
        "source": args.source,
        "reach": args.reach,
        "step": args.step,
        "max_drift": args.max_drift,
        "frames": len(written),
        "games": games,
    }
    (output / "dataset.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(written)} frames from {len(labels)} hand labels, written to {output}")
    for game, counts in games.items():
        print(f"  {game}: {counts['hand_labels']} hand labels -> {counts['frames']} frames")


if __name__ == "__main__":
    main()
