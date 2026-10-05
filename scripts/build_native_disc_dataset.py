#!/usr/bin/env python3
"""Build a disc detection dataset at the full resolution of the video (1920x1080).

A disc is about 17 pixels wide in a video frame. The Roboflow exports shrink the frames,
and with them the disc, so this dataset goes back to the original frames:

- `object_detection_disc.v1i` already holds unchanged 1920x1080 frames.
- `player disc detection.v4i` holds frames of one game stretched to 1280x1280. Their names
  carry the frame number, so the original frame is read from the video in data/raw/videos.
  Validation and test images are otherwise unchanged and keep their labels. Training
  images come in three augmented versions (slightly rotated, sheared, or cropped); the
  version that matches the original best is lined up with it and its labels are moved
  along.

Splits: the test frames are those of `players_discs_merged.v2`, so results stay comparable.
The validation set is the old one plus every seventh training frame, because a best epoch
picked on some 50 discs is partly luck.

The sources are read-only; the result is written to a new dataset directory.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
TRAINING_DATA = REPO / "data" / "raw" / "training_data"
VIDEOS = REPO / "data" / "raw" / "videos"

NATIVE_SOURCE = "object_detection_disc.v1i.yolov8"  # 1920x1080 frames, class 0 = disc
# Frames of these sources are held out for testing in players_discs_merged.v2
SAME_FRAMES_SOURCE = "object_detection.v3i.yolov8"  # the frames of NATIVE_SOURCE at 960x960
STRETCHED_SOURCE = "player disc detection.v4i.yolov8"  # class 0 = disc
STRETCHED_VIDEO = "portland_vs_san_francisco_2024.mp4"

SPLITS = ("train", "valid", "test")
FRAME_SIZE = (1920, 1080)
EXTRA_VALIDATION_EVERY = 7
MATCH_SIZE = (960, 540)  # Frames are compared and lined up at this size
FRAME_SEARCH = 2  # The frame number in a name can be off by this many frames
MIN_INLIERS = 60
DISC_WINDOW = 40  # Half the side of the area around a disc that frames are compared on

Box = Tuple[float, float, float, float]  # centre x, centre y, width, height in pixels


def source_frame(stem: str) -> str:
    """Original frame name of a Roboflow image (without its export hash)."""
    return re.sub(r"_(jpg|jpeg|png)$", "", re.sub(r"\.rf\.[0-9a-f]+$", "", stem))


def images_by_frame(dataset: str, split: str) -> Dict[str, List[Path]]:
    frames: Dict[str, List[Path]] = {}
    for image_path in sorted((TRAINING_DATA / dataset / split / "images").glob("*")):
        frames.setdefault(source_frame(image_path.stem), []).append(image_path)
    return frames


def disc_boxes(image_path: Path, size: Tuple[int, int] = FRAME_SIZE) -> List[Box]:
    """Disc labels of an image, scaled to a frame of the given (width, height)."""
    label_path = image_path.parents[1] / "labels" / f"{image_path.stem}.txt"
    boxes = []
    for line in label_path.read_text().splitlines() if label_path.exists() else []:
        class_id, x, y, w, h = line.split()
        if class_id == "0":
            boxes.append(
                (float(x) * size[0], float(y) * size[1], float(w) * size[0], float(h) * size[1])
            )
    return boxes


def small_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        cv2.resize(image, MATCH_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    )


def read_candidates(capture: cv2.VideoCapture, number: int) -> List[np.ndarray]:
    """The video frames around a frame number; the number in a name can be slightly off."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, number - FRAME_SEARCH))
    frames = []
    for _ in range(2 * FRAME_SEARCH + 1):
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    if not frames:
        sys.exit(f"Could not read frame {number} of {STRETCHED_VIDEO}")
    return frames


def difference(exported: np.ndarray, frame: np.ndarray, boxes: List[Box]) -> float:
    """How much an exported image (at frame size) differs from a video frame.

    Around the discs when there are any: a disc in flight moves many pixels per frame
    while the rest of the picture barely changes, so only there the right frame stands out.
    """
    if not boxes:
        a, b = small_gray(exported).astype(np.int16), small_gray(frame).astype(np.int16)
        return float(np.abs(a - b).mean())
    total = 0.0
    for x, y, _, _ in boxes:
        x1, y1 = int(max(0, x - DISC_WINDOW)), int(max(0, y - DISC_WINDOW))
        x2, y2 = int(x + DISC_WINDOW), int(y + DISC_WINDOW)
        a = cv2.cvtColor(exported[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.int16)
        b = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.int16)
        total += float(np.abs(a - b).mean()) if a.size else 255.0
    return total / len(boxes)


def line_up(augmented: np.ndarray, original: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    """Homography from an augmented image (at frame size) to the original frame.

    Returns the homography in frame pixels and the number of features that agree with it.
    """
    orb = cv2.ORB_create(3000)
    points_a, features_a = orb.detectAndCompute(small_gray(augmented), None)
    points_o, features_o = orb.detectAndCompute(small_gray(original), None)
    if features_a is None or features_o is None:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(features_a, features_o)
    if len(matches) < MIN_INLIERS:
        return None, 0
    source = np.float32([points_a[match.queryIdx].pt for match in matches])
    target = np.float32([points_o[match.trainIdx].pt for match in matches])
    matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
    if matrix is None:
        return None, 0
    scale = np.diag([FRAME_SIZE[0] / MATCH_SIZE[0], FRAME_SIZE[1] / MATCH_SIZE[1], 1.0])
    return scale @ matrix @ np.linalg.inv(scale), int(inliers.sum())


def move_boxes(boxes: List[Box], matrix: np.ndarray) -> List[Box]:
    """Boxes carried over by a homography (the box around the moved corners)."""
    moved = []
    for x, y, w, h in boxes:
        corners = np.float32(
            [
                [x - w / 2, y - h / 2],
                [x + w / 2, y - h / 2],
                [x + w / 2, y + h / 2],
                [x - w / 2, y + h / 2],
            ]
        ).reshape(-1, 1, 2)
        corners = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
        (x1, y1), (x2, y2) = corners.min(axis=0), corners.max(axis=0)
        moved.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return moved


def write_sample(
    output_dir: Path, split: str, name: str, frame: np.ndarray, boxes: List[Box]
) -> int:
    cv2.imwrite(
        str(output_dir / split / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    width, height = FRAME_SIZE
    lines = [
        f"0 {x / width:.6f} {y / height:.6f} {w / width:.6f} {h / height:.6f}"
        for x, y, w, h in boxes
        if 0 <= x < width and 0 <= y < height
    ]
    (output_dir / split / "labels" / f"{name}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else "")
    )
    return len(lines)


def build(output_dir: Path) -> None:
    if output_dir.exists():
        sys.exit(f"Refusing to overwrite existing dataset: {output_dir}")
    for split in SPLITS:
        (output_dir / split / "images").mkdir(parents=True)
        (output_dir / split / "labels").mkdir(parents=True)

    test_frames = set(images_by_frame(SAME_FRAMES_SOURCE, "test")) | set(
        images_by_frame(STRETCHED_SOURCE, "test")
    )
    validation_frames = set(images_by_frame(SAME_FRAMES_SOURCE, "valid")) | set(
        images_by_frame(STRETCHED_SOURCE, "valid")
    )
    counts = {split: [0, 0] for split in SPLITS}  # images, discs
    training_frames_seen = 0

    def split_of(frame_name: str, source_split: str) -> str:
        nonlocal training_frames_seen
        if frame_name in test_frames:
            return "test"
        if frame_name in validation_frames or source_split != "train":
            return "valid"
        training_frames_seen += 1
        return "valid" if training_frames_seen % EXTRA_VALIDATION_EVERY == 0 else "train"

    def add(split: str, name: str, frame: np.ndarray, boxes: List[Box]) -> None:
        counts[split][0] += 1
        counts[split][1] += write_sample(output_dir, split, name, frame, boxes)

    # Frames that are stored at full resolution already
    for source_split in SPLITS:
        for frame_name, paths in images_by_frame(NATIVE_SOURCE, source_split).items():
            frame = cv2.imread(str(paths[0]))
            if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
                sys.exit(f"Unexpected image size in {paths[0]}")
            add(split_of(frame_name, source_split), frame_name, frame, disc_boxes(paths[0]))

    # Frames that have to be fetched from the video
    capture = cv2.VideoCapture(str(VIDEOS / STRETCHED_VIDEO))
    skipped = 0
    for source_split in SPLITS:
        for frame_name, paths in images_by_frame(STRETCHED_SOURCE, source_split).items():
            number = int(re.search(r"frame_(\d+)", frame_name).group(1))
            candidates = read_candidates(capture, number)
            exported = [cv2.resize(cv2.imread(str(path)), FRAME_SIZE) for path in paths]

            if source_split == "train":
                # Augmented: line the best-fitting version up with each candidate frame
                # and move its labels along
                middle = candidates[len(candidates) // 2]
                version = max(range(len(exported)), key=lambda i: line_up(exported[i], middle)[1])
                options = []
                for frame in candidates:
                    matrix, inliers = line_up(exported[version], frame)
                    if matrix is None or inliers < MIN_INLIERS:
                        continue
                    boxes = move_boxes(disc_boxes(paths[version]), matrix)
                    restored = cv2.warpPerspective(exported[version], matrix, FRAME_SIZE)
                    options.append((difference(restored, frame, boxes), frame, boxes))
                if not options:
                    skipped += 1
                    continue
            else:
                boxes = disc_boxes(paths[0])
                options = [
                    (difference(exported[0], frame, boxes), frame, boxes) for frame in candidates
                ]
            _, original, boxes = min(options, key=lambda option: option[0])
            add(split_of(frame_name, source_split), f"portland_{frame_name}", original, boxes)
    capture.release()

    for split in SPLITS:
        print(f"{split}: {counts[split][0]} frames, {counts[split][1]} discs")
    print(f"Skipped {skipped} training frames whose augmented versions could not be lined up")

    data_yaml = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["disc"],
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"Wrote {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="discs_native.v3.yolov8")
    args = parser.parse_args()
    build(TRAINING_DATA / args.name)
