#!/usr/bin/env python3
"""Build a dataset for the field segmentation model out of the field labels.

A field label says where the whole field lies in a frame, so the areas the segmentation
model is to find can be drawn from it: the central field and the end zones, as far as
they are in the picture. The frames come from a dataset built by
`build_field_registration_dataset.py` (hand labels carried along the video).

Whole games are kept out of the training: frames of one game look alike, so a model is
only measured on games it has not seen. One game checks the training while it runs
(validation), others are kept for `benchmark_field_registration.py --games ... --model ...`.

The images are stored stretched to a square, as the app gives frames to the model (see
`processing/field_segmentation.py`); the outlines are stored as shares of the picture, so
the stretching does not change them.

    python scripts/build_field_mask_dataset.py rendered_field_v1 \\
        --validation chicago_vs_new_york --test san_francisco_vs_colorado Truck_Stop_VS_Revolver

With --negatives, pictures that show no field from above (close-ups; see
`collect_field_negatives.py`) are added with nothing labelled in them, split by game
like the rest and named `negative_...`. Without them a model learns that grass is field.

The sources are read-only; the result is written to a new dataset directory in YOLO
segmentation format, with `data.yaml` for the Model Training tab.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.utils import field_template  # noqa: E402

TRAINING_DATA = REPO / DEFAULT_PATHS["TRAINING_DATA"]
# The classes of the field model, as the app knows them (processing/field_registration.py)
CLASSES = ["central-field", "endzone"]
# The existing hand-drawn outlines of one game, added to the training
ROBOFLOW_SOURCE = "roboflow_field_finder_v8i"
MIN_AREA_SHARE = 0.0015  # An area smaller than this share of the picture is not drawn
OUTLINE_PRECISION = 1.5  # Pixels an outline may be off after thinning out its points


def areas_in_frame(
    field_to_image: np.ndarray, template, size: Tuple[int, int]
) -> List[Tuple[int, np.ndarray]]:
    """The areas of the field as outlines in the frame: [(class, (n, 2) pixels)]."""
    width, height = size
    image_to_field = np.linalg.inv(field_to_image)
    xs, ys = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
    places = pixels @ image_to_field.T
    across = places[..., 0] / places[..., 2]
    along = places[..., 1] / places[..., 2]
    # Only what lies in front of the camera is seen
    in_front = (
        across * field_to_image[2, 0] + along * field_to_image[2, 1] + field_to_image[2, 2] > 0
    )
    on_field = in_front & (across >= 0) & (across <= template.width)
    far_goal = template.length - template.end_zone
    regions = (
        (0, on_field & (along >= template.end_zone) & (along <= far_goal)),
        (1, on_field & (along > far_goal) & (along <= template.length)),
        (1, on_field & (along >= 0) & (along < template.end_zone)),
    )
    found = []
    for class_id, region in regions:
        contours, _ = cv2.findContours(
            region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA_SHARE * width * height:
                continue
            outline = cv2.approxPolyDP(contour, OUTLINE_PRECISION, True).reshape(-1, 2)
            if len(outline) >= 3:
                found.append((class_id, outline.astype(np.float64)))
    return found


def label_lines(areas: List[Tuple[int, np.ndarray]], size: Tuple[int, int]) -> str:
    lines = []
    for class_id, outline in areas:
        shares = (outline / np.array(size)).clip(0.0, 1.0)
        lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in shares.ravel()))
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="Name of the new dataset folder in data/raw/training_data")
    parser.add_argument("--source", default="propagated_field_v1")
    parser.add_argument("--validation", nargs="+", required=True, help="Games (parts of names)")
    parser.add_argument("--test", nargs="+", default=[], help="Games left out altogether")
    parser.add_argument("--size", type=int, default=960, help="Side of the stored square images")
    parser.add_argument(
        "--without-roboflow", action="store_true", help="Leave the hand-drawn outlines out"
    )
    parser.add_argument("--negatives", help="Folder of pictures without a field to add")
    args = parser.parse_args()

    source = TRAINING_DATA / args.source
    output = TRAINING_DATA / args.name
    if output.exists():
        sys.exit(f"Refusing to overwrite existing dataset: {output}")
    summary = json.loads((source / "dataset.json").read_text())
    template = field_template.TEMPLATES[summary["ruleset"]]
    games = sorted(summary["games"])

    def split_of(game: str) -> str:
        if any(part in game for part in args.test):
            return "test"
        return "valid" if any(part in game for part in args.validation) else "train"

    splits = {game: split_of(game) for game in games}
    for wanted, names in (("valid", args.validation), ("test", args.test)):
        for part in names:
            if not any(part in game for game in games):
                sys.exit(f"No game of {args.source} is called like '{part}'")
    for split in ("train", "valid", "test"):
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)

    counts: Dict[str, List[int]] = {split: [0, 0] for split in ("train", "valid", "test")}
    for label_path in sorted((source / "labels").glob("*.json")):
        stored = json.loads(label_path.read_text())
        split = splits[stored["game"]]
        frame = cv2.imread(str(source / "images" / f"{label_path.stem}.jpg"))
        size = (frame.shape[1], frame.shape[0])
        areas = areas_in_frame(np.array(stored["field_to_image"]), template, size)
        if not areas:
            continue
        cv2.imwrite(
            str(output / split / "images" / f"{label_path.stem}.jpg"),
            cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_AREA),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        (output / split / "labels" / f"{label_path.stem}.txt").write_text(label_lines(areas, size))
        counts[split][0] += 1
        counts[split][1] += len(areas)

    negatives = {split: 0 for split in counts}
    if args.negatives:
        folder = TRAINING_DATA / args.negatives
        excluded = set((folder / "excluded.txt").read_text().split())
        for image in sorted((folder / "images").glob("*.jpg")):
            if image.stem in excluded:
                continue
            game = re.sub(r"_snippet_\d+_\d+$", "", image.stem.rsplit("_frame_", 1)[0])
            split = split_of(game)
            frame = cv2.imread(str(image))
            cv2.imwrite(
                str(output / split / "images" / f"negative_{image.name}"),
                cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            (output / split / "labels" / f"negative_{image.stem}.txt").write_text("")
            negatives[split] += 1

    roboflow = 0
    if not args.without_roboflow:
        # Hand-drawn outlines of one game (Portland v San Francisco), all into the training
        for part in ("train", "valid", "test"):
            folder = TRAINING_DATA / ROBOFLOW_SOURCE / part
            for image in sorted((folder / "images").glob("*")):
                label = folder / "labels" / f"{image.stem}.txt"
                if not label.exists():
                    continue
                target = output / "train" / "images" / f"roboflow_{image.name}"
                target.write_bytes(image.read_bytes())
                (output / "train" / "labels" / f"roboflow_{image.stem}.txt").write_bytes(
                    label.read_bytes()
                )
                roboflow += 1

    data = {
        "path": str(output),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    (output / "dataset.json").write_text(
        json.dumps({"source": args.source, "games": splits, "size": args.size}, indent=2)
    )
    for split, (images, areas) in counts.items():
        named = [game for game, where in splits.items() if where == split]
        print(
            f"{split}: {images} frames, {areas} areas, {negatives[split]} pictures without a "
            f"field, games: {', '.join(named)}"
        )
    print(f"plus {roboflow} hand-outlined images of {ROBOFLOW_SOURCE} in train")
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
