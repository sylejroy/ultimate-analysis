#!/usr/bin/env python3
"""Build a disc dataset of tiles cut from full-resolution frames.

A disc is about 17 pixels wide in a 1920x1080 frame. Shrinking the frame to 1280 for the
model shrinks the disc to 11 pixels. A model trained on tiles sees the disc at its full
size without the cost of training on whole full-resolution frames; in use it runs on the
whole frame at 1920.

Only frames that exist at full resolution are used:

- the frames labelled with this app, with the splits they have in a combined dataset
  (scripts/build_combined_disc_dataset.py), so nothing next to a test frame is trained on
- `roboflow_object_detection_disc_v1i` for training, as in the merged datasets. Some of
  its frames are also among the validation and test images of the merged dataset, at a
  lower resolution and under another name; those are left out here as they are there.

Each frame is cut into overlapping tiles. Every tile with a disc is kept, and as many
tiles without one, which show the model what is not a disc.

    python scripts/build_disc_tile_dataset.py combined_discs_v2 tiles_discs_v1

The sources are read-only; the result is written to a new dataset directory.
"""

import argparse
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import yaml

REPO = Path(__file__).resolve().parents[1]
TRAINING_DATA = REPO / "data" / "raw" / "training_data"
ROBOFLOW_SOURCE = "roboflow_object_detection_disc_v1i"  # 1920x1080 frames, class 0 = disc
SPLITS = ("train", "valid", "test")
TILE = 960
MIN_VISIBLE = 0.5  # A disc cut by the tile's edge counts if this much of it is inside

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels


def read_boxes(label_path: Path, width: int, height: int) -> List[Box]:
    boxes = []
    for line in label_path.read_text().splitlines() if label_path.exists() else []:
        parts = line.split()
        if len(parts) < 5 or parts[0] != "0":
            continue
        cx, cy, w, h = (float(value) for value in parts[1:5])
        boxes.append(
            (
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            )
        )
    return boxes


def frame_of(image_name: str) -> Optional[Tuple[str, int]]:
    """Game and frame index in the name of a Roboflow image (with or without a source prefix)."""
    match = re.match(r"(?:s\d+_)?(.+?)_frame_?(\d+)", image_name)
    return (match.group(1), int(match.group(2))) if match else None


def tile_starts(length: int) -> List[int]:
    """Start positions of tiles that cover a side of the frame, overlapping by half."""
    if length <= TILE:
        return [0]
    count = max(2, round((length - TILE) / (TILE / 2)) + 1)
    return [round(i * (length - TILE) / (count - 1)) for i in range(count)]


def add_frame(
    image_path: Path, label_path: Path, output: Path, generator: random.Random
) -> Tuple[int, int]:
    """Write the tiles of one frame. Returns (tiles, discs) written."""
    frame = cv2.imread(str(image_path))
    if frame is None or frame.shape[0] < TILE:
        return 0, 0
    height, width = frame.shape[:2]
    boxes = read_boxes(label_path, width, height)

    with_disc, without_disc = [], []
    for top in tile_starts(height):
        for left in tile_starts(width):
            lines = []
            for x1, y1, x2, y2 in boxes:
                ix1, iy1 = max(x1, left), max(y1, top)
                ix2, iy2 = min(x2, left + TILE), min(y2, top + TILE)
                inside = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inside < MIN_VISIBLE * (x2 - x1) * (y2 - y1) or inside <= 0:
                    continue
                lines.append(
                    f"0 {((ix1 + ix2) / 2 - left) / TILE:.6f} {((iy1 + iy2) / 2 - top) / TILE:.6f} "
                    f"{(ix2 - ix1) / TILE:.6f} {(iy2 - iy1) / TILE:.6f}"
                )
            (with_disc if lines else without_disc).append((left, top, lines))

    # A frame without any disc still gives one tile, so empty scenes are seen as well
    keep = with_disc + generator.sample(
        without_disc, min(len(without_disc), max(1, len(with_disc)))
    )
    for left, top, lines in keep:
        name = f"{image_path.stem}_x{left}_y{top}"
        cv2.imwrite(
            str(output / "images" / f"{name}.jpg"),
            frame[top : top + TILE, left : left + TILE],
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        (output / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(keep), sum(len(lines) for _, _, lines in keep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "combined", help="Combined dataset whose labelled_ frames and splits are used"
    )
    parser.add_argument("name", help="Name of the new dataset folder in data/raw/training_data")
    args = parser.parse_args()

    output_dir = TRAINING_DATA / args.name
    if output_dir.exists():
        sys.exit(f"Refusing to overwrite existing dataset: {output_dir}")
    generator = random.Random(0)

    for split in SPLITS:
        output = output_dir / split
        (output / "images").mkdir(parents=True)
        (output / "labels").mkdir(parents=True)
        sources = [
            (image, TRAINING_DATA / args.combined / split / "labels" / f"{image.stem}.txt")
            for image in sorted(
                (TRAINING_DATA / args.combined / split / "images").glob("labelled_*")
            )
        ]
        if split == "train":
            held_out = {
                frame_of(image.name)
                for other in ("valid", "test")
                for image in (TRAINING_DATA / args.combined / other / "images").glob("*")
                if not image.name.startswith("labelled_")
            }
            for source_split in SPLITS:
                folder = TRAINING_DATA / ROBOFLOW_SOURCE / source_split
                sources += [
                    (image, folder / "labels" / f"{image.stem}.txt")
                    for image in sorted((folder / "images").glob("*"))
                    if frame_of(image.name) not in held_out
                ]
        tiles = discs = 0
        for image, label in sources:
            written = add_frame(image, label, output, generator)
            tiles += written[0]
            discs += written[1]
        print(f"{split}: {len(sources)} frames, {tiles} tiles, {discs} discs")

    data = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["disc"],
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Written to {output_dir}")


if __name__ == "__main__":
    main()
