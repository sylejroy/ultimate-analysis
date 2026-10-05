#!/usr/bin/env python3
"""Build a combined player + disc detection dataset from the Roboflow exports.

The source exports were stretched to a square (960x960 / 1280x1280), which distorts
the 16:9 video frames the models later see. Normalized YOLO labels are unaffected by
that stretch, so every image is resized back to 16:9 and the labels are only remapped
to the two classes the pipeline uses (disc, player).

With --player-model, the disc-only dataset is added to the training split as well. Its
frames are unaugmented 1920x1080 originals with hand-labelled discs but no players, so
the given model supplies the player boxes. Frames that are held out in the other
sources are skipped, which keeps the validation and test splits identical to v1.

The sources are read-only; the result is written to a new dataset directory.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
import yaml

TRAINING_DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "training_data"

CLASS_NAMES = ["disc", "player"]
DISC, PLAYER = 0, 1

# Source dataset -> {source class id: merged class id}
SOURCES = {
    # names: ['disc', 'player'] (referees were annotated as players)
    "roboflow_object_detection_v3i": {0: DISC, 1: PLAYER},
    # names: ['disc', 'player', 'player in possession', 'ref']
    "roboflow_player_disc_detection_v4i": {0: DISC, 1: PLAYER, 2: PLAYER, 3: PLAYER},
}
# names: ['disc']; players are not annotated
DISC_ONLY_SOURCE = "roboflow_object_detection_disc_v1i"
SPLITS = ("train", "valid", "test")


def source_frame(stem: str) -> str:
    """Original frame name of a Roboflow image (without its export hash)."""
    return re.sub(r"_(jpg|jpeg|png)$", "", re.sub(r"\.rf\.[0-9a-f]+$", "", stem))


def read_labels(image_path: Path, class_map: Dict[int, int]) -> List[str]:
    """Label lines of an image, remapped to the merged classes."""
    label_path = image_path.parents[1] / "labels" / f"{image_path.stem}.txt"
    lines = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                sys.exit(f"Unexpected label format in {label_path}: {line!r}")
            lines.append(" ".join([str(class_map[int(parts[0])])] + parts[1:]))
    return lines


def write_sample(
    output_dir: Path, split: str, stem: str, image: np.ndarray, lines: List[str], size: tuple
) -> None:
    """Write one image, resized to the output size, with its labels."""
    interpolation = cv2.INTER_AREA if image.shape[0] >= size[1] else cv2.INTER_LINEAR
    resized = cv2.resize(image, size, interpolation=interpolation)
    cv2.imwrite(
        str(output_dir / split / "images" / f"{stem}.jpg"),
        resized,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    (output_dir / split / "labels" / f"{stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else "")
    )


def count_classes(lines: List[str], counts: List[int]) -> None:
    for line in lines:
        counts[int(line.split()[0])] += 1


def add_disc_only_source(
    output_dir: Path,
    size: tuple,
    held_out_frames: Set[str],
    player_model: Path,
    confidence: float,
    device: str,
) -> None:
    """Add the disc-only frames to the training split with model-labelled players."""
    from ultralytics import YOLO

    model = YOLO(str(player_model))
    player_class = next(i for i, name in model.names.items() if name == "player")

    counts = [0] * len(CLASS_NAMES)
    added = skipped = 0
    for split in SPLITS:
        for image_path in sorted((TRAINING_DATA / DISC_ONLY_SOURCE / split / "images").glob("*")):
            if source_frame(image_path.stem) in held_out_frames:
                skipped += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                sys.exit(f"Unreadable image: {image_path}")

            lines = read_labels(image_path, {0: DISC})
            result = model.predict(
                image,
                imgsz=size[0],
                conf=confidence,
                classes=[player_class],
                device=device,
                verbose=False,
            )[0]
            for x, y, w, h in result.boxes.xywhn.tolist():
                lines.append(f"{PLAYER} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            count_classes(lines, counts)
            write_sample(
                output_dir, "train", f"s{len(SOURCES)}_{image_path.stem}", image, lines, size
            )
            added += 1

    print(
        f"{DISC_ONLY_SOURCE} -> train: {added} images, labels {dict(zip(CLASS_NAMES, counts))}"
        f" (players labelled by {player_model.name}); skipped {skipped} held-out frames"
    )


def build(
    output_dir: Path,
    width: int,
    height: int,
    player_model: Optional[Path] = None,
    confidence: float = 0.4,
    device: str = "cpu",
) -> None:
    if output_dir.exists():
        sys.exit(f"Refusing to overwrite existing dataset: {output_dir}")

    for split in SPLITS:
        (output_dir / split / "images").mkdir(parents=True)
        (output_dir / split / "labels").mkdir(parents=True)

    size = (width, height)
    held_out_frames: Set[str] = set()
    for index, (source, class_map) in enumerate(SOURCES.items()):
        for split in SPLITS:
            images = sorted((TRAINING_DATA / source / split / "images").glob("*"))
            counts = [0] * len(CLASS_NAMES)
            for image_path in images:
                image = cv2.imread(str(image_path))
                if image is None:
                    sys.exit(f"Unreadable image: {image_path}")

                lines = read_labels(image_path, class_map)
                count_classes(lines, counts)
                if split != "train":
                    held_out_frames.add(source_frame(image_path.stem))

                # Prefix keeps names unique across sources
                write_sample(output_dir, split, f"s{index}_{image_path.stem}", image, lines, size)
            print(
                f"{source} {split}: {len(images)} images, labels {dict(zip(CLASS_NAMES, counts))}"
            )

    if player_model is not None:
        add_disc_only_source(output_dir, size, held_out_frames, player_model, confidence, device)

    data_yaml = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"Wrote {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="roboflow_merged_players_discs_v1")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--player-model",
        type=Path,
        help="Trained weights used to label players in the disc-only dataset; "
        "without it that dataset is left out",
    )
    parser.add_argument("--player-confidence", type=float, default=0.4)
    parser.add_argument("--device", default="cpu", help="Device for the player model")
    args = parser.parse_args()
    build(
        TRAINING_DATA / args.name,
        args.width,
        args.height,
        args.player_model,
        args.player_confidence,
        args.device,
    )
