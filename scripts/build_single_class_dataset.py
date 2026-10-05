#!/usr/bin/env python3
"""Copy a YOLO detection dataset keeping the labels of one class only.

A model trained on the result detects just that class, e.g. a dedicated disc detector.
All images are kept: the ones without the class teach the model what is not one.

    python scripts/build_single_class_dataset.py roboflow_merged_players_discs_v2 disc roboflow_merged_discs_v2

The source is read-only; the result is written to a new dataset directory.
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

TRAINING_DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "training_data"
SPLITS = ("train", "valid", "test")


def build(source_dir: Path, class_name: str, output_dir: Path) -> None:
    names = yaml.safe_load((source_dir / "data.yaml").read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else names
    if class_name not in names:
        sys.exit(f"{source_dir.name} has no class {class_name!r}; its classes are {names}")
    if output_dir.exists():
        sys.exit(f"Refusing to overwrite existing dataset: {output_dir}")
    class_id = str(names.index(class_name))

    for split in SPLITS:
        images = sorted((source_dir / split / "images").glob("*"))
        (output_dir / split / "images").mkdir(parents=True)
        (output_dir / split / "labels").mkdir(parents=True)
        labelled = boxes = 0
        for image_path in images:
            shutil.copy2(image_path, output_dir / split / "images" / image_path.name)

            label_path = source_dir / split / "labels" / f"{image_path.stem}.txt"
            lines = label_path.read_text().splitlines() if label_path.exists() else []
            kept = [
                "0 " + line.split(maxsplit=1)[1] for line in lines if line.split()[0] == class_id
            ]
            (output_dir / split / "labels" / f"{image_path.stem}.txt").write_text(
                "\n".join(kept) + ("\n" if kept else "")
            )
            labelled += bool(kept)
            boxes += len(kept)
        print(f"{split}: {len(images)} images, {labelled} with a {class_name}, {boxes} boxes")

    data_yaml = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": [class_name],
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"Wrote {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="Dataset folder name in data/raw/training_data")
    parser.add_argument("class_name", help="Class to keep")
    parser.add_argument("name", help="Name of the new dataset folder")
    args = parser.parse_args()
    build(TRAINING_DATA / args.source, args.class_name, TRAINING_DATA / args.name)
