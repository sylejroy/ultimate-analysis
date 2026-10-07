#!/usr/bin/env python3
"""Build a disc or a player dataset from the Roboflow data and the frames labelled with this app.

The merged Roboflow dataset keeps its images and its splits, so a model trained on the
result can be scored on the old test set like the models before it. The frames labelled
in the Labelling tab and from the phone are added with their boxes of that one class
(the phone frames hold discs only and go into disc datasets only).

A labelled frame goes to the split its name gives it, unless it lies within a few seconds
of a frame of the old dataset in the same game: then it follows that frame's split. A
frame next to an old test frame would otherwise let a model train on what it is tested
on. Frames close to old frames of different splits are left out.

    python scripts/build_combined_disc_dataset.py combined_discs_v1
    python scripts/build_combined_disc_dataset.py combined_players_v1 --object player

The sources are read-only; the result is written to a new dataset directory.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ultimate_analysis.utils import label_files  # noqa: E402

TRAINING_DATA = REPO / "data" / "raw" / "training_data"
ROBOFLOW_SOURCES = {"disc": "roboflow_merged_discs_v2", "player": "roboflow_merged_players_v2"}
LABELLED_SOURCES = ("labelled_players_discs_v1", "labelled_discs_v1")
# The old frames whose names carry no game are all from this one
UNNAMED_VIDEO = "portland_vs_san_francisco_2024"
SPLITS = ("train", "valid", "test")
NEAR_FRAMES = 300  # About 10 seconds: the same players at about the same places

Frame = Tuple[str, int]  # (video name without ending, frame index)


def old_frame(image_name: str) -> Optional[Frame]:
    """Game and frame index of an image of the merged Roboflow dataset, from its name."""
    match = re.match(r"s\d+_(?:(.+?)_)?frame_?(\d+)", image_name)
    if not match:
        return None
    return (match.group(1) or UNNAMED_VIDEO, int(match.group(2)))


def split_for(frame: Frame, own_split: str, old_frames: Dict[str, List[Frame]]) -> Optional[str]:
    """Split a labelled frame goes to, or None if it has to be left out."""
    near = {
        split
        for split, frames in old_frames.items()
        if any(
            video == frame[0] and abs(index - frame[1]) <= NEAR_FRAMES for video, index in frames
        )
    }
    if not near:
        return own_split
    return near.pop() if len(near) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="Name of the new dataset folder in data/raw/training_data")
    parser.add_argument("--object", choices=sorted(ROBOFLOW_SOURCES), default="disc")
    args = parser.parse_args()
    kind = args.object

    output_dir = TRAINING_DATA / args.name
    if output_dir.exists():
        sys.exit(f"Refusing to overwrite existing dataset: {output_dir}")
    for split in SPLITS:
        (output_dir / split / "images").mkdir(parents=True)
        (output_dir / split / "labels").mkdir(parents=True)

    # The Roboflow data: as it is
    old_frames: Dict[str, List[Frame]] = {}
    source = TRAINING_DATA / ROBOFLOW_SOURCES[kind]
    for split in SPLITS:
        images = sorted((source / split / "images").glob("*"))
        old_frames[split] = [frame for image in images if (frame := old_frame(image.name))]
        for image in images:
            shutil.copy2(image, output_dir / split / "images" / image.name)
            label = source / split / "labels" / f"{image.stem}.txt"
            shutil.copy2(label, output_dir / split / "labels" / label.name)
        print(f"{ROBOFLOW_SOURCES[kind]} {split}: {len(images)} images")

    # The frames labelled here: boxes of the one class only
    for name in LABELLED_SOURCES:
        source = TRAINING_DATA / name
        if kind not in label_files.dataset_classes(source):
            continue  # No such boxes were drawn there: its frames do not say "none here"
        disc_id = str(label_files.dataset_classes(source).index(kind))
        counts = {split: [0, 0] for split in SPLITS}  # images, boxes
        moved = left_out = 0
        for stem in label_files.labelled_frames(source):
            video, index = stem.rsplit("_frame_", 1)
            own_split = label_files.split_of(stem).replace("val", "valid")
            split = split_for((video, int(index)), own_split, old_frames)
            if split is None:
                left_out += 1
                continue
            moved += split != own_split
            lines = (source / "labels" / f"{stem}.txt").read_text().splitlines()
            discs = [
                "0 " + line.split(maxsplit=1)[1] for line in lines if line.split()[0] == disc_id
            ]
            image = next((source / "images").glob(f"{stem}.*"))
            # Two sources may hold the same frame; the first one wins
            target = output_dir / split / "images" / f"labelled_{image.name}"
            if any((output_dir / other / "images" / target.name).exists() for other in SPLITS):
                continue
            shutil.copy2(image, target)
            (output_dir / split / "labels" / f"labelled_{stem}.txt").write_text(
                "\n".join(discs) + ("\n" if discs else "")
            )
            counts[split][0] += 1
            counts[split][1] += len(discs)
        summary = ", ".join(
            f"{split} {images} images / {discs} {kind}s"
            for split, (images, discs) in counts.items()
        )
        print(f"{name}: {summary}; {moved} followed an old frame nearby, {left_out} left out")

    data = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": [kind],
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Written to {output_dir}")


if __name__ == "__main__":
    main()
