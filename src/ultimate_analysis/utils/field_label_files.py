"""Frames with labelled field lines and marks on disk.

A dataset folder holds the frames at full resolution and one file per frame that says
where the elements of the field are in it:

    <dataset>/images/<video>_frame_<number>.jpg
    <dataset>/labels/<video>_frame_<number>.json
    <dataset>/train.txt, val.txt, test.txt     which frames belong to which split
    <dataset>/dataset.json                     the ruleset the field is drawn by

A label file holds the pixels as they were clicked, and the mapping from picture to field
they give, so a frame can be used as a verified calibration without fitting it again.
Frames are named and split as in the datasets of boxes (label_files.py).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import field_template
from .label_files import JPEG_QUALITY, SPLITS, split_of

Point = Tuple[float, float]


@dataclass
class FieldLabel:
    """The elements of the field labelled in one frame, in frame pixels."""

    lines: Dict[str, List[Point]] = field(default_factory=dict)  # Name -> two pixels on the line
    points: Dict[str, Point] = field(default_factory=dict)  # Name -> the pixel of the mark

    def is_empty(self) -> bool:
        return not self.lines and not self.points

    def copy(self) -> "FieldLabel":
        return FieldLabel(
            {name: [tuple(pixel) for pixel in pixels] for name, pixels in self.lines.items()},
            {name: tuple(pixel) for name, pixel in self.points.items()},
        )


def dataset_ruleset(dataset_dir: Path) -> str:
    """Ruleset a dataset's fields are drawn by (the default for a new dataset)."""
    path = Path(dataset_dir) / "dataset.json"
    if path.exists():
        return json.loads(path.read_text()).get("ruleset", field_template.DEFAULT_RULESET)
    return field_template.DEFAULT_RULESET


def labelled_frames(dataset_dir: Path, video_path: Optional[str] = None) -> List[str]:
    """Names of the labelled frames, of one video or of all, in order."""
    prefix = f"{Path(video_path).stem}_frame_" if video_path else ""
    labels = Path(dataset_dir) / "labels"
    if not labels.exists():
        return []
    return sorted(path.stem for path in labels.glob("*.json") if path.stem.startswith(prefix))


def load_label(dataset_dir: Path, name: str) -> Optional[FieldLabel]:
    """The label of a frame, or None if the frame is not in the dataset."""
    path = Path(dataset_dir) / "labels" / f"{name}.json"
    if not path.exists():
        return None
    stored = json.loads(path.read_text())
    return FieldLabel(
        {key: [tuple(pixel) for pixel in pixels] for key, pixels in stored["lines"].items()},
        {key: tuple(pixel) for key, pixel in stored["points"].items()},
    )


def save_frame(
    dataset_dir: Path,
    name: str,
    frame: np.ndarray,
    label: FieldLabel,
    ruleset: Optional[str] = None,
) -> None:
    """Store a frame with its label; the dataset's splits are brought up to date.

    The mapping the label gives is stored with it, or null if the elements do not fix one
    yet: such a frame still teaches where the lines are.
    """
    dataset_dir = Path(dataset_dir)
    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels").mkdir(parents=True, exist_ok=True)
    ruleset = ruleset or dataset_ruleset(dataset_dir)
    (dataset_dir / "dataset.json").write_text(json.dumps({"ruleset": ruleset}, indent=2))

    image_path = dataset_dir / "images" / f"{name}.jpg"
    if not image_path.exists():
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    fit = field_template.fit_field(field_template.TEMPLATES[ruleset], label.lines, label.points)
    stored = {
        "ruleset": ruleset,
        "image_size": [int(frame.shape[1]), int(frame.shape[0])],
        "lines": {
            key: [list(map(float, pixel)) for pixel in pixels]
            for key, pixels in label.lines.items()
        },
        "points": {key: list(map(float, pixel)) for key, pixel in label.points.items()},
        "image_to_field": fit.image_to_field.tolist() if fit else None,
        "misfit_pixels": round(fit.error, 3) if fit else None,
    }
    (dataset_dir / "labels" / f"{name}.json").write_text(json.dumps(stored, indent=2))
    write_splits(dataset_dir)


def remove_frame(dataset_dir: Path, name: str) -> None:
    """Take a frame out of the dataset again."""
    dataset_dir = Path(dataset_dir)
    for path in (dataset_dir / "labels" / f"{name}.json", dataset_dir / "images" / f"{name}.jpg"):
        if path.exists():
            path.unlink()
    write_splits(dataset_dir)


def write_splits(dataset_dir: Path) -> None:
    """Write the lists of which frames belong to which split."""
    dataset_dir = Path(dataset_dir)
    names = labelled_frames(dataset_dir)
    for split in SPLITS:
        lines = [f"./images/{name}.jpg" for name in names if split_of(name) == split]
        (dataset_dir / f"{split}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def summary(dataset_dir: Path) -> Dict[str, int]:
    """Counts for the dataset: frames, frames with a mapping, and frames per split."""
    names = labelled_frames(dataset_dir)
    counts = {"frames": len(names), "calibrated": 0, **{split: 0 for split in SPLITS}}
    for name in names:
        counts[split_of(name)] += 1
        stored = json.loads((Path(dataset_dir) / "labels" / f"{name}.json").read_text())
        counts["calibrated"] += stored.get("image_to_field") is not None
    return counts
