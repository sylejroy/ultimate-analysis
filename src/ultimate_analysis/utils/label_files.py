"""Labelled video frames on disk, as a dataset that can be trained on directly.

A dataset folder holds the labelled frames at full resolution and their boxes in YOLO
format (one line per box: class, centre x, centre y, width, height, all relative to the
frame size):

    <dataset>/images/<video>_frame_<number>.jpg
    <dataset>/labels/<video>_frame_<number>.txt
    <dataset>/train.txt, val.txt, test.txt     which frames belong to which split
    <dataset>/data.yaml                        what the training reads

The name of a frame says where it comes from, so it can always be found in its video again.
"""

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

CLASS_NAMES = ["disc", "player"]
SPLITS = ("train", "val", "test")
# Frames of the same stretch of play look alike and must not end up in different splits.
# A video is cut into stretches of this many frames; one in ten goes to validation and
# one in ten to testing.
SPLIT_GROUP_FRAMES = 300
JPEG_QUALITY = 95


@dataclass
class LabelBox:
    """A labelled object; corners in frame pixels."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def frame_name(video_path: str, frame_index: int) -> str:
    """Name under which a frame of a video is stored."""
    return f"{Path(video_path).stem}_frame_{frame_index:06d}"


def frame_index_of(name: str) -> int:
    """Position in its video of a stored frame."""
    return int(name.rsplit("_frame_", 1)[1])


def split_of(name: str) -> str:
    """Split a frame belongs to. It only depends on the name, so it never changes."""
    video, index = name.rsplit("_frame_", 1)
    group = f"{video}:{int(index) // SPLIT_GROUP_FRAMES}"
    bucket = zlib.crc32(group.encode()) % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def load_boxes(
    dataset_dir: Path, name: str, frame_size: Tuple[int, int]
) -> Optional[List[LabelBox]]:
    """Boxes of a labelled frame, or None if the frame has not been labelled.

    Args:
        dataset_dir: Dataset folder
        name: Frame name
        frame_size: (width, height) of the frame
    """
    label_path = Path(dataset_dir) / "labels" / f"{name}.txt"
    if not label_path.exists():
        return None
    width, height = frame_size
    boxes = []
    for line in label_path.read_text().splitlines():
        class_id, x, y, w, h = line.split()
        x, y, w, h = float(x) * width, float(y) * height, float(w) * width, float(h) * height
        boxes.append(LabelBox(int(class_id), x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return boxes


def save_frame(
    dataset_dir: Path,
    name: str,
    frame: np.ndarray,
    boxes: List[LabelBox],
    class_names: Optional[List[str]] = None,
) -> None:
    """Store a frame with its boxes and bring the split lists up to date.

    Args:
        dataset_dir: Dataset folder
        name: Frame name
        frame: The video frame (BGR)
        boxes: Its labelled objects
        class_names: Classes of a new dataset (default: disc and player); an existing
            dataset keeps the classes it was started with
    """
    dataset_dir = Path(dataset_dir)
    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels").mkdir(parents=True, exist_ok=True)

    image_path = dataset_dir / "images" / f"{name}.jpg"
    if not image_path.exists():
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    height, width = frame.shape[:2]
    lines = []
    for box in boxes:
        x1, x2 = sorted((min(max(box.x1, 0), width), min(max(box.x2, 0), width)))
        y1, y2 = sorted((min(max(box.y1, 0), height), min(max(box.y2, 0), height)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(
            f"{box.class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f}"
            f" {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    (dataset_dir / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    write_splits(dataset_dir, class_names)


def remove_frame(dataset_dir: Path, name: str) -> None:
    """Take a frame out of the dataset again."""
    dataset_dir = Path(dataset_dir)
    (dataset_dir / "images" / f"{name}.jpg").unlink(missing_ok=True)
    (dataset_dir / "labels" / f"{name}.txt").unlink(missing_ok=True)
    write_splits(dataset_dir)


def labelled_frames(dataset_dir: Path, video_path: Optional[str] = None) -> List[str]:
    """Names of the labelled frames, of one video or of all, in order."""
    prefix = f"{Path(video_path).stem}_frame_" if video_path else ""
    labels = Path(dataset_dir) / "labels"
    if not labels.exists():
        return []
    return sorted(path.stem for path in labels.glob("*.txt") if path.stem.startswith(prefix))


def dataset_classes(dataset_dir: Path) -> List[str]:
    """Class names of a dataset (disc and player if it does not say)."""
    data_yaml = Path(dataset_dir) / "data.yaml"
    if data_yaml.exists():
        names = (yaml.safe_load(data_yaml.read_text()) or {}).get("names")
        if names:
            return list(names.values()) if isinstance(names, dict) else list(names)
    return list(CLASS_NAMES)


def write_splits(dataset_dir: Path, class_names: Optional[List[str]] = None) -> None:
    """Write the split lists and data.yaml for the frames that are labelled now."""
    dataset_dir = Path(dataset_dir)
    # A dataset that only holds discs must not be turned into one that claims to hold
    # players as well: its frames would then say that there are none
    if (dataset_dir / "data.yaml").exists() or class_names is None:
        class_names = dataset_classes(dataset_dir)
    names: Dict[str, List[str]] = {split: [] for split in SPLITS}
    for name in labelled_frames(dataset_dir):
        names[split_of(name)].append(name)
    for split in SPLITS:
        lines = [f"./images/{name}.jpg" for name in names[split]]
        (dataset_dir / f"{split}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "nc": len(class_names),
        "names": list(class_names),
    }
    (dataset_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))


def summary(dataset_dir: Path) -> Dict[str, int]:
    """Number of labelled frames, of frames per split, and of boxes per class."""
    class_names = dataset_classes(dataset_dir)
    counts = {"frames": 0, **{split: 0 for split in SPLITS}, **{name: 0 for name in CLASS_NAMES}}
    for name in labelled_frames(dataset_dir):
        counts["frames"] += 1
        counts[split_of(name)] += 1
        for line in (Path(dataset_dir) / "labels" / f"{name}.txt").read_text().splitlines():
            counts[class_names[int(line.split()[0])]] += 1
    return counts
