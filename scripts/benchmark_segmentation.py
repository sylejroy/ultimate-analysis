#!/usr/bin/env python3
"""Score the field segmentation model on labelled images.

The field lines are fitted to the outline of the predicted field, so what counts is how
well the predicted area and its outline match the labelled ones:

- IoU of the whole field (the unified mask the app fits lines to) and of each class
- outline error: the average distance between the predicted and the labelled field
  outline, in pixels of a 1920x1080 frame. Outline along the image border is left out,
  since no field line runs there.

The model runs through the app's own segmentation code. The Roboflow export stores the
16:9 frames stretched to a square; by default they are stretched back first, which is how
the app sees a video frame. --as-stored scores the square images the model was trained on.

    python scripts/benchmark_segmentation.py
    python scripts/benchmark_segmentation.py path/to/best.pt --splits test
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ultimate_analysis.processing.field_analysis import create_unified_field_mask  # noqa: E402
from ultimate_analysis.processing.field_segmentation import (  # noqa: E402
    reset_segmentation_cache,
    run_field_segmentation,
    set_field_model,
)
from ultimate_analysis.utils.model_files import (  # noqa: E402
    default_model_path,
    get_training_args,
    model_display_name,
)

FRAME_SIZE = (1920, 1080)  # (width, height) the outline error is reported at
# Outline this close to the image border is where the frame cuts the field off, not a field
# line; labels and predictions stop at slightly different distances from the border
BORDER = 30
SEAM_KERNEL = np.ones((25, 25), dtype=np.uint8)


def labelled_masks(
    label_path: Path, names: List[str], size: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """{class name: mask of its labelled polygons} at the given (width, height)."""
    width, height = size
    masks = {name: np.zeros((height, width), dtype=np.uint8) for name in names}
    for line in label_path.read_text().splitlines() if label_path.exists() else []:
        values = line.split()
        polygon = np.array(values[1:], dtype=np.float32).reshape(-1, 2) * (width, height)
        cv2.fillPoly(masks[names[int(values[0])]], [polygon.round().astype(np.int32)], 1)
    return masks


def predicted_masks(
    frame: np.ndarray, names: List[str]
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """({class name: predicted mask}, unified field mask) from the app's segmentation."""
    height, width = frame.shape[:2]
    reset_segmentation_cache()
    results = run_field_segmentation(frame, 0)

    masks = {name: np.zeros((height, width), dtype=np.uint8) for name in names}
    for result in results:
        if result.masks is None:
            continue
        for mask, class_id in zip(np.asarray(result.masks.data), result.boxes.cls.cpu().numpy()):
            mask = cv2.resize(
                mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
            )
            name = result.names[int(class_id)]
            if name in masks:
                np.maximum(masks[name], (mask > 0.5).view(np.uint8), out=masks[name])

    unified = create_unified_field_mask(results, (height, width))
    return masks, unified if unified is not None else np.zeros((height, width), dtype=np.uint8)


def iou(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    """Intersection over union of two masks; None when both are empty."""
    union = np.count_nonzero(first | second)
    return np.count_nonzero(first & second) / union if union else None


def outline(mask: np.ndarray) -> np.ndarray:
    """Outer outline pixels of a mask, without the parts along the image border.

    Only the outer outline counts: the labelled classes are separate polygons, and the
    thin seams between them are not an edge of the field.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.zeros_like(mask)
    cv2.drawContours(edge, contours, -1, 1, 1)
    edge[:BORDER] = edge[-BORDER:] = 0
    edge[:, :BORDER] = edge[:, -BORDER:] = 0
    return edge


def outline_error(predicted: np.ndarray, labelled: np.ndarray) -> Optional[float]:
    """Average distance between two outlines in pixels, measured in both directions."""
    predicted_edge, labelled_edge = outline(predicted), outline(labelled)
    if not predicted_edge.any() or not labelled_edge.any():
        return None
    to_labelled = cv2.distanceTransform(1 - labelled_edge, cv2.DIST_L2, 3)
    to_predicted = cv2.distanceTransform(1 - predicted_edge, cv2.DIST_L2, 3)
    return float(
        (to_labelled[predicted_edge > 0].mean() + to_predicted[labelled_edge > 0].mean()) / 2
    )


def summarize(label: str, values: List[Optional[float]], unit: str, worst_is_high: bool) -> str:
    scores = np.array([value for value in values if value is not None])
    if not len(scores):
        return f"{label:<28} no data"
    worst = np.percentile(scores, 90 if worst_is_high else 10)
    missing = (
        f", not scored in {len(values) - len(scores)} images" if len(scores) < len(values) else ""
    )
    return (
        f"{label:<28} mean {scores.mean():.3f}{unit}  median {np.median(scores):.3f}{unit}"
        f"  worst tenth beyond {worst:.3f}{unit}{missing}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", nargs="?", default=default_model_path("segmentation"))
    parser.add_argument(
        "--dataset", type=Path, help="default: the dataset the model was trained on"
    )
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--as-stored", action="store_true", help="do not stretch images to 16:9")
    args = parser.parse_args()

    dataset = args.dataset or Path(get_training_args(args.weights)["data"]).parent
    names = yaml.safe_load((dataset / "data.yaml").read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else names
    if not set_field_model(str(args.weights)):
        sys.exit(f"Could not load {args.weights}")

    field_iou: List[Optional[float]] = []
    error: List[Optional[float]] = []
    class_iou: Dict[str, List[Optional[float]]] = {name: [] for name in names}
    count = 0
    for split in args.splits:
        for image_path in sorted((dataset / split / "images").glob("*")):
            image = cv2.imread(str(image_path))
            frame = image if args.as_stored else cv2.resize(image, FRAME_SIZE)
            size = (frame.shape[1], frame.shape[0])
            labelled = labelled_masks(
                dataset / split / "labels" / f"{image_path.stem}.txt", names, size
            )
            predicted, predicted_field = predicted_masks(frame, names)
            # The class polygons of a label do not always touch; close the seam between them
            labelled_field = cv2.morphologyEx(
                np.maximum.reduce(list(labelled.values())), cv2.MORPH_CLOSE, SEAM_KERNEL
            )

            field_iou.append(iou(predicted_field, labelled_field))
            for name in names:
                class_iou[name].append(iou(predicted[name], labelled[name]))
            # Reported at the size of a video frame, whatever size was scored
            at_frame_size = [
                cv2.resize(m, FRAME_SIZE, interpolation=cv2.INTER_NEAREST)
                for m in (predicted_field, labelled_field)
            ]
            error.append(outline_error(*at_frame_size))
            count += 1

    shape = "as stored" if args.as_stored else "stretched back to 16:9"
    print(f"{model_display_name(args.weights)}")
    print(f"{dataset.name} ({', '.join(args.splits)}): {count} images, {shape}")
    print(summarize("Field IoU", field_iou, "", worst_is_high=False))
    for name in names:
        print(summarize(f"{name} IoU", class_iou[name], "", worst_is_high=False))
    print(summarize("Field outline error", error, " px", worst_is_high=True))


if __name__ == "__main__":
    main()
