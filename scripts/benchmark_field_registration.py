"""Measure the estimate of where the field lies against the labelled frames.

For every frame of a field dataset (Labelling tab, "Field lines"), the field is estimated
from the field model's masks and compared with the label:

- picture: how far the places of the field that are in the frame lie from where the label
  has them, in pixels (median over a grid of places)
- field: how far the frame's pixels on the field are mapped from where the label maps them,
  in the field's unit (median over a grid of pixels)

The estimate is made twice: with the camera's focal length unknown, and with the focal
length the other labelled frames of the same game give (none of the frame's own label goes
into its estimate). The label is taken as the camera of the game's focal length that fits
its corners best, since corners at the far end alone leave the near end open.

Usage:
    python scripts/benchmark_field_registration.py
    python scripts/benchmark_field_registration.py --dataset labelled_field_v1 --save-pictures out/
"""

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.field_registration import estimate_field  # noqa: E402
from ultimate_analysis.processing.field_segmentation import (  # noqa: E402
    reset_segmentation_cache,
    run_field_segmentation,
)
from ultimate_analysis.utils import field_label_files, field_template  # noqa: E402
from ultimate_analysis.utils.field_camera import fit_camera  # noqa: E402

GRID_STEP = 5.0  # Field units between the places compared
PIXEL_STEP = 60  # Frame pixels between the pixels compared


def game_of(name: str) -> str:
    """The game a frame is from: its video's name without the snippet and frame numbers."""
    return re.sub(r"(_snippet_\d+_\d+)?_frame_\d+$", "", name)


def errors(estimate_to_image, label_to_image, template, frame_shape):
    """(median pixels, median field units) between an estimated mapping and the labelled one."""
    height, width = frame_shape
    xs = np.arange(0.0, template.width + 0.1, GRID_STEP)
    ys = np.arange(0.0, template.length + 0.1, GRID_STEP)
    places = np.array([[x, y, 1.0] for y in ys for x in xs])

    def to_image(mapping):
        mapped = places @ mapping.T
        front = mapped[:, 2] > 1e-9
        pixels = np.full((len(places), 2), np.nan)
        pixels[front] = mapped[front, :2] / mapped[front, 2:3]
        return pixels

    labelled, estimated = to_image(label_to_image), to_image(estimate_to_image)
    in_frame = (
        (labelled[:, 0] >= 0)
        & (labelled[:, 0] < width)
        & (labelled[:, 1] >= 0)
        & (labelled[:, 1] < height)
    )
    in_picture = np.linalg.norm(labelled[in_frame] - estimated[in_frame], axis=1)
    in_picture = np.where(np.isnan(in_picture), 10.0 * width, in_picture)

    pixels = np.array(
        [
            [x, y, 1.0]
            for y in range(PIXEL_STEP // 2, height, PIXEL_STEP)
            for x in range(PIXEL_STEP // 2, width, PIXEL_STEP)
        ]
    )

    def to_field(mapping):
        mapped = pixels @ np.linalg.inv(mapping).T
        return mapped[:, :2] / mapped[:, 2:3]

    labelled, estimated = to_field(label_to_image), to_field(estimate_to_image)
    on_field = (
        (labelled[:, 0] >= 0)
        & (labelled[:, 0] <= template.width)
        & (labelled[:, 1] >= 0)
        & (labelled[:, 1] <= template.length)
    )
    on_the_field = np.linalg.norm(labelled[on_field] - estimated[on_field], axis=1)
    return (
        float(np.median(in_picture)) if len(in_picture) else float("nan"),
        float(np.median(on_the_field)) if len(on_the_field) else float("nan"),
    )


def draw(frame, mapping, template, colour):
    fit = field_template.FieldFit(np.linalg.inv(mapping), mapping, 8, 0.0, ("", 0.0), 1.0)
    for start, end in template.lines.values():
        for part in field_template.field_segment_in_image(fit, start, end):
            cv2.polylines(frame, [np.int32(np.clip(part, -20000, 20000))], False, colour, 3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="labelled_field_v1")
    parser.add_argument("--save-pictures", type=Path, help="Folder for the frames as estimated")
    args = parser.parse_args()

    dataset = REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.dataset
    template = field_template.TEMPLATES[field_label_files.dataset_ruleset(dataset)]
    if args.save_pictures:
        args.save_pictures.mkdir(parents=True, exist_ok=True)

    # The corner dots of each label, and the focal length they give by themselves
    labels, focals = {}, {}
    for name in field_label_files.labelled_frames(dataset):
        label = field_label_files.load_label(dataset, name)
        frame = cv2.imread(str(dataset / "images" / f"{name}.jpg"))
        if frame is None or len(label.points) < 4:
            continue
        size = (frame.shape[1], frame.shape[0])
        free = fit_camera(template, {}, label.points, size)
        if free is not None:
            labels[name], focals[name] = (label, frame), free.focal

    print("Focal length each game's labels give (pixels):")
    for game in sorted({game_of(name) for name in labels}):
        values = sorted(round(focals[name]) for name in labels if game_of(name) == game)
        print(f"  {game}: {values}")

    rows = []
    for name, (label, frame) in labels.items():
        size = (frame.shape[1], frame.shape[0])
        others = [
            focals[other] for other in labels if other != name and game_of(other) == game_of(name)
        ]
        focal = float(np.median(others)) if others else None
        reference = fit_camera(template, {}, label.points, size, focal or focals[name])

        reset_segmentation_cache()
        results = run_field_segmentation(frame, 0)
        row = [name]
        for known in (None, focal):
            estimate = estimate_field(results, frame.shape[:2], template, focal=known)
            row.append(
                errors(estimate.field_to_image, reference.field_to_image, template, frame.shape[:2])
                if estimate is not None
                else None
            )
        row.append(sorted(estimate.lines) if estimate is not None else [])
        rows.append(row)
        if args.save_pictures:
            draw(frame, reference.field_to_image, template, (0, 255, 255))
            if estimate is not None:
                draw(frame, estimate.field_to_image, template, (255, 0, 255))
            cv2.imwrite(str(args.save_pictures / f"{name}.jpg"), frame)

    unit = template.unit
    print(f"\n{'':<46} {'focal unknown':>18} {'focal known':>18}")
    print(f"{'frame':<46} {'px':>9} {unit:>8} {'px':>9} {unit:>8}  lines")
    for name, free, known, lines in rows:
        short = name[:20] + ".." + name[-20:] if len(name) > 44 else name
        cells = "".join(
            f" {found[0]:>9.1f} {found[1]:>8.2f}" if found else f" {'-':>9} {'-':>8}"
            for found in (free, known)
        )
        print(f"{short:<46}{cells}  {', '.join(name[:-5] for name in lines)}")
    for title, column in (("focal unknown", 1), ("focal known", 2)):
        done = np.array([row[column] for row in rows if row[column] is not None])
        print(f"\n{title}: estimated {len(done)} of {len(rows)} frames")
        if len(done):
            print(
                f"  median over these: {np.median(done[:, 0]):.1f} px, "
                f"{np.median(done[:, 1]):.2f} {unit}"
            )
            print(
                "  within 1 / 2 / 5 "
                + unit
                + ": "
                + " / ".join(str(int(np.sum(done[:, 1] <= limit))) for limit in (1.0, 2.0, 5.0))
                + f" of {len(rows)} frames"
            )


if __name__ == "__main__":
    main()
