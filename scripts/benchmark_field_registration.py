"""Measure the estimate of where the field lies against the labelled frames.

For every frame of a field dataset (Labelling tab, "Field lines"), the field is estimated
from the field model's masks and compared with the label at the corners that were put on
the picture by hand, and only there: what a label says about the rest of the field
follows from those corners and is no measurement.

- pixels: how far the estimate draws a labelled corner from where it was put, the worst
  of a frame's corners
- field: how far the estimate maps the labelled pixel from the corner's place on the
  field, in the field's unit, the worst of a frame's corners. At the far end a pixel is
  about half a yard along the field, so this is the stricter number.

A frame counts as given if there is an estimate and it passes the check against the
players and the field mask (`implausible`); as wrong if it is given and more than
`--wrong` field units off. The focal length is that of the game's other labelled frames:
nothing of a frame's own label goes into its estimate.

Usage:
    python scripts/benchmark_field_registration.py
    python scripts/benchmark_field_registration.py --save-pictures out/
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ultimate_analysis.constants import DEFAULT_PATHS  # noqa: E402
from ultimate_analysis.processing.field_analysis import create_unified_field_mask  # noqa: E402
from ultimate_analysis.processing.field_registration import (  # noqa: E402
    estimate_field,
    implausible,
)
from ultimate_analysis.processing.field_segmentation import (  # noqa: E402
    reset_segmentation_cache,
    run_field_segmentation,
)
from ultimate_analysis.processing.inference import (  # noqa: E402
    detect_players,
    load_detection_model,
)
from ultimate_analysis.utils import field_label_files, field_template  # noqa: E402
from ultimate_analysis.utils.field_camera import fit_camera  # noqa: E402
from ultimate_analysis.utils.model_files import default_model_path  # noqa: E402


def game_of(name: str) -> str:
    """The game a frame is from: its video's name without the snippet and frame numbers."""
    return re.sub(r"(_snippet_\d+_\d+)?_frame_\d+$", "", name)


def corner_errors(
    field_to_image: np.ndarray, corners: Dict[str, Tuple[float, float]], template
) -> Tuple[float, float]:
    """(pixels, field units): how far off the estimate is at the worst labelled corner."""
    image_to_field = np.linalg.inv(field_to_image)
    pixels, units = [], []
    for name, (u, v) in corners.items():
        place = np.array(template.points[name])
        drawn = field_to_image @ [*place, 1.0]
        pixels.append(
            float(np.hypot(drawn[0] / drawn[2] - u, drawn[1] / drawn[2] - v))
            if drawn[2] > 0
            else float("inf")
        )
        mapped = image_to_field @ [u, v, 1.0]
        units.append(float(np.linalg.norm(mapped[:2] / mapped[2] - place)))
    return max(pixels), max(units)


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
    parser.add_argument("--wrong", type=float, default=2.0, help="Field units that count as wrong")
    parser.add_argument("--save-pictures", type=Path, help="Folder for the frames as estimated")
    args = parser.parse_args()

    dataset = REPO / DEFAULT_PATHS["TRAINING_DATA"] / args.dataset
    template = field_template.TEMPLATES[field_label_files.dataset_ruleset(dataset)]
    players = load_detection_model(default_model_path("player_detection"))
    if args.save_pictures:
        args.save_pictures.mkdir(parents=True, exist_ok=True)

    # The corners of each label that lie in the frame, and the focal length it gives
    labels = {}
    for name in field_label_files.labelled_frames(dataset):
        label = field_label_files.load_label(dataset, name)
        frame = cv2.imread(str(dataset / "images" / f"{name}.jpg"))
        if frame is None:
            continue
        size = (frame.shape[1], frame.shape[0])
        corners = {
            key: pixel
            for key, pixel in label.points.items()
            if 0 <= pixel[0] < size[0] and 0 <= pixel[1] < size[1]
        }
        free = fit_camera(template, {}, label.points, size)
        labels[name] = (frame, corners, free.focal if free is not None else None)

    rows = []  # (name, game, state, pixels, units)
    for name, (frame, corners, _) in labels.items():
        others = [
            focal
            for other, (_, _, focal) in labels.items()
            if other != name and game_of(other) == game_of(name) and focal is not None
        ]
        focal: Optional[float] = float(np.median(others)) if others else None

        reset_segmentation_cache()
        results = run_field_segmentation(frame, 0)
        estimate = estimate_field(results, frame.shape[:2], template, focal=focal)
        state, pixels, units = "none", float("nan"), float("nan")
        if estimate is not None:
            pixels, units = corner_errors(estimate.field_to_image, corners, template)
            boxes = np.array([d["bbox"] for d in detect_players(frame, *players)]).reshape(-1, 4)
            feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]])
            mask = create_unified_field_mask(results, frame.shape[:2])
            state = "left out" if implausible(estimate, template, mask, feet) else "given"
        rows.append((name, game_of(name), state, pixels, units))
        if args.save_pictures:
            for key, (u, v) in corners.items():
                cv2.circle(frame, (int(u), int(v)), 10, (0, 255, 255), 2)
            if estimate is not None:
                colour = (255, 0, 255) if state == "given" else (0, 0, 255)
                draw(frame, estimate.field_to_image, template, colour)
            cv2.imwrite(str(args.save_pictures / f"{name}.jpg"), frame)

    unit = template.unit
    print(f"{'game':<34} {'frames':>6} {'given':>6} {'wrong':>6} {'left out':>9} {'none':>5}")

    def line(title, chosen):
        given = [row for row in chosen if row[2] == "given"]
        wrong = sum(1 for row in given if row[4] > args.wrong)
        left_out = sum(1 for row in chosen if row[2] == "left out")
        none = sum(1 for row in chosen if row[2] == "none")
        print(
            f"{title[:34]:<34} {len(chosen):>6} {len(given):>6} {wrong:>6} {left_out:>9} {none:>5}"
        )

    for game in sorted({row[1] for row in rows}):
        line(game, [row for row in rows if row[1] == game])
    line("all", rows)

    given = np.array([[row[3], row[4]] for row in rows if row[2] == "given"])
    left_out = np.array([row[4] for row in rows if row[2] == "left out"])
    print(f"\nOf {len(rows)} frames, an estimate is given for {len(given)}.")
    if len(given):
        print(
            f"At the worst labelled corner of these: median {np.median(given[:, 0]):.1f} px / "
            f"{np.median(given[:, 1]):.2f} {unit}, 90% under "
            f"{np.percentile(given[:, 0], 90):.0f} px / {np.percentile(given[:, 1], 90):.1f} {unit}"
        )
        for limit in (5, 10, 20):
            print(f"  within {limit} px: {int((given[:, 0] <= limit).sum())} of {len(rows)} frames")
        print(
            f"  given and more than {args.wrong:g} {unit} off: "
            f"{int((given[:, 1] > args.wrong).sum())} of {len(given)}"
        )
    if len(left_out):
        kept_out_wrong = int((left_out > args.wrong).sum())
        print(
            f"Left out by the check: {len(left_out)}, of which {kept_out_wrong} were more than "
            f"{args.wrong:g} {unit} off"
        )


if __name__ == "__main__":
    main()
