#!/usr/bin/env python3
"""Build TensorRT engines for the models the app runs.

An engine is specific to a model, the video frame size, and this GPU and driver, and
takes a few minutes to build. The app picks it up on the next start and keeps using
PyTorch for any model without one.

Without arguments, engines are built for the default player, disc, and field
segmentation models from configs/default.yaml, for 1920x1080 video.
"""

import argparse
import os
import sys
from pathlib import Path

# Ultralytics otherwise pip-installs export helpers on its own, and their dependencies
# have replaced the CUDA build of PyTorch before. Missing packages should fail loudly.
os.environ["YOLO_AUTOINSTALL"] = "false"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ultimate_analysis.config.settings import get_setting  # noqa: E402
from ultimate_analysis.processing.inference import disc_window_image_size  # noqa: E402
from ultimate_analysis.processing.tensorrt_engines import (  # noqa: E402
    engine_path,
    export_engine,
    network_input_shape,
)
from ultimate_analysis.utils.model_files import (  # noqa: E402
    default_model_path,
    get_training_image_size,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "weights",
        nargs="*",
        type=Path,
        help="Detection weights (.pt) to build engines for; default: the configured models",
    )
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    parser.add_argument("--force", action="store_true", help="Rebuild engines that exist")
    args = parser.parse_args()

    half = bool(get_setting("models.inference.half_precision", False))
    frame_shape = (args.frame_height, args.frame_width)

    # (weights, network input shape, half precision)
    jobs = []
    detection_weights = args.weights or [
        Path(default_model_path("player_detection")),
        Path(default_model_path("disc_detection")),
    ]
    disc_weights = Path(default_model_path("disc_detection"))
    for weights in detection_weights:
        imgsz = get_training_image_size(weights)
        jobs.append((weights, network_input_shape(frame_shape, imgsz), half))
        # The disc model also searches a small window around a disc it is following
        window = disc_window_image_size(frame_shape, imgsz)
        if weights == disc_weights and window:
            jobs.append((weights, (window, window), half))
    if not args.weights:
        # Field segmentation always receives a square, letterboxed frame in full precision
        weights = Path(default_model_path("segmentation"))
        imgsz = get_training_image_size(weights)
        jobs.append((weights, (imgsz, imgsz), False))

    for weights, input_shape, half_precision in dict.fromkeys(jobs):
        weights = weights if weights.is_absolute() else REPO / weights
        target = engine_path(weights, input_shape, half_precision)
        if target.exists() and not args.force:
            print(f"Exists, skipping: {target}")
            continue
        print(f"Building {target.name} for {weights.parents[2].name} ...")
        try:
            export_engine(weights, input_shape, half_precision)
        except ValueError as e:
            print(f"Skipped: {e}")
            continue
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
