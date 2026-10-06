#!/usr/bin/env python3
"""Compare candidate scoring with color, grayscale, and reduced coverage warps.

All geometric objectives use original coordinates; only coverage rasterization changes.
Reports CPU time per population, fitness differences, and winning-candidate agreement.
This measures evaluation consistency, not whether the fitted field is physically correct.

    python scripts/benchmark_homography_optimizer.py
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ultimate_analysis.config.settings import get_config  # noqa: E402
from ultimate_analysis.optimization.homography_optimizer import (  # noqa: E402
    HomographyIndividual,
    HomographyOptimizer,
)


def load_frames(per_video):
    frames = []
    for video in sorted((REPO / "data" / "raw" / "videos").glob("*.mp4")):
        capture = cv2.VideoCapture(str(video))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for index in np.linspace(0.1 * count, 0.9 * count, per_video).astype(int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    frames.append(frame)
        finally:
            capture.release()
    if not frames:
        raise RuntimeError("No readable evaluation videos")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    frames = load_frames(args.frames_per_video)
    initial = yaml.safe_load((REPO / "configs" / "homography_params.yaml").read_text())[
        "homography_parameters"
    ]
    rng = np.random.default_rng(0)
    optimizer = HomographyOptimizer(initial, population_size=1)
    candidates = [HomographyIndividual(initial)]
    for _ in range(19):
        params = initial.copy()
        for key in params:
            amount = 0.0005 if key in ("H20", "H21") else 100 if key in ("H02", "H12") else 0.2
            params[key] += rng.normal(0, amount)
        candidates.append(HomographyIndividual(params))
    config = get_config()["optimization"]
    previous_scale = config["ga_coverage_scale"]
    modes = [("color", 1.0), ("gray", 1.0), ("gray", 0.5), ("gray", 0.25)]
    scores = {mode: [] for mode in modes}
    durations = {mode: [] for mode in modes}
    try:
        for frame in frames:
            h, w = frame.shape[:2]
            lines = [
                (np.array([0.2 * w, 0.4 * h]), np.array([0.8 * w, 0.4 * h])),
                (np.array([0.2 * w, 0.4 * h]), np.array([0.3 * w, 0.8 * h])),
            ]
            for repeat in range(args.repeats + 1):
                # Rotate order to reduce warmup and temperature bias.
                ordered = modes[repeat % len(modes) :] + modes[: repeat % len(modes)]
                for mode in ordered:
                    kind, scale = mode
                    config["ga_coverage_scale"] = scale
                    start = time.perf_counter()
                    source = frame if kind == "color" else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    result = [
                        optimizer.calculate_fitness(candidate, frame, lines, [1.0, 1.0], source)
                        for candidate in candidates
                    ]
                    elapsed = (time.perf_counter() - start) * 1000
                    if repeat:
                        durations[mode].append(elapsed)
                    if repeat == args.repeats:
                        scores[mode].append(result)
    finally:
        config["ga_coverage_scale"] = previous_scale
    reference = np.array(scores[modes[0]])
    print(f"{len(frames)} frames, {len(candidates)} candidates, {args.repeats} timed repeats")
    for mode in modes:
        values = np.array(scores[mode])
        agreement = np.mean(values.argmax(axis=1) == reference.argmax(axis=1))
        print(
            f"{mode[0]} scale={mode[1]}: median {np.median(durations[mode]):.2f} ms/population | max fitness difference {np.abs(values - reference).max():.6f} | same winner {agreement:.0%}"
        )


if __name__ == "__main__":
    main()
