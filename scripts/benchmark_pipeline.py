#!/usr/bin/env python3
"""Measure the complete analysis/rendering pipeline on a fixed video sequence.

Decode frames before timing, warm up without resetting tracking, then report stage costs
and optional cProfile results. Includes both rendered views, excludes decoding and Qt.
No training or export runs are started and existing model engines are reused.

    python scripts/benchmark_pipeline.py --output before.json
    python scripts/benchmark_pipeline.py --compare before.json     # after a change
    python scripts/benchmark_pipeline.py --profile                  # where the time goes

--compare reports on how many frames the track IDs, jersey numbers, or disc holder differ
from a saved run, and the largest box difference. Timings are only meaningful with
nothing else on the GPU; profiling adds overhead, so take throughput from unprofiled runs.
"""

import argparse
import collections
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ultimate_analysis.config.settings import get_setting  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.homography import load_default_matrix  # noqa: E402
from ultimate_analysis.processing.inference import set_disc_model, set_player_model  # noqa: E402
from ultimate_analysis.processing.player_id import initialize_player_id_system  # noqa: E402
from ultimate_analysis.utils.model_files import default_model_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output", type=Path, help="Optional JSON timings and track snapshots")
    parser.add_argument(
        "--compare", type=Path, help="Compare track snapshots with a saved reference"
    )
    args = parser.parse_args()
    if args.frames <= args.warmup or args.warmup < 0:
        parser.error("--frames must exceed --warmup, which must be nonnegative")
    if args.threads is not None:
        cv2.setNumThreads(args.threads)
    video = args.video or next(
        (REPO / "data" / "processed" / "dev_data").glob(get_setting("video.default_video"))
    )
    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        for _ in range(args.frames):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) <= args.warmup:
        raise RuntimeError("Not enough decoded frames for a timed run")
    if not set_player_model(default_model_path("player_detection")) or not set_disc_model(
        default_model_path("disc_detection")
    ):
        raise RuntimeError("Default detector unavailable")
    initialize_player_id_system()
    pipeline = AnalysisPipeline()
    pipeline.reset()
    pipeline.set_frame_rate(fps)
    pipeline.homography_matrix = load_default_matrix()
    options = PipelineOptions()
    profiler = cProfile.Profile()
    use_cuda = torch.cuda.is_available()
    stages = collections.defaultdict(list)
    totals, snapshots = [], []
    for index, frame in enumerate(frames):
        measured = index >= args.warmup
        if measured and args.profile:
            profiler.enable()
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = pipeline.process(frame, index, options)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if measured and args.profile:
            profiler.disable()
        if measured:
            totals.append(elapsed)
            for name, duration in result.timings.items():
                stages[name].append(duration)
            snapshots.append(
                {
                    "frame": index,
                    "tracks": [
                        {"id": track.track_id, "class": track.class_name, "bbox": track.bbox}
                        for track in result.tracks
                    ],
                    "numbers": {str(key): number for key, (number, _) in result.player_ids.items()},
                    "holder": result.holder_id,
                }
            )
    summary = {
        "video": video.name,
        "frames": len(totals),
        "warmup_frames": args.warmup,
        "profiled": args.profile,
        "cv_threads": cv2.getNumThreads(),
        "device": torch.cuda.get_device_name() if use_cuda else "CPU",
        "torch_version": torch.__version__,
        "fps": 1000 / np.mean(totals),
        "median_ms": float(np.median(totals)),
        "p95_ms": float(np.percentile(totals, 95)),
        "stages_mean_ms": {name: sum(values) / len(totals) for name, values in stages.items()},
    }
    print(json.dumps(summary, indent=2))
    report = {"summary": summary, "snapshots": snapshots}
    if args.compare:
        reference = json.loads(args.compare.read_text(encoding="utf-8"))["snapshots"]
        if [frame["frame"] for frame in reference] != [frame["frame"] for frame in snapshots]:
            raise ValueError("Comparison reports cover different frame indices")
        mismatches = {"track_ids": 0, "numbers": 0, "holder": 0}
        max_delta = 0.0
        for old, new in zip(reference, snapshots):
            old_tracks = {(track["id"], track["class"]): track["bbox"] for track in old["tracks"]}
            new_tracks = {(track["id"], track["class"]): track["bbox"] for track in new["tracks"]}
            mismatches["track_ids"] += old_tracks.keys() != new_tracks.keys()
            for key in old_tracks.keys() & new_tracks.keys():
                max_delta = max(
                    max_delta, float(np.max(np.abs(np.array(old_tracks[key]) - new_tracks[key])))
                )
            mismatches["numbers"] += old["numbers"] != new["numbers"]
            mismatches["holder"] += old["holder"] != new["holder"]
        report["comparison"] = {
            "comparison_frames": len(snapshots),
            "frames_differing": mismatches,
            "max_bbox_delta": max_delta,
        }
        print(json.dumps(report["comparison"], indent=2))
    if args.output:
        args.output.write_text(json.dumps(report), encoding="utf-8")
    if args.profile:
        pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(30)


if __name__ == "__main__":
    main()
