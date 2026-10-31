#!/usr/bin/env python3
"""Measure the complete analysis/rendering pipeline on a fixed video sequence.

Decode frames before timing, warm up without resetting tracking, then report stage costs
and optional cProfile results. Includes both rendered views, excludes decoding and Qt.
No training or export runs are started and existing model engines are reused.

    python scripts/benchmark_pipeline.py --output before.json
    python scripts/benchmark_pipeline.py --compare before.json     # after a change
    python scripts/benchmark_pipeline.py --profile                  # where the time goes
    python scripts/benchmark_pipeline.py --ahead                    # as the app plays

--compare reports on how many frames the track IDs, jersey numbers, or disc holder differ
from a saved run, and the largest box difference. Timings are only meaningful with
nothing else on the GPU; profiling adds overhead, so take throughput from unprofiled runs.

Before timing, the load of the graphics card and the processor is looked at for a few
seconds; if something else is using them the run stops (--busy-ok times it anyway and
marks it). The summary holds the git commit, the settings changed with --set, and that
load (and the day and the command line). What a part costs is found by switching it off:

    python scripts/benchmark_pipeline.py --ahead --set models.reid.enabled=false
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
# The clip the timings in docs/MEASUREMENTS.md were taken on; without it, the app's default video
BENCHMARK_CLIP = (
    REPO
    / "data"
    / "processed"
    / "benchmark_clips"
    / "portland_vs_san_francisco_2024_snippet_4_40912.mp4"
)
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ultimate_analysis.config.settings import get_setting  # noqa: E402
from ultimate_analysis.pipeline import AnalysisPipeline, PipelineOptions  # noqa: E402
from ultimate_analysis.processing.inference import set_disc_model, set_player_model  # noqa: E402
from ultimate_analysis.processing.player_id import initialize_player_id_system  # noqa: E402
from ultimate_analysis.utils.benchmarking import apply_settings, run_header  # noqa: E402
from ultimate_analysis.utils.model_files import default_model_path  # noqa: E402

# Above this the machine is taken to be busy with something else (percent, averaged)
BUSY_GPU = 10.0
BUSY_CPU = 25.0


def machine_load(seconds: float = 3.0) -> dict:
    """How busy the graphics card and the processor are while nothing of ours runs."""
    import psutil

    load = {"cpu_percent": psutil.cpu_percent(interval=seconds)}
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        samples = []
        for _ in range(5):
            samples.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            time.sleep(0.2)
        load["gpu_percent"] = float(np.mean(samples))
        load["gpu_processes"] = len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
        pynvml.nvmlShutdown()
    except Exception as error:  # No NVIDIA card or driver library: nothing to say
        load["gpu_error"] = str(error)
    load["busy"] = load["cpu_percent"] > BUSY_CPU or load.get("gpu_percent", 0.0) > BUSY_GPU
    return load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--ahead",
        action="store_true",
        help="Give the pipeline the next frame as well, as the app does while it plays",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON timings and track snapshots")
    parser.add_argument(
        "--compare", type=Path, help="Compare track snapshots with a saved reference"
    )
    parser.add_argument("--set", action="append", default=[], help="setting=value, as JSON")
    parser.add_argument(
        "--busy-ok", action="store_true", help="Time even if the machine is busy (marked)"
    )
    args = parser.parse_args()
    if args.frames <= args.warmup or args.warmup < 0:
        parser.error("--frames must exceed --warmup, which must be nonnegative")
    if args.threads is not None:
        cv2.setNumThreads(args.threads)
    run = run_header(apply_settings(args.set))
    load = machine_load()
    print(f"Before the run: {json.dumps(load)}")
    if load["busy"] and not args.busy_ok:
        sys.exit(
            "Something else is using the graphics card or the processor: the times would "
            "say nothing. Free the machine, or pass --busy-ok to time it anyway (marked)."
        )
    video = args.video or BENCHMARK_CLIP
    if not video.exists():
        video = REPO / "data" / "raw" / "videos" / get_setting("video.default_video")
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
        following = frames[index + 1] if args.ahead and index + 1 < len(frames) else None
        result = pipeline.process(
            frame, index, options, next_frame=None if following is None else lambda: following
        )
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
        "run": run,
        "machine_before": load,
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
