#!/usr/bin/env python3
"""Compare live OCR scheduling on ordered, labelled crop observations.

Uses the real player-ID pipeline, temporal votes, finalization, and optional crop selection.
Each manifest observation is one replay step (the crops were sampled every three video
frames). Stable labelled identities are supplied: this does not score tracking or occlusion.
Images are packed side by side so original crop dimensions and OCR coordinates are retained.

    python scripts/benchmark_player_id_scheduling.py
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ultimate_analysis.config.settings import get_config  # noqa: E402
from ultimate_analysis.processing import player_id  # noqa: E402
from ultimate_analysis.processing.jersey_crops import JerseyCropSelector  # noqa: E402
from ultimate_analysis.processing.jersey_tracker import reset_jersey_tracker  # noqa: E402

EVAL_DIR = REPO / "data" / "processed" / "jersey_eval"


def load_replay():
    labels = json.loads((EVAL_DIR / "labels.json").read_text())
    truth = {key: str(value) for key, value in labels["numbers"].items()}
    truth.update({key: None for key in labels["no_number"]})
    ids = {key: index + 1 for index, key in enumerate(sorted(truth))}
    by_clip = collections.defaultdict(lambda: collections.defaultdict(list))
    for entry in json.loads((EVAL_DIR / "manifest.json").read_text()):
        key = f"{entry['clip']}/{entry['track']}"
        if key in truth:
            by_clip[entry["clip"]][entry["frame"]].append((ids[key], entry["file"]))
    clips = []
    for clip, observations in sorted(by_clip.items()):
        frames = []
        for _, entries in sorted(observations.items()):
            crops = [(track_id, cv2.imread(str(EVAL_DIR / file))) for track_id, file in entries]
            if any(crop is None for _, crop in crops):
                raise ValueError(f"Missing crop in {clip}")
            height = max(crop.shape[0] for _, crop in crops)
            width = sum(crop.shape[1] + 1 for _, crop in crops)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            tracks, x = [], 0
            for track_id, crop in crops:
                h, w = crop.shape[:2]
                frame[:h, x : x + w] = crop
                tracks.append(
                    SimpleNamespace(
                        track_id=track_id,
                        class_name="player",
                        bbox=[x, 0, x + w, h],
                        time_since_update=0,
                    )
                )
                x += w + 1
            frames.append((frame, tracks))
        clips.append(frames)
    return clips, {ids[key]: number for key, number in truth.items()}


def replay(clips, truth, enabled):
    settings = get_config()["models"]["player_id"]
    previous = settings["crop_selection"]["enabled"]
    settings["crop_selection"]["enabled"] = enabled
    live, ever_wrong = {}, set()
    reads = 0
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for frames in clips:
            reset_jersey_tracker()
            selector, finalized = JerseyCropSelector(), set()
            for frame_index, (frame, tracks) in enumerate(frames):
                results, timing, finalized = player_id.run_player_id_on_tracks(
                    frame, tracks, frame_index, finalized, selector
                )
                reads += timing.get("tracks_ocr", 0)
                live.update({track_id: number for track_id, (number, _) in results.items()})
                for track in tracks:
                    number = live.get(track.track_id, "Unknown")
                    if number != "Unknown" and number != truth[track.track_id]:
                        ever_wrong.add(track.track_id)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
    finally:
        settings["crop_selection"]["enabled"] = previous
    numbered = {track_id: number for track_id, number in truth.items() if number is not None}
    right = sum(live.get(track_id) == number for track_id, number in numbered.items())
    wrong = sum(
        live.get(track_id, "Unknown") not in ("Unknown", number)
        for track_id, number in numbered.items()
    )
    false = sum(
        live.get(track_id, "Unknown") != "Unknown"
        for track_id, number in truth.items()
        if number is None
    )
    return {
        "right": right,
        "wrong": wrong,
        "unread": len(numbered) - right - wrong,
        "false_numbers": false,
        "ever_wrong": len(ever_wrong),
        "ocr_crops": reads,
        "seconds": round(seconds, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="parseq")
    args = parser.parse_args()
    clips, truth = load_replay()
    player_id.set_player_id_method(args.method)
    player_id.initialize_player_id_system()
    if player_id._easyocr_reader is None or (
        args.method != "easyocr" and player_id._get_active_reader() is None
    ):
        raise RuntimeError(
            f"Requested reader {args.method} is unavailable; refusing a fallback benchmark"
        )
    first_frame, tracks = clips[0][0]
    crops = [first_frame[y1:y2, x1:x2] for track in tracks for x1, y1, x2, y2 in [track.bbox]]
    player_id._read_jersey_numbers(crops)  # Load and warm up before timing either mode.
    print(
        f"{len(clips)} clips, {sum(map(len, clips))} replay steps, {len(truth)} labelled identities"
    )
    for enabled in (False, True):
        print(
            ("selected" if enabled else "fixed") + ": " + json.dumps(replay(clips, truth, enabled))
        )


if __name__ == "__main__":
    main()
