#!/usr/bin/env python3
"""Score the jersey number readers on the labelled player crops.

The test set (data/processed/jersey_eval) holds player crops per track from four games.
labels.json gives the true number of each track where one is clearly visible, and lists
tracks where no number can be seen. Every reader runs through the app's own player ID
code, so the numbers reflect what the app does.

    python scripts/benchmark_jersey_readers.py                # all readers
    python scripts/benchmark_jersey_readers.py parseq yolo_digits
"""

import collections
import contextlib
import io
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import torch  # noqa: E402

from ultimate_analysis.processing import jersey_readers, player_id  # noqa: E402

EVAL_DIR = REPO / "data" / "processed" / "jersey_eval"
CROPS_PER_CALL = 4  # the app reads a few players per frame
EVERY_NTH_CROP = 2


def load_samples():
    """[(track key, true number or None, crop image)] for the labelled tracks."""
    labels = json.loads((EVAL_DIR / "labels.json").read_text())
    truth = {key: number for key, number in labels["numbers"].items()}
    truth.update({key: None for key in labels["no_number"]})

    by_track = collections.defaultdict(list)
    for entry in json.loads((EVAL_DIR / "manifest.json").read_text()):
        key = f"{entry['clip']}/{entry['track']}"
        if key in truth:
            by_track[key].append(entry)

    samples = []
    for key, entries in by_track.items():
        for entry in sorted(entries, key=lambda e: e["frame"])[::EVERY_NTH_CROP]:
            samples.append((key, truth[key], cv2.imread(str(EVAL_DIR / entry["file"]))))
    return samples


def read_numbers(method, crops):
    """Numbers the app reads from the crops with this reader (None = no number)."""
    player_id.set_player_id_method(method)
    with contextlib.redirect_stdout(io.StringIO()):
        results, _ = player_id._read_jersey_numbers(crops)
    return [
        (None if number == "Unknown" else number, (details or {}).get("confidence", 0.0))
        for number, details in results
    ]


def main() -> None:
    methods = sys.argv[1:] or list(jersey_readers.READER_LABELS)
    samples = load_samples()
    numbered = [s for s in samples if s[1] is not None]
    tracks = {key: number for key, number, _ in numbered}
    print(
        f"{len(tracks)} players with a number ({len(numbered)} crops), "
        f"{len(samples) - len(numbered)} crops of players without a visible number\n"
    )

    for method in methods:
        if (
            method != "easyocr"
            and jersey_readers.get_reader(method, player_id._get_text_detector) is None
        ):
            print(f"{method:12s} not available (it would fall back to EasyOCR)")
            continue

        read_numbers(method, [s[2] for s in samples[:8]])  # load and warm up
        reads, seconds = [], 0.0
        for start in range(0, len(samples), CROPS_PER_CALL):
            crops = [s[2] for s in samples[start : start + CROPS_PER_CALL]]
            torch.cuda.synchronize()
            started = time.perf_counter()
            reads.extend(read_numbers(method, crops))
            torch.cuda.synchronize()
            seconds += time.perf_counter() - started

        correct = wrong = false_reads = 0
        votes = collections.defaultdict(collections.Counter)
        for (key, number, _), (read, confidence) in zip(samples, reads):
            if number is None:
                false_reads += read is not None
            elif read is not None:
                correct += read == number
                wrong += read != number
                votes[key][read] += confidence

        right = sum(v.most_common(1)[0][0] == tracks[key] for key, v in votes.items())
        print(
            f"{method:12s} players: {right} right, {len(votes) - right} wrong, "
            f"{len(tracks) - len(votes)} unread | per crop: {correct / len(numbered):.1%} correct, "
            f"{wrong / len(numbered):.1%} wrong | reads on players without a number: "
            f"{false_reads / max(1, len(samples) - len(numbered)):.1%} | "
            f"{seconds / len(samples) * 1000:.1f} ms per crop"
        )


if __name__ == "__main__":
    main()
