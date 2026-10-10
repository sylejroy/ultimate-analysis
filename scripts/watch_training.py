#!/usr/bin/env python3
"""Show how a training run of the small networks is getting on, as it runs.

scripts/train_holder.py and scripts/train_reid.py print a line per epoch. Started
with their output written to a file, they say nothing on the screen; this reads such
files as they grow and prints a line per epoch with a bar for the measure the best
epoch is chosen by, the best so far marked, and how long is left.

The benchmarks that run over the videos (scripts/benchmark_possession.py and the like)
say every so often how many of their frames are done; those lines are shown as a bar of
how far along the run is, with the time left, and the log lines between them are left
out.

    python scripts/watch_training.py run.log --epochs 100
    python scripts/watch_training.py first.log second.log     # one after the other

Several files are followed at once: whichever is written to is shown, under its name.
A file that is not there yet is waited for. Ctrl+C ends it; the training goes on.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional

# The measure shown as a bar, the first of these an epoch's line has
MEASURES = (
    "holder_picked",
    "across_points_teammates",
    "same_point_teammates",
    "average_precision",
    "accuracy",
)
BAR = 30
# How a benchmark says how far along it is: "40 of 563 labelled frames: ..."
DONE = re.compile(r"^(\d+) of (\d+) ([a-z ]*frames)\b(.*)$")
# A line of the app's log, which a benchmark writes hundreds of
LOGGED = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}|^\[\d{2}/\d{2}/\d{4}|^Loading ")


def epoch_of(line: str) -> Optional[dict]:
    """What a line of a training run says of an epoch, or None if it is none."""
    line = line.strip()
    if not line.startswith("{") or '"epoch"' not in line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def shown(epoch: dict, best: float, since: tuple, epochs: int) -> str:
    """An epoch as a line: its number, the loss, the measure as a bar, the time left.
    `since` is the first epoch seen as it was written, and when: the time left is told
    from the epochs watched, not from those that were in the file already."""
    measure = next((name for name in MEASURES if name in epoch), None)
    value = float(epoch.get(measure, 0.0)) if measure else 0.0
    filled = round(BAR * max(0.0, min(1.0, value)))
    number = int(epoch["epoch"])
    left = ""
    if epochs and number < epochs and number > since[0]:
        seconds = (time.time() - since[1]) / (number - since[0]) * (epochs - number)
        left = (
            f"  about {seconds / 60:.0f} min left" if seconds >= 90 else f"  {seconds:.0f} s left"
        )
    of = f"/{epochs}" if epochs else ""
    start = f"epoch {number:>3}{of}  loss {float(epoch.get('loss', 0.0)):6.3f}  "
    if measure is None:
        # An epoch that was not scored: some runs score every few epochs only
        return f"{start}{left}"
    return (
        f"{start}{measure} {value:6.1%} |{'#' * filled}{'.' * (BAR - filled)}|"
        f"{'  best so far' if value >= best else ''}{left}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Files a run's output is written to")
    parser.add_argument("--epochs", type=int, default=0, help="How many the runs are to make")
    args = parser.parse_args()

    read: Dict[Path, int] = {log: 0 for log in args.logs}  # Characters read of each
    best: Dict[Path, float] = {log: 0.0 for log in args.logs}
    since: Dict[Path, tuple] = {}  # The first epoch seen as it was written, and when
    first_look = True  # What is in the files already was not watched being written
    last: Optional[Path] = None
    begun = set()  # The files that are there
    print(
        f"{time.strftime('%H:%M')}  Waiting for {', '.join(log.name for log in args.logs)}: "
        "nothing shows here until the run begins to write",
        flush=True,
    )
    while True:
        for log in args.logs:
            if not log.exists():
                continue
            if log not in begun:
                # Said at once: a benchmark's first line of progress takes minutes
                begun.add(log)
                print(f"\n=== {log.name}: started at {time.strftime('%H:%M')}", flush=True)
                last = log
            text = log.read_text(encoding="utf-8", errors="replace")
            new, read[log] = text[read[log] :], len(text)
            # A line still being written is read the next time round
            whole, _, rest = new.rpartition("\n")
            read[log] -= len(rest)
            for line in whole.splitlines():
                epoch = epoch_of(line)
                words = line.strip()
                if not words or "Warning" in words or "warn" in words:
                    continue
                if last != log:
                    print(f"\n=== {log.name}", flush=True)
                    last = log
                done = DONE.match(words)
                if done is not None:
                    number, of = int(done.group(1)), int(done.group(2))
                    if not first_look:
                        since.setdefault(log, (number, time.time()))
                    first, when = since.get(log, (10**9, 0.0))
                    left = ""
                    if first < number < of:
                        seconds = (time.time() - when) / (number - first) * (of - number)
                        left = f"  about {seconds / 60:.0f} min left"
                    filled = round(BAR * number / max(of, 1))
                    print(
                        f"{number:>4} of {of} {done.group(3)} |{'#' * filled}"
                        f"{'.' * (BAR - filled)}| {number / max(of, 1):4.0%}{left}"
                        f"  {done.group(4).strip(': ')[:70]}",
                        flush=True,
                    )
                    continue
                if epoch is None:
                    if not LOGGED.match(words):
                        print(words[:160], flush=True)
                    continue
                if not first_look:
                    since.setdefault(log, (int(epoch["epoch"]), time.time()))
                measure = next((name for name in MEASURES if name in epoch), None)
                value = float(epoch.get(measure, 0.0)) if measure else 0.0
                watched = since.get(log, (10**9, 0.0))
                print(shown(epoch, best[log], watched, args.epochs), flush=True)
                best[log] = max(best[log], value)
        first_look = False
        time.sleep(1.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
