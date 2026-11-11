#!/usr/bin/env python3
"""Show how a run of scripts/train_spots_gpu.py is getting on, in a console of its own:
how many iterations are done and how long is left, how far the iteration under way is,
how the team plays against itself, and each check against the scripted teams as a bar.

    python scripts/watch_spots.py                      # The newest run
    python scripts/watch_spots.py data/models/movement/<run>
    python scripts/watch_spots.py <run> --log <file>   # A run whose output went elsewhere

It reads the run's log.txt (what the training prints) every two seconds. Ctrl+C ends it;
the training goes on.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[1]
BAR = 40
ITERATION = re.compile(
    r"^iteration (\d+): (\d+) points ended, (\d+) scored; completed (\d+)/(\d+); "
    r"(\d+) choices; (\d+) s playing, (\d+) s learning"
)
CHECK = re.compile(
    r"^\s+against (.+) \(iteration (\d+)\): scored (\d+), conceded \d+ of (\d+); "
    r"completed (\d+)/(\d+)"
)
FAILED = re.compile(r"Traceback|Error|out of memory|exited with")
RECENT = 10  # Iterations the time each takes is averaged over


def bar(share: float, width: int = BAR) -> str:
    done = int(round(min(1.0, max(0.0, share)) * width))
    return "[" + "#" * done + "." * (width - done) + "]"


def lasting(seconds: float) -> str:
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def newest_run() -> Optional[Path]:
    runs = sorted((REPO / "data/models/movement").glob("*_spots_gpu"))
    return runs[-1] if runs else None


def screen(run: Path, log: Path, total: Optional[int]) -> List[str]:
    """What to show of a run now."""
    lines = [f"Run {run.name}", ""]
    if not log.exists():
        return lines + [f"Waiting for {log} ..."]
    text = log.read_text(encoding="utf-8", errors="replace").splitlines()
    iterations = [m for m in map(ITERATION.match, text) if m]
    checks = [m for m in map(CHECK.match, text) if m]
    if total is None and (run / "settings.json").exists():
        total = json.loads((run / "settings.json").read_text()).get("iterations")
    waited = time.time() - log.stat().st_mtime
    if not iterations:
        return lines + [f"The first iteration is under way ({waited:.0f} s in)."]
    done = int(iterations[-1].group(1))
    took = [int(m.group(7)) + int(m.group(8)) for m in iterations[-RECENT:]]
    each = sum(took) / len(took)
    if total:
        left = lasting((total - done) * each)
        lines.append(f"Iterations  {bar(done / total)} {done} of {total}, about {left} left")
    else:
        lines.append(f"Iterations  {done} done")
    if total and done >= total:
        lines.append("Finished.")
    elif waited > 5 * each + 120:
        lines.append(f"Nothing written for {lasting(waited)}: the run has stopped, or hangs.")
    else:
        # The iteration under way: playing first, then learning (as the last one went)
        playing = int(iterations[-1].group(7))
        doing = "playing" if waited < playing else "learning"
        lines.append(
            f"Iteration {done + 1:<4d} {bar(waited / each)} {doing}, {waited:.0f} of about "
            f"{each:.0f} s"
        )
    last = iterations[-1]
    ended, scored, caught, thrown = (int(last.group(k)) for k in (2, 3, 4, 5))
    lines += [
        "",
        f"Against itself, iteration {done}: {ended} points ended, {scored} with a score; "
        f"{caught / max(1, thrown):.0%} of {thrown} passes completed",
    ]
    for line in text:
        if FAILED.search(line):
            lines += ["", "Something went wrong:", "  " + line.strip()]
            break
    names = list(dict.fromkeys(m.group(1) for m in checks))
    for name in names:
        mine = [m for m in checks if m.group(1) == name]
        best = max(int(m.group(3)) / int(m.group(4)) for m in mine)
        lines += ["", f"Against {name}: points scored (the best so far marked)"]
        for m in mine[-12:]:
            share = int(m.group(3)) / int(m.group(4))
            passes = int(m.group(5)) / max(1, int(m.group(6)))
            mark = " *" if share == best else ""
            lines.append(
                f"  iteration {int(m.group(2)):4d}  {bar(share, 30)} {share:4.0%}  "
                f"({m.group(3)} of {m.group(4)}; passes {passes:.0%}){mark}"
            )
    if not checks:
        lines += ["", "No check against the scripted teams yet."]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", help="The run's folder (the newest)")
    parser.add_argument("--log", type=Path, help="What the training printed (<run>/log.txt)")
    parser.add_argument("--iterations", type=int, help="How many the run does (its settings)")
    parser.add_argument("--once", action="store_true", help="Show it once and end")
    args = parser.parse_args()
    run = args.run or newest_run()
    if run is None:
        sys.exit("No run under data/models/movement")
    log = args.log or run / "log.txt"
    os.system("")  # Lets a Windows console take the codes that clear it
    try:
        while True:
            shown = screen(run, log, args.iterations)
            if args.once:
                print("\n".join(shown))
                return
            print("\x1b[H\x1b[J" + "\n".join(shown) + "\n\nCtrl+C ends this; the training goes on.")
            time.sleep(2.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
