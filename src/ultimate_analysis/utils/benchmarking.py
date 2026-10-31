"""What the benchmark scripts share: where a mean lies with the games drawn again, the
settings changed with --set, the lines that say which run a result is of, and the
frames of a video around a labelled frame."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config.settings import get_config

REPO = Path(__file__).resolve().parents[3]
ROUNDS = 2000  # How often the groups are drawn again


def drawn_again(
    values: Sequence[float], groups: Optional[Sequence] = None, rounds: int = ROUNDS
) -> Tuple[float, float, float]:
    """The mean of some values (a share, for values of 0 and 1), and where it lies with
    their groups drawn again (5th and 95th percentile).

    `groups` says which group each value belongs to: a game, a video, a stretch, an
    image. The values of one group are not independent of each other, so whole groups
    are drawn, with replacement, as many as there are. Without groups every value is a
    group of its own. The draw is the same every time, so that two runs differ only by
    their values. All three are nan without values.
    """
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    # Each group a number, whatever it is called by (a name, a pair)
    numbers: dict = {}
    labels = np.array(
        [numbers.setdefault(group, len(numbers)) for group in groups]
        if groups is not None
        else range(len(values))
    )
    sums = np.bincount(labels, weights=values)
    counts = np.bincount(labels)
    drawn = np.random.default_rng(0).integers(len(counts), size=(rounds, len(counts)))
    means = sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 5)),
        float(np.percentile(means, 95)),
    )


def with_spread(
    spread: Tuple[float, float, float], what: str = "games", form: str = ".0%", unit: str = ""
) -> str:
    """ "80% (76% to 83% with the games drawn again)" from what `drawn_again` gave. Without
    `what`, only "80% (76% to 83%)": for a table, with a line under it that says so."""
    mean, low, high = spread
    inside = f"{low:{form}} to {high:{form}}{unit}"
    if what:
        inside += f" with the {what} drawn again"
    return f"{mean:{form}}{unit} ({inside})"


def apply_settings(overrides: Sequence[str]) -> List[str]:
    """Change the settings as "key=value" items of --set say, the value as JSON, and give
    them back as written down in a summary. A key that is not in the settings is refused:
    a misspelt one would change nothing and the run would look like a result."""
    config = get_config()
    changed = []
    for item in overrides:
        key, _, text = item.partition("=")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(
                f"--set {item}: the value is not JSON (a string needs quotes: {key}='\"text\"')"
            ) from None
        node = config
        *path, last = key.split(".")
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict) or last not in node:
            raise ValueError(f"--set {item}: there is no setting {key} (see configs/default.yaml)")
        node[last] = value
        changed.append(f"{key}={json.dumps(value)}")
    return changed


def git_commit() -> str:
    """The commit the code is at, with "+changes" if the working tree differs from it."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return (commit or "unknown") + ("+changes" if changed else "")
    except OSError:
        return "unknown"


def run_header(settings_changed: Sequence[str] = ()) -> List[str]:
    """The lines that say which run a result is of: the commit (and whether the code was
    changed since), the day, the settings changed with --set, and the command line."""
    command = " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])
    return [
        f"Commit {git_commit()}, {datetime.now():%Y-%m-%d %H:%M}",
        "Settings changed: " + (", ".join(settings_changed) or "none"),
        f"Command: {command}",
    ]


def frame_rate_of(video: Path) -> float:
    """Frames a second of a video file."""
    capture = cv2.VideoCapture(str(video))
    try:
        return capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()


def frames_of(
    video: Path, first: int, last: int, step: int = 1
) -> Iterator[Tuple[int, np.ndarray]]:
    """(frame number, frame) of a video from `first` to `last`, every `step` frames from
    the first, as the app analyses a video in "Real time". Stops where the video ends."""
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for number in range(first, last + 1):
            if (number - first) % step:
                # Passed over without being decoded
                if not capture.grab():
                    break
                continue
            ok, frame = capture.read()
            if not ok:
                break
            yield number, frame
    finally:
        capture.release()


def frames_around(
    video: Path, anchor: int, before_seconds: float, after_seconds: float, step: int = 1
) -> Iterator[Tuple[int, np.ndarray]]:
    """(frame number, frame) from so many seconds before a labelled frame to so many
    after it, every `step` frames with the labelled frame among them."""
    rate = frame_rate_of(video)
    before = int(before_seconds * rate) // step * step
    start = max(anchor % step, anchor - before)
    return frames_of(video, start, anchor + int(after_seconds * rate), step)
