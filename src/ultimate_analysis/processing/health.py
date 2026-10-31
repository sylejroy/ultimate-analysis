"""What went wrong in a stage, kept so that it can be shown.

A stage that fails returns what it can (no detections, tracks without history, the slower
model) so that the app keeps running. Without a word about it that looks like a normal
result: no detections reads as "nobody there". The stages report their failures here, and
the pipeline shows them on the frame.
"""

import threading
from contextlib import contextmanager
from typing import Collection, Dict, Iterator, List, Optional, Tuple

# Stage -> what happened, for the frame being analysed
_problems: Dict[str, str] = {}
# Stage -> what holds until the app is restarted (a model that runs without its engine)
_standing: Dict[str, str] = {}
# What `captured` is collecting: stage -> (problem, standing), and which stages
Caught = Dict[str, Tuple[str, bool]]
_captures: List[Tuple[Optional[Collection[str]], Caught]] = []
# Reports come from the frame's thread, the threads beside it and the other tabs
_captures_lock = threading.Lock()


def report(stage: str, problem: str, standing: bool = False) -> None:
    """Note that a stage failed, for this frame or (standing) for good."""
    with _captures_lock:
        (_standing if standing else _problems)[stage] = problem
        for only, caught in _captures:
            if only is None or stage in only:
                caught[stage] = (problem, standing)


@contextmanager
def captured(only: Optional[Collection[str]] = None) -> Iterator[Caught]:
    """Collect what is reported meanwhile, from any thread, to `repeat` later.

    Work done beside the frames, or in another process, reports on a frame that is not
    the one being analysed: `start_frame` would wipe the report before that frame's
    turn, and a report made in another process never arrives at all. `only`: collect
    the reports of these stages alone.
    """
    caught: Caught = {}
    with _captures_lock:
        _captures.append((only, caught))
    try:
        yield caught
    finally:
        with _captures_lock:
            _captures[:] = [entry for entry in _captures if entry[1] is not caught]


def repeat(caught: Caught) -> None:
    """Report again what `captured` collected, on the frame it is about."""
    for stage, (problem, standing) in caught.items():
        report(stage, problem, standing)


def clear(stage: str) -> None:
    """A stage that failed for good works again (a model that loaded on a later try)."""
    with _captures_lock:
        _standing.pop(stage, None)
        _problems.pop(stage, None)


def start_frame() -> None:
    """Forget the problems of the frame before."""
    with _captures_lock:
        _problems.clear()


def failed(stage: str) -> bool:
    """Whether a stage failed on the frame being analysed."""
    return stage in _problems


def problems() -> List[str]:
    """The problems to show, one line each."""
    with _captures_lock:
        return [f"{stage}: {problem}" for stage, problem in {**_standing, **_problems}.items()]
