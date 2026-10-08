"""What went wrong in a stage, kept so that it can be shown.

A stage that fails returns what it can (no detections, tracks without history, the slower
model) so that the app keeps running. Without a word about it that looks like a normal
result: no detections reads as "nobody there". The stages report their failures here, and
the pipeline shows them on the frame.
"""

from typing import Dict, List

# Stage -> what happened, for the frame being analysed
_problems: Dict[str, str] = {}
# Stage -> what holds until the app is restarted (a model that runs without its engine)
_standing: Dict[str, str] = {}


def report(stage: str, problem: str, standing: bool = False) -> None:
    """Note that a stage failed, for this frame or (standing) for good."""
    (_standing if standing else _problems)[stage] = problem


def start_frame() -> None:
    """Forget the problems of the frame before."""
    _problems.clear()


def failed(stage: str) -> bool:
    """Whether a stage failed on the frame being analysed."""
    return stage in _problems


def problems() -> List[str]:
    """The problems to show, one line each."""
    return [f"{stage}: {problem}" for stage, problem in {**_standing, **_problems}.items()]
