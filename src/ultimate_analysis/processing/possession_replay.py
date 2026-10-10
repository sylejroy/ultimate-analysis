"""What the possession tracker is told, written down and played back.

Who holds the disc is worked out from the discs detected, the tracks and the camera's
motion. Detecting and tracking take nearly all the time of a run over a video, and a
change to the possession logic or its settings changes neither. So a run is recorded
once (`RecordingPossessionTracker` in place of the tracker, told with `frame_done`
when a frame is through), and the logic is tried on the recording afterwards: `replay`
gives the holder, since when, and the team in possession after every frame, in a moment.

A recording holds only what the tracker is given: it says nothing new once the
detections or the tracks themselves change.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .possession import PossessionTracker

# What the tracker says after a frame: the frame, the holder, since when, the team
Said = Tuple[int, Optional[int], int, Optional[int]]


class RecordingPossessionTracker(PossessionTracker):
    """A possession tracker that writes down all it is told, in `calls`."""

    def __init__(self):
        self.calls: List[tuple] = []
        super().__init__()

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "frame_rate":
            self.calls.append(("rate", float(value)))
        super().__setattr__(name, value)

    def frame_done(self, frame_index: int) -> None:
        """A frame is through: a replay says what holds here."""
        self.calls.append(("frame", frame_index))

    def reset(self) -> None:
        self.calls.append(("reset",))
        super().reset()

    def move(self, camera_motion: np.ndarray) -> None:
        self.calls.append(("move", np.array(camera_motion, dtype=np.float64)))
        super().move(camera_motion)

    def rename(self, player_id: int, new_player_id: int) -> None:
        self.calls.append(("rename", player_id, new_player_id))
        super().rename(player_id, new_player_id)

    def update(
        self,
        detections: List[Dict[str, Any]],
        tracks: List[Any],
        frame_index: Optional[int] = None,
        holder_scores: Optional[Dict[int, float]] = None,
    ) -> Optional[int]:
        discs = [
            {
                "class_name": "disc",
                "confidence": float(detection.get("confidence", 0.0)),
                "bbox": [float(value) for value in detection["bbox"]],
            }
            for detection in detections
            if detection.get("class_name") == "disc"
        ]
        seen = [
            (
                track.track_id,
                tuple(float(value) for value in track.to_ltrb()),
                getattr(track, "team", None),
                getattr(track, "class_name", None),
            )
            for track in tracks
        ]
        # What a network said of the players at the disc is written down only where
        # it said anything: recordings made without one stay as they were
        scores = (dict(holder_scores),) if holder_scores else ()
        self.calls.append(("update", discs, seen, frame_index, *scores))
        return super().update(detections, tracks, frame_index, holder_scores)


def _track(track_id: int, box: tuple, team: Optional[int], class_name: Optional[str]) -> Any:
    return SimpleNamespace(track_id=track_id, class_name=class_name, team=team, to_ltrb=lambda: box)


def replay(calls: List[tuple]) -> List[Said]:
    """Tell a new possession tracker what was recorded; what it says after every frame
    that was marked as done."""
    tracker = PossessionTracker()
    said: List[Said] = []
    for kind, *given in calls:
        if kind == "reset":
            tracker.reset()
        elif kind == "rate":
            tracker.frame_rate = given[0]
        elif kind == "move":
            tracker.move(given[0])
        elif kind == "rename":
            tracker.rename(*given)
        elif kind == "update":
            discs, seen, frame_index, *scores = given
            tracker.update(discs, [_track(*track) for track in seen], frame_index, *scores)
        elif kind == "frame":
            said.append((given[0], tracker.holder_id, tracker.since, tracker.team))
    return said
