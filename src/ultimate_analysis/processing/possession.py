"""Possession: which tracked player holds the disc.

The disc belongs to the player whose bounding box it is in. A disc in flight passes other
players on its way, so the holder only changes after the disc has been seen at the same new
place (another player, or no player at all) for several frames in a row. Frames without a
detected disc say nothing and leave the holder as it is.
"""

from typing import Any, Dict, List, Optional, Sequence

from ..config.settings import get_setting


def player_at_disc(
    disc_box: Sequence[float], tracks: List[Any], box_margin: float
) -> Optional[int]:
    """Track ID of the player whose box, widened by a margin, contains the disc centre.

    A disc held at arm's length lies just outside the player's box, hence the margin (a
    fraction of the box width and height). Of several players, the one whose centre is
    nearest to the disc relative to their size is chosen.
    """
    disc_x = (disc_box[0] + disc_box[2]) / 2
    disc_y = (disc_box[1] + disc_box[3]) / 2

    nearest_id, nearest_distance = None, float("inf")
    for track in tracks:
        if getattr(track, "class_name", None) != "player":
            continue
        x1, y1, x2, y2 = track.to_ltrb()
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        inside = (
            x1 - box_margin * width <= disc_x <= x2 + box_margin * width
            and y1 - box_margin * height <= disc_y <= y2 + box_margin * height
        )
        if not inside:
            continue
        distance = ((disc_x - (x1 + x2) / 2) ** 2 + (disc_y - (y1 + y2) / 2) ** 2) ** 0.5 / height
        if distance < nearest_distance:
            nearest_id, nearest_distance = track.track_id, distance
    return nearest_id


class PossessionTracker:
    """Follows the disc holder over the frames of a video."""

    def __init__(self):
        self.holder_id: Optional[int] = None  # None: nobody, e.g. the disc is in flight
        self._candidate_id: Optional[int] = None
        self._candidate_frames = 0

    def reset(self) -> None:
        """Forget the holder (new video, seek)."""
        self.holder_id = None
        self._candidate_id = None
        self._candidate_frames = 0

    def update(self, detections: List[Dict[str, Any]], tracks: List[Any]) -> Optional[int]:
        """Take a frame's detections and tracks into account; returns the holder's track ID."""
        discs = [detection for detection in detections if detection.get("class_name") == "disc"]
        if not discs:
            return self.holder_id

        disc = max(discs, key=lambda detection: detection.get("confidence", 0.0))
        at_disc = player_at_disc(
            disc["bbox"], tracks, float(get_setting("models.possession.box_margin", 0.15))
        )

        if at_disc == self.holder_id:
            self._candidate_frames = 0
            return self.holder_id

        # A different place than the holder: count how long the disc stays there
        if at_disc != self._candidate_id or self._candidate_frames == 0:
            self._candidate_id = at_disc
            self._candidate_frames = 0
        self._candidate_frames += 1

        if self._candidate_frames >= int(get_setting("models.possession.confirm_frames", 10)):
            self.holder_id = at_disc
            self._candidate_frames = 0
        return self.holder_id
