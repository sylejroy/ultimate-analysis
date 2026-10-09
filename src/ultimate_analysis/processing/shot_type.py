"""Whether the footage is the wide view from the drone or something else.

Edited games cut away from the drone: to a camera at ground level, a title card, a black
frame. Tracking, possession and the field make no sense there, and what they learned
before the cut does not hold after it.

The player model tells the two apart by itself. It was trained on drone footage and finds
a dozen small players in it; in a close-up it finds nobody, or one or two. Of 600 random
frames of five edited games, the 526 with six or more players were all drone footage;
of the 74 with fewer, some 60 were close-ups, title cards or black, and the rest drone
footage with hardly anybody in it (a huddle, an empty field), where nothing is lost.

Few players alone is not a cut away, though. After a long throw the drone swings to the
other end of the field: for a second it shows three or four players, blurred, and then
everybody again. A cut breaks the picture: from one frame to the next the camera's
motion cannot be told. So the drone footage is only held to have stopped once the
picture broke, or when there has been hardly anybody for much longer than a swing.
"""

import numpy as np

from ..config.settings import get_setting

# While the picture goes on unbroken, drone footage with few players in it is held to be
# drone footage this many times longer than otherwise (12 x 15 frames: 3 s at 60 a second)
UNBROKEN_PATIENCE = 12


class ShotWatcher:
    """Follows whether the frames are wide drone footage, without flickering."""

    def __init__(self) -> None:
        self.wide = True
        self._disagreeing = 0  # Frames in a row that looked like the other kind
        self._broken = True  # Whether the picture broke since it last showed enough players

    def reset(self) -> None:
        """Start again (another video, a seek): the next frame decides by itself."""
        self.wide = True
        self._disagreeing = self._confirm_frames()
        self._broken = True

    def pause(self) -> None:
        """Hold that this is no drone footage, after whatever else was reset.

        The way back needs its frames in a row like any change: one frame with enough
        players in it is not the drone yet.
        """
        self.wide = False
        self._disagreeing = 0

    @staticmethod
    def _confirm_frames() -> int:
        return max(1, int(get_setting("models.shot_type.confirm_frames", 15)))

    def update(self, player_count: int, continuous: bool = False) -> bool:
        """Take in how many players the detector found in a frame; True for drone footage.

        The kind only changes once that many frames in a row say so: players are missed
        in single frames, and one person walking through a close-up is not a team.

        Args:
            player_count: Players the detector found in the frame
            continuous: Whether the picture goes on from the frame before (the camera's
                motion between the two could be told). Drone footage that goes on
                unbroken stays drone footage with few players in it, for a while.
        """
        looks_wide = player_count >= int(get_setting("models.shot_type.min_players", 6))
        if looks_wide:
            self._broken = False
        elif not continuous:
            self._broken = True
        if looks_wide == self.wide:
            self._disagreeing = 0
            return self.wide
        self._disagreeing += 1
        needed = self._confirm_frames()
        if self.wide and not self._broken:
            needed *= UNBROKEN_PATIENCE  # A swing of the camera, as far as can be told
        if self._disagreeing >= needed:
            self.wide = looks_wide
            self._disagreeing = 0
        return self.wide


# A cut within drone footage: from one frame to the next the camera's motion cannot be
# told and the players are elsewhere. With at least this many players before, fewer than
# this share of them have a detection where they stood (boxes overlapping this much).
CUT_MIN_PLAYERS = 6
CUT_MAX_STAYING = 0.3
CUT_SAME_PLACE_IOU = 0.3


def players_are_elsewhere(before: np.ndarray, now: np.ndarray) -> bool:
    """Whether the players of one frame are not where those of the frame before stood.

    Between two frames of one shot nearly every player's box still overlaps their own.
    After a cut to another view (a replay, the other end of the field) hardly any does,
    and what was followed (tracks, the holder, the field) belongs to the view before.

    Args:
        before: Player boxes (n, 4) as x1, y1, x2, y2 of the frame before
        now: Player boxes of this frame
    """
    before, now = np.asarray(before, dtype=np.float64), np.asarray(now, dtype=np.float64)
    if len(before) < CUT_MIN_PLAYERS:
        return False
    if len(now) == 0:
        return True
    width = np.minimum(before[:, None, 2], now[None, :, 2]) - np.maximum(
        before[:, None, 0], now[None, :, 0]
    )
    height = np.minimum(before[:, None, 3], now[None, :, 3]) - np.maximum(
        before[:, None, 1], now[None, :, 1]
    )
    shared = np.clip(width, 0, None) * np.clip(height, 0, None)
    area_before = (before[:, 2] - before[:, 0]) * (before[:, 3] - before[:, 1])
    area_now = (now[:, 2] - now[:, 0]) * (now[:, 3] - now[:, 1])
    overlap = shared / (area_before[:, None] + area_now[None, :] - shared)
    staying = (overlap.max(axis=1) >= CUT_SAME_PLACE_IOU).mean()
    return bool(staying < CUT_MAX_STAYING)
