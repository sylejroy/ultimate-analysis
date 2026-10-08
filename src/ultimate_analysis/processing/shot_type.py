"""Whether the footage is the wide view from the drone or something else.

Edited games cut away from the drone: to a camera at ground level, a title card, a black
frame. Tracking, possession and the field make no sense there, and what they learned
before the cut does not hold after it.

The player model tells the two apart by itself. It was trained on drone footage and finds
a dozen small players in it; in a close-up it finds nobody, or one or two. Of 600 random
frames of five edited games, the 526 with six or more players were all drone footage;
of the 74 with fewer, some 60 were close-ups, title cards or black, and the rest drone
footage with hardly anybody in it (a huddle, an empty field), where nothing is lost.
"""

from ..config.settings import get_setting


class ShotWatcher:
    """Follows whether the frames are wide drone footage, without flickering."""

    def __init__(self) -> None:
        self.wide = True
        self._disagreeing = 0  # Frames in a row that looked like the other kind

    def reset(self) -> None:
        """Start again (another video, a seek): the next frame decides by itself."""
        self.wide = True
        self._disagreeing = self._confirm_frames()

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

    def update(self, player_count: int) -> bool:
        """Take in how many players the detector found in a frame; True for drone footage.

        The kind only changes once that many frames in a row say so: players are missed
        in single frames, and one person walking through a close-up is not a team.
        """
        looks_wide = player_count >= int(get_setting("models.shot_type.min_players", 6))
        if looks_wide == self.wide:
            self._disagreeing = 0
            return self.wide
        self._disagreeing += 1
        if self._disagreeing >= self._confirm_frames():
            self.wide = looks_wide
            self._disagreeing = 0
        return self.wide
