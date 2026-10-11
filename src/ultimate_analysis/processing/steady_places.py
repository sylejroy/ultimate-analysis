"""Where the players stand on the field, without the jumping about.

A player's place is where the middle of the lower edge of their box lies on the field.
That edge goes up and down by a pixel or two with every stride and every frame's
detection, and far from the camera a pixel is half a yard along the field: there a
player standing still is put a yard here and a yard there, more from one frame to the
next than a sprint would take them.

Here each player's place is followed over the frames. A player moves on as they were
moving, and each frame's place counts for as much as it can be trusted there: the
pixels of doubt about the box's edge are turned into yards at that spot of the field,
along it and across it. Near the camera, where a pixel is inches, the place follows
the box at once; at the far end it is held steady along the field and still follows
at once across it.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# How unsure the foot of a box is in the picture, in pixels: sideways, and up and down
# (the stride). Measured on two games: 0.5 to 2 px sideways and 1 to 2.6 up and down
# about a player's own half-second mean; a larger number here makes the places steadier
# and slower to follow.
FOOT_DOUBT_PIXELS = (1.5, 3.0)
# How hard a player may speed up or turn, in field units a second per second: what the
# place is allowed to do that it was not doing already
MAX_PUSH = 9.0
# A place further than this from where the player was expected, in field units and in
# times the doubt there, is somebody else under the same number, or the field found
# anew: the following starts again from it
JUMP_UNITS = 4.0
JUMP_DOUBTS = 6.0
# A player not seen for this long is followed anew when they are seen again (seconds)
FORGET_SECONDS = 1.0
# How unsure a new player's speed is, in field units a second, squared: standing or
# sprinting
NEW_SPEED_DOUBT = 36.0


class SteadyPlaces:
    """Follows the place of every player on the field (a filter per player that takes
    the place and the speed along as they were, and each new place by its doubt)."""

    def __init__(self) -> None:
        # Player -> [time last seen, state (x, y, speed x, speed y), its doubt (4x4)]
        self._players: Dict[int, list] = {}

    def reset(self) -> None:
        """Forget everybody (a cut, another video, the field lost)."""
        self._players.clear()

    def rename(self, was: int, now: int) -> None:
        """A player is known under another number from now on."""
        if was in self._players:
            self._players[now] = self._players.pop(was)

    def update(
        self,
        seconds: float,
        feet: Dict[int, Sequence[float]],
        image_to_field: Optional[np.ndarray],
    ) -> Dict[int, Tuple[float, float]]:
        """Take in a frame and say where its players stand.

        Args:
            seconds: The frame's time in the video
            feet: Player -> the pixel their feet are at (the middle of the box's lower edge)
            image_to_field: Where the field lies in the frame (pixel -> field units,
                3x3), or None if that is not known

        Returns:
            Player -> place on the field (x across, y along), for the players whose
            feet are on ground the frame shows
        """
        if image_to_field is None:
            self.reset()
            return {}
        for player in [
            p
            for p, known in self._players.items()
            if not 0.0 <= seconds - known[0] <= FORGET_SECONDS
        ]:
            del self._players[player]
        if not feet:
            return {}
        # All players at once: a frame has fourteen, and one by one costs a millisecond
        players = list(feet)
        pixels = np.array([feet[player] for player in players], dtype=np.float64).reshape(-1, 2)
        here = pixels @ image_to_field[:, :2].T + image_to_field[:, 2]
        on_ground = np.abs(here[:, 2]) > 1e-12
        depth = np.where(on_ground, here[:, 2], 1.0)
        place = here[:, :2] / depth[:, None]
        # How the place changes with the pixel, and with that the pixels of doubt in
        # field units at each player's spot
        per_pixel = (
            image_to_field[None, :2, :2] - place[:, :, None] * image_to_field[None, None, 2, :2]
        ) / depth[:, None, None]
        doubt = per_pixel * np.square(FOOT_DOUBT_PIXELS) @ per_pixel.transpose(0, 2, 1)
        doubt = doubt + 1e-6 * np.eye(2)
        on_ground &= np.isfinite(doubt).all(axis=(1, 2))

        count = len(players)
        known = [self._players.get(player) for player in players]
        followed = np.array([entry is not None for entry in known]) & on_ground
        time = np.array([seconds - entry[0] if entry is not None else 0.0 for entry in known])
        followed &= time > 0.0
        state = np.zeros((count, 4))
        state[:, :2] = place
        # Standing or running is not known of a new player: any speed a player can have
        spread = np.zeros((count, 4, 4))
        spread[:, 2, 2] = spread[:, 3, 3] = NEW_SPEED_DOUBT
        spread[:, :2, :2] = doubt
        if followed.any():
            was = np.array([entry[1] for entry, on in zip(known, followed) if on])
            was_spread = np.array([entry[2] for entry, on in zip(known, followed) if on])
            moved, moved_spread = _moved_on(was, was_spread, time[followed])
            off = place[followed] - moved[:, :2]
            expected = moved_spread[:, :2, :2] + doubt[followed]
            trust = np.linalg.inv(expected)
            far = np.einsum("ni,nij,nj->n", off, trust, off)
            # Somebody else under the same number, or the field found anew: from here
            jumped = (np.linalg.norm(off, axis=1) > JUMP_UNITS) & (far > JUMP_DOUBTS**2)
            gain = moved_spread[:, :, :2] @ trust
            kept = np.nonzero(followed)[0][~jumped]
            state[kept] = (moved + (gain @ off[:, :, None])[:, :, 0])[~jumped]
            spread[kept] = (moved_spread - gain @ moved_spread[:, :2, :])[~jumped]

        places: Dict[int, Tuple[float, float]] = {}
        for index, player in enumerate(players):
            if not on_ground[index]:
                self._players.pop(player, None)
                continue
            self._players[player] = [seconds, state[index], spread[index]]
            places[player] = (float(state[index, 0]), float(state[index, 1]))
        return places


def _moved_on(
    state: np.ndarray, spread: np.ndarray, time: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Some players' places and speeds (N, 4), and the doubt about them (N, 4, 4), each
    its own `time` in seconds on."""
    step = np.tile(np.eye(4), (len(state), 1, 1))
    step[:, 0, 2] = step[:, 1, 3] = time
    # What a push of unknown direction adds in that time
    push = np.zeros_like(spread)
    push[:, 0, 0] = push[:, 1, 1] = time**4 / 4
    push[:, 0, 2] = push[:, 2, 0] = push[:, 1, 3] = push[:, 3, 1] = time**3 / 2
    push[:, 2, 2] = push[:, 3, 3] = time**2
    moved = (step @ state[:, :, None])[:, :, 0]
    return moved, step @ spread @ step.transpose(0, 2, 1) + MAX_PUSH**2 * push
