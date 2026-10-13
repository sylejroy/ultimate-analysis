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

A player whose legs are covered by somebody standing in front of them (the thrower
behind the mark) has a box that ends where the other player begins, higher or lower
with every frame. Their feet are taken to be as far below the top of the box as the
player is tall: the camera says how many pixels a player of 1.80 m is tall at that spot,
and the player is as much taller or shorter than that as their box was while nothing
covered them (as the others' boxes are in this frame, for a player not yet seen free).
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
# A player's height is remembered as it was over about this long, while nobody stood in
# front of them (seconds)
HEIGHT_SECONDS = 1.0
# The player whose height in pixels the camera gives, in field units (yards): 1.80 m
PLAYER_HEIGHT = 1.9685
# A player not yet seen with nobody in front of them is taken to be as tall as the
# others, if at least this many of those are seen free in the frame
OTHERS_TO_GO_BY = 3
# Another player covers a player's feet if their box lies over this share of the
# player's width, reaches up past the box's lower edge and ends below it
COVERING_WIDTH = 0.25
# Feet put below a box are less sure than feet seen: this share of what was added to
# the box, on top of the doubt of any foot (pixels up and down)
COVERED_DOUBT = 0.5


class SteadyPlaces:
    """Follows the place of every player on the field (a filter per player that takes
    the place and the speed along as they were, and each new place by its doubt)."""

    def __init__(self) -> None:
        # Player -> [time last seen, state (x, y, speed x, speed y), its doubt (4x4)]
        self._players: Dict[int, list] = {}
        # Player -> [time last seen, how tall they are with nobody in front of them: the
        # height of the box in heights of a player of 1.80 m at their feet (without a
        # camera: in field units, as wide as that many pixels are there)]
        self._heights: Dict[int, list] = {}

    def reset(self) -> None:
        """Forget everybody (a cut, another video, the field lost)."""
        self._players.clear()
        self._heights.clear()

    def rename(self, was: int, now: int) -> None:
        """A player is known under another number from now on."""
        if was in self._players:
            self._players[now] = self._players.pop(was)
        if was in self._heights:
            self._heights[now] = self._heights.pop(was)

    def update(
        self,
        seconds: float,
        feet: Dict[int, Sequence[float]],
        image_to_field: Optional[np.ndarray],
        boxes: Optional[Dict[int, Sequence[float]]] = None,
        camera: Optional[np.ndarray] = None,
    ) -> Dict[int, Tuple[float, float]]:
        """Take in a frame and say where its players stand.

        Args:
            seconds: The frame's time in the video
            feet: Player -> the pixel their feet are at (the middle of the box's lower edge)
            image_to_field: Where the field lies in the frame (pixel -> field units,
                3x3), or None if that is not known
            boxes: Player -> their box (left, top, right, bottom), to put the feet of
                those right whose legs another of them covers; None: feet as they are
            camera: (x, y, height above the ground) -> pixel, 3x4, to know how tall a
                player is in the picture at a spot; None: by the players' own boxes
                alone, and nothing for a player not yet seen free

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
        for player in [
            p
            for p, known in self._heights.items()
            if not 0.0 <= seconds - known[0] <= FORGET_SECONDS
        ]:
            del self._heights[player]
        if not feet:
            return {}
        # All players at once: a frame has fourteen, and one by one costs a millisecond
        players = list(feet)
        pixels = np.array([feet[player] for player in players], dtype=np.float64).reshape(-1, 2)
        place, per_pixel, on_ground = _on_field(pixels, image_to_field)
        foot_doubt = np.tile(np.asarray(FOOT_DOUBT_PIXELS, dtype=np.float64), (len(players), 1))
        if boxes is not None:
            lower_edge = pixels[:, 1].copy()
            # A few times over: how tall a player is in pixels depends on how far off
            # their feet are, which is what is being found
            before = np.zeros(len(players))
            for _ in range(3):
                added = self._below_the_boxes(
                    seconds, players, boxes, place, per_pixel, on_ground, camera
                )
                if np.abs(added - before).max() < 0.25:
                    break  # No feet are covered, or they are found
                before = added
                pixels[:, 1] = lower_edge + added
                place, per_pixel, on_ground = _on_field(pixels, image_to_field)
                foot_doubt[:, 1] = FOOT_DOUBT_PIXELS[1] + COVERED_DOUBT * added
        # The pixels of doubt in field units at each player's spot
        doubt = per_pixel * np.square(foot_doubt)[:, None, :] @ per_pixel.transpose(0, 2, 1)
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

    def _below_the_boxes(
        self,
        seconds: float,
        players: Sequence[int],
        boxes: Dict[int, Sequence[float]],
        place: np.ndarray,
        per_pixel: np.ndarray,
        on_ground: np.ndarray,
        camera: Optional[np.ndarray],
    ) -> np.ndarray:
        """For each player, how many pixels below their box their feet are: nothing
        unless somebody stands in front of their legs and the box is shorter than they
        were tall. Takes in how tall those are whom nobody covers."""
        added = np.zeros(len(players))
        have = np.array([player in boxes for player in players]) & on_ground
        if not have.any():
            return added
        box = np.array(
            [boxes[player] if player in boxes else (0.0, 0.0, 0.0, 0.0) for player in players],
            dtype=np.float64,
        ).reshape(-1, 4)
        width = np.maximum(box[:, 2] - box[:, 0], 1e-6)
        shared = np.minimum(box[:, None, 2], box[None, :, 2]) - np.maximum(
            box[:, None, 0], box[None, :, 0]
        )
        # [a, b]: b stands in front of a's legs
        covers = (
            (shared > COVERING_WIDTH * width[:, None])
            & (box[None, :, 1] < box[:, None, 3])
            & (box[None, :, 3] > box[:, None, 3])
            & have[None, :]
        )
        covered = covers.any(axis=1)
        # What a player's height is counted in, in pixels at their feet: the height of
        # a player of 1.80 m there, by the camera; without one, a field unit sideways.
        # The same player is as many of these tall wherever the camera looks and
        # however near it has come
        if camera is not None:
            unit = _pixels_tall(place, camera)
        else:
            with np.errstate(divide="ignore"):
                unit = 1.0 / np.linalg.norm(per_pixel[:, :, 0], axis=1)
        have &= np.isfinite(unit) & (unit > 0.0)
        tall = (box[:, 3] - box[:, 1]) / np.where(have, unit, 1.0)
        free = have & ~covered
        # A player not yet seen free is like the others: boxes are a little taller or
        # shorter than the camera's player all over the frame
        like_others = (
            float(np.median(tall[free]))
            if camera is not None and free.sum() >= OTHERS_TO_GO_BY
            else None
        )
        for index, player in enumerate(players):
            if not have[index]:
                continue
            known = self._heights.get(player)
            if covered[index]:
                if known is not None:
                    known[0] = seconds
                is_tall = known[1] if known is not None else like_others
                if is_tall is not None:
                    added[index] = max(0.0, (is_tall - tall[index]) * unit[index])
            elif known is None:
                self._heights[player] = [seconds, float(tall[index])]
            else:
                share = 1.0 - float(np.exp(-max(seconds - known[0], 0.0) / HEIGHT_SECONDS))
                self._heights[player] = [seconds, known[1] + share * (tall[index] - known[1])]
        return added


def _pixels_tall(place: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """How many pixels a player of 1.80 m standing at each place (N, 2) is tall in the
    picture; not a number where the camera cannot say."""
    feet = np.column_stack([place, np.zeros(len(place)), np.ones(len(place))]) @ camera.T
    head = feet + PLAYER_HEIGHT * camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        tall = feet[:, 1] / feet[:, 2] - head[:, 1] / head[:, 2]
    return np.where((feet[:, 2] > 1e-9) & (head[:, 2] > 1e-9), tall, np.nan)


def _on_field(
    pixels: np.ndarray, image_to_field: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where some pixels (N, 2) lie on the field (N, 2), how the place changes with the
    pixel there (N, 2, 2), and whether they lie on ground the frame shows (N)."""
    here = pixels @ image_to_field[:, :2].T + image_to_field[:, 2]
    on_ground = np.abs(here[:, 2]) > 1e-12
    depth = np.where(on_ground, here[:, 2], 1.0)
    place = here[:, :2] / depth[:, None]
    per_pixel = (
        image_to_field[None, :2, :2] - place[:, :, None] * image_to_field[None, None, 2, :2]
    ) / depth[:, None, None]
    return place, per_pixel, on_ground


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
