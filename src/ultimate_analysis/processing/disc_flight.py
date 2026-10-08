"""Where a flying disc is over the field.

The top-down view is made by taking every pixel for a spot on the ground. A disc in the
air is not on the ground: seen from the drone it lies over ground further away, and in
the top-down view a pass at chest height lands yards behind the player who catches it.

One camera cannot tell how high a disc is. The disc is somewhere on the line from the
spot under the camera to the spot on the ground it is seen over, nearer the camera the
higher it flies. It is put where it would be at the height a disc is thrown and caught
at. That is right at both ends of a throw and too far away in between for a throw that
rises, but nearer than the ground everywhere.

Fitting the whole throw was tried and is worse: a steady flight in a straight line from
where the thrower stood fixes the height on paper, and on drawn throws it does, but on
game footage the thrower's place, the moment of release and the camera's place are each
off by enough that the fit put the disc further from the catcher than the ground does
(see docs/MEASUREMENTS.md).
"""

from typing import Optional, Sequence, Tuple

import numpy as np

# The height a flying disc is taken to be at, in field units: where it is released and
# caught. Of 0.5, 1, 1.5 and 2, this put the disc nearest to the catcher at the catch.
DISC_HEIGHT = 1.0
# With the camera lower than this many times the disc's height, nothing is said: the
# camera's place is then wrong, or the disc is nearly level with it
MIN_CAMERA_HEIGHTS = 3.0


def place_under_disc(
    seen_over: Sequence[float], camera: Sequence[float], height: float = DISC_HEIGHT
) -> Optional[Tuple[float, float]]:
    """The spot on the field a flying disc is over.

    Args:
        seen_over: The spot on the ground the disc is seen over (its pixel taken for
            ground), in field units
        camera: The camera's place in field units: x, y, and its height
        height: How high the disc is taken to fly, in field units

    Returns:
        The spot, or None if the camera's place does not allow one to be given
    """
    seen_over = np.asarray(seen_over, dtype=np.float64)
    camera = np.asarray(camera, dtype=np.float64)
    camera_height = abs(float(camera[2]))
    if not np.isfinite(camera).all() or camera_height < MIN_CAMERA_HEIGHTS * height:
        return None
    place = camera[:2] + (1.0 - height / camera_height) * (seen_over - camera[:2])
    return float(place[0]), float(place[1])
