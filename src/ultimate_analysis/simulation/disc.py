"""How a simulated disc flies: along a curve over the field, up and down again.

A throw is aimed at a spot, with a curve and a height. Over the field the disc follows a
quadratic Bezier from the hand to the spot, bent sideways by the curve at its middle:
an inside-out or outside-in throw leaves the hand in one direction and comes round to
the spot. Its height rises from the hand to the top of its arc halfway and comes down
to catching height at the end. A higher throw is slower over the ground: it hangs.

The disc's angle, from flat (0) to a steep blade (1), shapes the rest: a blade flies
faster, takes a strong curve and cuts through the wind, but comes in on edge and is
harder to catch; a flat disc floats, curves little and drifts.

The wind carries it: downwind it flies faster over the ground, upwind slower; across, it
is drifted sideways, more the longer it is in the air and the higher it flies, so that
it comes down off where it was aimed unless the thrower aimed off for it.

Anybody can get a hand to the disc where it passes within PATH_REACH of them and below
JUMP_REACH; at the end it comes down to catching height for the receiver.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

HAND_HEIGHT = 1.0  # Yards: where the disc leaves the hand and is caught
JUMP_REACH = 2.6  # Yards: the highest a player gets a hand to
PATH_REACH = 1.0  # Yards across from the disc's path
FLAT = 2.0  # Yards: the top of the arc of a flat throw
HIGHEST = 15.0  # Yards: the highest a throw goes
HANG = 0.03  # How much slower over the ground, per yard of height above a flat throw
RELEASE_SECONDS = 0.15
DISC_YARDS_PER_SECOND = 17.7  # Over the ground, a flat throw
LONGEST_THROW = 70.0  # Yards
WIND_ALONG = 0.5  # Yards a second faster over the ground, per yard a second of wind behind it
SLOWEST = 6.0  # Yards a second over the ground, into the strongest wind
WIND_DRIFT = (
    0.15  # Yards a flat throw drifts, per yard a second of wind across and second in the air
)
DRIFT_PER_HEIGHT = 0.05  # ... and more by this share per yard of height above a flat throw
# The disc's angle: the most curve a flat disc and a blade take, as shares of the length;
# how much faster a blade flies; how much more it is dropped
FLAT_CURVE, BLADE_CURVE = 0.1, 0.4
BLADE_SPEED, BLADE_DROPPED = 0.3, 0.06
# How far the disc is tipped from flat, flat (a flat throw is never quite flat) to a
# blade: a side wind pushes on the face it turns to the wind, so a blade drifts most
FLAT_ANGLE, BLADE_ANGLE = np.radians(15.0), np.radians(75.0)


def usual_height(distance: float) -> float:
    """The top of the arc of a throw thrown as most are: flat when short, higher long."""
    return float(np.clip(FLAT + 0.08 * distance, FLAT, 8.0))


def highest_for(distance: float) -> float:
    """A short throw cannot go very high."""
    return float(min(HIGHEST, HAND_HEIGHT + 0.4 * distance + 1.0))


@dataclass
class Flight:
    """A disc in the air."""

    start: np.ndarray  # Where it left the hand, on the field
    aimed: np.ndarray  # Where it was thrown to (where it comes down, without the wind)
    released: float  # When
    lands: float  # When it comes down
    thrower: int  # Index of the player who threw it (0 to 13)
    curve: float = 0.0  # Yards it is bent to the left of the straight line, at the middle
    height: float = FLAT  # Yards: the top of its arc
    drift: np.ndarray = field(default_factory=lambda: np.zeros(2))  # By the wind, at the end
    tilt: float = 0.0  # The disc's angle: 0 flat, 1 a steep blade
    pull: bool = False  # The pull that starts a point: nobody plays it on its way
    # Player -> when they got within reach of where it comes down (and stayed)
    in_reach_since: dict = field(default_factory=dict)

    @property
    def target(self) -> np.ndarray:
        """Where it comes down."""
        return self.aimed + self.drift

    @property
    def bend(self) -> np.ndarray:
        """The Bezier's middle point: twice as far off the line as the path at its middle."""
        line = self.aimed - self.start
        size = float(np.hypot(*line))
        left = np.array([-line[1], line[0]]) / max(size, 1e-9)
        return (self.start + self.aimed) / 2.0 + left * 2.0 * self.curve

    def share(self, seconds: float) -> float:
        """How far along its flight the disc is, 0 to 1."""
        return float(
            np.clip((seconds - self.released) / max(self.lands - self.released, 1e-9), 0, 1)
        )

    def point(self, s: float) -> Tuple[np.ndarray, float]:
        """Where the disc is over the field, and how high, a share `s` along its flight."""
        place = (1 - s) ** 2 * self.start + 2 * s * (1 - s) * self.bend + s**2 * self.aimed
        place = place + self.drift * s**2  # The wind carries it more the longer it flies
        return place, HAND_HEIGHT + (self.height - HAND_HEIGHT) * 4.0 * s * (1.0 - s)

    def at(self, seconds: float) -> Tuple[np.ndarray, float]:
        """Where the disc is over the field, and how high, at a time."""
        return self.point(self.share(seconds))

    def leaving(self) -> np.ndarray:
        """The direction the disc leaves the hand in."""
        way = self.bend - self.start
        return way / max(float(np.hypot(*way)), 1e-9)

    def length(self) -> float:
        """How far it flies over the ground, roughly."""
        points = [self.point(s / 8)[0] for s in range(9)]
        return float(sum(np.hypot(*(b - a)) for a, b in zip(points, points[1:])))


def flight_of(
    start: np.ndarray,
    target: np.ndarray,
    released: float,
    thrower: int,
    curve: float = 0.0,
    height: Optional[float] = None,
    wind: Optional[np.ndarray] = None,
    tilt: Optional[float] = None,
) -> Flight:
    """A throw from `start` to `target`: the curve within what an arm can put on a disc,
    the height within what the length allows (as most throws go if not given), its time
    in the air from both and the wind (yards a second, on the field), and how far the
    wind carries it. The angle is as steep as the curve asks if not given, and limits
    the curve if given."""
    start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
    offset = target - start
    distance = float(np.hypot(*offset))
    if distance > LONGEST_THROW:
        target = start + offset * (LONGEST_THROW / distance)
        distance = LONGEST_THROW
    height = usual_height(distance) if height is None else float(height)
    height = float(np.clip(height, HAND_HEIGHT, highest_for(distance)))
    if tilt is None:  # As steep as the curve asks
        tilt = (abs(curve) / max(distance, 1e-9) - FLAT_CURVE) / (BLADE_CURVE - FLAT_CURVE)
    tilt = float(np.clip(tilt, 0.0, 1.0))
    most = (FLAT_CURVE + (BLADE_CURVE - FLAT_CURVE) * tilt) * distance
    curve = float(np.clip(curve, -most, most))
    flight = Flight(start, target, released, released, thrower, curve, height)
    flight.tilt = tilt
    wind = np.zeros(2) if wind is None else np.asarray(wind, dtype=float)
    way = offset / max(distance, 1e-9)
    behind = float(wind @ way)
    speed = DISC_YARDS_PER_SECOND / (1.0 + HANG * max(0.0, height - FLAT))
    speed = max(SLOWEST, speed * (1.0 + BLADE_SPEED * tilt) + WIND_ALONG * behind)
    flying = RELEASE_SECONDS + flight.length() / speed
    flight.lands = released + flying
    across = wind - behind * way
    higher = 1.0 + DRIFT_PER_HEIGHT * max(0.0, height - FLAT)
    flight.drift = across * WIND_DRIFT * flying * higher * face_to_the_side(tilt)
    return flight


def face_to_the_side(tilt: float) -> float:
    """How much of the disc's face a side wind pushes on, against a flat throw's."""
    angle = FLAT_ANGLE + (BLADE_ANGLE - FLAT_ANGLE) * float(np.clip(tilt, 0.0, 1.0))
    return float(np.sin(angle) / np.sin(FLAT_ANGLE))


def drift_of(
    start, target, curve: float, height: Optional[float], wind: np.ndarray, tilt=None
) -> np.ndarray:
    """How far the wind would carry a throw from where it is aimed: for aiming off."""
    return flight_of(start, target, 0.0, 0, curve, height, wind, tilt).drift
