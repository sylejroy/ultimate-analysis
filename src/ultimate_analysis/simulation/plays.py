"""Set plays: called by the holder, they start the offence's movement.

A play gives each role a route: places to be at given seconds after the call, relative
to where the thrower stood then, as the offence attacks (y towards the end zone
attacked). Roles count from the holder: role 1 is the next player after them in the
team's order, and so on round, so that every player knows their role from who holds the
disc. The routes are plain numbers: learning may change them, and new plays may come of
it.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .game import LENGTH, PLAYERS, WIDTH


@dataclass
class Play:
    name: str
    routes: Dict[int, np.ndarray]  # Role -> (n, 3): seconds after the call, x, y offsets

    @property
    def seconds(self) -> float:
        return max(float(route[-1, 0]) for route in self.routes.values())

    def spot(self, role: int, since: float, origin: np.ndarray) -> Optional[np.ndarray]:
        """Where the play wants a role `since` seconds after the call; None if it has no
        route for the role or the route is over."""
        route = self.routes.get(role)
        if route is None or since > route[-1, 0]:
            return None
        x = np.interp(since, route[:, 0], route[:, 1])
        y = np.interp(since, route[:, 0], route[:, 2])
        place = origin + [x, y]
        return np.array([np.clip(place[0], 1.0, WIDTH - 1.0), np.clip(place[1], 1.0, LENGTH - 1.0)])


def role_of(player: int, holder: int) -> int:
    """A player's role from who holds the disc: 1 to 6."""
    return (player - holder) % PLAYERS


def _route(*points) -> np.ndarray:
    return np.array(points, dtype=float)


PLAYS: List[Play] = [
    # One cutter isolated deep: a step under, then away; the rest clear to the sidelines
    Play(
        "Iso deep",
        {
            2: _route((0.0, 0.0, 14.0), (0.8, 2.0, 9.0), (2.2, 4.0, 30.0), (3.5, 4.0, 40.0)),
            3: _route((0.0, 0.0, 19.0), (1.0, -14.0, 22.0), (3.5, -15.0, 24.0)),
            4: _route((0.0, 0.0, 24.0), (1.0, 14.0, 26.0), (3.5, 15.0, 28.0)),
            5: _route((0.0, 0.0, 29.0), (1.2, -14.0, 32.0), (3.5, -15.0, 34.0)),
            6: _route((0.0, 0.0, 34.0), (1.2, 14.0, 36.0), (3.5, 15.0, 38.0)),
        },
    ),
    # The dump moves to give a swing, and a cutter goes to the break side for it
    Play(
        "Dump swing",
        {
            1: _route((0.0, 6.0, -6.0), (0.6, 4.0, -4.0), (2.5, 4.0, -4.0)),
            3: _route((0.0, 0.0, 19.0), (1.0, -6.0, 14.0), (2.0, -16.0, 10.0), (3.0, -18.0, 14.0)),
        },
    ),
    # The stack splits to both sides and the middle opens for a cut from the back
    Play(
        "Split",
        {
            2: _route((0.0, 0.0, 14.0), (1.0, -10.0, 14.0), (3.0, -10.0, 16.0)),
            3: _route((0.0, 0.0, 19.0), (1.0, 10.0, 19.0), (3.0, 10.0, 21.0)),
            4: _route((0.0, 0.0, 24.0), (1.0, -10.0, 24.0), (3.0, -10.0, 26.0)),
            5: _route((0.0, 0.0, 29.0), (1.0, 10.0, 29.0), (3.0, 10.0, 31.0)),
            6: _route((0.0, 0.0, 34.0), (1.2, 0.0, 30.0), (2.2, 0.0, 14.0), (3.0, 2.0, 12.0)),
        },
    ),
]
