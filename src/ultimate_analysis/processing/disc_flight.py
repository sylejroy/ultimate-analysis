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

from typing import List, Optional, Sequence, Tuple

import numpy as np

# The height a flying disc is taken to be at, in field units: where it is released and
# caught. Of 0.5, 1, 1.5 and 2, this put the disc nearest to the catcher at the catch.
DISC_HEIGHT = 1.0
# With the camera lower than this many times the disc's height, nothing is said: the
# camera's place is then wrong, or the disc is nearly level with it
MIN_CAMERA_HEIGHTS = 3.0
# The path of the disc is not drawn across a time longer than this (seconds) in which
# the disc was neither seen nor held
PATH_GAP_SECONDS = 1.0
# A flight is tied to the thrower or the catcher only if they are no further than this
# from where the flight was seen to begin or end (field units): further, and it is not
# their throw or catch
MAX_CORRECTION = 6.0
# ... and only if the flight was seen to within this long of the release or the catch
ANCHOR_GAP_SECONDS = 0.5
# A sighting further from the last place of the disc than a disc flies in the time
# between (field units a second, and some room for where it is put) is something else:
# a cone, a line marker, a white shoe
MAX_DISC_SPEED = 30.0
SIGHTING_ROOM = 3.0
# The places of a carried disc are averaged over this long, as the trails of the
# players are: a player's box bobs with every step
CARRIED_SMOOTHING_SECONDS = 0.4


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


class DiscPath:
    """Where the disc has been on the field over the last seconds.

    In a player's hands the disc is where the player stands, on the ground it is where
    it is seen, and in the air it is put where a disc at throwing height would be.
    What the disc is doing is known late: that it was caught is confirmed a moment
    after the catch, that it lies on the ground a second after it came to rest. So
    every sighting is kept with all it could mean, and when the answer comes, the
    path is put right from the moment the answer holds from.

    A flight that has ended is known at both ends: it began where the thrower stood
    and ended where the catcher stands, or where the disc lies. Its places in between,
    taken at one fixed height, are shifted so that it begins and ends there, by as much
    as each end was off and evenly in between.
    """

    def __init__(self, seconds: float = 8.0):
        self.seconds = seconds  # How far back the path is kept
        self.reset()

    def reset(self) -> None:
        """Forget the path (new video, seek, cut)."""
        self._points: List[dict] = []
        self._state: Optional[str] = None
        self._last: Optional[Tuple[float, np.ndarray]] = None  # When and where it last was

    @property
    def last_place(self) -> Optional[np.ndarray]:
        """Where the disc was last, held or seen; None if it has not been yet."""
        return None if self._last is None else self._last[1]

    def _can_be_the_disc(self, seconds: float, place: np.ndarray) -> bool:
        """Whether the disc can have got to a place from where it last was."""
        if self._last is None:
            return True
        reach = MAX_DISC_SPEED * (seconds - self._last[0]) + SIGHTING_ROOM
        return float(np.linalg.norm(place - self._last[1])) <= reach

    def add(
        self,
        seconds: float,
        state: str,
        ground: Optional[Sequence[float]] = None,
        raised: Optional[Sequence[float]] = None,
        holder: Optional[Sequence[float]] = None,
        since: Optional[float] = None,
    ) -> None:
        """Take in a frame.

        Args:
            seconds: The time of the frame in the video
            state: "held", "air" or "ground" (PossessionTracker.disc_state)
            ground: The spot on the ground the disc is seen over, if it is seen
            raised: The spot it is over if it flies (place_under_disc), if it is seen
            holder: Where the player holding the disc stands, if one does
            since: The time from which `state` holds, which may be before this frame
        """
        if self._points and seconds < self._points[-1]["t"]:
            self.reset()
        seen = raised if raised is not None else ground
        if seen is not None and not self._can_be_the_disc(seconds, np.asarray(seen, float)):
            ground = raised = seen = None
        if state == "held" and holder is not None:
            self._last = (seconds, np.asarray(holder, dtype=float))
        elif seen is not None:
            self._last = (seconds, np.asarray(seen, dtype=float))
        self._points.append(
            {
                "t": seconds,
                "state": state,
                "ground": None if ground is None else np.asarray(ground, dtype=float),
                "raised": None if raised is None else np.asarray(raised, dtype=float),
                "holder": None if holder is None else np.asarray(holder, dtype=float),
            }
        )
        start = len(self._points) - 1
        if state != self._state:
            # What the disc does now, it has done since `since`: the frames from then on
            # were taken for what it did before
            while since is not None and start > 0 and self._points[start - 1]["t"] >= since:
                start -= 1
            for point in self._points[start:]:
                point["state"] = state
                if state == "held":
                    point["holder"] = self._points[-1]["holder"]
        for point in self._points[start:]:
            key = {"held": "holder", "ground": "ground"}.get(point["state"], "raised")
            point["place"] = point[key]
        if state != self._state and state != "air":
            self._tie_down_flight(start)
        self._state = state
        while self._points and self._points[0]["t"] < seconds - self.seconds:
            self._points.pop(0)

    def _tie_down_flight(self, end: int) -> None:
        """Shift the flight that ended before point `end` so that it begins where the
        thrower stood and ends where the disc was caught or came to lie."""
        begin = end
        while begin > 0 and self._points[begin - 1]["state"] == "air":
            begin -= 1
        flight = [p for p in self._points[begin:end] if p["place"] is not None]
        for point in self._points[begin:end]:
            point["state"] = "flown"
        landed = next((p for p in self._points[end:] if p["place"] is not None), None)
        thrown = self._points[begin - 1] if begin > 0 else None
        if thrown is not None and (thrown["state"] != "held" or thrown["place"] is None):
            thrown = None
        if not flight:
            return
        first, last = flight[0], flight[-1]
        duration = last["t"] - first["t"]
        # How fast the disc was seen to move at each end, to carry the first and last
        # sighting on to the moment of the release and of the catch
        along = flight[min(4, len(flight) - 1)]
        back = flight[max(0, len(flight) - 5)]
        leaving = (along["place"] - first["place"]) / max(along["t"] - first["t"], 1e-6)
        arriving = (last["place"] - back["place"]) / max(last["t"] - back["t"], 1e-6)
        if len(flight) < 2:
            leaving = arriving = np.zeros(2)

        def off(anchor: Optional[dict], seen: dict, speed: np.ndarray) -> np.ndarray:
            if anchor is None or abs(anchor["t"] - seen["t"]) > ANCHOR_GAP_SECONDS:
                return np.zeros(2)
            by = anchor["place"] - (seen["place"] + speed * (anchor["t"] - seen["t"]))
            return by if np.linalg.norm(by) <= MAX_CORRECTION else np.zeros(2)

        at_release, at_catch = off(thrown, first, leaving), off(landed, last, arriving)
        for point in flight:
            share = (point["t"] - first["t"]) / duration if duration > 0 else 1.0
            point["place"] = point["place"] + (1.0 - share) * at_release + share * at_catch

    def stretches(self) -> List[np.ndarray]:
        """The path as lines of places on the field (N, 2), oldest first: one line as
        long as the disc was followed, a new one after each time it was lost."""
        points = [point for point in self._points if point.get("place") is not None]
        if len(points) < 2:
            return []
        times = np.array([point["t"] for point in points])
        places = np.array([point["place"] for point in points], dtype=np.float64)
        held = np.array([point["state"] == "held" for point in points])
        # A carry: frames one after the other in which the disc was held and did not
        # jump, so by one player. Its places are averaged, as many before as after each,
        # so that the path does not lag.
        steps = np.linalg.norm(np.diff(places, axis=0), axis=1)
        carried_on = held[1:] & held[:-1] & (steps <= SIGHTING_ROOM)
        edges = [0, *(np.flatnonzero(~carried_on) + 1), len(points)]
        for begin, end in zip(edges[:-1], edges[1:]):
            if end - begin < 3 or not held[begin]:
                continue
            between = (times[end - 1] - times[begin]) / (end - begin - 1)
            reach = int(round(CARRIED_SMOOTHING_SECONDS / 2 / max(between, 1e-6)))
            index = np.arange(end - begin)
            half = np.minimum(reach, np.minimum(index, end - begin - 1 - index))
            sums = np.vstack([np.zeros((1, 2)), np.cumsum(places[begin:end], axis=0)])
            places[begin:end] = (sums[index + half + 1] - sums[index - half]) / (2 * half + 1)[
                :, None
            ]
        lost = np.flatnonzero(np.diff(times) > PATH_GAP_SECONDS) + 1
        return [line for line in np.split(places, lost) if len(line) >= 2]
