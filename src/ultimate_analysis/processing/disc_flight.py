"""Where a flying disc is over the field.

The top-down view is made by taking every pixel for a spot on the ground. A disc in the
air is not on the ground: seen from the drone it lies over ground further away, and in
the top-down view a pass at chest height lands yards behind the player who catches it.

One camera cannot tell how high a disc is. The disc is somewhere on the line from the
spot under the camera to the spot on the ground it is seen over, nearer the camera the
higher it flies. It is put where it would be at the height a disc is thrown and caught
at. That is right at both ends of a throw and too far away in between for a throw that
rises, but nearer than the ground everywhere.

That is while the disc flies. Once a throw has ended, both its ends are known: it left
the thrower's hands and came to the catcher's, or to the ground. The whole throw is then
fitted between the two (`fit_flight`): a course from the one to the other that may bend
to a side, rise and come down, and lose speed as a disc does, laid through the lines of
sight of all its sightings. That gives the height in between, and the path over the
ground is put right backwards.

Fitting a throw from the thrower's end alone, before it is known where it ends, was
tried and is worse than one fixed height: the thrower's place, the moment of release
and the camera's place are each off by enough that the far end of the throw is put
yards from the catcher (see docs/MEASUREMENTS.md).
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
# A throw is fitted as a whole if it was seen this often, and from within
# ANCHOR_GAP_SECONDS of both its ends
FIT_SIGHTINGS = 4
# What a throw is expected to do, each as far as it usually goes (the fit is held to
# these as strongly as to a sighting that is a yard off):
# - be ahead of a steady flight halfway: a disc loses speed. Share of the throw's length
#   by which it is ahead there, and how much that varies
FIT_AHEAD = (0.04, 0.04)
# - bend to a side, and rise: share of the throw's length, and field units on top
FIT_BEND = (0.1, 1.0)
FIT_RISE = (0.1, 1.0)
# A fit that leaves the sightings this far off on the ground (field units, root mean
# square), or rises higher than this, is not a throw between these two players
FIT_MISFIT = 3.0
FIT_MAX_HEIGHT = 20.0


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


def fit_flight(
    times: Sequence[float],
    seen_over: Sequence[Sequence[float]],
    cameras: Sequence[Sequence[float]],
    release: Tuple[float, Sequence[float], float],
    landing: Tuple[float, Sequence[float], float],
) -> Optional["FittedFlight"]:
    """The course of a throw between where it was released and where it came down.

    A sighting says that the disc is on the line from the camera to the spot on the
    ground it is seen over; how high on that line, one camera cannot tell. The course
    is the straight, steady flight from the release to the landing, and on top of it:
    being ahead or behind along the way, a bend to a side, and a rise. Each grows from
    nothing at the ends and may lean towards one of them. The course that lies nearest
    to all the lines of sight is taken, held to what throws usually do where the
    sightings leave it open (a throw straight away from the camera looks the same
    higher as further on).

    Args:
        times: When the disc was seen (seconds)
        seen_over: The spots on the ground it was seen over, (N, 2) in field units
        cameras: Where the camera was at each sighting: x, y and its height
        release: When, where (x, y) and how high the disc left the thrower
        landing: The same for where it was caught or came to lie

    Returns:
        The course, or None if the sightings are not those of such a throw
    """
    began, start, start_height = release[0], np.asarray(release[1], dtype=np.float64), release[2]
    ended, end, end_height = landing[0], np.asarray(landing[1], dtype=np.float64), landing[2]
    times = np.asarray(times, dtype=np.float64)
    seen_over = np.asarray(seen_over, dtype=np.float64).reshape(-1, 2)
    cameras = np.asarray(cameras, dtype=np.float64).reshape(-1, 3)
    if ended <= began or len(times) < FIT_SIGHTINGS or not np.isfinite(cameras).all():
        return None
    camera_height = np.abs(cameras[:, 2])
    if camera_height.min() < MIN_CAMERA_HEIGHTS * DISC_HEIGHT:
        return None
    way = end - start
    length = float(np.linalg.norm(way))
    side = np.array([-way[1], way[0]]) / length if length > 1e-6 else np.array([1.0, 0.0])
    share = np.clip((times - began) / (ended - began), 0.0, 1.0)
    bump, lean = share * (1.0 - share), share - 0.5
    # A disc this much higher is seen this much further from the camera
    per_height = (seen_over - cameras[:, :2]) / camera_height[:, None]
    level = start_height + (end_height - start_height) * share
    # Unknown: ahead, bend, bend's lean, rise, rise's lean. Two rows a sighting (x, y)
    columns = [
        bump[:, None] * way,
        bump[:, None] * side,
        (bump * lean)[:, None] * side,
        bump[:, None] * per_height,
        (bump * lean)[:, None] * per_height,
    ]
    wanted = seen_over - start - share[:, None] * way - level[:, None] * per_height
    bend = FIT_BEND[0] * length + FIT_BEND[1]
    rise = FIT_RISE[0] * length + FIT_RISE[1]
    # In the numbers of the unknowns: the bump is a quarter at its highest
    usual = np.array([4 * FIT_AHEAD[0], 0.0, 0.0, 4 * rise, 0.0])
    # A throw that bends one way and then the other, or rises late, is rare
    varies = np.array([4 * FIT_AHEAD[1], 4 * bend, 2 * bend, 4 * rise, 2 * rise])
    rows = np.vstack([np.concatenate([c[:, 0], c[:, 1]]) for c in columns]).T
    rows = np.vstack([rows, np.diag(1.0 / varies)])
    aim = np.concatenate([wanted[:, 0], wanted[:, 1], usual / varies])
    found = np.linalg.lstsq(rows, aim, rcond=None)[0]
    flight = FittedFlight(began, ended, start, end, start_height, end_height, side, found)
    places, heights = flight.at(times)
    on_line = cameras[:, :2] + (1.0 - heights / camera_height)[:, None] * (
        seen_over - cameras[:, :2]
    )
    misfit = float(np.sqrt(np.mean(np.sum((places - on_line) ** 2, axis=1))))
    if misfit > FIT_MISFIT or flight.at(np.linspace(began, ended, 21))[1].max() > FIT_MAX_HEIGHT:
        return None
    return flight


class FittedFlight:
    """The course of a throw as `fit_flight` found it."""

    def __init__(self, began, ended, start, end, start_height, end_height, side, found):
        self.began, self.ended = began, ended
        self._start, self._end = start, end
        self._heights = (start_height, end_height)
        self._side = side
        self._found = found

    def at(self, times: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Where the disc was over the field (N, 2) and how high (N), at some times."""
        span = self.ended - self.began
        share = np.clip((np.asarray(times, dtype=np.float64) - self.began) / span, 0.0, 1.0)
        bump, lean = share * (1.0 - share), share - 0.5
        ahead, bend, bend_lean, rise, rise_lean = self._found
        places = (
            self._start
            + (share + bump * ahead)[:, None] * (self._end - self._start)
            + (bump * (bend + bend_lean * lean))[:, None] * self._side
        )
        heights = (
            self._heights[0]
            + (self._heights[1] - self._heights[0]) * share
            + bump * (rise + rise_lean * lean)
        )
        return places, np.maximum(heights, 0.0)


class DiscPath:
    """Where the disc has been on the field over the last seconds.

    In a player's hands the disc is where the player stands, on the ground it is where
    it is seen, and in the air it is put where a disc at throwing height would be.
    What the disc is doing is known late: that it was caught is confirmed a moment
    after the catch, that it lies on the ground a second after it came to rest. So
    every sighting is kept with all it could mean, and when the answer comes, the
    path is put right from the moment the answer holds from.

    A flight that has ended is known at both ends: it began where the thrower stood
    and ended where the catcher stands, or where the disc lies. It is then fitted as a
    whole between the two (`fit_flight`), which says how high it was in between and so
    over which spots. A flight seen too little for that, or known at one end only, keeps
    its places at one fixed height, shifted so that it begins and ends where it did, by
    as much as each end was off and evenly in between.
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
        camera: Optional[Sequence[float]] = None,
    ) -> None:
        """Take in a frame.

        Args:
            seconds: The time of the frame in the video
            state: "held", "air" or "ground" (PossessionTracker.disc_state)
            ground: The spot on the ground the disc is seen over, if it is seen
            raised: The spot it is over if it flies (place_under_disc), if it is seen
            holder: Where the player holding the disc stands, if one does
            since: The time from which `state` holds, which may be before this frame
            camera: Where the camera is (x, y and its height in field units), for
                fitting a throw once it has ended; without it throws are not fitted
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
                "camera": None if camera is None else np.asarray(camera, dtype=float),
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
        if thrown is not None and landed is not None and self._fitted(begin, end, thrown, landed):
            return
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

    def _fitted(self, begin: int, end: int, thrown: dict, landed: dict) -> bool:
        """Put the flight in points `begin` to `end` where a fit of the whole throw has
        it; False if it cannot be fitted."""
        seen = [
            p
            for p in self._points[begin:end]
            if p["ground"] is not None and p["camera"] is not None and p["place"] is not None
        ]
        if (
            len(seen) < FIT_SIGHTINGS
            or seen[0]["t"] - thrown["t"] > ANCHOR_GAP_SECONDS
            or landed["t"] - seen[-1]["t"] > ANCHOR_GAP_SECONDS
        ):
            return False
        flight = fit_flight(
            [p["t"] for p in seen],
            [p["ground"] for p in seen],
            [p["camera"] for p in seen],
            (thrown["t"], thrown["place"], DISC_HEIGHT),
            (landed["t"], landed["place"], DISC_HEIGHT if landed["state"] == "held" else 0.0),
        )
        if flight is None:
            return False
        # Also the frames in which the disc was not found: the course says where it was
        places, heights = flight.at([p["t"] for p in self._points[begin:end]])
        for point, place, height in zip(self._points[begin:end], places, heights):
            point["place"], point["height"] = place, float(height)
        return True

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


# A disc is seen to fly if it was found this surely, in this many frames at least,
SURE_DISC = 0.5
FLIGHT_SIGHTINGS = 3
# ... no longer apart than this (seconds), the first and the last within that time,
FLIGHT_GAP_SECONDS = 0.12
FLIGHT_SECONDS = 0.5
# moved this fast (heights of a player a second: a soft pass is some 6, a pull 15),
FLIGHT_SPEED = 2.0
# went on the same way from one sighting to the next (the cosine of the turn), no more
# than so many times faster or slower than before,
FLIGHT_TURN = 0.7
FLIGHT_SPEED_CHANGE = 2.5
# and is at no player now: outside every box, widened by this share of it
FLIGHT_BOX_MARGIN = 0.15


class FlyingDisc:
    """Whether the disc is seen to fly: found well, frame after frame, on a clear and
    smooth course, away from every player.

    Who holds the disc is asked of a network that looks at one player at a time
    (processing/holder.py), and that network cannot know that nobody does: with the
    disc in the air it names a holder in a fifth of the frames. It is not asked while
    the disc is seen to fly. Where the disc is not found, or found only now and then,
    nothing is said here, and the network is asked: that is what it is for.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget the sightings (new video, seek, cut)."""
        self._seen: List[Tuple[float, np.ndarray]] = []  # Time, pixel; moved with the picture

    def move(self, camera_motion: np.ndarray) -> None:
        """Move the sightings along with the picture (homography from the last frame)."""
        for index, (seconds, place) in enumerate(self._seen):
            moved = camera_motion @ [place[0], place[1], 1.0]
            if abs(moved[2]) > 1e-9:
                self._seen[index] = (seconds, moved[:2] / moved[2])

    def update(
        self,
        seconds: float,
        discs: Sequence[dict],
        player_boxes: Sequence[Sequence[float]],
    ) -> bool:
        """Take in a frame; returns whether the disc is seen to fly in it.

        Args:
            seconds: The time of the frame in the video
            discs: The discs detected (bbox, confidence)
            player_boxes: The players' boxes (x1, y1, x2, y2)
        """
        if self._seen and seconds < self._seen[-1][0]:
            self.reset()
        found = max(discs, key=lambda disc: disc.get("confidence", 0.0), default=None)
        if found is None or found.get("confidence", 0.0) < SURE_DISC:
            # Not found, or not surely: the course ends here if it is not found again soon
            self._seen = [s for s in self._seen if seconds - s[0] <= FLIGHT_GAP_SECONDS]
            return False
        box = found["bbox"]
        place = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
        if self._seen and seconds - self._seen[-1][0] > FLIGHT_GAP_SECONDS:
            self._seen = []
        self._seen.append((seconds, place))
        self._seen = [s for s in self._seen if seconds - s[0] <= FLIGHT_SECONDS]
        for x1, y1, x2, y2 in player_boxes:
            pad_x, pad_y = FLIGHT_BOX_MARGIN * (x2 - x1), FLIGHT_BOX_MARGIN * (y2 - y1)
            if x1 - pad_x <= place[0] <= x2 + pad_x and y1 - pad_y <= place[1] <= y2 + pad_y:
                self._seen = [self._seen[-1]]  # At a player: caught, or about to be thrown
                return False
        if len(self._seen) < FLIGHT_SIGHTINGS:
            return False
        heights = [box[3] - box[1] for box in player_boxes]
        height = float(np.median(heights)) if heights else 80.0
        steps = [
            (later[1] - earlier[1]) / max(later[0] - earlier[0], 1e-6)
            for earlier, later in zip(self._seen, self._seen[1:])
        ]
        speeds = [float(np.linalg.norm(step)) for step in steps]
        if min(speeds) < FLIGHT_SPEED * height:
            return False
        for before, after, slow, fast in zip(steps, steps[1:], speeds, speeds[1:]):
            if float(before @ after) < FLIGHT_TURN * slow * fast:
                return False
            if max(slow, fast) > FLIGHT_SPEED_CHANGE * min(slow, fast):
                return False
        return True
