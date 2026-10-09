"""Possession: which tracked player holds the disc, and which team has it.

The disc belongs to the player whose bounding box it is in. A disc in flight passes other
players on its way, and the detections flicker, so the holder only changes once the disc
has been seen at the same new place (another player, or no player at all) for a while:
each frame it is seen there counts for that place, and half a frame against every other.
Frames without a detected disc say nothing and leave the holder as it is. Being sure
takes a moment, but the change did not happen when it was confirmed: it is dated back to
when the disc was first seen at the new place (`since`), for whoever keeps a record.
Live playback skips frames to keep up; a frame then counts for those skipped before it.

The player with the disc has an opponent standing right in front of them: the mark. Seen
from the drone, the mark often covers the thrower, or the disc is held out past the mark.
These keep the disc with the thrower then:

- While the disc is in the holder's box it is the holder's, however near it is to someone
  else. So is a disc at no player but within a body's height of the holder: a throw is
  further away than that within a moment, and a disc held out or badly placed by the
  detector is not.
- A holder whose track the tracker loses and begins again under a new number stays
  the holder.
- A holder who is not found for a moment (covered by the mark) is remembered where they
  were last seen; a disc there is still theirs.
- A disc that shows at a player standing right by the holder has most likely not changed
  hands: a pass goes somewhere. It must stay there much longer before that player is
  taken for the holder, which they are if the holder was chosen wrongly to begin with.

The team in possession is that of the last holder whose team is known. A player of the
other team takes longer to be taken for the holder: the disc changes teams far less often
than a defender stands where it is caught. A disc that lies still at no player for a
while is on the ground, and that is a turnover in nearly every case: the other team has
it from then on, before anyone has picked it up. Only the disc that was followed there
from a player's hands counts: a brick mark on the grass lies still at no player too.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting

Box = Tuple[float, float, float, float]

# A disc on the ground stays within this share of a player's height of where it was. A
# disc in the air does not, however slowly it floats.
GROUND_RADIUS = 0.15
GROUND_RADIUS_PIXELS = 12.0  # The same when no player is in the picture to measure by
# A disc on the ground counts as picked up or thrown again once it is this many times
# further from where it lay: the detection of a lying disc wanders by more than the radius
GROUND_LEFT = 4.0
# The fastest a disc flies across the picture, in heights of a player per second (a
# hard throw is some 30 m/s: 17 heights), and how long it may be unseen and still be
# taken for the same disc when it is seen again
MAX_DISC_SPEED = 25.0
MAX_UNSEEN_SECONDS = 1.5
# A disc not seen for this long is no longer said to be flying
FLIGHT_UNSEEN_SECONDS = 0.5
# A new track that covers this much (IoU) of where a lost holder stood is the holder
SAME_PLAYER_OVERLAP = 0.5
# What a frame with the disc elsewhere takes from a place's count, in frames. Less than
# one: of two places the disc is seen at in turn, the one it is seen at more often wins.
ELSEWHERE = 0.5
# What a frame with the disc at the holder takes from every other place's count
AT_HOLDER = 2.0


def _widened(box: Sequence[float], margin: float) -> Box:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return (x1 - margin * width, y1 - margin * height, x2 + margin * width, y2 + margin * height)


def _contains(box: Sequence[float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _overlap(first: Sequence[float], second: Sequence[float]) -> bool:
    return (
        first[0] < second[2]
        and second[0] < first[2]
        and first[1] < second[3]
        and second[1] < first[3]
    )


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    return shared / (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - shared
    )


def _disc_centre(disc_box: Sequence[float]) -> Tuple[float, float]:
    return (disc_box[0] + disc_box[2]) / 2, (disc_box[1] + disc_box[3]) / 2


def _is_player(track: Any) -> bool:
    return getattr(track, "class_name", None) == "player"


def player_at_disc(
    disc_box: Sequence[float], tracks: List[Any], box_margin: float
) -> Optional[int]:
    """Track ID of the player whose box, widened by a margin, contains the disc centre.

    A disc held at arm's length lies just outside the player's box, hence the margin (a
    fraction of the box width and height). Of several players, the one whose centre is
    nearest to the disc relative to their size is chosen.
    """
    disc_x, disc_y = _disc_centre(disc_box)

    nearest_id, nearest_distance = None, float("inf")
    for track in tracks:
        if not _is_player(track):
            continue
        x1, y1, x2, y2 = track.to_ltrb()
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        if not _contains(_widened((x1, y1, x2, y2), box_margin), disc_x, disc_y):
            continue
        distance = ((disc_x - (x1 + x2) / 2) ** 2 + (disc_y - (y1 + y2) / 2) ** 2) ** 0.5 / height
        if distance < nearest_distance:
            nearest_id, nearest_distance = track.track_id, distance
    return nearest_id


class PossessionTracker:
    """Follows the disc holder and the team in possession over the frames of a video."""

    def __init__(self):
        self.frame_rate = 30.0  # Of the video; the durations in the settings are seconds
        self.reset()

    def reset(self) -> None:
        """Forget the holder (new video, seek)."""
        self.holder_id: Optional[int] = None  # None: nobody, e.g. the disc is in flight
        self.team: Optional[int] = None  # The team in possession (0 or 1), if known
        self.on_ground = False  # The disc lies on the ground
        # Place (a player's track ID, or None for no player) -> frames the disc was seen there
        self._seen_at: Dict[Optional[int], float] = {}
        self._first_seen: Dict[Optional[int], int] = {}  # and the frame it was first seen there
        # The frame from which the holder and the team are as they are now: earlier than
        # the frame they were confirmed in
        self.since = 0
        self._step = 1  # Video frames from the frame before to this one
        self._holder_box: Optional[Box] = None  # Where the holder was last seen
        self._holder_unseen = 0  # Frames since then
        self._frame: Optional[int] = None  # The frame taken in last
        self._resting_at: Optional[Tuple[float, float]] = None  # A disc at no player: where
        self._resting_since = 0  # and since which frame it has not moved from there
        self._turned_over = False  # The disc has lain on the ground since someone last held it
        self._known_tracks: set = set()  # Every player track seen so far
        self._disc_seen: Optional[int] = None  # The frame the disc was last seen in
        self._holder_teams = [0, 0]  # Frames the present holder was seen as of each team
        # The last sighting of the disc (frame, pixel), and whether what is being seen
        # is the disc of the game: followed from a player's hands without a break
        self._last_sighting: Optional[Tuple[int, Tuple[float, float]]] = None
        self._followed_from_a_player = False

    @property
    def disc_state(self) -> str:
        """ "held", "ground", or "air" (which is also: not seen for a while)."""
        if self.holder_id is not None:
            return "held"
        return "ground" if self.on_ground else "air"

    @property
    def flight_seconds(self) -> Optional[float]:
        """How long the disc has been in the air, in seconds; None if it is held, lies on
        the ground, or has not been seen for a moment (then nothing says it still flies)."""
        if self.disc_state != "air" or self._frame is None or self._disc_seen is None:
            return None
        if self._frame - self._disc_seen > FLIGHT_UNSEEN_SECONDS * self.frame_rate:
            return None
        return max(0.0, (self._disc_seen - self.since) / self.frame_rate)

    def rename(self, player_id: int, new_player_id: int) -> None:
        """A player turned out to be another one known earlier; keep following them."""
        if self.holder_id == player_id:
            self.holder_id = new_player_id
        if player_id in self._seen_at:
            self._seen_at[new_player_id] = self._seen_at.pop(player_id)
            self._first_seen[new_player_id] = self._first_seen.pop(player_id)

    def move(self, camera_motion: np.ndarray) -> None:
        """Move what is remembered along with the picture (homography from the last frame)."""
        points = []
        if self._holder_box is not None:
            points += [self._holder_box[:2], self._holder_box[2:]]
        if self._resting_at is not None:
            points.append(self._resting_at)
        if self._last_sighting is not None:
            points.append(self._last_sighting[1])
        if not points:
            return
        moved = cv2.perspectiveTransform(
            np.array(points, dtype=np.float64).reshape(-1, 1, 2), camera_motion
        ).reshape(-1, 2)
        if self._holder_box is not None:
            self._holder_box = (*moved[0], *moved[1])
        if self._last_sighting is not None:
            self._last_sighting = (self._last_sighting[0], tuple(moved[-1]))
            moved = moved[:-1]
        if self._resting_at is not None:
            self._resting_at = tuple(moved[-1])

    def _frames(self, setting: str, default: float) -> int:
        seconds = float(get_setting(f"models.possession.{setting}", default))
        return max(1, round(seconds * self.frame_rate))

    def _follow_holder(self, tracks: List[Any]) -> None:
        """Note where the holder is and their team, or how long they have not been found."""
        if self.holder_id is None:
            self._holder_box = None
            return
        for track in tracks:
            if getattr(track, "track_id", None) == self.holder_id:
                self._holder_box = tuple(float(value) for value in track.to_ltrb())
                self._holder_unseen = 0
                # The team the holder was mostly seen as while they have the disc: the
                # tracker's view of a player's team wavers now and then (sun on a dark
                # shirt), and the disc has not changed teams when it does
                team = getattr(track, "team", None)
                if team is not None:
                    self._holder_teams[team] += self._step
                    self.team = int(self._holder_teams[1] > self._holder_teams[0])
                return
        # Not found. The tracker may have lost them and begun a new track where they
        # stand: a track never seen before that covers the place they were last seen at
        # is the holder under a new number. (One seen before is someone else, the mark
        # for one, whose track was missing for a frame.)
        if self._holder_box is not None:
            for track in tracks:
                if (
                    _is_player(track)
                    and track.track_id not in self._known_tracks
                    and _box_iou(self._holder_box, track.to_ltrb()) >= SAME_PLAYER_OVERLAP
                ):
                    self.rename(self.holder_id, track.track_id)
                    self._follow_holder(tracks)
                    return
        self._holder_unseen += self._step
        if self._holder_unseen > self._frames("holder_memory_seconds", 2.0):
            self._holder_box = None

    @property
    def holder_box(self) -> Optional[Box]:
        """Where the holder is: their box in this frame, or where they were last seen if
        they are not found (covered by the mark); None without a holder or a place."""
        return self._holder_box if self.holder_id is not None else None

    def _follow_disc(
        self, disc: Tuple[float, float], at_a_player: bool, player_height: float
    ) -> None:
        """Note whether what is seen is the disc of the game.

        The disc model also finds things that are no disc: a brick mark painted on the
        grass, a cone, a second disc beside the field. They lie still at no player,
        which is what a turnover looks like. The disc of the game was in a player's
        hands and got to where it is by flying there: from one sighting to the next it
        is no further than a disc flies in that time. Something that shows far from the
        last sighting, or after the disc was not seen for a while, is not known to be
        the disc until a player has it.
        """
        if at_a_player:
            self._followed_from_a_player = True
        elif self._last_sighting is None:
            self._followed_from_a_player = False
        else:
            seconds = (self._frame - self._last_sighting[0]) / self.frame_rate
            flown = float(np.hypot(*np.subtract(disc, self._last_sighting[1])))
            within_reach = MAX_DISC_SPEED * player_height * seconds + player_height
            if seconds > MAX_UNSEEN_SECONDS or flown > within_reach:
                self._followed_from_a_player = False
        self._last_sighting = (self._frame, disc)

    def _watch_ground(self, disc: Optional[Tuple[float, float]], tracks: List[Any]) -> None:
        """Follow a disc that is at no player (None: it is at one): does it lie still?"""
        if disc is None:
            self._resting_at = None
            self._off_the_ground()
            return
        heights = [t.to_ltrb()[3] - t.to_ltrb()[1] for t in tracks if _is_player(t)]
        radius = GROUND_RADIUS * float(np.median(heights)) if heights else GROUND_RADIUS_PIXELS
        moved = (
            float("inf")
            if self._resting_at is None
            else float(np.hypot(*np.subtract(disc, self._resting_at)))
        )
        if moved > (GROUND_LEFT * radius if self.on_ground else radius):
            self._resting_at = disc
            self._resting_since = self._frame
            self._off_the_ground()
        elif (
            not self.on_ground
            and self._followed_from_a_player
            and self._frame - self._resting_since >= self._frames("ground_seconds", 1.5)
        ):
            self.on_ground = True
            self.holder_id = None
            self.since = self._resting_since
            self._seen_at.clear()
            self._first_seen.clear()
            self._holder_box = None
            # Nobody throws the disc to the ground: the other team has it now. Once: a
            # disc that is seen to lie there again has not changed teams again.
            if self.team is not None and not self._turned_over:
                self.team = 1 - self.team
            self._turned_over = True

    def _off_the_ground(self) -> None:
        """The disc is not lying where it lay: picked up, or seen elsewhere."""
        if self.on_ground:
            self.on_ground = False
            self.since = self._frame

    def _frames_needed(self, at_disc: Optional[int], tracks: List[Any]) -> int:
        """How long the disc must be seen at a place before the holder changes to it."""
        needed = self._frames("confirm_seconds", 0.33)
        track = next((t for t in tracks if getattr(t, "track_id", None) == at_disc), None)
        if track is None:
            return needed
        if self._holder_box is not None and _overlap(self._holder_box, track.to_ltrb()):
            # Right by the holder: the mark, most likely
            needed = max(needed, self._frames("beside_holder_seconds", 1.5))
        # A defender at the catch is taken for the catcher easily, and an interception
        # is rare: a player of the team without the disc needs longer where a player of
        # the team with it stands right by. Alone with the disc (the pull, a pick-up
        # after a turnover the ground did not show) there is nobody to mistake them for.
        team = getattr(track, "team", None)
        if team is not None and self.team is not None and team != self.team:
            box = track.to_ltrb()
            if any(
                _is_player(other)
                and getattr(other, "team", None) == self.team
                and _overlap(box, other.to_ltrb())
                for other in tracks
            ):
                needed = max(needed, self._frames("other_team_seconds", 1.0))
        return needed

    def update(
        self,
        detections: List[Dict[str, Any]],
        tracks: List[Any],
        frame_index: Optional[int] = None,
    ) -> Optional[int]:
        """Take a frame's detections and tracks into account; returns the holder's track ID.

        Args:
            detections: The frame's detections; the discs among them are looked at
            tracks: The frame's tracks
            frame_index: The frame's number in the video, if frames may have been skipped
        """
        previous = self._frame
        if frame_index is not None:
            self._frame = frame_index
        else:
            self._frame = 0 if previous is None else previous + 1
        # A jump back or far ahead is a seek, after which the caller resets anyway
        skipped = 1 if previous is None else self._frame - previous
        self._step = min(max(1, skipped), max(1, round(self.frame_rate / 4)))
        self._follow_holder(tracks)
        self._known_tracks.update(track.track_id for track in tracks if _is_player(track))
        discs = [detection for detection in detections if detection.get("class_name") == "disc"]
        if not discs:
            return self.holder_id

        disc = max(discs, key=lambda detection: detection.get("confidence", 0.0))
        self._disc_seen = self._frame
        centre = _disc_centre(disc["bbox"])
        margin = float(get_setting("models.possession.box_margin", 0.15))
        at_disc = player_at_disc(disc["bbox"], tracks, margin)

        if self._holder_box is not None and at_disc != self.holder_id:
            reach = float(get_setting("models.possession.reach", 1.0))
            arm = reach * (self._holder_box[3] - self._holder_box[1])
            x1, y1, x2, y2 = self._holder_box
            if _contains(_widened(self._holder_box, margin), *centre):
                at_disc = self.holder_id  # Still in the holder's box, or where they were
            elif at_disc is None and _contains((x1 - arm, y1 - arm, x2 + arm, y2 + arm), *centre):
                at_disc = self.holder_id  # Held out at arm's length

        heights = [t.to_ltrb()[3] - t.to_ltrb()[1] for t in tracks if _is_player(t)]
        self._follow_disc(
            centre, at_disc is not None, float(np.median(heights)) if heights else 90.0
        )
        self._watch_ground(centre if at_disc is None else None, tracks)

        # Count where the disc is seen. With nobody holding it, a disc at nobody is a
        # place like any other; with a holder, a disc at the holder speaks against all.
        at_holder = at_disc == self.holder_id
        taken = AT_HOLDER if at_holder and self.holder_id is not None else ELSEWHERE
        for place in list(self._seen_at):
            if place != at_disc or at_holder:
                self._seen_at[place] -= taken * self._step
                if self._seen_at[place] <= 0:
                    del self._seen_at[place]
                    del self._first_seen[place]
        if at_holder:
            return self.holder_id
        self._seen_at[at_disc] = self._seen_at.get(at_disc, 0.0) + self._step
        self._first_seen.setdefault(at_disc, self._frame)

        if self._seen_at[at_disc] >= self._frames_needed(at_disc, tracks):
            self.holder_id = at_disc
            self._holder_teams = [0, 0]
            self.since = self._first_seen[at_disc]
            self._seen_at.clear()
            self._first_seen.clear()
            self._holder_box = None
            if at_disc is not None:
                self._turned_over = False
            self._follow_holder(tracks)
        return self.holder_id
