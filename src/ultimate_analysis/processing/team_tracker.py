"""Tracking players by where they are heading, within their team.

ByteTrack follows each box by its motion: a track continues with the detection its box
was moving towards. Two players who cross keep their tracks as long as they move
differently. What it cannot know is who is who once two boxes have covered each other;
there a track may carry on with the wrong player.

In Ultimate most of these meetings are between a player and the opponent marking them,
and the two teams wear different colours. The tracker here learns the two shirt colours
of the game while it runs, gives each track the team its player's shirt mostly looked
like, and never continues a track with a detection that clearly wears the other colour.
It also matches by where the feet are, which tells apart two players at different depths
whose boxes cover each other.
A track that ended up on the wrong player is thereby cut off when the two come apart;
reconnecting the player with their own track is left to the identity layer
(player_identity.py), which knows where they can have got to.

Swaps between teammates are not prevented by this.

The same colours tell the observers from the players: a track whose shirt is orange or
red nearly every time it is seen, without that being a team colour, can be left out.
"""

from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker, STrack

from . import appearance

# A detection is "clear" if no other box covers more of it than this (IoU). Only clear
# detections tell a shirt colour, and only they are held against a track's team.
CLEAR_OVERLAP = 0.35
# Shirt colours of boxes that stand alone are collected to learn the two team colours
SAMPLE_OVERLAP = 0.05
SAMPLE_MIN_CONFIDENCE = 0.5
MIN_SAMPLES = 60
MAX_SAMPLES = 3000
# The colour a team is shown in is the average of this many of its shirts at least, and
# is held from that many on
MIN_SHOWN_COLOUR_SAMPLES = 20
SHOWN_COLOUR_SAMPLES = 400
REFIT_FRAMES = 30
# Two colour clusters closer than this (Lab units) are not two teams
MIN_TEAM_DISTANCE = 40.0
# A shirt belongs to a team if it is nearer to its colour than to the other by this share
# of the distance between the two
MIN_COLOUR_MARGIN = 0.3
# Two team colours this far apart in colour alone (Lab a and b, without lightness) are
# told apart by colour; lightness then counts by this much only. Sun on a dark shirt
# makes it lighter, not another colour.
# In the four drone games the teams are no further apart in colour than 13 (dark green
# against white: the middle of a box has grass in it), so there lightness decides.
# Tried on them and not done: lightness counting less everywhere (a sighting says
# another team than the rest of its track on 20 to 26% with colour alone, against 1 to
# 3% as it is), and learning per game how shirts vary within a team (fewer such
# sightings on crops, 3.3% -> 2.3% for green against white, but in the running tracker,
# which goes by many sightings, a player's team changed no less often: 0, 0, 1 and 2
# times in four stretches before, 2, 0, 0 and 2 with it).
MIN_COLOUR_GAP = 15.0
LIGHTNESS_WEIGHT = 0.3
# How much the place of the feet counts in matching a detection to a track, next to how
# much the boxes overlap. Two players whose boxes cover each other mostly stand at
# different depths: their feet are apart when their boxes are not.
FEET_WEIGHT = 0.5
# A track has a team once this many clear sightings, and this share of them, agree
MIN_VOTES = 3
MIN_VOTE_SHARE = 0.75
# Observers wear orange or red. A track is one if nearly all its clear sightings show a
# shirt that is (1) orange or red: in Lab, where 128 is grey, clearly on the red and on
# the yellow side, and (2) not a team colour: the two team colours differ mostly in how
# light they are, and light and shadow move a shirt along the line between them, so a
# team in red lies on that line and an observer far off it. Without (1), players of teams
# in purple or green were taken for outsiders; the five observers in the 21 clips had
# shirts of (a, b) between (138, 152) and (177, 164) and were 22 to 54 off the line.
OBSERVER_MIN_RED = 134
OBSERVER_MIN_YELLOW = 146
OUTSIDER_DISTANCE = 18.0
OUTSIDER_MIN_SIGHTINGS = 20
OUTSIDER_SHARE = 0.85
# A single frame tells the two team colours if it shows at least this many shirts
MIN_SHIRTS_IN_FRAME = 8


def team_colours_of(shirts: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Shirt colours split into two groups: the two team colours, or None if they are one."""
    samples = np.array(shirts, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    # The same samples must give the same teams: fixed starting points
    cv2.setRNGSeed(0)
    _, _, colours = cv2.kmeans(samples, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    if np.linalg.norm(colours[0] - colours[1]) < MIN_TEAM_DISTANCE:
        return None
    return colours


def off_team_colours(
    shirt: Optional[np.ndarray], team_colours: Optional[np.ndarray]
) -> Optional[bool]:
    """Whether a shirt is orange or red without that being a team colour."""
    if team_colours is None or shirt is None:
        return None
    if shirt[1] < OBSERVER_MIN_RED or shirt[2] < OBSERVER_MIN_YELLOW:
        return False
    along = team_colours[1] - team_colours[0]
    along = along / np.linalg.norm(along)
    offset = shirt - team_colours[0]
    return bool(np.linalg.norm(offset - (offset @ along) * along) > OUTSIDER_DISTANCE)


def observers_in_frame(frame: np.ndarray, boxes: Sequence[Sequence[float]]) -> List[bool]:
    """Which of the player boxes of one frame show an observer rather than a player.

    The tracker decides this over many frames; here one frame must do. Its shirts give
    the two team colours, and a box counts as an observer's if its shirt is orange or
    red without that being a team colour. With too few shirts to tell the teams, nobody
    is taken for an observer.
    """
    kits = appearance.encode(frame, boxes)
    shirts = [None if kit is None else kit[:3] for kit in kits]
    seen = [shirt for shirt in shirts if shirt is not None]
    if len(seen) < MIN_SHIRTS_IN_FRAME:
        return [False] * len(shirts)
    team_colours = team_colours_of(seen)
    return [bool(off_team_colours(shirt, team_colours)) for shirt in shirts]


class DetectionBoxes:
    """Detections in the form the Ultralytics trackers take them."""

    def __init__(self, xyxy: Any, conf: Any):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        box = self.xyxy
        return np.column_stack(
            [
                (box[:, 0] + box[:, 2]) / 2,
                (box[:, 1] + box[:, 3]) / 2,
                box[:, 2] - box[:, 0],
                box[:, 3] - box[:, 1],
            ]
        )

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> "DetectionBoxes":
        return DetectionBoxes(self.xyxy[index], self.conf[index])


def tracker_settings(track_buffer: int) -> SimpleNamespace:
    """Ultralytics' ByteTrack settings, with our time a lost track is kept (frames)."""
    return SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=track_buffer,
        match_thresh=0.8,
        fuse_score=True,
    )


def largest_overlaps(boxes: np.ndarray) -> np.ndarray:
    """For each box (x1, y1, x2, y2) the largest IoU it has with any other of the boxes."""
    if len(boxes) < 2:
        return np.zeros(len(boxes))
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    shared = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    overlap = shared / (area[:, None] + area[None, :] - shared + 1e-9)
    np.fill_diagonal(overlap, 0.0)
    return overlap.max(axis=1)


def feet_distances(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """How far the feet of each detection are from where each track expects its player's.

    The feet are the bottom centre of a box (x1, y1, x2, y2). The distance is given in
    body heights, and 1 from one body height on.
    """
    track_feet = np.column_stack([(tracks[:, 0] + tracks[:, 2]) / 2, tracks[:, 3]])
    detection_feet = np.column_stack([(detections[:, 0] + detections[:, 2]) / 2, detections[:, 3]])
    apart = np.linalg.norm(track_feet[:, None, :] - detection_feet[None, :, :], axis=2)
    height = np.maximum(
        (tracks[:, 3] - tracks[:, 1])[:, None], (detections[:, 3] - detections[:, 1])[None, :]
    )
    return np.clip(apart / (height + 1e-9), 0.0, 1.0)


class TeamTrack(STrack):
    """A track that keeps count of which team its player's shirt looked like."""

    clear = False  # As a detection: no other box covers it
    shirt_team: Optional[int] = None  # As a detection: the team its shirt colour says
    off_colours: Optional[bool] = None  # As a detection: an observer's orange or red shirt
    team: Optional[int] = None  # As a track: the team most of its clear sightings said
    outsider = False  # As a track: an observer's shirt nearly every time

    def _count(self, detection: "TeamTrack") -> None:
        if not hasattr(self, "votes"):
            self.votes = [0, 0]
            self.colour_sightings = [0, 0]  # All with a known shirt colour, those off the teams'
        if detection.clear and detection.off_colours is not None:
            self.colour_sightings[0] += 1
            self.colour_sightings[1] += detection.off_colours
            seen, off = self.colour_sightings
            self.outsider = seen >= OUTSIDER_MIN_SIGHTINGS and off >= OUTSIDER_SHARE * seen
        if detection.clear and detection.shirt_team is not None:
            self.votes[detection.shirt_team] += 1
            best = int(self.votes[1] > self.votes[0])
            agreed = self.votes[best] >= max(MIN_VOTES, MIN_VOTE_SHARE * sum(self.votes))
            self.team = best if agreed else None

    def activate(self, kalman_filter: Any, frame_id: int) -> None:
        super().activate(kalman_filter, frame_id)
        self._count(self)

    def update(self, new_track: "TeamTrack", frame_id: int) -> None:
        super().update(new_track, frame_id)
        self._count(new_track)

    def re_activate(self, new_track: "TeamTrack", frame_id: int, new_id: bool = False) -> None:
        super().re_activate(new_track, frame_id, new_id)
        self._count(new_track)


class TeamTracker(BYTETracker):
    """ByteTrack that keeps every track within its team."""

    track_class = TeamTrack

    def __init__(self, args: Any):
        super().__init__(args)
        self._shirt_samples: List[np.ndarray] = []
        self.team_colours: Optional[np.ndarray] = None  # (2, 3) shirt colours in Lab
        # Per team: the colours of its shirts as far as looked at (BGR)
        self._shown_shirts: List[List[np.ndarray]] = [[], []]
        # ... and their middle value, with the number of shirts it was taken over
        self._shown_middle: List[Optional[Tuple[int, np.ndarray]]] = [None, None]
        self._frames_since_fit = 0

    def update(self, results: Any, img: Optional[np.ndarray] = None, *args, **kwargs):
        self._frames_since_fit += 1
        if len(self._shirt_samples) >= MIN_SAMPLES and (
            self.team_colours is None or self._frames_since_fit >= REFIT_FRAMES
        ):
            self._learn_team_colours()
        return super().update(results, img, *args, **kwargs)

    def _learn_team_colours(self) -> None:
        """Split the shirt colours seen so far into two groups."""
        self._frames_since_fit = 0
        del self._shirt_samples[:-MAX_SAMPLES]
        colours = team_colours_of(self._shirt_samples)
        if colours is None:
            self.team_colours = None
            return
        # Team 0 stays team 0: the tracks have counted their sightings by these numbers
        if self.team_colours is not None and np.linalg.norm(
            colours[0] - self.team_colours[0]
        ) > np.linalg.norm(colours[1] - self.team_colours[0]):
            colours = colours[::-1].copy()
        self.team_colours = colours

    def _team_of(self, shirt: Optional[np.ndarray]) -> Optional[int]:
        """The team a shirt colour clearly belongs to, if any."""
        if self.team_colours is None or shirt is None:
            return None
        between = self.team_colours[0] - self.team_colours[1]
        by_colour = float(np.hypot(between[1], between[2])) >= MIN_COLOUR_GAP
        weights = np.array([LIGHTNESS_WEIGHT if by_colour else 1.0, 1.0, 1.0])
        distance = np.linalg.norm((self.team_colours - shirt) * weights, axis=1)
        apart = np.linalg.norm(between * weights)
        if abs(distance[0] - distance[1]) <= MIN_COLOUR_MARGIN * apart:
            return None
        return int(distance[1] < distance[0])

    def _off_team_colours(self, shirt: Optional[np.ndarray]) -> Optional[bool]:
        """Whether a shirt is orange or red without that being a team colour."""
        return off_team_colours(shirt, self.team_colours)

    def outsiders(self) -> set:
        """IDs of the tracks that are followed but are no player of either team."""
        return {
            int(track.track_id)
            for track in self.tracked_stracks + self.lost_stracks
            if getattr(track, "outsider", False)
        }

    def shirt_colours(self) -> dict:
        """{team (0 or 1): its average shirt colour (BGR)}; empty until the teams are known.

        For showing a team by its colour: the middle value over its players of the
        middle value of each shirt (the colours the tracker matches by are means, dulled
        by the grass beside a player, and are fitted again and again). Twice the middle
        value: within a shirt against the number and the grass at its edges, over the
        shirts against the boxes that show mostly grass, of a player far away or bent
        over. It is held once enough shirts have been seen, so a team does not change
        colour on the screen.
        Until a team has a few shirts, it is the colour the tracker matches by.
        """
        if self.team_colours is None:
            return {}
        lab = np.clip(self.team_colours, 0, 255).astype(np.uint8).reshape(1, 2, 3)
        matched = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0]
        colours = {}
        for team in (0, 1):
            shirts = self._shown_shirts[team]
            colour = matched[team]
            if len(shirts) >= MIN_SHOWN_COLOUR_SAMPLES:
                # Asked for several times a frame, and new only with a new shirt
                middle = self._shown_middle[team]
                if middle is None or middle[0] != len(shirts):
                    middle = self._shown_middle[team] = (len(shirts), np.median(shirts, axis=0))
                colour = middle[1]
            colours[team] = tuple(int(round(value)) for value in colour)
        return colours

    def teams_of_tracks(self) -> dict:
        """{track ID: team (0 or 1)} for the tracks whose team is known."""
        return {
            int(track.track_id): track.team
            for track in self.tracked_stracks
            if getattr(track, "team", None) is not None
        }

    def leanings_of_tracks(self) -> dict:
        """{track ID: team (0 or 1)} as far as anything says: the team of a track that
        has one, and for a young track the team its few clear sightings lean to."""
        leanings = {}
        for track in self.tracked_stracks:
            team, votes = getattr(track, "team", None), getattr(track, "votes", (0, 0))
            if team is None and votes[0] != votes[1]:
                team = int(votes[1] > votes[0])
            if team is not None:
                leanings[int(track.track_id)] = team
        return leanings

    def init_track(self, results: Any, img: Optional[np.ndarray] = None) -> List[STrack]:
        detections = super().init_track(results, img)
        if img is None or not detections:
            return detections
        overlaps = largest_overlaps(results.xyxy)
        kits = appearance.encode(img, results.xyxy)
        for detection, kit, overlap, confidence in zip(detections, kits, overlaps, results.conf):
            shirt = None if kit is None else kit[:3]
            detection.clear = bool(overlap < CLEAR_OVERLAP)
            detection.shirt_team = self._team_of(shirt)
            detection.off_colours = self._off_team_colours(shirt)
            if (
                shirt is not None
                and overlap < SAMPLE_OVERLAP
                and confidence >= SAMPLE_MIN_CONFIDENCE
            ):
                self._shirt_samples.append(shirt)
                team = detection.shirt_team
                if team is not None and len(self._shown_shirts[team]) < SHOWN_COLOUR_SAMPLES:
                    shown = appearance.shirt_colour(img, detection.xyxy)
                    if shown is not None:
                        self._shown_shirts[team].append(shown)
        return detections

    def get_dists(self, tracks: List[STrack], detections: List[STrack]) -> np.ndarray:
        distances = super().get_dists(tracks, detections)
        if len(tracks) and len(detections):
            distances = (1 - FEET_WEIGHT) * distances + FEET_WEIGHT * feet_distances(
                np.array([track.xyxy for track in tracks], dtype=np.float64),
                np.array([detection.xyxy for detection in detections], dtype=np.float64),
            )
        for row, track in enumerate(tracks):
            team = getattr(track, "team", None)
            if team is None:
                continue
            for column, detection in enumerate(detections):
                other = getattr(detection, "shirt_team", None)
                if getattr(detection, "clear", False) and other is not None and other != team:
                    distances[row, column] = 1.0  # Never matched
        return distances

    def reset(self) -> None:
        super().reset()
        self._shirt_samples = []
        self.team_colours = None
        self._shown_shirts: List[List[np.ndarray]] = [[], []]
        # ... and their middle value, with the number of shirts it was taken over
        self._shown_middle: List[Optional[Tuple[int, np.ndarray]]] = [None, None]
        self._frames_since_fit = 0
