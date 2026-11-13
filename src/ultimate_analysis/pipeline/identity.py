"""Who the tracked players are: their jersey numbers, their looks, their teams.

The tracker follows boxes; this stage says which player a box is. It reads jersey
numbers (PlayerNumbers), knows players by their looks (PlayerLooks, with the roster of
the video), puts players in the team their looks say, joins tracks that turn out to be
one player, and exchanges two tracks whose players changed places in a crossing.

What is kept per box and not per player (who has the disc, who stands beside the field,
where each stands) belongs to the pipeline: a rename or an exchange is passed on to it.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config.settings import get_setting
from ..processing import appearance
from ..processing.jersey_tracker import get_best_jersey_number
from ..processing.player_looks import PlayerLooks
from ..processing.player_numbers import PlayerNumbers
from ..processing.tracking import PlayerTracker
from ..rendering.tracks import team_display_colour
from ..utils.logger import get_logger

logger = get_logger("PIPELINE")

# The colour a team is shown in, by the teams the players' looks give: a shirt is taken
# every so many frames, a colour is given from so many shirts, and no more are taken
# after so many, so that a team does not change colour on the screen
SHIRT_EVERY_FRAMES = 6
MIN_SHIRTS_PER_TEAM = 20
SHIRTS_PER_TEAM = 300


def _players(tracks: List[Any]) -> List[Any]:
    return [track for track in tracks if track.class_name == "player"]


class IdentityStage:
    """Jersey numbers, looks and teams of the tracked players."""

    def __init__(
        self,
        tracker: PlayerTracker,
        rename_boxes: Callable[[int, int], None],
        exchange_boxes: Callable[[int, int], None],
    ):
        """
        Args:
            tracker: The pipeline's tracker, whose players these are
            rename_boxes: Passes on that a track is known under another ID from now on
            exchange_boxes: Passes on that two tracks exchanged their IDs
        """
        self._tracker = tracker
        self._rename_boxes = rename_boxes
        self._exchange_boxes = exchange_boxes
        self.looks = PlayerLooks()
        # Jersey numbers persist across frames: only some players are read per frame
        self.numbers = PlayerNumbers(tracker)
        self.numbers.could_be = self.looks.could_be
        # Track -> jersey number of the player a track without a number looks like
        self.numbers_by_looks: Dict[int, str] = {}
        self.frame_rate = 30.0
        self._joined_at = 0  # The frame the tracks were last looked back over
        # The shirts of the players of each team as their looks give it, the colours
        # made of them, and a count of frames
        self._shirts: List[list] = [[], []]
        self._colours: Optional[dict] = None
        self._shirt_turn = 0

    @property
    def player_ids(self) -> Dict[int, Tuple[str, Any]]:
        """Player ID -> (jersey number or "Unknown", details of the reading)."""
        return self.numbers.numbers

    def new_video(self, video_path: Optional[str]) -> None:
        self.looks.new_video(video_path)

    def reset(self) -> None:
        """A seek or a cut: the tracks are over, and the tracker numbers its teams anew."""
        self.looks.cut()
        self.numbers.reset()
        self._forget_colours()

    def _forget_colours(self) -> None:
        self._shirts, self._colours, self._shirt_turn = [[], []], None, 0

    def set_looks_model(self, weights: Optional[str]) -> bool:
        """Know the players by another network; whether it changed."""
        if not self.looks.set_weights(weights):
            return False
        self.numbers_by_looks = {}
        self._forget_colours()
        return True

    # ------------------------------------------------------------------ per frame

    def teams_by_looks(self, frame: np.ndarray, tracks: List[Any]) -> None:
        """Put the players in the team their looks say, where those say another than
        the colour of the shirt area does: a kit is not always one colour. And take
        the colour a team is shown in from the shirts of those the looks put in it."""
        if not get_setting("models.reid.teams_by_looks", True):
            return
        players = _players(tracks)
        by_looks = self.looks.teams(
            {track.track_id: getattr(track, "team", None) for track in players}
        )
        if not by_looks:
            return
        for track in players:
            team = by_looks.get(track.track_id)
            if team is not None and team != getattr(track, "team", None):
                track.team = team
                self._tracker.set_team_of_player(track.track_id, team)
        # A shirt every few frames, of players who stand clear of the others, until
        # there are enough: after that a team's colour stays as it is
        self._shirt_turn += 1
        if self._shirt_turn % SHIRT_EVERY_FRAMES == 0:
            boxes = [track.to_ltrb() for track in players]
            touched = appearance.box_overlaps(boxes).max(axis=1) > 0 if boxes else []
            for track, box, is_touched in zip(players, boxes, touched):
                team = by_looks.get(track.track_id)
                if team is None or len(self._shirts[team]) >= SHIRTS_PER_TEAM or is_touched:
                    continue
                shirt = appearance.shirt_colour(frame, box)
                if shirt is not None:
                    self._shirts[team].append(shirt)
                    self._colours = None
        colours = self.team_colours()
        for track in players:
            if getattr(track, "team", None) is not None:
                track.team_colour = colours.get(track.team)

    def team_colours(self) -> Dict[int, Tuple[int, int, int]]:
        """{team: the colour of its shirts (BGR)}: of the shirts of those the looks put
        in the team once there are a few, and as the tracker has it until then."""
        if self._colours is None:
            colours = dict(self._tracker.team_shirt_colours())
            for team, shirts in enumerate(self._shirts):
                if len(shirts) >= MIN_SHIRTS_PER_TEAM:
                    colours[team] = tuple(int(round(v)) for v in np.median(shirts, axis=0))
            if all(len(shirts) < MIN_SHIRTS_PER_TEAM for shirts in self._shirts):
                return colours  # The tracker's, which it may still be changing
            self._colours = appearance.told_apart(colours)
        return self._colours

    def read_numbers(self, frame: np.ndarray, tracks: List[Any], frame_index: int) -> dict:
        """Read the jersey numbers of some players; returns how long that took."""
        timing, renamed = self.numbers.update(frame, tracks, frame_index)
        for was, now in renamed:
            self.rename(was, now)
        return timing

    def look(
        self,
        frame: np.ndarray,
        tracks: List[Any],
        frame_index: int,
        holder_id: Optional[int],
        places_on_field: Dict[int, np.ndarray],
        places: Dict[int, Tuple[float, float]],
        in_play: bool = True,
    ) -> None:
        """Look at the players: name the ones whose number is not read by their looks,
        join tracks that turn out to be one player, and settle crossings. `in_play`:
        False between points, lined up or after a score (no passes counted then)."""
        self.numbers_by_looks = self.looks.update(
            frame,
            tracks,
            frame_index,
            self.certain_numbers(tracks),
            holder_id,
            places_on_field,
            places,
            in_play,
        )
        self._join_looking_back(tracks, frame_index)
        self._settle_crossings(tracks, frame_index)

    def rename(self, was: int, now: int) -> None:
        """A track is known under another ID from now on: everything kept of it goes along."""
        self._rename_boxes(was, now)
        self.looks.rename(was, now)
        if was in self.numbers_by_looks:
            self.numbers_by_looks[now] = self.numbers_by_looks.pop(was)

    def _settle_crossings(self, tracks: List[Any], frame_index: int) -> None:
        """Exchange the players of two tracks that changed them while their boxes
        overlapped, as their looks before and after say."""
        players = {t.track_id: t for t in _players(tracks)}
        for first, second in self.looks.swapped(list(players), frame_index):
            if not self._tracker.swap_players(first, second):
                continue
            one, other = players[first], players[second]
            one.track_id, other.track_id = other.track_id, one.track_id
            one.team, other.team = other.team, one.team
            one.team_colour, other.team_colour = other.team_colour, one.team_colour
            # What belongs to the box goes with the box: who has the disc, where they
            # stand on the field and beside it, how sharp and how far they ran
            self._exchange_boxes(first, second)
            self.looks.swap(first, second)
            logger.info(f"Players {first} and {second} had changed tracks in a crossing")

    def _join_looking_back(self, tracks: List[Any], frame_index: int) -> None:
        """Give a player the tracker lost and began again their ID back, where the
        tracks since the last cut, looked back over, turn out to be one player
        (tracklet_linking.py: never in the picture at once, of one team, within a
        player's reach of each other on the field, and clearly most alike).

        Every so often (models.reid.join_looking_back_seconds, 0: never), and only into
        a player the tracker still remembers and does not see.
        """
        every = float(get_setting("models.reid.join_looking_back_seconds", 1.0))
        # A frame before the last joining is a seek back: joined at once, not only once
        # playback is past where it was. (Not reset at a cut: joining at once after a
        # cut joins on too little, measured on Chicago v San Francisco)
        if every <= 0 or 0 <= frame_index - self._joined_at < every * self.frame_rate:
            return
        self._joined_at = frame_index
        players = {t.track_id: t for t in _players(tracks)}
        naming = self.looks.players_of_tracks()
        missing = set(self._tracker.missing_players(set(players)))
        for was, track in list(players.items()):
            now = naming.get(was, was)
            if now == was or now in players or now not in missing:
                continue
            self.numbers.join(track, now)
            self.rename(was, now)
            players[now] = track
            missing.discard(now)
            logger.info(f"Player {was} is player {now} again (looking back over the tracks)")

    # ------------------------------------------------------------------ for showing

    def certain_numbers(self, tracks: List[Any]) -> Dict[int, str]:
        """Track -> jersey number, for the players whose number is read often enough to
        name them by."""
        needed = float(get_setting("models.tracking.identity.number_certainty", 0.6))
        certain = {}
        for track in _players(tracks):
            if track.track_id in self.player_ids:
                number, certainty = get_best_jersey_number(track.track_id)
                # Not a player shown without it, or going by another: the votes may be
                # of whoever the tracker swapped them with (player_numbers.py)
                shown = self.player_ids[track.track_id][0]
                if number and certainty >= needed and str(shown) == str(number):
                    certain[track.track_id] = str(number)
        return certain

    def shown_numbers(self) -> Dict[int, Tuple[str, Any]]:
        """The numbers to draw: those read, and for the others the number of the player
        they look like, marked with a tilde."""
        shown = dict(self.player_ids)
        for track_id, number in self.numbers_by_looks.items():
            if not str(shown.get(track_id, ("", None))[0]).isdigit():
                shown[track_id] = (f"~{number}", None)
        return shown

    def players_of_tracks(self) -> Dict[int, int]:
        """Track -> player, looking back over the tracks since the last cut."""
        return self.looks.players_of_tracks()

    def player_stats(self) -> List[dict]:
        """What is known of each player of the video and what they did: a row each
        (PlayerRoster.table), with the colour their team is drawn in."""
        rows = self.looks.roster.table()
        for row in rows:
            shirt = row.pop("team_colour")
            row["colour"] = None if shirt is None else team_display_colour(shirt)
        return rows
