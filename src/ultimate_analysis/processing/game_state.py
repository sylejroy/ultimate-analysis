"""Game state: what is happening on the field (lined up, pull, live play, stoppage, ...).

The state is derived from what the other stages see, all of which can be missing in any
frame: where the tracked players stand, how they move, who holds the disc, and where the
end zones are. Every decision therefore needs its evidence to hold for some time, and no
evidence means the state stays as it is.

    UNKNOWN ──> LINED_UP ──> PULL ──> LIVE <──> STOPPAGE
                   ^                    │
                   └── BETWEEN_POINTS <─┘ (score, or the footage cuts to the next point)

- LINED_UP: at least one team stands side by side on its goal line.
- PULL: the line breaks up and runs downfield.
- LIVE: a receiver has the disc after the pull, or players are spread out and moving.
- STOPPAGE: play was live and now nearly everybody stands still (foul, pick, timeout).
- BETWEEN_POINTS: a catch in an end zone after which the players stop, or a cut.

Positions are in image pixels. Speeds are measured relative to the other players, which
removes camera pans, and in player heights per second, which makes near and far players
comparable.
"""

from collections import deque
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting

SPEED_WINDOW_S = 0.5  # Player speed is measured over this time
MIN_SPEED_HISTORY_S = 0.25
CUT_THUMBNAIL = (64, 36)


class GameState(str, Enum):
    UNKNOWN = "Unknown"
    BETWEEN_POINTS = "Between points"
    LINED_UP = "Lined up"
    PULL = "Pull"
    LIVE = "Live play"
    STOPPAGE = "Stoppage"


def _setting(name: str, default: float) -> float:
    return float(get_setting(f"models.game_state.{name}", default))


Player = Tuple[float, float, float]  # foot x, foot y, height in image pixels


def _largest_row(players: List[Player]) -> List[Player]:
    """The largest group of players standing side by side.

    Seen from behind or in front, that is a row of players of similar size with their feet
    at the same image height, spread out sideways.
    """
    best: List[Player] = []
    for _, anchor_y, anchor_height in players:
        row = [
            player
            for player in players
            if abs(player[1] - anchor_y) < 0.4 * anchor_height
            and 0.65 * anchor_height < player[2] < 1.5 * anchor_height
        ]
        xs = [player[0] for player in row]
        # Side by side, not bunched together
        if len(row) > len(best) and max(xs) - min(xs) >= len(row) * 0.5 * anchor_height:
            best = row
    return best


def line_up(players: List[Player]) -> Tuple[int, float]:
    """(players in the largest row, share of all players standing in the two largest rows).

    Before a pull each team stands in a row on its goal line and nobody stands between
    them, so nearly every player is in one of two rows. A stack during play also forms a
    row, but the handlers and their defenders stand elsewhere.
    """
    if not players:
        return 0, 0.0
    first = _largest_row(players)
    others = [player for player in players if player not in first]
    second = _largest_row(others)
    # The other team's row is at the far end of the field, not next to the first one
    apart = (
        first and second and abs(first[0][1] - second[0][1]) > 2 * min(first[0][2], second[0][2])
    )
    in_rows = len(first) + (len(second) if apart and len(second) >= 3 else 0)
    return len(first), in_rows / len(players)


def in_end_zone(field_results: List[Any], x: float, y: float, frame_shape: Tuple[int, int]) -> bool:
    """Whether an image point lies in a segmented end zone."""
    frame_h, frame_w = frame_shape
    for result in field_results:
        if getattr(result, "masks", None) is None:
            continue
        masks = np.asarray(result.masks.data)
        classes = np.asarray(
            result.boxes.cls.cpu() if hasattr(result.boxes.cls, "cpu") else result.boxes.cls
        )
        for mask, class_id in zip(masks, classes):
            if "endzone" not in str(result.names[int(class_id)]).lower():
                continue
            mask_h, mask_w = mask.shape
            column = min(mask_w - 1, max(0, int(x / frame_w * mask_w)))
            row = min(mask_h - 1, max(0, int(y / frame_h * mask_h)))
            if mask[row, column] > 0.5:
                return True
    return False


class GameStateTracker:
    """Follows the game state over the frames of a video."""

    def __init__(self):
        self.state = GameState.UNKNOWN
        self.reset()

    def reset(self) -> None:
        """Forget everything (new video, seek)."""
        self.state = GameState.UNKNOWN
        self._state_since = 0.0
        self._positions: Dict[int, Deque[Tuple[float, float, float]]] = {}
        self._held_since: Dict[str, Optional[float]] = {}
        self._thumbnail: Optional[np.ndarray] = None
        self._holder_id: Optional[int] = None
        self._end_zone_catch_at: Optional[float] = None
        self.features: Dict[str, float] = {}

    # ------------------------------------------------------------------ evidence

    def _is_cut(self, frame: np.ndarray) -> bool:
        """Whether the footage jumps to another scene at this frame (or is blacked out)."""
        thumbnail = cv2.cvtColor(cv2.resize(frame, CUT_THUMBNAIL), cv2.COLOR_BGR2GRAY).astype(
            np.int16
        )
        previous, self._thumbnail = self._thumbnail, thumbnail
        if thumbnail.mean() < 12:
            return True
        return previous is not None and float(np.abs(thumbnail - previous).mean()) > _setting(
            "cut_difference", 35
        )

    def _moving_share(self, time_s: float, players: Dict[int, Tuple[float, float, float]]) -> float:
        """Share of players moving relative to the others (-1 if it cannot be told yet)."""
        velocities, heights = [], []
        for track_id, (x, y, height) in players.items():
            history = self._positions.setdefault(track_id, deque())
            history.append((time_s, x, y))
            while history and time_s - history[0][0] > SPEED_WINDOW_S:
                history.popleft()
            elapsed = time_s - history[0][0]
            if elapsed >= MIN_SPEED_HISTORY_S:
                velocities.append(((x - history[0][1]) / elapsed, (y - history[0][2]) / elapsed))
                heights.append(height)
        for track_id in [track_id for track_id in self._positions if track_id not in players]:
            del self._positions[track_id]

        if len(velocities) < 4:
            return -1.0
        velocities = np.array(velocities)
        # A camera pan moves everybody the same way; what is left is the players' own motion
        relative = np.linalg.norm(velocities - np.median(velocities, axis=0), axis=1)
        speeds = relative / np.array(heights)
        return float((speeds > _setting("moving_speed", 0.6)).mean())

    def _held_for(self, name: str, condition: bool, time_s: float) -> float:
        """Seconds a condition has held without interruption (0 if it does not hold)."""
        if not condition:
            self._held_since[name] = None
            return 0.0
        if self._held_since.get(name) is None:
            self._held_since[name] = time_s
        return time_s - self._held_since[name]

    def _enter(self, state: GameState, time_s: float) -> None:
        if state != self.state:
            self.state = state
            self._state_since = time_s
            self._end_zone_catch_at = None

    # ------------------------------------------------------------------ per frame

    def update(
        self,
        time_s: float,
        frame: np.ndarray,
        tracks: List[Any],
        holder_id: Optional[int],
        field_results: List[Any],
    ) -> GameState:
        """Take a frame's results into account; returns the game state.

        Args:
            time_s: Position of the frame in the video in seconds
            frame: The video frame, to notice cuts
            tracks: Tracked objects of the frame
            holder_id: Track ID of the player holding the disc, if any
            field_results: Field segmentation results, to place a catch in an end zone
        """
        if self._is_cut(frame):
            # The tracks before and after a cut have nothing to do with each other.
            # Edited footage cuts from a score straight to the next line-up.
            was_playing = self.state in (GameState.PULL, GameState.LIVE, GameState.STOPPAGE)
            self._positions.clear()
            self._held_since.clear()
            self._holder_id = None
            self._enter(GameState.BETWEEN_POINTS if was_playing else self.state, time_s)
            return self.state

        players = {}
        for track in tracks:
            if getattr(track, "class_name", None) == "player":
                x1, y1, x2, y2 = track.to_ltrb()
                if y2 > y1:
                    players[track.track_id] = ((x1 + x2) / 2, y2, y2 - y1)

        count = len(players)
        enough = count >= int(_setting("min_players", 6))
        moving = self._moving_share(time_s, players)
        in_line, in_rows = line_up(list(players.values()))
        lined = (
            in_line >= int(_setting("line_players", 5))
            and in_rows >= _setting("line_share", 0.8)
            and 0 <= moving < 0.4
        )
        self.features = {"players": count, "in_line": in_line, "in_rows": in_rows, "moving": moving}

        lined_for = self._held_for("lined", lined, time_s)
        # The line-up ends when the players start running, still in their rows at first
        dispersed_for = self._held_for("dispersed", enough and not lined, time_s)
        active_for = self._held_for("active", enough and not lined and moving >= 0.3, time_s)
        still_for = self._held_for("still", enough and not lined and 0 <= moving < 0.12, time_s)
        few_for = self._held_for("few", count < 4, time_s)
        in_state = time_s - self._state_since

        # A catch in an end zone is a score if the players then stop
        if holder_id is not None and holder_id != self._holder_id and holder_id in players:
            x, y, _ = players[holder_id]
            if self.state == GameState.LIVE and in_end_zone(field_results, x, y, frame.shape[:2]):
                self._end_zone_catch_at = time_s
        self._holder_id = holder_id
        catch_at = self._end_zone_catch_at
        if catch_at is not None and time_s - catch_at > _setting("score_window_s", 6):
            self._end_zone_catch_at = catch_at = None

        state = self.state
        if state == GameState.LINED_UP:
            if dispersed_for >= _setting("pull_start_s", 0.5):
                self._enter(GameState.PULL, time_s)
        elif (
            lined_for
            >= (2.0 if state in (GameState.LIVE, GameState.STOPPAGE) else 1.0)
            * _setting("line_up_s", 1.0)
            and state != GameState.PULL
        ):
            self._enter(GameState.LINED_UP, time_s)
        elif state == GameState.PULL:
            caught = holder_id is not None and in_state >= _setting("pull_min_s", 3)
            if caught or in_state >= _setting("pull_max_s", 10):
                self._enter(GameState.LIVE, time_s)
        elif state == GameState.LIVE:
            if catch_at is not None and 0 <= moving < 0.2 and time_s - catch_at >= 2.0:
                self._enter(GameState.BETWEEN_POINTS, time_s)
            elif still_for >= _setting("stoppage_s", 4):
                self._enter(GameState.STOPPAGE, time_s)
        elif state == GameState.STOPPAGE:
            if active_for >= 1.0:
                self._enter(GameState.LIVE, time_s)
        elif active_for >= _setting("live_s", 3):  # UNKNOWN or BETWEEN_POINTS, joined mid-play
            self._enter(GameState.LIVE, time_s)

        if few_for >= _setting("unknown_s", 5) and self.state != GameState.BETWEEN_POINTS:
            self._enter(GameState.UNKNOWN, time_s)
        return self.state
