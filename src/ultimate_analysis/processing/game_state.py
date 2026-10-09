"""Which phase a game is in: between points, lined up, the pull, live play, a score.

A game of Ultimate goes round in a circle. Both teams line up on their goal lines; one
pulls and runs down the field while the disc is in the air; the point is played until
someone catches the disc in the end zone they attack; everyone walks back, and the teams
line up again, the scorers where they scored. Each phase looks different from above:

- lined up: nearly every player of one team stands at one end of the field and nearly
  every player of the other at the other end
- the pull: the line breaks: the disc is in the air, or players run out of their end
- live: the teams are mixed over the field
- a score: a player holds the disc in the end zone their team attacks (which end that
  is, the line-up has said), or, where no line-up was seen, holds it in an end zone
  while everybody slows down
- between points: none of these; players walk

What is seen is rough (places on the field are off by yards, the disc is found in half
the frames, a player's team is not always known), so every change of phase must be seen
for a while before it is taken, and the order of the circle counts: a score comes out of
live play, a pull out of a line-up.

The state follows from what the pipeline already works out per frame; nothing here looks
at a picture.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

UNKNOWN, BETWEEN_POINTS, LINED_UP, PULL, LIVE, SCORE = (
    "unknown",
    "between points",
    "lined up",
    "pull",
    "live",
    "score",
)
STATES = (UNKNOWN, BETWEEN_POINTS, LINED_UP, PULL, LIVE, SCORE)

# A team is at an end of the field if this many of its players, and this share of them,
# stand no further than this beyond the goal line (field units; places are rough)
MIN_PLAYERS_AT_AN_END = 4
SHARE_AT_AN_END = 0.7
LINE_REACH = 8.0
# Seconds something must be seen for before the state changes on it
LINE_UP_SECONDS = 1.0
LINE_BROKEN_SECONDS = 0.5
SCORE_SECONDS = 0.5
# A line-up seen again this soon after a pull means the pull was none
FALSE_START_SECONDS = 6.0
# A pull is over when someone has the disc, at the earliest and at the latest after this
MIN_PULL_SECONDS = 2.0
MAX_PULL_SECONDS = 12.0
# After a score the state stays "score" this long, then it is between points
SCORE_SHOWN_SECONDS = 4.0
# Without the field for this long, nothing is known
LOST_SECONDS = 2.0
# Speeds, in field units per second: walking pace, and clearly running
WALKING, RUNNING = 2.0, 3.5
# A player's speed is taken over this long: places wobble from frame to frame
SPEED_SECONDS = 0.6
# Without a line-up seen: holding the disc in an end zone while everyone slows down for
# this long is a score
SLOW_SCORE_SECONDS = 2.5


@dataclass
class Player:
    """A player as the state model sees them."""

    player: int
    team: Optional[int]  # 0 or 1, if the tracker knows
    x: float  # Across the field
    y: float  # Along the field


@dataclass
class GameEvent:
    seconds: float
    kind: str  # "pull" or "score"
    team: Optional[int] = None  # Who scored


@dataclass
class _Seen:
    """Whether something has been seen without a break, and since when."""

    since: Optional[float] = None

    def update(self, seen: bool, seconds: float) -> float:
        """How long it has been seen for, 0 if it is not seen now."""
        if not seen:
            self.since = None
            return 0.0
        if self.since is None:
            self.since = seconds
        return seconds - self.since


@dataclass
class GameStateTracker:
    """Follows the phase of a game from frame to frame."""

    length: float = 110.0  # Of the field, back line to back line
    end_zone: float = 20.0
    state: str = UNKNOWN
    since: float = 0.0  # When the state was entered
    events: List[GameEvent] = field(default_factory=list)
    # Team -> +1 if it attacks the far end zone (y grows), -1 the near one; from the
    # last line-up
    attacks: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._places: Dict[int, Deque[Tuple[float, float, float]]] = {}
        self._lined_up, self._line_broken = _Seen(), _Seen()
        self._scoring, self._slow_in_end_zone = _Seen(), _Seen()
        self._lost, self._playing, self._walking = _Seen(), _Seen(), _Seen()

    def reset(self) -> None:
        """Start again (new video, seek); the events so far are forgotten."""
        self.state, self.since, self.events, self.attacks = UNKNOWN, 0.0, [], {}
        self.__post_init__()

    def cut(self) -> None:
        """The view has changed and the tracker has started again: where the game stands
        must be found anew, and the tracker counts its teams anew, so which end each
        attacks is no longer known. The events so far are kept."""
        self.state, self.attacks = UNKNOWN, {}
        self.__post_init__()

    # ------------------------------------------------------------------ what is seen

    def _speeds(self, players: Sequence[Player], seconds: float) -> Dict[int, float]:
        """Each player's speed over the last moment, as far as they were seen that long."""
        speeds = {}
        for player in players:
            trail = self._places.setdefault(player.player, deque())
            trail.append((seconds, player.x, player.y))
            while trail and seconds - trail[0][0] > 2 * SPEED_SECONDS:
                trail.popleft()
            earlier = next((p for p in trail if seconds - p[0] <= SPEED_SECONDS), trail[0])
            elapsed = seconds - earlier[0]
            if elapsed >= 0.5 * SPEED_SECONDS:
                speeds[player.player] = float(
                    np.hypot(player.x - earlier[1], player.y - earlier[2]) / elapsed
                )
        present = {player.player for player in players}
        for gone in [p for p, trail in self._places.items() if p not in present]:
            if seconds - self._places[gone][-1][0] > 2.0:
                del self._places[gone]
        return speeds

    def _line_up(self, players: Sequence[Player]) -> Optional[Dict[int, int]]:
        """If the teams stand at opposite ends: team -> the direction it will attack in."""
        near_line, far_line = self.end_zone + LINE_REACH, self.length - self.end_zone - LINE_REACH
        at_end = {}
        for team in (0, 1):
            along = np.array([p.y for p in players if p.team == team])
            if len(along) < MIN_PLAYERS_AT_AN_END:
                return None
            near, far = int((along <= near_line).sum()), int((along >= far_line).sum())
            needed = max(MIN_PLAYERS_AT_AN_END, SHARE_AT_AN_END * len(along))
            at_end[team] = +1 if near >= needed else -1 if far >= needed else 0
        if at_end[0] * at_end[1] != -1:
            return None
        return at_end

    def _in_end_zone(self, y: float) -> int:
        """+1 in the far end zone, -1 in the near one, 0 on the central field."""
        return +1 if y >= self.length - self.end_zone else -1 if y <= self.end_zone else 0

    # ------------------------------------------------------------------ per frame

    def update(
        self,
        seconds: float,
        players: Sequence[Player],
        disc_state: str = "air",
        holder: Optional[int] = None,
        flight_seconds: Optional[float] = None,
        known: bool = True,
    ) -> str:
        """Take in a frame; returns the state.

        Args:
            seconds: Time of the frame in the video
            players: The players on the field with their places
            disc_state: "held", "air" or "ground" (see possession.py)
            holder: The player who holds the disc, if one does
            flight_seconds: How long the disc has been seen flying, if it is
            known: Whether this is drone footage with the field found; without that
                the players' places say nothing
        """
        if self._lost.update(not known, seconds) > LOST_SECONDS and self.state != UNKNOWN:
            self._enter(UNKNOWN, seconds)
        if not known:
            return self.state

        speeds = self._speeds(players, seconds)
        typical_speed = float(np.median(list(speeds.values()))) if speeds else 0.0
        lined_up = self._line_up(players)
        holding = next((p for p in players if p.player == holder), None)
        flying = disc_state == "air" and (flight_seconds or 0.0) >= 0.3

        lined_for = self._lined_up.update(lined_up is not None, seconds)
        # The line is broken: the disc flies, or the teams are no longer at their ends
        # and run
        broken_for = self._line_broken.update(
            flying or (lined_up is None and typical_speed >= WALKING), seconds
        )
        playing_for = self._playing.update(
            lined_up is None and typical_speed >= WALKING and (flying or holding is not None),
            seconds,
        )
        walking_for = self._walking.update(typical_speed < WALKING, seconds)

        # A score: the holder stands in the end zone their team attacks
        end = self._in_end_zone(holding.y) if holding is not None else 0
        attacked = self.attacks.get(holding.team) if holding is not None else None
        scoring_for = self._scoring.update(end != 0 and attacked == end, seconds)
        slow_for = self._slow_in_end_zone.update(
            end != 0 and attacked is None and typical_speed < WALKING, seconds
        )

        if self.state != LINED_UP and lined_for >= LINE_UP_SECONDS:
            # Seen from live play: the point ended without the score being seen. Seen a
            # moment after a pull: there was none, someone only stepped off the line.
            pulled = self.events[-1] if self.events and self.events[-1].kind == "pull" else None
            if pulled is not None and seconds - pulled.seconds <= FALSE_START_SECONDS:
                self.events.pop()
            self.attacks = dict(lined_up)
            self._enter(LINED_UP, seconds)
        elif self.state == LINED_UP:
            if lined_up is not None:
                self.attacks = dict(lined_up)
            if broken_for >= LINE_BROKEN_SECONDS:
                self._enter(PULL, seconds)
                self.events.append(GameEvent(seconds, "pull"))
        elif self.state == PULL:
            lasted = seconds - self.since
            if (holding is not None and lasted >= MIN_PULL_SECONDS) or lasted >= MAX_PULL_SECONDS:
                self._enter(LIVE, seconds)
        elif self.state == LIVE:
            if scoring_for >= SCORE_SECONDS or slow_for >= SLOW_SCORE_SECONDS:
                self._enter(SCORE, seconds)
                self.events.append(GameEvent(seconds, "score", holding.team))
        elif self.state == SCORE:
            if seconds - self.since >= SCORE_SHOWN_SECONDS:
                self._enter(BETWEEN_POINTS, seconds)
        elif self.state in (UNKNOWN, BETWEEN_POINTS):
            # A point under way that did not begin with a line-up that was seen
            if playing_for >= 2.0 and typical_speed >= RUNNING:
                self._enter(LIVE, seconds)
            elif self.state == UNKNOWN and walking_for >= 3.0:
                self._enter(BETWEEN_POINTS, seconds)
        return self.state

    def _enter(self, state: str, seconds: float) -> None:
        self.state, self.since = state, seconds
        for seen in (self._lined_up, self._line_broken, self._scoring, self._slow_in_end_zone):
            seen.since = None
        self._playing.since = self._walking.since = None
