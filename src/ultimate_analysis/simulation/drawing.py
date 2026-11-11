"""Pictures of the simulated game: the field from above turned on its side, White
attacking to the right and Red to the left, the dangerous space for the team with the
disc shaded (processing/open_space.py), the disc in yellow with its shadow (the higher
it flies the bigger it is drawn) and its path, the wind as an arrow, the score and what
happens along the top. Used by the Simulation tab and scripts/render_simulation.py."""

from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..config.settings import get_setting
from ..processing.field_registration import field_to_canvas
from ..processing.open_space import FieldValue, dangerous_space
from ..processing.throw_targets import COLUMNS, ROWS, Moment
from ..rendering.field_maps import shade_from_above
from ..rendering.game_scene import GameScene, ScenePlayer
from ..rendering.top_down import draw_field_diagram
from ..utils.field_template import TEMPLATES
from . import game
from .plays import PLAYS
from .scripted import CALL, ScriptedTeam, ScriptedTeam2

COLOURS = ((235, 235, 235), (60, 60, 210))  # BGR: team 0 white, team 1 red
NAMES = ("White", "Red")
DISC = (21, 204, 250)
SHADOW = (20, 40, 20)
WIND = (255, 230, 120)
SIZE = (360, 900)  # Of the field from above, before it is turned on its side
SCALE = 1.5  # Of the picture after turning
BANNER = 80  # Pixels


def team_of(kind: str, seed: Optional[int] = None, team: int = 0):
    """The scripted team ("scripted"), the benchmark one ("scripted2", scripted.ScriptedTeam2),
    or one played by the networks in a folder: a team
    that chooses spots if it holds a spots.pt (spots.SpotTeam, drawing its choices as it
    learned them); trained by reinforcement if it holds a ppo.pt (then playing its
    likeliest moves); else as learned.LearnedTeam."""
    if kind == "scripted":
        return ScriptedTeam(seed)
    if kind == "scripted2":
        return ScriptedTeam2(seed, plays=False)
    from pathlib import Path

    folder = Path(kind)
    if (folder / "spots.pt").exists():
        from .spots import SpotTeam, load_spot_brains

        return SpotTeam(load_spot_brains(folder), team, np.random.default_rng(seed), False)
    if (folder / "ppo.pt").exists():
        from .ppo import PolicyTeam, load_brains

        return PolicyTeam(load_brains(folder), team, np.random.default_rng(seed), False, True)
    from .learned import LearnedTeam, load_minds

    return LearnedTeam(load_minds(folder), get_setting("models.throw_targets.weights"), seed)


def danger_of(state: game.GameState) -> Optional[np.ndarray]:
    """The dangerous space for the team with the disc, over the field as it lies."""
    if state.holder is None:
        return None
    team = state.offence
    seen = game.view(state, team)
    others = np.delete(seen.ours, seen.holder, axis=0)
    found = dangerous_space(
        Moment(tuple(seen.ours[seen.holder, :2]), others, seen.theirs), FieldValue()
    )
    grid = found.danger.reshape(ROWS, COLUMNS)
    return grid if game.attack_of(team) > 0 else grid[::-1, ::-1]


def caption_of(event: game.Event) -> Optional[str]:
    """What is said of an event along the top, if anything."""
    who = NAMES[event.team]
    return {
        "block": f"Block! {who} lose the disc",
        "mark block": f"The mark blocks {who}'s throw",
        "drop": f"{who}'s throw falls: turnover",
        "out": f"{who} catch it out of bounds: turnover",
        "stall": f"{who} stalled out",
        "pick up": f"{who} pick up the disc",
        "score": f"{who} score!",
        "pull caught": f"{who} catch the pull",
        "pull lands": f"The pull lands: {who} pick it up",
    }.get(event.kind)


def call_of(state: game.GameState, decisions: Sequence) -> Optional[str]:
    """The set play the team with the disc has called, if any."""
    if not decisions or state.holder is None:
        return None
    messages = decisions[state.offence].messages
    if messages is None:
        return None
    call = int(round(float(np.max(messages[:, CALL]))))
    if 0 < call <= len(PLAYS):
        return f"{NAMES[state.offence]} call a set play: {PLAYS[call - 1].name}"
    return None


def picture_of(
    state: game.GameState, score: List[int], caption: str = "", show_danger: bool = True
) -> np.ndarray:
    """The game as it stands (BGR)."""
    template = TEMPLATES["usau"]
    to_view = field_to_canvas(template, SIZE)

    def at(place) -> tuple:
        x, y, w = to_view @ [place[0], place[1], 1.0]
        return int(round(x / w)), int(round(y / w))

    players = [
        {
            "place": tuple(state.places[i]),
            "colour": COLOURS[game.team_of(i)],
            "label": "",
            "holder": i == state.holder,
        }
        for i in range(2 * game.PLAYERS)
    ]
    view = draw_field_diagram(
        SIZE,
        template.width,
        template.length,
        template.end_zone,
        to_view,
        players,
        None,
        field_map=danger_of(state) if show_danger else None,
    )
    if state.throw is not None:
        flight = state.throw
        path = np.array([at(flight.point(s)[0]) for s in np.linspace(0, 1, 24)], np.int32)
        cv2.polylines(view, [path], False, DISC, 1, cv2.LINE_AA)
        place, height = flight.at(state.seconds)
        cv2.circle(view, at(place), 4, SHADOW, -1, cv2.LINE_AA)
        # Drawn up off its shadow, and bigger, the higher it flies
        up = at(place)
        up = (up[0] - int(height * 1.2), up[1])
        cv2.circle(view, up, int(4 + 0.6 * height), DISC, -1, cv2.LINE_AA)
        cv2.circle(view, up, int(4 + 0.6 * height), (20, 20, 20), 1, cv2.LINE_AA)
    elif state.disc is not None:
        cv2.circle(view, at(state.disc), 5, DISC, -1, cv2.LINE_AA)

    # Turned on its side: team 0 attacks to the right. Numbers are written after, so
    # that they read upright
    height = view.shape[0]
    view = cv2.rotate(view, cv2.ROTATE_90_CLOCKWISE)
    view = cv2.resize(view, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

    def turned(place) -> tuple:
        x, y = at(place)
        return int(SCALE * (height - 1 - y)), int(SCALE * x)

    for i, place in enumerate(state.places):
        x, y = turned(place)
        for colour, thickness in (((20, 20, 20), 4), ((245, 245, 245), 1)):
            cv2.putText(
                view,
                str(i % game.PLAYERS + 1),
                (x + 9, y - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                thickness,
                cv2.LINE_AA,
            )
    if np.any(state.wind):
        # The wind, from the top left corner: on the field, y to the right
        base = np.array([60, 60])
        arrow = np.array([state.wind[1], state.wind[0]]) * 5.0
        cv2.arrowedLine(
            view,
            tuple(base),
            tuple((base + arrow).astype(int)),
            WIND,
            3,
            cv2.LINE_AA,
            tipLength=0.3,
        )
        speed = float(np.hypot(*state.wind))
        cv2.putText(
            view,
            f"wind {speed:.0f} yd/s",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WIND,
            1,
            cv2.LINE_AA,
        )

    banner = np.full((BANNER, view.shape[1], 3), 24, dtype=np.uint8)
    holder = "" if state.holder is None else f"   stall {state.stall:.0f}"
    pulled = state.throw is not None and state.throw.pull
    has = "receive the pull" if pulled else "has the disc"
    top = (
        f"{NAMES[0]} {score[0]} - {score[1]} {NAMES[1]}     {NAMES[state.offence]} {has}{holder}"
        f"     {state.seconds:5.1f} s"
    )
    cv2.putText(
        banner, top, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA
    )
    cv2.putText(
        banner, caption, (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 220, 250), 2, cv2.LINE_AA
    )
    return np.vstack([banner, view])


def scene_of_game(state: game.GameState) -> GameScene:
    """The game as the 3D view draws it: players in their team's colour, numbered 1 to 7,
    and the disc at its height."""
    players = [
        ScenePlayer(
            i,
            (float(state.places[i][0]), float(state.places[i][1])),
            COLOURS[game.team_of(i)],
            str(i % game.PLAYERS + 1),
            holding=i == state.holder,
        )
        for i in range(2 * game.PLAYERS)
    ]
    disc, disc_state = None, "air"
    if state.holder is not None:
        x, y = state.places[state.holder]
        disc, disc_state = (float(x), float(y), 1.0), "held"
    elif state.throw is not None:
        place, height = state.throw.at(state.seconds)
        disc = (float(place[0]), float(place[1]), float(height))
    elif state.disc is not None:
        disc, disc_state = (float(state.disc[0]), float(state.disc[1]), 0.0), "ground"
    return GameScene(int(round(state.seconds / game.STEP)), players, disc, disc_state)


def knowledge_picture(
    view,
    team: int,
    size: tuple = (240, 620),
    considered: Optional[list] = None,
    heat: Optional[tuple] = None,
    trail: Optional[list] = None,
) -> np.ndarray:
    """What one player knows, from above with their team attacking upwards: teammates
    and opponents where they believe them, solid where seen, a ring as wide as how
    unsure where remembered, faint where lost; the player themselves ringed in yellow;
    the disc where they judge it comes down. `considered`: the choices a player weighs,
    [(place on the field, or None for holding on, chance)], the one taken first: each
    ringed and numbered in that order, the one taken in yellow with its chance and an
    arrow from the player. `heat`: (places (n, 2), chances (n,)): every choice weighed,
    shaded over the field. `trail`: where the player has been, oldest first: the path
    they ran, drawn behind them."""
    from .learned import KNOWN_UNSURE

    template = TEMPLATES["usau"]
    to_view = field_to_canvas(template, size)
    picture = draw_field_diagram(
        size, template.width, template.length, template.end_zone, to_view, []
    )

    def at(place) -> tuple:
        x, y, w = to_view @ [place[0], place[1], 1.0]
        return int(round(x / w)), int(round(y / w))

    unit = abs((to_view @ [1.0, 0.0, 1.0])[0] - (to_view @ [0.0, 0.0, 1.0])[0])
    if heat is not None:
        _shade_chances(picture, to_view, template, *heat)
    if trail is not None and len(trail) > 1:
        points = [at(place) for place in trail]
        recent = len(points) - 30  # The last three seconds drawn thicker
        for n in range(1, len(points)):
            wide = 2 if n >= recent else 1
            cv2.line(picture, points[n - 1], points[n], TRAIL, wide, cv2.LINE_AA)
    groups = (
        (view.ours, view.ours_unsure, COLOURS[team]),
        (view.theirs, view.theirs_unsure, COLOURS[1 - team]),
    )
    for rows, unsure, colour in groups:
        for row, doubt in zip(rows, unsure):
            centre = at(row[:2])
            if doubt >= KNOWN_UNSURE:
                cv2.circle(picture, centre, 4, (90, 90, 90), 1, cv2.LINE_AA)  # Lost
                continue
            if doubt > 0.6:
                cv2.circle(picture, centre, max(5, int(doubt * unit)), colour, 1, cv2.LINE_AA)
            cv2.circle(picture, centre, 5, colour, -1, cv2.LINE_AA)
    cv2.circle(picture, at(view.ours[view.me, :2]), 9, DISC, 2, cv2.LINE_AA)
    if view.disc is not None:
        cv2.drawMarker(picture, at(view.disc), DISC, cv2.MARKER_TILTED_CROSS, 12, 2)
    if considered:
        _draw_choices(picture, at, at(view.ours[view.me, :2]), considered)
    return picture


HEAT_SPREAD = 2.0  # Yards: how far each choice's chance is spread over the field
TRAIL = (235, 200, 120)  # The path a player ran (blue, green, red)


def _shade_chances(picture, to_view, template, places, chances) -> None:
    """Every choice's chance spread over the field around it and shaded, in place."""
    across = np.arange(0.5, template.width, 1.0)
    along = np.arange(0.5, template.length, 1.0)
    grid = np.zeros((len(along), len(across)))
    for (x, y), chance in zip(np.asarray(places, dtype=float), chances):
        if chance <= 0:
            continue
        grid += chance * np.exp(
            -((along[:, None] - y) ** 2 + (across[None] - x) ** 2) / (2 * HEAT_SPREAD**2)
        )

    def in_view(points: np.ndarray) -> np.ndarray:
        rows = np.column_stack([points, np.ones(len(points))]) @ to_view.T
        return np.rint(rows[:, :2] / rows[:, 2:]).astype(int)

    shade_from_above(picture, grid, in_view, template.width, template.length)


def _draw_choices(picture: np.ndarray, at, start: tuple, considered) -> None:
    """The choices a player weighs, ringed and numbered in order, the one taken (the
    first) in yellow with its chance and an arrow from the player, the others with a
    thin line."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    def label(text: str, origin: tuple, colour: tuple) -> None:
        cv2.putText(picture, text, origin, font, 0.42, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(picture, text, origin, font, 0.42, colour, 1, cv2.LINE_AA)

    for n in reversed(range(len(considered))):  # The one taken drawn last, on top
        place, chance = considered[n]
        taken = n == 0
        colour = (64, 214, 255) if taken else (235, 235, 235)
        if place is None:  # Holding on is not a place: said at the top
            label(f"Hold on {chance:.0%}", (8, 18 if taken else 36), colour)
            continue
        end = at(place)
        if taken:
            long = float(np.hypot(end[0] - start[0], end[1] - start[1]))
            if long > 1.0:
                cv2.arrowedLine(
                    picture, start, end, colour, 2, cv2.LINE_AA, tipLength=min(0.5, 10.0 / long)
                )
        else:
            cv2.line(picture, start, end, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.circle(picture, end, 8 if taken else 5, colour, 2 if taken else 1, cv2.LINE_AA)
        said = f"{n + 1}  {chance:.0%}" if taken else str(n + 1)
        label(said, (end[0] + 11, end[1] + 5), colour)
