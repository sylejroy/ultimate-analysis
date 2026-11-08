"""The top-down view: the field drawn from above, with the players where they stand."""

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .field_maps import GIVES_UP_COLOUR, GIVES_UP_SHOWN, shade_from_above

# The colours of the drawn field (BGR): grass, the end zones a little lighter, what lies
# around the field darker, and the lines
DIAGRAM_GRASS = (45, 83, 20)
DIAGRAM_END_ZONE = (58, 96, 34)
DIAGRAM_SURROUNDINGS = (24, 22, 20)
DIAGRAM_LINES = (235, 235, 235)
DIAGRAM_DISC = (21, 204, 250)  # The disc, and the ring around who holds it
UNKNOWN_TEAM = (175, 175, 175)
OUT_OF_VIEW = 0.55  # How bright what the camera does not see is drawn, of what it sees
DIAGRAM_VIEW_EDGE = (200, 210, 80)  # The outline of what the camera sees
DIAGRAM_TRAIL_POINTS = 30  # A trail is drawn through at most this many of its points
_empty_field: dict = {}  # The field without players as last drawn, and what it was drawn of


def draw_field_diagram(
    size: Tuple[int, int],
    width: float,
    length: float,
    end_zone: float,
    field_to_view: np.ndarray,
    players: Sequence[dict],
    disc: Optional[Tuple[float, float]] = None,
    in_camera: Optional[np.ndarray] = None,
    disc_path: Sequence[np.ndarray] = (),
    field_map: Optional[np.ndarray] = None,
    map_lines: bool = False,
    opening: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The field drawn from above, with the players as dots in their team's colour.

    A drawing instead of the video warped onto the ground: in the warped picture the
    players are smeared along the ground away from the camera and nothing can be read.

    Args:
        size: (width, height) of the picture
        width, length, end_zone: The field's sizes in its unit
        field_to_view: 3x3, place on the field -> pixel of the picture
        players: For each player a dict: "place" (x, y on the field), "colour" (BGR, or
            None for a team not known), "label" (a jersey number or ""), "holder"
            (whether they have the disc), "trail" (places on the field, oldest first)
            and "gives_up" (a defender's share of the dangerous space, if shown)
        disc: Where the disc is on the field, if it is seen and nobody holds it
        in_camera: The outline, on the field, of what the camera sees (N, 2): drawn
            lighter than the rest, which is out of view
        disc_path: Where the disc has been: lines of places on the field (N, 2)
        field_map: A number per square of the field (rows along it from the near back
            line, columns across): where the next throw goes or the dangerous space,
            drawn over the field, the highest brightest
        map_lines: The map as lines of equal height instead of shading
        opening: How much more the map is than a moment ago, drawn as thin lines

    Returns:
        The picture (BGR)
    """

    def in_view(places: np.ndarray) -> np.ndarray:
        mapped = places @ field_to_view[:, :2].T + field_to_view[:, 2]
        return np.rint(mapped[:, :2] / mapped[:, 2:3]).astype(np.int32)

    corners = in_view(np.array([[0, 0], [width, 0], [width, length], [0, length]], dtype=float))
    unit = float(np.linalg.norm(corners[1] - corners[0])) / width  # Pixels per field unit
    line = max(1, int(round(unit * 0.25)))
    # The empty field is the same in every frame: drawn once
    drawn = (tuple(size), width, length, end_zone, field_to_view.tobytes())
    if _empty_field.get("of") != drawn:
        view = np.full((size[1], size[0], 3), DIAGRAM_SURROUNDINGS, dtype=np.uint8)
        cv2.fillConvexPoly(view, corners, DIAGRAM_GRASS)
        for near, far in ((0.0, end_zone), (length - end_zone, length)):
            zone = np.array([[0, near], [width, near], [width, far], [0, far]], dtype=float)
            cv2.fillConvexPoly(view, in_view(zone), DIAGRAM_END_ZONE)
        cv2.polylines(view, [corners], True, DIAGRAM_LINES, line, cv2.LINE_AA)
        for along in (end_zone, length - end_zone):
            ends = in_view(np.array([[0, along], [width, along]], dtype=float))
            cv2.line(view, tuple(ends[0]), tuple(ends[1]), DIAGRAM_LINES, line, cv2.LINE_AA)
        _empty_field.update(of=drawn, view=view)
    view = _empty_field["view"].copy()

    if in_camera is not None and len(in_camera) >= 3:
        # What is out of the camera's view is dimmed, and the view outlined
        seen = np.zeros(view.shape[:2], dtype=np.uint8)
        outline = in_view(np.asarray(in_camera, dtype=float))
        cv2.fillPoly(seen, [outline], 255)
        bright = view
        view = cv2.convertScaleAbs(bright, alpha=OUT_OF_VIEW)
        cv2.copyTo(bright, seen, view)
        cv2.polylines(view, [outline], True, DIAGRAM_VIEW_EDGE, max(1, line), cv2.LINE_AA)

    if field_map is not None:
        shade_from_above(view, field_map, in_view, width, length, map_lines, opening)

    radius = max(4, int(round(unit * 1.1)))
    for player in players:
        trail = np.asarray(player.get("trail", ()), dtype=float).reshape(-1, 2)
        if len(trail) >= 2:
            colour = player["colour"] or UNKNOWN_TEAM
            # A trail is smooth, and a smooth line costs by its corners: a few of its
            # points draw the same line (counted from the newest, where the player is)
            every = -(-len(trail) // DIAGRAM_TRAIL_POINTS)
            trail = trail[::-every][::-1]
            cv2.polylines(view, [in_view(trail)], False, colour, max(1, radius // 4), cv2.LINE_AA)
    for stretch in disc_path:
        if len(stretch) >= 2:
            cv2.polylines(
                view, [in_view(stretch)], False, DIAGRAM_DISC, max(2, radius // 3), cv2.LINE_AA
            )
    font = cv2.FONT_HERSHEY_SIMPLEX
    for player in players:
        x, y = in_view(np.array([player["place"]], dtype=float))[0]
        colour = player["colour"] or UNKNOWN_TEAM
        # A dark rim under a light team and a light one under a dark team
        rim = (20, 20, 20) if sum(colour) > 380 else (235, 235, 235)
        if player.get("holder"):
            cv2.circle(view, (x, y), radius + 5, DIAGRAM_DISC, 3, cv2.LINE_AA)
        gives_up = player.get("gives_up", 0.0)
        if gives_up >= GIVES_UP_SHOWN:
            # A defender who leaves much of the dangerous space: ringed red, with the share
            cv2.circle(view, (x, y), radius + 5, GIVES_UP_COLOUR, 3, cv2.LINE_AA)
            text = f"{gives_up:.0%}"
            scale = radius / 22.0 + 0.2
            origin = (x - radius, y + radius + int(18 * scale) + 6)
            cv2.putText(view, text, origin, font, scale, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(view, text, origin, font, scale, GIVES_UP_COLOUR, 2, cv2.LINE_AA)
        cv2.circle(view, (x, y), radius, colour, -1, cv2.LINE_AA)
        cv2.circle(view, (x, y), radius, rim, 1, cv2.LINE_AA)
        if player.get("label"):
            text = str(player["label"])
            scale = radius / 22.0 + 0.25
            (text_width, _), _ = cv2.getTextSize(text, font, scale, 2)
            origin = (x - text_width // 2, y - radius - 5)
            cv2.putText(view, text, origin, font, scale, (20, 20, 20), 5, cv2.LINE_AA)
            cv2.putText(view, text, origin, font, scale, (245, 245, 245), 2, cv2.LINE_AA)
    if disc is not None:
        x, y = in_view(np.array([disc], dtype=float))[0]
        cv2.circle(view, (x, y), max(3, radius // 2), DIAGRAM_DISC, -1, cv2.LINE_AA)
        cv2.circle(view, (x, y), max(3, radius // 2), (20, 20, 20), 1, cv2.LINE_AA)
    return view
