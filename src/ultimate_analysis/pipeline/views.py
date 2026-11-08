"""The two views of a frame: the camera view with what was found drawn on it, and the
field from above with the players where they stand.

Both are drawn from a Scene: what the analysis of the frame left, taken in one go on the
frame's own thread. The top-down view is drawn on another thread meanwhile, and reads
nothing the next frame's analysis changes.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config.settings import get_setting
from ..processing import health
from ..processing.field_registration import field_to_canvas
from ..rendering.field import (
    draw_field_segmentation,
    draw_unified_field_mask,
    get_primary_field_color,
)
from ..rendering.field_maps import mark_defenders, shade_camera_view
from ..rendering.overlays import draw_fps_overlay, draw_jersey_table, draw_notice
from ..rendering.top_down import draw_field_diagram
from ..rendering.tracks import (
    draw_detections,
    draw_disc_heat,
    draw_holder_guess,
    draw_possession,
    draw_tracks,
    draw_tracks_with_player_ids,
    team_display_colour,
)
from ..utils.field_template import FieldTemplate
from ..utils.logger import get_logger

logger = get_logger("PIPELINE")

Record = Callable[[str, float], float]


@dataclass
class Scene:
    """What the views of a frame are drawn from."""

    template: FieldTemplate  # The field the top-down view draws
    frame_size: Tuple[int, int]  # (width, height) of the frame
    wide: bool = True  # False on a close-up or title card
    fps: float = 0.0  # The processing rate shown
    detections: List[Dict[str, Any]] = field(default_factory=list)
    tracks: List[Any] = field(default_factory=list)
    trails: Dict[int, np.ndarray] = field(default_factory=dict)  # Track -> its feet's trail
    # Track -> the number shown: read, or "~17" for one known by the player's looks
    numbers: Dict[int, Tuple[str, Any]] = field(default_factory=dict)
    read_numbers: bool = False  # Whether any number was read (for the readings table)
    holder_id: Optional[int] = None
    holder_box: Optional[Any] = None  # Where the holder was last seen, if not now
    possession_colour: Optional[Tuple[int, int, int]] = None
    holder_guess: Optional[Dict[int, float]] = None  # The holder network's, if shown
    disc_heat: Optional[np.ndarray] = None  # The disc heat map, if shown
    # A map over the field, if shown: a number per square (rows along the field from the
    # near back line), where the next throw goes (processing/throw_targets.py) or the
    # dangerous space (processing/open_space.py); how much more it is than a moment ago;
    # whether it is drawn as lines; and each defender's share of the danger by track
    field_map: Optional[np.ndarray] = None
    map_opening: Optional[np.ndarray] = None
    map_lines: bool = False
    gives_up: Dict[int, float] = field(default_factory=dict)
    field_results: List[Any] = field(default_factory=list)  # The field model's masks
    field_mask: Optional[np.ndarray] = None  # The field the model sees, one mask
    image_to_field: Optional[np.ndarray] = None  # Where the field lies, if known
    places: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # On the field
    disc_place: Optional[Tuple[float, float]] = None  # Over which spot a flying disc is
    disc_path: Sequence[np.ndarray] = ()  # Where the disc has been, on the field
    show_holder_guess: bool = False
    show_disc_heat: bool = False


def draw_camera_view(frame: np.ndarray, scene: Scene, options: Any, record: Record) -> np.ndarray:
    """Draw what was found on a frame the caller owns."""
    if not scene.wide:
        draw_notice(frame, "Close-up: analysis paused")
        draw_fps_overlay(frame, scene.fps)
        return frame
    if not (options.field_segmentation or scene.detections or scene.tracks):
        draw_fps_overlay(frame, scene.fps)
        return frame

    start = time.perf_counter()

    # Field first, as background for the players
    if scene.field_results and options.field_segmentation:
        if get_setting("models.segmentation.show_raw_masks", False):
            frame = draw_field_segmentation(frame, scene.field_results)
        if scene.field_mask is not None:
            # The outline of the field the model sees
            frame = draw_unified_field_mask(
                frame, scene.field_mask, get_primary_field_color(), alpha=0.3, in_place=True
            )

    # The map over the field, on the ground under the players
    if scene.field_map is not None and scene.image_to_field is not None:
        shade_camera_view(
            frame,
            scene.field_map,
            scene.image_to_field,
            seen_by_camera(scene.image_to_field, scene.frame_size, scene.template),
            scene.template.width,
            scene.template.length,
            scene.map_lines,
            scene.map_opening,
        )

    # Plain detections only without tracking; tracks carry the same boxes
    if scene.detections and not options.tracking:
        frame = draw_detections(frame, scene.detections, in_place=True)
    elif scene.tracks and options.tracking:
        if options.player_id:
            frame = draw_tracks_with_player_ids(
                frame, scene.tracks, scene.trails, scene.numbers, in_place=True
            )
        else:
            frame = draw_tracks(frame, scene.tracks, scene.trails, in_place=True)
        draw_possession(
            frame, scene.tracks, scene.holder_id, scene.holder_box, scene.possession_colour
        )
        if scene.gives_up:
            mark_defenders(frame, scene.tracks, scene.gives_up)
        # As the network says it, in every frame: nothing is held back, so that what
        # it gets wrong can be seen
        if scene.show_holder_guess:
            draw_holder_guess(frame, scene.tracks, scene.holder_guess)
    if scene.show_disc_heat:
        draw_disc_heat(frame, scene.disc_heat)

    draw_fps_overlay(frame, scene.fps)
    if (
        options.player_id
        and scene.read_numbers
        and get_setting("models.player_id.show_reading_table", False)
    ):
        draw_jersey_table(frame)

    record("Visualization", start)
    return frame


def draw_top_down_view(
    scene: Scene, options: Any, record: Record
) -> Tuple[Optional[np.ndarray], str]:
    """Draw the field from above, with the players where they stand on it.

    Returns:
        (view, "") or (None, reason there is no view)
    """
    if not scene.wide:
        return None, "Close-up: no top-down view"
    if not options.field_segmentation:
        return None, "The top-down view needs field segmentation"
    if scene.image_to_field is None:
        return None, "The field has not been found in this view yet"
    start = time.perf_counter()
    try:
        view = _field_diagram(scene, options)
    except Exception as e:
        logger.exception(f"Error drawing the top-down view: {e}")
        health.report("Top-down view", f"failed ({type(e).__name__}); see the log")
        return None, f"Error: {e}"
    record("Top-down view", start)
    return view, ""


def _field_diagram(scene: Scene, options: Any) -> np.ndarray:
    """The top-down view as a drawing: the field, and the players where they stand."""
    to_field = scene.image_to_field
    template = scene.template
    # A drawing needs no more pixels than the panel it is shown in
    height = int(get_setting("homography.diagram_height", 900))
    size = (int(height * 1.25 * template.width / template.length), height)
    to_view = field_to_canvas(template, size)

    def on_field(pixels: np.ndarray) -> np.ndarray:
        mapped = pixels @ to_field[:, :2].T + to_field[:, 2]
        ahead = mapped[:, 2] * mapped[-1, 2] > 0  # Above the horizon is no ground
        return mapped[ahead, :2] / mapped[ahead, 2:3]

    players, disc = [], None
    for track in scene.tracks if options.tracking else ():
        x1, y1, x2, y2 = track.to_ltrb()
        if track.class_name != "player":
            if scene.holder_id is None:
                disc = scene.disc_place or tuple(
                    on_field(np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]))[0]
                )
            continue
        shirt = getattr(track, "team_colour", None)
        number = scene.numbers.get(track.track_id, ("", None))[0]
        trail = scene.trails.get(track.track_id)
        players.append(
            {
                "place": scene.places.get(track.track_id)
                or tuple(on_field(np.array([[(x1 + x2) / 2, y2]]))[0]),
                "colour": None if shirt is None else team_display_colour(shirt),
                # A number with a tilde is known by the player's looks, not read
                "label": f"#{number}"
                if str(number).isdigit()
                else (str(number) if str(number).startswith("~") else ""),
                "holder": track.track_id == scene.holder_id,
                "trail": on_field(np.asarray(trail, dtype=float)) if trail is not None else (),
                "gives_up": scene.gives_up.get(track.track_id, 0.0),
            }
        )
    return draw_field_diagram(
        size,
        template.width,
        template.length,
        template.end_zone,
        to_view,
        players,
        disc,
        seen_by_camera(to_field, scene.frame_size, template),
        scene.disc_path if options.tracking else (),
        scene.field_map,
        scene.map_lines,
        scene.map_opening,
    )


def seen_by_camera(
    to_field: np.ndarray, frame_size: Tuple[int, int], template: FieldTemplate
) -> Optional[np.ndarray]:
    """The outline on the field of what the frame shows: its border, as far as that lies
    on the ground (the part above the horizon does not). (N, 2) in field units."""
    width, height = frame_size
    along = np.linspace(0.0, 1.0, 24, endpoint=False)
    border = np.concatenate(
        [
            np.column_stack([along * width, np.zeros_like(along)]),
            np.column_stack([np.full_like(along, width), along * height]),
            np.column_stack([(1 - along) * width, np.full_like(along, height)]),
            np.column_stack([np.zeros_like(along), (1 - along) * height]),
        ]
    )
    mapped = np.column_stack([border, np.ones(len(border))]) @ to_field.T
    # The middle of the frame's lower edge is ground in front of the camera
    ground = to_field @ [width / 2.0, height - 1.0, 1.0]
    ahead = mapped[:, 2] * ground[2] > 1e-9 * abs(ground[2])
    if ahead.sum() < 3:
        return None
    places = mapped[ahead, :2] / mapped[ahead, 2:3]
    # Near the horizon the ground runs off to infinity: kept within sight of the field
    reach = 3.0 * template.length
    return np.clip(places, -reach, reach)
