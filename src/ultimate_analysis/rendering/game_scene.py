"""The game as a scene in three dimensions.

Where everybody stands on the field, where the disc is and where the camera is, taken
from what the pipeline found in a frame, and how that looks from a point of view of
one's own choosing. No Qt here: gui/scene/scene_tab.py paints what this works out.

Places are in field units (yards or metres, as the field's template says), x across the
field, y along it and z upwards.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..processing.disc_flight import DISC_HEIGHT
from ..utils.field_camera import camera_of
from ..utils.field_template import FieldTemplate

# Who stands further from the field than this (field units) is left out: spectators,
# and boxes above the horizon, which map to places that mean nothing
BESIDE_FIELD = 12.0
# How tall a player is drawn, in metres
PLAYER_METRES = 1.8
# Nothing nearer to the eye than this is drawn (field units)
NEAR = 0.5


@dataclass
class ScenePlayer:
    """One player on the ground."""

    player_id: int
    place: Tuple[float, float]
    colour: Optional[Tuple[int, int, int]]  # Their team's shirt colour (BGR), if known
    number: str  # Jersey number, "" while it is not known
    holding: bool = False


@dataclass
class GameScene:
    """What is known of the game in one frame."""

    frame_index: int
    players: List[ScenePlayer] = field(default_factory=list)
    disc: Optional[Tuple[float, float, float]] = None
    disc_state: str = "air"  # "held", "air" or "ground" (FrameResult.disc_state)
    # Where the camera that films the game is, and the spot of the ground in the middle
    # of its picture
    camera: Optional[Tuple[float, float, float]] = None
    looks_at: Optional[Tuple[float, float]] = None


def player_height(template: FieldTemplate) -> float:
    """How tall a player is in the template's units."""
    return PLAYER_METRES if template.unit == "m" else PLAYER_METRES / 0.9144


def _on_ground(to_field: np.ndarray, pixel: Sequence[float]) -> Optional[np.ndarray]:
    over = to_field @ np.array([pixel[0], pixel[1], 1.0])
    if abs(over[2]) < 1e-12:
        return None
    return over[:2] / over[2]


def _near_field(place: np.ndarray, template: FieldTemplate) -> bool:
    return bool(
        -BESIDE_FIELD <= place[0] <= template.width + BESIDE_FIELD
        and -BESIDE_FIELD <= place[1] <= template.length + BESIDE_FIELD
    )


def scene_of(result: Any, template: FieldTemplate) -> Optional[GameScene]:
    """The scene of a frame's results (FrameResult); None where the field is not known.

    A player stands where the pipeline has them (FrameResult.places, held steady over
    the frames), or else where the middle of their box's lower edge is on the ground. A held
    disc is with its holder, a flying one over the spot the pipeline worked out, both at
    the height a disc is thrown and caught at; any other lies where it is seen.
    """
    to_field = getattr(result, "image_to_field", None)
    if to_field is None or not getattr(result, "wide_shot", True):
        return None
    scene = GameScene(result.frame_index, disc_state=result.disc_state)
    for track in result.tracks:
        if track.class_name != "player":
            continue
        x1, _, x2, y2 = track.bbox
        steady = getattr(result, "places", {}).get(track.track_id)
        place = (
            np.asarray(steady, dtype=np.float64)
            if steady is not None
            else _on_ground(to_field, ((x1 + x2) / 2, y2))
        )
        if place is None or not _near_field(place, template):
            continue
        number = str(result.player_ids.get(track.track_id, ("", None))[0] or "")
        if not number.isdigit():
            number = str(result.numbers_by_looks.get(track.track_id, ""))
        scene.players.append(
            ScenePlayer(
                player_id=track.track_id,
                place=(float(place[0]), float(place[1])),
                colour=getattr(track, "team_colour", None),
                number=number if number.isdigit() else "",
                holding=track.track_id == result.holder_id,
            )
        )
    scene.disc = _disc_of(result, scene, to_field, template)
    scene.camera, scene.looks_at = _camera_of(result, to_field, template)
    return scene


def _disc_of(
    result: Any, scene: GameScene, to_field: np.ndarray, template: FieldTemplate
) -> Optional[Tuple[float, float, float]]:
    holder = next((player for player in scene.players if player.holding), None)
    if holder is not None:
        return (holder.place[0], holder.place[1], DISC_HEIGHT)
    if result.disc_place is not None:
        return (float(result.disc_place[0]), float(result.disc_place[1]), DISC_HEIGHT)
    discs = [d for d in result.detections if d["class_name"] == "disc"]
    if not discs:
        return None
    x1, y1, x2, y2 = max(discs, key=lambda d: d.get("confidence", 0.0))["bbox"]
    place = _on_ground(to_field, ((x1 + x2) / 2, (y1 + y2) / 2))
    if place is None or not _near_field(place, template):
        return None
    return (float(place[0]), float(place[1]), 0.0)


def _camera_of(
    result: Any, to_field: np.ndarray, template: FieldTemplate
) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float]]]:
    view = getattr(result, "main_view", None)
    if view is None:
        return None, None
    height, width = view.shape[:2]
    try:
        _, _, position = camera_of(np.linalg.inv(to_field), (width, height))
    except (np.linalg.LinAlgError, cv2.error):
        return None, None
    if not np.all(np.isfinite(position)):
        return None, None
    camera = (float(position[0]), float(position[1]), abs(float(position[2])))
    middle = _on_ground(to_field, (width / 2, height / 2))
    reach = 2 * template.length
    if middle is None or np.hypot(middle[0] - camera[0], middle[1] - camera[1]) > reach:
        return camera, None
    return camera, (float(middle[0]), float(middle[1]))


@dataclass
class ViewPoint:
    """An eye that circles a spot of the scene and looks at it."""

    target: Tuple[float, float, float]
    azimuth: float  # Degrees: the direction from the spot to the eye, from the x axis
    elevation: float  # Degrees above the ground
    distance: float
    focal: float = 1.2  # Focal length as a share of the picture's height
    # The eye and its axes for the values above as they were when last asked for: a
    # picture asks a hundred times and more
    _worked_out: Optional[tuple] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_eye(cls, eye: Sequence[float], target: Sequence[float]) -> "ViewPoint":
        """The eye at one place looking at another."""
        away = np.asarray(eye, dtype=np.float64) - np.asarray(target, dtype=np.float64)
        distance = max(float(np.linalg.norm(away)), 1e-6)
        return cls(
            target=(float(target[0]), float(target[1]), float(target[2])),
            azimuth=float(np.degrees(np.arctan2(away[1], away[0]))),
            elevation=float(np.degrees(np.arcsin(np.clip(away[2] / distance, -1.0, 1.0)))),
            distance=distance,
        )

    def _eye_and_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        key = (self.target, self.azimuth, self.elevation, self.distance)
        if self._worked_out is None or self._worked_out[0] != key:
            eye = self._eye()
            self._worked_out = (key, eye, self._axes(eye))
        return self._worked_out[1], self._worked_out[2]

    @property
    def eye(self) -> np.ndarray:
        return self._eye_and_axes()[0]

    @property
    def axes(self) -> np.ndarray:
        """Rows: to the right, upwards and ahead, as the eye sees them."""
        return self._eye_and_axes()[1]

    def _eye(self) -> np.ndarray:
        azimuth, elevation = np.radians(self.azimuth), np.radians(self.elevation)
        away = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        return np.asarray(self.target, dtype=np.float64) + self.distance * away

    def _axes(self, eye: np.ndarray) -> np.ndarray:
        ahead = np.asarray(self.target, dtype=np.float64) - eye
        ahead /= np.linalg.norm(ahead)
        right = np.cross(ahead, [0.0, 0.0, 1.0])
        if np.linalg.norm(right) < 1e-9:  # Straight down: any direction is "right"
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        return np.array([right, np.cross(right, ahead), ahead])

    def seen(self, places: Sequence[Sequence[float]]) -> np.ndarray:
        """Places of the scene (N, 3) -> right of, above and ahead of the eye (N, 3)."""
        eye, axes = self._eye_and_axes()
        return (np.asarray(places, dtype=np.float64).reshape(-1, 3) - eye) @ axes.T

    def on_picture(self, seen: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """What `seen` gave, all of it ahead of the eye -> pixels (N, 2) of a picture."""
        scale = self.focal * size[1] / seen[:, 2]
        return np.column_stack([size[0] / 2 + seen[:, 0] * scale, size[1] / 2 - seen[:, 1] * scale])

    def pixels_per_unit(self, ahead: float, size: Tuple[int, int]) -> float:
        """How large one field unit is drawn this far ahead of the eye."""
        return self.focal * size[1] / max(ahead, NEAR)


def clip_ahead(seen: np.ndarray, closed: bool = True, near: float = NEAR) -> np.ndarray:
    """Cut a polygon (or, not closed, a line of several points) off behind the eye.

    Args:
        seen: Its corners as the eye sees them (N, 3), from ViewPoint.seen
        closed: Whether the last corner is joined to the first
        near: How far ahead of the eye the cut is made

    Returns:
        The corners of what is left (M, 3); none if it is all behind the eye
    """
    kept: List[np.ndarray] = []
    count = len(seen)
    for index in range(count if closed else count - 1):
        here, there = seen[index], seen[(index + 1) % count]
        if here[2] >= near:
            kept.append(here)
        if (here[2] >= near) != (there[2] >= near):
            share = (near - here[2]) / (there[2] - here[2])
            kept.append(here + share * (there - here))
    if not closed and count and seen[-1][2] >= near:
        kept.append(seen[-1])
    return np.array(kept).reshape(-1, 3)


def circle(centre: Sequence[float], radius: float, corners: int = 20) -> np.ndarray:
    """A circle lying flat, as a polygon (corners, 3)."""
    angles = np.linspace(0.0, 2 * np.pi, corners, endpoint=False)
    return np.column_stack(
        [
            centre[0] + radius * np.cos(angles),
            centre[1] + radius * np.sin(angles),
            np.full(corners, centre[2] if len(centre) > 2 else 0.0),
        ]
    )
