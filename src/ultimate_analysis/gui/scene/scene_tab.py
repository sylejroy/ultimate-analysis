"""The 3D View tab: the game as the pipeline understands it, on a field one can walk around.

Shows what the Main Analysis tab last analysed: the players where they stand on the
field in their team's colours, the disc, who holds it, and the camera that films the
game. Drag to turn the view, right-drag to move it, the wheel to come closer.
"""

from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...config.settings import get_setting
from ...rendering.game_scene import (
    GameScene,
    ScenePlayer,
    ViewPoint,
    circle,
    clip_ahead,
    player_height,
    scene_of,
    within_rectangle,
)
from ...utils.field_template import DEFAULT_RULESET, TEMPLATES, FieldTemplate
from ..theme import FAINT_TEXT

SKY_TOP, SKY_BOTTOM = QColor(18, 22, 30), QColor(46, 56, 70)
SURROUND = QColor(34, 78, 48)
GRASS = (QColor(52, 124, 68), QColor(58, 134, 74))  # Mown in stripes
END_ZONE = QColor(44, 104, 86)
LINE = QColor(238, 242, 238)
SHADOW = QColor(0, 0, 0, 70)
UNKNOWN_TEAM = QColor(160, 164, 170)
LEGS = QColor(28, 30, 36)
SKIN = QColor(224, 186, 150)
HOLDER = QColor(255, 214, 64)
DISC = QColor(250, 250, 250)
CAMERA = QColor(120, 190, 255)
LABEL_GROUND = QColor(16, 18, 22, 190)

STRIPE = 10.0  # Field units from one mown stripe to the next
BESIDE = 6.0  # How far the grass goes on beside the field
PLAYER_WIDTH = 0.55  # Field units
DISC_RADIUS = 0.45  # Drawn three times its size: a disc is a few pixels otherwise
HOLDER_RING = 1.3
TRAIL_SECONDS = 2.0
# Frames skipped, or gone back over, by more than this begin the trails anew
JUMP_FRAMES = 15

VIEWS = ("From the sideline", "From behind the end zone", "From above", "From the camera")


def _colour(bgr: Optional[Sequence[int]]) -> QColor:
    if bgr is None:
        return UNKNOWN_TEAM
    return QColor(int(bgr[2]), int(bgr[1]), int(bgr[0]))


class SceneView(QWidget):
    """The painted scene, and the eye that is moved around it with the mouse."""

    def __init__(self, template: FieldTemplate, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setFocusPolicy(Qt.StrongFocus)
        self.template = template
        self.scene: Optional[GameScene] = None
        self.message = "Play a video in the Main Analysis tab with tracking and the field on"
        self.facts: List[str] = []
        self.team_colour: Optional[QColor] = None  # Of the team in possession
        self.show_trails = True
        self.show_seen = True
        self.frame_rate = 30.0
        self._places: Dict[int, np.ndarray] = {}  # Player -> where they are drawn
        self._trails: Dict[int, deque] = {}  # Player -> (frame, x, y) of late
        self._disc_trail: deque = deque()
        self._last_frame: Optional[int] = None
        self._dragged_from: Optional[QPointF] = None
        self.view = self.view_named(VIEWS[0])

    # -- what is shown ------------------------------------------------------------

    def view_named(self, name: str) -> ViewPoint:
        """One of the ready-made points of view."""
        width, length = self.template.width, self.template.length
        middle = (width / 2, length / 2, 0.0)
        if name == VIEWS[3] and self.scene is not None and self.scene.camera is not None:
            looks_at = self.scene.looks_at or middle[:2]
            return ViewPoint.from_eye(self.scene.camera, (looks_at[0], looks_at[1], 0.0))
        if name == VIEWS[1]:
            return ViewPoint(middle, -90.0, 24.0, 1.05 * length)
        if name == VIEWS[2]:
            return ViewPoint(middle, 0.0, 89.0, 0.9 * length)
        return ViewPoint(middle, 0.0, 33.0, 0.8 * length)

    def show_scene(self, scene: Optional[GameScene], message: str = "") -> None:
        """Show a frame's scene; with None the last one stays, under a message."""
        self.message = message
        if scene is not None:
            self._follow(scene)
            self.scene = scene
        self.update()

    def _follow(self, scene: GameScene) -> None:
        last = self._last_frame
        if last is None or not 0 <= scene.frame_index - last <= JUMP_FRAMES:
            self._places.clear()
            self._trails.clear()
            self._disc_trail.clear()
        self._last_frame = scene.frame_index
        oldest = scene.frame_index - TRAIL_SECONDS * self.frame_rate
        places = {}
        for player in scene.players:
            places[player.player_id] = np.array(player.place)
            trail = self._trails.setdefault(player.player_id, deque())
            trail.append((scene.frame_index, *places[player.player_id]))
        self._places = places
        for player_id in list(self._trails):
            trail = self._trails[player_id]
            while trail and trail[0][0] < oldest:
                trail.popleft()
            if not trail:
                del self._trails[player_id]
        if scene.disc is not None:
            self._disc_trail.append((scene.frame_index, *scene.disc))
        while self._disc_trail and self._disc_trail[0][0] < oldest:
            self._disc_trail.popleft()

    # -- the mouse ----------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        self._dragged_from = event.localPos()

    def mouseReleaseEvent(self, event) -> None:
        self._dragged_from = None

    def mouseMoveEvent(self, event) -> None:
        if self._dragged_from is None:
            return
        moved = event.localPos() - self._dragged_from
        self._dragged_from = event.localPos()
        view = self.view
        if event.buttons() & Qt.LeftButton:
            view.azimuth -= 0.35 * moved.x()
            view.elevation = float(np.clip(view.elevation + 0.3 * moved.y(), 4.0, 89.0))
        else:
            # The spot looked at moves over the ground with the mouse
            right, _, ahead = view.axes
            along = np.array([ahead[0], ahead[1]])
            if np.linalg.norm(along) < 1e-6:
                along = np.array([-right[1], right[0]])
            along /= np.linalg.norm(along)
            step = view.distance / (view.focal * max(self.height(), 1))
            shift = -moved.x() * step * right[:2] + moved.y() * step * along
            limit = self.template.length
            view.target = (
                float(np.clip(view.target[0] + shift[0], -limit, limit)),
                float(np.clip(view.target[1] + shift[1], -limit, 2 * limit)),
                0.0,
            )
        self.update()

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        self.view.distance = float(np.clip(self.view.distance * 0.88**steps, 6.0, 400.0))
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        self.view = self.view_named(VIEWS[0])
        self.update()

    # -- painting -----------------------------------------------------------------

    def _size(self) -> Tuple[int, int]:
        return max(self.width(), 1), max(self.height(), 1)

    def _flat(self, painter: QPainter, corners: Sequence[Sequence[float]], closed=True) -> None:
        """Paint a polygon, or a line of several points, of the scene with the painter's
        pen and brush."""
        seen = clip_ahead(self.view.seen(corners), closed)
        if len(seen) < (3 if closed else 2):
            return
        points = QPolygonF([QPointF(x, y) for x, y in self.view.on_picture(seen, self._size())])
        if closed:
            painter.drawPolygon(points)
        else:
            painter.drawPolyline(points)

    def _point(self, place: Sequence[float]) -> Optional[Tuple[QPointF, float]]:
        """A place of the scene -> its pixel and how far ahead of the eye it is; None
        behind the eye."""
        seen = self.view.seen([place])
        if seen[0, 2] < 0.5:
            return None
        x, y = self.view.on_picture(seen, self._size())[0]
        return QPointF(x, y), float(seen[0, 2])

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        sky = QLinearGradient(0, 0, 0, self.height())
        sky.setColorAt(0.0, SKY_TOP)
        sky.setColorAt(1.0, SKY_BOTTOM)
        painter.fillRect(self.rect(), QBrush(sky))
        self._paint_field(painter)
        if self.scene is not None:
            self._paint_seen(painter)
            self._paint_trails(painter)
            self._paint_shadows(painter)
            self._paint_things(painter)
        self._paint_facts(painter)
        painter.end()

    def _paint_field(self, painter: QPainter) -> None:
        width, length, zone = self.template.width, self.template.length, self.template.end_zone
        painter.setPen(Qt.NoPen)
        painter.setBrush(SURROUND)
        self._flat(
            painter,
            [
                (-BESIDE, -BESIDE, 0),
                (width + BESIDE, -BESIDE, 0),
                (width + BESIDE, length + BESIDE, 0),
                (-BESIDE, length + BESIDE, 0),
            ],
        )
        start, stripe = 0.0, 0
        while start < length - 1e-9:
            end = min(start + STRIPE, length)
            in_zone = end <= zone + 1e-9 or start >= length - zone - 1e-9
            painter.setBrush(END_ZONE if in_zone else GRASS[stripe % 2])
            self._flat(painter, [(0, start, 0), (width, start, 0), (width, end, 0), (0, end, 0)])
            start, stripe = end, stripe + 1
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(LINE, 2.0))
        for first, second in self.template.lines.values():
            self._flat(painter, [(*first, 0), (*second, 0)], closed=False)
        for name in ("near_brick", "far_brick", "midfield"):
            x, y = self.template.points[name]
            self._flat(painter, [(x - 0.5, y - 0.5, 0), (x + 0.5, y + 0.5, 0)], closed=False)
            self._flat(painter, [(x - 0.5, y + 0.5, 0), (x + 0.5, y - 0.5, 0)], closed=False)

    def _paint_seen(self, painter: QPainter) -> None:
        """The ground the camera's picture shows, lit a little, with its edge drawn."""
        seen = self.scene.seen
        if not self.show_seen or seen is None or len(seen) < 3:
            return
        # As far as the grass goes: towards the horizon the camera sees for miles
        seen = within_rectangle(
            seen,
            (-BESIDE, -BESIDE),
            (self.template.width + BESIDE, self.template.length + BESIDE),
        )
        if len(seen) < 3:
            return
        outline = [(x, y, 0.015) for x, y in seen]
        painter.setPen(QPen(QColor(CAMERA.red(), CAMERA.green(), CAMERA.blue(), 170), 1.4))
        painter.setBrush(QColor(255, 255, 255, 26))
        self._flat(painter, outline)

    def _paint_trails(self, painter: QPainter) -> None:
        if not self.show_trails:
            return
        colours = {player.player_id: _colour(player.colour) for player in self.scene.players}
        painter.setBrush(Qt.NoBrush)
        for player_id, trail in self._trails.items():
            if len(trail) < 2 or player_id not in colours:
                continue
            colour = QColor(colours[player_id])
            colour.setAlpha(150)
            painter.setPen(QPen(colour, 2.0))
            self._flat(painter, [(x, y, 0.02) for _, x, y in trail], closed=False)
        if len(self._disc_trail) >= 2:
            painter.setPen(QPen(QColor(255, 255, 255, 170), 1.6, Qt.DashLine))
            self._flat(painter, [(x, y, z) for _, x, y, z in self._disc_trail], closed=False)

    def _paint_shadows(self, painter: QPainter) -> None:
        painter.setPen(Qt.NoPen)
        painter.setBrush(SHADOW)
        for player in self.scene.players:
            x, y = self._places.get(player.player_id, player.place)
            self._flat(painter, circle((x + 0.25, y + 0.15, 0.01), 0.6, 14))
        if self.scene.disc is not None and self.scene.disc_state != "held":
            x, y, _ = self.scene.disc
            self._flat(painter, circle((x, y, 0.01), DISC_RADIUS, 14))

    def _paint_things(self, painter: QPainter) -> None:
        """Players, disc and camera, the furthest first so that the near ones cover them."""
        things: List[Tuple[float, Callable[[], None]]] = []
        for player in self.scene.players:
            x, y = self._places.get(player.player_id, player.place)
            feet = self._point((x, y, 0.0))
            if feet is not None:
                things.append(
                    (feet[1], lambda p=player, at=(x, y): self._paint_player(painter, p, at))
                )
        disc = self.scene.disc
        if disc is not None:
            if self.scene.disc_state == "held":
                # In the holder's hand: beside them as the eye sees it
                disc = tuple(np.array(disc) + 0.45 * self.view.axes[0])
            at = self._point(disc)
            if at is not None:
                things.append((at[1] - 0.3, lambda d=disc: self._paint_disc(painter, d)))
        if self.scene.camera is not None:
            at = self._point(self.scene.camera)
            if at is not None:
                things.append((at[1], lambda: self._paint_camera(painter)))
        for _, paint in sorted(things, key=lambda thing: -thing[0]):
            paint()

    def _paint_player(
        self, painter: QPainter, player: ScenePlayer, at: Tuple[float, float]
    ) -> None:
        height = player_height(self.template)
        x, y = at
        marks = [self._point((x, y, share * height)) for share in (0.0, 0.47, 0.84, 0.93)]
        if any(mark is None for mark in marks):
            return
        (feet, ahead), (hip, _), (shoulder, _), (head, _) = marks
        unit = self.view.pixels_per_unit(ahead, self._size())
        thick = max(PLAYER_WIDTH * unit, 3.0)
        colour = _colour(player.colour)
        if player.holding:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(HOLDER, max(0.18 * unit, 2.0)))
            self._flat(painter, circle((x, y, 0.03), HOLDER_RING, 28))
        painter.setPen(QPen(LEGS, 0.8 * thick, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(feet, hip)
        painter.setPen(QPen(colour.darker(160), thick + 2.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(hip, shoulder)
        painter.setPen(QPen(colour, thick, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(hip, shoulder)
        radius = max(0.22 * height * unit / 2, 2.0)
        painter.setPen(QPen(LEGS, 1.0))
        painter.setBrush(SKIN)
        painter.drawEllipse(head + QPointF(0, -radius * 0.6), radius, radius)
        text = player.number or (f"{player.player_id}" if unit > 9 else "")
        if player.observer:
            text = "observer" if unit > 9 else ""
        if text:
            self._label(
                painter,
                head + QPointF(0, -2.6 * radius - 8),
                f"#{text}" if player.number else text,
                HOLDER if player.holding else QColor(240, 240, 240),
                bold=bool(player.number),
            )

    def _paint_disc(self, painter: QPainter, disc: Sequence[float]) -> None:
        if disc[2] > 0.05 and self.scene.disc_state != "held":
            painter.setPen(QPen(QColor(255, 255, 255, 110), 1.0, Qt.DotLine))
            self._flat(painter, [(disc[0], disc[1], 0.0), disc], closed=False)
        painter.setPen(QPen(QColor(40, 40, 40), 1.2))
        painter.setBrush(HOLDER if self.scene.disc_state == "held" else DISC)
        self._flat(painter, circle(disc, DISC_RADIUS, 22))

    def _paint_camera(self, painter: QPainter) -> None:
        camera = self.scene.camera
        at = self._point(camera)
        painter.setPen(
            QPen(QColor(CAMERA.red(), CAMERA.green(), CAMERA.blue(), 120), 1.0, Qt.DashLine)
        )
        self._flat(painter, [(camera[0], camera[1], 0.0), camera], closed=False)
        if self.scene.looks_at is not None:
            self._flat(painter, [camera, (*self.scene.looks_at, 0.0)], closed=False)
        painter.setPen(QPen(QColor(20, 30, 40), 1.0))
        painter.setBrush(CAMERA)
        painter.drawRect(QRectF(at[0].x() - 6, at[0].y() - 4, 12, 8))
        self._label(painter, at[0] + QPointF(0, -16), "camera", CAMERA)

    def _label(
        self, painter: QPainter, middle: QPointF, text: str, colour: QColor, bold=False
    ) -> None:
        font = QFont(self.font())
        font.setPointSizeF(8.5)
        font.setBold(bold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        box = QRectF(0, 0, metrics.horizontalAdvance(text) + 8, metrics.height() + 2)
        box.moveCenter(middle)
        painter.setPen(Qt.NoPen)
        painter.setBrush(LABEL_GROUND)
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(colour)
        painter.drawText(box, Qt.AlignCenter, text)

    def _paint_facts(self, painter: QPainter) -> None:
        font = QFont(self.font())
        font.setPointSizeF(10.5)
        painter.setFont(font)
        line = painter.fontMetrics().height() + 4
        if self.facts:
            widest = max(painter.fontMetrics().horizontalAdvance(fact) for fact in self.facts)
            box = QRectF(12, 12, widest + 40, line * len(self.facts) + 12)
            painter.setPen(Qt.NoPen)
            painter.setBrush(LABEL_GROUND)
            painter.drawRoundedRect(box, 6, 6)
            if self.team_colour is not None:
                painter.setBrush(self.team_colour)
                painter.drawRoundedRect(QRectF(20, 22, 8, line * len(self.facts) - 8), 3, 3)
            painter.setPen(QColor(236, 238, 242))
            for row, fact in enumerate(self.facts):
                painter.drawText(QPointF(36, 12 + 6 + line * (row + 1) - 6), fact)
        if self.message:
            painter.setPen(QColor(236, 238, 242))
            box = QRectF(0, self.height() - 44, self.width(), 32)
            painter.fillRect(box, LABEL_GROUND)
            painter.drawText(box, Qt.AlignCenter, self.message)


class SceneTab(QWidget):
    """The view with its few controls. Fed by the Main Analysis tab (`show_result`)."""

    play_pause_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        ruleset = get_setting("homography.ruleset", DEFAULT_RULESET)
        self.template = TEMPLATES.get(ruleset, TEMPLATES[DEFAULT_RULESET])
        self.scene_view = SceneView(self.template)

        self.play_button = QPushButton("Play / pause")
        self.play_button.setToolTip("Plays the video of the Main Analysis tab (Space)")
        self.play_button.clicked.connect(self.play_pause_requested)
        self.view_combo = QComboBox()
        self.view_combo.addItems(VIEWS)
        self.view_combo.setToolTip(
            "Where the scene is looked at from. From the camera: from where the camera "
            "that films the game was worked out to be."
        )
        self.view_combo.activated[str].connect(self._on_view_chosen)
        self.trails_checkbox = QCheckBox("Trails")
        self.trails_checkbox.setChecked(True)
        self.trails_checkbox.setToolTip("Where each player and the disc went in the last 2 seconds")
        self.trails_checkbox.toggled.connect(self._on_trails_toggled)
        self.seen_checkbox = QCheckBox("Camera's view")
        self.seen_checkbox.setChecked(True)
        self.seen_checkbox.setToolTip("The ground the camera's picture shows, outlined")
        self.seen_checkbox.toggled.connect(self._on_seen_toggled)
        hint = QLabel("Drag: turn    Right-drag: move    Wheel: closer    Double-click: back")
        hint.setStyleSheet(f"color: {FAINT_TEXT};")

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(QLabel("View"))
        controls.addWidget(self.view_combo)
        controls.addWidget(self.trails_checkbox)
        controls.addWidget(self.seen_checkbox)
        controls.addStretch(1)
        controls.addWidget(hint)
        layout = QVBoxLayout(self)
        layout.addWidget(self.scene_view, 1)
        layout.addLayout(controls)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self.play_pause_requested.emit()
        else:
            super().keyPressEvent(event)

    def set_frame_rate(self, rate: float) -> None:
        if rate > 0:
            self.scene_view.frame_rate = rate

    def show_result(self, result) -> None:
        """Show what the pipeline found in a frame (FrameResult). Does nothing while the
        tab is not looked at."""
        if not self.isVisible():
            return
        scene = scene_of(result, self.template)
        view = self.scene_view
        view.facts = _facts(result, scene)
        view.team_colour = (
            _colour(result.possession_colour) if result.possession_colour is not None else None
        )
        if scene is not None:
            message = ""
        elif not result.wide_shot:
            message = "A close-up: nothing is analysed"
        else:
            message = "Where the field lies is not known in this frame"
        view.show_scene(scene, message)

    def _on_view_chosen(self, name: str) -> None:
        self.scene_view.view = self.scene_view.view_named(name)
        self.scene_view.update()

    def _on_seen_toggled(self, on: bool) -> None:
        self.scene_view.show_seen = on
        self.scene_view.update()

    def _on_trails_toggled(self, on: bool) -> None:
        self.scene_view.show_trails = on
        self.scene_view.update()


def _facts(result, scene: Optional[GameScene]) -> List[str]:
    """The lines written over the scene: the phase of the game, the disc, the score."""
    facts = [result.game_state.capitalize()]
    if result.point_seconds is not None:
        minutes, seconds = divmod(int(result.point_seconds), 60)
        facts[0] += f"  ·  point {result.points_begun}, {minutes}:{seconds:02d}"
    if result.holder_id is not None:
        number = str(result.player_ids.get(result.holder_id, ("", None))[0] or "")
        who = f"#{number}" if number.isdigit() else f"player {result.holder_id}"
        facts.append(f"Disc held by {who}")
    elif result.disc_state == "ground":
        facts.append("Disc on the ground")
    elif result.flight_seconds is not None:
        facts.append(f"Disc in the air for {result.flight_seconds:.1f} s")
    else:
        facts.append("Disc in the air or not seen")
    if result.score:
        facts.append("Score  " + "  :  ".join(str(points) for _, points in result.score))
    if scene is not None:
        observers = sum(1 for player in scene.players if player.observer)
        said = f"{len(scene.players) - observers} players on the field"
        if observers:
            said += f", {observers} observer{'s' if observers > 1 else ''}"
        facts.append(said)
    return facts
