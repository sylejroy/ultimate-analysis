"""The Simulation tab: two simulated teams play a point, in 3D or from above.

Each team is the scripted one or a team of learned players (a folder under
data/models/movement with a minds.pt: copied from the recorded players by
scripts/train_movement.py, changed by playing by scripts/evolve_teams.py, or trained by
reinforcement by scripts/train_ppo.py; or with a spots.pt: players who learned from
nothing where to go and where to throw, scripts/train_spots.py and train_spots_gpu.py). Play, pause, step a tenth of a second, start a
new point; the wind of a new point blows from any direction, as strong as set.

In 3D the field is the 3D View's: drag to turn it, right-drag to move it, the wheel to
come closer. From above, the dangerous space for the team with the disc is shaded.
Clicking a player in 3D shows their brain beside the field: what they know of the game
and the path they ran; for a player who chooses where to run, what they are weighing
now, drawn on the field around them; and their network as it works in the present step.
"""

from typing import List, Optional

import numpy as np
from PyQt5.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...simulation import game
from ...simulation.drawing import (
    NAMES,
    call_of,
    caption_of,
    close_up,
    knowledge_picture,
    picture_of,
    scene_of_game,
    team_of,
)
from ...simulation.learned import INPUTS, activations, weights_of
from ...simulation.match import Match
from ...utils.field_template import TEMPLATES
from ...utils.logger import get_logger
from ...utils.model_files import models_root
from ..scene.scene_tab import SceneView
from ..widgets.controls import ToggleSwitch
from ..widgets.images import frame_to_pixmap
from ..widgets.panels import compact_combo, side_panel
from .brain_view import NetworkView

logger = get_logger("GUI")

SPEEDS = (
    ("Half speed", 0.5),
    ("As fast as the game", 1.0),
    ("Twice as fast", 2.0),
    ("Four times", 4.0),
)
PICK_REACH = 25  # Pixels from a player's figure that a click picks them
TRAIL = 150  # Steps of the picked player's path kept to draw: 15 seconds


def teams_available() -> List[tuple]:
    """(label, kind) of every team one can pick: the scripted one, then the latest of
    each learned run whose networks see what the players see now, newest first."""
    import torch

    found = [("Scripted", "scripted"), ("Scripted 2 (plays as recorded teams)", "scripted2")]
    root = models_root() / "movement"
    if root.is_dir():
        # Teams that choose spots: the latest of each run, newest first
        runs = sorted(root.glob("*/latest/spots.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        from ...simulation.spot_features import SPOT_FEATURES
        from ...simulation.spots import HIDDEN

        for spots in runs:
            try:
                kept = torch.load(spots, map_location="cpu", weights_only=True)
                rows, inputs = kept["spots"]["offence"]["0.weight"].shape
                if rows != HIDDEN or inputs != SPOT_FEATURES:
                    continue  # Made for what players chose between, or saw of it, before
            except Exception:  # noqa: BLE001 (a file that does not load is not offered)
                continue
            found.append((f"{spots.parent.parent.name} (chooses spots)", str(spots.parent)))
        newest_first = sorted(root.rglob("minds.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        # A run's latest only, not every checkpoint it kept on the way
        newest_first = [
            m for m in newest_first if m.parent.parent == root or m.parent.name == "latest"
        ]
        for minds in newest_first:
            try:
                kept = torch.load(minds, map_location="cpu", weights_only=True)
                if kept["offence"]["0.weight"].shape[1] > INPUTS:
                    continue  # Made for inputs these players do not have
            except Exception:  # noqa: BLE001 (a file that does not load is not offered)
                continue
            folder = minds.parent
            name = folder.name if folder.parent == root else f"{folder.parent.name} / {folder.name}"
            found.append((name, str(folder)))
    return found


class SimulationView(SceneView):
    """The 3D view of the game, where a click on a player picks them."""

    picked = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__(TEMPLATES["usau"])
        self.frame_rate = 1.0 / game.STEP
        self.message = "Press New point"
        self._pressed_at: Optional[QPointF] = None

    def mousePressEvent(self, event) -> None:
        self._pressed_at = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if self._pressed_at is not None and (event.pos() - self._pressed_at).manhattanLength() < 5:
            self._pick(QPointF(event.pos()))
        self._pressed_at = None

    def _pick(self, at: QPointF) -> None:
        """The player whose figure is nearest to a click, if near enough."""
        if self.scene is None:
            return
        best, nearest = None, float(PICK_REACH)
        for player in self.scene.players:
            x, y = self._places.get(player.player_id, player.place)
            for height in (0.0, 1.0, 1.8):
                point = self._point((x, y, height))
                if point is None:
                    continue
                distance = float(np.hypot(point[0].x() - at.x(), point[0].y() - at.y()))
                if distance < nearest:
                    best, nearest = player.player_id, distance
        if best is not None:
            self.picked.emit(best)


class BrainPanel(QWidget):
    """What a picked player knows, and their network at work."""

    def __init__(self) -> None:
        super().__init__()
        self.title = QLabel("Click a player in 3D to look at their brain")
        self.title.setWordWrap(True)
        self.knowledge = QLabel()
        self.knowledge.setAlignment(Qt.AlignCenter)
        self.knowledge.setToolTip(
            "What they know, from above with their team attacking upwards: solid where\n"
            "they see a player, a ring as wide as they are unsure where they remember one,\n"
            "faint where they have lost one; themselves ringed in yellow, the path they ran\n"
            "behind them; the cross where they judge the disc comes down. A player who\n"
            "chooses where to run: the whole field shaded by how likely each place (the\n"
            "holder: each throw) is, and framed, the part shown large beside"
        )
        self.network = NetworkView()
        self.close = QLabel()
        self.close.setAlignment(Qt.AlignCenter)
        self.close.setMinimumHeight(320)
        self.close.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.close.setToolTip(
            "The field around them, their team attacking upwards. Shaded: how likely they\n"
            "are to run to each place (the holder: to throw there). The yellow arrow: where\n"
            "they run now, and how likely that was; the rings: what else they weigh, the\n"
            "wider the likelier. The blue line behind them: the path they ran. Everyone\n"
            "has an arrow to where they will be in half a second; the opponent they are\n"
            "matched with is ringed in blue; the cross is the disc"
        )
        self.pages = QStackedWidget()
        self.pages.addWidget(self.close)
        self.pages.addWidget(self.network)
        self.pages.setCurrentWidget(self.network)
        self.show_network = QPushButton("Show the network")
        self.show_network.setCheckable(True)
        self.show_network.setToolTip(
            "The network that scores each choice, at work on the one taken: what goes in,\n"
            "its hidden layers, and the connections that carry the most now"
        )
        self.show_network.toggled.connect(self._page)
        self.show_network.setVisible(False)
        self._trail: list = []  # (seconds, place as their team sees the field)
        self._trail_of: Optional[tuple] = None  # The player and the point it is of
        self.said = QLabel()
        self.said.setWordWrap(True)
        self.said.setVisible(False)  # Shown for the players whose network says something
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(self.title, 1)
        top.addWidget(self.show_network)
        layout.addLayout(top)
        row = QHBoxLayout()
        row.addWidget(self.pages, 1)
        row.addWidget(self.knowledge)
        layout.addLayout(row, 1)
        layout.addWidget(self.said)

    def _page(self, _checked: bool = False) -> None:
        """The choices in words, or the network, for a player who chooses."""
        chooses = self.show_network.isVisible() and not self.show_network.isChecked()
        self.pages.setCurrentWidget(self.close if chooses else self.network)

    def _follow(self, match: Match, player: int, view) -> list:
        """The path the picked player ran in this point, with where they are now."""
        if self._trail_of != (player, id(match)):
            self._trail_of, self._trail = (player, id(match)), []
        if not self._trail or view.seconds > self._trail[-1][0]:
            self._trail.append((view.seconds, view.ours[view.me, :2].copy()))
            del self._trail[:-TRAIL]
        return [place for _, place in self._trail]

    def show_player(self, match: Match, teams, player: int) -> None:
        team, k = game.team_of(player), player % game.PLAYERS
        views = match.views[team] if match.views and match.views[team] else None
        side = "offence" if match.state.offence == team else "defence"
        self.title.setText(f"{NAMES[team]} {k + 1}, on {side}")
        if views is None:
            self.network.clear("Play a step first")
            return
        view = views[k]
        trail = self._follow(match, player, view)
        chooses = hasattr(getattr(teams[team], "brains", None), "spots")
        self.show_network.setVisible(chooses)
        self.said.setVisible(not chooses)
        self._page()
        if chooses:
            self._team = team
            self._show_spots(teams[team], view, k, side, trail)
            return
        self.knowledge.setPixmap(frame_to_pixmap(knowledge_picture(view, team, trail=trail)))
        brain = getattr(teams[team], "minds", None) or getattr(
            getattr(teams[team], "brains", None), "minds", None
        )
        if brain is None:
            self.network.clear("A scripted player: rules, not a network")
            self.said.setText("")
            return
        policy = getattr(teams[team], "brains", None)
        network = (
            getattr(policy, side).movement
            if policy is not None
            else (brain.offence if side == "offence" else brain.defence)
        )
        from ...simulation.learned import view_features

        layers = activations(network, view_features(view))
        masks = getattr(brain, "masks", None)
        self.network.show_brain(layers, weights_of(network), None if not masks else masks.get(side))
        known_mates = int((view.ours_unsure < 3.0).sum()) - 1
        known_opponents = int((view.theirs_unsure < 3.0).sum())
        out = layers[-1]
        wired = [w != 0 for w in weights_of(network)]
        share = sum(int(w.sum()) for w in wired) / sum(w.size for w in wired)
        self.said.setText(
            f"Wired: {share:.0%} of the connections there could be. "
            f"Knows where {known_mates} of 6 teammates and {known_opponents} of 7 opponents are. "
            f"Changes how they run by ({out[0] * 8:+.1f}, {out[1] * 8:+.1f}) yd/s; says "
            + ", ".join(f"{v:+.2f}" for v in np.tanh(out[2:]))
        )

    def _show_spots(self, team, view, k: int, side: str, trail: list) -> None:
        """A player who chooses where to run: the network that scores each choice (the
        holder: holding on or each throw), at work on the one they took, with the
        choices it scores highest as its outputs, numbered as on their view."""
        import torch

        from ...simulation import spot_features as sf
        from ...simulation.spots import THROW_PRIOR
        from . import choices as said

        holder = view.attacking and view.holder == k
        with torch.no_grad():
            if holder:
                features = sf.throw_choices(sf.views_of([view]))[0]
                network = team.brains.throws
                scores = network(features)[:, 0].numpy() + THROW_PRIOR
            else:
                plan = team.plans[k].choice if k < len(team.plans) else None
                age = view.seconds - team.plans[k].decided_at if plan is not None else 0.0
                seeing = sf.views_of([view], [plan], ages=[age])
                found = sf.spot_choices(seeing)
                leans = sf.spot_leans(seeing, found[4])[0]
                features, kinds, players, params, allowed = (each[0] for each in found)
                network = team.brains.spots[side]
                scores = (network(features)[:, 0] + leans).numpy()
        chances = np.exp(scores - scores.max())
        chances = chances / chances.sum()
        rows = features.numpy()
        me = view.ours[view.me, :2]
        if holder:
            taken = int(np.argmax(chances))  # Drawn afresh every 0.3 s: the likeliest
            listed = said.ranked(chances, taken)
            labels = [said.throw_label(view, i) for i in listed]
            aims = sf.throw_aims(sf.views_of([view]))[0].numpy()
            places = [None if i == 0 else aims[i - 1] for i in listed]
            heat = (aims, chances[1:])
            notes = said.throw_notes(rows[taken])
            groups = sf.THROW_GROUPS
        else:
            where = _spot_places(view, kinds.numpy(), players.numpy(), params.numpy())
            # Where they run to now is the choice of going on, while they may
            taken = sf.KEEP if bool(allowed[sf.KEEP]) else int(np.argmax(chances))
            listed = said.ranked(chances, taken)
            labels = [
                said.spot_label(i, int(players[i]), params[i].numpy(), where[i] - me)
                for i in listed
            ]
            places = [where[i] for i in listed]
            heat = (where, chances)
            notes = said.spot_notes(rows[taken])
            groups = sf.SPOT_GROUPS
        options = [(label, float(chances[i])) for label, i in zip(labels, listed)]
        considered = [(place, chance) for place, (_, chance) in zip(places, options)]
        # The whole field small, with the part shown large framed
        whole = knowledge_picture(view, self._team, heat=heat, trail=trail, window=True)
        self.knowledge.setPixmap(frame_to_pixmap(whole))
        near = close_up(view, self._team, considered=considered, heat=heat, trail=trail)
        self.close.setPixmap(
            frame_to_pixmap(near).scaled(
                self.close.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        layers = activations(network, rows[taken])
        self.network.show_brain(
            layers, weights_of(network), None, groups, ("Score",), notes, options
        )
        self.said.setText("")
        self.said.setToolTip(
            "The holder chooses every 0.3 s between holding on and a throw: to any of 481 "
            "places over the field, or to a teammate, led."
            if holder
            else "They choose again "
            + ("every second on the way" if view.attacking else "every half second")
            + f", between {int(allowed.sum())} things: staying, going on, a cut, a place by "
            "the disc, a spot beside any other player."
        )


def _spot_places(view, kinds, players, params) -> np.ndarray:
    """Where each choice of spot_features.spot_choices is now, as the team sees the field."""
    from ...simulation import spot_features as sf

    places = np.array(params, dtype=float)
    them = kinds == sf.OPPONENT
    places[them] += view.theirs[players[them], :2]
    mates = kinds == sf.TEAMMATE
    places[mates] += view.ours[players[mates], :2]
    return places


class SimulationTab(QWidget):
    """Two simulated teams playing, with the controls to watch them."""

    def __init__(self) -> None:
        super().__init__()
        self.rng = np.random.default_rng()
        self.score = [0, 0]
        self.match: Optional[Match] = None
        self.teams: tuple = ()
        self.picked: Optional[int] = None
        self.caption, self._shown_for, self._events_seen = "", 0, 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step)

        controls = QWidget()
        column = QVBoxLayout(controls)
        teams = QGroupBox("Teams")
        form = QFormLayout()
        self.team_combos = []
        offered = teams_available()
        for name in NAMES:
            combo = compact_combo(QComboBox())
            for label, kind in offered:
                combo.addItem(label, kind)
            combo.setToolTip(
                "The scripted team, or a team of learned players: copied from the recorded\n"
                "players (scripts/train_movement.py), changed by playing\n"
                "(scripts/evolve_teams.py) or trained by reinforcement (scripts/train_ppo.py).\n"
                "Takes effect at the next point."
            )
            side = "attacks right" if name == "White" else "attacks left"
            form.addRow(QLabel(f"{name} ({side})"), combo)
            self.team_combos.append(combo)
        teams.setLayout(form)
        column.addWidget(teams)

        play = QGroupBox("Play")
        buttons = QVBoxLayout()
        row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        step_button = QPushButton("Step")
        step_button.setToolTip("A tenth of a second")
        step_button.clicked.connect(self.step)
        new_button = QPushButton("New point")
        new_button.clicked.connect(self.new_point)
        for button in (self.play_button, step_button, new_button):
            row.addWidget(button)
        buttons.addLayout(row)
        self.speed_combo = compact_combo(QComboBox())
        for label, speed in SPEEDS:
            self.speed_combo.addItem(label, speed)
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._set_speed)
        buttons.addWidget(self.speed_combo)
        wind_row = QHBoxLayout()
        wind_row.addWidget(QLabel("Wind"))
        self.wind_slider = QSlider(Qt.Horizontal)
        self.wind_slider.setRange(0, int(game.STRONGEST_WIND))
        self.wind_slider.setValue(3)
        self.wind_label = QLabel()
        self.wind_slider.valueChanged.connect(self._show_wind)
        self.wind_slider.setToolTip(
            "The strongest wind of a new point, in yards a second; it blows from any direction"
        )
        wind_row.addWidget(self.wind_slider)
        wind_row.addWidget(self.wind_label)
        buttons.addLayout(wind_row)
        self.from_above = ToggleSwitch("From above, with the dangerous space")
        self.from_above.setToolTip(
            "The field from above, the dangerous space for the team with the disc shaded\n"
            "(processing/open_space.py); off: in 3D, where a click on a player shows their brain"
        )
        self.from_above.stateChanged.connect(self._switch_view)
        buttons.addWidget(self.from_above)
        play.setLayout(buttons)
        column.addWidget(play)

        happened = QGroupBox("What happened")
        log = QVBoxLayout()
        self.events = QListWidget()
        log.addWidget(self.events)
        happened.setLayout(log)
        column.addWidget(happened, 1)

        self.view3d = SimulationView()
        self.view3d.picked.connect(self._pick)
        self.flat = QLabel("Press New point")
        self.flat.setAlignment(Qt.AlignCenter)
        self.flat.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.field = QStackedWidget()
        self.field.addWidget(self.view3d)
        self.field.addWidget(self.flat)
        self.brain = BrainPanel()

        right = QVBoxLayout()
        right.addWidget(self.field, 3)
        right.addWidget(self.brain, 2)
        layout = QHBoxLayout(self)
        layout.addWidget(side_panel(controls))
        layout.addLayout(right, 1)
        self._show_wind(self.wind_slider.value())

    # ------------------------------------------------------------------ playing

    def new_point(self) -> None:
        """Start a point with the teams picked: they take turns to receive the pull."""
        self.teams = tuple(
            team_of(combo.currentData(), int(self.rng.integers(1 << 30)), side)
            for side, combo in enumerate(self.team_combos)
        )
        unfinished = self.match is not None and self.match.state.scored is None
        offence = (sum(self.score) + unfinished) % 2
        start = game.pull_start(
            self.rng, offence, game.a_wind(self.rng, float(self.wind_slider.value()))
        )
        self.match = Match(self.teams, self.rng, start)
        self.caption, self._shown_for, self._events_seen = "The pull", 30, 0
        wind = float(np.hypot(*start.wind))
        self.events.addItem(f"New point: {NAMES[offence]} receive the pull, wind {wind:.0f} yd/s")
        self.redraw()

    def step(self) -> None:
        """Play a tenth of a second."""
        if self.match is None:
            self.new_point()
            return
        state = self.match.state
        if state.over:
            self.timer.stop()
            self.play_button.setText("Play")
            return
        self.match.step()
        for event in state.events[self._events_seen :]:
            said = caption_of(event)
            if said:
                self.caption, self._shown_for = said, 25
                self.events.addItem(f"{state.seconds:5.1f} s  {said}")
                self.events.scrollToBottom()
        self._events_seen = len(state.events)
        call = call_of(state, self.match.decisions)
        if call and self._shown_for <= 0:
            self.caption, self._shown_for = call, 20
        self._shown_for -= 1
        if state.over and state.scored is not None:
            self.score[state.scored] += 1
        self.redraw()

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            if self.match is None or self.match.state.over:
                self.new_point()
            self._set_speed()
            self.timer.start()
            self.play_button.setText("Pause")

    def _set_speed(self, _index: int = 0) -> None:
        self.timer.setInterval(int(1000 * game.STEP / self.speed_combo.currentData()))

    def _show_wind(self, value: int) -> None:
        self.wind_label.setText(f"up to {value} yd/s")

    def _switch_view(self, _state: int) -> None:
        self.field.setCurrentIndex(1 if self.from_above.isChecked() else 0)
        self.redraw()

    def _pick(self, player: int) -> None:
        self.picked = player
        self.redraw()

    def redraw(self) -> None:
        """Show the game as it stands, and the picked player's brain."""
        if self.match is None:
            return
        state = self.match.state
        if self.from_above.isChecked():
            picture = picture_of(state, self.score, self.caption, True)
            self.flat.setPixmap(
                frame_to_pixmap(picture).scaled(
                    self.flat.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        else:
            score = f"White {self.score[0]} - {self.score[1]} Red"
            self.view3d.facts = [score, f"{state.seconds:.1f} s", self.caption]
            self.view3d.show_scene(scene_of_game(state))
        if self.picked is not None:
            self.brain.show_player(self.match, self.teams, self.picked)
