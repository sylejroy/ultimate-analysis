"""Two ways of labelling that take one key per answer.

"Same player?" shows two tracks and asks whether they are one player; "Holding the
disc?" shows a player and asks whether they hold the disc, or takes a click on the one
who does where a box shows two players. What is asked and how the
answers are kept is in utils/judgement_files.py; this is the windows.
"""

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from ...config.settings import get_setting
from ...constants import DEFAULT_PATHS
from ...processing.model_lock import GPU_SETUP_LOCK
from ...utils import judgement_files
from ...utils.enlarging import enlarged
from ...utils.logger import get_logger
from ..theme import FAINT_TEXT
from ..widgets.images import frame_to_pixmap
from ..widgets.model_selection import populate_holder_model_combo

logger = get_logger("LABELLING")

PAIRS_SOURCE, PAIRS_ANSWERS = "reid_players_v2", "labelled_pairs_v1"
HOLDER_SOURCE, HOLDER_ANSWERS = "holder_sighted_v1", "labelled_holders_v1"
TRACK_CROPS = 8  # Crops shown of each track
CROP_HEIGHT = 250  # ... at this height
PLAYER_HEIGHT = 560  # The height the player asked about is shown at
PLAYER_PAD = 0.35  # ... with this share of their box around it: a disc is held out
ANSWERS_SHOWN = 10  # The last answers that can be taken back
ANSWER_HEIGHT = 130  # ... each shown as the player's picture at this height
SAID_YES, SAID_NO = (90, 200, 90), (80, 80, 230)  # The frame around it (BGR)
SAID_TWO = (60, 200, 230)  # ... and around a box said to show two players
# Holders said by hand to aim for before the next network is taught: with 79 of them a
# network learned next to nothing (10 Oct 2026), and one taught on the dataset's own
# 13,800, a quarter of them wrong, found most. A first goal, not a measured need
HOLDER_GOAL = 1000
GROUND = (34, 31, 30)  # Behind the pictures (BGR): the window's


def data_folder() -> Path:
    return Path(DEFAULT_PATHS["TRAINING_DATA"])


def faint(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {FAINT_TEXT};")
    label.setWordWrap(True)
    return label


class _Judging(QWidget):
    """What both have: where the answers come from and go to, the pictures, a row of
    answers each with its key, and a line saying how far along the work is."""

    def __init__(self, source: str, answers: str, file_name: str):
        super().__init__()
        self._file_name = file_name
        self._loaded = False
        self.rng = random.Random()

        self.source_edit, self.answers_edit = QLineEdit(source), QLineEdit(answers)
        for edit in (self.source_edit, self.answers_edit):
            edit.setFixedWidth(220)
            edit.editingFinished.connect(self._load_and_ask)
        top = QHBoxLayout()
        top.addWidget(faint("Asked about the dataset"))
        top.addWidget(self.source_edit)
        top.addWidget(faint("answers kept in"))
        top.addWidget(self.answers_edit)
        top.addStretch(1)

        self.pictures = QVBoxLayout()
        self.buttons = QHBoxLayout()
        self.status = faint("")
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(self.pictures, 1)
        layout.addLayout(self.buttons)
        layout.addWidget(self.status)

    def add_answer(self, text: str, key: str, answer: Callable[[], None]) -> None:
        button = QPushButton(f"{text}  [{key}]")
        button.setFixedHeight(40)
        button.clicked.connect(answer)
        self.buttons.addWidget(button)
        # Only while this page is the one shown: the other pages have keys of their own
        shortcut = QShortcut(key, self)
        shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        shortcut.activated.connect(answer)

    @property
    def source(self) -> Path:
        return data_folder() / self.source_edit.text().strip()

    @property
    def answers(self) -> Path:
        return data_folder() / self.answers_edit.text().strip() / self._file_name

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._loaded:
            self._load_and_ask()
        self.setFocus()

    def _load_and_ask(self) -> None:
        self._loaded = True
        try:
            self.load()
        except (OSError, KeyError, ValueError) as error:
            logger.warning(f"Cannot read {self.source}: {error}")
            self.status.setText(f"Cannot read {self.source}: {error}")
            return
        self.ask()

    def load(self) -> None:
        raise NotImplementedError

    def ask(self) -> None:
        raise NotImplementedError


class TrackVectors(QThread):
    """Works out, beside the window, how the network that tells players apart sees every
    track of a dataset: one vector each, from the pictures that are shown of it. Kept in
    data/cache, since reading eight pictures of two thousand tracks takes a while."""

    worked_out = pyqtSignal(object)  # Track -> vector; empty without a network

    def __init__(self, source: Path, tracks: Dict[judgement_files.TrackKey, judgement_files.Track]):
        super().__init__()
        self.source, self.tracks = source, tracks

    def run(self) -> None:
        try:
            self.worked_out.emit(self._vectors())
        except Exception as error:
            logger.warning(f"Pairs are drawn at random, the tracks' vectors failed: {error}")
            self.worked_out.emit({})

    def _vectors(self) -> dict:
        weights = Path(get_setting("models.reid.weights", ""))
        if not weights.is_file():
            return {}
        kept = Path("data/cache/track_vectors") / f"{self.source.name}__{weights.parent.name}.npz"
        names = ["|".join(key) for key in self.tracks]
        if kept.exists():
            saved = np.load(kept)
            if list(saved["tracks"]) == names:
                return dict(zip(self.tracks, saved["vectors"]))
        from ...processing.reid import embed, load_embedder, track_vector

        model = load_embedder(weights)
        vectors = []
        for track in self.tracks.values():
            crops = [cv2.imread(str(path)) for path in track.spread(TRACK_CROPS)]
            crops = [crop for crop in crops if crop is not None]
            # The network shares the GPU with the analysis of the main tab
            with GPU_SETUP_LOCK.work():
                vectors.append(track_vector(embed(model, crops)))
        kept.parent.mkdir(parents=True, exist_ok=True)
        np.savez(kept, tracks=np.array(names), vectors=np.array(vectors))
        return dict(zip(self.tracks, vectors))


class CropPicture(QLabel):
    """A picture of a track. A click takes it out (it shows somebody else than the
    rest of the track) or puts it back."""

    def __init__(self, frame: int, path: Path):
        super().__init__()
        self.frame = frame
        self.taken_out = False
        crop = cv2.imread(str(path))
        if crop is None:
            crop = np.full((CROP_HEIGHT, CROP_HEIGHT // 2, 3), GROUND, dtype=np.uint8)
        width = max(1, round(crop.shape[1] * CROP_HEIGHT / crop.shape[0]))
        self._crop = cv2.resize(crop, (width, CROP_HEIGHT), interpolation=cv2.INTER_CUBIC)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click if this is not the player of the rest of the track")
        self._draw()

    def _draw(self) -> None:
        shown = self._crop
        if self.taken_out:
            # Dimmed and crossed out
            shown = (shown * 0.3).astype(np.uint8)
            height, width = shown.shape[:2]
            cv2.line(shown, (4, 4), (width - 5, height - 5), (70, 70, 230), 3, cv2.LINE_AA)
            cv2.line(shown, (width - 5, 4), (4, height - 5), (70, 70, 230), 3, cv2.LINE_AA)
        self.setPixmap(frame_to_pixmap(shown))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.taken_out = not self.taken_out
            self._draw()


class CropRow(QWidget):
    """The pictures of a track side by side."""

    def __init__(self):
        super().__init__()
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._layout.addStretch(1)
        self.pictures: List[CropPicture] = []

    def show_crops(self, crops: List[Tuple[int, Path]]) -> None:
        for picture in self.pictures:
            self._layout.removeWidget(picture)
            picture.deleteLater()
        self.pictures = [CropPicture(frame, path) for frame, path in crops]
        for place, picture in enumerate(self.pictures):
            self._layout.insertWidget(place, picture)

    @property
    def shown(self) -> List[int]:
        return [picture.frame for picture in self.pictures]

    @property
    def taken_out(self) -> set:
        return {picture.frame for picture in self.pictures if picture.taken_out}


class SamePlayerWidget(_Judging):
    """Two tracks: one player or two. Pictures that show somebody else than the rest
    of their track are clicked out first; the answer is about those left."""

    def __init__(self):
        super().__init__(PAIRS_SOURCE, PAIRS_ANSWERS, "pairs.csv")
        self.tracks: Dict[judgement_files.TrackKey, judgement_files.Track] = {}
        self.judged: set = set()
        self.pair: Optional[Tuple[judgement_files.Track, judgement_files.Track]] = None
        self.why = judgement_files.RANDOM  # Why the pair is asked about
        self.vectors: dict = {}  # Track -> how the network sees it, once worked out
        self._working: Optional[TrackVectors] = None
        self.why_label = QLabel("")
        self.pictures.addWidget(self.why_label)
        self.captions = [faint(""), faint("")]
        self.strips = [CropRow(), CropRow()]
        for caption, strip in zip(self.captions, self.strips):
            self.pictures.addWidget(caption)
            self.pictures.addWidget(strip)
        self.pictures.addWidget(
            faint(
                "Click a picture that shows somebody else than the rest of its track to "
                "take it out, then answer for the pictures left."
            )
        )
        self.pictures.addStretch(1)
        self.add_answer("Same player", "1", lambda: self._answer("same"))
        self.add_answer("Different players", "2", lambda: self._answer("different"))
        self.add_answer("Cannot tell", "Space", lambda: self._answer("unsure"))

    def load(self) -> None:
        self.tracks = judgement_files.load_tracks(self.source)
        self.judged = judgement_files.judged_pairs(self.answers)
        # Until the network has looked at the tracks, pairs are drawn at random
        self.vectors = {}
        self._working = TrackVectors(self.source, self.tracks)
        self._working.worked_out.connect(self._on_vectors)
        self._working.start()

    def _on_vectors(self, vectors: dict) -> None:
        if self.sender() is self._working:
            self.vectors = vectors
            self._show_status()

    def _show_status(self) -> None:
        looked = (
            "pairs the network is unsure of are asked about most"
            if self.vectors
            else "pairs are drawn at random until the network has looked at the tracks"
        )
        self.status.setText(
            f"{len(self.judged)} pairs answered, {len(self.tracks)} tracks; {looked}"
        )

    def _likeness(self) -> Optional[float]:
        first, second = self.pair
        if first.key in self.vectors and second.key in self.vectors:
            return float(self.vectors[first.key] @ self.vectors[second.key])
        return None

    def ask(self) -> None:
        chosen = judgement_files.choose_pair(self.tracks, self.judged, self.rng, self.vectors)
        if chosen is None:
            self.pair = None
            self.status.setText(f"No pair of tracks left to ask about in {self.source.name}")
            return
        self.pair, self.why = chosen[:2], chosen[2]
        self.why_label.setText(
            "Asked because the network is unsure whether these are one player"
            if self.why == judgement_files.UNSURE
            else "A pair drawn at random"
        )
        for name, track, caption, strip in zip(
            ("First", "Second"), self.pair, self.captions, self.strips
        ):
            video, stretch, player = track.key
            number = f", number {track.number} read" if track.number else ""
            caption.setText(
                f"{name} track: {video}, stretch {stretch}, track {player}, "
                f"{len(track.crops)} pictures over frames {track.crops[0][0]} to "
                f"{track.crops[-1][0]}{number}"
            )
            strip.show_crops(track.chosen(TRACK_CROPS))
        self._show_status()

    def _answer(self, answer: str) -> None:
        if self.pair is None:
            return
        first, second = self.pair
        judgement_files.save_pair(self.answers, self.source.name, first, second, answer)
        judgement_files.save_asked(
            self.answers.with_name("asked.csv"), first, second, self.why, self._likeness()
        )
        for track, strip in zip(self.pair, self.strips):
            judgement_files.save_pictures(
                self.answers.with_name("pictures.csv"),
                self.source.name,
                track,
                strip.shown,
                strip.taken_out,
            )
        self.judged.add(judgement_files.pair_of(first.key, second.key))
        self.ask()


class ClickPicture(QLabel):
    """A picture that says where in it a click fell, as shares of its width and height."""

    clicked = pyqtSignal(float, float)
    right_clicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def mousePressEvent(self, event) -> None:
        shown = self.pixmap()
        if event.button() not in (Qt.LeftButton, Qt.RightButton) or shown is None:
            return
        if shown.isNull():
            return
        x, y = event.pos().x() / shown.width(), event.pos().y() / shown.height()
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            (self.clicked if event.button() == Qt.LeftButton else self.right_clicked).emit(x, y)


class GivenAnswer(QLabel):
    """One of the last answers: the player's picture, framed green for "holds the disc",
    red for "does not" and yellow for a box of two. A click takes the answer back and
    asks again."""

    clicked = pyqtSignal(object)

    def __init__(
        self,
        picture: np.ndarray,
        holding: bool,
        rows: List[dict],
        asked: int,
        two: bool = False,
    ):
        super().__init__()
        self.rows, self.asked, self.holding, self.two = rows, asked, holding, two
        self.entries: list = []  # What was written for it, and when
        self.time = ""
        width = max(1, round(picture.shape[1] * ANSWER_HEIGHT / picture.shape[0]))
        small = cv2.resize(picture, (width, ANSWER_HEIGHT), interpolation=cv2.INTER_AREA)
        colour = SAID_TWO if two else SAID_YES if holding else SAID_NO
        small = cv2.copyMakeBorder(small, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=colour)
        self.setPixmap(frame_to_pixmap(small))
        self.setCursor(Qt.PointingHandCursor)
        said = (
            "two players in this box"
            if two
            else "holds the disc"
            if holding
            else "does not hold the disc"
        )
        self.setToolTip(f"Answered: {said}. Click to take the answer back and give it again.")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self)


class HolderWidget(_Judging):
    """A player of a frame: do they hold the disc? Where the box shows two players,
    that is said instead, and a click on the one who holds the disc says where they
    are. The last answers stand in a row below; a click on one takes it back and asks
    about that player again. Answers given for wide boxes before a box of two could be
    said to be one are asked about again first."""

    def __init__(self):
        super().__init__(HOLDER_SOURCE, HOLDER_ANSWERS, "holders.csv")
        self.frames: Dict[judgement_files.FrameKey, List[dict]] = {}
        self.answered: set = set()
        # Frames nobody was asked about yet: those with another player in the holder's
        # box, and the rest
        self.open: List[judgement_files.FrameKey] = []
        self.open_crowded: List[judgement_files.FrameKey] = []
        self.why_label = QLabel("")
        self.pictures.addWidget(self.why_label)
        self.rows: List[dict] = []  # The players of the frame being asked about
        self.asked = 0  # Which of them
        self._videos: Dict[str, cv2.VideoCapture] = {}
        self.caption = faint("")
        self.player, self.surroundings = ClickPicture(), QLabel()
        self.player.clicked.connect(self._pointed_at)
        # The part of the frame the player's picture shows (left, top, right, bottom),
        # and whether a click on it is waited for: the holder in a box of two
        self._region: Optional[Tuple[int, int, int, int]] = None
        self._pointing = False
        # Answers for wide boxes to ask about again, and the one being asked about
        self.recheck: List[dict] = []
        self._rechecking: Optional[dict] = None
        side_by_side = QHBoxLayout()
        side_by_side.addWidget(self.player)
        side_by_side.addWidget(self.surroundings)
        side_by_side.addStretch(1)
        self.pictures.addWidget(self.caption)
        # Shown while a click on the holder is waited for: it must not be overlooked
        self.prompt = QLabel(
            "Two players in this box: now CLICK on the one who holds the disc, in the left "
            "picture.   Space: neither does, or it cannot be told.   Esc: not two after all."
        )
        self.prompt.setStyleSheet(
            "background: #e6c83c; color: #1e1f22; font-weight: bold; padding: 8px;"
        )
        self.prompt.hide()
        self.pictures.addWidget(self.prompt)
        # What a holder network says of the player asked about: the newest one, unless
        # another is chosen
        self.guess = faint("")
        self.guess_network = QComboBox()
        populate_holder_model_combo(self.guess_network)
        if self.guess_network.count() > 1:
            self.guess_network.setCurrentIndex(1)  # The newest: they are listed newest first
        self.guess_network.setToolTip(
            "How sure a holder network is that this player holds the disc.\n"
            "A network taught on the answers given here knows the players already answered;\n"
            "of the others it is a guess. Off: answer without being led."
        )
        self.guess_network.currentIndexChanged.connect(self._show_guess)
        guessing = QHBoxLayout()
        guessing.addWidget(faint("A network's guess:"))
        guessing.addWidget(self.guess_network)
        guessing.addWidget(self.guess, 1)
        self.pictures.addLayout(guessing)
        self._networks: Dict[str, object] = {}  # Weights -> the network, once loaded
        self.pictures.addLayout(side_by_side)
        self.pictures.addStretch(1)
        # The last answers, the newest on the left
        self.given: List[GivenAnswer] = []
        self._given_row = QHBoxLayout()
        self._given_row.setSpacing(6)
        self._given_row.addWidget(faint("Last answers (click one to give it again):"))
        self._given_row.addStretch(1)
        self.pictures.addLayout(self._given_row)
        self._close: Optional[np.ndarray] = None  # The picture of the player asked about
        # While an answer is given again: the frame that was being asked about, and
        # whether the answer taken back was a yes
        self._interrupted: Optional[Tuple[List[dict], int, str]] = None
        self._again: Optional[bool] = None
        # How far along the way to enough holders said by hand
        self.progress = QProgressBar()
        self.progress.setRange(0, HOLDER_GOAL)
        self.progress.setTextVisible(True)
        self.layout().addWidget(self.progress)
        self.add_answer("Holds the disc", "Y", lambda: self._answer(True))
        self.add_answer("Does not", "N", lambda: self._answer(False))
        self.add_answer("Two players in this box", "T", self._two_in_box)
        back = QShortcut("Esc", self)
        back.setContext(Qt.WidgetWithChildrenShortcut)
        back.activated.connect(lambda: self._wait_for_click(False))
        self.add_answer("Cannot tell", "Space", lambda: self._answer(None))

    def load(self) -> None:
        self.frames = judgement_files.load_holder_frames(self.source)
        self.answered = judgement_files.answered_players(self.answers)
        self.open, self.open_crowded = [], []
        for key, rows in self.frames.items():
            if not any((*key, row["player"]) in self.answered for row in rows):
                (self.open_crowded if judgement_files.crowded(rows) else self.open).append(key)
        self.rows, self.asked = [], 0
        self._pointing, self._rechecking = False, None
        self.recheck = judgement_files.wide_answers(self.answers, self.frames)
        self._show_last_answers()

    def _show_last_answers(self) -> None:
        """Fill the row of answers with the last ones in the file: those of an earlier
        sitting can be taken back too."""
        for old in self.given:
            self._given_row.removeWidget(old)
            old.deleteLater()
        self.given = []
        last = judgement_files.last_holder_answers(self.answers, ANSWERS_SHOWN)
        for answer in reversed(last):
            rows = self.frames.get(answer["frame"])
            players = [row["player"] for row in rows or []]
            if answer["player"] not in players:
                continue
            asked = players.index(answer["player"])
            picture = self._pictures(rows[asked], sharp=False)[0]
            # A box of two is not among these: it is written to a file of its own
            given = GivenAnswer(picture, answer["holding"], rows, asked)
            by_player = {row["player"]: row for row in rows}
            given.entries = [
                (by_player[player], holding, how)
                for player, holding, how in answer["lines"]
                if player in by_player
            ]
            given.time = answer["time"]
            self._show_given(given)

    def ask(self) -> None:
        if self.asked >= len(self.rows) and self._interrupted is not None:
            # Back to what was being asked about before an answer was given again
            self.rows, self.asked, why = self._interrupted
            self._interrupted = None
            self.why_label.setText(why)
        self._wait_for_click(False)
        while self.asked >= len(self.rows) and self.recheck:
            # An answer given for a wide box, before a box of two could be said to be one
            again = self.recheck.pop(0)
            rows = self.frames.get(again["frame"], [])
            players = [row["player"] for row in rows]
            if again["player"] in players:
                self.rows, self.asked = rows, players.index(again["player"])
                self._rechecking = again
                was = "holds the disc" if again["holding"] else "does not hold the disc"
                self.why_label.setText(
                    f"Asked again: a wide box that may show two players. Your answer was: "
                    f"{was}. {len(self.recheck)} more to look at again."
                )
        if self.asked >= len(self.rows):
            if not self.open and not self.open_crowded:
                self.status.setText(f"No frame left to ask about in {self.source.name}")
                return
            # Mostly frames in which two stand in one box, while there are any
            crowded = bool(self.open_crowded) and (
                not self.open or self.rng.random() < judgement_files.CROWDED_SHARE
            )
            left = self.open_crowded if crowded else self.open
            # Any of the frames left, taken out without moving the rest
            index = self.rng.randrange(len(left))
            left[index], left[-1] = left[-1], left[index]
            self.rows, self.asked = self.frames[left.pop()], 0
            self.why_label.setText(
                "Asked because another player stands in the holder's box"
                if crowded
                else "A frame drawn at random"
            )
        row = self.rows[self.asked]
        close, wide, self._region = self._pictures(row)
        self._close = close
        self._show_guess()
        self.player.setPixmap(frame_to_pixmap(close))
        self.surroundings.setPixmap(frame_to_pixmap(wide))
        self.caption.setText(
            f"{row['video']}, frame {row['frame']}, track {row['player']}: player "
            f"{self.asked + 1} of {len(self.rows)} of this frame. A yes makes the others "
            "of the frame a no."
        )
        self.status.setText(
            f"{len(self.answered)} players answered, {len(self.open) + len(self.open_crowded)} "
            f"frames not asked about ({len(self.open_crowded)} with two players in one box)"
        )
        holders = sum(judgement_files.holders_said(self.answers).values())
        self.progress.setValue(min(holders, HOLDER_GOAL))
        left = max(0, HOLDER_GOAL - holders)
        self.progress.setFormat(
            f"{holders} of {HOLDER_GOAL} holders confirmed"
            + (f" - {left} to go" if left else " - goal reached")
        )

    def _show_guess(self, *_: object) -> None:
        """Say what the chosen holder network makes of the player asked about."""
        weights = self.guess_network.currentData()
        if not weights or self.asked >= len(self.rows):
            self.guess.setText("")
            return
        row = self.rows[self.asked]
        frame = self._frame(row)
        if frame is None:
            self.guess.setText("")
            return
        try:
            from ...processing.holder import BOX_ALONE, crop_around, holding, load_holder_net

            with GPU_SETUP_LOCK.work():
                if weights not in self._networks:
                    self._networks[weights] = load_holder_net(Path(weights))
                box = [float(row[name]) for name in ("x1", "y1", "x2", "y2")]
                crop = crop_around(frame, box, BOX_ALONE)[0]
                sure = float(holding(self._networks[weights], [crop])[0])
        except Exception as error:  # No PyTorch, no boxes in the dataset, weights of another build
            logger.warning(f"The holder network has nothing to say: {error}")
            self.guess.setText("")
            return
        self.guess.setText(f"{sure:.0%} sure that this player holds the disc")

    def _frame(self, row: dict) -> Optional[np.ndarray]:
        """The frame of the video a player is in, if the video is there."""
        video = row["video"]
        if video not in self._videos:
            self._videos[video] = cv2.VideoCapture(
                str(Path(DEFAULT_PATHS["RAW_VIDEOS"]) / f"{video}.mp4")
            )
        capture = self._videos[video]
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        ok, frame = capture.read()
        return frame if ok else None

    def _pictures(
        self, row: dict, sharp: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """The player made large, where they stand, and the part of the frame the first
        shows (None without the video)."""
        frame = self._frame(row)
        if frame is None:
            folder = judgement_files.stretch_folder(
                self.source / "crops", row["video"], row["stretch"]
            )
            crop = cv2.imread(str(folder / f"{row['player']}_{row['frame']}.jpg"))
            if crop is None:
                crop = np.full((90, 40, 3), GROUND, dtype=np.uint8)
            return self._large(crop, sharp), np.full((10, 10, 3), GROUND, dtype=np.uint8), None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(float(row[name])) for name in ("x1", "y1", "x2", "y2"))

        def around(pad_x: float, pad_y: float) -> Tuple[int, int, int, int]:
            dx, dy = round(pad_x * (x2 - x1)), round(pad_y * (y2 - y1))
            return max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy)

        region = left, top, right, bottom = around(PLAYER_PAD, PLAYER_PAD)
        close = self._large(frame[top:bottom, left:right], sharp)
        left, top, right, bottom = around(4.0, 1.5)
        wide = frame[top:bottom, left:right].copy()
        cv2.rectangle(wide, (x1 - left, y1 - top), (x2 - left, y2 - top), (255, 255, 255), 1)
        scale = PLAYER_HEIGHT / wide.shape[0]
        wide = cv2.resize(wide, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return close, wide, region

    def _large(self, crop: np.ndarray, sharp: bool = True) -> np.ndarray:
        if not sharp:
            # For the small pictures of the answers given: the network is not worth its time
            scale = PLAYER_HEIGHT / crop.shape[0]
            return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # The network shares the GPU with the analysis of the main tab
        with GPU_SETUP_LOCK.work():
            crop = enlarged(crop)
        scale = PLAYER_HEIGHT / crop.shape[0]
        return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def _two_in_box(self) -> None:
        """The box shows two players: wait for a click on the one who holds the disc."""
        if self.asked >= len(self.rows):
            return
        if self._region is None:
            self._save_two(None)  # Without the video there is no frame to point in
            return
        self._wait_for_click(True)

    def _wait_for_click(self, waiting: bool) -> None:
        """Begin or end waiting for a click on the holder in a box of two."""
        self._pointing = waiting
        self.prompt.setVisible(waiting)
        self.player.setCursor(Qt.CrossCursor if waiting else Qt.ArrowCursor)
        self.player.setStyleSheet("border: 4px solid #e6c83c;" if waiting else "")

    def _pointed_at(self, x: float, y: float) -> None:
        if not self._pointing or self._region is None:
            return
        left, top, right, bottom = self._region
        self._save_two((left + x * (right - left), top + y * (bottom - top)))

    def _look_again(self, replaced: bool) -> None:
        """The answer being asked about again, if one is, has been looked at: taken
        out where another is given in its place."""
        old, self._rechecking = self._rechecking, None
        if old is None:
            return
        row = self.rows[self.asked]
        if replaced:
            by_player = {r["player"]: r for r in self.rows}
            entries = [
                (by_player[player], holding, how)
                for player, holding, how in old["lines"]
                if player in by_player
            ]
            judgement_files.keep_a_copy(self.answers)
            judgement_files.take_back_holders(self.answers, entries, old["time"])
            self.answered -= {
                (r["video"], r["stretch"], r["frame"], r["player"]) for r, _, _ in entries
            }
            for given in [g for g in self.given if g.time == old["time"] and not g.two]:
                self.given.remove(given)
                self._given_row.removeWidget(given)
                given.deleteLater()
        judgement_files.save_rechecked(self.answers, row)

    def _save_two(self, point: Optional[Tuple[float, float]]) -> None:
        """Write down that the box asked about shows two players, and where the one
        who holds the disc is, if that was pointed at."""
        row = self.rows[self.asked]
        self._again = None
        self._look_again(replaced=True)
        entries = judgement_files.two_in_box_answers(self.rows, row["player"], point)
        given = GivenAnswer(self._close, True, self.rows, self.asked, two=True)
        given.entries = entries
        given.time = judgement_files.save_two_in_box(
            self.answers, self.source.name, row, point, entries
        )
        self._show_given(given)
        self.answered.add((row["video"], row["stretch"], row["frame"], row["player"]))
        self.answered.update(
            (r["video"], r["stretch"], r["frame"], r["player"]) for r, _, _ in entries
        )
        self.asked = len(self.rows)  # The frame is settled
        self.ask()

    def _answer(self, holding: Optional[bool]) -> None:
        if self.asked >= len(self.rows):
            return
        if self._pointing and holding is None:
            self._save_two(None)
            return
        row = self.rows[self.asked]
        was_yes, self._again = self._again, None
        if self._rechecking is not None:
            # Looked at again: the others of the frame were asked about the first time
            self._look_again(replaced=holding is not None)
            was_yes = False
        if holding is None:
            self.asked += 1
        else:
            entries = judgement_files.holder_answers(self.rows, row["player"], holding)
            given = GivenAnswer(self._close, holding, self.rows, self.asked)
            given.entries = entries
            given.time = judgement_files.save_holders(self.answers, self.source.name, entries)
            self._show_given(given)
            self.answered.update(
                (r["video"], r["stretch"], r["frame"], r["player"]) for r, _, _ in entries
            )
            # A holder settles the frame; otherwise the next player of it is asked about
            self.asked = len(self.rows) if holding else self.asked + 1
        if was_yes is False or (was_yes and holding is None):
            # Given again where the others of the frame were asked about already
            self.asked = len(self.rows)
        self.ask()

    def _show_given(self, given: GivenAnswer) -> None:
        given.clicked.connect(self._give_again)
        self.given.insert(0, given)
        self._given_row.insertWidget(1, given)
        for old in self.given[ANSWERS_SHOWN:]:
            self._given_row.removeWidget(old)
            old.deleteLater()
        del self.given[ANSWERS_SHOWN:]

    def _give_again(self, given: GivenAnswer) -> None:
        """Take one of the last answers back and ask about that player again."""
        judgement_files.take_back_holders(self.answers, given.entries, given.time)
        self.answered -= {
            (r["video"], r["stretch"], r["frame"], r["player"]) for r, _, _ in given.entries
        }
        if given.two:
            row = given.rows[given.asked]
            judgement_files.take_back_two_in_box(self.answers, row, given.time)
            self.answered.discard((row["video"], row["stretch"], row["frame"], row["player"]))
        if self._rechecking is not None:
            # Put back: it is asked about again once this one is given
            self.recheck.insert(0, self._rechecking)
            self._rechecking = None
        self.given.remove(given)
        self._given_row.removeWidget(given)
        given.deleteLater()
        if self._interrupted is None and self.asked < len(self.rows):
            self._interrupted = (self.rows, self.asked, self.why_label.text())
        self.rows, self.asked, self._again = given.rows, given.asked, given.holding
        self.why_label.setText("An answer taken back: give it again")
        self.ask()
