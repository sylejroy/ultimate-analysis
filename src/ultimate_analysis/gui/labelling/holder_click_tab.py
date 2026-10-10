"""Who holds the disc, said with one click on the player.

"Click the holder" plays the second of the game that leads up to a frame of a holder
dataset, stops on the frame with a box around every player the player model finds, and
takes a click on the one who holds the disc. A right-click says that a box shows two
players, one of them with the disc: the box then opens large in a window of its own,
for a click on that one. A player standing still is not told from
one who has just caught the disc by a single picture; the second before shows it.

What is written is what the "Holding the disc?" page writes, to the same files
(utils/judgement_files.py): the player clicked is a yes and the others of the frame a
no. The dataset knows four players of a frame at most; a click on a player it does not
know makes all it knows a no, and where the click fell is kept beside the answers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QShortcut,
    QVBoxLayout,
)

from ...constants import DEFAULT_PATHS
from ...processing.model_lock import GPU_SETUP_LOCK, MODEL_LOCK
from ...utils import judgement_files
from ...utils.enlarging import enlarged
from ...utils.logger import get_logger
from ...utils.model_files import default_model_path
from ..widgets.images import frame_to_pixmap
from ..widgets.model_selection import populate_holder_model_combo
from .judgement_tabs import (
    HOLDER_ANSWERS,
    HOLDER_GOAL,
    HOLDER_SOURCE,
    ClickPicture,
    _Judging,
    faint,
)

logger = get_logger("LABELLING")

LEAD_SECONDS = 1.0  # How much of the game before the frame is played
PLAYED_EVERY = 2  # Every so many frames of it are shown
# The height the frame is shown at. All of it, always: "nobody holds the disc" can only
# be said of a picture that shows everybody
SHOWN_HEIGHT = 760
SAME_PLAYER = 0.5  # A box found is a player of the dataset from this overlap (IoU)
BOX = (255, 255, 255)  # Colour of the boxes drawn (BGR)
# What was stored is shown on the frame for this long before the next one comes: the
# box clicked in green as the holder, in yellow as a box of two; the others in red as
# those written down as not holding the disc
STORED_SHOWN_MS = 600
HOLDER, TWO, NOT_HOLDING = (90, 200, 90), (60, 200, 230), (80, 80, 230)
# The player a holder network takes for the holder is marked in this colour, if it is
# with a thick line if it is at least this sure of them
GUESSED = (255, 200, 60)
DISC = (0, 230, 255)  # The ring around a disc found flying (BGR): yellow
LEAST_SURE = 0.5
TWO_HEIGHT = 640  # The height a box of two players is shown at to click on one of them
TWO_ROOM = 0.35  # ... with this share of the box around it

Box = Tuple[float, float, float, float]


def overlap(first: Box, second: Box) -> float:
    """How much two boxes (x1, y1, x2, y2) share of what they cover together (IoU)."""
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    area = (first[2] - first[0]) * (first[3] - first[1])
    return shared / (area + (second[2] - second[0]) * (second[3] - second[1]) - shared)


def box_of(row: dict) -> Box:
    return tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))


def box_clicked(boxes: List[Box], x: float, y: float) -> Optional[int]:
    """Which of the boxes a click fell in: the smallest of several."""
    inside = [
        ((b[2] - b[0]) * (b[3] - b[1]), index)
        for index, b in enumerate(boxes)
        if b[0] <= x <= b[2] and b[1] <= y <= b[3]
    ]
    return min(inside)[1] if inside else None


class TwoPlayersDialog(QDialog):
    """A box that shows two players, made large: a click on the one who has the disc.

    `where` is afterwards the click as shares of the picture's width and height, or None
    if which of the two has it could not be told. Closed any other way, it was no box
    of two after all.
    """

    def __init__(self, picture: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Two players in this box")
        self.where: Optional[Tuple[float, float]] = None
        shown = ClickPicture()
        shown.setPixmap(frame_to_pixmap(picture))
        shown.setCursor(Qt.CrossCursor)
        shown.clicked.connect(self._chosen)
        unsure = QPushButton("Cannot tell which of them  [Space]")
        unsure.clicked.connect(self.accept)
        QShortcut("Space", self).activated.connect(self.accept)
        back = QPushButton("Not two players after all  [Esc]")
        back.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(unsure)
        buttons.addWidget(back)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("One of these two has the disc: click on that player."))
        layout.addWidget(shown)
        layout.addLayout(buttons)

    def _chosen(self, x: float, y: float) -> None:
        self.where = (x, y)
        self.accept()


class HolderClickWidget(_Judging):
    """The second before a frame, then the frame with its players: click the holder."""

    def __init__(self):
        super().__init__(HOLDER_SOURCE, HOLDER_ANSWERS, "holders.csv")
        self.frames: Dict[judgement_files.FrameKey, List[dict]] = {}
        self.answered: set = set()
        self.open: List[judgement_files.FrameKey] = []
        self.open_crowded: List[judgement_files.FrameKey] = []
        self.key: Optional[judgement_files.FrameKey] = None  # The frame asked about
        self.rows: List[dict] = []  # ... and the players of it the dataset knows
        self.boxes: List[Box] = []  # The players found in it, in frame pixels
        self._part = (0, 0, 1, 1)  # The part of the frame shown
        self._clip: List[np.ndarray] = []  # The second before, as shown; the frame last
        self._frame: Optional[np.ndarray] = None  # The frame asked about, whole
        # Frame -> where the disc was found flying in it
        self._flying: Dict[judgement_files.FrameKey, Tuple[float, float]] = {}
        self._at = 0
        self._last: Optional[dict] = None  # The answer given last, to take it back
        self._videos: Dict[str, cv2.VideoCapture] = {}
        self._detector: Any = None
        self._tried_detector = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_picture)

        self.caption = faint(
            "Click on the player who holds the disc. Right-click a box that shows two "
            "players, one of them with the disc: it opens large to click on that one."
        )
        self.picture = ClickPicture()
        self.picture.setCursor(Qt.PointingHandCursor)
        self.picture.clicked.connect(lambda x, y: self._clicked(x, y, two=False))
        self.picture.right_clicked.connect(lambda x, y: self._clicked(x, y, two=True))
        self.why_label = QLabel("")
        row = QHBoxLayout()
        row.addWidget(self.picture)
        row.addStretch(1)
        self.pictures.addWidget(self.why_label)
        self.pictures.addWidget(self.caption)
        # The player a holder network takes for the holder is marked on the frame: by
        # the newest network, unless another is chosen
        self.guess = faint("")
        self.guess_network = QComboBox()
        populate_holder_model_combo(self.guess_network)
        if self.guess_network.count() > 1:
            self.guess_network.setCurrentIndex(1)  # The newest: they are listed newest first
        self.guess_network.setToolTip(
            "The player this holder network is surest of is marked in light blue on the\n"
            "frame, with how sure it is. A network taught on the answers given here knows\n"
            "the players already answered. Off: answer without being led."
        )
        self.guess_network.currentIndexChanged.connect(self._guess_again)
        guessing = QHBoxLayout()
        guessing.addWidget(faint("A network's guess (light blue):"))
        guessing.addWidget(self.guess_network)
        guessing.addWidget(self.guess, 1)
        # Only the frames in which the dataset has no holder: the disc in the air
        self.only_flying = QCheckBox("Only show frames with a flying disc")
        self.only_flying.setToolTip(
            "Ask only about frames in which the dataset has no holder: the disc in the air,\n"
            "or hidden. N if nobody holds it; a click if somebody does after all."
        )
        self.only_flying.toggled.connect(self._only_flying_toggled)
        guessing.addWidget(self.only_flying)
        self.pictures.addLayout(guessing)
        self._networks: Dict[str, object] = {}  # Weights -> the network, once loaded
        self.pictures.addLayout(row)
        self.pictures.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setRange(0, HOLDER_GOAL)
        self.layout().addWidget(self.progress)
        self.add_answer("Nobody holds it", "N", self._nobody)
        self.add_answer("Cannot tell", "Space", self._skip)
        self.add_answer("Play again", "R", self._play)
        self.add_answer("Take the last answer back", "Backspace", self._take_back)

    # ------------------------------------------------------------------ what to ask

    def load(self) -> None:
        self.frames = judgement_files.load_holder_frames(self.source, without=True)
        self.answered = judgement_files.answered_players(self.answers)
        self.open, self.open_crowded, self.open_without = [], [], []
        flying = self._flying = judgement_files.flying_frames(self.source.name)
        for key, rows in self.frames.items():
            if any((*key, row["player"]) in self.answered for row in rows):
                continue
            if not any(row["holding"] == "1" for row in rows):
                # The dataset has no holder in it. Asked about only where the disc was
                # found to fly: in two of three of these frames it lies still, is at a
                # player or is not found again
                if key in flying:
                    self.open_without.append(key)
            elif judgement_files.crowded(rows):
                self.open_crowded.append(key)
            else:
                self.open.append(key)
        self.key, self._last = None, None

    def ask(self, key: Optional[judgement_files.FrameKey] = None) -> None:
        """Show the next frame to ask about (or the one given)."""
        self._timer.stop()
        while True:
            crowded = without = False
            if key is None:
                only_flying = self.only_flying.isChecked()
                if not self.open_without and (
                    only_flying or (not self.open and not self.open_crowded)
                ):
                    kind = "frame with a flying disc" if only_flying else "frame"
                    self.status.setText(
                        f"No {kind} left to ask about in {self.source.name}"
                        + (
                            ": scripts/find_flying_disc_frames.py looks for more"
                            if only_flying
                            else ""
                        )
                    )
                    self.key = None
                    return
                # Some frames in which the dataset has no holder: the disc in the air, or
                # hidden. Of the others, mostly ones in which two stand in one box
                with_holder = bool(self.open or self.open_crowded)
                without = bool(self.open_without) and (
                    only_flying
                    or not with_holder
                    or self.rng.random() < judgement_files.NO_HOLDER_SHARE
                )
                crowded = (
                    not without
                    and bool(self.open_crowded)
                    and (not self.open or self.rng.random() < judgement_files.CROWDED_SHARE)
                )
                left = self.open_without if without else self.open_crowded if crowded else self.open
                # A game drawn first, then a frame of it: every game is asked about as
                # often, however many of its frames there are
                game = self.rng.choice(sorted({frame[0] for frame in left}))
                index = self.rng.choice([i for i, frame in enumerate(left) if frame[0] == game])
                left[index], left[-1] = left[-1], left[index]
                key = left.pop()
            if self._show(key):
                break
            key = None  # The video is not there, or the frame not in it: the next one
        self.why_label.setText(
            "A frame in which the disc was found flying: N if nobody holds it"
            if without
            else "Asked because another player stands in the holder's box"
            if crowded
            else "A frame drawn at random"
        )
        holders = sum(judgement_files.holders_said(self.answers).values())
        self.progress.setValue(min(holders, HOLDER_GOAL))
        self.progress.setFormat(f"{holders} of {HOLDER_GOAL} holders confirmed")
        self.status.setText(
            f"{self.key[0]}, frame {self.key[2]}: {len(self.boxes)} players found. "
            f"{len(self.open) + len(self.open_crowded) + len(self.open_without)} frames not "
            "asked about."
        )

    # ------------------------------------------------------------------ the pictures

    def _capture(self, video: str) -> cv2.VideoCapture:
        if video not in self._videos:
            self._videos[video] = cv2.VideoCapture(
                str(Path(DEFAULT_PATHS["RAW_VIDEOS"]) / f"{video}.mp4")
            )
        return self._videos[video]

    def _players_in(self, frame: np.ndarray) -> List[Box]:
        """The players the player model finds in a frame; none without the model."""
        if not self._tried_detector:
            self._tried_detector = True
            try:
                from ...processing.inference import load_detection_model

                with MODEL_LOCK:
                    self._detector = load_detection_model(default_model_path("player_detection"))
            except Exception as error:  # No Ultralytics, no weights
                logger.warning(f"Players are not looked for in the frames asked about: {error}")
        if self._detector is None:
            return []
        from ...processing.inference import detect_players

        with MODEL_LOCK:
            found = detect_players(frame, *self._detector)
        return [tuple(float(value) for value in detection["bbox"]) for detection in found]

    def _show(self, key: judgement_files.FrameKey) -> bool:
        """Read the second before a frame and the frame, find its players, and play;
        False if the video does not give them."""
        rows = self.frames[key]
        capture = self._capture(key[0])
        if not capture.isOpened():
            return False
        rate = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_index = int(key[2])
        first = max(0, frame_index - int(round(LEAD_SECONDS * rate)))
        disc = self._flying.get(key)
        if disc is not None and self.only_flying.isChecked():
            # Only flying discs are asked about: the frame alone, with the disc marked.
            # What led up to it says little where nobody has the disc
            first = frame_index
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames = []
        for number in range(first, frame_index + 1):
            ok, frame = capture.read()
            if not ok:
                return False
            if (frame_index - number) % PLAYED_EVERY == 0:
                frames.append(frame)
        self._frame = frames[-1].copy()  # Before the boxes are drawn on it
        height, width = frames[-1].shape[:2]
        known = [box_of(row) for row in rows]
        self._part = left, top, right, bottom = 0, 0, width, height
        # The players found, and those of the dataset the model did not find as well
        found = self._players_in(frames[-1])
        found += [box for box in known if all(overlap(box, b) < SAME_PLAYER for b in found)]
        self.boxes = [
            b for b in found if b[2] > left and b[0] < right and b[3] > top and b[1] < bottom
        ]
        scale = SHOWN_HEIGHT / (bottom - top)
        self._clip = [
            cv2.resize(
                f[top:bottom, left:right], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            for f in frames
        ]
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(
                self._clip[-1],
                (round((x1 - left) * scale), round((y1 - top) * scale)),
                (round((x2 - left) * scale), round((y2 - top) * scale)),
                BOX,
                1,
                cv2.LINE_AA,
            )
        if disc is not None:
            # Where the disc was found flying: a fat ring, dark-rimmed to show on grass
            # and on a white shirt alike
            centre = (round((disc[0] - left) * scale), round((disc[1] - top) * scale))
            cv2.circle(self._clip[-1], centre, 22, (20, 20, 20), 7, cv2.LINE_AA)
            cv2.circle(self._clip[-1], centre, 22, DISC, 4, cv2.LINE_AA)
        guessed = self._guessed(frames[-1])
        self.guess.setText("")
        if guessed is not None:
            (x1, y1, x2, y2), sure = self.boxes[guessed[0]], guessed[1]
            first = (round((x1 - left) * scale) - 3, round((y1 - top) * scale) - 3)
            second = (round((x2 - left) * scale) + 3, round((y2 - top) * scale) + 3)
            # Thick where the network is sure of them, thin where they are only its best guess
            thick = 2 if sure >= LEAST_SURE else 1
            cv2.rectangle(self._clip[-1], first, second, GUESSED, thick, cv2.LINE_AA)
            cv2.putText(
                self._clip[-1],
                f"{sure:.0%}",
                (first[0], second[1] + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                GUESSED,
                1,
                cv2.LINE_AA,
            )
            unsure = "" if sure >= LEAST_SURE else ": its best guess, and not a sure one"
            self.guess.setText(f"{sure:.0%} sure of the player marked{unsure}")
        elif self.guess_network.currentData():
            self.guess.setText("sure of nobody")
        self.key, self.rows = key, rows
        self._interval = max(15, round(1000 * PLAYED_EVERY / rate))
        self._play()
        return True

    def _guessed(self, frame: np.ndarray) -> Optional[Tuple[int, float]]:
        """The player of the frame the chosen holder network is surest of, and how sure
        (index into `self.boxes`, 0 to 1); None if it is switched off."""
        weights = self.guess_network.currentData()
        if not weights or not self.boxes:
            return None
        try:
            from ...processing.holder import BOX_ALONE, crop_around, holding, load_holder_net

            with GPU_SETUP_LOCK.work():
                if weights not in self._networks:
                    self._networks[weights] = load_holder_net(Path(weights))
                crops = [crop_around(frame, box, BOX_ALONE)[0] for box in self.boxes]
                sure = holding(self._networks[weights], crops)
        except Exception as error:  # No PyTorch, weights of another build
            logger.warning(f"The holder network has nothing to say: {error}")
            return None
        best = int(np.argmax(sure))
        return best, float(sure[best])

    def _guess_again(self, *_: object) -> None:
        """Another network was chosen: show the frame asked about with its guess."""
        if self.key is not None:
            self._show(self.key)

    def _play(self) -> None:
        if not self._clip:
            return
        self._at = 0
        self.picture.setPixmap(frame_to_pixmap(self._clip[0]))
        self._timer.start(self._interval)

    def _next_picture(self) -> None:
        self._at += 1
        if self._at >= len(self._clip) - 1:
            self._at = len(self._clip) - 1
            self._timer.stop()
        self.picture.setPixmap(frame_to_pixmap(self._clip[self._at]))

    # ------------------------------------------------------------------ the answers

    def _written(
        self,
        entries: list,
        time: str,
        chosen: Optional[Box] = None,
        colour: Tuple[int, int, int] = HOLDER,
        said: str = "",
        **more: Any,
    ) -> None:
        """An answer is written: remember it, show on the frame what was stored (the box
        chosen in its colour, the players who do not hold the disc in red), and go on
        after a moment.

        Red are those written down as not holding it, and with them those for whom it
        follows and who become crops to learn from when the dataset is made
        (scripts/build_holder_by_hand_dataset.py --everybody): every player found where
        nobody holds the disc, and every one whose box does not touch the holder's
        where somebody does. Not where the box clicked shows two players."""
        self._last = {"key": self.key, "entries": entries, "time": time, **more}
        self.answered.update(
            (r["video"], r["stretch"], r["frame"], r["player"]) for r, _, _ in entries
        )
        self._timer.stop()
        left, top, _, bottom = self._part
        scale = SHOWN_HEIGHT / (bottom - top)

        def corners(box: Box) -> Tuple[Tuple[int, int], Tuple[int, int]]:
            return (
                (round((box[0] - left) * scale), round((box[1] - top) * scale)),
                (round((box[2] - left) * scale), round((box[3] - top) * scale)),
            )

        shown = self._clip[-1].copy()
        for row, holding, _ in entries:
            if not holding:
                cv2.rectangle(shown, *corners(box_of(row)), NOT_HOLDING, 2, cv2.LINE_AA)
        if colour != TWO:
            for box in self.boxes:
                if chosen is None or overlap(box, chosen) == 0:
                    cv2.rectangle(shown, *corners(box), NOT_HOLDING, 2, cv2.LINE_AA)
        if chosen is not None:
            first, second = corners(chosen)
            cv2.rectangle(shown, first, second, colour, 4, cv2.LINE_AA)
            cv2.putText(
                shown,
                said,
                (first[0], max(18, first[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
                cv2.LINE_AA,
            )
        self.picture.setPixmap(frame_to_pixmap(shown))
        self.status.setText(f"Stored: {said}" if said else "Stored")
        # Nothing is taken for this frame any more; the next comes in a moment
        self.key = None
        QTimer.singleShot(STORED_SHOWN_MS, self._after_stored)

    def _after_stored(self) -> None:
        if self.key is None:  # Unless an answer was taken back meanwhile
            self.ask()

    def _clicked(self, x: float, y: float, two: bool) -> None:
        """A click on the picture: on the holder, or (`two`) on the holder in a box
        that shows two players."""
        if self.key is None:
            return
        left, top, right, bottom = self._part
        point = (left + x * (right - left), top + y * (bottom - top))
        index = box_clicked(self.boxes, *point)
        if index is None:
            self.status.setText("That is no player's box: click inside one, or press N or Space")
            return
        box = self.boxes[index]
        if two:
            which = self._which_of_two(box)
            if which is None:
                return  # Not two players after all
            point = which[0]  # None: one of them has it, which cannot be told
        likeness = [overlap(box, box_of(row)) for row in self.rows]
        row = self.rows[int(np.argmax(likeness))] if max(likeness) >= SAME_PLAYER else None
        source = self.source.name
        if row is not None and two:
            entries = judgement_files.two_in_box_answers(self.rows, row["player"], point)
            time = judgement_files.save_two_in_box(self.answers, source, row, point, entries)
            self.answered.add((*self.key, row["player"]))
            self._written(entries, time, box, TWO, "two players in this box", two=row)
        elif row is not None:
            entries = judgement_files.holder_answers(self.rows, row["player"], True)
            time = judgement_files.save_holders(self.answers, source, entries)
            self._written(entries, time, box, HOLDER, "holds the disc")
        else:
            # A player the dataset does not know holds it: none of those it knows does
            entries = judgement_files.others_hold(self.rows)
            time = judgement_files.save_holders(self.answers, source, entries)
            where = point or ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            judgement_files.save_clicked(self.answers, source, self.key, box, where, two, time)
            said = "two players in this box" if two else "holds the disc"
            self._written(entries, time, box, TWO if two else HOLDER, said, clicked=True)

    def _which_of_two(self, box: Box) -> Optional[Tuple[Optional[Tuple[float, float]]]]:
        """Show a box of two players large and take a click on the one with the disc:
        (where in the frame,), (None,) if that cannot be told, or None if the box does
        not show two after all."""
        height, width = self._frame.shape[:2]
        pad_x, pad_y = TWO_ROOM * (box[2] - box[0]), TWO_ROOM * (box[3] - box[1])
        left, top = max(0, int(box[0] - pad_x)), max(0, int(box[1] - pad_y))
        right, bottom = min(width, int(box[2] + pad_x)), min(height, int(box[3] + pad_y))
        # The network shares the GPU with the analysis of the main tab
        with GPU_SETUP_LOCK.work():
            picture = enlarged(self._frame[top:bottom, left:right])
        scale = TWO_HEIGHT / picture.shape[0]
        picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        dialog = TwoPlayersDialog(picture, self)
        if dialog.exec_() != QDialog.Accepted:
            return None
        if dialog.where is None:
            return (None,)
        x, y = dialog.where
        return ((left + x * (right - left), top + y * (bottom - top)),)

    def _nobody(self) -> None:
        if self.key is None:
            return
        entries = judgement_files.nobody_holds(self.rows)
        time = judgement_files.save_holders(self.answers, self.source.name, entries)
        self._written(entries, time, said="nobody holds the disc")

    def _skip(self) -> None:
        if self.key is not None:
            self.ask()

    def _put_back(self) -> None:
        """What is being asked about is asked about later."""
        if self.key is not None:
            (
                self.open_without
                if not any(row["holding"] == "1" for row in self.rows)
                else self.open_crowded
                if judgement_files.crowded(self.rows)
                else self.open
            ).append(self.key)
            self.key = None

    def _only_flying_toggled(self, _checked: bool) -> None:
        """Another kind of frame is asked for: go on with one of that kind."""
        if self._loaded:
            self._put_back()
            self.ask()

    def _take_back(self) -> None:
        """Take the last answer out of the files again and ask about its frame anew."""
        last, self._last = self._last, None
        if last is None:
            self.status.setText("No answer to take back: only the last one can be")
            return
        judgement_files.take_back_holders(self.answers, last["entries"], last["time"])
        if last.get("two") is not None:
            judgement_files.take_back_two_in_box(self.answers, last["two"], last["time"])
            self.answered.discard((*last["key"], last["two"]["player"]))
        if last.get("clicked"):
            judgement_files.take_back_clicked(self.answers, last["key"], last["time"])
        self.answered -= {
            (r["video"], r["stretch"], r["frame"], r["player"]) for r, _, _ in last["entries"]
        }
        if self.key is not None and self.key != last["key"]:
            # What was being asked about is asked about later
            (
                self.open_without
                if not any(row["holding"] == "1" for row in self.rows)
                else self.open_crowded
                if judgement_files.crowded(self.rows)
                else self.open
            ).append(self.key)
        self.ask(last["key"])
        self.why_label.setText("An answer taken back: give it again")
