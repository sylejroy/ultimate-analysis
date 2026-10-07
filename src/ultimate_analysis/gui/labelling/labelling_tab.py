"""Labelling tab: mark players and discs on video frames to build a training dataset.

Every frame starts with the boxes the current models find (a prelabel). The user corrects
them and saves the frame; saved frames go into a dataset folder at full resolution, ready
to be selected in the Model Training tab.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...config.settings import get_setting
from ...constants import DEFAULT_PATHS
from ...processing.inference import detect_discs, detect_players, load_detection_model
from ...processing.model_lock import MODEL_LOCK
from ...processing.team_tracker import observers_in_frame
from ...utils import label_files
from ...utils.label_files import CLASS_NAMES, LabelBox
from ...utils.logger import get_logger
from ...utils.model_files import default_model_path
from ..widgets.panels import side_panel
from ..widgets.video_list import VideoListWidget
from .box_canvas import BoxCanvas

logger = get_logger("LABELLING")

DEFAULT_DATASET = "labelled_players_discs_v1"
RANDOM_FRAME_TRIES = 20
HELP_TEXT = (
    "Left drag: new disc box, right drag: new player\n"
    "    box (also across other boxes)\n"
    "Click a box: select it\n"
    "Drag the selected box: move, a grip: resize\n"
    "Delete: remove the selected box\n"
    "1 / 2: make the selected box a disc / a player\n"
    "Wheel: zoom, middle button drag: move view,\n"
    "    0: whole frame\n"
    "Enter: save and go on\n"
    "Left / Right: step without saving\n"
    "R: random frame of a random video;\n"
    "    saving then goes on at random too"
)


# How often a suggestion that had to wait for the models is tried again
SUGGESTION_RETRY_MS = 300

# A point is played by seven a side with one disc: no more are suggested, and the count
# of boxes shows red beyond
MOST_ON_THE_FIELD = {"player": 14, "disc": 1}
COUNT_FULL, COUNT_SHORT, COUNT_OVER = "#4caf50", "#d0d0d0", "#ff5252"


def most_certain(detections: List[dict], most: int) -> List[dict]:
    """The detections the model is surest of, at most so many."""
    return sorted(detections, key=lambda detection: -detection["confidence"])[:most]


def count_colour(name: str, number: int) -> str:
    """Colour for the number of boxes of a class on a frame."""
    most = MOST_ON_THE_FIELD.get(name)
    if most is None or number < most:
        return COUNT_SHORT
    return COUNT_FULL if number == most else COUNT_OVER


class LabellingTab(QWidget):
    """Tab for labelling players and discs on video frames."""

    def __init__(self):
        super().__init__()
        self._capture: Optional[cv2.VideoCapture] = None
        self._video_path = ""
        self._frame_count = 0
        self._frame_index = 0
        self._frame: Optional[np.ndarray] = None
        self._frame_counts: Dict[str, int] = {}  # Per video, for picking random frames
        self._saved = False  # The shown frame is in the dataset
        self._edited = False  # ... and its boxes were changed since
        # (model, image size) for players and discs; loaded when first needed
        # (video, frame) shown without its suggestion because the models were busy
        self._awaiting_suggestion: Optional[Tuple[str, int]] = None
        self._detectors: Optional[Tuple[Any, Any]] = None

        self._init_ui()
        self._init_shortcuts()
        self.video_list.reload()
        if self.video_list.count():
            self.video_list.setCurrentRow(0)

    # ------------------------------------------------------------------ interface

    def _init_ui(self) -> None:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)

        videos = QGroupBox("Video")
        videos_layout = QVBoxLayout(videos)
        self.video_list = VideoListWidget(show_duration=False)
        self.video_list.currentRowChanged.connect(self._on_video_selected)
        videos_layout.addWidget(self.video_list)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.video_list.reload)
        videos_layout.addWidget(refresh_button)
        panel_layout.addWidget(videos)

        frames = QGroupBox("Frame")
        frames_layout = QVBoxLayout(frames)
        self.frame_label = QLabel("No video")
        frames_layout.addWidget(self.frame_label)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setTracking(False)  # Only load the frame the slider is let go on
        self.frame_slider.valueChanged.connect(self._go_to)
        frames_layout.addWidget(self.frame_slider)
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step (frames):"))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 3000)
        self.step_spin.setValue(30)
        self.step_spin.setToolTip("How far Previous, Next, and saving move through the video")
        step_row.addWidget(self.step_spin)
        frames_layout.addLayout(step_row)
        move_row = QHBoxLayout()
        previous_button = QPushButton("Previous")
        previous_button.clicked.connect(lambda: self._step(-1))
        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda: self._step(1))
        move_row.addWidget(previous_button)
        move_row.addWidget(next_button)
        frames_layout.addLayout(move_row)
        labelled_row = QHBoxLayout()
        previous_labelled = QPushButton("Previous labelled")
        previous_labelled.setToolTip("Go to the frame labelled before this one (in any video)")
        previous_labelled.clicked.connect(lambda: self._go_to_labelled(-1))
        next_labelled = QPushButton("Next labelled")
        next_labelled.setToolTip("Go to the frame labelled after this one (in any video)")
        next_labelled.clicked.connect(lambda: self._go_to_labelled(1))
        labelled_row.addWidget(previous_labelled)
        labelled_row.addWidget(next_labelled)
        frames_layout.addLayout(labelled_row)
        random_button = QPushButton("Random frame (R)")
        random_button.setToolTip(
            "Go to a frame of any video that is not labelled yet. Frames from all over the "
            "games teach a model more than neighbouring frames of one point."
        )
        random_button.clicked.connect(self._go_to_random_frame)
        frames_layout.addWidget(random_button)
        self.random_check = QCheckBox("Go on at random after saving")
        self.random_check.setToolTip(
            "Save and go on opens another random frame instead of the next frame of this "
            "video. Ticked by Random frame."
        )
        frames_layout.addWidget(self.random_check)
        panel_layout.addWidget(frames)

        boxes = QGroupBox("Boxes")
        boxes_layout = QVBoxLayout(boxes)
        self.buttons_label = QLabel("")
        boxes_layout.addWidget(self.buttons_label)
        self.prelabel_check = QCheckBox("Suggest boxes with the current models")
        self.prelabel_check.setChecked(True)
        boxes_layout.addWidget(self.prelabel_check)
        # How many of each are boxed, large: too many or too few shows at a glance
        self.count_label = QLabel("")
        self.count_label.setTextFormat(Qt.RichText)
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setToolTip(
            "Boxes on this frame. Green: as many as a point has on the field (14 players, "
            "1 disc). Red: more than that."
        )
        boxes_layout.addWidget(self.count_label)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        boxes_layout.addWidget(self.status_label)
        self.save_button = QPushButton("Save and go on (Enter)")
        self.save_button.clicked.connect(self._save_and_go_on)
        boxes_layout.addWidget(self.save_button)
        suggest_button = QPushButton("Suggest again")
        suggest_button.setToolTip("Replace the boxes with what the models find on this frame")
        suggest_button.clicked.connect(self._suggest_again)
        boxes_layout.addWidget(suggest_button)
        clear_button = QPushButton("Remove all boxes")
        clear_button.clicked.connect(self._clear_boxes)
        boxes_layout.addWidget(clear_button)
        self.remove_button = QPushButton("Take frame out of the dataset")
        self.remove_button.clicked.connect(self._remove_frame)
        boxes_layout.addWidget(self.remove_button)
        panel_layout.addWidget(boxes)

        dataset = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset)
        self.dataset_edit = QLineEdit(DEFAULT_DATASET)
        self.dataset_edit.setToolTip(
            "Folder in data/raw/training_data. A name with 'players' or 'disc' in it is "
            "listed in the Model Training tab."
        )
        self.dataset_edit.editingFinished.connect(self._show_frame)
        dataset_layout.addWidget(self.dataset_edit)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        dataset_layout.addWidget(self.summary_label)
        panel_layout.addWidget(dataset)

        help_label = QLabel(HELP_TEXT)
        help_label.setStyleSheet("color: #999;")
        panel_layout.addWidget(help_label)
        panel_layout.addStretch()

        self.canvas = BoxCanvas()
        self.canvas.boxes_changed.connect(self._on_boxes_changed)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(side_panel(panel))
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

    def _init_shortcuts(self) -> None:
        shortcuts = {
            Qt.Key_Return: self._save_and_go_on,
            Qt.Key_Enter: self._save_and_go_on,
            Qt.Key_Right: lambda: self._step(1),
            Qt.Key_Left: lambda: self._step(-1),
            Qt.Key_1: lambda: self._set_class(0),
            Qt.Key_2: lambda: self._set_class(1),
            Qt.Key_0: self.canvas.reset_view,
            Qt.Key_R: self._go_to_random_frame,
        }
        for key, action in shortcuts.items():
            shortcut = QShortcut(key, self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(action)

    # ------------------------------------------------------------------ video and frames

    def _dataset_dir(self) -> Path:
        name = self.dataset_edit.text().strip() or DEFAULT_DATASET
        return Path(DEFAULT_PATHS["TRAINING_DATA"]) / name

    def _on_video_selected(self, row: int) -> None:
        if not 0 <= row < len(self.video_list.video_files):
            return
        self._keep_edits()
        if self._capture is not None:
            self._capture.release()
        self._video_path = self.video_list.video_files[row]
        self._capture = cv2.VideoCapture(self._video_path)
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, self._frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.canvas.reset_view()
        self._frame_index = 0
        self._show_frame()

    def _step(self, direction: int) -> None:
        self._go_to(self._frame_index + direction * self.step_spin.value())

    def _go_to_random_frame(self) -> None:
        """Open a frame that is not labelled yet, anywhere in any video.

        Videos with few labels for their length are drawn more often, so the labels
        spread evenly over the footage. From here on, saving goes on to another random frame.
        """
        self.random_check.setChecked(True)
        videos = self.video_list.video_files
        for video in videos:
            if video not in self._frame_counts:
                capture = cv2.VideoCapture(video)
                self._frame_counts[video] = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                capture.release()
        counts = [self._frame_counts[video] for video in videos]
        if not any(counts):
            return
        # Videos with few labels for their length come up more often
        labelled = [
            len(label_files.labelled_frames(self._dataset_dir(), video)) for video in videos
        ]
        weights = label_files.random_video_weights(counts, labelled)

        for _ in range(RANDOM_FRAME_TRIES):
            row = random.choices(range(len(videos)), weights=weights)[0]
            index = random.randrange(counts[row])
            name = label_files.frame_name(videos[row], index)
            if (self._dataset_dir() / "labels" / f"{name}.txt").exists():
                continue
            if row != self.video_list.currentRow():
                self.video_list.setCurrentRow(row)  # Opens the video at its first frame
            self._go_to(index)
            return

    def _go_to_labelled(self, direction: int) -> None:
        """Go to the frame labelled before or after the current one, in any video.

        The frames are taken in the order they were labelled in: with random frames, the
        one labelled last is in another video. From a frame that is not labelled yet,
        "previous" is the one labelled last.
        """
        labels = self._dataset_dir() / "labels"
        rows = {Path(video).stem: row for row, video in enumerate(self.video_list.video_files)}
        # Creation time: changing a frame's boxes later does not move it in the order
        names = sorted(
            (
                name
                for name in label_files.labelled_frames(self._dataset_dir())
                if name.rsplit("_frame_", 1)[0] in rows
            ),
            key=lambda name: ((labels / f"{name}.txt").stat().st_ctime, name),
        )
        current = label_files.frame_name(self._video_path, self._frame_index)
        if current in names:
            position = names.index(current) + direction
        else:
            position = len(names) - 1 if direction < 0 else len(names)
        if not 0 <= position < len(names):
            return
        row = rows[names[position].rsplit("_frame_", 1)[0]]
        if row != self.video_list.currentRow():
            self.video_list.setCurrentRow(row)  # Opens the video at its first frame
        self._go_to(label_files.frame_index_of(names[position]))

    def _go_to(self, frame_index: int) -> None:
        if self._capture is None:
            return
        frame_index = min(max(frame_index, 0), max(0, self._frame_count - 1))
        if frame_index == self._frame_index and self._frame is not None:
            return
        self._keep_edits()
        self._frame_index = frame_index
        self._show_frame()

    def _show_frame(self) -> None:
        """Read the current frame and show it with its saved or suggested boxes."""
        if self._capture is None:
            return
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, self._frame_index)
        ok, frame = self._capture.read()
        if not ok:
            self.status_label.setText("This frame cannot be read")
            return
        self._frame = frame

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self._frame_index)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"Frame {self._frame_index} of {self._frame_count}")

        self._apply_dataset_classes()
        name = label_files.frame_name(self._video_path, self._frame_index)
        boxes = label_files.load_boxes(self._dataset_dir(), name, (frame.shape[1], frame.shape[0]))
        self._saved = boxes is not None
        self._edited = False
        self._awaiting_suggestion = None
        if boxes is None:
            boxes = self._suggest(frame) if self.prelabel_check.isChecked() else []
            if boxes is None:
                # The models are busy: the frame shows now, the suggestion follows
                boxes = []
                self._awaiting_suggestion = (self._video_path, self._frame_index)
                QTimer.singleShot(SUGGESTION_RETRY_MS, self._suggest_when_free)
        self.canvas.set_frame(frame, boxes, unconfirmed=not self._saved)
        self._update_status()

    # ------------------------------------------------------------------ boxes

    def _suggest(self, frame: np.ndarray) -> Optional[List[LabelBox]]:
        """Boxes the current default models find on a frame; None while the models are busy.

        The Main Analysis tab holds the models while it loads them at startup, for some
        ten seconds. Waiting for that here would freeze the window.
        """
        if not MODEL_LOCK.acquire(blocking=False):
            return None
        try:
            with MODEL_LOCK:
                if self._detectors is None:
                    # Own copies: the models the Main Analysis tab plays with keep their state
                    players = load_detection_model(default_model_path("player_detection"))
                    discs = load_detection_model(default_model_path("disc_detection"))
                    self._detectors = (players, discs)
                players, discs = self._detectors
                detections = []
                if players is not None:
                    found = detect_players(frame, *players)
                    # Observers are no players: left out, as the tracker leaves them out
                    if get_setting("models.tracking.hide_non_players", True):
                        observers = observers_in_frame(frame, [d["bbox"] for d in found])
                        found = [d for d, observer in zip(found, observers) if not observer]
                    detections += most_certain(found, MOST_ON_THE_FIELD["player"])
                if discs is not None:
                    detections += most_certain(
                        detect_discs(frame, *discs), MOST_ON_THE_FIELD["disc"]
                    )
        except Exception as e:
            logger.exception(f"Could not suggest boxes: {e}")
            return []
        finally:
            MODEL_LOCK.release()
        classes = label_files.dataset_classes(self._dataset_dir())
        return [
            LabelBox(CLASS_NAMES.index(detection["class_name"]), *map(float, detection["bbox"]))
            for detection in detections
            if detection["class_name"] in classes
        ]

    def _apply_dataset_classes(self) -> None:
        """Offer only the classes the dataset holds.

        A dataset labelled from the phone holds discs only. A player box does not belong
        in it, and its frames must not be read as "no players here".
        """
        classes = label_files.dataset_classes(self._dataset_dir())
        self.canvas.classes = {CLASS_NAMES.index(name) for name in classes}
        self.buttons_label.setText(
            "Left drag: disc    Right drag: player"
            if "player" in classes
            else "Left drag: disc (this dataset holds discs only)"
        )

    def _set_class(self, class_id: int) -> None:
        """Change the selected box to another class."""
        self.canvas.set_class(class_id)

    def _on_boxes_changed(self) -> None:
        self._edited = True
        self._update_status()

    def _suggest_when_free(self) -> None:
        """Add the suggestion a frame was shown without, unless the user has moved on."""
        if (
            self._awaiting_suggestion != (self._video_path, self._frame_index)
            or self._saved
            or self._edited
            or self.canvas.boxes
        ):
            self._awaiting_suggestion = None
            return
        boxes = self._suggest(self._frame)
        if boxes is None:
            QTimer.singleShot(SUGGESTION_RETRY_MS, self._suggest_when_free)
            return
        self._awaiting_suggestion = None
        self.canvas.set_frame(self._frame, boxes, unconfirmed=True)
        self._update_status()

    def _suggest_again(self) -> None:
        if self._frame is None:
            return
        boxes = self._suggest(self._frame)
        if boxes is None:
            self.status_label.setText("The models are busy; try again in a moment")
            return
        self.canvas.set_frame(self._frame, boxes, self.canvas.unconfirmed)
        self._on_boxes_changed()

    def _clear_boxes(self) -> None:
        if self._frame is not None:
            self.canvas.set_frame(self._frame, [], self.canvas.unconfirmed)
            self._on_boxes_changed()

    def _save(self) -> None:
        if self._frame is None:
            return
        name = label_files.frame_name(self._video_path, self._frame_index)
        label_files.save_frame(self._dataset_dir(), name, self._frame, self.canvas.boxes)
        self._saved, self._edited = True, False
        self.canvas.unconfirmed = False
        self.canvas.update()
        self._update_status()

    def _keep_edits(self) -> None:
        """Changes to a frame that is in the dataset are kept when leaving it.

        Suggested boxes on a new frame are only stored with Save: stepping past a frame
        must not put it into the dataset.
        """
        if self._saved and self._edited:
            self._save()

    def _save_and_go_on(self) -> None:
        self._save()
        if self.random_check.isChecked():
            self._go_to_random_frame()
        else:
            self._step(1)

    def _remove_frame(self) -> None:
        if self._frame is None or not self._saved:
            return
        name = label_files.frame_name(self._video_path, self._frame_index)
        label_files.remove_frame(self._dataset_dir(), name)
        self._show_frame()

    def _update_status(self) -> None:
        count = {name: 0 for name in CLASS_NAMES}
        for box in self.canvas.boxes:
            count[CLASS_NAMES[box.class_id]] += 1
        classes = label_files.dataset_classes(self._dataset_dir())
        self.count_label.setText(
            "&nbsp;&nbsp;&nbsp;".join(
                f"<span style='font-size:26px; font-weight:600; color:{count_colour(name, number)}'>"
                f"{number}</span> <span style='font-size:14px'>{name}"
                f"{'s' if number != 1 else ''}</span>"
                for name, number in count.items()
                if name in classes
            )
        )
        if self._awaiting_suggestion is not None:
            state = "Not saved yet: the models are loading, suggestions follow"
        elif self._saved:
            state = "In the dataset" + (" (changes are kept)" if self._edited else "")
        else:
            state = "Not saved yet: boxes are suggestions" if self.canvas.boxes else "Not saved yet"
        self.status_label.setText(state)
        self.remove_button.setEnabled(self._saved)

        totals = label_files.summary(self._dataset_dir())
        here = len(label_files.labelled_frames(self._dataset_dir(), self._video_path))
        self.summary_label.setText(
            f"{totals['frames']} frames ({here} of this video)\n"
            f"{totals['disc']} discs, {totals['player']} players\n"
            f"train {totals['train']}, validation {totals['val']}, test {totals['test']}"
        )

    def closeEvent(self, event):
        self._keep_edits()
        if self._capture is not None:
            self._capture.release()
        super().closeEvent(event)
