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
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...constants import DEFAULT_PATHS
from ...processing.inference import detect_discs, detect_players, load_detection_model
from ...processing.model_lock import MODEL_LOCK
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
    "Drag: new box (also across other boxes)\n"
    "Click a box: select it\n"
    "Drag the selected box: move, a grip: resize\n"
    "Delete: remove the selected box\n"
    "1 / 2: disc / player\n"
    "Wheel: zoom, right drag: move view, 0: whole frame\n"
    "Enter: save and go on\n"
    "Left / Right: step without saving\n"
    "R: random frame of a random video;\n"
    "    saving then goes on at random too"
)


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
        class_row = QHBoxLayout()
        self.class_buttons = QButtonGroup(self)
        for class_id, name in enumerate(CLASS_NAMES):
            button = QRadioButton(f"{name.capitalize()} ({class_id + 1})")
            button.setChecked(class_id == 0)
            self.class_buttons.addButton(button, class_id)
            class_row.addWidget(button)
        self.class_buttons.idClicked.connect(self._set_class)
        boxes_layout.addLayout(class_row)
        self.prelabel_check = QCheckBox("Suggest boxes with the current models")
        self.prelabel_check.setChecked(True)
        boxes_layout.addWidget(self.prelabel_check)
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

        Every frame is equally likely, so a full game is drawn far more often than a
        short clip. From here on, saving goes on to another random frame.
        """
        self.random_check.setChecked(True)
        videos = self.video_list.video_files
        for video in videos:
            if video not in self._frame_counts:
                capture = cv2.VideoCapture(video)
                self._frame_counts[video] = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                capture.release()
        weights = [self._frame_counts[video] for video in videos]
        if not any(weights):
            return

        for _ in range(RANDOM_FRAME_TRIES):
            row = random.choices(range(len(videos)), weights=weights)[0]
            index = random.randrange(weights[row])
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
        if boxes is None:
            boxes = self._suggest(frame) if self.prelabel_check.isChecked() else []
        self.canvas.set_frame(frame, boxes, unconfirmed=not self._saved)
        self._update_status()

    # ------------------------------------------------------------------ boxes

    def _suggest(self, frame: np.ndarray) -> List[LabelBox]:
        """Boxes the current default models find on a frame."""
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
                    detections += detect_players(frame, *players)
                if discs is not None:
                    detections += detect_discs(frame, *discs)
        except Exception as e:
            logger.exception(f"Could not suggest boxes: {e}")
            return []
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
        for class_id, name in enumerate(CLASS_NAMES):
            self.class_buttons.button(class_id).setEnabled(name in classes)
        if CLASS_NAMES[self.canvas.current_class] not in classes:
            self._set_class(CLASS_NAMES.index(classes[0]))

    def _set_class(self, class_id: int) -> None:
        if not self.class_buttons.button(class_id).isEnabled():
            return
        self.class_buttons.button(class_id).setChecked(True)
        self.canvas.set_class(class_id)

    def _on_boxes_changed(self) -> None:
        self._edited = True
        self._update_status()

    def _suggest_again(self) -> None:
        if self._frame is not None:
            self.canvas.set_frame(self._frame, self._suggest(self._frame), self.canvas.unconfirmed)
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
        found = ", ".join(
            f"{number} {name}{'s' if number != 1 else ''}" for name, number in count.items()
        )
        if self._saved:
            state = "In the dataset" + (" (changes are kept)" if self._edited else "")
        else:
            state = "Not saved yet: boxes are suggestions" if self.canvas.boxes else "Not saved yet"
        self.status_label.setText(f"{state}\n{found}")
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
