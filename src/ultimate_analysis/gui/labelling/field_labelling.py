"""Labelling the field: where its lines and marks are in a frame.

The whole field is drawn over the frame in perspective, and the user pulls that drawing
onto the real field: lengthening or shortening a line with the mouse wheel, moving a line,
a corner, or all of it. The drawing always stays a view of a flat field. A label is right
when the drawing lies on the real lines. Saved frames are verified calibrations, and the
training data for a model that finds the field by itself.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
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

from ...constants import DEFAULT_PATHS
from ...processing.camera_motion import CameraMotionEstimator
from ...processing.field_registration import estimate_field, found_lines
from ...processing.field_segmentation import reset_segmentation_cache, run_field_segmentation
from ...processing.model_lock import MODEL_LOCK
from ...utils import field_label_files, field_template, label_files
from ...utils.field_camera import fit_camera
from ...utils.field_label_files import FieldLabel
from ...utils.logger import get_logger
from ...utils.painted_lines import painted_line_mask
from ..widgets.panels import side_panel
from ..widgets.video_list import VideoListWidget
from .field_canvas import FieldCanvas
from .field_diagram import FieldDiagram, title

logger = get_logger("FIELD_LABELLING")

DEFAULT_DATASET = "labelled_field_v1"
# Labels are carried over to a frame at most this far away; further off, following the
# camera through every frame in between takes too long and drifts
MAX_CARRY_FRAMES = 150
# How often the field model is asked again when it was busy
MODEL_RETRY_MS = 300
# How often a random frame is drawn again when it turns out to show no field
RANDOM_FRAME_DRAWS = 6
HELP_TEXT = (
    "Put the corner dots of the drawing on the\n"
    "corners of the field:\n"
    "Drag a dot: move that corner. Dots placed in\n"
    "    this frame (filled) stay where they are,\n"
    "    the rest follows; three or four are enough\n"
    "The spot under a dot shows magnified beside it\n"
    "A corner snaps so its line lies on a long painted\n"
    "    line (turns yellow); Shift: no snapping\n"
    "Wheel: zoom, also out past the frame to reach\n"
    "    corners outside it\n"
    "Drag elsewhere: move the picture, 0: whole frame\n"
    "Enter: save and go on\n"
    "Left / Right: step without saving\n"
    "R: random frame of a random video"
)


class FieldLabellingWidget(QWidget):
    """Labels the lines and marks of the field on video frames."""

    def __init__(self):
        super().__init__()
        self._capture: Optional[cv2.VideoCapture] = None
        self._video_path = ""
        self._frame_count = 0
        self._frame_index = 0
        self._frame: Optional[np.ndarray] = None
        self._frame_counts: Dict[str, int] = {}
        # Focal length of a video's camera as its labelled frames give it: (frames, focal)
        self._focals: Dict[str, Tuple[Tuple[str, ...], Optional[float]]] = {}
        # What the field model sees in the frame shown: (lines by name, unnamed goal lines)
        self._model_lines: Optional[tuple] = None
        # (video, frame) shown without the field model's view because the model was busy
        self._awaiting_model: Optional[Tuple[str, int]] = None
        self._saved = False  # The shown frame is in the dataset
        self._edited = False
        self._template = field_template.TEMPLATES[field_template.DEFAULT_RULESET]

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
        panel_layout.addWidget(videos)

        frames = QGroupBox("Frame")
        frames_layout = QVBoxLayout(frames)
        self.frame_label = QLabel("No video")
        frames_layout.addWidget(self.frame_label)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setTracking(False)
        self.frame_slider.valueChanged.connect(self._go_to)
        frames_layout.addWidget(self.frame_slider)
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step (frames):"))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 3000)
        self.step_spin.setValue(60)
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
        random_button = QPushButton("Random frame (R)")
        random_button.clicked.connect(self._go_to_random_frame)
        frames_layout.addWidget(random_button)
        self.random_check = QCheckBox("Go on at random after saving")
        self.random_check.setChecked(True)
        frames_layout.addWidget(self.random_check)
        self.carry_check = QCheckBox("Keep the labels when stepping")
        self.carry_check.setChecked(True)
        self.carry_check.setToolTip(
            "Stepping to a frame nearby that is not labelled yet takes the labels along, "
            "moved the way the camera moved. They then only need correcting."
        )
        frames_layout.addWidget(self.carry_check)
        panel_layout.addWidget(frames)

        field = QGroupBox("Field")
        field_layout = QVBoxLayout(field)
        self.diagram = FieldDiagram(self._template)
        self.diagram.setToolTip("The line under the mouse in the frame is shown here")
        field_layout.addWidget(self.diagram)
        self.element_label = QLabel("")
        self.element_label.setWordWrap(True)
        field_layout.addWidget(self.element_label)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        field_layout.addWidget(self.status_label)
        self.save_button = QPushButton("Save and go on (Enter)")
        self.save_button.clicked.connect(self._save_and_go_on)
        field_layout.addWidget(self.save_button)
        self.estimate_check = QCheckBox("Start from the field model")
        self.estimate_check.setChecked(True)
        self.estimate_check.setToolTip(
            "A frame without a label starts with the field where the field model sees it,\n"
            "as far as that goes; the corners then only need correcting."
        )
        field_layout.addWidget(self.estimate_check)
        self.painted_check = QCheckBox("Show the painted lines found")
        self.painted_check.setChecked(True)
        self.painted_check.setToolTip(
            "Thin white streaks on the grass, found in the picture and marked in yellow:\n"
            "faint lines are easier to see, and the corners are where they meet.\n"
            "A guide for the eye: far lines come out in parts, and other streaks too."
        )
        self.painted_check.toggled.connect(self._show_painted_lines)
        field_layout.addWidget(self.painted_check)
        self.model_lines_check = QCheckBox("Show the field model's lines")
        self.model_lines_check.setChecked(False)
        self.model_lines_check.setToolTip(
            "The sidelines, back line and goal lines as the field model sees them,\n"
            "drawn faintly: what the estimate of the field is made from."
        )
        self.model_lines_check.toggled.connect(self._show_model_lines)
        field_layout.addWidget(self.model_lines_check)
        suggest_button = QPushButton("Place from the field model")
        suggest_button.setToolTip(
            "Put the drawing where the field model sees the field in this frame.\n"
            "The more frames of a video are labelled, the better its camera is known\n"
            "and the better this gets."
        )
        suggest_button.clicked.connect(self._suggest)
        field_layout.addWidget(suggest_button)
        clear_button = QPushButton("Start again")
        clear_button.setToolTip("Put the drawing back to a first guess for this kind of view")
        clear_button.clicked.connect(self._clear)
        field_layout.addWidget(clear_button)
        self.remove_button = QPushButton("Take frame out of the dataset")
        self.remove_button.clicked.connect(self._remove_frame)
        field_layout.addWidget(self.remove_button)
        panel_layout.addWidget(field)

        dataset = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset)
        self.dataset_edit = QLineEdit(DEFAULT_DATASET)
        self.dataset_edit.setToolTip("Folder in data/raw/training_data")
        self.dataset_edit.editingFinished.connect(self._on_dataset_changed)
        dataset_layout.addWidget(self.dataset_edit)
        ruleset_row = QHBoxLayout()
        ruleset_row.addWidget(QLabel("Field:"))
        self.ruleset_combo = QComboBox()
        self.ruleset_combo.addItem("USA Ultimate (110 x 40 yd)", "usau")
        self.ruleset_combo.addItem("WFDF (100 x 37 m)", "wfdf")
        self.ruleset_combo.setToolTip("The sizes the field is drawn by; stored with the dataset")
        self.ruleset_combo.currentIndexChanged.connect(self._on_ruleset_changed)
        ruleset_row.addWidget(self.ruleset_combo, 1)
        dataset_layout.addLayout(ruleset_row)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        dataset_layout.addWidget(self.summary_label)
        panel_layout.addWidget(dataset)

        help_label = QLabel(HELP_TEXT)
        help_label.setStyleSheet("color: #999;")
        panel_layout.addWidget(help_label)
        panel_layout.addStretch()

        self.canvas = FieldCanvas(self._template)
        self.canvas.label_changed.connect(self._on_label_changed)
        self.canvas.hovered_changed.connect(self._on_hovered)

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
            Qt.Key_0: self.canvas.reset_view,
            Qt.Key_R: self._go_to_random_frame,
        }
        for key, action in shortcuts.items():
            shortcut = QShortcut(key, self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(action)

    # ------------------------------------------------------------------ dataset

    def _dataset_dir(self) -> Path:
        name = self.dataset_edit.text().strip() or DEFAULT_DATASET
        return Path(DEFAULT_PATHS["TRAINING_DATA"]) / name

    def _on_dataset_changed(self) -> None:
        ruleset = field_label_files.dataset_ruleset(self._dataset_dir())
        self.ruleset_combo.blockSignals(True)
        self.ruleset_combo.setCurrentIndex(max(0, self.ruleset_combo.findData(ruleset)))
        self.ruleset_combo.blockSignals(False)
        self._apply_ruleset()
        self._show_frame()

    def _on_ruleset_changed(self) -> None:
        self._apply_ruleset()
        self.canvas.update()
        self._on_label_changed()

    def _apply_ruleset(self) -> None:
        self._template = field_template.TEMPLATES[self.ruleset_combo.currentData()]
        self.diagram.set_template(self._template)
        self.canvas.template = self._template

    # ------------------------------------------------------------------ video and frames

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

    def _read(self, frame_index: int) -> Optional[np.ndarray]:
        if self._capture is None:
            return None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._capture.read()
        return frame if ok else None

    def _step(self, direction: int) -> None:
        self._go_to(self._frame_index + direction * self.step_spin.value(), carry=True)

    def _go_to(self, frame_index: int, carry: bool = False) -> None:
        if self._capture is None:
            return
        frame_index = min(max(frame_index, 0), max(0, self._frame_count - 1))
        if frame_index == self._frame_index and self._frame is not None:
            return
        self._keep_edits()
        carried = None
        if carry and self.carry_check.isChecked() and self.canvas.mapping is not None:
            carried = self._carried_field(self._frame_index, frame_index)
        self._frame_index = frame_index
        self._show_frame(carried)

    def _go_to_random_frame(self) -> None:
        """Open a frame that is not labelled yet, anywhere in any video."""
        self.random_check.setChecked(True)
        videos = self.video_list.video_files
        for video in videos:
            if video not in self._frame_counts:
                capture = cv2.VideoCapture(video)
                self._frame_counts[video] = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                capture.release()
        # Edited games cut to close-ups; a few more draws find drone footage again
        for _ in range(RANDOM_FRAME_DRAWS):
            picked = label_files.random_unlabelled_frame(
                videos,
                [self._frame_counts[video] for video in videos],
                [field_label_files.labelled_frames(self._dataset_dir(), video) for video in videos],
            )
            if picked is None:
                return
            row, index = picked
            if row != self.video_list.currentRow():
                self.video_list.setCurrentRow(row)  # Opens the video at its first frame
            self._go_to(index)
            # A frame in which the field model sees no field at all is no drone footage.
            # While the model is busy that cannot be told, and the frame is taken
            if self._awaiting_model is not None or self._model_lines is None:
                return
            named, unnamed = self._model_lines
            if named or unnamed:
                return

    def _show_frame(self, carried: Optional[np.ndarray] = None) -> None:
        """Read the current frame and show it with its saved field, or one to start from."""
        frame = self._read(self._frame_index)
        if frame is None:
            self.status_label.setText("This frame cannot be read")
            return
        self._frame = frame
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self._frame_index)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"Frame {self._frame_index} of {self._frame_count}")

        self._awaiting_model = None
        self._model_lines = self._lines_of_the_model()
        name = label_files.frame_name(self._video_path, self._frame_index)
        mapping = self._mapping_of(field_label_files.load_label(self._dataset_dir(), name))
        self._saved = mapping is not None
        self._edited = False
        if mapping is None:
            # To start from: the field of the frame just left, else where the field model
            # sees it, else that of the nearest labelled frame of this video, else a guess
            mapping = carried
            if mapping is None and self.estimate_check.isChecked():
                mapping = self._estimated_field()
            if mapping is None:
                mapping = self._nearest_labelled_field()
        self.canvas.focal = self._video_focal()
        self.canvas.set_frame(frame, mapping, unconfirmed=not self._saved)
        self._show_model_lines()
        self._update_status()
        # Finding the painted lines takes a moment: the frame shows first
        self.canvas.set_painted_lines(None)
        shown = (self._video_path, self._frame_index)
        QTimer.singleShot(0, lambda: self._show_painted_lines(only_for=shown))

    # ------------------------------------------------------------------ labels

    def _mapping_of(self, label: Optional[FieldLabel]) -> Optional[np.ndarray]:
        """The mapping from field to frame that a stored label gives."""
        if label is None:
            return None
        fit = field_template.fit_field(self._template, label.lines, label.points)
        return fit.field_to_image * fit.front_sign if fit is not None else None

    def _nearest_labelled_field(self) -> Optional[np.ndarray]:
        names = field_label_files.labelled_frames(self._dataset_dir(), self._video_path)
        if not names:
            return None
        nearest = min(
            names, key=lambda name: abs(label_files.frame_index_of(name) - self._frame_index)
        )
        return self._mapping_of(field_label_files.load_label(self._dataset_dir(), nearest))

    def _carried_field(self, from_index: int, to_index: int) -> Optional[np.ndarray]:
        """The field as it lies in another frame nearby, moved with the camera."""
        if abs(to_index - from_index) > MAX_CARRY_FRAMES or self._capture is None:
            return None
        first, last = sorted((from_index, to_index))
        estimator = CameraMotionEstimator()
        motion = np.eye(3)
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for index in range(first, last + 1):
            ok, frame = self._capture.read()
            if not ok:
                return None
            step = estimator.update(frame, [])
            if step is not None:
                motion = step @ motion
            elif index != first:
                return None  # A cut, or nothing to follow: the field does not carry over
        if to_index < from_index:
            motion = np.linalg.inv(motion)
        return motion @ self.canvas.mapping

    def _video_focal(self) -> Optional[float]:
        """Focal length of this video's camera, as the corners of its labelled frames give it."""
        names = tuple(field_label_files.labelled_frames(self._dataset_dir(), self._video_path))
        known = self._focals.get(self._video_path)
        if known is not None and known[0] == names:
            return known[1]
        size = (self._frame.shape[1], self._frame.shape[0])
        focals = []
        for name in names:
            label = field_label_files.load_label(self._dataset_dir(), name)
            fit = fit_camera(self._template, {}, label.points, size) if label else None
            if fit is not None:
                focals.append(fit.focal)
        focal = float(np.median(focals)) if focals else None
        self._focals[self._video_path] = (names, focal)
        return focal

    def _lines_of_the_model(self) -> Optional[tuple]:
        """The lines of the field that the field model sees in the current frame.

        None if the model is busy (the Main Analysis tab holds the models while it loads
        them at startup; waiting here would freeze the window) or fails.
        """
        if not MODEL_LOCK.acquire(blocking=False):
            self._awaiting_model = (self._video_path, self._frame_index)
            QTimer.singleShot(MODEL_RETRY_MS, self._model_when_free)
            return None
        try:
            reset_segmentation_cache()
            results = run_field_segmentation(self._frame, 0)
            reset_segmentation_cache()
            return found_lines(results, self._frame.shape[:2])
        except Exception as e:
            logger.exception(f"Could not run the field model: {e}")
            return None
        finally:
            MODEL_LOCK.release()

    def _model_when_free(self) -> None:
        """Bring in what the field model sees for a frame that was shown without it."""
        if self._awaiting_model != (self._video_path, self._frame_index):
            return
        self._awaiting_model = None
        self._model_lines = self._lines_of_the_model()
        if self._model_lines is None:
            return  # Still busy (asked again by itself), or failed
        self._show_model_lines()
        # The frame had to start without the estimate; it gets it if nothing was done yet
        if not self._saved and not self._edited and self.estimate_check.isChecked():
            mapping = self._estimated_field()
            if mapping is not None:
                self.canvas.set_frame(self._frame, mapping, unconfirmed=True)

    def _show_painted_lines(self, _checked: bool = True, only_for: Optional[tuple] = None) -> None:
        """Mark the painted lines found in the frame shown, if that is wanted."""
        if only_for is not None and only_for != (self._video_path, self._frame_index):
            return  # The user has moved on
        if self._frame is None or not self.painted_check.isChecked():
            self.canvas.set_painted_lines(None)
            return
        self.canvas.set_painted_lines(painted_line_mask(self._frame))

    def _show_model_lines(self) -> None:
        shown = self.model_lines_check.isChecked() and self._model_lines is not None
        named, unnamed = self._model_lines if shown else ({}, [])
        self.canvas.model_lines = [*named.values(), *unnamed]
        self.canvas.update()

    def _estimated_field(self) -> Optional[np.ndarray]:
        """Where the field model sees the field in the current frame, as a mapping."""
        if self._frame is None or self._model_lines is None:
            return None
        estimate = estimate_field(
            [],
            self._frame.shape[:2],
            self._template,
            focal=self._video_focal(),
            lines=self._model_lines,
        )
        return estimate.field_to_image if estimate is not None else None

    def _suggest(self) -> None:
        """Put the drawing where the field model sees the field."""
        mapping = self._estimated_field()
        if mapping is None:
            self.status_label.setText("The field model does not show enough of the field here")
            return
        self.canvas.set_frame(self._frame, mapping, unconfirmed=not self._saved)
        self._on_label_changed()

    def _clear(self) -> None:
        if self._frame is not None:
            self.canvas.set_frame(self._frame, None, unconfirmed=not self._saved)
            self._on_label_changed()

    def _on_hovered(self, name: str) -> None:
        self.diagram.set_state(set(), name or None)
        self.element_label.setText(title(name) if name else "")

    def _on_label_changed(self) -> None:
        self._edited = True
        self._update_status()

    def _save(self) -> None:
        if self._frame is None or self.canvas.mapping is None:
            return
        name = label_files.frame_name(self._video_path, self._frame_index)
        field_label_files.save_frame(
            self._dataset_dir(),
            name,
            self._frame,
            self.canvas.label(),
            self.ruleset_combo.currentData(),
        )
        self._saved, self._edited = True, False
        self.canvas.unconfirmed = False
        self.canvas.update()
        self._update_status()

    def _keep_edits(self) -> None:
        """Changes to a frame that is in the dataset are kept when leaving it."""
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
        field_label_files.remove_frame(self._dataset_dir(), name)
        self._show_frame()

    def _update_status(self) -> None:
        state = (
            "In the dataset" + (" (changes are kept)" if self._edited else "")
            if self._saved
            else "Not saved yet: pull the drawing onto the field, then save"
        )
        self.status_label.setText(state)
        self.remove_button.setEnabled(self._saved)
        self.save_button.setEnabled(self.canvas.mapping is not None)

        counts = field_label_files.summary(self._dataset_dir())
        self.summary_label.setText(
            f"{counts['frames']} frames, {counts['calibrated']} with the field placed\n"
            f"train {counts['train']}, validation {counts['val']}, test {counts['test']}"
        )
