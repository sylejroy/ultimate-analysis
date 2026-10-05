"""Main analysis tab: video list, playback, processing options, and the two views.

The tab only handles the interface. Decoding and analysis run on a worker thread
(pipeline_worker.py); the tab sends it requests and displays the frames that come back.
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...config.settings import get_config, get_setting
from ...constants import SHORTCUTS
from ...pipeline import PipelineOptions
from ...processing.homography import load_default_matrix
from ...processing.jersey_readers import READER_LABELS
from ...processing.player_id import get_player_id_method
from ...utils.logger import get_logger
from ...utils.model_files import default_model_path, models_root
from ..widgets.images import frame_to_pixmap
from ..widgets.model_selection import (
    populate_detection_model_combo,
    populate_segmentation_model_combo,
)
from ..widgets.panels import PANEL_WIDTH, collapsible, compact_combo, side_panel
from ..widgets.performance_widget import PerformanceWidget
from ..widgets.video_list import VideoListWidget
from ..widgets.zoomable_image_label import ZoomableImageLabel
from .pipeline_worker import PipelineWorker, ProcessedFrame

logger = get_logger("MAIN_TAB")

# Closing waits this long for the worker, which may still be loading the models
WORKER_SHUTDOWN_TIMEOUT_MS = 60000


class MainTab(QWidget):
    """Main video analysis tab with video player and processing controls."""

    video_changed = pyqtSignal(str)  # Path of the video that was loaded

    # Requests to the worker thread; they are queued and run in the order sent
    _load_video_requested = pyqtSignal(str, object, object)  # path, options, homography matrix
    _frame_requested = pyqtSignal(str, object, int)  # mode, options, generation
    _command_requested = pyqtSignal(object)  # callable taking the worker

    def __init__(self):
        super().__init__()

        self.video_files: List[str] = []
        self.current_video_index: int = 0
        self.video_info: Dict[str, Any] = {}  # Of the loaded video; empty while none is loaded
        self.is_playing: bool = False
        self.homography_enabled = True

        # Results of the frame on screen
        self.current_detections: List[Dict] = []
        self.current_tracks: List[Any] = []
        self.current_player_ids: Dict[int, Tuple[str, Any]] = {}

        # Frames in flight become stale when the video position or the models change;
        # their generation number then no longer matches and they are not shown.
        self._generation = 0
        self._busy = False  # A frame request is with the worker
        self._redraw_pending = False  # The current frame must be drawn again when it returns
        self._last_request_time = 0.0
        self._frame_interval_ms = 40.0

        self._start_worker()

        # Playback requests the next frame when the previous one is back, no sooner than
        # the video's frame interval.
        self.playback_timer = QTimer()
        self.playback_timer.setSingleShot(True)
        self.playback_timer.timeout.connect(lambda: self._request_frame("next"))

        # Several option changes in quick succession lead to one redraw
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(lambda: self._request_frame("current"))
        self.debounce_delay_ms = 100

        self._init_ui()
        self._init_shortcuts()

        # Models load on the worker thread; the window does not wait for them
        self._on_player_model_changed(self.player_model_combo.currentText())
        self._on_disc_model_changed(self.disc_model_combo.currentText())
        self._load_segmentation_models()

        self._reload_videos()
        self._select_default_video()

    def _start_worker(self) -> None:
        self._worker_thread = QThread(self)
        self._worker = PipelineWorker()
        self._worker.moveToThread(self._worker_thread)

        self._load_video_requested.connect(self._worker.load_video)
        self._frame_requested.connect(self._worker.process_frame)
        self._command_requested.connect(self._worker.execute)
        self._worker.frame_processed.connect(self._on_frame_processed)
        self._worker.video_loaded.connect(self._on_video_loaded)

        self._worker_thread.start()

    # ------------------------------------------------------------------ interface

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Video list and controls
        splitter.addWidget(side_panel(self._create_left_panel()))

        # Center panel: Main video display
        center_panel = self._create_center_panel()
        splitter.addWidget(center_panel)

        # Right panel: Homography controls and top-down view
        right_panel = self._create_right_panel()
        right_panel.setMinimumWidth(300)  # Minimum width for homography controls
        right_panel.setMaximumWidth(600)  # Maximum width to prevent taking too much space
        splitter.addWidget(right_panel)

        # Simple initial sizing - left takes ~20%, center takes ~50%, right takes ~30%
        # The video gets whatever the side panels leave free
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([PANEL_WIDTH, 1600, 420])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with video list and controls."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Video list section
        video_group = QGroupBox("Available Videos")
        video_layout = QVBoxLayout()

        # Video list with refresh button
        list_header = QHBoxLayout()
        list_header.addWidget(QLabel("Videos"))

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._reload_videos)
        refresh_button.setToolTip("Refresh video list")
        list_header.addWidget(refresh_button)

        video_layout.addLayout(list_header)

        # Video list widget
        self.video_list = VideoListWidget()
        self.video_list.currentRowChanged.connect(self._on_video_selection_changed)
        video_layout.addWidget(self.video_list)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # Processing controls section
        processing_group = QGroupBox("Processing Options")
        processing_layout = QVBoxLayout()

        # Checkboxes for processing features
        self.inference_checkbox = QCheckBox("Object Detection (Inference)")
        self.inference_checkbox.setToolTip(
            f"Enable/disable object detection [{SHORTCUTS['TOGGLE_INFERENCE']}]"
        )
        self.inference_checkbox.setChecked(True)  # Enable inference by default
        self.inference_checkbox.stateChanged.connect(self._on_inference_toggled)

        self.tracking_checkbox = QCheckBox("Object Tracking")
        self.tracking_checkbox.setToolTip(
            f"Enable/disable object tracking [{SHORTCUTS['TOGGLE_TRACKING']}]"
        )
        self.tracking_checkbox.setChecked(True)  # Enable tracking by default
        self.tracking_checkbox.stateChanged.connect(self._on_tracking_toggled)

        self.player_id_checkbox = QCheckBox("Player Identification")
        self.player_id_checkbox.setToolTip(
            f"Enable/disable player ID based on jersey numbers [{SHORTCUTS['TOGGLE_PLAYER_ID']}]"
        )
        # Enable player ID by default so OCR-based jersey identification starts automatically
        self.player_id_checkbox.setChecked(True)
        self.player_id_checkbox.stateChanged.connect(self._on_player_id_toggled)

        self.field_segmentation_checkbox = QCheckBox("Field Segmentation")
        self.field_segmentation_checkbox.setToolTip(
            f"Enable/disable field boundary detection [{SHORTCUTS['TOGGLE_FIELD_SEGMENTATION']}]"
        )
        self.field_segmentation_checkbox.stateChanged.connect(self._on_field_segmentation_toggled)
        self.field_segmentation_checkbox.setChecked(
            True
        )  # Enable by default to show advanced field line detection

        self.homography_checkbox = QCheckBox("Enable Top-Down View")
        self.homography_checkbox.setChecked(True)  # Enable by default
        self.homography_checkbox.setToolTip(
            "Enable/disable homography transformation for top-down view"
        )
        self.homography_checkbox.stateChanged.connect(self._on_homography_toggled)

        processing_layout.addWidget(self.inference_checkbox)
        processing_layout.addWidget(self.tracking_checkbox)
        processing_layout.addWidget(self.player_id_checkbox)
        processing_layout.addWidget(self.field_segmentation_checkbox)
        processing_layout.addWidget(self.homography_checkbox)

        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)

        # Model selection section
        models_group = QGroupBox("Model Settings")
        models_layout = QFormLayout()
        # Model names are long: each label goes above its dropdown
        models_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)

        self.player_model_combo = compact_combo(QComboBox())
        populate_detection_model_combo(
            self.player_model_combo,
            "player",
            default_model_path("player_detection"),
        )
        self.player_model_combo.currentTextChanged.connect(self._on_player_model_changed)
        models_layout.addRow("Player Detection Model:", self.player_model_combo)

        # Jersey number reader dropdown
        self.player_id_method_combo = compact_combo(QComboBox())
        for method, label in READER_LABELS.items():
            self.player_id_method_combo.addItem(label, method)
        self.player_id_method_combo.setCurrentIndex(
            self.player_id_method_combo.findData(get_player_id_method())
        )
        self.player_id_method_combo.setToolTip(
            "How jersey numbers are read.\n"
            "PARSeq + text detector: most accurate in testing, same speed as EasyOCR\n"
            "Florence-2: similar accuracy, about 4x slower\n"
            "YOLO digit detector: needs a model trained on a digits dataset\n"
            "A reader that cannot be loaded falls back to EasyOCR."
        )
        self.player_id_method_combo.currentIndexChanged.connect(self._on_player_id_method_changed)
        models_layout.addRow("Jersey Number Reader:", self.player_id_method_combo)

        # Disc detection model dropdown
        self.disc_model_combo = compact_combo(QComboBox())
        populate_detection_model_combo(
            self.disc_model_combo,
            "disc",
            default_model_path("disc_detection"),
        )
        self.disc_model_combo.currentTextChanged.connect(self._on_disc_model_changed)
        models_layout.addRow("Disc Detection Model:", self.disc_model_combo)

        models_group.setLayout(models_layout)
        layout.addWidget(models_group)

        # Field Segmentation Controls
        segmentation_group = QGroupBox("Field Segmentation")
        segmentation_layout = QVBoxLayout()

        # RANSAC line fitting checkbox
        self.ransac_checkbox = QCheckBox("Use RANSAC Line Fitting")
        self.ransac_checkbox.setChecked(True)  # Enable by default for advanced line detection
        self.ransac_checkbox.stateChanged.connect(self._on_ransac_toggled)
        self.ransac_checkbox.setToolTip(
            "Fit straight lines to contour segments using RANSAC algorithm"
        )
        segmentation_layout.addWidget(self.ransac_checkbox)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))

        self.segmentation_model_combo = compact_combo(QComboBox())
        self.segmentation_model_combo.currentTextChanged.connect(
            self._on_segmentation_model_changed
        )
        model_layout.addWidget(self.segmentation_model_combo)

        refresh_models_button = QPushButton("↻")
        refresh_models_button.setMaximumWidth(30)
        refresh_models_button.setToolTip("Refresh model list")
        refresh_models_button.clicked.connect(self._load_segmentation_models)
        model_layout.addWidget(refresh_models_button)

        segmentation_layout.addLayout(model_layout)
        segmentation_group.setLayout(segmentation_layout)
        layout.addWidget(segmentation_group)

        # Performance metrics section
        # Only needed when tuning, so it is folded away until asked for
        timings_group = QGroupBox("Stage Timings")
        timings_layout = QVBoxLayout()
        self.performance_widget = PerformanceWidget()
        timings_layout.addWidget(self.performance_widget)
        timings_group.setLayout(timings_layout)
        layout.addWidget(collapsible(timings_group, expanded=False))

        # Add stretch to push everything to top
        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def _create_center_panel(self) -> QWidget:
        """Create the center panel with main video display and controls."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Video display area with zoom capability
        self.video_scroll_area = QScrollArea()
        self.video_scroll_area.setWidgetResizable(True)
        self.video_scroll_area.setMinimumHeight(360)
        self.video_scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: 2px solid #555;
                background-color: #1a1a1a;
            }
        """
        )

        self.video_label = ZoomableImageLabel()
        self.video_label.setText("No video selected")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            """
            QLabel {
                background-color: #1a1a1a;
                color: #999;
                font-size: 14px;
            }
        """
        )
        self.video_scroll_area.setWidget(self.video_label)
        layout.addWidget(self.video_scroll_area, 1)  # Takes most space

        # Progress bar
        self.progress_bar = QSlider(Qt.Horizontal)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.sliderMoved.connect(self._on_seek)
        layout.addWidget(self.progress_bar)

        # Control buttons
        controls_layout = QHBoxLayout()

        # Previous video button
        self.prev_button = QPushButton("⏮")
        self.prev_button.setToolTip(f"Previous video [{SHORTCUTS['PREV_VIDEO']}]")
        self.prev_button.clicked.connect(self._prev_video)
        self.prev_button.setFixedSize(40, 40)
        controls_layout.addWidget(self.prev_button)

        # Play/Pause button
        self.play_pause_button = QPushButton("▶")
        self.play_pause_button.setToolTip(f"Play/Pause [{SHORTCUTS['PLAY_PAUSE']}]")
        self.play_pause_button.clicked.connect(self._toggle_play_pause)
        self.play_pause_button.setFixedSize(60, 40)
        controls_layout.addWidget(self.play_pause_button)

        # Next video button
        self.next_button = QPushButton("⏭")
        self.next_button.setToolTip(f"Next video [{SHORTCUTS['NEXT_VIDEO']}]")
        self.next_button.clicked.connect(self._next_video)
        self.next_button.setFixedSize(40, 40)
        controls_layout.addWidget(self.next_button)

        # Add some space
        controls_layout.addStretch()

        # Reset tracker button
        reset_button = QPushButton("Reset Tracker")
        reset_button.setToolTip(f"Reset object tracker [{SHORTCUTS['RESET_TRACKER']}]")
        reset_button.clicked.connect(self._reset_tracker)
        reset_button.setFixedHeight(40)
        controls_layout.addWidget(reset_button)

        layout.addLayout(controls_layout)

        panel.setLayout(layout)
        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with top-down view."""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins for more space

        # Top-down view display (expanded to fill the panel with zoom capability)
        view_group = QGroupBox("Top-Down View")
        view_layout = QVBoxLayout()
        view_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins inside group box

        self.homography_scroll_area = QScrollArea()
        self.homography_scroll_area.setWidgetResizable(True)
        self.homography_scroll_area.setMinimumHeight(300)  # Reasonable minimum
        self.homography_scroll_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.homography_scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: 2px solid #555;
                background-color: #1a1a1a;
            }
        """
        )

        self.homography_display_label = ZoomableImageLabel()
        self.homography_display_label.setText("Loading top-down view...")
        self.homography_display_label.setAlignment(Qt.AlignCenter)
        self.homography_display_label.setStyleSheet(
            """
            QLabel {
                background-color: #1a1a1a;
                color: #999;
                font-size: 14px;
            }
        """
        )
        self.homography_scroll_area.setWidget(self.homography_display_label)
        view_layout.addWidget(self.homography_scroll_area)

        view_group.setLayout(view_layout)
        layout.addWidget(
            view_group, 1
        )  # Give the view group stretch factor of 1 to fill available space

        panel.setLayout(layout)
        return panel

    def _init_shortcuts(self):
        """Initialize keyboard shortcuts."""
        # Play/Pause
        QShortcut(QKeySequence(SHORTCUTS["PLAY_PAUSE"]), self, self._toggle_play_pause)

        # Previous/Next video
        QShortcut(QKeySequence(SHORTCUTS["PREV_VIDEO"]), self, self._prev_video)
        QShortcut(QKeySequence(SHORTCUTS["NEXT_VIDEO"]), self, self._next_video)

        # Reset tracker
        QShortcut(QKeySequence(SHORTCUTS["RESET_TRACKER"]), self, self._reset_tracker)

        # Toggle processing features
        QShortcut(
            QKeySequence(SHORTCUTS["TOGGLE_INFERENCE"]),
            self,
            lambda: self.inference_checkbox.toggle(),
        )
        QShortcut(
            QKeySequence(SHORTCUTS["TOGGLE_TRACKING"]),
            self,
            lambda: self.tracking_checkbox.toggle(),
        )
        QShortcut(
            QKeySequence(SHORTCUTS["TOGGLE_PLAYER_ID"]),
            self,
            lambda: self.player_id_checkbox.toggle(),
        )
        QShortcut(
            QKeySequence(SHORTCUTS["TOGGLE_FIELD_SEGMENTATION"]),
            self,
            lambda: self.field_segmentation_checkbox.toggle(),
        )

    # ------------------------------------------------------------------ worker requests

    def _run_on_worker(self, command: Callable[[PipelineWorker], None]) -> None:
        """Queue a command for the worker thread."""
        self._command_requested.emit(command)

    def _options(self) -> PipelineOptions:
        """The processing options as currently ticked."""
        return PipelineOptions(
            detection=self.inference_checkbox.isChecked(),
            tracking=self.tracking_checkbox.isChecked(),
            player_id=self.player_id_checkbox.isChecked(),
            field_segmentation=self.field_segmentation_checkbox.isChecked(),
            top_down_view=self.homography_enabled,
        )

    def _request_frame(self, mode: str) -> None:
        """Ask the worker for the next frame ("next") or a redraw ("current").

        Only one request is in flight at a time, so playback never builds up a backlog.
        """
        if not self.video_info:
            return
        if self._busy:
            if mode == "current":
                self._redraw_pending = True
            return
        self._busy = True
        self._last_request_time = time.perf_counter()
        self._frame_requested.emit(mode, self._options(), self._generation)

    def _request_display_update(self, immediate: bool = False) -> None:
        """Redraw the current frame, by default after a short pause to merge changes."""
        if immediate:
            self.update_timer.stop()
            self._request_frame("current")
        else:
            self.update_timer.start(self.debounce_delay_ms)

    def _on_frame_processed(self, processed: ProcessedFrame) -> None:
        """Show a frame from the worker and keep playback going."""
        self._busy = False

        if processed.generation == self._generation:
            if processed.result is not None:
                self._show_frame(processed)
            elif processed.mode == "next":
                logger.info("End of video reached")
                self._stop_playback()

        if self._redraw_pending:
            self._redraw_pending = False
            self._request_frame("current")
        elif self.is_playing:
            elapsed_ms = (time.perf_counter() - self._last_request_time) * 1000
            self.playback_timer.start(max(0, int(self._frame_interval_ms - elapsed_ms)))

    def _show_frame(self, processed: ProcessedFrame) -> None:
        result = processed.result
        self.current_detections = result.detections
        self.current_tracks = result.tracks
        self.current_player_ids = result.player_ids

        display_start = time.perf_counter()
        self.video_label.set_image(frame_to_pixmap(result.main_view))
        if result.top_down_view is not None:
            self.homography_display_label.set_image(frame_to_pixmap(result.top_down_view))
        else:
            self.homography_display_label.setText(result.top_down_message)
        if processed.mode == "next":
            self.progress_bar.setValue(processed.video_position)
        display_ms = (time.perf_counter() - display_start) * 1000

        if self.performance_widget.isVisible():
            self.performance_widget.begin_frame()
            for stage, duration_ms in result.timings.items():
                self.performance_widget.add_processing_measurement(stage, duration_ms)
            self.performance_widget.add_processing_measurement("UI Display", display_ms)

    # ------------------------------------------------------------------ videos

    def _reload_videos(self) -> None:
        """Search the video folders again, keeping the current video selected."""
        current = self.video_info.get("path")
        self.video_files = self.video_list.reload()
        if current in self.video_files:
            self.current_video_index = self.video_files.index(current)
            self.video_list.blockSignals(True)
            self.video_list.setCurrentRow(self.current_video_index)
            self.video_list.blockSignals(False)
        logger.info(f"Found {len(self.video_files)} video files")

    def _select_default_video(self) -> None:
        """Open the configured default video, or the first one."""
        if not self.video_files:
            return
        default_name = get_setting("video.default_video", "")
        names = [Path(path).name for path in self.video_files]
        self.video_list.setCurrentRow(names.index(default_name) if default_name in names else 0)

    def _on_video_selection_changed(self, row: int):
        """Handle video selection change."""
        if 0 <= row < len(self.video_files):
            self.current_video_index = row
            self._load_selected_video()

    def _load_selected_video(self):
        """Have the worker open the selected video; _on_video_loaded continues."""
        if not self.video_files or self.current_video_index >= len(self.video_files):
            return

        video_path = self.video_files[self.current_video_index]
        self._stop_playback()
        self._generation += 1
        self.video_info = {}
        self.video_label.setText(f"Loading {Path(video_path).name}...")
        self._load_video_requested.emit(video_path, self._options(), load_default_matrix())

    def _on_video_loaded(self, info: Dict[str, Any]) -> None:
        """Show the first frame of a video the worker has opened."""
        selected = self.video_files[self.current_video_index] if self.video_files else None
        if info.get("path") != selected:
            return  # Another video was selected in the meantime

        if not info.get("loaded"):
            self.video_label.setText("Failed to load video")
            return

        self.video_info = info
        self._frame_interval_ms = 1000.0 / info["fps"] if info["fps"] > 0 else 40.0
        self.progress_bar.setMaximum(max(1, info["total_frames"] - 1))
        self.progress_bar.setValue(0)
        self._request_frame("current")
        self.video_changed.emit(info["path"])
        logger.info(f"Video loaded: {Path(info['path']).name}")

    def _prev_video(self):
        """Switch to previous video."""
        if self.video_files:
            # Wraps to the last video from the first
            self.video_list.setCurrentRow((self.current_video_index - 1) % len(self.video_files))

    def _next_video(self):
        """Switch to next video."""
        if self.video_files:
            # Wraps to the first video from the last
            self.video_list.setCurrentRow((self.current_video_index + 1) % len(self.video_files))

    # ------------------------------------------------------------------ playback

    def _toggle_play_pause(self):
        """Toggle video playback."""
        if not self.video_info:
            return
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        """Start video playback."""
        if not self.video_info:
            return
        self.is_playing = True
        self.play_pause_button.setText("⏸")
        self._request_frame("next")

    def _stop_playback(self):
        """Stop video playback."""
        self.playback_timer.stop()
        self.is_playing = False
        self.play_pause_button.setText("▶")

    def _on_seek(self, frame_idx: int):
        """Handle seek bar movement."""
        if not self.video_info:
            return
        self._generation += 1
        self._run_on_worker(lambda worker: worker.seek(frame_idx))
        self._request_display_update(immediate=True)

    def _reset_tracker(self):
        """Reset the object tracker and everything derived from earlier frames."""
        self._run_on_worker(lambda worker: worker.pipeline.reset())
        logger.info("Tracker reset")

    def hideEvent(self, event):
        """Pause when another tab is shown; nobody is watching the playback."""
        self._stop_playback()
        super().hideEvent(event)

    # ------------------------------------------------------------------ processing options

    def _on_inference_toggled(self, checked: bool):
        """Handle inference checkbox toggle."""
        self._request_display_update()

    def _on_tracking_toggled(self, checked: bool):
        """Handle tracking checkbox toggle."""
        if checked:
            # Tracking needs detections
            self.inference_checkbox.setChecked(True)
        self._request_display_update()

    def _on_player_id_toggled(self, checked: bool):
        """Handle player ID checkbox toggle."""
        if checked:
            # Player ID needs tracks
            self.tracking_checkbox.setChecked(True)
            self.inference_checkbox.setChecked(True)
        self._request_display_update()

    def _on_field_segmentation_toggled(self, checked: bool):
        """Handle field segmentation checkbox toggle."""
        self._request_display_update()

    def _on_homography_toggled(self, state: int):
        """Handle top-down view checkbox toggle."""
        self.homography_enabled = state == Qt.Checked
        if self.homography_enabled:
            self._request_display_update(immediate=True)
        else:
            self.homography_display_label.setText("Homography view disabled")

    def _on_ransac_toggled(self, state: int):
        """Handle RANSAC line fitting checkbox toggle."""
        # The drawing code reads this setting; override it in memory for this session
        ransac_config = get_config()
        for key in ("models", "segmentation", "contour", "ransac"):
            ransac_config = ransac_config.setdefault(key, {})
        ransac_config["enabled"] = state == Qt.Checked
        self._request_display_update(immediate=True)

    # ------------------------------------------------------------------ models

    def _on_player_model_changed(self, model_path: str):
        """Handle player detection model change."""
        if model_path:
            full_path = str(models_root() / model_path)
            self._run_on_worker(lambda worker: worker.set_player_model(full_path))
            self._request_display_update()

    def _on_disc_model_changed(self, model_path: str):
        """Handle disc detection model change."""
        if model_path:
            full_path = str(models_root() / model_path)
            self._run_on_worker(lambda worker: worker.set_disc_model(full_path))
            self._request_display_update()

    def _on_player_id_method_changed(self, index: int):
        """Handle jersey number reader selection change."""
        method = self.player_id_method_combo.itemData(index)
        if method:
            self._run_on_worker(lambda worker: worker.set_player_id_method(method))
            self._request_display_update()

    def _load_segmentation_models(self):
        """List the field segmentation models and use the selected one."""
        populate_segmentation_model_combo(
            self.segmentation_model_combo,
            default_model_path("segmentation"),
        )
        self._on_segmentation_model_changed(self.segmentation_model_combo.currentText())

    def _on_segmentation_model_changed(self, display_name: str):
        """Handle segmentation model selection change."""
        model_path = self.segmentation_model_combo.currentData()
        if model_path and Path(model_path).exists():
            self._run_on_worker(lambda worker: worker.set_field_model(model_path))
            self._request_display_update()

    # ------------------------------------------------------------------ shutdown

    def shutdown(self) -> None:
        """Stop playback and the worker thread. Safe to call more than once."""
        self._stop_playback()
        self.update_timer.stop()
        if self._worker_thread.isRunning():
            self._run_on_worker(lambda worker: worker.shutdown())
            # The worker finishes what it is doing first; loading the models at startup
            # can take a while, and the process must not exit while the thread still runs
            if not self._worker_thread.wait(WORKER_SHUTDOWN_TIMEOUT_MS):
                logger.warning("Analysis worker did not stop in time")

    def closeEvent(self, event):
        """Handle widget close event."""
        self.shutdown()
        super().closeEvent(event)
