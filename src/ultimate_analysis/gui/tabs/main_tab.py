"""
Main tab for the Ultimate Analysis GUI.
"""
import os
import time
import logging
from typing import List, Dict, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QCheckBox, 
    QPushButton, QTableWidget, QTableWidgetItem, QSlider, QShortcut, 
    QGroupBox, QFormLayout, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QIcon

from ultimate_analysis.gui.components.video_player import VideoPlayer
from ultimate_analysis.gui.components.runtime_dialog import RuntimesDialog
from ultimate_analysis.processing.inference import run_inference, set_detection_model, get_model_status
from ultimate_analysis.processing.tracking import run_tracking, reset_tracker, set_tracker_type
from ultimate_analysis.processing import field_segmentation
from ultimate_analysis.processing.player_id import run_player_id

logger = logging.getLogger("ultimate_analysis.gui.main_tab")


class MainTab(QWidget):
    """
    Main tab widget for video analysis and processing.
    """
    
    def __init__(self):
        super().__init__()
        self.video_folder = "data/processed/dev_data"
        self.player = VideoPlayer()
        self.video_files: List[str] = []
        self.current_video_index = 0
        self.is_paused = False
        self.tracks = []
        self.track_histories = {}
        self.runtimes_dialog = RuntimesDialog(self)
        
        self._init_ui()
        self._init_shortcuts()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout()
        
        # Left panel: video list and settings
        left_layout = QVBoxLayout()
        
        # Video list
        self.video_list = QListWidget()
        left_layout.addWidget(self.video_list, 1)
        
        # Settings panel
        config_panel = self._create_settings_panel()
        left_layout.addLayout(config_panel, 0)
        
        # Right panel: video display and controls
        right_layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel("Select a video to play")
        self.video_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.video_label, 8)
        
        # Progress bar
        self.progress_bar = QSlider(Qt.Horizontal)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setSingleStep(1)
        self.progress_bar.sliderMoved.connect(self._seek_video)
        right_layout.addWidget(self.progress_bar)
        
        # Controls
        controls_layout = self._create_controls_layout()
        right_layout.addLayout(controls_layout)
        
        # Add layouts to main layout
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 4)
        self.setLayout(layout)
        
        # Setup timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)
        
        # Connect signals
        self._connect_signals()
        
        # Load videos
        self._load_videos()

    def _create_settings_panel(self) -> QVBoxLayout:
        """Create the settings configuration panel."""
        config_panel = QVBoxLayout()
        
        # General Settings
        general_group = QGroupBox("General Settings")
        general_layout = QFormLayout()
        general_group.setLayout(general_layout)
        config_panel.addWidget(general_group)
        
        # Tracker Settings
        tracker_group = QGroupBox("Tracker Settings")
        tracker_layout = QFormLayout()
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["DeepSort", "Histogram"])
        tracker_layout.addRow("Tracker Type:", self.tracker_combo)
        tracker_group.setLayout(tracker_layout)
        config_panel.addWidget(tracker_group)
        
        # Detection Settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout()
        self.model_combo = QComboBox()
        detection_models = self._find_models("data/models/finetune", keyword="object_detection")
        self.model_combo.addItems(detection_models)
        detection_layout.addRow("Inference Model:", self.model_combo)
        detection_group.setLayout(detection_layout)
        config_panel.addWidget(detection_group)
        
        # Field Segmentation Settings
        field_group = QGroupBox("Field Segmentation Settings")
        field_layout = QFormLayout()
        self.field_model_combo = QComboBox()
        field_models = self._find_models("data/models/finetune", keyword="field_finder")
        self.field_model_combo.addItems(field_models)
        field_layout.addRow("Field Segmentation Model:", self.field_model_combo)
        field_group.setLayout(field_layout)
        config_panel.addWidget(field_group)
        
        # Player Identification Settings
        player_id_group = QGroupBox("Player Identification")
        player_id_layout = QFormLayout()
        self.player_id_method_combo = QComboBox()
        self.player_id_method_combo.addItems(["YOLO", "EasyOCR"])
        player_id_layout.addRow("Player ID Method:", self.player_id_method_combo)
        
        self.player_id_model_combo = QComboBox()
        player_id_models = self._find_models("data/models/finetune", keyword="digit_detector")
        self.player_id_model_combo.addItems(player_id_models)
        player_id_layout.addRow("Player ID YOLO Model:", self.player_id_model_combo)
        player_id_group.setLayout(player_id_layout)
        config_panel.addWidget(player_id_group)
        
        return config_panel

    def _create_controls_layout(self) -> QVBoxLayout:
        """Create the controls layout."""
        controls_layout = QVBoxLayout()
        
        # Checkboxes
        checkbox_row = QHBoxLayout()
        self.inference_checkbox = QCheckBox("Inference [I]")
        self.inference_checkbox.setChecked(True)
        self.tracking_checkbox = QCheckBox("Tracking [T]")
        self.tracking_checkbox.setChecked(True)
        self.player_id_checkbox = QCheckBox("Player ID [J]")
        self.field_checkbox = QCheckBox("Field [F]")
        
        checkbox_row.addWidget(self.inference_checkbox)
        checkbox_row.addWidget(self.tracking_checkbox)
        checkbox_row.addWidget(self.player_id_checkbox)
        checkbox_row.addWidget(self.field_checkbox)
        controls_layout.addLayout(checkbox_row)
        
        # Navigation buttons
        nav_row = QHBoxLayout()
        self.prev_button = QPushButton("←")
        self.prev_button.setMinimumHeight(36)
        self.prev_button.setToolTip("Prev [←]")
        self.prev_button.clicked.connect(self._prev_video)
        nav_row.addWidget(self.prev_button)
        
        self.play_pause_button = QPushButton("▶")
        self.play_pause_button.setMinimumHeight(36)
        self.play_pause_button.setToolTip("Play/Stop [Space]")
        self.play_pause_button.clicked.connect(self._toggle_play_pause)
        nav_row.addWidget(self.play_pause_button)
        
        self.next_button = QPushButton("→")
        self.next_button.setMinimumHeight(36)
        self.next_button.setToolTip("Next [→]")
        self.next_button.clicked.connect(self._next_video)
        nav_row.addWidget(self.next_button)
        
        controls_layout.addLayout(nav_row)
        
        # Utility buttons
        util_row = QHBoxLayout()
        self.reset_tracker_button = QPushButton("Reset Tracker [R]")
        self.reset_tracker_button.clicked.connect(self._reset_tracker)
        util_row.addWidget(self.reset_tracker_button)
        
        self.show_runtimes_button = QPushButton("Runtimes")
        self.show_runtimes_button.clicked.connect(self._open_runtimes_dialog)
        util_row.addWidget(self.show_runtimes_button)
        
        controls_layout.addLayout(util_row)
        
        return controls_layout

    def _init_shortcuts(self):
        """Initialize keyboard shortcuts."""
        shortcuts = [
            ("Space", self._toggle_play_pause),
            ("Left", self._prev_video),
            ("Right", self._next_video),
            ("I", lambda: self.inference_checkbox.setChecked(not self.inference_checkbox.isChecked())),
            ("T", lambda: self.tracking_checkbox.setChecked(not self.tracking_checkbox.isChecked())),
            ("J", lambda: self.player_id_checkbox.setChecked(not self.player_id_checkbox.isChecked())),
            ("F", lambda: self.field_checkbox.setChecked(not self.field_checkbox.isChecked())),
            ("R", self._reset_tracker),
        ]
        
        for key, func in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(func)

    def _connect_signals(self):
        """Connect widget signals to their handlers."""
        # Settings signals
        self.tracker_combo.currentTextChanged.connect(self._on_tracker_changed)
        self.model_combo.currentTextChanged.connect(self._on_detection_model_changed)
        self.field_model_combo.currentTextChanged.connect(self._on_field_model_changed)
        self.player_id_method_combo.currentTextChanged.connect(self._on_player_id_method_changed)
        self.player_id_model_combo.currentTextChanged.connect(self._on_player_id_model_changed)
        
        # Checkbox signals
        self.inference_checkbox.stateChanged.connect(self._handle_inference_checkbox)
        self.tracking_checkbox.stateChanged.connect(self._handle_tracking_checkbox)
        self.player_id_checkbox.stateChanged.connect(self._handle_player_id_checkbox)
        
        # Video list signal
        self.video_list.currentRowChanged.connect(self._load_selected_video)

    def _find_models(self, root_folder: str, keyword: Optional[str] = None) -> List[str]:
        """Find model files in the specified folder."""
        models = []
        for dirpath, dirnames, filenames in os.walk(root_folder):
            if "best.pt" in filenames:
                if keyword is None or keyword in dirpath:
                    models.append(os.path.relpath(os.path.join(dirpath, "best.pt"), root_folder))
        return models if models else ["None found"]

    def _load_videos(self):
        """Load available videos from the video folder."""
        if not os.path.exists(self.video_folder):
            logger.warning(f"Video folder not found: {self.video_folder}")
            return
            
        self.video_files = [
            f for f in os.listdir(self.video_folder) 
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
        
        self.video_list.clear()
        self.video_list.addItems(self.video_files)
        
        if self.video_files:
            self.video_list.setCurrentRow(0)

    def _load_selected_video(self, row: int):
        """Load the selected video."""
        if row < 0 or row >= len(self.video_files):
            return
            
        video_path = os.path.join(self.video_folder, self.video_files[row])
        if self.player.load_video(video_path):
            self.current_video_index = row
            self.progress_bar.setMaximum(self.player.get_frame_count() - 1)
            self._update_video_display()
        else:
            logger.error(f"Failed to load video: {video_path}")

    def _update_video_display(self):
        """Update the video display with the current frame."""
        frame = self.player.get_current_frame()
        if frame is not None:
            self._display_frame(frame)
            self.progress_bar.setValue(self.player.get_current_position())

    def _display_frame(self, frame):
        """Display a frame in the video label."""
        # Convert frame to QImage and display
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale image to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def _next_frame(self):
        """Process and display the next frame."""
        if self.is_paused:
            return
            
        frame = self.player.get_next_frame()
        if frame is None:
            self._toggle_play_pause()  # Stop playback
            return
            
        # Apply processing steps
        processed_frame = self._process_frame(frame)
        self._display_frame(processed_frame)
        
        # Update progress bar
        self.progress_bar.setValue(self.player.get_current_position())

    def _process_frame(self, frame):
        """Apply selected processing steps to the frame."""

        processed_frame = frame.copy()
        detections = None

        # Apply inference if enabled
        if self.inference_checkbox.isChecked():
            # Check if model is loaded before running inference
            try:
                status = get_model_status()
                if not status['model_loaded']:
                    logger.error(f"Model not loaded when inference requested. Status: {status}")
                    # Try to load the model
                    from ...config.settings import get_settings
                    settings = get_settings()
                    logger.info(f"Attempting to load model during playback: {settings.models.detection_model}")
                    set_detection_model(settings.models.detection_model)
                    logger.info("Model loaded successfully during playback")
            except Exception as e:
                logger.error(f"Failed to check/load model during playback: {e}")

            start_time = time.time()
            # Apply inference processing
            detections = run_inference(processed_frame)
            runtime = (time.time() - start_time) * 1000
            self.runtimes_dialog.log_runtime("Inference", runtime)

        # Apply tracking if enabled
        if self.tracking_checkbox.isChecked():
            start_time = time.time()
            # Apply tracking processing
            if detections is None:
                # If inference was not run, run it now
                detections = run_inference(processed_frame)
            tracks = run_tracking(processed_frame, detections)
            
            # Draw tracks on the frame
            from ultimate_analysis.gui.utils.visualization import draw_tracks
            from ultimate_analysis.processing.tracking import tracker_type
            processed_frame = draw_tracks(processed_frame, tracks, tracker_type)
            
            runtime = (time.time() - start_time) * 1000
            self.runtimes_dialog.log_runtime("Tracking", runtime)
        
        # Apply player ID if enabled
        if self.player_id_checkbox.isChecked():
            start_time = time.time()
            # Apply player ID processing
            digit_str, digits = run_player_id(processed_frame)
            from ultimate_analysis.gui.utils.visualization import draw_player_id_results
            processed_frame = draw_player_id_results(processed_frame, digits, digit_str)
            runtime = (time.time() - start_time) * 1000
            self.runtimes_dialog.log_runtime("Player ID", runtime)
        
        # Apply field segmentation if enabled
        if self.field_checkbox.isChecked():
            start_time = time.time()
            # Ensure field segmentation model is loaded
            try:
                if getattr(field_segmentation, 'field_model', None) is None:
                    from ...config.settings import get_settings
                    settings = get_settings()
                    logger.info(f"Loading field segmentation model: {settings.models.segmentation_model}")
                    field_segmentation.set_field_model(settings.models.segmentation_model)
                    logger.info("Field segmentation model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to check/load field segmentation model: {e}")
            # Apply field segmentation processing
            masks = field_segmentation.run_field_segmentation(processed_frame)
            from ultimate_analysis.gui.utils.visualization import overlay_segmentation_mask
            processed_frame = overlay_segmentation_mask(processed_frame, masks, color=(0, 255, 0), alpha=0.4)
            runtime = (time.time() - start_time) * 1000
            self.runtimes_dialog.log_runtime("Field Segmentation", runtime)
        
        return processed_frame

    # Event handlers
    def _toggle_play_pause(self):
        """Toggle play/pause state."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.timer.stop()
            self.play_pause_button.setText("▶")
        else:
            fps = self.player.get_fps()
            self.timer.start(int(1000 / fps))
            self.play_pause_button.setText("⏸")

    def _prev_video(self):
        """Go to previous video."""
        if self.current_video_index > 0:
            self.video_list.setCurrentRow(self.current_video_index - 1)

    def _next_video(self):
        """Go to next video."""
        if self.current_video_index < len(self.video_files) - 1:
            self.video_list.setCurrentRow(self.current_video_index + 1)

    def _seek_video(self, position: int):
        """Seek to a specific position in the video."""
        self.player.set_position(position)
        self._update_video_display()

    def _reset_tracker(self):
        """Reset the tracker."""
        reset_tracker()
        self.tracks = []
        self.track_histories = {}
        
        # Clear any cached track histories in the visualization module
        from ultimate_analysis.gui.utils.visualization import clear_track_history
        clear_track_history()
        
        logger.info("Tracker reset and track histories cleared")

    def _open_runtimes_dialog(self):
        """Open the runtimes dialog."""
        self.runtimes_dialog.show()

    # Settings handlers
    def _on_tracker_changed(self, tracker_type: str):
        """Handle tracker type change."""
        # Convert GUI names to internal tracker names
        if tracker_type == "DeepSort":
            internal_type = "deepsort"
        elif tracker_type == "Histogram":
            internal_type = "histogram"
        else:
            logger.warning(f"Unknown tracker type: {tracker_type}")
            return
        
        logger.info(f"Tracker changed to: {tracker_type} (internal: {internal_type})")
        
        # Set the new tracker type (this creates a new tracker instance)
        set_tracker_type(internal_type)
        
        # Clear any cached track histories in the visualization module
        from ultimate_analysis.gui.utils.visualization import clear_track_history
        clear_track_history()
        
        logger.info("Track histories cleared for new tracker")

    def _on_detection_model_changed(self, model_path: str):
        """Handle detection model change."""
        if model_path != "None found":
            set_detection_model(model_path)
            logger.info(f"Detection model changed to: {model_path}")

    def _on_field_model_changed(self, model_path: str):
        """Handle field segmentation model change."""
        if model_path != "None found":
            field_segmentation.set_field_model(model_path)
            logger.info(f"Field segmentation model changed to: {model_path}")

    def _on_player_id_method_changed(self, method: str):
        """Handle player ID method change."""
        logger.info(f"Player ID method changed to: {method}")

    def _on_player_id_model_changed(self, model_path: str):
        """Handle player ID model change."""
        if model_path != "None found":
            logger.info(f"Player ID model changed to: {model_path}")

    # Checkbox handlers
    def _handle_inference_checkbox(self, state: int):
        """Handle inference checkbox state change."""
        logger.info(f"Inference {'enabled' if state == Qt.Checked else 'disabled'}")

    def _handle_tracking_checkbox(self, state: int):
        """Handle tracking checkbox state change."""
        logger.info(f"Tracking {'enabled' if state == Qt.Checked else 'disabled'}")

    def _handle_player_id_checkbox(self, state: int):
        """Handle player ID checkbox state change."""
        logger.info(f"Player ID {'enabled' if state == Qt.Checked else 'disabled'}")

    def closeEvent(self, event):
        """Handle tab close event."""
        self.timer.stop()
        self.player.release()
        super().closeEvent(event)
