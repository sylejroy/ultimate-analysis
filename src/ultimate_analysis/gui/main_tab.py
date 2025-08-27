"""Main tab for Ultimate Analysis GUI.

This module contains the main video analysis interface with video list,
playback controls, and processing options.
"""

import os
import cv2
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, 
    QCheckBox, QPushButton, QSlider, QListWidgetItem, QGroupBox,
    QFormLayout, QComboBox, QShortcut, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QFont, QColor

from .video_player import VideoPlayer
from .visualization import draw_detections, draw_tracks, draw_tracks_with_player_ids
from .performance_widget import PerformanceWidget
from ..processing import (
    run_inference, run_tracking, run_player_id_on_tracks, run_field_segmentation,
    set_detection_model, set_field_model, set_tracker_type, 
    reset_tracker, get_track_histories
)
from ..processing.jersey_tracker import get_jersey_tracker
from ..processing.field_segmentation import visualize_segmentation
from ..config.settings import get_setting
from ..constants import SHORTCUTS, DEFAULT_PATHS, SUPPORTED_VIDEO_EXTENSIONS


class VideoListWidget(QListWidget):
    """Enhanced video list widget with duration information."""
    
    def __init__(self):
        super().__init__()
        
        self.setStyleSheet("""
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #2a2a2a;
            }
        """)


class MainTab(QWidget):
    """Main video analysis tab with video player and processing controls."""
    
    # Signals
    video_changed = pyqtSignal(str)  # Emitted when video selection changes
    
    def __init__(self):
        super().__init__()
        
        # Video player and state
        self.video_player = VideoPlayer()
        self.video_files: List[str] = []
        self.current_video_index: int = 0
        self.is_playing: bool = False
        
        # Processing state
        self.current_detections: List[Dict] = []
        self.current_tracks: List[Any] = []
        self.current_field_results: List[Any] = []
        self.current_player_ids: Dict[int, Tuple[str, Any]] = {}
        
        # FPS tracking for processed frames
        self.frame_times: List[float] = []
        self.max_frame_samples = 30  # Rolling average over 30 frames
        self.current_fps = 0.0
        
        # Playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._on_timer_tick)
        
        # Initialize UI
        self._init_ui()
        self._init_shortcuts()
        self._load_videos()
        
        # Load default video (portland_vs_san_francisco_2024_snippet_4_40912.mp4)
        if self.video_files:
            default_video_name = "portland_vs_san_francisco_2024_snippet_4_40912.mp4"
            default_index = 0  # Fallback to first video
            
            # Look for the specific default video
            for i, video_path in enumerate(self.video_files):
                if Path(video_path).name == default_video_name:
                    default_index = i
                    print(f"[MAIN_TAB] Loading default video: {default_video_name}")
                    break
            else:
                print(f"[MAIN_TAB] Default video '{default_video_name}' not found, loading first available video")
            
            self.video_list.setCurrentRow(default_index)
            self._load_selected_video()
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Video list and controls
        left_panel = self._create_left_panel()
        left_panel.setMinimumWidth(300)  # Minimum width to prevent collapse
        left_panel.setMaximumWidth(500)  # Maximum width to prevent taking too much space
        splitter.addWidget(left_panel)
        
        # Right panel: Video display
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Simple initial sizing - left takes ~20%, right takes ~80%
        splitter.setSizes([350, 1400])  # Initial sizes in pixels
        
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
        refresh_button.clicked.connect(self._load_videos)
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
        self.inference_checkbox.setToolTip(f"Enable/disable object detection [{SHORTCUTS['TOGGLE_INFERENCE']}]")
        self.inference_checkbox.stateChanged.connect(self._on_inference_toggled)
        
        self.tracking_checkbox = QCheckBox("Object Tracking")  
        self.tracking_checkbox.setToolTip(f"Enable/disable object tracking [{SHORTCUTS['TOGGLE_TRACKING']}]")
        self.tracking_checkbox.stateChanged.connect(self._on_tracking_toggled)
        
        self.player_id_checkbox = QCheckBox("Player Identification")
        self.player_id_checkbox.setToolTip(f"Enable/disable player ID based on jersey numbers [{SHORTCUTS['TOGGLE_PLAYER_ID']}]")
        self.player_id_checkbox.stateChanged.connect(self._on_player_id_toggled)
        
        self.field_segmentation_checkbox = QCheckBox("Field Segmentation")
        self.field_segmentation_checkbox.setToolTip(f"Enable/disable field boundary detection [{SHORTCUTS['TOGGLE_FIELD_SEGMENTATION']}]")
        self.field_segmentation_checkbox.stateChanged.connect(self._on_field_segmentation_toggled)
        
        processing_layout.addWidget(self.inference_checkbox)
        processing_layout.addWidget(self.tracking_checkbox)
        processing_layout.addWidget(self.player_id_checkbox)
        processing_layout.addWidget(self.field_segmentation_checkbox)
        
        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)
        
        # Model selection section
        models_group = QGroupBox("Model Settings")
        models_layout = QFormLayout()
        
        # Detection model dropdown
        self.detection_model_combo = QComboBox()
        self._populate_model_combo(self.detection_model_combo, "detection")
        self.detection_model_combo.currentTextChanged.connect(self._on_detection_model_changed)
        models_layout.addRow("Detection Model:", self.detection_model_combo)
        
        # Tracking method dropdown
        self.tracking_method_combo = QComboBox()
        self.tracking_method_combo.addItems(["DeepSORT", "Histogram"])
        self.tracking_method_combo.currentTextChanged.connect(self._on_tracking_method_changed)
        models_layout.addRow("Tracking Method:", self.tracking_method_combo)
        
        # Player ID method dropdown
        self.player_id_method_combo = QComboBox()
        self.player_id_method_combo.addItems(["EasyOCR"])
        self.player_id_method_combo.currentTextChanged.connect(self._on_player_id_method_changed)
        models_layout.addRow("Player ID Method:", self.player_id_method_combo)
        
        # Field segmentation model dropdown
        self.field_model_combo = QComboBox()
        self._populate_model_combo(self.field_model_combo, "segmentation")
        self.field_model_combo.currentTextChanged.connect(self._on_field_model_changed)
        models_layout.addRow("Field Model:", self.field_model_combo)
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # Performance metrics section
        self.performance_widget = PerformanceWidget()
        layout.addWidget(self.performance_widget)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with video display and controls."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Video display area
        self.video_label = QLabel("No video selected")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedHeight(1080)  # Much bigger for main tab - increased from 400 to 1080
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #1a1a1a;
                color: #999;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.video_label, 1)  # Takes most space
        
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
    
    def _init_shortcuts(self):
        """Initialize keyboard shortcuts."""
        # Play/Pause
        QShortcut(QKeySequence(SHORTCUTS['PLAY_PAUSE']), self, self._toggle_play_pause)
        
        # Previous/Next video
        QShortcut(QKeySequence(SHORTCUTS['PREV_VIDEO']), self, self._prev_video)
        QShortcut(QKeySequence(SHORTCUTS['NEXT_VIDEO']), self, self._next_video)
        
        # Reset tracker
        QShortcut(QKeySequence(SHORTCUTS['RESET_TRACKER']), self, self._reset_tracker)
        
        # Toggle processing features
        QShortcut(QKeySequence(SHORTCUTS['TOGGLE_INFERENCE']), self, 
                 lambda: self.inference_checkbox.toggle())
        QShortcut(QKeySequence(SHORTCUTS['TOGGLE_TRACKING']), self, 
                 lambda: self.tracking_checkbox.toggle())
        QShortcut(QKeySequence(SHORTCUTS['TOGGLE_PLAYER_ID']), self, 
                 lambda: self.player_id_checkbox.toggle())
        QShortcut(QKeySequence(SHORTCUTS['TOGGLE_FIELD_SEGMENTATION']), self, 
                 lambda: self.field_segmentation_checkbox.toggle())
    
    def _load_videos(self):
        """Load and display available video files."""
        print("[MAIN_TAB] Loading video files...")
        
        self.video_files.clear()
        self.video_list.clear()
        
        # Search paths for videos
        search_paths = [
            Path(DEFAULT_PATHS['DEV_DATA']),
            Path(DEFAULT_PATHS['RAW_VIDEOS'])
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                print(f"[MAIN_TAB] Search path does not exist: {search_path}")
                continue
            
            print(f"[MAIN_TAB] Searching for videos in: {search_path}")
            
            # Find video files
            for file_path in search_path.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    self.video_files.append(str(file_path))
        
        # Sort videos by name
        self.video_files.sort()
        
        # Populate list with video info
        for video_path in self.video_files:
            duration = self._get_video_duration(video_path)
            filename = Path(video_path).name
            
            # Create list item with filename and duration
            item_text = f"{filename} ({duration})"
            item = QListWidgetItem(item_text)
            item.setToolTip(video_path)
            self.video_list.addItem(item)
        
        print(f"[MAIN_TAB] Found {len(self.video_files)} video files")
    
    def _get_video_duration(self, video_path: str) -> str:
        """Get video duration as formatted string.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Duration string in format "MM:SS" or "Unknown"
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                
                if fps > 0:
                    duration_seconds = frame_count / fps
                    minutes = int(duration_seconds // 60)
                    seconds = int(duration_seconds % 60)
                    return f"{minutes:02d}:{seconds:02d}"
            
        except Exception as e:
            print(f"[MAIN_TAB] Error getting duration for {video_path}: {e}")
        
        return "Unknown"
    
    def _populate_model_combo(self, combo: QComboBox, model_type: str):
        """Populate a combo box with available models.
        
        Args:
            combo: QComboBox to populate
            model_type: Type of models to find ("detection", "segmentation", etc.)
        """
        combo.clear()
        
        # Look for models in the models directory
        models_path = Path(get_setting("models.base_path", DEFAULT_PATHS['MODELS']))
        
        if not models_path.exists():
            print(f"[MAIN_TAB] Models directory not found: {models_path}")
            return
        
        # Search for model files
        model_files = []
        for model_dir in models_path.rglob("*"):
            if model_dir.is_file() and model_dir.suffix == ".pt":
                # Skip last.pt files - we only want best.pt from finetuned models
                if model_dir.name == "last.pt":
                    continue
                    
                # Check if this model type is in the path
                if model_type in str(model_dir).lower():
                    relative_path = model_dir.relative_to(models_path)
                    model_files.append(str(relative_path))
        
        # Add pretrained models
        pretrained_path = models_path / "pretrained"
        if pretrained_path.exists():
            for model_file in pretrained_path.glob("*.pt"):
                if model_type in model_file.name.lower() or model_type == "detection":
                    relative_path = model_file.relative_to(models_path)
                    model_files.append(str(relative_path))
        
        # Sort and add to combo
        model_files.sort()
        combo.addItems(model_files)
        
        print(f"[MAIN_TAB] Found {len(model_files)} {model_type} models")
    
    def _on_video_selection_changed(self, row: int):
        """Handle video selection change."""
        if 0 <= row < len(self.video_files):
            self.current_video_index = row
            self._load_selected_video()
    
    def _load_selected_video(self):
        """Load the currently selected video."""
        if not self.video_files or self.current_video_index >= len(self.video_files):
            return
        
        video_path = self.video_files[self.current_video_index]
        print(f"[MAIN_TAB] Loading video: {video_path}")
        
        # Stop playback
        self._stop_playback()
        
        # Reset FPS tracking for new video
        self.frame_times.clear()
        self.current_fps = 0.0
        
        # Load video
        if self.video_player.load_video(video_path):
            # Update UI
            filename = Path(video_path).name
            self.video_label.setText(f"Loaded: {filename}")
            
            # Set progress bar range
            video_info = self.video_player.get_video_info()
            self.progress_bar.setMaximum(max(1, video_info['total_frames'] - 1))
            self.progress_bar.setValue(0)
            
            # Display first frame
            first_frame = self.video_player.get_current_frame()
            if first_frame is not None:
                self._display_frame(first_frame)
            
            # Emit signal
            self.video_changed.emit(video_path)
            
            print(f"[MAIN_TAB] Video loaded successfully: {filename}")
        else:
            self.video_label.setText("Failed to load video")
    
    def _display_frame(self, frame):
        """Display a frame in the video label.
        
        Args:
            frame: OpenCV frame (numpy array) to display
        """
        if frame is None:
            return
        
        # Apply processing if enabled
        processed_frame = self._process_frame(frame.copy())
        
        # Convert to Qt format and display
        height, width, channel = processed_frame.shape
        bytes_per_line = 3 * width
        
        q_image = QImage(processed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        # Use fixed dimensions instead of current label size to prevent growth
        label_width = self.video_label.width()
        label_height = 1200  # Use the fixed height we set
        scaled_pixmap = pixmap.scaled(
            label_width - 4, label_height - 4,  # Account for 2px border on each side
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def _process_frame(self, frame):
        """Apply enabled processing to a frame.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Processed frame with visualizations
        """
        # Start total runtime timer
        total_start_time = time.time()
        
        # Reset detection/tracking results
        self.current_detections = []
        self.current_tracks = []
        self.current_field_results = []
        self.current_player_ids = {}
        
        # Run inference if enabled
        if self.inference_checkbox.isChecked():
            print("[MAIN_TAB] Running inference...")
            start_time = time.time()
            self.current_detections = run_inference(frame)
            duration_ms = (time.time() - start_time) * 1000
            self.performance_widget.add_processing_measurement("Inference", duration_ms)
        
        # Run tracking if enabled
        if self.tracking_checkbox.isChecked() and self.current_detections:
            print("[MAIN_TAB] Running tracking...")
            start_time = time.time()
            self.current_tracks = run_tracking(frame, self.current_detections)
            duration_ms = (time.time() - start_time) * 1000
            self.performance_widget.add_processing_measurement("Tracking", duration_ms)
        
        # Run field segmentation if enabled
        if self.field_segmentation_checkbox.isChecked():
            print("[MAIN_TAB] Running field segmentation...")
            start_time = time.time()
            self.current_field_results = run_field_segmentation(frame)
            duration_ms = (time.time() - start_time) * 1000
            self.performance_widget.add_processing_measurement("Field Segmentation", duration_ms)
        
        # Run player ID if enabled (requires tracking to be active)
        if self.player_id_checkbox.isChecked() and self.current_tracks:
            print(f"[MAIN_TAB] Running player identification on {len(self.current_tracks)} tracks...")
            start_time = time.time()
            self.current_player_ids, player_id_timing = run_player_id_on_tracks(frame, self.current_tracks)
            duration_ms = (time.time() - start_time) * 1000
            
            # Debug timing values and results
            print(f"[MAIN_TAB] Player ID results: {len(self.current_player_ids)} tracks processed")
            for track_id, (jersey_number, details) in self.current_player_ids.items():
                confidence = details.get('confidence', 0.0) if details else 0.0
                print(f"[MAIN_TAB]   Track {track_id}: #{jersey_number} (conf: {confidence:.3f})")
            print(f"[MAIN_TAB] Raw timing: {player_id_timing}")
            print(f"[MAIN_TAB] Total duration: {duration_ms:.1f}ms")
            
            # Add detailed timing measurements (only if there are actual measurements)
            if player_id_timing['preprocessing_ms'] > 0 or player_id_timing['ocr_ms'] > 0:
                self.performance_widget.add_processing_measurement("Player ID - Preprocessing", player_id_timing['preprocessing_ms'])
                self.performance_widget.add_processing_measurement("Player ID - EasyOCR", player_id_timing['ocr_ms'])
            
            print(f"[MAIN_TAB] Identified {len(self.current_player_ids)} players")
            print(f"[MAIN_TAB] Player ID timing - Preprocessing: {player_id_timing['preprocessing_ms']:.1f}ms, OCR: {player_id_timing['ocr_ms']:.1f}ms")
        else:
            # Clear player IDs when not running
            self.current_player_ids = {}
        
        # Apply visualizations (with timing)
        viz_start_time = time.time()
        frame = self._apply_visualizations(frame)
        viz_duration_ms = (time.time() - viz_start_time) * 1000
        self.performance_widget.add_processing_measurement("Visualization", viz_duration_ms)
        
        # Record total runtime
        total_duration_ms = (time.time() - total_start_time) * 1000
        self.performance_widget.add_processing_measurement("Total Runtime", total_duration_ms)
        
        # Update FPS calculation
        self._update_fps(total_duration_ms)
        
        return frame
    
    def _apply_visualizations(self, frame):
        """Apply visualization overlays to frame.
        
        Args:
            frame: Frame to add visualizations to
            
        Returns:
            Frame with visualizations applied
        """
        # Apply field segmentation overlay first (as background)
        if self.current_field_results and self.field_segmentation_checkbox.isChecked():
            frame = visualize_segmentation(frame, self.current_field_results, alpha=0.3)
        
        # Show detections only if tracking is NOT enabled (to avoid visual clutter)
        if self.current_detections and not self.tracking_checkbox.isChecked():
            frame = draw_detections(frame, self.current_detections)
        
        # Show tracking visualization if tracking is enabled
        if self.current_tracks and self.tracking_checkbox.isChecked():
            # Get track histories for trajectory visualization
            track_histories = get_track_histories()
            
            # Use player ID visualization if player ID is enabled (even if no IDs detected yet)
            if self.player_id_checkbox.isChecked():
                frame = draw_tracks_with_player_ids(frame, self.current_tracks, track_histories, self.current_player_ids)
            else:
                frame = draw_tracks(frame, self.current_tracks, track_histories)
        
        # Add FPS overlay to top right
        self._draw_fps_overlay(frame)
        
        # Add jersey tracking table overlay if player ID is enabled
        if self.player_id_checkbox.isChecked():
            self._draw_jersey_table_overlay(frame)
        
        return frame
    
    def _update_fps(self, frame_time_ms: float) -> None:
        """Update FPS calculation with latest frame processing time.
        
        Args:
            frame_time_ms: Processing time for the current frame in milliseconds
        """
        # Add current frame time
        self.frame_times.append(frame_time_ms)
        
        # Keep only recent samples for rolling average
        if len(self.frame_times) > self.max_frame_samples:
            self.frame_times.pop(0)
        
        # Calculate average frame time and convert to FPS
        if len(self.frame_times) > 0:
            avg_frame_time_ms = sum(self.frame_times) / len(self.frame_times)
            if avg_frame_time_ms > 0:
                self.current_fps = 1000.0 / avg_frame_time_ms
            else:
                self.current_fps = 0.0
    
    def _draw_fps_overlay(self, frame) -> None:
        """Draw FPS overlay on the top right of the frame.
        
        Args:
            frame: OpenCV frame to draw on (modified in place)
        """
        if self.current_fps <= 0:
            return
            
        # Format FPS text
        fps_text = f"Processing: {self.current_fps:.1f} FPS"
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Set text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)  # Green color
        thickness = 2
        
        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
        
        # Position in top right with some padding, moved down slightly
        x = width - text_width - 15
        y = text_height + 45  # Increased from 15 to 45 to lower the position
        
        # Draw background rectangle for better visibility
        cv2.rectangle(frame, 
                     (x - 5, y - text_height - 5), 
                     (x + text_width + 5, y + baseline + 5), 
                     (0, 0, 0), -1)  # Black background
        
        # Draw the FPS text
        cv2.putText(frame, fps_text, (x, y), font, font_scale, color, thickness)
    
    def _toggle_play_pause(self):
        """Toggle video playback."""
        if not self.video_player.is_loaded():
            return
        
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()
    
    def _start_playback(self):
        """Start video playback."""
        if not self.video_player.is_loaded():
            return
        
        # Calculate timer interval based on FPS
        video_info = self.video_player.get_video_info()
        fps = video_info['fps']
        interval_ms = max(1, int(1000 / fps))
        
        self.playback_timer.start(interval_ms)
        self.is_playing = True
        self.play_pause_button.setText("⏸")
        
        print(f"[MAIN_TAB] Started playback at {fps} FPS (interval: {interval_ms}ms)")
    
    def _stop_playback(self):
        """Stop video playback."""
        self.playback_timer.stop()
        self.is_playing = False
        self.play_pause_button.setText("▶")
        
        print("[MAIN_TAB] Stopped playback")
    
    def _on_timer_tick(self):
        """Handle playback timer tick."""
        if not self.video_player.is_loaded():
            self._stop_playback()
            return
        
        # Get next frame
        frame = self.video_player.get_next_frame()
        
        if frame is not None:
            # Display frame
            self._display_frame(frame)
            
            # Update progress bar
            video_info = self.video_player.get_video_info()
            self.progress_bar.setValue(video_info['current_frame'])
        else:
            # End of video reached
            print("[MAIN_TAB] End of video reached")
            self._stop_playback()
    
    def _on_seek(self, frame_idx: int):
        """Handle seek bar movement."""
        if self.video_player.is_loaded():
            self.video_player.seek_to_frame(frame_idx)
            
            # Display current frame
            frame = self.video_player.get_current_frame()
            if frame is not None:
                self._display_frame(frame)
    
    def _prev_video(self):
        """Switch to previous video."""
        if not self.video_files:
            return
        
        # Loop to last video if on first
        new_index = (self.current_video_index - 1) % len(self.video_files)
        self.video_list.setCurrentRow(new_index)
    
    def _next_video(self):
        """Switch to next video."""
        if not self.video_files:
            return
        
        # Loop to first video if on last
        new_index = (self.current_video_index + 1) % len(self.video_files)
        self.video_list.setCurrentRow(new_index)
    
    def _reset_tracker(self):
        """Reset the object tracker."""
        reset_tracker()
        print("[MAIN_TAB] Tracker reset")
    
    # Processing control event handlers
    def _on_inference_toggled(self, checked: bool):
        """Handle inference checkbox toggle."""
        print(f"[MAIN_TAB] Inference {'enabled' if checked else 'disabled'}")
    
    def _on_tracking_toggled(self, checked: bool):
        """Handle tracking checkbox toggle."""
        print(f"[MAIN_TAB] Tracking {'enabled' if checked else 'disabled'}")
        if checked:
            # Enable inference if tracking is enabled
            self.inference_checkbox.setChecked(True)
    
    def _on_player_id_toggled(self, checked: bool):
        """Handle player ID checkbox toggle."""
        print(f"[MAIN_TAB] Player ID {'enabled' if checked else 'disabled'}")
        if checked:
            # Enable tracking and inference if player ID is enabled
            self.tracking_checkbox.setChecked(True)
            self.inference_checkbox.setChecked(True)
            
            print("[MAIN_TAB] Player ID using EasyOCR for jersey number recognition")
    
    def _on_field_segmentation_toggled(self, checked: bool):
        """Handle field segmentation checkbox toggle."""
        print(f"[MAIN_TAB] Field segmentation {'enabled' if checked else 'disabled'}")
        
        if checked:
            # Ensure field segmentation model is loaded with the default path
            default_model_path = Path(get_setting("models.base_path", DEFAULT_PATHS['MODELS'])) / "segmentation/field_finder_yolo11m-seg/segmentation_finetune/weights/best.pt"
            if default_model_path.exists():
                set_field_model(str(default_model_path))
                print(f"[MAIN_TAB] Field segmentation model set to: {default_model_path}")
            else:
                print(f"[MAIN_TAB] Default field segmentation model not found at: {default_model_path}")
                print("[MAIN_TAB] Will use fallback models or mock results")
    
    # Model selection event handlers
    def _on_detection_model_changed(self, model_path: str):
        """Handle detection model change."""
        if model_path:
            full_path = Path(get_setting("models.base_path", DEFAULT_PATHS['MODELS'])) / model_path
            set_detection_model(str(full_path))
            print(f"[MAIN_TAB] Detection model changed to: {model_path}")
    
    def _on_tracking_method_changed(self, method: str):
        """Handle tracking method change."""
        if method:
            set_tracker_type(method.lower())
            print(f"[MAIN_TAB] Tracking method changed to: {method}")
    
    def _on_player_id_method_changed(self, method: str):
        """Handle player ID method change. Only EasyOCR is supported."""
        print(f"[MAIN_TAB] Player ID method: {method} (EasyOCR only)")
    
    def _on_field_model_changed(self, model_path: str):
        """Handle field model change."""
        if model_path:
            full_path = Path(get_setting("models.base_path", DEFAULT_PATHS['MODELS'])) / model_path
            set_field_model(str(full_path))
            print(f"[MAIN_TAB] Field model changed to: {model_path}")
    
    def _draw_jersey_table_overlay(self, frame):
        """Draw jersey tracking information as an overlay on the frame."""
        try:
            tracker = get_jersey_tracker()
            
            # Get best and second-best jersey numbers for each track
            tracked_data = []
            for track_id in tracker._track_probabilities.keys():
                top_probs = tracker.get_top_probabilities(track_id, top_k=2)
                if top_probs:
                    best_jersey, best_prob = top_probs[0][0], top_probs[0][1]
                    second_jersey, second_prob = None, 0.0
                    if len(top_probs) > 1:
                        second_jersey, second_prob = top_probs[1][0], top_probs[1][1]
                    
                    tracked_data.append({
                        'track_id': track_id,
                        'best_jersey': best_jersey,
                        'best_prob': best_prob,
                        'second_jersey': second_jersey,
                        'second_prob': second_prob
                    })
            
            if not tracked_data:
                return
            
            # Sort by track ID
            tracked_data.sort(key=lambda x: x['track_id'])
            
            # Overlay position (top-left corner with margin, lowered slightly)
            start_x = 20
            start_y = 80  # Lowered from 30 to 80
            line_height = 30  # Increased to accommodate two lines per track
            
            # Draw semi-transparent background
            table_height = len(tracked_data) * line_height + 40
            table_width = 250  # Increased width for second jersey
            
            # Create overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (start_x - 10, start_y - 20), 
                         (start_x + table_width, start_y + table_height - 20), 
                         (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
            
            # Draw header
            cv2.putText(frame, "Jersey Tracking", (start_x, start_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw table entries
            y_offset = start_y + 25
            for data in tracked_data:
                # Determine color based on best confidence
                if data['best_prob'] >= 0.7:
                    color = (0, 255, 0)  # Green
                elif data['best_prob'] >= 0.4:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw best jersey number (primary line)
                best_text = f"Track {data['track_id']}: #{data['best_jersey']} ({data['best_prob']:.2f})"
                cv2.putText(frame, best_text, (start_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw second jersey number (secondary line) if available
                if data['second_jersey'] and data['second_prob'] > 0.1:  # Only show if reasonable confidence
                    second_color = (128, 128, 128)  # Gray for secondary
                    second_text = f"   Alt: #{data['second_jersey']} ({data['second_prob']:.2f})"
                    cv2.putText(frame, second_text, (start_x, y_offset + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, second_color, 1)
                
                y_offset += line_height
                
        except Exception as e:
            print(f"[MAIN_TAB] Error drawing jersey overlay: {e}")

    def closeEvent(self, event):
        """Handle widget close event."""
        self._stop_playback()
        self.video_player.close_video()
        super().closeEvent(event)
