"""Main tab for Ultimate Analysis GUI.

This module contains the main video analysis interface with video list,
playback controls, and processing options.
"""

import os
import cv2
import time
import hashlib
import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, 
    QCheckBox, QPushButton, QSlider, QListWidgetItem, QGroupBox,
    QFormLayout, QComboBox, QShortcut, QSplitter, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QKeySequence

from .video_player import VideoPlayer
from .homography_tab import ZoomableImageLabel
from .visualization import draw_detections, draw_tracks, draw_tracks_with_player_ids, draw_unified_field_mask, create_unified_field_mask, draw_all_field_lines, draw_field_segmentation
from .ransac_line_visualization import draw_ransac_field_lines
from .performance_widget import PerformanceWidget
from ..processing import (
    run_inference, run_tracking, run_player_id_on_tracks, run_field_segmentation,
    set_detection_model, set_field_model, set_tracker_type, 
    reset_tracker, get_track_histories
)
from ..processing.async_processor import AsyncVideoProcessor, ProcessingResult
from ..processing.line_extraction import extract_raw_lines_from_segmentation
from ..processing.jersey_tracker import get_jersey_tracker
from ..config.settings import get_setting
from ..constants import SHORTCUTS, DEFAULT_PATHS, SUPPORTED_VIDEO_EXTENSIONS, FALLBACK_DEFAULTS
from ..utils.video_utils import get_video_duration
from ..utils.segmentation_utils import (
    apply_segmentation_to_warped_frame,
    load_segmentation_models, populate_segmentation_model_combo
)


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
        
        # Field segmentation state
        self.show_segmentation = True
        self.current_segmentation_results = None
        self.segmentation_model_combo: Optional[QComboBox] = None
        self.show_segmentation_checkbox: Optional[QCheckBox] = None
        self.ransac_checkbox: Optional[QCheckBox] = None
        self.available_segmentation_models: List[str] = []
        self.ransac_lines: List[Tuple[np.ndarray, np.ndarray]] = []  # Store RANSAC-calculated field lines
        self.ransac_confidences: List[float] = []  # Store RANSAC line confidences
        self.all_lines_for_display: Dict[str, Tuple[np.ndarray, float, bool]] = {}  # Store all lines for display
        
        # Homography state
        self.homography_enabled = True  # Enable by default
        self.homography_matrix: Optional[np.ndarray] = None
        self.homography_warped_frame: Optional[np.ndarray] = None
        self.homography_display_label: Optional[QLabel] = None
        
        # Try to load homography matrix from file
        loaded_matrix = self._load_homography_params_from_file()
        if loaded_matrix is not None:
            self.homography_matrix = loaded_matrix
            print("[MAIN_TAB] Using homography matrix from file")
        else:
            print("[MAIN_TAB] Using default homography parameters (no file found)")
        
        # Frame-based caching system for optimization
        self.frame_cache: Dict[str, Dict[str, Any]] = {}  # hash -> {results, timestamp}
        self.cache_max_size = get_setting("performance.cache_size_frames", 50)
        self.cache_enabled = get_setting("performance.enable_frame_caching", True)
        self.last_frame_hash: Optional[str] = None
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # FPS tracking for processed frames
        self.frame_times: List[float] = []
        self.max_frame_samples = 30  # Rolling average over 30 frames
        self.current_fps = 0.0
        
        # Playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._on_timer_tick)
        
        # Debounced update system for immediate display refresh with pending processing
        self.pending_update = False
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_update_displays)
        self.debounce_delay_ms = get_setting("performance.debounce_delay_ms", 100)
        
        # Async processing system
        self.async_processor = AsyncVideoProcessor()
        self.async_processor.processing_complete.connect(self._on_async_processing_complete)
        self.async_processor.processing_error.connect(self._on_async_processing_error)
        self.async_processor.queue_status_changed.connect(self._on_async_queue_status_changed)
        
        # Async processing state (always enabled)
        self.enable_async_processing = True
        self.pending_processing_tasks: Dict[str, int] = {}  # task_id -> frame_index
        self.latest_async_results: Optional[ProcessingResult] = None
        self.async_queue_size = 0
        
        # Display throttling for smooth UI
        self.last_display_time = 0.0
        self.min_display_interval = 1.0 / 30.0  # Max 30 FPS display during async processing
        
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
        
        # Load segmentation models
        self._load_segmentation_models()
    
    def _load_homography_params_from_file(self) -> Optional[np.ndarray]:
        """Load homography matrix from the homography_params.yaml file.
        
        Returns:
            Homography matrix as numpy array, or None if loading fails
        """
        try:
            homography_file = Path("configs/homography_params.yaml")
            if not homography_file.exists():
                print(f"[MAIN_TAB] Homography params file not found: {homography_file}")
                return None
            
            with open(homography_file, 'r') as f:
                data = yaml.safe_load(f)
            
            if 'homography_parameters' not in data:
                print("[MAIN_TAB] No homography_parameters found in file")
                return None
            
            params = data['homography_parameters']
            
            # Reconstruct 3x3 matrix from individual elements
            matrix = np.array([
                [params['H00'], params['H01'], params['H02']],
                [params['H10'], params['H11'], params['H12']],
                [params['H20'], params['H21'], 1.0]  # H22 is typically 1.0
            ], dtype=np.float32)
            
            print(f"[MAIN_TAB] Loaded homography matrix from file: {homography_file}")
            print(f"[MAIN_TAB] Matrix:\n{matrix}")
            
            return matrix
            
        except Exception as e:
            print(f"[MAIN_TAB] Error loading homography parameters: {e}")
            return None
    
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
        
        # Center panel: Main video display
        center_panel = self._create_center_panel()
        splitter.addWidget(center_panel)
        
        # Right panel: Homography controls and top-down view
        right_panel = self._create_right_panel()
        right_panel.setMinimumWidth(300)  # Minimum width for homography controls
        right_panel.setMaximumWidth(600)  # Maximum width to prevent taking too much space
        splitter.addWidget(right_panel)
        
        # Simple initial sizing - left takes ~20%, center takes ~50%, right takes ~30%
        splitter.setSizes([350, 1000, 500])  # Initial sizes in pixels
        
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
        self.inference_checkbox.setChecked(True)  # Enable inference by default
        self.inference_checkbox.stateChanged.connect(self._on_inference_toggled)
        
        self.tracking_checkbox = QCheckBox("Object Tracking")  
        self.tracking_checkbox.setToolTip(f"Enable/disable object tracking [{SHORTCUTS['TOGGLE_TRACKING']}]")
        self.tracking_checkbox.setChecked(True)  # Enable tracking by default
        self.tracking_checkbox.stateChanged.connect(self._on_tracking_toggled)
        
        self.player_id_checkbox = QCheckBox("Player Identification")
        self.player_id_checkbox.setToolTip(f"Enable/disable player ID based on jersey numbers [{SHORTCUTS['TOGGLE_PLAYER_ID']}]")
        self.player_id_checkbox.stateChanged.connect(self._on_player_id_toggled)
        
        self.field_segmentation_checkbox = QCheckBox("Field Segmentation")
        self.field_segmentation_checkbox.setToolTip(f"Enable/disable field boundary detection [{SHORTCUTS['TOGGLE_FIELD_SEGMENTATION']}]")
        self.field_segmentation_checkbox.stateChanged.connect(self._on_field_segmentation_toggled)
        self.field_segmentation_checkbox.setChecked(True)  # Enable by default to show advanced field line detection
        
        self.homography_checkbox = QCheckBox("Enable Top-Down View")
        self.homography_checkbox.setChecked(True)  # Enable by default
        self.homography_checkbox.setToolTip("Enable/disable homography transformation for top-down view")
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
        
        # Note: Field segmentation model selection is now in the Field Segmentation section below
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # Field Segmentation Controls
        segmentation_group = QGroupBox("Field Segmentation")
        segmentation_layout = QVBoxLayout()
        
        # Show segmentation checkbox
        self.show_segmentation_checkbox = QCheckBox("Show Field Segmentation")
        self.show_segmentation_checkbox.setChecked(True)  # Enable by default to show field lines
        self.show_segmentation_checkbox.stateChanged.connect(self._on_field_segmentation_toggled)
        segmentation_layout.addWidget(self.show_segmentation_checkbox)
        
        # RANSAC line fitting checkbox
        self.ransac_checkbox = QCheckBox("Use RANSAC Line Fitting")
        self.ransac_checkbox.setChecked(True)  # Enable by default for advanced line detection
        self.ransac_checkbox.stateChanged.connect(self._on_ransac_toggled)
        self.ransac_checkbox.setToolTip("Fit straight lines to contour segments using RANSAC algorithm")
        segmentation_layout.addWidget(self.ransac_checkbox)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        
        self.segmentation_model_combo = QComboBox()
        self.segmentation_model_combo.setMinimumWidth(150)
        self.segmentation_model_combo.currentTextChanged.connect(self._on_segmentation_model_changed)
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
        self.performance_widget = PerformanceWidget()
        layout.addWidget(self.performance_widget)
        
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
        self.video_scroll_area.setMinimumHeight(1080)  # Much bigger for main tab
        self.video_scroll_area.setStyleSheet("""
            QScrollArea {
                border: 2px solid #555;
                background-color: #1a1a1a;
            }
        """)
        
        self.video_label = ZoomableImageLabel()
        self.video_label.setText("No video selected")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                color: #999;
                font-size: 14px;
            }
        """)
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
        self.homography_scroll_area.setStyleSheet("""
            QScrollArea {
                border: 2px solid #555;
                background-color: #1a1a1a;
            }
        """)
        
        self.homography_display_label = ZoomableImageLabel()
        self.homography_display_label.setText("Loading top-down view...")
        self.homography_display_label.setAlignment(Qt.AlignCenter)
        self.homography_display_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                color: #999;
                font-size: 14px;
            }
        """)
        self.homography_scroll_area.setWidget(self.homography_display_label)
        view_layout.addWidget(self.homography_scroll_area)
        
        view_group.setLayout(view_layout)
        layout.addWidget(view_group, 1)  # Give the view group stretch factor of 1 to fill available space
        
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
            duration = get_video_duration(video_path)
            filename = Path(video_path).name
            
            # Create list item with filename and duration
            item_text = f"{filename} ({duration})"
            item = QListWidgetItem(item_text)
            item.setToolTip(video_path)
            self.video_list.addItem(item)
        
        print(f"[MAIN_TAB] Found {len(self.video_files)} video files")
    
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
        
        # Auto-select the default model if available
        self._select_default_model(combo, model_type)
        
        print(f"[MAIN_TAB] Found {len(model_files)} {model_type} models")
    
    def _select_default_model(self, combo: QComboBox, model_type: str):
        """Auto-select the default model in the combo box.
        
        Args:
            combo: QComboBox to update
            model_type: Type of model ("detection" or "segmentation")
        """
        if combo.count() == 0:
            return
            
        # Get the default model path from configuration
        if model_type == "detection":
            default_model = get_setting("models.detection.default_model", FALLBACK_DEFAULTS['model_detection'])
        elif model_type == "segmentation":
            default_model = get_setting("models.segmentation.default_model", FALLBACK_DEFAULTS['model_segmentation'])
        else:
            return
        
        # Convert absolute path to relative path for comparison
        models_path = Path(get_setting("models.base_path", DEFAULT_PATHS['MODELS']))
        try:
            default_relative = Path(default_model).relative_to(models_path)
        except ValueError:
            # If default_model is not under models_path, try to find it directly
            default_relative = Path(default_model)
        
        # Search for matching item in combo
        for i in range(combo.count()):
            item_text = combo.itemText(i)
            if str(default_relative) == item_text or default_model in item_text:
                combo.setCurrentIndex(i)
                print(f"[MAIN_TAB] Auto-selected default {model_type} model: {item_text}")
                
                # Trigger the change handler to actually load the model
                if model_type == "detection":
                    self._on_detection_model_changed(item_text)
                elif model_type == "segmentation":
                    self._on_segmentation_model_changed(item_text)
                break
        else:
            print(f"[MAIN_TAB] Default {model_type} model not found in available models: {default_model}")
    
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
        
        # Clear frame cache when loading new video
        self.frame_cache.clear()
        self.last_frame_hash = None
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        print("[MAIN_TAB] Frame cache cleared for new video")
        
        # Reset homography matrix for new video
        self.homography_matrix = None
        if self.homography_enabled:
            # Reload homography matrix from file for new video
            loaded_matrix = self._load_homography_params_from_file()
            if loaded_matrix is not None:
                self.homography_matrix = loaded_matrix
                print("[MAIN_TAB] Reloaded homography matrix for new video")
        
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
        
        # Check if async processing is enabled
        if self.enable_async_processing:
            # Display frame immediately with previous processing results
            self._display_frame_with_cached_results(frame)
            
            # Submit frame for async processing if any processing is enabled
            enabled_options = self._get_enabled_processing_options()
            if any(enabled_options.values()):
                self._submit_frame_for_async_processing(frame)
        else:
            # Use synchronous processing (original behavior)
            processed_frame = self._process_frame_cached(frame.copy())
            self._display_processed_frame(processed_frame)
        
        # Update homography display if enabled
        if self.homography_enabled:
            self._update_homography_display()
    
    def _display_frame_with_cached_results(self, frame):
        """Display frame immediately using cached processing results.
        
        Args:
            frame: Raw video frame to display
        """
        # Always apply the latest async results we have, even if they're from a recent frame
        if (self.latest_async_results and 
            self.latest_async_results.success):
            
            # Use the latest processing results (don't check frame timing during playback)
            self.current_detections = self.latest_async_results.detections or []
            self.current_tracks = self.latest_async_results.tracks or []
            self.current_field_results = self.latest_async_results.field_results or []
            self.current_player_ids = self.latest_async_results.player_ids or {}
        
        # Process the frame with current results (fast visualization only)
        processed_frame = self._apply_visualizations(frame.copy())
        self._display_processed_frame(processed_frame)
    
    def _display_processed_frame(self, processed_frame):
        """Display a processed frame in the video label.
        
        Args:
            processed_frame: Frame with visualizations applied
        """
        # Throttle display updates for smooth UI during async processing
        current_time = time.time()
        if self.is_playing and (current_time - self.last_display_time) < self.min_display_interval:
            return  # Skip this frame to maintain smooth playback
        
        self.last_display_time = current_time
        
        # Record frame update for FPS tracking
        self.performance_widget.add_frame_update()
        
        # Convert to Qt format and display
        height, width, channel = processed_frame.shape
        bytes_per_line = 3 * width
        
        q_image = QImage(processed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        
        # Use ZoomableImageLabel's set_image method for zoom support
        self.video_label.set_image(pixmap)
    
    def _submit_frame_for_async_processing(self, frame):
        """Submit frame for asynchronous processing.
        
        Args:
            frame: Video frame to process
        """
        if not self.video_player.is_loaded():
            return
        
        # Get current frame index
        video_info = self.video_player.get_video_info()
        frame_index = video_info['current_frame']
        
        # Skip submission if we're playing and have too many pending tasks
        # This improves performance during fast playback
        max_pending = get_setting("processing.async.max_queue_size", 5)
        if self.is_playing and len(self.pending_processing_tasks) >= max_pending:
            # Skip every other frame during fast playback to maintain performance
            if not hasattr(self, '_last_submitted_frame'):
                self._last_submitted_frame = 0
                
            frame_skip = 2 if len(self.pending_processing_tasks) > max_pending // 2 else 1
            if (frame_index - self._last_submitted_frame) < frame_skip:
                return
        
        # Get enabled processing options
        enabled_options = self._get_enabled_processing_options()
        
        # Submit for async processing with high priority for current frame
        task_id = self.async_processor.process_frame_async(
            frame=frame,
            frame_index=frame_index,
            enabled_options=enabled_options,
            priority=10  # High priority for current frame
        )
        
        # Track pending task and last submitted frame
        self.pending_processing_tasks[task_id] = frame_index
        self._last_submitted_frame = frame_index
        
        print(f"[MAIN_TAB] Submitted frame {frame_index} for async processing (task: {task_id})")
    
    def _get_enabled_processing_options(self) -> Dict[str, bool]:
        """Get current enabled processing options.
        
        Returns:
            Dictionary of enabled processing options
        """
        return {
            'inference': self.inference_checkbox.isChecked(),
            'tracking': self.tracking_checkbox.isChecked(),
            'field_segmentation': self.field_segmentation_checkbox.isChecked(),
            'player_id': self.player_id_checkbox.isChecked()
        }
    
    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute a hash of the frame content for caching.
        
        Args:
            frame: Input frame
            
        Returns:
            Hash string representing frame content
        """
        # Use a smaller sample of the frame for faster hashing
        # Sample every 8th pixel to balance speed and collision resistance
        sample = frame[::8, ::8]
        return hashlib.md5(sample.tobytes()).hexdigest()
    
    def _get_processing_cache_key(self) -> str:
        """Generate cache key based on current processing settings.
        
        Returns:
            Cache key string representing current processing configuration
        """
        settings = [
            str(self.inference_checkbox.isChecked()),
            str(self.tracking_checkbox.isChecked()),
            str(self.field_segmentation_checkbox.isChecked()),
            str(self.player_id_checkbox.isChecked())
        ]
        return "|".join(settings)
    
    def _delayed_update_displays(self) -> None:
        """Delayed update handler for debounced UI updates."""
        if self.pending_update and self.video_player.is_loaded():
            frame = self.video_player.get_current_frame()
            if frame is not None:
                self._display_frame(frame)
            self.pending_update = False
            print("[MAIN_TAB] Executed debounced display update")
    
    def _request_display_update(self, immediate: bool = False) -> None:
        """Request a display update with debouncing to prevent excessive recomputation.
        
        Args:
            immediate: If True, force immediate update and set pending flag
        """
        if immediate:
            # Force immediate update but set pending flag for debounced processing
            self.pending_update = True
            if self.video_player.is_loaded():
                frame = self.video_player.get_current_frame()
                if frame is not None:
                    self._display_frame(frame)
        else:
            # Standard debounced update
            self.pending_update = True
            self.update_timer.start(self.debounce_delay_ms)
    def _cleanup_frame_cache(self) -> None:
        """Clean up old cache entries to maintain memory limits."""
        if len(self.frame_cache) <= self.cache_max_size:
            return
            
        # Sort by timestamp and remove oldest entries
        sorted_entries = sorted(
            self.frame_cache.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # Remove oldest entries to get back to 80% of max size
        target_size = int(self.cache_max_size * 0.8)
        entries_to_remove = len(sorted_entries) - target_size
        
        for i in range(entries_to_remove):
            cache_key = sorted_entries[i][0]
            del self.frame_cache[cache_key]
            
        print(f"[MAIN_TAB] Cache cleanup: removed {entries_to_remove} entries, {len(self.frame_cache)} remaining")
    
    def _process_frame_cached(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with caching optimization to avoid redundant computations.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Processed frame with visualizations
        """
        # Start total runtime timer at the very beginning
        total_runtime_start = time.time()
        
        if not self.cache_enabled:
            processed_frame = self._process_frame(frame)
            # Record total runtime for non-cached processing
            total_runtime_ms = (time.time() - total_runtime_start) * 1000
            self.performance_widget.add_processing_measurement("Total Runtime", total_runtime_ms)
            self._update_fps(total_runtime_ms)
            return processed_frame
            
        # Compute frame content hash
        frame_hash = self._compute_frame_hash(frame)
        processing_key = self._get_processing_cache_key()
        cache_key = f"{frame_hash}_{processing_key}"
        
        # Check cache hit
        current_time = time.time()
        if cache_key in self.frame_cache:
            # Cache hit - return cached results
            cached_data = self.frame_cache[cache_key]
            
            # Update cache timestamp and restore processing results
            cached_data['timestamp'] = current_time
            self.current_detections = cached_data['detections']
            self.current_tracks = cached_data['tracks'] 
            self.current_field_results = cached_data['field_results']
            self.current_player_ids = cached_data['player_ids']
            
            # Apply visualizations to the frame
            viz_start_time = time.time()
            processed_frame = self._apply_visualizations(frame.copy())
            viz_duration_ms = (time.time() - viz_start_time) * 1000
            
            # Record cache hit performance with total runtime from start of method
            total_runtime_ms = (time.time() - total_runtime_start) * 1000
            self.cache_hit_count += 1
            self.performance_widget.add_processing_measurement("Visualization", viz_duration_ms)
            self.performance_widget.add_processing_measurement("Total Runtime", total_runtime_ms)
            self._update_fps(total_runtime_ms)
            
            # Log cache efficiency periodically
            if (self.cache_hit_count + self.cache_miss_count) % 30 == 0:
                total_requests = self.cache_hit_count + self.cache_miss_count
                hit_rate = (self.cache_hit_count / total_requests) * 100
                print(f"[MAIN_TAB] Cache hit rate: {hit_rate:.1f}% ({self.cache_hit_count}/{total_requests})")
            
            return processed_frame
        
        # Cache miss - perform full processing
        self.cache_miss_count += 1
        processed_frame = self._process_frame(frame)
        
        # Store results in cache
        self.frame_cache[cache_key] = {
            'timestamp': current_time,
            'detections': self.current_detections.copy(),
            'tracks': self.current_tracks.copy() if self.current_tracks else [],
            'field_results': self.current_field_results.copy() if self.current_field_results else [],
            'player_ids': self.current_player_ids.copy()
        }
        
        # Clean up cache if needed
        self._cleanup_frame_cache()
        
        # Record total runtime from start of method
        total_runtime_ms = (time.time() - total_runtime_start) * 1000
        self.performance_widget.add_processing_measurement("Total Runtime", total_runtime_ms)
        self._update_fps(total_runtime_ms)
        
        self.last_frame_hash = frame_hash
        return processed_frame
    
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
        else:
            # Clear field results when disabled
            self.current_field_results = []
            self.ransac_lines = []
            self.ransac_confidences = []
            self.all_lines_for_display = {}
        
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
        
        # Add line extraction timing if RANSAC was used and field segmentation was enabled
        if self.all_lines_for_display and self.field_segmentation_checkbox.isChecked():
            # Estimate line extraction time (this is included in field segmentation timing)
            line_count = len(self.all_lines_for_display)
            estimated_line_extraction_ms = viz_duration_ms * 0.2  # Roughly 20% of viz time
            self.performance_widget.add_processing_measurement("Line Extraction", estimated_line_extraction_ms)
        elif not self.field_segmentation_checkbox.isChecked():
            # Set line extraction to 0 when field segmentation is disabled
            self.performance_widget.add_processing_measurement("Line Extraction", 0.0)
        
        # Add default homography measurements if homography is not enabled
        if not self.homography_enabled:
            self.performance_widget.add_processing_measurement("Homography Calculation", 0.0)
            self.performance_widget.add_processing_measurement("Homography Display", 0.0)
        
        return frame
    
    def _apply_visualizations(self, frame):
        """Apply visualization overlays to frame with memory and performance optimization.
        
        Args:
            frame: Frame to add visualizations to
            
        Returns:
            Frame with visualizations applied
        """
        # Early return for minimal processing if no overlays are enabled
        viz_enabled = (self.field_segmentation_checkbox.isChecked() or 
                      self.current_detections or self.current_tracks)
        
        if not viz_enabled:
            # Add only FPS overlay for minimal processing
            self._draw_fps_overlay(frame)
            return frame
        
        # Apply field segmentation overlay first (as background) - contour only for better performance
        if self.current_field_results and self.field_segmentation_checkbox.isChecked():
            # Show raw segmentation model output if enabled
            from ..config.settings import get_setting
            show_raw_masks = get_setting("models.segmentation.show_raw_masks", True)
            if show_raw_masks:
                frame = draw_field_segmentation(frame, self.current_field_results)
                print(f"[MAIN_TAB] Applied raw segmentation masks to frame")
            
            # Create and display unified mask with RANSAC line detection
            frame_shape = frame.shape[:2]  # (height, width)
            unified_mask = create_unified_field_mask(self.current_field_results, frame_shape)
            
            if unified_mask is not None:
                # Extract raw RANSAC lines for direct use
                detected_lines, confidences = extract_raw_lines_from_segmentation(
                    self.current_field_results, frame_shape)
                
                # Store RANSAC lines directly
                if detected_lines:
                    self.ransac_lines = detected_lines
                    self.ransac_confidences = confidences
                    print(f"[MAIN_TAB] Using {len(self.ransac_lines)} RANSAC lines directly")
                else:
                    self.ransac_lines = []
                    self.ransac_confidences = []
                
                # Use the same color as segmentation for consistency
                from .visualization import get_primary_field_color
                field_color = get_primary_field_color()  # Bright cyan (BGR) - same as segmentation
                frame, raw_lines_dict, self.all_lines_for_display = draw_unified_field_mask(
                    frame, unified_mask, field_color, alpha=0.3, fill_mask=False)
                print(f"[MAIN_TAB] Applied field contour (no fill) to frame: {np.sum(unified_mask)} pixels")
                
                # Draw raw RANSAC lines only in main view (tracking data for top-down view)
                if self.all_lines_for_display:
                    frame = draw_all_field_lines(frame, self.all_lines_for_display, 
                                               scale_factor=1.0, draw_raw_lines_only=True)
                    print(f"[MAIN_TAB] Added raw RANSAC lines for {len(self.all_lines_for_display)} field lines")
                
                # Overlay RANSAC field lines on main view
                if self.ransac_lines:
                    frame = draw_ransac_field_lines(frame, self.ransac_lines, 
                                                   self.ransac_confidences, transformation_matrix=None, 
                                                   scale_factor=1.0)
                    print(f"[MAIN_TAB] Added {len(self.ransac_lines)} RANSAC lines to main view")
            else:
                print("[MAIN_TAB] No unified mask could be created")
                self.ransac_lines = []
                self.ransac_confidences = []
                self.all_lines_for_display = {}
        
        # Conditional visualization based on enabled options (avoid unnecessary processing)
        
        # Show detections only if tracking is NOT enabled (to avoid visual clutter)
        if self.current_detections and not self.tracking_checkbox.isChecked():
            frame = draw_detections(frame, self.current_detections)
        
        # Show tracking visualization if tracking is enabled
        elif self.current_tracks and self.tracking_checkbox.isChecked():
            # Get track histories for trajectory visualization (cached in processing module)
            track_histories = get_track_histories()
            
            # Use player ID visualization if player ID is enabled (even if no IDs detected yet)
            if self.player_id_checkbox.isChecked():
                frame = draw_tracks_with_player_ids(frame, self.current_tracks, track_histories, self.current_player_ids)
            else:
                frame = draw_tracks(frame, self.current_tracks, track_histories)
        
        # Add FPS overlay to top right (lightweight)
        self._draw_fps_overlay(frame)
        
        # Add jersey tracking table overlay if player ID is enabled (conditional rendering)
        if self.player_id_checkbox.isChecked() and self.current_player_ids:
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
        """Handle seek bar movement with immediate display update."""
        if self.video_player.is_loaded():
            self.video_player.seek_to_frame(frame_idx)
            
            # Use immediate display update for scrubbing responsiveness
            self._request_display_update(immediate=True)
    
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
    
    # Processing control event handlers with debounced updates
    def _on_inference_toggled(self, checked: bool):
        """Handle inference checkbox toggle with debounced update."""
        print(f"[MAIN_TAB] Inference {'enabled' if checked else 'disabled'}")
        # Clear cache when processing settings change
        self.frame_cache.clear()
        self._request_display_update()
    
    def _on_tracking_toggled(self, checked: bool):
        """Handle tracking checkbox toggle with debounced update."""
        print(f"[MAIN_TAB] Tracking {'enabled' if checked else 'disabled'}")
        if checked:
            # Enable inference if tracking is enabled
            self.inference_checkbox.setChecked(True)
        # Clear cache when processing settings change
        self.frame_cache.clear()
        self._request_display_update()
    
    def _on_player_id_toggled(self, checked: bool):
        """Handle player ID checkbox toggle with debounced update."""
        print(f"[MAIN_TAB] Player ID {'enabled' if checked else 'disabled'}")
        if checked:
            # Enable tracking and inference if player ID is enabled
            self.tracking_checkbox.setChecked(True)
            self.inference_checkbox.setChecked(True)
            
            print("[MAIN_TAB] Player ID using EasyOCR for jersey number recognition")
        # Clear cache when processing settings change
        self.frame_cache.clear()
        self._request_display_update()
    
    def _on_field_segmentation_toggled(self, checked: bool):
        """Handle field segmentation checkbox toggle with debounced update."""
        print(f"[MAIN_TAB] Field segmentation {'enabled' if checked else 'disabled'}")
        
        if checked:
            # Ensure field segmentation model is loaded with the default from configuration
            default_model = get_setting(
                "models.segmentation.default_model", 
                FALLBACK_DEFAULTS['model_segmentation']
            )
            
            if Path(default_model).exists():
                set_field_model(str(default_model))
                print(f"[MAIN_TAB] Field segmentation model set to: {default_model}")
            else:
                print(f"[MAIN_TAB] Default field segmentation model not found at: {default_model}")
                print("[MAIN_TAB] Will use fallback models or mock results")
        # Clear cache when processing settings change
        self.frame_cache.clear()
        self._request_display_update()
    
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
    
    def _on_segmentation_model_changed(self, display_name: str):
        """Handle segmentation model selection change."""
        if self.segmentation_model_combo is None:
            return
            
        model_path = self.segmentation_model_combo.currentData()
        if model_path and os.path.exists(model_path):
            print(f"[MAIN_TAB] Changing segmentation model to: {model_path}")
            try:
                if set_field_model(model_path):
                    print(f"[MAIN_TAB] Successfully loaded segmentation model: {display_name}")
                    # Force re-run segmentation with new model
                    if self.show_segmentation:
                        self.current_segmentation_results = None
                        self._request_display_update()
                else:
                    print(f"[MAIN_TAB] Failed to load segmentation model: {model_path}")
            except Exception as e:
                print(f"[MAIN_TAB] Error loading segmentation model: {e}")
        else:
            print(f"[MAIN_TAB] Invalid model path: {model_path}")

    def _on_homography_toggled(self, state: int):
        """Handle homography checkbox toggle."""
        self.homography_enabled = state == 2  # Qt.Checked = 2
        
        print(f"[MAIN_TAB] Homography {'enabled' if self.homography_enabled else 'disabled'}")
        
        if self.homography_enabled:
            # Try to load homography matrix if not already loaded
            if self.homography_matrix is None:
                loaded_matrix = self._load_homography_params_from_file()
                if loaded_matrix is not None:
                    self.homography_matrix = loaded_matrix
            self._update_homography_display()
        else:
            if self.homography_display_label:
                self.homography_display_label.setText("Homography view disabled")
                self.homography_warped_frame = None
    
    def _on_segmentation_toggled(self, state: int):
        """Handle segmentation checkbox toggle."""
        self.show_segmentation = state == 2  # Qt.Checked = 2
        
        print(f"[MAIN_TAB] Segmentation toggle: state={state}, show_segmentation={self.show_segmentation}")
        
        if self.show_segmentation:
            print("[MAIN_TAB] Field segmentation enabled")
            self._update_displays()
        else:
            print("[MAIN_TAB] Field segmentation disabled")
            self.current_segmentation_results = None
            self._update_displays()
    
    def _on_ransac_toggled(self, state: int):
        """Handle RANSAC line fitting checkbox toggle."""
        ransac_enabled = state == 2  # Qt.Checked = 2
        
        print(f"[MAIN_TAB] RANSAC toggle: state={state}, ransac_enabled={ransac_enabled}")
        
        # Temporarily override the config value in memory
        from ..config.settings import get_config
        config = get_config()
        if 'models' not in config:
            config['models'] = {}
        if 'segmentation' not in config['models']:
            config['models']['segmentation'] = {}
        if 'contour' not in config['models']['segmentation']:
            config['models']['segmentation']['contour'] = {}
        if 'ransac' not in config['models']['segmentation']['contour']:
            config['models']['segmentation']['contour']['ransac'] = {}
        
        config['models']['segmentation']['contour']['ransac']['enabled'] = ransac_enabled
        
        # Update displays if field segmentation is currently shown
        if self.show_segmentation_checkbox.isChecked():
            self._update_frame_display()
    
    def _load_segmentation_models(self):
        """Load available field segmentation models using utility function."""
        from ..utils.segmentation_utils import load_segmentation_models, populate_segmentation_model_combo
        
        self.available_segmentation_models = load_segmentation_models()
        
        # Update combo box
        if hasattr(self, 'segmentation_model_combo') and self.segmentation_model_combo is not None:
            default_model_path = "data/models/segmentation/20250826_1_segmentation_yolo11s-seg_field finder.v8i.yolov8/finetune_20250826_092226/weights/best.pt"
            populate_segmentation_model_combo(
                self.segmentation_model_combo, 
                self.available_segmentation_models,
                default_model_path
            )
        
        print(f"[MAIN_TAB] Loaded {len(self.available_segmentation_models)} segmentation models")
    
    def _map_tracked_objects_to_top_down(self, warped_frame: np.ndarray) -> np.ndarray:
        """Map tracked objects to the top-down view using their foot positions.
        
        Args:
            warped_frame: The homography-transformed frame
            
        Returns:
            Frame with tracked objects mapped to top-down view
        """
        if not self.current_tracks or self.homography_matrix is None:
            return warped_frame
        
        result_frame = warped_frame.copy()
        
        for track in self.current_tracks:
            # Get track properties
            track_id = getattr(track, 'track_id', None)
            if track_id is None:
                continue
                
            # Get bounding box
            bbox = None
            if hasattr(track, 'to_ltrb'):
                bbox = track.to_ltrb()
            elif hasattr(track, 'to_tlbr'):
                bbox = track.to_tlbr()
            elif hasattr(track, 'bbox'):
                bbox = track.bbox
            
            if bbox is None or len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate foot position (bottom center of bounding box)
            foot_x = (x1 + x2) / 2
            foot_y = y2  # Bottom of bounding box represents feet
            
            # Transform foot position using homography matrix
            foot_point = np.array([[[foot_x, foot_y]]], dtype=np.float32)
            try:
                transformed_foot = cv2.perspectiveTransform(foot_point, self.homography_matrix)
                transformed_x = int(transformed_foot[0][0][0])
                transformed_y = int(transformed_foot[0][0][1])
                
                # Check if transformed position is within frame bounds
                frame_h, frame_w = warped_frame.shape[:2]
                if 0 <= transformed_x < frame_w and 0 <= transformed_y < frame_h:
                    # Generate unique color for each track ID
                    from .visualization import _get_track_color
                    color = _get_track_color(track_id)
                    
                    # Draw foot position as a circle (larger for top-down view)
                    cv2.circle(result_frame, (transformed_x, transformed_y), 12, color, -1)
                    
                    # Draw track ID label with larger font for top-down view
                    label_text = f"ID:{track_id}"
                    
                    # Add player jersey number if available
                    if track_id in self.current_player_ids:
                        jersey_number, _ = self.current_player_ids[track_id]
                        if jersey_number != "Unknown":
                            label_text = f"#{jersey_number}"
                    
                    # Draw label background for better visibility (larger for top-down view)
                    font_scale = 1.0  # Increased from 0.6 for better visibility in top-down view
                    font_thickness = 3  # Increased from 2 for better visibility
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    label_bg_x1 = transformed_x - label_size[0] // 2 - 5
                    label_bg_y1 = transformed_y - 35
                    label_bg_x2 = transformed_x + label_size[0] // 2 + 5
                    label_bg_y2 = transformed_y - 5
                    
                    cv2.rectangle(result_frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
                    
                    # Draw label text with larger font
                    cv2.putText(result_frame, label_text, 
                              (transformed_x - label_size[0] // 2, transformed_y - 15),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                    
                    # Draw direction indicator if track has history
                    track_histories = get_track_histories()
                    if track_histories and track_id in track_histories:
                        history = track_histories[track_id]
                        if len(history) >= 2:
                            # Get last two foot positions and transform them
                            last_pos = history[-1]
                            prev_pos = history[-2] if len(history) > 1 else history[-1]
                            
                            # Transform previous position
                            prev_point = np.array([[[prev_pos[0], prev_pos[1]]]], dtype=np.float32)
                            try:
                                transformed_prev = cv2.perspectiveTransform(prev_point, self.homography_matrix)
                                prev_x = int(transformed_prev[0][0][0])
                                prev_y = int(transformed_prev[0][0][1])
                                
                                # Draw direction arrow
                                if 0 <= prev_x < frame_w and 0 <= prev_y < frame_h:
                                    # Calculate direction vector
                                    dx = transformed_x - prev_x
                                    dy = transformed_y - prev_y
                                    length = (dx*dx + dy*dy) ** 0.5
                                    
                                    if length > 5:  # Only draw if significant movement
                                        # Normalize and scale
                                        dx = int(dx / length * 15)
                                        dy = int(dy / length * 15)
                                        
                                        # Draw arrow
                                        arrow_end_x = transformed_x + dx
                                        arrow_end_y = transformed_y + dy
                                        cv2.arrowedLine(result_frame, 
                                                      (transformed_x, transformed_y),
                                                      (arrow_end_x, arrow_end_y),
                                                      color, 2, tipLength=0.3)
                            except:
                                pass  # Skip if transformation fails
                    
            except Exception as e:
                print(f"[MAIN_TAB] Error transforming track {track_id} position: {e}")
                continue
        
        return result_frame

    def _calculate_output_canvas_size(self, input_width: int, input_height: int) -> Tuple[int, int]:
        """Calculate output canvas size with specified aspect ratio.
        
        Args:
            input_width: Original frame width
            input_height: Original frame height
            
        Returns:
            Tuple of (output_width, output_height) with 3:1 aspect ratio
        """
        # Get configuration settings
        buffer_factor = get_setting("homography.buffer_factor", 2.5)
        aspect_ratio = get_setting("homography.output_aspect_ratio", 3.0)  # height:width
        
        # Calculate total area we want to maintain (similar to original but with buffer)
        original_area = input_width * input_height
        target_area = int(original_area * buffer_factor)
        
        # Calculate output dimensions with specified aspect ratio
        # For aspect_ratio = height/width, we have: height = aspect_ratio * width
        # Area = width * height = width * (aspect_ratio * width) = aspect_ratio * width^2
        # Therefore: width = sqrt(area / aspect_ratio), height = aspect_ratio * width
        
        if aspect_ratio >= 1.0:
            # Height >= Width (e.g., 3:1 ratio means height = 3 * width)
            output_width = int(np.sqrt(target_area / aspect_ratio))
            output_height = int(output_width * aspect_ratio)
        else:
            # Width > Height (e.g., 1:3 ratio means width = 3 * height)
            output_height = int(np.sqrt(target_area * aspect_ratio))
            output_width = int(output_height / aspect_ratio)
        
        print(f"[MAIN_TAB] Canvas size: {input_width}x{input_height} -> {output_width}x{output_height} (aspect {aspect_ratio:.1f}:1, area: {input_width*input_height} -> {output_width*output_height})")
        return output_width, output_height

    def _update_homography_display(self):
        """Update the homography display with the current frame and transformation."""
        if not self.homography_enabled or self.homography_display_label is None:
            return
        
        homography_start_time = time.time()
        
        try:
            # Get current frame
            if self.video_player.is_loaded():
                frame = self.video_player.get_current_frame()
                if frame is None:
                    self.homography_display_label.setText("No frame available")
                    return
                
                # Apply homography transformation
                if self.homography_matrix is not None:
                    # Apply the transformation with timing
                    homography_calc_start = time.time()
                    height, width = frame.shape[:2]
                    
                    # Calculate output canvas size with 3:1 aspect ratio
                    output_width, output_height = self._calculate_output_canvas_size(width, height)
                    
                    warped_frame = cv2.warpPerspective(frame, self.homography_matrix, (output_width, output_height))
                    homography_calc_duration_ms = (time.time() - homography_calc_start) * 1000
                    self.performance_widget.add_processing_measurement("Homography Calculation", homography_calc_duration_ms)
                    
                    # Apply field segmentation to warped frame if available
                    if self.current_field_results and self.field_segmentation_checkbox.isChecked():
                        try:
                            # Transform the segmentation masks to match the warped frame
                            original_frame_shape = frame.shape[:2]  # (height, width)
                            warped_frame_with_segmentation = apply_segmentation_to_warped_frame(
                                warped_frame, self.current_field_results, self.homography_matrix, 
                                original_frame_shape, "MAIN_TAB"
                            )
                            if warped_frame_with_segmentation is not None:
                                warped_frame = warped_frame_with_segmentation
                                print(f"[MAIN_TAB] Applied transformed segmentation to homography view: {len(self.current_field_results)} results")
                            else:
                                print("[MAIN_TAB] Failed to apply transformed segmentation to homography view")
                        except Exception as e:
                            print(f"[MAIN_TAB] Error applying segmentation to homography view: {e}")
                    
                    # Map tracked objects to top-down view if tracking is enabled
                    if self.tracking_checkbox.isChecked() and self.current_tracks:
                        warped_frame = self._map_tracked_objects_to_top_down(warped_frame)
                        print(f"[MAIN_TAB] Mapped {len(self.current_tracks)} tracked objects to top-down view")
                    
                    # Add RANSAC field lines to top-down view
                    if self.ransac_lines:
                        # Use RANSAC lines with confidence display for top-down view
                        warped_frame = draw_ransac_field_lines(warped_frame, self.ransac_lines, 
                                                              self.ransac_confidences, self.homography_matrix, 
                                                              scale_factor=2.0)
                        print(f"[MAIN_TAB] Added RANSAC field lines to top-down view: {len(self.ransac_lines)} lines")
                    elif self.all_lines_for_display:
                        # Fallback to classified lines if no RANSAC lines available
                        warped_frame = draw_all_field_lines(warped_frame, self.all_lines_for_display, 
                                                          self.homography_matrix, scale_factor=2.0, 
                                                          draw_raw_lines_only=False)
                        print(f"[MAIN_TAB] Added classified field lines to top-down view (fallback): {len(self.all_lines_for_display)} lines")
                    
                    self.homography_warped_frame = warped_frame
                    
                    # Convert warped frame to Qt format and display (preserve aspect ratio)
                    warped_height, warped_width, warped_channel = warped_frame.shape
                    bytes_per_line = 3 * warped_width
                    
                    # Ensure the frame is contiguous for QImage
                    warped_frame = np.ascontiguousarray(warped_frame)
                    
                    q_image = QImage(warped_frame.data, warped_width, warped_height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                    
                    # Display with preserved aspect ratio using ZoomableImageLabel
                    pixmap = QPixmap.fromImage(q_image)
                    self.homography_display_label.set_image(pixmap)
                    
                    # Record homography processing time
                    homography_duration_ms = (time.time() - homography_start_time) * 1000
                    self.performance_widget.add_processing_measurement("Homography Display", homography_duration_ms)
                    
                    print("[MAIN_TAB] Updated homography display using loaded matrix")
                else:
                    self.homography_display_label.setText("Homography matrix not available")
            else:
                self.homography_display_label.setText("No video loaded")
                
        except Exception as e:
            print(f"[MAIN_TAB] Error updating homography display: {e}")
            self.homography_display_label.setText(f"Error: {str(e)}")
    
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
        
        # Shutdown async processor
        if hasattr(self, 'async_processor'):
            self.async_processor.shutdown()
        
        super().closeEvent(event)
    
    def _on_async_processing_complete(self, result: ProcessingResult):
        """Handle completion of async processing task.
        
        Args:
            result: ProcessingResult containing the analysis results
        """
        # Remove task from pending list
        if result.task_id in self.pending_processing_tasks:
            frame_index = self.pending_processing_tasks.pop(result.task_id)
            
            # Always update latest results if successful, regardless of frame timing
            if result.success:
                # Update processing results
                self.current_detections = result.detections or []
                self.current_tracks = result.tracks or []
                self.current_field_results = result.field_results or []
                self.current_player_ids = result.player_ids or {}
                
                # Store latest results
                self.latest_async_results = result
                
                # Update performance metrics
                if result.performance_metrics:
                    for name, time_ms in result.performance_metrics.items():
                        self.performance_widget.add_processing_measurement(name, time_ms)
                
                # Only request display update if we're not currently playing or it's been a while
                # (during playback, the timer will handle frame display and throttling will apply)
                current_time = time.time()
                if not self.is_playing or (current_time - self.last_display_time) >= self.min_display_interval:
                    self._request_display_update(immediate=True)
                
                print(f"[MAIN_TAB] Applied async results for frame {frame_index}")
            else:
                print(f"[MAIN_TAB] Failed async processing for frame {frame_index}")
        else:
            print(f"[MAIN_TAB] Received unexpected async result for task {result.task_id}")
    
    def _on_async_processing_error(self, task_id: str, error_message: str):
        """Handle async processing error.
        
        Args:
            task_id: ID of the failed task
            error_message: Error description
        """
        print(f"[MAIN_TAB] Async processing error for task {task_id}: {error_message}")
        
        # Remove from pending tasks
        if task_id in self.pending_processing_tasks:
            self.pending_processing_tasks.pop(task_id)
    
    def _on_async_queue_status_changed(self, queue_size: int):
        """Handle change in async processing queue size.
        
        Args:
            queue_size: Current number of tasks in queue
        """
        self.async_queue_size = queue_size
        # Could update UI indicator here if desired
