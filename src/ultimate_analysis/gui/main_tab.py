"""Main tab for Ultimate Analysis GUI.

This module contains the main video analysis interface with video list,
playback controls, and processing options.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config.settings import get_setting
from ..constants import (
    DEFAULT_PATHS,
    FALLBACK_DEFAULTS,
    SHORTCUTS,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from ..processing import (
    get_track_histories,
    reset_tracker,
    run_field_segmentation,
    run_inference,
    run_player_id_on_tracks,
    run_tracking,
    set_field_model,
)
from ..processing.inference import reset_inference_state, warmup_models
from ..processing.field_segmentation import reset_segmentation_cache
from ..processing.jersey_tracker import get_best_jersey_number, get_jersey_tracker
from ..processing.line_extraction import fit_lines_from_mask
from ..processing.player_id import initialize_player_id_system
from ..utils.logger import get_logger
from ..utils.segmentation_utils import (
    apply_segmentation_to_warped_frame,
)
from ..utils.video_utils import get_video_duration
from .homography_tab import ZoomableImageLabel
from .performance_widget import PerformanceWidget
from .ransac_line_visualization import draw_ransac_field_lines
from .video_player import VideoPlayer
from .visualization import (
    create_unified_field_mask,
    draw_all_field_lines,
    draw_detections,
    draw_field_segmentation,
    draw_tracks,
    draw_tracks_with_player_ids,
    draw_unified_field_mask,
)


class VideoListWidget(QListWidget):
    """Enhanced video list widget with duration information."""

    def __init__(self):
        super().__init__()

        self.setStyleSheet(
            """
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #2a2a2a;
            }
        """
        )


class MainTab(QWidget):
    """Main video analysis tab with video player and processing controls."""

    # Signals
    video_changed = pyqtSignal(str)  # Emitted when video selection changes

    def __init__(self):
        super().__init__()

        # Initialize logger
        self.logger = get_logger("MAIN_TAB")

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
        self.segmentation_model_combo: Optional[QComboBox] = None
        self.show_segmentation_checkbox: Optional[QCheckBox] = None
        self.ransac_checkbox: Optional[QCheckBox] = None
        self.available_segmentation_models: List[str] = []
        self.ransac_lines: List[
            Tuple[np.ndarray, np.ndarray]
        ] = []  # Store RANSAC-calculated field lines
        self.ransac_confidences: List[float] = []  # Store RANSAC line confidences
        self.all_lines_for_display: Dict[
            str, Tuple[np.ndarray, float, bool]
        ] = {}  # Store all lines for display
        # Geometry is derived solely from a segmentation result and the target frame
        # size.  Segmentation can intentionally be reused for several frames, so
        # avoid re-building its mask and re-running RANSAC for every redraw.
        self._field_geometry_cache_results: Optional[List[Any]] = None
        self._field_geometry_cache_frame_shape: Optional[Tuple[int, int]] = None
        self._cached_unified_field_mask: Optional[np.ndarray] = None
        self._cached_ransac_lines: List[Tuple[np.ndarray, np.ndarray]] = []
        self._cached_ransac_confidences: List[float] = []
        self._cached_field_contour: Optional[np.ndarray] = None
        self._cached_ransac_fit: Optional[tuple] = None

        # Homography state
        self.homography_enabled = True  # Enable by default
        self.homography_matrix: Optional[np.ndarray] = None
        self.homography_display_label: Optional[QLabel] = None
        # Undecorated frame currently on screen, reused by the top-down view
        self._last_raw_frame: Optional[np.ndarray] = None

        # Try to load homography matrix from file
        loaded_matrix = self._load_homography_params_from_file()
        if loaded_matrix is not None:
            self.homography_matrix = loaded_matrix
            print("[MAIN_TAB] Using homography matrix from file")
        else:
            print("[MAIN_TAB] Using default homography parameters (no file found)")

        # Frame-based caching system for optimization
        self.frame_cache: Dict[str, Dict[str, Any]] = {}  # cache key -> results
        self.cache_enabled = get_setting("performance.enable_frame_caching", True)
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        # FPS tracking for processed frames
        self.frame_times: List[float] = []
        self.max_frame_samples = 30  # Rolling average over 30 frames
        self.current_fps = 0.0
        # Global frame counter for optimization intervals
        self.global_frame_index: int = 0
        # Tracks whose jersey numbers are finalized (certainty threshold reached)
        self.finalized_player_id_tracks: set[int] = set()
        # Track last frame we updated a jersey number (for pruning stale entries)
        self.player_id_last_seen: Dict[int, int] = {}

        # Playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._on_timer_tick)

        # Debounced update system for immediate display refresh with pending processing
        self.pending_update = False
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_update_displays)
        self.debounce_delay_ms = get_setting("performance.debounce_delay_ms", 100)

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
                print(
                    f"[MAIN_TAB] Default video '{default_video_name}' not found, loading first available video"
                )

            if self.video_list.currentRow() == default_index:
                self._load_selected_video()
            else:
                # The selection signal loads the video; do not load and warm it twice.
                self.video_list.setCurrentRow(default_index)

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

            with open(homography_file, "r") as f:
                data = yaml.safe_load(f)

            if "homography_parameters" not in data:
                print("[MAIN_TAB] No homography_parameters found in file")
                return None

            params = data["homography_parameters"]

            # Reconstruct 3x3 matrix from individual elements
            matrix = np.array(
                [
                    [params["H00"], params["H01"], params["H02"]],
                    [params["H10"], params["H11"], params["H12"]],
                    [params["H20"], params["H21"], 1.0],  # H22 is typically 1.0
                ],
                dtype=np.float32,
            )

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

        # Player detection model dropdown
        self.player_model_combo = QComboBox()
        self._populate_model_combo(self.player_model_combo, "player_detection")
        self.player_model_combo.currentTextChanged.connect(self._on_player_model_changed)
        models_layout.addRow("Player Detection Model:", self.player_model_combo)

        # Disc detection model dropdown
        self.disc_model_combo = QComboBox()
        self._populate_model_combo(self.disc_model_combo, "disc_detection")
        self.disc_model_combo.currentTextChanged.connect(self._on_disc_model_changed)
        models_layout.addRow("Disc Detection Model:", self.disc_model_combo)

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
        self.ransac_checkbox.setToolTip(
            "Fit straight lines to contour segments using RANSAC algorithm"
        )
        segmentation_layout.addWidget(self.ransac_checkbox)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))

        self.segmentation_model_combo = QComboBox()
        self.segmentation_model_combo.setMinimumWidth(150)
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

    def _load_videos(self):
        """Load and display available video files."""
        print("[MAIN_TAB] Loading video files...")

        self.video_files.clear()
        self.video_list.clear()

        # Search paths for videos
        search_paths = [Path(DEFAULT_PATHS["DEV_DATA"]), Path(DEFAULT_PATHS["RAW_VIDEOS"])]

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
        """Populate a combo box with the trained detection models for one class.

        Args:
            combo: QComboBox to populate
            model_type: "player_detection" or "disc_detection"
        """
        combo.clear()

        # Look for models in the models directory
        models_path = Path(get_setting("models.base_path", DEFAULT_PATHS["MODELS"]))

        if not models_path.exists():
            print(f"[MAIN_TAB] Models directory not found: {models_path}")
            return

        # Only finished training runs (best.pt) whose dataset contains the class are useful
        # here; generic pretrained weights and models for the other class are left out.
        target_class = "disc" if model_type == "disc_detection" else "player"
        model_files = []
        for model_file in (models_path / "detection").rglob("best.pt"):
            class_names = self._get_model_class_names(model_file)
            if class_names is None or target_class in class_names:
                model_files.append(str(model_file.relative_to(models_path)))

        # Sort and add to combo
        model_files.sort()
        combo.addItems(model_files)

        # Auto-select the default model if available
        self._select_default_model(combo, model_type)

        print(f"[MAIN_TAB] Found {len(model_files)} {model_type} models")

    @staticmethod
    def _get_model_class_names(model_file: Path) -> Optional[List[str]]:
        """Class names a trained model was trained on, from its run's args.yaml and dataset.

        Returns None when they cannot be determined (e.g. the dataset was removed).
        """
        try:
            with open(model_file.parents[1] / "args.yaml", "r") as f:
                data_yaml = yaml.safe_load(f)["data"]
            with open(data_yaml, "r") as f:
                names = yaml.safe_load(f)["names"]
            return [str(name) for name in (names.values() if isinstance(names, dict) else names)]
        except Exception:
            return None

    def _select_default_model(self, combo: QComboBox, model_type: str):
        """Auto-select the default model in the combo box.

        Args:
            combo: QComboBox to update
            model_type: Type of model ("player_detection", "disc_detection", or "segmentation")
        """
        if combo.count() == 0:
            return

        # Get the default model path from configuration
        if model_type == "player_detection":
            default_model = get_setting(
                "models.player_detection.default_model", FALLBACK_DEFAULTS["model_player_detection"]
            )
        elif model_type == "disc_detection":
            default_model = get_setting(
                "models.disc_detection.default_model", FALLBACK_DEFAULTS["model_disc_detection"]
            )
        elif model_type == "segmentation":
            default_model = get_setting(
                "models.segmentation.default_model", FALLBACK_DEFAULTS["model_segmentation"]
            )
        else:
            return

        # Convert absolute path to relative path for comparison
        models_path = Path(get_setting("models.base_path", DEFAULT_PATHS["MODELS"]))
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
                if model_type == "player_detection":
                    self._on_player_model_changed(item_text)
                elif model_type == "disc_detection":
                    self._on_disc_model_changed(item_text)
                elif model_type == "segmentation":
                    self._on_segmentation_model_changed(item_text)
                break
        else:
            print(
                f"[MAIN_TAB] Default {model_type} model not found in available models: {default_model}"
            )

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
        self._reset_tracker()
        self.frame_times.clear()
        self.current_fps = 0.0

        # Clear frame cache when loading new video
        self.frame_cache.clear()
        self._clear_field_geometry_cache()
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        print("[MAIN_TAB] Frame cache cleared for new video")

        self._last_raw_frame = None

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
            # Pre-initialize player ID system to avoid first-frame delay
            # (Safe to call multiple times, only initializes once)
            if self.player_id_checkbox.isChecked():
                initialize_player_id_system()

            # Warmup inference models if configured
            if self.inference_checkbox.isChecked() and get_setting(
                "models.inference.warmup_on_load", True
            ):
                warmup_models()

            # Update UI
            filename = Path(video_path).name
            self.video_label.setText(f"Loaded: {filename}")  # Set progress bar range
            video_info = self.video_player.get_video_info()
            self.progress_bar.setMaximum(max(1, video_info["total_frames"] - 1))
            self.progress_bar.setValue(0)

            # Display first frame via unified pipeline to capture full per-frame timings
            # (includes Frame I/O, processing, visualization, UI display, and Total Runtime)
            self._process_and_display(mode="current")

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

        # Overlays are drawn in place on the copy; the raw frame stays clean for the
        # top-down view, which would otherwise have to decode it a second time.
        self._last_raw_frame = frame

        # Apply processing if enabled (with caching optimization)
        processed_frame = self._process_frame_cached(frame.copy())

        # Convert to Qt format and display
        ui_disp_start = time.time()
        height, width = processed_frame.shape[:2]
        bytes_per_line = 3 * width

        q_image = QImage(
            processed_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_BGR888,  # OpenCV's channel order, no swap copy
        )
        pixmap = QPixmap.fromImage(q_image)

        # Use ZoomableImageLabel's set_image method for zoom support
        self.video_label.set_image(pixmap)
        ui_disp_ms = (time.time() - ui_disp_start) * 1000
        try:
            self.performance_widget.add_processing_measurement("UI Display", ui_disp_ms)
        except Exception:
            pass

        # Update homography display if enabled
        if self.homography_enabled:
            self._update_homography_display()

    def _get_processing_cache_key(self) -> str:
        """Generate cache key based on current processing settings.

        Returns:
            Cache key string representing current processing configuration
        """
        settings = [
            str(self.inference_checkbox.isChecked()),
            str(self.tracking_checkbox.isChecked()),
            str(self.field_segmentation_checkbox.isChecked()),
            str(self.player_id_checkbox.isChecked()),
        ]
        return "|".join(settings)

    def _delayed_update_displays(self) -> None:
        """Delayed update handler for debounced UI updates."""
        if self.pending_update and self.video_player.is_loaded():
            # Process and display current frame with full timing coverage (no frame advance)
            self._process_and_display(mode="current")
            self.pending_update = False
            self.logger.debug("[MAIN_TAB] Executed debounced display update")

    def _request_display_update(self, immediate: bool = False) -> None:
        """Request a display update with debouncing to prevent excessive recomputation.

        Args:
            immediate: If True, force immediate update and set pending flag
        """
        if immediate:
            # Force immediate update but set pending flag for debounced processing
            self.pending_update = True
            if self.video_player.is_loaded():
                self._process_and_display(mode="current")
        else:
            # Standard debounced update
            self.pending_update = True
            self.update_timer.start(self.debounce_delay_ms)

    def _process_and_display(self, mode: str = "current") -> None:
        """End-to-end per-frame pipeline including Frame I/O and Total Runtime.

        Args:
            mode: "current" to use current frame without advancing; "next" to advance.
        """
        if not self.video_player.is_loaded():
            self._stop_playback()
            return

        # Begin new frame for metrics
        try:
            self.performance_widget.begin_frame()
        except Exception:
            pass

        # Total runtime timing starts before frame I/O
        total_start = time.time()

        # Frame I/O
        io_start = time.time()
        self.global_frame_index = self.video_player.current_frame_idx
        frame = (
            self.video_player.get_next_frame()
            if mode == "next"
            else self.video_player.get_current_frame()
        )
        io_ms = (time.time() - io_start) * 1000
        self.performance_widget.add_processing_measurement("Frame I/O", io_ms)

        if frame is not None:
            # Display/process frame (includes processing + homography display timings)
            self._display_frame(frame)

            # Update progress bar when advancing
            if mode == "next":
                video_info = self.video_player.get_video_info()
                self.progress_bar.setValue(video_info["current_frame"])
        else:
            # End of video reached when advancing
            if mode == "next":
                print("[MAIN_TAB] End of video reached")
                self._stop_playback()
            # No frame to process; still record totals

        # Record total runtime after full pipeline
        total_ms = (time.time() - total_start) * 1000
        self.performance_widget.add_processing_measurement("Total Runtime", total_ms)
        self._update_fps(total_ms)

    def _process_frame_cached(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with caching optimization to avoid redundant computations.

        Args:
            frame: Input frame to process

        Returns:
            Processed frame with visualizations
        """
        if not self.cache_enabled:
            processed_frame = self._process_frame(frame)
            return processed_frame

        # Video position distinguishes identical consecutive frames without hashing pixels.
        frame_hash = str(self.global_frame_index)
        processing_key = self._get_processing_cache_key()
        cache_key = f"{frame_hash}_{processing_key}"

        # Cache lookup timing
        cache_lookup_start = time.time()
        cache_hit = cache_key in self.frame_cache
        cache_lookup_ms = (time.time() - cache_lookup_start) * 1000
        self.performance_widget.add_processing_measurement("Cache - Lookup", cache_lookup_ms)

        if cache_hit:
            # Cache hit - return cached results
            cached_data = self.frame_cache[cache_key]

            # Restore processing results
            self.current_detections = cached_data["detections"]
            self.current_tracks = cached_data["tracks"]
            self.current_field_results = cached_data["field_results"]
            self.current_player_ids = cached_data["player_ids"]

            # Apply visualizations to the frame
            processed_frame = self._apply_visualizations(frame)

            self.cache_hit_count += 1
            # Visualization timing is emitted inside _apply_visualizations to avoid overlap

            # Log cache efficiency periodically
            if (self.cache_hit_count + self.cache_miss_count) % 30 == 0:
                total_requests = self.cache_hit_count + self.cache_miss_count
                hit_rate = (self.cache_hit_count / total_requests) * 100
                print(
                    f"[MAIN_TAB] Cache hit rate: {hit_rate:.1f}% ({self.cache_hit_count}/{total_requests})"
                )

            return processed_frame

        # Cache miss - perform full processing
        self.cache_miss_count += 1
        processed_frame = self._process_frame(frame)

        # Store results in cache (timed)
        cache_store_start = time.time()
        # Only redraws of the current frame can safely reuse stateful tracking/OCR results.
        self.frame_cache.clear()
        self.frame_cache[cache_key] = {
            "detections": self.current_detections.copy(),
            "tracks": self.current_tracks.copy() if self.current_tracks else [],
            "field_results": (self.current_field_results if self.current_field_results else []),
            "player_ids": self.current_player_ids.copy(),
        }
        cache_store_ms = (time.time() - cache_store_start) * 1000
        self.performance_widget.add_processing_measurement("Cache - Store", cache_store_ms)

        return processed_frame

    def _clear_field_geometry_cache(self) -> None:
        """Discard geometry derived from a previous segmentation result."""
        self._field_geometry_cache_results = None
        self._field_geometry_cache_frame_shape = None
        self._cached_unified_field_mask = None
        self._cached_ransac_lines = []
        self._cached_ransac_confidences = []
        self._cached_field_contour = None
        self._cached_ransac_fit = None

    def _get_field_geometry(
        self, field_results: List[Any], frame_shape: Tuple[int, int]
    ) -> Tuple[
        Optional[np.ndarray],
        List[Tuple[np.ndarray, np.ndarray]],
        List[float],
        Dict[str, float],
    ]:
        """Return the mask and RANSAC lines derived from a segmentation result.

        Field segmentation is intentionally sampled at an interval.  For frames
        between samples, the result object is the same, so recomputing its mask
        and RANSAC fit only adds latency without changing the displayed geometry.

        The contour and raw RANSAC fit are kept alongside (_cached_field_contour,
        _cached_ransac_fit) so the drawing code reuses this one randomized fit.
        """
        timings = {"mask_ms": 0.0, "line_extraction_ms": 0.0}
        if (
            field_results is not self._field_geometry_cache_results
            or frame_shape != self._field_geometry_cache_frame_shape
        ):
            mask_start = time.perf_counter()
            unified_mask = create_unified_field_mask(field_results, frame_shape)
            timings["mask_ms"] = (time.perf_counter() - mask_start) * 1000
            field_contour: Optional[np.ndarray] = None
            ransac_fit: Optional[tuple] = None
            if unified_mask is None:
                ransac_lines: List[Tuple[np.ndarray, np.ndarray]] = []
                ransac_confidences: List[float] = []
            else:
                line_extraction_start = time.perf_counter()
                ransac_lines, ransac_confidences, field_contour, ransac_fit = fit_lines_from_mask(
                    unified_mask
                )
                timings["line_extraction_ms"] = (time.perf_counter() - line_extraction_start) * 1000

            self._field_geometry_cache_results = field_results
            self._field_geometry_cache_frame_shape = frame_shape
            self._cached_unified_field_mask = unified_mask
            self._cached_ransac_lines = ransac_lines
            self._cached_ransac_confidences = ransac_confidences
            self._cached_field_contour = field_contour
            self._cached_ransac_fit = ransac_fit

        return (
            self._cached_unified_field_mask,
            self._cached_ransac_lines,
            self._cached_ransac_confidences,
            timings,
        )

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply enabled processing to a frame.

        Args:
            frame: Input frame to process

        Returns:
            Processed frame with visualizations
        """
        # Reset detection/tracking results (but keep existing jersey IDs for persistence)
        self.current_detections = []
        self.current_tracks = []
        self.current_field_results = []
        # Do NOT clear current_player_ids here; we may reuse previous frame's IDs when OCR is skipped

        # Run inference if enabled
        if self.inference_checkbox.isChecked():
            self.logger.debug("[MAIN_TAB] Running inference...")
            start_time = time.time()
            self.current_detections = run_inference(frame)
            duration_ms = (time.time() - start_time) * 1000
            self.performance_widget.add_processing_measurement("Inference", duration_ms)

        # Run tracking if enabled
        if self.tracking_checkbox.isChecked():
            self.logger.debug("[MAIN_TAB] Running tracking...")
            start_time = time.time()
            self.current_tracks = run_tracking(frame, self.current_detections)
            duration_ms = (time.time() - start_time) * 1000
            self.performance_widget.add_processing_measurement("Tracking", duration_ms)

        # Run field segmentation if enabled
        if self.field_segmentation_checkbox.isChecked():
            self.logger.debug("[MAIN_TAB] Running field segmentation...")
            start_time = time.time()
            self.current_field_results = run_field_segmentation(frame, self.global_frame_index)
            duration_ms = (time.time() - start_time) * 1000
            self.performance_widget.add_processing_measurement("Field Segmentation", duration_ms)
        else:
            # Clear field results when disabled
            self.current_field_results = []
            self.ransac_lines = []
            self.ransac_confidences = []
            self.all_lines_for_display = {}

        # Run player ID if enabled (requires tracking to be active)
        new_player_id_results: Dict[int, Tuple[str, Any]] = {}
        if self.player_id_checkbox.isChecked() and self.current_tracks:
            self.logger.debug(
                f"[MAIN_TAB] Running player identification on {len(self.current_tracks)} tracks..."
            )
            start_time = time.time()
            (
                new_player_id_results,
                player_id_timing,
                self.finalized_player_id_tracks,
            ) = run_player_id_on_tracks(
                frame,
                self.current_tracks,
                frame_index=self.global_frame_index,
                finalized_tracks=self.finalized_player_id_tracks,
            )
            duration_ms = (time.time() - start_time) * 1000

            # Debug timing values and results
            self.logger.debug(
                f"[MAIN_TAB] Player ID results: {len(new_player_id_results)} tracks processed"
            )
            for track_id, (jersey_number, details) in new_player_id_results.items():
                confidence = details.get("confidence", 0.0) if details else 0.0
                self.logger.debug(
                    f"[MAIN_TAB]   Track {track_id}: #{jersey_number} (conf: {confidence:.3f})"
                )
            self.logger.debug(f"[MAIN_TAB] Raw timing: {player_id_timing}")
            self.logger.debug(f"[MAIN_TAB] Total duration: {duration_ms:.1f}ms")

            # Add detailed timing measurements (only if there are actual measurements)
            if (
                player_id_timing["preprocessing_ms"] > 0
                or player_id_timing["ocr_ms"] > 0
                or player_id_timing.get("filtering_ms", 0) > 0
            ):
                self.performance_widget.add_processing_measurement(
                    "Player ID - Preprocessing", player_id_timing["preprocessing_ms"]
                )
                self.performance_widget.add_processing_measurement(
                    "Player ID - EasyOCR", player_id_timing["ocr_ms"]
                )
                if player_id_timing.get("filtering_ms", 0) > 0:
                    self.performance_widget.add_processing_measurement(
                        "Player ID - Jersey Number Filtering", player_id_timing["filtering_ms"]
                    )

            self.logger.debug(
                f"[MAIN_TAB] Identified {len(new_player_id_results)} players this frame"
            )
            from ..config.settings import get_setting as _ua_get_setting_verbose

            if _ua_get_setting_verbose(
                "player_id.verbose_debug",
                _ua_get_setting_verbose("models.player_id.verbose_debug", False),
            ):
                print(
                    f"[MAIN_TAB] Player ID timing - Preprocessing: {player_id_timing['preprocessing_ms']:.1f}ms, OCR: {player_id_timing['ocr_ms']:.1f}ms, Filtering: {player_id_timing.get('filtering_ms', 0.0):.1f}ms"
                )
        else:
            # Clear player IDs when not running
            if not self.player_id_checkbox.isChecked():
                self.current_player_ids = {}
                self.player_id_last_seen.clear()

        # Merge new OCR results into persistent map (if player ID enabled)
        if self.player_id_checkbox.isChecked() and self.current_tracks:
            # Update / add new results
            for track_id, value in new_player_id_results.items():
                self.current_player_ids[track_id] = value
                self.player_id_last_seen[track_id] = self.global_frame_index

            # For tracks not updated this frame, attempt to fill using tracker probabilities if missing or Unknown
            for track in self.current_tracks:
                track_id = getattr(track, "track_id", getattr(track, "id", None))
                if track_id is None:
                    continue
                if track_id not in self.current_player_ids or self.current_player_ids[track_id][
                    0
                ] in ("Unknown", None, ""):
                    best_num, best_prob = get_best_jersey_number(track_id)
                    if best_num and best_prob > 0.0:
                        # Create lightweight details structure
                        existing_details = (
                            self.current_player_ids.get(track_id, ("Unknown", {}))[1] or {}
                        )
                        existing_details.setdefault("best_tracked", {})
                        existing_details["best_tracked"] = {
                            "jersey_number": best_num,
                            "probability": best_prob,
                        }
                        self.current_player_ids[track_id] = (best_num, existing_details)
                        self.player_id_last_seen[track_id] = self.global_frame_index

            # Prune entries for tracks no longer present (simple approach)
            current_ids = {
                getattr(t, "track_id", getattr(t, "id", -1)) for t in self.current_tracks
            }
            stale_ids = [tid for tid in self.current_player_ids.keys() if tid not in current_ids]
            for tid in stale_ids:
                # Allow a short grace maybe? For now remove immediately to avoid clutter
                self.current_player_ids.pop(tid, None)
                self.player_id_last_seen.pop(tid, None)

        # Apply visualizations (Visualization timing emitted inside to avoid overlap)
        frame = self._apply_visualizations(frame)

        return frame

    def _apply_visualizations(self, frame):
        """Apply visualization overlays to frame with memory and performance optimization.

        Args:
            frame: Frame to add visualizations to (owned by the caller, drawn on in place)

        Returns:
            Frame with visualizations applied
        """
        # Early return for minimal processing if no overlays are enabled
        viz_enabled = (
            self.field_segmentation_checkbox.isChecked()
            or self.current_detections
            or self.current_tracks
        )

        if not viz_enabled:
            # Add only FPS overlay for minimal processing
            self._draw_fps_overlay(frame)
            return frame

        # Apply field segmentation overlay first (as background) - contour only for better performance
        _viz_start = time.time()
        _viz_excluded_ms = 0.0
        if self.current_field_results and self.field_segmentation_checkbox.isChecked():
            # Show raw segmentation model output if enabled
            from ..config.settings import get_setting

            show_raw_masks = get_setting("models.segmentation.show_raw_masks", True)
            if show_raw_masks:
                frame = draw_field_segmentation(frame, self.current_field_results)
                self.logger.debug("[MAIN_TAB] Applied raw segmentation masks to frame")

            # Create and display unified mask with RANSAC line detection
            frame_shape = frame.shape[:2]  # (height, width)
            unified_mask, detected_lines, confidences, geometry_timing = self._get_field_geometry(
                self.current_field_results, frame_shape
            )
            self.performance_widget.add_processing_measurement(
                "Mask Unification", geometry_timing["mask_ms"]
            )
            self.performance_widget.add_processing_measurement(
                "Line Extraction", geometry_timing["line_extraction_ms"]
            )
            _viz_excluded_ms += geometry_timing["mask_ms"] + geometry_timing["line_extraction_ms"]

            if unified_mask is not None:
                # Store RANSAC lines directly
                if detected_lines:
                    self.ransac_lines = detected_lines
                    self.ransac_confidences = confidences
                    self.logger.debug(
                        f"[MAIN_TAB] Using {len(self.ransac_lines)} RANSAC lines directly"
                    )
                else:
                    self.ransac_lines = []
                    self.ransac_confidences = []

                # Use the same color as segmentation for consistency
                from .visualization import get_primary_field_color

                field_color = get_primary_field_color()  # Bright cyan (BGR) - same as segmentation
                frame, _, self.all_lines_for_display = draw_unified_field_mask(
                    frame,
                    unified_mask,
                    field_color,
                    alpha=0.3,
                    fill_mask=False,
                    ransac_fit=self._cached_ransac_fit,
                    field_contour=self._cached_field_contour,
                    in_place=True,
                )
                self.logger.debug(
                    f"[MAIN_TAB] Applied field contour (no fill) to frame: {int(np.sum(unified_mask))} pixels"
                )

                # Draw raw RANSAC lines only in main view (tracking data for top-down view)
                if self.all_lines_for_display:
                    frame = draw_all_field_lines(
                        frame,
                        self.all_lines_for_display,
                        scale_factor=1.0,
                        draw_raw_lines_only=True,
                        in_place=True,
                    )
                    self.logger.debug(
                        f"[MAIN_TAB] Added raw RANSAC lines for {len(self.all_lines_for_display)} field lines"
                    )

                # Overlay RANSAC field lines on main view
                if self.ransac_lines:
                    frame = draw_ransac_field_lines(
                        frame,
                        self.ransac_lines,
                        self.ransac_confidences,
                        transformation_matrix=None,
                        scale_factor=1.0,
                        in_place=True,
                    )
                    self.logger.debug(
                        f"[MAIN_TAB] Added {len(self.ransac_lines)} RANSAC lines to main view"
                    )
            else:
                print("[MAIN_TAB] No unified mask could be created")
                self.ransac_lines = []
                self.ransac_confidences = []
                self.all_lines_for_display = {}

        # Conditional visualization based on enabled options (avoid unnecessary processing)

        # Show detections only if tracking is NOT enabled (to avoid visual clutter)
        if self.current_detections and not self.tracking_checkbox.isChecked():
            frame = draw_detections(frame, self.current_detections, in_place=True)

        # Show tracking visualization if tracking is enabled
        elif self.current_tracks and self.tracking_checkbox.isChecked():
            # Get track histories for trajectory visualization (cached in processing module)
            track_histories = get_track_histories()

            # Use player ID visualization if player ID is enabled (even if no IDs detected yet)
            if self.player_id_checkbox.isChecked():
                frame = draw_tracks_with_player_ids(
                    frame,
                    self.current_tracks,
                    track_histories,
                    self.current_player_ids,
                    in_place=True,
                )
            else:
                frame = draw_tracks(frame, self.current_tracks, track_histories, in_place=True)

        # Add FPS overlay to top right (lightweight)
        self._draw_fps_overlay(frame)

        # Add jersey tracking table overlay if player ID is enabled (conditional rendering)
        if self.player_id_checkbox.isChecked() and self.current_player_ids:
            self._draw_jersey_table_overlay(frame)

        # Emit Visualization timing excluding sub-steps measured separately
        _viz_total_ms = (time.time() - _viz_start) * 1000
        _viz_ms = max(0.0, _viz_total_ms - _viz_excluded_ms)
        self.performance_widget.add_processing_measurement("Visualization", _viz_ms)
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
        cv2.rectangle(
            frame,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            (0, 0, 0),
            -1,
        )  # Black background

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
        fps = video_info["fps"]
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
        self._process_and_display(mode="next")

    def _on_seek(self, frame_idx: int):
        """Handle seek bar movement with immediate display update."""
        if self.video_player.is_loaded():
            if not self.video_player.seek_to_frame(frame_idx):
                return
            self._reset_tracker()

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
        reset_inference_state()
        reset_segmentation_cache()
        self.frame_cache.clear()
        self._clear_field_geometry_cache()
        self.current_detections = []
        self.current_tracks = []
        self.current_field_results = []
        self.current_player_ids.clear()
        self.player_id_last_seen.clear()
        print("[MAIN_TAB] Tracker reset")
        # Clear finalized jersey numbers when tracker is reset
        self.finalized_player_id_tracks.clear()

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
                "models.segmentation.default_model", FALLBACK_DEFAULTS["model_segmentation"]
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
    def _on_player_model_changed(self, model_path: str):
        """Handle player detection model change."""
        if model_path:
            full_path = Path(get_setting("models.base_path", DEFAULT_PATHS["MODELS"])) / model_path
            from ..processing import set_player_model

            if set_player_model(str(full_path)):
                self._reset_tracker()
                self._request_display_update()
            print(f"[MAIN_TAB] Player detection model changed to: {model_path}")

    def _on_disc_model_changed(self, model_path: str):
        """Handle disc detection model change."""
        if model_path:
            full_path = Path(get_setting("models.base_path", DEFAULT_PATHS["MODELS"])) / model_path
            from ..processing import set_disc_model

            if set_disc_model(str(full_path)):
                self._reset_tracker()
                self._request_display_update()
            print(f"[MAIN_TAB] Disc detection model changed to: {model_path}")

    def _on_segmentation_model_changed(self, display_name: str):
        """Handle segmentation model selection change."""
        if self.segmentation_model_combo is None:
            return

        model_path = self.segmentation_model_combo.currentData()
        if model_path and os.path.exists(model_path):
            print(f"[MAIN_TAB] Changing segmentation model to: {model_path}")
            try:
                if set_field_model(model_path):
                    self.frame_cache.clear()
                    self._clear_field_geometry_cache()
                    print(f"[MAIN_TAB] Successfully loaded segmentation model: {display_name}")
                    # Force re-run segmentation with new model
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

    def _on_ransac_toggled(self, state: int):
        """Handle RANSAC line fitting checkbox toggle."""
        ransac_enabled = state == 2  # Qt.Checked = 2

        print(f"[MAIN_TAB] RANSAC toggle: state={state}, ransac_enabled={ransac_enabled}")

        # Temporarily override the config value in memory
        from ..config.settings import get_config

        ransac_config = get_config()
        for key in ("models", "segmentation", "contour", "ransac"):
            ransac_config = ransac_config.setdefault(key, {})
        ransac_config["enabled"] = ransac_enabled

        # Update displays if field segmentation is currently shown
        if self.show_segmentation_checkbox.isChecked():
            self._request_display_update(immediate=True)

    def _load_segmentation_models(self):
        """Load available field segmentation models using utility function."""
        from ..utils.segmentation_utils import (
            load_segmentation_models,
            populate_segmentation_model_combo,
        )

        self.available_segmentation_models = load_segmentation_models()

        # Update combo box
        if hasattr(self, "segmentation_model_combo") and self.segmentation_model_combo is not None:
            default_model_path = get_setting(
                "models.segmentation.default_model", FALLBACK_DEFAULTS["model_segmentation"]
            )
            populate_segmentation_model_combo(
                self.segmentation_model_combo,
                self.available_segmentation_models,
                default_model_path,
            )

        print(f"[MAIN_TAB] Loaded {len(self.available_segmentation_models)} segmentation models")

    def _map_tracked_objects_to_top_down(
        self, warped_frame: np.ndarray, matrix: np.ndarray, scale: float = 1.0
    ) -> np.ndarray:
        """Map tracked objects to the top-down view using their foot positions.

        Args:
            warped_frame: The homography-transformed frame (drawn on in place)
            matrix: Homography that produced warped_frame (including any display scaling)
            scale: Display scale of warped_frame, applied to marker and label sizes

        Returns:
            Frame with tracked objects mapped to top-down view
        """
        if not self.current_tracks or matrix is None:
            return warped_frame

        result_frame = warped_frame
        track_histories = get_track_histories()

        def px(value: float) -> int:
            return max(1, int(round(value * scale)))

        for track in self.current_tracks:
            # Get track properties
            track_id = getattr(track, "track_id", None)
            if track_id is None:
                continue

            # Get bounding box
            bbox = None
            if hasattr(track, "to_ltrb"):
                bbox = track.to_ltrb()
            elif hasattr(track, "to_tlbr"):
                bbox = track.to_tlbr()
            elif hasattr(track, "bbox"):
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
                transformed_foot = cv2.perspectiveTransform(foot_point, matrix)
                transformed_x = int(transformed_foot[0][0][0])
                transformed_y = int(transformed_foot[0][0][1])

                # Check if transformed position is within frame bounds
                frame_h, frame_w = warped_frame.shape[:2]
                if 0 <= transformed_x < frame_w and 0 <= transformed_y < frame_h:
                    # Generate unique color for each track ID
                    from .visualization import _get_track_color

                    color = _get_track_color(track_id)

                    # Draw foot position as a circle (larger for top-down view)
                    cv2.circle(result_frame, (transformed_x, transformed_y), px(12), color, -1)

                    # Draw track ID label with larger font for top-down view
                    label_text = f"ID:{track_id}"

                    # Add player jersey number if available
                    if track_id in self.current_player_ids:
                        jersey_number, _ = self.current_player_ids[track_id]
                        if jersey_number != "Unknown":
                            label_text = f"#{jersey_number}"

                    # Draw label background for better visibility (larger for top-down view)
                    font_scale = 1.0 * scale  # Sized for visibility in top-down view
                    font_thickness = px(3)
                    label_size = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                    )[0]
                    label_bg_x1 = transformed_x - label_size[0] // 2 - px(5)
                    label_bg_y1 = transformed_y - px(35)
                    label_bg_x2 = transformed_x + label_size[0] // 2 + px(5)
                    label_bg_y2 = transformed_y - px(5)

                    cv2.rectangle(
                        result_frame,
                        (label_bg_x1, label_bg_y1),
                        (label_bg_x2, label_bg_y2),
                        color,
                        -1,
                    )

                    # Draw label text with larger font
                    cv2.putText(
                        result_frame,
                        label_text,
                        (transformed_x - label_size[0] // 2, transformed_y - px(15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                    )

                    # Draw direction indicator if track has history
                    if track_histories and track_id in track_histories:
                        history = track_histories[track_id]
                        if len(history) >= 2:
                            # Get last two foot positions and transform them
                            prev_pos = history[-2] if len(history) > 1 else history[-1]

                            # Transform previous position
                            prev_point = np.array([[[prev_pos[0], prev_pos[1]]]], dtype=np.float32)
                            try:
                                transformed_prev = cv2.perspectiveTransform(prev_point, matrix)
                                prev_x = int(transformed_prev[0][0][0])
                                prev_y = int(transformed_prev[0][0][1])

                                # Draw direction arrow
                                if 0 <= prev_x < frame_w and 0 <= prev_y < frame_h:
                                    # Calculate direction vector
                                    dx = transformed_x - prev_x
                                    dy = transformed_y - prev_y
                                    length = (dx * dx + dy * dy) ** 0.5

                                    if length > 5 * scale:  # Only draw if significant movement
                                        # Normalize and scale
                                        dx = int(dx / length * 15 * scale)
                                        dy = int(dy / length * 15 * scale)

                                        # Draw arrow
                                        arrow_end_x = transformed_x + dx
                                        arrow_end_y = transformed_y + dy
                                        cv2.arrowedLine(
                                            result_frame,
                                            (transformed_x, transformed_y),
                                            (arrow_end_x, arrow_end_y),
                                            color,
                                            px(2),
                                            tipLength=0.3,
                                        )
                            except Exception:
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

        self.logger.debug(
            f"[MAIN_TAB] Canvas size: {input_width}x{input_height} -> {output_width}x{output_height} (aspect {aspect_ratio:.1f}:1, area: {input_width * input_height} -> {output_width * output_height})"
        )
        return output_width, output_height

    def _update_homography_display(self):
        """Update the homography display with the displayed frame and transformation."""
        if not self.homography_enabled or self.homography_display_label is None:
            return

        homography_start_time = time.time()
        homography_calc_duration_ms = 0.0

        try:
            # Get current frame
            if self.video_player.is_loaded():
                # Reuse the frame the main view is showing. Reading it again from the
                # decoder costs a decode plus a backwards seek, and after playback has
                # advanced it returns the following frame instead.
                frame = self._last_raw_frame
                if frame is None:
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

                    # The panel is far smaller than the full canvas, so render it at a
                    # reduced scale; warp cost is proportional to the output pixel count.
                    display_scale = float(get_setting("homography.display_scale", 0.5))
                    display_scale = min(1.0, max(0.1, display_scale))
                    display_matrix = self.homography_matrix
                    if display_scale != 1.0:
                        output_width = max(1, int(output_width * display_scale))
                        output_height = max(1, int(output_height * display_scale))
                        display_matrix = (
                            np.diag([display_scale, display_scale, 1.0]) @ self.homography_matrix
                        )

                    # Map the full frame to top-down view
                    warped_frame = cv2.warpPerspective(
                        frame, display_matrix, (output_width, output_height)
                    )

                    homography_calc_duration_ms = (time.time() - homography_calc_start) * 1000
                    self.performance_widget.add_processing_measurement(
                        "Homography Calculation", homography_calc_duration_ms
                    )

                    # Apply field segmentation to warped frame if available
                    if self.current_field_results and self.field_segmentation_checkbox.isChecked():
                        try:
                            # Transform the segmentation masks to match the warped frame
                            original_frame_shape = frame.shape[:2]  # (height, width)
                            # Normally a cache hit: the main view derived this geometry already
                            self._get_field_geometry(self.current_field_results, original_frame_shape)
                            warped_frame_with_segmentation = apply_segmentation_to_warped_frame(
                                warped_frame,
                                self.current_field_results,
                                display_matrix,
                                original_frame_shape,
                                "MAIN_TAB",
                                field_contour=self._cached_field_contour,
                                draw_scale=display_scale,
                                in_place=True,
                            )
                            if warped_frame_with_segmentation is not None:
                                warped_frame = warped_frame_with_segmentation
                                self.logger.debug(
                                    f"[MAIN_TAB] Applied transformed segmentation to homography view: {len(self.current_field_results)} results"
                                )
                            else:
                                print(
                                    "[MAIN_TAB] Failed to apply transformed segmentation to homography view"
                                )
                        except Exception as e:
                            print(f"[MAIN_TAB] Error applying segmentation to homography view: {e}")

                    # Map tracked objects to top-down view if tracking is enabled
                    if self.tracking_checkbox.isChecked() and self.current_tracks:
                        warped_frame = self._map_tracked_objects_to_top_down(
                            warped_frame, display_matrix, display_scale
                        )
                        self.logger.debug(
                            f"[MAIN_TAB] Mapped {len(self.current_tracks)} tracked objects to top-down view"
                        )

                    # Add RANSAC field lines to top-down view
                    if self.ransac_lines:
                        # Use RANSAC lines with confidence display for top-down view
                        warped_frame = draw_ransac_field_lines(
                            warped_frame,
                            self.ransac_lines,
                            self.ransac_confidences,
                            display_matrix,
                            scale_factor=2.0 * display_scale,
                            show_confidence=True,
                            in_place=True,
                        )
                        self.logger.debug(
                            f"[MAIN_TAB] Added RANSAC field lines to top-down view: {len(self.ransac_lines)} lines"
                        )
                    elif self.all_lines_for_display:
                        # Fallback to classified lines if no RANSAC lines available
                        warped_frame = draw_all_field_lines(
                            warped_frame,
                            self.all_lines_for_display,
                            display_matrix,
                            scale_factor=2.0 * display_scale,
                            draw_raw_lines_only=False,
                            in_place=True,
                        )
                        print(
                            f"[MAIN_TAB] Added classified field lines to top-down view (fallback): {len(self.all_lines_for_display)} lines"
                        )

                    # Convert warped frame to Qt format and display (preserve aspect ratio)
                    qt_convert_start = time.time()
                    warped_height, warped_width = warped_frame.shape[:2]
                    bytes_per_line = 3 * warped_width

                    # Ensure the frame is contiguous for QImage
                    if not warped_frame.flags["C_CONTIGUOUS"]:
                        warped_frame = np.ascontiguousarray(warped_frame)

                    q_image = QImage(
                        warped_frame.data,
                        warped_width,
                        warped_height,
                        bytes_per_line,
                        QImage.Format_BGR888,  # OpenCV's channel order, no swap copy
                    )

                    # Display with preserved aspect ratio using ZoomableImageLabel
                    pixmap = QPixmap.fromImage(q_image)
                    qt_convert_ms = (time.time() - qt_convert_start) * 1000
                    self.performance_widget.add_processing_measurement(
                        "Homography Qt Conversion", qt_convert_ms
                    )

                    # Set image timing
                    qt_display_start = time.time()
                    self.homography_display_label.set_image(pixmap)
                    qt_display_ms = (time.time() - qt_display_start) * 1000
                    self.performance_widget.add_processing_measurement(
                        "Homography Qt Display", qt_display_ms
                    )

                    # Record total homography processing time (excluding sub-components)
                    homography_duration_ms = (time.time() - homography_start_time) * 1000
                    # Exclude already-measured times to avoid double counting
                    homography_other_ms = max(
                        0.0,
                        homography_duration_ms
                        - homography_calc_duration_ms
                        - qt_convert_ms
                        - qt_display_ms,
                    )
                    if homography_other_ms > 1.0:  # Only report if significant
                        self.performance_widget.add_processing_measurement(
                            "Homography Other", homography_other_ms
                        )

                    self.logger.debug(
                        f"[MAIN_TAB] Updated homography display ({warped_width}x{warped_height}px) - Calc: {homography_calc_duration_ms:.1f}ms, Qt: {qt_convert_ms + qt_display_ms:.1f}ms"
                    )
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

                    tracked_data.append(
                        {
                            "track_id": track_id,
                            "best_jersey": best_jersey,
                            "best_prob": best_prob,
                            "second_jersey": second_jersey,
                            "second_prob": second_prob,
                        }
                    )

            if not tracked_data:
                return

            # Sort by track ID
            tracked_data.sort(key=lambda x: x["track_id"])

            # Overlay position (top-left corner with margin, lowered slightly)
            start_x = 20
            start_y = 80  # Lowered from 30 to 80
            line_height = 30  # Increased to accommodate two lines per track

            # Draw semi-transparent background
            table_height = len(tracked_data) * line_height + 40
            table_width = 250  # Increased width for second jersey

            # Darken only the table area; blending a copy of the whole frame costs far more
            background = frame[
                start_y - 20 : start_y + table_height - 19, start_x - 10 : start_x + table_width + 1
            ]
            cv2.convertScaleAbs(background, dst=background, alpha=0.7)

            # Draw header
            cv2.putText(
                frame,
                "Jersey Tracking",
                (start_x, start_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Draw table entries
            y_offset = start_y + 25
            for data in tracked_data:
                # Determine color based on best confidence
                if data["best_prob"] >= 0.7:
                    color = (0, 255, 0)  # Green
                elif data["best_prob"] >= 0.4:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 0, 255)  # Red

                # Draw best jersey number (primary line)
                best_text = (
                    f"Track {data['track_id']}: #{data['best_jersey']} ({data['best_prob']:.2f})"
                )
                cv2.putText(
                    frame, best_text, (start_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

                # Draw second jersey number (secondary line) if available
                if (
                    data["second_jersey"] and data["second_prob"] > 0.1
                ):  # Only show if reasonable confidence
                    second_color = (128, 128, 128)  # Gray for secondary
                    second_text = f"   Alt: #{data['second_jersey']} ({data['second_prob']:.2f})"
                    cv2.putText(
                        frame,
                        second_text,
                        (start_x, y_offset + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        second_color,
                        1,
                    )

                y_offset += line_height

        except Exception as e:
            print(f"[MAIN_TAB] Error drawing jersey overlay: {e}")

    def closeEvent(self, event):
        """Handle widget close event."""
        self._stop_playback()
        self.video_player.close_video()
        super().closeEvent(event)
