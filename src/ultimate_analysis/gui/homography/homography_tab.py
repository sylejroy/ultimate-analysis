"""Homography estimation tab for Ultimate Analysis GUI.

This module provides an interactive interface for adjusting homography parameters
with real-time perspective transformation visualization and YAML save/load functionality.
"""

import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...config.settings import get_setting
from ...processing.field_analysis import (
    create_unified_field_mask,
    extract_raw_lines_from_segmentation,
)
from ...processing.field_segmentation import run_field_segmentation, set_field_model
from ...processing.homography import (
    IDENTITY_PARAMETERS,
    PARAMETER_NAMES,
    default_parameters_file,
    load_parameters,
    output_canvas_size,
    parameter_range,
    parameters_to_matrix,
    save_parameters,
)
from ...processing.model_lock import MODEL_LOCK
from ...rendering.field import draw_unified_field_mask, get_primary_field_color
from ...rendering.field_lines import draw_ransac_field_lines
from ...rendering.top_down import apply_segmentation_to_warped_frame
from ...utils.logger import get_logger
from ...utils.model_files import default_model_path
from ...utils.video import VideoPlayer
from ..widgets.images import frame_to_pixmap
from ..widgets.model_selection import populate_segmentation_model_combo
from ..widgets.video_list import VideoListWidget
from ..widgets.zoomable_image_label import ZoomableImageLabel
from .fitness_chart import FitnessChart
from .runtime_dialog import RuntimeDialog

logger = get_logger("HOMOGRAPHY")


class HomographyTab(QWidget):
    """Interactive homography estimation tab with real-time transformation preview."""

    def __init__(self):
        super().__init__()

        # Video player and state
        self.video_player = VideoPlayer()
        self.video_files: List[str] = []
        self.current_video_index: int = 0
        self.current_frame: Optional[np.ndarray] = None

        # Homography parameters (H[2,2] = 1.0 fixed)
        # Default is identity matrix except for the bottom row scaling
        self.homography_params = dict(IDENTITY_PARAMETERS)

        # UI components
        self.video_list: Optional[QListWidget] = None
        self.frame_label: Optional[QLabel] = None
        self.original_display: Optional[ZoomableImageLabel] = None
        self.warped_display: Optional[ZoomableImageLabel] = None
        self.param_sliders: Dict[str, QSlider] = {}
        self.param_labels: Dict[str, QLabel] = {}
        self.param_inputs: Dict[str, QLineEdit] = {}  # For direct text input
        # Scrubbing controls
        self.scrubbing_slider: Optional[QSlider] = None
        self.scrubbing_frame_label: Optional[QLabel] = None
        self.current_video_label: Optional[QLabel] = None

        # Zoom functionality
        self.original_scroll_area: Optional[QScrollArea] = None
        self.warped_scroll_area: Optional[QScrollArea] = None

        # Field segmentation state
        self.show_segmentation = True  # Auto-enable for GA optimization
        self.current_segmentation_results = None
        self.segmentation_model_combo: Optional[QComboBox] = None
        # Auto-enable RANSAC line fitting for GA optimization
        from ...config.settings import get_config

        config = get_config()
        if "models" not in config:
            config["models"] = {}
        if "segmentation" not in config["models"]:
            config["models"]["segmentation"] = {}
        if "contour" not in config["models"]["segmentation"]:
            config["models"]["segmentation"]["contour"] = {}
        if "ransac" not in config["models"]["segmentation"]["contour"]:
            config["models"]["segmentation"]["contour"]["ransac"] = {}
        config["models"]["segmentation"]["contour"]["ransac"]["enabled"] = True
        self.ransac_lines: List[
            Tuple[np.ndarray, np.ndarray]
        ] = []  # Store RANSAC-calculated field lines
        self.ransac_confidences: List[float] = []  # Store RANSAC line confidences
        self.all_lines_for_display: Dict[
            str, Tuple[np.ndarray, float, bool]
        ] = {}  # Store all lines for display

        # Runtime performance tracking
        self.runtime_dialog = RuntimeDialog(self)
        self.runtime_button: Optional[QPushButton] = None

        # Lazy loading flags
        self._videos_loaded = False
        self._segmentation_models_loaded = False

        # Genetic algorithm state
        self.ga_optimizer = None
        self.ga_running = False
        self.ga_continuous_timer: Optional[QTimer] = None  # Timer for continuous evolution
        self.ga_generation_history = []
        self.ga_fitness_history = []

        # GA UI components
        self.ga_start_button: Optional[QPushButton] = None
        self.ga_next_gen_button: Optional[QPushButton] = None
        self.ga_multi_gen_button: Optional[QPushButton] = None
        self.ga_reset_button: Optional[QPushButton] = None
        self.ga_apply_button: Optional[QPushButton] = None
        self.ga_continuous_button: Optional[QPushButton] = None
        self.ga_stop_button: Optional[QPushButton] = None
        self.ga_generation_label: Optional[QLabel] = None
        self.ga_fitness_label: Optional[QLabel] = None
        self.ga_population_label: Optional[QLabel] = None

        self.ga_fitness_chart = FitnessChart()

        # Initialize UI only
        self._init_ui()

        logger.debug("Loading segmentation models...")
        # Load segmentation models list (just file discovery, no model loading)
        self._load_segmentation_models()

        logger.debug("Loading default parameters...")
        # Load default homography parameters from config
        self._load_default_parameters()

        logger.info("Tab initialization complete")

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()

        # Main content with splitter (top part)
        content_splitter = QSplitter(Qt.Horizontal)

        # Left panel: Video list and parameter controls (excluding GA)
        left_panel = self._create_left_panel()
        content_splitter.addWidget(left_panel)

        # Right panel: Side-by-side video displays
        right_panel = self._create_right_panel()
        content_splitter.addWidget(right_panel)

        # Set splitter proportions (25% left, 75% right for larger image display)
        content_splitter.setSizes([300, 1200])

        main_layout.addWidget(content_splitter)

        # GA controls below images for better chart visibility
        ga_panel = self._create_ga_panel()
        main_layout.addWidget(ga_panel)

        self.setLayout(main_layout)

    def showEvent(self, event):
        """Override showEvent to implement lazy loading when tab becomes visible."""
        super().showEvent(event)

        # Lazy load content when tab is first shown
        if not self._videos_loaded:
            logger.debug("Lazy loading videos...")
            self._reload_videos()
            self._videos_loaded = True

        if not self._segmentation_models_loaded:
            logger.debug("Lazy loading segmentation models...")
            self._load_segmentation_models()
            self._segmentation_models_loaded = True

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with video list and parameter controls."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Video selection section
        video_group = QGroupBox("Video Selection")
        video_layout = QVBoxLayout()

        # Video list header
        list_header = QHBoxLayout()
        list_header.addWidget(QLabel("Videos"))

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._reload_videos)
        refresh_button.setToolTip("Refresh video list")
        list_header.addWidget(refresh_button)

        video_layout.addLayout(list_header)

        # Video list widget
        self.video_list = VideoListWidget(show_duration=False)
        self.video_list.setMinimumHeight(150)
        self.video_list.currentRowChanged.connect(self._on_video_selection_changed)
        video_layout.addWidget(self.video_list)

        # Frame navigation
        frame_nav_layout = QHBoxLayout()
        frame_nav_layout.addWidget(QLabel("Frame:"))

        self.frame_label = QLabel("0 / 0")
        frame_nav_layout.addWidget(self.frame_label)
        frame_nav_layout.addStretch()

        video_layout.addLayout(frame_nav_layout)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # Homography parameters section
        params_group = QGroupBox("Homography Parameters")
        params_layout = QVBoxLayout()

        # Create parameter sliders
        self._create_parameter_controls(params_layout)

        # Control buttons
        button_layout = QHBoxLayout()

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._reset_homography)
        reset_button.setToolTip("Reset to identity matrix")
        button_layout.addWidget(reset_button)

        save_button = QPushButton("Save Params")
        save_button.clicked.connect(self._save_parameters)
        save_button.setToolTip("Save parameters to YAML file")
        button_layout.addWidget(save_button)

        load_button = QPushButton("Load Params")
        load_button.clicked.connect(self._load_parameters)
        load_button.setToolTip("Load parameters from YAML file")
        button_layout.addWidget(load_button)

        # Second row of buttons
        button_layout2 = QHBoxLayout()

        save_default_button = QPushButton("Save as Default")
        save_default_button.clicked.connect(self._save_as_default_parameters)
        save_default_button.setToolTip("Save current parameters as default startup values")
        save_default_button.setStyleSheet(
            "background-color: #2c5aa0; color: white; font-weight: bold;"
        )
        button_layout2.addWidget(save_default_button)

        load_default_button = QPushButton("Load Default")
        load_default_button.clicked.connect(self._load_default_parameters)
        load_default_button.setToolTip("Load default parameters from config")
        button_layout2.addWidget(load_default_button)

        params_layout.addLayout(button_layout)
        params_layout.addLayout(button_layout2)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Field Segmentation Controls (Auto-enabled for GA optimization)
        segmentation_group = QGroupBox("Field Segmentation")
        segmentation_layout = QVBoxLayout()

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
        # Runtime performance button
        runtime_group = QGroupBox("Performance Monitoring")
        runtime_layout = QVBoxLayout()

        self.runtime_button = QPushButton("Show Runtime Performance")
        self.runtime_button.setMinimumHeight(40)
        self.runtime_button.setStyleSheet(
            """
            QPushButton {
                background-color: #2c5aa0;
                color: white;
                font-weight: bold;
                border: 2px solid #1e3f73;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #3a6bb5;
            }
            QPushButton:pressed {
                background-color: #1e3f73;
            }
        """
        )
        self.runtime_button.clicked.connect(self._show_runtime_dialog)
        runtime_layout.addWidget(self.runtime_button)

        runtime_group.setLayout(runtime_layout)
        layout.addWidget(runtime_group)

        # Add stretch to push everything to top
        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def _create_ga_panel(self) -> QWidget:
        """Create the genetic algorithm optimization panel below images."""
        panel = QWidget()
        layout = QHBoxLayout()

        # Genetic Algorithm Optimization Panel
        ga_group = QGroupBox("Genetic Algorithm Optimization")
        ga_layout = QVBoxLayout()

        # Info label
        ga_info = QLabel("Optimize homography matrix using genetic algorithm")
        ga_info.setWordWrap(True)
        ga_info.setStyleSheet("color: #ccc; font-size: 11px; margin: 5px;")
        ga_layout.addWidget(ga_info)

        # Control buttons (Row 1)
        ga_buttons1 = QHBoxLayout()

        self.ga_start_button = QPushButton("Start GA")
        self.ga_start_button.setToolTip("Initialize genetic algorithm with current parameters")
        self.ga_start_button.clicked.connect(self._start_genetic_algorithm)
        ga_buttons1.addWidget(self.ga_start_button)

        self.ga_next_gen_button = QPushButton("Next Gen")
        self.ga_next_gen_button.setToolTip("Proceed to next generation")
        self.ga_next_gen_button.clicked.connect(self._evolve_ga_next_generation)
        self.ga_next_gen_button.setEnabled(False)
        ga_buttons1.addWidget(self.ga_next_gen_button)

        ga_layout.addLayout(ga_buttons1)

        # Control buttons (Row 2)
        ga_buttons2 = QHBoxLayout()

        self.ga_multi_gen_button = QPushButton("Skip 10 Gens")
        self.ga_multi_gen_button.setToolTip("Proceed by 10 generations")
        self.ga_multi_gen_button.clicked.connect(lambda: self._evolve_ga_generations(10))
        self.ga_multi_gen_button.setEnabled(False)
        ga_buttons2.addWidget(self.ga_multi_gen_button)

        self.ga_reset_button = QPushButton("Reset GA")
        self.ga_reset_button.setToolTip("Reset genetic algorithm")
        self.ga_reset_button.clicked.connect(self._reset_genetic_algorithm)
        self.ga_reset_button.setEnabled(False)
        ga_buttons2.addWidget(self.ga_reset_button)

        ga_layout.addLayout(ga_buttons2)

        # Continuous evolution controls (Row 3)
        ga_buttons3 = QHBoxLayout()

        self.ga_continuous_button = QPushButton("Evolve Continuously")
        self.ga_continuous_button.setToolTip(
            "Start continuous evolution - runs generations automatically"
        )
        self.ga_continuous_button.clicked.connect(self._start_continuous_evolution)
        self.ga_continuous_button.setEnabled(False)
        self.ga_continuous_button.setStyleSheet(
            """
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                border: 2px solid #1e7e34;
                border-radius: 3px;
                padding: 6px;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #34ce57;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
                border-color: #666;
            }
        """
        )
        ga_buttons3.addWidget(self.ga_continuous_button)

        self.ga_stop_button = QPushButton("Stop Evolution")
        self.ga_stop_button.setToolTip("Stop continuous evolution")
        self.ga_stop_button.clicked.connect(self._stop_continuous_evolution)
        self.ga_stop_button.setEnabled(False)
        self.ga_stop_button.setStyleSheet(
            """
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-weight: bold;
                border: 2px solid #c82333;
                border-radius: 3px;
                padding: 6px;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #e94966;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
                border-color: #666;
            }
        """
        )
        ga_buttons3.addWidget(self.ga_stop_button)

        ga_layout.addLayout(ga_buttons3)

        # Apply best button
        self.ga_apply_button = QPushButton("Apply Best Parameters")
        self.ga_apply_button.setToolTip("Apply best parameters found so far")
        self.ga_apply_button.clicked.connect(self._apply_ga_best_parameters)
        self.ga_apply_button.setEnabled(False)
        self.ga_apply_button.setStyleSheet(
            """
            QPushButton {
                background-color: #2c5aa0;
                color: white;
                font-weight: bold;
                border: 2px solid #1e3f73;
                border-radius: 3px;
                padding: 6px;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #3a6bb5;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
                border-color: #666;
            }
        """
        )
        ga_layout.addWidget(self.ga_apply_button)

        # Status display
        ga_status = QFormLayout()
        self.ga_generation_label = QLabel("0")
        self.ga_generation_label.setStyleSheet("font-family: monospace; color: #fff;")
        ga_status.addRow("Generation:", self.ga_generation_label)

        self.ga_fitness_label = QLabel("0.000")
        self.ga_fitness_label.setStyleSheet("font-family: monospace; color: #fff;")
        ga_status.addRow("Best Fitness:", self.ga_fitness_label)

        self.ga_population_label = QLabel("20")
        self.ga_population_label.setStyleSheet("font-family: monospace; color: #fff;")
        ga_status.addRow("Population:", self.ga_population_label)

        ga_layout.addLayout(ga_status)

        ga_group.setLayout(ga_layout)
        layout.addWidget(ga_group)

        # Fitness progress chart (if available) - now has more space
        if self.ga_fitness_chart.view is not None:
            chart_group = QGroupBox("Fitness Evolution")
            chart_layout = QVBoxLayout()
            self.ga_fitness_chart.view.setMinimumHeight(200)
            self.ga_fitness_chart.view.setMaximumHeight(300)
            chart_layout.addWidget(self.ga_fitness_chart.view)
            chart_group.setLayout(chart_layout)
            layout.addWidget(chart_group)
        else:
            chart_unavailable = QLabel("Fitness chart unavailable\n(PyQt5.QtChart not installed)")
            chart_unavailable.setAlignment(Qt.AlignCenter)
            chart_unavailable.setStyleSheet("color: #888; font-size: 10px; margin: 10px;")
            layout.addWidget(chart_unavailable)

        panel.setLayout(layout)
        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with side-by-side zoomable displays."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Display header
        header = QLabel("Homography Transformation Comparison (Mouse wheel to zoom)")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(header)

        # Create horizontal layout for side-by-side image displays
        images_layout = QHBoxLayout()

        # Original frame display with scroll area and scrubbing controls
        original_group = QGroupBox("Original Frame")
        original_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        original_layout = QVBoxLayout()

        self.original_scroll_area = QScrollArea()
        self.original_scroll_area.setWidgetResizable(True)
        self.original_scroll_area.setAlignment(Qt.AlignCenter)
        self.original_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.original_display = ZoomableImageLabel()
        self.original_display.setText("No video selected")
        self.original_display.setStyleSheet(
            """
            QLabel {
                border: 2px solid #555;
                background-color: #1a1a1a;
                color: #999;
                font-size: 12px;
            }
        """
        )

        self.original_scroll_area.setWidget(self.original_display)
        original_layout.addWidget(self.original_scroll_area)

        # Add video scrubbing controls under the original frame
        scrubbing_panel = self._create_scrubbing_panel()
        original_layout.addWidget(scrubbing_panel)

        original_group.setLayout(original_layout)
        images_layout.addWidget(original_group)

        # Warped frame display with scroll area
        warped_group = QGroupBox("Warped Frame (3:1 aspect ratio)")
        warped_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        warped_layout = QVBoxLayout()

        self.warped_scroll_area = QScrollArea()
        self.warped_scroll_area.setWidgetResizable(True)
        self.warped_scroll_area.setAlignment(Qt.AlignCenter)
        self.warped_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.warped_display = ZoomableImageLabel()
        self.warped_display.setText("No video selected")
        self.warped_display.setStyleSheet(
            """
            QLabel {
                border: 2px solid #555;
                background-color: #1a1a1a;
                color: #999;
                font-size: 12px;
            }
        """
        )

        self.warped_scroll_area.setWidget(self.warped_display)
        warped_layout.addWidget(self.warped_scroll_area)
        warped_group.setLayout(warped_layout)
        images_layout.addWidget(warped_group)

        # Add the side-by-side images layout to main layout
        layout.addLayout(images_layout)

        # Reset zoom buttons
        zoom_layout = QHBoxLayout()

        reset_original_btn = QPushButton("Reset Original Zoom")
        reset_original_btn.clicked.connect(lambda: self.original_display.set_zoom(1.0))
        zoom_layout.addWidget(reset_original_btn)

        reset_warped_btn = QPushButton("Reset Warped Zoom")
        reset_warped_btn.clicked.connect(lambda: self.warped_display.set_zoom(1.0))
        zoom_layout.addWidget(reset_warped_btn)

        fit_to_window_btn = QPushButton("Fit to Window")
        fit_to_window_btn.clicked.connect(self._fit_images_to_window)
        zoom_layout.addWidget(fit_to_window_btn)

        reset_both_btn = QPushButton("Reset Both Zoom")
        reset_both_btn.clicked.connect(self._reset_all_zoom)
        zoom_layout.addWidget(reset_both_btn)

        layout.addLayout(zoom_layout)

        # Grid overlay controls
        grid_layout = QHBoxLayout()

        # Grid toggle checkbox
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(True)  # Grid enabled by default
        self.grid_checkbox.toggled.connect(self._toggle_grid)
        grid_layout.addWidget(self.grid_checkbox)

        # Grid spacing control
        grid_layout.addWidget(QLabel("Spacing:"))
        self.grid_spacing_spinbox = QSpinBox()
        self.grid_spacing_spinbox.setMinimum(10)
        self.grid_spacing_spinbox.setMaximum(200)
        self.grid_spacing_spinbox.setValue(50)
        self.grid_spacing_spinbox.setSuffix(" px")
        self.grid_spacing_spinbox.valueChanged.connect(self._update_grid_spacing)
        grid_layout.addWidget(self.grid_spacing_spinbox)

        grid_layout.addStretch()  # Push controls to the left

        layout.addLayout(grid_layout)

        panel.setLayout(layout)
        return panel

    def _create_scrubbing_panel(self) -> QWidget:
        """Create the video scrubbing controls panel."""
        panel = QWidget()
        panel.setMaximumHeight(60)  # Compact size for under original frame
        panel.setMinimumHeight(60)
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(2)

        # Video info and controls
        info_layout = QHBoxLayout()
        info_layout.setSpacing(10)

        # Current video name
        self.current_video_label = QLabel("No video loaded")
        self.current_video_label.setStyleSheet("font-weight: bold; color: #fff; font-size: 11px;")
        info_layout.addWidget(self.current_video_label)

        info_layout.addStretch()

        # Frame info
        self.scrubbing_frame_label = QLabel("0 / 0")
        self.scrubbing_frame_label.setStyleSheet(
            "color: #ccc; font-family: monospace; font-size: 10px;"
        )
        info_layout.addWidget(self.scrubbing_frame_label)

        layout.addLayout(info_layout)

        # Compact scrubbing slider
        self.scrubbing_slider = QSlider(Qt.Horizontal)
        self.scrubbing_slider.setMinimum(0)
        self.scrubbing_slider.setMaximum(0)
        self.scrubbing_slider.setValue(0)
        self.scrubbing_slider.setMinimumHeight(20)  # Smaller for compact layout
        self.scrubbing_slider.valueChanged.connect(self._on_scrubbing_changed)
        self.scrubbing_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 8px;
                background: #2a2a2a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 2px solid #005a9e;
                width: 20px;
                height: 20px;
                border-radius: 10px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #106ebe;
            }
        """
        )
        layout.addWidget(self.scrubbing_slider)

        panel.setLayout(layout)
        return panel

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
        logger.info(f"Loading video: {video_path}")

        # Load video
        if self.video_player.load_video(video_path):
            # Update UI
            video_info = self.video_player.get_video_info()
            total_frames = video_info["total_frames"]
            self.frame_label.setText(f"0 / {total_frames}")

            # Update scrubbing controls
            if hasattr(self, "scrubbing_slider"):
                self.scrubbing_slider.setMaximum(max(1, total_frames - 1))
                self.scrubbing_slider.setValue(0)
            if hasattr(self, "scrubbing_frame_label"):
                self.scrubbing_frame_label.setText(f"Frame: 0 / {total_frames}")
            if hasattr(self, "current_video_label"):
                video_name = os.path.basename(video_path)
                self.current_video_label.setText(video_name)

            # Display first frame
            first_frame = self.video_player.get_current_frame()
            if first_frame is not None:
                self.current_frame = first_frame.copy()
                # Update displays WITHOUT running segmentation initially (for fast loading)
                self._update_displays_without_segmentation()

    def _update_displays_without_segmentation(self):
        """Update displays without running segmentation - for fast initial loading."""
        if self.current_frame is None:
            return

        # Display original frame without segmentation overlay
        self._display_frame(self.current_frame, self.original_display)

        # Create and display warped frame without segmentation overlay
        warped_frame = self._apply_homography(self.current_frame)
        self._display_frame(warped_frame, self.warped_display)

    def _on_frame_changed(self, frame_idx: int):
        """Handle frame slider change."""
        if self.video_player.is_loaded():
            self.video_player.seek_to_frame(frame_idx)

            # Update frame label
            video_info = self.video_player.get_video_info()
            total_frames = video_info["total_frames"]
            self.frame_label.setText(f"{frame_idx} / {total_frames}")

            # Sync scrubbing slider
            if hasattr(self, "scrubbing_slider"):
                self.scrubbing_slider.blockSignals(True)
                self.scrubbing_slider.setValue(frame_idx)
                self.scrubbing_slider.blockSignals(False)
                # Update scrubbing frame label
                if hasattr(self, "scrubbing_frame_label"):
                    self.scrubbing_frame_label.setText(f"Frame: {frame_idx} / {total_frames}")

            # Get and display current frame
            frame = self.video_player.get_current_frame()
            if frame is not None:
                self.current_frame = frame.copy()

                # Run segmentation on new frame if enabled
                if self.show_segmentation:
                    self._run_segmentation_on_current_frame()

                self._update_displays()

    def _on_scrubbing_changed(self, frame_idx: int):
        """Handle scrubbing slider change."""
        if self.video_player.is_loaded():
            # Update via main frame change handler
            self._on_frame_changed(frame_idx)

    def _update_displays(self):
        """Update both original and warped frame displays."""
        if self.current_frame is None:
            return

        update_start = time.perf_counter()
        # Apply segmentation overlay to original frame if enabled
        original_frame = self.current_frame.copy()
        if self.show_segmentation and self.current_segmentation_results:
            # Create and display unified mask on original frame
            morphological_start = time.perf_counter()
            frame_shape = self.current_frame.shape[:2]  # (height, width)
            unified_mask = create_unified_field_mask(self.current_segmentation_results, frame_shape)
            morphological_duration = (time.perf_counter() - morphological_start) * 1000
            self.runtime_dialog.add_measurement("Morphological Ops", morphological_duration)

            if unified_mask is not None:
                # Use same color as segmentation visualization for consistency
                field_color = get_primary_field_color()  # Cyan (BGR)

                # Time the line extraction and tracking steps
                extraction_start = time.perf_counter()

                # Extract raw RANSAC lines for direct use
                detected_lines, confidences = extract_raw_lines_from_segmentation(
                    self.current_segmentation_results, frame_shape
                )

                extraction_duration = (time.perf_counter() - extraction_start) * 1000
                self.runtime_dialog.add_measurement("Line Extraction", extraction_duration)

                # Store RANSAC lines directly
                if detected_lines:
                    self.ransac_lines = detected_lines
                    self.ransac_confidences = confidences
                    logger.debug(f"Using {len(self.ransac_lines)} RANSAC lines directly")
                else:
                    self.ransac_lines = []
                    self.ransac_confidences = []

                # Draw unified mask for visualization (lightweight version without RANSAC re-computation)
                original_frame, _, self.all_lines_for_display = draw_unified_field_mask(
                    original_frame, unified_mask, field_color, alpha=0.4, fill_mask=False
                )

                logger.debug(
                    f"Applied field contour (no fill) to original frame: {np.sum(unified_mask)} pixels"
                )
            else:
                logger.debug("No unified mask could be created for original frame")
                self.ransac_lines = []
                self.ransac_confidences = []
                self.all_lines_for_display = {}
        elif self.show_segmentation:
            logger.warning("Segmentation enabled but no results available")

        # Display original frame (with optional segmentation overlay)
        self._display_frame(original_frame, self.original_display)

        # Create warped frame
        homography_start = time.perf_counter()
        warped_frame = self._apply_homography(self.current_frame)
        homography_duration = (time.perf_counter() - homography_start) * 1000
        self.runtime_dialog.add_measurement("Homography Calculation", homography_duration)

        logger.debug(f"Warped frame shape: {warped_frame.shape}, dtype: {warped_frame.dtype}")

        # For warped view, apply segmentation overlay if enabled
        if self.show_segmentation and self.current_segmentation_results:
            try:
                # Transform the segmentation masks to match the warped frame
                original_frame_shape = self.current_frame.shape[:2]  # (height, width)
                h_matrix = parameters_to_matrix(self.homography_params)
                warped_frame_with_segmentation = apply_segmentation_to_warped_frame(
                    warped_frame,
                    self.current_segmentation_results,
                    h_matrix,
                    original_frame_shape,
                    "HOMOGRAPHY",
                )
                if warped_frame_with_segmentation is not None:
                    warped_frame = warped_frame_with_segmentation
                    logger.debug(
                        f"Applied transformed segmentation to warped frame: {len(self.current_segmentation_results)} results"
                    )

                # Add RANSAC lines to the warped frame if available
                if self.ransac_lines:
                    warped_frame = draw_ransac_field_lines(
                        warped_frame,
                        self.ransac_lines,
                        self.ransac_confidences,
                        h_matrix,
                        scale_factor=2.0,
                    )
                    logger.debug(
                        f"Added RANSAC field lines to top-down view: {len(self.ransac_lines)} lines"
                    )

                if warped_frame_with_segmentation is None:
                    logger.error("Failed to apply transformed segmentation to warped frame")
            except Exception as e:
                logger.error(f"Error applying segmentation to warped frame: {e}")
        elif self.show_segmentation:
            logger.warning("Segmentation enabled but no results available for warped frame")

        self._display_frame(warped_frame, self.warped_display)

        # Record total display update time
        update_duration = (time.perf_counter() - update_start) * 1000
        self.runtime_dialog.add_measurement("Homography Display", update_duration)

        # Don't show error dialog on startup - just log it

    # ------------------------------------------------------------------ parameters

    @staticmethod
    def _slider_position(param_name: str, value: float) -> int:
        """Slider position (0-1000) nearest to a parameter value."""
        low, high = parameter_range(param_name)
        return max(0, min(1000, int((value - low) / (high - low) * 1000)))

    def _create_parameter_controls(self, layout: QVBoxLayout):
        """Create a slider, text input, and value label for each homography parameter."""
        descriptions = {
            "H00": "Scale X",
            "H01": "Skew X",
            "H02": "Translate X",
            "H10": "Skew Y",
            "H11": "Scale Y",
            "H12": "Translate Y",
            "H20": "Perspective X",
            "H21": "Perspective Y",
        }
        form_layout = QFormLayout()

        for param_name in PARAMETER_NAMES:
            value = self.homography_params[param_name]

            # Slider with 1000 steps over the parameter's range
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(self._slider_position(param_name, value))
            slider.valueChanged.connect(
                lambda position, name=param_name: self._on_parameter_changed(name, position)
            )
            self.param_sliders[param_name] = slider

            # Text input for exact values
            text_input = QLineEdit()
            text_input.setText(f"{value:.6f}")
            text_input.setMaximumWidth(80)
            text_input.setStyleSheet("font-family: monospace; font-size: 10px;")
            text_input.editingFinished.connect(
                lambda name=param_name, widget=text_input: self._on_text_input_changed(
                    name, widget.text()
                )
            )
            self.param_inputs[param_name] = text_input

            value_label = QLabel(f"{value:.6f}")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet("font-family: monospace; font-size: 9px; color: #888;")
            self.param_labels[param_name] = value_label

            # Slider and input side by side, the value below
            control_layout = QHBoxLayout()
            control_layout.setContentsMargins(0, 0, 0, 0)
            control_layout.addWidget(slider, 1)
            control_layout.addWidget(text_input, 0)

            param_layout = QVBoxLayout()
            param_layout.setContentsMargins(0, 0, 0, 0)
            param_layout.addLayout(control_layout)
            param_layout.addWidget(value_label)

            param_container = QWidget()
            param_container.setLayout(param_layout)
            form_layout.addRow(f"{descriptions[param_name]} ({param_name}):", param_container)

        layout.addLayout(form_layout)

    def _set_parameters(self, parameters: Dict[str, float]) -> None:
        """Take over parameter values and show them in the sliders, inputs, and labels.

        The values are kept exactly; a slider only shows the nearest of its positions and
        must not write that rounded position back.
        """
        for name, value in parameters.items():
            if name not in self.homography_params:
                continue
            value = float(value)
            self.homography_params[name] = value

            slider = self.param_sliders[name]
            slider.blockSignals(True)
            slider.setValue(self._slider_position(name, value))
            slider.blockSignals(False)

            text_input = self.param_inputs[name]
            text_input.blockSignals(True)
            text_input.setText(f"{value:.6f}")
            text_input.blockSignals(False)

            self.param_labels[name].setText(f"{value:.6f}")

    def _on_parameter_changed(self, param_name: str, slider_value: int):
        """Handle homography parameter change from slider."""
        low, high = parameter_range(param_name)
        param_value = low + slider_value / 1000.0 * (high - low)
        self.homography_params[param_name] = param_value
        self.param_labels[param_name].setText(f"{param_value:.6f}")

        text_input = self.param_inputs[param_name]
        text_input.blockSignals(True)
        text_input.setText(f"{param_value:.6f}")
        text_input.blockSignals(False)

        self._update_displays()

    def _on_text_input_changed(self, param_name: str, text_value: str):
        """Handle homography parameter change from text input."""
        try:
            value = float(text_value)
        except ValueError:
            # Not a number: show the current value again
            value = self.homography_params[param_name]

        low, high = parameter_range(param_name)
        self._set_parameters({param_name: max(low, min(high, value))})
        self._update_displays()

    def _reset_homography(self):
        """Reset homography to identity matrix."""
        self._set_parameters(IDENTITY_PARAMETERS)
        self._update_displays()

    def _calibration_source(self) -> Tuple[Optional[str], int]:
        """Video name and frame index the current parameters were tuned on."""
        video = Path(self.video_files[self.current_video_index]).name if self.video_files else None
        frame = self.scrubbing_slider.value() if self.scrubbing_slider is not None else 0
        return video, frame

    def _save_parameters(self):
        """Save current homography parameters to a YAML file chosen by the user."""
        homography_dir = Path(get_setting("homography.save_directory", "configs"))
        homography_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Homography Parameters",
            str(homography_dir / f"homography_params_{timestamp}.yaml"),
            "YAML files (*.yaml *.yml);;All files (*.*)",
        )
        if not filename:
            return

        try:
            save_parameters(filename, self.homography_params, *self._calibration_source())
            QMessageBox.information(self, "Success", f"Parameters saved to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save parameters:\n{str(e)}")
            logger.error(f"Error saving parameters: {e}")

    def _save_as_default_parameters(self):
        """Save current parameters as the default parameters file."""
        default_file = default_parameters_file()
        try:
            save_parameters(
                default_file,
                self.homography_params,
                *self._calibration_source(),
                description="Default homography parameters (updated by user)",
            )
            QMessageBox.information(
                self,
                "Success",
                f"Parameters saved as default to:\n{default_file}\n\n"
                "These parameters will be loaded automatically on startup.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save default parameters:\n{str(e)}")
            logger.error(f"Error saving default parameters: {e}")

    def _load_parameters(self):
        """Load homography parameters from a YAML file chosen by the user."""
        homography_dir = Path(get_setting("homography.save_directory", "configs"))
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Homography Parameters",
            str(homography_dir) if homography_dir.exists() else "",
            "YAML files (*.yaml *.yml);;All files (*.*)",
        )
        if not filename:
            return

        try:
            self._set_parameters(load_parameters(filename))
            self._update_displays()
            QMessageBox.information(self, "Success", f"Parameters loaded from:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load parameters:\n{str(e)}")
            logger.error(f"Error loading parameters: {e}")

    def _load_default_parameters(self):
        """Load the default homography parameters on startup, if there are any."""
        default_file = default_parameters_file()
        if not default_file.exists():
            return
        try:
            self._set_parameters(load_parameters(default_file))
            self._update_displays()
        except Exception as e:
            logger.error(f"Error loading default parameters: {e}")

    # ------------------------------------------------------------------ videos and display

    def _reload_videos(self):
        """List the available videos and open the first one."""
        self.video_files = self.video_list.reload()
        if self.video_files:
            self.video_list.setCurrentRow(0)  # Selecting the row loads the video

    def _display_frame(self, frame: np.ndarray, label: ZoomableImageLabel):
        """Display a frame in the specified zoomable label widget."""
        if frame is not None:
            label.set_image(frame_to_pixmap(frame))

    def _apply_homography(self, frame: np.ndarray) -> np.ndarray:
        """Warp a frame to the top-down canvas with the current parameters."""
        height, width = frame.shape[:2]
        return cv2.warpPerspective(
            frame, parameters_to_matrix(self.homography_params), output_canvas_size(width, height)
        )

    def _show_runtime_dialog(self) -> None:
        """Show the runtime performance window."""
        self.runtime_dialog.show()
        self.runtime_dialog.raise_()
        self.runtime_dialog.activateWindow()

    def _record_fitness(self) -> None:
        """Add the optimizer's current best fitness to the chart."""
        if self.ga_optimizer:
            self.ga_fitness_chart.add_point(
                self.ga_optimizer.generation, self.ga_optimizer.best_fitness
            )

    def _reset_all_zoom(self):
        """Reset zoom for both image displays."""
        if self.original_display:
            self.original_display.set_zoom(1.0)
        if self.warped_display:
            self.warped_display.set_zoom(1.0)

    def _fit_images_to_window(self):
        """Fit both images to their current window size."""
        if self.original_display and self.original_display.original_pixmap:
            # Trigger a resize to fit current container
            self.original_display.set_image(self.original_display.original_pixmap)
        if self.warped_display and self.warped_display.original_pixmap:
            # Trigger a resize to fit current container
            self.warped_display.set_image(self.warped_display.original_pixmap)

    def _toggle_grid(self, checked: bool):
        """Toggle grid visibility on both image displays."""
        if self.original_display:
            self.original_display.set_grid_visible(checked)
        if self.warped_display:
            self.warped_display.set_grid_visible(checked)

    def _update_grid_spacing(self, spacing: int):
        """Update grid spacing on both image displays."""
        if self.original_display:
            self.original_display.set_grid_spacing(spacing)
        if self.warped_display:
            self.warped_display.set_grid_spacing(spacing)

    def _run_segmentation_on_current_frame(self):
        """Run field segmentation on the current frame."""
        if self.current_frame is None:
            logger.warning("No current frame available for segmentation")
            return

        try:
            logger.debug("Running field segmentation on current frame")
            segmentation_start = time.perf_counter()
            with MODEL_LOCK:
                self.current_segmentation_results = run_field_segmentation(self.current_frame)
            segmentation_duration = (time.perf_counter() - segmentation_start) * 1000
            self.runtime_dialog.add_measurement("Field Segmentation", segmentation_duration)

            if self.current_segmentation_results:
                logger.info(
                    f"Segmentation complete: {len(self.current_segmentation_results)} results ({segmentation_duration:.1f}ms)"
                )
                # Debug: Check if results have masks
                for i, result in enumerate(self.current_segmentation_results):
                    if hasattr(result, "masks") and result.masks is not None:
                        logger.debug(
                            f"Result {i}: has masks with shape {result.masks.data.shape if hasattr(result.masks, 'data') else 'unknown'}"
                        )
                    else:
                        logger.debug(f"Result {i}: no masks found")
            else:
                logger.debug("No segmentation results returned")
        except Exception as e:
            logger.error(f"Error running field segmentation: {e}")
            self.current_segmentation_results = None

    def _load_segmentation_models(self):
        """Load available field segmentation models using utility function."""
        if self.segmentation_model_combo is not None:
            populate_segmentation_model_combo(
                self.segmentation_model_combo, default_model_path("segmentation")
            )
            self._on_segmentation_model_changed(self.segmentation_model_combo.currentText())

    # Removed segmentation and RANSAC toggle methods - features are now auto-enabled

    def _on_segmentation_model_changed(self, display_name: str):
        """Handle segmentation model selection change."""
        if self.segmentation_model_combo is None:
            return

        model_path = self.segmentation_model_combo.currentData()
        if model_path and os.path.exists(model_path):
            logger.debug(f"Changing segmentation model to: {model_path}")
            try:
                with MODEL_LOCK:
                    model_set = set_field_model(model_path)
                if model_set:
                    logger.debug(f"Successfully loaded segmentation model: {display_name}")
                    # Re-run segmentation with new model if currently enabled
                    if self.show_segmentation:
                        self._run_segmentation_on_current_frame()
                        self._update_displays()
                else:
                    logger.error(f"Failed to load segmentation model: {model_path}")
            except Exception as e:
                logger.error(f"Error loading segmentation model: {e}")
        else:
            logger.warning(f"Invalid model path: {model_path}")

    # ===== GENETIC ALGORITHM METHODS =====

    def _validate_ga_prerequisites(self) -> bool:
        """Validate that genetic algorithm can be started.

        Returns:
            True if GA can be started, False otherwise
        """
        if self.current_frame is None:
            QMessageBox.warning(
                self,
                "Cannot Start GA",
                "No video frame loaded. Please load a video first.",
            )
            return False

        if not self.ransac_lines or len(self.ransac_lines) < 2:
            QMessageBox.warning(
                self,
                "Insufficient Line Data",
                "Need at least 2 detected field lines for optimization.\n\n"
                "Please:\n"
                "1. Enable field segmentation\n"
                "2. Ensure lines are detected in the current frame\n"
                "3. Try adjusting segmentation parameters if needed",
            )
            return False

        return True

    def _start_genetic_algorithm(self):
        """Initialize and start genetic algorithm with current parameters."""
        if not self._validate_ga_prerequisites():
            return

        try:
            # Import GA module
            from ...optimization.homography_optimizer import HomographyOptimizer

            # Initialize optimizer with current parameters
            self.ga_optimizer = HomographyOptimizer(
                initial_params=self.homography_params,
                population_size=get_setting("optimization.ga_population_size", 20),
                elite_size=get_setting("optimization.ga_elite_size", 2),
                mutation_rate=get_setting("optimization.ga_mutation_rate", 0.2),
                crossover_rate=get_setting("optimization.ga_crossover_rate", 0.7),
            )

            # Calculate initial fitness
            logger.debug("Evaluating initial GA population...")
            self.ga_optimizer.evaluate_population(
                self.current_frame, self.ransac_lines, self.ransac_confidences
            )

            # Update UI state
            self.ga_running = True
            self._update_ga_ui_state(running=True)
            self._update_ga_display()

            # Start the fitness chart with the initial population
            self.ga_fitness_chart.clear()
            self._record_fitness()

            logger.info(
                f"GA started with population size {self.ga_optimizer.population_size}, "
                f"initial best fitness: {self.ga_optimizer.best_fitness:.4f}"
            )

            # Show info message
            QMessageBox.information(
                self,
                "GA Started",
                f"Genetic algorithm initialized with current parameters.\n\n"
                f"Population size: {self.ga_optimizer.population_size}\n"
                f"Initial best fitness: {self.ga_optimizer.best_fitness:.4f}\n\n"
                f"Click 'Next Gen' to evolve the population or 'Skip 10 Gens' to run multiple generations.\n"
                f"You can apply the best solution at any time with 'Apply Best Parameters'.",
            )

        except Exception as e:
            logger.error(f"Error starting genetic algorithm: {e}")
            QMessageBox.critical(self, "GA Error", f"Failed to start genetic algorithm:\n{str(e)}")

    def _evolve_ga_next_generation(self):
        """Evolve genetic algorithm to the next generation."""
        if not self.ga_optimizer or not self.ga_running:
            return

        try:
            logger.debug(f"Evolving to generation {self.ga_optimizer.generation + 1}")

            # Evolve to next generation
            self.ga_optimizer.evolve()

            # Evaluate new population
            self.ga_optimizer.evaluate_population(
                self.current_frame, self.ransac_lines, self.ransac_confidences
            )

            # Update UI
            self._update_ga_display()
            self._record_fitness()

            # Preview best parameters automatically
            self._preview_ga_best_parameters()

            logger.info(
                f"Generation {self.ga_optimizer.generation} complete, "
                f"best fitness: {self.ga_optimizer.best_fitness:.4f}"
            )

        except Exception as e:
            logger.error(f"Error evolving generation: {e}")
            QMessageBox.critical(self, "GA Error", f"Error during evolution:\n{str(e)}")

    def _evolve_ga_generations(self, num_generations: int):
        """Evolve genetic algorithm for multiple generations.

        Args:
            num_generations: Number of generations to evolve
        """
        if not self.ga_optimizer or not self.ga_running:
            return

        # Disable buttons during processing
        self._update_ga_ui_state(running=True, processing=True)

        # Create progress dialog
        progress = QProgressDialog(
            f"Evolving {num_generations} generations...", "Cancel", 0, num_generations, self
        )
        progress.setWindowTitle("Genetic Algorithm Evolution")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately

        try:
            logger.debug(f"Evolving {num_generations} generations in batch")

            # Process generations
            for i in range(num_generations):
                if progress.wasCanceled():
                    logger.debug("GA evolution canceled by user")
                    break

                # Evolve and evaluate
                self.ga_optimizer.evolve()
                self.ga_optimizer.evaluate_population(
                    self.current_frame, self.ransac_lines, self.ransac_confidences
                )

                # Update progress
                progress.setValue(i + 1)
                progress.setLabelText(
                    f"Generation {self.ga_optimizer.generation}: "
                    f"Best fitness {self.ga_optimizer.best_fitness:.4f}"
                )

                # Process events to keep UI responsive
                QApplication.processEvents()

            # Update UI
            self._update_ga_display()
            self._record_fitness()

            # Preview best parameters automatically
            self._preview_ga_best_parameters()

            logger.info(
                f"Batch evolution complete. Final generation: {self.ga_optimizer.generation}, "
                f"best fitness: {self.ga_optimizer.best_fitness:.4f}"
            )

        except Exception as e:
            logger.error(f"Error during batch evolution: {e}")
            QMessageBox.critical(self, "GA Error", f"Error during batch evolution:\n{str(e)}")
        finally:
            # Re-enable buttons
            self._update_ga_ui_state(running=True, processing=False)

    def _preview_ga_best_parameters(self):
        """Preview the best parameters found by genetic algorithm without permanently applying them."""
        if not self.ga_optimizer:
            return

        try:
            # Store original parameters
            original_params = self.homography_params.copy()

            # Temporarily apply best parameters
            best_params = self.ga_optimizer.get_best_parameters()
            for name, value in best_params.items():
                if name in self.homography_params:
                    self.homography_params[name] = value

            # Update displays
            self._update_displays()

            # Restore original parameters (this keeps the display but reverts internal state)
            self.homography_params = original_params

        except Exception as e:
            logger.error(f"Error previewing GA parameters: {e}")

    def _apply_ga_best_parameters(self):
        """Apply the best parameters found by genetic algorithm."""
        if not self.ga_optimizer or not self.ga_running:
            return

        try:
            # Get best parameters
            best_params = self.ga_optimizer.get_best_parameters()

            # Apply to homography parameters and update UI
            for name, value in best_params.items():
                if name in self.homography_params:
                    self._set_parameters({name: value})

            # Update displays
            self._update_displays()

            # Show success message
            stats = self.ga_optimizer.get_population_stats()
            QMessageBox.information(
                self,
                "GA Parameters Applied",
                f"Applied best parameters from generation {self.ga_optimizer.generation}\n\n"
                f"Best fitness: {stats['best_fitness']:.4f}\n"
                f"Average fitness: {stats['average_fitness']:.4f}\n"
                f"Population std: {stats['fitness_std']:.4f}\n\n"
                f"Parameters have been permanently applied.",
            )

            logger.info(f"Applied GA best parameters with fitness {stats['best_fitness']:.4f}")

        except Exception as e:
            logger.error(f"Error applying GA parameters: {e}")
            QMessageBox.critical(self, "GA Error", f"Error applying parameters:\n{str(e)}")

    def _reset_genetic_algorithm(self):
        """Reset genetic algorithm to initial state."""
        # Stop continuous evolution if running
        if self.ga_continuous_timer and self.ga_continuous_timer.isActive():
            self.ga_continuous_timer.stop()

        self.ga_optimizer = None
        self.ga_running = False
        self.ga_generation_history.clear()
        self.ga_fitness_history.clear()

        # Update UI state
        self._update_ga_ui_state(running=False)
        self.ga_fitness_chart.clear()

        logger.info("Genetic algorithm reset")

    def _update_ga_ui_state(self, running: bool, processing: bool = False):
        """Update GA UI button states.

        Args:
            running: Whether GA is currently running
            processing: Whether GA is currently processing (disable all controls)
        """
        if processing:
            # Disable all buttons during processing
            self.ga_start_button.setEnabled(False)
            self.ga_next_gen_button.setEnabled(False)
            self.ga_multi_gen_button.setEnabled(False)
            self.ga_reset_button.setEnabled(False)
            self.ga_apply_button.setEnabled(False)
            if self.ga_continuous_button:
                self.ga_continuous_button.setEnabled(False)
            if self.ga_stop_button:
                self.ga_stop_button.setEnabled(False)
        elif running:
            # GA is running, enable evolution controls
            self.ga_start_button.setEnabled(False)
            self.ga_next_gen_button.setEnabled(True)
            self.ga_multi_gen_button.setEnabled(True)
            self.ga_reset_button.setEnabled(True)
            self.ga_apply_button.setEnabled(True)
            if self.ga_continuous_button:
                self.ga_continuous_button.setEnabled(True)
            if self.ga_stop_button:
                self.ga_stop_button.setEnabled(False)
        else:
            # GA not running, only enable start
            self.ga_start_button.setEnabled(True)
            self.ga_next_gen_button.setEnabled(False)
            self.ga_multi_gen_button.setEnabled(False)
            self.ga_reset_button.setEnabled(False)
            self.ga_apply_button.setEnabled(False)
            if self.ga_continuous_button:
                self.ga_continuous_button.setEnabled(False)
            if self.ga_stop_button:
                self.ga_stop_button.setEnabled(False)

    def _update_ga_display(self):
        """Update GA status display with current information."""
        if not self.ga_optimizer:
            self.ga_generation_label.setText("0")
            self.ga_fitness_label.setText("0.000")
            self.ga_population_label.setText("20")
            return

        # Update status labels
        self.ga_generation_label.setText(str(self.ga_optimizer.generation))
        self.ga_fitness_label.setText(f"{self.ga_optimizer.best_fitness:.4f}")
        self.ga_population_label.setText(str(self.ga_optimizer.population_size))

    def _start_continuous_evolution(self):
        """Start continuous evolution using a timer."""
        if not self.ga_optimizer or not self.ga_running:
            logger.error("Cannot start continuous evolution - GA not running")
            return

        logger.info("Starting continuous evolution...")

        # Initialize timer if not already created
        if not self.ga_continuous_timer:
            self.ga_continuous_timer = QTimer()
            self.ga_continuous_timer.timeout.connect(self._continuous_evolution_step)

        # Set timer interval (1 second between generations)
        evolution_interval = get_setting("optimization.continuous_evolution_interval_ms", 1000)
        self.ga_continuous_timer.start(evolution_interval)

        # Update button states
        if self.ga_continuous_button:
            self.ga_continuous_button.setEnabled(False)
        if self.ga_stop_button:
            self.ga_stop_button.setEnabled(True)

        # Disable other GA buttons during continuous evolution
        if self.ga_next_gen_button:
            self.ga_next_gen_button.setEnabled(False)
        if self.ga_multi_gen_button:
            self.ga_multi_gen_button.setEnabled(False)

    def _stop_continuous_evolution(self):
        """Stop continuous evolution."""
        logger.debug("Stopping continuous evolution...")

        if self.ga_continuous_timer and self.ga_continuous_timer.isActive():
            self.ga_continuous_timer.stop()

        # Update button states
        if self.ga_continuous_button:
            self.ga_continuous_button.setEnabled(True)
        if self.ga_stop_button:
            self.ga_stop_button.setEnabled(False)

        # Re-enable other GA buttons
        if self.ga_next_gen_button:
            self.ga_next_gen_button.setEnabled(True)
        if self.ga_multi_gen_button:
            self.ga_multi_gen_button.setEnabled(True)

    def _continuous_evolution_step(self):
        """Execute a single evolution step for continuous mode."""
        try:
            if not self.ga_optimizer or not self.ga_running:
                logger.info("GA stopped - stopping continuous evolution")
                self._stop_continuous_evolution()
                return

            # Run next generation
            self._evolve_ga_next_generation()

            # Optional: Stop if convergence criteria are met
            convergence_threshold = get_setting("optimization.convergence_threshold", 0.95)
            max_generations = get_setting("optimization.max_generations", 200)

            if (
                self.ga_optimizer.best_fitness >= convergence_threshold
                or self.ga_optimizer.generation >= max_generations
            ):
                logger.debug(
                    f"Stopping continuous evolution - convergence reached "
                    f"(fitness: {self.ga_optimizer.best_fitness:.4f}, generation: {self.ga_optimizer.generation})"
                )
                self._stop_continuous_evolution()

        except Exception as e:
            logger.error(f"Error in continuous evolution step: {e}")
            self._stop_continuous_evolution()
