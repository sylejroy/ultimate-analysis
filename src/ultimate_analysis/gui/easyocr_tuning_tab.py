"""EasyOCR Tuning Tab for Ultimate Analysis GUI.

This module provides a specialized interface for tuning EasyOCR parameters
on detected player bounding boxes for optimal jersey number recognition.
"""

import os
import cv2
import numpy as np
import yaml
import random
import traceback
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, 
    QPushButton, QSlider, QListWidgetItem, QGroupBox,
    QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QSplitter, QScrollArea, QGridLayout,
    QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont

from .video_player import VideoPlayer
from ..processing import run_inference, set_detection_model
from ..processing.player_id import run_player_id, _initialize_easyocr
from ..config.settings import get_setting, get_config
from ..constants import DEFAULT_PATHS, SUPPORTED_VIDEO_EXTENSIONS, JERSEY_NUMBER_MIN, JERSEY_NUMBER_MAX

# Try to import EasyOCR for parameter checking
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None


class EasyOCRTuningTab(QWidget):
    """EasyOCR parameter tuning tab for optimizing jersey number detection."""
    
    def __init__(self):
        super().__init__()
        
        # State
        self.video_player = VideoPlayer()
        self.video_files: List[str] = []
        self.current_video_index: int = 0
        self.current_frame: Optional[np.ndarray] = None
        self.current_detections: List[Dict] = []
        self.current_crops: List[Tuple[np.ndarray, Dict]] = []  # (crop_image, detection_info)
        
        # EasyOCR parameters (with optimized defaults)
        self.ocr_params = {
            'languages': ['en'],
            'gpu': True,
            'width_ths': 0.4,  # Updated from provided config
            'height_ths': 0.7,
            'paragraph': False,
            'adjust_contrast': 0.5,  # From provided config
            'filter_ths': 0.003,
            'text_threshold': 0.7,  # From provided config
            'low_text': 0.6,  # Updated from provided config
            'link_threshold': 0.4,
            'canvas_size': 2560,
            'mag_ratio': 2.0,  # Updated from provided config
            'slope_ths': 0.1,  # From provided config
            'ycenter_ths': 0.5,
            'y_ths': 0.5,
            'x_ths': 1.0,
            'detector': True,
            'recognizer': True,
            'allowlist': '0123456789',  # From provided config - digits only
            'detail': 1,  # From provided config
            'rotation_info': [0],  # From provided config
            'decoder': 'greedy',  # From provided config
            'beamWidth': 5,
            'workers': 0,  # Number of parallel workers (0 = auto)
            'batch_size': 1,
        }
        
        # Preprocessing parameters (with optimized defaults)
        self.preprocess_params = {
            'crop_top_fraction': 0.33,  # Use top third of detection
            'contrast_alpha': 1.0,     # Contrast adjustment
            'brightness_beta': 0,      # Brightness adjustment
            'gaussian_blur': 13,       # Updated from provided config (blur_ksize)
            'resize_factor': 1.0,      # Resize factor (multiplier)
            'resize_absolute_width': 0,  # Absolute width in pixels (0 = use factor)
            'resize_absolute_height': 0, # Absolute height in pixels (0 = use factor)
            'enhance_contrast': False,  # CLAHE enhancement (disabled per config)
            'clahe_clip_limit': 3.0,   # From provided config
            'clahe_grid_size': 8,      # From provided config
            'denoise': False,          # Apply denoising
            'sharpen': True,           # From provided config (enabled)
            'sharpen_strength': 0.05,  # From provided config
            'upscale': True,           # From provided config (enabled)
            'upscale_factor': 3.0,     # From provided config
            'upscale_to_size': True,   # From provided config
            'upscale_target_size': 256, # From provided config
            'colour_mode': True,       # From provided config
            'bw_mode': True,           # From provided config
        }
        
        # Initialize UI
        self._init_ui()
        self._load_videos()
        # Load parameters from config (including user.yaml) automatically on startup
        self._load_parameters_from_config()
        
        print("[EASYOCR_TUNING] EasyOCR Tuning Tab initialized with user configuration loaded")
    
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout()
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Video list and parameters
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Video display and results
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (35% left, 65% right - more space for parameters)
        splitter.setSizes([350, 1400])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with video list and parameter controls."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Video list section
        video_group = QGroupBox("Video Selection")
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
        self.video_list = QListWidget()
        self.video_list.setMinimumHeight(200)  # Make video list taller
        self.video_list.currentRowChanged.connect(self._on_video_selection_changed)
        video_layout.addWidget(self.video_list)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Model selection
        model_group = QGroupBox("Detection Model")
        model_layout = QFormLayout()
        
        self.detection_model_combo = QComboBox()
        self._populate_model_combo()
        self.detection_model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addRow("Model:", self.detection_model_combo)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Parameters in 2-column layout
        params_group = QGroupBox("Parameters")
        params_main_layout = QHBoxLayout()
        
        # Left column - Preprocessing
        left_column = QWidget()
        left_layout = QVBoxLayout()
        
        preprocess_group = QGroupBox("Preprocessing Parameters")
        preprocess_layout = QFormLayout()
        self._create_preprocessing_controls(preprocess_layout)
        preprocess_group.setLayout(preprocess_layout)
        left_layout.addWidget(preprocess_group)
        left_layout.addStretch()
        left_column.setLayout(left_layout)
        
        # Right column - EasyOCR
        right_column = QWidget()
        right_layout = QVBoxLayout()
        
        ocr_group = QGroupBox("EasyOCR Parameters")
        ocr_layout = QFormLayout()
        self._create_ocr_controls(ocr_layout)
        ocr_group.setLayout(ocr_layout)
        right_layout.addWidget(ocr_group)
        right_layout.addStretch()
        right_column.setLayout(right_layout)
        
        # Add columns to main layout
        params_main_layout.addWidget(left_column)
        params_main_layout.addWidget(right_column)
        params_group.setLayout(params_main_layout)
        layout.addWidget(params_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        load_button = QPushButton("Load from Config")
        load_button.clicked.connect(self._load_parameters_from_config)
        load_button.setToolTip("Load parameters from default.yaml")
        button_layout.addWidget(load_button)
        
        save_button = QPushButton("Save to Config")
        save_button.clicked.connect(self._save_parameters_to_config)
        save_button.setToolTip("Save current parameters to user.yaml")
        button_layout.addWidget(save_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with video display and results."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Video display area - fixed size
        video_container = QWidget()
        video_container.setFixedHeight(550)  # Increased from 450 to 550 for larger video
        video_layout = QVBoxLayout()
        
        self.video_label = QLabel("No video selected")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedHeight(450)  # Fixed height instead of minimum to prevent growth
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #1a1a1a;
                color: #999;
                font-size: 14px;
            }
        """)
        self.video_label.setScaledContents(False)  # Don't scale contents
        video_layout.addWidget(self.video_label)
        
        # Frame slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.setValue(0)
        self.frame_slider.sliderMoved.connect(self._on_frame_changed)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0 / 0")
        slider_layout.addWidget(self.frame_label)
        
        video_layout.addLayout(slider_layout)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.run_analysis_button = QPushButton("Run EasyOCR Analysis")
        self.run_analysis_button.clicked.connect(self._run_easyocr_analysis)
        self.run_analysis_button.setToolTip("Run detection + EasyOCR analysis on current frame")
        self.run_analysis_button.setEnabled(False)  # Disabled until video loads
        control_layout.addWidget(self.run_analysis_button)
        
        control_layout.addStretch()
        video_layout.addLayout(control_layout)
        
        video_container.setLayout(video_layout)
        layout.addWidget(video_container)
        
        # Crops display area - takes remaining space
        crops_group = QGroupBox("Detected Crops & OCR Results")
        crops_layout = QVBoxLayout()
        
        # Scroll area for crops - no height restriction, takes all remaining space
        self.crops_scroll = QScrollArea()
        self.crops_scroll.setWidgetResizable(True)
        self.crops_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.crops_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container for crop displays
        self.crops_container = QWidget()
        self.crops_container.setStyleSheet("QWidget { background-color: #1a1a1a; }")  # Dark background
        self.crops_layout = QGridLayout()
        self.crops_layout.setSpacing(8)  # Increased spacing for larger widgets
        self.crops_layout.setContentsMargins(8, 8, 8, 8)  # Increased margins
        self.crops_container.setLayout(self.crops_layout)
        self.crops_scroll.setWidget(self.crops_container)
        
        crops_layout.addWidget(self.crops_scroll)
        crops_group.setLayout(crops_layout)
        layout.addWidget(crops_group, 1)  # Give it stretch factor of 1 to take remaining space
        
        panel.setLayout(layout)
        return panel
    
    def _create_preprocessing_controls(self, layout: QFormLayout):
        """Create preprocessing parameter controls."""
        # Top crop fraction
        self.crop_fraction_spin = QDoubleSpinBox()
        self.crop_fraction_spin.setRange(0.1, 1.0)
        self.crop_fraction_spin.setSingleStep(0.05)
        self.crop_fraction_spin.setValue(self.preprocess_params['crop_top_fraction'])
        self.crop_fraction_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("Top Crop Fraction:", self.crop_fraction_spin)
        
        # Contrast
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.1, 3.0)
        self.contrast_spin.setSingleStep(0.1)
        self.contrast_spin.setValue(self.preprocess_params['contrast_alpha'])
        self.contrast_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("Contrast (α):", self.contrast_spin)
        
        # Brightness
        self.brightness_spin = QSpinBox()
        self.brightness_spin.setRange(-100, 100)
        self.brightness_spin.setValue(self.preprocess_params['brightness_beta'])
        self.brightness_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("Brightness (β):", self.brightness_spin)
        
        # Gaussian blur
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(0, 31)  # Must be odd
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(self.preprocess_params['gaussian_blur'])
        self.blur_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("Gaussian Blur (ksize):", self.blur_spin)
        
        # Contrast Enhancement Section
        layout.addRow(QLabel("=== Contrast Enhancement ==="))
        
        # CLAHE enhancement toggle
        self.enhance_check = QCheckBox()
        self.enhance_check.setChecked(self.preprocess_params['enhance_contrast'])
        self.enhance_check.stateChanged.connect(self._on_preprocess_param_changed)
        self.enhance_check.stateChanged.connect(self._update_clahe_controls)  # Update control states
        layout.addRow("Enable CLAHE Enhancement:", self.enhance_check)
        
        # CLAHE parameters (indented to show dependency)
        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setRange(1.0, 10.0)
        self.clahe_clip_spin.setSingleStep(0.5)
        self.clahe_clip_spin.setValue(self.preprocess_params['clahe_clip_limit'])
        self.clahe_clip_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("  → Clip Limit:", self.clahe_clip_spin)
        
        self.clahe_grid_spin = QSpinBox()
        self.clahe_grid_spin.setRange(2, 16)
        self.clahe_grid_spin.setValue(self.preprocess_params['clahe_grid_size'])
        self.clahe_grid_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("  → Grid Size:", self.clahe_grid_spin)
        
        # Denoising Section
        layout.addRow(QLabel("=== Denoising ==="))
        
        # Denoising toggle
        self.denoise_check = QCheckBox()
        self.denoise_check.setChecked(self.preprocess_params['denoise'])
        self.denoise_check.stateChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("Enable Denoising:", self.denoise_check)
        
        # Sharpening Section
        layout.addRow(QLabel("=== Sharpening ==="))
        
        # Sharpening toggle
        self.sharpen_check = QCheckBox()
        self.sharpen_check.setChecked(self.preprocess_params['sharpen'])
        self.sharpen_check.stateChanged.connect(self._on_preprocess_param_changed)
        self.sharpen_check.stateChanged.connect(self._update_sharpen_controls)  # Update control states
        layout.addRow("Enable Sharpening:", self.sharpen_check)
        
        # Sharpen strength (indented to show dependency)
        self.sharpen_strength_spin = QDoubleSpinBox()
        self.sharpen_strength_spin.setRange(0.01, 1.0)
        self.sharpen_strength_spin.setSingleStep(0.01)
        self.sharpen_strength_spin.setDecimals(3)
        self.sharpen_strength_spin.setValue(self.preprocess_params['sharpen_strength'])
        self.sharpen_strength_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("  → Strength:", self.sharpen_strength_spin)
        
        # Upscaling Section
        layout.addRow(QLabel("=== Upscaling ==="))
        
        # Upscaling toggle
        self.upscale_check = QCheckBox()
        self.upscale_check.setChecked(self.preprocess_params['upscale'])
        self.upscale_check.stateChanged.connect(self._on_preprocess_param_changed)
        self.upscale_check.stateChanged.connect(self._update_upscale_controls)  # Update control states
        layout.addRow("Enable Upscaling:", self.upscale_check)
        
        # Upscale parameters (indented to show dependency)
        self.upscale_factor_spin = QDoubleSpinBox()
        self.upscale_factor_spin.setRange(1.0, 8.0)
        self.upscale_factor_spin.setSingleStep(0.5)
        self.upscale_factor_spin.setValue(self.preprocess_params['upscale_factor'])
        self.upscale_factor_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("  → Factor:", self.upscale_factor_spin)
        
        self.upscale_to_size_check = QCheckBox()
        self.upscale_to_size_check.setChecked(self.preprocess_params['upscale_to_size'])
        self.upscale_to_size_check.stateChanged.connect(self._on_preprocess_param_changed)
        self.upscale_to_size_check.stateChanged.connect(self._update_upscale_controls)  # Update target size control
        layout.addRow("  → Scale to Fixed Size:", self.upscale_to_size_check)
        
        self.upscale_target_spin = QSpinBox()
        self.upscale_target_spin.setRange(64, 1024)
        self.upscale_target_spin.setSingleStep(32)
        self.upscale_target_spin.setValue(self.preprocess_params['upscale_target_size'])
        self.upscale_target_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("    → Target Size (px):", self.upscale_target_spin)
        
        # Color Processing Section
        layout.addRow(QLabel("=== Color Processing ==="))
        
        # Color mode processing
        self.colour_mode_check = QCheckBox()
        self.colour_mode_check.setChecked(self.preprocess_params['colour_mode'])
        self.colour_mode_check.stateChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("Color Mode:", self.colour_mode_check)
        
        # Black & white mode
        self.bw_mode_check = QCheckBox()
        self.bw_mode_check.setChecked(self.preprocess_params['bw_mode'])
        self.bw_mode_check.stateChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("B&W Mode:", self.bw_mode_check)
        
        # Size Adjustment Section
        layout.addRow(QLabel("=== Size Adjustment ==="))
        
        # Resize factor
        self.resize_spin = QDoubleSpinBox()
        self.resize_spin.setRange(0.5, 4.0)
        self.resize_spin.setSingleStep(0.1)
        self.resize_spin.setValue(self.preprocess_params['resize_factor'])
        self.resize_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("Resize Factor:", self.resize_spin)
        
        # Absolute resize width
        self.resize_width_spin = QSpinBox()
        self.resize_width_spin.setRange(0, 2048)
        self.resize_width_spin.setSingleStep(32)
        self.resize_width_spin.setValue(self.preprocess_params['resize_absolute_width'])
        self.resize_width_spin.valueChanged.connect(self._on_preprocess_param_changed)
        self.resize_width_spin.setToolTip("Absolute width in pixels (0 = use factor)")
        layout.addRow("Absolute Width (px):", self.resize_width_spin)
        
        # Absolute resize height
        self.resize_height_spin = QSpinBox()
        self.resize_height_spin.setRange(0, 2048)
        self.resize_height_spin.setSingleStep(32)
        self.resize_height_spin.setValue(self.preprocess_params['resize_absolute_height'])
        self.resize_height_spin.valueChanged.connect(self._on_preprocess_param_changed)
        self.resize_height_spin.setToolTip("Absolute height in pixels (0 = use factor)")
        layout.addRow("Absolute Height (px):", self.resize_height_spin)
        
        # Initialize control states based on checkboxes
        self._update_clahe_controls()
        self._update_sharpen_controls()
        self._update_upscale_controls()
    
    def _create_ocr_controls(self, layout: QFormLayout):
        """Create EasyOCR parameter controls."""
        if not EASYOCR_AVAILABLE:
            layout.addRow(QLabel("EasyOCR not available"))
            return
        
        # Core detection parameters
        layout.addRow(QLabel("=== Detection Parameters ==="))
        
        # Text confidence threshold
        self.text_threshold_spin = QDoubleSpinBox()
        self.text_threshold_spin.setRange(0.1, 1.0)
        self.text_threshold_spin.setSingleStep(0.05)
        self.text_threshold_spin.setValue(self.ocr_params['text_threshold'])
        self.text_threshold_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Text Threshold:", self.text_threshold_spin)
        
        # Low text threshold
        self.low_text_spin = QDoubleSpinBox()
        self.low_text_spin.setRange(0.1, 1.0)
        self.low_text_spin.setSingleStep(0.05)
        self.low_text_spin.setValue(self.ocr_params['low_text'])
        self.low_text_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Low Text:", self.low_text_spin)
        
        # Link threshold
        self.link_threshold_spin = QDoubleSpinBox()
        self.link_threshold_spin.setRange(0.1, 1.0)
        self.link_threshold_spin.setSingleStep(0.05)
        self.link_threshold_spin.setValue(self.ocr_params['link_threshold'])
        self.link_threshold_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Link Threshold:", self.link_threshold_spin)
        
        # Geometric constraints
        layout.addRow(QLabel("=== Geometric Constraints ==="))
        
        # Width threshold
        self.width_ths_spin = QDoubleSpinBox()
        self.width_ths_spin.setRange(0.1, 2.0)
        self.width_ths_spin.setSingleStep(0.1)
        self.width_ths_spin.setValue(self.ocr_params['width_ths'])
        self.width_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Width Threshold:", self.width_ths_spin)
        
        # Height threshold
        self.height_ths_spin = QDoubleSpinBox()
        self.height_ths_spin.setRange(0.1, 2.0)
        self.height_ths_spin.setSingleStep(0.1)
        self.height_ths_spin.setValue(self.ocr_params['height_ths'])
        self.height_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Height Threshold:", self.height_ths_spin)
        
        # X threshold
        self.x_ths_spin = QDoubleSpinBox()
        self.x_ths_spin.setRange(0.1, 3.0)
        self.x_ths_spin.setSingleStep(0.1)
        self.x_ths_spin.setValue(self.ocr_params['x_ths'])
        self.x_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("X Threshold:", self.x_ths_spin)
        
        # Y threshold
        self.y_ths_spin = QDoubleSpinBox()
        self.y_ths_spin.setRange(0.1, 1.0)
        self.y_ths_spin.setSingleStep(0.05)
        self.y_ths_spin.setValue(self.ocr_params['y_ths'])
        self.y_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Y Threshold:", self.y_ths_spin)
        
        # Y-center threshold
        self.ycenter_ths_spin = QDoubleSpinBox()
        self.ycenter_ths_spin.setRange(0.1, 1.0)
        self.ycenter_ths_spin.setSingleStep(0.05)
        self.ycenter_ths_spin.setValue(self.ocr_params['ycenter_ths'])
        self.ycenter_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Y-Center Threshold:", self.ycenter_ths_spin)
        
        # Slope threshold
        self.slope_ths_spin = QDoubleSpinBox()
        self.slope_ths_spin.setRange(0.01, 1.0)
        self.slope_ths_spin.setSingleStep(0.05)
        self.slope_ths_spin.setValue(self.ocr_params['slope_ths'])
        self.slope_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Slope Threshold:", self.slope_ths_spin)
        
        # Image processing parameters
        layout.addRow(QLabel("=== Image Processing ==="))
        
        # Canvas size
        self.canvas_size_spin = QSpinBox()
        self.canvas_size_spin.setRange(512, 4096)
        self.canvas_size_spin.setSingleStep(256)
        self.canvas_size_spin.setValue(self.ocr_params['canvas_size'])
        self.canvas_size_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Canvas Size:", self.canvas_size_spin)
        
        # Magnification ratio
        self.mag_ratio_spin = QDoubleSpinBox()
        self.mag_ratio_spin.setRange(0.5, 3.0)
        self.mag_ratio_spin.setSingleStep(0.1)
        self.mag_ratio_spin.setValue(self.ocr_params['mag_ratio'])
        self.mag_ratio_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Mag Ratio:", self.mag_ratio_spin)
        
        # Adjust contrast
        self.adjust_contrast_spin = QDoubleSpinBox()
        self.adjust_contrast_spin.setRange(0.0, 2.0)
        self.adjust_contrast_spin.setSingleStep(0.1)
        self.adjust_contrast_spin.setValue(self.ocr_params['adjust_contrast'])
        self.adjust_contrast_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Adjust Contrast:", self.adjust_contrast_spin)
        
        # Filter threshold
        self.filter_ths_spin = QDoubleSpinBox()
        self.filter_ths_spin.setRange(0.001, 0.1)
        self.filter_ths_spin.setSingleStep(0.001)
        self.filter_ths_spin.setDecimals(4)
        self.filter_ths_spin.setValue(self.ocr_params['filter_ths'])
        self.filter_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Filter Threshold:", self.filter_ths_spin)
        
        # Performance and parallel processing
        layout.addRow(QLabel("=== Performance ==="))
        
        # Number of workers
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(self.ocr_params['workers'])
        self.workers_spin.valueChanged.connect(self._on_ocr_param_changed)
        self.workers_spin.setToolTip("Number of parallel workers (0 = auto)")
        layout.addRow("Workers:", self.workers_spin)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(self.ocr_params['batch_size'])
        self.batch_size_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Batch Size:", self.batch_size_spin)
        
        # Beam width (for beam search decoder)
        self.beam_width_spin = QSpinBox()
        self.beam_width_spin.setRange(1, 20)
        self.beam_width_spin.setValue(self.ocr_params['beamWidth'])
        self.beam_width_spin.valueChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Beam Width:", self.beam_width_spin)
        
        # Options and flags
        layout.addRow(QLabel("=== Options ==="))
        
        # GPU usage
        self.gpu_check = QCheckBox()
        self.gpu_check.setChecked(self.ocr_params['gpu'])
        self.gpu_check.stateChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Use GPU:", self.gpu_check)
        
        # Paragraph mode
        self.paragraph_check = QCheckBox()
        self.paragraph_check.setChecked(self.ocr_params['paragraph'])
        self.paragraph_check.stateChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Paragraph Mode:", self.paragraph_check)
        
        # Detail level
        self.detail_spin = QSpinBox()
        self.detail_spin.setRange(0, 2)
        self.detail_spin.setValue(self.ocr_params['detail'])
        self.detail_spin.valueChanged.connect(self._on_ocr_param_changed)
        self.detail_spin.setToolTip("0=no detail, 1=bbox, 2=polygon")
        layout.addRow("Detail Level:", self.detail_spin)
        
        # Character filtering
        layout.addRow(QLabel("=== Character Filtering ==="))
        
        # Allowlist (only allow these characters)
        self.allowlist_edit = QLineEdit()
        if self.ocr_params['allowlist']:
            self.allowlist_edit.setText(self.ocr_params['allowlist'])
        self.allowlist_edit.textChanged.connect(self._on_ocr_param_changed)
        self.allowlist_edit.setPlaceholderText("e.g., 0123456789")
        layout.addRow("Allowlist:", self.allowlist_edit)
        
        # Additional parameters
        layout.addRow(QLabel("=== Additional Options ==="))
    
    def _load_videos(self):
        """Load and display available video files."""
        print("[EASYOCR_TUNING] Loading video files...")
        
        self.video_files.clear()
        self.video_list.clear()
        
        # Search paths for videos
        search_paths = [
            Path(DEFAULT_PATHS['DEV_DATA']),
            Path(DEFAULT_PATHS['RAW_VIDEOS'])
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                print(f"[EASYOCR_TUNING] Search path does not exist: {search_path}")
                continue
            
            print(f"[EASYOCR_TUNING] Searching for videos in: {search_path}")
            
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
        
        print(f"[EASYOCR_TUNING] Found {len(self.video_files)} video files")
        
        # Auto-select a random video if available
        if self.video_files:
            random_index = random.randint(0, len(self.video_files) - 1)
            self.video_list.setCurrentRow(random_index)
            self.current_video_index = random_index
            selected_video = self.video_files[random_index]
            print(f"[EASYOCR_TUNING] Auto-selected random video: {Path(selected_video).name}")
            
            # Load the selected video
            self._load_selected_video()
    
    def _get_video_duration(self, video_path: str) -> str:
        """Get video duration as formatted string."""
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
            print(f"[EASYOCR_TUNING] Error getting duration for {video_path}: {e}")
        
        return "Unknown"
    
    def _populate_model_combo(self):
        """Populate detection model combo box."""
        self.detection_model_combo.clear()
        
        # Look for models in the models directory
        models_path = Path(get_setting("models.base_path", DEFAULT_PATHS['MODELS']))
        
        if not models_path.exists():
            print(f"[EASYOCR_TUNING] Models directory not found: {models_path}")
            return
        
        # Search for detection model files
        model_files = []
        for model_dir in models_path.rglob("*"):
            if model_dir.is_file() and model_dir.suffix == ".pt":
                # Check if this is a detection model
                if "detection" in str(model_dir).lower() or "object" in str(model_dir).lower():
                    relative_path = model_dir.relative_to(models_path)
                    model_files.append(str(relative_path))
        
        # Add pretrained models
        pretrained_path = models_path / "pretrained"
        if pretrained_path.exists():
            for model_file in pretrained_path.glob("*.pt"):
                if not model_file.name.endswith("-seg.pt"):  # Exclude segmentation models
                    relative_path = model_file.relative_to(models_path)
                    model_files.append(str(relative_path))
        
        # Sort and add to combo
        model_files.sort()
        self.detection_model_combo.addItems(model_files)
        
        print(f"[EASYOCR_TUNING] Found {len(model_files)} detection models")
    
    # ========== EVENT HANDLERS ==========
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
        print(f"[EASYOCR_TUNING] Loading video: {video_path}")
        
        # Load video
        if self.video_player.load_video(video_path):
            # Update UI
            filename = Path(video_path).name
            
            # Set frame slider range
            video_info = self.video_player.get_video_info()
            total_frames = video_info['total_frames']
            self.frame_slider.setMaximum(max(1, total_frames - 1))
            self.frame_slider.setValue(0)
            self.frame_label.setText(f"0 / {total_frames}")
            
            # Display first frame
            first_frame = self.video_player.get_current_frame()
            if first_frame is not None:
                self.current_frame = first_frame.copy()
                self._display_frame(first_frame)
                
                # Enable analysis button when video is loaded
                self.run_analysis_button.setEnabled(True)
            
            print(f"[EASYOCR_TUNING] Video loaded successfully: {filename}")
        else:
            self.video_label.setText("Failed to load video")
    
    def _on_frame_changed(self, frame_idx: int):
        """Handle frame slider movement."""
        if self.video_player.is_loaded():
            self.video_player.seek_to_frame(frame_idx)
            
            # Update frame label
            video_info = self.video_player.get_video_info()
            self.frame_label.setText(f"{frame_idx} / {video_info['total_frames']}")
            
            # Display current frame
            frame = self.video_player.get_current_frame()
            if frame is not None:
                self.current_frame = frame.copy()
                self._display_frame(frame)
                
                # Clear previous results
                self.current_detections.clear()
                self.current_crops.clear()
                self._clear_crops_display()
    
    def _on_model_changed(self, model_path: str):
        """Handle detection model change."""
        if model_path:
            full_path = Path(get_setting("models.base_path", DEFAULT_PATHS['MODELS'])) / model_path
            set_detection_model(str(full_path))
            print(f"[EASYOCR_TUNING] Detection model changed to: {model_path}")
    
    def _on_preprocess_param_changed(self):
        """Handle preprocessing parameter change."""
        # Update parameters
        self.preprocess_params['crop_top_fraction'] = self.crop_fraction_spin.value()
        self.preprocess_params['contrast_alpha'] = self.contrast_spin.value()
        self.preprocess_params['brightness_beta'] = self.brightness_spin.value()
        self.preprocess_params['gaussian_blur'] = self.blur_spin.value()
        self.preprocess_params['enhance_contrast'] = self.enhance_check.isChecked()
        self.preprocess_params['clahe_clip_limit'] = self.clahe_clip_spin.value()
        self.preprocess_params['clahe_grid_size'] = self.clahe_grid_spin.value()
        self.preprocess_params['sharpen'] = self.sharpen_check.isChecked()
        self.preprocess_params['sharpen_strength'] = self.sharpen_strength_spin.value()
        self.preprocess_params['upscale'] = self.upscale_check.isChecked()
        self.preprocess_params['upscale_factor'] = self.upscale_factor_spin.value()
        self.preprocess_params['upscale_to_size'] = self.upscale_to_size_check.isChecked()
        self.preprocess_params['upscale_target_size'] = self.upscale_target_spin.value()
        self.preprocess_params['colour_mode'] = self.colour_mode_check.isChecked()
        self.preprocess_params['bw_mode'] = self.bw_mode_check.isChecked()
        self.preprocess_params['resize_factor'] = self.resize_spin.value()
        self.preprocess_params['resize_absolute_width'] = self.resize_width_spin.value()
        self.preprocess_params['resize_absolute_height'] = self.resize_height_spin.value()
        self.preprocess_params['denoise'] = self.denoise_check.isChecked()
        
        print(f"[EASYOCR_TUNING] Preprocessing parameters updated")
    
    def _on_ocr_param_changed(self):
        """Handle EasyOCR parameter change."""
        if not EASYOCR_AVAILABLE:
            return
        
        # Update parameters
        self.ocr_params['text_threshold'] = self.text_threshold_spin.value()
        self.ocr_params['low_text'] = self.low_text_spin.value()
        self.ocr_params['link_threshold'] = self.link_threshold_spin.value()
        self.ocr_params['width_ths'] = self.width_ths_spin.value()
        self.ocr_params['height_ths'] = self.height_ths_spin.value()
        self.ocr_params['x_ths'] = getattr(self, 'x_ths_spin', self).value() if hasattr(self, 'x_ths_spin') else self.ocr_params['x_ths']
        self.ocr_params['y_ths'] = getattr(self, 'y_ths_spin', self).value() if hasattr(self, 'y_ths_spin') else self.ocr_params['y_ths']
        self.ocr_params['ycenter_ths'] = getattr(self, 'ycenter_ths_spin', self).value() if hasattr(self, 'ycenter_ths_spin') else self.ocr_params['ycenter_ths']
        self.ocr_params['slope_ths'] = getattr(self, 'slope_ths_spin', self).value() if hasattr(self, 'slope_ths_spin') else self.ocr_params['slope_ths']
        self.ocr_params['canvas_size'] = self.canvas_size_spin.value()
        self.ocr_params['mag_ratio'] = self.mag_ratio_spin.value()
        self.ocr_params['adjust_contrast'] = getattr(self, 'adjust_contrast_spin', self).value() if hasattr(self, 'adjust_contrast_spin') else self.ocr_params['adjust_contrast']
        self.ocr_params['filter_ths'] = getattr(self, 'filter_ths_spin', self).value() if hasattr(self, 'filter_ths_spin') else self.ocr_params['filter_ths']
        self.ocr_params['workers'] = getattr(self, 'workers_spin', self).value() if hasattr(self, 'workers_spin') else self.ocr_params['workers']
        self.ocr_params['batch_size'] = getattr(self, 'batch_size_spin', self).value() if hasattr(self, 'batch_size_spin') else self.ocr_params['batch_size']
        self.ocr_params['beamWidth'] = getattr(self, 'beam_width_spin', self).value() if hasattr(self, 'beam_width_spin') else self.ocr_params['beamWidth']
        self.ocr_params['gpu'] = self.gpu_check.isChecked()
        self.ocr_params['paragraph'] = getattr(self, 'paragraph_check', self).isChecked() if hasattr(self, 'paragraph_check') else self.ocr_params['paragraph']
        self.ocr_params['detail'] = getattr(self, 'detail_spin', self).value() if hasattr(self, 'detail_spin') else self.ocr_params['detail']
        
        # Handle text inputs
        if hasattr(self, 'allowlist_edit'):
            allowlist_text = self.allowlist_edit.text().strip()
            self.ocr_params['allowlist'] = allowlist_text if allowlist_text else None
        
        print(f"[EASYOCR_TUNING] EasyOCR parameters updated")
    
    def _update_clahe_controls(self):
        """Enable/disable CLAHE controls based on enhance_contrast checkbox."""
        enabled = self.enhance_check.isChecked()
        self.clahe_clip_spin.setEnabled(enabled)
        self.clahe_grid_spin.setEnabled(enabled)
    
    def _update_sharpen_controls(self):
        """Enable/disable sharpen controls based on sharpen checkbox."""
        enabled = self.sharpen_check.isChecked()
        self.sharpen_strength_spin.setEnabled(enabled)
    
    def _update_upscale_controls(self):
        """Enable/disable upscale controls based on upscale checkbox."""
        enabled = self.upscale_check.isChecked()
        self.upscale_factor_spin.setEnabled(enabled)
        self.upscale_to_size_check.setEnabled(enabled)
        self.upscale_target_spin.setEnabled(enabled and self.upscale_to_size_check.isChecked())
    
    # ========== CORE PROCESSING METHODS ==========
    def _run_easyocr_analysis(self):
        """Run combined inference and EasyOCR analysis."""
        if self.current_frame is None:
            print("[EASYOCR_TUNING] No frame loaded")
            return
        
        try:
            # Step 1: Run YOLO detection
            detections = run_inference(self.current_frame)
            
            # Filter for person detections (including various person-related classes)
            person_detections = []
            for det in detections:
                class_name = det.get('class_name', '').lower()
                # Check for person, player, or similar classes
                if 'person' in class_name or 'player' in class_name or class_name == 'person':
                    person_detections.append(det)
            
            self.current_detections = person_detections
            
            if not person_detections:
                # Clear crops display and show message
                self._clear_crops_display()
                # Still display frame with no detections
                self._display_frame(self.current_frame)
                return
            
            # Step 2: Extract crops from detections
            self._extract_crops_from_detections()
            
            # Step 3: Run EasyOCR if crops available
            if self.current_crops:
                self._run_easyocr_on_crops()
            else:
                # Clear crops display and show frame with detections only
                self._clear_crops_display()
                annotated_frame = self._draw_detections(self.current_frame.copy())
                self._display_frame(annotated_frame)
            
            print(f"[EASYOCR_TUNING] Analysis complete: {len(person_detections)} detections, {len(self.current_crops)} crops")
            
        except Exception as e:
            # Clear crops display and show error
            self._clear_crops_display()
            error_msg = f"Analysis error: {str(e)}"
            print(f"[EASYOCR_TUNING] {error_msg}")
            traceback.print_exc()
    
    def _extract_crops_from_detections(self):
        """Extract crops from person detections."""
        if self.current_frame is None or not self.current_detections:
            return
        
        self.current_crops.clear()
        
        for i, detection in enumerate(self.current_detections):
            bbox = detection['bbox']
            # bbox format from inference is [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            
            # Calculate width and height
            w, h = x2 - x1, y2 - y1
            
            # Apply crop fraction (crop from top)
            crop_fraction = self.preprocess_params['crop_top_fraction']
            crop_height = int(h * crop_fraction)
            
            # Adjust crop coordinates
            crop_y2 = y1 + crop_height
            
            # Ensure bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(self.current_frame.shape[1], int(x2))
            crop_y2 = min(self.current_frame.shape[0], crop_y2)
            
            if x2 > x1 and crop_y2 > y1:
                crop = self.current_frame[y1:crop_y2, x1:x2]
                
                # Apply preprocessing
                processed_crop = self._preprocess_crop(crop)
                
                self.current_crops.append({
                    'detection_idx': i,
                    'bbox': bbox,
                    'crop_bbox': [x1, y1, x2 - x1, crop_y2 - y1],
                    'original_crop': crop,
                    'processed_crop': processed_crop
                })
        
        print(f"[EASYOCR_TUNING] Extracted {len(self.current_crops)} crops")
    
    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Apply preprocessing to a crop."""
        processed = crop.copy()
        
        # Resize (absolute takes priority over factor)
        abs_width = self.preprocess_params['resize_absolute_width']
        abs_height = self.preprocess_params['resize_absolute_height']
        resize_factor = self.preprocess_params['resize_factor']
        
        if abs_width > 0 and abs_height > 0:
            # Absolute resize
            processed = cv2.resize(processed, (abs_width, abs_height))
        elif abs_width > 0:
            # Absolute width, maintain aspect ratio
            current_height, current_width = processed.shape[:2]
            new_height = int(current_height * abs_width / current_width)
            processed = cv2.resize(processed, (abs_width, new_height))
        elif abs_height > 0:
            # Absolute height, maintain aspect ratio
            current_height, current_width = processed.shape[:2]
            new_width = int(current_width * abs_height / current_height)
            processed = cv2.resize(processed, (new_width, abs_height))
        elif resize_factor != 1.0:
            # Factor-based resize
            new_height = int(processed.shape[0] * resize_factor)
            new_width = int(processed.shape[1] * resize_factor)
            processed = cv2.resize(processed, (new_width, new_height))
        
        # Color mode conversion
        if self.preprocess_params['bw_mode']:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        elif not self.preprocess_params.get('colour_mode', False):
            # Default grayscale processing
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Denoising
        if self.preprocess_params['denoise']:
            if len(processed.shape) == 3:
                processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
            else:
                processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
        
        # Bilateral filter
        # Contrast and brightness
        alpha = self.preprocess_params['contrast_alpha']
        beta = self.preprocess_params['brightness_beta']
        if alpha != 1.0 or beta != 0:
            processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
        
        # Gaussian blur
        blur_kernel = self.preprocess_params['gaussian_blur']
        if blur_kernel > 0:
            # Ensure kernel size is odd
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)
        
        # CLAHE enhancement
        if self.preprocess_params['enhance_contrast']:
            clip_limit = self.preprocess_params['clahe_clip_limit']
            grid_size = self.preprocess_params['clahe_grid_size']
            
            if len(processed.shape) == 3:
                # Convert to LAB, apply CLAHE to L channel
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Grayscale
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
                processed = clahe.apply(processed)
        
        # Sharpening
        if self.preprocess_params['sharpen']:
            strength = self.preprocess_params['sharpen_strength']
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * strength
            kernel[1,1] = 1 + (8 * strength)  # Adjust center to maintain brightness
            processed = cv2.filter2D(processed, -1, kernel)
        
        # Upscaling
        if self.preprocess_params['upscale']:
            if self.preprocess_params['upscale_to_size']:
                # Upscale to fixed size
                target_size = self.preprocess_params['upscale_target_size']
                processed = cv2.resize(processed, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            else:
                # Upscale by factor
                factor = self.preprocess_params['upscale_factor']
                new_height = int(processed.shape[0] * factor)
                new_width = int(processed.shape[1] * factor)
                processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return processed
    
    def _run_easyocr_on_crops(self):
        """Run EasyOCR on all detected crops."""
        if not EASYOCR_AVAILABLE:
            self._clear_crops_display()
            return
        
        if not self.current_crops:
            self._clear_crops_display()
            return
        
        try:
            # Initialize EasyOCR reader with current parameters
            gpu = self.ocr_params['gpu']
            reader = easyocr.Reader(
                self.ocr_params['languages'], 
                gpu=gpu
            )
            
            results = []
            
            for crop_data in self.current_crops:
                processed_crop = crop_data['processed_crop']
                
                # Prepare parameters for readtext
                readtext_params = {
                    'text_threshold': self.ocr_params['text_threshold'],
                    'low_text': self.ocr_params['low_text'],
                    'link_threshold': self.ocr_params['link_threshold'],
                    'width_ths': self.ocr_params['width_ths'],
                    'height_ths': self.ocr_params['height_ths'],
                    'canvas_size': self.ocr_params['canvas_size'],
                    'mag_ratio': self.ocr_params['mag_ratio'],
                    'slope_ths': self.ocr_params['slope_ths'],
                    'ycenter_ths': self.ocr_params['ycenter_ths'],
                    'y_ths': self.ocr_params['y_ths'],
                    'x_ths': self.ocr_params['x_ths'],
                    'paragraph': self.ocr_params['paragraph'],
                    'adjust_contrast': self.ocr_params['adjust_contrast'],
                    'filter_ths': self.ocr_params['filter_ths'],
                    'batch_size': self.ocr_params['batch_size'],
                    'workers': self.ocr_params['workers'],
                    'decoder': self.ocr_params['decoder'],
                    'beamWidth': self.ocr_params['beamWidth'],
                    'detail': self.ocr_params['detail']
                }
                
                # Add character filtering if specified
                if self.ocr_params['allowlist']:
                    readtext_params['allowlist'] = self.ocr_params['allowlist']
                
                # Run EasyOCR
                ocr_results = reader.readtext(processed_crop, **readtext_params)
                
                # Process results - find best numeric text (likely jersey number)
                best_text = ""
                best_confidence = 0.0
                
                for bbox, text, confidence in ocr_results:
                    # Clean text and check if it's a valid jersey number
                    clean_text = ''.join(filter(str.isdigit, text))
                    if clean_text and confidence > best_confidence:
                        best_text = clean_text
                        best_confidence = confidence
                
                results.append({
                    'detection_idx': crop_data['detection_idx'],
                    'text': best_text,
                    'confidence': best_confidence,
                    'ocr_results': ocr_results
                })
            
            # Display results visually
            self._display_crops_with_results(results)
            
            # Draw results on frame
            if self.current_frame is not None:
                annotated_frame = self._draw_ocr_results(self.current_frame.copy(), results)
                self._display_frame(annotated_frame)
            
            print(f"[EASYOCR_TUNING] EasyOCR complete: {len(results)} results")
            
        except Exception as e:
            # Clear crops display and show error
            self._clear_crops_display()
            error_msg = f"EasyOCR error: {str(e)}"
            print(f"[EASYOCR_TUNING] {error_msg}")
            traceback.print_exc()
    
    # ========== VISUALIZATION METHODS ==========
    def _display_frame(self, frame: np.ndarray):
        """Display a frame in the video label."""
        if frame is None:
            return
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Create pixmap and scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        
        # Get the fixed display size (excluding borders)
        label_size = self.video_label.size()
        display_width = label_size.width() - 4  # Account for 2px border on each side
        display_height = label_size.height() - 4
        
        # Scale pixmap to fit within the fixed dimensions
        scaled_pixmap = pixmap.scaled(
            display_width, display_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection bounding boxes on frame."""
        for i, detection in enumerate(self.current_detections):
            bbox = detection['bbox']
            confidence = detection.get('confidence', 0.0)
            
            # bbox format is [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw detection info
            label = f"Person {i+1}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def _display_crops_with_results(self, results: List[Dict]):
        """Display preprocessed crops with OCR results visually."""
        # Clear existing crop displays
        self._clear_crops_display()
        
        if not self.current_crops or not results:
            return
        
        # Calculate grid layout (max 5 columns for optimal space usage with larger widgets)
        max_cols = 5
        num_crops = len(self.current_crops)
        cols = min(num_crops, max_cols)
        rows = (num_crops + cols - 1) // cols
        
        for i, (crop_data, result) in enumerate(zip(self.current_crops, results)):
            row = i // cols
            col = i % cols
            
            # Create crop display widget
            crop_widget = self._create_crop_display(crop_data, result, i + 1)
            self.crops_layout.addWidget(crop_widget, row, col)
        
        print(f"[EASYOCR_TUNING] Displayed {num_crops} crops in {rows}x{cols} grid")
    
    def _create_crop_display(self, crop_data: Dict, result: Dict, crop_num: int) -> QWidget:
        """Create a widget to display a single crop with OCR result."""
        widget = QWidget()
        widget.setStyleSheet("QWidget { background-color: #2a2a2a; }")  # Dark background
        layout = QVBoxLayout()
        layout.setSpacing(2)  # Increased spacing for larger layout
        
        # Crop image display - increased size to fill more space
        crop_label = QLabel()
        crop_label.setAlignment(Qt.AlignCenter)
        crop_label.setFixedSize(160, 120)  # Increased from 100x60 to 160x120
        crop_label.setStyleSheet("border: 1px solid #555; background-color: #2a2a2a;")
        
        # Convert processed crop to QPixmap with OCR highlights
        processed_crop = crop_data['processed_crop']
        if processed_crop is not None and processed_crop.size > 0:
            # Create annotated version of the crop with OCR detections highlighted
            annotated_crop = self._draw_ocr_on_crop(processed_crop.copy(), result)
            
            # Convert BGR to RGB
            rgb_crop = cv2.cvtColor(annotated_crop, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_crop.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_crop.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit label
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                crop_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            crop_label.setPixmap(scaled_pixmap)
        else:
            crop_label.setText("Error")
        
        layout.addWidget(crop_label)
        
        # OCR result display
        text = result.get('text', '')
        confidence = result.get('confidence', 0.0)
        
        if text:
            # Color code based on confidence
            if confidence > 0.7:
                color = "#00ff00"  # Green for high confidence
            elif confidence > 0.4:
                color = "#ffa500"  # Orange for medium confidence
            else:
                color = "#ff6666"  # Light red for low confidence
            
            result_text = f"#{text}"
            conf_text = f"{confidence:.3f}"
        else:
            color = "#ff0000"  # Red for no detection
            result_text = "No text"
            conf_text = "0.000"
        
        # Result label - increased size for better visibility
        result_label = QLabel(result_text)
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-weight: bold;
                font-size: 12px;
                background-color: #1a1a1a;
                border: 1px solid {color};
                border-radius: 3px;
                padding: 2px;
            }}
        """)
        layout.addWidget(result_label)
        
        # Confidence label - increased size
        conf_label = QLabel(f"Conf: {conf_text}")
        conf_label.setAlignment(Qt.AlignCenter)
        conf_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 10px;
            }
        """)
        layout.addWidget(conf_label)
        
        # Crop number label - increased size
        num_label = QLabel(f"Crop {crop_num}")
        num_label.setAlignment(Qt.AlignCenter)
        num_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 9px;
            }
        """)
        layout.addWidget(num_label)
        
        widget.setLayout(layout)
        widget.setFixedSize(170, 180)  # Increased from 110x120 to 170x180
        return widget
    
    def _draw_ocr_on_crop(self, crop_image: np.ndarray, result: Dict) -> np.ndarray:
        """Draw OCR detection boxes and text on the crop image."""
        if 'ocr_results' not in result or not result['ocr_results']:
            return crop_image
        
        # Get image dimensions
        img_height, img_width = crop_image.shape[:2]
        
        # Draw each OCR detection
        for bbox, text, confidence in result['ocr_results']:
            if confidence < 0.1:  # Skip very low confidence detections
                continue
                
            # Convert bbox coordinates (EasyOCR returns normalized coordinates)
            if isinstance(bbox[0], (list, tuple)):
                # Polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                points = np.array(bbox, dtype=np.int32)
                points[:, 0] = np.clip(points[:, 0], 0, img_width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, img_height - 1)
                
                # Draw polygon outline
                color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)  # Green for high conf, orange for low
                cv2.polylines(crop_image, [points], True, color, 2)  # Increased thickness from 1 to 2
                
                # Fill polygon with semi-transparent overlay
                overlay = crop_image.copy()
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(crop_image, 0.8, overlay, 0.2, 0, crop_image)
                
                # Draw text near the detection
                if text.strip() and any(c.isdigit() for c in text):  # Only show if contains digits
                    text_pos = (int(np.min(points[:, 0])), int(np.min(points[:, 1]) - 3))
                    text_pos = (max(0, text_pos[0]), max(12, text_pos[1]))
                    
                    # Draw text background - increased font size
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]  # Increased from 0.3 to 0.5
                    cv2.rectangle(crop_image, text_pos, 
                                (text_pos[0] + text_size[0], text_pos[1] - text_size[1] - 3), 
                                color, -1)
                    cv2.putText(crop_image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Increased from 0.3 to 0.5
        
        return crop_image
    
    def _clear_crops_display(self):
        """Clear all crop displays from the grid."""
        # Remove all widgets from the grid layout
        while self.crops_layout.count():
            child = self.crops_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def _draw_ocr_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Draw OCR results on frame."""
        for result in results:
            det_idx = result['detection_idx']
            text = result['text']
            confidence = result['confidence']
            
            if det_idx < len(self.current_detections):
                detection = self.current_detections[det_idx]
                bbox = detection['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Draw detection box in blue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw OCR result
                if text:
                    ocr_label = f"'{text}' ({confidence:.3f})"
                    color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)  # Green if high conf, orange if low
                else:
                    ocr_label = "No text"
                    color = (0, 0, 255)  # Red for no detection
                
                # Draw label
                label_size = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y2), (x1 + label_size[0] + 10, y2 + label_size[1] + 10), color, -1)
                cv2.putText(frame, ocr_label, (x1 + 5, y2 + label_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    # ========== CONFIGURATION METHODS ==========
    def _load_parameters_from_config(self):
        """Load parameters from configuration."""
        try:
            # Load user.yaml overrides if they exist
            user_config = self._load_user_config()
            
            print(f"[EASYOCR_TUNING] Loading parameters from config...")
            print(f"[EASYOCR_TUNING] User config found: {bool(user_config)}")
            if user_config:
                print(f"[EASYOCR_TUNING] User config keys: {list(user_config.keys())}")
                if 'player_id' in user_config:
                    print(f"[EASYOCR_TUNING] Player ID config keys: {list(user_config['player_id'].keys())}")
            
            # Load preprocessing parameters (using user overrides if available)
            user_preprocess = user_config.get('player_id', {}).get('preprocessing', {})
            print(f"[EASYOCR_TUNING] User preprocessing config: {user_preprocess}")
            
            self.preprocess_params['crop_top_fraction'] = user_preprocess.get('crop_top_fraction', 
                get_setting('player_id.preprocessing.crop_top_fraction', 0.33))
            self.preprocess_params['contrast_alpha'] = user_preprocess.get('contrast_alpha',
                get_setting('player_id.preprocessing.contrast_alpha', 1.0))
            self.preprocess_params['brightness_beta'] = user_preprocess.get('brightness_beta',
                get_setting('player_id.preprocessing.brightness_beta', 0))
            self.preprocess_params['gaussian_blur'] = user_preprocess.get('gaussian_blur',
                get_setting('player_id.preprocessing.gaussian_blur', 13))
            self.preprocess_params['enhance_contrast'] = user_preprocess.get('enhance_contrast',
                get_setting('player_id.preprocessing.enhance_contrast', True))
            self.preprocess_params['clahe_clip_limit'] = user_preprocess.get('clahe_clip_limit',
                get_setting('player_id.preprocessing.clahe_clip_limit', 2.0))
            self.preprocess_params['clahe_grid_size'] = user_config.get('player_id', {}).get('preprocessing', {}).get('clahe_grid_size',
                get_setting('player_id.preprocessing.clahe_grid_size', 8))
            self.preprocess_params['sharpen'] = user_config.get('player_id', {}).get('preprocessing', {}).get('sharpen',
                get_setting('player_id.preprocessing.sharpen', True))
            self.preprocess_params['sharpen_strength'] = user_config.get('player_id', {}).get('preprocessing', {}).get('sharpen_strength',
                get_setting('player_id.preprocessing.sharpen_strength', 0.05))
            self.preprocess_params['upscale'] = user_config.get('player_id', {}).get('preprocessing', {}).get('upscale',
                get_setting('player_id.preprocessing.upscale', True))
            self.preprocess_params['upscale_factor'] = user_config.get('player_id', {}).get('preprocessing', {}).get('upscale_factor',
                get_setting('player_id.preprocessing.upscale_factor', 3.0))
            self.preprocess_params['upscale_to_size'] = user_config.get('player_id', {}).get('preprocessing', {}).get('upscale_to_size',
                get_setting('player_id.preprocessing.upscale_to_size', True))
            self.preprocess_params['upscale_target_size'] = user_config.get('player_id', {}).get('preprocessing', {}).get('upscale_target_size',
                get_setting('player_id.preprocessing.upscale_target_size', 256))
            self.preprocess_params['colour_mode'] = user_config.get('player_id', {}).get('preprocessing', {}).get('colour_mode',
                get_setting('player_id.preprocessing.colour_mode', False))
            self.preprocess_params['bw_mode'] = user_config.get('player_id', {}).get('preprocessing', {}).get('bw_mode',
                get_setting('player_id.preprocessing.bw_mode', True))
            self.preprocess_params['resize_factor'] = user_config.get('player_id', {}).get('preprocessing', {}).get('resize_factor',
                get_setting('player_id.preprocessing.resize_factor', 1.0))
            self.preprocess_params['resize_absolute_width'] = user_config.get('player_id', {}).get('preprocessing', {}).get('resize_absolute_width',
                get_setting('player_id.preprocessing.resize_absolute_width', 0))
            self.preprocess_params['resize_absolute_height'] = user_config.get('player_id', {}).get('preprocessing', {}).get('resize_absolute_height',
                get_setting('player_id.preprocessing.resize_absolute_height', 0))
            self.preprocess_params['denoise'] = user_config.get('player_id', {}).get('preprocessing', {}).get('denoise',
                get_setting('player_id.preprocessing.denoise', False))
            
            # Load OCR parameters (using user overrides if available)
            if EASYOCR_AVAILABLE:
                ocr_config = user_config.get('player_id', {}).get('easyocr', {})
                print(f"[EASYOCR_TUNING] User OCR config: {ocr_config}")
                
                self.ocr_params['text_threshold'] = ocr_config.get('text_threshold', get_setting('player_id.easyocr.text_threshold', 0.7))
                self.ocr_params['low_text'] = ocr_config.get('low_text', get_setting('player_id.easyocr.low_text', 0.6))
                self.ocr_params['link_threshold'] = ocr_config.get('link_threshold', get_setting('player_id.easyocr.link_threshold', 0.4))
                self.ocr_params['width_ths'] = ocr_config.get('width_ths', get_setting('player_id.easyocr.width_ths', 0.4))
                self.ocr_params['height_ths'] = ocr_config.get('height_ths', get_setting('player_id.easyocr.height_ths', 0.7))
                self.ocr_params['x_ths'] = ocr_config.get('x_ths', get_setting('player_id.easyocr.x_ths', 1.0))
                self.ocr_params['y_ths'] = ocr_config.get('y_ths', get_setting('player_id.easyocr.y_ths', 0.5))
                self.ocr_params['ycenter_ths'] = ocr_config.get('ycenter_ths', get_setting('player_id.easyocr.ycenter_ths', 0.5))
                self.ocr_params['slope_ths'] = ocr_config.get('slope_ths', get_setting('player_id.easyocr.slope_ths', 0.1))
                self.ocr_params['canvas_size'] = ocr_config.get('canvas_size', get_setting('player_id.easyocr.canvas_size', 2560))
                self.ocr_params['mag_ratio'] = ocr_config.get('mag_ratio', get_setting('player_id.easyocr.mag_ratio', 2.0))
                self.ocr_params['adjust_contrast'] = ocr_config.get('adjust_contrast', get_setting('player_id.easyocr.adjust_contrast', 0.5))
                self.ocr_params['filter_ths'] = ocr_config.get('filter_ths', get_setting('player_id.easyocr.filter_ths', 0.003))
                self.ocr_params['workers'] = ocr_config.get('workers', get_setting('player_id.easyocr.workers', 0))
                self.ocr_params['batch_size'] = ocr_config.get('batch_size', get_setting('player_id.easyocr.batch_size', 1))
                self.ocr_params['beamWidth'] = ocr_config.get('beamWidth', get_setting('player_id.easyocr.beamWidth', 5))
                self.ocr_params['gpu'] = ocr_config.get('gpu', get_setting('player_id.easyocr.gpu', True))
                self.ocr_params['paragraph'] = ocr_config.get('paragraph', get_setting('player_id.easyocr.paragraph', False))
                self.ocr_params['detail'] = ocr_config.get('detail', get_setting('player_id.easyocr.detail', 1))
                self.ocr_params['allowlist'] = ocr_config.get('allowlist', get_setting('player_id.easyocr.allowlist', '0123456789'))
            
            # Update UI controls with loaded values
            self._update_controls_from_params()
            
            # Debug output to verify parameter loading
            print(f"[EASYOCR_TUNING] Final parameter values:")
            print(f"  crop_top_fraction: {self.preprocess_params['crop_top_fraction']}")
            print(f"  contrast_alpha: {self.preprocess_params['contrast_alpha']}")
            print(f"  gaussian_blur: {self.preprocess_params['gaussian_blur']}")
            if EASYOCR_AVAILABLE:
                print(f"  text_threshold: {self.ocr_params['text_threshold']}")
                print(f"  low_text: {self.ocr_params['low_text']}")
            
            print("[EASYOCR_TUNING] Parameters loaded from configuration")
        except Exception as e:
            print(f"[EASYOCR_TUNING] Error loading parameters from config: {e}")
            traceback.print_exc()
            # Continue with default values if config loading fails
    
    def _load_user_config(self) -> Dict[str, Any]:
        """Load user.yaml configuration if it exists.
        
        Returns:
            User configuration dictionary or empty dict if not found
        """
        try:
            # Find project root
            current_dir = Path(__file__).parent
            project_root = None
            
            for parent in current_dir.parents:
                if (parent / "configs").exists():
                    project_root = parent
                    break
            
            if project_root is None:
                return {}
            
            user_config_file = project_root / "configs" / "user.yaml"
            
            if not user_config_file.exists():
                return {}
            
            with open(user_config_file, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                
            if user_config:
                print(f"[EASYOCR_TUNING] User configuration loaded from: {user_config_file}")
                return user_config
            else:
                return {}
                
        except Exception as e:
            print(f"[EASYOCR_TUNING] Error loading user config: {e}")
            return {}
    
    def _update_controls_from_params(self):
        """Update UI controls with current parameter values."""
        # Temporarily disconnect signals to prevent triggering parameter updates
        self._disconnect_param_signals()
        
        try:
            # Preprocessing controls
            self.crop_fraction_spin.setValue(self.preprocess_params['crop_top_fraction'])
            self.contrast_spin.setValue(self.preprocess_params['contrast_alpha'])
            self.brightness_spin.setValue(self.preprocess_params['brightness_beta'])
            self.blur_spin.setValue(self.preprocess_params['gaussian_blur'])
            if hasattr(self, 'enhance_check'):
                self.enhance_check.setChecked(self.preprocess_params['enhance_contrast'])
            if hasattr(self, 'clahe_clip_spin'):
                self.clahe_clip_spin.setValue(self.preprocess_params['clahe_clip_limit'])
            if hasattr(self, 'clahe_grid_spin'):
                self.clahe_grid_spin.setValue(self.preprocess_params['clahe_grid_size'])
            if hasattr(self, 'sharpen_check'):
                self.sharpen_check.setChecked(self.preprocess_params['sharpen'])
            if hasattr(self, 'sharpen_strength_spin'):
                self.sharpen_strength_spin.setValue(self.preprocess_params['sharpen_strength'])
            if hasattr(self, 'upscale_check'):
                self.upscale_check.setChecked(self.preprocess_params['upscale'])
            if hasattr(self, 'upscale_factor_spin'):
                self.upscale_factor_spin.setValue(self.preprocess_params['upscale_factor'])  
            if hasattr(self, 'upscale_to_size_check'):
                self.upscale_to_size_check.setChecked(self.preprocess_params['upscale_to_size'])
            if hasattr(self, 'upscale_target_spin'):
                self.upscale_target_spin.setValue(self.preprocess_params['upscale_target_size'])
            if hasattr(self, 'colour_mode_check'):
                self.colour_mode_check.setChecked(self.preprocess_params['colour_mode'])
            if hasattr(self, 'bw_mode_check'):
                self.bw_mode_check.setChecked(self.preprocess_params['bw_mode'])
            self.resize_spin.setValue(self.preprocess_params['resize_factor'])
            self.resize_width_spin.setValue(self.preprocess_params['resize_absolute_width'])
            self.resize_height_spin.setValue(self.preprocess_params['resize_absolute_height'])
            self.denoise_check.setChecked(self.preprocess_params['denoise'])
            
            # OCR controls
            if EASYOCR_AVAILABLE:
                self.text_threshold_spin.setValue(self.ocr_params['text_threshold'])
                self.low_text_spin.setValue(self.ocr_params['low_text'])
                self.link_threshold_spin.setValue(self.ocr_params['link_threshold'])
                self.width_ths_spin.setValue(self.ocr_params['width_ths'])
                self.height_ths_spin.setValue(self.ocr_params['height_ths'])
                self.canvas_size_spin.setValue(self.ocr_params['canvas_size'])
                self.mag_ratio_spin.setValue(self.ocr_params['mag_ratio'])
                self.gpu_check.setChecked(self.ocr_params['gpu'])
                
                # Extended controls (check if they exist)
                if hasattr(self, 'x_ths_spin'):
                    self.x_ths_spin.setValue(self.ocr_params['x_ths'])
                if hasattr(self, 'y_ths_spin'):
                    self.y_ths_spin.setValue(self.ocr_params['y_ths'])
                if hasattr(self, 'ycenter_ths_spin'):
                    self.ycenter_ths_spin.setValue(self.ocr_params['ycenter_ths'])
                if hasattr(self, 'slope_ths_spin'):
                    self.slope_ths_spin.setValue(self.ocr_params['slope_ths'])
                if hasattr(self, 'adjust_contrast_spin'):
                    self.adjust_contrast_spin.setValue(self.ocr_params['adjust_contrast'])
                if hasattr(self, 'filter_ths_spin'):
                    self.filter_ths_spin.setValue(self.ocr_params['filter_ths'])
                if hasattr(self, 'workers_spin'):
                    self.workers_spin.setValue(self.ocr_params['workers'])
                if hasattr(self, 'batch_size_spin'):
                    self.batch_size_spin.setValue(self.ocr_params['batch_size'])
                if hasattr(self, 'beam_width_spin'):
                    self.beam_width_spin.setValue(self.ocr_params['beamWidth'])
                if hasattr(self, 'paragraph_check'):
                    self.paragraph_check.setChecked(self.ocr_params['paragraph'])
                if hasattr(self, 'detail_spin'):
                    self.detail_spin.setValue(self.ocr_params['detail'])
                if hasattr(self, 'allowlist_edit'):
                    self.allowlist_edit.setText(self.ocr_params['allowlist'] or '')
        
        finally:
            # Reconnect signals
            self._connect_param_signals()
    
    def _disconnect_param_signals(self):
        """Temporarily disconnect parameter change signals."""
        # Disconnect preprocessing signals - spinboxes
        self.crop_fraction_spin.valueChanged.disconnect()
        self.contrast_spin.valueChanged.disconnect()
        self.brightness_spin.valueChanged.disconnect()
        self.blur_spin.valueChanged.disconnect()
        self.resize_spin.valueChanged.disconnect()
        self.resize_width_spin.valueChanged.disconnect()
        self.resize_height_spin.valueChanged.disconnect()
        
        # Disconnect preprocessing signals - checkboxes
        self.denoise_check.stateChanged.disconnect()
        
        # Disconnect optional preprocessing controls
        if hasattr(self, 'enhance_check'):
            self.enhance_check.stateChanged.disconnect()
        if hasattr(self, 'clahe_clip_spin'):
            self.clahe_clip_spin.valueChanged.disconnect()
        if hasattr(self, 'clahe_grid_spin'):
            self.clahe_grid_spin.valueChanged.disconnect()
        if hasattr(self, 'sharpen_check'):
            self.sharpen_check.stateChanged.disconnect()
        if hasattr(self, 'sharpen_strength_spin'):
            self.sharpen_strength_spin.valueChanged.disconnect()
        if hasattr(self, 'upscale_check'):
            self.upscale_check.stateChanged.disconnect()
        if hasattr(self, 'upscale_factor_spin'):
            self.upscale_factor_spin.valueChanged.disconnect()
        if hasattr(self, 'upscale_to_size_check'):
            self.upscale_to_size_check.stateChanged.disconnect()
        if hasattr(self, 'upscale_target_spin'):
            self.upscale_target_spin.valueChanged.disconnect()
        if hasattr(self, 'colour_mode_check'):
            self.colour_mode_check.stateChanged.disconnect()
        if hasattr(self, 'bw_mode_check'):
            self.bw_mode_check.stateChanged.disconnect()
        
        # Disconnect OCR signals if available
        if EASYOCR_AVAILABLE:
            self.text_threshold_spin.valueChanged.disconnect()
            self.low_text_spin.valueChanged.disconnect()
            self.link_threshold_spin.valueChanged.disconnect()
            self.width_ths_spin.valueChanged.disconnect()
            self.height_ths_spin.valueChanged.disconnect()
            self.canvas_size_spin.valueChanged.disconnect()
            self.mag_ratio_spin.valueChanged.disconnect()
            self.gpu_check.stateChanged.disconnect()
            
            # Disconnect optional OCR controls
            if hasattr(self, 'x_ths_spin'):
                self.x_ths_spin.valueChanged.disconnect()
            if hasattr(self, 'y_ths_spin'):
                self.y_ths_spin.valueChanged.disconnect()
            if hasattr(self, 'ycenter_ths_spin'):
                self.ycenter_ths_spin.valueChanged.disconnect()
            if hasattr(self, 'slope_ths_spin'):
                self.slope_ths_spin.valueChanged.disconnect()
            if hasattr(self, 'adjust_contrast_spin'):
                self.adjust_contrast_spin.valueChanged.disconnect()
            if hasattr(self, 'filter_ths_spin'):
                self.filter_ths_spin.valueChanged.disconnect()
            if hasattr(self, 'workers_spin'):
                self.workers_spin.valueChanged.disconnect()
            if hasattr(self, 'batch_size_spin'):
                self.batch_size_spin.valueChanged.disconnect()
            if hasattr(self, 'beam_width_spin'):
                self.beam_width_spin.valueChanged.disconnect()
            if hasattr(self, 'paragraph_check'):
                self.paragraph_check.stateChanged.disconnect()
            if hasattr(self, 'detail_spin'):
                self.detail_spin.valueChanged.disconnect()
            if hasattr(self, 'allowlist_edit'):
                self.allowlist_edit.textChanged.disconnect()
    
    def _connect_param_signals(self):
        """Reconnect parameter change signals."""
        # Reconnect preprocessing signals - spinboxes
        self.crop_fraction_spin.valueChanged.connect(self._on_preprocess_param_changed)
        self.contrast_spin.valueChanged.connect(self._on_preprocess_param_changed)
        self.brightness_spin.valueChanged.connect(self._on_preprocess_param_changed)
        self.blur_spin.valueChanged.connect(self._on_preprocess_param_changed)
        self.resize_spin.valueChanged.connect(self._on_preprocess_param_changed)
        self.resize_width_spin.valueChanged.connect(self._on_preprocess_param_changed)
        self.resize_height_spin.valueChanged.connect(self._on_preprocess_param_changed)
        
        # Reconnect preprocessing signals - checkboxes
        self.denoise_check.stateChanged.connect(self._on_preprocess_param_changed)
        
        # Reconnect optional preprocessing controls
        if hasattr(self, 'enhance_check'):
            self.enhance_check.stateChanged.connect(self._on_preprocess_param_changed)
        if hasattr(self, 'clahe_clip_spin'):
            self.clahe_clip_spin.valueChanged.connect(self._on_preprocess_param_changed)
        if hasattr(self, 'clahe_grid_spin'):
            self.clahe_grid_spin.valueChanged.connect(self._on_preprocess_param_changed)
        if hasattr(self, 'sharpen_check'):
            self.sharpen_check.stateChanged.connect(self._on_preprocess_param_changed)
        if hasattr(self, 'sharpen_strength_spin'):
            self.sharpen_strength_spin.valueChanged.connect(self._on_preprocess_param_changed)
        if hasattr(self, 'upscale_check'):
            self.upscale_check.stateChanged.connect(self._on_preprocess_param_changed)
        if hasattr(self, 'upscale_factor_spin'):
            self.upscale_factor_spin.valueChanged.connect(self._on_preprocess_param_changed)
        if hasattr(self, 'upscale_to_size_check'):
            self.upscale_to_size_check.stateChanged.connect(self._on_preprocess_param_changed)
        if hasattr(self, 'upscale_target_spin'):
            self.upscale_target_spin.valueChanged.connect(self._on_preprocess_param_changed)
        if hasattr(self, 'colour_mode_check'):
            self.colour_mode_check.stateChanged.connect(self._on_preprocess_param_changed)
        if hasattr(self, 'bw_mode_check'):
            self.bw_mode_check.stateChanged.connect(self._on_preprocess_param_changed)
        
        # Reconnect OCR signals if available
        if EASYOCR_AVAILABLE:
            self.text_threshold_spin.valueChanged.connect(self._on_ocr_param_changed)
            self.low_text_spin.valueChanged.connect(self._on_ocr_param_changed)
            self.link_threshold_spin.valueChanged.connect(self._on_ocr_param_changed)
            self.width_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
            self.height_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
            self.canvas_size_spin.valueChanged.connect(self._on_ocr_param_changed)
            self.mag_ratio_spin.valueChanged.connect(self._on_ocr_param_changed)
            self.gpu_check.stateChanged.connect(self._on_ocr_param_changed)
            
            # Reconnect optional OCR controls
            if hasattr(self, 'x_ths_spin'):
                self.x_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
            if hasattr(self, 'y_ths_spin'):
                self.y_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
            if hasattr(self, 'ycenter_ths_spin'):
                self.ycenter_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
            if hasattr(self, 'slope_ths_spin'):
                self.slope_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
            if hasattr(self, 'adjust_contrast_spin'):
                self.adjust_contrast_spin.valueChanged.connect(self._on_ocr_param_changed)
            if hasattr(self, 'filter_ths_spin'):
                self.filter_ths_spin.valueChanged.connect(self._on_ocr_param_changed)
            if hasattr(self, 'workers_spin'):
                self.workers_spin.valueChanged.connect(self._on_ocr_param_changed)
            if hasattr(self, 'batch_size_spin'):
                self.batch_size_spin.valueChanged.connect(self._on_ocr_param_changed)
            if hasattr(self, 'beam_width_spin'):
                self.beam_width_spin.valueChanged.connect(self._on_ocr_param_changed)
            if hasattr(self, 'paragraph_check'):
                self.paragraph_check.stateChanged.connect(self._on_ocr_param_changed)
            if hasattr(self, 'detail_spin'):
                self.detail_spin.valueChanged.connect(self._on_ocr_param_changed)
            if hasattr(self, 'allowlist_edit'):
                self.allowlist_edit.textChanged.connect(self._on_ocr_param_changed)
    
    def _save_parameters_to_config(self):
        """Save current parameters to configuration file."""
        try:
            # Find project root and construct absolute path to user.yaml
            current_dir = Path(__file__).parent
            project_root = None
            
            for parent in current_dir.parents:
                if (parent / "configs").exists():
                    project_root = parent
                    break
            
            if project_root is None:
                print("[EASYOCR_TUNING] Error: Could not find project root with configs directory")
                return
            
            config_path = project_root / "configs" / "user.yaml"
            
            # Create config updates with all parameters
            config_updates = {
                'player_id': {
                    'preprocessing': {
                        'crop_top_fraction': self.preprocess_params['crop_top_fraction'],
                        'contrast_alpha': self.preprocess_params['contrast_alpha'],
                        'brightness_beta': self.preprocess_params['brightness_beta'],
                        'gaussian_blur': self.preprocess_params['gaussian_blur'],
                        'resize_factor': self.preprocess_params['resize_factor'],
                        'resize_absolute_width': self.preprocess_params['resize_absolute_width'],
                        'resize_absolute_height': self.preprocess_params['resize_absolute_height'],
                        'enhance_contrast': self.preprocess_params['enhance_contrast'],
                        'clahe_clip_limit': self.preprocess_params['clahe_clip_limit'],
                        'clahe_grid_size': self.preprocess_params['clahe_grid_size'],
                        'denoise': self.preprocess_params['denoise'],
                        'sharpen': self.preprocess_params['sharpen'],
                        'sharpen_strength': self.preprocess_params['sharpen_strength'],
                        'upscale': self.preprocess_params['upscale'],
                        'upscale_factor': self.preprocess_params['upscale_factor'],
                        'upscale_to_size': self.preprocess_params['upscale_to_size'],
                        'upscale_target_size': self.preprocess_params['upscale_target_size'],
                        'colour_mode': self.preprocess_params['colour_mode'],
                        'bw_mode': self.preprocess_params['bw_mode']
                    }
                }
            }
            
            if EASYOCR_AVAILABLE:
                config_updates['player_id']['easyocr'] = {
                    'text_threshold': self.ocr_params['text_threshold'],
                    'low_text': self.ocr_params['low_text'],
                    'link_threshold': self.ocr_params['link_threshold'],
                    'width_ths': self.ocr_params['width_ths'],
                    'height_ths': self.ocr_params['height_ths'],
                    'x_ths': self.ocr_params['x_ths'],
                    'y_ths': self.ocr_params['y_ths'],
                    'ycenter_ths': self.ocr_params['ycenter_ths'],
                    'slope_ths': self.ocr_params['slope_ths'],
                    'canvas_size': self.ocr_params['canvas_size'],
                    'mag_ratio': self.ocr_params['mag_ratio'],
                    'adjust_contrast': self.ocr_params['adjust_contrast'],
                    'filter_ths': self.ocr_params['filter_ths'],
                    'workers': self.ocr_params['workers'],
                    'batch_size': self.ocr_params['batch_size'],
                    'beamWidth': self.ocr_params['beamWidth'],
                    'gpu': self.ocr_params['gpu'],
                    'paragraph': self.ocr_params['paragraph'],
                    'detail': self.ocr_params['detail'],
                    'allowlist': self.ocr_params['allowlist']
                }
            
            # Ensure configs directory exists
            config_path.parent.mkdir(exist_ok=True)
            
            # Load existing config or create new
            existing_config = {}
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        existing_config = yaml.safe_load(f) or {}
                except Exception as e:
                    print(f"[EASYOCR_TUNING] Warning: Could not read existing config: {e}")
                    existing_config = {}
            
            # Merge updates
            self._deep_update(existing_config, config_updates)
            
            # Save updated config
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_config, f, default_flow_style=False, indent=2)
            
            print(f"[EASYOCR_TUNING] Parameters saved successfully to {config_path}")
            print(f"[EASYOCR_TUNING]   - Preprocessing parameters: {len(config_updates['player_id']['preprocessing'])} saved")
            if EASYOCR_AVAILABLE and 'easyocr' in config_updates['player_id']:
                print(f"[EASYOCR_TUNING]   - EasyOCR parameters: {len(config_updates['player_id']['easyocr'])} saved")
            
            # Optional: Add visual feedback to user (you could replace this with a status bar message)
            # For now, console output should be sufficient for debugging
            
        except Exception as e:
            print(f"[EASYOCR_TUNING] Error saving parameters to config: {e}")
            traceback.print_exc()
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
