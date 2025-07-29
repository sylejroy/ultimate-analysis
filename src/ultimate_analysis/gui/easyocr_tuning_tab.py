"""EasyOCR Tuning Tab for Ultimate Analysis GUI.

This module provides a specialized interface for tuning EasyOCR parameters
on detected player bounding boxes for optimal jersey number recognition.
"""

import os
import cv2
import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, 
    QPushButton, QSlider, QListWidgetItem, QGroupBox,
    QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QSplitter, QScrollArea, QGridLayout
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
        
        # EasyOCR parameters (with defaults)
        self.ocr_params = {
            'languages': ['en'],
            'gpu': True,
            'width_ths': 0.7,
            'height_ths': 0.7,
            'paragraph': False,
            'adjust_contrast': 0.5,
            'filter_ths': 0.003,
            'text_threshold': 0.7,
            'low_text': 0.4,
            'link_threshold': 0.4,
            'canvas_size': 2560,
            'mag_ratio': 1.0,
            'slope_ths': 0.1,
            'ycenter_ths': 0.5,
            'height_ths': 0.7,
            'width_ths': 0.7,
            'y_ths': 0.5,
            'x_ths': 1.0,
        }
        
        # Preprocessing parameters
        self.preprocess_params = {
            'crop_top_fraction': 0.33,  # Use top third of detection
            'contrast_alpha': 1.0,     # Contrast adjustment
            'brightness_beta': 0,      # Brightness adjustment
            'gaussian_blur': 0,        # Blur kernel size (0 = no blur)
            'resize_factor': 1.0,      # Resize factor
            'enhance_contrast': False,  # CLAHE enhancement
        }
        
        # Initialize UI
        self._init_ui()
        self._load_videos()
        self._load_parameters_from_config()
        
        print("[EASYOCR_TUNING] EasyOCR Tuning Tab initialized")
    
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
        
        # Set splitter proportions (40% left, 60% right)
        splitter.setSizes([400, 600])
        
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
        
        # Preprocessing parameters
        preprocess_group = QGroupBox("Preprocessing Parameters")
        preprocess_layout = QFormLayout()
        
        # Create parameter controls
        self._create_preprocessing_controls(preprocess_layout)
        
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
        # EasyOCR parameters
        ocr_group = QGroupBox("EasyOCR Parameters")
        ocr_layout = QFormLayout()
        
        # Create OCR parameter controls
        self._create_ocr_controls(ocr_layout)
        
        ocr_group.setLayout(ocr_layout)
        layout.addWidget(ocr_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        load_button = QPushButton("Load from Config")
        load_button.clicked.connect(self._load_parameters_from_config)
        load_button.setToolTip("Load parameters from default.yaml")
        button_layout.addWidget(load_button)
        
        save_button = QPushButton("Save to Config")
        save_button.clicked.connect(self._save_parameters_to_config)
        save_button.setToolTip("Save current parameters to default.yaml")
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
        
        # Video display area
        self.video_label = QLabel("No video selected")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(300)
        self.video_label.setMaximumHeight(500)  # Fixed maximum height
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #1a1a1a;
                color: #999;
                font-size: 14px;
            }
        """)
        self.video_label.setScaledContents(False)  # Don't scale contents
        layout.addWidget(self.video_label)
        
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
        
        layout.addLayout(slider_layout)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.run_analysis_button = QPushButton("Run EasyOCR Analysis")
        self.run_analysis_button.clicked.connect(self._run_easyocr_analysis)
        self.run_analysis_button.setToolTip("Run detection + EasyOCR analysis on current frame")
        self.run_analysis_button.setEnabled(False)  # Disabled until video loads
        control_layout.addWidget(self.run_analysis_button)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Crops display area
        crops_group = QGroupBox("Detected Crops & OCR Results")
        crops_layout = QVBoxLayout()
        
        # Scroll area for crops
        self.crops_scroll = QScrollArea()
        self.crops_scroll.setWidgetResizable(True)
        self.crops_scroll.setMaximumHeight(250)
        self.crops_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.crops_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container for crop displays
        self.crops_container = QWidget()
        self.crops_layout = QGridLayout()
        self.crops_container.setLayout(self.crops_layout)
        self.crops_scroll.setWidget(self.crops_container)
        
        crops_layout.addWidget(self.crops_scroll)
        crops_group.setLayout(crops_layout)
        layout.addWidget(crops_group)
        
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
        self.blur_spin.setRange(0, 15)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(self.preprocess_params['gaussian_blur'])
        self.blur_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("Gaussian Blur:", self.blur_spin)
        
        # Resize factor
        self.resize_spin = QDoubleSpinBox()
        self.resize_spin.setRange(0.5, 4.0)
        self.resize_spin.setSingleStep(0.1)
        self.resize_spin.setValue(self.preprocess_params['resize_factor'])
        self.resize_spin.valueChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("Resize Factor:", self.resize_spin)
        
        # CLAHE enhancement
        self.enhance_check = QCheckBox()
        self.enhance_check.setChecked(self.preprocess_params['enhance_contrast'])
        self.enhance_check.stateChanged.connect(self._on_preprocess_param_changed)
        layout.addRow("Enhance Contrast:", self.enhance_check)
    
    def _create_ocr_controls(self, layout: QFormLayout):
        """Create EasyOCR parameter controls."""
        if not EASYOCR_AVAILABLE:
            layout.addRow(QLabel("EasyOCR not available"))
            return
        
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
        
        # GPU usage
        self.gpu_check = QCheckBox()
        self.gpu_check.setChecked(self.ocr_params['gpu'])
        self.gpu_check.stateChanged.connect(self._on_ocr_param_changed)
        layout.addRow("Use GPU:", self.gpu_check)
    
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
    
    # Event Handlers
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
        self.preprocess_params['resize_factor'] = self.resize_spin.value()
        self.preprocess_params['enhance_contrast'] = self.enhance_check.isChecked()
        
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
        self.ocr_params['canvas_size'] = self.canvas_size_spin.value()
        self.ocr_params['mag_ratio'] = self.mag_ratio_spin.value()
        self.ocr_params['gpu'] = self.gpu_check.isChecked()
        
        print(f"[EASYOCR_TUNING] EasyOCR parameters updated")
    
    # Core Processing Methods
    def _run_easyocr_analysis(self):
        """Run combined inference and EasyOCR analysis."""
        if self.current_frame is None:
            self.results_text.setText("No frame loaded")
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
            import traceback
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
        
        # Resize
        resize_factor = self.preprocess_params['resize_factor']
        if resize_factor != 1.0:
            new_height = int(processed.shape[0] * resize_factor)
            new_width = int(processed.shape[1] * resize_factor)
            processed = cv2.resize(processed, (new_width, new_height))
        
        # Contrast and brightness
        alpha = self.preprocess_params['contrast_alpha']
        beta = self.preprocess_params['brightness_beta']
        if alpha != 1.0 or beta != 0:
            processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
        
        # Gaussian blur
        blur_kernel = self.preprocess_params['gaussian_blur']
        if blur_kernel > 0:
            processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)
        
        # CLAHE enhancement
        if self.preprocess_params['enhance_contrast']:
            # Convert to LAB, apply CLAHE to L channel
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
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
            reader = easyocr.Reader(['en'], gpu=gpu)
            
            results = []
            
            for crop_data in self.current_crops:
                processed_crop = crop_data['processed_crop']
                
                # Run EasyOCR
                ocr_results = reader.readtext(
                    processed_crop,
                    text_threshold=self.ocr_params['text_threshold'],
                    low_text=self.ocr_params['low_text'],
                    link_threshold=self.ocr_params['link_threshold'],
                    width_ths=self.ocr_params['width_ths'],
                    height_ths=self.ocr_params['height_ths'],
                    canvas_size=self.ocr_params['canvas_size'],
                    mag_ratio=self.ocr_params['mag_ratio']
                )
                
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
            import traceback
            traceback.print_exc()
    
    # Visualization Methods
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
        
        # Calculate grid layout (max 4 columns)
        max_cols = 4
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
        layout = QVBoxLayout()
        layout.setSpacing(2)
        
        # Crop image display
        crop_label = QLabel()
        crop_label.setAlignment(Qt.AlignCenter)
        crop_label.setFixedSize(120, 80)
        crop_label.setStyleSheet("border: 1px solid #555; background-color: #2a2a2a;")
        
        # Convert processed crop to QPixmap
        processed_crop = crop_data['processed_crop']
        if processed_crop is not None and processed_crop.size > 0:
            # Convert BGR to RGB
            rgb_crop = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2RGB)
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
        
        # Result label
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
        
        # Confidence label
        conf_label = QLabel(f"Conf: {conf_text}")
        conf_label.setAlignment(Qt.AlignCenter)
        conf_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 10px;
            }
        """)
        layout.addWidget(conf_label)
        
        # Crop number label
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
        widget.setFixedSize(130, 150)
        return widget
    
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
    
    # Configuration Methods
    def _load_parameters_from_config(self):
        """Load parameters from configuration."""
        try:
            # Load preprocessing parameters
            self.preprocess_params['crop_top_fraction'] = get_setting('player_id.preprocessing.crop_top_fraction', 0.5)
            self.preprocess_params['contrast_alpha'] = get_setting('player_id.preprocessing.contrast_alpha', 1.0)
            self.preprocess_params['brightness_beta'] = get_setting('player_id.preprocessing.brightness_beta', 0)
            self.preprocess_params['gaussian_blur'] = get_setting('player_id.preprocessing.gaussian_blur', 0)
            self.preprocess_params['resize_factor'] = get_setting('player_id.preprocessing.resize_factor', 1.0)
            self.preprocess_params['enhance_contrast'] = get_setting('player_id.preprocessing.enhance_contrast', False)
            
            # Load OCR parameters
            if EASYOCR_AVAILABLE:
                self.ocr_params['text_threshold'] = get_setting('player_id.easyocr.text_threshold', 0.7)
                self.ocr_params['low_text'] = get_setting('player_id.easyocr.low_text', 0.4)
                self.ocr_params['link_threshold'] = get_setting('player_id.easyocr.link_threshold', 0.4)
                self.ocr_params['width_ths'] = get_setting('player_id.easyocr.width_ths', 0.7)
                self.ocr_params['height_ths'] = get_setting('player_id.easyocr.height_ths', 0.7)
                self.ocr_params['canvas_size'] = get_setting('player_id.easyocr.canvas_size', 2560)
                self.ocr_params['mag_ratio'] = get_setting('player_id.easyocr.mag_ratio', 1.0)
                self.ocr_params['gpu'] = get_setting('player_id.easyocr.gpu', True)
            
            # Update UI controls with loaded values
            self._update_controls_from_params()
            
            print("[EASYOCR_TUNING] Parameters loaded from configuration")
            
        except Exception as e:
            print(f"[EASYOCR_TUNING] Error loading parameters: {e}")
    
    def _update_controls_from_params(self):
        """Update UI controls with current parameter values."""
        # Preprocessing controls
        self.crop_fraction_spin.setValue(self.preprocess_params['crop_top_fraction'])
        self.contrast_spin.setValue(self.preprocess_params['contrast_alpha'])
        self.brightness_spin.setValue(self.preprocess_params['brightness_beta'])
        self.blur_spin.setValue(self.preprocess_params['gaussian_blur'])
        self.resize_spin.setValue(self.preprocess_params['resize_factor'])
        self.enhance_check.setChecked(self.preprocess_params['enhance_contrast'])
        
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
    
    def _save_parameters_to_config(self):
        """Save current parameters to configuration file."""
        try:
            # Create config updates
            config_updates = {
                'player_id': {
                    'preprocessing': {
                        'crop_top_fraction': self.preprocess_params['crop_top_fraction'],
                        'contrast_alpha': self.preprocess_params['contrast_alpha'],
                        'brightness_beta': self.preprocess_params['brightness_beta'],
                        'gaussian_blur': self.preprocess_params['gaussian_blur'],
                        'resize_factor': self.preprocess_params['resize_factor'],
                        'enhance_contrast': self.preprocess_params['enhance_contrast']
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
                    'canvas_size': self.ocr_params['canvas_size'],
                    'mag_ratio': self.ocr_params['mag_ratio'],
                    'gpu': self.ocr_params['gpu']
                }
            
            # Save to user config
            config_path = Path('configs') / 'user.yaml'
            config_path.parent.mkdir(exist_ok=True)
            
            # Load existing config or create new
            if config_path.exists():
                with open(config_path, 'r') as f:
                    existing_config = yaml.safe_load(f) or {}
            else:
                existing_config = {}
            
            # Merge updates
            self._deep_update(existing_config, config_updates)
            
            # Save updated config
            with open(config_path, 'w') as f:
                yaml.dump(existing_config, f, default_flow_style=False, indent=2)
            
            print(f"[EASYOCR_TUNING] Parameters saved to {config_path}")
            
        except Exception as e:
            print(f"[EASYOCR_TUNING] Error saving parameters: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
