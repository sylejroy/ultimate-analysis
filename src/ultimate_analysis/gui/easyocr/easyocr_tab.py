"""EasyOCR Tuning Tab for Ultimate Analysis GUI.

This module provides a specialized interface for tuning EasyOCR parameters
on detected player bounding boxes for optimal jersey number recognition.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...processing.inference import detect_players, load_detection_model
from ...processing.jersey_crops import best_number, easyocr_readtext_parameters, preprocess_crop
from ...processing.model_lock import MODEL_LOCK
from ...utils.logger import get_logger
from ...utils.model_files import default_model_path, models_root
from ...utils.video import VideoPlayer
from ..widgets.images import frame_to_pixmap
from ..widgets.model_selection import populate_detection_model_combo
from ..widgets.parameter_form import build_form, control_names, read_control, write_control
from ..widgets.video_list import VideoListWidget
from .parameters import OCR_FORM, PREPROCESS_FORM

# Try to import EasyOCR for parameter checking
try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

# Parameter name -> attribute of the control that edits it
PREPROCESS_CONTROLS = control_names(PREPROCESS_FORM)
OCR_CONTROLS = control_names(OCR_FORM)

logger = get_logger("EASYOCR_TUNING")


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
        self._detector = None  # (model, image size) of the selected model, loaded on first run
        self._readers: Dict[bool, Any] = {}  # EasyOCR reader per GPU setting

        # EasyOCR parameters (with optimized defaults)
        self.ocr_params = {
            "languages": ["en"],
            "gpu": True,
            "width_ths": 0.4,  # Updated from provided config
            "height_ths": 0.7,
            "paragraph": False,
            "adjust_contrast": 0.5,  # From provided config
            "filter_ths": 0.003,
            "text_threshold": 0.7,  # From provided config
            "low_text": 0.6,  # Updated from provided config
            "link_threshold": 0.4,
            "canvas_size": 2560,
            "mag_ratio": 2.0,  # Updated from provided config
            "slope_ths": 0.1,  # From provided config
            "ycenter_ths": 0.5,
            "y_ths": 0.5,
            "x_ths": 1.0,
            "detector": True,
            "recognizer": True,
            "allowlist": "0123456789",  # From provided config - digits only
            "detail": 1,  # From provided config
            "rotation_info": [0],  # From provided config
            "decoder": "greedy",  # From provided config
            "beamWidth": 5,
            "workers": 0,  # Number of parallel workers (0 = auto)
            "batch_size": 1,
        }

        # Preprocessing parameters (with optimized defaults)
        self.preprocess_params = {
            "crop_top_fraction": 0.33,  # Use top third of detection
            "contrast_alpha": 1.0,  # Contrast adjustment
            "brightness_beta": 0,  # Brightness adjustment
            "gaussian_blur": 13,  # Updated from provided config (blur_ksize)
            "resize_factor": 1.0,  # Resize factor (multiplier)
            "resize_absolute_width": 0,  # Absolute width in pixels (0 = use factor)
            "resize_absolute_height": 0,  # Absolute height in pixels (0 = use factor)
            "enhance_contrast": False,  # CLAHE enhancement (disabled per config)
            "clahe_clip_limit": 3.0,  # From provided config
            "clahe_grid_size": 8,  # From provided config
            "denoise": False,  # Apply denoising
            "sharpen": True,  # From provided config (enabled)
            "sharpen_strength": 0.05,  # From provided config
            "upscale": True,  # From provided config (enabled)
            "upscale_factor": 3.0,  # From provided config
            "upscale_to_size": True,  # From provided config
            "upscale_target_size": 256,  # From provided config
            "colour_mode": True,  # From provided config
            "bw_mode": True,  # From provided config
            # Minimum crop size filtering (skip OCR on crops too small to be readable)
            "min_crop_width": 20,  # Minimum crop width in pixels for OCR processing
            "min_crop_height": 30,  # Minimum crop height in pixels for OCR processing
        }

        # Initialize UI
        self._init_ui()
        self._reload_videos()
        # Load parameters from config (including easyocr_params.yaml) automatically on startup
        self._load_parameters_from_config()

        logger.info("EasyOCR Tuning Tab initialized with user configuration loaded")

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

    def _create_section_header(self, text: str) -> QLabel:
        """Create a styled section header label.

        Args:
            text: The header text (should include === formatting)

        Returns:
            Styled QLabel for section header
        """
        header = QLabel(text)
        header.setStyleSheet(
            """
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
                padding: 5px 0px;
                border-bottom: 1px solid #555555;
                margin-top: 10px;
            }
        """
        )
        return header

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
        refresh_button.clicked.connect(self._reload_videos)
        refresh_button.setToolTip("Refresh video list")
        list_header.addWidget(refresh_button)

        video_layout.addLayout(list_header)

        # Video list widget
        self.video_list = VideoListWidget()
        self.video_list.setMinimumHeight(200)  # Make video list taller
        self.video_list.currentRowChanged.connect(self._on_video_selection_changed)
        video_layout.addWidget(self.video_list)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # Model selection
        model_group = QGroupBox("Player Detection Model")
        model_layout = QFormLayout()

        self.detection_model_combo = QComboBox()
        populate_detection_model_combo(
            self.detection_model_combo,
            "player",
            default_model_path("player_detection"),
        )
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
        build_form(
            self,
            preprocess_layout,
            PREPROCESS_FORM,
            self.preprocess_params,
            self._on_preprocess_param_changed,
            self._create_section_header,
        )
        # Dependent controls are only editable while their feature is switched on
        self.enhance_check.stateChanged.connect(self._update_clahe_controls)
        self.sharpen_check.stateChanged.connect(self._update_sharpen_controls)
        self.upscale_check.stateChanged.connect(self._update_upscale_controls)
        self.upscale_to_size_check.stateChanged.connect(self._update_upscale_controls)
        self._update_clahe_controls()
        self._update_sharpen_controls()
        self._update_upscale_controls()
        preprocess_group.setLayout(preprocess_layout)
        left_layout.addWidget(preprocess_group)
        left_layout.addStretch()
        left_column.setLayout(left_layout)

        # Right column - EasyOCR
        right_column = QWidget()
        right_layout = QVBoxLayout()

        ocr_group = QGroupBox("EasyOCR Parameters")
        ocr_layout = QFormLayout()
        if EASYOCR_AVAILABLE:
            build_form(
                self,
                ocr_layout,
                OCR_FORM,
                self.ocr_params,
                self._on_ocr_param_changed,
                self._create_section_header,
            )
        else:
            ocr_layout.addRow(QLabel("EasyOCR not available"))
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
        save_button.setToolTip("Save current parameters to easyocr_params.yaml")
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
        self.video_label.setStyleSheet(
            """
            QLabel {
                border: 2px solid #555;
                background-color: #1a1a1a;
                color: #999;
                font-size: 14px;
            }
        """
        )
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
        self.crops_container.setStyleSheet(
            "QWidget { background-color: #1a1a1a; }"
        )  # Dark background
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

    # ========== EVENT HANDLERS ==========
    def _reload_videos(self):
        """List the available videos and open a random one."""
        self.video_files = self.video_list.reload()
        logger.info(f"Found {len(self.video_files)} video files")
        if self.video_files:
            # Selecting the row loads the video
            self.video_list.setCurrentRow(random.randint(0, len(self.video_files) - 1))

    def _on_model_changed(self, model_path: str):
        """Handle player detection model change: the new model loads on the next run."""
        self._detector = None

    def _get_detector(self):
        """The selected detection model as (model, image size), loaded on first use.

        The tab has its own copy; choosing a model here leaves the main tab's alone.
        """
        if self._detector is None and self.detection_model_combo.currentText():
            weights = models_root() / self.detection_model_combo.currentText()
            self._detector = load_detection_model(str(weights))
        return self._detector

    def _get_reader(self):
        """EasyOCR reader for the current GPU setting, created once and kept."""
        gpu = self.ocr_params["gpu"]
        if gpu not in self._readers:
            self._readers[gpu] = easyocr.Reader(self.ocr_params["languages"], gpu=gpu)
        return self._readers[gpu]

    def _display_frame(self, frame: np.ndarray):
        """Display a frame in the video label, scaled to fit."""
        if frame is None:
            return

        # Leave room for the label's 2px border
        label_size = self.video_label.size()
        scaled_pixmap = frame_to_pixmap(frame).scaled(
            label_size.width() - 4,
            label_size.height() - 4,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

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
            filename = Path(video_path).name

            # Set frame slider range
            video_info = self.video_player.get_video_info()
            total_frames = video_info["total_frames"]
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

            logger.info(f"Video loaded successfully: {filename}")
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

    def _on_preprocess_param_changed(self):
        """Handle preprocessing parameter change."""
        for key, control in PREPROCESS_CONTROLS.items():
            self.preprocess_params[key] = read_control(getattr(self, control))

    def _on_ocr_param_changed(self):
        """Handle EasyOCR parameter change."""
        if not EASYOCR_AVAILABLE:
            return

        for key, control in OCR_CONTROLS.items():
            self.ocr_params[key] = read_control(getattr(self, control))
        # An empty allowlist means no character restriction
        self.ocr_params["allowlist"] = self.ocr_params["allowlist"].strip() or None

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
            logger.debug("No frame loaded")
            return

        try:
            # Step 1: Detect the players
            detector = self._get_detector()
            if detector is None:
                logger.error("No player detection model could be loaded")
                return
            with MODEL_LOCK:
                person_detections = detect_players(self.current_frame, *detector)

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

            logger.info(
                f"Analysis complete: {len(person_detections)} detections, {len(self.current_crops)} crops"
            )

        except Exception as e:
            # Clear crops display and show error
            self._clear_crops_display()
            error_msg = f"Analysis error: {str(e)}"
            logger.debug(f"{error_msg}")
            logger.exception("Unexpected error")

    def _extract_crops_from_detections(self):
        """Extract crops from person detections."""
        if self.current_frame is None or not self.current_detections:
            return

        self.current_crops.clear()

        for i, detection in enumerate(self.current_detections):
            bbox = detection["bbox"]
            # bbox format from inference is [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            # Calculate width and height
            _, h = x2 - x1, y2 - y1

            # Apply crop fraction (crop from top)
            crop_fraction = self.preprocess_params["crop_top_fraction"]
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
                processed_crop = preprocess_crop(crop, self.preprocess_params)

                self.current_crops.append(
                    {
                        "detection_idx": i,
                        "bbox": bbox,
                        "crop_bbox": [x1, y1, x2 - x1, crop_y2 - y1],
                        "original_crop": crop,
                        "processed_crop": processed_crop,
                    }
                )

        logger.debug(f"Extracted {len(self.current_crops)} crops")

    def _run_easyocr_on_crops(self):
        """Run EasyOCR on all detected crops."""
        if not EASYOCR_AVAILABLE:
            self._clear_crops_display()
            return

        if not self.current_crops:
            self._clear_crops_display()
            return

        try:
            reader = self._get_reader()

            results = []

            for crop_data in self.current_crops:
                # Check minimum crop size BEFORE OCR processing (using original crop size)
                original_crop = crop_data["original_crop"]
                orig_height, orig_width = original_crop.shape[:2]
                min_crop_width = self.preprocess_params["min_crop_width"]
                min_crop_height = self.preprocess_params["min_crop_height"]

                if orig_width < min_crop_width or orig_height < min_crop_height:
                    logger.debug(
                        f"Original crop too small ({orig_width}x{orig_height}), skipping OCR (min: {min_crop_width}x{min_crop_height})"
                    )
                    # Add result showing crop was skipped
                    results.append(
                        {
                            "detection_idx": crop_data["detection_idx"],
                            "text": "SKIPPED",
                            "confidence": 0.0,
                            "ocr_results": [],
                            "skipped_reason": f"Original crop too small ({orig_width}x{orig_height})",
                        }
                    )
                    continue

                # Use processed crop for OCR
                processed_crop = crop_data["processed_crop"]

                readtext_params = easyocr_readtext_parameters(self.ocr_params)

                # Run EasyOCR
                with MODEL_LOCK:
                    ocr_results = reader.readtext(processed_crop, **readtext_params)

                # The most confident numeric reading is taken as the jersey number
                best_text, best_confidence = best_number(ocr_results)

                results.append(
                    {
                        "detection_idx": crop_data["detection_idx"],
                        "text": best_text,
                        "confidence": best_confidence,
                        "ocr_results": ocr_results,
                    }
                )

            # Display results visually
            self._display_crops_with_results(results)

            # Draw results on frame
            if self.current_frame is not None:
                annotated_frame = self._draw_ocr_results(self.current_frame.copy(), results)
                self._display_frame(annotated_frame)

            # Print summary with skipped crop information
            skipped_count = sum(1 for r in results if "skipped_reason" in r)
            processed_count = len(results) - skipped_count
            logger.info(
                f"EasyOCR complete: {processed_count} crops processed, {skipped_count} crops skipped (too small)"
            )

        except Exception as e:
            # Clear crops display and show error
            self._clear_crops_display()
            error_msg = f"EasyOCR error: {str(e)}"
            logger.debug(f"{error_msg}")
            logger.exception("Unexpected error")

    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection bounding boxes on frame."""
        for i, detection in enumerate(self.current_detections):
            bbox = detection["bbox"]
            confidence = detection.get("confidence", 0.0)

            # bbox format is [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw detection info
            label = f"Person {i + 1}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), (0, 255, 0), -1
            )
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

        logger.debug(f"Displayed {num_crops} crops in {rows}x{cols} grid")

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
        processed_crop = crop_data["processed_crop"]
        if processed_crop is not None and processed_crop.size > 0:
            # Create annotated version of the crop with OCR detections highlighted
            annotated_crop = self._draw_ocr_on_crop(processed_crop.copy(), result)

            # Convert BGR to RGB
            rgb_crop = cv2.cvtColor(annotated_crop, cv2.COLOR_BGR2RGB)
            height, width = rgb_crop.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(rgb_crop.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Scale to fit label
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                crop_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            crop_label.setPixmap(scaled_pixmap)
        else:
            crop_label.setText("Error")

        layout.addWidget(crop_label)

        # OCR result display
        text = result.get("text", "")
        confidence = result.get("confidence", 0.0)
        skipped_reason = result.get("skipped_reason", "")

        if skipped_reason:
            # Crop was skipped due to minimum size filtering
            color = "#888888"  # Gray for skipped
            result_text = "SKIPPED"
            conf_text = skipped_reason
        elif text and text != "SKIPPED":
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
        result_label.setStyleSheet(
            f"""
            QLabel {{
                color: {color};
                font-weight: bold;
                font-size: 12px;
                background-color: #1a1a1a;
                border: 1px solid {color};
                border-radius: 3px;
                padding: 2px;
            }}
        """
        )
        layout.addWidget(result_label)

        # Confidence label - increased size
        if skipped_reason:
            conf_label = QLabel(conf_text)  # Show skip reason instead of confidence
            conf_label.setStyleSheet(
                """
                QLabel {
                    color: #999999;
                    font-size: 9px;
                }
            """
            )
        else:
            conf_label = QLabel(f"Conf: {conf_text}")
            conf_label.setStyleSheet(
                """
                QLabel {
                    color: #cccccc;
                    font-size: 10px;
                }
            """
            )
        conf_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(conf_label)

        # Crop number label - increased size
        num_label = QLabel(f"Crop {crop_num}")
        num_label.setAlignment(Qt.AlignCenter)
        num_label.setStyleSheet(
            """
            QLabel {
                color: #888888;
                font-size: 9px;
            }
        """
        )
        layout.addWidget(num_label)

        widget.setLayout(layout)
        widget.setFixedSize(170, 180)  # Increased from 110x120 to 170x180
        return widget

    def _draw_ocr_on_crop(self, crop_image: np.ndarray, result: Dict) -> np.ndarray:
        """Draw OCR detection boxes and text on the crop image."""
        # Check if crop was skipped
        if "skipped_reason" in result:
            # Draw "SKIPPED" overlay on the image
            overlay = crop_image.copy()
            h, w = crop_image.shape[:2]

            # Add semi-transparent gray overlay
            cv2.rectangle(overlay, (0, 0), (w, h), (128, 128, 128), -1)
            cv2.addWeighted(crop_image, 0.7, overlay, 0.3, 0, crop_image)

            # Add "SKIPPED" text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "SKIPPED"
            text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(crop_image, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2)

            return crop_image

        if "ocr_results" not in result or not result["ocr_results"]:
            return crop_image

        # Get image dimensions
        img_height, img_width = crop_image.shape[:2]

        # Draw each OCR detection
        for bbox, text, confidence in result["ocr_results"]:
            if confidence < 0.1:  # Skip very low confidence detections
                continue

            # Convert bbox coordinates (EasyOCR returns normalized coordinates)
            if isinstance(bbox[0], (list, tuple)):
                # Polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                points = np.array(bbox, dtype=np.int32)
                points[:, 0] = np.clip(points[:, 0], 0, img_width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, img_height - 1)

                # Draw polygon outline
                color = (
                    (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                )  # Green for high conf, orange for low
                cv2.polylines(
                    crop_image, [points], True, color, 2
                )  # Increased thickness from 1 to 2

                # Fill polygon with semi-transparent overlay
                overlay = crop_image.copy()
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(crop_image, 0.8, overlay, 0.2, 0, crop_image)

                # Draw text near the detection
                if text.strip() and any(c.isdigit() for c in text):  # Only show if contains digits
                    text_pos = (int(np.min(points[:, 0])), int(np.min(points[:, 1]) - 3))
                    text_pos = (max(0, text_pos[0]), max(12, text_pos[1]))

                    # Draw text background - increased font size
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[
                        0
                    ]  # Increased from 0.3 to 0.5
                    cv2.rectangle(
                        crop_image,
                        text_pos,
                        (text_pos[0] + text_size[0], text_pos[1] - text_size[1] - 3),
                        color,
                        -1,
                    )
                    cv2.putText(
                        crop_image,
                        text,
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )  # Increased from 0.3 to 0.5

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
            det_idx = result["detection_idx"]
            text = result["text"]
            confidence = result["confidence"]

            if det_idx < len(self.current_detections):
                detection = self.current_detections[det_idx]
                bbox = detection["bbox"]
                x1, y1, x2, y2 = [int(coord) for coord in bbox]

                # Draw detection box in blue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw OCR result
                if text:
                    ocr_label = f"'{text}' ({confidence:.3f})"
                    color = (
                        (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                    )  # Green if high conf, orange if low
                else:
                    ocr_label = "No text"
                    color = (0, 0, 255)  # Red for no detection

                # Draw label
                label_size = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    frame, (x1, y2), (x1 + label_size[0] + 10, y2 + label_size[1] + 10), color, -1
                )
                cv2.putText(
                    frame,
                    ocr_label,
                    (x1 + 5, y2 + label_size[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        return frame

    # ========== CONFIGURATION METHODS ==========
    def _load_parameters_from_config(self):
        """Load parameters saved in easyocr_params.yaml, keeping defaults for the rest."""
        try:
            user_config = self._load_user_config().get("player_id", {})

            saved_preprocess = user_config.get("preprocessing", {})
            for key in PREPROCESS_CONTROLS:
                if key in saved_preprocess:
                    self.preprocess_params[key] = saved_preprocess[key]

            if EASYOCR_AVAILABLE:
                saved_ocr = user_config.get("easyocr", {})
                for key in OCR_CONTROLS:
                    if key in saved_ocr:
                        self.ocr_params[key] = saved_ocr[key]

            # Update UI controls with loaded values
            self._update_controls_from_params()

            logger.info("Parameters loaded from configuration")
        except Exception as e:
            logger.exception(f"Error loading parameters from config: {e}")
            # Continue with default values if config loading fails

    def _load_user_config(self) -> Dict[str, Any]:
        """Load easyocr_params.yaml configuration if it exists.

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

            user_config_file = project_root / "configs" / "easyocr_params.yaml"

            if not user_config_file.exists():
                return {}

            with open(user_config_file, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)

            if user_config:
                logger.debug(f"User configuration loaded from: {user_config_file}")
                return user_config
            else:
                return {}

        except Exception as e:
            logger.error(f"Error loading user config: {e}")
            return {}

    def _update_controls_from_params(self):
        """Update UI controls with current parameter values."""
        bindings = [(self.preprocess_params, PREPROCESS_CONTROLS)]
        if EASYOCR_AVAILABLE:
            bindings.append((self.ocr_params, OCR_CONTROLS))

        for params, controls in bindings:
            for key, control_name in controls.items():
                control = getattr(self, control_name)
                # Blocked so a control does not write the other, not yet updated, controls
                # back into the parameters
                control.blockSignals(True)
                try:
                    write_control(control, params[key])
                finally:
                    control.blockSignals(False)

        self._update_clahe_controls()
        self._update_sharpen_controls()
        self._update_upscale_controls()

    def _save_parameters_to_config(self):
        """Save current parameters to configuration file."""
        try:
            # Find project root and construct absolute path to easyocr_params.yaml
            current_dir = Path(__file__).parent
            project_root = None

            for parent in current_dir.parents:
                if (parent / "configs").exists():
                    project_root = parent
                    break

            if project_root is None:
                logger.error("Error: Could not find project root with configs directory")
                return

            config_path = project_root / "configs" / "easyocr_params.yaml"

            # Create config updates with all parameters
            config_updates = {
                "player_id": {
                    "preprocessing": {
                        key: self.preprocess_params[key] for key in PREPROCESS_CONTROLS
                    }
                }
            }

            if EASYOCR_AVAILABLE:
                config_updates["player_id"]["easyocr"] = {
                    key: self.ocr_params[key] for key in OCR_CONTROLS
                }

            # Ensure configs directory exists
            config_path.parent.mkdir(exist_ok=True)

            # Load existing config or create new
            existing_config = {}
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        existing_config = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.error(f"Warning: Could not read existing config: {e}")
                    existing_config = {}

            # Merge updates
            self._deep_update(existing_config, config_updates)

            # Save updated config
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing_config, f, default_flow_style=False, indent=2)

            logger.info(f"Parameters saved successfully to {config_path}")
            logger.info(
                f"  - Preprocessing parameters: {len(config_updates['player_id']['preprocessing'])} saved"
            )
            if EASYOCR_AVAILABLE and "easyocr" in config_updates["player_id"]:
                logger.info(
                    f"  - EasyOCR parameters: {len(config_updates['player_id']['easyocr'])} saved"
                )

        except Exception as e:
            logger.exception(f"Error saving parameters to config: {e}")

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
