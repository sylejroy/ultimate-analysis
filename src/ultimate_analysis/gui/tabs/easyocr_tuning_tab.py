"""
EasyOCR tuning tab - complete implementation.
"""
import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, 
    QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QFormLayout, QGroupBox, 
    QScrollArea, QGridLayout, QCheckBox, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from ...config.settings import get_settings

logger = logging.getLogger("ultimate_analysis.gui.easyocr_tuning")

# Parameter descriptions for tooltips
EASYOCR_PARAM_DESCRIPTIONS = {
    "detail": "0: Only text, 1: Text and box coordinates.",
    "allowlist": "Characters allowed in recognition.",
    "min_size": "Minimum text height to be detected.",
    "rotation_info": "Angles (degrees) to try for rotated text.",
    "text_threshold": "Confidence threshold for text regions.",
    "low_text": "Threshold for low-confidence text regions.",
    "mag_ratio": "Image magnification ratio before detection.",
    "contrast_ths": "Contrast threshold for text region filtering.",
    "adjust_contrast": "Adjust image contrast before detection.",
    "slope_ths": "Slope threshold for text region filtering.",
    "width_ths": "Width threshold for text region filtering.",
    "decoder": "Text decoding algorithm (greedy/beamsearch)."
}

PREPROC_PARAM_DESCRIPTIONS = {
    "blur_ksize": "Gaussian blur kernel size (odd integer).",
    "clahe_clip": "CLAHE clip limit (contrast enhancement).",
    "clahe_grid": "CLAHE grid size (tile size).",
    "sharpen": "Sharpening strength (0=off, 1=default, >1=stronger).",
    "upscale": "Upscale factor for OCR crop."
}

EASYOCR_PARAMS = [
    ("detail", QComboBox, [0, 1], 1),
    ("allowlist", QComboBox, ["None", "0123456789", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"], "None"),
    ("min_size", QSpinBox, 1, 100, 20),
    ("rotation_info", QComboBox, ["[0]", "[0, 90, 180, 270]", "[0, 45, 90, 135, 180, 225, 270, 315]"], "[0]"),
    ("text_threshold", QDoubleSpinBox, 0.0, 1.0, 0.01, 0.7),
    ("low_text", QDoubleSpinBox, 0.0, 1.0, 0.01, 0.4),
    ("mag_ratio", QDoubleSpinBox, 1.0, 10.0, 0.1, 1.5),
    ("contrast_ths", QDoubleSpinBox, 0.0, 1.0, 0.01, 0.1),
    ("adjust_contrast", QDoubleSpinBox, 0.0, 1.0, 0.01, 0.5),
    ("slope_ths", QDoubleSpinBox, 0.0, 1.0, 0.01, 0.1),
    ("width_ths", QDoubleSpinBox, 0.0, 1.0, 0.01, 0.4),
    ("decoder", QComboBox, ["greedy", "beamsearch"], "greedy"),
]

PREPROC_PARAMS = [
    ("blur_ksize", QSpinBox, 1, 21, 3),
    ("clahe_clip", QDoubleSpinBox, 1.0, 10.0, 0.1, 2.0),
    ("clahe_grid", QSpinBox, 1, 16, 4),
    ("sharpen", QDoubleSpinBox, 0.0, 3.0, 0.1, 1.0),
    ("upscale", QDoubleSpinBox, 1.0, 5.0, 0.1, 2.0),
]


class DevEasyOCRTuningTab(QWidget):
    """
    Tab for EasyOCR tuning development tools.
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.input_folder = self.settings.paths.processed_dev_data
        self.cap = None
        self.current_video = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.ocr_params = self._default_ocr_params()
        self.preproc_params = self._default_preproc_params()
        self.player_crops = []
        self.player_results = []
        self.bw_mode = False
        self.colour_mode = False
        self._init_ui()
        self._populate_video_list()

    def _default_ocr_params(self) -> Dict[str, Any]:
        """Get default OCR parameters optimized for better performance."""
        return {
            "detail": 1,
            "allowlist": None,
            "min_size": 18,
            "rotation_info": [0, 90, 180, 270],
            "text_threshold": 0.6,
            "low_text": 0.3,
            "mag_ratio": 2.0,
            "contrast_ths": 0.05,
            "adjust_contrast": 0.7,
            "slope_ths": 0.1,
            "width_ths": 0.3,
            "decoder": "greedy",
        }

    def _default_preproc_params(self) -> Dict[str, Any]:
        """Get default preprocessing parameters."""
        return {
            "blur_ksize": 1,         # No blur by default
            "clahe_clip": 3.0,       # Stronger contrast
            "clahe_grid": 8,         # Finer grid
            "sharpen": 2.0,          # Stronger sharpening
            "upscale": 3.0,          # Larger crops
        }

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout()
        
        # Video list on the left
        self.video_list = QListWidget()
        self.video_list.itemSelectionChanged.connect(self._load_selected_video)
        layout.addWidget(self.video_list, 1)

        # Right panel with controls and display
        right = QVBoxLayout()
        
        # Video display area
        self.frame_label = QLabel("No video loaded")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(400, 300)
        self.frame_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        right.addWidget(self.frame_label, 2)

        # Frame slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._slider_moved)
        right.addWidget(self.slider)

        # Preprocessing parameter controls
        preproc_group = self._create_preproc_controls()
        right.addWidget(preproc_group)

        # OCR parameter controls
        param_group = self._create_ocr_controls()
        right.addWidget(param_group)

        # Save/load parameter set buttons
        param_btn_row = QHBoxLayout()
        self.save_params_btn = QPushButton("Save Param Set")
        self.save_params_btn.clicked.connect(self._save_param_set)
        self.load_params_btn = QPushButton("Load Param Set")
        self.load_params_btn.clicked.connect(self._load_param_set)
        param_btn_row.addWidget(self.save_params_btn)
        param_btn_row.addWidget(self.load_params_btn)
        right.addLayout(param_btn_row)

        # Run button
        self.run_button = QPushButton("Run Player Detection + OCR on Frame")
        self.run_button.clicked.connect(self._run_detection_and_ocr)
        right.addWidget(self.run_button)

        # Mosaic area for results
        self.mosaic_area = QScrollArea()
        self.mosaic_widget = QWidget()
        self.mosaic_layout = QGridLayout()
        self.mosaic_widget.setLayout(self.mosaic_layout)
        self.mosaic_area.setWidget(self.mosaic_widget)
        self.mosaic_area.setWidgetResizable(True)
        right.addWidget(self.mosaic_area, 6)

        # Add the right panel to the main layout
        layout.addLayout(right, 4)
        self.setLayout(layout)

    def _create_preproc_controls(self) -> QGroupBox:
        """Create preprocessing parameter controls."""
        preproc_group = QGroupBox("Preprocessing Parameters")
        preproc_layout = QFormLayout()
        
        self.preproc_widgets = {}
        self.preproc_enables = {}
        
        for name, widget_type, *args in PREPROC_PARAMS:
            row = QHBoxLayout()
            
            # Enable checkbox
            enable = QCheckBox()
            enable.setChecked(True)
            enable.stateChanged.connect(self._update_preproc_params)
            self.preproc_enables[name] = enable
            
            # Parameter widget
            if widget_type == QSpinBox:
                widget = QSpinBox()
                widget.setRange(args[0], args[1])
                widget.setValue(self.preproc_params[name])
                widget.valueChanged.connect(self._update_preproc_params)
            elif widget_type == QDoubleSpinBox:
                widget = QDoubleSpinBox()
                widget.setRange(args[0], args[1])
                widget.setSingleStep(args[2])
                widget.setValue(self.preproc_params[name])
                widget.valueChanged.connect(self._update_preproc_params)
            
            widget.setToolTip(PREPROC_PARAM_DESCRIPTIONS.get(name, ""))
            row.addWidget(enable)
            row.addWidget(widget)
            preproc_layout.addRow(name, row)
            self.preproc_widgets[name] = widget

        # Upscale mode controls
        self.upscale_to_size_checkbox = QCheckBox("Upscale to pixel size")
        self.upscale_to_size_checkbox.setChecked(False)
        self.upscale_to_size_checkbox.setToolTip("If checked, upscaling resizes the largest crop dimension to the target size in pixels.")
        self.upscale_to_size_checkbox.stateChanged.connect(self._update_preproc_params)
        
        self.upscale_target_size_spin = QSpinBox()
        self.upscale_target_size_spin.setRange(16, 512)
        self.upscale_target_size_spin.setValue(64)
        self.upscale_target_size_spin.setToolTip("Target pixel size for upscaling (largest dimension). Only used if 'Upscale to pixel size' is checked.")
        self.upscale_target_size_spin.valueChanged.connect(self._update_preproc_params)
        
        upscale_row = QHBoxLayout()
        upscale_row.addWidget(self.upscale_to_size_checkbox)
        upscale_row.addWidget(self.upscale_target_size_spin)
        preproc_layout.addRow("Upscale Mode", upscale_row)

        # B/W and color mode controls
        self.bw_checkbox = QCheckBox("Black/White Invert")
        self.bw_checkbox.setChecked(False)
        self.bw_checkbox.stateChanged.connect(self._toggle_bw_mode)
        preproc_layout.addRow("Invert B/W", self.bw_checkbox)

        self.colour_checkbox = QCheckBox("Use Colour Input")
        self.colour_checkbox.setChecked(False)
        self.colour_checkbox.stateChanged.connect(self._toggle_colour_mode)
        preproc_layout.addRow("Use Colour Input", self.colour_checkbox)

        preproc_group.setLayout(preproc_layout)
        return preproc_group

    def _create_ocr_controls(self) -> QGroupBox:
        """Create OCR parameter controls."""
        param_group = QGroupBox("EasyOCR Parameters")
        param_layout = QFormLayout()
        
        self.param_widgets = {}
        
        for name, widget_type, *args in EASYOCR_PARAMS:
            if widget_type == QComboBox:
                widget = QComboBox()
                for v in args[0]:
                    widget.addItem(str(v))
                widget.setCurrentText(str(args[1]))
                widget.currentTextChanged.connect(self._update_params)
            elif widget_type == QSpinBox:
                widget = QSpinBox()
                widget.setRange(args[0], args[1])
                widget.setValue(args[2])
                widget.valueChanged.connect(self._update_params)
            elif widget_type == QDoubleSpinBox:
                widget = QDoubleSpinBox()
                widget.setRange(args[0], args[1])
                widget.setSingleStep(args[2])
                widget.setValue(args[3])
                widget.valueChanged.connect(self._update_params)
            
            widget.setToolTip(EASYOCR_PARAM_DESCRIPTIONS.get(name, ""))
            param_layout.addRow(name, widget)
            self.param_widgets[name] = widget
        
        param_group.setLayout(param_layout)
        return param_group

    def _populate_video_list(self):
        """Populate the video list with available video files."""
        self.video_list.clear()
        
        try:
            if not os.path.exists(self.input_folder):
                logger.warning(f"Input folder does not exist: {self.input_folder}")
                return
            
            for filename in os.listdir(self.input_folder):
                filepath = os.path.join(self.input_folder, filename)
                if os.path.isfile(filepath) and filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    self.video_list.addItem(filename)
        except Exception as e:
            logger.error(f"Error populating video list: {e}")
            QMessageBox.warning(self, "Error", f"Could not load video list: {e}")

    def _load_selected_video(self):
        """Load the selected video."""
        items = self.video_list.selectedItems()
        if not items:
            return
        
        filename = items[0].text()
        filepath = os.path.join(self.input_folder, filename)
        
        try:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(filepath)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video: {filepath}")
            
            self.current_video = filepath
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            
            self.slider.setMaximum(self.total_frames - 1)
            self.slider.setValue(0)
            self._show_frame(0)
            
            logger.info(f"Loaded video: {filename} ({self.total_frames} frames)")
            
        except Exception as e:
            logger.error(f"Error loading video {filename}: {e}")
            QMessageBox.warning(self, "Error", f"Could not load video: {e}")

    def _slider_moved(self, value: int):
        """Handle slider movement."""
        if not self.cap:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        self._show_frame(value)

    def _show_frame(self, frame_idx: int):
        """Show the current frame."""
        if not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.current_frame_img = frame
        
        # Convert to QPixmap and display
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        
        self.frame_label.setPixmap(pixmap.scaled(
            self.frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def _update_preproc_params(self):
        """Update preprocessing parameters."""
        for name, widget in self.preproc_widgets.items():
            self.preproc_params[name] = widget.value()
        
        self.upscale_to_size = self.upscale_to_size_checkbox.isChecked()
        self.upscale_target_size = self.upscale_target_size_spin.value()

    def _update_params(self):
        """Update OCR parameters."""
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QComboBox):
                val = widget.currentText()
                if name == "allowlist" and val == "None":
                    val = None
                elif name == "rotation_info":
                    try:
                        val = eval(val)
                    except:
                        val = [0]
                elif name == "detail":
                    val = int(val)
                self.ocr_params[name] = val
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                self.ocr_params[name] = widget.value()

    def _toggle_bw_mode(self, state):
        """Toggle black/white mode."""
        self.bw_mode = bool(state)

    def _toggle_colour_mode(self, state):
        """Toggle colour mode."""
        self.colour_mode = bool(state)

    def _preprocess_crop(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess a crop for EasyOCR."""
        try:
            # Import here to avoid circular imports
            from ...processing.player_id import preprocess_for_easyocr
            
            preproc_params = self.preproc_params.copy()
            preproc_enables = {k: v.isChecked() for k, v in self.preproc_enables.items()}
            
            return preprocess_for_easyocr(
                crop, preproc_params, preproc_enables,
                colour_mode=self.colour_mode, bw_mode=self.bw_mode,
                upscale_to_size=getattr(self, 'upscale_to_size', False),
                upscale_target_size=getattr(self, 'upscale_target_size', 64)
            )
        except Exception as e:
            logger.error(f"Error preprocessing crop: {e}")
            # Return original crop as fallback
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            return crop, gray

    def _save_param_set(self):
        """Save the current parameter set to a JSON file."""
        params = {
            'ocr': self.ocr_params,
            'preproc': self.preproc_params,
            'preproc_enables': {k: v.isChecked() for k, v in self.preproc_enables.items()},
            'colour_mode': self.colour_mode,
            'bw_mode': self.bw_mode,
            'upscale_to_size': getattr(self, 'upscale_to_size', False),
            'upscale_target_size': getattr(self, 'upscale_target_size', 64)
        }
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Parameter Set", "easyocr_params.json", "JSON Files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(params, f, indent=2)
                logger.info(f"Saved parameter set to: {filename}")
                QMessageBox.information(self, "Success", "Parameter set saved successfully!")
            except Exception as e:
                logger.error(f"Error saving parameter set: {e}")
                QMessageBox.warning(self, "Error", f"Could not save parameter set: {e}")

    def _load_param_set(self):
        """Load a parameter set from a JSON file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Parameter Set", "", "JSON Files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    params = json.load(f)
                
                # Load OCR params
                for k, v in params.get('ocr', {}).items():
                    if k in self.param_widgets:
                        widget = self.param_widgets[k]
                        if isinstance(widget, QComboBox):
                            idx = widget.findText(str(v))
                            if idx >= 0:
                                widget.setCurrentIndex(idx)
                        else:
                            widget.setValue(v)
                
                # Load preproc params
                for k, v in params.get('preproc', {}).items():
                    if k in self.preproc_widgets:
                        self.preproc_widgets[k].setValue(v)
                
                # Load enables
                for k, v in params.get('preproc_enables', {}).items():
                    if k in self.preproc_enables:
                        self.preproc_enables[k].setChecked(v)
                
                # Load mode settings
                if 'colour_mode' in params:
                    self.colour_checkbox.setChecked(params['colour_mode'])
                if 'bw_mode' in params:
                    self.bw_checkbox.setChecked(params['bw_mode'])
                if 'upscale_to_size' in params:
                    self.upscale_to_size_checkbox.setChecked(params['upscale_to_size'])
                if 'upscale_target_size' in params:
                    self.upscale_target_size_spin.setValue(params['upscale_target_size'])
                
                logger.info(f"Loaded parameter set from: {filename}")
                QMessageBox.information(self, "Success", "Parameter set loaded successfully!")
                
            except Exception as e:
                logger.error(f"Error loading parameter set: {e}")
                QMessageBox.warning(self, "Error", f"Could not load parameter set: {e}")

    def _run_detection_and_ocr(self):
        """Run player detection and OCR on the current frame."""
        if not hasattr(self, "current_frame_img"):
            QMessageBox.warning(self, "Error", "No frame loaded")
            return
        
        try:
            # Import processing modules
            from ...processing.inference import run_inference
            from ...processing.tracking import run_tracking
            from ...processing.player_id import _easyocr_reader, set_easyocr
            
            frame = self.current_frame_img.copy()
            
            # Run detection and tracking
            detections = run_inference(frame)
            tracks = run_tracking(frame, detections)
            
            player_crops = []
            player_results = []
            
            # Initialize EasyOCR if needed
            if _easyocr_reader is None:
                set_easyocr()
            
            # Process each track
            for track in tracks:
                # Check if this is a player
                track_class = getattr(track, "det_class", None)
                PLAYER_CLASS_IDX = 1
                if track_class is not None and track_class != PLAYER_CLASS_IDX:
                    continue
                
                # Get bounding box
                bbox = self._get_bbox_from_track(track)
                if bbox is None:
                    continue
                
                # Extract and preprocess crop
                crop = self._extract_crop(frame, bbox)
                if crop is None:
                    continue
                
                preproc_rgb, preproc_gray = self._preprocess_crop(crop)
                
                # Run OCR
                ocr_results = _easyocr_reader.readtext(preproc_rgb, **self.ocr_params)
                
                # Process results
                digit_str = ''.join([r[1] for r in ocr_results])
                ocr_boxes = [r[0] for r in ocr_results]
                ocr_confs = [r[2] if len(r) > 2 else None for r in ocr_results]
                
                player_crops.append(preproc_gray)
                player_results.append((digit_str, ocr_boxes, ocr_confs))
            
            # Display results
            self._display_mosaic(player_crops, player_results)
            
        except Exception as e:
            logger.error(f"Error running detection and OCR: {e}")
            QMessageBox.warning(self, "Error", f"Could not run detection and OCR: {e}")

    def _get_bbox_from_track(self, track) -> Optional[List[int]]:
        """Extract bounding box from track object."""
        bbox = None
        
        if hasattr(track, "to_ltrb"):
            bbox = track.to_ltrb()
        elif hasattr(track, "to_tlwh"):
            x, y, w, h = track.to_tlwh()
            bbox = [x, y, x + w, y + h]
        elif hasattr(track, "bbox"):
            bbox = track.bbox
        elif isinstance(track, dict):
            bbox = track.get('bbox', None)
        
        if bbox is not None:
            bbox = np.round(np.array(bbox)).astype(int)
            if len(bbox) == 4:
                return bbox.tolist()
        
        return None

    def _extract_crop(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Extract crop from frame using bounding box."""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        
        # Take top half for jersey number detection
        half_h = max(1, h // 2)
        crop = frame[y1:y1+half_h, x1:x2]
        
        return crop if crop.size > 0 else None

    def _display_mosaic(self, crops: List[np.ndarray], results: List[Tuple]):
        """Display crops and OCR results in a mosaic layout."""
        # Clear existing widgets
        for i in reversed(range(self.mosaic_layout.count())):
            widget = self.mosaic_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        
        # Display crops in a grid
        cols = 6
        for idx, (crop, (digit_str, ocr_boxes, ocr_confs)) in enumerate(zip(crops, results)):
            # Create display crop with OCR boxes
            crop_display = self._create_crop_display(crop, ocr_boxes, ocr_confs, digit_str)
            
            # Create QLabel with the crop
            label = QLabel()
            pixmap = self._array_to_pixmap(crop_display)
            label.setPixmap(pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Create container with crop and text
            if digit_str:
                label.setToolTip(f"Digits: {digit_str}")
                label_text = QLabel(digit_str)
                label_text.setAlignment(Qt.AlignCenter)
                label_text.setStyleSheet("color: #00FF00; font-weight: bold;")
                
                container = QWidget()
                vbox = QVBoxLayout()
                vbox.addWidget(label)
                vbox.addWidget(label_text)
                container.setLayout(vbox)
                
                self.mosaic_layout.addWidget(container, idx // cols, idx % cols)
            else:
                self.mosaic_layout.addWidget(label, idx // cols, idx % cols)

    def _create_crop_display(self, crop: np.ndarray, ocr_boxes: List, ocr_confs: List, digit_str: str) -> np.ndarray:
        """Create a display version of the crop with OCR boxes and borders."""
        crop_display = crop.copy()
        
        # Draw OCR boxes
        for i, box in enumerate(ocr_boxes):
            pts = np.array(box, dtype=np.int32)
            conf = ocr_confs[i] if ocr_confs and i < len(ocr_confs) else None
            color = (255, 0, 0)  # Red boxes
            cv2.polylines(crop_display, [pts], isClosed=True, color=color, thickness=2)
            
            # Draw confidence as text near the box
            if conf is not None:
                pt = pts[0]
                cv2.putText(crop_display, f"{conf:.2f}", (pt[0], pt[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        
        # Add green border if digits detected
        if digit_str and any(char.isdigit() for char in digit_str):
            border_color = (0, 255, 0)  # Green
            border_thickness = 8
            crop_display = cv2.copyMakeBorder(
                crop_display, border_thickness, border_thickness, 
                border_thickness, border_thickness,
                cv2.BORDER_CONSTANT, value=border_color
            )
        
        return crop_display

    def _array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap."""
        h, w = array.shape[:2]
        
        if len(array.shape) == 2:
            # Grayscale
            qimg = QImage(array.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # Color
            ch = array.shape[2]
            qimg = QImage(array.data, w, h, ch * w, QImage.Format_BGR888)
        
        return QPixmap.fromImage(qimg)
