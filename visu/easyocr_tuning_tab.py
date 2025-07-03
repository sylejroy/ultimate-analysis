import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QFormLayout, QGroupBox, QScrollArea, QGridLayout, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from processing.inference import run_inference
from processing.tracking import run_tracking
from processing.player_id import run_player_id, easyocr_reader

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

import json
from PyQt5.QtWidgets import QFileDialog

class DevEasyOCRTuningTab(QWidget):
    def __init__(self, input_folder="input/dev_data"):
        super().__init__()
        self.input_folder = input_folder
        self.cap = None
        self.current_video = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.ocr_params = self.default_ocr_params()
        self.preproc_params = self.default_preproc_params()
        self.player_crops = []
        self.player_results = []
        self.bw_mode = False
        self.colour_mode = False
        self.init_ui()
        self.populate_video_list()

    def default_ocr_params(self):
        # Use EasyOCR's documented defaults
        return {
            "detail": 1,
            "allowlist": None,
            "min_size": 20,
            "rotation_info": [0],
            "text_threshold": 0.7,
            "low_text": 0.4,
            "mag_ratio": 1.5,
            "contrast_ths": 0.1,
            "adjust_contrast": 0.5,
            "slope_ths": 0.1,
            "width_ths": 0.4,
            "decoder": "greedy",
        }

    def default_preproc_params(self):
        return {
            "blur_ksize": 3,
            "clahe_clip": 2.0,
            "clahe_grid": 4,
            "sharpen": 1.0,
            "upscale": 2.0,
        }

    def init_ui(self):
        layout = QHBoxLayout()
        self.video_list = QListWidget()
        self.video_list.itemSelectionChanged.connect(self.load_selected_video)
        layout.addWidget(self.video_list, 1)

        right = QVBoxLayout()
        self.frame_label = QLabel("No video loaded")
        self.frame_label.setAlignment(Qt.AlignCenter)
        right.addWidget(self.frame_label, 2)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        right.addWidget(self.slider)

        # Preprocessing parameter controls
        preproc_group = QGroupBox("Preprocessing Parameters")
        preproc_layout = QFormLayout()
        self.preproc_widgets = {}
        self.preproc_enables = {}
        for name, widget_type, *args in PREPROC_PARAMS:
            row = QHBoxLayout()
            enable = QCheckBox()
            enable.setChecked(True)
            enable.stateChanged.connect(self.update_preproc_params)
            self.preproc_enables[name] = enable
            if widget_type == QSpinBox:
                widget = QSpinBox()
                widget.setRange(args[0], args[1])
                widget.setValue(self.preproc_params[name])
                widget.valueChanged.connect(self.update_preproc_params)
            elif widget_type == QDoubleSpinBox:
                widget = QDoubleSpinBox()
                widget.setRange(args[0], args[1])
                widget.setSingleStep(args[2])
                widget.setValue(self.preproc_params[name])
                widget.valueChanged.connect(self.update_preproc_params)
            widget.setToolTip(PREPROC_PARAM_DESCRIPTIONS.get(name, ""))
            row.addWidget(enable)
            row.addWidget(widget)
            preproc_layout.addRow(name, row)
            self.preproc_widgets[name] = widget
        # Add black/white switch
        self.bw_checkbox = QCheckBox("Black/White Invert")
        self.bw_checkbox.setChecked(False)
        self.bw_checkbox.stateChanged.connect(self.toggle_bw_mode)
        preproc_layout.addRow("Invert B/W", self.bw_checkbox)
        # Add colour input switch
        self.colour_checkbox = QCheckBox("Use Colour Input")
        self.colour_checkbox.setChecked(False)
        self.colour_checkbox.stateChanged.connect(self.toggle_colour_mode)
        preproc_layout.addRow("Use Colour Input", self.colour_checkbox)

        preproc_group.setLayout(preproc_layout)
        right.addWidget(preproc_group)

        # OCR parameter controls
        param_group = QGroupBox("EasyOCR Parameters")
        param_layout = QFormLayout()
        self.param_widgets = {}
        for name, widget_type, *args in EASYOCR_PARAMS:
            if widget_type == QComboBox:
                widget = QComboBox()
                for v in args[0]:
                    widget.addItem(str(v))
                widget.setCurrentText(str(args[1]))
                widget.currentTextChanged.connect(self.update_params)
            elif widget_type == QSpinBox:
                widget = QSpinBox()
                widget.setRange(args[0], args[1])
                widget.setValue(args[2])
                widget.valueChanged.connect(self.update_params)
            elif widget_type == QDoubleSpinBox:
                widget = QDoubleSpinBox()
                widget.setRange(args[0], args[1])
                widget.setSingleStep(args[2])
                widget.setValue(args[3])
                widget.valueChanged.connect(self.update_params)
            widget.setToolTip(EASYOCR_PARAM_DESCRIPTIONS.get(name, ""))
            param_layout.addRow(name, widget)
            self.param_widgets[name] = widget
        param_group.setLayout(param_layout)
        right.addWidget(param_group)

        # Save/load parameter set buttons
        param_btn_row = QHBoxLayout()
        self.save_params_btn = QPushButton("Save Param Set")
        self.save_params_btn.clicked.connect(self.save_param_set)
        self.load_params_btn = QPushButton("Load Param Set")
        self.load_params_btn.clicked.connect(self.load_param_set)
        param_btn_row.addWidget(self.save_params_btn)
        param_btn_row.addWidget(self.load_params_btn)
        right.addLayout(param_btn_row)

        self.run_button = QPushButton("Run Player Detection + OCR on Frame")
        self.run_button.clicked.connect(self.run_detection_and_ocr)
        right.addWidget(self.run_button)

        # Mosaic area
        self.mosaic_area = QScrollArea()
        self.mosaic_widget = QWidget()
        self.mosaic_layout = QGridLayout()
        self.mosaic_widget.setLayout(self.mosaic_layout)
        self.mosaic_area.setWidget(self.mosaic_widget)
        self.mosaic_area.setWidgetResizable(True)
        right.addWidget(self.mosaic_area, 6)

        # Add the right panel to the main layout
        layout.addLayout(right, 4)
        # Only set the layout once, at the end of init_ui
        self.setLayout(layout)

    def toggle_bw_mode(self, state):
        self.bw_mode = bool(state)

    def populate_video_list(self):
        self.video_list.clear()
        if not os.path.exists(self.input_folder):
            return
        for f in os.listdir(self.input_folder):
            path = os.path.join(self.input_folder, f)
            if os.path.isfile(path) and f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                self.video_list.addItem(f)

    def load_selected_video(self):
        items = self.video_list.selectedItems()
        if not items:
            return
        filename = items[0].text()
        path = os.path.join(self.input_folder, filename)
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.current_video = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.slider.setMaximum(self.total_frames-1)
        self.slider.setValue(0)
        self.show_frame(0)

    def slider_moved(self, value):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        self.show_frame(value)

    def show_frame(self, frame_idx):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.frame_label.setPixmap(pixmap.scaled(
            self.frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.current_frame_img = frame

    def update_preproc_params(self):
        for name, widget in self.preproc_widgets.items():
            self.preproc_params[name] = widget.value()
        # No return needed, enables are handled in preprocess_crop

    def update_params(self):
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
            elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                self.ocr_params[name] = widget.value()

    def preprocess_crop(self, crop):
        # Step enables
        enabled = self.preproc_enables
        # Colour mode: skip grayscale and CLAHE, use original crop
        if self.colour_mode:
            proc = crop.copy()
        else:
            # CLAHE
            if enabled["clahe_clip"].isChecked() or enabled["clahe_grid"].isChecked():
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                proc = cv2.createCLAHE(clipLimit=self.preproc_params["clahe_clip"], tileGridSize=(self.preproc_params["clahe_grid"], self.preproc_params["clahe_grid"])) .apply(gray)
            else:
                proc = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Black/White invert
            if self.bw_mode:
                proc = cv2.bitwise_not(proc)
        # Sharpening
        if enabled["sharpen"].isChecked() and self.preproc_params["sharpen"] > 0:
            sharpen_strength = self.preproc_params["sharpen"]
            kernel = np.array([[0, -1, 0], [-1, 5 + sharpen_strength, -1], [0, -1, 0]])
            proc = cv2.filter2D(proc, -1, kernel)
        # Upscale
        if enabled["upscale"].isChecked() and self.preproc_params["upscale"] > 1.0:
            scale = self.preproc_params["upscale"]
            proc = cv2.resize(proc, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Blur
        if enabled["blur_ksize"].isChecked() and self.preproc_params["blur_ksize"] > 1:
            ksize = self.preproc_params["blur_ksize"]
            if ksize % 2 == 0:
                ksize += 1
            proc = cv2.GaussianBlur(proc, (ksize, ksize), 0)
        # Convert to 3-channel RGB for easyocr if not already
        if len(proc.shape) == 2:
            upscaled_rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
        else:
            upscaled_rgb = proc
        return upscaled_rgb, proc

    def save_param_set(self):
        params = {
            'ocr': self.ocr_params,
            'preproc': self.preproc_params,
            'preproc_enables': {k: v.isChecked() for k, v in self.preproc_enables.items()},
            'colour_mode': getattr(self, 'colour_mode', False),
            'bw_mode': getattr(self, 'bw_mode', False)
        }
        fname, _ = QFileDialog.getSaveFileName(self, "Save Parameter Set", "easyocr_params.json", "JSON Files (*.json)")
        if fname:
            with open(fname, 'w') as f:
                json.dump(params, f, indent=2)

    def load_param_set(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Parameter Set", "", "JSON Files (*.json)")
        if fname:
            with open(fname, 'r') as f:
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
            # Load colour mode if present
            if 'colour_mode' in params:
                self.colour_checkbox.setChecked(params['colour_mode'])
            # Load B/W mode if present
            if 'bw_mode' in params:
                self.bw_checkbox.setChecked(params['bw_mode'])

    def run_detection_and_ocr(self):
        if not hasattr(self, "current_frame_img"):
            return
        frame = self.current_frame_img.copy()
        detections = run_inference(frame)
        tracks = run_tracking(frame, detections)
        player_crops = []
        player_results = []
        for track in tracks:
            track_class = getattr(track, "det_class", None)
            PLAYER_CLASS_IDX = 1
            if track_class is not None and track_class != PLAYER_CLASS_IDX:
                continue
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
                    x1, y1, x2, y2 = bbox
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    w, h = x2 - x1, y2 - y1
                    if w > 0 and h > 0:
                        half_h = max(1, h // 2)
                        obj_crop = frame[y1:y1+half_h, x1:x2]
                        if obj_crop.size > 0:
                            preproc_rgb, preproc_gray = self.preprocess_crop(obj_crop)
                            ocr_results = easyocr_reader.readtext(
                                preproc_rgb,
                                **self.ocr_params
                            )
                            digit_str = ''.join([r[1] for r in ocr_results])
                            ocr_boxes = [r[0] for r in ocr_results]
                            ocr_confs = [r[2] if len(r) > 2 else None for r in ocr_results]
                            player_crops.append(preproc_gray)  # Show preprocessed grayscale in mosaic
                            player_results.append((digit_str, ocr_boxes, ocr_confs))
        self.display_mosaic(player_crops, player_results)

    def display_mosaic(self, crops, results):
        # Remove old widgets
        for i in reversed(range(self.mosaic_layout.count())):
            widget = self.mosaic_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        # Display crops in a grid
        cols = 6
        for idx, (crop, (digit_str, ocr_boxes, ocr_confs)) in enumerate(zip(crops, results)):
            label = QLabel()
            crop_disp = crop.copy()
            # Draw OCR boxes
            for i, box in enumerate(ocr_boxes):
                pts = np.array(box, dtype=np.int32)
                conf = ocr_confs[i] if ocr_confs and i < len(ocr_confs) else None
                color = (255, 0, 0)
                cv2.polylines(crop_disp, [pts], isClosed=True, color=color, thickness=2)
                # Draw confidence as text near the box
                if conf is not None:
                    pt = pts[0]
                    cv2.putText(crop_disp, f"{conf:.2f}", (pt[0], pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
            # If a digit was detected, add a thick green border
            if digit_str and any(char.isdigit() for char in digit_str):
                border_color = (0, 255, 0)  # Green
                border_thickness = 8
                crop_disp = cv2.copyMakeBorder(
                    crop_disp, border_thickness, border_thickness, border_thickness, border_thickness,
                    cv2.BORDER_CONSTANT, value=border_color
                )
            h, w = crop_disp.shape[:2]
            if len(crop_disp.shape) == 2:
                qimg = QImage(crop_disp.data, w, h, w, QImage.Format_Grayscale8)
            else:
                ch = crop_disp.shape[2]
                qimg = QImage(crop_disp.data, w, h, ch * w, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimg)
            label.setPixmap(pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            # Show the detected number underneath the image
            if digit_str:
                label.setToolTip(f"Digits: {digit_str}")
                label_text = QLabel(digit_str)
                label_text.setAlignment(Qt.AlignCenter)
                label_text.setStyleSheet("color: #00FF00; font-weight: bold;")
                vbox = QVBoxLayout()
                vbox.addWidget(label)
                vbox.addWidget(label_text)
                container = QWidget()
                container.setLayout(vbox)
                self.mosaic_layout.addWidget(container, idx // cols, idx % cols)
            else:
                self.mosaic_layout.addWidget(label, idx // cols, idx % cols)


    def toggle_colour_mode(self, state):
        self.colour_mode = bool(state)