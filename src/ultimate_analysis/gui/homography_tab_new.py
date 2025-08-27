"""Homography estimation tab for Ultimate Analysis GUI."""

import os
import yaml
import datetime
from typing import List, Dict, Optional
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, 
    QPushButton, QSlider, QListWidgetItem, QGroupBox,
    QFormLayout, QSplitter, QFileDialog, QMessageBox,
    QScrollArea, QSizePolicy, QCheckBox, QSpinBox, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent, QPainter, QPen, QColor

from .video_player import VideoPlayer
from ..config.settings import get_setting
from ..constants import DEFAULT_PATHS, SUPPORTED_VIDEO_EXTENSIONS
from ..processing.field_segmentation import run_field_segmentation, set_field_model
from ..gui.visualization import draw_field_segmentation


class ZoomableImageLabel(QLabel):
    """Custom QLabel that supports mouse wheel zooming."""
    
    zoom_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.original_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)
        
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if self.original_pixmap is None:
            return
        zoom_in = event.angleDelta().y() > 0
        zoom_delta = 0.15 if zoom_in else -0.15
        new_zoom = max(0.1, min(10.0, self.zoom_factor + zoom_delta))
        
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self._update_display()
            self.zoom_changed.emit(self.zoom_factor)
    
    def set_image(self, pixmap: QPixmap):
        """Set the image."""
        self.original_pixmap = pixmap
        self.zoom_factor = 1.0
        self._update_display()
        self.zoom_changed.emit(self.zoom_factor)
    
    def set_zoom(self, zoom_factor: float):
        """Set zoom factor."""
        self.zoom_factor = max(0.1, min(10.0, zoom_factor))
        self._update_display()
    
    def _update_display(self):
        """Update the displayed image with current zoom."""
        if self.original_pixmap is None:
            return
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.original_pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
        self.setMinimumSize(scaled_pixmap.size())


class HomographyTab(QWidget):
    """Interactive homography estimation tab."""
    
    def __init__(self):
        super().__init__()
        
        # Video player and state
        self.video_player = VideoPlayer()
        self.video_files: List[str] = []
        self.current_video_index: int = 0
        self.current_frame: Optional[np.ndarray] = None
        
        # Homography parameters
        self.homography_params = {
            'H00': 1.0, 'H01': 0.0, 'H02': 0.0,
            'H10': 0.0, 'H11': 1.0, 'H12': 0.0,
            'H20': 0.0, 'H21': 0.0
        }
        
        # UI components
        self.video_list: Optional[QListWidget] = None
        self.frame_label: Optional[QLabel] = None
        self.original_display: Optional[ZoomableImageLabel] = None
        self.warped_display: Optional[ZoomableImageLabel] = None
        self.param_sliders: Dict[str, QSlider] = {}
        self.param_labels: Dict[str, QLabel] = {}
        
        # Field segmentation state
        self.show_segmentation = False
        self.current_segmentation_results = None
        self.segmentation_model_combo: Optional[QComboBox] = None
        self.show_segmentation_checkbox: Optional[QCheckBox] = None
        self.available_segmentation_models: List[str] = []
        
        # Initialize UI
        self._init_ui()
        self._load_videos()
        self._load_segmentation_models()
        self._load_default_parameters()
        
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        content_splitter = QSplitter(Qt.Horizontal)
        
        left_panel = self._create_left_panel()
        content_splitter.addWidget(left_panel)
        
        right_panel = self._create_right_panel()
        content_splitter.addWidget(right_panel)
        
        content_splitter.setSizes([300, 1200])
        main_layout.addWidget(content_splitter)
        self.setLayout(main_layout)
        
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with controls."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Video selection
        video_group = QGroupBox("Video Selection")
        video_layout = QVBoxLayout()
        
        self.video_list = QListWidget()
        self.video_list.setMinimumHeight(150)
        self.video_list.currentRowChanged.connect(self._on_video_selection_changed)
        video_layout.addWidget(self.video_list)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._load_videos)
        video_layout.addWidget(refresh_button)
        
        self.frame_label = QLabel("0 / 0")
        video_layout.addWidget(self.frame_label)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Homography parameters
        params_group = QGroupBox("Homography Parameters")
        params_layout = QVBoxLayout()
        
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._reset_homography)
        params_layout.addWidget(reset_button)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with displays."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        header = QLabel("Homography Transformation")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Original frame
        original_group = QGroupBox("Original Frame")
        original_layout = QVBoxLayout()
        
        self.original_display = ZoomableImageLabel()
        self.original_display.setText("No video selected")
        original_layout.addWidget(self.original_display)
        
        original_group.setLayout(original_layout)
        content_splitter.addWidget(original_group)
        
        # Warped frame
        warped_group = QGroupBox("Warped Frame")
        warped_layout = QVBoxLayout()
        
        self.warped_display = ZoomableImageLabel()
        self.warped_display.setText("No video selected")
        warped_layout.addWidget(self.warped_display)
        
        warped_group.setLayout(warped_layout)
        content_splitter.addWidget(warped_group)
        
        content_splitter.setSizes([600, 600])
        layout.addWidget(content_splitter, 1)
        
        panel.setLayout(layout)
        return panel
    
    def _load_videos(self):
        """Load available video files."""
        self.video_files.clear()
        self.video_list.clear()
        
        search_paths = [
            Path(DEFAULT_PATHS['DEV_DATA']),
            Path(DEFAULT_PATHS['RAW_VIDEOS'])
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            for file_path in search_path.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    self.video_files.append(str(file_path))
        
        self.video_files.sort()
        
        for video_path in self.video_files:
            filename = Path(video_path).name
            item = QListWidgetItem(filename)
            item.setToolTip(video_path)
            self.video_list.addItem(item)
        
        if self.video_files:
            self.video_list.setCurrentRow(0)
            self._load_selected_video()
    
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
        
        if self.video_player.load_video(video_path):
            video_info = self.video_player.get_video_info()
            total_frames = video_info['total_frames']
            self.frame_label.setText(f"0 / {total_frames}")
            
            first_frame = self.video_player.get_current_frame()
            if first_frame is not None:
                self.current_frame = first_frame.copy()
                self._update_displays()
    
    def _update_displays(self):
        """Update both displays."""
        if self.current_frame is None:
            return
            
        # Display original frame
        self._display_frame(self.current_frame, self.original_display)
        
        # Create and display warped frame
        warped_frame = self._apply_homography(self.current_frame)
        self._display_frame(warped_frame, self.warped_display)
    
    def _display_frame(self, frame, display_label):
        """Display a frame in the given label."""
        if frame is None:
            return
        
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        display_label.set_image(pixmap)
    
    def _apply_homography(self, frame: np.ndarray) -> np.ndarray:
        """Apply homography transformation to frame."""
        h_matrix = self._get_homography_matrix()
        original_height, original_width = frame.shape[:2]
        warped = cv2.warpPerspective(frame, h_matrix, (original_width, original_height))
        return warped
    
    def _get_homography_matrix(self) -> np.ndarray:
        """Get current homography matrix."""
        return np.array([
            [self.homography_params['H00'], self.homography_params['H01'], self.homography_params['H02']],
            [self.homography_params['H10'], self.homography_params['H11'], self.homography_params['H12']],
            [self.homography_params['H20'], self.homography_params['H21'], 1.0]
        ])
    
    def _reset_homography(self):
        """Reset homography to identity matrix."""
        self.homography_params = {
            'H00': 1.0, 'H01': 0.0, 'H02': 0.0,
            'H10': 0.0, 'H11': 1.0, 'H12': 0.0,
            'H20': 0.0, 'H21': 0.0
        }
        self._update_displays()
    
    def _load_segmentation_models(self):
        """Load segmentation models."""
        self.available_segmentation_models = []
    
    def _load_default_parameters(self):
        """Load default homography parameters."""
        pass
    
    def _on_segmentation_toggled(self, state: int):
        """Handle segmentation toggle."""
        pass
    
    def _on_segmentation_model_changed(self, model_name: str):
        """Handle segmentation model change."""
        pass