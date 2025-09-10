"""Homography estimation tab for Ultimate Analysis GUI.

This module provides an interactive interface for adjusting homography parameters
with real-time perspective transformation visualization and YAML save/load functionality.
"""

import os
import yaml
import datetime
import time
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, 
    QPushButton, QSlider, QListWidgetItem, QGroupBox,
    QFormLayout, QSplitter, QFileDialog, QMessageBox,
    QScrollArea, QSizePolicy, QCheckBox, QSpinBox, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent, QPainter, QPen, QColor, QMouseEvent

from .video_player import VideoPlayer
from ..config.settings import get_setting
from ..constants import DEFAULT_PATHS, SUPPORTED_VIDEO_EXTENSIONS
from ..processing.field_segmentation import run_field_segmentation, set_field_model
from ..gui.visualization import (
    draw_field_segmentation, 
    create_unified_field_mask, 
    draw_unified_field_mask,
    calculate_field_contour,
    draw_field_contour,
    draw_field_lines_ransac,
    draw_classified_field_lines,
    draw_all_field_lines
)
from ..processing.line_extraction import extract_raw_lines_from_segmentation
from .ransac_line_visualization import draw_ransac_field_lines


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
        
        # Grid overlay properties
        self.show_grid = True
        self.grid_spacing = 50  # pixels at 1.0 zoom
        self.grid_color = QColor(255, 255, 255, 80)  # Semi-transparent white
        self.grid_line_width = 1
        
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming to mouse position."""
        if self.original_pixmap is None:
            return
            
        # Get the scroll area parent
        scroll_area = None
        parent = self.parent()
        while parent:
            if isinstance(parent, QScrollArea):
                scroll_area = parent
                break
            parent = parent.parent()
        
        if scroll_area is None:
            # Fallback to center zoom if no scroll area found
            zoom_in = event.angleDelta().y() > 0
            zoom_delta = 0.15 if zoom_in else -0.15
            new_zoom = max(0.1, min(10.0, self.zoom_factor + zoom_delta))
            
            if new_zoom != self.zoom_factor:
                self.zoom_factor = new_zoom
                self._update_display()
                self.zoom_changed.emit(self.zoom_factor)
            return
        
        # Get mouse position
        mouse_pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        
        # Calculate zoom change
        zoom_in = event.angleDelta().y() > 0
        zoom_delta = 0.15 if zoom_in else -0.15
        old_zoom = self.zoom_factor
        new_zoom = max(0.1, min(10.0, old_zoom + zoom_delta))
        
        if new_zoom == old_zoom:
            return
        
        # Get scrollbars
        h_scroll = scroll_area.horizontalScrollBar()
        v_scroll = scroll_area.verticalScrollBar()
        
        # Store old scroll positions
        old_h = h_scroll.value()
        old_v = v_scroll.value()
        
        # Calculate mouse position in the "image coordinate system"
        # This is the key: we need to find what point in the original image
        # is currently under the mouse cursor
        
        # For a QLabel with pixmap, the coordinate calculation is:
        # 1. Mouse position relative to the label widget
        # 2. Account for the label's alignment (center alignment means offset)
        # 3. Account for scroll position
        # 4. Account for current zoom level
        
        widget_rect = self.rect()
        if self.pixmap():
            pixmap_size = self.pixmap().size()
            
            # Calculate where the pixmap is positioned within the widget
            # QLabel centers the pixmap when it's smaller than the widget
            x_offset = max(0, (widget_rect.width() - pixmap_size.width()) // 2)
            y_offset = max(0, (widget_rect.height() - pixmap_size.height()) // 2)
            
            # Mouse position relative to the actual image (accounting for centering)
            img_mouse_x = mouse_pos.x() - x_offset
            img_mouse_y = mouse_pos.y() - y_offset
            
            # Account for scroll position and current zoom to get original image coordinates
            orig_img_x = (img_mouse_x + old_h) / old_zoom
            orig_img_y = (img_mouse_y + old_v) / old_zoom
            
            # Apply new zoom
            self.zoom_factor = new_zoom
            self._update_display()
            
            # Calculate new offsets after zoom
            if self.pixmap():
                new_pixmap_size = self.pixmap().size()
                new_x_offset = max(0, (widget_rect.width() - new_pixmap_size.width()) // 2)
                new_y_offset = max(0, (widget_rect.height() - new_pixmap_size.height()) // 2)
                
                # Calculate where the scroll position should be to keep the same
                # original image point under the mouse
                target_img_x = orig_img_x * new_zoom
                target_img_y = orig_img_y * new_zoom
                
                new_h = target_img_x - (mouse_pos.x() - new_x_offset)
                new_v = target_img_y - (mouse_pos.y() - new_y_offset)
                
                # Apply new scroll positions with bounds checking
                h_scroll.setValue(max(0, min(h_scroll.maximum(), int(new_h))))
                v_scroll.setValue(max(0, min(v_scroll.maximum(), int(new_v))))
        else:
            # If no pixmap, just update zoom
            self.zoom_factor = new_zoom
            self._update_display()
        
        self.zoom_changed.emit(self.zoom_factor)
    
    def set_image(self, pixmap: QPixmap):
        """Set the image and reset zoom to fit the container."""
        self.original_pixmap = pixmap
        # Calculate initial zoom to fit the container while maintaining aspect ratio
        if pixmap and not pixmap.isNull():
            container_size = self.size()
            pixmap_size = pixmap.size()
            
            # Calculate scale factors for width and height
            scale_w = container_size.width() / pixmap_size.width() if pixmap_size.width() > 0 else 1.0
            scale_h = container_size.height() / pixmap_size.height() if pixmap_size.height() > 0 else 1.0
            
            # Use the smaller scale factor to ensure the image fits completely
            initial_zoom = min(scale_w, scale_h, 1.0)  # Don't scale up beyond original size initially
            self.zoom_factor = max(0.1, initial_zoom)
        else:
            self.zoom_factor = 1.0
            
        self._update_display()
        self.zoom_changed.emit(self.zoom_factor)
    
    def set_zoom(self, zoom_factor: float):
        """Set zoom factor programmatically."""
        self.zoom_factor = max(0.1, min(10.0, zoom_factor))
        self._update_display()
    
    def _update_display(self):
        """Update the displayed image with current zoom."""
        if self.original_pixmap is None:
            return
            
        # Scale the pixmap
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.original_pixmap.scaled(
            scaled_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap)
        
        # Set the minimum size so the scroll area recognizes the content size
        self.setMinimumSize(scaled_pixmap.size())
        
        # Also set the size hint for proper scroll area calculation
        self.resize(scaled_pixmap.size())
        
        # Update the scroll area geometry
        parent = self.parent()
        if isinstance(parent, QScrollArea):
            parent.updateGeometry()
    
    def paintEvent(self, event):
        """Override paint event to draw grid overlay on top of the image."""
        # First, let the parent QLabel draw the image
        super().paintEvent(event)
        
        # Draw grid overlay if enabled and we have an image
        if self.show_grid and self.pixmap() and not self.pixmap().isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            
            # Set up the grid pen
            pen = QPen(self.grid_color)
            pen.setWidth(self.grid_line_width)
            painter.setPen(pen)
            
            # Get the image area within the widget
            pixmap_rect = self.pixmap().rect()
            widget_rect = self.rect()
            
            # Calculate the image position (centered in widget)
            x_offset = max(0, (widget_rect.width() - pixmap_rect.width()) // 2)
            y_offset = max(0, (widget_rect.height() - pixmap_rect.height()) // 2)
            
            # Calculate actual grid spacing based on current zoom
            actual_grid_spacing = self.grid_spacing * self.zoom_factor
            
            # Draw vertical lines
            image_left = x_offset
            image_right = x_offset + pixmap_rect.width()
            image_top = y_offset
            image_bottom = y_offset + pixmap_rect.height()
            
            # Start from the first grid line within the image
            start_x = image_left + (actual_grid_spacing - (image_left % actual_grid_spacing)) % actual_grid_spacing
            x = start_x
            while x < image_right:
                painter.drawLine(int(x), image_top, int(x), image_bottom)
                x += actual_grid_spacing
            
            # Draw horizontal lines
            start_y = image_top + (actual_grid_spacing - (image_top % actual_grid_spacing)) % actual_grid_spacing
            y = start_y
            while y < image_bottom:
                painter.drawLine(image_left, int(y), image_right, int(y))
                y += actual_grid_spacing
            
            painter.end()
    
    def set_grid_visible(self, visible: bool):
        """Toggle grid visibility."""
        self.show_grid = visible
        self.update()  # Trigger a repaint
    
    def set_grid_spacing(self, spacing: int):
        """Set grid spacing in pixels at 1.0 zoom."""
        self.grid_spacing = spacing
        self.update()  # Trigger a repaint
    
    def set_grid_color(self, color: QColor):
        """Set grid color."""
        self.grid_color = color
        self.update()  # Trigger a repaint


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
        
        # Scrubbing controls
        self.scrubbing_slider: Optional[QSlider] = None
        self.scrubbing_frame_label: Optional[QLabel] = None
        self.current_video_label: Optional[QLabel] = None
        
        # Zoom functionality
        self.original_scroll_area: Optional[QScrollArea] = None
        self.warped_scroll_area: Optional[QScrollArea] = None
        
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
        
        # Runtime performance tracking
        self.runtime_popup: Optional[QDialog] = None
        self.runtime_table: Optional[QTableWidget] = None
        self.runtime_button: Optional[QPushButton] = None
        self.processing_times: Dict[str, List[float]] = {
            'Field Segmentation': [],
            'Morphological Ops': [],
            'RANSAC Fitting': [],
            'Line Tracking': [],
            'Homography Calc': [],
            'Display Update': []
        }
        
        # Lazy loading flags
        self._videos_loaded = False
        self._segmentation_models_loaded = False
        self._ui_initialized = False
        
        # Initialize UI only
        self._init_ui()
        
        # Load default homography parameters from config
        self._load_default_parameters()
        
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Main content with splitter
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Video list and parameter controls
        left_panel = self._create_left_panel()
        content_splitter.addWidget(left_panel)
        
        # Right panel: Side-by-side video displays
        right_panel = self._create_right_panel()
        content_splitter.addWidget(right_panel)
        
        # Set splitter proportions (25% left, 75% right for larger image display)
        content_splitter.setSizes([300, 1200])
        
        main_layout.addWidget(content_splitter)
        
        self.setLayout(main_layout)
    
    def showEvent(self, event):
        """Override showEvent to implement lazy loading when tab becomes visible."""
        super().showEvent(event)
        
        # Lazy load content when tab is first shown
        if not self._videos_loaded:
            print("[HOMOGRAPHY] Lazy loading videos...")
            self._load_videos()
            self._videos_loaded = True
            
        if not self._segmentation_models_loaded:
            print("[HOMOGRAPHY] Lazy loading segmentation models...")
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
        refresh_button.clicked.connect(self._force_reload_videos)
        refresh_button.setToolTip("Refresh video list")
        list_header.addWidget(refresh_button)
        
        video_layout.addLayout(list_header)
        
        # Video list widget
        self.video_list = QListWidget()
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
        save_default_button.setStyleSheet("background-color: #2c5aa0; color: white; font-weight: bold;")
        button_layout2.addWidget(save_default_button)
        
        load_default_button = QPushButton("Load Default")
        load_default_button.clicked.connect(self._load_default_parameters)
        load_default_button.setToolTip("Load default parameters from config")
        button_layout2.addWidget(load_default_button)
        
        params_layout.addLayout(button_layout)
        params_layout.addLayout(button_layout2)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Field Segmentation Controls
        segmentation_group = QGroupBox("Field Segmentation")
        segmentation_layout = QVBoxLayout()
        
        # Show segmentation checkbox
        self.show_segmentation_checkbox = QCheckBox("Show Field Segmentation")
        self.show_segmentation_checkbox.setChecked(self.show_segmentation)
        self.show_segmentation_checkbox.stateChanged.connect(self._on_segmentation_toggled)
        segmentation_layout.addWidget(self.show_segmentation_checkbox)
        
        # RANSAC line fitting checkbox
        self.ransac_checkbox = QCheckBox("Use RANSAC Line Fitting")
        self.ransac_checkbox.setChecked(get_setting("models.segmentation.contour.ransac.enabled", True))
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
        
        # Runtime performance button
        runtime_group = QGroupBox("Performance Monitoring")
        runtime_layout = QVBoxLayout()
        
        self.runtime_button = QPushButton("Show Runtime Performance")
        self.runtime_button.setMinimumHeight(40)
        self.runtime_button.setStyleSheet("""
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
        """)
        self.runtime_button.clicked.connect(self._show_runtime_popup)
        runtime_layout.addWidget(self.runtime_button)
        
        runtime_group.setLayout(runtime_layout)
        layout.addWidget(runtime_group)
        
        # Initialize popup window (hidden)
        self.runtime_popup = None
        self.runtime_table = None
        
        # Add stretch to push everything to top
        layout.addStretch()
        
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
        self.original_display.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #1a1a1a;
                color: #999;
                font-size: 12px;
            }
        """)
        
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
        self.warped_display.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #1a1a1a;
                color: #999;
                font-size: 12px;
            }
        """)
        
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
        self.scrubbing_frame_label.setStyleSheet("color: #ccc; font-family: monospace; font-size: 10px;")
        info_layout.addWidget(self.scrubbing_frame_label)
        
        layout.addLayout(info_layout)
        
        # Compact scrubbing slider
        self.scrubbing_slider = QSlider(Qt.Horizontal)
        self.scrubbing_slider.setMinimum(0)
        self.scrubbing_slider.setMaximum(0)
        self.scrubbing_slider.setValue(0)
        self.scrubbing_slider.setMinimumHeight(20)  # Smaller for compact layout
        self.scrubbing_slider.valueChanged.connect(self._on_scrubbing_changed)
        self.scrubbing_slider.setStyleSheet("""
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
        """)
        layout.addWidget(self.scrubbing_slider)
        
        panel.setLayout(layout)
        return panel
        
    def _create_parameter_controls(self, layout: QVBoxLayout):
        """Create slider controls for homography parameters."""
        # Get slider ranges from configuration with proper type conversion
        h_range_main = get_setting("homography.slider_range_main", [-50.0, 50.0])
        h_range_persp = get_setting("homography.slider_range_perspective", [-0.2, 0.2])
        
        # Ensure ranges are numeric (convert from strings if needed)
        if isinstance(h_range_main, list) and len(h_range_main) == 2:
            h_range_main = [float(h_range_main[0]), float(h_range_main[1])]
        else:
            h_range_main = [-50.0, 50.0]  # Expanded fallback range
            
        if isinstance(h_range_persp, list) and len(h_range_persp) == 2:
            h_range_persp = [float(h_range_persp[0]), float(h_range_persp[1])]
        else:
            h_range_persp = [-0.2, 0.2]  # Expanded fallback range
        
        # Parameters with their ranges and default values
        param_config = [
            ('H00', h_range_main, 1.0, "Scale X"),
            ('H01', h_range_main, 0.0, "Skew X"),
            ('H02', [-10000.0, 10000.0], 0.0, "Translate X"),
            ('H10', h_range_main, 0.0, "Skew Y"), 
            ('H11', h_range_main, 1.0, "Scale Y"),
            ('H12', [-10000.0, 10000.0], 0.0, "Translate Y"),
            ('H20', h_range_persp, 0.0, "Perspective X"),
            ('H21', h_range_persp, 0.0, "Perspective Y"),
        ]
        
        # Create form layout for parameters
        form_layout = QFormLayout()
        
        for param_name, param_range, default_val, label_text in param_config:
            # Parameter label and value display
            param_layout = QVBoxLayout()
            
            # Slider for parameter
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)  # Use 1000 steps for precision
            
            # Map default value to slider position
            slider_val = int(((default_val - param_range[0]) / (param_range[1] - param_range[0])) * 1000)
            slider.setValue(slider_val)
            
            # Connect to update function
            slider.valueChanged.connect(lambda val, name=param_name: self._on_parameter_changed(name, val))
            self.param_sliders[param_name] = slider
            
            # Value label
            value_label = QLabel(f"{default_val:.6f}")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet("font-family: monospace; font-size: 10px;")
            self.param_labels[param_name] = value_label
            
            # Combine slider and label
            param_layout.addWidget(slider)
            param_layout.addWidget(value_label)
            
            # Add to form
            combined_widget = QWidget()
            combined_widget.setLayout(param_layout)
            form_layout.addRow(f"{label_text} ({param_name}):", combined_widget)
            
        layout.addLayout(form_layout)
        
    def _load_videos(self):
        """Load and display available video files."""
        print("[HOMOGRAPHY] Loading video files...")
        
        self.video_files.clear()
        self.video_list.clear()
        
        # Search paths for videos
        search_paths = [
            Path(DEFAULT_PATHS['DEV_DATA']),
            Path(DEFAULT_PATHS['RAW_VIDEOS'])
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            # Find video files
            for file_path in search_path.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    self.video_files.append(str(file_path))
        
        # Sort videos by name
        self.video_files.sort()
        
        # Populate list
        for video_path in self.video_files:
            filename = Path(video_path).name
            item = QListWidgetItem(filename)
            item.setToolTip(video_path)
            self.video_list.addItem(item)
        
        print(f"[HOMOGRAPHY] Found {len(self.video_files)} video files")
        
        # Auto-select first video if available
        if self.video_files:
            self.video_list.setCurrentRow(0)
            self._load_selected_video()
    
    def _force_reload_videos(self):
        """Force reload videos even if already loaded (for refresh button)."""
        print("[HOMOGRAPHY] Force reloading videos...")
        self._videos_loaded = False  # Reset flag to allow reloading
        self._load_videos()
        self._videos_loaded = True
            
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
        print(f"[HOMOGRAPHY] Loading video: {video_path}")
        
        # Load video
        if self.video_player.load_video(video_path):
            # Update UI
            video_info = self.video_player.get_video_info()
            total_frames = video_info['total_frames']
            self.frame_label.setText(f"0 / {total_frames}")
            
            # Update scrubbing controls
            if hasattr(self, 'scrubbing_slider'):
                self.scrubbing_slider.setMaximum(max(1, total_frames - 1))
                self.scrubbing_slider.setValue(0)
            if hasattr(self, 'scrubbing_frame_label'):
                self.scrubbing_frame_label.setText(f"Frame: 0 / {total_frames}")
            if hasattr(self, 'current_video_label'):
                video_name = os.path.basename(video_path)
                self.current_video_label.setText(video_name)
            
            # Display first frame
            first_frame = self.video_player.get_current_frame()
            if first_frame is not None:
                self.current_frame = first_frame.copy()
                self._update_displays()
                
    def _on_frame_changed(self, frame_idx: int):
        """Handle frame slider change."""
        if self.video_player.is_loaded():
            self.video_player.seek_to_frame(frame_idx)
            
            # Update frame label
            video_info = self.video_player.get_video_info()
            total_frames = video_info['total_frames']
            self.frame_label.setText(f"{frame_idx} / {total_frames}")
            
            # Sync scrubbing slider
            if hasattr(self, 'scrubbing_slider'):
                self.scrubbing_slider.blockSignals(True)
                self.scrubbing_slider.setValue(frame_idx)
                self.scrubbing_slider.blockSignals(False)
                # Update scrubbing frame label
                if hasattr(self, 'scrubbing_frame_label'):
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
                
    def _on_parameter_changed(self, param_name: str, slider_value: int):
        """Handle homography parameter change from slider."""
        # Get parameter range with proper type conversion
        if param_name in ['H00', 'H01', 'H10', 'H11']:
            param_range = get_setting("homography.slider_range_main", [-50.0, 50.0])
            if isinstance(param_range, list) and len(param_range) == 2:
                param_range = [float(param_range[0]), float(param_range[1])]
            else:
                param_range = [-50.0, 50.0]
        elif param_name in ['H20', 'H21']:
            param_range = get_setting("homography.slider_range_perspective", [-0.2, 0.2])
            if isinstance(param_range, list) and len(param_range) == 2:
                param_range = [float(param_range[0]), float(param_range[1])]
            else:
                param_range = [-0.2, 0.2]
        else:  # Translation parameters
            param_range = [-10000.0, 10000.0]
            
        # Convert slider value (0-1000) to parameter value
        normalized_val = slider_value / 1000.0
        param_value = param_range[0] + normalized_val * (param_range[1] - param_range[0])
        
        # Update parameter
        self.homography_params[param_name] = param_value
        
        # Update value label
        self.param_labels[param_name].setText(f"{param_value:.6f}")
        
        # Update displays
        self._update_displays()
        
    def _update_displays(self):
        """Update both original and warped frame displays."""
        if self.current_frame is None:
            return
        
        update_start = time.time()
        
        # Apply segmentation overlay to original frame if enabled
        original_frame = self.current_frame.copy()
        if self.show_segmentation and self.current_segmentation_results:
            # Create and display unified mask on original frame
            morphological_start = time.time()
            frame_shape = self.current_frame.shape[:2]  # (height, width)
            unified_mask = create_unified_field_mask(self.current_segmentation_results, frame_shape)
            morphological_duration = (time.time() - morphological_start) * 1000
            self._add_runtime_measurement('Morphological Ops', morphological_duration)
            
            if unified_mask is not None:
                # Use bright green for unified field mask - contour only for consistency with main tab
                field_color = (0, 255, 0)  # Bright green (BGR)
                
                # Time the line extraction and tracking steps
                extraction_start = time.time()
                
                # Extract raw RANSAC lines for direct use
                detected_lines, confidences = extract_raw_lines_from_segmentation(
                    self.current_segmentation_results, frame_shape)
                
                extraction_duration = (time.time() - extraction_start) * 1000
                self._add_runtime_measurement('Line Extraction', extraction_duration)
                
                # Store RANSAC lines directly
                if detected_lines:
                    self.ransac_lines = detected_lines
                    self.ransac_confidences = confidences
                    print(f"[HOMOGRAPHY] Using {len(self.ransac_lines)} RANSAC lines directly")
                else:
                    self.ransac_lines = []
                    self.ransac_confidences = []
                
                # Draw unified mask for visualization (lightweight version without RANSAC re-computation)
                original_frame, raw_lines_dict, self.all_lines_for_display = draw_unified_field_mask(
                    original_frame, unified_mask, field_color, alpha=0.4, fill_mask=False)
                
                print(f"[HOMOGRAPHY] Applied field contour (no fill) to original frame: {np.sum(unified_mask)} pixels")
            else:
                print("[HOMOGRAPHY] No unified mask could be created for original frame")
                self.ransac_lines = []
                self.ransac_confidences = []
                self.all_lines_for_display = {}
        elif self.show_segmentation:
            print("[HOMOGRAPHY] Segmentation enabled but no results available")
        
        # Display original frame (with optional segmentation overlay)
        self._display_frame(original_frame, self.original_display)
        
        # Create warped frame
        homography_start = time.time()
        warped_frame = self._apply_homography(self.current_frame)
        homography_duration = (time.time() - homography_start) * 1000
        self._add_runtime_measurement('Homography Calc', homography_duration)
        
        print(f"[HOMOGRAPHY] Warped frame shape: {warped_frame.shape}, dtype: {warped_frame.dtype}")
        
        # For warped view, apply segmentation overlay if enabled
        if self.show_segmentation and self.current_segmentation_results:
            try:
                # Transform the segmentation masks to match the warped frame
                warped_frame_with_segmentation = self._apply_segmentation_to_warped_frame(warped_frame, self.current_segmentation_results)
                if warped_frame_with_segmentation is not None:
                    warped_frame = warped_frame_with_segmentation
                    print(f"[HOMOGRAPHY] Applied transformed segmentation to warped frame: {len(self.current_segmentation_results)} results")
                else:
                    print("[HOMOGRAPHY] Failed to apply transformed segmentation to warped frame")
            except Exception as e:
                print(f"[HOMOGRAPHY] Error applying segmentation to warped frame: {e}")
        elif self.show_segmentation:
            print("[HOMOGRAPHY] Segmentation enabled but no results available for warped frame")
        
        self._display_frame(warped_frame, self.warped_display)
        
        # Record total display update time
        update_duration = (time.time() - update_start) * 1000
        self._add_runtime_measurement('Display Update', update_duration)
        
    def _apply_segmentation_to_warped_frame(self, warped_frame: np.ndarray, segmentation_results: list) -> np.ndarray:
        """Apply segmentation overlay to warped frame by transforming contour points from original image."""
        if not segmentation_results:
            return warped_frame
            
        try:
            # Create unified mask from segmentation results on original frame
            original_frame_shape = self.current_frame.shape[:2]  # (height, width)
            unified_mask = create_unified_field_mask(segmentation_results, original_frame_shape)
            
            if unified_mask is None:
                print("[HOMOGRAPHY] No unified mask could be created")
                return warped_frame
            
            print(f"[HOMOGRAPHY] Created unified mask with shape {unified_mask.shape}, {np.sum(unified_mask)} pixels")
            
            # Calculate contour on the original image
            original_contour = calculate_field_contour(unified_mask)
            
            if original_contour is None or len(original_contour) == 0:
                print("[HOMOGRAPHY] No contour found in original unified mask")
                return warped_frame
            
            # Transform contour points using homography matrix
            h_matrix = self._get_homography_matrix()
            transformed_contour = self._transform_contour_points(original_contour, h_matrix)
            
            if transformed_contour is None:
                print("[HOMOGRAPHY] Failed to transform contour points")
                return warped_frame
            
            # Create mask from transformed contour points
            warped_mask = np.zeros((warped_frame.shape[0], warped_frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(warped_mask, [transformed_contour], 1)
            
            # Apply overlay and draw contour on warped frame - contour only for consistency
            field_color = (0, 255, 0)  # Bright green (BGR)
            result_frame, _, _ = draw_unified_field_mask(warped_frame, warped_mask, field_color, 
                                                       alpha=0.4, draw_contour=False, fill_mask=False)
            
            # Draw the transformed contour directly
            result_frame = draw_field_contour(result_frame, transformed_contour)
            
            # Draw all lines (classified and unclassified) if available
            if self.all_lines_for_display:
                homography_matrix = self._get_homography_matrix()
                result_frame = draw_all_field_lines(result_frame, self.all_lines_for_display, 
                                                  homography_matrix, scale_factor=2.0, 
                                                  draw_raw_lines_only=False)  # Show classified lines with simple labels
                print(f"[HOMOGRAPHY] Drew {len(self.all_lines_for_display)} total field lines on warped frame")
            elif self.ransac_lines:
                # Draw RANSAC lines in top-down view
                homography_matrix = self._get_homography_matrix()
                result_frame = draw_ransac_field_lines(result_frame, self.ransac_lines, 
                                                      self.ransac_confidences, homography_matrix, 
                                                      scale_factor=2.0)
                print(f"[HOMOGRAPHY] Drew {len(self.ransac_lines)} RANSAC field lines on warped frame")
            
            print(f"[HOMOGRAPHY] Applied transformed contour to warped frame: {len(transformed_contour)} points")
            return result_frame
            
        except Exception as e:
            print(f"[HOMOGRAPHY] Error applying transformed segmentation to warped frame: {e}")
            import traceback
            traceback.print_exc()
            return warped_frame
    
    def _get_homography_matrix(self) -> np.ndarray:
        """Get the homography transformation matrix from current parameters."""
        # Construct homography matrix from H parameters
        h_matrix = np.array([
            [self.homography_params['H00'], self.homography_params['H01'], self.homography_params['H02']],
            [self.homography_params['H10'], self.homography_params['H11'], self.homography_params['H12']],
            [self.homography_params['H20'], self.homography_params['H21'], 1.0]
        ], dtype=np.float32)
        
        return h_matrix
    
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
        
        print(f"[HOMOGRAPHY] Canvas size: {input_width}x{input_height} -> {output_width}x{output_height} (aspect {aspect_ratio:.1f}:1, area: {input_width*input_height} -> {output_width*output_height})")
        return output_width, output_height
    
    def _transform_contour_points(self, contour: np.ndarray, h_matrix: np.ndarray) -> Optional[np.ndarray]:
        """Transform contour points using homography matrix.
        
        Args:
            contour: Contour points as numpy array of shape (N, 1, 2)
            h_matrix: 3x3 homography transformation matrix
            
        Returns:
            Transformed contour points in same format, or None if transformation fails
        """
        if contour is None or len(contour) == 0:
            return None
            
        try:
            # Reshape contour points for perspective transformation
            # contour is (N, 1, 2), we need (N, 2) for cv2.perspectiveTransform
            points = contour.reshape(-1, 1, 2).astype(np.float32)
            
            # Apply perspective transformation
            transformed_points = cv2.perspectiveTransform(points, h_matrix)
            
            # Reshape back to contour format (N, 1, 2)
            transformed_contour = transformed_points.reshape(-1, 1, 2).astype(np.int32)
            
            print(f"[HOMOGRAPHY] Transformed {len(contour)} contour points")
            return transformed_contour
            
        except Exception as e:
            print(f"[HOMOGRAPHY] Error transforming contour points: {e}")
            return None
    
    def _display_frame(self, frame: np.ndarray, label: ZoomableImageLabel):
        """Display a frame in the specified zoomable label widget."""
        if frame is None:
            return
            
        # Convert to Qt format
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Create pixmap and set it to the zoomable label
        pixmap = QPixmap.fromImage(q_image)
        label.set_image(pixmap)
        
    def _apply_homography(self, frame: np.ndarray) -> np.ndarray:
        """Apply homography transformation to frame.
        
        Args:
            frame: Input frame to transform
            
        Returns:
            Warped frame using 3:1 aspect ratio canvas
        """
        # Use the same homography matrix calculation as for segmentation
        h_matrix = self._get_homography_matrix()
        
        # Get original dimensions
        original_height, original_width = frame.shape[:2]
        
        # Calculate output canvas size with 3:1 aspect ratio
        output_width, output_height = self._calculate_output_canvas_size(original_width, original_height)
        
        # Apply perspective transform using calculated canvas size
        warped = cv2.warpPerspective(frame, h_matrix, (output_width, output_height))
        
        return warped
        
    def _reset_homography(self):
        """Reset homography to identity matrix."""
        # Reset parameters to identity
        identity_params = {
            'H00': 1.0, 'H01': 0.0, 'H02': 0.0,
            'H10': 0.0, 'H11': 1.0, 'H12': 0.0,
            'H20': 0.0, 'H21': 0.0
        }
        
        # Update sliders and parameters
        for param_name, value in identity_params.items():
            self.homography_params[param_name] = value
            
            # Update slider position with proper type conversion
            if param_name in ['H00', 'H01', 'H10', 'H11']:
                param_range = get_setting("homography.slider_range_main", [-50.0, 50.0])
                if isinstance(param_range, list) and len(param_range) == 2:
                    param_range = [float(param_range[0]), float(param_range[1])]
                else:
                    param_range = [-50.0, 50.0]
            elif param_name in ['H20', 'H21']:
                param_range = get_setting("homography.slider_range_perspective", [-0.2, 0.2])
                if isinstance(param_range, list) and len(param_range) == 2:
                    param_range = [float(param_range[0]), float(param_range[1])]
                else:
                    param_range = [-0.2, 0.2]
            else:
                param_range = [-10000.0, 10000.0]
                
            slider_val = int(((value - param_range[0]) / (param_range[1] - param_range[0])) * 1000)
            self.param_sliders[param_name].setValue(slider_val)
            
            # Update label
            self.param_labels[param_name].setText(f"{value:.6f}")
        
        # Update displays
        self._update_displays()
        
        print("[HOMOGRAPHY] Reset to identity matrix")
        
    def _save_parameters(self):
        """Save current homography parameters to YAML file."""
        # Use configs directory as default save location
        homography_dir = Path(get_setting("homography.save_directory", "configs"))
        homography_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate default filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"homography_params_{timestamp}.yaml"
        
        # Show save dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Homography Parameters",
            str(homography_dir / default_filename),
            "YAML files (*.yaml *.yml);;All files (*.*)"
        )
        
        if filename:
            try:
                # Prepare data for saving
                save_data = {
                    'homography_parameters': self.homography_params.copy(),
                    'metadata': {
                        'created_at': datetime.datetime.now().isoformat(),
                        'video_file': Path(self.video_files[self.current_video_index]).name if self.video_files else None,
                        'frame_index': self.scrubbing_slider.value() if hasattr(self, 'scrubbing_slider') else 0,
                        'application': 'Ultimate Analysis',
                        'version': '1.0'
                    }
                }
                
                # Save to YAML
                with open(filename, 'w', encoding='utf-8') as f:
                    yaml.dump(save_data, f, default_flow_style=False, sort_keys=False)
                
                QMessageBox.information(self, "Success", f"Parameters saved to:\n{filename}")
                print(f"[HOMOGRAPHY] Saved parameters to: {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save parameters:\n{str(e)}")
                print(f"[HOMOGRAPHY] Error saving parameters: {e}")
    
    def _save_as_default_parameters(self):
        """Save current parameters as the default parameters file."""
        try:
            # Get the default parameters file path from config
            default_params_file = get_setting("homography.default_params_file", "configs/homography_params.yaml")
            
            # Ensure the directory exists
            Path(default_params_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for saving
            save_data = {
                'homography_parameters': self.homography_params.copy(),
                'metadata': {
                    'created_at': datetime.datetime.now().isoformat(),
                    'description': "Default homography parameters (updated by user)",
                    'video_file': Path(self.video_files[self.current_video_index]).name if self.video_files else None,
                    'frame_index': self.scrubbing_slider.value() if hasattr(self, 'scrubbing_slider') else 0,
                    'application': 'Ultimate Analysis',
                    'version': '1.0'
                }
            }
            
            # Save to YAML
            with open(default_params_file, 'w', encoding='utf-8') as f:
                yaml.dump(save_data, f, default_flow_style=False, sort_keys=False)
            
            QMessageBox.information(self, "Success", 
                                  f"Parameters saved as default to:\n{default_params_file}\n\n"
                                  "These parameters will be loaded automatically on startup.")
            print(f"[HOMOGRAPHY] Saved default parameters to: {default_params_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save default parameters:\n{str(e)}")
            print(f"[HOMOGRAPHY] Error saving default parameters: {e}")
                
    def _load_parameters(self):
        """Load homography parameters from YAML file."""
        # Show load dialog - use configs directory as default
        homography_dir = Path(get_setting("homography.save_directory", "configs"))
        
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Homography Parameters",
            str(homography_dir) if homography_dir.exists() else "",
            "YAML files (*.yaml *.yml);;All files (*.*)"
        )
        
        if filename:
            try:
                # Load from YAML
                with open(filename, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                # Extract parameters
                if 'homography_parameters' in data:
                    loaded_params = data['homography_parameters']
                else:
                    # Try loading direct parameter format
                    loaded_params = data
                
                # Validate and update parameters
                for param_name in self.homography_params.keys():
                    if param_name in loaded_params:
                        value = float(loaded_params[param_name])
                        self.homography_params[param_name] = value
                        
                        # Update slider position with proper type conversion
                        if param_name in ['H00', 'H01', 'H10', 'H11']:
                            param_range = get_setting("homography.slider_range_main", [-50.0, 50.0])
                            if isinstance(param_range, list) and len(param_range) == 2:
                                param_range = [float(param_range[0]), float(param_range[1])]
                            else:
                                param_range = [-50.0, 50.0]
                        elif param_name in ['H20', 'H21']:
                            param_range = get_setting("homography.slider_range_perspective", [-0.2, 0.2])
                            if isinstance(param_range, list) and len(param_range) == 2:
                                param_range = [float(param_range[0]), float(param_range[1])]
                            else:
                                param_range = [-0.2, 0.2]
                        else:
                            param_range = [-10000.0, 10000.0]
                            
                        slider_val = int(((value - param_range[0]) / (param_range[1] - param_range[0])) * 1000)
                        slider_val = max(0, min(1000, slider_val))  # Clamp to valid range
                        self.param_sliders[param_name].setValue(slider_val)
                        
                        # Update label
                        self.param_labels[param_name].setText(f"{value:.6f}")
                
                # Update displays
                self._update_displays()
                
                QMessageBox.information(self, "Success", f"Parameters loaded from:\n{filename}")
                print(f"[HOMOGRAPHY] Loaded parameters from: {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load parameters:\n{str(e)}")
                print(f"[HOMOGRAPHY] Error loading parameters: {e}")
    
    def _load_default_parameters(self):
        """Load default homography parameters from config file on startup."""
        try:
            # Get the default parameters file path from config
            default_params_file = get_setting("homography.default_params_file", "configs/homography_params.yaml")
            
            if not Path(default_params_file).exists():
                print(f"[HOMOGRAPHY] Default parameters file not found: {default_params_file}")
                return
                
            print(f"[HOMOGRAPHY] Loading default parameters from: {default_params_file}")
            
            # Load from YAML
            with open(default_params_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Extract parameters
            if 'homography_parameters' in data:
                loaded_params = data['homography_parameters']
            else:
                # Try loading direct parameter format
                loaded_params = data
            
            # Validate and update parameters
            for param_name in self.homography_params.keys():
                if param_name in loaded_params:
                    value = float(loaded_params[param_name])
                    self.homography_params[param_name] = value
                    
                    # Update slider position with proper type conversion
                    if param_name in ['H00', 'H01', 'H10', 'H11']:
                        param_range = get_setting("homography.slider_range_main", [-50.0, 50.0])
                        if isinstance(param_range, list) and len(param_range) == 2:
                            param_range = [float(param_range[0]), float(param_range[1])]
                        else:
                            param_range = [-50.0, 50.0]
                    elif param_name in ['H20', 'H21']:
                        param_range = get_setting("homography.slider_range_perspective", [-0.2, 0.2])
                        if isinstance(param_range, list) and len(param_range) == 2:
                            param_range = [float(param_range[0]), float(param_range[1])]
                        else:
                            param_range = [-0.2, 0.2]
                    else:
                        param_range = [-10000.0, 10000.0]
                        
                    slider_val = int(((value - param_range[0]) / (param_range[1] - param_range[0])) * 1000)
                    slider_val = max(0, min(1000, slider_val))  # Clamp to valid range
                    self.param_sliders[param_name].setValue(slider_val)
                    
                    # Update label
                    self.param_labels[param_name].setText(f"{value:.6f}")
            
            # Update displays
            self._update_displays()
            
            print(f"[HOMOGRAPHY] Default parameters loaded successfully from: {default_params_file}")
            
        except Exception as e:
            print(f"[HOMOGRAPHY] Error loading default parameters: {e}")
            # Don't show error dialog on startup - just log it
    
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
            print("[HOMOGRAPHY] No current frame available for segmentation")
            return
            
        try:
            print("[HOMOGRAPHY] Running field segmentation on current frame")
            segmentation_start = time.time()
            self.current_segmentation_results = run_field_segmentation(self.current_frame)
            segmentation_duration = (time.time() - segmentation_start) * 1000
            self._add_runtime_measurement('Field Segmentation', segmentation_duration)
            
            if self.current_segmentation_results:
                print(f"[HOMOGRAPHY] Segmentation complete: {len(self.current_segmentation_results)} results ({segmentation_duration:.1f}ms)")
                # Debug: Check if results have masks
                for i, result in enumerate(self.current_segmentation_results):
                    if hasattr(result, 'masks') and result.masks is not None:
                        print(f"[HOMOGRAPHY] Result {i}: has masks with shape {result.masks.data.shape if hasattr(result.masks, 'data') else 'unknown'}")
                    else:
                        print(f"[HOMOGRAPHY] Result {i}: no masks found")
            else:
                print("[HOMOGRAPHY] No segmentation results returned")
        except Exception as e:
            print(f"[HOMOGRAPHY] Error running field segmentation: {e}")
            self.current_segmentation_results = None
            
    def _load_segmentation_models(self):
        """Load available field segmentation models using the same logic as main tab."""
        self.available_segmentation_models.clear()
        
        # Look for models in the models directory (same logic as main tab)
        models_path = Path(get_setting("models.base_path", DEFAULT_PATHS['MODELS']))
        
        if not models_path.exists():
            print(f"[HOMOGRAPHY] Models directory not found: {models_path}")
            return
        
        # Search for segmentation model files
        model_files = []
        for model_dir in models_path.rglob("*"):
            if model_dir.is_file() and model_dir.suffix == ".pt":
                # Skip last.pt files - we only want best.pt from finetuned models
                if model_dir.name == "last.pt":
                    continue
                    
                # Check if this is a segmentation model
                if "segmentation" in str(model_dir).lower():
                    model_files.append(str(model_dir))
        
        # Add pretrained segmentation models
        pretrained_path = models_path / "pretrained"
        if pretrained_path.exists():
            for model_file in pretrained_path.glob("*seg*.pt"):
                model_files.append(str(model_file))
        
        # Store the full paths
        self.available_segmentation_models = model_files
        
        # Update combo box
        if self.segmentation_model_combo is not None:
            self.segmentation_model_combo.clear()
            for model_path in self.available_segmentation_models:
                # Create display name from path
                if 'pretrained' in model_path:
                    display_name = f"Pretrained: {Path(model_path).stem}"
                else:
                    # Extract model name from path
                    path_parts = Path(model_path).parts
                    if 'segmentation' in path_parts:
                        seg_index = path_parts.index('segmentation')
                        if seg_index + 1 < len(path_parts):
                            display_name = path_parts[seg_index + 1]
                        else:
                            display_name = Path(model_path).parent.parent.name
                    else:
                        display_name = Path(model_path).parent.parent.name
                
                self.segmentation_model_combo.addItem(display_name, model_path)
                
        print(f"[HOMOGRAPHY] Loaded {len(self.available_segmentation_models)} segmentation models")
        
        # Auto-select the new default model if available
        if self.segmentation_model_combo is not None:
            default_model_path = "data/models/segmentation/20250826_1_segmentation_yolo11s-seg_field finder.v8i.yolov8/finetune_20250826_092226/weights/best.pt"
            for i in range(self.segmentation_model_combo.count()):
                item_path = self.segmentation_model_combo.itemData(i)
                if item_path and default_model_path in str(item_path):
                    self.segmentation_model_combo.setCurrentIndex(i)
                    print(f"[HOMOGRAPHY] Auto-selected default segmentation model: {self.segmentation_model_combo.itemText(i)}")
                    break
        
    def _on_segmentation_toggled(self, state: int):
        """Handle segmentation checkbox toggle."""
        self.show_segmentation = state == 2  # Qt.Checked = 2
        
        print(f"[HOMOGRAPHY] Segmentation toggle: state={state}, show_segmentation={self.show_segmentation}")
        
        if self.show_segmentation:
            print("[HOMOGRAPHY] Field segmentation enabled")
            self._run_segmentation_on_current_frame()
        else:
            print("[HOMOGRAPHY] Field segmentation disabled")
            self.current_segmentation_results = None
            
        self._update_displays()
    
    def _on_ransac_toggled(self, state: int):
        """Handle RANSAC line fitting checkbox toggle."""
        ransac_enabled = state == 2  # Qt.Checked = 2
        
        print(f"[HOMOGRAPHY] RANSAC toggle: state={state}, ransac_enabled={ransac_enabled}")
        
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
        
        # Update displays if segmentation is currently shown
        if self.show_segmentation and self.current_segmentation_results:
            self._update_displays()
        
    def _on_segmentation_model_changed(self, display_name: str):
        """Handle segmentation model selection change."""
        if self.segmentation_model_combo is None:
            return
            
        model_path = self.segmentation_model_combo.currentData()
        if model_path and os.path.exists(model_path):
            print(f"[HOMOGRAPHY] Changing segmentation model to: {model_path}")
            try:
                if set_field_model(model_path):
                    print(f"[HOMOGRAPHY] Successfully loaded segmentation model: {display_name}")
                    # Re-run segmentation with new model if currently enabled
                    if self.show_segmentation:
                        self._run_segmentation_on_current_frame()
                        self._update_displays()
                else:
                    print(f"[HOMOGRAPHY] Failed to load segmentation model: {model_path}")
            except Exception as e:
                print(f"[HOMOGRAPHY] Error loading segmentation model: {e}")
        else:
            print(f"[HOMOGRAPHY] Invalid model path: {model_path}")
    
    def _show_runtime_popup(self) -> None:
        """Show the runtime performance popup window."""
        if self.runtime_popup is None:
            # Create the popup window
            self.runtime_popup = QDialog(self)
            self.runtime_popup.setWindowTitle("Processing Runtime Performance")
            self.runtime_popup.setModal(False)  # Allow interaction with main window
            self.runtime_popup.resize(500, 350)
            
            # Set dark background
            self.runtime_popup.setStyleSheet("""
                QDialog {
                    background-color: #1a1a1a;
                    color: #ffffff;
                }
                QLabel {
                    color: #ffffff;
                    font-size: 12px;
                }
                QTableWidget {
                    background-color: #000000;
                    color: #ffffff;
                    gridline-color: #333333;
                    border: 1px solid #555555;
                    selection-background-color: #2c5aa0;
                }
                QTableWidget::item {
                    padding: 4px;
                    border-bottom: 1px solid #333333;
                }
                QTableWidget QHeaderView::section {
                    background-color: #2a2a2a;
                    color: #ffffff;
                    padding: 4px;
                    border: 1px solid #555555;
                    font-weight: bold;
                }
            """)
            
            # Create layout
            popup_layout = QVBoxLayout()
            
            # Add title
            title_label = QLabel("Processing Runtime Performance (ms)")
            title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
            popup_layout.addWidget(title_label)
            
            # Create table
            self.runtime_table = QTableWidget()
            self.runtime_table.setColumnCount(4)
            self.runtime_table.setHorizontalHeaderLabels(["Process", "Current", "Average", "Max"])
            
            # Set table properties
            self.runtime_table.horizontalHeader().setStretchLastSection(True)
            self.runtime_table.verticalHeader().setVisible(False)
            self.runtime_table.setAlternatingRowColors(True)
            
            # Initialize table with processing steps
            processes = ['Field Segmentation', 'Morphological Ops', 'RANSAC Fitting', 
                        'Line Classification', 'Homography Calc', 'Display Update']
            self.runtime_table.setRowCount(len(processes))
            
            for i, process in enumerate(processes):
                self.runtime_table.setItem(i, 0, QTableWidgetItem(process))
                self.runtime_table.setItem(i, 1, QTableWidgetItem("0.0"))
                self.runtime_table.setItem(i, 2, QTableWidgetItem("0.0"))
                self.runtime_table.setItem(i, 3, QTableWidgetItem("0.0"))
            
            # Set column widths
            header = self.runtime_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.Fixed)
            header.setSectionResizeMode(2, QHeaderView.Fixed)
            header.setSectionResizeMode(3, QHeaderView.Stretch)
            self.runtime_table.setColumnWidth(1, 80)
            self.runtime_table.setColumnWidth(2, 80)
            
            popup_layout.addWidget(self.runtime_table)
            
            # Add info label
            info_label = QLabel("Real-time performance monitoring. Window can be kept open while using the application.")
            info_label.setStyleSheet("font-size: 10px; color: #cccccc; margin: 5px;")
            popup_layout.addWidget(info_label)
            
            # Add close button
            close_button = QPushButton("Close")
            close_button.setStyleSheet("""
                QPushButton {
                    background-color: #555555;
                    color: white;
                    border: 1px solid #777777;
                    border-radius: 3px;
                    padding: 6px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #666666;
                }
                QPushButton:pressed {
                    background-color: #444444;
                }
            """)
            close_button.clicked.connect(self.runtime_popup.close)
            
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            button_layout.addWidget(close_button)
            popup_layout.addLayout(button_layout)
            
            self.runtime_popup.setLayout(popup_layout)
        
        # Show the popup
        self.runtime_popup.show()
        self.runtime_popup.raise_()
        self.runtime_popup.activateWindow()
    
    def _add_runtime_measurement(self, process_name: str, duration_ms: float) -> None:
        """Add a runtime measurement for a specific process.
        
        Args:
            process_name: Name of the process (e.g., 'Field Segmentation')
            duration_ms: Duration in milliseconds
        """
        if process_name in self.processing_times:
            # Add to history (keep last 10 measurements for rolling average)
            self.processing_times[process_name].append(duration_ms)
            if len(self.processing_times[process_name]) > 10:
                self.processing_times[process_name].pop(0)
            
            # Update table
            self._update_runtime_table()
    
    def _update_runtime_table(self) -> None:
        """Update the runtime performance table with current measurements."""
        if not self.runtime_table:
            return
        
        processes = ['Field Segmentation', 'Morphological Ops', 'RANSAC Fitting', 
                    'Line Classification', 'Homography Calc', 'Display Update']
        
        for i, process in enumerate(processes):
            if process in self.processing_times and self.processing_times[process]:
                times = self.processing_times[process]
                current = times[-1]  # Most recent measurement
                average = sum(times) / len(times)  # Rolling average
                maximum = max(times)  # Maximum in current history
                
                # Update table cells
                self.runtime_table.setItem(i, 1, QTableWidgetItem(f"{current:.1f}"))
                self.runtime_table.setItem(i, 2, QTableWidgetItem(f"{average:.1f}"))
                self.runtime_table.setItem(i, 3, QTableWidgetItem(f"{maximum:.1f}"))
            else:
                # No measurements yet
                self.runtime_table.setItem(i, 1, QTableWidgetItem("0.0"))
                self.runtime_table.setItem(i, 2, QTableWidgetItem("0.0"))
                self.runtime_table.setItem(i, 3, QTableWidgetItem("0.0"))
