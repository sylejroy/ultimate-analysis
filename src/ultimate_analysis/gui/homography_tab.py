"""Homography estimation tab for Ultimate Analysis GUI.

This module provides an interactive interface for adjusting homography parameters
with real-time perspective transformation visualization and YAML save/load functionality.
"""

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
    QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QWheelEvent, QPainter, QPen, QColor, QMouseEvent

from .video_player import VideoPlayer
from ..config.settings import get_setting
from ..constants import DEFAULT_PATHS, SUPPORTED_VIDEO_EXTENSIONS


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
        """Handle mouse wheel for zooming to mouse position."""
        if self.original_pixmap is None:
            return
            
        # Get the scroll area parent to adjust scrollbars
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
        
        # Get mouse position relative to the image label
        mouse_pos = event.position().toPoint()
        
        # Get current scroll positions
        h_scroll = scroll_area.horizontalScrollBar()
        v_scroll = scroll_area.verticalScrollBar()
        old_h_value = h_scroll.value()
        old_v_value = v_scroll.value()
        
        # Calculate mouse position relative to the image content
        old_zoom = self.zoom_factor
        zoom_in = event.angleDelta().y() > 0
        zoom_delta = 0.15 if zoom_in else -0.15
        new_zoom = max(0.1, min(10.0, old_zoom + zoom_delta))
        
        if new_zoom == old_zoom:
            return
            
        # Calculate the point in the image that should remain under the mouse
        image_point_x = (mouse_pos.x() + old_h_value) / old_zoom
        image_point_y = (mouse_pos.y() + old_v_value) / old_zoom
        
        # Update zoom
        self.zoom_factor = new_zoom
        self._update_display()
        
        # Calculate new scroll positions to keep the same image point under the mouse
        new_h_value = image_point_x * new_zoom - mouse_pos.x()
        new_v_value = image_point_y * new_zoom - mouse_pos.y()
        
        # Apply the new scroll positions
        h_scroll.setValue(int(new_h_value))
        v_scroll.setValue(int(new_v_value))
        
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


class InteractiveImageLabel(ZoomableImageLabel):
    """Interactive image label that supports line drawing for homography calculation."""
    
    lines_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing_mode = False
        self.current_line_points: List[QPoint] = []
        self.lines: List[List[QPoint]] = []  # List of lines, each line is a list of 2 points
        self.temp_point: Optional[QPoint] = None
        
        # Line types: 'parallel1', 'parallel2', 'perpendicular'
        self.line_types: List[str] = []
        self.current_line_type = 'parallel1'
        
    def start_drawing_mode(self, line_type: str):
        """Start drawing mode for a specific line type."""
        self.drawing_mode = True
        self.current_line_type = line_type
        self.current_line_points = []
        self.setCursor(Qt.CrossCursor)
        
    def stop_drawing_mode(self):
        """Stop drawing mode."""
        self.drawing_mode = False
        self.current_line_points = []
        self.temp_point = None
        self.setCursor(Qt.ArrowCursor)
        self.update()
        
    def clear_lines(self):
        """Clear all drawn lines."""
        self.lines = []
        self.line_types = []
        self.update()
        self.lines_changed.emit()
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for line drawing."""
        if self.drawing_mode and event.button() == Qt.LeftButton:
            if self.original_pixmap is None:
                return
                
            # Convert click position to image coordinates
            image_pos = self._widget_to_image_coords(event.pos())
            if image_pos is not None:
                self.current_line_points.append(image_pos)
                
                # Complete line when we have 2 points
                if len(self.current_line_points) == 2:
                    self.lines.append(self.current_line_points.copy())
                    self.line_types.append(self.current_line_type)
                    self.current_line_points = []
                    self.temp_point = None
                    self.drawing_mode = False
                    self.setCursor(Qt.ArrowCursor)
                    self.lines_changed.emit()
                    
                self.update()
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for temporary line drawing."""
        if self.drawing_mode and len(self.current_line_points) == 1:
            image_pos = self._widget_to_image_coords(event.pos())
            if image_pos is not None:
                self.temp_point = image_pos
                self.update()
        else:
            super().mouseMoveEvent(event)
            
    def _widget_to_image_coords(self, widget_pos: QPoint) -> Optional[QPoint]:
        """Convert widget coordinates to image coordinates."""
        if self.original_pixmap is None:
            return None
            
        # Get the displayed pixmap size and position
        displayed_pixmap = self.pixmap()
        if displayed_pixmap is None:
            return None
            
        # Calculate the offset of the displayed image within the widget
        widget_size = self.size()
        pixmap_size = displayed_pixmap.size()
        
        x_offset = (widget_size.width() - pixmap_size.width()) // 2
        y_offset = (widget_size.height() - pixmap_size.height()) // 2
        
        # Convert to image coordinates
        image_x = (widget_pos.x() - x_offset) / self.zoom_factor
        image_y = (widget_pos.y() - y_offset) / self.zoom_factor
        
        # Check if click is within original image bounds
        original_size = self.original_pixmap.size()
        if 0 <= image_x < original_size.width() and 0 <= image_y < original_size.height():
            return QPoint(int(image_x), int(image_y))
        
        return None
        
    def paintEvent(self, event):
        """Paint the image and overlay lines."""
        super().paintEvent(event)
        
        if self.original_pixmap is None:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get display parameters
        displayed_pixmap = self.pixmap()
        if displayed_pixmap is None:
            return
            
        widget_size = self.size()
        pixmap_size = displayed_pixmap.size()
        x_offset = (widget_size.width() - pixmap_size.width()) // 2
        y_offset = (widget_size.height() - pixmap_size.height()) // 2
        
        # Draw completed lines
        for i, line_points in enumerate(self.lines):
            line_type = self.line_types[i] if i < len(self.line_types) else 'parallel1'
            
            # Set color based on line type
            if line_type == 'parallel1':
                color = QColor(255, 0, 0)  # Red
            elif line_type == 'parallel2':
                color = QColor(0, 255, 0)  # Green
            else:  # perpendicular
                color = QColor(0, 0, 255)  # Blue
                
            pen = QPen(color, 3)
            painter.setPen(pen)
            
            if len(line_points) >= 2:
                start = line_points[0]
                end = line_points[1]
                
                # Convert to widget coordinates
                start_widget = QPoint(
                    int(start.x() * self.zoom_factor + x_offset),
                    int(start.y() * self.zoom_factor + y_offset)
                )
                end_widget = QPoint(
                    int(end.x() * self.zoom_factor + x_offset),
                    int(end.y() * self.zoom_factor + y_offset)
                )
                
                painter.drawLine(start_widget, end_widget)
        
        # Draw current line being drawn
        if self.drawing_mode and len(self.current_line_points) == 1 and self.temp_point is not None:
            # Set color for current line type
            if self.current_line_type == 'parallel1':
                color = QColor(255, 100, 100)  # Light red
            elif self.current_line_type == 'parallel2':
                color = QColor(100, 255, 100)  # Light green
            else:  # perpendicular
                color = QColor(100, 100, 255)  # Light blue
                
            pen = QPen(color, 2, Qt.DashLine)
            painter.setPen(pen)
            
            start = self.current_line_points[0]
            end = self.temp_point
            
            start_widget = QPoint(
                int(start.x() * self.zoom_factor + x_offset),
                int(start.y() * self.zoom_factor + y_offset)
            )
            end_widget = QPoint(
                int(end.x() * self.zoom_factor + x_offset),
                int(end.y() * self.zoom_factor + y_offset)
            )
            
            painter.drawLine(start_widget, end_widget)


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
        self.frame_slider: Optional[QSlider] = None
        self.frame_label: Optional[QLabel] = None
        self.original_display: Optional[InteractiveImageLabel] = None
        self.warped_display: Optional[ZoomableImageLabel] = None
        self.param_sliders: Dict[str, QSlider] = {}
        self.param_labels: Dict[str, QLabel] = {}
        
        # Line drawing components
        self.draw_parallel1_btn: Optional[QPushButton] = None
        self.draw_parallel2_btn: Optional[QPushButton] = None
        self.draw_perpendicular_btn: Optional[QPushButton] = None
        self.clear_lines_btn: Optional[QPushButton] = None
        self.calculate_homography_btn: Optional[QPushButton] = None
        
        # Zoom functionality
        self.original_scroll_area: Optional[QScrollArea] = None
        self.warped_scroll_area: Optional[QScrollArea] = None
        
        # Initialize UI
        self._init_ui()
        self._load_videos()
        
    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout()
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Video list and parameter controls
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Side-by-side video displays
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (25% left, 75% right for larger image display)
        splitter.setSizes([300, 1200])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
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
        refresh_button.clicked.connect(self._load_videos)
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
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        video_layout.addWidget(self.frame_slider)
        
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
        
        params_layout.addLayout(button_layout)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
        
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with vertically stacked zoomable displays."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Display header
        header = QLabel("Homography Transformation Comparison (Mouse wheel to zoom)")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(header)
        
        # Line drawing controls
        line_controls = self._create_line_drawing_controls()
        layout.addWidget(line_controls)
        
        # Original frame display with scroll area
        original_group = QGroupBox("Original Frame (Click to draw lines)")
        original_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        original_layout = QVBoxLayout()
        
        self.original_scroll_area = QScrollArea()
        self.original_scroll_area.setWidgetResizable(True)
        self.original_scroll_area.setAlignment(Qt.AlignCenter)
        self.original_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.original_display = InteractiveImageLabel()
        self.original_display.setText("No video selected")
        self.original_display.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #1a1a1a;
                color: #999;
                font-size: 12px;
            }
        """)
        self.original_display.lines_changed.connect(self._on_lines_changed)
        
        self.original_scroll_area.setWidget(self.original_display)
        original_layout.addWidget(self.original_scroll_area)
        original_group.setLayout(original_layout)
        layout.addWidget(original_group)
        
        # Warped frame display with scroll area
        warped_group = QGroupBox("Warped Frame (with buffer)")
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
        layout.addWidget(warped_group)
        
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
        
        panel.setLayout(layout)
        return panel
        
    def _create_parameter_controls(self, layout: QVBoxLayout):
        """Create slider controls for homography parameters."""
        # Get slider ranges from configuration with proper type conversion
        h_range_main = get_setting("homography.slider_range_main", [-5.0, 5.0])
        h_range_persp = get_setting("homography.slider_range_perspective", [-0.01, 0.01])
        
        # Ensure ranges are numeric (convert from strings if needed)
        if isinstance(h_range_main, list) and len(h_range_main) == 2:
            h_range_main = [float(h_range_main[0]), float(h_range_main[1])]
        else:
            h_range_main = [-5.0, 5.0]  # Fallback
            
        if isinstance(h_range_persp, list) and len(h_range_persp) == 2:
            h_range_persp = [float(h_range_persp[0]), float(h_range_persp[1])]
        else:
            h_range_persp = [-0.01, 0.01]  # Fallback
        
        # Parameters with their ranges and default values
        param_config = [
            ('H00', h_range_main, 1.0, "Scale X"),
            ('H01', h_range_main, 0.0, "Skew X"),
            ('H02', [-500.0, 500.0], 0.0, "Translate X"),
            ('H10', h_range_main, 0.0, "Skew Y"), 
            ('H11', h_range_main, 1.0, "Scale Y"),
            ('H12', [-500.0, 500.0], 0.0, "Translate Y"),
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
        
    def _create_line_drawing_controls(self) -> QWidget:
        """Create controls for line drawing functionality."""
        group = QGroupBox("Line Drawing for Homography Calculation")
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "Draw lines to calculate homography:\n"
            "1. Draw first parallel line (red)\n"
            "2. Draw second parallel line (green)\n"
            "3. Draw perpendicular line (blue)\n"
            "4. Calculate homography from lines"
        )
        instructions.setStyleSheet("font-size: 10px; color: #ccc; margin: 5px;")
        layout.addWidget(instructions)
        
        # Line drawing buttons
        button_layout = QHBoxLayout()
        
        self.draw_parallel1_btn = QPushButton("Draw Parallel 1")
        self.draw_parallel1_btn.setStyleSheet("background-color: #8B0000; color: white;")
        self.draw_parallel1_btn.clicked.connect(lambda: self._start_line_drawing('parallel1'))
        button_layout.addWidget(self.draw_parallel1_btn)
        
        self.draw_parallel2_btn = QPushButton("Draw Parallel 2")
        self.draw_parallel2_btn.setStyleSheet("background-color: #006400; color: white;")
        self.draw_parallel2_btn.clicked.connect(lambda: self._start_line_drawing('parallel2'))
        button_layout.addWidget(self.draw_parallel2_btn)
        
        self.draw_perpendicular_btn = QPushButton("Draw Perpendicular")
        self.draw_perpendicular_btn.setStyleSheet("background-color: #00008B; color: white;")
        self.draw_perpendicular_btn.clicked.connect(lambda: self._start_line_drawing('perpendicular'))
        button_layout.addWidget(self.draw_perpendicular_btn)
        
        layout.addLayout(button_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.clear_lines_btn = QPushButton("Clear Lines")
        self.clear_lines_btn.clicked.connect(self._clear_lines)
        action_layout.addWidget(self.clear_lines_btn)
        
        self.calculate_homography_btn = QPushButton("Calculate Homography")
        self.calculate_homography_btn.setStyleSheet("background-color: #FF8C00; color: white; font-weight: bold;")
        self.calculate_homography_btn.clicked.connect(self._calculate_homography_from_lines)
        self.calculate_homography_btn.setEnabled(False)
        action_layout.addWidget(self.calculate_homography_btn)
        
        layout.addLayout(action_layout)
        
        group.setLayout(layout)
        return group
        
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
            self.frame_slider.setMaximum(max(1, total_frames - 1))
            self.frame_slider.setValue(0)
            self.frame_label.setText(f"0 / {total_frames}")
            
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
            
            # Get and display current frame
            frame = self.video_player.get_current_frame()
            if frame is not None:
                self.current_frame = frame.copy()
                self._update_displays()
                
    def _on_parameter_changed(self, param_name: str, slider_value: int):
        """Handle homography parameter change from slider."""
        # Get parameter range with proper type conversion
        if param_name in ['H00', 'H01', 'H10', 'H11']:
            param_range = get_setting("homography.slider_range_main", [-5.0, 5.0])
            if isinstance(param_range, list) and len(param_range) == 2:
                param_range = [float(param_range[0]), float(param_range[1])]
            else:
                param_range = [-5.0, 5.0]
        elif param_name in ['H20', 'H21']:
            param_range = get_setting("homography.slider_range_perspective", [-0.01, 0.01])
            if isinstance(param_range, list) and len(param_range) == 2:
                param_range = [float(param_range[0]), float(param_range[1])]
            else:
                param_range = [-0.01, 0.01]
        else:  # Translation parameters
            param_range = [-500.0, 500.0]
            
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
            
        # Display original frame
        self._display_frame(self.current_frame, self.original_display)
        
        # Create and display warped frame
        warped_frame = self._apply_homography(self.current_frame)
        self._display_frame(warped_frame, self.warped_display)
        
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
            Warped frame using original frame dimensions
        """
        # Construct homography matrix
        H = np.array([
            [self.homography_params['H00'], self.homography_params['H01'], self.homography_params['H02']],
            [self.homography_params['H10'], self.homography_params['H11'], self.homography_params['H12']],
            [self.homography_params['H20'], self.homography_params['H21'], 1.0]
        ], dtype=np.float32)
        
        # Get original dimensions
        original_height, original_width = frame.shape[:2]
        
        # Apply perspective transform using original frame size
        warped = cv2.warpPerspective(frame, H, (original_width, original_height))
        
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
                param_range = get_setting("homography.slider_range_main", [-5.0, 5.0])
                if isinstance(param_range, list) and len(param_range) == 2:
                    param_range = [float(param_range[0]), float(param_range[1])]
                else:
                    param_range = [-5.0, 5.0]
            elif param_name in ['H20', 'H21']:
                param_range = get_setting("homography.slider_range_perspective", [-0.01, 0.01])
                if isinstance(param_range, list) and len(param_range) == 2:
                    param_range = [float(param_range[0]), float(param_range[1])]
                else:
                    param_range = [-0.01, 0.01]
            else:
                param_range = [-500.0, 500.0]
                
            slider_val = int(((value - param_range[0]) / (param_range[1] - param_range[0])) * 1000)
            self.param_sliders[param_name].setValue(slider_val)
            
            # Update label
            self.param_labels[param_name].setText(f"{value:.6f}")
        
        # Update displays
        self._update_displays()
        
        print("[HOMOGRAPHY] Reset to identity matrix")
        
    def _save_parameters(self):
        """Save current homography parameters to YAML file."""
        # Create homography directory if it doesn't exist
        homography_dir = Path(get_setting("homography.save_directory", "data/homography_params"))
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
                        'frame_index': self.frame_slider.value(),
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
                
    def _load_parameters(self):
        """Load homography parameters from YAML file."""
        # Show load dialog
        homography_dir = Path(get_setting("homography.save_directory", "data/homography_params"))
        
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
                            param_range = get_setting("homography.slider_range_main", [-5.0, 5.0])
                            if isinstance(param_range, list) and len(param_range) == 2:
                                param_range = [float(param_range[0]), float(param_range[1])]
                            else:
                                param_range = [-5.0, 5.0]
                        elif param_name in ['H20', 'H21']:
                            param_range = get_setting("homography.slider_range_perspective", [-0.01, 0.01])
                            if isinstance(param_range, list) and len(param_range) == 2:
                                param_range = [float(param_range[0]), float(param_range[1])]
                            else:
                                param_range = [-0.01, 0.01]
                        else:
                            param_range = [-500.0, 500.0]
                            
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
    
    def _start_line_drawing(self, line_type: str):
        """Start drawing mode for a specific line type."""
        if self.original_display:
            self.original_display.start_drawing_mode(line_type)
            
    def _clear_lines(self):
        """Clear all drawn lines."""
        if self.original_display:
            self.original_display.clear_lines()
            
    def _on_lines_changed(self):
        """Handle when lines are changed."""
        if self.original_display:
            lines = self.original_display.lines
            line_types = self.original_display.line_types
            
            # Check if we have the required lines for homography calculation
            has_parallel1 = 'parallel1' in line_types
            has_parallel2 = 'parallel2' in line_types
            has_perpendicular = 'perpendicular' in line_types
            
            can_calculate = has_parallel1 and has_parallel2 and has_perpendicular
            self.calculate_homography_btn.setEnabled(can_calculate)
            
    def _calculate_homography_from_lines(self):
        """Calculate homography from drawn lines using perspective correction."""
        if not self.original_display or not self.current_frame is not None:
            QMessageBox.warning(self, "Warning", "No image loaded or lines drawn.")
            return
            
        lines = self.original_display.lines
        line_types = self.original_display.line_types
        
        # Find lines by type
        parallel1_line = None
        parallel2_line = None
        perpendicular_line = None
        
        for i, line_type in enumerate(line_types):
            if line_type == 'parallel1' and i < len(lines):
                parallel1_line = lines[i]
            elif line_type == 'parallel2' and i < len(lines):
                parallel2_line = lines[i]
            elif line_type == 'perpendicular' and i < len(lines):
                perpendicular_line = lines[i]
                
        if not all([parallel1_line, parallel2_line, perpendicular_line]):
            QMessageBox.warning(self, "Warning", "Please draw all three lines (two parallel and one perpendicular).")
            return
            
        try:
            # Convert QPoint to numpy arrays
            p1_start = np.array([parallel1_line[0].x(), parallel1_line[0].y()])
            p1_end = np.array([parallel1_line[1].x(), parallel1_line[1].y()])
            p2_start = np.array([parallel2_line[0].x(), parallel2_line[0].y()])
            p2_end = np.array([parallel2_line[1].x(), parallel2_line[1].y()])
            perp_start = np.array([perpendicular_line[0].x(), perpendicular_line[0].y()])
            perp_end = np.array([perpendicular_line[1].x(), perpendicular_line[1].y()])
            
            # Calculate homography for perspective correction
            # This creates a transformation that makes the parallel lines truly parallel
            # and the perpendicular line truly perpendicular
            
            # Handle partially visible lines with robust geometric approach
            # Method 1: Try intersection-based approach first
            intersection1 = self._line_intersection(p1_start, p1_end, perp_start, perp_end)
            intersection2 = self._line_intersection(p2_start, p2_end, perp_start, perp_end)
            
            # Check if we have good intersections within reasonable bounds
            img_height, img_width = self.current_frame.shape[:2]
            valid_intersections = True
            
            if intersection1 is not None:
                if not (0 <= intersection1[0] <= img_width and 0 <= intersection1[1] <= img_height):
                    # Intersection is outside image bounds - lines may be too short
                    valid_intersections = False
            else:
                valid_intersections = False
                
            if intersection2 is not None:
                if not (0 <= intersection2[0] <= img_width and 0 <= intersection2[1] <= img_height):
                    valid_intersections = False
            else:
                valid_intersections = False
            
            # Method 2: Robust approach for partially visible lines
            if not valid_intersections:
                QMessageBox.information(self, "Info", 
                    "Lines don't intersect within image bounds. Using robust approach for partial lines...")
                
                # Extend lines to find better intersection points
                p1_extended = self._extend_line_to_image_bounds(p1_start, p1_end, img_width, img_height)
                p2_extended = self._extend_line_to_image_bounds(p2_start, p2_end, img_width, img_height)
                perp_extended = self._extend_line_to_image_bounds(perp_start, perp_end, img_width, img_height)
                
                # Try intersections with extended lines
                intersection1 = self._line_intersection(p1_extended[0], p1_extended[1], perp_extended[0], perp_extended[1])
                intersection2 = self._line_intersection(p2_extended[0], p2_extended[1], perp_extended[0], perp_extended[1])
                
                if intersection1 is None or intersection2 is None:
                    # Method 3: Use vanishing point approach for severely partial lines
                    src_points, dst_points = self._calculate_homography_vanishing_point_method(
                        p1_start, p1_end, p2_start, p2_end, perp_start, perp_end, img_width, img_height)
                    
                    if src_points is None:
                        QMessageBox.warning(self, "Warning", 
                            "Cannot calculate homography from these line segments. "
                            "Try drawing longer lines or lines that intersect within the image.")
                        return
                else:
                    # Use extended intersections
                    perp_vector = perp_extended[1] - perp_extended[0]
                    perp_length = np.linalg.norm(perp_vector)
                    
                    if perp_length > 0:
                        perp_unit = perp_vector / perp_length
                        # Create quadrilateral from intersections and perpendicular direction
                        field_width = perp_length * 0.7  # Use portion of perpendicular line
                        
                        src_points = np.float32([
                            intersection1,
                            intersection1 + perp_unit * field_width,
                            intersection2,
                            intersection2 + perp_unit * field_width
                        ])
                        
                        # Create destination rectangle
                        spacing = np.linalg.norm(intersection2 - intersection1)
                        center_x, center_y = img_width/2, img_height/2
                        scale = min(img_width * 0.7 / field_width, img_height * 0.7 / spacing)
                        
                        rect_width = field_width * scale
                        rect_height = spacing * scale
                        
                        dst_points = np.float32([
                            [center_x - rect_width/2, center_y - rect_height/2],
                            [center_x + rect_width/2, center_y - rect_height/2],
                            [center_x - rect_width/2, center_y + rect_height/2],
                            [center_x + rect_width/2, center_y + rect_height/2]
                        ])
                    else:
                        QMessageBox.warning(self, "Warning", "Invalid line geometry detected.")
                        return
            else:
                # Method 1: Standard approach with good intersections
                # Project perpendicular line start/end onto parallel lines
                proj1_start = self._project_point_to_line(perp_start, p1_start, p1_end)
                proj1_end = self._project_point_to_line(perp_end, p1_start, p1_end)
                proj2_start = self._project_point_to_line(perp_start, p2_start, p2_end)
                proj2_end = self._project_point_to_line(perp_end, p2_start, p2_end)
                
                # Use the closest projections to form a quadrilateral
                if proj1_start is not None and proj2_start is not None and proj1_end is not None and proj2_end is not None:
                    # Source points: the quadrilateral corners in the distorted image
                    src_points = np.float32([
                        proj1_start,  # Top-left
                        proj1_end,    # Top-right  
                        proj2_start,  # Bottom-left
                        proj2_end     # Bottom-right
                    ])
                    
                    # Calculate dimensions for the destination rectangle
                    perp_length = np.linalg.norm(perp_end - perp_start)
                    dist1 = np.linalg.norm(proj1_start - proj2_start)
                    dist2 = np.linalg.norm(proj1_end - proj2_end)
                    parallel_spacing = (dist1 + dist2) / 2
                    
                    # Create destination rectangle (corrected perspective)
                    center_x = img_width / 2
                    center_y = img_height / 2
                    
                    # Scale to fit nicely in the image
                    scale_factor = min(img_width * 0.8 / perp_length, img_height * 0.8 / parallel_spacing)
                    rect_width = perp_length * scale_factor
                    rect_height = parallel_spacing * scale_factor
                    
                    dst_points = np.float32([
                        [center_x - rect_width/2, center_y - rect_height/2],  # Top-left
                        [center_x + rect_width/2, center_y - rect_height/2],  # Top-right
                        [center_x - rect_width/2, center_y + rect_height/2],  # Bottom-left
                        [center_x + rect_width/2, center_y + rect_height/2]   # Bottom-right
                    ])
                else:
                    QMessageBox.warning(self, "Warning", "Could not project points to create quadrilateral.")
                    return
            
            # Calculate perspective transform
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Update homography parameters
            self.homography_params['H00'] = float(matrix[0, 0])
            self.homography_params['H01'] = float(matrix[0, 1])
            self.homography_params['H02'] = float(matrix[0, 2])
            self.homography_params['H10'] = float(matrix[1, 0])
            self.homography_params['H11'] = float(matrix[1, 1])
            self.homography_params['H12'] = float(matrix[1, 2])
            self.homography_params['H20'] = float(matrix[2, 0])
            self.homography_params['H21'] = float(matrix[2, 1])
            
            # Update all sliders with the new values
            self._update_sliders_from_params()
            
            # Update the display
            self._update_displays()
            
            QMessageBox.information(self, "Success", 
                                  "Homography calculated successfully from drawn lines!\n"
                                  "The transformation corrects perspective to make lines parallel.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate homography:\n{str(e)}")
            print(f"[HOMOGRAPHY] Error calculating homography from lines: {e}")
    
    def _line_intersection(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
        """Calculate intersection point of two lines defined by points (p1,p2) and (p3,p4)."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # Lines are parallel
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        
        return np.array([intersection_x, intersection_y])
    
    def _project_point_to_line(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> Optional[np.ndarray]:
        """Project a point onto a line defined by two points."""
        # Vector from line start to line end
        line_vec = line_end - line_start
        line_length_sq = np.dot(line_vec, line_vec)
        
        if line_length_sq < 1e-10:  # Line is too short
            return line_start
        
        # Vector from line start to point
        point_vec = point - line_start
        
        # Project point onto line
        t = np.dot(point_vec, line_vec) / line_length_sq
        projection = line_start + t * line_vec
        
        return projection
    
    def _extend_line_to_image_bounds(self, start: np.ndarray, end: np.ndarray, img_width: int, img_height: int) -> tuple:
        """Extend a line segment to intersect with image boundaries."""
        # Calculate line direction
        direction = end - start
        if np.linalg.norm(direction) < 1e-10:
            return start, end
        
        # Parametric line: point = start + t * direction
        # Find intersections with image boundaries
        intersections = []
        
        # Left boundary (x = 0)
        if abs(direction[0]) > 1e-10:
            t = -start[0] / direction[0]
            y = start[1] + t * direction[1]
            if 0 <= y <= img_height:
                intersections.append((0, y))
        
        # Right boundary (x = img_width)
        if abs(direction[0]) > 1e-10:
            t = (img_width - start[0]) / direction[0]
            y = start[1] + t * direction[1]
            if 0 <= y <= img_height:
                intersections.append((img_width, y))
        
        # Top boundary (y = 0)
        if abs(direction[1]) > 1e-10:
            t = -start[1] / direction[1]
            x = start[0] + t * direction[0]
            if 0 <= x <= img_width:
                intersections.append((x, 0))
        
        # Bottom boundary (y = img_height)
        if abs(direction[1]) > 1e-10:
            t = (img_height - start[1]) / direction[1]
            x = start[0] + t * direction[0]
            if 0 <= x <= img_width:
                intersections.append((x, img_height))
        
        # Remove duplicates and sort by distance from start
        unique_intersections = []
        for pt in intersections:
            is_duplicate = False
            for existing_pt in unique_intersections:
                if abs(pt[0] - existing_pt[0]) < 1 and abs(pt[1] - existing_pt[1]) < 1:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(pt)
        
        if len(unique_intersections) >= 2:
            # Return the two points that span the longest distance
            intersections_array = np.array(unique_intersections)
            max_dist = 0
            best_pair = (start, end)
            
            for i in range(len(intersections_array)):
                for j in range(i + 1, len(intersections_array)):
                    dist = np.linalg.norm(intersections_array[i] - intersections_array[j])
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (intersections_array[i], intersections_array[j])
            
            return np.array(best_pair[0]), np.array(best_pair[1])
        else:
            # Return original points if we can't extend properly
            return start, end
    
    def _calculate_homography_vanishing_point_method(self, p1_start, p1_end, p2_start, p2_end, 
                                                   perp_start, perp_end, img_width, img_height):
        """Calculate homography using vanishing point method for severely partial lines."""
        # Find vanishing point of parallel lines
        vp = self._find_vanishing_point(p1_start, p1_end, p2_start, p2_end)
        
        if vp is None:
            return None, None
        
        # Use the vanishing point to create a more robust homography
        # This is a simplified approach - in practice, you'd want more sophisticated methods
        
        # Create source points using available line segments and vanishing point constraints
        # Use the midpoints of the lines and project towards vanishing point
        p1_mid = (p1_start + p1_end) / 2
        p2_mid = (p2_start + p2_end) / 2
        perp_mid = (perp_start + perp_end) / 2
        
        # Calculate perpendicular directions
        p1_dir = p1_end - p1_start
        p1_perp = np.array([-p1_dir[1], p1_dir[0]])  # Perpendicular to p1
        p1_perp = p1_perp / np.linalg.norm(p1_perp) if np.linalg.norm(p1_perp) > 0 else np.array([0, 1])
        
        # Create a rectangular grid based on available information
        field_width = np.linalg.norm(perp_end - perp_start)
        field_height = np.linalg.norm(p2_mid - p1_mid)
        
        # Source quadrilateral
        src_points = np.float32([
            p1_mid - p1_perp * field_width * 0.3,
            p1_mid + p1_perp * field_width * 0.3,
            p2_mid - p1_perp * field_width * 0.3,
            p2_mid + p1_perp * field_width * 0.3
        ])
        
        # Destination rectangle
        center_x, center_y = img_width / 2, img_height / 2
        scale = min(img_width * 0.6 / field_width, img_height * 0.6 / field_height)
        rect_width = field_width * scale
        rect_height = field_height * scale
        
        dst_points = np.float32([
            [center_x - rect_width/2, center_y - rect_height/2],
            [center_x + rect_width/2, center_y - rect_height/2],
            [center_x - rect_width/2, center_y + rect_height/2],
            [center_x + rect_width/2, center_y + rect_height/2]
        ])
        
        return src_points, dst_points
    
    def _find_vanishing_point(self, p1_start, p1_end, p2_start, p2_end):
        """Find the vanishing point of two parallel lines."""
        # Extend both lines and find their intersection
        line1_extended = self._extend_line_to_image_bounds(p1_start, p1_end, 10000, 10000)
        line2_extended = self._extend_line_to_image_bounds(p2_start, p2_end, 10000, 10000)
        
        return self._line_intersection(line1_extended[0], line1_extended[1], 
                                     line2_extended[0], line2_extended[1])
    
    def _update_sliders_from_params(self):
        """Update all sliders to match current homography parameters."""
        for param_name, value in self.homography_params.items():
            if param_name in self.param_sliders:
                slider = self.param_sliders[param_name]
                
                # Determine the range for this parameter
                if param_name in ['H00', 'H01', 'H10', 'H11']:
                    param_range = get_setting("homography.slider_range_main", [-5.0, 5.0])
                elif param_name in ['H02', 'H12']:
                    param_range = get_setting("homography.slider_range_translation", [-1000.0, 1000.0])
                else:  # H20, H21
                    param_range = get_setting("homography.slider_range_perspective", [-0.01, 0.01])
                
                # Convert value to slider position
                slider_min = param_range[0]
                slider_max = param_range[1]
                slider_value = int(((value - slider_min) / (slider_max - slider_min)) * 1000)
                slider_value = max(0, min(1000, slider_value))
                
                # Update slider and label
                slider.setValue(slider_value)
                if param_name in self.param_labels:
                    self.param_labels[param_name].setText(f"{value:.6f}")
