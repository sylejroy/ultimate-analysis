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
        
        # Get mouse position relative to the widget
        mouse_pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        
        # Calculate zoom change
        zoom_in = event.angleDelta().y() > 0
        zoom_delta = 0.15 if zoom_in else -0.15
        old_zoom = self.zoom_factor
        new_zoom = max(0.1, min(10.0, old_zoom + zoom_delta))
        
        if new_zoom == old_zoom:
            return
        
        if scroll_area is None:
            # Fallback to center zoom if no scroll area found
            self.zoom_factor = new_zoom
            self._update_display()
            self.zoom_changed.emit(self.zoom_factor)
            return
        
        # Get current scroll positions
        h_scroll = scroll_area.horizontalScrollBar()
        v_scroll = scroll_area.verticalScrollBar()
        old_h_value = h_scroll.value()
        old_v_value = v_scroll.value()
        
        # Get current image display properties
        if not self.pixmap():
            return
            
        widget_size = self.size()
        old_pixmap_size = self.pixmap().size()
        
        # Calculate current image offset within widget
        old_x_offset = max(0, (widget_size.width() - old_pixmap_size.width()) // 2)
        old_y_offset = max(0, (widget_size.height() - old_pixmap_size.height()) // 2)
        
        # Calculate the point in the original image under the mouse cursor
        # Account for widget offset, scroll position, and current zoom
        original_image_x = (mouse_pos.x() - old_x_offset + old_h_value) / old_zoom
        original_image_y = (mouse_pos.y() - old_y_offset + old_v_value) / old_zoom
        
        # Update the zoom
        self.zoom_factor = new_zoom
        self._update_display()
        
        # Calculate new image properties after zoom
        new_pixmap_size = self.pixmap().size()
        new_x_offset = max(0, (widget_size.width() - new_pixmap_size.width()) // 2)
        new_y_offset = max(0, (widget_size.height() - new_pixmap_size.height()) // 2)
        
        # Calculate where that same point should be positioned to stay under the mouse
        # We want: mouse_pos = (original_image_point * new_zoom + new_offset - new_scroll)
        # So: new_scroll = original_image_point * new_zoom + new_offset - mouse_pos
        target_x_in_widget = original_image_x * new_zoom + new_x_offset
        target_y_in_widget = original_image_y * new_zoom + new_y_offset
        
        new_h_value = target_x_in_widget - mouse_pos.x()
        new_v_value = target_y_in_widget - mouse_pos.y()
        
        # Clamp scroll values to valid ranges
        new_h_value = max(0, min(h_scroll.maximum(), int(new_h_value)))
        new_v_value = max(0, min(v_scroll.maximum(), int(new_v_value)))
        
        # Apply the new scroll positions
        h_scroll.setValue(new_h_value)
        v_scroll.setValue(new_v_value)
        
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
        self.original_display: Optional[ZoomableImageLabel] = None
        self.warped_display: Optional[ZoomableImageLabel] = None
        self.param_sliders: Dict[str, QSlider] = {}
        self.param_labels: Dict[str, QLabel] = {}
        
        # Zoom functionality
        self.original_scroll_area: Optional[QScrollArea] = None
        self.warped_scroll_area: Optional[QScrollArea] = None
        
        # Initialize UI
        self._init_ui()
        self._load_videos()
        
        # Load default homography parameters from config
        self._load_default_parameters()
        
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
        
        # Original frame display with scroll area
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
                    'frame_index': self.frame_slider.value() if self.frame_slider else 0,
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
