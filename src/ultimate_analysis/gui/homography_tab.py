"""Homography estimation tab for Ultimate Analysis GUI.

This module provides an interactive interface for adjusting homography parameters
with real-time perspective transformation visualization and YAML save/load functionality.
"""

import os
import random
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, 
    QPushButton, QSlider, QListWidgetItem, QGroupBox,
    QFormLayout, QSplitter, QFileDialog, QMessageBox,
    QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from .video_player import VideoPlayer
from ..config.settings import get_setting
from ..constants import DEFAULT_PATHS, SUPPORTED_VIDEO_EXTENSIONS


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
        self.original_display: Optional[QLabel] = None
        self.warped_display: Optional[QLabel] = None
        self.param_sliders: Dict[str, QSlider] = {}
        self.param_labels: Dict[str, QLabel] = {}
        
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
        
        # Set splitter proportions (30% left, 70% right)
        splitter.setSizes([400, 1200])
        
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
        """Create the right panel with vertically stacked displays."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Display header
        header = QLabel("Homography Transformation Comparison")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(header)
        
        # Original frame display
        original_group = QGroupBox("Original Frame")
        original_layout = QVBoxLayout()
        
        self.original_display = QLabel("No video selected")
        self.original_display.setAlignment(Qt.AlignCenter)
        self.original_display.setFixedHeight(350)
        self.original_display.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #1a1a1a;
                color: #999;
                font-size: 12px;
            }
        """)
        original_layout.addWidget(self.original_display)
        original_group.setLayout(original_layout)
        layout.addWidget(original_group)
        
        # Warped frame display
        warped_group = QGroupBox("Warped Frame (with buffer)")
        warped_layout = QVBoxLayout()
        
        self.warped_display = QLabel("No video selected")
        self.warped_display.setAlignment(Qt.AlignCenter)
        self.warped_display.setFixedHeight(350)
        self.warped_display.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #1a1a1a;
                color: #999;
                font-size: 12px;
            }
        """)
        warped_layout.addWidget(self.warped_display)
        warped_group.setLayout(warped_layout)
        layout.addWidget(warped_group)
        
        # Add stretch to fill remaining space
        layout.addStretch()
        
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
        
    def _display_frame(self, frame: np.ndarray, label: QLabel):
        """Display a frame in the specified label widget."""
        if frame is None:
            return
            
        # Convert to Qt format
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
        
    def _apply_homography(self, frame: np.ndarray) -> np.ndarray:
        """Apply homography transformation to frame with buffer for extended transformations.
        
        Args:
            frame: Input frame to transform
            
        Returns:
            Warped frame with buffer to show transformations beyond original bounds
        """
        # Construct homography matrix
        H = np.array([
            [self.homography_params['H00'], self.homography_params['H01'], self.homography_params['H02']],
            [self.homography_params['H10'], self.homography_params['H11'], self.homography_params['H12']],
            [self.homography_params['H20'], self.homography_params['H21'], 1.0]
        ], dtype=np.float32)
        
        # Get original dimensions
        original_height, original_width = frame.shape[:2]
        
        # Create larger output canvas with buffer (50% larger in each direction)
        buffer_factor = get_setting("homography.buffer_factor", 1.5)
        output_width = int(original_width * buffer_factor)
        output_height = int(original_height * buffer_factor)
        
        # Calculate offset to center the original frame in the larger canvas
        offset_x = (output_width - original_width) // 2
        offset_y = (output_height - original_height) // 2
        
        # Adjust homography matrix to account for the centering offset
        # This ensures the transformation is centered in the larger canvas
        center_offset_matrix = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y], 
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combine the centering offset with the user's homography
        H_buffered = center_offset_matrix @ H
        
        # Apply perspective transform to larger canvas
        warped = cv2.warpPerspective(frame, H_buffered, (output_width, output_height))
        
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
