"""
Development video preprocessing tab - comprehensive implementation.
"""
import os
import cv2
import logging
from typing import List, Optional, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QPushButton, 
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QProgressBar,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QTextEdit, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from ..components.video_player import VideoPlayer
from ...utils.dataset_generation import (
    sample_all_videos, crop_images_in_folder, augment_dataset,
    make_test_videos, make_test_screenshot
)

logger = logging.getLogger("ultimate_analysis.gui.dev_video_preprocessing")


class VideoProcessingWorker(QThread):
    """Worker thread for video processing operations."""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, operation: str, params: Dict[str, Any]):
        super().__init__()
        self.operation = operation
        self.params = params
        
    def run(self):
        """Run the video processing operation."""
        try:
            if self.operation == "sample_frames":
                self._sample_frames()
            elif self.operation == "crop_images":
                self._crop_images()
            elif self.operation == "augment_dataset":
                self._augment_dataset()
            elif self.operation == "make_snippets":
                self._make_snippets()
            else:
                raise ValueError(f"Unknown operation: {self.operation}")
                
            self.finished.emit(True, "Operation completed successfully")
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.finished.emit(False, str(e))
    
    def _sample_frames(self):
        """Sample frames from videos."""
        self.status_updated.emit("Sampling frames from videos...")
        input_folder = self.params.get("input_folder", "data/raw/videos")
        output_folder = self.params.get("output_folder", "data/processed/sampled_frames")
        interval = self.params.get("interval", 20)
        
        sample_all_videos(input_folder, output_folder, interval)
        
    def _crop_images(self):
        """Crop images to specified size."""
        self.status_updated.emit("Cropping images...")
        input_folder = self.params.get("input_folder", "data/processed/sampled_frames")
        output_folder = self.params.get("output_folder", "data/processed/cropped_images")
        crop_size = self.params.get("crop_size", (640, 640))
        
        crop_images_in_folder(input_folder, output_folder, crop_size)
        
    def _augment_dataset(self):
        """Augment dataset with transformations."""
        self.status_updated.emit("Augmenting dataset...")
        input_folder = self.params.get("input_folder", "data/processed/cropped_images")
        output_folder = self.params.get("output_folder", "data/processed/augmented_dataset")
        augmentation_factor = self.params.get("augmentation_factor", 3)
        
        augment_dataset(input_folder, output_folder, augmentation_factor)
        
    def _make_snippets(self):
        """Create video snippets."""
        self.status_updated.emit("Creating video snippets...")
        input_folder = self.params.get("input_folder", "data/raw/videos")
        snippet_duration = self.params.get("snippet_duration", 7)
        num_snippets = self.params.get("num_snippets", 4)
        
        make_test_videos(input_folder, snippet_duration, num_snippets)


class DevVideoPreprocessingTab(QWidget):
    """
    Tab for video preprocessing development tools.
    """
    
    def __init__(self):
        super().__init__()
        self.video_player = VideoPlayer()
        self.current_video_files: List[str] = []
        self.processing_worker: Optional[VideoProcessingWorker] = None
        self._init_ui()
        self._load_video_list()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout()
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Video list and controls
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Video display and processing
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set initial sizes
        splitter.setSizes([400, 800])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
    
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with video list and controls."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Video list
        video_group = QGroupBox("Video Files")
        video_layout = QVBoxLayout()
        
        self.video_list = QListWidget()
        self.video_list.itemSelectionChanged.connect(self._on_video_selected)
        video_layout.addWidget(self.video_list)
        
        # Video list controls
        video_controls = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._load_video_list)
        video_controls.addWidget(self.refresh_button)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_video_folder)
        video_controls.addWidget(self.browse_button)
        
        video_layout.addLayout(video_controls)
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Processing settings
        settings_group = self._create_settings_group()
        layout.addWidget(settings_group)
        
        # Processing controls
        processing_group = self._create_processing_group()
        layout.addWidget(processing_group)
        
        # Status and progress
        status_group = self._create_status_group()
        layout.addWidget(status_group)
        
        panel.setLayout(layout)
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with video display and info."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel("Select a video to preview")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(400)
        self.video_label.setStyleSheet("border: 1px solid #444; background: #1a1a1a;")
        layout.addWidget(self.video_label)
        
        # Video info
        info_group = QGroupBox("Video Information")
        info_layout = QVBoxLayout()
        
        self.video_info_text = QTextEdit()
        self.video_info_text.setMaximumHeight(150)
        self.video_info_text.setReadOnly(True)
        info_layout.addWidget(self.video_info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Processing log
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        panel.setLayout(layout)
        return panel
    
    def _create_settings_group(self) -> QGroupBox:
        """Create the processing settings group."""
        settings_group = QGroupBox("Processing Settings")
        layout = QFormLayout()
        
        # Frame sampling settings
        self.sample_interval_spin = QSpinBox()
        self.sample_interval_spin.setRange(1, 300)
        self.sample_interval_spin.setValue(20)
        self.sample_interval_spin.setSuffix(" seconds")
        layout.addRow("Frame Sampling Interval:", self.sample_interval_spin)
        
        # Crop size settings
        self.crop_width_spin = QSpinBox()
        self.crop_width_spin.setRange(64, 2048)
        self.crop_width_spin.setValue(640)
        layout.addRow("Crop Width:", self.crop_width_spin)
        
        self.crop_height_spin = QSpinBox()
        self.crop_height_spin.setRange(64, 2048)
        self.crop_height_spin.setValue(640)
        layout.addRow("Crop Height:", self.crop_height_spin)
        
        # Augmentation settings
        self.augmentation_factor_spin = QSpinBox()
        self.augmentation_factor_spin.setRange(1, 10)
        self.augmentation_factor_spin.setValue(3)
        layout.addRow("Augmentation Factor:", self.augmentation_factor_spin)
        
        # Snippet settings
        self.snippet_duration_spin = QSpinBox()
        self.snippet_duration_spin.setRange(1, 60)
        self.snippet_duration_spin.setValue(7)
        self.snippet_duration_spin.setSuffix(" seconds")
        layout.addRow("Snippet Duration:", self.snippet_duration_spin)
        
        self.num_snippets_spin = QSpinBox()
        self.num_snippets_spin.setRange(1, 20)
        self.num_snippets_spin.setValue(4)
        layout.addRow("Number of Snippets:", self.num_snippets_spin)
        
        settings_group.setLayout(layout)
        return settings_group
    
    def _create_processing_group(self) -> QGroupBox:
        """Create the processing controls group."""
        processing_group = QGroupBox("Processing Operations")
        layout = QVBoxLayout()
        
        # Processing buttons
        self.sample_frames_button = QPushButton("Sample Frames")
        self.sample_frames_button.clicked.connect(self._sample_frames)
        layout.addWidget(self.sample_frames_button)
        
        self.crop_images_button = QPushButton("Crop Images")
        self.crop_images_button.clicked.connect(self._crop_images)
        layout.addWidget(self.crop_images_button)
        
        self.augment_dataset_button = QPushButton("Augment Dataset")
        self.augment_dataset_button.clicked.connect(self._augment_dataset)
        layout.addWidget(self.augment_dataset_button)
        
        self.make_snippets_button = QPushButton("Make Snippets")
        self.make_snippets_button.clicked.connect(self._make_snippets)
        layout.addWidget(self.make_snippets_button)
        
        # Screenshot button
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        layout.addWidget(separator)
        
        self.screenshot_button = QPushButton("Take Screenshot")
        self.screenshot_button.clicked.connect(self._take_screenshot)
        layout.addWidget(self.screenshot_button)
        
        processing_group.setLayout(layout)
        return processing_group
    
    def _create_status_group(self) -> QGroupBox:
        """Create the status and progress group."""
        status_group = QGroupBox("Status")
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        status_group.setLayout(layout)
        return status_group
    
    def _load_video_list(self):
        """Load the list of available videos."""
        video_folder = "data/raw/videos"
        if not os.path.exists(video_folder):
            self.video_list.clear()
            self.current_video_files = []
            return
        
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        video_files = [
            f for f in os.listdir(video_folder)
            if f.lower().endswith(video_extensions)
        ]
        
        self.current_video_files = video_files
        self.video_list.clear()
        self.video_list.addItems(video_files)
        
        self._log(f"Found {len(video_files)} video files")
    
    def _browse_video_folder(self):
        """Browse for video folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if folder:
            # Update the video folder and reload list
            self._log(f"Selected folder: {folder}")
            # For now, just refresh the current list
            self._load_video_list()
    
    def _on_video_selected(self):
        """Handle video selection."""
        current_item = self.video_list.currentItem()
        if not current_item:
            return
        
        video_file = current_item.text()
        video_path = os.path.join("data/raw/videos", video_file)
        
        if os.path.exists(video_path):
            self._load_video_preview(video_path)
            self._display_video_info(video_path)
    
    def _load_video_preview(self, video_path: str):
        """Load video preview."""
        if self.video_player.load_video(video_path):
            frame = self.video_player.get_next_frame()
            if frame is not None:
                self._display_frame(frame)
    
    def _display_frame(self, frame):
        """Display a frame in the video label."""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def _display_video_info(self, video_path: str):
        """Display video information."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.video_info_text.setText("Could not open video")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            
            info_text = f"""
File: {os.path.basename(video_path)}
Resolution: {width}x{height}
FPS: {fps:.2f}
Frame Count: {frame_count}
Duration: {duration:.2f} seconds
File Size: {file_size:.2f} MB
            """.strip()
            
            self.video_info_text.setText(info_text)
            cap.release()
            
        except Exception as e:
            self.video_info_text.setText(f"Error reading video info: {e}")
    
    def _log(self, message: str):
        """Add message to the log."""
        self.log_text.append(f"[{self._get_timestamp()}] {message}")
        logger.info(message)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def _start_processing(self, operation: str, params: Dict[str, Any]):
        """Start a processing operation."""
        if self.processing_worker and self.processing_worker.isRunning():
            QMessageBox.warning(self, "Processing", "Another operation is already running")
            return
        
        self.processing_worker = VideoProcessingWorker(operation, params)
        self.processing_worker.progress_updated.connect(self.progress_bar.setValue)
        self.processing_worker.status_updated.connect(self._on_status_updated)
        self.processing_worker.finished.connect(self._on_processing_finished)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._set_processing_enabled(False)
        
        self.processing_worker.start()
    
    def _on_status_updated(self, message: str):
        """Handle status update."""
        self.status_label.setText(message)
        self._log(message)
    
    def _on_processing_finished(self, success: bool, message: str):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self._set_processing_enabled(True)
        
        if success:
            self.status_label.setText("Ready")
            self._log(f"✓ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_label.setText("Error")
            self._log(f"✗ {message}")
            QMessageBox.critical(self, "Error", message)
    
    def _set_processing_enabled(self, enabled: bool):
        """Enable/disable processing buttons."""
        self.sample_frames_button.setEnabled(enabled)
        self.crop_images_button.setEnabled(enabled)
        self.augment_dataset_button.setEnabled(enabled)
        self.make_snippets_button.setEnabled(enabled)
        self.screenshot_button.setEnabled(enabled)
    
    # Processing operations
    def _sample_frames(self):
        """Sample frames from videos."""
        params = {
            "input_folder": "data/raw/videos",
            "output_folder": "data/processed/sampled_frames",
            "interval": self.sample_interval_spin.value()
        }
        self._start_processing("sample_frames", params)
    
    def _crop_images(self):
        """Crop images to specified size."""
        params = {
            "input_folder": "data/processed/sampled_frames",
            "output_folder": "data/processed/cropped_images",
            "crop_size": (self.crop_width_spin.value(), self.crop_height_spin.value())
        }
        self._start_processing("crop_images", params)
    
    def _augment_dataset(self):
        """Augment dataset with transformations."""
        params = {
            "input_folder": "data/processed/cropped_images",
            "output_folder": "data/processed/augmented_dataset",
            "augmentation_factor": self.augmentation_factor_spin.value()
        }
        self._start_processing("augment_dataset", params)
    
    def _make_snippets(self):
        """Create video snippets."""
        params = {
            "input_folder": "data/raw/videos",
            "snippet_duration": self.snippet_duration_spin.value(),
            "num_snippets": self.num_snippets_spin.value()
        }
        self._start_processing("make_snippets", params)
    
    def _take_screenshot(self):
        """Take a screenshot from the selected video."""
        current_item = self.video_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Screenshot", "Please select a video first")
            return
        
        video_file = current_item.text()
        video_path = os.path.join("data/raw/videos", video_file)
        
        try:
            screenshot_path = make_test_screenshot(video_path, "data/processed")
            if screenshot_path:
                self._log(f"Screenshot saved: {screenshot_path}")
                QMessageBox.information(self, "Screenshot", f"Screenshot saved to {screenshot_path}")
            else:
                QMessageBox.warning(self, "Screenshot", "Failed to take screenshot")
        except Exception as e:
            QMessageBox.critical(self, "Screenshot", f"Error taking screenshot: {e}")
    
    def closeEvent(self, event):
        """Handle tab close event."""
        if self.processing_worker and self.processing_worker.isRunning():
            self.processing_worker.terminate()
            self.processing_worker.wait()
        
        self.video_player.release()
        super().closeEvent(event)
