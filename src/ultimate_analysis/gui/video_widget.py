"""Video player widget for displaying and controlling video playback."""

import cv2
import numpy as np
from typing import Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QCheckBox, QSizePolicy, QShortcut
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QKeySequence

from ultimate_analysis.config import get_setting


class VideoWidget(QWidget):
    """Widget for video display and playback controls."""
    
    # Signals
    frame_changed = pyqtSignal(np.ndarray)  # Emits current frame
    playback_state_changed = pyqtSignal(bool)  # Emits True if playing, False if paused
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Video state
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.current_video_path = ""
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 25.0
        self.is_playing = False
        
        # Processing state
        self.current_frame_data: Optional[np.ndarray] = None
        
        self.init_ui()
        self.init_timer()
        self.setup_shortcuts()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Video display area
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #444444;
                background-color: #1e1e1e;
                color: #cccccc;
                font-size: 14px;
            }
        """)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label, 1)
        
        # Progress bar
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(100)
        self.progress_slider.setValue(0)
        self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.on_slider_released)
        self.progress_slider.valueChanged.connect(self.on_slider_moved)
        layout.addWidget(self.progress_slider)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        # Previous video button
        self.prev_button = QPushButton("⏮")
        self.prev_button.setToolTip("Previous recording [←]")
        self.prev_button.setMaximumWidth(50)
        controls_layout.addWidget(self.prev_button)
        
        # Play/Pause button
        self.play_pause_button = QPushButton("▶")
        self.play_pause_button.setToolTip("Play/Pause [Space]")
        self.play_pause_button.setMaximumWidth(50)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        controls_layout.addWidget(self.play_pause_button)
        
        # Next video button
        self.next_button = QPushButton("⏭")
        self.next_button.setToolTip("Next recording [→]")
        self.next_button.setMaximumWidth(50)
        controls_layout.addWidget(self.next_button)
        
        # Processing checkboxes
        controls_layout.addSpacing(20)
        
        self.inference_checkbox = QCheckBox("Inference")
        self.inference_checkbox.setToolTip("Enable object detection [I]")
        controls_layout.addWidget(self.inference_checkbox)
        
        self.tracking_checkbox = QCheckBox("Tracking")
        self.tracking_checkbox.setToolTip("Enable object tracking [T]")
        controls_layout.addWidget(self.tracking_checkbox)
        
        self.player_id_checkbox = QCheckBox("Player ID")
        self.player_id_checkbox.setToolTip("Enable player identification [P]")
        controls_layout.addWidget(self.player_id_checkbox)
        
        self.field_segmentation_checkbox = QCheckBox("Field Segmentation")
        self.field_segmentation_checkbox.setToolTip("Enable field segmentation [F]")
        controls_layout.addWidget(self.field_segmentation_checkbox)
        
        # Add stretch to push controls to left
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        self.setLayout(layout)
    
    def init_timer(self):
        """Initialize the playback timer."""
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.next_frame)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Space - Play/Pause
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_play_pause)
        
        # Arrow keys - Previous/Next video
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self.prev_button.click())
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.next_button.click())
        
        # Processing toggles
        QShortcut(QKeySequence(Qt.Key_I), self, lambda: self.inference_checkbox.toggle())
        QShortcut(QKeySequence(Qt.Key_T), self, lambda: self.tracking_checkbox.toggle())
        QShortcut(QKeySequence(Qt.Key_P), self, lambda: self.player_id_checkbox.toggle())
        QShortcut(QKeySequence(Qt.Key_F), self, lambda: self.field_segmentation_checkbox.toggle())
    
    def load_video(self, video_path: str) -> bool:
        """
        Load a video file for playback.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video loaded successfully, False otherwise
        """
        # Stop current playback
        self.stop_playback()
        
        # Release current video
        if self.video_cap:
            self.video_cap.release()
        
        # Load new video
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            self.video_label.setText(f"Failed to load: {video_path}")
            return False
        
        # Get video properties
        self.current_video_path = video_path
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.current_frame = 0
        
        # Update UI
        self.progress_slider.setMaximum(max(0, self.total_frames - 1))
        self.progress_slider.setValue(0)
        
        # Load first frame
        self.seek_to_frame(0)
        
        video_name = video_path.split('/')[-1].split('\\')[-1]
        self.video_label.setToolTip(f"Loaded: {video_name} ({self.total_frames} frames, {self.fps:.1f} fps)")
        
        return True
    
    def toggle_play_pause(self):
        """Toggle between play and pause."""
        if not self.video_cap:
            return
        
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start video playback."""
        if not self.video_cap:
            return
        
        self.is_playing = True
        self.play_pause_button.setText("⏸")
        
        # Calculate timer interval based on FPS and frame skip
        frame_skip = get_setting("video.frame_skip", 1)
        interval_ms = int((1000 / self.fps) * frame_skip)
        
        self.playback_timer.start(interval_ms)
        self.playback_state_changed.emit(True)
    
    def pause_playback(self):
        """Pause video playback."""
        self.is_playing = False
        self.play_pause_button.setText("▶")
        self.playback_timer.stop()
        self.playback_state_changed.emit(False)
    
    def stop_playback(self):
        """Stop video playback and reset to beginning."""
        self.pause_playback()
        self.seek_to_frame(0)
    
    def next_frame(self):
        """Advance to next frame."""
        if not self.video_cap:
            return
        
        frame_skip = get_setting("video.frame_skip", 1)
        
        ret, frame = self.video_cap.read()
        if not ret:
            # End of video
            self.pause_playback()
            return
        
        # Skip additional frames if configured
        for _ in range(frame_skip - 1):
            ret, _ = self.video_cap.read()
            if not ret:
                self.pause_playback()
                return
        
        self.current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.current_frame_data = frame
        
        # Update progress slider (don't trigger seek)
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(self.current_frame)
        self.progress_slider.blockSignals(False)
        
        # Display frame
        self.display_frame(frame)
        
        # Emit frame for processing
        self.frame_changed.emit(frame.copy())
    
    def seek_to_frame(self, frame_number: int):
        """
        Seek to specific frame.
        
        Args:
            frame_number: Frame number to seek to
        """
        if not self.video_cap:
            return
        
        frame_number = max(0, min(frame_number, self.total_frames - 1))
        
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_cap.read()
        
        if ret:
            self.current_frame = frame_number
            self.current_frame_data = frame
            self.display_frame(frame)
            self.frame_changed.emit(frame.copy())
        
        # Update progress slider
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(frame_number)
        self.progress_slider.blockSignals(False)
    
    def display_frame(self, frame: np.ndarray):
        """
        Display a frame in the video label.
        
        Args:
            frame: Frame to display (BGR format)
        """
        if frame is None:
            return
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage and QPixmap
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def on_slider_pressed(self):
        """Handle slider press (pause playback during seeking)."""
        self.was_playing = self.is_playing
        if self.is_playing:
            self.pause_playback()
    
    def on_slider_released(self):
        """Handle slider release (resume playback if was playing)."""
        if hasattr(self, 'was_playing') and self.was_playing:
            self.start_playback()
    
    def on_slider_moved(self, value: int):
        """
        Handle slider movement.
        
        Args:
            value: New slider value (frame number)
        """
        if self.video_cap and not self.playback_timer.isActive():
            self.seek_to_frame(value)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame data.
        
        Returns:
            Current frame as numpy array, or None if no video loaded
        """
        return self.current_frame_data.copy() if self.current_frame_data is not None else None
    
    def get_processing_settings(self) -> dict:
        """
        Get current processing settings from checkboxes.
        
        Returns:
            Dictionary of processing settings
        """
        return {
            'inference': self.inference_checkbox.isChecked(),
            'tracking': self.tracking_checkbox.isChecked(),
            'player_id': self.player_id_checkbox.isChecked(),
            'field_segmentation': self.field_segmentation_checkbox.isChecked()
        }
    
    def closeEvent(self, event):
        """Handle widget close event."""
        self.stop_playback()
        if self.video_cap:
            self.video_cap.release()
        super().closeEvent(event)
