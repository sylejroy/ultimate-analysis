"""Main window for Ultimate Analysis application."""

import logging
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
    QTabWidget, QSplitter, QApplication
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon

from ultimate_analysis.config import get_setting
from ultimate_analysis.gui.video_list_widget import VideoListWidget
from ultimate_analysis.gui.video_widget import VideoWidget
from ultimate_analysis.processing.inference import run_inference
from ultimate_analysis.processing.tracking import run_tracking
from ultimate_analysis.processing.field_segmentation import run_field_segmentation
from ultimate_analysis.processing.player_id import run_player_id

logger = logging.getLogger("ultimate_analysis.gui.main_window")


class MainTab(QWidget):
    """Main tab containing video player and controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.connect_signals()
        
        # Processing state
        self.current_detections = []
        self.current_tracks = []
        self.current_field_results = None
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout()
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Video list
        self.video_list_widget = VideoListWidget()
        self.video_list_widget.setMaximumWidth(400)
        self.video_list_widget.setMinimumWidth(300)
        splitter.addWidget(self.video_list_widget)
        
        # Right panel - Video player
        self.video_widget = VideoWidget()
        splitter.addWidget(self.video_widget)
        
        # Set splitter proportions (30% left, 70% right)
        splitter.setSizes([300, 700])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
    
    def connect_signals(self):
        """Connect widget signals."""
        # Video selection
        self.video_list_widget.video_selected.connect(self.load_video)
        
        # Video playback
        self.video_widget.frame_changed.connect(self.process_frame)
        
        # Button connections
        self.video_widget.prev_button.clicked.connect(self.video_list_widget.select_previous_video)
        self.video_widget.next_button.clicked.connect(self.video_list_widget.select_next_video)
    
    @pyqtSlot(str)
    def load_video(self, video_path: str):
        """
        Load a video file.
        
        Args:
            video_path: Path to video file
        """
        logger.info(f"Loading video: {video_path}")
        success = self.video_widget.load_video(video_path)
        
        if success:
            logger.info("Video loaded successfully")
            # Auto-start playback if configured
            if get_setting("gui.auto_play", False):
                self.video_widget.start_playback()
        else:
            logger.error("Failed to load video")
    
    @pyqtSlot(np.ndarray)
    def process_frame(self, frame: np.ndarray):
        """
        Process a video frame with enabled processing modules.
        
        Args:
            frame: Video frame to process
        """
        # Get processing settings
        settings = self.video_widget.get_processing_settings()
        
        # Reset processing results
        self.current_detections = []
        self.current_tracks = []
        
        try:
            # Object detection
            if settings['inference']:
                logger.debug("Running inference")
                self.current_detections = run_inference(frame)
                logger.debug(f"Detected {len(self.current_detections)} objects")
            
            # Object tracking
            if settings['tracking'] and settings['inference']:
                logger.debug("Running tracking")
                self.current_tracks = run_tracking(frame, self.current_detections)
                logger.debug(f"Tracking {len(self.current_tracks)} objects")
            
            # Field segmentation
            if settings['field_segmentation']:
                logger.debug("Running field segmentation")
                self.current_field_results = run_field_segmentation(frame)
            
            # Player identification
            if settings['player_id'] and settings['tracking']:
                self.process_player_id(frame)
            
            # TODO: Apply visualizations to frame
            # processed_frame = self.apply_visualizations(frame)
            # self.video_widget.display_frame(processed_frame)
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def process_player_id(self, frame: np.ndarray):
        """
        Run player ID on tracked objects.
        
        Args:
            frame: Current video frame
        """
        for track in self.current_tracks:
            try:
                # Get bounding box
                if hasattr(track, 'to_ltrb'):
                    x1, y1, x2, y2 = track.to_ltrb()
                else:
                    continue
                
                # Crop player region (top half for jersey number)
                crop_height = (y2 - y1) // 2
                player_crop = frame[y1:y1 + crop_height, x1:x2]
                
                if player_crop.size > 0:
                    digit_str, details = run_player_id(player_crop)
                    logger.debug(f"Player ID for track {track.track_id}: {digit_str}")
                    
            except Exception as e:
                logger.error(f"Error in player ID processing: {e}")
    
    def apply_visualizations(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply visualization overlays to frame.
        
        Args:
            frame: Original frame
            
        Returns:
            Frame with visualizations applied
        """
        # TODO: Implement visualization overlays
        # This would draw bounding boxes, track histories, field segmentation, etc.
        return frame


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.apply_styling()
    
    def init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle(get_setting("gui.window_title", "Ultimate Analysis"))
        self.setWindowIcon(QIcon())  # TODO: Add application icon
        
        # Set window size
        width = get_setting("gui.window_width", 1920)
        height = get_setting("gui.window_height", 1080)
        self.resize(width, height)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create main tab
        self.main_tab = MainTab()
        self.tab_widget.addTab(self.main_tab, "Main")
        
        # TODO: Add additional tabs as needed
        # self.tab_widget.addTab(DevTab(), "Development")
        
        self.setCentralWidget(self.tab_widget)
        
        # Load a random video on startup
        self.load_random_video_on_startup()
    
    def apply_styling(self):
        """Apply dark theme styling."""
        if get_setting("gui.theme", "dark") == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #232323;
                    color: #f0f0f0;
                }
                QTabWidget::pane {
                    border: 1px solid #444444;
                    background-color: #232323;
                }
                QTabWidget::tab-bar {
                    alignment: left;
                }
                QTabBar::tab {
                    background-color: #333333;
                    color: #f0f0f0;
                    padding: 8px 16px;
                    margin-right: 2px;
                    border: 1px solid #444444;
                    border-bottom: none;
                }
                QTabBar::tab:selected {
                    background-color: #444444;
                    border-bottom: 1px solid #444444;
                }
                QTabBar::tab:hover {
                    background-color: #3a3a3a;
                }
                QWidget {
                    background-color: #232323;
                    color: #f0f0f0;
                }
                QLabel {
                    color: #f0f0f0;
                }
                QPushButton {
                    background-color: #333333;
                    color: #f0f0f0;
                    border: 1px solid #444444;
                    padding: 6px 12px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #3a3a3a;
                }
                QPushButton:pressed {
                    background-color: #222222;
                }
                QCheckBox {
                    color: #f0f0f0;
                    spacing: 6px;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                }
                QCheckBox::indicator:unchecked {
                    background-color: #333333;
                    border: 1px solid #555555;
                    border-radius: 2px;
                }
                QCheckBox::indicator:checked {
                    background-color: #0078d4;
                    border: 1px solid #0078d4;
                    border-radius: 2px;
                }
                QSlider::groove:horizontal {
                    border: 1px solid #444444;
                    height: 8px;
                    background-color: #333333;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background-color: #0078d4;
                    border: 1px solid #0078d4;
                    width: 18px;
                    margin: -5px 0;
                    border-radius: 9px;
                }
                QSlider::handle:horizontal:hover {
                    background-color: #106ebe;
                }
                QTableWidget {
                    background-color: #1e1e1e;
                    alternate-background-color: #2a2a2a;
                    selection-background-color: #0078d4;
                    gridline-color: #444444;
                    border: 1px solid #444444;
                }
                QTableWidget QHeaderView::section {
                    background-color: #333333;
                    color: #f0f0f0;
                    padding: 6px;
                    border: 1px solid #444444;
                }
                QSplitter::handle {
                    background-color: #444444;
                }
                QSplitter::handle:hover {
                    background-color: #555555;
                }
            """)
    
    def load_random_video_on_startup(self):
        """Load a random video on application startup."""
        try:
            # Small delay to ensure widgets are fully initialized
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, self.main_tab.video_list_widget.select_random_video)
        except Exception as e:
            logger.warning(f"Could not load random video on startup: {e}")
    
    def closeEvent(self, event):
        """Handle application close event."""
        logger.info("Closing Ultimate Analysis")
        
        # Stop video playback
        if hasattr(self, 'main_tab') and hasattr(self.main_tab, 'video_widget'):
            self.main_tab.video_widget.stop_playback()
        
        # Accept the close event
        event.accept()
