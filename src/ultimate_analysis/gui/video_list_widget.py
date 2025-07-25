"""Video list widget for displaying available video files."""

import os
import cv2
from typing import List, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, 
    QPushButton, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from ultimate_analysis.config import get_setting


class VideoListWidget(QWidget):
    """Widget for displaying and managing video files."""
    
    # Signals
    video_selected = pyqtSignal(str)  # Emits video file path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_folder = ""
        self.video_files: List[Tuple[str, str, int]] = []  # (filename, path, duration_seconds)
        
        self.init_ui()
        self.load_default_folder()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Header with refresh button
        header_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("🔄 Refresh")
        self.refresh_button.setToolTip("Refresh video list [F5]")
        self.refresh_button.clicked.connect(self.refresh_video_list)
        header_layout.addWidget(self.refresh_button)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Video table
        self.video_table = QTableWidget()
        self.video_table.setColumnCount(3)
        self.video_table.setHorizontalHeaderLabels(["File", "Duration", "Size"])
        
        # Configure table
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.video_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.video_table.setAlternatingRowColors(True)
        
        # Set column widths
        header = self.video_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # File name stretches
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Duration fits content
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Size fits content
        
        # Connect selection signal
        self.video_table.itemSelectionChanged.connect(self.on_selection_changed)
        
        layout.addWidget(self.video_table)
        self.setLayout(layout)
    
    def load_default_folder(self):
        """Load video files from default folder."""
        data_dir = get_setting("data.dev_data_dir", "data/processed/dev_data")
        self.set_video_folder(data_dir)
    
    def set_video_folder(self, folder_path: str):
        """
        Set the folder to scan for video files.
        
        Args:
            folder_path: Path to folder containing video files
        """
        self.video_folder = folder_path
        self.refresh_video_list()
    
    def refresh_video_list(self):
        """Refresh the list of video files."""
        self.video_files.clear()
        self.video_table.setRowCount(0)
        
        if not os.path.exists(self.video_folder):
            os.makedirs(self.video_folder, exist_ok=True)
            return
        
        # Get supported video formats
        supported_formats = get_setting("video.supported_formats", [".mp4", ".avi", ".mov", ".mkv", ".wmv"])
        
        # Scan folder for video files
        for filename in sorted(os.listdir(self.video_folder)):
            if any(filename.lower().endswith(ext) for ext in supported_formats):
                file_path = os.path.join(self.video_folder, filename)
                
                # Get file info
                duration = self.get_video_duration(file_path)
                file_size = os.path.getsize(file_path)
                
                self.video_files.append((filename, file_path, duration))
                self.add_video_to_table(filename, duration, file_size)
        
        # Select first video if available
        if self.video_table.rowCount() > 0:
            self.video_table.selectRow(0)
            self.on_selection_changed()
    
    def add_video_to_table(self, filename: str, duration: int, file_size: int):
        """
        Add a video file to the table.
        
        Args:
            filename: Name of video file
            duration: Duration in seconds
            file_size: File size in bytes
        """
        row = self.video_table.rowCount()
        self.video_table.insertRow(row)
        
        # File name
        name_item = QTableWidgetItem(filename)
        name_item.setToolTip(filename)
        self.video_table.setItem(row, 0, name_item)
        
        # Duration
        duration_str = self.format_duration(duration)
        duration_item = QTableWidgetItem(duration_str)
        duration_item.setData(Qt.UserRole, duration)  # Store raw duration for sorting
        self.video_table.setItem(row, 1, duration_item)
        
        # File size
        size_str = self.format_file_size(file_size)
        size_item = QTableWidgetItem(size_str)
        size_item.setData(Qt.UserRole, file_size)  # Store raw size for sorting
        self.video_table.setItem(row, 2, size_item)
    
    def get_video_duration(self, file_path: str) -> int:
        """
        Get video duration in seconds.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Duration in seconds, or 0 if unable to determine
        """
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return 0
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if fps > 0:
                return int(frame_count / fps)
            return 0
            
        except Exception:
            return 0
    
    def format_duration(self, seconds: int) -> str:
        """
        Format duration as MM:SS or HH:MM:SS.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds <= 0:
            return "0:00"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Formatted size string
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def on_selection_changed(self):
        """Handle selection change in video table."""
        selected_rows = self.video_table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self.video_files):
                filename, file_path, duration = self.video_files[row]
                self.video_selected.emit(file_path)
    
    def get_current_video(self) -> str:
        """
        Get currently selected video file path.
        
        Returns:
            Path to selected video file, or empty string if none selected
        """
        selected_rows = self.video_table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self.video_files):
                return self.video_files[row][1]
        return ""
    
    def select_next_video(self):
        """Select next video in the list."""
        current_row = self.video_table.currentRow()
        if current_row < self.video_table.rowCount() - 1:
            self.video_table.selectRow(current_row + 1)
            self.on_selection_changed()
    
    def select_previous_video(self):
        """Select previous video in the list."""
        current_row = self.video_table.currentRow()
        if current_row > 0:
            self.video_table.selectRow(current_row - 1)
            self.on_selection_changed()
    
    def select_random_video(self):
        """Select a random video from the list."""
        if self.video_table.rowCount() > 0:
            import random
            random_row = random.randint(0, self.video_table.rowCount() - 1)
            self.video_table.selectRow(random_row)
            self.on_selection_changed()
