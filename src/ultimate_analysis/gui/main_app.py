"""Main application window for Ultimate Analysis.

This module contains the main PyQt5 application with tabbed interface
for video analysis functionality.
"""

import sys
from typing import Optional

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPalette, QColor, QIcon

from .main_tab import MainTab
from .easyocr_tuning_tab import EasyOCRTuningTab
from .model_tuning_tab import ModelTuningTab
from ..config.settings import get_setting
from ..constants import DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT


class UltimateAnalysisApp(QMainWindow):
    """Main application window with tabbed interface."""
    
    def __init__(self):
        super().__init__()
        
        # Application state
        self._current_video_path: Optional[str] = None
        
        # Initialize UI
        self._init_ui()
        self._setup_dark_theme()
        
        print("[APP] Ultimate Analysis application initialized")
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Window properties
        self.setWindowTitle(get_setting("app.name", "Ultimate Analysis"))
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        
        # Show window maximized
        self.showMaximized()
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        layout.addWidget(self.tab_widget)
        
        # Create main tab
        self.main_tab = MainTab()
        self.main_tab.video_changed.connect(self._on_video_changed)
        self.tab_widget.addTab(self.main_tab, "Main Analysis")
        
        # Create EasyOCR tuning tab
        self.easyocr_tab = EasyOCRTuningTab()
        self.tab_widget.addTab(self.easyocr_tab, "EasyOCR Tuning")
        
        # Create Model Tuning tab
        self.model_tuning_tab = ModelTuningTab()
        self.tab_widget.addTab(self.model_tuning_tab, "Model Training")
        
        # TODO: Add more tabs as needed
        # Example placeholder tabs:
        # self.tab_widget.addTab(QWidget(), "Data Preprocessing") 
        # self.tab_widget.addTab(QWidget(), "Performance Analysis")
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        print("[APP] UI initialized with main tab")
    
    def _center_window(self):
        """Center the application window on screen."""
        screen = QApplication.desktop().screenGeometry()
        window = self.geometry()
        self.move(
            (screen.width() - window.width()) // 2,
            (screen.height() - window.height()) // 2
        )
    
    def _setup_dark_theme(self):
        """Setup dark theme for the application."""
        # Create dark palette
        dark_palette = QPalette()
        
        # Set colors for dark theme
        dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(60, 60, 60))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        
        # Apply palette
        self.setPalette(dark_palette)
        
        # Additional stylesheet for enhanced dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d2d;
                color: #ffffff;
            }
            
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2d2d2d;
            }
            
            QTabWidget::tab-bar {
                alignment: center;
            }
            
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: #2d2d2d;
                border-bottom: 2px solid #42a5f5;
            }
            
            QTabBar::tab:hover {
                background-color: #4a4a4a;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QPushButton {
                background-color: #4a4a4a;
                border: 1px solid #666666;
                padding: 6px 12px;
                border-radius: 3px;
                color: #ffffff;
            }
            
            QPushButton:hover {
                background-color: #5a5a5a;
                border: 1px solid #777777;
            }
            
            QPushButton:pressed {
                background-color: #333333;
            }
            
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            
            QCheckBox::indicator:unchecked {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:checked {
                background-color: #42a5f5;
                border: 1px solid #42a5f5;
                border-radius: 3px;
            }
            
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                padding: 4px 8px;
                border-radius: 3px;
                color: #ffffff;
            }
            
            QComboBox:hover {
                border: 1px solid #777777;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
                margin-right: 5px;
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #666666;
                height: 6px;
                background: #3c3c3c;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background: #42a5f5;
                border: 1px solid #42a5f5;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #64b5f6;
                border: 1px solid #64b5f6;
            }
            
            QListWidget {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                color: #ffffff;
                selection-background-color: #42a5f5;
                outline: none;
            }
            
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #555555;
            }
            
            QListWidget::item:selected {
                background-color: #42a5f5;
                color: #ffffff;
            }
            
            QListWidget::item:hover {
                background-color: #4a4a4a;
            }
            
            QStatusBar {
                background-color: #3c3c3c;
                color: #ffffff;
                border-top: 1px solid #666666;
            }
            
            QLabel {
                color: #ffffff;
                background-color: transparent;
            }
            
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                padding: 4px 8px;
                border-radius: 3px;
                color: #ffffff;
            }
            
            QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #777777;
            }
            
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                padding: 4px 8px;
                border-radius: 3px;
                color: #ffffff;
            }
            
            QLineEdit:hover {
                border: 1px solid #777777;
            }
            
            QLineEdit:focus {
                border: 1px solid #42a5f5;
            }
            
            QTextEdit {
                background-color: #3c3c3c;
                border: 1px solid #666666;
                color: #ffffff;
            }
            
            QFormLayout QLabel {
                color: #ffffff;
                font-weight: normal;
            }
            
            QToolTip {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #666666;
                padding: 4px;
                border-radius: 3px;
            }
            
            QScrollArea {
                background-color: #2d2d2d;
                border: 1px solid #555555;
            }
            
            QScrollBar:vertical {
                background-color: #3c3c3c;
                width: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #666666;
                border-radius: 6px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #777777;
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        print("[APP] Dark theme applied")
    
    def _on_video_changed(self, video_path: str):
        """Handle video change from main tab.
        
        Args:
            video_path: Path to the newly loaded video
        """
        self._current_video_path = video_path
        
        # Update window title
        import os
        video_name = os.path.basename(video_path)
        app_name = get_setting("app.name", "Ultimate Analysis")
        self.setWindowTitle(f"{app_name} - {video_name}")
        
        # Update status bar
        self.status_bar.showMessage(f"Loaded: {video_name}")
        
        print(f"[APP] Video changed to: {video_name}")
    
    def get_current_video_path(self) -> Optional[str]:
        """Get the path of the currently loaded video.
        
        Returns:
            Path to current video or None if no video loaded
        """
        return self._current_video_path
    
    def closeEvent(self, event):
        """Handle application close event."""
        print("[APP] Application closing...")
        
        # Cleanup main tab
        if hasattr(self, 'main_tab'):
            self.main_tab.close()
        
        # Accept the close event
        event.accept()
        print("[APP] Application closed")


def create_application() -> QApplication:
    """Create and configure the PyQt5 QApplication.
    
    Returns:
        Configured QApplication instance
    """
    # Create application
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName(get_setting("app.name", "Ultimate Analysis"))
    app.setApplicationVersion(get_setting("app.version", "0.1.0"))
    app.setOrganizationName("Ultimate Analysis Team")
    
    # Set high DPI scaling
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    return app


def main():
    """Main entry point for the Ultimate Analysis application."""
    print("[APP] Starting Ultimate Analysis...")
    
    # Create application
    app = create_application()
    
    # Create main window
    main_window = UltimateAnalysisApp()
    main_window.show()
    
    print("[APP] Application started, entering event loop")
    
    # Run event loop
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("[APP] Application interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
