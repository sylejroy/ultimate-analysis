"""
Main GUI application for Ultimate Analysis Visualization.
"""
import sys
import os
import logging
from PyQt5.QtWidgets import QApplication, QTabWidget, QLabel
from PyQt5.QtGui import QPalette, QColor, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt

from .tabs.main_tab import MainTab
from .tabs.dev_video_preprocessing_tab import DevVideoPreprocessingTab
from .tabs.dev_yolo_training_tab import DevYoloTrainingTab
from .tabs.easyocr_tuning_tab import DevEasyOCRTuningTab
from ..config.settings import get_settings

logger = logging.getLogger("ultimate_analysis.gui.app")


class VisualizationApp(QTabWidget):
    """
    Main application window for Ultimate Analysis Visualization.
    Only GUI logic is present here.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Analysis Visualization")
        self.showMaximized()
        self._initialize_models()
        self._setup_tabs()
        self._setup_dark_mode()

    def _initialize_models(self):
        """Initialize models from settings."""
        try:
            settings = get_settings()
            
            # Initialize detection model
            try:
                from ..processing.inference import set_detection_model
                set_detection_model(settings.models.detection_model)
                logger.info(f"Detection model loaded: {settings.models.detection_model}")
            except ImportError as e:
                logger.warning(f"Detection model not available - inference module not imported: {e}")
            except Exception as e:
                logger.error(f"Failed to load detection model: {e}")
                logger.error(f"Model path: {settings.models.detection_model}")
            
            # Initialize field segmentation model
            try:
                from ..processing import field_segmentation
                field_segmentation.set_field_model(settings.models.segmentation_model)
                logger.info(f"Field segmentation model loaded: {settings.models.segmentation_model}")
            except ImportError as e:
                logger.warning(f"Field segmentation model not available - field_segmentation module not imported: {e}")
            except Exception as e:
                logger.error(f"Failed to load field segmentation model: {e}")
                logger.error(f"Model path: {settings.models.segmentation_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")

    def _setup_tabs(self):
        """Initialize and add all tabs to the application."""
        self.main_tab = MainTab()
        self.addTab(self.main_tab, "Main")
        
        self.dev_video_preprocessing_tab = DevVideoPreprocessingTab()
        self.addTab(self.dev_video_preprocessing_tab, "Dev-Video Preprocessing")
        
        self.dev_yolo_training_tab = DevYoloTrainingTab()
        self.addTab(self.dev_yolo_training_tab, "Dev-YOLO Training")
        
        self.easyocr_tuning_tab = DevEasyOCRTuningTab()
        self.addTab(self.easyocr_tuning_tab, "EasyOCR Tuning")
        
        # Add more tabs as needed

    def _setup_dark_mode(self):
        """Apply dark mode theme to the application."""
        app = QApplication.instance()
        if app is None:
            return
            
        # Set dark palette
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(dark_palette)
        
        # Set custom stylesheet
        app.setStyleSheet("""
            QToolTip { color: #ffffff; background-color: #2a2a2a; border: 1px solid white; }
            QWidget { background-color: #232323; color: #f0f0f0; }
            QLineEdit, QTextEdit, QPlainTextEdit { background-color: #1e1e1e; color: #f0f0f0; }
            QMenuBar, QMenu, QTabBar::tab { background-color: #232323; color: #f0f0f0; }
            QTabWidget::pane { border: 1px solid #444444; }
            QPushButton { background-color: #333333; color: #f0f0f0; border: 1px solid #444444; }
            QPushButton:pressed { background-color: #222222; }
            QComboBox, QSpinBox, QSlider, QListWidget, QTableWidget, QLabel, QCheckBox {
                background-color: #232323; color: #f0f0f0;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #232323; width: 12px; margin: 0px;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #444444; min-height: 20px; border-radius: 4px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background: none;
            }
            QTableWidget, QTableView, QHeaderView {
                background-color: #232323; color: #f0f0f0;
                gridline-color: #444444;
                selection-background-color: #2a2a2a;
                selection-color: #ffffff;
                border: 1px solid #444444;
            }
            QTableWidget QTableCornerButton::section, QTableView QTableCornerButton::section {
                background-color: #232323;
                border: 1px solid #444444;
            }
            QHeaderView::section {
                background-color: #232323;
                color: #f0f0f0;
                border: 1px solid #444444;
            }
        """)
