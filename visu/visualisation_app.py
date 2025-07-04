
import sys
import os
import logging
from PyQt5.QtWidgets import QApplication, QTabWidget, QLabel
from PyQt5.QtGui import QPalette, QColor, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt

# Ensure the parent directory is in sys.path for relative imports to work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Also add the current directory (visu) to the path for local imports
sys.path.append(os.path.dirname(__file__))

from main_tab import MainTab
from dev_video_preprocessing_tab import DevVideoPreprocessingTab
from dev_yolo_training_tab import DevYoloTrainingTab
from easyocr_tuning_tab import DevEasyOCRTuningTab

logger = logging.getLogger("ultimate_analysis.visualisation_app")

class VisualizationApp(QTabWidget):
    """
    Main application window for Ultimate Analysis Visualization.
    Only GUI logic is present here.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Analysis Visualization")
        self.showMaximized()
        self.main_tab = MainTab()
        self.addTab(self.main_tab, "Main")
        self.dev_video_preprocessing_tab = DevVideoPreprocessingTab()
        self.addTab(self.dev_video_preprocessing_tab, "Dev-Video Preprocessing")
        self.dev_yolo_training_tab = DevYoloTrainingTab()
        self.addTab(self.dev_yolo_training_tab, "Dev-YOLO Training")
        self.easyocr_tuning_tab = DevEasyOCRTuningTab()
        self.addTab(self.easyocr_tuning_tab, "EasyOCR Tuning")
        # Add more tabs as needed

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # ---- DARK MODE ----
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
    # ---- END DARK MODE ----

    window = VisualizationApp()
    window.show()
    sys.exit(app.exec_())