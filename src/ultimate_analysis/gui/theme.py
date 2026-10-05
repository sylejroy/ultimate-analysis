"""The application's dark theme."""

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QWidget


def apply_dark_theme(window: QWidget) -> None:
    """Apply the dark palette and stylesheet to a window and everything in it."""
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
    window.setPalette(dark_palette)

    # Additional stylesheet for enhanced dark theme
    window.setStyleSheet(
        """
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

        QCheckBox, QRadioButton {
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
            min-height: 18px;
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
            min-height: 18px;
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
            min-height: 18px;
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
    """
    )
