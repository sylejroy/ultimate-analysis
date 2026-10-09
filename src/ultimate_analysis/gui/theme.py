"""The application's dark theme."""

from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QWidget

# The colours everything is drawn in. Three greys from the window's ground to what lies
# on it, two greys of text, one colour that marks what is chosen or can be pressed.
GROUND = "#1e1f22"  # The window, and what pictures are shown on
PANEL = "#26282c"  # Groups of controls
CONTROL = "#30333a"  # Buttons, fields, lists
CONTROL_HOVER = "#3a3e46"
EDGE = "#3d4148"  # Lines around panels and controls
TEXT = "#e6e7ea"
FAINT_TEXT = "#9aa0aa"
ACCENT = "#3b9eff"
ACCENT_HOVER = "#62b1ff"
ACCENT_TEXT = "#0b1420"  # Text on the accent colour

STYLE = f"""
    QMainWindow, QDialog {{
        background-color: {GROUND};
        color: {TEXT};
    }}

    QWidget {{
        color: {TEXT};
    }}

    QLabel {{
        background-color: transparent;
    }}

    QTabWidget::pane {{
        border: none;
        border-top: 1px solid {EDGE};
        background-color: {GROUND};
    }}

    QTabWidget::tab-bar {{
        alignment: center;
    }}

    QTabBar::tab {{
        background-color: transparent;
        color: {FAINT_TEXT};
        padding: 9px 22px;
        margin: 0 2px;
        min-width: 110px;
        border-bottom: 2px solid transparent;
        font-weight: 600;
    }}

    QTabBar::tab:selected {{
        color: {TEXT};
        border-bottom: 2px solid {ACCENT};
    }}

    QTabBar::tab:hover:!selected {{
        color: {TEXT};
        border-bottom: 2px solid {EDGE};
    }}

    QGroupBox {{
        background-color: {PANEL};
        border: 1px solid {EDGE};
        border-radius: 8px;
        margin-top: 16px;
        padding: 12px 8px 8px 8px;
        font-weight: 600;
    }}

    QGroupBox:flat {{
        background-color: transparent;
        border: none;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: {FAINT_TEXT};
    }}

    QGroupBox::indicator {{
        width: 14px;
        height: 14px;
        border-radius: 3px;
        border: 1px solid {EDGE};
        background-color: {CONTROL};
    }}

    QGroupBox::indicator:checked {{
        background-color: {ACCENT};
        border: 1px solid {ACCENT};
    }}

    QPushButton {{
        background-color: {CONTROL};
        border: 1px solid {EDGE};
        padding: 6px 14px;
        border-radius: 6px;
        color: {TEXT};
    }}

    QPushButton:hover {{
        background-color: {CONTROL_HOVER};
        border: 1px solid {ACCENT};
    }}

    QPushButton:pressed {{
        background-color: {GROUND};
    }}

    QPushButton:disabled {{
        color: {FAINT_TEXT};
        background-color: {PANEL};
        border: 1px solid {EDGE};
    }}

    QPushButton:checked, QPushButton[primary="true"] {{
        background-color: {ACCENT};
        border: 1px solid {ACCENT};
        color: {ACCENT_TEXT};
        font-weight: 600;
    }}

    QPushButton[primary="true"]:hover {{
        background-color: {ACCENT_HOVER};
        border: 1px solid {ACCENT_HOVER};
    }}

    QCheckBox, QRadioButton {{
        spacing: 8px;
        padding: 2px 0;
        font-weight: normal;
    }}

    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        background-color: {CONTROL};
        border: 1px solid {EDGE};
    }}

    QCheckBox::indicator {{
        border-radius: 4px;
    }}

    QRadioButton::indicator {{
        border-radius: 8px;
    }}

    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border: 1px solid {ACCENT};
    }}

    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {ACCENT};
        border: 1px solid {ACCENT};
    }}

    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background-color: {CONTROL};
        border: 1px solid {EDGE};
        padding: 4px 8px;
        min-height: 20px;
        border-radius: 6px;
        color: {TEXT};
        font-weight: normal;
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_TEXT};
    }}

    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
        border: 1px solid {FAINT_TEXT};
    }}

    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border: 1px solid {ACCENT};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 22px;
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {FAINT_TEXT};
        margin-right: 8px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {CONTROL};
        border: 1px solid {EDGE};
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_TEXT};
        outline: none;
    }}

    QSlider::groove:horizontal {{
        height: 4px;
        background: {CONTROL};
        border-radius: 2px;
    }}

    QSlider::sub-page:horizontal {{
        background: {ACCENT};
        border-radius: 2px;
    }}

    QSlider::handle:horizontal {{
        background: {TEXT};
        width: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}

    QSlider::handle:horizontal:hover {{
        background: {ACCENT_HOVER};
    }}

    QTreeWidget {{
        alternate-background-color: {PANEL};
    }}

    QListWidget, QTextEdit, QPlainTextEdit, QTableWidget, QTreeWidget {{
        background-color: {GROUND};
        border: 1px solid {EDGE};
        border-radius: 6px;
        color: {TEXT};
        font-weight: normal;
        outline: none;
    }}

    QListWidget::item {{
        padding: 6px 8px;
        border-radius: 4px;
        margin: 1px 2px;
    }}

    QListWidget::item:selected {{
        background-color: {ACCENT};
        color: {ACCENT_TEXT};
    }}

    QListWidget::item:hover:!selected {{
        background-color: {CONTROL};
    }}

    QHeaderView::section {{
        background-color: {PANEL};
        color: {FAINT_TEXT};
        border: none;
        border-bottom: 1px solid {EDGE};
        padding: 4px 8px;
    }}

    QProgressBar {{
        background-color: {CONTROL};
        border: none;
        border-radius: 4px;
        text-align: center;
        color: {TEXT};
        min-height: 8px;
    }}

    QProgressBar::chunk {{
        background-color: {ACCENT};
        border-radius: 4px;
    }}

    QStatusBar {{
        background-color: {GROUND};
        color: {FAINT_TEXT};
        border-top: 1px solid {EDGE};
    }}

    QToolTip {{
        background-color: {PANEL};
        color: {TEXT};
        border: 1px solid {EDGE};
        padding: 6px 8px;
        border-radius: 6px;
    }}

    QScrollArea {{
        background-color: transparent;
        border: none;
    }}

    QScrollBar:vertical, QScrollBar:horizontal {{
        background: transparent;
        width: 10px;
        height: 10px;
        margin: 0;
    }}

    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background-color: {EDGE};
        border-radius: 5px;
        min-height: 24px;
        min-width: 24px;
    }}

    QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
        background-color: {FAINT_TEXT};
    }}

    QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
        border: none;
        background: none;
        width: 0;
        height: 0;
    }}

    QSplitter::handle {{
        background-color: {GROUND};
    }}

    QSplitter::handle:hover {{
        background-color: {EDGE};
    }}
"""

# What a picture (the video, the top-down view) is shown on
PICTURE_FRAME = f"""
    QScrollArea {{
        border: 1px solid {EDGE};
        border-radius: 8px;
        background-color: #111214;
    }}
"""
PICTURE_PLACEHOLDER = f"""
    QLabel {{
        background-color: #111214;
        color: {FAINT_TEXT};
        font-size: 14px;
    }}
"""


def apply_dark_theme(window: QWidget) -> None:
    """Apply the dark palette and stylesheet to a window and everything in it."""
    palette = QPalette()
    for role, colour in (
        (QPalette.Window, GROUND),
        (QPalette.WindowText, TEXT),
        (QPalette.Base, GROUND),
        (QPalette.AlternateBase, PANEL),
        (QPalette.ToolTipBase, PANEL),
        (QPalette.ToolTipText, TEXT),
        (QPalette.Text, TEXT),
        (QPalette.Button, CONTROL),
        (QPalette.ButtonText, TEXT),
        (QPalette.BrightText, "#ff5c5c"),
        (QPalette.Link, ACCENT),
        (QPalette.Highlight, ACCENT),
        (QPalette.HighlightedText, ACCENT_TEXT),
    ):
        palette.setColor(role, QColor(colour))
    window.setPalette(palette)

    font = QFont("Segoe UI")
    font.setPointSizeF(9.5)
    window.setFont(font)
    window.setStyleSheet(STYLE)
