"""Main application window for Ultimate Analysis.

This module contains the main PyQt5 application with tabbed interface
for video analysis functionality.
"""

import sys
from typing import Callable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config.settings import get_setting
from ..constants import (
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
)
from ..utils.logger import get_logger
from .main.main_tab import MainTab
from .theme import apply_dark_theme

logger = get_logger("APP")


class LazyLoadingTab(QWidget):
    """Wrapper widget for lazy loading tabs."""

    def __init__(self, tab_factory: Callable[[], QWidget], tab_name: str):
        super().__init__()
        self.tab_factory = tab_factory
        self.tab_name = tab_name
        self.actual_tab: Optional[QWidget] = None
        self._is_loaded = False

        # Create placeholder layout
        layout = QVBoxLayout()
        self.placeholder_label = QLabel(f"Loading {tab_name}...")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet(
            """
            QLabel {
                color: #888;
                font-size: 16px;
                font-style: italic;
            }
        """
        )
        layout.addWidget(self.placeholder_label)
        self.setLayout(layout)

    def load_actual_tab(self):
        """Load the actual tab content when first accessed."""
        if self._is_loaded:
            return

        logger.debug(f"Lazy loading {self.tab_name} tab...")

        try:
            # Create the actual tab
            self.actual_tab = self.tab_factory()

            # Replace placeholder with actual content
            layout = self.layout()
            layout.removeWidget(self.placeholder_label)
            self.placeholder_label.hide()
            self.placeholder_label.deleteLater()
            layout.addWidget(self.actual_tab)

            self._is_loaded = True
            logger.info(f"{self.tab_name} tab loaded successfully")

        except Exception as e:
            logger.error(f"Error loading {self.tab_name} tab: {e}")
            # Show error message instead of placeholder
            error_label = QLabel(f"Error loading {self.tab_name}: {str(e)}")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet(
                """
                QLabel {
                    color: #ff4444;
                    font-size: 14px;
                }
            """
            )
            layout = self.layout()
            layout.removeWidget(self.placeholder_label)
            self.placeholder_label.deleteLater()
            layout.addWidget(error_label)

    def is_loaded(self) -> bool:
        """Check if the actual tab has been loaded."""
        return self._is_loaded

    def get_actual_tab(self) -> Optional[QWidget]:
        """Get the actual tab widget if loaded."""
        return self.actual_tab


class UltimateAnalysisApp(QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self):
        super().__init__()

        # Application state

        # Tab references for lazy loading
        self.main_tab: Optional[MainTab] = None
        self.easyocr_tab: Optional[LazyLoadingTab] = None
        self.model_training_tab: Optional[LazyLoadingTab] = None
        self.homography_tab: Optional[LazyLoadingTab] = None
        self.labelling_tab: Optional[LazyLoadingTab] = None

        # Initialize UI
        self._init_ui()
        apply_dark_theme(self)

        logger.info("Ultimate Analysis application initialized")

    def _create_easyocr_tab(self) -> QWidget:
        """Factory method to create EasyOCR tuning tab."""
        from .easyocr.easyocr_tab import EasyOCRTuningTab

        return EasyOCRTuningTab()

    def _create_model_training_tab(self) -> QWidget:
        """Factory method to create model training tab."""
        from .training.training_tab import ModelTrainingTab

        return ModelTrainingTab()

    def _create_homography_tab(self) -> QWidget:
        """Factory method to create homography tab."""
        from .homography.homography_tab import HomographyTab

        return HomographyTab()

    def _create_labelling_tab(self) -> QWidget:
        """Factory method to create labelling tab."""
        from .labelling.labelling_modes import LabellingModes

        return LabellingModes()

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
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tab_widget)

        # Create main tab (always loaded immediately)
        self.main_tab = MainTab()
        self.main_tab.video_changed.connect(self._on_video_changed)
        self.tab_widget.addTab(self.main_tab, "Main Analysis")

        # Create lazy loading tabs
        # After the analysis itself, in the order the work is done: label frames, train
        # models on them, calibrate the top-down view, tune the jersey number reading
        self.labelling_tab = LazyLoadingTab(self._create_labelling_tab, "Labelling")
        self.tab_widget.addTab(self.labelling_tab, "Labelling")

        self.model_training_tab = LazyLoadingTab(self._create_model_training_tab, "Model Training")
        self.tab_widget.addTab(self.model_training_tab, "Model Training")

        self.homography_tab = LazyLoadingTab(self._create_homography_tab, "Field Calibration")
        self.tab_widget.addTab(self.homography_tab, "Field Calibration")

        self.easyocr_tab = LazyLoadingTab(self._create_easyocr_tab, "Jersey Number Tuning")
        self.tab_widget.addTab(self.easyocr_tab, "Jersey Number Tuning")

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        logger.info("UI initialized with main tab and lazy loading tabs")

    def _on_tab_changed(self, index: int):
        """Handle tab change to trigger lazy loading."""
        current_widget = self.tab_widget.widget(index)

        # If it's a lazy loading tab that hasn't been loaded yet, load it
        if isinstance(current_widget, LazyLoadingTab) and not current_widget.is_loaded():
            current_widget.load_actual_tab()

    def _on_video_changed(self, video_path: str):
        """Handle video change from main tab.

        Args:
            video_path: Path to the newly loaded video
        """

        # Update window title
        import os

        video_name = os.path.basename(video_path)
        app_name = get_setting("app.name", "Ultimate Analysis")
        self.setWindowTitle(f"{app_name} - {video_name}")

        # Update status bar
        self.status_bar.showMessage(f"Loaded: {video_name}")

        logger.info(f"Video changed to: {video_name}")

    def closeEvent(self, event):
        """Handle application close event."""
        logger.info("Application closing...")

        # The window disappears at once; stopping the worker threads can take a moment
        self.hide()

        # Cleanup main tab
        if hasattr(self, "main_tab") and self.main_tab:
            self.main_tab.close()

        # Cleanup lazy loading tabs if they were loaded
        for tab_attr in ["easyocr_tab", "model_training_tab", "homography_tab", "labelling_tab"]:
            if hasattr(self, tab_attr):
                tab = getattr(self, tab_attr)
                if isinstance(tab, LazyLoadingTab) and tab.is_loaded():
                    actual_tab = tab.get_actual_tab()
                    if actual_tab and hasattr(actual_tab, "close"):
                        actual_tab.close()

        # Accept the close event
        event.accept()
        logger.info("Application closed")


def create_application() -> QApplication:
    """Create and configure the PyQt5 QApplication.

    Returns:
        Configured QApplication instance
    """
    # On a scaled display (e.g. 150% on a 4K screen) Qt lays the interface out in scaled
    # units and draws it sharply. This only takes effect when set before the application
    # object exists; otherwise the window is either tiny or blurred by Windows.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName(get_setting("app.name", "Ultimate Analysis"))
    app.setApplicationVersion(get_setting("app.version", "0.1.0"))
    app.setOrganizationName("Ultimate Analysis Team")

    return app


def main():
    """Main entry point for the Ultimate Analysis application."""
    logger.info("Starting Ultimate Analysis...")

    # Create application
    app = create_application()

    # Create main window
    main_window = UltimateAnalysisApp()
    main_window.show()

    logger.info("Application started, entering event loop")

    # Run event loop
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        logger.warning("Application interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
