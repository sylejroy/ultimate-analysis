"""Main entry point for Ultimate Analysis application."""

import sys
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ultimate_analysis.gui.main_window import MainWindow
from ultimate_analysis.config import get_setting


def setup_logging():
    """Setup application logging."""
    log_level = get_setting("logging.level", "INFO")
    log_format = get_setting("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Optionally enable file logging
    if get_setting("logging.file_enabled", True):
        log_file = get_setting("logging.file_path", "logs/ultimate_analysis.log")
        log_dir = Path(log_file).parent
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def main():
    """Main application entry point."""
    # Setup logging first
    setup_logging()
    logger = logging.getLogger("ultimate_analysis.main")
    
    logger.info("Starting Ultimate Analysis")
    
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName(get_setting("app.name", "Ultimate Analysis"))
    app.setApplicationVersion(get_setting("app.version", "0.1.0"))
    app.setOrganizationName("Ultimate Analysis Team")
    
    # Create and show main window
    try:
        window = MainWindow()
        window.show()
        
        logger.info("Application window created and shown")
        
        # Start event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
