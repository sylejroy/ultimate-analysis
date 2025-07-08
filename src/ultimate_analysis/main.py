#!/usr/bin/env python3
"""
Main entry point for Ultimate Analysis application.

This script initializes and runs the PyQt5 GUI application.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

# Add src directory to Python path for imports
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from ultimate_analysis.config.settings import get_settings
from ultimate_analysis.core.exceptions import UltimateAnalysisError
from ultimate_analysis.core.utils import setup_logging


def main(config_path: Optional[str] = None) -> int:
    """
    Main entry point for the Ultimate Analysis application.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Set up logging
        logger = setup_logging(level="INFO")
        logger.info("Starting Ultimate Analysis application")
        
        # Load settings
        settings = get_settings(config_path)
        logger.info(f"Settings loaded from: {config_path or 'default'}")
        
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("Ultimate Analysis")
        app.setApplicationVersion("0.1.0")
        
        # Import and create main window (after QApplication is created)
        try:
            from ultimate_analysis.gui.app import VisualizationApp
            
            window = VisualizationApp()
            window.show()
            
            logger.info("Application window created and shown")
            
            # Run the application
            exit_code = app.exec_()
            logger.info(f"Application exiting with code: {exit_code}")
            return exit_code
            
        except ImportError as e:
            logger.error(f"Failed to import GUI components: {e}")
            QMessageBox.critical(None, "Error", f"Could not import GUI application: {e}")
            return 1
            
    except UltimateAnalysisError as e:
        logger.error(f"Application error: {e.message}")
        if e.details:
            logger.error(f"Details: {e.details}")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    # Parse command line arguments if needed
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    exit_code = main(config_path)
    sys.exit(exit_code)
