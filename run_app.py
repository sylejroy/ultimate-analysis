#!/usr/bin/env python3
"""Run the Ultimate Analysis GUI application."""

import sys
import os

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from PyQt5.QtWidgets import QApplication
from ultimate_analysis.gui.main_app import UltimateAnalysisApp

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = UltimateAnalysisApp()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
