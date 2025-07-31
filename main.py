"""Main entry point for Ultimate Analysis application."""

import sys
import os
from pathlib import Path

# Add the src directory to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from ultimate_analysis.gui.main_app import main

if __name__ == "__main__":
    main()
