#!/usr/bin/env python3
"""
Simple launcher script for Ultimate Analysis application.

This script sets up the environment and launches the application.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def main():
    """Launch the Ultimate Analysis application."""
    try:
        # Change to project directory to ensure relative paths work
        os.chdir(project_root)
        
        # Import and run the main application
        from ultimate_analysis.main import main as app_main
        
        # Parse command line arguments
        config_path = sys.argv[1] if len(sys.argv) > 1 else None
        
        # Run the application
        return app_main(config_path)
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        return 0
    except Exception as e:
        print(f"Failed to start application: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
