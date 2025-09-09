"""Main entry point for Ultimate Analysis application."""

import sys
import argparse
from pathlib import Path

# Add the src directory to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from ultimate_analysis.gui.main_app import main

def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Ultimate Analysis - AI-powered Ultimate Frisbee video analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Launch in realtime mode with only the main analysis tab (streamlined interface)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(realtime_mode=args.realtime)
