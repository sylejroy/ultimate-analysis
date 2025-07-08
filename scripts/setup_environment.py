#!/usr/bin/env python3
"""
Environment setup script for Ultimate Analysis.

Sets up the development environment, creates necessary directories,
and downloads required models if needed.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up the development environment."""
    project_root = Path(__file__).parent.parent
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/models/detection",
        "data/models/segmentation",
        "data/models/player_id",
        "data/cache",
        "logs",
    ]
    
    print("Creating project directories...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("\nEnvironment setup completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Place your video files in input/dev_data/")
    print("3. Download/train models and place in finetune/ directories")
    print("4. Run the application: python src/ultimate_analysis/main.py")

if __name__ == "__main__":
    setup_environment()
