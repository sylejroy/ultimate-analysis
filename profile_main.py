#!/usr/bin/env python3
"""
Profile the main Ultimate Analysis application using cProfile.

This script profiles the execution of the main application function,
handling the sys.exit() call gracefully.
"""

import cProfile
import sys
import os

# Add src directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from ultimate_analysis.gui.main_app import main

def profile_main():
    """Profile the main application function."""
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        main()
    except SystemExit:
        # Handle the sys.exit() from the GUI app gracefully
        pass

    profiler.disable()
    profiler.dump_stats('profile_output.prof')
    print("Profiling complete. Profile data saved to 'profile_output.prof'")
    print("Run 'visualize_profile.py' to view the results with snakeviz.")

if __name__ == "__main__":
    profile_main()