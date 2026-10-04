#!/usr/bin/env python3
"""
Profile the main Ultimate Analysis application using cProfile.

This script profiles the execution of the main application function,
handling the sys.exit() call gracefully.
"""

import cProfile
from pathlib import Path

from main import main

PROFILE_PATH = Path(__file__).resolve().parent / "profile_output.prof"


def profile_main() -> None:
    """Profile the main application function."""
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        main()
    except SystemExit:
        # Handle the sys.exit() from the GUI app gracefully
        pass
    finally:
        profiler.disable()
        profiler.dump_stats(str(PROFILE_PATH))
        print(f"Profile data saved to '{PROFILE_PATH}'")
        print("Run 'visualize_profile.py' to view the results with snakeviz.")


if __name__ == "__main__":
    profile_main()
