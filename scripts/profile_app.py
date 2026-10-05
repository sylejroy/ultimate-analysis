#!/usr/bin/env python3
"""Profile a session of the application with cProfile.

Use the application as usual and close it; the profile is written to profile_output.prof
in the repository root. View it with scripts/view_profile.py.
"""

import cProfile
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from main import main  # noqa: E402

PROFILE_PATH = REPO / "profile_output.prof"


def profile_app() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        main()
    except SystemExit:
        # The GUI ends with sys.exit(); the profile is still written
        pass
    finally:
        profiler.disable()
        profiler.dump_stats(str(PROFILE_PATH))
        print(f"Profile data saved to '{PROFILE_PATH}'")
        print("Run 'python scripts/view_profile.py' to view it with snakeviz.")


if __name__ == "__main__":
    profile_app()
