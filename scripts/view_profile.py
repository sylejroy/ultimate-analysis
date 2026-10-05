#!/usr/bin/env python3
"""Open the profile written by scripts/profile_app.py in snakeviz."""

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parents[1] / "profile_output.prof"


def view_profile() -> None:
    if not PROFILE_PATH.is_file():
        sys.exit(f"Profile file '{PROFILE_PATH}' not found. Run scripts/profile_app.py first.")
    if find_spec("snakeviz") is None:
        sys.exit("snakeviz is not installed. Install it with: python -m pip install snakeviz")

    print(f"Opening profile data in snakeviz: {PROFILE_PATH}")
    try:
        subprocess.run([sys.executable, "-m", "snakeviz", str(PROFILE_PATH)], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error running snakeviz: {e}")


if __name__ == "__main__":
    view_profile()
