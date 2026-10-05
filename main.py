"""Main entry point for Ultimate Analysis application."""

import os
import sys
from pathlib import Path


def main() -> None:
    """Launch the application."""
    root = Path(__file__).resolve().parent
    # Config, model, and video paths are relative to the repository root
    os.chdir(root)

    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from ultimate_analysis.gui.main_app import main as run_application

    run_application()


if __name__ == "__main__":
    main()
