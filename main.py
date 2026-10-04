"""Main entry point for Ultimate Analysis application."""

import sys
from pathlib import Path


def main() -> None:
    """Launch the application from the repository's source directory."""
    src_path = Path(__file__).resolve().parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from ultimate_analysis.gui.main_app import main as run_application

    run_application()


if __name__ == "__main__":
    main()
