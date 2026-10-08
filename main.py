"""Main entry point for Ultimate Analysis application."""

import faulthandler
import os
import sys
from pathlib import Path


def main() -> None:
    """Launch the application."""
    root = Path(__file__).resolve().parent
    # Config, model, and video paths are relative to the repository root
    os.chdir(root)

    # Ultralytics installs packages by itself when it misses one, and has replaced the
    # CUDA build of PyTorch that way
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    # A crash inside a native library (TensorRT, CUDA) leaves the Python stacks of all
    # threads here; without this there is no trace of where it happened
    crash_log = root / "data" / "cache" / "crash_traces.log"
    crash_log.parent.mkdir(parents=True, exist_ok=True)
    faulthandler.enable(file=open(crash_log, "a", encoding="utf-8"), all_threads=True)

    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from ultimate_analysis.gui.main_app import main as run_application

    run_application()


if __name__ == "__main__":
    main()
