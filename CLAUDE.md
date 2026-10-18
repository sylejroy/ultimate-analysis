# Ultimate Analysis

PyQt5 desktop app that analyses Ultimate Frisbee video: player and disc detection (YOLO),
tracking (ByteTrack), possession, jersey numbers, field segmentation, and a top-down view.

The README says what the app does and how to run it. Keep checked-in documents short:
what the app does and how to use it, not how it works inside.

## Environment

- Run Python as `.venv/Scripts/python.exe` (Windows, Python 3.12). Entry point: `main.py`.
- Checks before finishing a change:

  ```bash
  python -m ruff check src tests scripts
  python -m ruff format --check src tests scripts
  python -m unittest discover -s tests
  ```

  Tests are `unittest` (pytest is not installed) and need no videos, weights, or GPU.
- `requirements-lock.txt` is the environment as it is known to work (`pip freeze`);
  update it when a package changes.
- PyTorch must stay at `2.7.1+cu128`. setuptools stays below 81 until it is checked that
  no package still imports `pkg_resources` (DeepSORT, which did, is gone).
- Never let Ultralytics install packages on its own (`YOLO_AUTOINSTALL=false`); it has
  replaced the CUDA build of PyTorch before. Install with `--no-deps` when in doubt.

## Rules

- Nothing under `data/` is in Git. Never delete or overwrite anything there without asking;
  write new datasets and runs to new folders.
- Start model training through the app's Model Training tab so the progress is visible,
  not headless from a script. Dataset building, benchmarks, and TensorRT export are fine
  from the terminal. The tab only trains YOLO models: the networks that tell players
  apart by their looks (`scripts/train_reid.py`), tell who holds the disc
  (`scripts/train_holder.py`) and draw the disc heat map (`scripts/train_disc_heatmap.py`)
  are trained from the terminal, with their progress printed per epoch.
- Commit only when asked.
- Source files use Windows line endings (CRLF); keep them.
