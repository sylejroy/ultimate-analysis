# Ultimate Analysis

PyQt5 desktop app that analyses Ultimate Frisbee video: player and disc detection (YOLO),
tracking (DeepSORT), possession, jersey numbers, field segmentation, and a top-down view.

Layout, code style, and how a frame flows through the app are in
`docs/DEVELOPMENT_GUIDELINES.md`. The README has the pipeline overview;
`docs/MEASUREMENTS.md` has the measured accuracy and speed of every model. Read those instead of guessing; this file only holds
what they do not say.

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
- PyTorch must stay at `2.7.1+cu128` and setuptools below 81. Newer setuptools removes
  `pkg_resources`, and DeepSORT then fails to load without an error: tracking silently
  falls back to a much worse tracker.
- Never let Ultralytics install packages on its own (`YOLO_AUTOINSTALL=false`); it has
  replaced the CUDA build of PyTorch before. Install with `--no-deps` when in doubt.

## Rules

- Nothing under `data/` is in Git. Never delete or overwrite anything there without asking;
  write new datasets and runs to new folders.
- Start model training through the app's Model Training tab so the progress is visible,
  not headless from a script. Dataset building, benchmarks, and TensorRT export are fine
  from the terminal.
- Commit only when asked.
- Source files use Windows line endings (CRLF); keep them.

## Things that are easy to get wrong

- The Roboflow dataset exports are 16:9 frames stretched to a square. Detection datasets
  are converted back to 16:9 (`scripts/build_merged_detection_dataset.py`). The field
  segmentation model is trained on the stretched images, so the app stretches frames the
  same way before segmenting; padding instead costs outline accuracy.
- A model runs at the image size it was trained at, read from the `args.yaml` of its run.
- TensorRT engines sit next to the weights and are tied to the model, frame size, GPU, and
  driver. A new or retrained model needs `scripts/export_tensorrt.py`, otherwise it runs
  in PyTorch, about three times slower.
- Tracking, possession, and jersey numbers build on earlier frames. Feed the pipeline
  frames in order and reset it after a seek.
- Qt's offscreen mode (`QT_QPA_PLATFORM=offscreen`) is fine for tests but renders no text;
  screenshots need a real window.
- Timings measured while a training run uses the GPU are meaningless.

## Measuring

Change a model, a threshold, or preprocessing only with a before and after number from
the benchmark scripts listed under "Testing" in `docs/DEVELOPMENT_GUIDELINES.md`. The
validation and test sets are small (98 discs, 48 field images), so small differences are
noise. Record results in the tables of `docs/MEASUREMENTS.md`.
