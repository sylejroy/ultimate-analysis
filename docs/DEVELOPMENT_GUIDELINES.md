# Ultimate Analysis — Development Guidelines

**Keep it simple.** Prefer the plainest solution that works, and readable code over
clever code.

## Layout

- `src/ultimate_analysis/gui/` — PyQt5 window, tabs, and drawing (`visualization.py`)
- `src/ultimate_analysis/processing/` — detection, tracking, jersey OCR, field
  segmentation, and line fitting
- `src/ultimate_analysis/optimization/` — genetic homography optimizer
- `src/ultimate_analysis/training/` — the subprocess the Model Training tab launches
- `src/ultimate_analysis/config/` — `get_setting("dot.path", default)` over
  `configs/default.yaml`
- `configs/` — `default.yaml`, `easyocr_params.yaml`, `training.yaml`,
  `homography_params.yaml`
- `data/` — local videos, datasets, and trained models. Not tracked by Git, so
  anything deleted there is gone; leave it alone during cleanups.
- `tests/` — regression tests on synthetic frames and mocked models

## Code

- Formatting: `black` and `ruff`, 100-character lines (settings in `pyproject.toml`).
- Type hints on function signatures; short docstrings.
- Tunable values go in `configs/default.yaml` and are read with `get_setting()`.
  Fixed limits and fallbacks go in `constants.py`.
- Keep new modules small (around 500 lines). The four GUI tabs are far above that;
  do not grow them, and move logic into `processing/` or helpers when touching them.
- Remove a setting, option, or function when its last use goes away.

## Performance

The pipeline runs on the UI thread, so per-frame cost is felt directly.

- Load models once and reuse them.
- Cache anything that only changes when its input changes (field mask, contour,
  line fit), and draw in place on frames the caller owns.
- Run expensive stages on an interval (segmentation, OCR) and reuse the last result.
- Measure before and after; record the numbers, not an estimate.

## Workflow

- For a new algorithm, agree on the approach in plain words or pseudocode before
  writing code.
- Start model training from the app's Model Training tab so progress is visible.
- Before finishing a change:

  ```bash
  python -m ruff check src tests
  python -m unittest discover -s tests -v
  ```
