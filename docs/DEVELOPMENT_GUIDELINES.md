# Ultimate Analysis — Development Guidelines

**Keep it simple.** Prefer the plainest solution that works, and readable code over
clever code.

## Layout

The code is layered. A module only imports from its own layer and the ones above it in
this table; `tests/test_layering.py` fails when one does not.

| Package | Contents |
| --- | --- |
| `config/`, `constants.py`, `utils/` | Settings (`get_setting("dot.path", default)`), fixed limits, logging, video files, model files, label files |
| `processing/` | Analysis stages: detection, camera motion, tracking, possession, game state, jersey numbers, field segmentation and geometry, homography, TensorRT engines |
| `rendering/` | Drawing results on frames with OpenCV. No Qt. |
| `pipeline.py` | `AnalysisPipeline`: one frame in, results and rendered views out. No Qt. |
| `gui/` | Everything Qt. One package per tab (`main/`, `easyocr/`, `training/`, `homography/`, `labelling/`), shared widgets in `widgets/`, the window in `main_app.py` |
| `web/` | The phone labelling page and its server. No Qt. |
| `optimization/`, `training/` | Genetic homography optimizer; the training subprocess |

Outside `src/`:

- `configs/` — `default.yaml`, `easyocr_params.yaml`, `training.yaml`,
  `homography_params.yaml`
- `scripts/` — dataset building, TensorRT export, benchmarks, profiling
- `tests/` — one file per area; synthetic frames and mocked models
- `data/` — local videos, datasets, and trained models. Not tracked by Git, so anything
  deleted there is gone; leave it alone during cleanups.

## How a frame flows

1. The main tab asks its worker thread (`gui/main/pipeline_worker.py`) for a frame.
2. The worker decodes it and calls `AnalysisPipeline.process`, which runs the enabled
   stages from `processing/` and draws the views with `rendering/`.
3. The finished `FrameResult` goes back to the tab, which only converts it to pixmaps.

The worker owns the video reader and the pipeline; the tab never touches them directly
and sends every change (seek, model, reader) as a queued command.

### Shared models

The `processing` modules keep their loaded models in module-level variables: there is one
player model, one disc model, one field model, and one jersey reader for the whole
process. All tabs use them, from different threads, so every call into `processing` that
runs or swaps a model is made while holding `processing.model_lock.MODEL_LOCK`. The
pipeline worker does this for the main tab; the other tabs work on single frames from the
GUI thread and take the lock themselves. A tab that needs a different model than the one
the main tab plays with loads its own copy (`inference.load_detection_model`) instead of
swapping the shared one.

## Code

- Formatting: `ruff format` and `ruff check`, 100-character lines (settings in
  `pyproject.toml`).
- Type hints on function signatures; short docstrings.
- Logging: `logger = get_logger("TAG")` at the top of the module. No `print` in `src/`
  (the training subprocess is the exception; the GUI reads its output). Per-frame
  messages are `debug`.
- Tunable values go in `configs/default.yaml` and are read with `get_setting()`.
  Fixed limits and fallbacks go in `constants.py`.
- A tab module holds interface code only. Anything that computes or draws goes to
  `processing/`, `rendering/`, or a widget of its own.
- One home per helper: video discovery and reading in `utils/video.py`, model lookup in
  `utils/model_files.py`, homography parameters in `processing/homography.py`, crop
  preprocessing in `processing/jersey_crops.py`.
- Remove a setting, option, or function when its last use goes away.

Modules that are still too large and should be split when they are next worked on:
`gui/homography/homography_tab.py` (1,600 lines), `gui/easyocr/easyocr_tab.py` and
`gui/training/training_tab.py` (1,100 each), `processing/field_analysis.py` (800).

## Testing

- Logic that carries state from frame to frame gets a unit test with synthetic input:
  tracking, possession, jersey number bookkeeping, caches, the pipeline's result reuse.
  Models are mocked; the tests need no videos, weights, or GPU.
- How good a model is cannot be unit tested. Measure it with the benchmark scripts and
  record the result in the README:
  - `scripts/benchmark_detectors.py` — players and discs
  - `scripts/benchmark_segmentation.py` — field area and outline
  - `scripts/benchmark_field_registration.py` — where the field lies, against the labelled field frames
  - `scripts/benchmark_jersey_readers.py` — jersey numbers
  - `scripts/benchmark_player_id_scheduling.py` — temporal crop selection, votes and OCR work
  - `scripts/benchmark_homography_optimizer.py` — coverage sampling speed and candidate agreement
  - `scripts/benchmark_pipeline.py` — analysis/rendering throughput, stage costs and track-output comparison
- GUI code is checked by starting the app, visiting every tab, playing a video, and
  closing it without an error in the log.
- A bug that got through gets a test that would have caught it.

## Performance

- Load models once and reuse them.
- Cache anything that only changes when its input changes (field mask, contour, line
  fit), and draw in place on frames the caller owns.
- Run expensive stages on an interval (segmentation, OCR) and reuse the last result.
- Nothing slow on the GUI thread of the main tab.
- Measure before and after, with nothing else using the GPU; record the numbers, not an
  estimate. The Performance panel of the main tab shows the time per stage.
- Use unprofiled pipeline runs for throughput; cProfile is for locating costs and adds
  overhead. Warm up models first and compare saved outputs as well as frame times.

## Workflow

- For a new algorithm, agree on the approach in plain words or pseudocode before
  writing code.
- Start model training from the app's Model Training tab so progress is visible.
- Commit a finished change before starting the next one.
- Before finishing a change:

  ```bash
  python -m ruff check src tests scripts
  python -m ruff format --check src tests scripts
  python -m unittest discover -s tests -v
  ```
