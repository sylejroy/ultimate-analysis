# Ultimate Analysis

A PyQt5 desktop application for analysing Ultimate Frisbee video with YOLO detection,
DeepSORT tracking, OCR-based player identification, and a top-down field view.

## Features

- **Object detection**: players and discs, with separately selectable models.
- **Tracking**: consistent player and disc identities across frames (DeepSORT), with
  trails and foot-level positions.
- **Player identification**: jersey numbers read with EasyOCR and aggregated over time,
  plus a tuning tab for the OCR and crop-preprocessing parameters.
- **Field segmentation**: field mask, contour, and RANSAC boundary lines.
- **Homography**: interactive perspective correction with a genetic-algorithm
  assistant; the result drives the top-down view in the main tab.
- **Model training**: train YOLO11/YOLO26 detection and segmentation models from the
  GUI with live output, progress, and metric plots against a baseline.
- **Performance monitoring**: per-stage timings while analysis runs.

## Quick Start

Requirements: Python 3.12 (the version it is developed on), a CUDA-capable GPU
(recommended), 8 GB RAM.

```bash
git clone <repository-url>
cd ultimate-analysis
python -m venv .venv
.venv\Scripts\activate  # Windows
python -m pip install -r requirements.txt
python main.py
```

1. Put videos in `data/raw/videos` and select one in the Main Analysis tab.
2. Choose the player, disc, and field-segmentation models.
3. Press play. Toggle detection, tracking, player ID, segmentation, and the top-down
   view independently.
4. Use the Homography tab to calibrate the top-down view, and the EasyOCR Tuning and
   Model Training tabs for the specialist workflows.

## Configuration

Configuration files are in `configs/`:

- `default.yaml` — application, model, tracking, segmentation, and homography settings
- `easyocr_params.yaml` — EasyOCR and crop-preprocessing parameters
- `training.yaml` — training defaults (model, dataset, epochs, image size, ...)
- `homography_params.yaml` — the saved default perspective transform

## Models and Data

Everything under `data/` is local and not tracked by Git.

- `data/models/pretrained/` — base weights; missing YOLO11/YOLO26 weights download
  automatically when selected for training.
- `data/models/detection/`, `data/models/segmentation/` — one folder per training run.
  The model dropdowns list each run's `weights/best.pt`.
- `data/raw/training_data/` — datasets in YOLO format.
  `scripts/build_merged_detection_dataset.py` builds the merged player + disc dataset
  at 1280×720 from the Roboflow exports.

## Development

See `docs/DEVELOPMENT_GUIDELINES.md` for layout and conventions, and
`docs/REBUILD_DESIGN_DOCUMENT.md` for a description of what each tab does.

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check src tests
python -m unittest discover -s tests -v
```

The tests use synthetic frames and mocked models, so they need no videos or weights.

Behaviour worth knowing when changing the pipeline:

- Redraws of the current frame reuse its processing results. Seeking or switching
  videos clears tracking, OCR identities, and cached segmentation.
- Field segmentation runs every few frames; the mask, contour, and line fit are
  cached until the next run.
- Selecting the same model for players and discs runs it once per frame and splits its
  detections by class. A model only reports the classes it is selected for.
- The disc model is skipped after a stretch with no disc and retried periodically
  (`models.disc_detection.skip_threshold` and `retry_interval`, 30 frames each).

### TensorRT engines (optional)

The detection and field segmentation models run about three times faster as TensorRT
engines. An engine is tied to one model, the video frame size, and this GPU and driver.

```bash
python scripts/export_tensorrt.py                 # default models, 1920x1080 video
python scripts/export_tensorrt.py path/to/weights/best.pt
```

Engines are stored next to the weights (`best.384x640.fp16.engine`). On the next start
the app uses an engine when one matches the model and frame size, and PyTorch otherwise;
`models.inference.tensorrt: false` switches engines off. Rebuild after a GPU driver or
TensorRT update.

Setting up a new environment for exporting needs care. The export uses `tensorrt-cu12`,
`onnx`, `onnxslim`, and `nvidia-modelopt`. Installing `nvidia-modelopt` with its
dependencies replaces the CUDA build of PyTorch and upgrades setuptools past the version
DeepSORT works with, after which the app silently falls back to the simple tracker.
Restore both afterwards:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 --no-deps
python -m pip install setuptools==70.2.0
```

The export script disables Ultralytics' automatic package installation for this reason.

To profile a GUI session, run `python profile_main.py`, close the application, then
run `python visualize_profile.py`. Both use `profile_output.prof` in the repository
root.

## License

This project declares GNU General Public License v3.0. A LICENSE file is not
currently included in the repository.
