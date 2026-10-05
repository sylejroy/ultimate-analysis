# Ultimate Analysis

A PyQt5 desktop application for analysing Ultimate Frisbee video with YOLO detection,
DeepSORT tracking, OCR-based player identification, and a top-down field view.

## Features

- **Object detection**: players and discs, with separately selectable models.
- **Tracking**: consistent player and disc identities across frames (DeepSORT), with
  trails and foot-level positions.
- **Possession**: the player holding the disc is highlighted in both views.
- **Player identification**: jersey numbers read by a selectable reader (PARSeq,
  Florence-2, a YOLO digit detector, or EasyOCR) and aggregated over time, plus a tuning
  tab for the EasyOCR and crop-preprocessing parameters.
- **Field segmentation**: field mask, contour, and RANSAC boundary lines.
- **Homography**: interactive perspective correction with a genetic-algorithm
  assistant; the result drives the top-down view in the main tab.
- **Model training**: train YOLO11/YOLO26 detection and segmentation models from the
  GUI with live output, progress, and metric plots against a baseline.
- **Performance monitoring**: per-stage timings while analysis runs.

## Screenshots

**Main Analysis** — detection, tracking, jersey numbers, the disc holder (gold box), the
field outline, and the top-down view:

![Main Analysis tab](docs/gui_example_main_analysis.png)

**Model Training** — live output and metric plots:

![Model Training tab](docs/gui_example_model_training.png)

**Homography Estimation** — manual calibration and the genetic-algorithm assistant:

![Homography Estimation tab](docs/gui_example_homography.png)

**EasyOCR Tuning** — crop preprocessing and reader parameters on single frames:

![EasyOCR Tuning tab](docs/gui_example_ocr_tuning.png)

## Pipeline

Each frame of the Main Analysis tab goes through these stages, every one of which can be
switched off:

1. **Detection** — one YOLO26s model for players and one for discs, both at image size
   1280 and run as TensorRT engines when built. The disc model is skipped while no disc
   has been seen for a while.
2. **Tracking** — DeepSORT gives players and the disc stable IDs and trails.
3. **Possession** — the player whose box holds the detected disc, confirmed over several
   frames.
4. **Jersey numbers** — a few tracks per frame are read (PARSeq by default) and the
   readings are accumulated per track.
5. **Field segmentation** — every fifth frame: field mask, outline, and RANSAC boundary
   lines.
6. **Rendering** — overlays on the camera view, and the frame warped to the top-down view
   with players mapped by their foot positions.

The pipeline runs on a worker thread at about 20 frames per second on an RTX 5060 Ti with
all stages on, and the window stays responsive meanwhile.

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
  at 1280×720 from the Roboflow exports, and `scripts/build_single_class_dataset.py`
  copies a dataset with the labels of one class, for a detector of discs or players only.

## Development

See `docs/DEVELOPMENT_GUIDELINES.md` for layout and conventions, and
`docs/REBUILD_DESIGN_DOCUMENT.md` for a description of what each tab does.

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m unittest discover -s tests -v
```

The tests use synthetic frames and mocked models, so they need no videos or weights.

The code under `src/ultimate_analysis/` is layered: `processing/` holds the analysis
stages, `rendering/` draws their results with OpenCV, `pipeline.py` combines both into
one call per frame, and `gui/` holds everything Qt, with one package per tab. The Main
Analysis tab runs the pipeline on a worker thread, so the window stays responsive while
a frame is processed.

Behaviour worth knowing when changing the pipeline:

- Redraws of the current frame reuse its processing results. Seeking or switching
  videos clears tracking, OCR identities, and cached segmentation.
- Field segmentation runs every few frames; the mask, contour, and line fit are
  cached until the next run.
- Selecting the same model for players and discs runs it once per frame and splits its
  detections by class. A model only reports the classes it is selected for.
- Possession goes to the player whose box contains the detected disc. The holder only
  changes after the disc has been seen at another player, or at no player, for
  `models.possession.confirm_frames` frames in a row (10), so a disc flying past someone
  does not change it. Frames without a detected disc leave the holder as it is.
- The disc model is skipped after a stretch with no disc and retried periodically
  (`models.disc_detection.skip_threshold` and `retry_interval`, 30 frames each).

### Detection models

Players and discs are detected by two separate models (the defaults in
`configs/default.yaml`). A dedicated disc model finds clearly more discs than one model
trained on both classes; for players it makes no difference. Selecting the same model in
both roles runs it once per frame instead of twice.

`scripts/benchmark_detectors.py` scores models on the validation and test images of the
default training dataset (141 images with 1,981 players and 98 discs) and times them on
video frames. A detection counts when it overlaps a labelled box of its class with
IoU ≥ 0.5; recall is at the app's confidence threshold (players 0.5, discs 0.3). All
models below were trained on the `merged.v2` data; times are with TensorRT engines unless
noted.

| Model | Player AP50 | Player recall | Disc AP50 | Disc recall | Time per frame |
| --- | --- | --- | --- | --- | --- |
| YOLO26s at 1280, players only (default player model) | 0.982 | 0.977 | | | 6.5 ms |
| YOLO26s at 1280, discs only (default disc model) | | | 0.505 | 0.449 | 5.3 ms |
| YOLO26s at 1280, both classes | 0.980 | 0.976 | 0.384 | 0.296 | 5.4 ms |
| RT-DETR-L at 960, both classes | 0.977 | 0.970 | 0.280 | 0.316 | 30.5 ms (PyTorch) |

The previous defaults (YOLO11s trained on the older, smaller datasets) scored 0.853 AP50
for players and 0.324 for discs.

A disc is about 11 pixels wide in a 1280×720 frame, so it needs the full image size and
is still the weak spot: the best model finds less than half of the discs.

To train a detector for one class, build a single-class copy of a dataset with
`scripts/build_single_class_dataset.py` and select it in the Model Training tab.

The tab also lists RT-DETR (`rtdetr-l.pt`, `rtdetr-x.pt`), a transformer detector, as an
alternative to YOLO. It stretches the frame to a square and runs in PyTorch only. On this
data it matched YOLO for players, was worse for discs (a quarter of its disc detections
at the app's threshold were right), and took twice as long to train.

### Field segmentation

The field lines are fitted to the outline of the predicted field. `scripts/benchmark_segmentation.py`
scores the segmentation model on its validation and test images (48): IoU of the whole
field and of each class, and the average distance between the predicted and the labelled
field outline in pixels of a 1920×1080 frame.

| | Field IoU | Central field IoU | End zone IoU | Outline error |
| --- | --- | --- | --- | --- |
| Default model (YOLO11s-seg) | 0.995 | 0.991 | 0.957 | 2.5 px |

The training images are 16:9 frames stretched to a square, so the app stretches each frame
the same way before segmenting it. Padding the frame to a square instead, as the app did
before, gave 0.983 field IoU, 0.892 for the end zones, and a 14 px outline error.

### Jersey number readers

The reader is chosen with "Jersey Number Reader" in the Main Analysis tab or
`models.player_id.method`. Measured on 1,016 hand-labelled crops of 23 players from four
games, plus crops of 14 players with no visible number:

| Reader | Players identified | Wrong | Correct / wrong reads per crop | Time per crop |
| --- | --- | --- | --- | --- |
| `parseq` (default) | 17 of 23 | 0 | 18.9% / 0.3% | 8 ms |
| `florence` | 19 of 23 | 1 | 21.3% / 1.6% | 38 ms |
| `easyocr` | 11 of 23 | 4 | 7.6% / 3.7% | 9 ms |
| `yolo_digits` | not measured: no digit model trained yet | | | |

Most crops show a player from the front or side, so a low per-crop rate is expected; what
counts is that the reads that do come are right. None of the readers reported a number
for the players without one.

- `parseq` runs the PARSeq recognizer on the regions EasyOCR's text detector finds. It is
  downloaded through `torch.hub` on first use.
- `florence` is the Florence-2 vision-language model. It answers for every crop, so reads
  below `models.player_id.florence.min_confidence` (0.7) are discarded.
- `yolo_digits` uses the newest detection run trained on a dataset with "digits" in its
  name (train one in the Model Training tab; `digits.v1i.yolov8` is house numbers, a rough
  starting point). Until one exists the app falls back to EasyOCR, as it does for any
  reader that fails to load.

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

### Profiling

Run `python scripts/profile_app.py`, use the application, and close it; then run
`python scripts/view_profile.py` to open the result in snakeviz. Both use
`profile_output.prof` in the repository root. The profile covers the GUI thread; the
Performance panel in the Main Analysis tab shows the time per pipeline stage.

## License

This project declares GNU General Public License v3.0. A LICENSE file is not
currently included in the repository.
