# Ultimate Analysis — Rebuild Design Document

## 1. Purpose

Ultimate Analysis is a local desktop application for reviewing Ultimate Frisbee video with computer-vision assistance. Its primary job is to let an analyst load game footage, identify players and the disc, maintain identities over time, infer jersey numbers, understand the playable field, and inspect the action from a corrected top-down view.

This document is a feature-parity specification for recreating the product with a newer implementation stack or an LLM-assisted development workflow. It describes the current observable product behavior and the required semantic outcomes; it does **not** require retaining PyQt5, YOLO, DeepSORT, EasyOCR, global state, or the legacy module structure.

## 2. Product scope

### In scope

- Local analysis of a single Ultimate Frisbee video at a time.
- Interactive, frame-by-frame and playback-based visual analysis.
- Detection of players and discs, object tracking, and jersey-number identification.
- Field segmentation, field-line extraction, and perspective correction.
- Tuning and training workflows for the underlying vision models.
- Persistent YAML-based configuration and calibration files.
- Performance visibility while analysis is running.

### Explicitly not present in the current application

- Cloud upload, accounts, teams, collaboration, or permissions.
- A database, match library, tagging timeline, annotations, statistics dashboard, or report generation.
- Export of annotated video, CSV/JSON analysis results, or a formal API.
- Automated play/event recognition (throws, scores, turnovers, formations, etc.).
- Manual correction of a track ID or jersey identity in the GUI.

Those can be considered future additions, but should not be represented as feature parity.

## 3. Primary users and user goals

| User | Goal |
| --- | --- |
| Video analyst / coach | Inspect a game quickly, follow players and disc, and understand position relative to the field. |
| CV developer | Switch trained models, assess output quality, and profile the pipeline. |
| Dataset/model developer | Train a detection or field-segmentation model and compare its learning curves to a baseline. |
| Calibration user | Create and save a perspective transform that turns the camera view into a usable top-down field view. |

## 4. Functional requirements

### 4.1 Application shell

1. The application is a dark-themed desktop GUI, launched locally.
2. It has four top-level tabs:
   - Main Analysis
   - EasyOCR Tuning
   - Model Training
   - Homography Estimation
3. Main Analysis is ready at startup. The other three tabs may be initialized on first use to improve startup time.
4. The current video filename is reflected in the window/status context after a video loads.
5. The application must safely release video/model resources when closed.

### 4.2 Video discovery and playback

1. The app discovers compatible local videos from its configured video directory. Supported extensions are MP4, AVI, MOV, MKV, and WMV.
2. Each relevant tab exposes a refreshable video list. Selecting a video loads it into that tab's own video reader.
3. Video metadata used by the UI includes path, width, height, FPS, total frame count, current frame, and duration.
4. The analyst can:
   - view the current frame;
   - play/pause;
   - scrub to an arbitrary frame;
   - move to adjacent videos;
   - reset tracking state;
   - navigate with keyboard shortcuts where practical.
5. Frames must retain their original aspect ratio and be viewable in a scrollable/zoomable surface where the tab requires it.

### 4.3 Main Analysis tab

The main tab is the live video-analysis workspace. It contains a left configuration panel, a central processed-video view with playback controls, and an optional right-side top-down view.

#### Processing controls

The user can independently enable or disable:

- Object detection/inference.
- Object tracking.
- Player identification (jersey OCR).
- Field segmentation.
- Top-down (homography) view.
- Rendering of the segmentation overlay.
- RANSAC-based field-line fitting.

Dependencies must be clear in the new UI: player identification requires detected/tracked players; a useful top-down view requires a calibration matrix; field lines require segmentation output.

#### Model controls

1. The user can select distinct player-detection and disc-detection models. Each list shows only finished training runs whose dataset contains that class.
2. DeepSORT is the tracking method; there is no selector. A simple fallback is used
   when it is unavailable; it assigns fresh IDs rather than preserving identity.
3. The user can select the field-segmentation model and refresh the model list from the model directory.
4. The user can select the jersey-number reader: PARSeq with a text detector (default), Florence-2, a YOLO digit detector, or EasyOCR. A reader that cannot be loaded falls back to EasyOCR.

#### Main-view output

For each processed frame, render the original video plus enabled overlays:

- Player and disc bounding boxes.
- Detection class, confidence, and originating model category.
- Stable track IDs and per-track colours.
- Track trails/history and an approximate foot-level location.
- Best current jersey number and confidence, derived from multiple OCR observations rather than a single frame.
- Field segmentation mask/contour.
- Optional RANSAC field-boundary lines, inliers, outliers, and filtered edge points.
- A live FPS indicator and, when identification data exists, a jersey-number summary table overlay.

#### Top-down view

When enabled and a valid homography is available:

1. Warp the frame into the calibrated output canvas.
2. Transform tracked object positions into the same coordinate system.
3. Render the mapped player/disc positions and identities over the warped image.
4. Show an informative empty/error state if no video, frame, or transform is available.

### 4.4 Detection and tracking behavior

#### Object detection

1. Run separate models for players and discs on a BGR video frame.
2. Normalize each output to a common detection record:

```text
Detection {
  bbox: [left, top, right, bottom],
  confidence: number,
  classId: integer,
  className: "player" | "disc",
  modelType: "player_model" | "disc_model"
}
```

3. Make confidence threshold, NMS/IoU threshold, maximum detections, image size, and selected model configurable per model type.
4. Load models lazily and provide a warm-up operation so the first frame does not have a large cold-start delay.

#### Object tracking

1. Convert detections to persistent tracks that include ID, bounding box, confidence, class, confirmation state, and elapsed time since update.
2. Preserve track identity across nearby frames and short occlusions.
3. Keep a bounded history of visual positions for trails and later coordinate mapping.
4. Support reset on video change and on explicit user request.
5. Preserve DeepSORT-quality behavior in the default path; provide a simpler fallback tracker for environments where it is unavailable or for debugging.

### 4.5 Jersey-number identification

The product reads numerical jersey identifiers from player bounding boxes and aggregates uncertain readings over time.

1. Only player tracks are eligible for OCR.
2. Crop the upper portion of each eligible player box, reject undersized crops, and apply configurable preprocessing.
3. Preprocessing options presently include crop fraction, contrast/brightness, blur, denoising, CLAHE enhancement, sharpening, resizing/upscaling, colour/grayscale mode, and black-and-white mode.
4. OCR is constrained to a numeric allowlist by default and returns text, confidence, and digit-box information where available.
5. Maintain a probability distribution/history per track. Weight recent results more strongly and give a configurable bonus to readings near the centre of the player box.
6. Expose the best jersey number, confidence, measurement count, and the leading alternatives for each track internally; show the best result in the live overlay.
7. Mark an identity as finalized after a configurable certainty threshold, and skip future OCR work for that track unless state is reset.

### 4.6 EasyOCR Tuning tab

This is a diagnostic single-frame workspace—not a full tracking analysis.

1. Let the user select a video, player-detection model, and frame using a slider.
2. On “Run EasyOCR Analysis,” run player detection on that frame, crop detected players, preprocess each crop, and run OCR.
3. Display the source frame with detection/OCR annotations.
4. Display a grid of individual crops with their OCR result, confidence, and detected number.
5. Provide editable controls for the preprocessing and EasyOCR parameters described in §4.5.
6. Load settings from and save settings to `configs/easyocr_params.yaml`, preserving unrelated YAML keys.

### 4.7 Field segmentation and geometry

1. Run a selectable segmentation model on video frames to identify the playing field.
2. Filter by segmentation confidence and IoU thresholds; support model-input preprocessing and output-mask postprocessing.
3. Combine field-class masks into a unified binary field mask.
4. Support configurable morphology: opening, closing, and hole filling.
5. Extract a dominant field contour after filtering small areas and simplify it for drawing.
6. Optionally interpolate contour points, remove points near image edges, and fit several boundary-line segments using RANSAC.
7. Use the resulting mask, contour, and/or fitted lines as visual diagnostics and as input to homography calibration/optimization.

### 4.8 Homography Estimation tab

This tab calibrates a projective transformation from the camera image to a top-down field-oriented canvas.

1. Let the user select a video, scrub to a calibration frame, and choose a segmentation model.
2. Display original and warped frames side-by-side. Both views support zoom reset, fit-to-window, and an optional grid with configurable spacing.
3. Represent the transform as eight editable parameters (`H00`, `H01`, `H02`, `H10`, `H11`, `H12`, `H20`, `H21`); the ninth homogeneous element is fixed.
4. Support sliders and direct numeric entry; every change updates the preview.
5. Allow reset, save/load transform to a chosen YAML file, and save/load the default transform at `configs/homography_params.yaml`.
6. Saved metadata includes creation time, description, source video, frame index, application name, and version.
7. Run segmentation on the current calibration frame so the user can see field geometry used by calibration.
8. Provide a runtime-performance popup/table for segmentation, contour/line processing, and warping work.

#### Genetic-algorithm calibration assistant

1. Provide controls to start, reset, evolve one generation, evolve a batch, continuously evolve, stop, preview, and apply the current best transform.
2. Show generation number, best fitness, population size, and a fitness-over-time chart when the plotting dependency is available.
3. Score transforms using a configurable weighted combination of field-line alignment, field coverage, line visibility, field proportion, perspective distortion, and vertical/horizontal orientation balance.
4. Validate prerequisites (video/frame and usable field geometry) before starting optimization.

### 4.9 Model Training tab

The training tab manages Ultralytics-compatible detection or field-segmentation training.

1. Let the user choose task type: detection or field segmentation.
2. Discover compatible base model files (YOLO11 and YOLO26) and datasets from configured project locations; preselect the defaults saved in `configs/training.yaml`.
3. Show selected-model and selected-dataset information.
4. Expose training settings:
   - epochs, patience, batch size, learning rate, image size;
   - optimizer, momentum, weight decay, cosine learning-rate schedule, workers;
   - enable/disable augmentation plus mosaic, mixup, copy-paste, HSV hue/saturation/value controls.
5. Load and save training configuration YAML files.
6. Start training in a non-blocking subprocess/worker, stream raw output, update progress/status and elapsed time, and allow the user to stop training.
7. Discover the output results directory and plot training metrics from `results.csv` while training progresses or after it finishes.
8. Support baseline/reference results for visual comparison when a reference path is supplied.

### 4.10 Configuration and local data

Configuration is editable YAML. A rebuild should keep a stable, documented schema or include a migration layer.

| File | Responsibility |
| --- | --- |
| `configs/default.yaml` | App, video, models, tracking, OCR scheduling, segmentation, logging, homography, and GA defaults. |
| `configs/easyocr_params.yaml` | OCR engine and crop preprocessing parameters. |
| `configs/training.yaml` | Detection and segmentation training defaults. |
| `configs/homography_params.yaml` | Default transform and calibration metadata. |

Expected local folders are `data/raw/videos`, `data/models`, `data/raw/dataset`, `data/processed`, `logs`, `output`, `temp`, and `configs`. The current app discovers models from the local model tree and groups many model artifacts by training run.

## 5. Current processing flow

```text
Select video → load metadata / reset state / optionally warm models
    → decode frame
    → [player detector] + [disc detector]
    → normalize detections
    → [tracker] → persistent object tracks + trails
    → [OCR scheduler] → crop / preprocess / OCR / jersey probability tracker
    → [field segmenter] → unified mask → contour / optional RANSAC lines
    → compose source-frame overlays
    → [homography enabled] warp frame + map tracks → top-down overlay
    → render views + update timing/FPS
```

Each bracketed stage is optional according to the main-tab controls. The current implementation runs these stages serially, one frame at a time, on a worker thread for the Main Analysis tab; the tuning and calibration tabs still process their single frames on the GUI thread.

## 6. Performance baseline and redesign requirements

No reliable all-features baseline has been recorded; earlier figures in this repository were estimates, not measurements. Phase 0 of the phase plan measures one on the agreed clips and hardware. Actual performance is hardware, model, and footage dependent.

### Existing mitigations to preserve or improve

- Cache loaded models and warm them after video selection.
- Optionally use FP16 model inference on compatible GPUs.
- Run field segmentation every *N* frames and reuse the last result between runs.
- Reduce disc-model calls after an empty scene, but periodically retry so a disc
  entering the frame can be detected again.
- Run OCR only for scheduled tracks, stagger the schedule by track ID, and stop OCR for finalized jersey IDs.
- Bound track histories and periodic cleanup caches.
- Cache the current frame's processing results and field geometry (mask, contour,
  line fit) for repeat display updates; do not replay historical tracker/OCR state
  from a frame cache.
- Warp the top-down view at reduced scale (`homography.display_scale`).
- Lazy-load secondary GUI tabs.

### Rebuild performance requirements

1. The UI must remain responsive during decode, inference, OCR, segmentation, optimization, and training. No heavy compute may execute on the UI/event loop.
2. Split work into cancellable background stages with bounded queues. During playback, favour the newest frame over processing an obsolete backlog.
3. Share decoded/resized frame representations across models and avoid needless CPU↔GPU copies.
4. Cache results with keys that include video/frame identity, selected models, and processing parameters; invalidate correctly when any input changes.
5. Make scheduling policy explicit: interactive scrubbing needs the selected frame quickly; playback needs stable latency; batch/offline analysis may prioritize throughput.
6. Instrument every stage with latency, queue depth, FPS, dropped-frame count, GPU memory, CPU memory, and cache hit rate. Retain the current per-stage timing view.
7. Make every speed/accuracy trade-off visible and persistable in configuration.

### Suggested measurable acceptance targets

Targets should be re-baselined on the intended hardware and model set, but a practical starting point is:

| Scenario | Target |
| --- | --- |
| App startup to usable Main Analysis | under 3 seconds excluding one-time model downloads. |
| First processed frame after models are ready | under 500 ms. |
| Scrub preview using cached results | under 150 ms. |
| Playback with all features | maintain UI interaction and report true processed/displayed FPS; drop stale frames instead of accumulating lag. |
| Playback with detection + tracking only | aim for source-frame rate on the target GPU or document the attainable rate. |
| Long video session | bounded memory growth; no unbounded frame, track, crop, or result cache. |

## 7. Recommended modern architecture

This is a recommended implementation approach, not a requirement to use a specific language or framework.

```text
Desktop UI
  ↕ commands, immutable state snapshots, events
Application/controller layer
  ├─ Video session service (decode, seek, metadata)
  ├─ Pipeline scheduler (priorities, cancellation, latest-frame policy)
  ├─ Vision adapters (detection, tracking, OCR, segmentation)
  ├─ Geometry service (mask, lines, homography, GA)
  ├─ Training service (subprocess + log/result monitoring)
  ├─ Configuration repository (validated YAML + migrations)
  └─ Telemetry/cache service
       ↕
Local filesystem, GPU/accelerator runtime, model artifacts
```

Key boundaries:

- **UI:** rendering and user intent only; never owns inference state.
- **Vision adapters:** return typed, model-agnostic results so YOLO/EasyOCR/DeepSORT can be replaced independently.
- **Pipeline scheduler:** owns dependencies, work cancellation, quality mode, and cache invalidation.
- **Session state:** one explicit `VideoSession` per opened video/tab; avoid mutable module-global trackers/models that leak state across videos.
- **Persistence:** validate configuration on read, expose errors in the UI, and write atomically.

## 8. Core data contracts

Use typed schemas (for example Pydantic/JSON Schema/TypeScript types) and version them. At minimum:

```text
VideoInfo { path, width, height, fps, frameCount, durationSeconds }
FrameRef { videoId, frameIndex, timestampSeconds, image }
Detection { bbox, confidence, classId, className, modelType }
Track { id, bbox, confidence, className, confirmed, age, timeSinceUpdate, history }
JerseyIdentity { trackId, bestNumber?, confidence, alternatives, measurementCount, finalized }
FieldGeometry { mask?, contour?, lineSegments?, diagnostics }
HomographyCalibration { matrix, sourceVideo?, sourceFrame?, createdAt, description? }
StageTiming { stage, durationMs, cacheHit, frameIndex? }
AnalysisFrame { source, detections, tracks, jerseyIdentities, fieldGeometry, topDown?, timings }
```

Coordinates must document their reference frame (source image vs. warped canvas) and use one rectangle convention consistently. This prevents the legacy class of errors around crop/resize/OCR-box coordinate transforms.

## 9. Quality requirements

### Reliability

- A missing model, unsupported video, invalid YAML, no field mask, or unavailable GPU must generate a clear, recoverable UI state rather than crash the app.
- Model changes, video changes, and tracking resets must not reuse stale analysis results.
- Training failure or cancellation must leave the GUI usable and preserve logs/results created so far.

### Usability

- Display enabled features and unavailable dependencies clearly.
- Keep configuration controls grouped by purpose, with sensible defaults and a reset-to-default option.
- Surface progress for any operation longer than a moment: warm-up, model load, training, genetic optimization, and lengthy seek/decode.
- Keep both original and top-down views visually legible at high-DPI resolutions.

### Privacy and deployment

- The current product is fully local: videos, models, OCR results, and calibration files stay on the user's machine.
- Preserve that default. Any future remote/LLM feature must be opt-in and state exactly what data leaves the machine.

## 10. Rebuild priorities

**Implementation must follow the [incremental phase plan](REBUILD_PHASE_PLAN.md).**
The three releases below are product groupings, not single implementation tasks.
The phase plan splits them into runnable checkpoints, beginning with a baseline
and ending with an explicit cutover decision. After each phase, stop for the user's
hands-on review and acceptance before starting the next one. Keep the current app
runnable alongside the rebuild and preserve every file in `data/`.

### Release 1 — analyst workflow

- Video selection, playback, seeking, and responsive rendering.
- Player/disc detection, tracking, toggles, and overlays.
- Model selection and basic timing/FPS display.
- Validated configuration and robust failure states.

### Release 2 — field and identity quality

- Field segmentation/mask/contour/RANSAC visualization.
- OCR-based jersey identity with aggregation, scheduling, and tuning tab.
- Saved/loadable homography and integrated top-down track mapping.

### Release 3 — advanced developer workflows

- Homography calibration UI, runtime inspector, and genetic optimizer.
- Model-training UI with subprocess isolation, live logs, metrics, and baseline comparison.

This order gives users useful analysis much earlier while isolating expensive or specialized tools.

## 11. Verification checklist

Before calling a rebuild feature-complete, verify:

- A supported video loads, plays, pauses, seeks, and closes without resource leaks.
- Each processing toggle correctly changes computation and overlay output.
- Player and disc models can be switched independently and do not contaminate cached output.
- Track IDs remain stable on representative footage; reset and video change clear state.
- OCR shows numeric jersey results and converges/finalizes across frames.
- Field segmentation produces a mask, contour, and optional line diagnostics; cached-frame behavior is correct.
- A saved homography reopens and produces the same warped image; mapped tracks align to the transformed source.
- GA controls work and never freeze the UI.
- Training can start, stream output, stop, and show generated metrics.
- All heavy work is cancellable/backgrounded, and telemetry exposes the limiting stage.

## 12. Legacy implementation reference

The present code places the main GUI in `src/ultimate_analysis/gui/`, vision processing in `src/ultimate_analysis/processing/`, calibration optimization in `src/ultimate_analysis/optimization/`, and configuration under `configs/`. The most relevant legacy sources for implementation details are:

- `pipeline.py` — live analysis orchestration: one frame in, results and rendered views out.
- `gui/main/main_tab.py` and `gui/main/pipeline_worker.py` — the Main Analysis tab and the thread its pipeline runs on.
- `processing/inference.py`, `tracking.py`, `player_id.py`, `jersey_readers.py`, and `jersey_tracker.py` — analysis pipeline behavior.
- `processing/field_segmentation.py` and `field_analysis.py` — mask and line extraction.
- `rendering/` — overlays and the top-down view.
- `gui/homography/homography_tab.py`, `processing/homography.py`, and `optimization/homography_optimizer.py` — calibration tooling.
- `gui/easyocr/easyocr_tab.py` and `gui/training/training_tab.py` — specialist workflows.
