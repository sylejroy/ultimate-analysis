# Rebuild phase plan

This is the delivery plan for [the rebuild design](REBUILD_DESIGN_DOCUMENT.md).
The design remains the feature specification; this plan defines the order of work
and the points where you inspect the application. No rebuild implementation starts
just because this plan exists.

## Delivery rules

- Implement **one phase at a time**. At its end, provide a runnable build, exact
  launch instructions, a short change list, test results, and known limitations.
- **Stop for your review after every phase.** Continue only when you explicitly
  accept that phase. Fix review findings within the same phase before advancing.
- Every phase adds a usable, visible workflow. Infrastructure work must support
  that phase's demonstration rather than grow into a separate framework project.
- Keep the current application runnable through its existing entry point. Give
  the rebuild a separate entry point and implementation location, selected in
  Phase 0. Keep dependencies isolated if the stacks conflict.
- Never delete files in `data/`. Use existing videos and models as read-only inputs.
  Tests use synthetic fixtures or a separate temporary directory. Training creates
  a new output directory and never replaces an existing run or dataset.
- Until final acceptance, save rebuild settings/calibrations separately. Reading
  legacy YAML is allowed; migration must operate on copies and preserve unknown
  keys. Do not silently rewrite the current app's configuration.
- Record a reproducible revision/checkpoint for each accepted phase. Returning to
  the previous accepted build must not require undoing data/config migrations.
- Do not remove legacy functionality before its replacement has been accepted.
  A feature scheduled for a later phase should be clearly marked as unavailable
  in the rebuild, rather than presented as working with fabricated output.

## Phase map

| Phase | What you can inspect at the end | Design coverage |
| --- | --- | --- |
| 0 | Current-app baseline and agreed rebuild approach | Sections 6–9 |
| 1 | New shell with reliable video playback | 4.1, 4.2, basic 4.10 |
| 2 | Responsive player/disc detection | Detection in 4.3–4.4 |
| 3 | Stable tracking and complete basic analyst workflow | Tracking in 4.3–4.4; Release 1 |
| 4 | Field masks, contours, and line diagnostics | 4.7 |
| 5 | Live jersey-number identification | 4.5 |
| 6 | OCR tuning workspace | 4.6 |
| 7 | Manual calibration and integrated top-down view | Manual 4.8; completes Release 2 |
| 8 | Genetic calibration assistant | GA in 4.8 |
| 9 | Training workspace | 4.9; completes Release 3 features |
| 10 | Full-workflow validation and optional cutover | Sections 9, 11 |

Phases run in this order by default. A phase may be split further if its demo is
too large to review comfortably. Reordering or expanding scope should be agreed
at a checkpoint, not silently bundled into implementation.

## Phase 0 — Establish a baseline and choose the implementation approach

**Deliver:** A short implementation decision record, a baseline report, and a
repeatable review script for the current application. Agree on the UI/runtime
stack, separate launch command, supported hardware, and configuration-copy policy.
Keep existing model engines initially unless there is a concrete reason to change
them; a GUI rebuild should not also require retraining models.

Choose a small set of representative clips: ordinary play, an empty scene followed
by a returning disc, overlapping players, and a camera movement or difficult field
view. Record model paths, image sizes, thresholds, enabled features, hardware, and
source FPS. Use the same inputs/settings for later comparisons.

**You check:** Run the current app through load, playback, pause, seek, video switch,
and close. Identify which outputs and interactions matter most to preserve. Review
the proposed rebuild stack and the definition of acceptable responsiveness.

**Evidence:** Separate cold startup/model loading from warm processing. Record
processed/displayed FPS, stage latency, and memory where measurable. Carry forward
the regression cases in `tests/`.

**Stop:** Approve the approach and baseline before implementing the new shell.

## Phase 1 — Runnable shell and video playback

**Deliver:** A separately launched desktop app with Main Analysis, local video
discovery/refresh, metadata, play/pause, seek, adjacent-video navigation, and a
zoomable aspect-correct image view. Show the selected filename and actionable load
errors. Defer the three specialist workspaces until their phases.

Introduce an explicit video session, background decoding, cancellation on seek or
video change, validated path/settings reads, and a minimal frame/timing contract.
Model inference is not part of this phase.

**You check:** Open each baseline clip; resize the window; play, pause, and scrub
rapidly; switch videos during playback; try a missing/unsupported file; close and
reopen. The selected frame must appear without frames from the previous video.

**Evidence:** Tests cover failed opens/seeks, frame-index semantics, resource
release, and late results from cancelled sessions. Record startup, seek latency,
and decode/display FPS with no vision features enabled.

**Stop:** Accept playback and basic layout before adding detection.

## Phase 2 — Background player and disc detection

**Deliver:** Independent player/disc model selectors, thresholds, detection toggle,
bounding boxes, class/confidence labels, model-load progress, and inference timings.
Missing models or unavailable acceleration produce clear recoverable states.

Introduce the scheduler with bounded work queues and cancellation. Every result
must identify its video, frame, and settings/model revision. Reject stale results
after a seek, model switch, or toggle. Load/warm models off the UI thread and reuse
loaded models. Do not silently change image size or precision to meet a speed goal.

**You check:** Play with detection enabled, scrub while inference is busy, change
each model independently, toggle detection, and test a missing model. Move/resize
the window during model loading. Confirm player and disc labels are distinct even
when both models use local class ID zero.

**Evidence:** Adapter and scheduler tests cover output coordinates, bounded queues,
invalidation, failure recovery, and repeated model selection without reloading.
Measure warm stage latency and dropped frames against Phase 0. If adaptive disc
skipping is implemented, test that a returning disc is detected again.

**Stop:** Accept detection quality and interaction responsiveness.

## Phase 3 — Tracking and the basic analyst workflow

**Deliver:** Stable IDs, colours, foot positions, bounded trails, tracking toggle,
reset control, and the basic timing/FPS view. Clearly label any simple fallback's
limitations; it must not masquerade as persistent tracking.

Tracking owns ordered state for a session. Define how it handles skipped source
frames: age/predict by elapsed source frames or reset on a declared discontinuity.
It must never receive out-of-order updates from background workers. Empty
detections still advance tracker state. Paused redraws do not advance it.

**You check:** Follow players through short occlusions; compare an empty scene and
its return; replay/reset; seek backwards; switch videos and models. Verify that old
identities/trails disappear on reset and no other tab changes this session's state.

**Evidence:** Tests cover class identity, empty-frame ageing, history eviction,
ordered delivery, dropped-frame policy, and reset without unnecessary model reloads.
Run detection + tracking continuously and compare throughput and memory to baseline.

**Stop:** Accept this as the first useful analyst build (Release 1).

## Phase 4 — Field segmentation and geometry

**Deliver:** Segmentation model selection, enable/overlay toggles, masks and contours,
morphology controls, optional RANSAC lines/diagnostics, and per-stage timings.
Geometry runs in the background and is reused when its inputs have not changed.

**You check:** Compare a clear field, camera motion, and a frame with no usable
field. Toggle overlays separately from computation; change the model; seek and
switch videos. Inspect the mask and line alignment at different zoom levels.

**Evidence:** Tests cover original-image coordinates, empty-result caching, interval
scheduling, and invalidation after shape/frame/model/parameter changes. Record
segmentation and geometry invocation counts as well as latency.

**Stop:** Accept geometry quality before using it for calibration.

## Phase 5 — Live jersey-number identification

**Deliver:** Player-only crops, configured preprocessing/OCR, per-track aggregation,
confidence/finalization, scheduling, and jersey overlays/summary. Explain the
detection/tracking dependency in the UI and show unavailable OCR as an error state.

Run OCR off the UI thread with bounded pending crops. Associate every reading with
its session, track, and frame; discard results from before a reset or track retirement.

**You check:** Follow several readable jerseys; watch uncertain readings converge;
inspect small/unreadable crops; disable/re-enable OCR; reset or switch video while
OCR is busy. Confirm disc tracks never receive jersey numbers.

**Evidence:** Test numeric filtering, crop coordinates, scheduling, aggregation,
finalization, and stale-result rejection. Compare OCR-on versus OCR-off latency and
pending work. Check output quality alongside reduced OCR invocation counts.

**Stop:** Accept live identity behavior before adding tuning controls.

## Phase 6 — OCR tuning workspace

**Deliver:** A lazily initialized tuning tab with its own video session, frame/model
selection, explicit Run Analysis action, annotated source frame, crop grid, parameter
controls, and load/save of a separate OCR-settings file compatible with legacy YAML.

**You check:** Tune one readable and one difficult frame; inspect crop/confidence
changes; save and reload settings; switch between tuning and Main Analysis. Tuning
must not silently change Main Analysis's tracker, frame, model, or live parameters.
Applying tuning settings to live analysis must be an explicit action.

**Evidence:** Test YAML round trips and unknown-key preservation, invalid values,
tab/session isolation, and consistent preprocessing between tuning and live analysis.

**Stop:** Accept the tuning workflow and settings-application behavior.

## Phase 7 — Manual homography and top-down analysis

**Deliver:** A lazily initialized calibration tab with independent video selection,
original/warped views, eight editable transform parameters, grid/zoom controls,
reset, save/load with metadata, and segmentation diagnostics. Connect accepted
calibration to Main Analysis's top-down image and mapped track/jersey positions.

Keep preview warping off the UI thread and cancel obsolete slider previews. Warn
when a calibration belongs to a different source video instead of silently applying
it. Genetic optimization is reserved for Phase 8.

**You check:** Calibrate a recognizable frame, save/reload, inspect image landmarks
and mapped player feet, resize both views, and use the result in Main Analysis.
Try missing, malformed, singular, and wrong-video calibration files.

**Evidence:** Test known-point coordinate transforms, save/load equivalence, invalid
matrix handling, correct source/canvas coordinates, and latest-preview delivery.
Measure warp latency with tracking and OCR enabled.

**Stop:** Accept manual calibration and the complete analyst workflow (Release 2).

## Phase 8 — Genetic calibration assistant

**Deliver:** Start/reset, one generation, batch evolution, continuous evolution,
stop, best-transform preview/apply, fitness chart, and runtime inspection. Require
usable geometry before starting. Preserve manual calibration until Apply is used.

**You check:** Evolve a clear-field frame, stop during a batch, change frames while
running, compare candidate previews, and explicitly apply a result. Try starting
without usable geometry. The GUI must remain usable throughout.

**Evidence:** Use seeded runs to check scoring and reproducibility. Test cancellation,
invalid prerequisites, bounded history, and rejection of results from an old frame.
Quality requires visual review; higher fitness alone is not acceptance.

**Stop:** Accept optimizer controls and calibration usefulness.

## Phase 9 — Model training workspace

**Deliver:** A lazy training tab with task selection, model/dataset discovery,
parameter and augmentation controls, YAML load/save, subprocess execution, live
logs/status, elapsed time, stop, metrics plots, and baseline comparison.

Use a new output directory for every run. Start with a fake training subprocess to
verify lifecycle/log handling, then perform an explicitly launched short smoke run
on a disposable fixture. Do not begin a full training job automatically.

**You check:** Review run settings/output location; launch a short run; watch logs
and plots; stop it; inspect retained partial results; load a completed result and
baseline. Try an invalid dataset/model. Navigate the GUI while training is active.

**Evidence:** Test process launch/failure/stop/close behavior, partial CSV reads,
configuration round trips, output discovery, and preservation of existing datasets
and runs. State how training and live inference share or contend for the GPU.

**Stop:** Accept training lifecycle and output handling (Release 3 features).

## Phase 10 — Full-workflow validation and cutover decision

**Deliver:** An end-to-end parity report against design Section 11, reproducible
launch/setup instructions, dependency/error-state checks, and performance results
on the agreed clips/hardware. Resolve regressions before offering a default-launch
change. Final acceptance does not authorize deleting legacy code or data.

**You check:** Perform a complete analyst session, OCR tuning, calibration/save/load,
optimizer run, and short training run. Reopen the app and verify persisted choices.
Exercise rapid seeks, resets, model changes, tab switching, and clean shutdown.
Run a longer session to check that memory and queues remain bounded.

**Evidence:** Report cold/warm latency separately, processed and displayed FPS,
dropped frames, queue depth, CPU/GPU memory, and visual quality. Compare like-for-like
settings; document unmet targets rather than hiding them through reduced accuracy.
Verify the old app can still open its original configuration and inputs.

**Stop:** You decide whether the rebuild becomes the default app. Keep the previous
accepted build and legacy entry point available for rollback.

## Review record (complete at every checkpoint)

```text
Phase / revision:
Launch command:
Completed scope:
Manual steps and expected results:
Automated checks and results:
Performance evidence (hardware, clips, models, settings):
Known limitations / deferred work:
Files or settings written:
How to run the previous accepted build:
User decision: pending / accepted / changes requested
Review findings:
```

Current status: **planning only; no rebuild phase has been started or accepted.**
