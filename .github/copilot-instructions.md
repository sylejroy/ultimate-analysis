Strictly adhere to TaskSync Protocol #tasksync.md.
Always launch Python with the project .venv activated.

# Ultimate Analysis — AI Coding Agent Guide

PyQt5 video analysis app for Ultimate Frisbee using YOLO, tracking, OCR, and real-time processing.

## Agentic Workflow (TaskSync) ✅
- Start by creating a concise todo list and set exactly one item in-progress.
- Gather context: read relevant files/directories before editing; prefer larger focused reads over many tiny ones.
- Make minimal, targeted edits; keep files ≤ 500 lines (see `constants.MAX_FILE_SIZE_LINES`).
- After edits, run formatting and quality gates. Iterate until clean.
- Explain new code with short, clear comments/docstrings; assume the user isn’t a CV/Python expert.

## How to Run and Validate
- Primary entrypoint: `main.py`
- Recommended: use VS Code tasks (Windows, .venv assumed)
	- Run GUI: “Run Ultimate Analysis GUI”
	- Format All Code: sorts imports and runs Black (line length 100)
	- Full Code Quality Check: file size, Ruff, MyPy, Bandit, Vulture, Radon
- Manual run (when needed): use `./.venv/Scripts/python.exe` to execute Python.

## Repository Map (current)
- `src/ultimate_analysis/config/` — YAML config access and helpers
	- `settings.py` exposes `get_config()` and `get_setting("dot.path")` with `UA_` env overrides
- `src/ultimate_analysis/gui/` — PyQt5 UI
	- `main_app.py`, `video_player.py`, tabs: `main_tab.py`, `model_tuning_tab.py`, `easyocr_tuning_tab.py`, `homography_tab.py`
- `src/ultimate_analysis/processing/` — CV/ML pipeline
	- `inference.py`, `tracking.py`, `field_segmentation.py`, `player_id.py`, `field_analysis.py`, `line_extraction.py`, `jersey_tracker.py`
- `src/ultimate_analysis/constants.py` — immutable limits, fallbacks, colors, paths
- `configs/` — `default.yaml`, and tuning YAMLs
- `data/models/` — finetuned runs + `pretrained/`
- `scripts/check_file_sizes.py` — dev guard for 500-line limit

## Config vs Constants
- Use `get_setting()` for anything that may vary per environment/runtime.
- Use `constants.py` for immutable system constraints, validation bounds, fallbacks.

Config access example:
```python
from ultimate_analysis.config.settings import get_setting
conf = get_setting("models.detection.confidence_threshold", 0.5)
```

Environment overrides (always prefixed with UA_, use double underscores for nesting):
- Example: `UA_MODELS__DETECTION__CONFIDENCE_THRESHOLD=0.7`

## Model Management and Loading
- Preferred search order for model weights:
	1) Latest finetuned run under `data/models/<task>/<run>/finetune_*/weights/best.pt`
	2) Fallback to `data/models/pretrained/*.pt`
- Keep models cached in-process; load once and reuse.
- Respect `constants.DEFAULT_PATHS` for path roots; prefer `pathlib.Path`.
- Current pretrained examples (available): `data/models/pretrained/yolo11*.pt`

## Coding Standards (KISS)
- Max 500 lines per file; split modules before exceeding.
- Type hints on all public functions; concise docstrings.
- Imports: standard lib → third-party → local. No cyclic deps.
- Style: `black` (line length 100) + `isort --profile black` + `ruff`.
- Avoid cleverness; choose readable, testable solutions.

## Quality Gates to Keep Green
- Format All Code → sorts imports, formats with Black.
- Full Code Quality Check → runs:
	- File size check (`scripts/check_file_sizes.py`)
	- Ruff (lint), MyPy (types), Bandit (security), Vulture (dead code), Radon (complexity)
- Treat any failure as a blocker; fix before proceeding.

## GUI ↔ Processing Contract
- Exchange simple, serializable structures (dicts, lists, tuples, numpy arrays).
- Long-running or heavy work stays in `processing/` with clear function boundaries:
	- Inputs: frame (np.ndarray BGR), settings (dict), cached models/trackers
	- Outputs: detections/tracks/masks as arrays or small dicts
- Keep blocking work off the UI thread (respect PyQt event loop).

## Key Modules and Roles
- `processing/inference.py`: YOLO inference and model caching
- `processing/tracking.py`: track management and state
- `processing/field_segmentation.py`: pitch/field masks
- `processing/player_id.py`: OCR (EasyOCR) for jerseys; batch heavy work
- `gui/video_player.py`: frame IO and playback control
- `gui/main_app.py`: app wiring and tabs

## Performance Tips
- Cache models; batch OCR; skip frames for expensive steps (see `constants.OPTIMIZATION`).
- Monitor/frame timing (`constants.PERFORMANCE_MONITORING`); avoid per-frame allocations.
- Profile flows with `profile_main.py` and inspect using `visualize_profile.py` (optional).

## Data and Paths
- Use `constants.DEFAULT_PATHS` for canonical dirs:
	- MODELS, PRETRAINED, DEV_DATA, RAW_VIDEOS, OUTPUT, LOGS, CACHE
- Video extensions: see `constants.SUPPORTED_VIDEO_EXTENSIONS`.
- Keep repository data under `data/` (raw, processed, models).

## Minimal “Contract” for New Functions
- Inputs/outputs documented, types hinted.
- Error modes: invalid path, missing model, CPU-only fallback.
- Edge cases: empty frame, no detections, slow GPU, large frames.

## When Adding/Changing Behavior
1) Touch only the smallest surface area; keep public APIs stable.
2) Use `get_setting()` for knobs; validate against `constants` bounds.
3) Wire UI with simple dicts; don’t block the UI thread.
4) Update docstrings/comments for user clarity.
5) Run Format + Full Quality checks and ensure clean.

## Quick References
- Entry point: `main.py`
- Config API: `ultimate_analysis.config.settings.get_setting()`
- Constants: `ultimate_analysis/constants.py`
- Tasks: “Run Ultimate Analysis GUI”, “Format All Code”, “Full Code Quality Check”

Keep it simple, keep it fast, keep it clear.

