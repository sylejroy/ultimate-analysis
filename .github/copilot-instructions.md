# Ultimate Analysis — AI Coding Agent Guide

PyQt5 video analysis app for Ultimate Frisbee: YOLO detection, DeepSORT tracking,
EasyOCR jersey numbers, field segmentation, and a homography top-down view.

- Follow `docs/DEVELOPMENT_GUIDELINES.md` for layout, style, and workflow.
- `docs/REBUILD_DESIGN_DOCUMENT.md` describes what each tab does.
- Run Python with the project environment: `./.venv/Scripts/python.exe`.
- Entry point: `main.py`.
- Settings: `ultimate_analysis.config.settings.get_setting("models.player_detection.confidence_threshold", 0.5)`.
  There are no environment-variable overrides.
- Trained models live in `data/models/<task>/<run>/finetune_*/weights/best.pt`;
  base weights in `data/models/pretrained/`.
- Never delete or overwrite anything under `data/` without being asked.
- Checks before finishing: `python -m ruff check src tests` and
  `python -m unittest discover -s tests -v`.
