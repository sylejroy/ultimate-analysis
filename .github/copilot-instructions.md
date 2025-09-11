Strictly adhere to TaskSync Protocol #tasksync.md

# Ultimate Analysis - AI Coding Agent Instructions

This is a PyQt5-based video analysis application for Ultimate Frisbee game analysis using computer vision, YOLO models, and real-time processing.

## Project Architecture

**Core Philosophy**: KISS (Keep It Simple, Stupid) - maximum 500 lines per file, simple solutions over clever ones. Strictly adhere to TaskSync Protocol #tasksync.md.

### Key Components
- **`src/ultimate_analysis/config/`**: YAML-based configuration with environment overrides (`default.yaml`)
- **`src/ultimate_analysis/processing/`**: ML inference, YOLO models, tracking, field segmentation, player ID (OCR)
- **`src/ultimate_analysis/gui/`**: PyQt5 interface with video player, tabs, and visualization
- **`data/models/`**: Custom YOLO models (detection, segmentation, pose) organized by type and training runs

### Configuration Access
```python
from ultimate_analysis.config.settings import get_setting, get_config
confidence = get_setting("models.detection.confidence_threshold")  # Use dot notation
```

**Key Principle**: Clear separation between constants and configuration:
- **`constants.py`**: Immutable system constraints, validation bounds, fallback defaults
- **`*.yaml`**: Runtime-configurable settings that can change per environment
- **Always use `get_setting()` for configurable values**, not constants

### Model Loading Pattern
Models are cached and loaded from `data/models/` with fallback to pretrained. Check existing pattern in finetune directories:
```
data/models/detection/object_detection_yolo11l/finetune3/weights/best.pt
data/models/pretrained/yolo11l.pt
```

### Development Workflow
1. **File size check**: Keep under 500 lines (enforced in constants)
2. **Type hints**: Required for all function signatures
3. **Import order**: Standard → Third-party → Local (see development guidelines)
4. **Documentation**: Docstrings for public functions/classes
5. **Explanations**: Check with user on all new pieces of code for clarity and understanding - assume user is not an expert in python or computer vision
6. **Clarify**: If unsure about a feature, ask for clarification before implementing
7. **Understanding**: When modifying this codebase, always check these files first to understand existing patterns and maintain consistency with the established architecture.

## Performance Considerations

### Real-time Processing
- **Model caching**: Models loaded once, cached for reuse
- **Batch processing**: Use for player ID and heavy computations
- **Runtime optimization**: Profile and optimize bottlenecks

### GUI-Processing Bridge
PyQt5 GUI communicates with processing modules through simple data structures (dicts, lists). No complex ORM patterns.

### Model Management
- **Pretrained models**: Download to `data/models/pretrained/`
- **Finetuned models**: Organized by task and training run in `data/models/`
- **Training configs**: Each model has `args.yaml` with hyperparameters

