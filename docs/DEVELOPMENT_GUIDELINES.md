## The Most Important Guideline: KISS

**Keep It Simple, Stupid (KISS):**
- Always strive for the simplest solution that works.
- Avoid unnecessary complexity, cleverness, or over-engineering.
- Prefer clear, readable, and maintainable code over “smart” tricks.
- Simplicity makes code easier to test, debug, and extend.

---

# Ultimate Analysis - Development Guidelines

## Project Overview

**Ultimate Analysis** is a PyQt5-based video analysis application designed for Ultimate Frisbee game analysis. The application provides real-time video processing capabilities including:

- **Object Detection & Tracking**: YOLO-based player and disc detection with multi-object tracking
- **Player Identification**: Jersey number recognition using both YOLO and EasyOCR methods
- **Field Segmentation**: Automated field boundary detection and visualization
- **Performance Monitoring**: Real-time runtime analysis with sparkline visualizations
- **Batch Processing**: Optimized parallel processing for player ID and other computationally intensive tasks

### Key Components
- **GUI Layer** (`visu/`): PyQt5 interface with video player, controls, and visualizations
- **Processing Layer** (`processing/`): Core ML inference, tracking, and analysis algorithms
- **Training Data** (`training_data/`, `finetune/`): Custom YOLO models for sports-specific detection
- **Utilities** (`utils/`): Dataset generation and helper functions

## Code Quality Standards

### File Size and Structure
- **Maximum file size**: 500 lines per file
- **Refactoring trigger**: When approaching 500 lines, refactor into logical modules
- **Module organization**: Group by feature or responsibility, not file type
- **Clear separation**: Distinguish between GUI, processing, utilities, and configuration

### Python Standards
- **Primary language**: Python 3.8+
- **Code formatting**: Use `black` for consistent formatting
- **Style guide**: Follow PEP8 strictly
- **Type hints**: Required for all function signatures and class attributes
- **Data validation**: Use `pydantic` for data models and validation

### Import Organization
```python
# Standard library imports
import os
import time
from typing import List, Dict, Optional

# Third-party imports
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget
from pydantic import BaseModel

# Local imports (prefer relative within packages)
from .video_player import VideoPlayer
from ..processing.inference import run_inference
```

### Documentation Requirements
- **README.md**: Place in the project root. Update when adding features, changing dependencies, or modifying setup.
- **DEVELOPMENT_GUIDELINES.md**: Place in the project root. Update as standards and best practices evolve.
- **LICENSE, CONTRIBUTING.md**: Place in the project root for visibility and compliance.
- **docs/**: Place extended or detailed documentation (user guides, API docs, diagrams) in a `docs/` directory at the root.
- **Inline comments**: Required for non-obvious code logic
- **Reasoning comments**: Use `# Reason:` prefix to explain complex logic decisions
- **Docstrings**: Required for all public functions and classes
- **Type documentation**: Include types in docstrings for complex parameters

### Code Organization Principles

#### Module Structure
```
ultimate-analysis/
├── README.md                           # Project overview and setup instructions
├── LICENSE                            # Project license
├── CONTRIBUTING.md                    # Contribution guidelines
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup configuration
├── .gitignore                        # Git ignore patterns
│
├── docs/                             # Documentation
│   ├── DEVELOPMENT_GUIDELINES.md     # Development standards (this file)
│   └── assets/                       # Documentation images/diagrams
│
├── src/                              # Source code
│   └── ultimate_analysis/            # Main package
│       ├── __init__.py
│       ├── main.py                   # Application entry point
│       ├── config/                   # Configuration management
│       │   ├── __init__.py
│       │   ├── settings.py           # Application settings
│       │   └── constants.py          # Global constants
│       │
│       ├── core/                     # Core business logic
│       │   ├── __init__.py
│       │   ├── models.py             # Data models (Pydantic)
│       │   ├── exceptions.py         # Custom exceptions
│       │   └── utils.py              # Core utilities
│       │
│       ├── processing/               # ML and video processing
│       │   ├── __init__.py
│       │   ├── inference.py          # YOLO inference engine
│       │   ├── tracking.py           # Object tracking algorithms
│       │   ├── field_segmentation.py # Field detection and segmentation
│       │   ├── player_id.py          # Player identification (YOLO/OCR)
│       │   └── video_processor.py    # Video processing pipeline
│       │
│       ├── gui/                      # User interface
│       │   ├── __init__.py
│       │   ├── app.py                # Main application window
│       │   ├── components/           # Reusable UI components
│       │   │   ├── __init__.py
│       │   │   ├── video_player.py   # Video player widget
│       │   │   ├── controls.py       # Control panels
│       │   │   └── dialogs.py        # Dialog windows
│       │   │
│       │   ├── tabs/                 # Tab implementations
│       │   │   ├── __init__.py
│       │   │   ├── main_tab.py       # Main analysis tab
│       │   │   ├── preprocessing_tab.py # Video preprocessing
│       │   │   ├── training_tab.py   # Model training interface
│       │   │   └── tuning_tab.py     # Parameter tuning
│       │   │
│       │   └── visualization/        # Visualization components
│       │       ├── __init__.py
│       │       ├── detection_viz.py  # Detection overlays
│       │       ├── tracking_viz.py   # Tracking visualizations
│       │       ├── field_viz.py      # Field segmentation display
│       │       └── player_id_viz.py  # Player ID visualizations
│       │
│       └── utils/                    # Utility modules
│           ├── __init__.py
│           ├── file_utils.py         # File operations
│           ├── image_utils.py        # Image processing utilities
│           ├── video_utils.py        # Video manipulation utilities
│           ├── dataset_utils.py      # Dataset generation and management
│           └── performance_utils.py  # Performance monitoring
│
├── data/                             # Data directories (gitignored)
│   ├── raw/                          # Raw input data
│   ├── processed/                    # Processed datasets
│   ├── models/                       # Trained models
│   │   ├── detection/                # Object detection models
│   │   ├── segmentation/             # Field segmentation models
│   │   └── player_id/                # Player identification models
│
├── configs/                          # Configuration files
│   ├── default.yaml                 # Default configuration
│   ├── development.yaml             # Development settings
│   ├── production.yaml              # Production settings
│   └── model_configs/               # Model-specific configurations
│       ├── yolo_detection.yaml
│       ├── field_segmentation.yaml
│       └── player_id.yaml
```

### Key Structure Principles

1. **Clear Separation of Concerns**: GUI, processing, and utilities are separate
2. **Testability**: Test structure mirrors source structure
3. **Scalability**: Modular design allows easy addition of new features
4. **Configuration Management**: Centralized configuration with environment-specific settings
5. **Data Organization**: Structured data directories with clear purposes
6. **Documentation**: Comprehensive documentation structure
7. **Modern Python**: Follows `src/` layout best practices

#### Naming Conventions
- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Error Handling and Logging
- **Structured logging**: Use Python's `logging` module with appropriate levels
- **Exception handling**: Catch specific exceptions, not broad `except:` blocks
- **Error context**: Include relevant context in error messages
- **Graceful degradation**: Application should handle missing models/files gracefully

### Performance Considerations
- **Caching**: Cache expensive operations (model loading, computations)
- **Batch processing**: Use parallel processing for CPU-intensive tasks
- **Memory management**: Clean up resources and clear caches when switching contexts
- **Frame rate optimization**: Adjust processing frequency based on computational load

### Testing Standards
- **Test files**: Prefix with `test_` for easy identification
- **Test coverage**: Focus on critical paths and edge cases
- **Integration tests**: Test complete workflows, not just individual functions
- **Performance tests**: Monitor and validate optimization improvements

## Development Workflow

## Algorithm Implementation Protocol

**Before implementing any algorithm in code:**
1. Write and present clear pseudocode for the algorithm.
2. Ask the user if the pseudocode or approach needs to be modified or clarified.
3. Only proceed to code implementation after user approval of the pseudocode.

This ensures transparency, correctness, and alignment with project requirements for all algorithmic work.

### Before Making Changes
1. **Understand context**: Review existing code structure and patterns
2. **Verify dependencies**: Confirm all imports and file paths exist
3. **Check file size**: If approaching 500 lines, plan refactoring strategy
4. **Consider impact**: Assess how changes affect related components

### Making Changes
1. **Incremental development**: Make small, focused changes
2. **Consistent patterns**: Follow existing code patterns and conventions
3. **Type safety**: Add type hints for new functions and variables
4. **Documentation**: Update comments and docstrings as needed

### Code Review Checklist
- [ ] File size under 500 lines
- [ ] PEP8 compliance
- [ ] Type hints present
- [ ] Appropriate error handling
- [ ] Clear variable/function names
- [ ] Reasoning comments for complex logic
- [ ] No hardcoded values (use constants/config)
- [ ] Proper resource cleanup
- [ ] Updated documentation

## Critical Guidelines

### Never Do
- **Assume missing context**: Always ask for clarification when uncertain
- **Hallucinate libraries**: Only use verified, existing Python packages
- **Delete existing code**: Unless explicitly instructed or part of defined task
- **Ignore file paths**: Always verify paths and module names exist
- **Use broad exceptions**: Catch specific exception types
- **Hardcode values**: Use configuration or constants

### Always Do
- **Verify before referencing**: Check that files, modules, and functions exist
- **Maintain consistency**: Follow existing patterns and conventions
- **Document changes**: Update README.md for significant modifications
- **Consider performance**: Optimize for the real-time nature of video processing
- **Always consider runtime**: When implementing features, always account for runtime performance and efficiency, especially for AI/ML components and video processing tasks.
- **Handle errors gracefully**: Provide meaningful error messages and fallback behavior

## Project-Specific Considerations

### Video Processing
- **Real-time constraints**: Balance accuracy with performance
- **Memory usage**: Monitor and optimize for large video files
- **Model loading**: Cache loaded models to avoid repeated initialization
- **Frame rate adaptation**: Adjust processing based on computational load

### ML Model Integration
- **Model paths**: Use relative paths and verify existence
- **Error handling**: Gracefully handle missing or corrupted models
- **Performance monitoring**: Track inference times and optimize bottlenecks
- **Batch processing**: Utilize parallel processing for efficiency

### GUI Responsiveness
- **Non-blocking operations**: Use threading for long-running tasks
- **Progress indication**: Provide feedback for lengthy operations
- **Resource cleanup**: Properly dispose of Qt resources
- **User feedback**: Clear error messages and status updates

## Questions to Ask
When uncertain about implementation details:
1. What is the expected input/output format?
2. Are there existing patterns or utilities I should use?
3. What are the performance requirements?
4. How should errors be handled in this context?
5. Are there dependencies on other modules I should be aware of?

---

*This document should be referenced before making any code changes to ensure consistency and quality across the Ultimate Analysis project.*
The project structure and organization should follow best practices for modern Python computer vision and AI applications, including clear separation of concerns, modularity, and maintainability. Use the provided module structure as a baseline, and adapt as needed to ensure scalability and clarity as the project evolves.
