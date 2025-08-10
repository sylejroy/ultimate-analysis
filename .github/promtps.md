# Ultimate Analysis - Development Prompts

This document contains structured development prompts optimized for AI coding agents working on the Ultimate Frisbee analysis application.

## 📋 Project Context

**Architecture**: PyQt5 GUI + YOLO ML processing + OpenCV visualization
**Key Principle**: KISS (Keep It Simple, Stupid) - max 500 lines per file
**Configuration**: YAML-based with environment overrides
**Data Flow**: Video → YOLO Detection → Tracking → Player ID → Visualization

**Required Reading**: Always review `copilot-instructions.md` before implementing any feature.

---

## 🎯 Active Development Tasks

### ✅ COMPLETED
- [x] Main Tab with video player and processing controls
- [x] Inference processing (YOLO detection)
- [x] Player ID with EasyOCR jersey number detection
- [x] EasyOCR Tuning Tab with parameter optimization
- [x] Performance monitoring display
- [x] Jersey Number Tracking with probabilistic fusion
- [x] Aggressive parallel processing optimization

### 🚧 IN PROGRESS
- [ ] Model Training Tab (current focus)

### 📋 PENDING
- [ ] Advanced field segmentation visualization
- [ ] Batch video processing pipeline
- [ ] Export/import functionality for analysis results

---

## 🏗️ Model Training Tab Specification

**Objective**: Create a comprehensive model training interface using Ultralytics YOLO

### Core Requirements

#### Task Type Selection
```
Radio buttons: [Detection] [Field Segmentation]
- Detection: Players, discs, general object detection
- Field Segmentation: End zones, field boundaries, playing areas
```

#### Model Selection Pipeline
```
1. Base Model Dropdown:
   - Detection: Non-seg models from `data/models/pretrained/` and `data/models/detection/`
   - Segmentation: -seg models from `data/models/pretrained/` and `data/models/segmentation/`

2. Model Info Display:
   - Model architecture (YOLOv11n/s/m/l/x)
   - File size, parameters count
   - Input resolution, class count
   - Last modified date
```

#### Training Data Management
```
1. Dataset Dropdown:
   - Detection: Scan `data/raw/training_data/` for "*object_detection*" or "*player*disc*"
   - Segmentation: Look for "*field*finder*" datasets

2. Dataset Info Display:
   - Number of images (train/val/test splits)
   - Class distribution chart
   - Sample image with annotations overlay
   - Data format (YOLO, COCO, etc.)
```

#### Training Configuration
```
Parameters Panel:
├── Core Settings
│   ├── Epochs: [50] (1-1000)
│   ├── Patience: [10] (1-100)  
│   ├── Batch Size: [16] or [0.8] (auto GPU memory)
│   └── Learning Rate: [0.01] (0.0001-1.0)
├── Advanced Settings
│   ├── Image Size: [640] (320-1280)
│   ├── Optimizer: [SGD|Adam|AdamW]
│   ├── Workers: [8] (0-16)
│   └── Device: [0|cpu] (GPU selection)
└── Configuration Actions
    ├── [Load from training.yaml]
    ├── [Save to training.yaml]
    └── [Reset to Defaults]
```

#### Training Execution
```
Progress Monitoring:
├── Status Bar: "Epoch 25/100 (ETA: 15:30)"
├── Progress Bar: Visual epoch completion
├── Live Metrics Table:
│   ├── Current Loss (train/val)
│   ├── mAP@0.5 (detection)
│   ├── Precision/Recall
│   └── Learning Rate Schedule
└── Results Visualization:
    ├── Training curves (from results.csv)
    ├── Loss progression graph
    └── Validation metrics plot
```

#### Output Management
```
Save Strategy:
- Create timestamped folders: `data/models/{task}/{model_name}/finetune_{timestamp}/`
- Preserve all training artifacts: weights/, results.csv, args.yaml
- Auto-backup previous training runs
- Generate training summary report
```

### Implementation Architecture

#### File Structure
```
src/ultimate_analysis/gui/model_tuning_tab.py  # Main UI implementation
src/ultimate_analysis/training/train_model.py  # Training logic wrapper
configs/training.yaml                          # Training parameters
```

#### Key Classes
```python
class ModelTuningTab(QWidget):
    """Main training interface with all UI components"""
    
class TrainingThread(QThread):
    """Background training execution with progress signals"""
    
class DatasetScanner:
    """Utility for discovering and analyzing training datasets"""
    
class ModelAnalyzer:
    """Extract metadata and info from YOLO model files"""
```

#### Integration Points
- Configuration system: Use `get_setting()` for all parameters
- Error handling: Graceful failures with user feedback
- Resource monitoring: CPU/GPU usage during training
- Logging: Comprehensive training logs in `logs/training/`

### User Experience Flow
1. **Select Task Type** → Updates available models and datasets
2. **Choose Base Model** → Displays model information and capabilities  
3. **Select Training Data** → Shows dataset stats and sample images
4. **Configure Parameters** → Load/save configs, validation checks
5. **Start Training** → Background execution with live progress
6. **Monitor Results** → Real-time metrics and visualization
7. **Completion** → Model saved, summary report generated

### Success Criteria
- [ ] Intuitive task-based workflow
- [ ] Comprehensive model and dataset information
- [ ] Robust training parameter management
- [ ] Real-time progress monitoring
- [ ] Reliable output organization
- [ ] Integration with existing configuration system
- [ ] Error handling and user feedback
- [ ] Performance monitoring during training

---

## 🔧 Implementation Guidelines

### Code Quality Standards
- **File Size Limit**: Maximum 500 lines per file
- **Type Hints**: Required for all function signatures
- **Documentation**: Docstrings for all public methods
- **Error Handling**: Comprehensive try-catch with user feedback
- **Configuration**: Use YAML settings, avoid hardcoded values

### Architecture Patterns
- **Separation of Concerns**: GUI ↔ Processing ↔ Configuration
- **Signal-Slot Communication**: PyQt signals for async operations
- **Resource Management**: Proper cleanup of threads and file handles
- **Performance**: Monitor memory usage, optimize for real-time processing

### Testing Strategy
- **Integration Testing**: Test with actual model files and datasets
- **UI Testing**: Verify all controls work with valid/invalid inputs
- **Performance Testing**: Monitor resource usage during training
- **Error Testing**: Handle missing files, corrupted data, training failures

---

## 📝 Development Notes

### Known Constraints
- Windows PowerShell v5.1 environment
- PyQt5 GUI framework (not Qt6)
- YOLO model compatibility with Ultralytics library
- Real-time processing requirements for video analysis

### Future Enhancements
- Multi-GPU training support
- Custom data augmentation pipeline
- Model ensemble training
- Automated hyperparameter optimization
- Integration with cloud training platforms

---

*Last Updated: August 2, 2025*
*Branch: feature/parallel_processing*





## Field projection

Read through the repository to understand how it functionns. Keep the copilot-instructions.md in mind.

New feature to be developed: from the field segmentation output, I need to map points in the image to real-world points on a 2 dimensional ultimate field of dimensions 100x37m in order to determine the position of each player. Make an algorithm suggestion keeping the following in mind:
- you need to unify the field segmentation output, as the output is often several segmented areas which can overlap
- ideally from the unified output, build up to 4 lines to describe the outer border of the field
- bear in mind: most of the time only a portion of the field is in view. The footage is from a drone looking down the field towards the horizon. For the most part, the full width of the field is in view, but not the length
- estimate the position of the field in the image (parts of the field will be outside the bounds of the image, you need to consider that)
- use a probabilistic approach that seems reasonable (kalman, extended, particle filter etc. be sure to consider all options)
