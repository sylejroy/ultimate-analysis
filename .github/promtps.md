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


"""
Feature: Sideline-based homography estimation from YOLO field segmentation
File: src/ultimate_analysis/processing/homography.py (new file under 500 lines)

Goal:
Given:
  - A video frame (np.ndarray, BGR)
  - A YOLO segmentation mask of the field (same resolution as frame)
  - Known world coordinates for a regulation ultimate field (100m x 37m)
Only the two far corners are visible in each frame. The near corners are hidden.
We want to estimate a homography that maps the world field coordinates to image coordinates.

Steps for implementation:
1. **Inputs**
   - image: np.ndarray
   - field_mask: np.ndarray (binary mask from YOLO segmentation model)
   - template_corners: np.ndarray([[0,0],[100,0],[100,37],[0,37]]) in meters
   - num_steps: int (resolution of near-corner search along sidelines)

2. **Edge Detection**
   - Apply morphological cleanup to `field_mask` to smooth edges.
   - Detect boundaries and fit:
     a) Far endline (Hough or RANSAC)
     b) Left sideline
     c) Right sideline

3. **Fixed Points**
   - Find far-left corner = intersection of far endline and left sideline.
   - Find far-right corner = intersection of far endline and right sideline.

4. **Search**
   - For s_L in linspace(0,1,num_steps) and s_R in linspace(0,1,num_steps):
       - Near-left = far_left + s_L * (vector along left sideline toward camera)
       - Near-right = far_right + s_R * (vector along right sideline toward camera)
       - Assemble world→image correspondences:
           (0,0) → far_left
           (100,0) → far_right
           (0,37) → near_left
           (100,37) → near_right
       - Compute H_candidate = cv2.getPerspectiveTransform()
       - Warp a binary field template (scaled to meters) into image space
       - Compute IoU between warped template and `field_mask`

5. **Selection**
   - Keep track of H with the highest IoU.
   - Return H_best and best IoU score.

6. **Integration**
   - This should be callable from existing processing pipeline after YOLO segmentation step.
   - Expose via `estimate_field_homography(image, field_mask, template_corners, num_steps=50) -> Tuple[np.ndarray, float]`.

Constraints:
- Keep file under 500 lines.
- Use type hints and docstrings.
- Follow import order (standard, third-party, local).
- Use get_setting() for configurable parameters.
- Do not break real-time performance: allow optional `frame_skip` from config.
"""

"""
Feature: GUI Tab for Interactive Homography Estimation
File: src/ultimate_analysis/gui/tabs/homography_tab.py (new file under 500 lines)

Goal:
Create a new experimental PyQt5 tab that allows a user to:
  - Select a video recording from the left column (reuse existing video selection UI).
  - Scroll through frames of the selected recording.
  - Adjust each of the 8 homography parameters (H[0,0] ... H[2,1], H[2,2] fixed at 1.0) with sliders.
  - See side-by-side display:
      Left: raw frame
      Right: frame warped with current homography matrix (cv2.warpPerspective).
  - Update warp in real time as sliders move.

Layout & Interaction:
1. **Base Layout**
   - Split into two main sections:
       a) Left sidebar: video selection list + frame scroll slider (reuse existing tab pattern).
       b) Main display: two QLabel widgets for images (original + warped), stacked horizontally.

2. **Homography Controls**
   - Under the displays, add 8 sliders (QSlider, horizontal), one per homography matrix parameter except H[2,2].
   - Sliders should be labeled with parameter index (e.g., "H00", "H01", ... "H21").
   - Sliders map to float ranges: default -0.005 to 0.005 for first two rows, -1e-5 to 1e-5 for third row (fine tuning perspective). These ranges should be configurable via `get_setting()`.
   - Also include a "Reset" button to restore the identity homography.

3. **Video Frame Navigation**
   - Use existing frame seek/scroll implementation from other experimental tabs.
   - When frame changes, update both displays immediately with current H.

4. **Warping**
   - Implement `apply_homography(frame: np.ndarray, H: np.ndarray) -> np.ndarray` using `cv2.warpPerspective`.
   - Output should be same resolution as input frame.

5. **Integration**
   - Tab should register in GUI alongside other experimental tabs.
   - No heavy processing — must remain interactive at video framerate.
   - Ensure consistent styling and naming with existing experimental tabs.

Constraints:
- Keep file under 500 lines.
- Use type hints and docstrings for all public methods.
- Follow repo import order (standard, third-party, local).
- Configuration (slider ranges, step size, video dir) should use get_setting().
- Do not block the GUI thread — use PyQt signals/slots if needed for slider updates.
"""