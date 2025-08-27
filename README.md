# Ultimate Analysis

A PyQt5-based video analysis application for Ultimate Frisbee game analysis using computer vision and YOLO models.

![Main Analysis Interface](docs/gui_example_main_analysis.png)

## Features

- **Real-time Object Detection**: YOLO-based player and disc tracking
- **Player Identification**: Jersey number recognition with OCR
- **Field Segmentation**: Automated field boundary detection
- **Homography Estimation**: Interactive perspective correction and field mapping
- **Model Training**: Custom YOLO model training interface
- **Performance Monitoring**: Built-in timing and memory analysis

## Screenshots

### Main Analysis Interface
![Main Analysis](docs/gui_example_main_analysis.png)
*Real-time video analysis with object detection and tracking*

### Homography Estimation Interface
![Homography Estimation](docs/gui_example_homography.png)
*Interactive perspective correction with real-time transformation preview*

### Model Training Interface  
![Model Training](docs/gui_example_model_training.png)
*Train custom YOLO models for improved detection accuracy*

### OCR Tuning Interface
![OCR Tuning](docs/gui_example_ocr_tuning.png)
*Fine-tune player identification and jersey number recognition*

## Quick Start

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB RAM minimum

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd ultimate-analysis
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python main.py
   ```

### Basic Usage

1. Load a video file through the File menu
2. Select appropriate YOLO models in settings
3. Click play to start real-time analysis
4. Use the Homography tab for perspective correction and field mapping
5. Access training and OCR tuning features through their respective tabs

## Key Features Detail

### Homography Estimation
Interactive perspective correction tool with:
- **Real-time Preview**: Side-by-side original and transformed views
- **Parameter Sliders**: Fine-tune transformation matrix values
- **Field Segmentation Overlay**: Visualize field boundaries on corrected perspective
- **Save/Load**: Persist homography parameters for different camera angles
- **Zoom & Grid**: Enhanced visualization aids for precise alignment

### DeepSORT Tracking
Reliable multi-object tracking featuring:
- **Consistent IDs**: Maintains player identities across frames
- **Jersey Integration**: Links jersey numbers with tracking IDs
- **Trajectory History**: Visualizes player movement patterns
- **Foot-level Tracking**: Accurate ground-plane position tracking

## Configuration

Configuration files are in `configs/`:
- `default.yaml` - Base settings
- `user.yaml` - User overrides
- `training.yaml` - Model training parameters
- `homography_params.yaml` - Saved homography transformations

## Development

The project follows the KISS principle with a 500-line file limit. Key directories:

- `src/ultimate_analysis/gui/` - PyQt5 interface
- `src/ultimate_analysis/processing/` - ML inference and tracking
- `src/ultimate_analysis/config/` - YAML configuration management
- `data/models/` - YOLO models (detection, segmentation, pose)

See `docs/DEVELOPMENT_GUIDELINES.md` for detailed development standards.

## Models

Uses custom-trained YOLO models:
- **Detection**: Player and disc detection (`yolo11l.pt`)
- **Segmentation**: Field boundary detection (`yolo11l-seg.pt`)  
- **Player ID**: EasyOCR for jersey number recognition

Models are cached for performance and organized by training runs in `data/models/`.

## License

GNU General Public License v3.0 - see LICENSE file for details.
