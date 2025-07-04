
<p align="center">
  <img src="docs/gui_example.png" alt="Ultimate Analysis GUI Example" width="700"/>
</p>

# Ultimate Analysis

## Project Overview

Ultimate Analysis is a computer vision and AI-powered tool for automated analysis of Ultimate Frisbee games from video footage. The system extracts meaningful information from game videos, including player and disc tracking, field mapping, and game state detection, to enable advanced analytics and insights.

## Features (2025)

- **Playing Field Detection**
  - Detect and segment the playing field from video frames using YOLO segmentation models.
  - Visualize field segmentation overlays and field boundaries.

- **Player Detection and Tracking**
  - Detect players using YOLO object detection models.
  - Track player movements with DeepSort or a custom histogram-based tracker (selectable in the GUI).
  - Identify players by jersey number using either YOLO digit detector or EasyOCR (fully configurable in the main tab).

- **Disc Detection and Tracking**
  - Detect and track the disc (if included in your detection model and pipeline).

- **Unified Configuration Panel**
  - All settings (model selection, tracker, player ID method, etc.) are now centralized in the main tab, organized into logical sections (General, Tracker, Detection, Field Segmentation, Player Identification).
  - Changing any setting immediately updates both the backend and the UI.

- **Modern GUI Application**
  - PyQt5-based GUI for visualizing detections, tracks, field segmentation, and player IDs.
  - Step through video frames, play/pause, reset trackers, and visualization state.
  - Select and switch between YOLO and EasyOCR for player identification directly in the main tab.
  - All configuration is now in the left panel under the video list for a streamlined workflow.

- **Developer Tools**
  - **Dev-Runtimes Tab:** View live runtime statistics for each processing and visualization step (with max value and units, no rolling average).
  - **Dev-Video Preprocessing Tab:** Browse, preview, and snip video segments directly from the GUI.
  - **Dev-YOLO Training Tab:** View available training datasets and select a YOLO model architecture for training (UI mockup).

- **Testing and Scripts**
  - Example scripts for training, running, and visualizing detection, tracking, field segmentation, and digit/jersey number recognition on video files.

- **Robust, Modular Codebase**
  - Centralized all player ID logic (YOLO/EasyOCR) and parameter management in `processing/player_id.py`.
  - Visualization files contain only visualization logic, with robust error handling and logging.
  - All settings/configuration logic is in the main tab; the settings tab has been removed.
  - Public getter functions provide safe access to internal state for player ID methods and models.
  - Imports and type hints are fixed throughout for maintainability.
  - Only top-down (naive) pitch projection is used; camera-model-based projection logic has been removed.
  - Optimized for runtime performance (import placement, reduced frame copies, batch/parallel processing where possible).
  - Example scripts for training, running and visualizing detection, tracking, field segmentation, and digit/jersey number recognition on video files.

## Nice-to-Have / Future Features

- **Game State Detection**
  - Classify game states (live play, stoppage, pre-pull, between points) from video.
    - Implement a classifier for game state transitions.
    - Annotate and label training data for different game states.
    - Integrate state detection into the main analysis pipeline.
  - Detect turnovers, team possession, and player in possession.
    - Develop logic to infer turnovers from disc and player tracking.
    - Track team possession changes.
    - Identify which player is in possession of the disc at each frame.

- **Disc Possession and Event Detection**
  - More robust disc tracking and event detection (e.g., passes, turnovers).
    - Improve disc detection accuracy in crowded scenes.
    - Detect disc throw and catch events.
    - Automatically log passes, turnovers, and other key events.

- **Perspective Transformation**
  - Apply perspective transformation to generate a top-down view for tactical analysis.
    - Calibrate camera and field coordinates.
    - Map player and disc positions to a bird’s-eye view.
    - Visualize team formations and movement patterns on the transformed field.

- **Custom Dataset Tools**
  - Tools for creating and managing custom datasets (e.g., via Roboflow) to improve detection accuracy.
    - Build annotation tools for labeling players, discs, and field boundaries.
    - Automate dataset export and augmentation.
    - Integrate with Roboflow or similar platforms for dataset management.

- **Advanced Analytics**
  - Generate player heatmaps, movement statistics, and tactical insights.
    - Aggregate player positions over time to create heatmaps.
    - Calculate player speed, distance covered, and other statistics.
    - Analyze team strategies and formations.

- **Export and Reporting**
  - Export results to CSV, JSON, or annotated video.
    - Implement export functions for detections, tracks, and analytics.
    - Generate summary reports and visualizations.

- **Model Weights Hosting**
  - Host trained YOLO and segmentation weights for easy download and reproducibility.
    - Upload models to [Hugging Face Hub](https://huggingface.co/), [Zenodo](https://zenodo.org/), or [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
    - Provide download scripts or links in the documentation.

## Getting Started

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ultimate-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install ultralytics PyQt5 deep_sort_realtime easyocr scipy matplotlib
   ```

3. **(Optional) Download and install CUDA and cuDNN:**
   - For GPU acceleration, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) according to your system configuration.

> **Note:** You do not need to manually download YOLO weights; the Ultralytics library will automatically download required weights when you run the code.

### Usage

To run Ultimate Analysis on a video file:

```bash
python main.py --video <path-to-video> --output <output-directory>
```

For example, to analyze a video file `game_video.mp4` and save the output to the `outputs/` directory:

```bash
python main.py --video game_video.mp4 --output outputs/
```

## Dependencies

To run Ultimate Analysis, you will need the following Python packages and system dependencies:

### Python Packages

- **Python 3.8+**
- [opencv-python](https://pypi.org/project/opencv-python/) (`cv2`)
- [numpy](https://pypi.org/project/numpy/)
- [torch](https://pytorch.org/) (PyTorch, with CUDA support recommended)
- [ultralytics](https://pypi.org/project/ultralytics/) (for YOLO models)
- [PyQt5](https://pypi.org/project/PyQt5/) (for GUI)
- [deep_sort_realtime](https://pypi.org/project/deep-sort-realtime/) (for DeepSort tracking)
- [easyocr](https://pypi.org/project/easyocr/) (for jersey number/digit recognition, optional)
- [scipy](https://pypi.org/project/scipy/) (optional, for some image processing utilities)
- [matplotlib](https://pypi.org/project/matplotlib/) (optional, for plotting/debugging)

> **Note:** `pillow` is not required unless you add explicit image I/O or manipulation outside of OpenCV. You can remove it from the requirements unless you use it elsewhere.

### System Dependencies

- **CUDA Toolkit** (for GPU acceleration, recommended)
- **cuDNN** (for GPU acceleration, recommended)
- **ffmpeg** (for video reading/writing, required by OpenCV)