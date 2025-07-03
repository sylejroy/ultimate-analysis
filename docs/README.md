# Ultimate Analysis

## Project Overview

Ultimate Analysis is a computer vision and AI-powered tool for automated analysis of Ultimate Frisbee games from video footage. The system extracts meaningful information from game videos, including player and disc tracking, field mapping, and game state detection, to enable advanced analytics and insights.

## Features (Currently Implemented)

- **Playing Field Detection**
  - Detect and segment the playing field from video frames using YOLO segmentation models.
  - Visualize field segmentation overlays and field boundaries.

- **Player Detection and Tracking**
  - Detect players on the field using YOLO object detection models.
  - Track player movements across frames using DeepSort or a custom histogram-based tracker.
  - Identify players using jersey numbers via YOLO digit detector or EasyOCR (configurable in the GUI).

- **Disc Detection and Tracking**
  - Detect the disc in each frame (if included in your detection model).
  - Track disc movement (if included in your tracking logic).

- **GUI Application**
  - Visualize detections, tracks, field segmentation, and player IDs in a PyQt5-based GUI.
  - Switch between different models and trackers in the settings tab.
  - Step through video frames, play/pause, and reset trackers and visualisation state.
  - Select and switch between YOLO and EasyOCR for player identification.

- **Testing and Scripts**
  - Example scripts for training, running and visualizing detection, tracking, field segmentation, and digit/jersey number recognition on video files.

## Nice-to-Have / Future Features

- **Game State Detection**
  - Classify game states (live play, stoppage, pre-pull, between points) from video.
  - Detect turnovers, team possession, and player in possession.

- **Disc Possession and Event Detection**
  - More robust disc tracking and event detection (e.g., passes, turnovers).

- **Perspective Transformation**
  - Apply perspective transformation to generate a top-down view for tactical analysis.

- **Custom Dataset Tools**
  - Tools for creating and managing custom datasets (e.g., via Roboflow) to improve detection accuracy.

- **Advanced Analytics**
  - Generate player heatmaps, movement statistics, and tactical insights.

- **Export and Reporting**
  - Export results to CSV, JSON, or annotated video.

- **Model Weights Hosting**
  - Host trained YOLO and segmentation weights for easy download and reproducibility (e.g., on [Hugging Face Hub](https://huggingface.co/), [Zenodo](https://zenodo.org/), or [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)).

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