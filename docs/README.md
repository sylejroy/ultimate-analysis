# Ultimate Analysis

## Project Overview

Ultimate Analysis is a computer vision and AI-powered tool for automated analysis of Ultimate Frisbee games from video footage. The system aims to extract meaningful information from game videos, including player and disc tracking, field mapping, and game state detection, to enable advanced analytics and insights.

## Planned Features

- **Playing Field Detection**
  - Automatically detect and segment the playing field from video frames.
  - Apply perspective transformation to generate a top-down view for tactical analysis.

- **Player Detection and Tracking**
  - Detect players on the field using object detection models.
  - Track player movements across frames.
  - (Future) Identify players using jersey numbers or other distinguishing features.

- **Disc Detection and Tracking**
  - Accurately locate the disc in each frame.
  - Track disc movement and estimate possession changes.

- **Game State Detection**
  - Classify game states (live play, stoppage, pre-pull, between points) from video.
  - Detect turnovers, team possession, and player in possession.

- **Custom Dataset Support**
  - Tools for creating and managing custom datasets (e.g., via Roboflow) to improve detection accuracy, especially for frisbee/disc detection.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## Setup
- install CUDA from Nvidia website
- install CCDNN from Nvidia website
- install pytorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
- install ultralytics
```
pip3 install ultralytics
```

## Current State Example:
![YOLO + DeepSort Tracking Example](tracking_example.png)