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

## Use Cases

- Automated game statistics and analytics
- Tactical breakdowns and visualizations
- Player performance tracking
- Highlight and event detection

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

[MIT License](LICENSE)

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

## Devlog (To clean up and put elsewhere)
04.06.25
- initial project & commit to GitHub
- find example video to develop project on: https://www.youtube.com/watch?v=iWrF4S3TkoU
- create functions to export random screenshots and video snippets from video to use for development

05.06.25
- initial training data for field detection using semantic segmentation (annotated 45 images using Roboflow)
- expanded training data to 100 images & used Roboflow to augment data - flip vertically & shear
- https://app.roboflow.com/sylvain-5mcdp/field-finder/2

