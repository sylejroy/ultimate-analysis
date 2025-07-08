# Ultimate Analysis

A PyQt5-based video analysis application designed for Ultimate Frisbee game analysis, featuring real-time object detection, tracking, player identification, and field segmentation.

## Features

- **Object Detection & Tracking**: YOLO-based player and disc detection with multi-object tracking
- **Player Identification**: Jersey number recognition using both YOLO and EasyOCR methods  
- **Field Segmentation**: Automated field boundary detection and visualization
- **Performance Monitoring**: Real-time runtime analysis with sparkline visualizations
- **Ground Plane Estimation**: Map pixel coordinates to real-world field positions using player height calibration

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyQt5
- OpenCV
- Ultralytics YOLO
- EasyOCR

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ultimate-analysis
```

2. Set up the environment:
```bash
python scripts/setup_environment.py
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your video files in `input/dev_data/`

5. Run the application:
```bash
python src/ultimate_analysis/main.py
```

## Project Structure

```
ultimate-analysis/
├── src/ultimate_analysis/     # Main application code
│   ├── core/                  # Core business logic and models
│   ├── processing/            # ML inference and video processing
│   ├── gui/                   # PyQt5 user interface
│   ├── utils/                 # Utility functions
│   └── config/                # Configuration management
├── tests/                     # Test suite
├── configs/                   # Configuration files
├── scripts/                   # Utility scripts
├── data/                      # Data directories (gitignored)
└── docs/                      # Documentation
```

## Configuration

The application uses YAML configuration files located in `configs/`:

- `default.yaml` - Base configuration
- `development.yaml` - Development settings  
- `production.yaml` - Production optimized settings

You can specify a custom configuration file:
```bash
python src/ultimate_analysis/main.py configs/custom.yaml
```

## Development

### Code Quality

This project follows strict development guidelines:

- Maximum 500 lines per file
- PEP8 compliance with black formatting
- Type hints required
- Comprehensive documentation
- Structured error handling

See `docs/DEVELOPMENT_GUIDELINES.md` for complete standards.

### Testing

Run the test suite:
```bash
pytest tests/
```

Run performance benchmarks:
```bash
python scripts/benchmark.py
```

## Architecture

### Core Components

- **Processing Pipeline**: ML inference → Object tracking → Player ID → Field mapping
- **Real-time Analysis**: Optimized for live video processing with configurable performance targets
- **Modular Design**: Clear separation between GUI, processing, and utilities
- **Extensible**: Easy to add new detection models or tracking algorithms

### Data Models

The application uses Pydantic models for type safety and validation:

- `Detection`: Object detection results
- `Track`: Multi-frame object tracking
- `PlayerID`: Player identification results
- `FieldPosition`: Real-world coordinates
- `AnalysisResult`: Complete frame analysis

## Contributing

1. Follow the development guidelines in `docs/DEVELOPMENT_GUIDELINES.md`
2. Write tests for new features
3. Update documentation as needed
4. Submit pull requests with clear descriptions

## License

[License information to be added]

## Support

For questions or issues, please [contact information to be added].
