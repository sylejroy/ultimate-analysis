# Ultimate Analysis

A PyQt5-based video analysis application designed for Ultimate Frisbee game analysis with real-time computer vision capabilities.

## Features

- **Object Detection & Tracking**: YOLO-based player and disc detection with multi-object tracking
- **Player Identification**: Jersey number recognition using both YOLO and EasyOCR methods
- **Field Segmentation**: Automated field boundary detection and visualization
- **Performance Monitoring**: Real-time runtime analysis with performance metrics
- **Batch Processing**: Optimized parallel processing for computationally intensive tasks

## System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB RAM minimum, 16GB recommended
- 2GB available disk space for models and cache

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ultimate-analysis
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

4. **Set up environment**:
   ```bash
   python scripts/setup_environment.py
   ```

### Basic Usage

1. **Run the application**:
   ```bash
   python -m ultimate_analysis
   ```

2. **Load a video**: Use the file menu to select a video file
3. **Configure models**: Select appropriate YOLO models in the settings
4. **Start analysis**: Click play to begin real-time analysis

## Project Structure

```
ultimate-analysis/
├── src/ultimate_analysis/      # Main application source
│   ├── config/                 # Configuration management
│   ├── core/                   # Core utilities and exceptions
│   ├── gui/                    # PyQt5 interface components
│   ├── processing/             # ML inference and video processing
│   └── utils/                  # Utility functions
├── configs/                    # Configuration files
├── data/                       # Data directories (gitignored)
│   ├── models/                 # Trained models
│   ├── raw/                    # Raw input data
│   └── processed/              # Processed datasets
├── docs/                       # Documentation
├── scripts/                    # Setup and utility scripts
└── tests/                      # Test suite
```

## Configuration

The application uses YAML configuration files located in `configs/`:

- `default.yaml`: Base configuration
- `development.yaml`: Development-specific overrides
- `production.yaml`: Production settings

Set the environment with:
```bash
export ULTIMATE_ANALYSIS_ENV=development  # or production
```

## Development

### Development Setup

1. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Run tests**:
   ```bash
   pytest
   ```

3. **Code formatting**:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

### Development Guidelines

- Follow the **KISS principle**: Keep implementations simple and focused
- **File size limit**: Maximum 500 lines per file
- **Type hints**: Required for all function signatures
- **Documentation**: Docstrings for all public functions and classes
- **Testing**: Write tests for new functionality

See `docs/DEVELOPMENT_GUIDELINES.md` for complete development standards.

### Key Development Principles

1. **Separation of Concerns**: GUI, processing, and utilities are separate
2. **Performance First**: Optimize for real-time video processing
3. **Modularity**: Easy addition of new features
4. **Error Handling**: Graceful degradation and meaningful error messages

## Models

The application uses custom-trained YOLO models for sports-specific detection:

- **Object Detection**: `yolo11l.pt` for players and disc detection
- **Field Segmentation**: `yolo11l-seg.pt` for field boundary detection
- **Player ID**: Combined YOLO + EasyOCR for jersey number recognition

Models are automatically downloaded during setup or can be placed manually in `data/models/`.

## Performance

### Optimization Tips

- Use GPU acceleration when available
- Adjust processing frequency based on computational load
- Enable model result caching for repeated analysis
- Use appropriate YOLO model size for your hardware

### Monitoring

The application includes built-in performance monitoring:
- Processing step timing
- Memory usage tracking
- FPS monitoring
- Real-time performance metrics

## Troubleshooting

### Common Issues

**Models not loading**:
- Ensure models are in `data/models/` directory
- Check model file permissions
- Verify CUDA installation for GPU models

**Poor performance**:
- Reduce video resolution or FPS
- Use smaller YOLO models (yolo11n vs yolo11l)
- Enable result caching
- Close other GPU-intensive applications

**GUI not responding**:
- Check that processing is running in background threads
- Reduce visualization complexity
- Monitor memory usage

### Logging

Logs are written to `logs/ultimate_analysis.log` by default. Adjust log level in configuration:

```yaml
app:
  log_level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow development guidelines
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](docs/LICENSE) file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review existing issues on GitHub
- Create a new issue with detailed information

---

**Note**: This application is optimized for Ultimate Frisbee analysis but can be adapted for other sports with similar requirements.
