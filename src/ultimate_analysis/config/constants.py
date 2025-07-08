"""
Application constants and default configuration.

Contains all constant values used throughout the application.
"""

from typing import Dict, Any

# Ultimate Frisbee field dimensions (in meters)
FIELD_LENGTH = 70.0  # meters
FIELD_WIDTH = 37.0   # meters
GOAL_ZONE_LENGTH = 18.0  # meters

# Player height assumption for ground plane estimation
DEFAULT_PLAYER_HEIGHT = 1.85  # meters

# Model configuration defaults
DEFAULT_DETECTION_MODEL = "data/models/finetune/object_detection_yolo11l/finetune3/weights/best.pt"
DEFAULT_FIELD_MODEL = "data/models/finetune/field_finder_yolo11x-seg/segmentation_finetune4/weights/best.pt"
DEFAULT_PLAYER_ID_MODEL = "data/models/finetune/digit_detector_yolo11m/finetune/weights/best.pt"

# Detection classes
DETECTION_CLASSES = {
    0: "disc",
    1: "player", 
    2: "referee"
}

# Performance thresholds
MIN_FPS = 10.0  # Minimum acceptable FPS
MAX_INFERENCE_TIME_MS = 100.0  # Maximum inference time in milliseconds
MAX_MEMORY_USAGE_MB = 2048  # Maximum memory usage in MB

# Video processing defaults
DEFAULT_INPUT_SIZE = 960  # Default input size for models
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.45

# Tracking parameters
MAX_TRACK_AGE = 30  # Maximum frames to keep inactive tracks
MIN_TRACK_HITS = 3  # Minimum hits before confirming track

# Player ID parameters
PLAYER_ID_CACHE_SIZE = 100  # Maximum number of cached player IDs
PLAYER_ID_UPDATE_INTERVAL = 10  # Update player ID every N frames

# GUI defaults
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080
DEFAULT_VIDEO_FOLDER = "input/dev_data"

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Default application configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "application": {
        "name": "Ultimate Analysis",
        "version": "0.1.0",
        "debug": False,
    },
    "gui": {
        "window_width": DEFAULT_WINDOW_WIDTH,
        "window_height": DEFAULT_WINDOW_HEIGHT,
        "video_folder": DEFAULT_VIDEO_FOLDER,
        "dark_mode": True,
    },
    "models": {
        "detection_model": DEFAULT_DETECTION_MODEL,
        "field_model": DEFAULT_FIELD_MODEL,
        "player_id_model": DEFAULT_PLAYER_ID_MODEL,
        "input_size": DEFAULT_INPUT_SIZE,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "iou_threshold": DEFAULT_IOU_THRESHOLD,
    },
    "tracking": {
        "method": "deepsort",
        "max_age": MAX_TRACK_AGE,
        "min_hits": MIN_TRACK_HITS,
    },
    "player_id": {
        "method": "easyocr",
        "cache_size": PLAYER_ID_CACHE_SIZE,
        "update_interval": PLAYER_ID_UPDATE_INTERVAL,
    },
    "field": {
        "length": FIELD_LENGTH,
        "width": FIELD_WIDTH,
        "goal_zone_length": GOAL_ZONE_LENGTH,
        "player_height": DEFAULT_PLAYER_HEIGHT,
    },
    "performance": {
        "min_fps": MIN_FPS,
        "max_inference_time_ms": MAX_INFERENCE_TIME_MS,
        "max_memory_usage_mb": MAX_MEMORY_USAGE_MB,
    },
    "logging": {
        "level": LOG_LEVEL,
        "format": LOG_FORMAT,
        "file": None,  # Log to console by default
    },
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed", 
        "models_dir": "data/models",
        "cache_dir": "data/cache",
    },
}
