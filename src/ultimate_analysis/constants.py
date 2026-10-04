"""Constants for Ultimate Analysis application.

This file contains immutable system constraints, validation bounds, and fallback defaults.
Use configuration files for runtime-configurable settings.
"""

# Video processing constraints
MIN_FPS = 1
MAX_FPS = 120

# GUI constraints
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800

# Keyboard shortcuts (immutable system bindings)
SHORTCUTS = {
    "PLAY_PAUSE": "Space",
    "PREV_VIDEO": "Left",
    "NEXT_VIDEO": "Right",
    "RESET_TRACKER": "R",
    "TOGGLE_INFERENCE": "I",
    "TOGGLE_TRACKING": "T",
    "TOGGLE_PLAYER_ID": "J",
    "TOGGLE_FIELD_SEGMENTATION": "F",
}

# Processing pipeline constraints
TRACK_HISTORY_MAX_LENGTH = 100

# Color scheme for visualization (BGR format for OpenCV)
VISUALIZATION_COLORS = {
    "DETECTION_BOX": (0, 255, 0),  # Green (default)
    "TRACKING_BOX": (255, 0, 0),  # Blue
    "PLAYER_ID_BOX": (0, 255, 255),  # Yellow
    "FIELD_MASK": (0, 0, 255),  # Red
    "BACKGROUND": (30, 30, 30),  # Dark gray
    # Class-specific colors
    "DISC": (0, 255, 255),  # Bright cyan - easy to spot
    "PLAYER": (128, 128, 128),  # Subtle gray
    # Model-specific colors for differentiation
    "PLAYER_MODEL": (0, 200, 0),  # Bright green for player model detections
    "DISC_MODEL": (
        0,
        0,
        255,
    ),  # Bright red for disc model detections (changed from orange for better visibility)
}

# File system paths (relative to project root)
DEFAULT_PATHS = {
    "MODELS": "data/models",
    "PRETRAINED": "data/models/pretrained",
    "DEV_DATA": "data/processed/dev_data",
    "RAW_VIDEOS": "data/raw/videos",
    "OUTPUT": "output",
    "LOGS": "logs",
    "CACHE": "data/cache",
}

# Fallback defaults (used when configuration is not available)
FALLBACK_DEFAULTS = {
    "video_fps": 25,
    "confidence_threshold": 0.5,
    "nms_threshold": 0.45,
    "tracker_type": "deepsort",
    "model_player_detection": "data/models/detection/20250802_1_detection_yolo11s_object_detection.v3i.yolov8/finetune_20250802_102035/weights/best.pt",
    "model_disc_detection": "data/models/detection/20250913_4_detection_disc_yolo11s_object_detection_disc.v1i.yolov8/finetune_20250913_205313/weights/best.pt",
    "model_segmentation": "data/models/segmentation/20250826_1_segmentation_yolo11s-seg_field finder.v8i.yolov8/finetune_20250826_092226/weights/best.pt",
}

# Video file extensions (system constraint)
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")

# Jersey number constraints
JERSEY_NUMBER_MIN = 0
JERSEY_NUMBER_MAX = 99
