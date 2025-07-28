"""Constants for Ultimate Analysis application.

This file contains immutable system constraints, validation bounds, and fallback defaults.
Use configuration files for runtime-configurable settings.
"""

from typing import List, Tuple

# File size constraints (enforced during development)
MAX_FILE_SIZE_LINES = 500

# Video processing constraints
MIN_FPS = 1
MAX_FPS = 120
MIN_FRAME_SKIP = 1
MAX_FRAME_SKIP = 10

# Model validation bounds
MIN_CONFIDENCE_THRESHOLD = 0.1
MAX_CONFIDENCE_THRESHOLD = 1.0
MIN_NMS_THRESHOLD = 0.1
MAX_NMS_THRESHOLD = 1.0

# GUI constraints
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800

# Keyboard shortcuts (immutable system bindings)
SHORTCUTS = {
    'PLAY_PAUSE': 'Space',
    'PREV_VIDEO': 'Left',
    'NEXT_VIDEO': 'Right',
    'RESET_TRACKER': 'R',
    'TOGGLE_INFERENCE': 'I',
    'TOGGLE_TRACKING': 'T',
    'TOGGLE_PLAYER_ID': 'J',
    'TOGGLE_FIELD_SEGMENTATION': 'F'
}

# YOLO class indices (standard COCO classes relevant to Ultimate Frisbee)
YOLO_CLASSES = {
    'PERSON': 0,
    'PLAYER': 1,  # Custom class for trained models
    'DISC': 2,    # Custom class for trained models
}

# Processing pipeline constraints
MAX_DETECTIONS_PER_FRAME = 100
MAX_TRACKS_ACTIVE = 50
TRACK_HISTORY_MAX_LENGTH = 100

# Color scheme for visualization (BGR format for OpenCV)
VISUALIZATION_COLORS = {
    'DETECTION_BOX': (0, 255, 0),      # Green
    'TRACKING_BOX': (255, 0, 0),       # Blue
    'PLAYER_ID_BOX': (0, 255, 255),    # Yellow
    'FIELD_MASK': (0, 0, 255),         # Red
    'BACKGROUND': (30, 30, 30),        # Dark gray
}

# Performance monitoring
PERFORMANCE_MONITORING = {
    'LOG_EVERY_N_FRAMES': 10,
    'CACHE_CLEANUP_INTERVAL': 100,
    'MAX_MEMORY_MB': 2048,
}

# File system paths (relative to project root)
DEFAULT_PATHS = {
    'MODELS': 'data/models',
    'PRETRAINED': 'data/models/pretrained',
    'DEV_DATA': 'data/processed/dev_data',
    'RAW_VIDEOS': 'data/raw/videos',
    'OUTPUT': 'output',
    'LOGS': 'logs',
    'CACHE': 'data/cache',
}

# Fallback defaults (used when configuration is not available)
FALLBACK_DEFAULTS = {
    'video_fps': 25,
    'confidence_threshold': 0.5,
    'nms_threshold': 0.45,
    'tracker_type': 'deepsort',
    'model_detection': 'yolo11l.pt',
    'model_segmentation': 'yolo11l-seg.pt',
}

# Video file extensions (system constraint)
SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')

# Image file extensions  
SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

# Model file extensions
SUPPORTED_MODEL_EXTENSIONS = ('.pt', '.onnx', '.engine')

# OCR languages supported
SUPPORTED_OCR_LANGUAGES = ['en', 'es', 'fr', 'de', 'it']

# Jersey number constraints
JERSEY_NUMBER_MIN = 0
JERSEY_NUMBER_MAX = 99

# Field dimensions (Ultimate Frisbee field in meters)
FIELD_DIMENSIONS = {
    'LENGTH': 100,      # meters
    'WIDTH': 37,        # meters
    'END_ZONE': 25,     # meters
}

# Processing optimization settings
OPTIMIZATION = {
    'BATCH_SIZE_PLAYER_ID': 8,
    'FRAME_SKIP_FIELD_SEGMENTATION': 5,
    'CACHE_SIZE_DETECTIONS': 50,
    'PARALLEL_PROCESSES': 4,
}
