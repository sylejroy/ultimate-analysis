"""Constants for Ultimate Analysis application.

This file contains immutable system constraints, validation bounds, and fallback defaults.
Use configuration files for runtime-configurable settings.
"""

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
    "PLAY_PAUSE": "Space",
    "PREV_VIDEO": "Left",
    "NEXT_VIDEO": "Right",
    "RESET_TRACKER": "R",
    "TOGGLE_INFERENCE": "I",
    "TOGGLE_TRACKING": "T",
    "TOGGLE_PLAYER_ID": "J",
    "TOGGLE_FIELD_SEGMENTATION": "F",
}

# YOLO class indices for Ultimate Frisbee finetuned models
# Note: Actual class names come from the model itself via model.names
YOLO_CLASSES = {
    "DISC": 0,  # Frisbee disc (finetuned model class 0)
    "PLAYER": 1,  # Ultimate player (finetuned model class 1)
    "PERSON": 0,  # Fallback mapping for standard COCO models
}

# Processing pipeline constraints
MAX_DETECTIONS_PER_FRAME = 100
MAX_TRACKS_ACTIVE = 50
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

# Performance monitoring
PERFORMANCE_MONITORING = {
    "LOG_EVERY_N_FRAMES": 10,
    "CACHE_CLEANUP_INTERVAL": 100,
    "MAX_MEMORY_MB": 2048,
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
    "model_detection": "data/models/detection/20250802_1_detection_yolo11s_object_detection.v3i.yolov8/finetune_20250802_102035/weights/best.pt",
    "model_player_detection": "data/models/detection/20250802_1_detection_yolo11s_object_detection.v3i.yolov8/finetune_20250802_102035/weights/best.pt",
    "model_disc_detection": "data/models/detection/20250913_4_detection_disc_yolo11s_object_detection_disc.v1i.yolov8/finetune_20250913_205313/weights/best.pt",
    "model_segmentation": "data/models/segmentation/20250826_1_segmentation_yolo11s-seg_field finder.v8i.yolov8/finetune_20250826_092226/weights/best.pt",
}

# Video file extensions (system constraint)
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")

# Image file extensions
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# Model file extensions
SUPPORTED_MODEL_EXTENSIONS = (".pt", ".onnx", ".engine")

# OCR languages supported
SUPPORTED_OCR_LANGUAGES = ["en", "es", "fr", "de", "it"]

# Jersey number constraints
JERSEY_NUMBER_MIN = 0
JERSEY_NUMBER_MAX = 99

# Field dimensions (Ultimate Frisbee field in meters)
FIELD_DIMENSIONS = {
    "LENGTH": 100,  # meters
    "WIDTH": 37,  # meters
    "END_ZONE": 25,  # meters
}

# Processing optimization settings
OPTIMIZATION = {
    "BATCH_SIZE_PLAYER_ID": 8,
    "FRAME_SKIP_FIELD_SEGMENTATION": 5,
    "CACHE_SIZE_DETECTIONS": 50,
    "PARALLEL_PROCESSES": 4,
}
