"""Constants for Ultimate Analysis - Immutable system constraints and validation bounds."""

# File size constraints (as per copilot instructions)
MAX_FILE_SIZE_LINES = 500  # Maximum lines per file (KISS principle)

# Video processing constraints
MIN_FPS = 1
MAX_FPS = 120
MIN_FRAME_DIMENSION = 32
MAX_FRAME_DIMENSION = 4096

# Model inference constraints
MIN_CONFIDENCE_THRESHOLD = 0.01
MAX_CONFIDENCE_THRESHOLD = 1.0
MIN_NMS_THRESHOLD = 0.01
MAX_NMS_THRESHOLD = 1.0

# Tracking constraints
MIN_MAX_TRACKS = 1
MAX_MAX_TRACKS = 1000
MIN_TRACK_HISTORY = 1
MAX_TRACK_HISTORY = 1000

# GUI constraints
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600
MAX_WINDOW_WIDTH = 7680  # 8K width
MAX_WINDOW_HEIGHT = 4320  # 8K height

# File format constants
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Color constants (BGR format for OpenCV)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# Default fallback values (when configuration is unavailable)
DEFAULT_FPS = 25
DEFAULT_CONFIDENCE = 0.5
DEFAULT_NMS_THRESHOLD = 0.45
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080

# Logging levels
LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
