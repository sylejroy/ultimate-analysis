"""
Processing module for Ultimate Analysis application.

Contains ML inference, tracking, and video processing algorithms.
"""

# Import processing functions that can be used externally
try:
    from .inference import run_inference, set_detection_model
    _inference_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"Inference module could not be imported: {e}")
    _inference_available = False

try:
    from .tracking import run_tracking, reset_tracker
    _tracking_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"Tracking module could not be imported: {e}")
    _tracking_available = False

try:
    from . import field_segmentation
    _field_segmentation_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"Field segmentation module could not be imported: {e}")
    _field_segmentation_available = False

try:
    from .player_id import run_player_id
    _player_id_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"Player ID module could not be imported: {e}")
    _player_id_available = False

__all__ = []

# Add available functions to __all__
if _inference_available:
    __all__.extend(["run_inference", "set_detection_model"])
if _tracking_available:
    __all__.extend(["run_tracking", "reset_tracker"])
if _field_segmentation_available:
    __all__.extend(["field_segmentation"])
if _player_id_available:
    __all__.extend(["run_player_id"])

# Always include module names
__all__.extend([
    "inference",
    "tracking",
    "field_segmentation", 
    "player_id",
    "video_processor",
])
