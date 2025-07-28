"""Processing package initialization."""

# Import main processing functions for easy access
from .inference import run_inference, set_detection_model
from .tracking import run_tracking, reset_tracker, set_tracker_type
from .player_id import run_player_id, set_player_id_method
from .field_segmentation import run_field_segmentation, set_field_model

__all__ = [
    'run_inference',
    'set_detection_model', 
    'run_tracking',
    'reset_tracker',
    'set_tracker_type',
    'run_player_id',
    'set_player_id_method',
    'run_field_segmentation',
    'set_field_model'
]
