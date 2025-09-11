"""Processing package initialization."""

# Import main processing functions for easy access
from .inference import run_inference, set_detection_model
from .tracking import run_tracking, reset_tracker, set_tracker_type, get_track_histories
from .player_id import run_player_id_on_tracks
from .field_segmentation import run_field_segmentation, set_field_model
from .async_processor import AsyncVideoProcessor, ProcessingResult, ProcessingTask, ProcessingTaskType

__all__ = [
    'run_inference',
    'set_detection_model', 
    'run_tracking',
    'reset_tracker',
    'set_tracker_type',
    'get_track_histories',
    'run_player_id_on_tracks',
    'run_field_segmentation',
    'set_field_model',
    'AsyncVideoProcessor',
    'ProcessingResult',
    'ProcessingTask', 
    'ProcessingTaskType'
]
