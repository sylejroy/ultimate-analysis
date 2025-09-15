"""Processing package initialization."""

# Import main processing functions for easy access
from .field_segmentation import run_field_segmentation, set_field_model
from .inference import (
    run_inference,
    set_detection_model,
    set_disc_model,
    set_player_model,
)
from .player_id import run_player_id_on_tracks
from .tracking import get_track_histories, reset_tracker, run_tracking, set_tracker_type

__all__ = [
    "run_inference",
    "set_detection_model",  # Deprecated but kept for compatibility
    "set_player_model",
    "set_disc_model",
    "run_tracking",
    "reset_tracker",
    "set_tracker_type",
    "get_track_histories",
    "run_player_id_on_tracks",
    "run_field_segmentation",
    "set_field_model",
]
