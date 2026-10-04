"""Processing package initialization."""

import cv2

# Import main processing functions for easy access
from .field_segmentation import run_field_segmentation, set_field_model
from .inference import (
    run_inference,
    set_disc_model,
    set_player_model,
)
from .player_id import run_player_id_on_tracks
from .tracking import get_track_histories, reset_tracker, run_tracking, set_tracker_type

# Ultralytics switches OpenCV multithreading off when it is imported, to protect its
# training data loaders. Training runs in its own process, so give the app's resizing,
# warping, and mask operations all cores back.
cv2.setNumThreads(-1)

__all__ = [
    "run_inference",
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
