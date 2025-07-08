"""
GUI components module.

Contains reusable UI components like video player, controls, and dialogs.
"""
from .video_player import VideoPlayer
from .runtime_dialog import RuntimesDialog

__all__ = [
    "VideoPlayer",
    "RuntimesDialog",
    "video_player",
    "controls",
    "dialogs",
]
