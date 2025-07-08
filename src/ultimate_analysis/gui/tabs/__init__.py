"""
GUI tabs module.

Contains tab implementations for the main application interface.
"""
from .main_tab import MainTab
from .dev_video_preprocessing_tab import DevVideoPreprocessingTab
from .dev_yolo_training_tab import DevYoloTrainingTab
from .easyocr_tuning_tab import DevEasyOCRTuningTab

__all__ = [
    "MainTab",
    "DevVideoPreprocessingTab", 
    "DevYoloTrainingTab",
    "DevEasyOCRTuningTab",
    "main_tab",
    "preprocessing_tab",
    "training_tab",
    "tuning_tab",
]
