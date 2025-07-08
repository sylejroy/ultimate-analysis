"""
Ultimate Analysis - Ultimate Frisbee Video Analysis Application

A PyQt5-based application for real-time video analysis of Ultimate Frisbee games,
featuring object detection, tracking, player identification, and field segmentation.
"""

__version__ = "0.1.0"
__author__ = "Ultimate Analysis Team"
__email__ = "contact@ultimate-analysis.com"

# Make core components easily importable
from .core.models import *
from .core.exceptions import *

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
]
