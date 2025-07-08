"""
Custom exceptions for Ultimate Analysis application.

Defines application-specific exceptions for better error handling
and debugging throughout the system.
"""

from typing import Optional, Any


class UltimateAnalysisError(Exception):
    """Base exception class for Ultimate Analysis application."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class VideoProcessingError(UltimateAnalysisError):
    """Raised when video processing operations fail."""
    pass


class ModelLoadError(UltimateAnalysisError):
    """Raised when ML model loading fails."""
    pass


class InferenceError(UltimateAnalysisError):
    """Raised when model inference fails."""
    pass


class TrackingError(UltimateAnalysisError):
    """Raised when object tracking operations fail."""
    pass


class CalibrationError(UltimateAnalysisError):
    """Raised when ground plane calibration fails."""
    pass


class PlayerIDError(UltimateAnalysisError):
    """Raised when player identification fails."""
    pass


class FieldSegmentationError(UltimateAnalysisError):
    """Raised when field segmentation operations fail."""
    pass


class ConfigurationError(UltimateAnalysisError):
    """Raised when configuration is invalid or missing."""
    pass


class FileNotFoundError(UltimateAnalysisError):
    """Raised when required files are not found."""
    pass


class ValidationError(UltimateAnalysisError):
    """Raised when data validation fails."""
    pass


class PerformanceError(UltimateAnalysisError):
    """Raised when performance requirements are not met."""
    pass
