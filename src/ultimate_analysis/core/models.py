"""
Data models for Ultimate Analysis application using Pydantic.

Contains all data structures used throughout the application for
validation, serialization, and type safety.
"""

import os
import time
from typing import List, Dict, Optional, Tuple, Union, Any
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, validator


class DetectionClass(str, Enum):
    """Enumeration of detectable object classes."""
    PLAYER = "player"
    DISC = "disc"
    REFEREE = "referee"
    UNKNOWN = "unknown"


class TrackingMethod(str, Enum):
    """Available tracking methods."""
    DEEPSORT = "deepsort"
    HISTOGRAM = "histogram"


class PlayerIDMethod(str, Enum):
    """Available player identification methods."""
    YOLO = "yolo"
    EASYOCR = "easyocr"


class BoundingBox(BaseModel):
    """Represents a bounding box in image coordinates."""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate") 
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    
    @validator('x2')
    def x2_must_be_greater_than_x1(cls, v, values):
        if 'x1' in values and v <= values['x1']:
            raise ValueError('x2 must be greater than x1')
        return v
    
    @validator('y2')
    def y2_must_be_greater_than_y1(cls, v, values):
        if 'y1' in values and v <= values['y1']:
            raise ValueError('y2 must be greater than y1')
        return v
    
    @property
    def width(self) -> float:
        """Calculate bounding box width."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Calculate bounding box height."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Detection(BaseModel):
    """Represents an object detection result."""
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    class_id: int = Field(..., ge=0, description="Class ID from model")
    class_name: DetectionClass = Field(default=DetectionClass.UNKNOWN)
    frame_id: Optional[int] = Field(None, description="Frame number where detected")
    timestamp: Optional[float] = Field(None, description="Detection timestamp")
    
    class Config:
        use_enum_values = True


class Track(BaseModel):
    """Represents a tracked object across multiple frames."""
    track_id: int = Field(..., description="Unique track identifier")
    detections: List[Detection] = Field(default_factory=list)
    is_active: bool = Field(default=True)
    created_frame: int = Field(..., description="Frame where track was created")
    last_seen_frame: int = Field(..., description="Last frame where track was updated")
    
    @property
    def current_detection(self) -> Optional[Detection]:
        """Get the most recent detection for this track."""
        return self.detections[-1] if self.detections else None
    
    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        """Get the trajectory as a list of center points."""
        return [det.bbox.center for det in self.detections]


class PlayerID(BaseModel):
    """Represents player identification result."""
    player_number: str = Field(..., description="Detected player number")
    confidence: float = Field(..., ge=0.0, le=1.0)
    method: PlayerIDMethod
    details: Optional[Dict[str, Any]] = Field(None, description="Method-specific details")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box of detected number")


class FieldPosition(BaseModel):
    """Represents a position on the Ultimate field in real-world coordinates."""
    x: float = Field(..., description="X coordinate in meters (field length)")
    y: float = Field(..., description="Y coordinate in meters (field width)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    @validator('x')
    def x_within_field_bounds(cls, v):
        # Ultimate field is 70m long
        if not -35 <= v <= 35:
            raise ValueError('X coordinate must be within field bounds (-35m to 35m)')
        return v
    
    @validator('y')
    def y_within_field_bounds(cls, v):
        # Ultimate field is 37m wide
        if not -18.5 <= v <= 18.5:
            raise ValueError('Y coordinate must be within field bounds (-18.5m to 18.5m)')
        return v


class GroundPlaneCalibration(BaseModel):
    """Represents ground plane calibration data."""
    scale_factor: float = Field(..., description="Pixels per meter")
    perspective_model: Dict[str, Any] = Field(..., description="Perspective correction parameters")
    reference_height: float = Field(default=1.85, description="Reference height in meters")
    calibration_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    frame_id: int = Field(..., description="Frame used for calibration")
    timestamp: float = Field(default_factory=time.time)


class ProcessingStats(BaseModel):
    """Represents processing performance statistics."""
    step_name: str
    runtime_ms: float = Field(..., ge=0.0)
    frame_id: int
    timestamp: float = Field(default_factory=time.time)
    memory_usage_mb: Optional[float] = Field(None, ge=0.0)
    
    class Config:
        # Allow for numpy types
        arbitrary_types_allowed = True


class VideoMetadata(BaseModel):
    """Represents video file metadata."""
    file_path: str
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fps: float = Field(..., gt=0.0)
    total_frames: int = Field(..., ge=0)
    duration_seconds: float = Field(..., ge=0.0)
    codec: Optional[str] = None
    
    @validator('file_path')
    def file_must_exist(cls, v):
        if not os.path.exists(v):
            raise ValueError(f'Video file does not exist: {v}')
        return v


class AnalysisResult(BaseModel):
    """Represents the complete analysis result for a frame."""
    frame_id: int
    timestamp: float
    detections: List[Detection] = Field(default_factory=list)
    tracks: List[Track] = Field(default_factory=list)
    player_ids: Dict[int, PlayerID] = Field(default_factory=dict)  # track_id -> PlayerID
    field_positions: Dict[int, FieldPosition] = Field(default_factory=dict)  # track_id -> FieldPosition
    ground_plane_calibration: Optional[GroundPlaneCalibration] = None
    processing_stats: List[ProcessingStats] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


# Type aliases for commonly used types
ImageArray = np.ndarray
BBoxArray = np.ndarray  # Shape: (N, 4) for N bounding boxes
ConfidenceArray = np.ndarray  # Shape: (N,) for N confidence scores
