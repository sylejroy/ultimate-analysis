"""
Core utilities for Ultimate Analysis application.

Contains common utility functions used across the application.
"""

import os
import time
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np


def setup_logging(
    level: str = "INFO", 
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        filename=log_file
    )
    
    return logging.getLogger("ultimate_analysis")


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root
    """
    return Path(__file__).parent.parent.parent.parent


def validate_file_exists(file_path: Union[str, Path]) -> Path:
    """
    Validate that a file exists and return Path object.
    
    Args:
        file_path: Path to validate
        
    Returns:
        Path object if file exists
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path_obj


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate frames per second based on elapsed time and frame count.
    
    Args:
        start_time: Start timestamp
        frame_count: Number of frames processed
        
    Returns:
        Calculated FPS
    """
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0.0


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_value: Minimum bound
        max_value: Maximum bound
        
    Returns:
        Clamped value
    """
    return max(min_value, min(max_value, value))


def normalize_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """
    Normalize bounding box coordinates to [0, 1] range.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Normalized bounding box coordinates
    """
    x1, y1, x2, y2 = bbox
    return [
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height
    ]


def denormalize_bbox(
    normalized_bbox: List[float], 
    image_width: int, 
    image_height: int
) -> List[float]:
    """
    Denormalize bounding box coordinates from [0, 1] range to pixel coordinates.
    
    Args:
        normalized_bbox: Normalized bounding box as [x1, y1, x2, y2]
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Denormalized bounding box coordinates
    """
    x1, y1, x2, y2 = normalized_bbox
    return [
        x1 * image_width,
        y1 * image_height,
        x2 * image_width,
        y2 * image_height
    ]


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box as [x1, y1, x2, y2]
        bbox2: Second bounding box as [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Calculate intersection area
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def moving_average(values: List[float], window_size: int) -> List[float]:
    """
    Calculate moving average of a list of values.
    
    Args:
        values: List of values
        window_size: Size of the moving window
        
    Returns:
        List of moving averages
    """
    if len(values) < window_size:
        return values
    
    averages = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        averages.append(sum(window) / window_size)
    
    return averages


def format_runtime(runtime_ms: float) -> str:
    """
    Format runtime in milliseconds to human-readable string.
    
    Args:
        runtime_ms: Runtime in milliseconds
        
    Returns:
        Formatted runtime string
    """
    if runtime_ms < 1000:
        return f"{runtime_ms:.1f}ms"
    elif runtime_ms < 60000:
        return f"{runtime_ms / 1000:.2f}s"
    else:
        minutes = int(runtime_ms // 60000)
        seconds = (runtime_ms % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"
