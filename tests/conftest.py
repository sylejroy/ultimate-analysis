"""
Test configuration for Ultimate Analysis test suite.

Provides fixtures and configuration for pytest tests.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np

from src.ultimate_analysis.core.models import (
    BoundingBox, Detection, Track, DetectionClass
)


@pytest.fixture
def sample_bbox() -> BoundingBox:
    """Create a sample bounding box for testing."""
    return BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0)


@pytest.fixture
def sample_detection(sample_bbox: BoundingBox) -> Detection:
    """Create a sample detection for testing."""
    return Detection(
        bbox=sample_bbox,
        confidence=0.85,
        class_id=1,
        class_name=DetectionClass.PLAYER,
        frame_id=0
    )


@pytest.fixture
def sample_track(sample_detection: Detection) -> Track:
    """Create a sample track for testing."""
    return Track(
        track_id=1,
        detections=[sample_detection],
        created_frame=0,
        last_seen_frame=0
    )


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config() -> dict:
    """Create a sample configuration for testing."""
    return {
        "models": {
            "detection_model": "test_model.pt",
            "confidence_threshold": 0.5,
        },
        "tracking": {
            "method": "deepsort",
            "max_age": 30,
        },
        "logging": {
            "level": "DEBUG",
        },
    }
