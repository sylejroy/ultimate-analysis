"""Shared data types for field processing modules.

This module contains data structures used across multiple field processing modules
to avoid circular import dependencies.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class FieldLine:
    """Represents a field line with its endpoints and properties."""
    line_type: str  # 'left_sideline', 'right_sideline', 'close_field', 'far_field'
    point1: Tuple[float, float]  # (x, y) start point
    point2: Tuple[float, float]  # (x, y) end point
    confidence: float = 1.0  # Confidence score for the line detection
    visible: bool = True  # Whether line is fully/partially visible in image
    extrapolated: bool = False  # Whether line extends beyond image bounds
    
    def length(self) -> float:
        """Calculate the length of the field line."""
        x1, y1 = self.point1
        x2, y2 = self.point2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def angle(self) -> float:
        """Calculate the angle of the field line in degrees."""
        import math
        x1, y1 = self.point1
        x2, y2 = self.point2
        return math.degrees(math.atan2(y2 - y1, x2 - x1))
    
    def midpoint(self) -> Tuple[float, float]:
        """Calculate the midpoint of the field line."""
        x1, y1 = self.point1
        x2, y2 = self.point2
        return ((x1 + x2) / 2, (y1 + y2) / 2)
