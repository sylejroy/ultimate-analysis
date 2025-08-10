"""Field coverage estimation module.

This module analyzes detected field lines to estimate how much of the Ultimate field
is visible in the frame and where the missing boundaries would be located.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import math

from .field_types import FieldLine


@dataclass
class FieldCoverage:
    """Represents field coverage analysis results."""
    visible_width_ratio: float     # Fraction of field width visible (0.0 to 1.0)
    visible_length_ratio: float    # Fraction of field length visible (0.0 to 1.0)
    left_sideline_cut: bool        # Whether left sideline is cut by image edge
    right_sideline_cut: bool       # Whether right sideline is cut by image edge
    near_boundary_cut: bool        # Whether near field boundary is cut
    far_boundary_cut: bool         # Whether far field boundary is cut
    estimated_full_field: List[Tuple[float, float]]  # Estimated complete field corners
    confidence: float              # Confidence in coverage estimation (0.0 to 1.0)


def estimate_field_coverage(field_lines: List[FieldLine], 
                          image_shape: Tuple[int, int]) -> Optional[FieldCoverage]:
    """Estimate field coverage based on detected lines and their relationships.
    
    Args:
        field_lines: List of detected field lines
        image_shape: Shape of the image (height, width)
        
    Returns:
        FieldCoverage object with analysis results, or None if insufficient data
        
    Strategy:
        1. Analyze which lines are cut by image boundaries
        2. Use line intersections and angles to extrapolate missing boundaries
        3. Apply Ultimate field geometry constraints (100m x 37m, right angles)
        4. Estimate what fraction of the field is visible
    """
    if len(field_lines) < 2:
        print("[FIELD_COV] Insufficient field lines for coverage analysis")
        return None
    
    h, w = image_shape
    print(f"[FIELD_COV] Analyzing coverage for {len(field_lines)} lines in {w}x{h} image")
    
    # Categorize lines and analyze boundary intersections
    sidelines = []
    field_boundaries = []
    
    for line in field_lines:
        # Determine if line is cut by image boundaries
        x1, y1 = line.point1
        x2, y2 = line.point2
        
        is_cut_by_edge = _is_line_cut_by_boundary(x1, y1, x2, y2, image_shape)
        
        # Calculate line angle to classify
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            angle = 90.0
        else:
            angle = abs(math.atan2(dy, dx) * 180 / math.pi)
            if angle > 90:
                angle = 180 - angle
        
        if angle > 30:  # Diagonal lines are sidelines
            sidelines.append({
                'line': line,
                'angle': angle,
                'is_cut': is_cut_by_edge,
                'position': 'left' if (x1 + x2) / 2 < w / 2 else 'right'
            })
        else:  # Horizontal lines are field boundaries
            field_boundaries.append({
                'line': line,
                'angle': angle,
                'is_cut': is_cut_by_edge,
                'position': 'near' if (y1 + y2) / 2 > h / 2 else 'far'
            })
    
    print(f"[FIELD_COV] Found {len(sidelines)} sidelines, {len(field_boundaries)} field boundaries")
    
    # Analyze sideline coverage
    left_sideline_cut = False
    right_sideline_cut = False
    left_sideline = None
    right_sideline = None
    
    for sl in sidelines:
        if sl['position'] == 'left':
            left_sideline = sl
            left_sideline_cut = sl['is_cut']
        else:
            right_sideline = sl
            right_sideline_cut = sl['is_cut']
    
    # Analyze field boundary coverage
    near_boundary_cut = False
    far_boundary_cut = False
    near_boundary = None
    far_boundary = None
    
    for fb in field_boundaries:
        if fb['position'] == 'near':
            near_boundary = fb
            near_boundary_cut = fb['is_cut']
        else:
            far_boundary = fb
            far_boundary_cut = fb['is_cut']
    
    # Estimate field coverage ratios
    visible_width_ratio = _estimate_width_coverage(
        left_sideline, right_sideline, left_sideline_cut, right_sideline_cut, image_shape
    )
    
    visible_length_ratio = _estimate_length_coverage(
        near_boundary, far_boundary, near_boundary_cut, far_boundary_cut, image_shape
    )
    
    # Extrapolate complete field boundaries
    estimated_full_field = _extrapolate_full_field(
        sidelines, field_boundaries, image_shape
    )
    
    # Calculate confidence based on number of detected lines and their completeness
    confidence = _calculate_coverage_confidence(field_lines, sidelines, field_boundaries)
    
    coverage = FieldCoverage(
        visible_width_ratio=visible_width_ratio,
        visible_length_ratio=visible_length_ratio,
        left_sideline_cut=left_sideline_cut,
        right_sideline_cut=right_sideline_cut,
        near_boundary_cut=near_boundary_cut,
        far_boundary_cut=far_boundary_cut,
        estimated_full_field=estimated_full_field,
        confidence=confidence
    )
    
    print(f"[FIELD_COV] Coverage estimate: {visible_width_ratio:.1%} width, {visible_length_ratio:.1%} length")
    print(f"[FIELD_COV] Boundaries cut: L={left_sideline_cut}, R={right_sideline_cut}, "
          f"N={near_boundary_cut}, F={far_boundary_cut}")
    
    return coverage


def _is_line_cut_by_boundary(x1: float, y1: float, x2: float, y2: float, 
                           image_shape: Tuple[int, int]) -> bool:
    """Check if a line is cut by image boundaries.
    
    Args:
        x1, y1, x2, y2: Line endpoints
        image_shape: Shape of the image (height, width)
        
    Returns:
        True if line appears to be cut by image edge
    """
    h, w = image_shape
    edge_threshold = 10  # pixels from edge to consider "cut"
    
    # Check if either endpoint is very close to image boundary
    near_left = x1 <= edge_threshold or x2 <= edge_threshold
    near_right = x1 >= w - edge_threshold or x2 >= w - edge_threshold
    near_top = y1 <= edge_threshold or y2 <= edge_threshold
    near_bottom = y1 >= h - edge_threshold or y2 >= h - edge_threshold
    
    return near_left or near_right or near_top or near_bottom


def _estimate_width_coverage(left_sideline: Optional[Dict], right_sideline: Optional[Dict],
                           left_cut: bool, right_cut: bool, 
                           image_shape: Tuple[int, int]) -> float:
    """Estimate what fraction of field width is visible.
    
    Args:
        left_sideline, right_sideline: Sideline analysis data
        left_cut, right_cut: Whether sidelines are cut by image boundaries
        image_shape: Shape of the image (height, width)
        
    Returns:
        Estimated fraction of field width visible (0.0 to 1.0)
    """
    h, w = image_shape
    
    if not left_sideline or not right_sideline:
        # Only one sideline detected - estimate based on cut status
        if left_cut or right_cut:
            return 0.7  # Estimate 70% visible if sideline is cut
        else:
            return 0.9  # Estimate 90% visible if sideline is complete
    
    # Both sidelines detected - analyze their positions and cut status
    left_line = left_sideline['line']
    right_line = right_sideline['line']
    
    # Get average x positions of sidelines
    left_x = (left_line.point1[0] + left_line.point2[0]) / 2
    right_x = (right_line.point1[0] + right_line.point2[0]) / 2
    
    visible_width_pixels = right_x - left_x
    
    # Estimate total field width based on cut status
    if left_cut and right_cut:
        # Both cut - field extends beyond both sides
        # Use perspective analysis to estimate extension
        estimated_total_width = visible_width_pixels * 1.4  # Rough estimate
        coverage_ratio = visible_width_pixels / estimated_total_width
    elif left_cut:
        # Left cut, right complete - field extends left
        extension_factor = _estimate_extension_factor(left_sideline, image_shape)
        estimated_total_width = visible_width_pixels * (1 + extension_factor)
        coverage_ratio = visible_width_pixels / estimated_total_width
    elif right_cut:
        # Right cut, left complete - field extends right
        extension_factor = _estimate_extension_factor(right_sideline, image_shape)
        estimated_total_width = visible_width_pixels * (1 + extension_factor)
        coverage_ratio = visible_width_pixels / estimated_total_width
    else:
        # Neither cut - most of field width is visible
        coverage_ratio = 0.95
    
    return min(1.0, max(0.1, coverage_ratio))


def _estimate_length_coverage(near_boundary: Optional[Dict], far_boundary: Optional[Dict],
                            near_cut: bool, far_cut: bool,
                            image_shape: Tuple[int, int]) -> float:
    """Estimate what fraction of field length is visible.
    
    Args:
        near_boundary, far_boundary: Field boundary analysis data
        near_cut, far_cut: Whether boundaries are cut by image edges
        image_shape: Shape of the image (height, width)
        
    Returns:
        Estimated fraction of field length visible (0.0 to 1.0)
    """
    h, w = image_shape
    
    if not near_boundary or not far_boundary:
        # Only one boundary detected
        if near_cut or far_cut:
            return 0.6  # Estimate 60% visible if boundary is cut
        else:
            return 0.8  # Estimate 80% visible if boundary is complete
    
    # Both boundaries detected - analyze their positions
    near_line = near_boundary['line']
    far_line = far_boundary['line']
    
    # Get average y positions
    near_y = (near_line.point1[1] + near_line.point2[1]) / 2
    far_y = (far_line.point1[1] + far_line.point2[1]) / 2
    
    visible_length_pixels = near_y - far_y  # Near should be larger y (bottom of image)
    
    # Estimate coverage based on cut status
    if near_cut and far_cut:
        # Both cut - significant field extension beyond frame
        estimated_total_length = visible_length_pixels * 1.6
        coverage_ratio = visible_length_pixels / estimated_total_length
    elif near_cut:
        # Near cut (bottom of image) - field extends toward camera
        coverage_ratio = 0.7  # Common in drone footage
    elif far_cut:
        # Far cut (top of image) - field extends away from camera
        coverage_ratio = 0.8  # Less common but possible
    else:
        # Neither cut - most of field length visible
        coverage_ratio = 0.9
    
    return min(1.0, max(0.2, coverage_ratio))


def _estimate_extension_factor(sideline_data: Dict, image_shape: Tuple[int, int]) -> float:
    """Estimate how much a cut sideline extends beyond the image.
    
    Args:
        sideline_data: Sideline analysis data
        image_shape: Shape of the image (height, width)
        
    Returns:
        Extension factor (0.0 to 1.0) representing fraction of additional field
    """
    # Use line angle and position to estimate perspective extension
    angle = sideline_data['angle']
    line = sideline_data['line']
    
    # Steeper angles suggest more perspective distortion
    if angle > 50:
        return 0.3  # High perspective - moderate extension
    elif angle > 35:
        return 0.2  # Medium perspective - small extension
    else:
        return 0.1  # Low perspective - minimal extension


def _extrapolate_full_field(sidelines: List[Dict], field_boundaries: List[Dict],
                          image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
    """Extrapolate complete field corner positions.
    
    Args:
        sidelines: List of sideline analysis data
        field_boundaries: List of field boundary analysis data
        image_shape: Shape of the image (height, width)
        
    Returns:
        List of estimated complete field corners
    """
    h, w = image_shape
    corners = []
    
    # Use detected lines to estimate corners
    # This is a simplified approach - could be enhanced with full perspective analysis
    
    if len(sidelines) >= 2 and len(field_boundaries) >= 2:
        # Full line detection - use intersections
        left_sl = next((sl for sl in sidelines if sl['position'] == 'left'), None)
        right_sl = next((sl for sl in sidelines if sl['position'] == 'right'), None)
        near_fb = next((fb for fb in field_boundaries if fb['position'] == 'near'), None)
        far_fb = next((fb for fb in field_boundaries if fb['position'] == 'far'), None)
        
        if all([left_sl, right_sl, near_fb, far_fb]):
            # Calculate intersections
            corners = [
                _find_line_intersection(left_sl['line'], near_fb['line']),   # Bottom-left
                _find_line_intersection(right_sl['line'], near_fb['line']),  # Bottom-right
                _find_line_intersection(right_sl['line'], far_fb['line']),   # Top-right
                _find_line_intersection(left_sl['line'], far_fb['line'])     # Top-left
            ]
            # Filter out None intersections
            corners = [c for c in corners if c is not None]
    
    # Fallback to image boundary estimation if insufficient intersections
    if len(corners) < 4:
        margin = 50
        corners = [
            (margin, h - margin),        # Bottom-left
            (w - margin, h - margin),    # Bottom-right
            (w - margin, margin),        # Top-right
            (margin, margin)             # Top-left
        ]
    
    return corners


def _find_line_intersection(line1: FieldLine, line2: FieldLine) -> Optional[Tuple[float, float]]:
    """Find intersection point between two field lines."""
    x1, y1 = line1.point1
    x2, y2 = line1.point2
    x3, y3 = line2.point1
    x4, y4 = line2.point2
    
    # Calculate intersection using line equations
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:  # Lines are parallel
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    # Calculate intersection point
    intersect_x = x1 + t * (x2 - x1)
    intersect_y = y1 + t * (y2 - y1)
    
    return (intersect_x, intersect_y)


def _calculate_coverage_confidence(field_lines: List[FieldLine], 
                                 sidelines: List[Dict], 
                                 field_boundaries: List[Dict]) -> float:
    """Calculate confidence in coverage estimation.
    
    Args:
        field_lines: All detected field lines
        sidelines: Analyzed sideline data
        field_boundaries: Analyzed field boundary data
        
    Returns:
        Confidence score (0.0 to 1.0)
    """
    base_confidence = 0.3
    
    # Bonus for number of detected lines
    line_bonus = min(0.4, len(field_lines) * 0.1)
    
    # Bonus for having both sidelines
    sideline_bonus = 0.2 if len(sidelines) >= 2 else 0.1 if len(sidelines) >= 1 else 0.0
    
    # Bonus for having field boundaries
    boundary_bonus = 0.2 if len(field_boundaries) >= 2 else 0.1 if len(field_boundaries) >= 1 else 0.0
    
    # Penalty for many cut lines (uncertainty)
    cut_count = sum(1 for sl in sidelines if sl['is_cut']) + sum(1 for fb in field_boundaries if fb['is_cut'])
    cut_penalty = min(0.3, cut_count * 0.1)
    
    confidence = base_confidence + line_bonus + sideline_bonus + boundary_bonus - cut_penalty
    
    return min(1.0, max(0.1, confidence))
