"""Field projection module - unified field segmentation and line estimation.

This module handles:
1. Unifying overlapping field segmentation masks into a single unified field
2. Extracting field lines (sidelines, close/far field lines) from the unified segmentation
3. Handling partial field coverage from drone footage
4. Estimating field position even when parts extend beyond image bounds
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import math

from ..config.settings import get_setting
from ..constants import FIELD_DIMENSIONS


@dataclass
class FieldLine:
    """Represents a field line with its endpoints and properties."""
    line_type: str  # 'left_sideline', 'right_sideline', 'close_field', 'far_field'
    point1: Tuple[float, float]  # (x, y) start point
    point2: Tuple[float, float]  # (x, y) end point
    confidence: float  # Confidence in line detection (0.0 to 1.0)
    visible: bool  # Whether line is fully/partially visible in image
    extrapolated: bool  # Whether line extends beyond image bounds


@dataclass
class UnifiedField:
    """Represents a unified field segmentation with estimated lines."""
    mask: np.ndarray  # Unified binary field mask
    lines: List[FieldLine]  # Detected/estimated field lines
    field_corners: List[Tuple[float, float]]  # Estimated field corner points
    coverage_ratio: float  # Fraction of field visible (0.0 to 1.0)
    image_bounds: Tuple[int, int]  # (width, height) of source image


def unify_field_segmentation(segmentation_results: List[Any], 
                           frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Unify multiple overlapping field segmentation masks into a single mask.
    
    Args:
        segmentation_results: Raw YOLO segmentation results from field_segmentation module
        frame_shape: Shape of the frame (height, width)
        
    Returns:
        Unified binary field mask, or None if no field detected
        
    Implementation:
        - Combines all field masks using logical OR
        - Applies morphological operations to fill gaps and smooth boundaries
        - Uses connected components to keep only the largest field region
        - Performs extensive denoising with erosion/dilation operations
    """
    if not segmentation_results:
        return None
        
    print(f"[FIELD_PROJ] Unifying {len(segmentation_results)} field segmentation results")
    
    # Initialize unified mask
    unified_mask = np.zeros(frame_shape, dtype=np.uint8)
    mask_count = 0
    
    # Combine all field masks with weighted averaging for overlapping regions
    accumulated_mask = np.zeros(frame_shape, dtype=np.float32)
    
    for result in segmentation_results:
        if not hasattr(result, 'masks') or result.masks is None:
            continue
            
        try:
            # Get mask data
            if hasattr(result.masks, 'data'):
                mask_data = result.masks.data
            else:
                continue
                
            # Convert to numpy if needed
            if hasattr(mask_data, 'cpu'):
                masks = mask_data.cpu().numpy()
            else:
                masks = mask_data.numpy() if hasattr(mask_data, 'numpy') else mask_data
            
            # Process each mask
            for mask in masks:
                # Resize mask to match frame shape if necessary
                if mask.shape != frame_shape:
                    mask_resized = cv2.resize(
                        mask.astype(np.float32), 
                        (frame_shape[1], frame_shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask_resized = mask.astype(np.float32)
                
                # Accumulate mask values (handles overlapping regions better)
                accumulated_mask += mask_resized
                mask_count += 1
                
        except Exception as e:
            print(f"[FIELD_PROJ] Error processing segmentation result: {e}")
            continue
    
    if mask_count == 0:
        print("[FIELD_PROJ] No valid field masks found in segmentation results")
        return None
    
    # Convert accumulated mask to binary (threshold at 0.5 for overlapping regions)
    unified_mask = (accumulated_mask > 0.5).astype(np.uint8)
    
    if np.sum(unified_mask) == 0:
        print("[FIELD_PROJ] No field pixels found after unification")
        return None
    
    print(f"[FIELD_PROJ] Combined {mask_count} masks, {np.sum(unified_mask)} initial field pixels")
    
    # Apply morphological operations to clean up the mask
    unified_mask = _apply_morphological_cleanup(unified_mask)
    
    # Keep only the largest connected component (main field area)
    unified_mask = _extract_largest_component(unified_mask)
    
    print(f"[FIELD_PROJ] Unified field mask created with {np.sum(unified_mask)} field pixels after cleanup")
    return unified_mask


def _fit_field_polygon(field_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Fit a polygon to the field mask boundary and extract field lines.
    
    Args:
        field_mask: Binary mask of field area
        
    Returns:
        List of line tuples (x1, y1, x2, y2) representing polygon edges
    """
    # Find contours of the field mask
    contours, _ = cv2.findContours(
        field_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        print("[FIELD_PROJ] No contours found in field mask")
        return []
    
    # Get the largest contour (should be the main field boundary)
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 4:
        print("[FIELD_PROJ] Contour too small for polygon fitting")
        return []
    
    print(f"[FIELD_PROJ] Found contour with {len(largest_contour)} points")
    
    # Approximate the contour with a polygon
    # Start with a reasonable epsilon value (2% of perimeter)
    epsilon = get_setting("field_projection.polygon_epsilon", 0.02) * cv2.arcLength(largest_contour, True)
    polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    print(f"[FIELD_PROJ] Polygon approximation has {len(polygon)} vertices")
    
    if len(polygon) < 3:
        print("[FIELD_PROJ] Polygon too simple (less than 3 vertices)")
        return []
    
    # Convert polygon vertices to line segments
    lines = []
    for i in range(len(polygon)):
        # Get current and next vertex (wrapping around)
        p1 = polygon[i][0]
        p2 = polygon[(i + 1) % len(polygon)][0]
        
        # Create line tuple (x1, y1, x2, y2)
        line = (int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
        lines.append(line)
        
        # Debug: print each polygon edge
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            angle = 90.0
        else:
            angle = abs(math.atan2(dy, dx) * 180 / math.pi)
        print(f"[FIELD_PROJ] Polygon edge {i}: ({x1},{y1})-({x2},{y2}), length={length:.1f}, angle={angle:.1f}°")
    
    print(f"[FIELD_PROJ] Extracted {len(lines)} polygon edges as potential field lines")
    
    # Filter out very short lines and edge lines that are likely noise
    min_line_length = get_setting("field_projection.min_polygon_line_length", 50)
    edge_margin = get_setting("field_projection.edge_margin", 5)
    h, w = field_mask.shape
    filtered_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Check if line is too short
        if length < min_line_length:
            print(f"[FIELD_PROJ] Filtered out short line: length={length:.1f}")
            continue
            
        # For polygon edges from field segmentation, be more permissive with edge filtering
        # Only filter lines that are VERY close to edges (likely true artifacts)
        # Use a smaller margin for polygon-based detection since these are real field boundaries
        strict_margin = 3  # Much more permissive for polygon edges
        if (_is_line_at_edge(x1, y1, x2, y2, w, h, strict_margin)):
            print(f"[FIELD_PROJ] Filtered out edge artifact: ({x1},{y1})-({x2},{y2}) - within {strict_margin}px of image boundary")
            continue
            
        print(f"[FIELD_PROJ] Kept field boundary: ({x1},{y1})-({x2},{y2}), length={length:.1f}")
        filtered_lines.append(line)
    
    print(f"[FIELD_PROJ] {len(filtered_lines)} lines after length and edge filtering")
    
    return filtered_lines


def _is_line_at_edge(x1: int, y1: int, x2: int, y2: int, w: int, h: int, margin: int) -> bool:
    """Check if a line is an edge artifact (close to edge AND parallel to edge).
    
    Args:
        x1, y1, x2, y2: Line endpoints
        w, h: Image dimensions
        margin: Distance from edge to consider as "at edge"
        
    Returns:
        True if line is an edge artifact (close to edge AND parallel to edge), False otherwise
    """
    # Calculate line angle
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        angle = 90.0
    else:
        angle = abs(math.atan2(dy, dx) * 180 / math.pi)
    
    # Threshold for considering a line parallel to edges (nearly horizontal or vertical)
    parallel_threshold = 15.0  # degrees
    
    # Check if line is nearly horizontal (parallel to top/bottom edges)
    is_horizontal = angle <= parallel_threshold or angle >= (180 - parallel_threshold)
    
    # Check if line is nearly vertical (parallel to left/right edges)  
    is_vertical = abs(angle - 90) <= parallel_threshold
    
    # Only filter lines that are BOTH near edges AND parallel to those edges
    
    # Check top/bottom edges (horizontal lines)
    if is_horizontal:
        if (y1 <= margin and y2 <= margin) or (y1 >= h - margin and y2 >= h - margin):
            return True
    
    # Check left/right edges (vertical lines)
    if is_vertical:
        if (x1 <= margin and x2 <= margin) or (x1 >= w - margin and x2 >= w - margin):
            return True
    
    # If line is near edge but NOT parallel to it, it's probably a real field line
    return False


def estimate_field_lines(unified_mask: np.ndarray, original_frame: np.ndarray = None) -> List[FieldLine]:
    """Estimate field lines from unified field segmentation mask using polygon fitting.
    
    Args:
        unified_mask: Unified binary field mask
        original_frame: Original video frame (not used in polygon approach)
        
    Returns:
        List of estimated field lines (sidelines and field lines)
        
    Implementation:
        - Finds the boundary contour of the unified field mask
        - Fits a polygon to approximate the field boundary
        - Identifies which polygon edges correspond to field lines based on perspective
        - Returns the major field boundary lines (sidelines and field lines)
    """
    if unified_mask is None or np.sum(unified_mask) == 0:
        return []
        
    print("[FIELD_PROJ] Estimating field lines from unified mask using polygon fitting")
    
    # Get configuration parameters
    edge_margin = get_setting("field_projection.edge_margin", 10)
    
    # Create a mask that excludes edge regions to avoid boundary artifacts
    h, w = unified_mask.shape
    edge_free_mask = unified_mask.copy()
    edge_free_mask[:edge_margin, :] = 0  # Top edge
    edge_free_mask[-edge_margin:, :] = 0  # Bottom edge
    edge_free_mask[:, :edge_margin] = 0  # Left edge
    edge_free_mask[:, -edge_margin:] = 0  # Right edge

    # Find the field boundary using polygon fitting
    field_lines = _fit_field_polygon(edge_free_mask)
    
    if not field_lines:
        print("[FIELD_PROJ] No field lines detected from polygon fitting")
        return []
    
    print(f"[FIELD_PROJ] Polygon fitting detected {len(field_lines)} field lines")
    
    # Classify the lines based on perspective and position
    classified_lines = _classify_field_lines_perspective(field_lines, (h, w))
    
    print(f"[FIELD_PROJ] Classified {len(classified_lines)} field lines")
    
    return classified_lines


def create_unified_field(segmentation_results: List[Any], 
                        frame_shape: Tuple[int, int],
                        original_frame: np.ndarray = None) -> Optional[UnifiedField]:
    """Create a complete unified field representation with lines and metadata.
    
    Args:
        segmentation_results: Raw YOLO segmentation results
        frame_shape: Shape of the frame (height, width)
        original_frame: Original video frame for line detection (optional)
        
    Returns:
        UnifiedField object with mask, lines, and metadata, or None if failed
    """
    print(f"[FIELD_PROJ] Creating unified field from {len(segmentation_results)} segmentation results")
    
    # Step 1: Unify field segmentation masks
    unified_mask = unify_field_segmentation(segmentation_results, frame_shape)
    if unified_mask is None:
        return None
    
    # Step 2: Estimate field lines
    field_lines = estimate_field_lines(unified_mask, original_frame)
    
    # Step 3: Estimate field corners
    field_corners = _estimate_field_corners(field_lines, frame_shape)
    
    # Step 4: Calculate field coverage ratio
    coverage_ratio = _calculate_field_coverage(unified_mask, field_corners, frame_shape)
    
    # Create unified field object
    unified_field = UnifiedField(
        mask=unified_mask,
        lines=field_lines,
        field_corners=field_corners,
        coverage_ratio=coverage_ratio,
        image_bounds=(frame_shape[1], frame_shape[0])  # (width, height)
    )
    
    print(f"[FIELD_PROJ] Created unified field with {len(field_lines)} lines, "
          f"coverage ratio: {coverage_ratio:.2f}")
    
    return unified_field


def _apply_morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """Apply morphological operations to clean up the field mask.
    
    Args:
        mask: Binary field mask
        
    Returns:
        Cleaned binary mask with noise removed and boundaries smoothed
    """
    # Get morphological parameters from configuration
    erosion_kernel_size = get_setting("field_projection.erosion_kernel_size", 3)
    dilation_kernel_size = get_setting("field_projection.dilation_kernel_size", 5)
    closing_kernel_size = get_setting("field_projection.closing_kernel_size", 7)
    
    print(f"[FIELD_PROJ] Applying morphological cleanup: erosion({erosion_kernel_size}), "
          f"dilation({dilation_kernel_size}), closing({closing_kernel_size})")
    
    # Step 1: Erosion to remove small noise and thin connections
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
    mask = cv2.erode(mask, erosion_kernel, iterations=1)
    
    # Step 2: Dilation to restore the main field area
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    
    # Step 3: Closing operation to fill internal holes and smooth boundaries
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
    
    # Step 4: Opening operation to remove remaining small noise
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    
    # Step 5: Final median blur for smooth edges
    mask = cv2.medianBlur(mask, 5)
    
    return mask


def _extract_largest_component(mask: np.ndarray) -> np.ndarray:
    """Extract the largest connected component from the mask.
    
    Args:
        mask: Binary field mask
        
    Returns:
        Mask containing only the largest connected component
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:  # Only background
        return mask
    
    # Find the largest component (excluding background at index 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create mask with only the largest component
    largest_mask = (labels == largest_label).astype(np.uint8)
    
    return largest_mask


def _filter_boundary_lines(lines: np.ndarray, mask_shape: Tuple[int, int], 
                          edge_margin: int) -> List[Tuple[int, int, int, int]]:
    """Filter out lines that touch image boundaries (edge artifacts).
    
    Args:
        lines: Array of detected lines from HoughLinesP
        mask_shape: Shape of the mask (height, width)
        edge_margin: Pixels from edge to consider as boundary
        
    Returns:
        List of filtered line segments that don't touch boundaries
    """
    h, w = mask_shape
    filtered_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Check if any endpoint is too close to image boundaries
        if (x1 <= edge_margin or x1 >= w - edge_margin or
            x2 <= edge_margin or x2 >= w - edge_margin or
            y1 <= edge_margin or y1 >= h - edge_margin or
            y2 <= edge_margin or y2 >= h - edge_margin):
            continue  # Skip lines touching boundaries
        
        filtered_lines.append((x1, y1, x2, y2))
    
    print(f"[FIELD_PROJ] Filtered {len(lines) - len(filtered_lines)} boundary lines, "
          f"{len(filtered_lines)} remain")
    return filtered_lines


def _classify_field_lines_perspective(lines: List[Tuple[int, int, int, int]], 
                                    mask_shape: Tuple[int, int]) -> List[FieldLine]:
    """Classify detected lines into field line types considering drone perspective.
    
    Args:
        lines: List of filtered line segments
        mask_shape: Shape of the mask (height, width)
        
    Returns:
        List of classified FieldLine objects
        
    Note:
        In drone perspective:
        - Sidelines appear diagonal (not vertical)
        - Field lines (goal lines) appear more horizontal
        - The perspective creates converging lines toward the horizon
    """
    field_lines = []
    
    if not lines:
        return field_lines
    
    # Get angle thresholds from configuration
    sideline_min = get_setting("field_projection.sideline_angle_min", 25)
    sideline_max = get_setting("field_projection.sideline_angle_max", 75)
    field_line_max = get_setting("field_projection.field_line_angle_max", 25)
    
    # Separate lines by orientation considering perspective
    diagonal_lines = []    # Sidelines (appear diagonal due to perspective)
    horizontal_lines = []  # Field lines (appear more horizontal)
    
    for line in lines:
        x1, y1, x2, y2 = line
        
        # Calculate line angle relative to horizontal
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0:  # Vertical line
            angle = 90.0
        else:
            angle = math.atan2(dy, dx) * 180 / math.pi
            # Normalize angle to 0-180 range (absolute orientation, not direction)
            angle = abs(angle)
            if angle > 90:
                angle = 180 - angle  # Convert obtuse to acute equivalent
        
        # In drone perspective:
        # - Sidelines appear diagonal (configurable range, including steep angles)
        # - Field lines appear more horizontal (configurable threshold)
        
        # Check for diagonal sidelines (25-75° range covers most perspective angles)
        is_diagonal = (sideline_min <= angle <= sideline_max)
        
        # Also accept steep diagonal lines that wrap around (e.g., 152° -> 28°)
        # These are common in drone perspective where sidelines appear very angled
        if not is_diagonal:
            # Check if this is a steep sideline that appears as obtuse angle
            raw_angle = abs(math.atan2(dy, dx) * 180 / math.pi)
            if raw_angle > 90:
                # Convert back: 152° means it's really 28° steep diagonal
                equivalent_angle = 180 - raw_angle
                is_diagonal = (sideline_min <= equivalent_angle <= sideline_max)
                angle = equivalent_angle  # Use the equivalent acute angle
        
        if is_diagonal:  # Diagonal lines - likely sidelines
            diagonal_lines.append((x1, y1, x2, y2, angle))
            print(f"[FIELD_PROJ] Line ({x1},{y1})-({x2},{y2}) classified as SIDELINE (angle={angle:.1f}°)")
        elif angle <= field_line_max:  # More horizontal lines - likely field lines
            horizontal_lines.append((x1, y1, x2, y2, angle))
            print(f"[FIELD_PROJ] Line ({x1},{y1})-({x2},{y2}) classified as FIELD LINE (angle={angle:.1f}°)")
        else:
            raw_angle = abs(math.atan2(dy, dx) * 180 / math.pi)
            print(f"[FIELD_PROJ] Line ({x1},{y1})-({x2},{y2}) REJECTED (angle={angle:.1f}° from raw {raw_angle:.1f}°, not in valid ranges)")
        # Skip lines outside these ranges as they're likely artifacts
    
    print(f"[FIELD_PROJ] Classified lines: {len(diagonal_lines)} diagonal (sidelines), "
          f"{len(horizontal_lines)} horizontal (field lines)")
    
    # Process diagonal lines (sidelines)
    if diagonal_lines:
        left_sideline, right_sideline = _find_diagonal_sidelines(diagonal_lines, mask_shape)
        if left_sideline:
            field_lines.append(left_sideline)
        if right_sideline:
            field_lines.append(right_sideline)
    
    # Process horizontal lines (field lines)
    if horizontal_lines:
        close_line, far_line = _find_horizontal_field_lines(horizontal_lines, mask_shape)
        if close_line:
            field_lines.append(close_line)
        if far_line:
            field_lines.append(far_line)
    
    return field_lines


def _find_diagonal_sidelines(diagonal_lines: List[Tuple[int, int, int, int, float]], 
                           mask_shape: Tuple[int, int]) -> Tuple[Optional[FieldLine], Optional[FieldLine]]:
    """Find left and right sidelines from diagonal line segments.
    
    Args:
        diagonal_lines: List of diagonal line segments with angles
        mask_shape: Shape of the mask (height, width)
        
    Returns:
        Tuple of (left_sideline, right_sideline) FieldLine objects
    """
    if not diagonal_lines:
        return None, None
    
    # Group lines by general x-position (left vs right side of image)
    h, w = mask_shape
    left_lines = []
    right_lines = []
    
    for x1, y1, x2, y2, angle in diagonal_lines:
        # Calculate center x position of the line
        center_x = (x1 + x2) / 2
        
        if center_x < w / 2:  # Left half of image
            left_lines.append((x1, y1, x2, y2, angle))
        else:  # Right half of image
            right_lines.append((x1, y1, x2, y2, angle))
    
    # Find the best line in each group
    left_sideline = _select_best_sideline(left_lines, 'left_sideline') if left_lines else None
    right_sideline = _select_best_sideline(right_lines, 'right_sideline') if right_lines else None
    
    return left_sideline, right_sideline


def _find_horizontal_field_lines(horizontal_lines: List[Tuple[int, int, int, int, float]], 
                               mask_shape: Tuple[int, int]) -> Tuple[Optional[FieldLine], Optional[FieldLine]]:
    """Find close and far field lines from horizontal line segments.
    
    Args:
        horizontal_lines: List of horizontal line segments with angles
        mask_shape: Shape of the mask (height, width)
        
    Returns:
        Tuple of (close_field_line, far_field_line) FieldLine objects
    """
    if not horizontal_lines:
        return None, None
    
    # Group lines by y-position (top vs bottom of image)
    h, w = mask_shape
    top_lines = []
    bottom_lines = []
    
    for x1, y1, x2, y2, angle in horizontal_lines:
        # Calculate center y position of the line
        center_y = (y1 + y2) / 2
        
        if center_y < h / 2:  # Top half of image (far field)
            top_lines.append((x1, y1, x2, y2, angle))
        else:  # Bottom half of image (close field)
            bottom_lines.append((x1, y1, x2, y2, angle))
    
    # Find the best line in each group
    far_line = _select_best_field_line(top_lines, 'far_field') if top_lines else None
    close_line = _select_best_field_line(bottom_lines, 'close_field') if bottom_lines else None
    
    return close_line, far_line


def _select_best_sideline(lines: List[Tuple[int, int, int, int, float]], 
                         line_type: str) -> Optional[FieldLine]:
    """Select the best sideline from candidate lines.
    
    Args:
        lines: List of candidate line segments with angles
        line_type: Type of line ('left_sideline' or 'right_sideline')
        
    Returns:
        Best FieldLine object or None
    """
    if not lines:
        return None
    
    # Select the longest line as the best candidate
    best_line = None
    best_length = 0
    
    for x1, y1, x2, y2, angle in lines:
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > best_length:
            best_length = length
            best_line = (x1, y1, x2, y2, angle)
    
    if best_line:
        x1, y1, x2, y2, angle = best_line
        # Calculate confidence based on length and angle consistency
        confidence = min(0.9, best_length / 200.0)  # Longer lines = higher confidence
        
        return FieldLine(
            line_type=line_type,
            point1=(x1, y1),
            point2=(x2, y2),
            confidence=confidence,
            visible=True,
            extrapolated=False
        )
    
    return None


def _select_best_field_line(lines: List[Tuple[int, int, int, int, float]], 
                          line_type: str) -> Optional[FieldLine]:
    """Select the best field line from candidate lines.
    
    Args:
        lines: List of candidate line segments with angles
        line_type: Type of line ('close_field' or 'far_field')
        
    Returns:
        Best FieldLine object or None
    """
    if not lines:
        return None
    
    # Select the longest, most horizontal line
    best_line = None
    best_score = 0
    
    for x1, y1, x2, y2, angle in lines:
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        # Score combines length and horizontalness (lower angle is better)
        horizontalness = 1.0 - (angle / 90.0)  # 1.0 for horizontal, 0.0 for vertical
        score = length * horizontalness
        
        if score > best_score:
            best_score = score
            best_line = (x1, y1, x2, y2, angle)
    
    if best_line:
        x1, y1, x2, y2, angle = best_line
        # Calculate confidence based on length and horizontalness
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        horizontalness = 1.0 - (angle / 90.0)
        confidence = min(0.9, (length / 200.0) * horizontalness)
        
        return FieldLine(
            line_type=line_type,
            point1=(x1, y1),
            point2=(x2, y2),
            confidence=confidence,
            visible=True,
            extrapolated=False
        )
    
    return None


def _classify_field_lines(lines: np.ndarray, mask_shape: Tuple[int, int]) -> List[FieldLine]:
    """Classify detected lines into field line types.
    
    Args:
        lines: Array of detected lines from HoughLinesP
        mask_shape: Shape of the mask (height, width)
        
    Returns:
        List of classified FieldLine objects
    """
    field_lines = []
    
    # Group lines by orientation (vertical = sidelines, horizontal = field lines)
    vertical_lines = []    # Sidelines (left/right)
    horizontal_lines = []  # Field lines (close/far)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line angle
        dx = x2 - x1
        dy = y2 - y1
        angle = abs(math.atan2(dy, dx) * 180 / math.pi)
        
        # Classify based on angle (allowing some tolerance)
        if angle > 45:  # More vertical than horizontal
            vertical_lines.append((x1, y1, x2, y2))
        else:  # More horizontal than vertical  
            horizontal_lines.append((x1, y1, x2, y2))
    
    # Process vertical lines (sidelines)
    if vertical_lines:
        left_sideline, right_sideline = _find_sidelines(vertical_lines, mask_shape)
        if left_sideline:
            field_lines.append(left_sideline)
        if right_sideline:
            field_lines.append(right_sideline)
    
    # Process horizontal lines (field lines)
    if horizontal_lines:
        close_line, far_line = _find_field_lines(horizontal_lines, mask_shape)
        if close_line:
            field_lines.append(close_line)
        if far_line:
            field_lines.append(far_line)
    
    return field_lines


def _find_sidelines(vertical_lines: List[Tuple[int, int, int, int]], 
                   mask_shape: Tuple[int, int]) -> Tuple[Optional[FieldLine], Optional[FieldLine]]:
    """Find left and right sidelines from vertical line segments.
    
    Args:
        vertical_lines: List of vertical line segments
        mask_shape: Shape of the mask (height, width)
        
    Returns:
        Tuple of (left_sideline, right_sideline) FieldLine objects
    """
    if not vertical_lines:
        return None, None
    
    # Group lines by x-coordinate and merge nearby lines
    merged_lines = _merge_parallel_lines(vertical_lines, orientation='vertical')
    
    if not merged_lines:
        return None, None
    
    # Sort by x-coordinate to find leftmost and rightmost
    merged_lines.sort(key=lambda line: (line[0] + line[2]) / 2)  # Sort by center x
    
    left_line = None
    right_line = None
    
    if len(merged_lines) >= 1:
        # Leftmost line is left sideline
        x1, y1, x2, y2 = merged_lines[0]
        left_line = FieldLine(
            line_type='left_sideline',
            point1=(x1, y1),
            point2=(x2, y2),
            confidence=0.8,  # TODO: Calculate based on line quality
            visible=True,
            extrapolated=False
        )
    
    if len(merged_lines) >= 2:
        # Rightmost line is right sideline
        x1, y1, x2, y2 = merged_lines[-1]
        right_line = FieldLine(
            line_type='right_sideline',
            point1=(x1, y1),
            point2=(x2, y2),
            confidence=0.8,
            visible=True,
            extrapolated=False
        )
    
    return left_line, right_line


def _find_field_lines(horizontal_lines: List[Tuple[int, int, int, int]], 
                     mask_shape: Tuple[int, int]) -> Tuple[Optional[FieldLine], Optional[FieldLine]]:
    """Find close and far field lines from horizontal line segments.
    
    Args:
        horizontal_lines: List of horizontal line segments
        mask_shape: Shape of the mask (height, width)
        
    Returns:
        Tuple of (close_field_line, far_field_line) FieldLine objects
    """
    if not horizontal_lines:
        return None, None
    
    # Group lines by y-coordinate and merge nearby lines
    merged_lines = _merge_parallel_lines(horizontal_lines, orientation='horizontal')
    
    if not merged_lines:
        return None, None
    
    # Sort by y-coordinate (assuming drone view where y increases downward)
    merged_lines.sort(key=lambda line: (line[1] + line[3]) / 2)  # Sort by center y
    
    close_line = None
    far_line = None
    
    if len(merged_lines) >= 1:
        # Line closest to bottom of image (highest y) is the close field line
        x1, y1, x2, y2 = merged_lines[-1]
        close_line = FieldLine(
            line_type='close_field',
            point1=(x1, y1),
            point2=(x2, y2),
            confidence=0.8,
            visible=True,
            extrapolated=False
        )
    
    if len(merged_lines) >= 2:
        # Line closest to top of image (lowest y) is the far field line
        x1, y1, x2, y2 = merged_lines[0]
        far_line = FieldLine(
            line_type='far_field',
            point1=(x1, y1),
            point2=(x2, y2),
            confidence=0.8,
            visible=True,
            extrapolated=False
        )
    
    return close_line, far_line


def _merge_parallel_lines(lines: List[Tuple[int, int, int, int]], 
                         orientation: str) -> List[Tuple[int, int, int, int]]:
    """Merge nearby parallel lines into single longer lines.
    
    Args:
        lines: List of line segments
        orientation: 'vertical' or 'horizontal'
        
    Returns:
        List of merged line segments
    """
    if not lines:
        return []
    
    # TODO: Implement sophisticated line merging algorithm
    # For now, return the longest lines after basic filtering
    
    # Filter out very short lines
    min_length = get_setting("field_projection.min_line_length", 50)
    filtered_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length >= min_length:
            filtered_lines.append(line)
    
    return filtered_lines


def _estimate_missing_lines(field_lines: List[FieldLine], 
                          unified_mask: np.ndarray) -> List[FieldLine]:
    """Estimate missing field lines by extrapolating from detected lines and field geometry.
    
    Args:
        field_lines: List of detected field lines
        unified_mask: Unified field mask for reference
        
    Returns:
        List of field lines including estimated missing lines
    """
    # TODO: Implement sophisticated line estimation
    # This would use field geometry knowledge and perspective transformation
    
    # For now, return the detected lines
    # Future implementation would:
    # 1. Identify which lines are missing
    # 2. Use field dimensions and perspective to estimate missing lines
    # 3. Mark estimated lines with extrapolated=True
    
    return field_lines


def _estimate_field_corners(field_lines: List[FieldLine], 
                          frame_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
    """Estimate field corner points from detected lines.
    
    Args:
        field_lines: List of detected/estimated field lines
        frame_shape: Shape of the frame (height, width)
        
    Returns:
        List of estimated field corner points (x, y)
    """
    corners = []
    
    # Find line intersections to estimate corners
    # TODO: Implement robust corner detection from line intersections
    
    # For now, return empty list
    # Future implementation would find intersections between:
    # - Left sideline and close field line
    # - Left sideline and far field line  
    # - Right sideline and close field line
    # - Right sideline and far field line
    
    return corners


def _calculate_field_coverage(unified_mask: np.ndarray, 
                            field_corners: List[Tuple[float, float]], 
                            frame_shape: Tuple[int, int]) -> float:
    """Calculate what fraction of the field is visible in the image.
    
    Args:
        unified_mask: Unified field mask
        field_corners: Estimated field corner points
        frame_shape: Shape of the frame (height, width)
        
    Returns:
        Coverage ratio between 0.0 and 1.0
    """
    if unified_mask is None:
        return 0.0
    
    # Simple approximation: ratio of visible field pixels to total image area
    # TODO: Implement more sophisticated coverage calculation using field geometry
    
    field_pixels = np.sum(unified_mask)
    total_pixels = frame_shape[0] * frame_shape[1]
    
    # Rough estimate - assumes field could fill entire frame
    coverage_ratio = min(1.0, field_pixels / (total_pixels * 0.6))  # 60% max expected coverage
    
    return coverage_ratio


def _draw_field_polygon(frame: np.ndarray, field_mask: np.ndarray) -> None:
    """Draw the polygon boundary that was fitted to the field mask.
    
    Args:
        frame: Frame to draw on (modified in place)
        field_mask: Binary field mask used for polygon fitting
    """
    try:
        # Find contours of the field mask
        contours, _ = cv2.findContours(
            field_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return
        
        # Get the largest contour (should be the main field boundary)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 4:
            return
        
        # Approximate the contour with a polygon
        epsilon = get_setting("field_projection.polygon_epsilon", 0.02) * cv2.arcLength(largest_contour, True)
        polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(polygon) < 3:
            return
        
        # Draw the polygon boundary
        cv2.polylines(frame, [polygon], True, (255, 0, 255), 2)  # Magenta polygon outline
        
        # Draw polygon vertices
        for i, point in enumerate(polygon):
            pt = tuple(point[0])
            cv2.circle(frame, pt, 4, (255, 0, 255), -1)  # Magenta vertices
            # Label vertices with numbers
            cv2.putText(frame, str(i), (pt[0] + 5, pt[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add polygon info text
        polygon_text = f"Polygon: {len(polygon)} vertices"
        cv2.putText(frame, polygon_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                   
    except Exception as e:
        print(f"[FIELD_PROJ] Error drawing polygon: {e}")


def visualize_unified_field(frame: np.ndarray, unified_field: UnifiedField) -> np.ndarray:
    """Visualize the unified field with detected lines on a frame.
    
    Args:
        frame: Input frame to draw on
        unified_field: UnifiedField object with mask and lines
        
    Returns:
        Frame with unified field visualization
    """
    if unified_field is None:
        return frame
    
    vis_frame = frame.copy()
    
    # Draw unified field mask
    if unified_field.mask is not None:
        # Create colored overlay for field area
        overlay = vis_frame.copy()
        field_color = (0, 200, 0)  # Green
        overlay[unified_field.mask > 0] = field_color
        vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
        
        # Draw the polygon boundary used for line detection
        _draw_field_polygon(vis_frame, unified_field.mask)
    
    # Draw field lines
    line_colors = {
        'left_sideline': (255, 0, 0),   # Blue
        'right_sideline': (255, 0, 0),  # Blue
        'close_field': (0, 255, 255),   # Yellow
        'far_field': (0, 255, 255),     # Yellow
    }
    
    for line in unified_field.lines:
        color = line_colors.get(line.line_type, (255, 255, 255))  # White default
        
        # Line thickness based on confidence
        thickness = int(2 + line.confidence * 2)
        
        # Draw line
        pt1 = (int(line.point1[0]), int(line.point1[1]))
        pt2 = (int(line.point2[0]), int(line.point2[1]))
        
        if line.extrapolated:
            # Draw dashed line for extrapolated lines
            _draw_dashed_line(vis_frame, pt1, pt2, color, thickness)
        else:
            cv2.line(vis_frame, pt1, pt2, color, thickness)
        
        # Add line type label
        mid_x = int((line.point1[0] + line.point2[0]) / 2)
        mid_y = int((line.point1[1] + line.point2[1]) / 2)
        cv2.putText(vis_frame, line.line_type, (mid_x, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw field corners
    for corner in unified_field.field_corners:
        cv2.circle(vis_frame, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), -1)
    
    # Add coverage ratio text
    coverage_text = f"Field Coverage: {unified_field.coverage_ratio:.1%}"
    cv2.putText(vis_frame, coverage_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis_frame


def _draw_dashed_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                     color: Tuple[int, int, int], thickness: int) -> None:
    """Draw a dashed line on the image.
    
    Args:
        img: Image to draw on
        pt1: Start point (x, y)
        pt2: End point (x, y)
        color: Line color (B, G, R)
        thickness: Line thickness
    """
    dash_length = 10
    x1, y1 = pt1
    x2, y2 = pt2
    
    total_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if total_length == 0:
        return
    
    num_dashes = int(total_length / (dash_length * 2))
    if num_dashes == 0:
        cv2.line(img, pt1, pt2, color, thickness)
        return
    
    dx = (x2 - x1) / total_length
    dy = (y2 - y1) / total_length
    
    for i in range(num_dashes):
        start_x = x1 + i * 2 * dash_length * dx
        start_y = y1 + i * 2 * dash_length * dy
        end_x = x1 + (i * 2 + 1) * dash_length * dx
        end_y = y1 + (i * 2 + 1) * dash_length * dy
        
        cv2.line(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color, thickness)
