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


def estimate_field_lines(unified_mask: np.ndarray) -> List[FieldLine]:
    """Estimate field lines from unified field segmentation mask.
    
    Args:
        unified_mask: Unified binary field mask
        
    Returns:
        List of estimated field lines (sidelines and field lines)
        
    Implementation:
        - Uses edge detection on the field mask
        - Filters out edges touching image boundaries (artifacts)
        - Applies Hough line transform to detect straight lines
        - Classifies lines based on drone perspective (diagonal sidelines, horizontal field lines)
        - Estimates missing lines by extrapolating from field geometry
    """
    if unified_mask is None or np.sum(unified_mask) == 0:
        return []
        
    print("[FIELD_PROJ] Estimating field lines from unified mask")
    
    # Get configuration parameters
    min_line_length = get_setting("field_projection.min_line_length", 80)  # Increased for better lines
    max_line_gap = get_setting("field_projection.max_line_gap", 15)
    hough_threshold = get_setting("field_projection.hough_threshold", 40)  # Increased threshold
    edge_margin = get_setting("field_projection.edge_margin", 10)  # Pixels from edge to ignore
    
    # Create a mask that excludes edge regions to avoid boundary artifacts
    h, w = unified_mask.shape
    edge_free_mask = unified_mask.copy()
    edge_free_mask[:edge_margin, :] = 0  # Top edge
    edge_free_mask[-edge_margin:, :] = 0  # Bottom edge
    edge_free_mask[:, :edge_margin] = 0  # Left edge
    edge_free_mask[:, -edge_margin:] = 0  # Right edge
    
    # Detect edges in the field mask (excluding boundary regions)
    # Use adaptive thresholds based on mask intensity
    mask_uint8 = edge_free_mask * 255
    
    # Apply Gaussian blur before edge detection for smoother results
    blurred_mask = cv2.GaussianBlur(mask_uint8, (3, 3), 0)
    
    # Use Canny edge detection with optimized parameters for field boundaries
    low_threshold = get_setting("field_projection.canny_low_threshold", 50)
    high_threshold = get_setting("field_projection.canny_high_threshold", 150)
    edges = cv2.Canny(blurred_mask, low_threshold, high_threshold, apertureSize=3)
    
    # Apply morphological operations to connect nearby edges and clean up
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel)
    
    # Remove very short edge segments that are likely noise
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, edge_kernel)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        print("[FIELD_PROJ] No lines detected with Hough transform")
        return []
    
    # Filter out lines that touch image boundaries (edge artifacts)
    filtered_lines = _filter_boundary_lines(lines, unified_mask.shape, edge_margin)
    
    if not filtered_lines:
        print("[FIELD_PROJ] No valid lines after boundary filtering")
        return []
    
    # Classify and filter lines for drone perspective
    field_lines = _classify_field_lines_perspective(filtered_lines, unified_mask.shape)
    
    # Estimate missing lines by extrapolation
    field_lines = _estimate_missing_lines(field_lines, unified_mask)
    
    print(f"[FIELD_PROJ] Estimated {len(field_lines)} field lines")
    return field_lines


def create_unified_field(segmentation_results: List[Any], 
                        frame_shape: Tuple[int, int]) -> Optional[UnifiedField]:
    """Create a complete unified field representation with lines and metadata.
    
    Args:
        segmentation_results: Raw YOLO segmentation results
        frame_shape: Shape of the frame (height, width)
        
    Returns:
        UnifiedField object with mask, lines, and metadata, or None if failed
    """
    print(f"[FIELD_PROJ] Creating unified field from {len(segmentation_results)} segmentation results")
    
    # Step 1: Unify field segmentation masks
    unified_mask = unify_field_segmentation(segmentation_results, frame_shape)
    if unified_mask is None:
        return None
    
    # Step 2: Estimate field lines
    field_lines = estimate_field_lines(unified_mask)
    
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
            angle = abs(math.atan2(dy, dx) * 180 / math.pi)
        
        # In drone perspective:
        # - Sidelines appear diagonal (configurable range)
        # - Field lines appear more horizontal (configurable threshold)
        if sideline_min <= angle <= sideline_max:  # Diagonal lines - likely sidelines
            diagonal_lines.append((x1, y1, x2, y2, angle))
        elif angle <= field_line_max:  # More horizontal lines - likely field lines
            horizontal_lines.append((x1, y1, x2, y2, angle))
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
