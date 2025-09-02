"""Field analysis processing functions.

This module contains computational algorithms for field boundary detection,
line fitting, and field geometry analysis. Separated from visualization
to maintain clear separation between processing and display logic.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from ..config.settings import get_setting


def create_unified_field_mask_processing(segmentation_results: List[Any], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Create a unified mask combining all segmentation classes into one binary mask.
    
    This is a pure processing function that creates the base mask data.
    
    Args:
        segmentation_results: List of segmentation result objects
        frame_shape: (height, width) of the target frame
        
    Returns:
        Unified binary mask (H, W) where 1 indicates field area, or None if no results
    """
    if not segmentation_results:
        return None
    
    frame_h, frame_w = frame_shape
    unified_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    
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
            
            # Combine all class masks into unified mask
            for mask in masks:
                # Ensure mask is 2D
                if len(mask.shape) == 3:
                    mask = mask[0]
                
                # Resize to target frame size if needed
                if mask.shape != (frame_h, frame_w):
                    mask_resized = cv2.resize(
                        mask.astype(np.float32), 
                        (frame_w, frame_h), 
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask_resized = mask
                
                # Add to unified mask (any field class becomes 1)
                unified_mask = np.logical_or(unified_mask, mask_resized > 0.5).astype(np.uint8)
                
        except Exception as e:
            print(f"[FIELD_ANALYSIS] Error creating unified mask: {e}")
    
    # Apply morphological operations to smooth the mask
    if np.any(unified_mask):
        unified_mask = apply_morphological_smoothing(unified_mask)
            
    return unified_mask if np.any(unified_mask) else None


def calculate_field_contour_processing(unified_mask: np.ndarray, 
                           simplify_epsilon: float = None,
                           min_contour_area: int = None) -> Optional[np.ndarray]:
    """Calculate and simplify the contour of the field mask.
    
    Args:
        unified_mask: Binary mask (H, W) where 1 indicates field area
        simplify_epsilon: Epsilon parameter for contour simplification (as fraction of perimeter)
        min_contour_area: Minimum area threshold for contours
        
    Returns:
        Simplified contour points as numpy array of shape (N, 1, 2), or None if no contour found
    """
    if unified_mask is None or not np.any(unified_mask):
        return None
    
    # Use config values if parameters not provided
    if simplify_epsilon is None:
        simplify_epsilon = get_setting("models.segmentation.contour.simplify_epsilon", 0.01)
    if min_contour_area is None:
        min_contour_area = get_setting("models.segmentation.contour.min_area", 5000)
    
    try:
        # Find contours
        contours, _ = cv2.findContours(unified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (main field boundary)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour meets minimum area requirement
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < min_contour_area:
            print(f"[FIELD_ANALYSIS] Contour area {contour_area} below threshold {min_contour_area}")
            return None
        
        # Simplify contour using Douglas-Peucker algorithm
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = simplify_epsilon * perimeter
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        print(f"[FIELD_ANALYSIS] Original contour points: {len(largest_contour)}, simplified: {len(simplified_contour)}")
        
        return simplified_contour
        
    except Exception as e:
        print(f"[FIELD_ANALYSIS] Error calculating field contour: {e}")
        return None


def apply_morphological_smoothing(mask: np.ndarray, 
                                 opening_kernel_size: int = None,
                                 closing_kernel_size: int = None,
                                 fill_holes: bool = None) -> np.ndarray:
    """Apply morphological operations to smooth a binary mask.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        opening_kernel_size: Size of kernel for opening operation (removes noise)
        closing_kernel_size: Size of kernel for closing operation (fills gaps)
        fill_holes: Whether to apply hole filling
        
    Returns:
        Smoothed binary mask
    """
    if not np.any(mask):
        return mask
    
    # Use config values if parameters not provided
    if opening_kernel_size is None:
        opening_kernel_size = get_setting("models.segmentation.morphological.opening_kernel_size", 5)
    if closing_kernel_size is None:
        closing_kernel_size = get_setting("models.segmentation.morphological.closing_kernel_size", 15)
    if fill_holes is None:
        fill_holes = get_setting("models.segmentation.morphological.fill_holes", True)
    
    try:
        # Ensure mask is binary
        mask_binary = (mask > 0).astype(np.uint8)
        
        # 1. Opening operation: erosion followed by dilation
        # This removes small noise and disconnected components
        if opening_kernel_size > 0:
            opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                     (opening_kernel_size, opening_kernel_size))
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, opening_kernel)
        
        # 2. Closing operation: dilation followed by erosion
        # This fills small gaps and holes within the field area
        if closing_kernel_size > 0:
            closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                     (closing_kernel_size, closing_kernel_size))
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, closing_kernel)
        
        # 3. Fill remaining holes using flood fill
        if fill_holes:
            mask_binary = _fill_holes_flood_fill(mask_binary)
        
        return mask_binary
        
    except Exception as e:
        print(f"[FIELD_ANALYSIS] Error in morphological smoothing: {e}")
        return mask


def _fill_holes_flood_fill(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask using flood fill from the borders.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        
    Returns:
        Mask with holes filled
    """
    try:
        # Create a copy to work with
        filled_mask = mask.copy()
        h, w = mask.shape
        
        # Create a mask that is 2 pixels larger in each dimension
        # This allows flood fill to work from the border
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        
        # Copy the original mask to the center of the flood mask
        flood_mask[1:-1, 1:-1] = 1 - filled_mask  # Invert: 0 becomes 1, 1 becomes 0
        
        # Flood fill from the top-left corner (which should be background)
        cv2.floodFill(flood_mask, None, (0, 0), 0)
        
        # Extract the filled region and invert back
        filled_region = flood_mask[1:-1, 1:-1]
        filled_mask = 1 - filled_region
        
        return filled_mask.astype(np.uint8)
        
    except Exception as e:
        print(f"[FIELD_ANALYSIS] Error in hole filling: {e}")
        return mask

try:
    from sklearn.linear_model import RANSACRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[FIELD_ANALYSIS] Warning: sklearn not available, using fallback RANSAC implementation")

from ..config.settings import get_setting


def filter_edge_points(contour: np.ndarray, 
                      frame_shape: tuple,
                      edge_margin: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Filter out contour points that are too close to image edges.
    
    Points near the edge are often artifacts from segmentation models
    and should not be considered for field boundary fitting.
    
    Args:
        contour: Contour points as numpy array of shape (N, 1, 2) or (N, 2)
        frame_shape: Shape of the frame (height, width) or (height, width, channels)
        edge_margin: Distance from edge in pixels to filter out
        
    Returns:
        Tuple of (filtered_contour, edge_points):
        - filtered_contour: Points away from edges
        - edge_points: Points near edges that were filtered out
    """
    if contour is None or len(contour) == 0:
        return contour, np.array([]).reshape(0, 1, 2)
    
    # Ensure contour is in shape (N, 2)
    if contour.ndim == 3 and contour.shape[1] == 1:
        points = contour.reshape(-1, 2)
    else:
        points = contour.reshape(-1, 2)
    
    # Get frame dimensions
    height, width = frame_shape[:2]
    
    # Create mask for points away from edges
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Points are kept if they are sufficiently far from all edges
    valid_mask = (
        (x_coords >= edge_margin) &           # Not too close to left edge
        (x_coords <= width - edge_margin) &   # Not too close to right edge
        (y_coords >= edge_margin) &           # Not too close to top edge
        (y_coords <= height - edge_margin)    # Not too close to bottom edge
    )
    
    # Split points into valid and edge points
    valid_points = points[valid_mask]
    edge_points = points[~valid_mask]
    
    # Convert back to original format (N, 1, 2)
    valid_contour = valid_points.reshape(-1, 1, 2) if len(valid_points) > 0 else np.array([]).reshape(0, 1, 2)
    edge_contour = edge_points.reshape(-1, 1, 2) if len(edge_points) > 0 else np.array([]).reshape(0, 1, 2)
    
    return valid_contour, edge_contour


def interpolate_contour_points(contour: np.ndarray, 
                              max_distance: float = 10.0,
                              min_distance: float = 3.0) -> np.ndarray:
    """Interpolate contour points to ensure even spacing.
    
    This function adds points between existing contour points to ensure 
    that no two consecutive points are more than max_distance apart,
    while avoiding over-densification with min_distance constraint.
    
    Args:
        contour: Contour points as numpy array of shape (N, 1, 2) or (N, 2)
        max_distance: Maximum allowed distance between consecutive points
        min_distance: Minimum distance to maintain between points
        
    Returns:
        Interpolated contour points as numpy array of shape (M, 1, 2)
    """
    if contour is None or len(contour) < 2:
        return contour
    
    # Ensure contour is in shape (N, 2)
    if contour.ndim == 3 and contour.shape[1] == 1:
        points = contour.reshape(-1, 2)
    else:
        points = contour.reshape(-1, 2)
    
    interpolated_points = []
    
    for i in range(len(points)):
        current_point = points[i]
        next_point = points[(i + 1) % len(points)]  # Wrap around for closed contour
        
        # Always add the current point
        interpolated_points.append(current_point)
        
        # Calculate distance to next point
        distance = np.linalg.norm(next_point - current_point)
        
        # If distance is too large, add interpolated points
        if distance > max_distance:
            # Calculate number of points needed
            num_intermediate = int(np.ceil(distance / max_distance)) - 1
            
            # Add intermediate points
            for j in range(1, num_intermediate + 1):
                alpha = j / (num_intermediate + 1)
                intermediate_point = current_point + alpha * (next_point - current_point)
                
                # Check minimum distance constraint with the last added point
                if len(interpolated_points) == 0 or \
                   np.linalg.norm(intermediate_point - interpolated_points[-1]) >= min_distance:
                    interpolated_points.append(intermediate_point)
    
    # Convert back to original format (N, 1, 2)
    if len(interpolated_points) > 0:
        interpolated_array = np.array(interpolated_points, dtype=np.float32)
        return interpolated_array.reshape(-1, 1, 2)
    else:
        return contour


def fit_field_lines_ransac(contour: np.ndarray, 
                          frame: np.ndarray,
                          num_lines: int = 4,
                          distance_threshold: float = 5.0,
                          min_samples: int = 2,
                          max_trials: int = 100) -> Optional[Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict, Dict]]:
    """Fit straight lines to contour segments using RANSAC.
    
    This function segments the contour and fits straight lines to each segment,
    which is useful for field boundary detection where we expect rectangular shapes.
    
    Args:
        contour: Contour points as numpy array of shape (N, 1, 2)
        frame: Input frame for determining shape (used for edge filtering)
        num_lines: Number of line segments to fit (typically 3-4 for field boundaries)
        distance_threshold: Maximum distance from point to line to be considered inlier
        min_samples: Minimum number of points needed to fit a line
        max_trials: Maximum RANSAC iterations per line segment
        
    Returns:
        Tuple of (fitted_lines, outlier_points, inlier_points, edge_filtered_points, classified_lines, all_lines_for_display) where:
        - fitted_lines: List of (start_point, end_point) tuples for each fitted line
        - outlier_points: List of outlier point arrays for each segment
        - inlier_points: List of inlier point arrays for each segment
        - edge_filtered_points: Points that were filtered out during edge filtering
        - classified_lines: Dictionary of high-confidence classified lines
        - all_lines_for_display: Dictionary of all lines (classified and unclassified) for visualization
        Returns (None, None, None, None, {}, {}) if fitting fails
    """
    if contour is None or len(contour) < num_lines * min_samples:
        return None, None, None, None, {}, {}
    
    try:
        # Convert contour to 2D points array
        points = contour.reshape(-1, 2).astype(np.float32)
        edge_filtered_points = np.array([]).reshape(0, 2)  # Store edge-filtered points
        
        # Apply interpolation if enabled (before edge filtering)
        interpolation_enabled = get_setting("models.segmentation.contour.interpolation.enabled", False)
        if interpolation_enabled:
            max_distance = get_setting("models.segmentation.contour.interpolation.max_point_distance", 10)
            min_distance = get_setting("models.segmentation.contour.interpolation.min_point_distance", 3)
            
            # Convert to contour format for interpolation
            contour_format = points.reshape(-1, 1, 2)
            interpolated_contour = interpolate_contour_points(contour_format, max_distance, min_distance)
            points = interpolated_contour.reshape(-1, 2).astype(np.float32)
        
        # Apply edge filtering after interpolation if enabled
        edge_filtering_enabled = get_setting("models.segmentation.contour.ransac.edge_filtering.enabled", False)
        if edge_filtering_enabled:
            edge_margin = get_setting("models.segmentation.contour.ransac.edge_filtering.margin", 20)
            
            # Convert to contour format for edge filtering
            contour_format = points.reshape(-1, 1, 2)
            filtered_contour, edge_points = filter_edge_points(contour_format, frame.shape, edge_margin)
            
            # Update points to use only non-edge points
            if len(filtered_contour) > 0:
                points = filtered_contour.reshape(-1, 2).astype(np.float32)
                edge_filtered_points = edge_points.reshape(-1, 2).astype(np.float32) if len(edge_points) > 0 else np.array([]).reshape(0, 2)
        
        # Sequential RANSAC: Find lines one by one, removing inliers each time
        remaining_points = points.copy()
        fitted_lines = []
        line_confidences = []  # Store confidence for each line
        all_outliers = []
        all_inliers = []
        
        for iteration in range(num_lines):
            if len(remaining_points) < min_samples:
                # Insufficient points remaining for line fitting
                break
            
            # Fit line to remaining points using RANSAC
            result = _fit_line_ransac_with_outliers(remaining_points, distance_threshold, min_samples, max_trials)
            
            if result is not None:
                line_points, outliers, inliers, confidence = result
                fitted_lines.append(line_points)
                line_confidences.append(confidence)
                all_inliers.append(inliers)
                
                # Remove inliers from remaining points for next iteration
                remaining_points = outliers
            else:
                # Failed to fit line, stopping sequential RANSAC
                break
        
        # All remaining points after all iterations are final outliers
        if len(remaining_points) > 0:
            all_outliers.append(remaining_points)
        else:
            all_outliers.append(np.array([]).reshape(0, 2))
        
        # Filter lines by confidence threshold for classification
        classification_threshold = get_setting("models.segmentation.contour.ransac.classification_confidence_threshold", 0.5)
        
        # Separate high-confidence lines for classification vs all lines for display
        high_confidence_lines = []
        high_confidence_confidences = []
        
        for i, (line, confidence) in enumerate(zip(fitted_lines, line_confidences)):
            if line is not None and confidence >= classification_threshold:
                high_confidence_lines.append(line)
                high_confidence_confidences.append(confidence)
        
        # Classify only the high-confidence lines
        if high_confidence_lines:
            classified_lines = _classify_field_lines(high_confidence_lines, frame.shape, high_confidence_confidences)
        else:
            classified_lines = {}
        
        # Create a dictionary of all lines (including low-confidence ones) for display purposes
        all_lines_for_display = {}
        
        # Add all fitted lines with their original indices for display
        for i, (line, confidence) in enumerate(zip(fitted_lines, line_confidences)):
            if line is not None:
                is_classified = confidence >= classification_threshold
                line_type = f"unclassified_line_{i}" if not is_classified else None
                
                # If it's a high-confidence line, it will be overwritten by the classified version below
                if not is_classified:
                    all_lines_for_display[line_type] = (line, confidence, False)
        
        # Add classified lines (these will overwrite any unclassified entries for the same lines)
        for line_type, (line_coords, confidence) in classified_lines.items():
            all_lines_for_display[line_type] = (line_coords, confidence, True)
        
        # Filter out None entries from fitted_lines but keep all for display
        valid_lines = [line for line in fitted_lines if line is not None]
        return (valid_lines, all_outliers, all_inliers, edge_filtered_points, classified_lines, all_lines_for_display) if valid_lines else (None, None, None, edge_filtered_points, {}, {})
        
    except Exception as e:
        print(f"[FIELD_ANALYSIS] Error in RANSAC line fitting: {e}")
        return None, None, None, np.array([]).reshape(0, 2), {}, {}


def _fit_line_ransac_with_outliers(points: np.ndarray, distance_threshold: float, 
                                   min_samples: int, max_trials: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Fit a line using RANSAC and return the line, outliers, inliers, and confidence."""
    if len(points) < min_samples:
        return None
    
    if SKLEARN_AVAILABLE:
        return _fit_line_ransac_sklearn(points, distance_threshold, min_samples, max_trials)
    else:
        return _fit_line_ransac_fallback(points, distance_threshold, min_samples, max_trials)


def _fit_line_ransac_sklearn(points: np.ndarray, distance_threshold: float, 
                            min_samples: int, max_trials: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Fit a line using sklearn RANSAC."""
    try:
        # Prepare data for RANSAC
        X = points[:, 0].reshape(-1, 1)  # x coordinates
        y = points[:, 1]                 # y coordinates
        
        # Create RANSAC regressor
        ransac = RANSACRegressor(
            estimator=None,  # Use default LinearRegression
            min_samples=min_samples,
            residual_threshold=distance_threshold,
            max_trials=max_trials,
            stop_probability=0.99,
            random_state=None
        )
        
        # Fit RANSAC
        ransac.fit(X, y)
        
        # Get inlier mask
        inlier_mask = ransac.inlier_mask_
        outlier_mask = ~inlier_mask
        
        # Extract inliers and outliers
        inliers = points[inlier_mask]
        outliers = points[outlier_mask]
        
        # Calculate confidence as ratio of inliers
        confidence = np.sum(inlier_mask) / len(points)
        
        # Get line endpoints from inliers
        if len(inliers) < 2:
            return None
            
        # Find extreme points along the fitted line
        x_coords = inliers[:, 0]
        y_coords = inliers[:, 1]
        
        # Use min/max x coordinates to define line endpoints
        x_min, x_max = x_coords.min(), x_coords.max()
        
        # Predict y values for these x coordinates
        y_min = ransac.predict([[x_min]])[0]
        y_max = ransac.predict([[x_max]])[0]
        
        # Create line endpoints
        line_points = np.array([[x_min, y_min], [x_max, y_max]])
        
        return line_points, outliers, inliers, confidence
        
    except Exception as e:
        print(f"[FIELD_ANALYSIS] Error in sklearn RANSAC fitting: {e}")
        return None


def _fit_line_ransac_fallback(points: np.ndarray, distance_threshold: float, 
                             min_samples: int, max_trials: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Fallback RANSAC implementation when sklearn is not available."""
    try:
        best_inliers = None
        best_line = None
        best_inlier_count = 0
        
        for trial in range(max_trials):
            # Randomly sample min_samples points
            if len(points) < min_samples:
                break
                
            sample_indices = np.random.choice(len(points), min_samples, replace=False)
            sample_points = points[sample_indices]
            
            # Fit line to sample points (simple least squares for 2 points)
            if min_samples == 2:
                # Two points define a line
                p1, p2 = sample_points[0], sample_points[1]
                
                # Skip if points are too close
                if np.linalg.norm(p2 - p1) < 1e-6:
                    continue
                    
                # Calculate line parameters: ax + by + c = 0
                # Using point-direction form
                direction = p2 - p1
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-6:
                    continue
                    
                # Normal vector to the line
                normal = np.array([-direction[1], direction[0]]) / direction_norm
                a, b = normal[0], normal[1]
                c = -(a * p1[0] + b * p1[1])
                
                # Calculate distances from all points to this line
                distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
                
                # Find inliers
                inlier_mask = distances <= distance_threshold
                inlier_count = np.sum(inlier_mask)
                
                # Update best model if this is better
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_inliers = inlier_mask
                    
                    # Calculate line endpoints from all inliers
                    inlier_points = points[inlier_mask]
                    if len(inlier_points) >= 2:
                        x_coords = inlier_points[:, 0]
                        x_min, x_max = x_coords.min(), x_coords.max()
                        
                        # Calculate corresponding y values on the line
                        if abs(b) > 1e-6:  # Line is not vertical
                            y_min = -(a * x_min + c) / b
                            y_max = -(a * x_max + c) / b
                        else:  # Vertical line
                            y_coords = inlier_points[:, 1]
                            y_min, y_max = y_coords.min(), y_coords.max()
                            x_min = x_max = -c / a
                        
                        best_line = np.array([[x_min, y_min], [x_max, y_max]])
        
        if best_inliers is not None and best_line is not None:
            inliers = points[best_inliers]
            outliers = points[~best_inliers]
            confidence = best_inlier_count / len(points)
            return best_line, outliers, inliers, confidence
        
        return None
        
    except Exception as e:
        print(f"[FIELD_ANALYSIS] Error in fallback RANSAC fitting: {e}")
        return None


def _classify_field_lines(fitted_lines: List[np.ndarray], frame_shape: Tuple[int, int, int], 
                         line_confidences: List[float] = None) -> Dict[str, Tuple[np.ndarray, float]]:
    """Classify fitted lines into field components (sidelines, endzone lines, etc.).
    
    Args:
        fitted_lines: List of line segments as [start_point, end_point] arrays
        frame_shape: Shape of the frame (height, width, channels)
        line_confidences: List of confidence scores for each line
        
    Returns:
        Dictionary mapping line types to (line_coordinates, confidence) tuples
    """
    if not fitted_lines:
        return {}
    
    # If no confidences provided, use default values
    if line_confidences is None:
        line_confidences = [1.0] * len(fitted_lines)
    
    frame_height, frame_width = frame_shape[:2]
    classified = {}
    
    # Calculate line properties
    line_info = []
    for i, line in enumerate(fitted_lines):
        if line is None or len(line) != 2:
            continue
            
        start_point, end_point = line[0], line[1]
        
        # Calculate line angle (in degrees)
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Normalize angle to [0, 180)
        if angle < 0:
            angle += 180
        
        # Calculate average Y position (vertical position in image)
        avg_y = (start_point[1] + end_point[1]) / 2
        avg_x = (start_point[0] + end_point[0]) / 2
        
        # Determine if line is horizontal or vertical
        # Lines within 20 degrees of horizontal are considered horizontal
        is_horizontal = abs(angle) <= 20 or abs(angle - 180) <= 20
        
        line_info.append({
            'index': i,
            'line': line,
            'angle': angle,
            'avg_y': avg_y,
            'avg_x': avg_x,
            'horizontal': is_horizontal,
            'confidence': line_confidences[i]
        })
    
    # Sort lines by position and angle for classification
    horizontal_lines = [l for l in line_info if l['horizontal']]
    vertical_lines = [l for l in line_info if not l['horizontal']]
    
    # Classify horizontal lines (endzone lines) - sort by Y position
    horizontal_lines.sort(key=lambda x: x['avg_y'])
    for i, line_data in enumerate(horizontal_lines):
        if i == 0 and line_data['avg_y'] < frame_height * 0.6:
            # Top line is far endzone back
            classified['far_endzone_back'] = (line_data['line'], line_data['confidence'])
        elif i == 1 and line_data['avg_y'] < frame_height * 0.8:
            # Second line might be far endzone front
            classified['far_endzone_front'] = (line_data['line'], line_data['confidence'])
    
    # Classify vertical lines (sidelines) - sort by X position  
    vertical_lines.sort(key=lambda x: x['avg_x'])
    for i, line_data in enumerate(vertical_lines):
        if i == 0 and line_data['avg_x'] < frame_width * 0.6:
            # Leftmost line is left sideline
            classified['left_sideline'] = (line_data['line'], line_data['confidence'])
        elif i == len(vertical_lines) - 1 and line_data['avg_x'] > frame_width * 0.4:
            # Rightmost line is right sideline  
            classified['right_sideline'] = (line_data['line'], line_data['confidence'])
    
    return classified
