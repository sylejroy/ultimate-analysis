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
        Tuple of (fitted_lines, outlier_points, inlier_points, edge_filtered_points, empty_dict, all_lines_for_display) where:
        - fitted_lines: List of (start_point, end_point) tuples for each fitted line
        - outlier_points: List of outlier point arrays for each segment
        - inlier_points: List of inlier point arrays for each segment
        - edge_filtered_points: Points that were filtered out during edge filtering
        - empty_dict: Empty dictionary (classification removed)
        - all_lines_for_display: Dictionary of all lines for visualization
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
        
        # Create a dictionary of all lines for display purposes (no classification)
        all_lines_for_display = {}
        
        # Add all fitted lines for display with simple numbering
        for i, (line, confidence) in enumerate(zip(fitted_lines, line_confidences)):
            if line is not None:
                line_type = f"line_{i}"
                all_lines_for_display[line_type] = (line, confidence, False)
        
        # Filter out None entries from fitted_lines but keep all for display
        valid_lines = [line for line in fitted_lines if line is not None]
        return (valid_lines, all_outliers, all_inliers, edge_filtered_points, {}, all_lines_for_display) if valid_lines else (None, None, None, edge_filtered_points, {}, {})
        
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
        # Ensure points is 2D with shape (n_points, 2)
        if len(points.shape) != 2 or points.shape[1] != 2:
            print(f"[PROCESSING] Invalid points shape: {points.shape}, expected (n, 2)")
            return None
            
        if len(points) < min_samples:
            print(f"[PROCESSING] Insufficient points: {len(points)} < {min_samples}")
            return None
        
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
        if inlier_mask is None:
            print(f"[PROCESSING] RANSAC returned None inlier mask")
            return None
            
        outlier_mask = ~inlier_mask
        
        # Validate mask dimensions
        if len(inlier_mask) != len(points):
            print(f"[PROCESSING] Mask length mismatch: mask={len(inlier_mask)}, points={len(points)}")
            return None
        
        # Extract inliers and outliers - check mask dtype
        if inlier_mask.dtype != bool:
            print(f"[PROCESSING] Converting mask from {inlier_mask.dtype} to bool")
            inlier_mask = inlier_mask.astype(bool)
            outlier_mask = ~inlier_mask
        
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
            # Validate mask dimensions before using
            if len(best_inliers) != len(points):
                print(f"[PROCESSING] Warning: fallback mask length ({len(best_inliers)}) != points length ({len(points)})")
                return None
            
            inliers = points[best_inliers]
            outliers = points[~best_inliers]
            confidence = best_inlier_count / len(points)
            return best_line, outliers, inliers, confidence
        
        return None
        
    except Exception as e:
        print(f"[FIELD_ANALYSIS] Error in fallback RANSAC fitting: {e}")
        return None


def extract_field_lines_ransac_processing(contour: np.ndarray, frame: np.ndarray, 
                                         num_lines: int = 4, distance_threshold: float = 15.0,
                                         min_samples: int = 30, max_trials: int = 1000) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """Heavy RANSAC processing function for extracting field lines from contour.
    
    This function contains all the heavy computational processing for line extraction
    and should be called from processing modules, not visualization code.
    
    Args:
        contour: Input contour points
        frame: Frame for shape information
        num_lines: Maximum number of lines to extract
        distance_threshold: RANSAC distance threshold
        min_samples: Minimum samples per line
        max_trials: Maximum RANSAC trials
        
    Returns:
        Tuple of (fitted_lines, line_confidences, all_outliers, all_inliers, edge_filtered_points, processing_stats)
    """
    if contour is None or len(contour) < num_lines * min_samples:
        return [], [], [], [], np.array([]).reshape(0, 2), {}
    
    processing_stats = {
        'original_points': 0,
        'interpolated_points': 0, 
        'edge_filtered_points': 0,
        'lines_fitted': 0,
        'processing_time_ms': 0
    }
    
    import time
    start_time = time.time()
    
    try:
        # Convert contour to 2D points array
        points = contour.reshape(-1, 2).astype(np.float32)
        processing_stats['original_points'] = len(points)
        edge_filtered_points = np.array([]).reshape(0, 2)
        
        # Apply interpolation if enabled (before edge filtering)
        interpolation_enabled = get_setting("models.segmentation.contour.interpolation.enabled", False)
        if interpolation_enabled:
            max_distance = get_setting("models.segmentation.contour.interpolation.max_point_distance", 10)
            min_distance = get_setting("models.segmentation.contour.interpolation.min_point_distance", 3)
            
            # Convert to contour format for interpolation
            contour_format = points.reshape(-1, 1, 2)
            interpolated_contour = interpolate_contour_points(contour_format, max_distance, min_distance)
            points = interpolated_contour.reshape(-1, 2).astype(np.float32)
            processing_stats['interpolated_points'] = len(points)
            
            # Only log occasionally to avoid spam
            if processing_stats['original_points'] % 100 == 0:  # Log every 100th frame
                print(f"[PROCESSING] Interpolated contour: {processing_stats['original_points']} -> {len(points)} points")
        
        # Apply edge filtering after interpolation if enabled
        edge_filtering_enabled = get_setting("models.segmentation.contour.ransac.edge_filtering.enabled", False)
        if edge_filtering_enabled:
            edge_margin = get_setting("models.segmentation.contour.ransac.edge_filtering.margin", 20)
            
            # Convert to contour format for edge filtering
            contour_format = points.reshape(-1, 1, 2)
            filtered_contour, edge_points = filter_edge_points(contour_format, frame.shape, edge_margin)
            
            # Update points to use only non-edge points
            if len(filtered_contour) > 0:
                original_count = len(points)
                points = filtered_contour.reshape(-1, 2).astype(np.float32)
                edge_filtered_points = edge_points.reshape(-1, 2).astype(np.float32) if len(edge_points) > 0 else np.array([]).reshape(0, 2)
                processing_stats['edge_filtered_points'] = len(edge_filtered_points)
                
                # Only log occasionally to avoid spam
                if original_count % 100 == 0:  # Log every 100th frame
                    print(f"[PROCESSING] Edge filtering: {original_count} -> {len(points)} points ({len(edge_filtered_points)} filtered)")
            else:
                print(f"[PROCESSING] Warning: Edge filtering removed all points!")
        
        # Sequential RANSAC: Find lines one by one, removing inliers each time
        remaining_points = points.copy()
        fitted_lines = []
        line_confidences = []
        all_outliers = []
        all_inliers = []
        
        for iteration in range(num_lines):
            if len(remaining_points) < min_samples:
                # Only log occasionally to avoid spam
                if iteration == 0:  # Log for first iteration only
                    print(f"[PROCESSING] Not enough remaining points ({len(remaining_points)} < {min_samples})")
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
                processing_stats['lines_fitted'] += 1
                
                # Only log occasionally to avoid spam
                if iteration == 0:  # Log first line only
                    print(f"[PROCESSING] Found {len(fitted_lines)} field lines with avg confidence {np.mean(line_confidences):.3f}")
            else:
                break
        
        # All remaining points after all iterations are final outliers
        if len(remaining_points) > 0:
            all_outliers.append(remaining_points)
        else:
            all_outliers.append(np.array([]).reshape(0, 2))
        
        processing_stats['processing_time_ms'] = (time.time() - start_time) * 1000
        
        return fitted_lines, line_confidences, all_outliers, all_inliers, edge_filtered_points, processing_stats
        
    except Exception as e:
        print(f"[PROCESSING] Error in RANSAC processing: {e}")
        return [], [], [], [], np.array([]).reshape(0, 2), processing_stats
