"""Field analysis processing functions.

This module contains computational algorithms for field boundary detection,
line fitting, and field geometry analysis. Separated from visualization
to maintain clear separation between processing and display logic.

Performance optimizations:
- Cached morphological kernels
- Vectorized operations for contour processing
- Optimized RANSAC with squared distance calculations
- Reduced logging frequency for real-time processing
- Efficient array operations with minimal reshaping
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting
from ..utils.logger import get_logger

logger = get_logger("FIELD_ANALYSIS")

# Cache for morphological kernels to avoid recreation
_kernel_cache = {}


def _normalize_contour_to_points(contour: np.ndarray) -> np.ndarray:
    """Normalize contour input to (N, 2) points format efficiently.

    Args:
        contour: Contour as (N, 1, 2) or (N, 2) array

    Returns:
        Points as (N, 2) array
    """
    if contour.ndim == 3 and contour.shape[1] == 1:
        return contour.reshape(-1, 2)
    elif contour.ndim == 2 and contour.shape[1] == 2:
        return contour
    else:
        return contour.reshape(-1, 2)


def _points_to_contour_format(points: np.ndarray) -> np.ndarray:
    """Convert points to standard contour format (N, 1, 2).

    Args:
        points: Points as (N, 2) array

    Returns:
        Contour as (N, 1, 2) array
    """
    return points.reshape(-1, 1, 2) if len(points) > 0 else np.array([]).reshape(0, 1, 2)


def _get_morphological_kernel(size: int, shape=cv2.MORPH_ELLIPSE) -> np.ndarray:
    """Get a cached morphological kernel or create and cache a new one.

    Args:
        size: Kernel size
        shape: Kernel shape (default: cv2.MORPH_ELLIPSE)

    Returns:
        Cached or newly created kernel
    """
    cache_key = (size, shape)
    if cache_key not in _kernel_cache:
        _kernel_cache[cache_key] = cv2.getStructuringElement(shape, (size, size))
    return _kernel_cache[cache_key]


def create_unified_field_mask(
    segmentation_results: List[Any], frame_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
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
        if not hasattr(result, "masks") or result.masks is None:
            continue

        try:
            # Get mask data
            if hasattr(result.masks, "data"):
                mask_data = result.masks.data
            else:
                continue

            # Convert to numpy if needed
            if hasattr(mask_data, "cpu"):
                masks = mask_data.cpu().numpy()
            else:
                masks = mask_data.numpy() if hasattr(mask_data, "numpy") else mask_data

            masks = np.asarray(masks)
            if masks.ndim == 4:
                masks = masks[:, 0]
            if len(masks) == 0:
                continue

            # Combine the class masks first, so only one image is scaled to the frame
            combined = masks.max(axis=0)
            if combined.shape != (frame_h, frame_w):
                combined = cv2.resize(
                    combined.astype(np.float32), (frame_w, frame_h), interpolation=cv2.INTER_LINEAR
                )

            # Any field class becomes 1
            np.maximum(unified_mask, (combined > 0.5).view(np.uint8), out=unified_mask)

        except Exception as e:
            logger.error(f"Error creating unified mask: {e}")

    # Apply morphological operations to smooth the mask
    if cv2.countNonZero(unified_mask):
        unified_mask = apply_morphological_smoothing(unified_mask)

    return unified_mask if cv2.countNonZero(unified_mask) else None


def calculate_field_contour(
    unified_mask: np.ndarray, simplify_epsilon: float = None, min_contour_area: int = None
) -> Optional[np.ndarray]:
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
            logger.debug(f"Contour area {contour_area} below threshold {min_contour_area}")
            return None

        # Simplify contour using Douglas-Peucker algorithm
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = simplify_epsilon * perimeter
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        logger.debug(
            f"Original contour points: {len(largest_contour)}, simplified: {len(simplified_contour)}"
        )

        return simplified_contour

    except Exception as e:
        logger.error(f"Error calculating field contour: {e}")
        return None


def apply_morphological_smoothing(
    mask: np.ndarray,
    opening_kernel_size: int = None,
    closing_kernel_size: int = None,
    fill_holes: bool = None,
) -> np.ndarray:
    """Apply morphological operations to smooth a binary mask.

    Args:
        mask: Binary mask (H, W) with values 0 or 1
        opening_kernel_size: Size of kernel for opening operation (removes noise)
        closing_kernel_size: Size of kernel for closing operation (fills gaps)
        fill_holes: Whether to apply hole filling

    Returns:
        Smoothed binary mask
    """
    if not cv2.countNonZero(mask):
        return mask

    # Use config values if parameters not provided
    if opening_kernel_size is None:
        opening_kernel_size = get_setting(
            "models.segmentation.morphological.opening_kernel_size", 5
        )
    if closing_kernel_size is None:
        closing_kernel_size = get_setting(
            "models.segmentation.morphological.closing_kernel_size", 15
        )
    if fill_holes is None:
        fill_holes = get_setting("models.segmentation.morphological.fill_holes", True)

    try:
        # Ensure mask is binary
        _, mask_binary = cv2.threshold(mask.astype(np.uint8, copy=False), 0, 1, cv2.THRESH_BINARY)

        # 1. Opening operation: erosion followed by dilation
        # This removes small noise and disconnected components
        if opening_kernel_size > 0:
            opening_kernel = _get_morphological_kernel(opening_kernel_size)
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, opening_kernel)

        # 2. Closing operation: dilation followed by erosion
        # This fills small gaps and holes within the field area
        if closing_kernel_size > 0:
            closing_kernel = _get_morphological_kernel(closing_kernel_size)
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, closing_kernel)

        # 3. Fill remaining holes using optimized flood fill
        if fill_holes:
            mask_binary = _fill_holes_flood_fill_optimized(mask_binary)

        return mask_binary

    except Exception as e:
        logger.error(f"Error in morphological smoothing: {e}")
        return mask


def _fill_holes_flood_fill_optimized(mask: np.ndarray) -> np.ndarray:
    """Optimized hole filling using flood fill from the borders.

    Args:
        mask: Binary mask (H, W) with values 0 or 1

    Returns:
        Mask with holes filled
    """
    try:
        h, w = mask.shape

        # Early return if mask is empty or full
        if not np.any(mask) or np.all(mask):
            return mask

        # Create a padded mask for flood fill (more efficient than copying)
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Copy inverted mask to center (vectorized operation)
        flood_mask[1:-1, 1:-1] = 1 - mask

        # Flood fill from corner - only if corner is actually background
        if flood_mask[0, 0] == 1:
            cv2.floodFill(flood_mask, None, (0, 0), 0)

        # Extract filled region and invert back (vectorized)
        filled_mask = 1 - flood_mask[1:-1, 1:-1]

        return filled_mask.astype(np.uint8)

    except Exception as e:
        logger.error(f"Error in optimized hole filling: {e}")
        return mask


try:
    from sklearn.linear_model import RANSACRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    # Fallback implementation will be used automatically when needed


def filter_edge_points(
    contour: np.ndarray, frame_shape: tuple, edge_margin: int = 20
) -> tuple[np.ndarray, np.ndarray]:
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
        empty_result = np.array([]).reshape(0, 1, 2)
        return contour if contour is not None else empty_result, empty_result

    # Use helper function to normalize input
    points = _normalize_contour_to_points(contour)

    # Get frame dimensions
    height, width = frame_shape[:2]

    # Vectorized edge filtering - much faster than individual checks
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # Create boolean mask for valid points (vectorized operations)
    valid_mask = (
        (x_coords >= edge_margin)
        & (x_coords <= width - edge_margin)
        & (y_coords >= edge_margin)
        & (y_coords <= height - edge_margin)
    )

    # Split points using boolean indexing
    valid_points = points[valid_mask]
    edge_points = points[~valid_mask]

    # Convert back to (N, 1, 2) format using helper function
    valid_contour = _points_to_contour_format(valid_points)
    edge_contour = _points_to_contour_format(edge_points)

    return valid_contour, edge_contour


def interpolate_contour_points(
    contour: np.ndarray, max_distance: float = 10.0, min_distance: float = 3.0
) -> np.ndarray:
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

    # Use helper function to normalize input
    points = _normalize_contour_to_points(contour)
    num_points = len(points)
    interpolated_points = []

    for i in range(num_points):
        current_point = points[i]
        next_point = points[(i + 1) % num_points]  # Wrap around for closed contour

        # Always add the current point
        interpolated_points.append(current_point)

        # Calculate distance to next point (vectorized)
        diff = next_point - current_point
        distance = np.linalg.norm(diff)

        # If distance is too large, add interpolated points
        if distance > max_distance:
            # Calculate number of intermediate points needed
            num_intermediate = int(np.ceil(distance / max_distance)) - 1

            # Generate intermediate points using vectorized operations
            if num_intermediate > 0:
                # Create interpolation factors
                alphas = np.linspace(
                    1 / (num_intermediate + 1),
                    num_intermediate / (num_intermediate + 1),
                    num_intermediate,
                )

                # Vectorized interpolation
                intermediate_points = current_point + alphas[:, np.newaxis] * diff

                # Apply minimum distance constraint
                for intermediate_point in intermediate_points:
                    if (
                        len(interpolated_points) == 0
                        or np.linalg.norm(intermediate_point - interpolated_points[-1])
                        >= min_distance
                    ):
                        interpolated_points.append(intermediate_point)

    # Convert to numpy array and reshape to original format
    if len(interpolated_points) > 0:
        interpolated_array = np.array(interpolated_points, dtype=np.float32)
        return _points_to_contour_format(interpolated_array)
    else:
        return contour


def fit_field_lines_ransac(
    contour: np.ndarray,
    frame: np.ndarray,
    num_lines: int = 4,
    distance_threshold: float = 5.0,
    min_samples: int = 2,
    max_trials: int = 100,
) -> Optional[
    Tuple[
        List[Tuple[np.ndarray, np.ndarray]],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        Dict,
        Dict,
    ]
]:
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
        interpolation_enabled = get_setting(
            "models.segmentation.contour.interpolation.enabled", False
        )
        if interpolation_enabled:
            max_distance = get_setting(
                "models.segmentation.contour.interpolation.max_point_distance", 10
            )
            min_distance = get_setting(
                "models.segmentation.contour.interpolation.min_point_distance", 3
            )

            # Convert to contour format for interpolation
            contour_format = points.reshape(-1, 1, 2)
            interpolated_contour = interpolate_contour_points(
                contour_format, max_distance, min_distance
            )
            points = interpolated_contour.reshape(-1, 2).astype(np.float32)

        # Apply edge filtering after interpolation if enabled
        edge_filtering_enabled = get_setting(
            "models.segmentation.contour.ransac.edge_filtering.enabled", False
        )
        if edge_filtering_enabled:
            edge_margin = get_setting(
                "models.segmentation.contour.ransac.edge_filtering.margin", 20
            )

            # Convert to contour format for edge filtering
            contour_format = points.reshape(-1, 1, 2)
            filtered_contour, edge_points = filter_edge_points(
                contour_format, frame.shape, edge_margin
            )

            # Update points to use only non-edge points
            if len(filtered_contour) > 0:
                points = filtered_contour.reshape(-1, 2).astype(np.float32)
                edge_filtered_points = (
                    edge_points.reshape(-1, 2).astype(np.float32)
                    if len(edge_points) > 0
                    else np.array([]).reshape(0, 2)
                )

        # Sequential RANSAC: Find lines one by one, removing inliers each time
        remaining_points = points.copy()
        fitted_lines = []
        line_confidences = []  # Store confidence for each line
        all_outliers = []
        all_inliers = []

        for _ in range(num_lines):
            if len(remaining_points) < min_samples:
                # Insufficient points remaining for line fitting
                break

            # Fit line to remaining points using RANSAC
            result = _fit_line_ransac_with_outliers(
                remaining_points, distance_threshold, min_samples, max_trials
            )

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
        return (
            (
                valid_lines,
                all_outliers,
                all_inliers,
                edge_filtered_points,
                {},
                all_lines_for_display,
            )
            if valid_lines
            else (None, None, None, edge_filtered_points, {}, {})
        )

    except Exception as e:
        logger.error(f"Error in RANSAC line fitting: {e}")
        return None, None, None, np.array([]).reshape(0, 2), {}, {}


def _fit_line_ransac_with_outliers(
    points: np.ndarray, distance_threshold: float, min_samples: int, max_trials: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Fit a line using RANSAC and return the line, outliers, inliers, and confidence."""
    if len(points) < min_samples:
        return None

    # The numpy fitter is the default: sklearn spends most of its time validating
    # inputs on every trial and regresses y on x, which cannot fit vertical lines.
    backend = get_setting("models.segmentation.contour.ransac.backend", "numpy")
    if backend == "sklearn" and SKLEARN_AVAILABLE:
        return _fit_line_ransac_sklearn(points, distance_threshold, min_samples, max_trials)
    return _fit_line_ransac_numpy(points, distance_threshold, min_samples, max_trials)


def _fit_line_ransac_sklearn(
    points: np.ndarray, distance_threshold: float, min_samples: int, max_trials: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Fit a line using sklearn RANSAC."""

    try:
        # Validate input dimensions once
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < min_samples:
            return None

        # Prepare data for RANSAC (avoid reshape when possible)
        X = points[:, 0:1]  # Keep as 2D without reshape
        y = points[:, 1]  # 1D array for y values

        # Create RANSAC regressor
        ransac = RANSACRegressor(
            estimator=None,  # Use default LinearRegression
            min_samples=min_samples,
            residual_threshold=distance_threshold,
            max_trials=max_trials,
            stop_probability=0.99,
            random_state=None,
        )

        # Fit RANSAC
        ransac.fit(X, y)

        # Get and validate inlier mask
        inlier_mask = ransac.inlier_mask_
        if inlier_mask is None or len(inlier_mask) != len(points):
            return None

        # Ensure boolean mask
        if inlier_mask.dtype != bool:
            inlier_mask = inlier_mask.astype(bool)

        # Extract inliers and outliers using boolean indexing
        inliers = points[inlier_mask]
        outliers = points[~inlier_mask]

        # Early return if insufficient inliers
        if len(inliers) < 2:
            return None

        # Calculate confidence
        confidence = inlier_mask.sum() / len(points)

        # Find line endpoints efficiently
        x_coords = inliers[:, 0]
        x_min, x_max = x_coords.min(), x_coords.max()

        # Batch predict for endpoints
        endpoints_x = np.array([[x_min], [x_max]])
        endpoints_y = ransac.predict(endpoints_x)

        # Create line endpoints
        line_points = np.column_stack([endpoints_x.ravel(), endpoints_y])

        return line_points, outliers, inliers, confidence

    except Exception as e:
        logger.error(f"Error in sklearn RANSAC fitting: {e}")
        return None


def _fit_line_ransac_numpy(
    points: np.ndarray, distance_threshold: float, min_samples: int, max_trials: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Fit a line with RANSAC using perpendicular distances, then refit on the inliers."""

    try:
        num_points = len(points)
        if points.ndim != 2 or points.shape[1] != 2 or num_points < max(2, min_samples):
            return None

        xs = points[:, 0].astype(np.float64)
        ys = points[:, 1].astype(np.float64)
        threshold_sq = distance_threshold**2  # Use squared distance to avoid sqrt

        best_inliers = None
        best_inlier_count = 0

        # A line hypothesis only ever needs two points
        samples = np.random.randint(0, num_points, size=(max_trials, 2))
        for i1, i2 in samples:
            x1, y1 = xs[i1], ys[i1]
            dx, dy = xs[i2] - x1, ys[i2] - y1
            norm_sq = dx * dx + dy * dy

            # Skip if points are too close
            if norm_sq < 1e-12:
                continue

            # Squared perpendicular distance of every point to the line through the pair
            cross = dx * (ys - y1) - dy * (xs - x1)
            inlier_mask = cross * cross <= threshold_sq * norm_sq
            inlier_count = int(np.count_nonzero(inlier_mask))

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inlier_mask

        if best_inliers is None or best_inlier_count < 2:
            return None

        inliers = points[best_inliers]
        outliers = points[~best_inliers]

        # Total least squares refit on the consensus set (independent of orientation)
        inliers_f64 = inliers.astype(np.float64)
        centroid = inliers_f64.mean(axis=0)
        _, _, vt = np.linalg.svd(inliers_f64 - centroid, full_matrices=False)
        direction = vt[0]
        if direction[0] < 0:
            direction = -direction  # Keep endpoints ordered by increasing x

        # Endpoints are the extreme inlier projections onto the fitted line
        projections = (inliers_f64 - centroid) @ direction
        line_points = np.array(
            [
                centroid + projections.min() * direction,
                centroid + projections.max() * direction,
            ]
        )

        confidence = best_inlier_count / num_points
        return line_points, outliers, inliers, confidence

    except Exception as e:
        logger.error(f"Error in numpy RANSAC fitting: {e}")
        return None


def extract_raw_lines_from_segmentation(
    segmentation_results: List[Any], frame_shape: Tuple[int, int]
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """Extract raw RANSAC lines directly from segmentation results.

    Args:
        segmentation_results: YOLO segmentation results
        frame_shape: Shape of the frame (height, width)

    Returns:
        Tuple of (detected_lines, confidences) where:
        - detected_lines: List of (start_point, end_point) tuples
        - confidences: List of confidence scores for each line
    """
    unified_mask = create_unified_field_mask(segmentation_results, frame_shape)
    detected_lines, confidences, _, _ = fit_lines_from_mask(unified_mask)
    return detected_lines, confidences


def fit_lines_from_mask(
    unified_mask: np.ndarray,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float], Optional[np.ndarray], Optional[tuple]]:
    """Fit RANSAC lines to a unified mask, keeping the intermediate geometry.

    RANSAC is randomized, so callers that both use and draw the lines should
    run it once through this function and reuse the returned fit.

    Args:
        unified_mask: Binary mask where 1 indicates field area

    Returns:
        Tuple of (detected_lines, confidences, simplified_contour, ransac_fit) where
        ransac_fit is the raw fit_field_lines_ransac result (None if no contour was found)
    """
    detected_lines = []
    confidences = []
    simplified_contour = None
    result = None

    if unified_mask is None or not np.any(unified_mask):
        return detected_lines, confidences, simplified_contour, result

    try:
        # Import here to avoid circular imports

        # Calculate contour for RANSAC
        simplified_contour = calculate_field_contour(unified_mask)

        if simplified_contour is not None:
            # Get RANSAC parameters
            num_lines = get_setting("models.segmentation.contour.ransac.num_lines", 4)
            distance_threshold = get_setting(
                "models.segmentation.contour.ransac.distance_threshold", 10.0
            )
            min_samples = get_setting("models.segmentation.contour.ransac.min_samples", 2)
            max_trials = get_setting("models.segmentation.contour.ransac.max_trials", 1000)

            # Create dummy frame for RANSAC (only shape is used)
            frame_shape = unified_mask.shape
            dummy_frame = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)

            # Run RANSAC line fitting
            result = fit_field_lines_ransac(
                simplified_contour,
                dummy_frame,
                num_lines=num_lines,
                distance_threshold=distance_threshold,
                min_samples=min_samples,
                max_trials=max_trials,
            )

            if result and result[0]:  # Check if fitted_lines exist
                fitted_lines, outlier_points, inlier_points, edge_filtered_points, _, _ = result

                # Extract lines with confidence based on inlier ratio
                total_contour_points = len(simplified_contour)

                for i, (start_point, end_point) in enumerate(fitted_lines):
                    detected_lines.append((start_point, end_point))

                    # Calculate confidence based on inlier count
                    if i < len(inlier_points) and len(inlier_points[i]) > 0:
                        inlier_count = len(inlier_points[i])
                        # Confidence based on inlier ratio (normalized to expected points per line)
                        expected_points_per_line = max(1, total_contour_points // num_lines)
                        confidence = min(0.95, inlier_count / expected_points_per_line)
                    else:
                        confidence = 0.3  # Low confidence for lines without inliers

                    confidences.append(confidence)

                logger.debug(
                    f"Extracted {len(detected_lines)} lines from mask with confidences: {[f'{c:.3f}' for c in confidences]}"
                )

    except Exception as e:
        logger.error(f"Error extracting lines from mask: {e}")

    return detected_lines, confidences, simplified_contour, result
