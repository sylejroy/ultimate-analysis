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

from typing import Any, List, Optional, Tuple

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


# Result of fit_field_lines_ransac: (lines, points on no line, points of each line, points
# left out for being at the frame border). A line is a (2, 2) array of its two end points.
RansacFit = Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]


def fit_field_lines_ransac(
    contour: np.ndarray,
    frame_shape: Tuple[int, ...],
    num_lines: int = 4,
    distance_threshold: float = 5.0,
    min_samples: int = 2,
    max_trials: int = 100,
) -> RansacFit:
    """Fit straight lines to the outline of the field, strongest line first.

    Args:
        contour: Outline points, shape (N, 1, 2)
        frame_shape: Shape of the frame; outline points at its border are left out
        num_lines: The most lines to look for
        distance_threshold: A point this close to a line (pixels) belongs to it
        min_samples: Fewest points a line needs
        max_trials: Point pairs tried per line

    Returns:
        (lines, outliers, inliers, edge_points); `lines` is empty if none was found.
        `outliers` holds one array, the points on no line; `inliers` one array per line.
    """
    no_points = np.empty((0, 2), dtype=np.float32)
    if contour is None or len(contour) < num_lines * min_samples:
        return [], [], [], no_points

    try:
        points = contour.reshape(-1, 2).astype(np.float32)
        edge_points = no_points

        if get_setting("models.segmentation.contour.interpolation.enabled", False):
            points = interpolate_contour_points(
                points.reshape(-1, 1, 2),
                get_setting("models.segmentation.contour.interpolation.max_point_distance", 10),
                get_setting("models.segmentation.contour.interpolation.min_point_distance", 3),
            )
            points = points.reshape(-1, 2).astype(np.float32)

        if get_setting("models.segmentation.contour.ransac.edge_filtering.enabled", False):
            kept, at_edge = filter_edge_points(
                points.reshape(-1, 1, 2),
                frame_shape,
                get_setting("models.segmentation.contour.ransac.edge_filtering.margin", 20),
            )
            if len(kept) > 0:
                points = kept.reshape(-1, 2).astype(np.float32)
                edge_points = at_edge.reshape(-1, 2).astype(np.float32)

        # One line after the other, each from the points the lines before it left over
        min_line_length = get_setting("models.segmentation.contour.ransac.min_line_length", 60)
        remaining = points.copy()
        lines: List[np.ndarray] = []
        inliers: List[np.ndarray] = []
        for _ in range(num_lines):
            if len(remaining) < min_samples:
                break
            found = _fit_line_ransac_numpy(remaining, distance_threshold, min_samples, max_trials)
            if found is None:
                break
            line, rest, on_line, _ = found
            # The lines come strongest first. A view often shows fewer sides of the
            # field than num_lines: what is left then are corners and dents of the
            # outline, and a line through a few of those is not a field line.
            if np.linalg.norm(line[1] - line[0]) < min_line_length:
                break
            lines.append(line)
            inliers.append(on_line)
            # The points of this line are taken out, and those just outside the
            # threshold with them: they would otherwise give the same line again
            remaining = rest[_distance_to_segment(rest, line[0], line[1]) > 2 * distance_threshold]

        return lines, [remaining], inliers, edge_points

    except Exception as e:
        logger.error(f"Error in RANSAC line fitting: {e}")
        return [], [], [], no_points


def _distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Distance of each point (N, 2) to the segment from start to end."""
    start = np.asarray(start, dtype=np.float64)
    along = np.asarray(end, dtype=np.float64) - start
    offsets = points.astype(np.float64) - start
    position = np.clip(offsets @ along / max(float(along @ along), 1e-12), 0.0, 1.0)
    return np.linalg.norm(offsets - position[:, None] * along, axis=1)


def _fit_line_ransac_numpy(
    points: np.ndarray, distance_threshold: float, min_samples: int, max_trials: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Fit a line with RANSAC using perpendicular distances, then refit on the inliers.

    The same points always give the same line: the pairs are drawn with a fixed seed. A
    fit that changed from one run to the next made the field lines flicker.
    """

    try:
        num_points = len(points)
        if points.ndim != 2 or points.shape[1] != 2 or num_points < max(2, min_samples):
            return None

        points_f64 = points.astype(np.float64)
        xs, ys = points_f64[:, 0], points_f64[:, 1]
        threshold_sq = distance_threshold**2  # Use squared distance to avoid sqrt

        # A line hypothesis only ever needs two points. All pairs are tried at once.
        samples = np.random.default_rng(0).integers(0, num_points, size=(max_trials, 2))
        x1, y1 = xs[samples[:, 0]], ys[samples[:, 0]]
        dx, dy = xs[samples[:, 1]] - x1, ys[samples[:, 1]] - y1
        norm_sq = dx * dx + dy * dy
        # Two points close together give the direction of the line badly
        usable = norm_sq > (4 * distance_threshold) ** 2
        if not usable.any():
            usable = norm_sq > 1e-12
        if not usable.any():
            return None
        x1, y1, dx, dy, norm_sq = x1[usable], y1[usable], dx[usable], dy[usable], norm_sq[usable]

        # Squared perpendicular distance of every point to the line through each pair
        cross = dx[:, None] * (ys[None, :] - y1[:, None]) - dy[:, None] * (
            xs[None, :] - x1[:, None]
        )
        inlier_masks = cross * cross <= threshold_sq * norm_sq[:, None]
        best_inliers = inlier_masks[int(np.argmax(inlier_masks.sum(axis=1)))]
        if int(best_inliers.sum()) < 2:
            return None

        # Total least squares refit on the consensus set (independent of orientation).
        # The refitted line has slightly different inliers than the pair it came from;
        # taking those and fitting again settles on the line through all of its points.
        for _ in range(3):
            centroid = points_f64[best_inliers].mean(axis=0)
            _, _, vt = np.linalg.svd(points_f64[best_inliers] - centroid, full_matrices=False)
            direction = vt[0]
            offsets = points_f64 - centroid
            distance = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])
            refined = distance <= distance_threshold
            if int(refined.sum()) < 2 or np.array_equal(refined, best_inliers):
                break
            best_inliers = refined
        best_inlier_count = int(best_inliers.sum())

        inliers = points[best_inliers]
        outliers = points[~best_inliers]
        inliers_f64 = points_f64[best_inliers]
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
) -> Tuple[List[np.ndarray], List[float], Optional[np.ndarray], Optional[RansacFit]]:
    """Fit the field lines to a field mask, keeping the outline and the fit for drawing.

    Args:
        unified_mask: Binary mask where 1 indicates field area

    Returns:
        (lines, confidences, outline, fit). The fit is None if the mask has no outline.
    """
    if unified_mask is None or not np.any(unified_mask):
        return [], [], None, None

    try:
        contour = calculate_field_contour(unified_mask)
        if contour is None:
            return [], [], None, None

        num_lines = get_setting("models.segmentation.contour.ransac.num_lines", 4)
        fit = fit_field_lines_ransac(
            contour,
            unified_mask.shape,
            num_lines=num_lines,
            distance_threshold=get_setting(
                "models.segmentation.contour.ransac.distance_threshold", 10.0
            ),
            min_samples=get_setting("models.segmentation.contour.ransac.min_samples", 2),
            max_trials=get_setting("models.segmentation.contour.ransac.max_trials", 1000),
        )
        lines, _, inliers, _ = fit
        # Confidence of a line: its share of the points a line would have if the
        # outline were split evenly between the lines looked for
        points_per_line = max(1, len(contour) // num_lines)
        confidences = [
            min(0.95, len(on_line) / points_per_line) if len(on_line) > 0 else 0.3
            for on_line in inliers
        ]
        return list(lines), confidences, contour, fit

    except Exception as e:
        logger.error(f"Error extracting lines from mask: {e}")
        return [], [], None, None
