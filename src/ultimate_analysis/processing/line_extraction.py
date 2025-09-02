"""Line extraction utilities for direct RANSAC line access.

This module provides functions to extract raw lines from field segmentation
for Kalman tracking, bypassing the classification system.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any

from ..config.settings import get_setting


def extract_raw_lines_from_segmentation(segmentation_results: List[Any], 
                                       frame_shape: Tuple[int, int]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """Extract raw RANSAC lines directly from segmentation results.
    
    Args:
        segmentation_results: YOLO segmentation results
        frame_shape: Shape of the frame (height, width)
        
    Returns:
        Tuple of (detected_lines, confidences) where:
        - detected_lines: List of (start_point, end_point) tuples
        - confidences: List of confidence scores for each line
    """
    detected_lines = []
    confidences = []
    
    try:
        # Import here to avoid circular imports
        from ..gui.visualization import calculate_field_contour
        from ..processing.field_analysis import fit_field_lines_ransac, create_unified_field_mask_processing
        
        # Create unified mask from segmentation
        unified_mask = create_unified_field_mask_processing(segmentation_results, frame_shape)
        
        if unified_mask is not None and np.any(unified_mask):
            # Calculate contour for RANSAC
            simplified_contour = calculate_field_contour(unified_mask)
            
            if simplified_contour is not None:
                # Get RANSAC parameters
                num_lines = get_setting("models.segmentation.contour.ransac.num_lines", 4)
                distance_threshold = get_setting("models.segmentation.contour.ransac.distance_threshold", 10.0)
                min_samples = get_setting("models.segmentation.contour.ransac.min_samples", 2)
                max_trials = get_setting("models.segmentation.contour.ransac.max_trials", 1000)
                
                # Create dummy frame for RANSAC (only shape is used)
                dummy_frame = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
                
                # Run RANSAC line fitting
                result = fit_field_lines_ransac(
                    simplified_contour,
                    dummy_frame,
                    num_lines=num_lines,
                    distance_threshold=distance_threshold,
                    min_samples=min_samples,
                    max_trials=max_trials
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
                    
                    print(f"[LINE_EXTRACTION] Extracted {len(detected_lines)} raw lines with confidences: {[f'{c:.3f}' for c in confidences]}")
                
    except Exception as e:
        print(f"[LINE_EXTRACTION] Error extracting lines: {e}")
    
    return detected_lines, confidences


def extract_lines_from_mask(unified_mask: np.ndarray) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """Extract raw RANSAC lines directly from a unified mask.
    
    Args:
        unified_mask: Binary mask where 1 indicates field area
        
    Returns:
        Tuple of (detected_lines, confidences)
    """
    detected_lines = []
    confidences = []
    
    if unified_mask is None or not np.any(unified_mask):
        return detected_lines, confidences
    
    try:
        # Import here to avoid circular imports
        from ..gui.visualization import calculate_field_contour
        from ..processing.field_analysis import fit_field_lines_ransac
        
        # Calculate contour for RANSAC
        simplified_contour = calculate_field_contour(unified_mask)
        
        if simplified_contour is not None:
            # Get RANSAC parameters
            num_lines = get_setting("models.segmentation.contour.ransac.num_lines", 4)
            distance_threshold = get_setting("models.segmentation.contour.ransac.distance_threshold", 10.0)
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
                max_trials=max_trials
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
                
                print(f"[LINE_EXTRACTION] Extracted {len(detected_lines)} lines from mask with confidences: {[f'{c:.3f}' for c in confidences]}")
        
    except Exception as e:
        print(f"[LINE_EXTRACTION] Error extracting lines from mask: {e}")
    
    return detected_lines, confidences
