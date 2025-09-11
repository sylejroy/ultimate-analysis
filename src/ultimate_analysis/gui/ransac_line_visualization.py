"""Drawing functions for RANSAC-calculated field lines.

This module provides visualization functions for displaying
RANSAC field lines in the main and top-down views.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def draw_ransac_field_lines(frame: np.ndarray, 
                           ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
                           confidences: List[float],
                           transformation_matrix: Optional[np.ndarray] = None,
                           scale_factor: float = 1.0) -> np.ndarray:
    """Draw RANSAC-calculated field lines with clean visualization.
    
    Args:
        frame: Frame to draw on
        ransac_lines: List of (start_point, end_point) tuples from RANSAC
        confidences: List of confidence scores for each line
        transformation_matrix: Optional homography matrix to transform lines to warped view
        scale_factor: Scale factor for text and line thickness (useful for top-down view)
        
    Returns:
        Frame with RANSAC lines drawn
    """
    if not ransac_lines:
        return frame
        
    result = frame.copy()
    
    try:
        # Color scheme based on confidence
        line_colors = {
            'excellent': (64, 255, 64),      # Bright green - excellent confidence (>0.8)
            'good': (0, 200, 255),           # Orange-yellow - good confidence (0.6-0.8)
            'fair': (0, 165, 255),           # Orange - fair confidence (0.4-0.6)
            'low': (0, 100, 255)             # Red-orange - low confidence (<0.4)
        }
        
        # Line thickness based on scale factor
        base_thickness = max(1, int(2 * scale_factor))
        
        for i, ((start_point, end_point), confidence) in enumerate(zip(ransac_lines, confidences)):
            # Transform line if homography matrix provided
            if transformation_matrix is not None:
                # Convert points to homogeneous coordinates
                start_homo = np.array([start_point[0], start_point[1], 1.0])
                end_homo = np.array([end_point[0], end_point[1], 1.0])
                
                # Apply transformation
                start_transformed = transformation_matrix @ start_homo
                end_transformed = transformation_matrix @ end_homo
                
                # Convert back to 2D coordinates
                if start_transformed[2] != 0 and end_transformed[2] != 0:
                    start_2d = (start_transformed[:2] / start_transformed[2]).astype(int)
                    end_2d = (end_transformed[:2] / end_transformed[2]).astype(int)
                else:
                    continue  # Skip if transformation fails
            else:
                start_2d = start_point.astype(int)
                end_2d = end_point.astype(int)
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = line_colors['excellent']
            elif confidence > 0.6:
                color = line_colors['good']
            elif confidence > 0.4:
                color = line_colors['fair']
            else:
                color = line_colors['low']
            
            # Draw the line
            cv2.line(result, tuple(start_2d), tuple(end_2d), color, base_thickness)
            
            # Optionally add confidence text near the line (for debugging)
            if scale_factor >= 1.5:  # Only show text at larger scales
                mid_point = ((start_2d + end_2d) // 2).astype(int)
                font_scale = 0.4 * scale_factor
                cv2.putText(result, f"{confidence:.2f}", tuple(mid_point), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        # Add summary info if there are lines
        if len(ransac_lines) > 0:
            _draw_ransac_summary(result, ransac_lines, confidences, scale_factor)
        
        return result
        
    except Exception as e:
        print(f"[RANSAC_LINES] Error drawing RANSAC lines: {e}")
        return frame


def _draw_ransac_summary(frame: np.ndarray, 
                        ransac_lines: List[Tuple[np.ndarray, np.ndarray]], 
                        confidences: List[float], 
                        scale_factor: float):
    """Draw summary information about RANSAC lines."""
    total_lines = len(ransac_lines)
    excellent_count = len([c for c in confidences if c > 0.8])
    good_count = len([c for c in confidences if 0.6 < c <= 0.8])
    avg_confidence = np.mean(confidences)
    
    # Position for text overlay
    text_y_start = 30
    font_scale = 0.5 * scale_factor
    text_color = (255, 255, 255)  # White
    
    # Draw summary
    summary_lines = [
        f"RANSAC Lines: {total_lines}",
        f"Excellent: {excellent_count}, Good: {good_count}",
        f"Avg Confidence: {avg_confidence:.2f}"
    ]
    
    for i, line in enumerate(summary_lines):
        y_pos = text_y_start + int(i * 20 * scale_factor)
        cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, text_color, 1)
