"""Drawing functions for Kalman-tracked field lines.

This module provides visualization functions specifically for displaying
tracked field lines in the top-down view with Kalman filtering.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from ..config.settings import get_setting


def draw_tracked_field_lines(frame: np.ndarray, tracked_lines: List[Dict[str, Any]],
                           transformation_matrix: Optional[np.ndarray] = None,
                           scale_factor: float = 1.0) -> np.ndarray:
    """Draw Kalman-tracked field lines with clean, modern visualization.
    
    Args:
        frame: Frame to draw on
        tracked_lines: List of tracked line dictionaries from Kalman filter
        transformation_matrix: Optional homography matrix to transform lines to warped view
        scale_factor: Scale factor for text and line thickness (useful for top-down view)
        
    Returns:
        Frame with tracked lines drawn
    """
    if not tracked_lines:
        return frame
        
    result = frame.copy()
    
    # Modern, clean color scheme
    line_colors = {
        'excellent': (64, 255, 64),      # Bright green - excellent tracking (>0.8)
        'good': (0, 200, 255),           # Orange-yellow - good tracking (0.6-0.8)
        'fair': (0, 165, 255),           # Orange - fair tracking (0.4-0.6)
        'poor': (0, 100, 200),           # Dark red - poor tracking (<0.4)
        'predicted': (200, 0, 200),      # Purple - predicted lines (no recent detection)
        'new': (255, 200, 0),            # Light blue - newly created tracks
    }
    
    # Cleaner scaling
    line_thickness = max(2, int(3 * scale_factor))
    text_scale = max(0.5, 0.7 * scale_factor)
    font = cv2.FONT_HERSHEY_DUPLEX  # Cleaner font
    
    # Group lines by quality for better visual hierarchy
    excellent_lines = []
    good_lines = []
    other_lines = []
    
    try:
        for track_data in tracked_lines:
            confidence = track_data['confidence']
            if confidence > 0.8:
                excellent_lines.append(track_data)
            elif confidence > 0.6:
                good_lines.append(track_data)
            else:
                other_lines.append(track_data)
        
        # Draw lines in order: other -> good -> excellent (excellent on top)
        for line_group in [other_lines, good_lines, excellent_lines]:
            for track_data in line_group:
                _draw_single_tracked_line(result, track_data, transformation_matrix, 
                                        line_colors, line_thickness, text_scale, font, scale_factor)
        
        # Add clean summary overlay
        if len(tracked_lines) > 0:
            _draw_tracking_summary(result, tracked_lines, scale_factor)
        
    except Exception as e:
        print(f"[TRACKED_LINES] Error drawing tracked lines: {e}")
    
    return result


def _draw_single_tracked_line(frame: np.ndarray, track_data: Dict[str, Any], 
                            transformation_matrix: Optional[np.ndarray],
                            line_colors: Dict[str, Tuple[int, int, int]], 
                            line_thickness: int, text_scale: float, font: int, scale_factor: float):
    """Draw a single tracked line with clean styling."""
    start_point = track_data['start_point']
    end_point = track_data['end_point']
    confidence = track_data['confidence']
    track_id = track_data['track_id']
    age = track_data['age']
    is_predicted = track_data.get('is_predicted', False)
    
    # Transform line if homography matrix provided
    if transformation_matrix is not None:
        # Convert to homogeneous coordinates
        start_homo = np.array([start_point[0], start_point[1], 1], dtype=np.float32)
        end_homo = np.array([end_point[0], end_point[1], 1], dtype=np.float32)
        
        # Apply transformation
        start_transformed = transformation_matrix @ start_homo
        end_transformed = transformation_matrix @ end_homo
        
        # Convert back to Cartesian coordinates
        if start_transformed[2] != 0 and end_transformed[2] != 0:
            start_point = start_transformed[:2] / start_transformed[2]
            end_point = end_transformed[:2] / end_transformed[2]
        else:
            return  # Skip invalid transformations
    
    # Determine line color and style based on confidence and state
    if is_predicted:
        color = line_colors['predicted']
        status = "PRED"
        alpha = 0.6  # Semi-transparent for predicted
    elif age < 3:
        color = line_colors['new']
        status = "NEW"
        alpha = 0.8
    elif confidence > 0.8:
        color = line_colors['excellent']
        status = "★"  # Star for excellent
        alpha = 1.0
    elif confidence > 0.6:
        color = line_colors['good']
        status = "●"  # Dot for good
        alpha = 0.9
    elif confidence > 0.4:
        color = line_colors['fair']
        status = "○"  # Circle for fair
        alpha = 0.8
    else:
        color = line_colors['poor']
        status = "◦"  # Small circle for poor
        alpha = 0.7
    
    # Convert points to integers for drawing
    start = tuple(start_point.astype(np.int32))
    end = tuple(end_point.astype(np.int32))
    
    # Draw the main line with slight glow effect for better visibility
    if confidence > 0.6:  # Add glow for high-confidence lines
        glow_thickness = line_thickness + 2
        glow_color = tuple(int(c * 0.3) for c in color)  # Darker glow
        cv2.line(frame, start, end, glow_color, glow_thickness)
    
    cv2.line(frame, start, end, color, line_thickness)
    
    # Draw clean endpoint markers
    marker_radius = max(3, int(4 * scale_factor))
    cv2.circle(frame, start, marker_radius, color, -1)
    cv2.circle(frame, end, marker_radius, color, -1)
    
    # Add clean text label
    if get_setting("models.segmentation.contour.show_labels", True):
        # Calculate label position (offset from midpoint to avoid overlap)
        mid_x = int((start[0] + end[0]) / 2)
        mid_y = int((start[1] + end[1]) / 2)
        
        # Offset label slightly to avoid line overlap
        label_x = mid_x + int(10 * scale_factor)
        label_y = mid_y - int(5 * scale_factor)
        
        # Create clean, minimal label
        if confidence > 0.6:
            label = f"{status} {confidence:.2f}"  # Clean format for good lines
        else:
            label = f"T{track_id} {confidence:.2f}"  # Show ID for lower confidence
        
        # Get text size for clean background
        (text_width, text_height), baseline = cv2.getTextSize(label, font, text_scale, 1)
        
        # Draw semi-transparent background
        bg_alpha = 0.8
        bg_color = (20, 20, 20)  # Dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                    (label_x - 3, label_y - text_height - 3),
                    (label_x + text_width + 3, label_y + baseline + 3),
                    bg_color, -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
        
        # Draw crisp white text
        text_color = (255, 255, 255)
        cv2.putText(frame, label, (label_x, label_y), font, text_scale, text_color, 1, cv2.LINE_AA)


def _draw_tracking_summary(frame: np.ndarray, tracked_lines: List[Dict[str, Any]], scale_factor: float):
    """Draw a clean tracking summary overlay."""
    # Calculate statistics
    total_tracks = len(tracked_lines)
    excellent_count = len([t for t in tracked_lines if t['confidence'] > 0.8])
    good_count = len([t for t in tracked_lines if 0.6 < t['confidence'] <= 0.8])
    predicted_count = len([t for t in tracked_lines if t.get('is_predicted', False)])
    avg_confidence = np.mean([t['confidence'] for t in tracked_lines])
    
    # Position summary in top-right corner
    frame_height, frame_width = frame.shape[:2]
    summary_x = frame_width - int(200 * scale_factor)
    summary_y = int(30 * scale_factor)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.4, 0.5 * scale_factor)
    line_height = int(20 * scale_factor)
    
    # Summary text with clean formatting
    summary_lines = [
        f"Field Lines: {total_tracks}",
        f"★ Excellent: {excellent_count}",
        f"● Good: {good_count}",
        f"◦ Predicted: {predicted_count}",
        f"Avg: {avg_confidence:.2f}"
    ]
    
    # Draw background for summary
    bg_width = int(180 * scale_factor)
    bg_height = len(summary_lines) * line_height + int(10 * scale_factor)
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                (summary_x - 10, summary_y - 10),
                (summary_x + bg_width, summary_y + bg_height),
                (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw summary text
    for i, text in enumerate(summary_lines):
        y_pos = summary_y + i * line_height
        if i == 0:  # Title
            color = (255, 255, 255)
        elif "★" in text:  # Excellent
            color = (64, 255, 64)
        elif "●" in text:  # Good
            color = (0, 200, 255)
        else:  # Other
            color = (200, 200, 200)
        
        cv2.putText(frame, text, (summary_x, y_pos), font, font_scale, color, 1, cv2.LINE_AA)


def draw_tracking_overlay(frame: np.ndarray, tracked_lines: List[Dict[str, Any]]) -> np.ndarray:
    """Draw tracking overlay with statistics.
    
    Args:
        frame: Frame to draw on
        tracked_lines: List of tracked line data
        
    Returns:
        Frame with tracking overlay
    """
    if not tracked_lines:
        return frame
    
    result = frame.copy()
    
    # Calculate tracking statistics
    total_tracks = len(tracked_lines)
    high_conf_tracks = len([t for t in tracked_lines if t['confidence'] > 0.7])
    predicted_tracks = len([t for t in tracked_lines if t.get('is_predicted', False)])
    avg_confidence = np.mean([t['confidence'] for t in tracked_lines])
    
    # Draw statistics overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White
    thickness = 1
    
    stats_text = [
        f"Tracked Lines: {total_tracks}",
        f"High Confidence: {high_conf_tracks}",
        f"Predicted: {predicted_tracks}",
        f"Avg Confidence: {avg_confidence:.3f}"
    ]
    
    y_offset = 30
    for i, text in enumerate(stats_text):
        y_pos = y_offset + i * 25
        
        # Draw text background
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(result, (10, y_pos - text_height - 5), 
                     (20 + text_width, y_pos + baseline + 5), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(result, text, (15, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return result


def classify_tracked_lines(tracked_lines: List[Dict[str, Any]], 
                         frame_shape: Tuple[int, int]) -> Dict[str, Dict[str, Any]]:
    """Classify tracked lines into field components for top-down view.
    
    Args:
        tracked_lines: List of tracked line data
        frame_shape: Shape of the frame (height, width)
        
    Returns:
        Dictionary mapping line types to tracked line data
    """
    if not tracked_lines:
        return {}
    
    frame_height, frame_width = frame_shape
    classified = {}
    
    # Separate lines by orientation
    horizontal_lines = []
    vertical_lines = []
    
    for track in tracked_lines:
        start_point = track['start_point']
        end_point = track['end_point']
        
        # Calculate line angle
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Normalize angle to [0, 180)
        if angle < 0:
            angle += 180
        
        # Determine orientation (15-degree tolerance)
        is_horizontal = abs(angle - 0) < 15 or abs(angle - 180) < 15
        
        if is_horizontal:
            avg_y = (start_point[1] + end_point[1]) / 2
            horizontal_lines.append((track, avg_y))
        else:
            avg_x = (start_point[0] + end_point[0]) / 2
            vertical_lines.append((track, avg_x))
    
    # Sort and classify horizontal lines (top to bottom)
    horizontal_lines.sort(key=lambda x: x[1])
    for i, (track, avg_y) in enumerate(horizontal_lines):
        if i == 0:
            classified['far_endzone'] = track
        elif i == 1:
            classified['near_endzone'] = track
        else:
            classified[f'horizontal_line_{i}'] = track
    
    # Sort and classify vertical lines (left to right)
    vertical_lines.sort(key=lambda x: x[1])
    for i, (track, avg_x) in enumerate(vertical_lines):
        if i == 0:
            classified['left_sideline'] = track
        elif i == 1:
            classified['right_sideline'] = track
        else:
            classified[f'vertical_line_{i}'] = track
    
    print(f"[VISUALIZATION] Classified {len(classified)} tracked lines: {list(classified.keys())}")
    return classified
