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
    """Draw Kalman-tracked field lines with classification colors in top-down view.
    
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
    
    # Define colors for different confidence levels and tracking states
    line_colors = {
        'high_confidence': (0, 255, 0),    # Green - high confidence tracked lines
        'medium_confidence': (0, 255, 255), # Yellow - medium confidence
        'low_confidence': (0, 165, 255),   # Orange - low confidence
        'predicted': (255, 0, 255),        # Magenta - predicted lines (no recent detection)
        'new_track': (255, 255, 0),        # Cyan - newly created tracks
    }
    
    # Calculate text and line scaling
    text_scale = max(0.4, 0.6 * scale_factor)
    line_thickness = max(1, int(2 * scale_factor))
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    try:
        for i, track_data in enumerate(tracked_lines):
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
                    continue  # Skip invalid transformations
            
            # Determine line color based on confidence and state
            if is_predicted:
                color = line_colors['predicted']
                line_type = "PRED"
            elif age < 3:
                color = line_colors['new_track']
                line_type = "NEW"
            elif confidence > 0.7:
                color = line_colors['high_confidence']
                line_type = "HIGH"
            elif confidence > 0.4:
                color = line_colors['medium_confidence']
                line_type = "MED"
            else:
                color = line_colors['low_confidence']
                line_type = "LOW"
            
            # Convert points to integers for drawing
            start = tuple(start_point.astype(np.int32))
            end = tuple(end_point.astype(np.int32))
            
            # Draw the line
            cv2.line(result, start, end, color, line_thickness)
            
            # Draw endpoint markers
            marker_radius = max(2, int(3 * scale_factor))
            cv2.circle(result, start, marker_radius, color, -1)
            cv2.circle(result, end, marker_radius, color, -1)
            
            # Add text label with track info
            if get_setting("models.segmentation.contour.show_labels", True):
                # Calculate midpoint for label placement
                mid_x = int((start[0] + end[0]) / 2)
                mid_y = int((start[1] + end[1]) / 2)
                
                # Create label with track ID, confidence, and type
                label = f"T{track_id}({line_type}:{confidence:.2f})"
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(label, font, text_scale, 1)
                
                # Draw text background
                bg_color = (0, 0, 0)  # Black background
                cv2.rectangle(result, 
                            (mid_x - text_width//2 - 2, mid_y - text_height - baseline - 2),
                            (mid_x + text_width//2 + 2, mid_y + baseline + 2),
                            bg_color, -1)
                
                # Draw text
                text_color = (255, 255, 255)  # White text
                cv2.putText(result, label, (mid_x - text_width//2, mid_y - baseline),
                          font, text_scale, text_color, 1, cv2.LINE_AA)
        
        print(f"[VISUALIZATION] Drew {len(tracked_lines)} tracked field lines")
        
    except Exception as e:
        print(f"[VISUALIZATION] Error drawing tracked lines: {e}")
    
    return result


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
