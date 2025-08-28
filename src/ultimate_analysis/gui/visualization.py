"""Visualization functions for Ultimate Analysis GUI.

This module provides functions for drawing detection boxes, tracking overlays,
player IDs, and field segmentation on video frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from ..constants import VISUALIZATION_COLORS
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


def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw detection bounding boxes and labels on frame.
    
    Args:
        frame: Input frame to draw on
        detections: List of detection dictionaries
        
    Returns:
        Frame with detection overlays
    """
    if not detections:
        return frame
    
    # Create a copy to avoid modifying original
    vis_frame = frame.copy()
    
    for detection in detections:
        bbox = detection.get('bbox', [])
        confidence = detection.get('confidence', 0.0)
        class_name = detection.get('class_name', 'unknown')
        
        if len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Choose color based on class - disc should be bright and easy to spot, player subtle
        color = VISUALIZATION_COLORS['DETECTION_BOX']  # Default fallback (green)
        
        # Ensure we have a valid class_name
        if class_name and isinstance(class_name, str):
            class_name_lower = class_name.lower().strip()
            
            if class_name_lower == 'disc' or 'disc' in class_name_lower:
                color = VISUALIZATION_COLORS['DISC']    # Bright cyan for disc - very easy to spot
            elif class_name_lower == 'player' or 'player' in class_name_lower:
                color = VISUALIZATION_COLORS['PLAYER']  # Subtle gray for player
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Draw label background
        cv2.rectangle(
            vis_frame, 
            (x1, y1 - label_size[1] - 10), 
            (x1 + label_size[0], y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            vis_frame, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
    
    return vis_frame


def draw_tracks_with_player_ids(frame: np.ndarray, tracks: List[Any], 
                                track_histories: Optional[Dict[int, List[Tuple[int, int]]]] = None,
                                player_ids: Optional[Dict[int, Tuple[str, Any]]] = None) -> np.ndarray:
    """Draw tracking bounding boxes with player jersey numbers and confidence.
    
    Args:
        frame: Input frame to draw on
        tracks: List of track objects
        track_histories: Optional dictionary of track histories
        player_ids: Optional dictionary mapping track_id -> (jersey_number, details)
        
    Returns:
        Frame with tracking and player ID overlays
    """
    if not tracks:
        return frame
    
    vis_frame = frame.copy()
    
    for track in tracks:
        # Get track properties
        track_id = getattr(track, 'track_id', None)
        if track_id is None:
            continue
            
        # Get bounding box
        bbox = None
        if hasattr(track, 'to_ltrb'):
            bbox = track.to_ltrb()
        elif hasattr(track, 'to_tlbr'):
            bbox = track.to_tlbr()
        elif hasattr(track, 'bbox'):
            bbox = track.bbox
        
        if bbox is None or len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Generate unique color for each track ID
        color = _get_track_color(track_id)
        
        # Draw thinner bounding box (thickness 1 instead of 3)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 1)
        
        # Handle player ID display
        jersey_number = "Unknown"
        details = None
        
        if player_ids and track_id in player_ids:
            jersey_number, details = player_ids[track_id]
        
        # Always show track ID and jersey number when using player ID mode
        if jersey_number != "Unknown":
            # Create simple jersey number label without confidence
            jersey_label = f"#{jersey_number}"
        else:
            # Show compact "?" for unknown tracks
            jersey_label = f"{track_id}:?"
        
        # Calculate label size and position using smaller font
        label_size = cv2.getTextSize(jersey_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(
            vis_frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            color,
            -1
        )
        
        # Draw jersey number text using smaller, slightly bold font
        cv2.putText(
            vis_frame,
            jersey_label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
        # Draw detection regions if OCR results are available for known players
        if jersey_number != "Unknown" and isinstance(details, dict) and 'ocr_results' in details:
            ocr_results = details['ocr_results']
            if ocr_results:
                # Get transformation information
                original_width = details.get('original_width', x2 - x1)
                original_height = details.get('original_height', y2 - y1)
                crop_width = details.get('crop_width', original_width)
                crop_height = details.get('crop_height', original_height)
                final_width = details.get('final_width', crop_width)
                final_height = details.get('final_height', crop_height)
                crop_fraction = details.get('crop_fraction', 0.33)
                
                # Calculate the actual dimensions of the jersey area in the track
                track_width = x2 - x1
                track_height = y2 - y1
                jersey_area_height = int(track_height * crop_fraction)
                
                # Draw bounding boxes for each OCR detection
                for bbox_ocr, text, conf in ocr_results:
                    if isinstance(bbox_ocr, list) and len(bbox_ocr) == 4:
                        # EasyOCR bbox format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                        ocr_points = np.array(bbox_ocr, dtype=np.float32)
                        ocr_x1 = int(np.min(ocr_points[:, 0]))
                        ocr_y1 = int(np.min(ocr_points[:, 1]))
                        ocr_x2 = int(np.max(ocr_points[:, 0]))
                        ocr_y2 = int(np.max(ocr_points[:, 1]))
                        
                        # Correct coordinate transformation accounting for all processing steps:
                        # 1. Original track -> Cropped jersey area (crop_fraction)
                        # 2. Cropped area -> Resized for processing (128x64)
                        # 3. OCR results are in the final processed image coordinates
                        
                        if final_width > 0 and final_height > 0 and crop_width > 0 and crop_height > 0:
                            # Step 1: Scale from final processed coordinates back to cropped coordinates
                            # The final processed image maintains aspect ratio, so we need to account for padding
                            crop_to_final_scale_x = crop_width / final_width
                            crop_to_final_scale_y = crop_height / final_height
                            
                            # Scale OCR coordinates back to cropped image space
                            crop_x1 = ocr_x1 * crop_to_final_scale_x
                            crop_y1 = ocr_y1 * crop_to_final_scale_y
                            crop_x2 = ocr_x2 * crop_to_final_scale_x
                            crop_y2 = ocr_y2 * crop_to_final_scale_y
                            
                            # Step 2: Scale from cropped coordinates to jersey area coordinates
                            # The cropped area is the top crop_fraction of the track
                            jersey_scale_x = track_width / crop_width
                            jersey_scale_y = jersey_area_height / crop_height
                            
                            # Scale and position within the jersey area of the track
                            final_x1 = x1 + int(crop_x1 * jersey_scale_x)
                            final_y1 = y1 + int(crop_y1 * jersey_scale_y)
                            final_x2 = x1 + int(crop_x2 * jersey_scale_x)
                            final_y2 = y1 + int(crop_y2 * jersey_scale_y)
                            
                            # Ensure minimum box size
                            if final_x2 - final_x1 < 3:
                                final_x2 = final_x1 + 3
                            if final_y2 - final_y1 < 3:
                                final_y2 = final_y1 + 3
                            
                            # Clamp to jersey area bounds
                            final_x1 = max(x1, min(final_x1, x2 - 3))
                            final_y1 = max(y1, min(final_y1, y1 + jersey_area_height - 3))
                            final_x2 = max(final_x1 + 3, min(final_x2, x2))
                            final_y2 = max(final_y1 + 3, min(final_y2, y1 + jersey_area_height))
                            
                            # Color based on confidence: Red (low) -> Orange -> Green (high)
                            if conf >= 0.7:
                                bbox_color = (0, 255, 0)  # Green for high confidence
                            elif conf >= 0.4:
                                bbox_color = (0, 165, 255)  # Orange for medium confidence
                            else:
                                bbox_color = (0, 0, 255)  # Red for low confidence
                            
                            # Draw the OCR bounding box
                            cv2.rectangle(vis_frame, (final_x1, final_y1), (final_x2, final_y2), bbox_color, 2)
                            
                            # Draw detected text and confidence
                            if text and text.strip():
                                text_label = f"'{text}' ({conf:.2f})"
                                # Draw text above the box
                                cv2.putText(
                                    vis_frame,
                                    text_label,
                                    (final_x1, final_y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    bbox_color,
                                    1
                                )
        
        # Draw trajectory history if available
        if track_histories and track_id in track_histories:
            history = track_histories[track_id]
            if len(history) > 1:
                # Draw trajectory line
                points = np.array(history, dtype=np.int32)
                cv2.polylines(vis_frame, [points], False, color, 2)
                
                # Draw trajectory points
                for point in history[-10:]:  # Show last 10 points
                    cv2.circle(vis_frame, tuple(point), 3, color, -1)
    
    return vis_frame


def draw_tracks(frame: np.ndarray, tracks: List[Any], track_histories: Optional[Dict[int, List[Tuple[int, int]]]] = None) -> np.ndarray:
    """Draw tracking bounding boxes, IDs, and history trails.
    
    Args:
        frame: Input frame to draw on
        tracks: List of track objects
        track_histories: Optional dictionary of track histories
        
    Returns:
        Frame with tracking overlays
    """
    if not tracks:
        return frame
    
    vis_frame = frame.copy()
    
    for track in tracks:
        # Get track properties
        track_id = getattr(track, 'track_id', None)
        if track_id is None:
            continue
            
        # Get bounding box
        bbox = None
        if hasattr(track, 'to_ltrb'):
            bbox = track.to_ltrb()
        elif hasattr(track, 'bbox'):
            bbox = track.bbox
        
        if bbox is None or len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Generate unique color for each track ID (better for tracking visualization)
        color = _get_track_color(track_id)
        
        # Draw bounding box with unique track color
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)  # Thicker line for better visibility
        
        # Draw track ID with background for better visibility
        track_label = f"ID:{track_id}"
        label_size = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Draw label background
        cv2.rectangle(
            vis_frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 4, y1),
            color,
            -1
        )
        
        # Draw track ID text
        cv2.putText(
            vis_frame, 
            track_label, 
            (x1 + 2, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255),  # White text for contrast
            2
        )
        
        # Draw class information if available
        class_info = ""
        if hasattr(track, 'class_name') and track.class_name:
            class_info = track.class_name
        elif hasattr(track, 'det_class') and track.det_class is not None:
            class_info = str(track.det_class)
        elif hasattr(track, 'class_id') and track.class_id is not None:
            class_info = f"C{track.class_id}"
            
        if class_info:
            cv2.putText(
                vis_frame, 
                class_info, 
                (x1, y2 + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
        
        # Draw track history if available (tracks are at foot level)
        if track_histories and track_id in track_histories:
            history = track_histories[track_id]
            if len(history) > 1:
                # Draw trajectory lines with decreasing opacity for older points
                for i in range(1, len(history)):
                    pt1 = history[i-1]
                    pt2 = history[i]
                    
                    # Calculate line thickness and opacity based on recency
                    alpha = min(1.0, (i / len(history)) + 0.3)  # Newer points more visible
                    thickness = max(1, int(3 * alpha))
                    
                    # Draw trajectory line (foot-level tracking)
                    cv2.line(vis_frame, pt1, pt2, color, thickness)
                
                # Draw small circles at trajectory points (representing foot positions)
                for i, point in enumerate(history[-10:]):  # Only last 10 points
                    alpha = (i + 1) / min(10, len(history))
                    radius = max(2, int(4 * alpha))  # Slightly larger for foot positions
                    cv2.circle(vis_frame, point, radius, color, -1)
                    
                    # Add small ground indicator for most recent position
                    if i == len(history[-10:]) - 1:  # Most recent point
                        # Draw small line below the point to indicate ground level
                        cv2.line(vis_frame, 
                                (point[0] - 5, point[1]), 
                                (point[0] + 5, point[1]), 
                                color, 2)
    
    return vis_frame


def draw_player_ids(frame: np.ndarray, tracks: List[Any], player_id_results: Dict[int, Tuple[str, Any]]) -> np.ndarray:
    """Draw player ID information on tracked players with both single-frame and historical results.
    
    Args:
        frame: Input frame to draw on
        tracks: List of track objects
        player_id_results: Dictionary mapping track_id to (jersey_number, details)
        
    Returns:
        Frame with player ID overlays
    """
    if not tracks or not player_id_results:
        return frame
    
    vis_frame = frame.copy()
    
    for track in tracks:
        track_id = getattr(track, 'track_id', None)
        if track_id is None or track_id not in player_id_results:
            continue
            
        # Get bounding box
        bbox = None
        if hasattr(track, 'to_ltrb'):
            bbox = track.to_ltrb()
        elif hasattr(track, 'bbox'):
            bbox = track.bbox
            
        if bbox is None or len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get player ID result
        jersey_number, details = player_id_results[track_id]
        
        if jersey_number and jersey_number != "Unknown":
            # Extract tracking details if available
            single_frame_result = None
            tracking_history = []
            best_tracked = None
            
            if details and isinstance(details, dict):
                single_frame_result = details.get('single_frame')
                tracking_history = details.get('tracking_history', [])
                best_tracked = details.get('best_tracked')
            
            # Main jersey number display (using primary result)
            main_color = VISUALIZATION_COLORS['PLAYER_ID_BOX']
            
            # Distinguish colors for single-frame vs tracked
            if best_tracked and best_tracked.get('probability', 0) > 0.5:
                # High confidence tracked result - use green
                main_color = (0, 200, 0)  # Green for reliable tracked result
            else:
                # Single-frame or low confidence - use orange  
                main_color = (0, 165, 255)  # Orange for single-frame
            
            # Draw background for main jersey number
            main_label = f"#{jersey_number}"
            main_label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            
            cv2.rectangle(
                vis_frame,
                (x2 + 5, y1),
                (x2 + 15 + main_label_size[0], y1 + main_label_size[1] + 10),
                main_color,
                -1
            )
            
            cv2.putText(
                vis_frame,
                main_label,
                (x2 + 10, y1 + main_label_size[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),  # White text for better contrast
                2
            )
            
            # Draw tracking history (top 3 probabilities) below main label
            if tracking_history:
                y_offset = y1 + main_label_size[1] + 20
                
                for i, (hist_number, probability, count) in enumerate(tracking_history[:3]):
                    # Color coding: highest probability in bright green, others in muted colors
                    if i == 0:  # Highest probability
                        hist_color = (0, 255, 0) if probability > 0.5 else (0, 200, 200)
                    elif i == 1:  # Second highest
                        hist_color = (0, 150, 255)  # Orange
                    else:  # Third highest
                        hist_color = (100, 100, 255)  # Light red
                    
                    # Create probability label
                    prob_label = f"#{hist_number}: {probability:.1%}"
                    if count > 1:
                        prob_label += f" ({count})"
                    
                    prob_label_size = cv2.getTextSize(prob_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    
                    # Draw probability background
                    cv2.rectangle(
                        vis_frame,
                        (x2 + 5, y_offset),
                        (x2 + 10 + prob_label_size[0], y_offset + prob_label_size[1] + 6),
                        hist_color,
                        -1
                    )
                    
                    # Draw probability text
                    cv2.putText(
                        vis_frame,
                        prob_label,
                        (x2 + 7, y_offset + prob_label_size[1] + 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),  # White text
                        1
                    )
                    
                    y_offset += prob_label_size[1] + 10
            
            # Draw single-frame confidence if different from tracked result
            if single_frame_result and single_frame_result.get('jersey_number') != jersey_number:
                sf_number = single_frame_result.get('jersey_number', 'Unknown')
                sf_confidence = single_frame_result.get('confidence', 0.0)
                
                if sf_number != "Unknown":
                    sf_label = f"SF: #{sf_number} ({sf_confidence:.2f})"
                    sf_label_size = cv2.getTextSize(sf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    
                    # Use different position (top left)
                    cv2.rectangle(
                        vis_frame,
                        (x1 - sf_label_size[0] - 10, y1 - sf_label_size[1] - 8),
                        (x1 - 2, y1 - 2),
                        (128, 128, 128),  # Gray for single-frame
                        -1
                    )
                    
                    cv2.putText(
                        vis_frame,
                        sf_label,
                        (x1 - sf_label_size[0] - 7, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )
            
            # Draw digit detection boxes if available (legacy support)
            if details and isinstance(details, list):
                for item in details:
                    if isinstance(item, tuple) and len(item) == 2:
                        digit_bbox, digit_class = item
                        if len(digit_bbox) == 4:
                            dx1, dy1, dx2, dy2 = map(int, digit_bbox)
                            # Offset by track bbox position
                            cv2.rectangle(
                                vis_frame,
                                (x1 + dx1, y1 + dy1),
                                (x1 + dx2, y1 + dy2),
                                (0, 255, 0),  # Green for digit boxes
                                1
                            )
    
    return vis_frame


def draw_field_segmentation(frame: np.ndarray, segmentation_results: List[Any]) -> np.ndarray:
    """Draw field segmentation masks and boundaries.
    
    Args:
        frame: Input frame to draw on
        segmentation_results: List of segmentation result objects
        
    Returns:
        Frame with field segmentation overlays
    """
    if not segmentation_results:
        return frame
    
    vis_frame = frame.copy()
    
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
                mask = mask_data.cpu().numpy()
            else:
                mask = mask_data.numpy() if hasattr(mask_data, 'numpy') else mask_data
            
            # Draw field segmentation overlay
            vis_frame = _draw_segmentation_masks(vis_frame, mask)
            
        except Exception as e:
            print(f"[VISUALIZATION] Error drawing field segmentation: {e}")
    
    return vis_frame


def _draw_segmentation_masks(frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Draw segmentation masks with color overlays.
    
    Args:
        frame: Input frame
        masks: Mask array with shape (N, H, W)
        
    Returns:
        Frame with mask overlays
    """
    if masks.size == 0:
        return frame
    
    overlay = frame.copy()
    color_mask = np.zeros_like(frame)
    
    # Define colors for different field regions - made much brighter for better visibility
    color_dict = {
        0: (0, 255, 255),    # Central Field: bright cyan (BGR)
        1: (255, 0, 255)     # Endzone: bright magenta (BGR)
    }
    
    name_dict = {
        0: "Central Field", 
        1: "Endzone"
    }
    
    n_classes = min(masks.shape[0], 2)  # Only process class 0 and 1
    frame_h, frame_w = frame.shape[:2]
    
    for cls in range(n_classes):
        # Resize each class mask to match frame size
        class_mask = masks[cls]
        
        # Skip if mask is empty
        if np.sum(class_mask) == 0:
            continue
            
        class_mask_resized = cv2.resize(
            class_mask.astype(np.uint8), 
            (frame_w, frame_h), 
            interpolation=cv2.INTER_NEAREST
        )
        mask_bool = class_mask_resized > 0.5
        
        # Skip if no pixels in mask
        if not np.any(mask_bool):
            continue
        
        color = color_dict.get(cls, (200, 200, 200))
        color_mask[mask_bool] = color
        
        # Draw border for the mask
        contours, _ = cv2.findContours(class_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            border_color = tuple(int(c * 0.7) for c in color)
            cv2.drawContours(overlay, contours, -1, border_color, 2)
            
            # Find center of mask for label
            ys, xs = np.where(mask_bool)
            if len(xs) > 0 and len(ys) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                label = name_dict.get(cls, str(cls))
                
                cv2.putText(
                    overlay, 
                    label, 
                    (cx, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    border_color, 
                    2, 
                    cv2.LINE_AA
                )
    
    # Blend with higher alpha for maximum visibility
    cv2.addWeighted(color_mask, 0.4, overlay, 0.6, 0, overlay)
    return overlay


def calculate_field_contour(unified_mask: np.ndarray, 
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
    
    # Import here to avoid circular imports
    from ..config.settings import get_setting
    
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
            print(f"[VISUALIZATION] Contour area {contour_area} below threshold {min_contour_area}")
            return None
        
        # Simplify contour using Douglas-Peucker algorithm
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = simplify_epsilon * perimeter
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        print(f"[VISUALIZATION] Original contour points: {len(largest_contour)}, simplified: {len(simplified_contour)}")
        
        return simplified_contour
        
    except Exception as e:
        print(f"[VISUALIZATION] Error calculating field contour: {e}")
        return None


def draw_field_contour(frame: np.ndarray, contour: np.ndarray,
                      contour_color: Tuple[int, int, int] = None,
                      point_color: Tuple[int, int, int] = None,
                      line_thickness: int = None,
                      point_radius: int = None,
                      draw_points: bool = None) -> np.ndarray:
    """Draw field contour lines and points on the frame.
    
    Args:
        frame: Input frame to draw on
        contour: Contour points as numpy array of shape (N, 1, 2)
        contour_color: BGR color for contour lines
        point_color: BGR color for contour points
        line_thickness: Thickness of contour lines
        point_radius: Radius of contour points
        draw_points: Whether to draw individual contour points
        
    Returns:
        Frame with contour overlay
    """
    if contour is None or len(contour) == 0:
        return frame
    
    # Import here to avoid circular imports
    from ..config.settings import get_setting
    
    # Use config values if parameters not provided
    if contour_color is None:
        # Get color from config as list and convert to tuple
        color_list = get_setting("models.segmentation.contour.line_color", [255, 255, 0])
        contour_color = tuple(color_list) if isinstance(color_list, list) else (255, 255, 0)
    if point_color is None:
        color_list = get_setting("models.segmentation.contour.point_color", [0, 255, 255])
        point_color = tuple(color_list) if isinstance(color_list, list) else (0, 255, 255)
    if line_thickness is None:
        line_thickness = get_setting("models.segmentation.contour.line_thickness", 3)
    if point_radius is None:
        point_radius = get_setting("models.segmentation.contour.point_radius", 5)
    if draw_points is None:
        draw_points = get_setting("models.segmentation.contour.draw_points", True)
    
    result = frame.copy()
    
    try:
        # Draw contour lines
        cv2.drawContours(result, [contour], -1, contour_color, line_thickness)
        
        # Draw contour points if enabled
        if draw_points:
            for point in contour:
                center = tuple(point[0])  # point is shape (1, 2), so point[0] is (x, y)
                cv2.circle(result, center, point_radius, point_color, -1)
                # Add small white border for better visibility
                cv2.circle(result, center, point_radius + 1, (255, 255, 255), 1)
        
        print(f"[VISUALIZATION] Drew contour with {len(contour)} points")
        
    except Exception as e:
        print(f"[VISUALIZATION] Error drawing field contour: {e}")
    
    return result


def fit_field_lines_ransac(contour: np.ndarray, 
                          frame: np.ndarray,
                          num_lines: int = 4,
                          distance_threshold: float = 10.0,
                          min_samples: int = 2,
                          max_trials: int = 1000) -> Optional[Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
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
        Tuple of (fitted_lines, outlier_points) where:
        - fitted_lines: List of (start_point, end_point) tuples for each fitted line
        - outlier_points: List of outlier point arrays for each segment
        Returns (None, None) if fitting fails
    """
    if contour is None or len(contour) < num_lines * min_samples:
        return None, None, None, None
    
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
            
            print(f"[VISUALIZATION] Interpolated contour: {len(contour_format.reshape(-1, 2))} -> {len(points)} points")
        
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
                print(f"[VISUALIZATION] Edge filtering: {original_count} -> {len(points)} points ({len(edge_filtered_points)} filtered)")
            else:
                print(f"[VISUALIZATION] Warning: Edge filtering removed all points!")
        
        # Sequential RANSAC: Find lines one by one, removing inliers each time
        remaining_points = points.copy()
        fitted_lines = []
        all_outliers = []
        all_inliers = []
        
        for iteration in range(num_lines):
            if len(remaining_points) < min_samples:
                print(f"[VISUALIZATION] Iteration {iteration}: Not enough remaining points ({len(remaining_points)} < {min_samples})")
                break
            
            print(f"[VISUALIZATION] Iteration {iteration}: Fitting line to {len(remaining_points)} remaining points")
            
            # Fit line to remaining points using RANSAC
            result = _fit_line_ransac_with_outliers(remaining_points, distance_threshold, min_samples, max_trials)
            
            if result is not None:
                line_points, outliers, inliers = result
                fitted_lines.append(line_points)
                all_inliers.append(inliers)
                
                # Remove inliers from remaining points for next iteration
                remaining_points = outliers
                
                print(f"[VISUALIZATION] Line {iteration}: Found line with {len(inliers)} inliers, {len(outliers)} points remaining")
            else:
                print(f"[VISUALIZATION] Iteration {iteration}: Failed to fit line, stopping sequential RANSAC")
                break
        
        # All remaining points after all iterations are final outliers
        if len(remaining_points) > 0:
            all_outliers.append(remaining_points)
            print(f"[VISUALIZATION] Final outliers: {len(remaining_points)} points")
        else:
            all_outliers.append(np.array([]).reshape(0, 2))
        
        print(f"[VISUALIZATION] Sequential RANSAC: Successfully fitted {len(fitted_lines)} out of {num_lines} lines")
        
        # Classify the fitted lines
        if fitted_lines:
            classified_lines = _classify_field_lines(fitted_lines, frame.shape)
        else:
            classified_lines = {}
        
        # Filter out None entries from fitted_lines
        valid_lines = [line for line in fitted_lines if line is not None]
        return (valid_lines, all_outliers, all_inliers, edge_filtered_points, classified_lines) if valid_lines else (None, None, None, edge_filtered_points, {})
        
    except Exception as e:
        print(f"[VISUALIZATION] Error in RANSAC line fitting: {e}")
        return None, None, None, np.array([]).reshape(0, 2), {}


def _classify_field_lines(fitted_lines: List[np.ndarray], frame_shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    """Classify fitted lines into field components (sidelines, endzone lines, etc.).
    
    Args:
        fitted_lines: List of line segments as [start_point, end_point] arrays
        frame_shape: Shape of the frame (height, width, channels)
        
    Returns:
        Dictionary mapping line types to line coordinates
    """
    if not fitted_lines:
        return {}
    
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
        # Lines within 15 degrees of horizontal are considered horizontal (more strict)
        is_horizontal = abs(angle - 0) < 15 or abs(angle - 180) < 15
        
        line_info.append({
            'index': i,
            'line': line,
            'angle': angle,
            'avg_y': avg_y,
            'avg_x': avg_x,
            'is_horizontal': is_horizontal,
            'start': start_point,
            'end': end_point
        })
        
        print(f"[VISUALIZATION] Line {i}: angle={angle:.1f}°, avg_y={avg_y:.1f}, avg_x={avg_x:.1f}, horizontal={is_horizontal}")
    
    # Separate horizontal and non-horizontal lines
    horizontal_lines = [info for info in line_info if info['is_horizontal']]
    vertical_lines = [info for info in line_info if not info['is_horizontal']]
    
    # Sort horizontal lines by Y position (top to bottom)
    horizontal_lines.sort(key=lambda x: x['avg_y'])
    
    # Sort vertical lines by X position (left to right)
    vertical_lines.sort(key=lambda x: x['avg_x'])
    
    # Classify horizontal lines based on Y position
    # Ultimate frisbee field has up to 4 horizontal lines: far endzone back, far endzone front, near endzone front, near endzone back
    # Use quarter-based regions but ensure each classification is only assigned once
    
    quarter_height = frame_height / 4
    used_classifications = set()
    
    for i, line_info_item in enumerate(horizontal_lines):
        y_pos = line_info_item['avg_y']
        
        # Determine preferred classification based on position
        if y_pos < quarter_height:
            preferred = 'far_endzone_back'
        elif y_pos < quarter_height * 2:
            preferred = 'far_endzone_front'
        elif y_pos < quarter_height * 3:
            preferred = 'near_endzone_front'
        else:
            preferred = 'near_endzone_back'
        
        # If preferred classification is already used, find an alternative
        if preferred not in used_classifications:
            classified[preferred] = line_info_item['line']
            used_classifications.add(preferred)
            print(f"[VISUALIZATION] Classified line {line_info_item['index']} as {preferred}")
        else:
            # Find the next available horizontal line classification
            alternatives = ['far_endzone_back', 'far_endzone_front', 'near_endzone_front', 'near_endzone_back']
            assigned = False
            for alt in alternatives:
                if alt not in used_classifications:
                    classified[alt] = line_info_item['line']
                    used_classifications.add(alt)
                    print(f"[VISUALIZATION] Classified line {line_info_item['index']} as {alt} (alternative)")
                    assigned = True
                    break
            
            if not assigned:
                # If all standard classifications are used, create additional ones
                classified[f'horizontal_line_{i}'] = line_info_item['line']
                print(f"[VISUALIZATION] Classified line {line_info_item['index']} as horizontal_line_{i} (overflow)")
    
    # Classify vertical lines (sidelines)
    # Ultimate frisbee field typically has 2 sidelines, but may detect more segments
    for i, line_info_item in enumerate(vertical_lines):
        if i == 0:
            classified['left_sideline'] = line_info_item['line']
            print(f"[VISUALIZATION] Classified line {line_info_item['index']} as left sideline")
        elif i == 1:
            classified['right_sideline'] = line_info_item['line']
            print(f"[VISUALIZATION] Classified line {line_info_item['index']} as right sideline")
        else:
            # Additional vertical lines - classify as additional sideline segments
            classified[f'sideline_segment_{i}'] = line_info_item['line']
            print(f"[VISUALIZATION] Classified line {line_info_item['index']} as sideline segment {i}")
    
    print(f"[VISUALIZATION] Line classification complete: {list(classified.keys())}")
    return classified


def _segment_contour_points(points: np.ndarray, num_segments: int) -> List[np.ndarray]:
    """Divide contour points into segments for line fitting.
    
    Args:
        points: Array of 2D points (N, 2)
        num_segments: Number of segments to create
        
    Returns:
        List of point arrays, one for each segment
    """
    n_points = len(points)
    segment_size = n_points // num_segments
    segments = []
    
    for i in range(num_segments):
        start_idx = i * segment_size
        if i == num_segments - 1:  # Last segment gets remaining points
            end_idx = n_points
        else:
            end_idx = (i + 1) * segment_size
        
        segment_points = points[start_idx:end_idx]
        segments.append(segment_points)
    
    return segments


def _fit_line_ransac(points: np.ndarray, 
                    distance_threshold: float,
                    min_samples: int,
                    max_trials: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Fit a line to points using RANSAC algorithm.
    
    Args:
        points: Array of 2D points (N, 2)
        distance_threshold: Maximum distance for inliers
        min_samples: Minimum samples to fit line
        max_trials: Maximum RANSAC iterations
        
    Returns:
        Tuple of (start_point, end_point) for the fitted line, or None if fitting fails
    """
    if len(points) < min_samples:
        return None
    
    best_line = None
    best_inliers = 0
    
    for trial in range(max_trials):
        # Randomly sample minimum points needed to fit a line
        sample_indices = np.random.choice(len(points), min_samples, replace=False)
        sample_points = points[sample_indices]
        
        # Fit line to sample points
        if min_samples == 2:
            # Simple case: line through two points
            p1, p2 = sample_points[0], sample_points[1]
            
            # Skip if points are too close (degenerate case)
            if np.linalg.norm(p2 - p1) < 1e-6:
                continue
                
            # Calculate distances from all points to this line
            line_vec = p2 - p1
            line_length = np.linalg.norm(line_vec)
            line_unit = line_vec / line_length
            
        else:
            # Fit line using least squares for more points
            try:
                # Use SVD to fit line
                centroid = np.mean(sample_points, axis=0)
                centered_points = sample_points - centroid
                
                # SVD: line direction is first principal component
                _, _, vh = np.linalg.svd(centered_points.T)
                line_unit = vh[0]  # First row is the principal direction
                p1 = centroid
                
            except np.linalg.LinAlgError:
                continue
        
        # Count inliers: points within distance_threshold of the line
        inlier_count = 0
        for point in points:
            if min_samples == 2:
                # Distance from point to line defined by p1, p2
                to_point = point - p1
                projection_length = np.dot(to_point, line_unit)
                closest_point_on_line = p1 + projection_length * line_unit
                distance = np.linalg.norm(point - closest_point_on_line)
            else:
                # Distance from point to line through centroid with direction line_unit
                to_point = point - p1
                projection_length = np.dot(to_point, line_unit)
                closest_point_on_line = p1 + projection_length * line_unit
                distance = np.linalg.norm(point - closest_point_on_line)
            
            if distance <= distance_threshold:
                inlier_count += 1
        
        # Update best line if this one has more inliers
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            
            # Find extent of inliers along the line
            inlier_projections = []
            for point in points:
                if min_samples == 2:
                    to_point = point - p1
                    projection_length = np.dot(to_point, line_unit)
                    closest_point_on_line = p1 + projection_length * line_unit
                    distance = np.linalg.norm(point - closest_point_on_line)
                else:
                    to_point = point - p1
                    projection_length = np.dot(to_point, line_unit)
                    closest_point_on_line = p1 + projection_length * line_unit
                    distance = np.linalg.norm(point - closest_point_on_line)
                
                if distance <= distance_threshold:
                    if min_samples == 2:
                        inlier_projections.append(np.dot(to_point, line_unit))
                    else:
                        inlier_projections.append(projection_length)
            
            if inlier_projections:
                min_proj = min(inlier_projections)
                max_proj = max(inlier_projections)
                
                if min_samples == 2:
                    start_point = p1 + min_proj * line_unit
                    end_point = p1 + max_proj * line_unit
                else:
                    start_point = p1 + min_proj * line_unit
                    end_point = p1 + max_proj * line_unit
                
                best_line = (start_point, end_point)
    
    # Return best line if it has enough inliers
    min_inliers = max(min_samples, len(points) // 4)  # At least 25% of points should be inliers
    if best_inliers >= min_inliers:
        return best_line
    
    return None


def _fit_line_ransac_with_outliers(points: np.ndarray, 
                                  distance_threshold: float,
                                  min_samples: int,
                                  max_trials: int) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]]:
    """Fit a line to points using RANSAC algorithm and return outliers and inliers.
    
    Args:
        points: Array of 2D points (N, 2)
        distance_threshold: Maximum distance for inliers
        min_samples: Minimum samples to fit line
        max_trials: Maximum RANSAC iterations
        
    Returns:
        Tuple of ((start_point, end_point), outlier_points, inlier_points) for the fitted line, outliers, and inliers,
        or None if fitting fails
    """
    if len(points) < min_samples:
        return None
    
    best_line = None
    best_inliers = 0
    best_inlier_mask = None
    
    for trial in range(max_trials):
        # Randomly sample minimum points needed to fit a line
        sample_indices = np.random.choice(len(points), min_samples, replace=False)
        sample_points = points[sample_indices]
        
        # Fit line to sample points
        if min_samples == 2:
            # Simple case: line through two points
            p1, p2 = sample_points[0], sample_points[1]
            
            # Skip if points are too close (degenerate case)
            if np.linalg.norm(p2 - p1) < 1e-6:
                continue
                
            # Calculate distances from all points to this line
            line_vec = p2 - p1
            line_length = np.linalg.norm(line_vec)
            line_unit = line_vec / line_length
            
        else:
            # Fit line using least squares for more points
            try:
                # Use SVD to fit line
                centroid = np.mean(sample_points, axis=0)
                centered_points = sample_points - centroid
                
                # SVD: line direction is first principal component
                _, _, vh = np.linalg.svd(centered_points.T)
                line_unit = vh[0]  # First row is the principal direction
                p1 = centroid
                
            except np.linalg.LinAlgError:
                continue
        
        # Count inliers and create mask: points within distance_threshold of the line
        inlier_mask = np.zeros(len(points), dtype=bool)
        inlier_count = 0
        
        for idx, point in enumerate(points):
            if min_samples == 2:
                # Distance from point to line defined by p1, p2
                to_point = point - p1
                projection_length = np.dot(to_point, line_unit)
                closest_point_on_line = p1 + projection_length * line_unit
                distance = np.linalg.norm(point - closest_point_on_line)
            else:
                # Distance from point to line through centroid with direction line_unit
                to_point = point - p1
                projection_length = np.dot(to_point, line_unit)
                closest_point_on_line = p1 + projection_length * line_unit
                distance = np.linalg.norm(point - closest_point_on_line)
            
            if distance <= distance_threshold:
                inlier_mask[idx] = True
                inlier_count += 1
        
        # Update best line if this one has more inliers
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_inlier_mask = inlier_mask.copy()
            
            # Find extent of inliers along the line
            inlier_projections = []
            for idx, point in enumerate(points):
                if inlier_mask[idx]:
                    if min_samples == 2:
                        to_point = point - p1
                        projection_length = np.dot(to_point, line_unit)
                    else:
                        to_point = point - p1
                        projection_length = np.dot(to_point, line_unit)
                    
                    inlier_projections.append(projection_length)
            
            if inlier_projections:
                min_proj = min(inlier_projections)
                max_proj = max(inlier_projections)
                
                if min_samples == 2:
                    start_point = p1 + min_proj * line_unit
                    end_point = p1 + max_proj * line_unit
                else:
                    start_point = p1 + min_proj * line_unit
                    end_point = p1 + max_proj * line_unit
                
                best_line = (start_point, end_point)
    
    # Return best line, outliers, and inliers if it has enough inliers
    min_inliers = max(min_samples, len(points) // 4)  # At least 25% of points should be inliers
    if best_inliers >= min_inliers and best_inlier_mask is not None:
        outliers = points[~best_inlier_mask]  # Points that are NOT inliers
        inliers = points[best_inlier_mask]    # Points that ARE inliers
        return best_line, outliers, inliers
    
    return None


def draw_field_lines_ransac(frame: np.ndarray, 
                           fitted_lines: List[Tuple[np.ndarray, np.ndarray]],
                           line_color: Tuple[int, int, int] = None,
                           line_thickness: int = None) -> np.ndarray:
    """Draw RANSAC-fitted field lines on the frame.
    
    Args:
        frame: Input frame to draw on
        fitted_lines: List of (start_point, end_point) tuples for each line
        line_color: BGR color for the lines
        line_thickness: Thickness of the lines
        
    Returns:
        Frame with fitted lines drawn
    """
    if not fitted_lines:
        return frame
    
    # Import here to avoid circular imports
    from ..config.settings import get_setting
    
    # Use config values if parameters not provided
    if line_color is None:
        color_list = get_setting("models.segmentation.contour.line_color", [0, 255, 0])
        line_color = tuple(color_list) if isinstance(color_list, list) else (0, 255, 0)
    if line_thickness is None:
        line_thickness = get_setting("models.segmentation.contour.line_thickness", 3)
    
    result = frame.copy()
    
    try:
        for i, (start_point, end_point) in enumerate(fitted_lines):
            # Convert points to integers for drawing
            start = tuple(start_point.astype(np.int32))
            end = tuple(end_point.astype(np.int32))
            
            # Draw the line
            cv2.line(result, start, end, line_color, line_thickness)
            
            # Draw endpoint markers
            cv2.circle(result, start, line_thickness + 2, line_color, -1)
            cv2.circle(result, end, line_thickness + 2, line_color, -1)
        
        print(f"[VISUALIZATION] Drew {len(fitted_lines)} RANSAC-fitted field lines")
        
    except Exception as e:
        print(f"[VISUALIZATION] Error drawing RANSAC lines: {e}")
    
    return result


def draw_field_lines_ransac_with_outliers(frame: np.ndarray, 
                                         fitted_lines: List[Tuple[np.ndarray, np.ndarray]],
                                         outlier_points: List[np.ndarray],
                                         line_color: Tuple[int, int, int] = None,
                                         line_thickness: int = None,
                                         outlier_color: Tuple[int, int, int] = None,
                                         outlier_radius: int = None,
                                         show_outliers: bool = None) -> np.ndarray:
    """Draw RANSAC-fitted field lines and outlier points on the frame.
    
    Args:
        frame: Input frame to draw on
        fitted_lines: List of (start_point, end_point) tuples for each line
        outlier_points: List of outlier point arrays for each segment
        line_color: BGR color for the lines
        line_thickness: Thickness of the lines
        outlier_color: BGR color for outlier points
        outlier_radius: Radius for outlier points
        show_outliers: Whether to draw outlier points
        
    Returns:
        Frame with fitted lines and outliers drawn
    """
    if not fitted_lines and not outlier_points:
        return frame
    
    try:
        result = frame.copy()
        
        # Draw RANSAC lines first
        if fitted_lines:
            result = draw_field_lines_ransac(result, fitted_lines, line_color, line_thickness)
        
        # Draw outlier points if enabled
        if outlier_points and (show_outliers if show_outliers is not None else get_setting("models.segmentation.contour.ransac.show_outliers", True)):
            # Get default values from config
            outlier_color = outlier_color or tuple(get_setting("models.segmentation.contour.ransac.outlier_color", [0, 0, 255]))
            outlier_radius = outlier_radius if outlier_radius is not None else get_setting("models.segmentation.contour.ransac.outlier_radius", 3)
            
            outlier_count = 0
            for segment_outliers in outlier_points:
                if segment_outliers is not None and len(segment_outliers) > 0:
                    for point in segment_outliers:
                        center = (int(point[0]), int(point[1]))
                        cv2.circle(result, center, outlier_radius, outlier_color, -1)
                        outlier_count += 1
            
            if outlier_count > 0:
                print(f"[VISUALIZATION] Drew {outlier_count} RANSAC outlier points")
        
    except Exception as e:
        print(f"[VISUALIZATION] Error drawing RANSAC lines and outliers: {e}")
    
    return result


def _apply_morphological_smoothing(mask: np.ndarray, 
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
    
    # Import here to avoid circular imports
    from ..config.settings import get_setting
    
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
        print(f"[VISUALIZATION] Error in morphological smoothing: {e}")
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
        print(f"[VISUALIZATION] Error in hole filling: {e}")
        return mask


def create_unified_field_mask(segmentation_results: List[Any], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Create a unified mask combining all segmentation classes into one binary mask.
    
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
            print(f"[VISUALIZATION] Error creating unified mask: {e}")
    
    # Apply morphological operations to smooth the mask
    if np.any(unified_mask):
        unified_mask = _apply_morphological_smoothing(unified_mask)
            
    return unified_mask if np.any(unified_mask) else None


def draw_unified_field_mask(frame: np.ndarray, unified_mask: np.ndarray, 
                           color: Tuple[int, int, int] = (0, 255, 0), 
                           alpha: float = 0.4,
                           draw_contour: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Draw a unified field mask with a single color overlay and optional contour.
    
    Args:
        frame: Input frame to draw on
        unified_mask: Binary mask (H, W) where 1 indicates field area
        color: BGR color tuple for the overlay
        alpha: Transparency for overlay (0.0 = transparent, 1.0 = opaque)
        draw_contour: Whether to calculate and draw simplified contours
        
    Returns:
        Tuple of (frame with unified mask overlay and optional contour, classified_lines dictionary)
    """
    if unified_mask is None or not np.any(unified_mask):
        return frame, {}
    
    # Import here to avoid circular imports
    from ..config.settings import get_setting
    
    overlay = frame.copy()
    classified_lines = {}  # Initialize empty classified lines dictionary
    
    # Create colored overlay where mask is present
    overlay[unified_mask == 1] = color
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    # Draw basic border contours for mask visibility
    contours, _ = cv2.findContours(unified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        border_color = tuple(int(c * 0.7) for c in color)
        cv2.drawContours(result, contours, -1, border_color, 2)
    
    # Draw simplified contour or RANSAC lines if requested
    if draw_contour:
        # Check if RANSAC line fitting is enabled
        ransac_enabled = get_setting("models.segmentation.contour.ransac.enabled", False)
        
        if ransac_enabled:
            # Use RANSAC line fitting approach
            simplified_contour = calculate_field_contour(unified_mask)
            if simplified_contour is not None:
                # Fit lines using RANSAC
                num_lines = get_setting("models.segmentation.contour.ransac.num_lines", 4)
                distance_threshold = get_setting("models.segmentation.contour.ransac.distance_threshold", 10.0)
                min_samples = get_setting("models.segmentation.contour.ransac.min_samples", 2)
                max_trials = get_setting("models.segmentation.contour.ransac.max_trials", 1000)
                
                fitted_lines, outlier_points, inlier_points, edge_filtered_points, classified_lines = fit_field_lines_ransac(
                    simplified_contour, 
                    frame,
                    num_lines=num_lines,
                    distance_threshold=distance_threshold,
                    min_samples=min_samples,
                    max_trials=max_trials
                )
                
                if fitted_lines:
                    # Draw RANSAC-fitted lines and outliers
                    line_color_list = get_setting("models.segmentation.contour.ransac.line_color", [0, 255, 0])
                    line_color = tuple(line_color_list) if isinstance(line_color_list, list) else (0, 255, 0)
                    result = draw_field_lines_ransac_with_outliers(
                        result, 
                        fitted_lines, 
                        outlier_points,
                        line_color=line_color
                    )
                    
                    # Draw edge-filtered points if enabled
                    edge_filtering_enabled = get_setting("models.segmentation.contour.ransac.edge_filtering.enabled", False)
                    show_edge_points = get_setting("models.segmentation.contour.ransac.edge_filtering.show_edge_points", True)
                    if edge_filtering_enabled and show_edge_points and len(edge_filtered_points) > 0:
                        edge_color_list = get_setting("models.segmentation.contour.ransac.edge_filtering.edge_point_color", [0, 0, 255])
                        edge_color = tuple(edge_color_list) if isinstance(edge_color_list, list) else (0, 0, 255)
                        edge_radius = get_setting("models.segmentation.contour.ransac.edge_filtering.edge_point_radius", 2)
                        
                        for point in edge_filtered_points:
                            x, y = int(point[0]), int(point[1])
                            if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                                cv2.circle(result, (x, y), edge_radius, edge_color, -1)
                        
                        print(f"[VISUALIZATION] Drew {len(edge_filtered_points)} edge-filtered points")
                    
                    # Draw inlier points if enabled
                    show_inliers = get_setting("models.segmentation.contour.ransac.inliers.show_inliers", True)
                    if show_inliers and inlier_points and len(inlier_points) > 0:
                        inlier_color_list = get_setting("models.segmentation.contour.ransac.inliers.inlier_color", [0, 255, 0])
                        inlier_color = tuple(inlier_color_list) if isinstance(inlier_color_list, list) else (0, 255, 0)
                        inlier_radius = get_setting("models.segmentation.contour.ransac.inliers.inlier_radius", 2)
                        
                        # Draw inliers for each segment
                        total_inliers = 0
                        for inlier_segment in inlier_points:
                            if len(inlier_segment) > 0:
                                for point in inlier_segment:
                                    x, y = int(point[0]), int(point[1])
                                    if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                                        cv2.circle(result, (x, y), inlier_radius, inlier_color, -1)
                                        total_inliers += 1
                        
                        print(f"[VISUALIZATION] Drew {total_inliers} inlier points")
                else:
                    print("[VISUALIZATION] RANSAC line fitting failed, falling back to contour")
                    result = draw_field_contour(result, simplified_contour)
            else:
                print("[VISUALIZATION] No contour found for RANSAC line fitting")
        else:
            # Use traditional contour approach
            simplified_contour = calculate_field_contour(unified_mask)
            if simplified_contour is not None:
                result = draw_field_contour(result, simplified_contour)
    
    return result, classified_lines


def draw_classified_field_lines(frame: np.ndarray, classified_lines: Dict[str, np.ndarray],
                               transformation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """Draw classified field lines with different colors for each type.
    
    Args:
        frame: Frame to draw on
        classified_lines: Dictionary mapping line types to line coordinates
        transformation_matrix: Optional homography matrix to transform lines to warped view
        
    Returns:
        Frame with classified lines drawn
    """
    if not classified_lines:
        return frame
        
    result = frame.copy()
    
    # Define colors for different line types
    line_colors = {
        'left_sideline': (255, 0, 0),      # Blue
        'right_sideline': (255, 0, 255),   # Magenta  
        'far_endzone_back': (0, 255, 255), # Yellow
        'far_endzone_front': (0, 165, 255), # Orange
        'near_endzone_front': (0, 255, 0), # Green
        'near_endzone_back': (255, 255, 0)  # Cyan
    }
    
    line_thickness = 3
    
    for line_type, line_coords in classified_lines.items():
        if line_coords is None or len(line_coords) != 2:
            continue
            
        color = line_colors.get(line_type, (255, 255, 255))  # Default white
        
        start_point = line_coords[0].copy()
        end_point = line_coords[1].copy()
        
        # Transform points if transformation matrix is provided
        if transformation_matrix is not None:
            # Convert to homogeneous coordinates
            start_homo = np.array([start_point[0], start_point[1], 1.0])
            end_homo = np.array([end_point[0], end_point[1], 1.0])
            
            # Apply transformation
            start_transformed = transformation_matrix @ start_homo
            end_transformed = transformation_matrix @ end_homo
            
            # Convert back to 2D coordinates
            if start_transformed[2] != 0:
                start_point = start_transformed[:2] / start_transformed[2]
            if end_transformed[2] != 0:
                end_point = end_transformed[:2] / end_transformed[2]
        
        # Draw the line
        start_int = (int(start_point[0]), int(start_point[1]))
        end_int = (int(end_point[0]), int(end_point[1]))
        
        # Check if points are within frame bounds
        h, w = frame.shape[:2]
        if (0 <= start_int[0] < w and 0 <= start_int[1] < h and
            0 <= end_int[0] < w and 0 <= end_int[1] < h):
            cv2.line(result, start_int, end_int, color, line_thickness)
            
            # Add text label with offset to avoid covering the line
            mid_point = ((start_int[0] + end_int[0]) // 2, (start_int[1] + end_int[1]) // 2)
            
            # Calculate offset perpendicular to the line
            line_vec = (end_int[0] - start_int[0], end_int[1] - start_int[1])
            line_length = max(1, (line_vec[0]**2 + line_vec[1]**2)**0.5)  # Avoid division by zero
            
            # Perpendicular vector (rotate 90 degrees)
            perp_vec = (-line_vec[1], line_vec[0])
            
            # Normalize and scale for offset distance
            offset_distance = 25  # pixels
            offset_x = int((perp_vec[0] / line_length) * offset_distance)
            offset_y = int((perp_vec[1] / line_length) * offset_distance)
            
            # Apply offset to text position
            text_pos = (mid_point[0] + offset_x, mid_point[1] + offset_y)
            
            # Ensure text position is within frame bounds
            text_pos = (max(10, min(w - 10, text_pos[0])), max(20, min(h - 10, text_pos[1])))
            
            # Draw text with larger font size and better visibility
            font_scale = 0.8  # Increased from 0.5
            font_thickness = 2  # Increased from 1
            cv2.putText(result, line_type.replace('_', ' ').title(), text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
            
            print(f"[VISUALIZATION] Drew {line_type} line from {start_int} to {end_int}")
    
    return result


def _get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a consistent, distinct color for a track ID.
    
    Args:
        track_id: Unique track identifier
        
    Returns:
        BGR color tuple
    """
    # Predefined distinct colors for better visual separation
    distinct_colors = [
        (0, 255, 255),    # Cyan
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Yellow
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 165, 255),    # Orange
        (128, 0, 128),    # Purple
        (255, 20, 147),   # Deep Pink
        (0, 255, 127),    # Spring Green
        (255, 69, 0),     # Red Orange
        (30, 144, 255),   # Dodger Blue
        (255, 215, 0),    # Gold
        (50, 205, 50),    # Lime Green
        (255, 105, 180),  # Hot Pink
        (0, 206, 209),    # Dark Turquoise
        (255, 140, 0),    # Dark Orange
    ]
    
    # Use modulo to cycle through distinct colors
    color_index = track_id % len(distinct_colors)
    base_color = distinct_colors[color_index]
    
    # Add slight variation based on track_id for uniqueness when cycling
    if track_id >= len(distinct_colors):
        variation = (track_id // len(distinct_colors)) * 30
        r, g, b = base_color
        # Apply variation while keeping colors bright
        r = max(50, min(255, r + (variation % 100)))
        g = max(50, min(255, g + ((variation * 2) % 100)))
        b = max(50, min(255, b + ((variation * 3) % 100)))
        return (int(b), int(g), int(r))  # Return as BGR
    
    return base_color
