"""Visualization functions for Ultimate Analysis GUI.

This module provides functions for drawing detection boxes, tracking overlays,
player IDs, and field segmentation on video frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from ..constants import VISUALIZATION_COLORS


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
                           draw_contour: bool = True) -> np.ndarray:
    """Draw a unified field mask with a single color overlay and optional contour.
    
    Args:
        frame: Input frame to draw on
        unified_mask: Binary mask (H, W) where 1 indicates field area
        color: BGR color tuple for the overlay
        alpha: Transparency for overlay (0.0 = transparent, 1.0 = opaque)
        draw_contour: Whether to calculate and draw simplified contours
        
    Returns:
        Frame with unified mask overlay and optional contour
    """
    if unified_mask is None or not np.any(unified_mask):
        return frame
    
    overlay = frame.copy()
    
    # Create colored overlay where mask is present
    overlay[unified_mask == 1] = color
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    # Draw basic border contours for mask visibility
    contours, _ = cv2.findContours(unified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        border_color = tuple(int(c * 0.7) for c in color)
        cv2.drawContours(result, contours, -1, border_color, 2)
    
    # Draw simplified contour if requested
    if draw_contour:
        simplified_contour = calculate_field_contour(unified_mask)
        if simplified_contour is not None:
            result = draw_field_contour(result, simplified_contour)
    
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
