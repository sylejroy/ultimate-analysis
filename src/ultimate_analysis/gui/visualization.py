"""Visualization functions for Ultimate Analysis GUI.

This module provides functions for drawing detection boxes, tracking overlays,
player IDs, and field segmentation on video frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from ..config.settings import get_setting
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
    """Draw tracking bounding boxes, IDs, player jersey numbers, and history trails.
    
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
        
        # Draw bounding box with unique track color
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
        
        # Create track label with player ID if available
        track_label = f"ID:{track_id}"
        if player_ids and track_id in player_ids:
            jersey_number, _ = player_ids[track_id]
            if jersey_number != "Unknown":
                track_label = f"ID:{track_id} #{jersey_number}"
        
        # Calculate label size and position
        label_size = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Draw label background
        cv2.rectangle(
            vis_frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            color,
            -1
        )
        
        # Draw track ID text
        cv2.putText(
            vis_frame,
            track_label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
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
    """Draw player ID information on tracked players.
    
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
            # Draw jersey number
            color = VISUALIZATION_COLORS['PLAYER_ID_BOX']
            
            # Draw background for jersey number
            label = f"#{jersey_number}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            
            cv2.rectangle(
                vis_frame,
                (x2 + 5, y1),
                (x2 + 15 + label_size[0], y1 + label_size[1] + 10),
                color,
                -1
            )
            
            cv2.putText(
                vis_frame,
                label,
                (x2 + 10, y1 + label_size[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),  # Black text
                2
            )
            
            # Draw digit detection boxes if available (for YOLO method)
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
    
    # Define colors for different field regions
    color_dict = {
        0: (200, 217, 37),   # Central Field: teal (BGR)
        1: (114, 38, 249)    # Endzone: pink (BGR)
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
    
    # Blend with low alpha for subtle overlay
    cv2.addWeighted(color_mask, 0.12, overlay, 0.88, 0, overlay)
    return overlay


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


def apply_all_visualizations(
    frame: np.ndarray,
    detections: List[Dict[str, Any]] = None,
    tracks: List[Any] = None,
    track_histories: Dict[int, List[Tuple[int, int]]] = None,
    player_ids: List[Dict[str, Any]] = None,
    field_result: Any = None,
    show_detections: bool = True,
    show_tracking: bool = True,
    show_player_ids: bool = True,
    show_field_segmentation: bool = True
) -> np.ndarray:
    """Apply all enabled visualizations to a frame.
    
    Args:
        frame: Input frame
        detections: Detection results from run_inference
        tracks: Tracking results from run_tracking
        track_histories: Track history data for drawing trails
        player_ids: Player ID results from run_player_id
        field_result: Field segmentation results from run_field_segmentation
        show_detections: Whether to show detection boxes
        show_tracking: Whether to show tracking overlays
        show_player_ids: Whether to show player IDs
        show_field_segmentation: Whether to show field segmentation
        
    Returns:
        Frame with all enabled visualizations applied
    """
    vis_frame = frame.copy()
    
    # Apply visualizations in order (background to foreground)
    
    # Field segmentation (background layer)
    if show_field_segmentation and field_result is not None:
        # Handle both single result and list of results
        field_results = [field_result] if not isinstance(field_result, list) else field_result
        vis_frame = draw_field_segmentation(vis_frame, field_results)
    
    # Detections (if tracking is not enabled, show raw detections)
    if show_detections and detections and not show_tracking:
        vis_frame = draw_detections(vis_frame, detections)
    
    # Tracking (includes bounding boxes, replaces detections)
    if show_tracking and tracks:
        vis_frame = draw_tracks(vis_frame, tracks, track_histories)
    
    # Player IDs (foreground layer)
    if show_player_ids and player_ids and tracks:
        # Convert player_ids list to dictionary format expected by draw_player_ids
        player_id_dict = {}
        if isinstance(player_ids, list):
            # Assume player_ids is a list of dicts with track_id and jersey_number
            for pid in player_ids:
                if isinstance(pid, dict) and 'track_id' in pid:
                    track_id = pid['track_id']
                    jersey_number = pid.get('jersey_number', 'Unknown')
                    details = pid.get('details', None)
                    player_id_dict[track_id] = (jersey_number, details)
        elif isinstance(player_ids, dict):
            player_id_dict = player_ids
        
        if player_id_dict:
            vis_frame = draw_player_ids(vis_frame, tracks, player_id_dict)
    
    return vis_frame
