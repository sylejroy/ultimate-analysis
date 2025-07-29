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
        
        # Choose color based on class
        color = VISUALIZATION_COLORS['DETECTION_BOX']
        if 'disc' in class_name.lower():
            color = (114, 38, 249)  # Pink/purple for disc
        elif 'player' in class_name.lower():
            color = (200, 217, 37)  # Teal for player
        
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
        
        # Generate consistent color for track ID
        color = _get_track_color(track_id)
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID
        label = f"ID: {track_id}"
        cv2.putText(
            vis_frame, 
            label, 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            color, 
            2
        )
        
        # Draw class if available
        class_name = None
        if hasattr(track, 'det_class') and track.det_class is not None:
            class_name = f"Class: {track.det_class}"
        elif hasattr(track, 'class_id') and track.class_id is not None:
            class_name = f"Class: {track.class_id}"
            
        if class_name:
            cv2.putText(
                vis_frame, 
                class_name, 
                (x1, y2 + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                1
            )
        
        # Draw track history if available
        if track_histories and track_id in track_histories:
            history = track_histories[track_id]
            if len(history) > 1:
                # Draw lines connecting history points
                for i in range(1, len(history)):
                    pt1 = history[i-1]
                    pt2 = history[i]
                    cv2.line(vis_frame, pt1, pt2, color, 2)
    
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
    """Generate a consistent color for a track ID.
    
    Args:
        track_id: Unique track identifier
        
    Returns:
        BGR color tuple
    """
    # Use track ID to generate consistent color
    np.random.seed(track_id)
    color = tuple(int(x) for x in np.random.randint(0, 255, 3))
    return color


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
