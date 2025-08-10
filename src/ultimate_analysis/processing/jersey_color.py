"""Jersey color estimation module for Ultimate Analysis.

This module analyzes player bounding boxes to estimate jersey colors using HSV color analysis
and histogram-based dominant hue detection.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import colorsys

# Cache for jersey colors by track ID
_jersey_color_cache = defaultdict(list)
_max_history_length = 10


def estimate_jersey_color(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                         track_id: int) -> Tuple[int, int, int]:
    """Estimate the dominant jersey color for a player.
    
    Args:
        frame: Input video frame
        bbox: Player bounding box (x1, y1, x2, y2)
        track_id: Track ID for color history tracking
        
    Returns:
        RGB color tuple representing the estimated jersey color
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Validate bounding box
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(x1+1, min(x2, w))
    y2 = max(y1+1, min(y2, h))
    
    # Extract jersey region from bounding box
    # Focus on the torso area where jersey is most visible
    jersey_crop = _extract_jersey_region(frame, x1, y1, x2, y2)
    
    if jersey_crop.size == 0:
        return (128, 128, 128)  # Gray fallback
    
    # Analyze dominant hue in the jersey region
    dominant_hue = _analyze_dominant_hue(jersey_crop)
    
    if dominant_hue is None:
        return (128, 128, 128)  # Gray fallback
    
    # Add to color history for temporal consistency
    _jersey_color_cache[track_id].append(dominant_hue)
    if len(_jersey_color_cache[track_id]) > _max_history_length:
        _jersey_color_cache[track_id].pop(0)
    
    # Get temporally-consistent color
    consistent_hue = _get_consistent_color(track_id)
    
    # Convert HSV to RGB for display
    rgb_color = _hue_to_rgb(consistent_hue)
    
    return rgb_color


def _extract_jersey_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Extract the jersey region from a player bounding box.
    
    Args:
        frame: Input video frame
        x1, y1, x2, y2: Bounding box coordinates
        
    Returns:
        Cropped region focusing on the jersey area
    """
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Focus on the upper-middle portion of the bounding box where jersey is most visible
    # Skip the head (top 20%) and focus on torso (20%-70% of height)
    jersey_y1 = y1 + int(bbox_height * 0.2)  # Skip head area
    jersey_y2 = y1 + int(bbox_height * 0.7)  # Focus on torso, before legs
    
    # Use middle 60% of width to avoid arm/background edges
    jersey_x1 = x1 + int(bbox_width * 0.2)
    jersey_x2 = x2 - int(bbox_width * 0.2)
    
    # Ensure valid crop region
    jersey_y1 = max(y1, min(jersey_y1, y2-1))
    jersey_y2 = max(jersey_y1+1, min(jersey_y2, y2))
    jersey_x1 = max(x1, min(jersey_x1, x2-1))
    jersey_x2 = max(jersey_x1+1, min(jersey_x2, x2))
    
    # Extract the jersey region
    jersey_crop = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
    
    return jersey_crop


def _analyze_dominant_hue(jersey_crop: np.ndarray) -> Optional[float]:
    """Analyze the dominant hue in the jersey region.
    
    Args:
        jersey_crop: Cropped jersey region
        
    Returns:
        Dominant hue value (0-360 degrees), or None if analysis fails
    """
    if jersey_crop.size == 0:
        return None
    
    # Convert to HSV for better color analysis
    hsv_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
    
    # Create mask to filter out low saturation and very dark/bright pixels
    # This helps exclude white/black/gray areas and focuses on actual colors
    h, s, v = cv2.split(hsv_crop)
    
    # Filter criteria:
    # - Saturation > 30 (avoid gray/white areas)
    # - Value > 40 and < 220 (avoid very dark and very bright areas)
    color_mask = (s > 30) & (v > 40) & (v < 220)
    
    if np.sum(color_mask) < 10:  # Not enough colored pixels
        return None
    
    # Get hue values for colored pixels
    colored_hues = h[color_mask]
    
    if len(colored_hues) == 0:
        return None
    
    # Create hue histogram (OpenCV hue is 0-179, convert to 0-360)
    hue_hist = np.histogram(colored_hues, bins=18, range=(0, 180))[0]  # 20-degree bins
    
    # Find the bin with maximum count
    max_bin = np.argmax(hue_hist)
    
    # Convert bin to hue value (center of bin, scaled to 0-360)
    dominant_hue = (max_bin * 10 + 5) * 2  # Center of 20-degree bin, scaled to 360
    
    return float(dominant_hue)


def _get_consistent_color(track_id: int) -> float:
    """Get temporally consistent color for a track using color history.
    
    Args:
        track_id: Track ID
        
    Returns:
        Consistent hue value based on recent history
    """
    if track_id not in _jersey_color_cache or len(_jersey_color_cache[track_id]) == 0:
        return 0.0  # Red fallback
    
    hue_history = _jersey_color_cache[track_id]
    
    if len(hue_history) == 1:
        return hue_history[0]
    
    # Group similar hues (within 30 degrees) and find most common group
    hue_groups = defaultdict(list)
    
    for hue in hue_history:
        # Find existing group within 30 degrees, accounting for hue wraparound
        assigned = False
        for group_center in hue_groups.keys():
            hue_diff = min(abs(hue - group_center), 360 - abs(hue - group_center))
            if hue_diff <= 30:
                hue_groups[group_center].append(hue)
                assigned = True
                break
        
        if not assigned:
            hue_groups[hue] = [hue]
    
    # Find the group with most entries
    largest_group = max(hue_groups.values(), key=len)
    
    # Return average hue of the largest group
    return sum(largest_group) / len(largest_group)


def _hue_to_rgb(hue: float) -> Tuple[int, int, int]:
    """Convert hue value to RGB color.
    
    Args:
        hue: Hue value in degrees (0-360)
        
    Returns:
        RGB color tuple (0-255 range)
    """
    # Normalize hue to 0-1 range
    hue_normalized = hue / 360.0
    
    # Use high saturation and medium value for vibrant jersey colors
    saturation = 0.8
    value = 0.9
    
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue_normalized, saturation, value)
    
    # Scale to 0-255 range
    return (int(r * 255), int(g * 255), int(b * 255))


def get_jersey_colors_for_tracks(frame: np.ndarray, tracks: List[Any]) -> Dict[int, Tuple[int, int, int]]:
    """Get jersey colors for all tracks in a frame.
    
    Args:
        frame: Input video frame
        tracks: List of track objects
        
    Returns:
        Dictionary mapping track_id -> RGB color tuple
    """
    jersey_colors = {}
    
    for track in tracks:
        # Skip disc tracks - only process players
        if hasattr(track, 'class_id') and track.class_id == 0:  # 0 = disc
            continue
        elif hasattr(track, 'class_name') and track.class_name.lower() == 'disc':
            continue
        
        # Get track ID and bounding box
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
        
        # Estimate jersey color
        jersey_color = estimate_jersey_color(frame, bbox, track_id)
        jersey_colors[track_id] = jersey_color
    
    return jersey_colors


def reset_jersey_color_cache():
    """Reset the jersey color cache (useful for new videos)."""
    global _jersey_color_cache
    _jersey_color_cache.clear()


def get_jersey_color_name(rgb_color: Tuple[int, int, int]) -> str:
    """Get a human-readable name for a jersey color.
    
    Args:
        rgb_color: RGB color tuple
        
    Returns:
        Color name string
    """
    r, g, b = rgb_color
    
    # Convert to HSV for better color classification
    hsv = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    hue = hsv[0] * 360
    saturation = hsv[1]
    value = hsv[2]
    
    # Low saturation = gray/white/black
    if saturation < 0.3:
        if value > 0.7:
            return "White"
        elif value < 0.3:
            return "Black"
        else:
            return "Gray"
    
    # Color classification by hue ranges
    if hue < 15 or hue >= 345:
        return "Red"
    elif hue < 45:
        return "Orange"
    elif hue < 75:
        return "Yellow"
    elif hue < 150:
        return "Green"
    elif hue < 210:
        return "Blue"
    elif hue < 270:
        return "Purple"
    elif hue < 315:
        return "Pink"
    else:
        return "Red"
