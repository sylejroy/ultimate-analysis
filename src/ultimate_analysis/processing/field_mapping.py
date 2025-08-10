"""Field coordinate mapping module for top-down visualization.

This module handles the conversion between image pixel coordinates and real field coordinates,
enabling top-down tactical views of player positions on the Ultimate Frisbee field.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import math

from ..config.settings import get_setting
from ..constants import FIELD_DIMENSIONS
from .field_projection import UnifiedField
from .field_types import FieldLine


@dataclass
class FieldCoordinate:
    """Represents a position on the Ultimate field in real-world coordinates."""
    x: float  # Distance along field length (0-100m)
    y: float  # Distance across field width (0-37m)
    confidence: float  # Confidence in mapping accuracy


@dataclass 
class PlayerPosition:
    """Represents a player's position on the field."""
    track_id: int
    jersey_number: Optional[str]
    field_coord: FieldCoordinate
    pixel_coord: Tuple[int, int]  # Original pixel position (feet center)
    jersey_color: Optional[Tuple[int, int, int]]  # RGB jersey color


def create_perspective_transform(field_lines: List[FieldLine], 
                                image_shape: Tuple[int, int],
                                coverage_analysis: Any = None) -> Optional[np.ndarray]:
    """Create perspective transformation matrix from field lines to top-down view.
    
    Args:
        field_lines: Detected field lines (sidelines and field lines)
        image_shape: Shape of the image (height, width)
        coverage_analysis: Optional FieldCoverage analysis for better estimation
        
    Returns:
        3x3 transformation matrix, or None if insufficient lines detected
        
    Strategy:
        1. Identify key field boundaries (left/right sidelines, near/far field lines)
        2. Use coverage analysis to extrapolate missing boundaries if available
        3. Map these to real field dimensions (100m x 37m)
        4. Use cv2.getPerspectiveTransform() to create mapping
    """
    if len(field_lines) < 3:  # Need at least 3 lines for reasonable mapping
        print(f"[FIELD_MAP] Insufficient field lines ({len(field_lines)}) for perspective transform")
        return None
    
    # Categorize detected lines by position and orientation
    left_sideline = None
    right_sideline = None
    near_field_line = None
    far_field_line = None
    
    h, w = image_shape
    
    for line in field_lines:
        # Calculate line center for position analysis
        x1, y1 = line.point1
        x2, y2 = line.point2
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate line angle to distinguish sidelines from field lines
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            angle = 90.0
        else:
            angle = abs(math.atan2(dy, dx) * 180 / math.pi)
            if angle > 90:
                angle = 180 - angle
        
        print(f"[FIELD_MAP] Analyzing line {line.line_type}: center=({center_x:.1f},{center_y:.1f}), angle={angle:.1f}°")
        
        # Classify based on actual line type if available
        if line.line_type == 'left_sideline':
            left_sideline = line
        elif line.line_type == 'right_sideline':
            right_sideline = line
        elif line.line_type == 'close_field':
            near_field_line = line
        elif line.line_type == 'far_field':
            far_field_line = line
        else:
            # Fallback classification based on position and angle
            if angle > 30:  # Diagonal lines are likely sidelines (run along field length)
                if center_x < w / 2:  # Left side of image
                    if left_sideline is None:
                        left_sideline = line
                        print(f"[FIELD_MAP] Assigned left sideline based on position")
                else:  # Right side of image
                    if right_sideline is None:
                        right_sideline = line
                        print(f"[FIELD_MAP] Assigned right sideline based on position")
            else:  # More horizontal lines are field boundaries (run across field width)
                if center_y > h / 2:  # Bottom of image (closer to camera)
                    if near_field_line is None:
                        near_field_line = line
                        print(f"[FIELD_MAP] Assigned near field line based on position")
                else:  # Top of image (farther from camera)
                    if far_field_line is None:
                        far_field_line = line
                        print(f"[FIELD_MAP] Assigned far field line based on position")
    
    # Find intersections to define field corners
    field_corners = []
    
    # Try to find field corner intersections
    if left_sideline and near_field_line:
        # Bottom-left corner
        corner = _find_line_intersection(left_sideline, near_field_line)
        if corner:
            field_corners.append(corner)
    
    if right_sideline and near_field_line:
        # Bottom-right corner
        corner = _find_line_intersection(right_sideline, near_field_line)
        if corner:
            field_corners.append(corner)
            
    if right_sideline and far_field_line:
        # Top-right corner
        corner = _find_line_intersection(right_sideline, far_field_line)
        if corner:
            field_corners.append(corner)
            
    if left_sideline and far_field_line:
        # Top-left corner  
        corner = _find_line_intersection(left_sideline, far_field_line)
        if corner:
            field_corners.append(corner)
    
    # If we don't have enough corners, use coverage analysis to estimate them
    if len(field_corners) < 4 and coverage_analysis:
        print("[FIELD_MAP] Using coverage analysis to estimate missing corners")
        field_corners = coverage_analysis.estimated_full_field
        if len(field_corners) >= 4:
            print(f"[FIELD_MAP] Coverage analysis provided {len(field_corners)} corners")
    
    # Fallback estimation if still insufficient corners
    if len(field_corners) < 4:
        field_corners = _estimate_field_corners(field_lines, image_shape)
    
    if len(field_corners) < 4:
        print(f"[FIELD_MAP] Cannot determine field corners from {len(field_lines)} lines")
        return None
    
    # Sort corners to ensure consistent ordering: bottom-left, bottom-right, top-right, top-left
    field_corners = _sort_field_corners(field_corners, image_shape)
    
    # Define target field dimensions in our coordinate system (in meters)
    # Ultimate field: 100m length x 37m width
    # Since drone views down the LENGTH of the field:
    # - Horizontal lines in image = field width boundaries (37m apart)
    # - Diagonal lines in image = field length boundaries (100m apart)
    field_length_m = 100.0  # Length of field (extends away from camera)
    field_width_m = 37.0    # Width of field (horizontal in image)
    
    # Define target corners in field coordinate system
    # Using padding for better visualization
    padding = 5.0  # 5m padding around field
    target_corners = np.float32([
        [padding, padding],                                     # Bottom-left (near camera, left side)
        [field_width_m + padding, padding],                     # Bottom-right (near camera, right side)
        [field_width_m + padding, field_length_m + padding],    # Top-right (far from camera, right side)
        [padding, field_length_m + padding]                     # Top-left (far from camera, left side)
    ])
    
    # Source corners from image
    source_corners = np.float32(field_corners)
    
    print(f"[FIELD_MAP] Creating perspective transform from corners:")
    for i, (src, tgt) in enumerate(zip(source_corners, target_corners)):
        print(f"[FIELD_MAP]   Corner {i}: ({src[0]:.1f},{src[1]:.1f}) -> ({tgt[0]:.1f},{tgt[1]:.1f})")
    
    # Create perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(source_corners, target_corners)
    
    return transform_matrix


def _find_line_intersection(line1: FieldLine, line2: FieldLine) -> Optional[Tuple[float, float]]:
    """Find intersection point between two field lines.
    
    Args:
        line1: First field line
        line2: Second field line
        
    Returns:
        Intersection point (x, y) or None if lines don't intersect
    """
    x1, y1 = line1.point1
    x2, y2 = line1.point2
    x3, y3 = line2.point1
    x4, y4 = line2.point2
    
    # Calculate intersection using line equations
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:  # Lines are parallel
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    # Calculate intersection point
    intersect_x = x1 + t * (x2 - x1)
    intersect_y = y1 + t * (y2 - y1)
    
    return (intersect_x, intersect_y)


def _estimate_field_corners(field_lines: List[FieldLine], 
                          image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
    """Estimate field corners when not all lines are detected.
    
    Args:
        field_lines: Available field lines
        image_shape: Shape of the image (height, width)
        
    Returns:
        List of estimated corner points
    """
    h, w = image_shape
    corners = []
    
    # Simple estimation: use image corners as fallback
    # This isn't perfect but provides a basic mapping
    corners = [
        (50, h - 50),      # Bottom-left (with small margin)
        (w - 50, h - 50),  # Bottom-right
        (w - 50, 50),      # Top-right
        (50, 50)           # Top-left
    ]
    
    print(f"[FIELD_MAP] Using estimated corners due to insufficient field lines")
    return corners


def _sort_field_corners(corners: List[Tuple[float, float]], 
                       image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
    """Sort corners into consistent order: bottom-left, bottom-right, top-right, top-left.
    
    Args:
        corners: List of corner points
        image_shape: Shape of the image (height, width)
        
    Returns:
        Sorted list of corner points
    """
    if len(corners) != 4:
        return corners
    
    h, w = image_shape
    
    # Sort by y coordinate first (bottom vs top)
    corners_sorted = sorted(corners, key=lambda p: p[1], reverse=True)
    
    # Get bottom two points (higher y values) and top two points (lower y values)
    bottom_points = sorted(corners_sorted[:2], key=lambda p: p[0])  # Sort by x
    top_points = sorted(corners_sorted[2:], key=lambda p: p[0])     # Sort by x
    
    # Return in order: bottom-left, bottom-right, top-right, top-left
    return [bottom_points[0], bottom_points[1], top_points[1], top_points[0]]


def map_players_to_field(tracks: List[Any], player_ids: Dict[int, Tuple[str, Any]], 
                        transform_matrix: np.ndarray,
                        jersey_colors: Dict[int, Tuple[int, int, int]] = None) -> List[PlayerPosition]:
    """Map player pixel positions to field coordinates.
    
    Args:
        tracks: List of player tracks
        player_ids: Dictionary mapping track_id -> (jersey_number, details)
        transform_matrix: Perspective transformation matrix
        
    Returns:
        List of PlayerPosition objects with field coordinates
    """
    player_positions = []
    
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
            
        # Get bounding box - use feet position (bottom center)
        bbox = None
        if hasattr(track, 'to_ltrb'):
            bbox = track.to_ltrb()
        elif hasattr(track, 'to_tlbr'):
            bbox = track.to_tlbr()
        elif hasattr(track, 'bbox'):
            bbox = track.bbox
        
        if bbox is None or len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = map(float, bbox)
        
        # Calculate feet position (bottom center of bounding box)
        feet_x = (x1 + x2) / 2
        feet_y = y2  # Bottom of bounding box
        
        # Transform to field coordinates
        pixel_point = np.array([[feet_x, feet_y]], dtype=np.float32)
        field_point = cv2.perspectiveTransform(pixel_point.reshape(1, 1, 2), transform_matrix)
        field_x, field_y = field_point[0, 0]
        
        # Get jersey number and color
        jersey_number = None
        jersey_color = None
        
        if track_id in player_ids:
            jersey_number = player_ids[track_id][0]
            if jersey_number == "Unknown":
                jersey_number = None
        
        if jersey_colors and track_id in jersey_colors:
            jersey_color = jersey_colors[track_id]
        
        # Create field coordinate with confidence based on jersey detection
        confidence = 0.8 if jersey_number else 0.5  # Higher confidence if player ID detected
        
        field_coord = FieldCoordinate(
            x=float(field_x),
            y=float(field_y), 
            confidence=confidence
        )
        
        player_position = PlayerPosition(
            track_id=track_id,
            jersey_number=jersey_number,
            field_coord=field_coord,
            pixel_coord=(int(feet_x), int(feet_y)),
            jersey_color=jersey_color
        )
        
        player_positions.append(player_position)
    
    print(f"[FIELD_MAP] Mapped {len(player_positions)} players to field coordinates")
    return player_positions


def create_top_down_field_view(player_positions: List[PlayerPosition], 
                              field_size: Tuple[int, int] = (300, 800),  # Increased size
                              coverage_analysis: Any = None) -> np.ndarray:
    """Create a top-down view of the Ultimate field with player positions.
    
    Args:
        player_positions: List of player positions in field coordinates
        field_size: Size of the output image (width, height) - swapped to match field orientation
        
    Returns:
        RGB image showing top-down field view with player positions
    """
    # Ultimate field dimensions: 100m length x 37m width
    # Since drone views down the length, in our top-down view:
    # - Width of image = field width (37m)
    # - Height of image = field length (100m)
    field_length_m = 100.0  # Length (extends away from camera in original view)
    field_width_m = 37.0    # Width (horizontal in original view)
    padding_m = 5.0  # 5m padding around field
    
    total_width_m = field_width_m + 2 * padding_m   # 47m total width
    total_length_m = field_length_m + 2 * padding_m # 110m total length
    
    img_width, img_height = field_size
    
    # Create field image
    field_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    field_img.fill(34)  # Dark green background
    
    # Calculate scaling factors
    scale_x = img_width / total_width_m   # Scale for field width
    scale_y = img_height / total_length_m # Scale for field length
    
    # Draw field boundaries
    field_left = int(padding_m * scale_x)
    field_right = int((field_width_m + padding_m) * scale_x)
    field_top = int(padding_m * scale_y)
    field_bottom = int((field_length_m + padding_m) * scale_y)
    
    # Field boundary (bright green)
    cv2.rectangle(field_img, (field_left, field_top), (field_right, field_bottom), (0, 150, 0), 2)
    
    # Draw field lines
    # End zone lines (goal lines) - these are horizontal lines in the original image
    # End zones are 25m deep
    goal_line_1 = int((padding_m + 25) * scale_y)  # 25m from near end
    goal_line_2 = int((padding_m + 75) * scale_y)  # 25m from far end
    
    cv2.line(field_img, (field_left, goal_line_1), (field_right, goal_line_1), (255, 255, 255), 1)
    cv2.line(field_img, (field_left, goal_line_2), (field_right, goal_line_2), (255, 255, 255), 1)
    
    # Center line (middle of the field length)
    center_y = int((padding_m + field_length_m / 2) * scale_y)
    cv2.line(field_img, (field_left, center_y), (field_right, center_y), (255, 255, 255), 1)
    
    # Draw players
    for player_pos in player_positions:
        # Convert field coordinates to image coordinates
        # Note: field_coord.x = field width position, field_coord.y = field length position
        img_x = int(player_pos.field_coord.x * scale_x)
        img_y = int(player_pos.field_coord.y * scale_y)
        
        # Skip if outside image bounds
        if img_x < 0 or img_x >= img_width or img_y < 0 or img_y >= img_height:
            continue
        
        # Player color based on estimated jersey color
        if player_pos.jersey_color:
            # Use the estimated jersey color (convert from RGB to BGR for OpenCV)
            r, g, b = player_pos.jersey_color
            player_color = (b, g, r)  # BGR format for OpenCV
        elif player_pos.jersey_number:
            # Fallback: Different colors for different jersey numbers
            jersey_hash = hash(player_pos.jersey_number) % 6
            colors = [
                (255, 0, 0),    # Red
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 128, 0)   # Orange
            ]
            player_color = colors[jersey_hash]
        else:
            player_color = (128, 128, 128)  # Gray for unknown players
        
        # Draw player circle - smaller size
        radius = max(2, int(4 * min(scale_x, scale_y)))  # Reduced from 8 to 4
        cv2.circle(field_img, (img_x, img_y), radius, player_color, -1)
        cv2.circle(field_img, (img_x, img_y), radius, (255, 255, 255), 1)  # White border
        
        # Draw jersey number - positioned above the player circle
        if player_pos.jersey_number:
            font_scale = 0.4  # Increased from 0.3 for better visibility
            text_size = cv2.getTextSize(player_pos.jersey_number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = img_x - text_size[0] // 2
            text_y = img_y - radius - 3  # Position above the circle
            
            # Draw text background for better visibility
            cv2.rectangle(field_img, 
                         (text_x - 2, text_y - text_size[1] - 1),
                         (text_x + text_size[0] + 2, text_y + 3),
                         (0, 0, 0), -1)  # Black background
            
            cv2.putText(field_img, player_pos.jersey_number, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        else:
            # Draw track ID for players without jersey numbers
            track_text = str(player_pos.track_id)
            font_scale = 0.3
            text_size = cv2.getTextSize(track_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = img_x - text_size[0] // 2
            text_y = img_y - radius - 3  # Position above the circle
            
            # Draw text background
            cv2.rectangle(field_img, 
                         (text_x - 1, text_y - text_size[1] - 1),
                         (text_x + text_size[0] + 1, text_y + 2),
                         (0, 0, 0), -1)  # Black background
            
            cv2.putText(field_img, track_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (128, 128, 128), 1)  # Gray text
    
    # Add field labels and coverage information
    cv2.putText(field_img, "Ultimate Field Top-Down View", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(field_img, f"Players: {len(player_positions)}", (10, img_height - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add coverage information if available
    if coverage_analysis:
        coverage_text = f"Coverage: {coverage_analysis.visible_width_ratio:.1%}w x {coverage_analysis.visible_length_ratio:.1%}l"
        cv2.putText(field_img, coverage_text, (10, img_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Add cut indicators
        cut_indicators = []
        if coverage_analysis.left_sideline_cut:
            cut_indicators.append("L")
        if coverage_analysis.right_sideline_cut:
            cut_indicators.append("R")
        if coverage_analysis.near_boundary_cut:
            cut_indicators.append("N")
        if coverage_analysis.far_boundary_cut:
            cut_indicators.append("F")
        
        if cut_indicators:
            cut_text = f"Cut: {','.join(cut_indicators)}"
            cv2.putText(field_img, cut_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    
    return field_img
