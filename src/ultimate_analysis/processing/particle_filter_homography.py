"""Particle filter-based homography estimation for Ultimate Frisbee field detection.

This module implements a particle filter approach to estimate camera perspective
based on field segmentation results. It combines field masks, extracts field lines,
and uses geometric constraints to find the optimal homography transformation.
"""

import cv2
import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..config.settings import get_setting


class FieldLine:
    """Represents a detected field line with geometric properties."""
    
    def __init__(self, line_type: str, points: np.ndarray, angle: Optional[float] = None):
        """Initialize a field line.
        
        Args:
            line_type: Type of line ('sideline_left', 'sideline_right', 'endzone_back')
            points: Line points as (N, 2) array
            angle: Line angle in radians (optional)
        """
        self.line_type = line_type
        self.points = points
        self.angle = angle
        
    def get_line_coefficients(self) -> Tuple[float, float, float]:
        """Get line coefficients for line equation ax + by + c = 0.
        
        Returns:
            Tuple of (a, b, c) coefficients
        """
        if len(self.points) < 2:
            return 0.0, 0.0, 0.0
            
        # Use first and last points to define the line
        p1, p2 = self.points[0], self.points[-1]
        
        # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        a = p2[1] - p1[1]
        b = -(p2[0] - p1[0])
        c = (p2[0] - p1[0]) * p1[1] - (p2[1] - p1[1]) * p1[0]
        
        # Normalize coefficients
        norm = np.sqrt(a*a + b*b)
        if norm > 0:
            a, b, c = a/norm, b/norm, c/norm
            
        return a, b, c


class ParticleFilterHomography:
    """Particle filter for homography estimation based on field segmentation."""
    
    def __init__(self, num_particles: int = 500, field_length: float = 100.0, field_width: float = 37.0):
        """Initialize the particle filter.
        
        Args:
            num_particles: Number of particles to use
            field_length: Ultimate Frisbee field length in meters
            field_width: Ultimate Frisbee field width in meters
        """
        self.num_particles = num_particles
        self.field_length = field_length
        self.field_width = field_width
        
        # Particle state (homography parameters)
        self.particles = np.zeros((num_particles, 8))  # H00, H01, H02, H10, H11, H12, H20, H21
        self.weights = np.ones(num_particles) / num_particles
        
        # Field lines detected from segmentation
        self.detected_lines: List[FieldLine] = []
        
    def combine_field_masks(self, segmentation_results: List[Any]) -> np.ndarray:
        """Combine all field segmentation masks into a single unified mask.
        
        Args:
            segmentation_results: List of segmentation results from YOLO model
            
        Returns:
            Combined binary mask of the field area
        """
        if not segmentation_results:
            return np.array([])
            
        combined_mask = None
        
        for result in segmentation_results:
            if hasattr(result, 'masks') and result.masks is not None:
                # Get masks data
                masks_data = result.masks.data
                if hasattr(masks_data, 'cpu'):
                    masks_data = masks_data.cpu().numpy()
                elif hasattr(masks_data, 'numpy'):
                    masks_data = masks_data.numpy()
                
                # Combine all masks from this result
                for mask in masks_data:
                    if combined_mask is None:
                        combined_mask = mask.copy()
                    else:
                        # Union of masks
                        combined_mask = np.maximum(combined_mask, mask)
        
        if combined_mask is None:
            return np.array([])
            
        # Convert to binary mask
        return (combined_mask > 0.5).astype(np.uint8)
    
    def simplify_mask_and_find_contour(self, mask: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Simplify the field mask and find the main field contour.
        
        Args:
            mask: Binary field mask
            
        Returns:
            Tuple of (simplified_mask, contours)
        """
        if mask.size == 0:
            return np.array([]), []
            
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small gaps
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small noise
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask_opened, []
            
        # Find the largest contour (main field area)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to reduce points
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        return mask_opened, [approx_contour]
    
    def extract_field_lines_from_mask(self, mask: np.ndarray, frame_shape: Tuple[int, int]) -> List[FieldLine]:
        """Extract field lines from the outer boundary of the segmentation mask.
        
        Args:
            mask: Binary field segmentation mask
            frame_shape: (height, width) of the frame
            
        Returns:
            List of detected field lines along the field boundaries
        """
        if mask.size == 0:
            return []
            
        lines = []
        
        # Find the outer contour of the field
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("[PARTICLE_FILTER] No contours found in mask")
            return []
            
        # Get the largest contour (main field boundary)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to get key boundary points
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        boundary_points = approx_contour.reshape(-1, 2)
        
        print(f"[PARTICLE_FILTER] Processing field boundary with {len(boundary_points)} key points")
        
        # Extract field boundary lines using boundary-focused approach
        frame_height, frame_width = frame_shape
        
        # 1. Find horizontal endzone line (top boundary)
        endzone_line = self._extract_top_boundary_line(boundary_points, frame_shape)
        if len(endzone_line) >= 2:
            lines.append(FieldLine('endzone_back', endzone_line))
            print(f"[PARTICLE_FILTER] Found endzone boundary with {len(endzone_line)} points")
        
        # 2. Extract left and right boundary edges
        left_boundary, right_boundary = self._extract_boundary_sidelines(boundary_points, frame_shape)
        
        if len(left_boundary) >= 2:
            lines.append(FieldLine('sideline_left', left_boundary))
            
        if len(right_boundary) >= 2:
            lines.append(FieldLine('sideline_right', right_boundary))
        
        print(f"[PARTICLE_FILTER] Total boundary lines extracted: {len(lines)}")
        return lines
    
    def _extract_top_boundary_line(self, boundary_points: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Extract the top horizontal boundary line from field boundary points."""
        frame_height = frame_shape[0]
        
        # Focus on points in the upper portion of the image
        upper_threshold = frame_height * 0.4
        upper_points = boundary_points[boundary_points[:, 1] <= upper_threshold]
        
        if len(upper_points) < 2:
            return np.array([])
        
        # Find the topmost points that form a roughly horizontal line
        top_y = np.min(upper_points[:, 1])
        tolerance = frame_height * 0.05  # 5% tolerance
        
        # Get points near the top boundary
        top_boundary_points = upper_points[
            abs(upper_points[:, 1] - top_y) <= tolerance
        ]
        
        if len(top_boundary_points) >= 2:
            # Sort by x-coordinate for a proper horizontal line
            sorted_points = top_boundary_points[np.argsort(top_boundary_points[:, 0])]
            print(f"[PARTICLE_FILTER] Found top boundary with {len(sorted_points)} points")
            return sorted_points
        
        return np.array([])
    
    def _extract_boundary_sidelines(self, boundary_points: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract left and right boundary sidelines from field boundary points."""
        frame_height, frame_width = frame_shape
        
        # Split boundary points into left and right halves
        left_half = boundary_points[boundary_points[:, 0] <= frame_width / 2]
        right_half = boundary_points[boundary_points[:, 0] > frame_width / 2]
        
        print(f"[PARTICLE_FILTER] Boundary analysis: {len(left_half)} left points, {len(right_half)} right points")
        
        # Extract the leftmost boundary edge
        left_boundary = self._extract_side_boundary(left_half, 'left', frame_shape)
        
        # Extract the rightmost boundary edge
        right_boundary = self._extract_side_boundary(right_half, 'right', frame_shape)
        
        return left_boundary, right_boundary
    
    def _extract_side_boundary(self, side_points: np.ndarray, side: str, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Extract the boundary edge for one side of the field."""
        if len(side_points) < 2:
            print(f"[PARTICLE_FILTER] {side} boundary: insufficient points ({len(side_points)})")
            return np.array([])
        
        frame_height, frame_width = frame_shape
        
        # For each y-level, find the extreme point (leftmost for left, rightmost for right)
        y_levels = np.unique(side_points[:, 1])
        boundary_edge_points = []
        
        for y in y_levels:
            points_at_y = side_points[side_points[:, 1] == y]
            
            if side == 'left':
                # Find leftmost point at this y-level
                extreme_point = points_at_y[np.argmin(points_at_y[:, 0])]
            else:
                # Find rightmost point at this y-level
                extreme_point = points_at_y[np.argmax(points_at_y[:, 0])]
            
            boundary_edge_points.append(extreme_point)
        
        if len(boundary_edge_points) < 2:
            print(f"[PARTICLE_FILTER] {side} boundary: insufficient edge points")
            return np.array([])
        
        # Sort boundary points by y-coordinate (top to bottom)
        boundary_edge = np.array(boundary_edge_points)
        boundary_edge = boundary_edge[np.argsort(boundary_edge[:, 1])]
        
        # Filter to keep only points that form a reasonable boundary line
        filtered_boundary = self._filter_boundary_line(boundary_edge, side)
        
        print(f"[PARTICLE_FILTER] {side} boundary: extracted {len(filtered_boundary)} edge points")
        return filtered_boundary
    
    def _filter_boundary_line(self, boundary_points: np.ndarray, side: str) -> np.ndarray:
        """Filter boundary points to form a clean boundary line."""
        if len(boundary_points) < 3:
            return boundary_points
        
        # Remove outliers that deviate significantly from the boundary trend
        filtered_points = [boundary_points[0]]  # Always keep first point
        
        for i in range(1, len(boundary_points) - 1):
            prev_point = filtered_points[-1]
            curr_point = boundary_points[i]
            next_point = boundary_points[i + 1]
            
            # Check if current point maintains reasonable slope
            slope1 = (curr_point[0] - prev_point[0]) / max(1, curr_point[1] - prev_point[1])
            slope2 = (next_point[0] - curr_point[0]) / max(1, next_point[1] - curr_point[1])
            
            # Keep point if slopes are reasonably consistent
            if abs(slope1 - slope2) < 2.0:  # Slope difference threshold
                filtered_points.append(curr_point)
        
        filtered_points.append(boundary_points[-1])  # Always keep last point
        
        return np.array(filtered_points)
    
    def extract_field_lines(self, contour: np.ndarray, frame_shape: Tuple[int, int]) -> List[FieldLine]:
        """Extract field lines (sidelines and endzone) from the field contour.
        
        Args:
            contour: Main field contour points
            frame_shape: (height, width) of the frame
            
        Returns:
            List of detected field lines
        """
        if len(contour) < 4:
            return []
            
        lines = []
        
        # Reshape contour to (N, 2)
        contour_points = contour.reshape(-1, 2)
        
        # Find the topmost points (back endzone line - should be roughly horizontal)
        # Sort by y-coordinate (top of image has smaller y values)
        sorted_by_y = contour_points[np.argsort(contour_points[:, 1])]
        
        # Get points in the top portion of the contour
        top_portion_height = frame_shape[0] * 0.3  # Top 30% of frame
        top_points = sorted_by_y[sorted_by_y[:, 1] < top_portion_height]
        
        if len(top_points) >= 2:
            # Find the most horizontal line segment in the top portion
            endzone_points = self._find_horizontal_line_segment(top_points)
            if len(endzone_points) >= 2:
                lines.append(FieldLine('endzone_back', endzone_points))
        
        # Find sidelines (should appear as diagonal lines)
        left_points, right_points = self._extract_sidelines(contour_points, frame_shape)
        
        if len(left_points) >= 2:
            lines.append(FieldLine('sideline_left', left_points))
            
        if len(right_points) >= 2:
            lines.append(FieldLine('sideline_right', right_points))
        
        return lines
    
    def _find_horizontal_line_segment(self, points: np.ndarray) -> np.ndarray:
        """Find the most horizontal line segment from a set of points.
        
        Args:
            points: Array of points (N, 2)
            
        Returns:
            Points forming the most horizontal line
        """
        if len(points) < 2:
            return np.array([])
            
        # Sort points by x-coordinate
        sorted_points = points[np.argsort(points[:, 0])]
        
        # Find consecutive points that form a roughly horizontal line
        best_line = []
        current_line = [sorted_points[0]]
        
        for i in range(1, len(sorted_points)):
            prev_point = current_line[-1]
            curr_point = sorted_points[i]
            
            # Calculate angle with horizontal
            dx = curr_point[0] - prev_point[0]
            dy = curr_point[1] - prev_point[1]
            
            if dx != 0:
                angle = abs(np.arctan(dy / dx))
                # If angle is close to horizontal (< 20 degrees)
                if angle < np.pi / 9:  # 20 degrees
                    current_line.append(curr_point)
                else:
                    # Check if current line is better than best
                    if len(current_line) > len(best_line):
                        best_line = current_line[:]
                    current_line = [curr_point]
            else:
                current_line.append(curr_point)
        
        # Check final line
        if len(current_line) > len(best_line):
            best_line = current_line
            
        return np.array(best_line) if best_line else np.array([])
    
    def _point_to_line_distance(self, point: np.ndarray, line_p1: np.ndarray, line_p2: np.ndarray) -> float:
        """Calculate distance from a point to a line segment.
        
        Args:
            point: Point coordinates
            line_p1: First point of line
            line_p2: Second point of line
            
        Returns:
            Distance from point to line
        """
        # Line equation: ax + by + c = 0
        a = line_p2[1] - line_p1[1]
        b = line_p1[0] - line_p2[0]
        c = line_p2[0] * line_p1[1] - line_p1[0] * line_p2[1]
        
        # Distance formula
        return abs(a * point[0] + b * point[1] + c) / np.sqrt(a*a + b*b)
    
    def initialize_particles_from_config(self, config_path: str) -> None:
        """Initialize particles from homography parameters in config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            base_params = config.get('homography_parameters', {})
            
            # Extract base homography parameters
            base_H = np.array([
                [base_params.get('H00', 1.0), base_params.get('H01', 0.0), base_params.get('H02', 0.0)],
                [base_params.get('H10', 0.0), base_params.get('H11', 1.0), base_params.get('H12', 0.0)],
                [base_params.get('H20', 0.0), base_params.get('H21', 0.0), 1.0]
            ])
            
            # Initialize particles with random variations around base parameters
            for i in range(self.num_particles):
                noise_scale = 0.1
                self.particles[i, 0] = base_H[0, 0] + np.random.normal(0, noise_scale * abs(base_H[0, 0]))
                self.particles[i, 1] = base_H[0, 1] + np.random.normal(0, noise_scale * abs(base_H[0, 1]) + 0.1)
                self.particles[i, 2] = base_H[0, 2] + np.random.normal(0, noise_scale * abs(base_H[0, 2]) + 50)
                self.particles[i, 3] = base_H[1, 0] + np.random.normal(0, noise_scale * abs(base_H[1, 0]) + 0.1)
                self.particles[i, 4] = base_H[1, 1] + np.random.normal(0, noise_scale * abs(base_H[1, 1]))
                self.particles[i, 5] = base_H[1, 2] + np.random.normal(0, noise_scale * abs(base_H[1, 2]) + 50)
                self.particles[i, 6] = base_H[2, 0] + np.random.normal(0, noise_scale * abs(base_H[2, 0]) + 0.001)
                self.particles[i, 7] = base_H[2, 1] + np.random.normal(0, noise_scale * abs(base_H[2, 1]) + 0.001)
                
            self.weights = np.ones(self.num_particles) / self.num_particles
            
        except Exception as e:
            print(f"[PARTICLE_FILTER] Error loading config {config_path}: {e}")
            self._initialize_identity_particles()
    
    def initialize_particles_from_config_dict(self, config_dict: dict) -> None:
        """Initialize particles from homography parameters in config dictionary."""
        try:
            base_params = config_dict.get('homography_parameters', {})
            
            # Extract base homography parameters
            base_H = np.array([
                [base_params.get('H00', 1.0), base_params.get('H01', 0.0), base_params.get('H02', 0.0)],
                [base_params.get('H10', 0.0), base_params.get('H11', 1.0), base_params.get('H12', 0.0)],
                [base_params.get('H20', 0.0), base_params.get('H21', 0.0), 1.0]
            ])
            
            # Initialize particles with random variations around base parameters
            for i in range(self.num_particles):
                noise_scale = 0.1
                self.particles[i, 0] = base_H[0, 0] + np.random.normal(0, noise_scale * abs(base_H[0, 0]))
                self.particles[i, 1] = base_H[0, 1] + np.random.normal(0, noise_scale * abs(base_H[0, 1]) + 0.1)
                self.particles[i, 2] = base_H[0, 2] + np.random.normal(0, noise_scale * abs(base_H[0, 2]) + 50)
                self.particles[i, 3] = base_H[1, 0] + np.random.normal(0, noise_scale * abs(base_H[1, 0]) + 0.1)
                self.particles[i, 4] = base_H[1, 1] + np.random.normal(0, noise_scale * abs(base_H[1, 1]))
                self.particles[i, 5] = base_H[1, 2] + np.random.normal(0, noise_scale * abs(base_H[1, 2]) + 50)
                self.particles[i, 6] = base_H[2, 0] + np.random.normal(0, noise_scale * abs(base_H[2, 0]) + 0.001)
                self.particles[i, 7] = base_H[2, 1] + np.random.normal(0, noise_scale * abs(base_H[2, 1]) + 0.001)
                
            self.weights = np.ones(self.num_particles) / self.num_particles
            
        except Exception as e:
            print(f"[PARTICLE_FILTER] Error loading config dict: {e}")
            self._initialize_identity_particles()
    
    def _initialize_identity_particles(self) -> None:
        """Initialize particles around identity transformation."""
        for i in range(self.num_particles):
            noise_scale = 0.1
            self.particles[i, 0] = 1.0 + np.random.normal(0, noise_scale)
            self.particles[i, 1] = np.random.normal(0, noise_scale)
            self.particles[i, 2] = np.random.normal(0, 50)
            self.particles[i, 3] = np.random.normal(0, noise_scale)
            self.particles[i, 4] = 1.0 + np.random.normal(0, noise_scale)
            self.particles[i, 5] = np.random.normal(0, 50)
            self.particles[i, 6] = np.random.normal(0, 0.001)
            self.particles[i, 7] = np.random.normal(0, 0.001)
            
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def get_homography_matrix(self, particle_idx: int) -> np.ndarray:
        """Get homography matrix for a specific particle."""
        params = self.particles[particle_idx]
        return np.array([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0]
        ])
    
    def calculate_particle_weights(self, frame_shape: Tuple[int, int]) -> None:
        """Calculate weights for all particles based on geometric constraints."""
        if not self.detected_lines:
            return
            
        # Find sidelines and endzone line
        sidelines = [line for line in self.detected_lines if 'sideline' in line.line_type]
        endzone_lines = [line for line in self.detected_lines if 'endzone' in line.line_type]
        
        for i in range(self.num_particles):
            weight = 1.0
            
            # Constraint 1: Sidelines should be parallel
            if len(sidelines) >= 2:
                weight *= self._evaluate_parallel_constraint(sidelines, i, frame_shape)
            
            # Constraint 2: Endzone line should be perpendicular to sidelines
            if len(sidelines) >= 1 and len(endzone_lines) >= 1:
                weight *= self._evaluate_perpendicular_constraint(sidelines[0], endzone_lines[0], i, frame_shape)
            
            # Constraint 3: Field aspect ratio should be reasonable after transformation
            weight *= self._evaluate_aspect_ratio_constraint(i, frame_shape)
            
            self.weights[i] = weight
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def _evaluate_parallel_constraint(self, sidelines: List[FieldLine], particle_idx: int, frame_shape: Tuple[int, int]) -> float:
        """Evaluate how well sidelines are parallel under this particle's homography."""
        if len(sidelines) < 2:
            return 1.0
            
        H = self.get_homography_matrix(particle_idx)
        
        try:
            angles = []
            for sideline in sidelines:
                if len(sideline.points) >= 2:
                    points_homogeneous = np.hstack([sideline.points, np.ones((len(sideline.points), 1))])
                    transformed_points = (H @ points_homogeneous.T).T
                    transformed_points = transformed_points[:, :2] / transformed_points[:, 2:3]
                    
                    if len(transformed_points) >= 2:
                        dx = transformed_points[-1, 0] - transformed_points[0, 0]
                        dy = transformed_points[-1, 1] - transformed_points[0, 1]
                        angle = np.arctan2(dy, dx)
                        angles.append(angle)
            
            if len(angles) >= 2:
                angle_diff = abs(angles[0] - angles[1])
                angle_diff = min(angle_diff, np.pi - angle_diff)
                return np.exp(-10 * angle_diff)
            
        except Exception as e:
            return 0.01
            
        return 1.0
    
    def _evaluate_perpendicular_constraint(self, sideline: FieldLine, endzone: FieldLine, 
                                         particle_idx: int, frame_shape: Tuple[int, int]) -> float:
        """Evaluate perpendicularity between sideline and endzone line."""
        H = self.get_homography_matrix(particle_idx)
        
        try:
            sideline_angle = self._get_transformed_line_angle(sideline, H)
            endzone_angle = self._get_transformed_line_angle(endzone, H)
            
            if sideline_angle is not None and endzone_angle is not None:
                angle_diff = abs(sideline_angle - endzone_angle)
                angle_diff = min(angle_diff, np.pi - angle_diff)
                perpendicular_error = abs(angle_diff - np.pi/2)
                return np.exp(-10 * perpendicular_error)
                
        except Exception as e:
            return 0.01
            
        return 1.0
    
    def _get_transformed_line_angle(self, line: FieldLine, H: np.ndarray) -> Optional[float]:
        """Get the angle of a line after homography transformation."""
        if len(line.points) < 2:
            return None
            
        try:
            points_homogeneous = np.hstack([line.points, np.ones((len(line.points), 1))])
            transformed_points = (H @ points_homogeneous.T).T
            transformed_points = transformed_points[:, :2] / transformed_points[:, 2:3]
            
            dx = transformed_points[-1, 0] - transformed_points[0, 0]
            dy = transformed_points[-1, 1] - transformed_points[0, 1]
            
            return np.arctan2(dy, dx)
            
        except Exception:
            return None
    
    def _evaluate_aspect_ratio_constraint(self, particle_idx: int, frame_shape: Tuple[int, int]) -> float:
        """Evaluate if the transformation produces a reasonable field aspect ratio."""
        H = self.get_homography_matrix(particle_idx)
        
        try:
            h, w = frame_shape
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
            
            transformed_corners = (H @ corners_homogeneous.T).T
            transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:3]
            
            min_x, min_y = np.min(transformed_corners, axis=0)
            max_x, max_y = np.max(transformed_corners, axis=0)
            
            width = max_x - min_x
            height = max_y - min_y
            
            if width > 0 and height > 0:
                aspect_ratio = width / height
                expected_ratio = self.field_length / self.field_width
                ratio_error = abs(aspect_ratio - expected_ratio) / expected_ratio
                return np.exp(-5 * ratio_error)
            
        except Exception:
            return 0.01
            
        return 0.01
    
    def resample_particles(self) -> None:
        """Resample particles based on their weights using systematic resampling."""
        cumsum = np.cumsum(self.weights)
        new_particles = np.zeros_like(self.particles)
        
        u = np.random.uniform(0, 1/self.num_particles)
        indices = np.zeros(self.num_particles, dtype=int)
        
        for i in range(self.num_particles):
            sample = u + i / self.num_particles
            indices[i] = np.searchsorted(cumsum, sample)
        
        for i in range(self.num_particles):
            new_particles[i] = self.particles[indices[i]]
        
        self.particles = new_particles
        
        # Add noise to prevent degeneracy
        noise_scale = 0.01
        for i in range(self.num_particles):
            self.particles[i, 0] += np.random.normal(0, noise_scale * abs(self.particles[i, 0]))
            self.particles[i, 1] += np.random.normal(0, noise_scale * abs(self.particles[i, 1]) + 0.01)
            self.particles[i, 2] += np.random.normal(0, noise_scale * abs(self.particles[i, 2]) + 5)
            self.particles[i, 3] += np.random.normal(0, noise_scale * abs(self.particles[i, 3]) + 0.01)
            self.particles[i, 4] += np.random.normal(0, noise_scale * abs(self.particles[i, 4]))
            self.particles[i, 5] += np.random.normal(0, noise_scale * abs(self.particles[i, 5]) + 5)
            self.particles[i, 6] += np.random.normal(0, noise_scale * abs(self.particles[i, 6]) + 0.0001)
            self.particles[i, 7] += np.random.normal(0, noise_scale * abs(self.particles[i, 7]) + 0.0001)
        
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def get_best_homography(self) -> np.ndarray:
        """Get the homography matrix of the best particle (highest weight)."""
        best_idx = np.argmax(self.weights)
        return self.get_homography_matrix(best_idx)
    
    def generate_top_down_view(self, frame: np.ndarray, output_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """Generate a top-down view of the field using the best particle's homography."""
        H = self.get_best_homography()
        
        try:
            field_to_image_scale_x = output_size[0] / self.field_length
            field_to_image_scale_y = output_size[1] / self.field_width
            scale = min(field_to_image_scale_x, field_to_image_scale_y)
            
            scaled_width = int(self.field_length * scale)
            scaled_height = int(self.field_width * scale)
            
            warped = cv2.warpPerspective(frame, H, (scaled_width, scaled_height))
            
            if (scaled_width, scaled_height) != output_size:
                result = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
                start_x = (output_size[0] - scaled_width) // 2
                start_y = (output_size[1] - scaled_height) // 2
                result[start_y:start_y+scaled_height, start_x:start_x+scaled_width] = warped
            else:
                result = warped
            
            return result
            
        except Exception as e:
            print(f"[PARTICLE_FILTER] Error generating top-down view: {e}")
            return frame
    
    def generate_top_down_view_with_lines(self, frame: np.ndarray, output_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """Generate a top-down view with detected field lines overlaid."""
        # Generate basic top-down view
        top_down = self.generate_top_down_view(frame, output_size)
        
        if not self.detected_lines:
            return top_down
        
        try:
            H = self.get_best_homography()
            
            # Colors for different line types
            colors = {
                'sideline_left': (255, 0, 0),    # Red
                'sideline_right': (0, 255, 0),   # Green
                'endzone_back': (0, 0, 255)      # Blue
            }
            
            # Calculate scaling and offset for the output image
            field_to_image_scale_x = output_size[0] / self.field_length
            field_to_image_scale_y = output_size[1] / self.field_width
            scale = min(field_to_image_scale_x, field_to_image_scale_y)
            
            scaled_width = int(self.field_length * scale)
            scaled_height = int(self.field_width * scale)
            start_x = (output_size[0] - scaled_width) // 2
            start_y = (output_size[1] - scaled_height) // 2
            
            # Draw transformed field lines
            for line in self.detected_lines:
                if len(line.points) < 2:
                    continue
                    
                color = colors.get(line.line_type, (255, 255, 255))
                
                # Transform line points to top-down view
                points_homogeneous = np.hstack([line.points, np.ones((len(line.points), 1))])
                transformed_points = (H @ points_homogeneous.T).T
                
                # Normalize homogeneous coordinates
                transformed_points = transformed_points[:, :2] / transformed_points[:, 2:3]
                
                # Scale transformed points to fit the warped image size
                if scaled_width > 0 and scaled_height > 0:
                    # The transformed points are already in the warped image coordinate system
                    # Scale them to the actual warped size, then offset to center in output
                    scale_x = scaled_width / frame.shape[1] if frame.shape[1] > 0 else 1
                    scale_y = scaled_height / frame.shape[0] if frame.shape[0] > 0 else 1
                    
                    transformed_points[:, 0] = transformed_points[:, 0] * scale_x + start_x
                    transformed_points[:, 1] = transformed_points[:, 1] * scale_y + start_y
                    
                    # Convert to integer points and draw
                    transformed_points = transformed_points.astype(np.int32)
                    
                    # Clip points to image bounds
                    transformed_points[:, 0] = np.clip(transformed_points[:, 0], 0, output_size[0] - 1)
                    transformed_points[:, 1] = np.clip(transformed_points[:, 1], 0, output_size[1] - 1)
                    
                    # Draw line segments
                    for i in range(len(transformed_points) - 1):
                        cv2.line(top_down, tuple(transformed_points[i]), tuple(transformed_points[i+1]), color, 3)
                    
                    # Draw line points
                    for point in transformed_points:
                        cv2.circle(top_down, tuple(point), 5, color, -1)
                    
                    # Add label
                    if len(transformed_points) > 0:
                        label_pos = tuple(transformed_points[len(transformed_points)//2])
                        cv2.putText(top_down, line.line_type, label_pos, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw ideal field outline for reference
            field_corners = np.array([
                [0, 0], [self.field_length, 0], 
                [self.field_length, self.field_width], [0, self.field_width]
            ], dtype=np.float32)
            
            # Scale and offset field corners
            field_corners[:, 0] = (field_corners[:, 0] / self.field_length) * scaled_width + start_x
            field_corners[:, 1] = (field_corners[:, 1] / self.field_width) * scaled_height + start_y
            field_corners = field_corners.astype(np.int32)
            
            # Draw field outline
            cv2.polylines(top_down, [field_corners], True, (255, 255, 255), 2)
            
            return top_down
            
        except Exception as e:
            print(f"[PARTICLE_FILTER] Error adding lines to top-down view: {e}")
            return top_down
    
    def update_homography_continuously(self, segmentation_results: List[Any], frame: np.ndarray, 
                                      num_iterations: int = 3) -> Tuple[bool, np.ndarray]:
        """Continuously update homography with new field segmentation data.
        
        This method performs a lightweight update suitable for real-time processing.
        
        Args:
            segmentation_results: Current field segmentation results
            frame: Current video frame
            num_iterations: Number of filter iterations (fewer for real-time)
            
        Returns:
            Tuple of (success, homography_matrix)
        """
        try:
            # Combine field masks
            combined_mask = self.combine_field_masks(segmentation_results)
            
            if combined_mask.size == 0:
                return False, self.get_best_homography()
            
            # Simplify mask and find contour
            simplified_mask, contours = self.simplify_mask_and_find_contour(combined_mask)
            
            if not contours:
                return False, self.get_best_homography()
            
            # Extract field lines using improved RANSAC method
            # Pass the simplified mask directly instead of contours
            new_lines = self.extract_field_lines_from_mask(simplified_mask, frame.shape[:2])
            
            if not new_lines:
                return False, self.get_best_homography()
            
            # Update detected lines
            self.detected_lines = new_lines
            
            # Perform lightweight particle filter update
            for iteration in range(num_iterations):
                self.calculate_particle_weights(frame.shape[:2])
                
                # Only resample if effective sample size is low
                effective_sample_size = 1.0 / np.sum(self.weights ** 2)
                if effective_sample_size < self.num_particles / 3:
                    self.resample_particles()
            
            return True, self.get_best_homography()
            
        except Exception as e:
            print(f"[PARTICLE_FILTER] Error in continuous update: {e}")
            return False, self.get_best_homography()
    
    def is_initialized(self) -> bool:
        """Check if the particle filter has been properly initialized.
        
        Returns:
            True if particles have been initialized, False otherwise
        """
        # Check if particles have been initialized (not all zeros)
        return not np.allclose(self.particles, 0.0)

    def run_particle_filter(self, segmentation_results: List[Any], frame: np.ndarray, 
                          num_iterations: int = 10) -> np.ndarray:
        """Run the complete particle filter process."""
        combined_mask = self.combine_field_masks(segmentation_results)
        
        if combined_mask.size == 0:
            print("[PARTICLE_FILTER] No field mask found, using identity homography")
            return np.eye(3)
        
        simplified_mask, contours = self.simplify_mask_and_find_contour(combined_mask)
        
        if not contours:
            print("[PARTICLE_FILTER] No contours found, using identity homography")
            return np.eye(3)
        
        self.detected_lines = self.extract_field_lines_from_mask(simplified_mask, frame.shape[:2])
        
        if not self.detected_lines:
            print("[PARTICLE_FILTER] No field lines detected, using identity homography")
            return np.eye(3)
        
        for iteration in range(num_iterations):
            self.calculate_particle_weights(frame.shape[:2])
            self.resample_particles()
        
        return self.get_best_homography()
    
    def visualize_field_lines(self, frame: np.ndarray) -> np.ndarray:
        """Visualize detected field lines on the frame."""
        if not self.detected_lines:
            return frame
            
        viz_frame = frame.copy()
        
        colors = {
            'sideline_left': (255, 0, 0),
            'sideline_right': (0, 255, 0),
            'endzone_back': (0, 0, 255)
        }
        
        for line in self.detected_lines:
            color = colors.get(line.line_type, (255, 255, 255))
            
            if len(line.points) >= 2:
                points = line.points.astype(np.int32)
                for i in range(len(points) - 1):
                    cv2.line(viz_frame, tuple(points[i]), tuple(points[i+1]), color, 3)
                
                for point in points:
                    cv2.circle(viz_frame, tuple(point), 5, color, -1)
                
                label_pos = tuple(points[len(points)//2])
                cv2.putText(viz_frame, line.line_type, label_pos, 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return viz_frame
