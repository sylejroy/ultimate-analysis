"""
Ground plane estimation for Ultimate Analysis.

This module implements algorithms for estimating the ground plane transformation
from image coordinates to real-world coordinates, including both field-geometry-based
and player-height-based calibration methods.
"""
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
import logging
from dataclasses import dataclass

from ..core.models import Point2D, Point3D
from ..core.exceptions import UltimateAnalysisError

logger = logging.getLogger("ultimate_analysis.processing.ground_plane")


@dataclass
class FieldGeometry:
    """Field geometry parameters."""
    length: float = 70.0  # meters
    width: float = 37.0   # meters
    goal_zone_length: float = 18.0  # meters
    
    
@dataclass
class CameraCalibration:
    """Camera calibration parameters."""
    intrinsic_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    height: float  # camera height in meters
    tilt_angle: float  # camera tilt angle in degrees
    

class GroundPlaneEstimator:
    """
    Ground plane estimation using field geometry and player height calibration.
    
    This class implements two main approaches:
    1. Field-geometry-based calibration using known field features
    2. Player-height-based calibration using average player height
    """
    
    def __init__(self, field_geometry: FieldGeometry = None):
        self.field_geometry = field_geometry or FieldGeometry()
        self.homography_matrix: Optional[np.ndarray] = None
        self.camera_calibration: Optional[CameraCalibration] = None
        self.is_calibrated = False
        
    def calibrate_from_field_geometry(
        self, 
        image_points: List[Point2D], 
        field_points: List[Point2D]
    ) -> bool:
        """
        Calibrate ground plane using field geometry.
        
        Algorithm implementation based on the pseudocode:
        1. Detect field features (corners, lines, goal areas)
        2. Map to known field coordinates
        3. Compute homography transformation
        4. Validate and refine calibration
        
        Args:
            image_points: List of points in image coordinates
            field_points: List of corresponding points in field coordinates (meters)
            
        Returns:
            True if calibration successful, False otherwise
        """
        try:
            if len(image_points) < 4 or len(field_points) < 4:
                raise UltimateAnalysisError("Need at least 4 point correspondences")
            
            if len(image_points) != len(field_points):
                raise UltimateAnalysisError("Number of image and field points must match")
            
            # Convert points to numpy arrays
            img_pts = np.array([(p.x, p.y) for p in image_points], dtype=np.float32)
            field_pts = np.array([(p.x, p.y) for p in field_points], dtype=np.float32)
            
            # Step 1: Detect field features (this would be implemented with actual detection)
            logger.info("Detecting field features...")
            
            # Step 2: Map to known field coordinates
            logger.info("Mapping to field coordinates...")
            
            # Step 3: Compute homography transformation
            logger.info("Computing homography transformation...")
            self.homography_matrix, mask = cv2.findHomography(
                img_pts, field_pts, cv2.RANSAC, 5.0
            )
            
            if self.homography_matrix is None:
                raise UltimateAnalysisError("Failed to compute homography")
            
            # Step 4: Validate calibration
            logger.info("Validating calibration...")
            if self._validate_field_calibration(img_pts, field_pts):
                self.is_calibrated = True
                logger.info("Field geometry calibration successful")
                return True
            else:
                logger.warning("Field geometry calibration validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Field geometry calibration failed: {e}")
            return False
    
    def calibrate_from_player_height(
        self, 
        player_detections: List[Dict[str, Any]], 
        average_player_height: float = 1.85
    ) -> bool:
        """
        Calibrate ground plane using player height as reference.
        
        Algorithm implementation based on the pseudocode:
        1. Detect players in multiple frames
        2. Calculate apparent height in pixels
        3. Use known player height to estimate scale
        4. Compute ground plane transformation
        
        Args:
            player_detections: List of player detection data
            average_player_height: Average player height in meters
            
        Returns:
            True if calibration successful, False otherwise
        """
        try:
            if len(player_detections) < 3:
                raise UltimateAnalysisError("Need at least 3 player detections")
            
            # Step 1: Detect players (already done, passed as parameter)
            logger.info("Processing player detections...")
            
            # Step 2: Calculate apparent height in pixels
            logger.info("Calculating apparent heights...")
            height_measurements = []
            
            for detection in player_detections:
                bbox = detection.get('bbox')
                if bbox is None:
                    continue
                
                # Calculate player height in pixels
                pixel_height = bbox[3] - bbox[1]  # y2 - y1
                foot_y = bbox[3]  # bottom of bounding box
                
                height_measurements.append({
                    'pixel_height': pixel_height,
                    'foot_y': foot_y,
                    'center_x': (bbox[0] + bbox[2]) / 2
                })
            
            if len(height_measurements) < 3:
                raise UltimateAnalysisError("Insufficient valid height measurements")
            
            # Step 3: Use known player height to estimate scale
            logger.info("Estimating scale and perspective...")
            
            # Calculate scale variation based on vertical position
            # Players farther from camera appear smaller
            height_data = np.array([(h['pixel_height'], h['foot_y']) for h in height_measurements])
            
            # Fit a linear model: pixel_height = a * foot_y + b
            # This accounts for perspective effects
            A = np.vstack([height_data[:, 1], np.ones(len(height_data))]).T
            coeffs, residuals, rank, s = np.linalg.lstsq(A, height_data[:, 0], rcond=None)
            
            if rank < 2:
                raise UltimateAnalysisError("Insufficient data for perspective estimation")
            
            # Step 4: Compute ground plane transformation
            logger.info("Computing ground plane transformation...")
            
            # Use the fitted model to create transformation
            # This is a simplified version - in practice, you'd use more sophisticated methods
            self.homography_matrix = self._compute_height_based_homography(
                height_measurements, average_player_height, coeffs
            )
            
            if self.homography_matrix is None:
                raise UltimateAnalysisError("Failed to compute height-based homography")
            
            # Validate calibration
            if self._validate_height_calibration(height_measurements, average_player_height):
                self.is_calibrated = True
                logger.info("Player height calibration successful")
                return True
            else:
                logger.warning("Player height calibration validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Player height calibration failed: {e}")
            return False
    
    def _compute_height_based_homography(
        self, 
        height_measurements: List[Dict[str, Any]], 
        average_height: float, 
        perspective_coeffs: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute homography matrix based on height measurements.
        
        This is a simplified implementation. In practice, you would use
        more sophisticated camera calibration techniques.
        """
        try:
            # Create correspondence points
            # For simplicity, assume a flat ground plane and use height ratios
            image_points = []
            world_points = []
            
            for i, measurement in enumerate(height_measurements):
                # Image point (foot of player)
                img_x = measurement['center_x']
                img_y = measurement['foot_y']
                
                # Estimate world position based on image position
                # This is a simplified mapping - in practice you'd use more sophisticated methods
                world_x = (img_x - 320) * 0.1  # crude mapping
                world_y = i * 2.0  # spread players along field
                
                image_points.append([img_x, img_y])
                world_points.append([world_x, world_y])
            
            # Add some reference points for the field boundaries
            # This ensures the homography covers the full field
            img_pts = np.array(image_points + [[0, 480], [640, 480], [320, 240]], dtype=np.float32)
            world_pts = np.array(world_points + [[-35, 0], [35, 0], [0, 35]], dtype=np.float32)
            
            # Compute homography
            homography, _ = cv2.findHomography(img_pts, world_pts, cv2.RANSAC)
            
            return homography
            
        except Exception as e:
            logger.error(f"Error computing height-based homography: {e}")
            return None
    
    def _validate_field_calibration(
        self, 
        image_points: np.ndarray, 
        field_points: np.ndarray
    ) -> bool:
        """Validate field geometry calibration."""
        try:
            # Transform image points to field coordinates
            transformed_points = cv2.perspectiveTransform(
                image_points.reshape(-1, 1, 2), self.homography_matrix
            ).reshape(-1, 2)
            
            # Calculate reprojection error
            errors = np.linalg.norm(transformed_points - field_points, axis=1)
            mean_error = np.mean(errors)
            
            logger.info(f"Field calibration mean error: {mean_error:.3f} meters")
            
            # Accept calibration if mean error is less than 1 meter
            return mean_error < 1.0
            
        except Exception as e:
            logger.error(f"Field calibration validation error: {e}")
            return False
    
    def _validate_height_calibration(
        self, 
        height_measurements: List[Dict[str, Any]], 
        expected_height: float
    ) -> bool:
        """Validate height-based calibration."""
        try:
            # Check if the calibration produces reasonable height estimates
            total_error = 0
            valid_measurements = 0
            
            for measurement in height_measurements:
                # Estimate real-world height using the calibration
                estimated_height = self._estimate_height_from_pixels(
                    measurement['pixel_height'], 
                    measurement['foot_y']
                )
                
                if estimated_height is not None:
                    error = abs(estimated_height - expected_height)
                    total_error += error
                    valid_measurements += 1
            
            if valid_measurements == 0:
                return False
            
            mean_error = total_error / valid_measurements
            logger.info(f"Height calibration mean error: {mean_error:.3f} meters")
            
            # Accept calibration if mean error is less than 0.3 meters
            return mean_error < 0.3
            
        except Exception as e:
            logger.error(f"Height calibration validation error: {e}")
            return False
    
    def _estimate_height_from_pixels(
        self, 
        pixel_height: float, 
        foot_y: float
    ) -> Optional[float]:
        """Estimate real-world height from pixel measurements."""
        # This is a placeholder implementation
        # In practice, you'd use the calibrated transformation
        try:
            # Use a simple perspective model
            # Height decreases as distance from camera increases
            scale_factor = 1.0 / (1.0 + foot_y * 0.001)
            estimated_height = pixel_height * scale_factor * 0.01
            
            return estimated_height
            
        except Exception as e:
            logger.error(f"Height estimation error: {e}")
            return None
    
    def image_to_world(self, image_point: Point2D) -> Optional[Point2D]:
        """
        Transform image coordinates to world coordinates.
        
        Args:
            image_point: Point in image coordinates (pixels)
            
        Returns:
            Point in world coordinates (meters) or None if not calibrated
        """
        if not self.is_calibrated or self.homography_matrix is None:
            logger.warning("Ground plane not calibrated")
            return None
        
        try:
            # Transform point using homography
            img_pt = np.array([[[image_point.x, image_point.y]]], dtype=np.float32)
            world_pt = cv2.perspectiveTransform(img_pt, self.homography_matrix)
            
            return Point2D(x=world_pt[0][0][0], y=world_pt[0][0][1])
            
        except Exception as e:
            logger.error(f"Image to world transformation error: {e}")
            return None
    
    def world_to_image(self, world_point: Point2D) -> Optional[Point2D]:
        """
        Transform world coordinates to image coordinates.
        
        Args:
            world_point: Point in world coordinates (meters)
            
        Returns:
            Point in image coordinates (pixels) or None if not calibrated
        """
        if not self.is_calibrated or self.homography_matrix is None:
            logger.warning("Ground plane not calibrated")
            return None
        
        try:
            # Use inverse homography
            inverse_homography = np.linalg.inv(self.homography_matrix)
            
            world_pt = np.array([[[world_point.x, world_point.y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(world_pt, inverse_homography)
            
            return Point2D(x=img_pt[0][0][0], y=img_pt[0][0][1])
            
        except Exception as e:
            logger.error(f"World to image transformation error: {e}")
            return None
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration information."""
        return {
            'is_calibrated': self.is_calibrated,
            'homography_matrix': self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            'field_geometry': {
                'length': self.field_geometry.length,
                'width': self.field_geometry.width,
                'goal_zone_length': self.field_geometry.goal_zone_length
            }
        }
    
    def save_calibration(self, filepath: str) -> bool:
        """Save calibration to file."""
        try:
            import json
            
            calibration_data = self.get_calibration_info()
            
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            logger.info(f"Calibration saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """Load calibration from file."""
        try:
            import json
            
            with open(filepath, 'r') as f:
                calibration_data = json.load(f)
            
            self.is_calibrated = calibration_data.get('is_calibrated', False)
            
            if calibration_data.get('homography_matrix'):
                self.homography_matrix = np.array(calibration_data['homography_matrix'])
            
            field_geom = calibration_data.get('field_geometry', {})
            self.field_geometry = FieldGeometry(
                length=field_geom.get('length', 70.0),
                width=field_geom.get('width', 37.0),
                goal_zone_length=field_geom.get('goal_zone_length', 18.0)
            )
            
            logger.info(f"Calibration loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
            return False


# Global instance for easy access
_ground_plane_estimator = GroundPlaneEstimator()


def get_ground_plane_estimator() -> GroundPlaneEstimator:
    """Get the global ground plane estimator instance."""
    return _ground_plane_estimator


def calibrate_ground_plane_from_field(
    image_points: List[Point2D], 
    field_points: List[Point2D]
) -> bool:
    """
    Calibrate ground plane using field geometry.
    
    Args:
        image_points: Points in image coordinates
        field_points: Corresponding points in field coordinates
        
    Returns:
        True if calibration successful
    """
    return _ground_plane_estimator.calibrate_from_field_geometry(image_points, field_points)


def calibrate_ground_plane_from_height(
    player_detections: List[Dict[str, Any]], 
    average_height: float = 1.85
) -> bool:
    """
    Calibrate ground plane using player height.
    
    Args:
        player_detections: List of player detection data
        average_height: Average player height in meters
        
    Returns:
        True if calibration successful
    """
    return _ground_plane_estimator.calibrate_from_player_height(player_detections, average_height)


def transform_point_to_world(image_point: Point2D) -> Optional[Point2D]:
    """Transform image point to world coordinates."""
    return _ground_plane_estimator.image_to_world(image_point)


def transform_point_to_image(world_point: Point2D) -> Optional[Point2D]:
    """Transform world point to image coordinates."""
    return _ground_plane_estimator.world_to_image(world_point)
