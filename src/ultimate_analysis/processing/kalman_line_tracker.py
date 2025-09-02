"""Kalman filter-based line tracking for field lines.

This module implements Kalman filtering to track field lines over time,
providing smooth and stable line detection for the top-down view.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time

from ..config.settings import get_setting


@dataclass
class LineState:
    """State of a tracked line."""
    start_point: np.ndarray  # [x, y]
    end_point: np.ndarray    # [x, y]
    velocity_start: np.ndarray  # [vx, vy]
    velocity_end: np.ndarray    # [vx, vy]
    confidence: float
    last_update: float
    track_id: int
    age: int


class KalmanLineFilter:
    """Kalman filter for tracking a single line."""
    
    def __init__(self, initial_start: np.ndarray, initial_end: np.ndarray, 
                 track_id: int, confidence: float = 1.0):
        """Initialize Kalman filter for line tracking.
        
        Args:
            initial_start: Initial start point [x, y]
            initial_end: Initial end point [x, y]
            track_id: Unique identifier for this track
            confidence: Initial confidence score
        """
        self.track_id = track_id
        self.confidence = confidence
        self.age = 0
        self.last_update = time.time()
        
        # State vector: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
        # Two endpoints with position and velocity
        self.state_dim = 8
        self.measure_dim = 4  # [x1, y1, x2, y2]
        
        # Initialize Kalman filter
        self.kf = cv2.KalmanFilter(self.state_dim, self.measure_dim)
        
        # Transition matrix (constant velocity model)
        dt = 1.0  # Time step
        self.kf.transitionMatrix = np.eye(self.state_dim, dtype=np.float32)
        self.kf.transitionMatrix[0, 2] = dt  # x1 += vx1 * dt
        self.kf.transitionMatrix[1, 3] = dt  # y1 += vy1 * dt
        self.kf.transitionMatrix[4, 6] = dt  # x2 += vx2 * dt
        self.kf.transitionMatrix[5, 7] = dt  # y2 += vy2 * dt
        
        # Measurement matrix (we observe positions only)
        self.kf.measurementMatrix = np.zeros((self.measure_dim, self.state_dim), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0  # measure x1
        self.kf.measurementMatrix[1, 1] = 1.0  # measure y1
        self.kf.measurementMatrix[2, 4] = 1.0  # measure x2
        self.kf.measurementMatrix[3, 5] = 1.0  # measure y2
        
        # Process noise covariance
        process_noise = get_setting("tracking.kalman.process_noise", 1e-3)
        self.kf.processNoiseCov = np.eye(self.state_dim, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        measurement_noise = get_setting("tracking.kalman.measurement_noise", 1e-1)
        self.kf.measurementNoiseCov = np.eye(self.measure_dim, dtype=np.float32) * measurement_noise
        
        # Error covariance matrix
        self.kf.errorCovPost = np.eye(self.state_dim, dtype=np.float32) * 1.0
        
        # Initialize state with zero velocities
        self.kf.statePre = np.array([
            initial_start[0], initial_start[1], 0.0, 0.0,  # x1, y1, vx1, vy1
            initial_end[0], initial_end[1], 0.0, 0.0       # x2, y2, vx2, vy2
        ], dtype=np.float32)
        
        self.kf.statePost = self.kf.statePre.copy()
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next line position.
        
        Returns:
            Tuple of (predicted_start, predicted_end) points
        """
        prediction = self.kf.predict()
        
        start_point = np.array([prediction[0], prediction[1]], dtype=np.float32)
        end_point = np.array([prediction[4], prediction[5]], dtype=np.float32)
        
        return start_point, end_point
    
    def update(self, measurement_start: np.ndarray, measurement_end: np.ndarray, 
               confidence: float) -> Tuple[np.ndarray, np.ndarray]:
        """Update filter with new measurement.
        
        Args:
            measurement_start: Measured start point [x, y]
            measurement_end: Measured end point [x, y]
            confidence: Confidence of this measurement
            
        Returns:
            Tuple of (corrected_start, corrected_end) points
        """
        # Create measurement vector
        measurement = np.array([
            measurement_start[0], measurement_start[1],
            measurement_end[0], measurement_end[1]
        ], dtype=np.float32)
        
        # Update filter
        self.kf.correct(measurement)
        
        # Update tracking state
        self.confidence = 0.8 * self.confidence + 0.2 * confidence  # Exponential smoothing
        self.age += 1
        self.last_update = time.time()
        
        # Extract corrected positions
        state = self.kf.statePost
        start_point = np.array([state[0], state[1]], dtype=np.float32)
        end_point = np.array([state[4], state[5]], dtype=np.float32)
        
        return start_point, end_point
    
    def get_current_line(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current filtered line position.
        
        Returns:
            Tuple of (start_point, end_point)
        """
        state = self.kf.statePost
        start_point = np.array([state[0], state[1]], dtype=np.float32)
        end_point = np.array([state[4], state[5]], dtype=np.float32)
        
        return start_point, end_point
    
    def is_expired(self, max_age_seconds: float = 2.0) -> bool:
        """Check if track is expired due to lack of updates.
        
        Args:
            max_age_seconds: Maximum age in seconds before expiry
            
        Returns:
            True if track should be removed
        """
        return (time.time() - self.last_update) > max_age_seconds


class KalmanLineTracker:
    """Multi-line tracker using Kalman filters."""
    
    def __init__(self):
        """Initialize the line tracker."""
        self.filters: Dict[int, KalmanLineFilter] = {}
        self.next_track_id = 0
        self.max_tracks = get_setting("tracking.kalman.max_tracks", 8)
        self.association_threshold = get_setting("tracking.kalman.association_threshold", 50.0)
        self.min_confidence = get_setting("tracking.kalman.min_confidence", 0.3)
        
        print(f"[KALMAN_TRACKER] Initialized with max_tracks={self.max_tracks}, "
              f"association_threshold={self.association_threshold}")
    
    def update(self, detected_lines: List[Tuple[np.ndarray, np.ndarray]], 
               confidences: List[float]) -> List[Dict[str, Any]]:
        """Update tracker with new line detections.
        
        Args:
            detected_lines: List of (start_point, end_point) tuples
            confidences: List of confidence scores for each line
            
        Returns:
            List of tracked lines with metadata
        """
        if not detected_lines:
            # Predict all existing tracks
            results = []
            for track_id, filter_obj in list(self.filters.items()):
                start, end = filter_obj.predict()
                
                # Check if track is expired
                if filter_obj.is_expired():
                    del self.filters[track_id]
                    print(f"[KALMAN_TRACKER] Removed expired track {track_id}")
                    continue
                
                results.append({
                    'start_point': start,
                    'end_point': end,
                    'confidence': filter_obj.confidence,
                    'track_id': track_id,
                    'age': filter_obj.age,
                    'is_predicted': True
                })
            
            return results
        
        # Data association between detections and existing tracks
        matched_pairs, unmatched_detections, unmatched_tracks = self._associate_detections(
            detected_lines, confidences)
        
        results = []
        
        # Update matched tracks
        for track_id, detection_idx in matched_pairs:
            filter_obj = self.filters[track_id]
            start_point, end_point = detected_lines[detection_idx]
            confidence = confidences[detection_idx]
            
            # Update filter
            corrected_start, corrected_end = filter_obj.update(start_point, end_point, confidence)
            
            results.append({
                'start_point': corrected_start,
                'end_point': corrected_end,
                'confidence': filter_obj.confidence,
                'track_id': track_id,
                'age': filter_obj.age,
                'is_predicted': False
            })
            
            print(f"[KALMAN_TRACKER] Updated track {track_id}: confidence={filter_obj.confidence:.3f}, age={filter_obj.age}")
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            if len(self.filters) >= self.max_tracks:
                # Remove oldest low-confidence track
                oldest_id = min(self.filters.keys(), 
                              key=lambda x: (self.filters[x].confidence, -self.filters[x].age))
                if self.filters[oldest_id].confidence < self.min_confidence:
                    del self.filters[oldest_id]
                    print(f"[KALMAN_TRACKER] Removed low-confidence track {oldest_id}")
                else:
                    continue  # Skip creating new track if can't remove old one
            
            start_point, end_point = detected_lines[detection_idx]
            confidence = confidences[detection_idx]
            
            if confidence >= self.min_confidence:
                new_filter = KalmanLineFilter(start_point, end_point, 
                                            self.next_track_id, confidence)
                self.filters[self.next_track_id] = new_filter
                
                results.append({
                    'start_point': start_point,
                    'end_point': end_point,
                    'confidence': confidence,
                    'track_id': self.next_track_id,
                    'age': 0,
                    'is_predicted': False
                })
                
                print(f"[KALMAN_TRACKER] Created new track {self.next_track_id}")
                self.next_track_id += 1
        
        # Predict unmatched existing tracks
        for track_id in unmatched_tracks:
            filter_obj = self.filters[track_id]
            start, end = filter_obj.predict()
            
            # Check if track is expired
            if filter_obj.is_expired():
                del self.filters[track_id]
                print(f"[KALMAN_TRACKER] Removed expired track {track_id}")
                continue
            
            results.append({
                'start_point': start,
                'end_point': end,
                'confidence': filter_obj.confidence * 0.9,  # Reduce confidence for predicted
                'track_id': track_id,
                'age': filter_obj.age,
                'is_predicted': True
            })
        
        return results
    
    def _associate_detections(self, detected_lines: List[Tuple[np.ndarray, np.ndarray]], 
                            confidences: List[float]) -> Tuple[List[Tuple[int, int]], 
                                                             List[int], List[int]]:
        """Associate detections with existing tracks using Hungarian algorithm.
        
        Args:
            detected_lines: List of detected line segments
            confidences: List of detection confidences
            
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if not self.filters or not detected_lines:
            return [], list(range(len(detected_lines))), list(self.filters.keys())
        
        # Calculate cost matrix
        track_ids = list(self.filters.keys())
        cost_matrix = np.full((len(track_ids), len(detected_lines)), np.inf)
        
        for i, track_id in enumerate(track_ids):
            # Predict track position
            pred_start, pred_end = self.filters[track_id].predict()
            
            for j, (det_start, det_end) in enumerate(detected_lines):
                # Calculate line-to-line distance
                distance = self._line_distance(
                    (pred_start, pred_end), (det_start, det_end))
                
                if distance < self.association_threshold:
                    # Consider confidence in cost (lower is better)
                    confidence_factor = max(0.1, 1.0 - confidences[j])
                    cost_matrix[i, j] = distance * confidence_factor
        
        # Simple greedy assignment (for now - could use Hungarian algorithm)
        matched_pairs = []
        unmatched_tracks = set(range(len(track_ids)))
        unmatched_detections = set(range(len(detected_lines)))
        
        # Find best assignments
        while True:
            # Find minimum cost
            min_cost = np.inf
            best_track_idx = -1
            best_det_idx = -1
            
            for i in unmatched_tracks:
                for j in unmatched_detections:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        best_track_idx = i
                        best_det_idx = j
            
            # If no valid assignment found, break
            if min_cost == np.inf:
                break
            
            # Make assignment
            track_id = track_ids[best_track_idx]
            matched_pairs.append((track_id, best_det_idx))
            unmatched_tracks.remove(best_track_idx)
            unmatched_detections.remove(best_det_idx)
        
        # Convert remaining indices to track IDs
        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
        unmatched_detection_indices = list(unmatched_detections)
        
        return matched_pairs, unmatched_detection_indices, unmatched_track_ids
    
    def _line_distance(self, line1: Tuple[np.ndarray, np.ndarray], 
                      line2: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate distance between two line segments.
        
        Args:
            line1: First line as (start_point, end_point)
            line2: Second line as (start_point, end_point)
            
        Returns:
            Distance metric between lines
        """
        start1, end1 = line1
        start2, end2 = line2
        
        # Calculate endpoint distances
        d1 = np.linalg.norm(start1 - start2)  # start to start
        d2 = np.linalg.norm(end1 - end2)      # end to end
        d3 = np.linalg.norm(start1 - end2)    # start to end (flipped)
        d4 = np.linalg.norm(end1 - start2)    # end to start (flipped)
        
        # Use minimum of aligned and flipped assignments
        aligned_distance = (d1 + d2) / 2
        flipped_distance = (d3 + d4) / 2
        
        return min(aligned_distance, flipped_distance)
    
    def reset(self):
        """Reset tracker, removing all tracks."""
        self.filters.clear()
        self.next_track_id = 0
        print("[KALMAN_TRACKER] Reset - all tracks removed")
    
    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Get all currently active tracks.
        
        Returns:
            List of active track information
        """
        results = []
        current_time = time.time()
        
        for track_id, filter_obj in self.filters.items():
            start, end = filter_obj.get_current_line()
            
            results.append({
                'start_point': start,
                'end_point': end,
                'confidence': filter_obj.confidence,
                'track_id': track_id,
                'age': filter_obj.age,
                'last_update': filter_obj.last_update,
                'time_since_update': current_time - filter_obj.last_update
            })
        
        return results


# Global tracker instance
_line_tracker = None


def get_line_tracker() -> KalmanLineTracker:
    """Get the global line tracker instance."""
    global _line_tracker
    if _line_tracker is None:
        _line_tracker = KalmanLineTracker()
    return _line_tracker


def track_lines(detected_lines: List[Tuple[np.ndarray, np.ndarray]], 
               confidences: List[float]) -> List[Dict[str, Any]]:
    """Track field lines using Kalman filtering.
    
    Args:
        detected_lines: List of (start_point, end_point) tuples from RANSAC
        confidences: List of confidence scores for each line
        
    Returns:
        List of tracked lines with filtering applied
    """
    tracker = get_line_tracker()
    return tracker.update(detected_lines, confidences)


def reset_line_tracker():
    """Reset the line tracker."""
    global _line_tracker
    if _line_tracker is not None:
        _line_tracker.reset()


def get_tracked_lines() -> List[Dict[str, Any]]:
    """Get current tracked lines without updating.
    
    Returns:
        List of currently tracked lines
    """
    tracker = get_line_tracker()
    return tracker.get_active_tracks()
