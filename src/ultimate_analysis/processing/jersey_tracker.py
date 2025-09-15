"""Jersey number tracking module with probabilistic fusion.

This module implements probabilistic jersey number tracking for player objects across
their entire tracking history. It combines noisy EasyOCR measurements with confidence
scores and spatial weighting to provide reliable jersey number identification.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


from ..config.settings import get_setting


@dataclass
class JerseyMeasurement:
    """Single jersey number measurement from EasyOCR."""

    jersey_number: str
    confidence: float
    spatial_weight: float  # Weight based on digit position within bounding box
    timestamp: float
    bbox_center_x: float  # Normalized position within bounding box (0-1)
    ocr_results: List[Any] = field(default_factory=list)


@dataclass
class JerseyProbability:
    """Jersey number probability entry."""

    jersey_number: str
    probability: float
    last_seen: float
    measurement_count: int


class JerseyNumberTracker:
    """Probabilistic jersey number tracker for player objects."""

    def __init__(self):
        """Initialize jersey number tracker."""
        # Track measurements for each object
        self._track_measurements: Dict[int, deque] = defaultdict(lambda: deque())

        # Track probabilities for each object
        self._track_probabilities: Dict[int, Dict[str, JerseyProbability]] = defaultdict(dict)

        # Load configuration parameters
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration parameters."""
        self.max_history_length = get_setting("player_id.tracking.max_history_length", 30)
        self.confidence_decay_factor = get_setting(
            "player_id.tracking.confidence_decay_factor", 0.95
        )
        self.spatial_weight_center_bonus = get_setting(
            "player_id.tracking.spatial_weight_center_bonus", 0.3
        )
        self.min_confidence_threshold = get_setting(
            "player_id.tracking.min_confidence_threshold", 0.1
        )
        self.center_region_width = get_setting(
            "player_id.tracking.center_region_width", 0.4
        )  # 40% of width is "center"
        self.measurement_weight_recent = get_setting(
            "player_id.tracking.measurement_weight_recent", 1.0
        )
        self.measurement_weight_old = get_setting("player_id.tracking.measurement_weight_old", 0.5)
        self.probability_smoothing = get_setting("player_id.tracking.probability_smoothing", 0.1)

    def _calculate_spatial_weight(self, bbox_center_x: float) -> float:
        """Calculate spatial weight based on digit position within bounding box.

        Digits closer to the center laterally get higher weight.

        Args:
            bbox_center_x: Normalized x position within bounding box (0-1)

        Returns:
            Spatial weight factor (0.5 to 1 + center_bonus)
        """
        # Calculate distance from center (0.5)
        distance_from_center = abs(bbox_center_x - 0.5)

        # Check if within center region
        if distance_from_center <= self.center_region_width / 2:
            # In center region - apply bonus
            center_factor = 1.0 - (distance_from_center / (self.center_region_width / 2))
            return 1.0 + (self.spatial_weight_center_bonus * center_factor)
        else:
            # Outside center - linear decay
            edge_distance = distance_from_center - (self.center_region_width / 2)
            max_edge_distance = 0.5 - (self.center_region_width / 2)
            decay_factor = 1.0 - (edge_distance / max_edge_distance) * 0.5
            return max(0.5, decay_factor)

    def add_measurement(
        self,
        track_id: int,
        jersey_number: str,
        confidence: float,
        bbox_center_x: float,
        ocr_results: List[Any] = None,
    ) -> None:
        """Add a new jersey number measurement for a track.

        Args:
            track_id: Unique identifier for the tracked object
            jersey_number: Detected jersey number string
            confidence: EasyOCR confidence score (0-1)
            bbox_center_x: Normalized x position of detection within bounding box (0-1)
            ocr_results: Raw OCR results for debugging
        """
        if not jersey_number or jersey_number == "Unknown":
            return

        # Calculate spatial weight
        spatial_weight = self._calculate_spatial_weight(bbox_center_x)

        # Create measurement
        measurement = JerseyMeasurement(
            jersey_number=jersey_number,
            confidence=confidence,
            spatial_weight=spatial_weight,
            timestamp=time.time(),
            bbox_center_x=bbox_center_x,
            ocr_results=ocr_results or [],
        )

        # Add to history
        measurements = self._track_measurements[track_id]
        measurements.append(measurement)

        # Limit history length
        if len(measurements) > self.max_history_length:
            measurements.popleft()

        # Update probabilities
        self._update_probabilities(track_id)

    def _update_probabilities(self, track_id: int) -> None:
        """Update probability distribution for a track based on all measurements.

        Args:
            track_id: Track to update probabilities for
        """
        measurements = self._track_measurements[track_id]
        if not measurements:
            return

        current_time = time.time()
        probabilities = self._track_probabilities[track_id]

        # Calculate weighted scores for each jersey number
        jersey_scores = defaultdict(float)
        jersey_counts = defaultdict(int)

        for i, measurement in enumerate(measurements):
            # Time-based weight (recent measurements weighted higher)
            age_factor = i / max(1, len(measurements) - 1)  # 0 (oldest) to 1 (newest)
            time_weight = (
                self.measurement_weight_old
                + (self.measurement_weight_recent - self.measurement_weight_old) * age_factor
            )

            # Combined weight
            total_weight = measurement.confidence * measurement.spatial_weight * time_weight

            jersey_scores[measurement.jersey_number] += total_weight
            jersey_counts[measurement.jersey_number] += 1

        # Convert scores to probabilities
        total_score = sum(jersey_scores.values())
        if total_score > 0:
            for jersey_number, score in jersey_scores.items():
                probability = score / total_score

                # Update or create probability entry
                if jersey_number in probabilities:
                    # Smooth probability update
                    old_prob = probabilities[jersey_number].probability
                    new_prob = (
                        old_prob * (1 - self.probability_smoothing)
                        + probability * self.probability_smoothing
                    )
                    probabilities[jersey_number].probability = new_prob
                    probabilities[jersey_number].last_seen = current_time
                    probabilities[jersey_number].measurement_count = jersey_counts[jersey_number]
                else:
                    probabilities[jersey_number] = JerseyProbability(
                        jersey_number=jersey_number,
                        probability=probability,
                        last_seen=current_time,
                        measurement_count=jersey_counts[jersey_number],
                    )

        # Apply confidence decay to old entries
        for jersey_number, prob_entry in list(probabilities.items()):
            time_since_seen = current_time - prob_entry.last_seen
            if time_since_seen > 0:
                # Decay probability if not seen recently
                decay_frames = int(time_since_seen * 30)  # Assume 30 FPS for frame conversion
                decayed_prob = prob_entry.probability * (self.confidence_decay_factor**decay_frames)

                if decayed_prob < self.min_confidence_threshold:
                    # Remove very low confidence entries
                    del probabilities[jersey_number]
                else:
                    prob_entry.probability = decayed_prob

        # Renormalize probabilities
        total_prob = sum(p.probability for p in probabilities.values())
        if total_prob > 0:
            for prob_entry in probabilities.values():
                prob_entry.probability /= total_prob

    def get_top_probabilities(self, track_id: int, top_k: int = 3) -> List[Tuple[str, float, int]]:
        """Get the top K most probable jersey numbers for a track.

        Args:
            track_id: Track to get probabilities for
            top_k: Number of top predictions to return

        Returns:
            List of (jersey_number, probability, measurement_count) tuples, sorted by probability
        """
        probabilities = self._track_probabilities[track_id]
        if not probabilities:
            return []

        # Sort by probability (descending)
        sorted_probs = sorted(probabilities.values(), key=lambda x: x.probability, reverse=True)

        # Return top K
        result = []
        for prob_entry in sorted_probs[:top_k]:
            result.append(
                (prob_entry.jersey_number, prob_entry.probability, prob_entry.measurement_count)
            )

        return result

    def get_best_jersey_number(self, track_id: int) -> Tuple[Optional[str], float]:
        """Get the most probable jersey number for a track.

        Args:
            track_id: Track to get best prediction for

        Returns:
            Tuple of (jersey_number, probability). Returns (None, 0.0) if no predictions.
        """
        top_probs = self.get_top_probabilities(track_id, top_k=1)
        if top_probs:
            return top_probs[0][0], top_probs[0][1]
        return None, 0.0

    def get_measurement_count(self, track_id: int) -> int:
        """Get total number of measurements for a track.

        Args:
            track_id: Track to get measurement count for

        Returns:
            Total number of measurements
        """
        return len(self._track_measurements[track_id])

    def clear_track(self, track_id: int) -> None:
        """Clear all data for a specific track.

        Args:
            track_id: Track to clear
        """
        if track_id in self._track_measurements:
            del self._track_measurements[track_id]
        if track_id in self._track_probabilities:
            del self._track_probabilities[track_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics for debugging.

        Returns:
            Dictionary with tracking statistics
        """
        total_tracks = len(self._track_measurements)
        total_measurements = sum(
            len(measurements) for measurements in self._track_measurements.values()
        )
        total_probabilities = sum(len(probs) for probs in self._track_probabilities.values())

        return {
            "total_tracks": total_tracks,
            "total_measurements": total_measurements,
            "total_probabilities": total_probabilities,
            "avg_measurements_per_track": total_measurements / max(1, total_tracks),
            "avg_probabilities_per_track": total_probabilities / max(1, total_tracks),
        }


# Global tracker instance
_jersey_tracker: Optional[JerseyNumberTracker] = None


def get_jersey_tracker() -> JerseyNumberTracker:
    """Get the global jersey number tracker instance."""
    global _jersey_tracker
    if _jersey_tracker is None:
        _jersey_tracker = JerseyNumberTracker()
    return _jersey_tracker


def add_jersey_measurement(
    track_id: int,
    jersey_number: str,
    confidence: float,
    bbox_center_x: float,
    ocr_results: List[Any] = None,
) -> None:
    """Add a jersey number measurement to the global tracker.

    Args:
        track_id: Unique identifier for the tracked object
        jersey_number: Detected jersey number string
        confidence: EasyOCR confidence score (0-1)
        bbox_center_x: Normalized x position of detection within bounding box (0-1)
        ocr_results: Raw OCR results for debugging
    """
    tracker = get_jersey_tracker()
    tracker.add_measurement(track_id, jersey_number, confidence, bbox_center_x, ocr_results)


def get_jersey_probabilities(track_id: int, top_k: int = 3) -> List[Tuple[str, float, int]]:
    """Get top K jersey number probabilities for a track.

    Args:
        track_id: Track to get probabilities for
        top_k: Number of top predictions to return

    Returns:
        List of (jersey_number, probability, measurement_count) tuples
    """
    tracker = get_jersey_tracker()
    return tracker.get_top_probabilities(track_id, top_k)


def get_best_jersey_number(track_id: int) -> Tuple[Optional[str], float]:
    """Get the most probable jersey number for a track.

    Args:
        track_id: Track to get best prediction for

    Returns:
        Tuple of (jersey_number, probability)
    """
    tracker = get_jersey_tracker()
    return tracker.get_best_jersey_number(track_id)


def reset_jersey_tracker() -> None:
    """Reset the jersey tracker state and clear all tracking history.

    This should be called when the main tracker is reset or when switching videos.
    """
    global _jersey_tracker
    from ..utils.logger import get_logger
    logger = get_logger("JERSEY_TRACKER")
    
    logger.info("Resetting jersey tracking state")
    _jersey_tracker = None  # This will force recreation on next access
    logger.info("Jersey tracking state reset complete")
