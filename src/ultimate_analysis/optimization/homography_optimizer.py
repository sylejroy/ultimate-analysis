"""Genetic algorithm for optimizing homography transformation parameters.

This module implements a genetic algorithm to automatically optimize the 8 parameters
of a homography matrix for improved field perspective transformation in Ultimate Analysis.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config.settings import get_setting


class HomographyIndividual:
    """Represents a single homography matrix solution in the genetic algorithm.
    
    This class encapsulates the 8 homography parameters and provides methods for
    mutation, matrix generation, and parameter management.
    """

    def __init__(
        self,
        params: Optional[Dict[str, float]] = None,
        mutation_rate: float = 0.2,
        mutation_amount: float = 0.1,
    ):
        """Initialize homography individual with given parameters or random values.

        Args:
            params: Dictionary of homography parameters or None for random initialization
            mutation_rate: Probability of parameter mutation (0.0-1.0)
            mutation_amount: Scale factor for mutation magnitude
        """
        self.param_names = ["H00", "H01", "H02", "H10", "H11", "H12", "H20", "H21"]
        self.params = {}
        self.fitness = 0.0
        self.mutation_rate = mutation_rate
        self.mutation_amount = mutation_amount

        # Parameter ranges for initialization and constraints
        self.param_ranges = {
            "H00": get_setting("homography.slider_range_main", [-100.0, 100.0]),
            "H01": get_setting("homography.slider_range_main", [-100.0, 100.0]),
            "H02": [-10000.0, 10000.0],
            "H10": get_setting("homography.slider_range_main", [-100.0, 100.0]),
            "H11": get_setting("homography.slider_range_main", [-100.0, 100.0]),
            "H12": [-10000.0, 10000.0],
            "H20": get_setting("homography.slider_range_perspective", [-0.2, 0.2]),
            "H21": get_setting("homography.slider_range_perspective", [-0.2, 0.2]),
        }

        # Initialize parameters
        if params is None:
            # Random initialization within specified ranges
            for name in self.param_names:
                range_min, range_max = self.param_ranges[name]
                self.params[name] = random.uniform(range_min, range_max)
        else:
            # Use provided parameters
            self.params = params.copy()

    def get_matrix(self) -> np.ndarray:
        """Convert parameter dictionary to homography matrix.

        Returns:
            3x3 homography matrix as numpy array
        """
        return np.array(
            [
                [self.params["H00"], self.params["H01"], self.params["H02"]],
                [self.params["H10"], self.params["H11"], self.params["H12"]],
                [self.params["H20"], self.params["H21"], 1.0],
            ],
            dtype=np.float32,
        )

    def mutate(self) -> None:
        """Apply random mutations to parameters based on mutation rate."""
        for name in self.param_names:
            if random.random() < self.mutation_rate:
                # Get range for this parameter
                range_min, range_max = self.param_ranges[name]
                range_size = range_max - range_min

                # Apply mutation proportional to parameter range
                delta = random.gauss(0, self.mutation_amount * range_size)
                self.params[name] += delta

                # Clamp to valid range
                self.params[name] = max(range_min, min(range_max, self.params[name]))

    def copy(self) -> "HomographyIndividual":
        """Create a deep copy of this individual.

        Returns:
            New HomographyIndividual with copied parameters
        """
        return HomographyIndividual(
            params=self.params.copy(),
            mutation_rate=self.mutation_rate,
            mutation_amount=self.mutation_amount,
        )


class HomographyOptimizer:
    """Genetic algorithm optimizer for homography parameters.
    
    This class manages a population of HomographyIndividual instances and
    evolves them over generations to find optimal homography transformations.
    """

    def __init__(
        self,
        initial_params: Dict[str, float],
        population_size: int = 20,
        elite_size: int = 2,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
    ):
        """Initialize the genetic optimizer.

        Args:
            initial_params: Starting homography parameters
            population_size: Number of individuals in population
            elite_size: Number of top individuals to preserve unchanged
            mutation_rate: Probability of parameter mutation
            crossover_rate: Probability of crossover between individuals
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
        self.best_fitness = 0.0
        self.best_individual = None
        self.population = []

        # Initialize population with variations of initial parameters
        # Add original unchanged as first individual
        self.population.append(
            HomographyIndividual(initial_params, mutation_rate=0)
        )

        # Add random variations for the rest of the population
        for _ in range(population_size - 1):
            individual = HomographyIndividual(
                initial_params,
                mutation_rate=1.0,  # Force mutation of all parameters
                mutation_amount=0.3,  # Larger initial variation
            )
            individual.mutate()  # Apply initial mutation
            self.population.append(individual)

    def calculate_fitness(
        self,
        individual: HomographyIndividual,
        frame: np.ndarray,
        ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
        confidences: List[float],
    ) -> float:
        """Calculate fitness score for an individual based on multiple criteria.

        Args:
            individual: HomographyIndividual to evaluate
            frame: Original video frame
            ransac_lines: List of (start_point, end_point) tuples for detected lines
            confidences: Confidence scores for each line

        Returns:
            Fitness score (higher is better)
        """
        try:
            # Get homography matrix and apply transformation
            h_matrix = individual.get_matrix()
            height, width = frame.shape[:2]

            # Calculate output canvas size with proper aspect ratio
            buffer_factor = get_setting("homography.buffer_factor", 1.8)
            aspect_ratio = get_setting("homography.output_aspect_ratio", 3.0)

            original_area = width * height
            target_area = int(original_area * buffer_factor)

            if aspect_ratio >= 1.0:
                output_width = int(np.sqrt(target_area / aspect_ratio))
                output_height = int(output_width * aspect_ratio)
            else:
                output_height = int(np.sqrt(target_area * aspect_ratio))
                output_width = int(output_height / aspect_ratio)

            # Apply transformation
            warped = cv2.warpPerspective(frame, h_matrix, (output_width, output_height))

            # Calculate fitness components
            alignment_score = self._evaluate_line_alignment(
                h_matrix, ransac_lines, confidences, (output_width, output_height)
            )
            coverage_score = self._evaluate_field_coverage(warped)
            visibility_score = self._evaluate_line_visibility(
                h_matrix, ransac_lines, (output_width, output_height)
            )
            proportion_score = self._evaluate_field_proportions(
                h_matrix, ransac_lines, (output_width, output_height)
            )
            distortion_score = self._evaluate_perspective_distortion(h_matrix)
            orientation_score = self._evaluate_line_orientation_balance(
                h_matrix, ransac_lines, (output_width, output_height)
            )

            # Get fitness weights from configuration
            weights = {
                "alignment": get_setting("optimization.ga_fitness_weights.alignment", 0.35),
                "coverage": get_setting("optimization.ga_fitness_weights.coverage", 0.2),
                "visibility": get_setting("optimization.ga_fitness_weights.visibility", 0.15),
                "proportion": get_setting("optimization.ga_fitness_weights.proportion", 0.1),
                "distortion": get_setting("optimization.ga_fitness_weights.distortion", 0.05),
                "orientation": get_setting("optimization.ga_fitness_weights.orientation", 0.15),
            }

            # Combined weighted score
            final_score = (
                weights["alignment"] * alignment_score
                + weights["coverage"] * coverage_score
                + weights["visibility"] * visibility_score
                + weights["proportion"] * proportion_score
                + weights["distortion"] * distortion_score
                + weights["orientation"] * orientation_score
            )

            # Update individual's fitness
            individual.fitness = final_score

            return final_score

        except Exception as e:
            print(f"[GA_OPTIMIZER] Error calculating fitness: {e}")
            individual.fitness = 0.0
            return 0.0

    def _evaluate_line_alignment(
        self,
        h_matrix: np.ndarray,
        ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
        confidences: List[float],
        output_size: Tuple[int, int],
    ) -> float:
        """Evaluate how well lines align to vertical/horizontal in transformed space.

        Args:
            h_matrix: Homography transformation matrix
            ransac_lines: List of detected field lines
            confidences: Line detection confidence scores
            output_size: (width, height) of output canvas

        Returns:
            Alignment score (0.0 to 1.0)
        """
        if not ransac_lines:
            return 0.0

        output_width, output_height = output_size
        vertical_tolerance = get_setting("optimization.ga_line_alignment.vertical_tolerance", 5.0)
        horizontal_tolerance = get_setting(
            "optimization.ga_line_alignment.horizontal_tolerance", 5.0
        )
        min_line_length = get_setting("optimization.ga_line_alignment.min_line_length", 10)

        total_score = 0.0
        total_weight = 0.0

        for i, (start_point, end_point) in enumerate(ransac_lines):
            if i >= len(confidences):
                continue

            confidence = confidences[i]

            # Transform line points to warped space
            start_homo = np.array([start_point[0], start_point[1], 1.0])
            end_homo = np.array([end_point[0], end_point[1], 1.0])

            # Apply transformation
            start_transformed = h_matrix @ start_homo
            end_transformed = h_matrix @ end_homo

            # Check for valid transformation
            if start_transformed[2] != 0 and end_transformed[2] != 0:
                start_2d = start_transformed[:2] / start_transformed[2]
                end_2d = end_transformed[:2] / end_transformed[2]

                # Calculate line angle in degrees
                dx = end_2d[0] - start_2d[0]
                dy = end_2d[1] - start_2d[1]

                # Skip very short lines
                length = np.sqrt(dx * dx + dy * dy)
                if length < min_line_length:
                    continue

                angle = np.degrees(np.arctan2(dy, dx))

                # Check if line is within warped frame bounds
                if (
                    0 <= start_2d[0] < output_width
                    and 0 <= start_2d[1] < output_height
                    and 0 <= end_2d[0] < output_width
                    and 0 <= end_2d[1] < output_height
                ):

                    # Calculate alignment score
                    # Check vertical alignment (90° or 270°)
                    vertical_deviation = min(abs(90 - abs(angle)), abs(270 - abs(angle)))
                    # Check horizontal alignment (0° or 180°)
                    horizontal_deviation = min(abs(angle), abs(180 - abs(angle)))

                    # Choose the better alignment
                    if vertical_deviation <= horizontal_deviation:
                        # This line is more vertical
                        alignment_score = max(0.0, 1.0 - vertical_deviation / vertical_tolerance)
                    else:
                        # This line is more horizontal
                        alignment_score = max(0.0, 1.0 - horizontal_deviation / horizontal_tolerance)

                    # Weight by confidence and line length
                    weight = confidence * (length / 100.0)  # Normalize length
                    total_score += alignment_score * weight
                    total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _evaluate_field_coverage(self, warped_frame: np.ndarray) -> float:
        """Evaluate how much of the original field content is visible (rewards showing more original content).

        Args:
            warped_frame: Transformed frame

        Returns:
            Coverage score (0.0 to 1.0), higher when more original content is visible
        """
        black_threshold = get_setting("optimization.ga_field_coverage.black_pixel_threshold", 10)
        optimal_coverage = get_setting("optimization.ga_field_coverage.optimal_coverage_ratio", 0.75)

        # Convert to grayscale and count non-black pixels
        gray_warped = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        non_black_pixels = np.count_nonzero(gray_warped > black_threshold)
        total_pixels = gray_warped.shape[0] * gray_warped.shape[1]

        current_coverage = non_black_pixels / total_pixels
        
        # Reward coverage close to optimal ratio (not 100% which means zoomed in too much)
        # This encourages showing maximum original field while maintaining some borders
        if current_coverage <= optimal_coverage:
            # Reward higher coverage up to optimal point
            coverage_score = current_coverage / optimal_coverage
        else:
            # Penalize over-coverage (too zoomed in, likely cropping original field)
            excess = current_coverage - optimal_coverage
            max_excess = 1.0 - optimal_coverage
            penalty = excess / max_excess if max_excess > 0 else 0
            coverage_score = 1.0 - (penalty * 0.5)  # Gentle penalty for over-coverage
        
        return max(0.0, min(1.0, coverage_score))

    def _evaluate_line_visibility(
        self,
        h_matrix: np.ndarray,
        ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
        output_size: Tuple[int, int],
    ) -> float:
        """Evaluate how many field lines remain visible in transformed space.

        Args:
            h_matrix: Homography transformation matrix
            ransac_lines: List of detected field lines
            output_size: (width, height) of output canvas

        Returns:
            Visibility score (0.0 to 1.0)
        """
        if not ransac_lines:
            return 0.0

        output_width, output_height = output_size
        visible_lines = 0

        for start_point, end_point in ransac_lines:
            # Transform line points to warped space
            start_homo = np.array([start_point[0], start_point[1], 1.0])
            end_homo = np.array([end_point[0], end_point[1], 1.0])

            # Apply transformation
            start_transformed = h_matrix @ start_homo
            end_transformed = h_matrix @ end_homo

            # Check for valid transformation
            if start_transformed[2] != 0 and end_transformed[2] != 0:
                start_2d = start_transformed[:2] / start_transformed[2]
                end_2d = end_transformed[:2] / end_transformed[2]

                # Check if any part of the line is within bounds
                if (
                    (0 <= start_2d[0] < output_width and 0 <= start_2d[1] < output_height)
                    or (0 <= end_2d[0] < output_width and 0 <= end_2d[1] < output_height)
                ):
                    visible_lines += 1

        return visible_lines / len(ransac_lines)

    def _evaluate_field_proportions(
        self,
        h_matrix: np.ndarray,
        ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
        output_size: Tuple[int, int],
    ) -> float:
        """Evaluate if the field maintains realistic proportions.

        Args:
            h_matrix: Homography transformation matrix
            ransac_lines: List of detected field lines
            output_size: (width, height) of output canvas

        Returns:
            Proportion score (0.0 to 1.0)
        """
        # For now, return a neutral score
        # In a more sophisticated implementation, we would:
        # 1. Identify parallel sidelines and goal lines
        # 2. Calculate the field dimensions in the transformed space
        # 3. Compare to expected Ultimate field ratio (≈2.7:1)
        expected_ratio = get_setting("optimization.ga_field_proportion.expected_ratio", 2.7)
        ratio_tolerance = get_setting("optimization.ga_field_proportion.ratio_tolerance", 0.5)

        # Simple approximation: use the output canvas ratio
        output_width, output_height = output_size
        actual_ratio = output_height / output_width  # height:width ratio

        ratio_deviation = abs(actual_ratio - expected_ratio)
        proportion_score = max(0.0, 1.0 - ratio_deviation / ratio_tolerance)

        return proportion_score

    def _evaluate_perspective_distortion(self, h_matrix: np.ndarray) -> float:
        """Evaluate perspective distortion by checking matrix properties.

        Args:
            h_matrix: Homography transformation matrix

        Returns:
            Distortion score (0.0 to 1.0, higher is better)
        """
        try:
            # Check if matrix is invertible
            det = np.linalg.det(h_matrix)
            if abs(det) < 1e-10:
                return 0.0  # Nearly singular matrix

            # Check for extreme perspective distortion
            # Perspective parameters should be reasonable
            h20, h21 = h_matrix[2, 0], h_matrix[2, 1]
            perspective_magnitude = np.sqrt(h20 * h20 + h21 * h21)

            # Penalize extreme perspective values
            max_perspective = get_setting("homography.slider_range_perspective", [-0.2, 0.2])[1]
            if perspective_magnitude > abs(max_perspective):
                return 0.0

            # Check for reasonable scale (avoid extreme scaling)
            scale_x = np.sqrt(h_matrix[0, 0] ** 2 + h_matrix[0, 1] ** 2)
            scale_y = np.sqrt(h_matrix[1, 0] ** 2 + h_matrix[1, 1] ** 2)

            if scale_x < 0.1 or scale_x > 10.0 or scale_y < 0.1 or scale_y > 10.0:
                return 0.5  # Extreme scaling

            return 1.0  # Good transformation

        except Exception:
            return 0.0

    def evaluate_population(
        self,
        frame: np.ndarray,
        ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
        confidences: List[float],
    ) -> None:
        """Calculate fitness for entire population.

        Args:
            frame: Original video frame
            ransac_lines: List of detected lines
            confidences: Line detection confidence scores
        """
        for individual in self.population:
            self.calculate_fitness(individual, frame, ransac_lines, confidences)

        # Sort population by fitness (descending)
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        # Update best individual if found better solution
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_individual = self.population[0].copy()

    def select_parent(self) -> HomographyIndividual:
        """Select parent using tournament selection.

        Returns:
            Selected individual for reproduction
        """
        # Tournament selection (select best from random sample)
        tournament_size = 3
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda ind: ind.fitness)

    def crossover(
        self, parent1: HomographyIndividual, parent2: HomographyIndividual
    ) -> HomographyIndividual:
        """Create child by crossing over two parents.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            New individual created from parents
        """
        if random.random() > self.crossover_rate:
            # No crossover, just clone parent1
            return parent1.copy()

        # Uniform crossover (randomly select from either parent for each parameter)
        child_params = {}
        for name in parent1.param_names:
            if random.random() < 0.5:
                child_params[name] = parent1.params[name]
            else:
                child_params[name] = parent2.params[name]

        child = HomographyIndividual(child_params, mutation_rate=self.mutation_rate)
        return child

    def evolve(self) -> None:
        """Evolve population to next generation."""
        # Increment generation counter
        self.generation += 1

        # Keep elites unchanged
        new_population = self.population[: self.elite_size]

        # Generate rest of population through selection, crossover and mutation
        for _ in range(self.population_size - self.elite_size):
            parent1 = self.select_parent()
            parent2 = self.select_parent()

            # Create child through crossover
            child = self.crossover(parent1, parent2)

            # Apply mutation
            child.mutate()

            # Add to new population
            new_population.append(child)

        # Replace old population
        self.population = new_population

    def get_best_parameters(self) -> Dict[str, float]:
        """Get parameters of the best individual.

        Returns:
            Dictionary of best homography parameters found
        """
        if self.best_individual is not None:
            return self.best_individual.params.copy()
        elif self.population:
            return self.population[0].params.copy()
        else:
            return {}  # Empty if no population exists

    def get_population_stats(self) -> Dict[str, float]:
        """Get statistics about the current population.

        Returns:
            Dictionary with population statistics
        """
        if not self.population:
            return {}

        fitnesses = [ind.fitness for ind in self.population]
        return {
            "best_fitness": max(fitnesses),
            "worst_fitness": min(fitnesses),
            "average_fitness": np.mean(fitnesses),
            "fitness_std": np.std(fitnesses),
            "generation": self.generation,
        }

    def _evaluate_line_orientation_balance(
        self,
        h_matrix: np.ndarray,
        ransac_lines: List[Tuple[np.ndarray, np.ndarray]],
        output_size: Tuple[int, int],
    ) -> float:
        """Evaluate the balance of vertical and horizontal lines in the transformed view.

        Rewards having both vertical and horizontal lines, punishes if either is missing.
        Awards bonus points for having more lines of each orientation.

        Args:
            h_matrix: Homography transformation matrix
            ransac_lines: List of detected field lines
            output_size: (width, height) of output canvas

        Returns:
            Line orientation balance score (0.0 to 1.0)
        """
        if not ransac_lines:
            return 0.0

        output_width, output_height = output_size
        vertical_tolerance = get_setting("optimization.ga_line_orientation.vertical_tolerance", 15.0)
        horizontal_tolerance = get_setting("optimization.ga_line_orientation.horizontal_tolerance", 15.0)
        min_line_length = get_setting("optimization.ga_line_orientation.min_line_length", 20)
        
        vertical_lines = 0
        horizontal_lines = 0

        for start_point, end_point in ransac_lines:
            # Transform line points to warped space
            start_homo = np.array([start_point[0], start_point[1], 1.0])
            end_homo = np.array([end_point[0], end_point[1], 1.0])

            # Apply transformation
            start_transformed = h_matrix @ start_homo
            end_transformed = h_matrix @ end_homo

            # Check for valid transformation
            if start_transformed[2] != 0 and end_transformed[2] != 0:
                start_2d = start_transformed[:2] / start_transformed[2]
                end_2d = end_transformed[:2] / end_transformed[2]

                # Check if line is within bounds and long enough
                x1, y1 = start_2d
                x2, y2 = end_2d
                
                # Skip lines that are completely outside bounds
                if (min(x1, x2) > output_width or max(x1, x2) < 0 or 
                    min(y1, y2) > output_height or max(y1, y2) < 0):
                    continue
                
                # Calculate line length and angle
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if line_length < min_line_length:
                    continue

                # Calculate angle in degrees (0° = horizontal, 90° = vertical)
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                
                # Normalize angle to 0-90 degrees
                if angle > 90:
                    angle = 180 - angle

                # Classify as vertical or horizontal
                if angle >= (90 - vertical_tolerance):
                    vertical_lines += 1
                elif angle <= horizontal_tolerance:
                    horizontal_lines += 1

        # Calculate base score
        base_score = 0.0
        
        # Penalty for missing either orientation
        missing_penalty = get_setting("optimization.ga_line_orientation.missing_orientation_penalty", 0.5)
        if vertical_lines == 0 or horizontal_lines == 0:
            base_score -= missing_penalty
        
        # Reward for having both orientations
        if vertical_lines > 0 and horizontal_lines > 0:
            base_score += 0.6  # Base reward for having both
            
            # Bonus for more lines (diminishing returns)
            vertical_bonus = min(vertical_lines * 0.1, 0.2)  # Max 0.2 bonus for verticals
            horizontal_bonus = min(horizontal_lines * 0.1, 0.2)  # Max 0.2 bonus for horizontals
            base_score += vertical_bonus + horizontal_bonus

        # Ensure score is in valid range
        return max(0.0, min(1.0, base_score))