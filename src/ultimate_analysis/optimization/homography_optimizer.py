"""Genetic algorithm for optimizing homography transformation parameters.

This module implements a genetic algorithm to automatically optimize the 8 parameters
of a homography matrix for improved field perspective transformation in Ultimate Analysis.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config.settings import get_setting
from ._homography_individual import HomographyIndividual
from ._homography_metrics import (
    evaluate_field_coverage,
    evaluate_field_proportions,
    evaluate_line_alignment,
    evaluate_line_orientation_balance,
    evaluate_line_visibility,
    evaluate_perspective_distortion,
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
        self.population.append(HomographyIndividual(initial_params, mutation_rate=0))

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
            import cv2  # local import to keep module light

            warped = cv2.warpPerspective(frame, h_matrix, (output_width, output_height))

            # Calculate fitness components
            alignment_score = evaluate_line_alignment(
                h_matrix, ransac_lines, confidences, (output_width, output_height)
            )
            coverage_score = evaluate_field_coverage(warped)
            visibility_score = evaluate_line_visibility(
                h_matrix, ransac_lines, (output_width, output_height)
            )
            proportion_score = evaluate_field_proportions(
                h_matrix, ransac_lines, (output_width, output_height)
            )
            distortion_score = evaluate_perspective_distortion(h_matrix)
            orientation_score = evaluate_line_orientation_balance(
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

    # orientation balance moved to _homography_metrics
