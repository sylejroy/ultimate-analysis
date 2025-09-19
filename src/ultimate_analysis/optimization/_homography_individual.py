"""Homography individual representation for the genetic optimizer.

Moved out of `homography_optimizer.py` to keep files within size limits.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np

from ..config.settings import get_setting


class HomographyIndividual:
    """Represents a single homography matrix solution for GA."""

    def __init__(
        self,
        params: Optional[Dict[str, float]] = None,
        mutation_rate: float = 0.2,
        mutation_amount: float = 0.1,
    ):
        self.param_names = ["H00", "H01", "H02", "H10", "H11", "H12", "H20", "H21"]
        self.params: Dict[str, float] = {}
        self.fitness = 0.0
        self.mutation_rate = mutation_rate
        self.mutation_amount = mutation_amount

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

        if params is None:
            for name in self.param_names:
                range_min, range_max = self.param_ranges[name]
                self.params[name] = random.uniform(range_min, range_max)
        else:
            self.params = params.copy()

    def get_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.params["H00"], self.params["H01"], self.params["H02"]],
                [self.params["H10"], self.params["H11"], self.params["H12"]],
                [self.params["H20"], self.params["H21"], 1.0],
            ],
            dtype=np.float32,
        )

    def mutate(self) -> None:
        for name in self.param_names:
            if random.random() < self.mutation_rate:
                range_min, range_max = self.param_ranges[name]
                range_size = range_max - range_min
                delta = random.gauss(0, self.mutation_amount * range_size)
                self.params[name] = max(range_min, min(range_max, self.params[name] + delta))

    def copy(self) -> "HomographyIndividual":
        return HomographyIndividual(
            params=self.params.copy(),
            mutation_rate=self.mutation_rate,
            mutation_amount=self.mutation_amount,
        )
