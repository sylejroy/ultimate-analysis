"""Coverage sampling must not change the coordinates of the geometry objectives."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np
from support import load_module


class HomographyOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("optimization.homography_optimizer")
        self.params = {
            "H00": 1,
            "H01": 0,
            "H02": 0,
            "H10": 0,
            "H11": 1,
            "H12": 0,
            "H20": 0,
            "H21": 0,
        }
        self.optimizer = self.module.HomographyOptimizer(self.params, population_size=2)
        self.frame = np.full((120, 160, 3), 180, dtype=np.uint8)

    def test_population_shares_one_grayscale_source(self):
        with patch.object(self.module.cv2, "warpPerspective", wraps=cv2.warpPerspective) as warp:
            self.optimizer.evaluate_population(self.frame, [], [])
        self.assertEqual(warp.call_count, 2)
        sources = [call.args[0] for call in warp.call_args_list]
        self.assertEqual(sources[0].ndim, 2)
        self.assertIs(sources[0], sources[1])

    def test_reduced_coverage_changes_only_the_raster_not_line_coordinates(self):
        original_setting = self.module.get_setting
        individual = self.optimizer.population[0]
        with (
            patch.object(
                self.module,
                "get_setting",
                side_effect=lambda key, default=None: 0.25
                if key == "optimization.ga_coverage_scale"
                else original_setting(key, default),
            ),
            patch.object(self.module.cv2, "warpPerspective", wraps=cv2.warpPerspective) as warp,
            patch.object(self.optimizer, "_evaluate_line_alignment", return_value=1) as alignment,
        ):
            score = self.optimizer.calculate_fitness(individual, self.frame, [], [])
        width, height = alignment.call_args.args[3]
        self.assertEqual(warp.call_args.args[2], (round(width * 0.25), round(height * 0.25)))
        np.testing.assert_array_equal(alignment.call_args.args[0], individual.get_matrix())
        np.testing.assert_allclose(
            warp.call_args.args[1][:2, :2],
            np.diag([round(width * 0.25) / width, round(height * 0.25) / height]),
        )
        self.assertTrue(np.isfinite(score))

    def test_full_resolution_grayscale_preserves_coverage_on_gray_input(self):
        gray = self.frame[..., 0].copy()
        gray[:, :40] = 0
        self.frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        matrix = np.array([[1, 0.1, 10], [0.2, 1, 0], [0.001, 0, 1]], dtype=np.float32)
        color_warp = cv2.warpPerspective(self.frame, matrix, (200, 200))
        gray_warp = cv2.warpPerspective(gray, matrix, (200, 200))
        self.assertAlmostEqual(
            self.optimizer._evaluate_field_coverage(color_warp),
            self.optimizer._evaluate_field_coverage(gray_warp),
        )


if __name__ == "__main__":
    unittest.main()
