"""Homography parameters, files, and the top-down canvas."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from support import load_module


class HomographyTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.homography")

    def test_parameters_round_trip_through_a_file(self):
        parameters = dict(self.module.IDENTITY_PARAMETERS, H02=-120.5, H21=0.0148)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "homography.yaml"
            self.module.save_parameters(path, parameters, video_file="game.mp4", frame_index=20)
            self.assertEqual(self.module.load_parameters(path), parameters)

            # Files with the bare parameters are accepted as well
            path.write_text("H00: 2\nH11: 3\n", encoding="utf-8")
            self.assertEqual(self.module.load_parameters(path), {"H00": 2.0, "H11": 3.0})

    def test_matrix_layout_and_fixed_corner(self):
        parameters = {name: float(i) for i, name in enumerate(self.module.PARAMETER_NAMES)}
        matrix = self.module.parameters_to_matrix(parameters)
        np.testing.assert_array_equal(matrix, [[0, 1, 2], [3, 4, 5], [6, 7, 1]])

    def test_canvas_keeps_the_configured_area_and_aspect_ratio(self):
        settings = {"homography.buffer_factor": 1.8, "homography.output_aspect_ratio": 3.0}
        with patch.object(
            self.module,
            "get_setting",
            side_effect=lambda key, default=None: settings.get(key, default),
        ):
            width, height = self.module.output_canvas_size(1920, 1080)
        self.assertEqual((width, height), (1115, 3345))

    def test_slider_ranges_come_from_the_settings_per_parameter_group(self):
        settings = {"homography.slider_range_main": [-100, 100]}
        with patch.object(
            self.module,
            "get_setting",
            side_effect=lambda key, default=None: settings.get(key, default),
        ):
            self.assertEqual(self.module.parameter_range("H00"), (-100.0, 100.0))
            self.assertEqual(self.module.parameter_range("H20"), (-0.2, 0.2))
            self.assertEqual(self.module.parameter_range("H02"), (-10000.0, 10000.0))


if __name__ == "__main__":
    unittest.main()
