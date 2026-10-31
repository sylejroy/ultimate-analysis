"""The field mask cleaned up: holes filled."""

import unittest

import numpy as np
from support import load_module


class HoleFillingTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.field_analysis")

    def test_a_hole_inside_the_field_is_filled(self):
        mask = np.ones((9, 9), dtype=np.uint8)
        mask[3:6, 3:6] = 0
        filled = self.module._fill_holes_flood_fill_optimized(mask)
        np.testing.assert_array_equal(filled, np.ones((9, 9), dtype=np.uint8))

    def test_a_notch_open_to_the_edge_is_kept(self):
        mask = np.ones((9, 9), dtype=np.uint8)
        mask[0:4, 3:6] = 0  # Background from the top edge inwards
        mask[6:8, 6:8] = 0  # ... and a hole
        filled = self.module._fill_holes_flood_fill_optimized(mask)
        expected = np.ones((9, 9), dtype=np.uint8)
        expected[0:4, 3:6] = 0
        np.testing.assert_array_equal(filled, expected)


if __name__ == "__main__":
    unittest.main()
