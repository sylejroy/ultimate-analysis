"""Field lines over time: moved with the camera, blended at a new fit."""

import unittest
from unittest.mock import patch

import numpy as np
from support import load_module


def line(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y2]], dtype=np.float64)


class FieldLineFilterTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.field_line_filter")
        patcher = patch.object(
            self.module, "get_setting", side_effect=lambda key, default=None: default
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.filter = self.module.FieldLineFilter()

    def shown(self):
        return self.filter.current()[0]

    def test_lines_move_with_the_camera_between_fits(self):
        self.filter.update([line(100, 200, 900, 200)], [0.9])
        pan = np.array([[1.0, 0, -20], [0, 1, 5], [0, 0, 1]])
        self.filter.move(pan)
        np.testing.assert_allclose(self.shown()[0], line(80, 205, 880, 205), atol=1e-3)

    def test_a_new_fit_is_blended_in(self):
        self.filter.update([line(100, 200, 900, 200)], [0.9])
        # The same line, found 10 pixels lower and with its ends the other way round
        self.filter.update([line(900, 210, 100, 210)], [0.8])
        np.testing.assert_allclose(self.shown()[0], line(100, 204, 900, 204), atol=1e-6)
        self.assertEqual(self.filter.current()[1], [0.8])

    def test_a_new_line_is_shown_once_two_fits_have_it(self):
        sideline, other = line(100, 200, 900, 200), line(300, 100, 320, 700)
        self.filter.update([sideline], [0.9])
        self.filter.update([sideline, other], [0.9, 0.5])
        self.assertEqual(len(self.shown()), 1)
        self.filter.update([sideline, other], [0.9, 0.5])
        self.assertEqual(len(self.shown()), 2)

    def test_a_line_one_fit_misses_is_kept_once(self):
        sideline, other = line(100, 200, 900, 200), line(300, 100, 320, 700)
        self.filter.update([sideline, other], [0.9, 0.5])
        self.assertEqual(len(self.shown()), 2)  # Nothing was shown: shown at once
        self.filter.update([sideline], [0.9])
        self.assertEqual(len(self.shown()), 2)
        self.filter.update([sideline], [0.9])
        self.assertEqual(len(self.shown()), 1)

    def test_reset_forgets_the_lines(self):
        self.filter.update([line(100, 200, 900, 200)], [0.9])
        self.filter.reset()
        self.assertEqual(self.filter.current(), ([], []))


if __name__ == "__main__":
    unittest.main()
