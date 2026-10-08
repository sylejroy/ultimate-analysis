"""The painted lines of a field are found in a picture; grass and blobs are not."""

import unittest

import cv2
import numpy as np
from support import load_module


class PaintedLinesTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("utils.painted_lines")
        noise = np.random.default_rng(0).normal(0, 12, (480, 640, 1))
        grass = np.clip(np.array([60.0, 140.0, 70.0]) + noise, 0, 255)
        self.frame = grass.astype(np.uint8)

    def test_a_white_line_on_grass_is_found_and_the_grass_is_not(self):
        cv2.line(self.frame, (40, 400), (600, 330), (235, 235, 235), 4, cv2.LINE_AA)
        mask = self.module.painted_line_mask(self.frame)
        on_line = np.zeros(mask.shape, dtype=np.uint8)
        cv2.line(on_line, (40, 400), (600, 330), 255, 9)
        self.assertGreater((mask[on_line > 0] > 0).mean(), 0.2)  # Its middle, along most of it
        self.assertLess((mask[on_line == 0] > 0).mean(), 0.002)

    def test_a_white_blob_is_no_line(self):
        cv2.circle(self.frame, (320, 240), 30, (235, 235, 235), -1)
        self.assertEqual(int(self.module.painted_line_mask(self.frame).max()), 0)

    def test_a_point_snaps_so_that_its_line_lies_on_a_long_painted_line(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(mask, (40, 400), (600, 330), 255, 2)
        anchor = (120.0, 390.0)  # On the line
        # Put four pixels off the line: moved back onto it, the same way along
        moved = self.module.snap_onto_line(mask, anchor, (440.0, 354.0), reach=8.0)
        self.assertIsNotNone(moved)
        on_line_y = 400 + (moved[0] - 40) * (330 - 400) / (600 - 40)
        self.assertAlmostEqual(moved[1], on_line_y, delta=1.0)
        self.assertAlmostEqual(moved[0], 440.0, delta=2.0)
        # Too far from the line, and a line too short to go by: left alone
        self.assertIsNone(self.module.snap_onto_line(mask, anchor, (440.0, 380.0), reach=8.0))
        short = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(short, (120, 390), (200, 380), 255, 2)
        self.assertIsNone(self.module.snap_onto_line(short, anchor, (440.0, 354.0), reach=8.0))


if __name__ == "__main__":
    unittest.main()
