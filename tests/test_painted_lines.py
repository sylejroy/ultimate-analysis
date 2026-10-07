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


if __name__ == "__main__":
    unittest.main()
