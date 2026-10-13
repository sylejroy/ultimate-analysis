"""The heat map of where a disc is likely is drawn as a glow where it is sure."""

import unittest

import numpy as np
from support import load_module


class DiscHeatTest(unittest.TestCase):
    def setUp(self):
        self.draw = load_module("rendering.tracks").draw_disc_heat
        self.frame = np.full((540, 960, 3), 90, dtype=np.uint8)

    def test_a_sure_spot_glows_and_the_rest_of_the_picture_is_left_alone(self):
        heat = np.full((270, 480), 0.004, dtype=np.float32)  # A little everywhere
        heat[100:103, 200:203] = 0.9
        self.draw(self.frame, heat)
        self.assertGreater(int(self.frame[202, 402].max()), 200)
        self.assertTrue((self.frame[:150] == 90).all())
        self.assertTrue((self.frame[:, 600:] == 90).all())

    def test_nothing_is_drawn_without_a_map_or_where_it_is_not_sure(self):
        self.draw(self.frame, None)
        self.draw(self.frame, np.full((270, 480), 0.01, dtype=np.float32))
        self.assertTrue((self.frame == 90).all())


if __name__ == "__main__":
    unittest.main()
