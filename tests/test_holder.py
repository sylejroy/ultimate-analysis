"""Whether a player holds the disc: the crops and the speeds the network is taught with."""

import unittest

import numpy as np
from support import load_module


class HolderHelpersTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.holder")

    def test_a_crop_is_wider_than_the_box_and_within_the_frame(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        crop, box = self.module.crop_around(frame, (10, 20, 50, 80), pad=0.25)
        self.assertEqual(box, (0, 5, 60, 95))
        self.assertEqual(crop.shape, (90, 60, 3))
        _, box = self.module.crop_around(frame, (150, 50, 200, 100), pad=0.25)
        self.assertEqual(box[2:], (200, 100))

    def test_a_disc_is_on_a_player_by_its_centre(self):
        box = (10, 20, 50, 80)
        self.assertTrue(self.module.disc_on(box, [(45, 15, 55, 25)]))
        self.assertFalse(self.module.disc_on(box, [(60, 15, 70, 25)]))
        self.assertFalse(self.module.disc_on(box, []))

    def test_speed_is_taken_over_half_a_second_where_the_field_is_known(self):
        places = {frame: {7: (10.0, 20.0 + 0.1 * frame)} for frame in range(0, 60, 2)}
        self.assertAlmostEqual(self.module.speed_of(places, 7, 50, 60.0), 6.0, places=5)
        self.assertIsNone(self.module.speed_of(places, 7, 2, 60.0))  # Nothing long enough before
        self.assertIsNone(self.module.speed_of(places, 8, 50, 60.0))
        self.assertIsNone(self.module.place_on_field(None, (0, 0, 10, 10)))
        self.assertEqual(self.module.place_on_field(np.eye(3), (0, 0, 10, 10)), (5.0, 10.0))


if __name__ == "__main__":
    unittest.main()
