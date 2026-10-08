"""A flying disc is placed over the field at the height it is thrown and caught at."""

import unittest

import numpy as np
from support import load_module

CAMERA = (20.0, -10.0, 9.0)


def seen_over(place, height):
    """The spot on the ground a disc at a height over a place is seen over."""
    camera, place = np.array(CAMERA[:2]), np.asarray(place, float)
    return camera + (place - camera) / (1.0 - height / CAMERA[2])


class DiscFlightTest(unittest.TestCase):
    def setUp(self):
        self.place_under_disc = load_module("processing.disc_flight").place_under_disc

    def test_a_disc_at_catching_height_is_placed_where_it_is(self):
        place = (30.0, 60.0)
        over = seen_over(place, 1.0)
        self.assertGreater(np.linalg.norm(over - place), 8.0)  # Taken for ground: yards behind
        np.testing.assert_allclose(self.place_under_disc(over, CAMERA), place, atol=1e-9)

    def test_a_higher_disc_is_placed_nearer_than_the_ground_and_not_past_itself(self):
        place = np.array([30.0, 60.0])
        over = seen_over(place, 3.0)
        given = np.array(self.place_under_disc(over, CAMERA))
        self.assertLess(np.linalg.norm(given - place), np.linalg.norm(over - place))
        # Still on the far side: never nearer to the camera than the disc is
        self.assertGreater(np.linalg.norm(given - CAMERA[:2]), np.linalg.norm(place - CAMERA[:2]))

    def test_no_place_with_a_camera_that_cannot_be_right(self):
        self.assertIsNone(self.place_under_disc((30.0, 60.0), (20.0, -10.0, 1.5)))
        self.assertIsNone(self.place_under_disc((30.0, 60.0), (np.nan, -10.0, 9.0)))


if __name__ == "__main__":
    unittest.main()
