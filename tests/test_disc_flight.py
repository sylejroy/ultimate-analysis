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


DiscPath = load_module("processing.disc_flight").DiscPath


class DiscPathTest(unittest.TestCase):
    def test_held_disc_is_where_its_holder_stands(self):
        path = DiscPath()
        for step in range(5):
            path.add(step / 30, "held", holder=(10.0 + step, 20.0), since=0.0)
        (line,) = path.stretches()
        np.testing.assert_allclose(line[:, 0], [10, 11, 12, 13, 14])

    def test_flight_is_tied_to_thrower_and_catcher_once_it_is_caught(self):
        path = DiscPath()
        path.add(0.0, "held", holder=(10.0, 20.0), since=0.0)
        # Seen flying one yard to the side of where it flies all the way
        for step in range(1, 10):
            path.add(step / 10, "air", ground=(13.0, 20.0 + step), raised=(12.0, 20.0 + step))
        (line,) = path.stretches()
        np.testing.assert_allclose(line[1:, 0], 12.0)
        path.add(1.0, "held", holder=(10.0, 30.0), since=1.0)
        (line,) = path.stretches()
        np.testing.assert_allclose(line[:, 0], 10.0, atol=1e-6)
        np.testing.assert_allclose(line[:, 1], 20.0 + np.arange(11), atol=1e-6)

    def test_catch_confirmed_late_counts_from_when_it_was_made(self):
        path = DiscPath()
        for step in range(6):
            path.add(step / 10, "air", raised=(float(step), 0.0))
        # Confirmed at 0.6 s, made at 0.4 s: the disc has been with the catcher since
        path.add(0.6, "held", holder=(4.0, 1.0), since=0.4)
        (line,) = path.stretches()
        np.testing.assert_allclose(line[-3:], [[4.0, 1.0]] * 3)

    def test_disc_come_to_rest_is_on_the_ground_since_it_stopped(self):
        path = DiscPath()
        for step in range(5):
            path.add(step / 10, "air", ground=(float(step), 0.0), raised=(float(step), 2.0))
        for step in range(5, 10):
            path.add(step / 10, "air", ground=(5.0, 0.0), raised=(5.0, 2.0))
        path.add(1.0, "ground", ground=(5.0, 0.0), raised=(5.0, 2.0), since=0.5)
        (line,) = path.stretches()
        np.testing.assert_allclose(line[-6:], [[5.0, 0.0]] * 6)
        # The flight before it ends on the ground too
        np.testing.assert_allclose(line[4], [4.0, 0.0], atol=1e-6)

    def test_thrower_far_from_the_flight_is_not_its_thrower(self):
        path = DiscPath()
        path.add(0.0, "held", holder=(30.0, 20.0), since=0.0)
        # Lost for a while, and then seen flying far from who had it
        for step in range(9, 13):
            path.add(step / 10, "air", raised=(12.0, 20.0 + step))
        path.add(1.3, "ground", ground=(12.0, 33.0), since=1.3)
        (line,) = path.stretches()
        np.testing.assert_allclose(line[1:5, 0], 12.0, atol=1e-6)

    def test_sighting_the_disc_cannot_have_flown_to_is_left_out(self):
        path = DiscPath()
        path.add(0.0, "held", holder=(10.0, 20.0), since=0.0)
        for step in range(1, 6):
            far = step == 3  # A cone at the other end of the field
            place = (60.0, 90.0) if far else (10.0, 20.0 + step)
            path.add(step / 10, "air", ground=place, raised=place)
        (line,) = path.stretches()
        self.assertEqual(len(line), 5)
        self.assertLess(line[:, 1].max(), 30.0)
        np.testing.assert_allclose(path.last_place, (10.0, 25.0))

    def test_carried_disc_does_not_bob_with_its_holder(self):
        path = DiscPath()
        for step in range(30):
            path.add(step / 30, "held", holder=(10.0, 20.0 + (step % 2)), since=0.0)
        (line,) = path.stretches()
        self.assertLess(np.ptp(line[3:-3, 1]), 0.2)

    def test_path_breaks_where_the_disc_was_lost_and_forgets_what_is_old(self):
        path = DiscPath(seconds=4.0)
        for step in range(3):
            path.add(step / 10, "air", raised=(float(step), 0.0))
        for step in range(30, 33):
            path.add(step / 10, "air", raised=(float(step), 0.0))
        self.assertEqual([len(line) for line in path.stretches()], [3, 3])
        for step in range(33, 60):
            path.add(step / 10, "air", raised=(float(step), 0.0))
        self.assertEqual(len(path.stretches()), 1)


class FlyingDiscTest(unittest.TestCase):
    """The disc is seen to fly on a clear, smooth course, found well, away from players."""

    def setUp(self):
        self.module = load_module("processing.disc_flight")
        self.flying = self.module.FlyingDisc()
        self.players = [(100, 100, 140, 200), (600, 100, 640, 200)]  # 100 pixels tall

    def disc(self, x, y, confidence=0.9):
        return [{"bbox": [x - 5, y - 5, x + 5, y + 5], "confidence": confidence}]

    def watch(self, places, confidence=0.9, rate=30.0):
        said = False
        for frame, (x, y) in enumerate(places):
            said = self.flying.update(frame / rate, self.disc(x, y, confidence), self.players)
        return said

    def test_a_disc_on_a_straight_course_between_two_players_flies(self):
        self.assertTrue(self.watch([(200 + 20 * step, 150) for step in range(5)]))

    def test_it_takes_three_sightings(self):
        self.assertFalse(self.watch([(200, 150), (220, 150)]))

    def test_a_disc_that_lies_still_or_jumps_about_does_not(self):
        self.assertFalse(self.watch([(300, 300)] * 6))
        self.flying.reset()
        self.assertFalse(self.watch([(300, 150), (340, 150), (300, 150), (340, 150)]))

    def test_a_disc_at_a_player_does_not(self):
        self.assertFalse(self.watch([(60 + 20 * step, 150) for step in range(4)]))

    def test_a_disc_found_unsurely_does_not(self):
        self.assertFalse(self.watch([(200 + 20 * step, 150) for step in range(5)], confidence=0.3))

    def test_a_course_ends_where_the_disc_is_not_found_for_a_while(self):
        self.assertTrue(self.watch([(200 + 20 * step, 150) for step in range(4)]))
        self.assertFalse(self.flying.update(1.0, self.disc(300, 150), self.players))

    def test_the_sightings_move_with_the_picture(self):
        # The camera pans by 20 pixels a frame against a disc that lies still
        shift = np.array([[1.0, 0.0, -20.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        said = False
        for frame in range(5):
            self.flying.move(shift)
            said = self.flying.update(frame / 30.0, self.disc(400 - 20 * frame, 300), self.players)
        self.assertFalse(said)


if __name__ == "__main__":
    unittest.main()
