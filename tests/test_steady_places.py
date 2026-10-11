"""Players' places on the field are held steady where a pixel is worth yards."""

import unittest

import numpy as np
from support import load_module

RATE = 60.0


def camera_view():
    """A view along the field from behind the near end, some yards up: pixel -> field."""
    camera = load_module("utils.field_camera")
    to_image = camera.camera_mapping(
        1700.0, (np.radians(90.0 + 20.0), 0.0, 0.0), (20.0, -15.0, 10.0), (1920, 1080)
    )
    return to_image, np.linalg.inv(to_image)


def pixel_of(to_image, place):
    drawn = to_image @ [place[0], place[1], 1.0]
    return drawn[:2] / drawn[2]


class SteadyPlacesTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.steady_places")
        self.to_image, self.to_field = camera_view()
        self.rng = np.random.default_rng(3)

    def followed(self, place_at, frames=180, shake=1.5):
        """(places as the boxes give them, places as followed, true places) of a player
        whose box's lower edge shakes by some pixels."""
        steady = self.module.SteadyPlaces()
        raw, held, true = [], [], []
        for frame in range(frames):
            place = np.asarray(place_at(frame / RATE), dtype=float)
            foot = pixel_of(self.to_image, place) + self.rng.normal(0.0, [0.5, shake])
            seen = self.to_field @ [foot[0], foot[1], 1.0]
            raw.append(seen[:2] / seen[2])
            held.append(steady.update(frame / RATE, {1: foot}, self.to_field)[1])
            true.append(place)
        return np.array(raw), np.array(held), np.array(true)

    def test_a_player_standing_at_the_far_end_stands_still(self):
        raw, held, true = self.followed(lambda seconds: (20.0, 100.0))
        self.assertGreater(np.abs(np.diff(raw[:, 1])).mean(), 0.3)  # A pixel is yards there
        steps = np.abs(np.diff(held[60:, 1]))
        self.assertLess(steps.mean(), 0.2 * np.abs(np.diff(raw[60:, 1])).mean())
        self.assertLess(np.abs(held[60:] - true[60:]).mean(), np.abs(raw[60:] - true[60:]).mean())

    def test_a_player_running_at_the_far_end_is_not_left_behind(self):
        # Eight yards a second along the field
        raw, held, true = self.followed(lambda seconds: (20.0, 80.0 + 8.0 * seconds))
        behind = (true[60:, 1] - held[60:, 1]).mean()
        self.assertLess(abs(behind), 0.4)  # Under a twentieth of a second of running
        self.assertLess(
            np.abs(held[60:, 1] - true[60:, 1]).mean(), np.abs(raw[60:, 1] - true[60:, 1]).mean()
        )

    def test_near_the_camera_the_place_follows_the_box(self):
        raw, held, _ = self.followed(lambda seconds: (20.0, 12.0 + 6.0 * seconds), frames=90)
        self.assertLess(np.abs(held[30:] - raw[30:]).max(), 0.25)

    def test_somebody_else_under_the_same_number_is_taken_at_once(self):
        steady = self.module.SteadyPlaces()
        for frame in range(30):
            steady.update(frame / RATE, {1: pixel_of(self.to_image, (10.0, 90.0))}, self.to_field)
        place = steady.update(30 / RATE, {1: pixel_of(self.to_image, (30.0, 60.0))}, self.to_field)[
            1
        ]
        np.testing.assert_allclose(place, (30.0, 60.0), atol=1e-6)

    def test_a_player_keeps_their_place_under_another_number(self):
        steady = self.module.SteadyPlaces()
        foot = pixel_of(self.to_image, (10.0, 90.0))
        for frame in range(30):
            steady.update(frame / RATE, {1: foot}, self.to_field)
        steady.rename(1, 7)
        shaken = foot + [0.0, 2.0]
        seen = self.to_field @ [shaken[0], shaken[1], 1.0]
        place = steady.update(30 / RATE, {7: shaken}, self.to_field)[7]
        # Held near where the player was, not put where the shaken box says
        self.assertLess(abs(place[1] - 90.0), 0.5 * abs(seen[1] / seen[2] - 90.0))

    def test_nothing_is_said_or_kept_without_the_field(self):
        steady = self.module.SteadyPlaces()
        steady.update(0.0, {1: (900.0, 400.0)}, self.to_field)
        self.assertEqual(steady.update(1 / RATE, {1: (900.0, 400.0)}, None), {})
        self.assertEqual(steady._players, {})


if __name__ == "__main__":
    unittest.main()
