"""The dangerous space: who gets where first, the mark, one throw ahead, and who gives it up."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from support import load_module

space = load_module("processing.open_space")
targets = load_module("processing.throw_targets")

# A thrower trapped on the right sideline at midfield, the mark inside them; a dump behind
# and inside, cutters on the line side covered, and one alone on the far side
THROWER = (37.0, 50.0)
OFFENCE = [[30.0, 45.0, 0, 0], [36.0, 65.0, 0, 1], [36.0, 75.0, 0, 1], [34.0, 85.0, 0, 0]]
OFFENCE += [[30.0, 70.0, 0, 0], [10.0, 58.0, 0, 1]]
DEFENCE = [[35.5, 51.0, 0, 0], [31.5, 46.5, 0, 0], [36.0, 66.5, 0, 1], [36.0, 76.5, 0, 1]]
DEFENCE += [[34.0, 86.5, 0, 0], [30.0, 72.0, 0, 0]]
DUMP, FAR_SIDE = (30.0, 45.0), (10.0, 58.0)


def moment(far_side_defender):
    return targets.Moment(
        THROWER,
        np.array(OFFENCE, dtype=np.float32),
        np.array(DEFENCE + [far_side_defender], dtype=np.float32),
    )


class DangerousSpaceTest(unittest.TestCase):
    def setUp(self):
        self.value = space.FieldValue()
        self.open = space.dangerous_space(moment([20.0, 70.0, 0, 0]), self.value)
        self.covered = space.dangerous_space(moment([11.5, 59.5, 0, 1]), self.value)

    def test_a_player_alone_is_open_and_one_covered_is_not(self):
        far = targets.square_of(FAR_SIDE)
        self.assertGreater(self.open.open[far], 2 * self.covered.open[far])

    def test_a_dump_is_dangerous_when_it_opens_the_far_side(self):
        dump = targets.square_of(DUMP)
        # It goes backwards, but from there the far side is open and unmarked
        self.assertLess(self.open.worth[dump], self.value(np.array([THROWER[1]]))[0])
        self.assertGreater(self.open.options[dump], self.covered.options[dump] + 0.05)
        self.assertGreater(self.open.danger[dump], 1.5 * self.covered.danger[dump])

    def test_the_mark_makes_their_side_harder(self):
        thrower = np.array([20.0, 50.0], dtype=np.float32)
        offence = np.array([[30.0, 50.0, 0, 0], [10.0, 50.0, 0, 0]], dtype=np.float32)
        mark = np.array([[22.0, 50.0, 0, 0]], dtype=np.float32)  # On the right
        squares = space.SQUARES[[targets.square_of((30.0, 50.0)), targets.square_of((10.0, 50.0))]]
        right, left = space.open_to(thrower, offence, mark, squares)
        self.assertLess(right, 0.75 * left)

    def test_each_defender_gives_up_the_danger_nearest_to_them(self):
        shares = self.open.given_up / self.open.given_up.sum()
        self.assertAlmostEqual(float(shares.sum()), 1.0, places=5)
        self.assertEqual(len(shares), len(DEFENCE) + 1)


class SpaceWatcherTest(unittest.TestCase):
    def test_the_map_is_turned_back_and_shares_are_by_track(self):
        watcher = space.SpaceWatcher()
        found = watcher.update(
            0.0, moment([20.0, 70.0, 0, 0]), -1, list(range(10, 17)), space.FieldValue()
        )
        self.assertEqual(found.danger.shape, (targets.ROWS, targets.COLUMNS))
        turned = space.dangerous_space(moment([20.0, 70.0, 0, 0]), space.FieldValue())
        np.testing.assert_allclose(
            found.danger, turned.danger.reshape(targets.ROWS, targets.COLUMNS)[::-1, ::-1]
        )
        self.assertEqual(sorted(found.given_up), list(range(10, 17)))
        self.assertIsNone(found.opening)

    def test_space_that_became_more_dangerous_is_opening(self):
        watcher = space.SpaceWatcher()
        value = space.FieldValue()
        watcher.update(0.0, moment([11.5, 59.5, 0, 1]), 1, list(range(7)), value)
        found = watcher.update(0.6, moment([20.0, 70.0, 0, 0]), 1, list(range(7)), value)
        far = np.unravel_index(targets.square_of(FAR_SIDE), (targets.ROWS, targets.COLUMNS))
        self.assertGreater(found.opening[far], 0.0)


class FieldValueTest(unittest.TestCase):
    def test_a_catch_in_the_end_zone_is_a_score_and_further_on_is_worth_more(self):
        value = space.FieldValue()
        near, far, end_zone = value(np.array([20.0, 80.0, 95.0]))
        self.assertLess(near, far)
        self.assertEqual(end_zone, 1.0)

    def test_a_fitted_curve_is_read_and_without_one_the_rough_one_is_used(self):
        path = Path(tempfile.mkdtemp()) / "value.json"
        path.write_text(json.dumps({"a": 1.0, "b": 0.0}))
        self.assertAlmostEqual(float(space.load_field_value(path)(np.array([50.0]))[0]), 0.731, 3)
        self.assertEqual(space.load_field_value("").a, space.ROUGH_VALUE[0])


if __name__ == "__main__":
    unittest.main()
