"""Where the next throw goes: what the network is shown, and the field turned for it."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from support import load_module

targets = load_module("processing.throw_targets")


class Network:
    """Stands in for the network: gives a fixed map and keeps what it was shown."""

    def __init__(self):
        self.grid = np.zeros((targets.ROWS, targets.COLUMNS), dtype=np.float32)
        self.grid[50, 3] = 1.0  # Near the far back line, on the left
        self.shown = []

    def chances(self, moment):
        self.shown.append(moment)
        return self.grid


def frame(reader, seconds, players, holder=1, team=0, attacks=None):
    reader.update(seconds, players, holder, team, attacks or {})


class ThrowReaderTest(unittest.TestCase):
    def setUp(self):
        self.reader = targets.ThrowReader()
        self.network = Network()

    def test_nothing_is_said_while_nobody_holds_the_disc(self):
        frame(self.reader, 0.0, [(1, 0, 20.0, 50.0)], holder=None, attacks={0: 1})
        self.assertIsNone(self.reader.chances(self.network))

    def test_nothing_is_said_while_the_end_attacked_is_not_known(self):
        frame(self.reader, 0.0, [(1, 0, 20.0, 50.0)])
        self.assertIsNone(self.reader.chances(self.network))

    def test_the_line_up_says_which_end_is_attacked(self):
        frame(self.reader, 0.0, [(1, 0, 20.0, 50.0), (2, 0, 25.0, 60.0), (3, 1, 21.0, 51.0)])
        frame(
            self.reader,
            0.5,
            [(1, 0, 20.0, 50.0), (2, 0, 25.0, 65.0), (3, 1, 21.0, 51.0)],
            attacks={0: 1},
        )
        np.testing.assert_array_equal(self.reader.chances(self.network), self.network.grid)
        moment = self.network.shown[-1]
        self.assertEqual(moment.thrower, (20.0, 50.0))
        # The thrower is not among the offence; a runner's speed in yards a second
        np.testing.assert_allclose(moment.offence, [[25.0, 65.0, 0.0, 10.0]])
        np.testing.assert_allclose(moment.defence, [[21.0, 51.0, 0.0, 0.0]])

    def test_a_team_attacking_the_near_end_is_turned_and_its_map_turned_back(self):
        frame(self.reader, 0.0, [(1, 0, 10.0, 30.0), (2, 0, 15.0, 20.0)], attacks={0: -1})
        grid = self.reader.chances(self.network)
        moment = self.network.shown[-1]
        self.assertEqual(moment.thrower, (30.0, 80.0))
        np.testing.assert_allclose(moment.offence, [[25.0, 90.0, 0.0, 0.0]])
        # The likeliest square is near the near back line, on the right, as the field lies
        self.assertEqual(np.unravel_index(grid.argmax(), grid.shape), (4, 16))

    def test_without_a_line_up_the_way_the_throws_went_says_it(self):
        frame(self.reader, 0.0, [(1, 0, 20.0, 50.0), (2, 0, 20.0, 40.0)], holder=1)
        frame(self.reader, 1.0, [(1, 0, 20.0, 50.0), (2, 0, 20.0, 40.0)], holder=2)
        self.reader.chances(self.network)
        # Back from 50 to 40: towards the near end
        self.assertEqual(self.network.shown[-1].thrower, (20.0, 70.0))

    def test_a_reset_forgets_where_the_throws_went(self):
        frame(self.reader, 0.0, [(1, 0, 20.0, 50.0), (2, 0, 20.0, 40.0)], holder=1)
        frame(self.reader, 1.0, [(1, 0, 20.0, 50.0), (2, 0, 20.0, 40.0)], holder=2)
        self.reader.reset()
        frame(self.reader, 2.0, [(1, 0, 20.0, 50.0), (2, 0, 20.0, 40.0)], holder=2)
        self.assertIsNone(self.reader.chances(self.network))


class NetworkFileTest(unittest.TestCase):
    def test_a_saved_network_gives_chances_over_the_whole_field(self):
        names = targets.PLACE_FEATURES + targets.PLAYER_FEATURES
        network = targets.square_scorer(len(names))
        path = Path(tempfile.mkdtemp()) / "model.pt"
        torch.save(
            {
                "state": network.state_dict(),
                "names": names,
                "mean": [0.0] * len(names),
                "std": [1.0] * len(names),
                "epoch": 1,
            },
            path,
        )
        loaded = targets.load_throw_targets(path)
        moment = targets.Moment(
            (20.0, 50.0),
            np.array([[25.0, 60.0, 0.0, 5.0]], dtype=np.float32),
            np.array([[21.0, 51.0, 0.0, 0.0], [26.0, 61.0, 0.0, 4.0]], dtype=np.float32),
        )
        chances = loaded.chances(moment)
        self.assertEqual(chances.shape, (targets.ROWS, targets.COLUMNS))
        self.assertAlmostEqual(float(chances.sum()), 1.0, places=4)

    def test_no_file_is_no_network(self):
        self.assertIsNone(targets.load_throw_targets(""))
        self.assertIsNone(targets.load_throw_targets("no/such/model.pt"))


class FeaturesTest(unittest.TestCase):
    def test_the_mark_makes_the_squares_behind_them_harder(self):
        moment = targets.Moment(
            (20.0, 50.0),
            np.zeros((0, 4), dtype=np.float32),
            np.array([[22.0, 50.0, 0.0, 0.0]], dtype=np.float32),  # Two yards to the right
        )
        towards = targets.features(moment, ["towards_mark"])[:, 0]
        right = targets.square_of((35.0, 50.0))
        left = targets.square_of((5.0, 50.0))
        self.assertGreater(towards[right], 0.9)
        self.assertLess(towards[left], -0.9)


if __name__ == "__main__":
    unittest.main()
