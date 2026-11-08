"""Teams that learn where to go and where to throw, from nothing (simulation/spots.py)."""

import unittest

import numpy as np
import torch
from support import load_module

game = load_module("simulation.game")
perception = load_module("simulation.perception")
spots = load_module("simulation.spots")
sf = load_module("simulation.spot_features")
scripted = load_module("simulation.scripted")
match = load_module("simulation.match")


def first_views(team=0):
    state = game.line_up(np.random.default_rng(0))  # Team 0 holds the disc
    still = game.Decision(np.zeros((game.PLAYERS, 2)))
    game.step(state, (still, still), np.random.default_rng(0))
    return state, perception.Senses(team, np.random.default_rng(0)).look(state)


class ChoicesTest(unittest.TestCase):
    def test_every_player_has_the_same_number_of_spots_to_choose_from(self):
        _, views = first_views()
        features, kinds, players, params = sf.spot_choices(sf.views_of(views))
        self.assertEqual(tuple(features.shape), (game.PLAYERS, sf.SPOT_CHOICES, sf.SPOT_FEATURES))
        self.assertEqual(tuple(kinds.shape), (game.PLAYERS, sf.SPOT_CHOICES))
        self.assertTrue(torch.isfinite(features).all())

    def test_a_spot_beside_a_player_follows_them(self):
        _, views = first_views()
        view = views[3]
        _, kinds, players, params = sf.spot_choices(sf.views_of([view]))
        index = int(torch.nonzero((kinds[0] == sf.OPPONENT) & (players[0] == 5))[0])
        beside = (sf.OPPONENT, 5, params[0, index].numpy())
        np.testing.assert_allclose(spots._spot_of(view, beside), view.theirs[5, :2] + beside[2])

    def test_a_disc_on_the_ground_is_described_from_each_spot(self):
        state, _ = first_views()
        state.holder, state.disc = None, np.array([20.0, 40.0])
        views = perception.Senses(0, np.random.default_rng(0)).look(state)
        v = sf.views_of(views)
        self.assertTrue(bool((v["disc_mode"] == sf.ON_GROUND).all()))
        features = sf.spot_choices(v)[0]
        on_ground = 2 + 2 + 1 + 4 + 1 + 2 * sf.NEAREST * sf.AROUND + 2  # Third of the disc's
        self.assertTrue(bool((features[..., on_ground] == 1.0).all()))

    def test_the_holder_may_hold_on_or_throw_anywhere(self):
        _, views = first_views()
        features = sf.throw_choices(sf.views_of([views[0]]))[0]
        self.assertEqual(tuple(features.shape), (sf.THROW_CHOICES, sf.THROW_FEATURES))
        self.assertEqual(float(features[0, 0]), 1.0)  # Holding on
        self.assertEqual(float(features[1:, 0].sum()), 0.0)
        # At first the holder holds on most of the time
        chances = np.exp(spots.THROW_PRIOR) / np.exp(spots.THROW_PRIOR).sum()
        self.assertAlmostEqual(float(chances[0]), spots.HOLD_AT_FIRST, places=5)


class SpotTeamTest(unittest.TestCase):
    def test_a_team_that_knows_nothing_plays_a_point_and_keeps_its_choices(self):
        rng = np.random.default_rng(1)
        ours = spots.SpotTeam(spots.SpotBrains(), 0, rng)
        start = game.pull_start(rng, 0)
        point = match.play_point((ours, scripted.ScriptedTeam(2)), rng, start, limit=20.0)
        self.assertGreater(point.seconds, 0.0)
        self.assertGreater(len(ours.choices), game.PLAYERS)
        sides = {c.side for c in ours.choices}
        self.assertTrue(sides <= {"offence", "defence", "throw"})
        for c in ours.choices:
            self.assertEqual(c.value_inputs.shape[0], load_module("simulation.ppo").CRITIC_INPUTS)


if __name__ == "__main__":
    unittest.main()
