"""Many points at once (simulation/batch.py), on the processor: the rules as game.py has them."""

import unittest

import numpy as np
import torch
from support import load_module

game = load_module("simulation.game")
batch = load_module("simulation.batch")
spots = load_module("simulation.spots")
sf = load_module("simulation.spot_features")


class BatchGameTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.game = batch.BatchGame(8, "cpu", seed=0)
        self.players = batch.BatchPlayers(self.game, spots.SpotBrains())

    def play(self, steps):
        self.game.begin_rollout(steps)
        with torch.no_grad():
            for _ in range(steps):
                self.game.step(*self.players.decide())
                self.players.forget(self.game.done[self.game.step_count - 1])

    def test_points_start_with_a_pull_from_the_far_goal_line(self):
        g = self.game
        self.assertTrue(bool(g.air.all()))
        self.assertTrue(bool(g.flight.pull.all()))
        for b in range(g.B):
            receiving = int(g.offence[b])
            thrower = int(g.flight.thrower[b])
            self.assertEqual(game.team_of(thrower), 1 - receiving)
            along = game.turn(g.places[b, thrower].numpy(), game.attack_of(receiving))[1]
            self.assertAlmostEqual(along, game.LENGTH - game.END_ZONE, places=4)

    def test_nobody_runs_faster_than_people_do_and_points_go_on(self):
        self.play(80)
        speeds = torch.linalg.norm(self.game.velocities, dim=-1)
        self.assertLessEqual(float(speeds.max()), game.RUN_SPEED + 1e-4)
        self.assertGreater(self.game.tally.get("throw", 0), 0)
        self.assertGreater(len(self.players.kept), 0)
        kept = self.players.kept[0]
        self.assertEqual(set(kept["view"]), set(sf.VIEW_FIELDS))

    def test_a_turnover_in_the_end_zone_the_other_team_attacks_goes_to_its_goal_line(self):
        g = self.game
        g.begin_rollout(1)
        idx = torch.tensor([0])
        g.offence[0], g.air[0] = 0, False
        g._turnover(
            idx,
            torch.tensor([[12.0, 8.0]]),
            torch.tensor([-1]),
            torch.tensor([-1]),
            torch.tensor([-1]),
        )
        self.assertEqual(int(g.offence[0]), 1)
        np.testing.assert_allclose(g.disc[0].numpy(), [12.0, game.END_ZONE], atol=1e-5)
        # The team losing it loses for it, the other gains
        self.assertLess(float(g.rewards[0, 0, 0]), 0.0)
        self.assertGreater(float(g.rewards[0, 0, 1]), 0.0)

    def test_the_values_see_each_team_as_it_attacks(self):
        for team in (0, 1):
            seen = batch.critic_inputs(self.game, team)
            self.assertEqual(
                tuple(seen.shape[1:]), (game.PLAYERS, load_module("simulation.ppo").CRITIC_INPUTS)
            )
            np.testing.assert_allclose(seen[0, :, -game.PLAYERS :].numpy(), np.eye(game.PLAYERS))


if __name__ == "__main__":
    unittest.main()
