"""How simulated players learn: brains that rewire, and the rewards of reinforcement."""

import unittest

import numpy as np
import torch
from support import load_module

game = load_module("simulation.game")
learned = load_module("simulation.learned")
ppo = load_module("simulation.ppo")


class InputsTest(unittest.TestCase):
    def test_a_network_made_for_fewer_inputs_plays_as_it_did(self):
        torch.manual_seed(0)
        network = learned.movement_network()
        weights = network.state_dict()
        older = dict(weights)
        older["0.weight"] = weights["0.weight"][:, :111].clone()  # Before role and the rest
        network.load_state_dict(learned.with_inputs(older))
        x = torch.randn(3, learned.INPUTS)
        cut = x.clone()
        cut[:, 111:] = 0.0
        with torch.no_grad():
            torch.testing.assert_close(network(x), network(cut))

    def test_players_see_their_role_how_long_the_disc_is_held_and_who_speeds_up(self):
        perception = load_module("simulation.perception")
        state = game.line_up(np.random.default_rng(0))  # Team 0 holds the disc
        senses = perception.Senses(0, np.random.default_rng(0))
        still = game.Decision(np.zeros((game.PLAYERS, 2)))
        run = game.Decision(np.tile([0.0, 8.0], (game.PLAYERS, 1)))
        for _ in range(6):
            views = senses.look(state)
            game.step(state, (run, still), np.random.default_rng(0))
        views = senses.look(state)
        groups = dict(learned.INPUT_GROUPS)
        starts = np.cumsum([0] + [n for _, n in learned.INPUT_GROUPS])
        at = {name: starts[i] for i, (name, _) in enumerate(learned.INPUT_GROUPS)}
        x = learned.view_features(views[2])
        role = x[at["Role"] : at["Role"] + groups["Role"]]
        self.assertEqual(int(np.argmax(role)), 2)  # Two places on from the holder
        self.assertAlmostEqual(float(x[at["Holding for"]]), views[2].held / 5.0, places=5)
        self.assertGreater(views[2].held, 0.0)
        accelerations = x[at["Accelerations"] : at["Accelerations"] + groups["Accelerations"]]
        self.assertGreater(accelerations[1], 0.0)  # Speeding up along the field
        # Nobody holds the disc: no role
        state.holder, state.disc = None, np.array([20.0, 40.0])
        x = learned.view_features(senses.look(state)[2])
        self.assertEqual(float(x[at["Role"] : at["Role"] + groups["Role"]].sum()), 0.0)


class MoreInputsTest(unittest.TestCase):
    def setUp(self):
        perception = load_module("simulation.perception")
        self.state = game.line_up(np.random.default_rng(0))  # Team 0 holds the disc
        self.senses = perception.Senses(0, np.random.default_rng(0))
        still = game.Decision(np.zeros((game.PLAYERS, 2)))
        game.step(self.state, (still, still), np.random.default_rng(0))
        starts = np.cumsum([0] + [n for _, n in learned.INPUT_GROUPS])
        self.at = {
            name: slice(starts[i], starts[i] + n)
            for i, (name, n) in enumerate(learned.INPUT_GROUPS)
        }

    def inputs(self, k):
        return learned.view_features(self.senses.look(self.state)[k])

    def test_a_disc_on_the_ground_is_seen(self):
        self.assertEqual(float(self.inputs(3)[self.at["Disc on the ground"]][0]), 0.0)
        self.state.holder, self.state.disc = None, np.array([20.0, 40.0])
        ground = self.inputs(3)[self.at["Disc on the ground"]]
        self.assertEqual(float(ground[0]), 1.0)
        mine = self.state.places[3]
        self.assertAlmostEqual(float(ground[2]), (40.0 - mine[1]) / learned.SPREAD, places=4)

    def test_the_player_of_their_number_comes_first_among_the_other_team(self):
        k = 4
        x = self.inputs(k)
        first = x[self.at["Opponents"]][: learned.OTHER]
        offset = self.state.places[game.PLAYERS + k] - self.state.places[k]
        np.testing.assert_allclose(first[:2], offset / learned.SPREAD, atol=0.1)

    def test_what_a_teammate_said_comes_with_where_they_are(self):
        said = np.zeros((game.PLAYERS, 4))
        said[5, 1] = 0.7
        self.senses.hear(said)
        x = self.inputs(4)
        by_slot = x[self.at["Heard, by teammate"]].reshape(-1, 4)
        mates = x[self.at["Teammates"]].reshape(-1, learned.OTHER)
        slot = int(np.flatnonzero(by_slot[:, 1])[0])
        offset = self.state.places[5] - self.state.places[4]
        np.testing.assert_allclose(mates[slot, :2], offset / learned.SPREAD, atol=0.1)

    def test_players_see_which_way_they_and_others_face(self):
        facing = self.inputs(2)[self.at["Facing"]]
        self.assertAlmostEqual(float(np.hypot(*facing[:2])), 1.0, places=5)

    def test_the_values_see_the_game_as_it_is_for_each_player(self):
        seen = ppo.critic_inputs(self.state, 1)
        self.assertEqual(seen.shape, (game.PLAYERS, ppo.CRITIC_INPUTS))
        np.testing.assert_allclose(seen[:, -game.PLAYERS :], np.eye(game.PLAYERS))
        self.assertEqual(float(seen[0, -game.PLAYERS - 1]), 0.0)  # Team 1 has not the disc
        self.assertEqual(float(ppo.critic_inputs(self.state, 0)[0, -game.PLAYERS - 1]), 1.0)


class RewiringTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.network = learned.movement_network()
        self.rng = np.random.default_rng(0)

    def wired(self):
        return sum(int(w.sum()) for w in learned.wiring_of(self.network))

    def test_the_weakest_connections_are_cut_and_fewer_grow_while_densely_wired(self):
        before = self.wired()
        learned.rewire(self.network, self.rng)
        # Fully wired: the weakest are cut, and there is nowhere else to grow
        self.assertAlmostEqual(self.wired(), 0.95 * before, delta=0.002 * before)
        before = self.wired()
        learned.rewire(self.network, self.rng)
        # Still dense: half as many grow as are cut
        self.assertAlmostEqual(self.wired(), before - 0.5 * 0.05 * before, delta=0.002 * before)

    def test_once_sparse_as_many_grow_as_are_cut(self):
        for _ in range(40):
            learned.rewire(self.network, self.rng)
        wiring = learned.wiring_of(self.network)
        density = sum(int(w.sum()) for w in wiring) / sum(w.size for w in wiring)
        self.assertLess(density, 0.6)
        before = self.wired()
        learned.rewire(self.network, self.rng)
        self.assertAlmostEqual(self.wired(), before, delta=0.01 * before)

    def test_learning_does_not_regrow_what_was_cut(self):
        wiring = learned.rewire(self.network, self.rng)
        with torch.no_grad():
            for layer in self.network:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.add_(0.1)
        learned.keep_wiring(self.network, wiring)
        self.assertEqual(learned.wiring_of(self.network)[0].sum(), wiring[0].sum())


class RewardsTest(unittest.TestCase):
    def point(self, events, scored=None):
        state = game.line_up(np.random.default_rng(0))
        state.events = [game.Event(*event) for event in events]
        state.scored = scored
        return state

    def test_a_completion_earns_its_pass_and_the_yards_it_gained(self):
        # Team 0 attacks up the field: a pass from 20 to 30 yards, one back to 25, a drop
        state = self.point(
            [
                (0.5, "throw", 0, (20.0, 20.0)),
                (1.0, "catch", 0, (20.0, 30.0)),
                (1.5, "throw", 0, (20.0, 30.0)),
                (2.0, "catch", 0, (20.0, 25.0)),
                (3.5, "throw", 0, (20.0, 25.0)),
                (4.0, "drop", 0, (20.0, 70.0)),
            ]
        )
        rewards = ppo.rewards_of(state, 60, game.STEP)
        self.assertAlmostEqual(rewards[0, 9], ppo.COMPLETION + ppo.PER_YARD * 10.0, places=5)
        self.assertAlmostEqual(rewards[0, 19], ppo.COMPLETION - ppo.PER_YARD * 5.0, places=5)
        self.assertAlmostEqual(rewards[0, 39], -ppo.TURNOVER, places=5)
        np.testing.assert_allclose(rewards[1], -rewards[0])  # The other team: the opposite

    def test_a_score_is_rewarded_and_counts_to_the_goal_line_wherever_caught(self):
        state = self.point(
            [
                (1.0, "catch", 1, (20.0, 30.0)),  # No throw before it: no yards
                (1.5, "throw", 1, (20.0, 30.0)),
                (2.0, "catch", 1, (20.0, 15.0)),
                (2.0, "score", 1, (20.0, 15.0)),
            ],
            scored=1,
        )
        rewards = ppo.rewards_of(state, 40, game.STEP)
        self.assertAlmostEqual(rewards[1, 9], ppo.COMPLETION, places=5)
        # Team 1 attacks down the field: 30 is 80 yards from its end, its goal line 90.
        # The scoring catch, 5 yards into the end zone: the pass, 10 yards and the score
        self.assertAlmostEqual(
            rewards[1, 19], ppo.COMPLETION + ppo.PER_YARD * 10.0 + ppo.SCORE, places=5
        )

    def test_each_player_is_credited_for_what_they_did(self):
        state = self.point(
            [
                (0.5, "throw", 0, (20.0, 20.0), 2),
                (1.0, "catch", 0, (20.0, 30.0), 4),
                (1.5, "throw", 0, (20.0, 30.0), 4),
                (2.0, "block", 0, (20.0, 40.0), 4, 9),  # Thrown by 4, blocked by 9
            ]
        )
        own = ppo.player_rewards(state, 30, game.STEP)
        self.assertAlmostEqual(own[2, 9], ppo.THROWN)
        self.assertAlmostEqual(own[4, 9], ppo.CAUGHT)
        self.assertAlmostEqual(own[4, 19], ppo.LOST)
        self.assertAlmostEqual(own[9, 19], ppo.BLOCKED)
        self.assertEqual(np.count_nonzero(own), 4)

    def test_the_defence_gains_while_its_mark_counts_the_stall(self):
        state = self.point([])
        counted = [-1, -1, -1, 0, 0, 0, -1, -1, -1, -1]  # Team 0 counts for 0.3 s
        rewards = ppo.rewards_of(state, 10, game.STEP, counted, stall_reward=1.0)
        np.testing.assert_allclose(rewards[0, 3:6], 0.1)
        self.assertEqual(rewards[0, :3].sum() + rewards[0, 6:].sum(), 0.0)
        np.testing.assert_allclose(rewards[1], 0.0)  # Not taken from the offence

    def test_the_stall_is_seen_counted_only_while_a_defender_is_near_the_thrower(self):
        rng = np.random.default_rng(0)
        state = game.line_up(rng)  # Team 0 holds the disc, its mark beside the thrower
        counted, watch = ppo.stall_watch()
        still = game.Decision(np.zeros((game.PLAYERS, 2)))
        for _ in range(5):
            game.step(state, (still, still), rng)
            watch(state)
        state.places[7] = [5.0, 5.0]  # The mark walks away
        for _ in range(5):
            game.step(state, (still, still), rng)
            watch(state)
        self.assertEqual(counted[:5], [1] * 5)
        self.assertEqual(counted[-3:], [-1] * 3)


def lone_receiver(covered):
    """Team 0 holds the disc at midfield, its players in a stack; one runs deep alone, or
    with a defender beside them."""
    state = game.line_up(np.random.default_rng(0))
    places = state.places
    places[:7] = [[20.0, 50.0]] + [[20.0, 60.0 + 3.0 * k] for k in range(5)] + [[30.0, 80.0]]
    places[7:] = places[:7] + [1.0, 1.0]
    places[7] = [19.0, 49.0]  # The mark, behind: not in the way of a throw upfield
    # The deep receiver's defender: goal side of them, or far behind
    places[13] = [30.5, 82.0] if covered else [5.0, 30.0]
    state.velocities[:] = 0.0
    state.velocities[6] = [0.0, 7.0]
    state.velocities[13] = [0.0, 7.0] if covered else [0.0, 0.0]
    return state


class HolderTest(unittest.TestCase):
    def test_the_holder_may_throw_to_any_square_or_to_a_teammate_led(self):
        perception = load_module("simulation.perception")
        throw_targets = load_module("processing.throw_targets")

        def look(covered):
            state = lone_receiver(covered)
            state.seconds, state.held_since = 2.0, 0.0
            view = perception.Senses(0, np.random.default_rng(0)).look(state)[0]
            return view, ppo.holder_inputs(view)

        view, (choices, prior, throw, targets) = look(False)
        squares = len(throw_targets.CENTRES)
        self.assertEqual(choices.shape, (squares + 6, ppo.SQUARE_FEATURES))
        self.assertEqual(throw.shape, (ppo.THROW_FEATURES,))
        self.assertEqual(len(prior), len(targets))
        # The last six are the teammates, led: the deep runner further on than they are
        np.testing.assert_allclose(choices[squares:, 10], 1.0)
        self.assertGreater(targets[-1][1], view.ours[6, 1])
        # A throw up through the stack passes its defenders; one across the field does not
        through = int(np.argmin(np.hypot(*(targets[:squares] - [20.0, 80.0]).T)))
        across = int(np.argmin(np.hypot(*(targets[:squares] - [38.0, 50.0]).T)))
        self.assertLess(choices[through, 15], 0.5)  # Nearest defender to the path, / 5 yd
        self.assertGreater(choices[across, 15], choices[through, 15] + 0.4)
        # Longer throws are longer in the air
        self.assertGreater(choices[through, 17], choices[across, 17])
        # How open they are where led: the deep runner more so alone than covered
        _, (covered, *_) = look(True)
        self.assertGreater(choices[-1, 11], covered[-1, 11] + 0.3)


if __name__ == "__main__":
    unittest.main()
