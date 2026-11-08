"""The simulated game's rules and physics."""

import unittest

import numpy as np
from support import load_module

game = load_module("simulation.game")


def still(throw_to=None):
    return game.Decision(np.zeros((game.PLAYERS, 2)), throw_to)


def state_with(offence_places, defence_places, holder=0):
    places = np.array(offence_places + defence_places, dtype=float)
    return game.GameState(places, np.zeros_like(places), offence=0, holder=holder)


def far_away(n=7, y=100.0):
    return [[2.0 + 5.0 * i, y] for i in range(n)]


class MovementTest(unittest.TestCase):
    def test_nobody_runs_faster_or_speeds_up_faster_than_people_do(self):
        state = game.line_up(np.random.default_rng(0))
        rng = np.random.default_rng(0)
        fast = np.tile([0.0, 50.0], (game.PLAYERS, 1))
        speeds = []
        for _ in range(30):
            game.step(state, (game.Decision(fast), game.Decision(fast)), rng)
            speeds.append(np.hypot(*state.velocities[1:].T).max())
        self.assertAlmostEqual(speeds[0], game.ACCELERATION * game.STEP, places=6)
        self.assertLessEqual(max(speeds), game.RUN_SPEED + 1e-9)
        self.assertAlmostEqual(speeds[-1], game.RUN_SPEED, places=6)

    def test_the_holder_does_not_run(self):
        state = game.line_up(np.random.default_rng(0))
        before = state.places[0].copy()
        run = game.Decision(np.tile([3.0, 3.0], (game.PLAYERS, 1)))
        game.step(state, (run, still()), np.random.default_rng(0))
        np.testing.assert_allclose(state.places[0], before)

    def test_each_team_says_its_velocities_as_it_sees_the_game(self):
        state = game.line_up(np.random.default_rng(0))
        forwards = game.Decision(np.tile([0.0, 5.0], (game.PLAYERS, 1)))
        game.step(state, (forwards, forwards), np.random.default_rng(0))
        self.assertGreater(state.velocities[1, 1], 0)  # Team 0 forwards: y grows
        self.assertLess(state.velocities[8, 1], 0)  # Team 1 forwards: y shrinks


class ThrowTest(unittest.TestCase):
    def play(self, state, throw_to, seed=0, steps=40):
        rng = np.random.default_rng(seed)
        game.step(state, (still(np.array(throw_to)), still()), rng)
        for _ in range(steps):
            if state.throw is None:
                break
            game.step(state, (still(), still()), rng)
        return state

    def test_a_throw_to_an_open_teammate_is_caught_and_takes_its_time(self):
        state = state_with([[20.0, 30.0], [20.0, 48.0]] + far_away(5, 5.0), far_away())
        self.play(state, [20.0, 48.0])
        self.assertEqual(state.holder, 1)
        self.assertEqual([e.kind for e in state.events], ["throw", "catch"])
        flight = state.events[1].seconds - state.events[0].seconds
        expected = game.flight_of([20.0, 30.0], [20.0, 48.0], 0.0, 0).lands
        # As long as the flight to where it was aimed, give or take how far off it went
        self.assertAlmostEqual(flight, expected, delta=2 * game.STEP)

    def test_a_defender_alone_where_it_comes_down_blocks_it(self):
        state = state_with([[20.0, 30.0]] + far_away(6, 5.0), [[30.0, 50.0]] + far_away(6))
        self.play(state, [30.0, 50.0])
        self.assertEqual(state.offence, 1)
        self.assertEqual(state.holder, game.PLAYERS)
        self.assertEqual(state.events[-1].kind, "block")

    def test_a_throw_to_nobody_falls_and_the_other_team_picks_it_up(self):
        state = state_with([[20.0, 30.0]] + far_away(6, 5.0), far_away())
        self.play(state, [5.0, 60.0])
        self.assertEqual((state.offence, state.holder), (1, None))
        # Where it was aimed, as far off as throws go
        self.assertLess(np.hypot(*(state.disc - [5.0, 60.0])), 5.0)
        state.places[game.PLAYERS] = state.disc + [0.5, 0.0]
        game.step(state, (still(), still()), np.random.default_rng(0))
        self.assertEqual(state.holder, game.PLAYERS)

    def test_a_tight_window_is_lost_sometimes_and_an_open_one_hardly_ever(self):
        def caught(defender_gap, seed):
            receiver = [20.0, 50.0]
            defender = [20.0 + defender_gap, 50.0]
            state = state_with(
                [[20.0, 30.0], receiver] + far_away(5, 5.0), [defender] + far_away(6)
            )
            return self.play(state, receiver, seed).offence == 0

        tight = [caught(0.6, seed) for seed in range(60)]
        self.assertTrue(0 < sum(tight) < 60)
        # Open: lost only now and then, to a drop or a throw that goes further off than
        # a receiver who does not move can reach
        self.assertGreaterEqual(sum(caught(6.0, seed) for seed in range(40)), 32)

    def test_a_catch_in_the_end_zone_scores_the_point(self):
        state = state_with([[20.0, 75.0], [20.0, 95.0]] + far_away(5, 5.0), far_away(7, 40.0))
        self.play(state, [20.0, 95.0])
        self.assertEqual(state.scored, 0)
        self.assertTrue(state.over)

    def test_ten_seconds_marked_is_a_stall(self):
        state = state_with([[20.0, 30.0]] + far_away(6, 5.0), [[21.0, 30.0]] + far_away(6))
        rng = np.random.default_rng(0)
        for _ in range(int(game.STALL / game.STEP) + 1):
            game.step(state, (still(), still()), rng)
        self.assertEqual(state.events[-1].kind, "stall")
        self.assertEqual(state.offence, 1)


class EndZoneTest(unittest.TestCase):
    def test_a_disc_lost_in_the_end_zone_the_other_team_attacks_goes_to_its_goal_line(self):
        state = game.line_up(np.random.default_rng(0))  # Team 0 attacks the far end
        # Team 0 loses it in its own end zone, the one team 1 attacks
        game._turnover(state, "drop", np.array([12.0, 8.0]), None)
        self.assertEqual(state.offence, 1)
        np.testing.assert_allclose(state.disc, [12.0, game.END_ZONE])
        self.assertIsNone(state.scored)
        # Lost in midfield: where it fell
        state = game.line_up(np.random.default_rng(0))
        game._turnover(state, "drop", np.array([12.0, 50.0]), None)
        np.testing.assert_allclose(state.disc, [12.0, 50.0])

    def test_a_defender_who_catches_it_in_the_end_zone_they_attack_scores(self):
        state = game.line_up(np.random.default_rng(0))
        state.places[9] = [20.0, 10.0]  # A team 1 defender in team 0's end zone
        game._turnover(state, "block", state.places[9].copy(), 9, 0, 9)
        self.assertEqual(state.scored, 1)
        self.assertEqual(state.events[-1].kind, "score")
        self.assertEqual(state.events[-1].player, 9)


class FacingTest(unittest.TestCase):
    def setUp(self):
        # Team 0 holds the disc at midfield; one of its players stands 10 yd behind it
        self.state = game.line_up(np.random.default_rng(0))
        self.state.places[0] = [20.0, 50.0]
        self.state.places[3] = [20.0, 40.0]
        self.state.places[7:] = [[39.0, 109.0 - k] for k in range(7)]  # The others far off

    def move(self, velocity, steps):
        wanted = np.zeros((game.PLAYERS, 2))
        wanted[3] = velocity
        still = game.Decision(np.zeros((game.PLAYERS, 2)))
        speeds = []
        for _ in range(steps):
            game.step(self.state, (game.Decision(wanted), still), np.random.default_rng(0))
            speeds.append(float(np.hypot(*self.state.velocities[3])))
        return speeds

    def test_backing_away_from_the_disc_is_slower_than_running(self):
        # Facing the disc, backing away no faster than they can backpedal
        speeds = self.move([0.0, -game.BACKPEDAL_SPEED], 30)
        self.assertLessEqual(max(speeds), game.BACKPEDAL_SPEED + 1e-9)
        self.assertGreater(self.state.facing[3][1], 0.9)  # Still facing the disc
        # Towards it, full speed
        self.setUp()
        self.assertAlmostEqual(self.move([0.0, 8.0], 30)[-1], game.RUN_SPEED, places=6)

    def test_running_from_the_disc_flat_out_turns_them_round(self):
        speeds = self.move([0.0, -8.0], 30)
        self.assertLess(self.state.facing[3][1], -0.9)  # Turned: facing away from the disc
        self.assertAlmostEqual(speeds[-1], game.RUN_SPEED, places=6)
        # A quarter of a second in they are still turning round
        self.setUp()
        self.move([0.0, -8.0], 3)
        self.assertGreater(self.state.facing[3][1], -0.9)


class WindTest(unittest.TestCase):
    def test_a_side_wind_carries_a_blade_further_than_a_flat_throw(self):
        disc = load_module("simulation.disc")
        side_wind = np.array([5.0, 0.0])
        flat = disc.flight_of([20.0, 30.0], [20.0, 60.0], 0.0, 0, 0.0, None, side_wind, 0.0)
        blade = disc.flight_of([20.0, 30.0], [20.0, 60.0], 0.0, 0, 0.0, None, side_wind, 1.0)
        self.assertGreater(flat.drift[0], 0.0)  # Downwind
        self.assertGreater(blade.drift[0], 2.0 * flat.drift[0])
        # A wind straight behind carries neither sideways
        behind = np.array([0.0, 5.0])
        still = disc.flight_of([20.0, 30.0], [20.0, 60.0], 0.0, 0, 0.0, None, behind, 1.0)
        self.assertAlmostEqual(float(still.drift[0]), 0.0, places=6)


class PullTest(unittest.TestCase):
    def test_the_pull_goes_from_the_far_line_to_the_receiving_half(self):
        for receiving in (0, 1):
            state = game.pull_start(np.random.default_rng(1), receiving)
            pull = state.throw
            self.assertTrue(pull.pull)
            self.assertIsNone(state.holder)
            self.assertEqual(state.offence, receiving)
            self.assertEqual(game.team_of(pull.thrower), 1 - receiving)
            # Seen by the receiving team, the pull comes from far ahead to near
            start = game.turn(pull.start[None], game.attack_of(receiving))[0]
            target = game.turn(pull.aimed[None], game.attack_of(receiving))[0]
            self.assertGreater(start[1], game.LENGTH / 2)
            self.assertLess(target[1], game.LENGTH / 2)
            self.assertGreater(pull.lands - pull.released, 3.0)

    def test_the_receiving_team_gets_the_pull_and_nobody_turns_it_over(self):
        for seed in range(6):
            rng = np.random.default_rng(seed)
            state = game.pull_start(rng, 0)
            receiving = range(game.PLAYERS)
            while state.throw is not None:
                # Everyone runs to where it comes down, the pulling team too
                wanted = (state.throw.target - state.places) * 5.0
                game.step(state, (game.Decision(wanted[:7]), game.Decision(-wanted[7:])), rng)
            kinds = [e.kind for e in state.events]
            self.assertEqual(state.offence, 0)
            self.assertTrue(set(kinds) <= {"pull caught", "pull lands"}, kinds)
            if state.holder is not None:
                self.assertIn(state.holder, receiving)


class ViewTest(unittest.TestCase):
    def test_each_team_sees_itself_attacking_the_far_end(self):
        state = game.line_up(np.random.default_rng(0))
        mine, theirs = game.view(state, 0), game.view(state, 1)
        self.assertTrue(mine.attacking)
        self.assertFalse(theirs.attacking)
        np.testing.assert_allclose(mine.ours[:, :2], state.places[:7])
        np.testing.assert_allclose(theirs.ours[:, :2], [40.0, 110.0] - state.places[7:])
        self.assertEqual(theirs.holder, 0)


if __name__ == "__main__":
    unittest.main()
