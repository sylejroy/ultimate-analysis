"""What simulated players know: seeing late, remembering what is behind them, the
defence reading a throw, teammates hearing each other; and teams that play by them."""

import unittest

import numpy as np
from support import load_module

game = load_module("simulation.game")
perception = load_module("simulation.perception")
scripted = load_module("simulation.scripted")
match = load_module("simulation.match")


def standing(places, holder=0, offence=0):
    places = np.array(places, dtype=float)
    return game.GameState(places, np.zeros_like(places), offence=offence, holder=holder)


def spread():
    """Team 0 spread across the field around y 50, team 1 a little further on."""
    ours = [[5.0 + 5.0 * i, 50.0] for i in range(7)]
    theirs = [[5.0 + 5.0 * i, 55.0] for i in range(7)]
    return ours + theirs


class SensesTest(unittest.TestCase):
    # The senses as they are with perception.PERFECT off
    def setUp(self):
        self.perfect, perception.PERFECT = perception.PERFECT, False

    def tearDown(self):
        perception.PERFECT = self.perfect

    def test_a_move_is_seen_late_and_the_mover_knows_it_at_once(self):
        state = standing(spread())
        senses = perception.Senses(1, np.random.default_rng(0))
        senses.look(state)
        state.places[0] += [0.0, 3.0]  # The holder of team 0 is moved on
        state.places[7] += [0.0, 3.0]  # ... and a player of team 1
        views = senses.look(state)
        # Team 1 sees team 0's player where they were (turned: the field is upside down)
        self.assertAlmostEqual(views[1].theirs[0, 1], 110.0 - 50.0, delta=1.5)
        # The player of team 1 knows where they are themselves
        self.assertAlmostEqual(views[0].ours[0, 1], 110.0 - 58.0, delta=1e-6)

    def test_what_is_behind_a_player_becomes_unsure(self):
        state = standing(spread())
        senses = perception.Senses(0, np.random.default_rng(0))
        senses.look(state)
        for _ in range(30):  # Three seconds looking at the holder, ahead of the player
            state.seconds += game.STEP
            state.places[0] = [20.0, 100.0]
            views = senses.look(state)
        player = views[3]  # At (20, 50): the holder ahead, team 1 at y 55 ahead too
        self.assertLess(player.theirs_unsure.max(), 1.0)
        # Team 0's player furthest along the line behind them? None: all are beside.
        # A player looking ahead does not see one far behind
        state.places[13] = [20.0, 10.0]
        for _ in range(20):
            state.seconds += game.STEP
            views = senses.look(state)
        self.assertGreater(views[3].theirs_unsure[6], 1.5)

    def test_the_defence_reads_a_throw_off_at_first_and_the_offence_knows_a_short_one(self):
        state = standing(spread())
        state.throw = game.flight_of([5.0, 50.0], [15.0, 58.0], 0.0, 0)  # 13 yd
        state.holder = None
        offence = perception.Senses(0, np.random.default_rng(0)).look(state)
        defence = perception.Senses(1, np.random.default_rng(1)).look(state)
        np.testing.assert_allclose(offence[2].disc, [15.0, 58.0])
        read = [game.turn(v.disc, -1) for v in defence]
        self.assertGreater(max(np.hypot(*(r - [15.0, 58.0])) for r in read), 0.5)

    def test_a_long_throw_is_misjudged_by_the_offence_too_less_as_it_comes_down(self):
        state = standing(spread())
        state.throw = game.flight_of([5.0, 30.0], [30.0, 90.0], 0.0, 0)  # 65 yd
        state.holder = None
        senses = perception.Senses(0, np.random.default_rng(0))
        first = np.array([np.hypot(*(v.disc - [30.0, 90.0])) for v in senses.look(state)])
        self.assertGreater(first.max(), 0.5)
        state.seconds = state.throw.lands - 0.1
        late = np.array([np.hypot(*(v.disc - [30.0, 90.0])) for v in senses.look(state)])
        self.assertLess(late.max(), first.max() / 5)

    def test_what_a_player_says_is_heard_by_the_team_the_step_after(self):
        state = standing(spread())
        senses = perception.Senses(0, np.random.default_rng(0))
        said = np.zeros((7, perception.MESSAGE_SIZE))
        said[2, 1] = 5.0
        senses.hear(said)
        self.assertEqual(senses.look(state)[4].heard[2, 1], 5.0)


class ScriptedTeamsTest(unittest.TestCase):
    def test_scripted_teams_play_points_to_the_end(self):
        rng = np.random.default_rng(0)
        for i in range(3):
            state = match.play_point((scripted.ScriptedTeam(i), scripted.ScriptedTeam(9 + i)), rng)
            self.assertTrue(state.over)
            self.assertIn("throw", [e.kind for e in state.events])

    def test_the_benchmark_team_plays_wider_and_holds_longer(self):
        rng = np.random.default_rng(1)
        teams = (scripted.ScriptedTeam2(0, plays=False), scripted.ScriptedTeam(1, plays=False))
        state = match.play_point(teams, rng)
        self.assertTrue(state.over)
        # Its stack zig-zags across the field, a good deal wider than the old one's
        old, new = scripted.ScriptedTeam, scripted.ScriptedTeam2
        self.assertGreater(new.stack_gap, old.stack_gap)
        self.assertGreater(new.stack_across, 0.0)
        # Its holder waits longer before throwing
        self.assertGreater(new.shortest_hold, old.shortest_hold)

    def test_defenders_switch_when_both_call_it(self):
        team = scripted.ScriptedTeam(0)
        state = standing(spread(), holder=None, offence=1)
        # Team 0 defends; its players 0 and 1 stand at each other's man
        state.places[0], state.places[1] = [10.0, 55.0], [5.0, 55.0]
        state.holder = 7
        senses = perception.Senses(0, np.random.default_rng(0))
        for _ in range(3):
            decision = team.decide(senses.look(state))
            senses.hear(decision.messages)
        self.assertEqual(team.guarding[:2], [1, 0])


if __name__ == "__main__":
    unittest.main()
