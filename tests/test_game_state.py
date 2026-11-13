"""The phase of a game follows from where the players stand and what the disc does."""

import unittest

from support import load_module

RATE = 10.0  # Frames per second of the made-up footage


class GameStateTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.game_state")
        self.tracker = self.module.GameStateTracker()
        self.time = 0.0

    def player(self, number, team, x, y):
        return self.module.Player(number, team, x, y)

    def lined_up(self, moved=0.0):
        """Team 0 on the near goal line, team 1 on the far one; `moved`: how far team 0
        has run out."""
        near = [self.player(i, 0, 5 + 5 * i, 20 + moved) for i in range(7)]
        far = [self.player(10 + i, 1, 5 + 5 * i, 90) for i in range(7)]
        return near + far

    def mixed(self, shift=0.0):
        """Both teams spread over the middle of the field, on the move."""
        return [self.player(i, i % 2, 5 + 2.5 * i, 40 + 2 * i + shift) for i in range(14)]

    def see(self, seconds, players, **disc):
        """Show the same kind of frame for a while; `players` may be a function of the
        time since the start of it."""
        state = self.tracker.state
        for step in range(int(seconds * RATE)):
            now = players(step / RATE) if callable(players) else players
            state = self.tracker.update(self.time, now, **disc)
            self.time += 1 / RATE
        return state

    def walking(self, t):
        """Both teams spread over the field and nobody hurrying."""
        return self.mixed(shift=1.0 * t)

    def throw(self, flying_for, catcher):
        """The disc held by player 3, in the air, then held by `catcher`."""
        self.see(0.3, self.walking, disc_state="held", holder=3)
        for step in range(int(flying_for * RATE)):
            self.see(0.1, self.walking, disc_state="air", flight_seconds=(step + 1) / RATE)
        return self.see(0.3, self.walking, disc_state="held", holder=catcher)

    def test_a_throw_says_there_is_play_though_nobody_runs(self):
        self.assertEqual(self.throw(0.6, catcher=5), self.module.LIVE)

    def test_a_toss_between_points_is_no_play(self):
        self.see(4.0, self.walking, disc_state="held", holder=3)
        self.assertEqual(self.tracker.state, self.module.BETWEEN_POINTS)
        self.assertEqual(self.throw(0.6, catcher=5), self.module.BETWEEN_POINTS)

    def test_a_holder_who_changes_with_no_flight_is_no_throw(self):
        self.see(1.0, self.walking, disc_state="held", holder=3)
        state = self.see(1.0, self.walking, disc_state="held", holder=5)
        self.assertEqual(state, self.module.UNKNOWN)

    def test_a_moment_in_the_air_is_no_throw(self):
        self.assertEqual(self.throw(0.2, catcher=5), self.module.UNKNOWN)

    def test_the_disc_back_in_the_same_hands_is_no_throw(self):
        self.assertEqual(self.throw(0.6, catcher=3), self.module.UNKNOWN)

    def test_a_flight_before_a_cut_is_not_a_throw_after_it(self):
        self.see(0.3, self.walking, disc_state="held", holder=3)
        self.see(0.6, self.walking, disc_state="air", flight_seconds=0.6)
        self.tracker.cut()
        state = self.see(0.3, self.walking, disc_state="held", holder=5)
        self.assertEqual(state, self.module.UNKNOWN)

    def test_a_point_goes_from_line_up_to_pull_to_live_to_score(self):
        m = self.module
        self.assertEqual(self.see(0.5, self.lined_up()), m.UNKNOWN)
        self.assertEqual(self.see(1.0, self.lined_up(), disc_state="held", holder=3), m.LINED_UP)
        self.assertEqual(self.tracker.attacks, {0: 1, 1: -1})

        # The pull: the disc flies, team 0 runs down the field
        state = self.see(
            1.5,
            lambda t: self.lined_up(moved=6 * t),
            disc_state="air",
            flight_seconds=1.0,
        )
        self.assertEqual(state, m.PULL)
        # Caught by team 1: play is live
        self.assertEqual(
            self.see(1.5, lambda t: self.mixed(4 * t), disc_state="held", holder=11), m.LIVE
        )
        self.assertEqual(self.see(3.0, lambda t: self.mixed(6 + 4 * t)), m.LIVE)

        # Player 2 of team 0 catches the disc in the far end zone, which team 0 attacks
        def catch(t):
            players = self.mixed(18)
            players[2] = self.player(2, 0, 20, 100)
            return players

        self.assertEqual(self.see(1.0, catch, disc_state="held", holder=2), m.SCORE)
        self.assertEqual(
            [(event.kind, event.team) for event in self.tracker.events],
            [("pull", None), ("score", 0)],
        )
        # And then it is between points until the teams line up again, the other way
        self.assertEqual(self.see(5.0, catch), m.BETWEEN_POINTS)
        swapped = [self.player(p.player, 1 - p.team, p.x, p.y) for p in self.lined_up()]
        self.assertEqual(self.see(1.5, swapped), m.LINED_UP)
        self.assertEqual(self.tracker.attacks, {1: 1, 0: -1})

    def test_a_line_that_is_only_left_for_a_moment_is_no_pull(self):
        m = self.module
        self.see(1.5, self.lined_up())
        # Some players jog out and back: the line is not seen for a moment
        self.see(2.0, lambda t: self.lined_up(moved=12 * t))
        self.assertEqual(self.tracker.state, m.PULL)
        self.assertEqual(self.see(1.5, self.lined_up()), m.LINED_UP)
        self.assertEqual(self.tracker.events, [])

    def test_holding_the_disc_in_the_end_zone_one_defends_is_no_score(self):
        m = self.module
        self.see(1.5, self.lined_up())
        self.see(1.5, lambda t: self.lined_up(moved=6 * t), disc_state="air", flight_seconds=1.0)

        # Team 1 catches the pull in the end zone it defends (the far one)
        def catch(t):
            players = self.mixed(4 * t)
            players[1] = self.player(1, 1, 20, 100)
            return players

        self.assertEqual(self.see(3.0, catch, disc_state="held", holder=1), m.LIVE)
        self.assertEqual([event.kind for event in self.tracker.events], ["pull"])

    def test_a_line_up_must_be_both_teams_at_opposite_ends(self):
        m = self.module
        # Everybody at one end: a huddle after a score, not a line-up
        huddle = [self.player(i, i % 2, 5 + 2 * i, 15) for i in range(14)]
        self.see(3.0, huddle)
        self.assertNotEqual(self.tracker.state, m.LINED_UP)
        # A team spread over the field
        self.tracker.reset()
        self.see(3.0, self.mixed())
        self.assertNotEqual(self.tracker.state, m.LINED_UP)

    def test_without_the_field_nothing_is_known(self):
        m = self.module
        self.see(1.5, self.lined_up())
        self.assertEqual(self.see(1.0, self.lined_up(), known=False), m.LINED_UP)
        self.assertEqual(self.see(2.0, self.lined_up(), known=False), m.UNKNOWN)
        # Back on a game in full swing: live play, without a line-up having been seen
        state = self.see(
            3.0, lambda t: self.mixed(5 * t), disc_state="held", holder=4, flight_seconds=None
        )
        self.assertEqual(state, m.LIVE)

    def test_players_standing_about_are_between_points(self):
        m = self.module
        self.assertEqual(self.see(4.0, self.mixed()), m.BETWEEN_POINTS)

    def test_a_point_seen_again_after_a_seek_back_is_not_counted_twice(self):
        def point():
            self.see(1.5, self.lined_up(), disc_state="held", holder=3)
            self.see(
                1.5, lambda t: self.lined_up(moved=6 * t), disc_state="air", flight_seconds=1.0
            )
            self.see(1.5, lambda t: self.mixed(4 * t), disc_state="held", holder=11)
            self.see(3.0, lambda t: self.mixed(6 + 4 * t))

            def catch(t):
                players = self.mixed(18)
                players[2] = self.player(2, 0, 20, 100)
                return players

            self.see(1.0, catch, disc_state="held", holder=2)

        point()
        events = [(event.kind, event.team) for event in self.tracker.events]
        self.assertEqual(events, [("pull", None), ("score", 0)])
        self.tracker.cut()
        self.time = 0.0
        point()
        self.assertEqual([(event.kind, event.team) for event in self.tracker.events], events)

    def live_without_a_line_up(self):
        """Back on a game in full swing after a cut: which end a team attacks not known."""
        self.tracker.cut()
        self.see(3.0, lambda t: self.mixed(5 * t), disc_state="held", holder=4)
        self.assertEqual(self.tracker.state, self.module.LIVE)
        self.assertEqual(self.tracker.attacks, {})

    def in_the_far_end_zone(self, t):
        """Play stops; player 2 of team 0 stands in the far end zone."""
        players = self.mixed(30)
        players[2] = self.player(2, 0, 20, 100)
        return players

    def test_without_a_line_up_a_disc_picked_up_in_an_end_zone_is_no_score(self):
        self.live_without_a_line_up()
        state = self.see(3.5, self.in_the_far_end_zone, disc_state="held", holder=2)
        self.assertEqual(state, self.module.LIVE)
        self.assertNotIn("score", [event.kind for event in self.tracker.events])

    def test_without_a_line_up_a_catch_in_an_end_zone_as_play_stops_is_a_score(self):
        self.live_without_a_line_up()
        self.see(0.6, self.in_the_far_end_zone, disc_state="air", flight_seconds=0.6)
        state = self.see(3.5, self.in_the_far_end_zone, disc_state="held", holder=2)
        self.assertEqual(state, self.module.SCORE)
        self.assertEqual([(e.kind, e.team) for e in self.tracker.events], [("score", 0)])


if __name__ == "__main__":
    unittest.main()
