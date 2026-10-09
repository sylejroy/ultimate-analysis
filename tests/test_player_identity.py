"""Player identities: a lost player who returns as a new track is the same player again."""

import unittest
from unittest.mock import patch

import numpy as np
from support import load_module


def kit(*values):
    """Colour of shirt and shorts."""
    return np.array(values, dtype=np.float32)


RED_TALL = kit(120, 170, 150, 60, 128, 128)
RED_SHORT = kit(124, 168, 151, 62, 128, 128)  # A teammate: nearly the same colours
BLUE = kit(110, 130, 80, 60, 128, 128)
RED_IN_SHADOW = kit(95, 160, 140, 50, 128, 128)  # The same kit, darker


class PlayerIdentityTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.player_identity")
        # Matching by position is off by default; these tests are about how it works
        settings = {"models.tracking.identity.position_match_seconds": 2.0}
        patcher = patch.object(
            self.module,
            "get_setting",
            side_effect=lambda key, default=None: settings.get(key, default),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.identities = self.module.PlayerIdentities()

    def see(self, time_s, *players, alive=None, teams=None):
        """players: (track id, feature, x) -> {track id: player id}"""
        observations = [
            self.module.Observation(
                track_id, (x, 500.0), 100.0, feature, (teams or {}).get(track_id)
            )
            for track_id, feature, x in players
        ]
        return self.identities.assign(time_s, observations, alive)

    def test_a_new_track_of_the_other_team_is_not_a_missing_player_returning(self):
        # Two teams whose kits look alike in this light; the tracker tells them apart
        first = self.see(0.0, (1, RED_TALL, 100), (2, BLUE, 800), teams={1: 0, 2: 1})
        self.see(0.5, (2, BLUE, 800), teams={2: 1})
        # A new track appears where player 1 could be, in a kit like theirs, of team 1
        new = self.see(1.0, (2, BLUE, 800), (7, RED_SHORT, 120), teams={2: 1, 7: 1})
        self.assertNotEqual(new[7], first[1])
        # One of their own team, or of a team not yet known, is taken for them as before
        self.identities.reset()
        first = self.see(0.0, (1, RED_TALL, 100), (2, BLUE, 800), teams={1: 0, 2: 1})
        self.see(0.5, (2, BLUE, 800), teams={2: 1})
        new = self.see(1.0, (2, BLUE, 800), (7, RED_SHORT, 120), teams={2: 1})
        self.assertEqual(new[7], first[1])

    def test_a_track_that_misses_some_frames_keeps_its_player(self):
        first = self.see(0.0, (1, RED_TALL, 100), (2, BLUE, 800), alive={1, 2})
        # The detector misses player 1 for a while; the tracker still follows the track
        for step in range(1, 20):
            self.see(0.1 * step, (2, BLUE, 800), alive={1, 2})
        again = self.see(2.0, (1, RED_TALL, 120), (2, BLUE, 800), alive={1, 2})
        self.assertEqual(again, first)

        # Once the tracker has given the track up, the same track ID means nothing
        self.see(2.1, (2, BLUE, 800), alive={2})
        self.see(40.0, (2, BLUE, 800), alive={2})
        self.assertNotEqual(self.see(40.1, (1, RED_TALL, 120), alive={1, 2})[1], first[1])

    def test_a_player_taken_over_by_a_new_track_leaves_the_old_one(self):
        first = self.see(0.0, (1, RED_TALL, 100), alive={1})
        self.see(0.5, alive={1})
        # The tracker starts a new track for the same player while the old one lingers
        new = self.see(1.0, (4, RED_TALL, 110), alive={1, 4})
        self.assertEqual(new[4], first[1])
        # Should the old track show up again after all, it is somebody else
        both = self.see(1.1, (4, RED_TALL, 112), (1, RED_TALL, 300), alive={1, 4})
        self.assertEqual(both[4], first[1])
        self.assertNotEqual(both[1], first[1])

    def test_a_returning_player_gets_their_identity_back_under_a_new_track(self):
        first = self.see(0.0, (1, RED_TALL, 100), (2, BLUE, 800))
        self.assertEqual(sorted(first.values()), [1, 2])
        self.see(0.5, (1, RED_TALL, 110), (2, BLUE, 800))

        # Track 1 is lost; the player comes back as track 7 a second later, close by.
        # The kit need not look the same as before (here: in a shadow).
        self.see(1.0, (2, BLUE, 800))
        back = self.see(1.5, (2, BLUE, 800), (7, RED_IN_SHADOW, 130))
        self.assertEqual(back[7], first[1])
        self.assertEqual(back[2], first[2])

    def test_a_new_track_in_the_other_teams_kit_is_not_the_missing_player(self):
        first = self.see(0.0, (1, RED_TALL, 100))
        self.see(0.5)
        self.assertNotIn(self.see(1.0, (7, BLUE, 110))[7], first.values())

    def test_opponents_who_come_apart_are_told_apart_by_their_kit(self):
        # A player and their marker covered each other and both tracks were lost. Both
        # new tracks are within reach of both players; place alone could not decide.
        first = self.see(0.0, (1, RED_TALL, 100), (2, BLUE, 120))
        self.see(0.5)
        back = self.see(1.0, (8, BLUE, 105), (9, RED_TALL, 115))
        self.assertEqual(back[9], first[1])
        self.assertEqual(back[8], first[2])

    def test_after_a_few_seconds_place_no_longer_says_who_it_is(self):
        first = self.see(0.0, (1, RED_TALL, 100))
        self.see(1.0)
        self.assertNotIn(self.see(4.0, (5, RED_TALL, 110))[5], first.values())

    def test_a_player_still_seen_while_the_new_track_existed_is_not_that_track(self):
        first = self.see(0.0, (1, RED_TALL, 100), (2, BLUE, 800))
        # Player 1 is missed for a single frame while a new track next to them is reported,
        # which the tracker had been building up for 0.05 s: somebody stepped out from behind
        new = self.identities.assign(
            0.02, [self.module.Observation(9, (120.0, 500.0), 100.0, RED_TALL)], {1, 2, 9}, 0.05
        )
        self.assertNotIn(new[9], first.values())
        self.assertEqual(self.see(0.04, (1, RED_TALL, 100), alive={1, 2, 9})[1], first[1])

    def test_two_missing_teammates_at_the_same_distance_are_not_guessed_between(self):
        first = self.see(0.0, (1, RED_TALL, 100), (2, RED_SHORT, 200))
        self.see(0.5)
        back = self.see(1.0, (9, RED_TALL, 150))
        self.assertNotIn(back[9], first.values())

    def test_a_player_cannot_reappear_further_away_than_they_can_run(self):
        first = self.see(0.0, (1, RED_TALL, 100))
        self.see(0.5)
        # 15 body heights away after half a second
        self.assertNotEqual(self.see(1.0, (4, RED_TALL, 1600))[4], first[1])

    def test_a_runner_is_expected_ahead_the_way_they_were_going(self):
        # A sprints to the right at 300 px per second, their teammate B stands still
        for step in range(6):
            first = self.see(0.1 * step, (1, RED_TALL, 500 + 30 * step), (2, RED_SHORT, 700))
        runner, stander = first[1], first[2]

        # Both are lost. A second later a red player appears 100 px beyond B: where A
        # would be by now. B would have had to start sprinting at once.
        self.see(0.6)
        back = self.see(1.5, (8, RED_TALL, 920))
        self.assertEqual(back[8], runner)

        # ... and one that appears where B stood is B
        self.assertEqual(self.see(1.6, (8, RED_TALL, 950), (9, RED_SHORT, 705))[9], stander)

    def test_a_runner_cannot_turn_on_the_spot(self):
        for step in range(6):
            first = self.see(0.1 * step, (1, RED_TALL, 500 + 30 * step))
        self.see(0.6)
        # Half a second later they cannot be well behind where they were lost
        self.assertNotEqual(self.see(1.0, (5, RED_TALL, 350))[5], first[1])

    def test_matching_by_position_is_off_unless_asked_for(self):
        with patch.object(self.module, "get_setting", side_effect=lambda key, d=None: d):
            first = self.see(0.0, (1, RED_TALL, 100))
            self.see(0.5)
            self.assertNotEqual(self.see(1.0, (7, RED_TALL, 105))[7], first[1])

    def test_a_player_on_screen_is_never_given_to_a_second_track(self):
        first = self.see(0.0, (1, RED_TALL, 100))
        both = self.see(1.0, (1, RED_TALL, 110), (2, RED_TALL, 130))
        self.assertEqual(both[1], first[1])
        self.assertNotEqual(both[2], first[1])

    def test_players_are_forgotten_after_a_while_and_follow_the_camera(self):
        first = self.see(0.0, (1, RED_TALL, 100))
        self.identities.apply_camera_motion(np.array([[1.0, 0, 900], [0, 1, 0], [0, 0, 1]]))
        # The picture moved 900 px to the right, and the player with it
        self.assertEqual(self.see(1.0, (3, RED_TALL, 1010))[3], first[1])

        self.see(2.0)
        self.assertEqual(self.identities.missing_players(set()), [first[1]])
        self.see(40.0)
        self.assertEqual(self.identities.missing_players(set()), [])

    def test_tracks_without_an_appearance_vector_keep_and_get_identities(self):
        first = self.see(0.0, (1, None, 100))
        self.assertEqual(self.see(1.0, (1, RED_TALL, 100)), first)
        self.assertEqual(self.see(2.0, (1, None, 100)), first)
        self.assertIsNone(self.identities.kit_distance(first[1], first[1] + 1))

    def test_merging_declares_a_new_player_to_be_an_earlier_one(self):
        first = self.see(0.0, (1, RED_TALL, 100))
        self.see(1.0)
        later = self.see(5.0, (5, BLUE, 900))
        self.assertEqual(self.identities.missing_players(set(later.values())), [first[1]])
        self.assertGreater(self.identities.kit_distance(later[5], first[1]), 30)

        self.identities.merge(later[5], first[1])
        self.assertEqual(self.see(6.0, (5, BLUE, 900))[5], first[1])


if __name__ == "__main__":
    unittest.main()
