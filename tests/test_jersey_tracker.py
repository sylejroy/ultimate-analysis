"""Jersey numbers from many readings: one reading decides nothing, agreement does."""

import unittest
from unittest.mock import patch

from support import load_module


class JerseyTrackerTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.jersey_tracker")
        patcher = patch.object(
            self.module, "get_setting", side_effect=lambda key, default=None: default
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tracker = self.module.JerseyNumberTracker()

    def read(self, number, times=1, confidence=0.9, player=1, center_x=0.5):
        for _ in range(times):
            self.tracker.add_measurement(player, number, confidence, center_x)
        return self.tracker.get_best_jersey_number(player)

    def test_a_single_reading_is_not_reported_and_never_certain(self):
        self.assertEqual(self.read("17"), (None, 0.0))

        number, certainty = self.read("17")
        self.assertEqual(number, "17")
        self.assertLess(certainty, 0.75)

        # Certainty grows with agreement and approaches, but never reaches, one
        _, after_ten = self.read("17", times=8)
        _, after_thirty = self.read("17", times=20)
        self.assertGreater(after_ten, 0.9)
        self.assertGreater(after_thirty, after_ten)
        self.assertLess(after_thirty, 1.0)

    def test_a_misread_is_outvoted_and_time_does_not_matter(self):
        self.read("71", times=2)
        self.assertEqual(self.read("17", times=2, confidence=0.5)[0], "71")
        number, certainty = self.read("17", times=6)
        self.assertEqual(number, "17")
        self.assertLess(certainty, 0.8)  # The doubt stays visible

        # Nothing is read for a long while: the number stays as it is
        self.assertEqual(self.tracker.get_best_jersey_number(1)[0], "17")
        self.assertEqual(
            [entry[0] for entry in self.tracker.get_top_probabilities(1)], ["17", "71"]
        )

    def test_a_single_digit_backs_the_two_digit_number_it_is_part_of(self):
        self.read("17", times=2)
        number, with_partials = self.read("7", times=3)
        self.assertEqual(number, "17")

        # The same single digit alone is a one-digit number
        self.assertEqual(self.read("7", times=3, player=2)[0], "7")
        # ... and it does not back a number it is not part of
        self.read("23", times=2, player=3)
        self.assertEqual(self.read("7", times=3, player=3)[0], "7")

    def test_a_number_read_twice_leads_a_digit_read_about_as_often(self):
        self.read("2", times=6)
        self.read("12")
        self.assertEqual(self.tracker.get_best_jersey_number(1)[0], "2")  # A stray "12"
        self.assertEqual(self.read("12")[0], "12")

    def test_a_number_read_first_stays_ahead_of_a_digit_read_many_times_after(self):
        self.read("12", times=2)
        self.assertEqual(self.read("2", times=12)[0], "12")

    def test_two_misreads_do_not_overturn_a_digit_read_for_long(self):
        self.read("2", times=30)
        number, certainty = self.read("23", times=2)
        self.assertEqual(number, "2")
        self.assertGreater(certainty, 0.9)

    def test_a_number_leading_on_borrowed_votes_is_not_final_and_more_of_the_digit_undoes_it(self):
        self.read("2", times=5)
        self.read("23", times=2)
        number, certainty = self.read("2")
        self.assertEqual(number, "23")
        self.assertLess(certainty, 0.9)  # finalized_certainty_threshold: it is read on
        self.assertEqual(self.read("2", times=9)[0], "2")

    def test_digits_at_the_side_of_the_box_are_a_neighbours_and_not_counted(self):
        self.read("79", times=3, center_x=0.15)
        self.read("79", times=3, center_x=0.85)
        self.assertEqual(self.tracker.tracked_ids(), [])

    def test_it_is_known_which_of_two_players_a_number_was_read_on_last(self):
        self.read("19", times=2, player=6)
        self.read("19", player=32)
        self.assertGreater(self.tracker.last_read(32, "19"), self.tracker.last_read(6, "19"))
        self.assertEqual(self.tracker.last_read(32, "77"), 0)

    def test_readings_at_the_edge_of_the_box_and_unreadable_ones_count_less_or_not(self):
        self.read("5", times=2, center_x=0.5)
        self.assertEqual(self.read("8", times=3, center_x=0.02)[0], "5")

        self.assertEqual(self.read("Unknown", times=5, player=4), (None, 0.0))
        self.assertEqual(self.read("12", times=3, confidence=0.0, player=4), (None, 0.0))
        self.assertEqual(self.tracker.tracked_ids(), [1])

    def test_merging_two_tracks_of_one_player_pools_their_readings(self):
        self.read("44", times=1, player=10)
        self.read("44", times=1, player=11)
        self.assertEqual(self.tracker.get_best_jersey_number(10), (None, 0.0))

        self.tracker.merge(11, 10)
        self.assertEqual(self.tracker.get_best_jersey_number(10)[0], "44")
        self.assertEqual(self.tracker.tracked_ids(), [10])

    def test_module_functions_share_one_tracker_until_reset(self):
        module = self.module
        module.reset_jersey_tracker()
        module.add_jersey_measurement(7, "9", 0.9)
        module.add_jersey_measurement(7, "9", 0.9)
        self.assertEqual(module.get_best_jersey_number(7)[0], "9")
        self.assertEqual(module.get_jersey_probabilities(7)[0][2], 2)
        module.reset_jersey_tracker()
        self.assertEqual(module.get_best_jersey_number(7), (None, 0.0))


if __name__ == "__main__":
    unittest.main()
