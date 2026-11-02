"""The numbers shown: one stays until another clearly leads it, and a team has one of each."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from support import load_module


def player(track_id, team=0):
    return SimpleNamespace(track_id=track_id, team=team, class_name="player")


class PlayerNumbersTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.player_numbers")
        self.votes = sys.modules[self.module.__package__ + ".jersey_tracker"]
        self.votes.reset_jersey_tracker()
        self.addCleanup(self.votes.reset_jersey_tracker)
        tracker = Mock()
        tracker.missing_players.return_value = []
        self.numbers = self.module.PlayerNumbers(tracker)
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)

    def show(self, tracks, read=None):
        """A frame on which the numbers in `read` (player -> [number]) were read."""

        def reader(frame, tracks, **rest):
            for player_id, numbers in (read or {}).items():
                for number in numbers:
                    self.votes.add_jersey_measurement(player_id, number, 0.9)
            answer = {player_id: ("Unknown", None) for player_id in read or {}}
            return answer, {}, set(self.numbers._finalized)

        with patch.object(self.module, "run_player_id_on_tracks", reader):
            self.numbers.update(self.frame, tracks, 0)
        return {key: value[0] for key, value in self.numbers.numbers.items()}

    def test_a_number_shown_stays_until_another_clearly_leads_it(self):
        one = player(1)
        self.assertEqual(self.show([one], {1: ["23", "23", "23"]}), {1: "23"})
        self.assertEqual(self.show([one], {1: ["45", "45", "45"]}), {1: "23"})  # Level
        self.assertEqual(self.show([one], {1: ["45"] * 6}), {1: "45"})

    def test_a_digit_gives_way_at_once_to_the_number_that_holds_it(self):
        one = player(1)
        self.assertEqual(self.show([one], {1: ["7", "7", "7"]}), {1: "7"})
        self.assertEqual(self.show([one], {1: ["17", "17"]}), {1: "17"})

    def test_a_digit_read_for_long_is_not_given_up_for_two_misreads(self):
        one = player(1)
        self.assertEqual(self.show([one], {1: ["2"] * 30}), {1: "2"})
        self.assertEqual(self.show([one], {1: ["23", "23"]}), {1: "2"})

    def test_of_two_players_of_a_team_only_the_one_read_last_shows_a_number(self):
        swapped, wearer = player(6), player(32)
        self.show([swapped, wearer], {6: ["19", "19"]})
        shown = self.show([swapped, wearer], {32: ["19", "19"]})
        self.assertEqual(shown, {6: "Unknown", 32: "19"})

    def test_a_player_joined_goes_by_the_votes_of_both(self):
        earlier, newcomer = player(12), player(40)
        self.show([earlier], {12: ["44"] * 6})
        self.numbers.numbers.pop(12)  # Gone from the picture
        self.show([newcomer], {40: ["7", "7"]})
        self.numbers.join(newcomer, 12)
        self.assertEqual(self.numbers.numbers[12][0], "44")

    def test_players_of_two_teams_may_wear_the_same_number(self):
        shown = self.show([player(1, team=0), player(2, team=1)], {1: ["12"] * 2, 2: ["12"] * 2})
        self.assertEqual(shown, {1: "12", 2: "12"})

    def test_nothing_is_taken_away_while_the_team_is_not_known(self):
        shown = self.show([player(1, team=None), player(2)], {1: ["12"] * 2, 2: ["12"] * 2})
        self.assertEqual(shown, {1: "12", 2: "12"})


if __name__ == "__main__":
    unittest.main()
