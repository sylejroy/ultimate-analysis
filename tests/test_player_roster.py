"""The roster: one entry per player of a video, matched by looks, named by a number."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from support import load_module

WHITE, DARK = (230, 230, 230), (60, 40, 40)


def looks_like(*direction, noise=0.0, seed=0):
    """A feature vector of length 1 pointing mostly one way."""
    vector = np.zeros(8)
    vector[: len(direction)] = direction
    vector = vector + noise * np.random.default_rng(seed).normal(size=8)
    return vector / np.linalg.norm(vector)


ANNA, BEN, CARL = looks_like(1, 0, 0), looks_like(0, 1, 0), looks_like(0, 0, 1)


class PlayerRosterTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.player_roster")
        self.roster = self.module.PlayerRoster()

    def see(self, track, vector, colour=WHITE, times=None):
        for look in range(times or self.module.MIN_LOOKS):
            self.roster.look(track, looks_like(*vector[:3], noise=0.05, seed=look), colour)

    def test_a_player_keeps_their_entry_from_one_point_to_the_next(self):
        self.see(1, ANNA)
        self.see(2, BEN)
        first = self.roster.match([1, 2])
        self.assertEqual(len(set(first.values())), 2)

        # A cut: the tracker starts again and numbers its tracks anew, the other way round
        self.roster.tracks_ended()
        self.see(1, BEN)
        self.see(2, ANNA)
        second = self.roster.match([1, 2])
        self.assertEqual(second[2], first[1])
        self.assertEqual(second[1], first[2])
        self.assertEqual(len(self.roster.entries), 2)

    def test_a_track_is_not_matched_before_it_has_been_looked_at_enough(self):
        self.see(1, ANNA, times=self.module.MIN_LOOKS - 1)
        self.assertEqual(self.roster.match([1]), {})

    def test_someone_who_looks_like_nobody_known_is_a_new_player(self):
        self.see(1, ANNA)
        self.roster.match([1])
        self.roster.tracks_ended()
        self.see(5, CARL)
        self.roster.match([5])
        self.assertEqual(len(self.roster.entries), 2)

    def test_two_tracks_at_the_same_moment_are_two_players_however_alike(self):
        self.see(1, ANNA)
        self.roster.match([1])
        self.see(2, ANNA)
        both = self.roster.match([1, 2])
        self.assertNotEqual(both[1], both[2])

    def test_players_of_the_other_team_are_not_considered(self):
        self.see(1, ANNA, WHITE)
        self.roster.match([1])
        self.roster.tracks_ended()
        self.see(1, ANNA, DARK)
        self.roster.match([1])
        teams = sorted(entry.team for entry in self.roster.entries.values())
        self.assertEqual(teams, [0, 1])

    def test_a_number_read_once_names_the_player_in_every_later_point(self):
        self.see(1, ANNA)
        self.roster.match([1])
        self.roster.count(1, "touches")
        self.roster.number_read(1, "23")
        self.assertEqual(self.roster.number_of(1), "23")

        self.roster.tracks_ended()
        self.see(7, ANNA)
        self.roster.match([7])
        self.assertEqual(self.roster.number_of(7), "23")  # Never read on this track
        self.roster.count(7, "touches")
        self.assertEqual(self.roster.table()[0]["touches"], 2.0)

    def test_what_a_player_did_is_theirs_once_their_number_is_read(self):
        self.see(1, ANNA)
        self.roster.match([1])
        self.roster.count(1, "touches", 3)
        self.assertEqual(self.roster.table()[0]["number"], "")
        self.roster.number_read(1, "8")
        row = self.roster.table()[0]
        self.assertEqual((row["number"], row["touches"]), ("8", 3.0))

    def test_two_entries_with_one_number_are_one_player(self):
        # The same player looked different enough in two points to get two entries
        self.see(1, ANNA)
        self.roster.match([1])
        self.roster.count(1, "touches", 2)
        self.roster.number_read(1, "8")
        self.roster.tracks_ended()
        self.see(1, CARL)
        self.roster.match([1])
        self.roster.count(1, "touches", 1)
        self.assertEqual(len(self.roster.entries), 2)
        self.roster.number_read(1, "8")
        self.assertEqual(len(self.roster.entries), 1)
        self.assertEqual(self.roster.table()[0]["touches"], 3.0)

    def test_the_roster_is_written_and_read_again(self):
        self.see(1, ANNA)
        self.see(2, BEN, DARK)
        self.roster.match([1, 2])
        self.roster.number_read(1, "23")
        self.roster.count(2, "seconds_with_disc", 4.5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roster.json"
            self.roster.save(path)
            again = self.module.PlayerRoster.load(path)
        self.assertEqual(again.table(), self.roster.table())
        # A player seen after loading is found in it
        for look in range(self.module.MIN_LOOKS):
            again.look(9, looks_like(1, 0, 0, noise=0.05, seed=look), WHITE)
        again.match([9])
        self.assertEqual(again.number_of(9), "23")


if __name__ == "__main__":
    unittest.main()
