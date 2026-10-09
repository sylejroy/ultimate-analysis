"""Tracks joined into players after the fact."""

import unittest

import numpy as np
from support import load_module


def looks(*weights: float) -> np.ndarray:
    vector = np.array(weights, dtype=float)
    return vector / np.linalg.norm(vector)


class TrackletLinkingTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.tracklet_linking")
        self.book = self.module.TrackletBook()

    def track(self, track_id, first, last, vector, team=0, number="", start=None, end=None):
        for frame in (first, last):
            place = (start if frame == first else end) or (10.0, 10.0)
            self.book.see(frame, track_id, team, number, place)
        for _ in range(self.module.MIN_LOOKS):
            self.book.learn(track_id, vector)

    def test_a_track_that_looks_like_an_earlier_one_is_that_player(self):
        self.track(1, 0, 100, looks(1, 0, 0))
        self.track(2, 150, 300, looks(1, 0.1, 0))
        self.track(3, 0, 300, looks(0, 1, 0))
        naming = self.book.players(60.0)
        self.assertEqual(naming[2], 1)
        self.assertEqual(naming[3], 3)

    def test_tracks_seen_at_the_same_time_are_two_players(self):
        self.track(1, 0, 100, looks(1, 0, 0))
        self.track(2, 50, 300, looks(1, 0, 0))
        self.assertEqual(self.book.players(60.0), {1: 1, 2: 2})

    def test_teams_and_numbers_keep_tracks_apart(self):
        self.track(1, 0, 100, looks(1, 0, 0), team=0)
        self.track(2, 150, 300, looks(1, 0, 0), team=1)
        self.track(3, 0, 100, looks(0, 1, 0), number="7")
        self.track(4, 150, 300, looks(0, 1, 0), number="9")
        naming = self.book.players(60.0)
        self.assertEqual(len(set(naming.values())), 4)

    def test_the_same_number_in_a_team_is_one_player_whatever_the_looks(self):
        self.track(1, 0, 100, looks(1, 0, 0), number="7")
        self.track(2, 150, 300, looks(0, 1, 0), number="7")
        self.assertEqual(self.book.players(60.0)[2], 1)

    def test_nobody_is_joined_to_a_track_further_off_than_they_can_run(self):
        self.track(1, 0, 100, looks(1, 0, 0), end=(10.0, 10.0))
        self.track(2, 130, 300, looks(1, 0, 0), start=(60.0, 10.0))  # 50 yards in half a second
        self.assertEqual(self.book.players(60.0), {1: 1, 2: 2})
        self.track(3, 0, 100, looks(0, 1, 0), end=(10.0, 30.0))
        self.track(4, 130, 300, looks(0, 1, 0), start=(13.0, 30.0))
        self.assertEqual(self.book.players(60.0)[4], 3)

    def test_two_candidates_alike_are_left_alone(self):
        self.track(1, 0, 100, looks(1, 0, 0))
        self.track(2, 0, 100, looks(1, 0.05, 0))  # Two who look alike, at the same time
        self.track(3, 150, 300, looks(1, 0.02, 0))
        self.assertEqual(self.book.players(60.0)[3], 3)

    def test_three_pieces_of_one_player_are_not_each_other_s_rivals(self):
        self.track(1, 0, 100, looks(1, 0, 0))
        self.track(2, 150, 200, looks(1, 0.05, 0))
        self.track(3, 250, 300, looks(1, 0.02, 0))
        self.assertEqual(self.book.players(60.0), {1: 1, 2: 1, 3: 1})

    def test_a_track_the_tracker_renamed_is_known_under_both_names(self):
        self.track(1, 0, 100, looks(1, 0, 0))
        self.track(2, 150, 300, looks(0, 1, 0))
        self.book.rename(2, 1)
        naming = self.book.players(60.0)
        self.assertEqual((naming[1], naming[2]), (1, 1))

    def test_looks_alone_join_nobody_where_the_field_is_not_known(self):
        for track_id, (first, last) in enumerate(((0, 100), (150, 300)), 1):
            self.book.see(first, track_id, 0)
            self.book.see(last, track_id, 0)
            for _ in range(self.module.MIN_LOOKS):
                self.book.learn(track_id, looks(1, 0, 0))
        self.assertEqual(self.book.players(60.0), {1: 1, 2: 2})

    def test_a_track_hardly_looked_at_is_left_alone(self):
        self.track(1, 0, 100, looks(1, 0, 0))
        self.book.see(150, 2, 0)
        self.book.learn(2, looks(1, 0, 0))
        self.assertEqual(self.book.players(60.0)[2], 2)


if __name__ == "__main__":
    unittest.main()
