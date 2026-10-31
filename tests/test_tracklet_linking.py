"""Tracks joined into players after the fact."""

import unittest
from unittest.mock import patch

import numpy as np
from support import load_module


def looks(*weights: float) -> np.ndarray:
    vector = np.array(weights, dtype=float)
    return vector / np.linalg.norm(vector)


def connect_pair_by_pair(module, tracklets, frame_rate, min_likeness, min_lead, needs_place):
    """`connect` as it was before the pairs were kept from one join to the next: what
    it gave is what the quick one must give."""
    groups = [[t] for t in tracklets]

    def compatible(one, other):
        return all(module.may_be_one_player(a, b, frame_rate) for a in one for b in other)

    def placed(one, other):
        return all(module.reach_known(a, b) for a in one for b in other)

    def numbered_alike(one, other):
        numbers = {t.number for t in one if t.number}, {t.number for t in other if t.number}
        teams = {t.team for t in one + other if t.team is not None}
        return bool(numbers[0] & numbers[1]) and len(teams) == 1

    while len(groups) > 1:
        count = len(groups)
        allowed = np.zeros((count, count), dtype=bool)
        for i in range(count):
            for j in range(i + 1, count):
                allowed[i, j] = allowed[j, i] = compatible(groups[i], groups[j])
        pair = next(
            (
                (i, j)
                for i in range(count)
                for j in range(i + 1, count)
                if allowed[i, j] and numbered_alike(groups[i], groups[j])
            ),
            None,
        )
        if pair is None:
            means = [module._mean(group) for group in groups]
            likeness = np.full((count, count), -1.0)
            for i in range(count):
                for j in range(i + 1, count):
                    if not allowed[i, j] or means[i] is None or means[j] is None:
                        continue
                    if not needs_place or placed(groups[i], groups[j]):
                        likeness[i, j] = likeness[j, i] = float(means[i] @ means[j])
            best = 0.0
            for i in range(count):
                for j in range(i + 1, count):
                    if likeness[i, j] < max(min_likeness, best):
                        continue
                    rivals = [
                        likeness[a, c]
                        for a, b in ((i, j), (j, i))
                        for c in range(count)
                        if c not in (i, j) and not allowed[b, c]
                    ]
                    if likeness[i, j] - max(rivals, default=-1.0) >= min_lead:
                        best, pair = likeness[i, j], (i, j)
        if pair is None:
            break
        i, j = pair
        groups[i] = groups[i] + groups[j]
        del groups[j]
    return groups


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
        # ... unless asked to (models.reid.join_needs_place)
        settings = {"models.reid.join_needs_place": False}
        with patch.object(
            self.module, "get_setting", side_effect=lambda key, d=None: settings.get(key, d)
        ):
            self.assertEqual(self.book.players(60.0), {1: 1, 2: 1})

    def test_joining_pair_by_pair_or_keeping_the_pairs_gives_the_same_groups(self):
        module = self.module
        for seed in range(12):
            rng = np.random.default_rng(seed)
            players = [looks(*rng.normal(size=6)) for _ in range(5)]
            tracklets = []
            for track_id in range(1, 41):
                player = int(rng.integers(5))
                first = int(rng.integers(0, 2000))
                tracklet = module.Tracklet(track_id, first, first + int(rng.integers(5, 200)))
                if rng.random() < 0.8:
                    tracklet.team_sightings[player % 2] = 10
                if rng.random() < 0.3:
                    tracklet.number = str(player) if rng.random() < 0.9 else "99"
                if rng.random() < 0.8:
                    tracklet.first_place = rng.uniform(0, 100, size=2)
                    tracklet.last_place = tracklet.first_place + rng.normal(size=2)
                for _ in range(int(rng.integers(0, 6))):
                    tracklet.learn(players[player] + rng.normal(scale=0.2, size=6))
                tracklets.append(tracklet)
            for needs_place in (True, False):
                with self.subTest(seed=seed, needs_place=needs_place):
                    slow = connect_pair_by_pair(module, tracklets, 30.0, 0.5, 0.1, needs_place)
                    quick = module.connect(tracklets, 30.0, 0.5, 0.1, needs_place)
                    self.assertEqual(
                        [[t.track_id for t in group] for group in quick],
                        [[t.track_id for t in group] for group in slow],
                    )

    def test_a_track_hardly_looked_at_is_left_alone(self):
        self.track(1, 0, 100, looks(1, 0, 0))
        self.book.see(150, 2, 0)
        self.book.learn(2, looks(1, 0, 0))
        self.assertEqual(self.book.players(60.0)[2], 2)


if __name__ == "__main__":
    unittest.main()
