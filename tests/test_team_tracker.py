"""The team rule of the player tracker, on drawn frames with the real ByteTrack."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ultimate_analysis.processing import team_tracker
except ImportError:  # Ultralytics is not installed
    team_tracker = None

GRASS, WHITE, DARK = (60, 140, 70), (235, 235, 235), (60, 30, 30)
SIZE = (40, 90)  # Width and height of a player


def box(x, y=200):
    return [x, y, x + SIZE[0], y + SIZE[1]]


def frame_with(*players):
    """players: (x, colour) -> a frame showing them, and their boxes."""
    frame = np.full((480, 900, 3), GRASS, dtype=np.uint8)
    for x, colour in players:
        x1, y1, x2, y2 = (int(value) for value in box(x))
        frame[y1:y2, x1:x2] = colour
    return frame, [box(x) for x, _ in players]


@unittest.skipIf(team_tracker is None, "needs Ultralytics")
class TeamTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = team_tracker.TeamTracker(team_tracker.tracker_settings(90))

    def see(self, *players):
        """{position in `players`: track ID} of the players that were given a track."""
        frame, boxes = frame_with(*players)
        rows = self.tracker.update(team_tracker.DetectionBoxes(boxes, [0.9] * len(boxes)), frame)
        return {int(row[7]): int(row[4]) for row in rows}

    def learn_the_teams(self):
        # Four players standing apart, long enough for the two colours to be learned
        for _ in range(40):
            ids = self.see((100, WHITE), (300, WHITE), (500, DARK), (700, DARK))
        self.assertIsNotNone(self.tracker.team_colours)
        return ids

    def test_the_two_shirt_colours_are_learned_and_tracks_get_their_team(self):
        self.learn_the_teams()
        teams = [track.team for track in self.tracker.tracked_stracks]
        self.assertEqual(sorted(teams), [0, 0, 1, 1])

    def test_a_track_is_not_continued_by_a_player_of_the_other_team(self):
        ids = self.learn_the_teams()
        # The white player at 300 vanishes and a dark one stands exactly there instead:
        # by motion alone the track would simply carry on
        for _ in range(5):
            now = self.see((100, WHITE), (300, DARK), (500, DARK), (700, DARK))
        self.assertNotEqual(now.get(1), ids[1])
        self.assertEqual(now[0], ids[0])

    def test_a_player_in_their_own_colours_keeps_the_track(self):
        ids = self.learn_the_teams()
        for step in range(1, 30):
            now = self.see((100, WHITE), (300 + 3 * step, WHITE), (500, DARK), (700, DARK))
        self.assertEqual(now, ids)

    def test_someone_in_neither_teams_colours_is_an_outsider(self):
        orange = (30, 130, 250)
        # Three players a side and one observer, as few as the observers are in a game
        people = [(40 + 120 * place, WHITE if place < 3 else DARK) for place in range(6)]
        for _ in range(60):
            ids = self.see(*people, (780, orange))
        self.assertEqual(self.tracker.outsiders(), {ids[6]})

    def test_feet_distances_are_in_body_heights(self):
        tracks = np.array([box(0)], dtype=np.float64)
        detections = np.array([box(0), box(0, 200 + SIZE[1] / 2), box(500)], dtype=np.float64)
        np.testing.assert_allclose(
            team_tracker.feet_distances(tracks, detections), [[0.0, 0.5, 1.0]], atol=1e-6
        )

    def test_largest_overlaps(self):
        boxes = np.array([box(0), box(20), box(500)], dtype=np.float32)
        overlaps = team_tracker.largest_overlaps(boxes)
        self.assertAlmostEqual(overlaps[0], 1 / 3, places=3)
        self.assertEqual(overlaps[2], 0.0)


if __name__ == "__main__":
    unittest.main()
