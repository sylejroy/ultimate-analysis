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

    def test_tracks_are_given_the_shirt_colour_of_their_team(self):
        ids = self.learn_the_teams()
        teams, colours = self.tracker.teams_of_tracks(), self.tracker.shirt_colours()
        self.assertEqual(teams[ids[0]], teams[ids[1]])
        self.assertEqual(teams[ids[2]], teams[ids[3]])
        self.assertNotEqual(teams[ids[0]], teams[ids[2]])
        for track_id, shirt in ((ids[0], WHITE), (ids[2], DARK)):
            self.assertLess(np.abs(np.subtract(colours[teams[track_id]], shirt)).max(), 12)

    def test_the_colour_a_team_is_shown_in_is_held_once_enough_shirts_were_seen(self):
        self.learn_the_teams()
        for _ in range(260):
            self.see((100, WHITE), (300, WHITE), (500, DARK), (700, DARK))
        shown = dict(self.tracker.shirt_colours())
        # The light changes: shirts are seen darker from now on
        for _ in range(60):
            self.see((100, (200, 200, 200)), (300, (200, 200, 200)), (500, DARK), (700, DARK))
        self.assertEqual(self.tracker.shirt_colours(), shown)
        self.tracker.reset()
        self.assertEqual(self.tracker.shirt_colours(), {})

    def test_the_grass_beside_a_player_is_not_in_the_colour_a_team_is_shown_in(self):
        # A slim player: much of what counts as the shirt in the box is grass
        frame, boxes = frame_with((100, WHITE))
        x1, y1, x2, y2 = (int(value) for value in boxes[0])
        frame[y1:y2, x1 : x1 + 14] = GRASS
        frame[y1:y2, x2 - 14 : x2] = GRASS
        colour, left = team_tracker.appearance.shirt_colour(frame, boxes[0])
        self.assertLess(np.abs(colour - WHITE).max(), 12)
        self.assertGreater(left, 0.3)
        with_grass = team_tracker.appearance.encode(frame, boxes)[0][:3]
        self.assertGreater(abs(float(with_grass[1]) - 128), 5)  # Greenish, in Lab
        # Nothing but grass tells no shirt
        frame[y1:y2, x1:x2] = GRASS
        self.assertIsNone(team_tracker.appearance.shirt_colour(frame, boxes[0])[0])

    def test_a_team_in_green_is_shown_in_green_not_in_the_grey_of_its_print(self):
        # Green like grass, with a grey number on the chest
        def green_shirts(*places):
            frame, boxes = frame_with(*[(x, WHITE if white else GRASS) for x, white in places])
            for (x, white), shirt in zip(places, boxes):
                if not white:
                    x1, y1, x2, y2 = (int(value) for value in shirt)
                    frame[y1 + 22 : y1 + 34, x1 + 14 : x1 + 26] = (120, 120, 120)
                    frame[y1:y2, x1 - 3 : x1] = (20, 20, 20)  # Told from the grass around
                    frame[y1:y2, x2 : x2 + 3] = (20, 20, 20)
            return frame, boxes

        for _ in range(140):
            frame, boxes = green_shirts((100, True), (300, True), (500, False), (700, False))
            self.tracker.update(team_tracker.DetectionBoxes(boxes, [0.9] * 4), frame)
        colours = sorted(self.tracker.shirt_colours().values())
        self.assertEqual(len(colours), 2)
        blue, green, red = colours[0]  # The darker of the two
        self.assertGreater(green, 1.3 * max(blue, red))

    def test_a_dark_green_shirt_in_the_sun_is_still_green_against_white(self):
        green, shining = (40, 90, 20), (125, 185, 105)
        for _ in range(40):
            self.see((100, WHITE), (300, WHITE), (500, green), (700, green))
        teams = self.tracker.teams_of_tracks()
        self.assertEqual(len(set(teams.values())), 2)
        lab = team_tracker.appearance.encode(*frame_with((100, shining)))[0][:3]
        green_lab = team_tracker.appearance.encode(*frame_with((100, green)))[0][:3]
        white_lab = team_tracker.appearance.encode(*frame_with((100, WHITE)))[0][:3]
        # Lighter than halfway to white, and still the green team's
        self.assertGreater(lab[0], (green_lab[0] + white_lab[0]) / 2)
        self.assertEqual(self.tracker._team_of(lab), self.tracker._team_of(green_lab))
        self.assertNotEqual(self.tracker._team_of(lab), self.tracker._team_of(white_lab))

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

    def test_an_observer_is_told_from_the_players_in_a_single_frame(self):
        orange = (30, 130, 250)
        people = [(20 + 95 * place, WHITE if place < 4 else DARK) for place in range(8)]
        frame, boxes = frame_with(*people, (800, orange))
        self.assertEqual(team_tracker.observers_in_frame(frame, boxes), [False] * 8 + [True])
        # Too few people to tell the teams: nobody is taken for an observer
        frame, boxes = frame_with((100, WHITE), (300, DARK), (500, orange))
        self.assertEqual(team_tracker.observers_in_frame(frame, boxes), [False] * 3)

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
