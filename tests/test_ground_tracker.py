"""Players followed on the ground: lost ones are found again by their place on the field.

Unlike the other tests these run the real tracker, so they need Ultralytics.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ultralytics.trackers.basetrack import TrackState

    from ultimate_analysis.processing import ground_tracker, team_tracker
except ImportError:  # pragma: no cover - the app's environment has Ultralytics
    TrackState = None


@unittest.skipIf(TrackState is None, "Ultralytics is not installed")
class GroundTrackerTest(unittest.TestCase):
    def setUp(self):
        self.module, self.team_tracker = ground_tracker, team_tracker
        settings = self.team_tracker.tracker_settings(track_buffer=360)
        self.tracker = self.module.GroundTeamTracker(settings, picture_seconds=3.0)
        # A field seen straight from above: ten pixels to the yard
        self.to_field = np.diag([0.1, 0.1, 1.0])

    def boxes(self, *feet):
        """Player boxes 20 x 60 pixels standing on the given spots of the picture."""
        xyxy = [[x - 10, y - 60, x + 10, y] for x, y in feet]
        return self.team_tracker.DetectionBoxes(xyxy, [0.9] * len(xyxy))

    def step(self, *feet, field=True):
        self.tracker.look(self.to_field if field else None, 1 / 30)
        rows = self.tracker.update(self.boxes(*feet))
        return {tuple(np.round([(r[0] + r[2]) / 2, r[3]]).astype(int)): int(r[4]) for r in rows}

    def follow(self, frames, *feet, field=True):
        found = {}
        for _ in range(frames):
            found = self.step(*feet, field=field)
        return found

    def test_feet_are_placed_on_the_field(self):
        places = self.module.feet_on_field(self.to_field, [[90, 140, 110, 200]])
        np.testing.assert_allclose(places, [[10.0, 20.0]])

    def test_a_lost_player_is_found_again_where_they_can_have_got_to(self):
        first = self.follow(5, (100, 200), (400, 200))
        player = first[(100, 200)]
        self.follow(45, (400, 200))  # A second and a half unseen
        # Back four yards on: no overlap with the box they were lost in
        found = self.step((140, 200), (400, 200))
        self.assertEqual(found.get((140, 200)), player)

    def test_a_lost_player_is_not_found_further_off_than_they_can_run(self):
        first = self.follow(5, (100, 200), (400, 200))
        player = first[(100, 200)]
        self.follow(6, (400, 200))  # A fifth of a second
        found = self.follow(3, (300, 200), (400, 200))  # Twenty yards on
        self.assertNotEqual(found.get((300, 200)), player)

    def test_without_the_field_a_lost_player_is_found_as_by_the_team_tracker(self):
        first = self.follow(5, (100, 200), (400, 200), field=False)
        player = first[(100, 200)]
        self.follow(45, (400, 200), field=False)
        found = self.follow(3, (140, 200), (400, 200), field=False)
        self.assertNotEqual(found.get((140, 200)), player)  # No overlap, nothing to go by
        self.assertIsNone(self.tracker.lost_stracks[0].ground)

    def test_a_shift_of_the_field_moves_the_lost_players_places_along(self):
        crowd = [(300 + 40 * i, 300) for i in range(6)]
        self.follow(5, (100, 200), *crowd)
        self.follow(3, *crowd)
        lost = next(t for t in self.tracker.lost_stracks)
        before = lost.ground.copy()
        # The field's place is now judged two yards further along for everybody
        self.to_field = np.array([[0.1, 0, 2.0], [0, 0.1, 0], [0, 0, 1.0]])
        self.step(*crowd)
        self.assertGreater(lost.ground[0] - before[0], 0.5)

    def test_reach_grows_with_the_time_lost_up_to_a_limit(self):
        self.assertLess(self.module.reach_after(0.0), self.module.reach_after(1.0))
        self.assertEqual(self.module.reach_after(60.0), self.module.MAX_REACH)


if __name__ == "__main__":
    unittest.main()
