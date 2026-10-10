"""A recorded run of the possession tracker played back says what the run said."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from support import load_module


def player(track_id, x1, y1, x2, y2, team=None):
    box = [x1, y1, x2, y2]
    return SimpleNamespace(track_id=track_id, class_name="player", to_ltrb=lambda: box, team=team)


def disc(x, y):
    return {"class_name": "disc", "confidence": 0.8, "bbox": [x - 5, y - 5, x + 5, y + 5]}


class PossessionReplayTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.possession_replay")
        self.settings = {"models.possession.confirm_seconds": 0.3}
        possession = load_module("processing.possession")
        patcher = patch.object(
            possession,
            "get_setting",
            side_effect=lambda key, default=None: self.settings.get(key, default),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def recorded(self):
        """A catch, a throw and a catch by another, with a renamed track on the way."""
        tracker = self.module.RecordingPossessionTracker()
        tracker.frame_rate = 10.0
        tracks = [player(1, 100, 100, 160, 300, team=0), player(2, 600, 100, 660, 300, team=1)]
        said = []
        for frame, x in enumerate((130, 130, 130, 130, 400, 400, 400, 630, 630, 630, 630)):
            if frame == 8:
                tracker.rename(2, 7)
                tracks[1] = player(7, 600, 100, 660, 300, team=1)
            tracker.update(
                [disc(x, 150), {"class_name": "player", "bbox": [0, 0, 1, 1]}], tracks, frame
            )
            tracker.frame_done(frame)
            said.append((frame, tracker.holder_id, tracker.since, tracker.team))
        return tracker.calls, said

    def test_a_replay_says_what_the_run_said(self):
        calls, said = self.recorded()
        self.assertEqual({holder for _, holder, _, _ in said}, {None, 1, 7})
        self.assertEqual(self.module.replay(calls), said)

    def test_a_replay_follows_the_settings_of_the_moment(self):
        calls, said = self.recorded()
        self.settings["models.possession.confirm_seconds"] = 5.0
        self.assertEqual({holder for _, holder, _, _ in self.module.replay(calls)}, {None})


if __name__ == "__main__":
    unittest.main()
