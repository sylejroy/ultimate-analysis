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

    def test_a_date_speaks_of_the_frames_since_it_but_not_of_a_name_or_a_reset(self):
        said = [(10, None, 0, None), (12, 1, 11, 0), (14, 1, 11, 0)]
        self.assertEqual(self.module.spoken_from(said), [10, 11, 11])
        # The tracker joins the holder with an earlier track: frames 12 and 14 keep the name
        said += [(16, 7, 11, 0), (18, 7, 11, 0), (20, 2, 19, 0)]
        self.assertEqual(self.module.spoken_from(said)[3:], [16, 16, 19])
        # A cut: the reset says nothing of the frames before it
        self.assertEqual(self.module.spoken_from(said + [(22, None, 0, None)])[-1], 22)

    def test_frames_keep_the_holder_they_had_when_a_cut_or_a_new_name_comes(self):
        said = [(10, 1, 8, 0), (12, 1, 8, 0), (14, 7, 8, 0), (16, None, 0, None), (18, 2, 17, 0)]
        holders = {f: spoken[1] for f, spoken in self.module.said_of_frames(said).items()}
        # Not the holder of the frames after the cut, nor the name they got later
        self.assertEqual(holders, {10: 1, 12: 1, 14: 7, 16: None, 18: 2})

    def test_the_holder_found_late_holds_from_when_it_began(self):
        said = [(10, None, 5, None), (12, None, 5, None), (14, None, 5, None), (16, 3, 12, 0)]
        holders = {f: spoken[1] for f, spoken in self.module.said_of_frames(said).items()}
        self.assertEqual(holders, {10: None, 12: 3, 14: 3, 16: 3})

    def test_a_replay_follows_the_settings_of_the_moment(self):
        calls, said = self.recorded()
        self.settings["models.possession.confirm_seconds"] = 5.0
        self.assertEqual({holder for _, holder, _, _ in self.module.replay(calls)}, {None})


if __name__ == "__main__":
    unittest.main()
