"""The tracker's IDs and joins as benchmark_tracking.py scores them, across restarts."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class FakePipeline:
    """IDs from 1 in each stretch between restarts; a join known only until the next one."""

    def __init__(self):
        self.restarts = 0
        self.joined = {}

    def new_video(self, path):
        pass

    def set_frame_rate(self, rate):
        pass

    def reset(self):
        self.restarts += 1
        self.joined = {}

    def players_of_tracks(self):
        return dict(self.joined)

    def process(self, frame, number, options):
        if number == 2:
            self.joined = {2: 1}  # Before the restart: track 2 is player 1
        if number == 3:
            self.reset()  # As a cut or a close-up does, on the pipeline itself
        if number == 4:
            self.joined = {3: 1}
        track = SimpleNamespace(
            track_id=1, class_name="player", to_ltrb=lambda: [number, 0.0, 10.0, 10.0]
        )
        return SimpleNamespace(tracks=[track])


class RunTrackerTest(unittest.TestCase):
    def setUp(self):
        import benchmark_tracking

        self.module = benchmark_tracking
        patches = {
            "AnalysisPipeline": FakePipeline,
            "PipelineOptions": lambda: None,
            "frame_rate_of": lambda path: 30.0,
            "frames_of": lambda path, first, last, step: (
                (n, None) for n in range(first, last + 1)
            ),
        }
        for name, value in patches.items():
            patcher = patch.object(self.module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_one_number_either_side_of_a_restart_is_two_ids(self):
        found, _ = self.module.run_tracker("video", 0, 5)
        before, after = found[0][0][0], found[4][0][0]
        self.assertEqual(before, found[2][0][0])
        self.assertEqual(after, found[3][0][0])  # From the frame it happened in on
        self.assertEqual(after - before, self.module.AFTER_A_CUT)

    def test_joins_of_every_stretch_are_kept_each_under_its_own_ids(self):
        _, naming = self.module.run_tracker("video", 0, 5)
        at = self.module.AFTER_A_CUT
        self.assertEqual(naming, {2: 1, at + 3: at + 1})


if __name__ == "__main__":
    unittest.main()
