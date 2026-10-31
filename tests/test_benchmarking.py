"""What the benchmark scripts share: the games drawn again, --set, the run's header, the
frames around a labelled frame."""

import math
import unittest
from unittest.mock import patch

from support import load_module


class FakeCapture:
    """A video of `length` frames; each frame is its own number."""

    def __init__(self, length: int = 100, rate: float = 25.0):
        self.length, self.rate, self.at = length, rate, 0
        self.read_frames, self.grabbed = [], []

    def get(self, _property):
        return self.rate

    def set(self, _property, position):
        self.at = int(position)

    def grab(self):
        self.grabbed.append(self.at)
        self.at += 1
        return self.at <= self.length

    def read(self):
        if self.at >= self.length:
            return False, None
        self.read_frames.append(self.at)
        self.at += 1
        return True, self.at - 1

    def release(self):
        pass


class DrawnAgainTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("utils.benchmarking")

    def test_a_share_lies_near_itself_with_many_groups(self):
        # 200 games of five frames, half of them right: the share is 0.5 and the games
        # drawn again move it little
        values = [1, 0] * 500
        groups = [index // 5 for index in range(1000)]
        mean, low, high = self.module.drawn_again(values, groups)
        self.assertEqual(mean, 0.5)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertGreater(low, 0.4)
        self.assertLess(high, 0.6)

    def test_all_equal_values_give_a_zero_width_interval(self):
        self.assertEqual(self.module.drawn_again([1, 1, 1, 1], ["a", "a", "b", "b"]), (1, 1, 1))
        self.assertEqual(self.module.drawn_again([0.25] * 10), (0.25, 0.25, 0.25))

    def test_the_mean_is_of_the_values_and_the_draw_is_of_the_groups(self):
        # Two groups: the mean counts every value, and a draw of one group alone gives
        # that group's own share, so the interval reaches both
        mean, low, high = self.module.drawn_again([1, 1, 1, 1, 0], ["a", "a", "a", "a", "b"])
        self.assertEqual(mean, 0.8)
        self.assertEqual((low, high), (0.0, 1.0))

    def test_groups_may_be_named_by_anything_hashable(self):
        groups = [("video", 1), ("video", 1), ("video", 2)]
        mean, low, high = self.module.drawn_again([2.0, 4.0, 9.0], groups)
        self.assertEqual(mean, 5.0)
        self.assertEqual((low, high), (3.0, 9.0))

    def test_the_same_values_give_the_same_interval(self):
        values = [float(index % 7) for index in range(40)]
        self.assertEqual(self.module.drawn_again(values), self.module.drawn_again(values))

    def test_nothing_gives_nan(self):
        self.assertTrue(all(math.isnan(v) for v in self.module.drawn_again([])))

    def test_the_text(self):
        self.assertEqual(
            self.module.with_spread((0.8, 0.76, 0.83)),
            "80% (76% to 83% with the games drawn again)",
        )
        self.assertEqual(
            self.module.with_spread((0.852, 0.755, 0.948), "stretches", ".3f"),
            "0.852 (0.755 to 0.948 with the stretches drawn again)",
        )
        self.assertEqual(
            self.module.with_spread((3.14, 2.8, 3.5), "images", ".1f", " px"),
            "3.1 px (2.8 to 3.5 px with the images drawn again)",
        )
        self.assertEqual(self.module.with_spread((0.8, 0.76, 0.83), ""), "80% (76% to 83%)")


class ApplySettingsTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("utils.benchmarking")
        self.config = {"models": {"possession": {"reach": 1.0, "on": True}}, "video": {"fps": 25}}
        patcher = patch.object(self.module, "get_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_nested_key_is_changed_and_written_down(self):
        changed = self.module.apply_settings(["models.possession.reach=0.5", "video.fps=30"])
        self.assertEqual(self.config["models"]["possession"]["reach"], 0.5)
        self.assertEqual(self.config["video"]["fps"], 30)
        self.assertEqual(changed, ["models.possession.reach=0.5", "video.fps=30"])

    def test_the_value_is_json(self):
        self.module.apply_settings(["models.possession.on=false", "models.possession.reach=[1, 2]"])
        self.assertIs(self.config["models"]["possession"]["on"], False)
        self.assertEqual(self.config["models"]["possession"]["reach"], [1, 2])
        with self.assertRaises(ValueError) as said:
            self.module.apply_settings(["models.possession.reach=half"])
        self.assertIn("not JSON", str(said.exception))

    def test_an_unknown_key_is_refused(self):
        for item in ("models.possession.reech=0.5", "models.posession.reach=0.5", "reach=0.5"):
            with self.assertRaises(ValueError) as said:
                self.module.apply_settings([item])
            self.assertIn("no setting", str(said.exception))
            self.assertIn(item.split("=")[0], str(said.exception))
        self.assertEqual(self.config["models"]["possession"]["reach"], 1.0)

    def test_the_header_names_the_commit_the_settings_and_the_command(self):
        with patch.object(self.module, "git_commit", return_value="abc1234+changes"):
            lines = self.module.run_header(["video.fps=30"])
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("Commit abc1234+changes, "))
        self.assertEqual(lines[1], "Settings changed: video.fps=30")
        self.assertTrue(lines[2].startswith("Command: "))
        self.assertEqual(self.module.run_header()[1], "Settings changed: none")


class FramesAroundTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("utils.benchmarking")
        self.capture = FakeCapture()
        patcher = patch.object(self.module.cv2, "VideoCapture", return_value=self.capture)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_so_many_frames_with_the_labelled_frame_among_them(self):
        # At 25 fps, 0.1 s is two frames either side of frame 11; with a step of 2 the
        # frames before are rounded down to the step, and the start keeps the step
        frames = list(self.module.frames_around("v.mp4", 11, 0.1, 0.1, step=2))
        self.assertEqual(frames, [(9, 9), (11, 11), (13, 13)])
        # The frames passed over are not decoded
        self.assertEqual(self.capture.read_frames, [9, 11, 13])
        self.assertEqual(self.capture.grabbed, [10, 12])

    def test_the_start_is_not_before_the_video(self):
        frames = [number for number, _ in self.module.frames_around("v.mp4", 1, 1.0, 0.0, step=2)]
        self.assertEqual(frames, [1])
        frames = [number for number, _ in self.module.frames_around("v.mp4", 1, 1.0, 0.0)]
        self.assertEqual(frames, [0, 1])

    def test_it_stops_where_the_video_ends(self):
        frames = [number for number, _ in self.module.frames_of("v.mp4", 97, 110)]
        self.assertEqual(frames, [97, 98, 99])
        frames = [number for number, _ in self.module.frames_of("v.mp4", 96, 110, step=3)]
        self.assertEqual(frames, [96, 99])


if __name__ == "__main__":
    unittest.main()
