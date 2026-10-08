"""Drone footage is told from close-ups by how many players the detector finds."""

import unittest

from support import load_module


class ShotWatcherTest(unittest.TestCase):
    def setUp(self):
        self.watcher = load_module("processing.shot_type").ShotWatcher()
        self.watcher.reset()

    def test_the_first_frame_decides_by_itself(self):
        self.assertFalse(self.watcher.update(0))

    def test_single_frames_do_not_change_the_kind(self):
        self.assertTrue(self.watcher.update(14))
        # Players are missed in a frame or two
        for count in (3, 2, 14, 14, 0):
            self.assertTrue(self.watcher.update(count))

    def test_a_cut_is_believed_after_some_frames_and_so_is_the_way_back(self):
        self.watcher.update(14)
        kinds = [self.watcher.update(1) for _ in range(20)]
        self.assertTrue(kinds[0])
        self.assertFalse(kinds[-1])
        kinds = [self.watcher.update(13) for _ in range(20)]
        self.assertFalse(kinds[0])
        self.assertTrue(kinds[-1])

    def test_after_a_pause_one_frame_with_players_is_not_the_drone_yet(self):
        self.watcher.update(14)
        self.watcher.reset()  # What a pipeline reset does on the way into a pause
        self.watcher.pause()
        self.assertFalse(self.watcher.update(14))
        kinds = [self.watcher.update(14) for _ in range(20)]
        self.assertTrue(kinds[-1])


if __name__ == "__main__":
    unittest.main()
