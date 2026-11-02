"""Drone footage is told from close-ups by how many players the detector finds."""

import unittest

import numpy as np
from support import load_module


class CutWithAMotionTest(unittest.TestCase):
    """Two views of one field can give a motion: most is asked of a cut then."""

    SECOND = 1 / 30

    def setUp(self):
        self.is_a_cut = load_module("processing.shot_type").is_a_cut

    @staticmethod
    def boxes(*places):
        return [[x, y, x + 40, y + 90] for x, y in places]

    def row(self, x=100, y=300, count=8):
        return self.boxes(*[(x + 150 * i, y) for i in range(count)])

    def test_a_cut_with_a_small_motion_found_is_one(self):
        # As Chicago v San Francisco, frame 219: 13 players, then 6, with 3 px of motion
        before, now = self.row(count=13), self.row(170, 600, 6)
        self.assertTrue(self.is_a_cut(before, now, np.eye(3), 2 * self.SECOND))

    def test_a_cut_without_a_motion_is_told_as_ever(self):
        self.assertTrue(self.is_a_cut(self.row(), self.row(170, 600, 2), None, self.SECOND))
        self.assertTrue(self.is_a_cut(self.row(), [], None, self.SECOND))

    def test_a_swing_with_few_players_left_is_no_cut(self):
        # The drone swings after a long throw: 8 players, then 2 somewhere else
        self.assertFalse(self.is_a_cut(self.row(), self.row(170, 600, 2), np.eye(3), self.SECOND))

    def test_a_frame_without_players_with_a_motion_is_no_cut(self):
        self.assertFalse(self.is_a_cut(self.row(), [], np.eye(3), self.SECOND))

    def test_a_hitch_is_no_cut(self):
        # Half a second later nobody stands where they stood, whatever the motion
        for motion in (np.eye(3), None):
            self.assertFalse(self.is_a_cut(self.row(), self.row(170, 600), motion, 0.5))
            self.assertTrue(self.is_a_cut(self.row(), self.row(170, 600), motion, 0.05))

    def test_a_pan_the_motion_explains_is_no_cut(self):
        # Half way to the next player: by the picture alone, nobody stayed
        pan = np.array([[1.0, 0.0, 375.0], [0.0, 1.0, -40.0], [0.0, 0.0, 1.0]])
        self.assertFalse(self.is_a_cut(self.row(), self.row(480, 262), pan, self.SECOND))

    def test_a_wrong_motion_on_steady_play_is_no_cut(self):
        # As San Francisco v Colorado, frames 73970 and 75584: everybody stays by the
        # picture, and the motion found sends them elsewhere
        wrong = np.array([[1.0, 0.0, 75.0], [0.0, 1.0, 200.0], [0.0, 0.0, 1.0]])
        self.assertFalse(self.is_a_cut(self.row(), self.row(), wrong, self.SECOND))


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

    def test_a_swing_of_the_camera_with_few_players_in_view_is_still_the_drone(self):
        self.watcher.update(14, continuous=True)
        kinds = [self.watcher.update(4, continuous=True) for _ in range(60)]
        self.assertTrue(all(kinds))
        # ... but not for ever: an unbroken fade to a title card ends it too
        kinds = [self.watcher.update(0, continuous=True) for _ in range(200)]
        self.assertFalse(kinds[-1])

    def test_few_players_after_the_picture_broke_is_a_cut_away(self):
        self.watcher.update(14, continuous=True)
        self.watcher.update(1, continuous=False)
        kinds = [self.watcher.update(1, continuous=True) for _ in range(20)]
        self.assertFalse(kinds[-1])

    def test_after_a_cut_seen_to_be_drone_footage_one_poor_frame_changes_nothing(self):
        self.watcher.reset()  # What everything forgotten leaves
        self.assertFalse(self.watcher.update(2))  # ... the next frame decides by itself
        self.watcher.reset()
        self.watcher.go_on()
        self.assertTrue(self.watcher.update(2, continuous=True))


class CutTest(unittest.TestCase):
    def setUp(self):
        self.elsewhere = load_module("processing.shot_type").players_are_elsewhere

    @staticmethod
    def boxes(*places):
        return [[x, y, x + 40, y + 90] for x, y in places]

    def test_players_who_moved_a_little_are_where_they_were(self):
        before = self.boxes(*[(100 + 150 * i, 300) for i in range(8)])
        now = self.boxes(*[(110 + 150 * i, 305) for i in range(7)])  # One is missed
        self.assertFalse(self.elsewhere(before, now))

    def test_after_a_cut_the_players_are_elsewhere(self):
        before = self.boxes(*[(100 + 150 * i, 300) for i in range(8)])
        now = self.boxes(*[(170 + 150 * i, 600) for i in range(8)])
        self.assertTrue(self.elsewhere(before, now))
        self.assertTrue(self.elsewhere(before, []))

    def test_boxes_move_by_a_homography_corner_by_corner(self):
        moved_with_camera = load_module("processing.shot_type").moved_with_camera
        zoom = np.diag([2.0, 2.0, 1.0])
        self.assertEqual(moved_with_camera([[1, 2, 3, 4]], zoom).tolist(), [[2, 4, 6, 8]])

    def test_a_few_players_tell_nothing(self):
        before = self.boxes((100, 300), (300, 300))
        self.assertFalse(self.elsewhere(before, self.boxes((600, 600), (900, 600))))


if __name__ == "__main__":
    unittest.main()
