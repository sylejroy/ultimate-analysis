"""Video metadata and the frame reader."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from support import load_module


class VideoPlayerTests(unittest.TestCase):
    def test_metadata_opens_video_only_once(self):
        module = load_module("utils.video")
        cap = Mock()
        properties = {
            module.cv2.CAP_PROP_FPS: 25,
            module.cv2.CAP_PROP_FRAME_COUNT: 1500,
            module.cv2.CAP_PROP_FRAME_WIDTH: 1920,
            module.cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }
        cap.get.side_effect = properties.__getitem__
        with patch.object(module.cv2, "VideoCapture", return_value=cap) as open_video:
            info = module.get_video_info("test.mp4")
        self.assertEqual(info["duration_formatted"], "01:00")
        open_video.assert_called_once_with("test.mp4")
        cap.release.assert_called_once()

    def test_metadata_exception_releases_capture(self):
        module = load_module("utils.video")
        cap = Mock()
        cap.get.side_effect = ValueError("invalid metadata")
        with patch.object(module.cv2, "VideoCapture", return_value=cap):
            self.assertIsNone(module.get_video_info("test.mp4"))
        cap.release.assert_called_once()

    def test_failed_open_releases_capture(self):
        module = load_module("utils.video")
        cap = Mock()
        cap.isOpened.return_value = False
        with (
            patch.object(module.Path, "exists", return_value=True),
            patch.object(module.cv2, "VideoCapture", return_value=cap),
        ):
            player = module.VideoPlayer()
            self.assertFalse(player.load_video("test.mp4"))
        cap.release.assert_called_once()
        self.assertIsNone(player.cap)

    def test_failed_seek_does_not_change_position(self):
        module = load_module("utils.video")
        player = module.VideoPlayer()
        player.cap = Mock()
        player.cap.set.return_value = False
        player.total_frames = 100
        player.current_frame_idx = 5
        self.assertFalse(player.seek_to_frame(20))
        self.assertEqual(player.current_frame_idx, 5)

    def test_with_a_step_the_frames_between_are_passed_over(self):
        module = load_module("utils.video")

        class Film:
            """Hands out its position as the frame."""

            def __init__(self):
                self.position = 0

            def isOpened(self):
                return True

            def release(self):
                pass

            def grab(self):
                self.position += 1
                return True

            def read(self):
                self.position += 1
                return True, np.full((1, 1), self.position - 1)

        player = module.VideoPlayer()
        player.cap = Film()
        player.set_frame_step(2)
        handed_out = []
        for _ in range(3):
            handed_out.append((player.current_frame_idx, int(player.get_next_frame()[0, 0])))
        # Every frame again from here: what was decoded ahead is the frame at the position
        player.set_frame_step(1)
        for _ in range(3):
            handed_out.append((player.current_frame_idx, int(player.get_next_frame()[0, 0])))
        self.assertEqual(handed_out, [(0, 0), (2, 2), (4, 4), (6, 6), (7, 7), (8, 8)])

    def test_decode_ahead_returns_frames_in_order_and_is_dropped_on_seek(self):
        module = load_module("utils.video")
        player = module.VideoPlayer()
        frames = iter(np.full((2, 2, 3), value, dtype=np.uint8) for value in range(10))
        player.cap = Mock()
        player.cap.read.side_effect = lambda: (True, next(frames))
        player.total_frames = 100

        self.assertEqual(player.get_next_frame()[0, 0, 0], 0)
        # The paused view shows the frame at the current position: the one decoded ahead
        self.assertEqual(player.get_current_frame()[0, 0, 0], 1)
        self.assertEqual(player.get_next_frame()[0, 0, 0], 1)
        self.assertEqual(player.current_frame_idx, 2)

        # Frame 2 was decoded ahead; after a seek it must not be returned
        self.assertTrue(player.seek_to_frame(50))
        self.assertIsNone(player._decode_ahead)
        self.assertEqual(player.get_next_frame()[0, 0, 0], 3)
        self.assertEqual(player.current_frame_idx, 51)
        player.cap = None


if __name__ == "__main__":
    unittest.main()
