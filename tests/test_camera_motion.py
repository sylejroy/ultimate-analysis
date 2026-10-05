"""Camera motion: the background motion is recovered and the players do not disturb it."""

import unittest

import cv2
import numpy as np
from support import load_module


def textured_scene(seed=0, size=(1400, 2400)):
    """A random blocky texture, larger than a frame so that it can be panned."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (size[0] // 20, size[1] // 20), dtype=np.uint8)
    scene = cv2.resize(coarse, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(cv2.GaussianBlur(scene, (5, 5), 0), cv2.COLOR_GRAY2BGR)


def view(scene, x, y, width=1280, height=720):
    return scene[y : y + height, x : x + width].copy()


class CameraMotionTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.camera_motion")
        self.estimator = self.module.CameraMotionEstimator()
        self.scene = textured_scene()

    def test_a_pan_is_recovered_frame_by_frame_and_over_many_frames(self):
        self.assertIsNone(self.estimator.update(view(self.scene, 400, 200), []))

        total = np.eye(3)
        for step in range(1, 41):
            # The camera moves right and down: the picture moves left and up
            motion = self.estimator.update(view(self.scene, 400 + 6 * step, 200 + 2 * step), [])
            self.assertIsNotNone(motion)
            np.testing.assert_allclose(motion @ [640, 360, 1], [634, 358, 1], atol=1.0)
            total = motion @ total

        # 40 frames span more than one key frame; the errors must not add up
        np.testing.assert_allclose(total @ [640, 360, 1], [640 - 240, 360 - 80, 1], atol=3.0)

    def test_a_masked_player_moving_the_other_way_is_ignored(self):
        def frame_with_player(step):
            frame = view(self.scene, 400 + 5 * step, 200)
            x = 300 + 40 * step
            frame[200:500, x : x + 150] = textured_scene(seed=1)[0:300, 0:150]
            return frame, [x, 200, x + 150, 500]

        frame, box = frame_with_player(0)
        self.estimator.update(frame, [box])
        for step in range(1, 6):
            frame, box = frame_with_player(step)
            motion = self.estimator.update(frame, [box])
            np.testing.assert_allclose(motion @ [640, 360, 1], [635, 360, 1], atol=1.0)

    def test_no_motion_is_reported_without_texture_or_after_a_cut(self):
        blank = np.full((720, 1280, 3), 90, dtype=np.uint8)
        self.estimator.update(blank, [])
        self.assertIsNone(self.estimator.update(blank, []))

        self.estimator.reset()
        self.estimator.update(view(self.scene, 400, 200), [])
        other_scene = view(textured_scene(seed=7), 100, 100)
        self.assertIsNone(self.estimator.update(other_scene, []))

    def test_positions_move_with_the_picture(self):
        shift = np.array([[1, 0, -6], [0, 1, 2], [0, 0, 1]], dtype=float)
        self.assertEqual(self.module.move_points([(100, 50), (0, 0)], shift), [(94, 52), (-6, 2)])
        self.assertEqual(self.module.move_points([], shift), [])


if __name__ == "__main__":
    unittest.main()
