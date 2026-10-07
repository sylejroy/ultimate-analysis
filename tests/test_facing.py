"""Which way a player faces, from the place of their shoulders."""

import unittest

import numpy as np
from support import load_module


def person(left_shoulder_x, right_shoulder_x, confidence=0.9):
    """Keypoints (COCO order) of a person 100 pixels from shoulders to hips."""
    keypoints = np.zeros((17, 2), dtype=np.float32)
    confidences = np.full(17, confidence, dtype=np.float32)
    keypoints[5], keypoints[6] = (left_shoulder_x, 40), (right_shoulder_x, 40)
    keypoints[11], keypoints[12] = (left_shoulder_x, 140), (right_shoulder_x, 140)
    return keypoints, confidences


class FacingTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.facing")

    def test_seen_from_behind_the_left_shoulder_is_on_the_left(self):
        self.assertEqual(self.module.facing_of(*person(20, 60), 200), self.module.BACK)
        self.assertEqual(self.module.facing_of(*person(60, 20), 200), self.module.FRONT)

    def test_shoulders_on_top_of_each_other_are_a_side_view(self):
        self.assertEqual(self.module.facing_of(*person(40, 48), 200), self.module.SIDE)

    def test_unsure_shoulders_say_nothing(self):
        self.assertIsNone(self.module.facing_of(*person(20, 60, confidence=0.1), 200))


if __name__ == "__main__":
    unittest.main()
