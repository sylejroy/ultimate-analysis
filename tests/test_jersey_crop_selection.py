"""Temporal crop selection, backoff, and cache ownership."""

import unittest

import cv2
import numpy as np
from support import load_module


class JerseyCropSelectionTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.jersey_crops")
        self.selector = self.module.JerseyCropSelector()
        self.sharp = np.zeros((80, 40, 3), dtype=np.uint8)
        self.sharp[:, ::4] = 255
        self.blurred = cv2.GaussianBlur(self.sharp, (15, 15), 3)

    def test_sharp_visible_torso_outscores_blur_and_occlusion(self):
        quality = self.module.crop_quality
        sharp = quality(self.sharp, 0.5, 0, 0)
        self.assertGreater(sharp, quality(self.blurred, 0.5, 0, 0))
        self.assertGreater(sharp, quality(self.sharp, 0.5, 0.5, 0))
        self.assertEqual(quality(self.sharp, 0.5, 1.0, 0), 0)
        self.assertEqual(quality(np.zeros_like(self.sharp), 0.5, 0, 5), 0)

    def test_best_recent_crop_is_copied_and_consumed_only_once(self):
        self.selector.begin_frame(1, {7})
        self.selector.observe(7, self.sharp, 1, 5, 100)
        self.selector.observe(7, self.blurred, 2, 5, 10)
        expected = self.sharp.copy()
        self.sharp[:] = 0  # The video buffer can be reused after observation.
        np.testing.assert_array_equal(self.selector.take(7, 5), expected)
        self.assertIsNone(self.selector.take(7, 5))
        self.selector.observe(7, self.blurred, 2, 5, 1000)  # Duplicate frame, no new vote.
        self.assertIsNone(self.selector.take(7, 5))

    def test_expired_best_crop_is_replaced_by_a_recent_candidate(self):
        self.selector.observe(7, self.sharp, 1, 5, 100)
        self.selector.observe(7, self.blurred, 6, 5, 10)
        np.testing.assert_array_equal(self.selector.take(7, 6), self.blurred)

    def test_unreadable_windows_back_off_and_success_restores_the_cadence(self):
        self.selector.observe(7, self.sharp, 0, 5, 100)
        self.selector.take(7, 0)
        self.selector.record_read(7, 0, False, 5, 4)
        self.selector.observe(7, self.sharp, 5, 5, 100)
        self.assertIsNone(self.selector.take(7, 5))
        self.selector.observe(7, self.sharp, 10, 5, 100)
        self.assertIsNotNone(self.selector.take(7, 10))
        self.selector.record_read(7, 10, False, 5, 4)
        self.selector.observe(7, self.sharp, 15, 5, 100)
        self.assertIsNone(self.selector.take(7, 20))
        self.selector.observe(7, self.sharp, 30, 5, 100)
        self.assertIsNotNone(self.selector.take(7, 30))
        self.selector.record_read(7, 30, True, 5, 4)
        self.selector.observe(7, self.sharp, 34, 5, 100)
        self.assertIsNotNone(self.selector.take(7, 35))

    def test_crop_expires_even_without_another_observation(self):
        self.selector.observe(7, self.sharp, 1, 5, 100)
        self.assertIsNone(self.selector.take(7, 6))

    def test_missing_players_and_seeks_release_cached_images(self):
        self.selector.observe(7, self.sharp, 10, 5, 100)
        self.selector.begin_frame(10, set())
        self.assertIsNone(self.selector.take(7, 10))
        self.selector.observe(7, self.sharp, 10, 5, 100)
        self.selector.begin_frame(2, {7})
        self.assertIsNone(self.selector.take(7, 2))


if __name__ == "__main__":
    unittest.main()
