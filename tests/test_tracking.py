"""DeepSORT tracking state and class handling."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from support import load_module


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.tracking")
        self.module._track_histories.clear()

    def test_reset_preserves_loaded_embedder_and_discards_identity_state(self):
        tracker = Mock()
        tracker.tracker.metric.samples = {1: ["old embedding"]}
        with patch.object(self.module, "_deepsort_tracker", tracker):
            self.module.reset_tracker()
            self.assertIs(self.module._deepsort_tracker, tracker)
        tracker.delete_all_tracks.assert_called_once()
        self.assertEqual(tracker.tracker.metric.samples, {})

    def test_empty_frame_ages_tracker_and_prunes_retired_histories(self):
        tracker = Mock()
        tracker.update_tracks.return_value = []
        self.module._track_histories[7] = [(1, 2)]
        frame = np.zeros((8, 8, 3))
        with patch.multiple(self.module, _deepsort_tracker=tracker, DEEPSORT_AVAILABLE=True):
            self.assertEqual(self.module.run_tracking(frame, []), [])
        tracker.update_tracks.assert_called_once_with([], embeds=None, frame=frame)
        self.assertEqual(self.module.get_track_histories(), {})

    def test_separate_model_class_zero_does_not_turn_players_into_discs(self):
        tracker = Mock()
        tracker.update_tracks.return_value = []
        detections = [
            {"bbox": [0, 0, 4, 4], "confidence": 0.9, "class_id": 0, "class_name": name}
            for name in ("player", "disc")
        ]
        with patch.multiple(self.module, _deepsort_tracker=tracker, DEEPSORT_AVAILABLE=True):
            self.module.run_tracking(np.zeros((8, 8, 3)), detections)
        raw = tracker.update_tracks.call_args.args[0]
        self.assertEqual([detection[2] for detection in raw], [1, 0])

    def test_real_deepsort_ages_tracks_without_loading_an_embedder(self):
        from deep_sort_realtime.deepsort_tracker import DeepSort

        tracker = DeepSort(embedder=None, n_init=2, max_age=1)
        tracker.embedder = Mock()
        tracker.generate_embeds = Mock(return_value=[np.ones(128, dtype=np.float32)])
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = [
            {"bbox": [0, 0, 4, 4], "confidence": 0.9, "class_id": 0, "class_name": "player"}
        ]
        with patch.multiple(self.module, _deepsort_tracker=tracker, DEEPSORT_AVAILABLE=True):
            self.module.run_tracking(frame, detections)
            tracks = self.module.run_tracking(frame, detections)
            self.assertEqual([track.class_name for track in tracks], ["player"])
            self.assertEqual(self.module.run_tracking(frame, []), [])
            self.assertEqual(self.module.run_tracking(frame, []), [])
            self.assertEqual(tracker.tracker.tracks, [])
            self.module.reset_tracker()
            self.assertIs(self.module._deepsort_tracker, tracker)
            self.assertEqual(tracker.tracker.metric.samples, {})

    def test_lost_tracks_are_kept_for_seconds_whatever_the_frame_rate(self):
        tracker = Mock()
        settings = {"models.tracking.max_age_seconds": 3.0}
        with (
            patch.object(self.module, "_deepsort_tracker", tracker),
            patch.object(
                self.module, "get_setting", side_effect=lambda key, d=None: settings.get(key, d)
            ),
        ):
            self.module.set_frame_rate(60)
            self.assertEqual(tracker.tracker.max_age, 180)
            self.module.set_frame_rate(29.97)
            self.assertEqual(tracker.tracker.max_age, 90)
            # A video without a usable frame rate keeps the last one
            self.module.set_frame_rate(0)
            self.assertEqual(tracker.tracker.max_age, 90)

    def test_camera_motion_moves_trails_and_where_tracks_expect_their_players(self):
        from types import SimpleNamespace

        # x, y, aspect, height and their speeds
        track = SimpleNamespace(mean=np.array([400.0, 300.0, 0.4, 100.0, 5.0, 0.0, 0.0, 1.0]))
        tracker = Mock()
        tracker.tracker.tracks = [track]
        self.module._track_histories[1] = [(400, 350), (410, 350)]

        with patch.object(self.module, "_deepsort_tracker", tracker):
            # The camera pans: the picture moves 20 px left and 4 px up
            pan = np.array([[1.0, 0, -20], [0, 1, -4], [0, 0, 1]])
            self.module.apply_camera_motion(pan)
            np.testing.assert_allclose(track.mean, [380, 296, 0.4, 100, 5, 0, 0, 1])
            self.assertEqual(self.module._track_histories[1], [(380, 346), (390, 346)])

            # The camera zooms in by 10% around the picture's corner
            zoom = np.diag([1.1, 1.1, 1.0])
            self.module.apply_camera_motion(zoom)
            np.testing.assert_allclose(track.mean, [418, 325.6, 0.4, 110, 5.5, 0, 0, 1.1])


if __name__ == "__main__":
    unittest.main()
