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


if __name__ == "__main__":
    unittest.main()
