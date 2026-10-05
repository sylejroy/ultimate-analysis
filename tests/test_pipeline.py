"""The per-frame pipeline: result reuse, state resets, and jersey number bookkeeping."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from support import load_module

STAGES = (
    "run_inference",
    "run_tracking",
    "run_field_segmentation",
    "run_player_id_on_tracks",
    "create_unified_field_mask",
    "fit_lines_from_mask",
    "get_track_histories",
    "get_best_jersey_number",
    "reset_tracker",
    "reset_inference_state",
    "reset_segmentation_cache",
    "reset_jersey_tracker",
)
DRAWING = (
    "draw_field_segmentation",
    "draw_unified_field_mask",
    "draw_all_field_lines",
    "draw_ransac_field_lines",
    "draw_detections",
    "draw_tracks",
    "draw_tracks_with_player_ids",
    "draw_possession",
    "draw_fps_overlay",
    "draw_jersey_table",
    "apply_segmentation_to_warped_frame",
    "draw_tracks_top_down",
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("pipeline")
        self.mocks = {}
        for name in STAGES + DRAWING:
            patcher = patch.object(self.module, name)
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

        # Drawing functions hand back the frame they were given
        for name in DRAWING:
            self.mocks[name].side_effect = lambda frame, *args, **kwargs: frame
        self.mocks["draw_unified_field_mask"].side_effect = lambda frame, *a, **k: (frame, {}, {})

        self.track = SimpleNamespace(track_id=7, class_name="player", bbox=[1, 1, 5, 5])
        self.mocks["run_inference"].return_value = [{"class_name": "player"}]
        self.mocks["run_tracking"].return_value = [self.track]
        self.mocks["run_field_segmentation"].return_value = [object()]
        self.mocks["run_player_id_on_tracks"].return_value = (
            {7: ("17", {})},
            {"preprocessing_ms": 1.0, "ocr_ms": 2.0, "filtering_ms": 0.0},
            set(),
        )
        self.mocks["create_unified_field_mask"].return_value = np.ones((8, 8), dtype=np.uint8)
        self.mocks["fit_lines_from_mask"].return_value = ([], [], None, None)
        self.mocks["get_track_histories"].return_value = {}
        self.mocks["get_best_jersey_number"].return_value = (None, 0.0)

        self.pipeline = self.module.AnalysisPipeline()
        self.options = self.module.PipelineOptions(top_down_view=False)
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)

    def test_redrawing_a_frame_reuses_its_analysis_and_new_frames_advance_it(self):
        first = self.pipeline.process(self.frame, 0, self.options)
        self.pipeline.process(self.frame, 0, self.options)
        self.assertEqual(self.mocks["run_inference"].call_count, 1)
        self.assertEqual(self.mocks["run_tracking"].call_count, 1)

        # Identical pixels at a new position are a new frame for the tracker
        for index in range(1, 100):
            self.pipeline.process(self.frame, index, self.options)
        self.assertEqual(self.mocks["run_tracking"].call_count, 100)

        self.assertIsNot(first.main_view, self.frame)
        self.assertEqual(first.player_ids, {7: ("17", {})})
        self.assertIn("Inference", first.timings)

    def test_changed_options_and_reset_analyse_the_frame_again(self):
        self.pipeline.process(self.frame, 3, self.options)

        without_tracking = self.module.PipelineOptions(tracking=False, top_down_view=False)
        result = self.pipeline.process(self.frame, 3, without_tracking)
        self.assertEqual(self.mocks["run_inference"].call_count, 2)
        self.assertEqual(result.tracks, [])

        self.pipeline.process(self.frame, 3, self.options)
        self.pipeline.reset()
        self.mocks["reset_tracker"].assert_called_once()
        self.mocks["reset_segmentation_cache"].assert_called_once()
        self.assertEqual(self.pipeline.player_ids, {})
        self.pipeline.process(self.frame, 3, self.options)
        self.assertEqual(self.mocks["run_inference"].call_count, 4)

    def test_field_geometry_is_computed_once_per_segmentation_result(self):
        # Segmentation returns the same result object for the frames in its interval
        for index in range(5):
            self.pipeline.process(self.frame, index, self.options)
        self.assertEqual(self.mocks["create_unified_field_mask"].call_count, 1)
        self.assertEqual(self.mocks["fit_lines_from_mask"].call_count, 1)

        self.mocks["run_field_segmentation"].return_value = [object()]
        self.pipeline.process(self.frame, 5, self.options)
        self.assertEqual(self.mocks["create_unified_field_mask"].call_count, 2)

    def test_jersey_numbers_persist_fall_back_to_the_tracker_and_follow_the_tracks(self):
        self.pipeline.process(self.frame, 0, self.options)

        # Not read this frame: the earlier number stays
        self.mocks["run_player_id_on_tracks"].return_value = (
            {},
            {"preprocessing_ms": 0, "ocr_ms": 0},
            set(),
        )
        result = self.pipeline.process(self.frame, 1, self.options)
        self.assertEqual(result.player_ids[7][0], "17")

        # A track read as unknown takes the jersey tracker's best number
        other = SimpleNamespace(track_id=9, class_name="player", bbox=[1, 1, 5, 5])
        self.mocks["run_tracking"].return_value = [other]
        self.mocks["run_player_id_on_tracks"].return_value = (
            {9: ("Unknown", None)},
            {"preprocessing_ms": 0, "ocr_ms": 0},
            set(),
        )
        self.mocks["get_best_jersey_number"].return_value = ("23", 0.8)
        result = self.pipeline.process(self.frame, 2, self.options)
        self.assertEqual(result.player_ids[9][0], "23")
        # Track 7 is gone, and so is its number
        self.assertNotIn(7, result.player_ids)

    def test_top_down_view_needs_a_homography(self):
        options = self.module.PipelineOptions()
        result = self.pipeline.process(self.frame, 0, options)
        self.assertIsNone(result.top_down_view)
        self.assertEqual(result.top_down_message, "Homography matrix not available")

        self.pipeline.homography_matrix = np.eye(3, dtype=np.float32)
        with patch.object(
            self.module, "get_setting", side_effect=lambda key, default=None: default
        ):
            result = self.pipeline.process(self.frame, 0, options)
        self.assertIsNotNone(result.top_down_view)
        self.assertIn("Homography Calculation", result.timings)


if __name__ == "__main__":
    unittest.main()
