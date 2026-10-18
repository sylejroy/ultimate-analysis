"""The per-frame pipeline: result reuse, state resets, and jersey number bookkeeping."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from support import load_module

STAGES = (
    "detect_players_and_discs",
    "run_tracking",
    "run_field_segmentation",
    "create_unified_field_mask",
    "fit_lines_from_mask",
    "get_track_histories",
    "apply_camera_motion",
    "set_frame_rate",
    "reset_tracker",
    "reset_inference_state",
    "reset_segmentation_cache",
)
# The stages the jersey numbers use (processing/player_numbers.py)
NUMBER_STAGES = (
    "run_player_id_on_tracks",
    "missing_players",
    "kit_distance",
    "merge_players",
    "merge_jersey_readings",
    "get_best_jersey_number",
    "get_jersey_read_count",
    "reset_jersey_tracker",
)
DRAWING = (
    "draw_field_segmentation",
    "draw_unified_field_mask",
    "draw_detections",
    "draw_tracks",
    "draw_tracks_with_player_ids",
    "draw_possession",
    "draw_fps_overlay",
    "draw_notice",
    "draw_jersey_table",
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("pipeline")
        self.mocks = {}
        numbers = sys.modules[self.module.PlayerNumbers.__module__]
        for owner, names in ((self.module, STAGES + DRAWING), (numbers, NUMBER_STAGES)):
            for name in names:
                patcher = patch.object(owner, name)
                self.mocks[name] = patcher.start()
                self.addCleanup(patcher.stop)

        # No network in the tests: nobody is known by their looks
        looks = patch.object(self.module, "PlayerLooks")
        stage = looks.start().return_value
        stage.update.return_value, stage.returned.return_value = {}, []
        stage.teams.return_value, stage.swapped.return_value = {}, []
        stage.swap.return_value = None
        self.addCleanup(looks.stop)

        # Drawing functions hand back the frame they were given
        for name in DRAWING:
            self.mocks[name].side_effect = lambda frame, *args, **kwargs: frame
        self.mocks["draw_unified_field_mask"].side_effect = lambda frame, *a, **k: frame

        self.track = SimpleNamespace(track_id=7, class_name="player", bbox=[1, 1, 5, 5])
        # Drone footage: a team's worth of players in view
        self.players = [{"class_name": "player", "bbox": [1, 1, 5, 5]}] * 7
        self.mocks["detect_players_and_discs"].return_value = self.players
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
        self.mocks["missing_players"].return_value = []
        self.mocks["get_best_jersey_number"].return_value = (None, 0.0)
        self.mocks["get_jersey_read_count"].return_value = 5

        # Each frame waits for the field unless a test says otherwise: what a test
        # sees must not depend on how fast another thread is
        self.beside_the_frames = False
        setting = self.module.get_setting
        patcher = patch.object(
            self.module,
            "get_setting",
            lambda key, default=None: (
                self.beside_the_frames
                if key == "models.segmentation.in_background"
                else setting(key, default)
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.pipeline = self.module.AnalysisPipeline()
        self.options = self.module.PipelineOptions(top_down_view=False)
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)

    def test_redrawing_a_frame_reuses_its_analysis_and_new_frames_advance_it(self):
        first = self.pipeline.process(self.frame, 0, self.options)
        self.pipeline.process(self.frame, 0, self.options)
        self.assertEqual(self.mocks["detect_players_and_discs"].call_count, 1)
        self.assertEqual(self.mocks["run_tracking"].call_count, 1)

        # Identical pixels at a new position are a new frame for the tracker
        for index in range(1, 100):
            self.pipeline.process(self.frame, index, self.options)
        self.assertEqual(self.mocks["run_tracking"].call_count, 100)

        self.assertIsNot(first.main_view, self.frame)
        self.assertEqual(first.player_ids, {7: ("17", {})})
        self.assertIn("Inference", first.timings)

    def test_a_frame_is_analysed_alike_with_the_views_left_undrawn(self):
        undrawn = self.module.PipelineOptions(top_down_view=False, draw=False)
        result = self.pipeline.process(self.frame, 0, undrawn)

        self.assertIs(result.main_view, self.frame)
        self.assertIsNone(result.top_down_view)
        for name in ("draw_tracks", "draw_tracks_with_player_ids", "draw_possession"):
            self.mocks[name].assert_not_called()
        self.assertEqual(self.mocks["run_tracking"].call_count, 1)
        self.assertEqual(result.player_ids, {7: ("17", {})})
        # Whether the views are drawn is no new analysis of the frame
        self.pipeline.process(self.frame, 0, self.options)
        self.assertEqual(self.mocks["run_tracking"].call_count, 1)

    def test_changed_options_and_reset_analyse_the_frame_again(self):
        self.pipeline.process(self.frame, 3, self.options)

        without_tracking = self.module.PipelineOptions(tracking=False, top_down_view=False)
        result = self.pipeline.process(self.frame, 3, without_tracking)
        self.assertEqual(self.mocks["detect_players_and_discs"].call_count, 2)
        self.assertEqual(result.tracks, [])

        self.pipeline.process(self.frame, 3, self.options)
        self.pipeline.reset()
        self.mocks["reset_tracker"].assert_called_once()
        self.mocks["reset_segmentation_cache"].assert_called_once()
        self.assertEqual(self.pipeline.player_ids, {})
        self.pipeline.process(self.frame, 3, self.options)
        self.assertEqual(self.mocks["detect_players_and_discs"].call_count, 4)

    def test_nothing_is_followed_during_a_close_up_and_tracking_starts_afresh_after_it(self):
        self.pipeline.process(self.frame, 0, self.options)
        tracked = self.mocks["run_tracking"].call_count
        resets = self.mocks["reset_tracker"].call_count

        # The footage cuts to a close-up: the detector finds nobody
        self.mocks["detect_players_and_discs"].return_value = []
        results = [self.pipeline.process(self.frame, index, self.options) for index in range(1, 40)]
        self.assertFalse(results[-1].wide_shot)
        self.assertEqual(results[-1].tracks, [])
        self.assertIsNone(results[-1].top_down_view)
        # A few frames pass before the cut is believed; from then on nothing is tracked
        self.assertLess(self.mocks["run_tracking"].call_count - tracked, 20)
        self.assertEqual(self.mocks["reset_tracker"].call_count, resets + 1)

        # Back on the drone
        self.mocks["detect_players_and_discs"].return_value = self.players
        results = [
            self.pipeline.process(self.frame, index, self.options) for index in range(40, 80)
        ]
        self.assertTrue(results[-1].wide_shot)
        self.assertEqual([track.track_id for track in results[-1].tracks], [7])

    def test_a_failed_stage_is_shown_and_is_not_taken_for_a_close_up(self):
        health = sys.modules[self.module.health.__name__]
        self.pipeline.process(self.frame, 0, self.options)

        def failing(frame, **_):
            health.report("Detection", "failed (RuntimeError); see the log")
            return []

        self.mocks["detect_players_and_discs"].side_effect = failing
        results = [self.pipeline.process(self.frame, index, self.options) for index in range(1, 40)]
        self.assertTrue(results[-1].wide_shot)
        self.assertEqual(
            list(results[-1].problems), ["Detection: failed (RuntimeError); see the log"]
        )

        # Working again: nothing is shown any more
        self.mocks["detect_players_and_discs"].side_effect = None
        self.assertEqual(list(self.pipeline.process(self.frame, 40, self.options).problems), [])

    def test_field_geometry_is_computed_once_per_segmentation_result(self):
        # Segmentation returns the same result object for the frames in its interval
        for index in range(5):
            self.pipeline.process(self.frame, index, self.options)
        self.assertEqual(self.mocks["create_unified_field_mask"].call_count, 1)
        self.assertEqual(self.mocks["fit_lines_from_mask"].call_count, 1)

        self.mocks["run_field_segmentation"].return_value = [object()]
        self.pipeline.process(self.frame, 5, self.options)
        self.assertEqual(self.mocks["create_unified_field_mask"].call_count, 2)

    def test_field_found_beside_the_frames_is_taken_in_a_frame_later(self):
        self.beside_the_frames = True
        seen = [object()]
        with (
            patch.object(self.module, "prepare_for_segmentation", return_value=self.frame),
            patch.object(self.module, "segment_prepared", return_value=seen) as model,
        ):
            first = self.pipeline.process(self.frame, 0, self.options)
            self.pipeline._field_job["work"].result()
            # Begun in the first frame, there in the second, and not begun again at once
            self.assertEqual(first.frame_index, 0)
            self.assertEqual(self.pipeline.field_results, [])
            self.pipeline.process(self.frame, 1, self.options)
            self.assertIs(self.pipeline.field_results, seen)
            self.assertIsNotNone(self.pipeline._geometry.mask)
            for index in range(2, 5):
                self.pipeline.process(self.frame, index, self.options)
            self.assertEqual(model.call_count, 1)
            self.pipeline.process(self.frame, 5, self.options)
            self.pipeline._field_job["work"].result()
            self.assertEqual(model.call_count, 2)
            # After a seek, what was found is of the view before
            self.pipeline.reset()
            self.assertIsNone(self.pipeline._field_job)
            self.assertEqual(self.pipeline._field_seen, [])

    def test_video_and_reader_resets_discard_player_crop_snapshots(self):
        for reset in (self.pipeline.reset, self.pipeline.reset_player_ids):
            with self.subTest(reset=reset.__name__):
                self.pipeline._numbers.crops.observe(7, self.frame, 3, 5, 100)
                reset()
                self.assertIsNone(self.pipeline._numbers.crops.take(7, 3))

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

    def test_a_player_with_the_number_of_a_missing_player_becomes_that_player(self):
        numbers = {7: ("17", 0.8), 3: ("17", 0.9), 5: ("17", 0.9)}
        self.mocks["get_best_jersey_number"].side_effect = lambda player: numbers.get(
            player, (None, 0.0)
        )

        numbers_module = sys.modules[self.module.PlayerNumbers.__module__]
        teams = patch.object(numbers_module, "team_of_player", side_effect={7: 0, 3: 0, 5: 1}.get)
        teams.start()
        self.addCleanup(teams.stop)

        # Player 3 is missing, of the same team as the new player 7 and looks like them:
        # one and the same
        self.mocks["missing_players"].return_value = [3]
        self.mocks["kit_distance"].return_value = 5.0
        result = self.pipeline.process(self.frame, 0, self.options)
        self.mocks["merge_players"].assert_called_once_with(7, 3)
        self.mocks["merge_jersey_readings"].assert_called_once_with(7, 3)
        self.assertEqual([track.track_id for track in result.tracks], [3])
        self.assertEqual(list(result.player_ids), [3])

        # The other team's number 17 looks different and stays somebody else
        self.track.track_id = 7
        self.mocks["missing_players"].return_value = [5]
        self.mocks["kit_distance"].return_value = 90.0
        result = self.pipeline.process(self.frame, 1, self.options)
        self.assertEqual([track.track_id for track in result.tracks], [7])
        self.mocks["merge_players"].assert_called_once()

        # Nor does one whose kit looks alike: the tracker has the two in different teams
        self.mocks["kit_distance"].return_value = 5.0
        result = self.pipeline.process(self.frame, 2, self.options)
        self.assertEqual([track.track_id for track in result.tracks], [7])
        self.mocks["merge_players"].assert_called_once()

        # Nor one whose team the tracker does not know yet
        numbers[9] = ("17", 0.9)
        self.mocks["missing_players"].return_value = [9]
        result = self.pipeline.process(self.frame, 3, self.options)
        self.assertEqual([track.track_id for track in result.tracks], [7])
        self.mocks["merge_players"].assert_called_once()

        # Nor one of the right team and kit whose number was read only twice
        self.mocks["missing_players"].return_value = [3]
        self.mocks["get_jersey_read_count"].return_value = 2
        result = self.pipeline.process(self.frame, 4, self.options)
        self.assertEqual([track.track_id for track in result.tracks], [7])
        self.mocks["merge_players"].assert_called_once()

        # Nor one who was seen somewhere the missing player cannot have been
        self.mocks["get_jersey_read_count"].return_value = 5
        self.pipeline._numbers.could_be = lambda player, earlier: False
        result = self.pipeline.process(self.frame, 5, self.options)
        self.assertEqual([track.track_id for track in result.tracks], [7])
        self.mocks["merge_players"].assert_called_once()

    def test_the_top_down_view_is_drawn_once_the_field_is_found(self):
        options = self.module.PipelineOptions(tracking=False)
        result = self.pipeline.process(self.frame, 0, options)
        self.assertIsNone(result.top_down_view)
        self.assertEqual(result.top_down_message, "The field has not been found in this view yet")

        self.pipeline._field_follower.image_to_field = np.eye(3)
        with patch.object(self.module, "draw_field_diagram", return_value="drawn") as diagram:
            result = self.pipeline.process(self.frame, 1, options)
        self.assertEqual((result.top_down_view, result.top_down_message), ("drawn", ""))
        diagram.assert_called_once()
        self.assertIn("Top-down view", result.timings)

        without = self.module.PipelineOptions(tracking=False, field_segmentation=False)
        result = self.pipeline.process(self.frame, 2, without)
        self.assertEqual(result.top_down_message, "The top-down view needs field segmentation")

    def test_a_crossing_exchanges_what_belongs_to_the_boxes(self):
        one = SimpleNamespace(track_id=7, class_name="player", team=0, team_colour=(1, 1, 1))
        other = SimpleNamespace(track_id=8, class_name="player", team=1, team_colour=(2, 2, 2))
        self.pipeline.tracks = [one, other]
        self.pipeline._looks.swapped.return_value = [(7, 8)]
        # Where each box stands on the field, followed over the frames
        self.pipeline._steady._players = {7: ["left"], 8: ["right"]}
        with patch.object(self.module, "swap_players", return_value=True):
            self.pipeline._settle_crossings(0)
        self.assertEqual((one.track_id, other.track_id), (8, 7))
        self.assertEqual(self.pipeline._steady._players, {8: ["left"], 7: ["right"]})


if __name__ == "__main__":
    unittest.main()
