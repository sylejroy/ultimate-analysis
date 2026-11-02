"""The per-frame pipeline: result reuse, state resets, and jersey number bookkeeping."""

import gc
import sys
import unittest
import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from support import load_module

STAGES = (
    "detect_players_and_discs",
    "reset_inference_state",
    "reset_segmentation_cache",
)
# The field model and what is made of its masks (pipeline/field.py)
FIELD_STAGES = (
    "run_field_segmentation",
    "create_unified_field_mask",
    "fit_lines_from_mask",
)
# What the pipeline and the jersey numbers ask of the tracker (processing/tracking.py)
TRACKER = (
    "run_tracking",
    "get_track_histories",
    "apply_camera_motion",
    "set_frame_rate",
    "reset",
    "missing_players",
    "kit_distance",
    "merge_players",
)
# The stages the jersey numbers use (processing/player_numbers.py)
NUMBER_STAGES = (
    "run_player_id_on_tracks",
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
        self.module = load_module("pipeline.analysis")
        self.mocks = {}
        self.identity = sys.modules[self.module.IdentityStage.__module__]
        numbers = sys.modules[self.identity.PlayerNumbers.__module__]
        tracker = self.module.PlayerTracker
        self.field = sys.modules[self.module.FieldStage.__module__]
        self.views = sys.modules[self.module.Scene.__module__]
        for owner, names in (
            (self.module, STAGES + ("draw_notice",)),
            (self.views, DRAWING),
            (self.field, FIELD_STAGES),
            (numbers, NUMBER_STAGES),
            (tracker, TRACKER),
        ):
            for name in names:
                patcher = patch.object(owner, name)
                self.mocks[name] = patcher.start()
                self.addCleanup(patcher.stop)

        # No network in the tests: nobody is known by their looks
        looks = patch.object(self.identity, "PlayerLooks")
        stage = looks.start().return_value
        stage.update.return_value = {}
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
        setting = self.field.get_setting
        patcher = patch.object(
            self.field,
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

    def test_a_new_pipeline_frees_the_one_dropped(self):
        # Only the pipeline may collect: dropped, one stays, in cycles with its parts
        gc.disable()
        self.addCleanup(gc.enable)
        gc.collect()
        dropped = weakref.ref(self.module.AnalysisPipeline())
        self.assertIsNotNone(dropped())
        self.module.AnalysisPipeline()
        self.assertIsNone(dropped())

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
        self.mocks["reset"].assert_called_once()
        self.mocks["reset_segmentation_cache"].assert_called_once()
        self.assertEqual(self.pipeline.player_ids, {})
        self.pipeline.process(self.frame, 3, self.options)
        self.assertEqual(self.mocks["detect_players_and_discs"].call_count, 4)

    @staticmethod
    def row_of_players(top, count=8):
        return [
            {
                "bbox": [10 + 100 * i, top, 40 + 100 * i, top + 80],
                "class_name": "player",
                "confidence": 0.9,
                "class_id": 1,
            }
            for i in range(count)
        ]

    def followed_with_a_motion(self, motion=np.eye(3)):
        """Two frames of drone footage with the camera's motion known; patched from then on."""
        patcher = patch.object(self.pipeline, "_camera_motion_of", return_value=motion)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mocks["detect_players_and_discs"].return_value = self.row_of_players(10)
        self.pipeline.process(self.frame, 0, self.options)
        self.pipeline.process(self.frame, 1, self.options)
        return self.mocks["reset"].call_count, self.pipeline.restarts

    def test_a_cut_is_told_also_where_a_motion_is_found_between_the_two_views(self):
        resets, restarts = self.followed_with_a_motion()
        self.pipeline.process(self.frame, 2, self.options)  # The same view: nothing happens
        self.assertEqual(self.mocks["reset"].call_count, resets)
        self.mocks["detect_players_and_discs"].return_value = self.row_of_players(400)
        self.pipeline.process(self.frame, 3, self.options)
        self.assertEqual(self.mocks["reset"].call_count, resets + 1)
        self.assertEqual(self.pipeline.restarts, restarts + 1)
        # Players missed in the frame after are no reason to stop
        self.mocks["detect_players_and_discs"].return_value = self.row_of_players(400, 2)
        self.assertTrue(self.pipeline.process(self.frame, 4, self.options).wide_shot)

    def test_a_swing_of_the_camera_with_few_players_left_forgets_nothing(self):
        resets, _ = self.followed_with_a_motion()
        self.mocks["detect_players_and_discs"].return_value = self.row_of_players(400, 2)
        result = self.pipeline.process(self.frame, 2, self.options)
        self.assertEqual(self.mocks["reset"].call_count, resets)
        self.assertTrue(result.wide_shot)

    def test_a_hitch_in_playback_forgets_nothing(self):
        resets, _ = self.followed_with_a_motion()
        self.mocks["detect_players_and_discs"].return_value = self.row_of_players(400)
        self.pipeline.process(self.frame, 31, self.options)  # A second on
        self.assertEqual(self.mocks["reset"].call_count, resets)

    def test_a_hitch_with_the_motion_lost_forgets_nothing(self):
        resets, _ = self.followed_with_a_motion(motion=None)
        self.pipeline._motion_known = True
        self.mocks["detect_players_and_discs"].return_value = self.row_of_players(400)
        self.pipeline.process(self.frame, 91, self.options)
        self.assertEqual(self.mocks["reset"].call_count, resets)

    def test_a_frame_without_players_forgets_nothing(self):
        resets, _ = self.followed_with_a_motion()
        self.mocks["detect_players_and_discs"].return_value = []
        self.pipeline.process(self.frame, 2, self.options)
        self.assertEqual(self.mocks["reset"].call_count, resets)

    def test_a_detector_that_failed_says_nothing_of_a_cut(self):
        resets, _ = self.followed_with_a_motion(motion=None)
        self.pipeline._motion_known = True  # The motion is lost in this frame only
        self.mocks["detect_players_and_discs"].return_value = []
        with patch.object(self.module.health, "failed", return_value=True):
            self.pipeline.process(self.frame, 2, self.options)
        self.assertEqual(self.mocks["reset"].call_count, resets)

    def test_detection_switched_off_says_nothing_of_a_cut(self):
        resets, _ = self.followed_with_a_motion(motion=None)
        self.pipeline._motion_known = True
        self.pipeline.process(self.frame, 2, replace(self.options, detection=False))
        self.assertEqual(self.mocks["reset"].call_count, resets)

    def test_nothing_is_followed_during_a_close_up_and_tracking_starts_afresh_after_it(self):
        self.pipeline.process(self.frame, 0, self.options)
        tracked = self.mocks["run_tracking"].call_count
        resets, restarts = self.mocks["reset"].call_count, self.pipeline.restarts

        # The footage cuts to a close-up: the detector finds nobody
        self.mocks["detect_players_and_discs"].return_value = []
        results = [self.pipeline.process(self.frame, index, self.options) for index in range(1, 40)]
        self.assertFalse(results[-1].wide_shot)
        self.assertEqual(results[-1].tracks, [])
        self.assertIsNone(results[-1].top_down_view)
        # A few frames pass before the cut is believed; from then on nothing is tracked
        self.assertLess(self.mocks["run_tracking"].call_count - tracked, 20)
        self.assertEqual(self.mocks["reset"].call_count, resets + 1)
        self.assertEqual(self.pipeline.restarts, restarts + 1)  # IDs begin again here too

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

    def test_a_failure_of_the_detectors_looking_ahead_is_shown_on_their_frame(self):
        """The next frame's detectors run during this frame and report then; what
        they say must reach the frame it is about, not be forgotten with this one."""
        health = sys.modules[self.module.health.__name__]

        def failing(frame, **_):
            health.report("Detection", "failed (RuntimeError); see the log")
            return []

        self.mocks["detect_players_and_discs"].side_effect = failing
        frame = self.frame
        results = [self.pipeline.process(frame, 0, self.options, next_frame=lambda: frame)]
        for index in range(1, 40):
            # The detectors are done with the frame before its turn, as in the app
            self.pipeline._ahead.result()
            results.append(
                self.pipeline.process(frame, index, self.options, next_frame=lambda: frame)
            )
        self.assertTrue(results[-1].wide_shot)
        self.assertEqual(
            {tuple(result.problems) for result in results},
            {("Detection: failed (RuntimeError); see the log",)},
        )

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
            patch.object(self.field, "prepare_for_segmentation", return_value=self.frame),
            patch.object(self.field, "segment_prepared", return_value=seen) as model,
        ):
            first = self.pipeline.process(self.frame, 0, self.options)
            self.pipeline._field._job["work"].result()
            # Begun in the first frame, there in the second, and not begun again at once
            self.assertEqual(first.frame_index, 0)
            self.assertEqual(self.pipeline._field.results, [])
            self.pipeline.process(self.frame, 1, self.options)
            self.assertIs(self.pipeline._field.results, seen)
            self.assertIsNotNone(self.pipeline._field.mask)
            for index in range(2, 5):
                self.pipeline.process(self.frame, index, self.options)
            self.assertEqual(model.call_count, 1)
            self.pipeline.process(self.frame, 5, self.options)
            self.pipeline._field._job["work"].result()
            self.assertEqual(model.call_count, 2)
            # After a seek, what was found is of the view before
            self.pipeline.reset()
            self.assertIsNone(self.pipeline._field._job)
            self.assertEqual(self.pipeline._field._seen, [])

    def test_video_and_reader_resets_discard_player_crop_snapshots(self):
        for reset in (self.pipeline.reset, self.pipeline.reset_player_ids):
            with self.subTest(reset=reset.__name__):
                self.pipeline._identity.numbers.crops.observe(7, self.frame, 3, 5, 100)
                reset()
                self.assertIsNone(self.pipeline._identity.numbers.crops.take(7, 3))

    def test_a_player_shown_without_a_number_or_with_another_is_not_named_by_it(self):
        identity = self.pipeline._identity
        tracks = [
            SimpleNamespace(track_id=track_id, class_name="player") for track_id in (6, 32, 40)
        ]
        identity.numbers.numbers.update({6: ("Unknown", None), 32: ("19", None), 40: ("7", None)})
        module = sys.modules[type(identity).__module__]
        with patch.object(module, "get_best_jersey_number", return_value=("19", 0.8)):
            self.assertEqual(identity.certain_numbers(tracks), {32: "19"})

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

        teams = patch.object(
            self.module.PlayerTracker, "team_of_player", side_effect={7: 0, 3: 0, 5: 1}.get
        )
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
        self.pipeline._identity.numbers.could_be = lambda player, earlier: False
        result = self.pipeline.process(self.frame, 5, self.options)
        self.assertEqual([track.track_id for track in result.tracks], [7])
        self.mocks["merge_players"].assert_called_once()

    def test_the_top_down_view_is_drawn_once_the_field_is_found(self):
        options = self.module.PipelineOptions(tracking=False)
        result = self.pipeline.process(self.frame, 0, options)
        self.assertIsNone(result.top_down_view)
        self.assertEqual(result.top_down_message, "The field has not been found in this view yet")

        self.pipeline._field.follower.image_to_field = np.eye(3)
        with patch.object(self.views, "draw_field_diagram", return_value="drawn") as diagram:
            result = self.pipeline.process(self.frame, 1, options)
        self.assertEqual((result.top_down_view, result.top_down_message), ("drawn", ""))
        diagram.assert_called_once()
        self.assertIn("Top-down view", result.timings)

        without = self.module.PipelineOptions(tracking=False, field_segmentation=False)
        result = self.pipeline.process(self.frame, 2, without)
        self.assertEqual(result.top_down_message, "The top-down view needs field segmentation")

    def test_a_top_down_view_that_fails_is_said_on_its_frame(self):
        """It is drawn on another thread while the camera view is drawn; its failure
        must be among this frame's problems, not land after they were read."""
        options = self.module.PipelineOptions(tracking=False)
        self.pipeline._field.follower.image_to_field = np.eye(3)
        with patch.object(self.views, "draw_field_diagram", side_effect=RuntimeError("no")):
            result = self.pipeline.process(self.frame, 1, options)
        self.assertIsNone(result.top_down_view)
        self.assertIn("Top-down view: failed (RuntimeError); see the log", result.problems)

    def test_the_players_table_comes_every_so_many_frames_analysed(self):
        """Not every so many video frames: playback passing over every second frame
        from an odd one would never land on a multiple of thirty."""
        every = self.module.STATS_EVERY_FRAMES
        with patch.object(self.pipeline, "player_stats", return_value=[{"player": 1}]):
            results = [
                self.pipeline.process(self.frame, index, self.options)
                for index in range(1, 2 * every + 2, 2)
            ]
        self.assertEqual(sum(result.player_stats is not None for result in results), 1)

    def test_a_crossing_exchanges_what_belongs_to_the_boxes(self):
        one = SimpleNamespace(track_id=7, class_name="player", team=0, team_colour=(1, 1, 1))
        other = SimpleNamespace(track_id=8, class_name="player", team=1, team_colour=(2, 2, 2))
        self.pipeline.tracks = [one, other]
        self.pipeline._identity.looks.swapped.return_value = [(7, 8)]
        # Where each box stands on the field, followed over the frames
        self.pipeline._steady._players = {7: ["left"], 8: ["right"]}
        with patch.object(self.module.PlayerTracker, "swap_players", return_value=True):
            self.pipeline._identity._settle_crossings(self.pipeline.tracks, 0)
        self.assertEqual((one.track_id, other.track_id), (8, 7))
        self.assertEqual(self.pipeline._steady._players, {8: ["left"], 7: ["right"]})


if __name__ == "__main__":
    unittest.main()
