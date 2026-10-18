"""Tracking state and class handling, with stand-ins for the ByteTrack trackers."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from support import load_module


def bytetrack(rows=(), tracks=()):
    """A stand-in for a ByteTrack tracker: `rows` are what each update gives back
    (x1, y1, x2, y2, track ID, confidence, class, detection), `tracks` its tracks."""
    tracker = Mock()
    tracker.update.return_value = np.array(rows, dtype=float).reshape(-1, 8)
    tracker.outsiders.return_value = ()
    tracker.teams_of_tracks.return_value = {}
    tracker.leanings_of_tracks.return_value = {}
    tracker.shirt_colours.return_value = {}
    tracker.tracked_stracks, tracker.lost_stracks = list(tracks), []
    tracker.args = SimpleNamespace()
    tracker.keep_frames = 180  # Video frames a lost track is kept
    return tracker


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.tracking")
        self.module._track_histories.clear()

    def test_reset_keeps_the_trackers_and_forgets_their_tracks(self):
        player, disc = bytetrack(), bytetrack()
        with patch.multiple(self.module, _player_tracker=player, _disc_tracker=disc):
            self.module.reset_tracker()
            self.assertIs(self.module._player_tracker, player)
        player.reset.assert_called_once()
        disc.reset.assert_called_once()

    def test_empty_frame_ages_tracker_and_prunes_retired_histories(self):
        player, disc = bytetrack(), bytetrack()
        self.module._track_histories[7] = np.array([(1, 2)], dtype=np.float32)
        frame = np.zeros((8, 8, 3))
        with patch.multiple(self.module, _player_tracker=player, _disc_tracker=disc):
            self.assertEqual(self.module.run_tracking(frame, []), [])
        player.update.assert_called_once()
        self.assertEqual(self.module.get_track_histories(), {})

    def test_separate_model_class_zero_does_not_turn_players_into_discs(self):
        # A player model whose only class is 0, beside a disc model: by class name
        player = bytetrack(rows=[(0, 0, 4, 4, 3, 0.9, 0, 0)])
        disc = bytetrack()
        detections = [
            {"bbox": [0, 0, 4, 4], "confidence": 0.9, "class_id": 0, "class_name": "player"}
        ]
        settings = {"models.tracking.identity.enabled": False}
        with (
            patch.multiple(self.module, _player_tracker=player, _disc_tracker=disc),
            patch.object(
                self.module, "get_setting", side_effect=lambda key, d=None: settings.get(key, d)
            ),
        ):
            tracks = self.module.run_tracking(np.zeros((8, 8, 3)), detections)
        self.assertEqual([track.class_name for track in tracks], ["player"])
        self.assertEqual(len(disc.update.call_args.args[0].conf), 0)
        self.module.reset_tracker()

    def test_lost_tracks_are_kept_for_seconds_whatever_the_frame_rate(self):
        player, disc = bytetrack(), bytetrack()
        settings = {"models.tracking.max_age_seconds": 3.0}
        with (
            patch.multiple(self.module, _player_tracker=player, _disc_tracker=disc),
            patch.object(
                self.module, "get_setting", side_effect=lambda key, d=None: settings.get(key, d)
            ),
        ):
            self.module.set_frame_rate(60)
            self.assertEqual(player.max_frames_lost, 180)
            self.module.set_frame_rate(29.97)
            self.assertEqual(player.max_frames_lost, 90)
            # A video without a usable frame rate keeps the last one
            self.module.set_frame_rate(0)
            self.assertEqual(player.max_frames_lost, 90)
        self.module.set_frame_rate(30)

    def test_a_bigger_step_keeps_lost_tracks_as_long_and_moves_them_as_far(self):
        # ByteTrack counts its calls: x, y, aspect, height and their speeds per call
        track = SimpleNamespace(
            mean=np.array([10.0, 20.0, 0.5, 90.0, 2.0, -1.0, 0.0, 0.5]), covariance=np.eye(8)
        )
        player, disc = (
            SimpleNamespace(args=SimpleNamespace(), tracked_stracks=tracks, lost_stracks=[])
            for tracks in ([track], [])
        )
        settings = {"models.tracking.max_age_seconds": 3.0}
        with (
            patch.multiple(self.module, _player_tracker=player, _disc_tracker=disc),
            patch.object(
                self.module, "get_setting", side_effect=lambda key, d=None: settings.get(key, d)
            ),
        ):
            self.module.set_frame_rate(60)
            self.assertEqual(player.max_frames_lost, 180)
            self.module._frames_per_step = 2  # "Real time": every second frame
            self.module._step_changed(1)
        self.module._frames_per_step = 1
        self.assertEqual((player.max_frames_lost, disc.max_frames_lost), (90, 90))
        np.testing.assert_allclose(track.mean, [10, 20, 0.5, 90, 4, -2, 0, 1])
        self.assertEqual((track.covariance[4, 4], track.covariance[0, 0]), (4.0, 1.0))

    def test_the_clock_counts_video_frames_not_calls(self):
        self.module.reset_tracker()
        with patch.multiple(self.module, _player_tracker=bytetrack(), _disc_tracker=bytetrack()):
            for index in (100, 102, 104):
                self.module.run_tracking(np.zeros((8, 8, 3)), [], index)
        self.assertEqual((self.module._frame_count, self.module._calls), (5, 3))
        self.module.reset_tracker()

    def test_a_trail_reaches_back_a_time_however_many_frames_are_skipped(self):
        settings = {
            "models.tracking.track_history_length": 300,
            "models.tracking.trail_seconds": 4.0,
        }
        self.module.set_frame_rate(60.0)
        for step, points in ((1, 240), (10, 24)):
            self.module._track_histories.clear()
            self.module._frames_per_step = step
            with patch.object(
                self.module, "get_setting", side_effect=lambda key, default=None: settings[key]
            ):
                for index in range(400):
                    self.module._update_track_history(1, (index, 0))
            self.assertEqual(len(self.module._track_histories[1]), points)
        self.module._frames_per_step = 1
        self.module._track_histories.clear()

    def test_a_trail_does_not_bob_with_the_stride(self):
        # A player runs to the right; the bottom of the box goes up and down 6 px a stride
        steps = np.arange(60)
        trail = np.column_stack([100 + 5.0 * steps, 500 + 6 * np.sin(steps * 2 * np.pi / 10)])
        evened = self.module.evened_out(trail, 5)
        self.assertLess(np.abs(evened[10:-10, 1] - 500).max(), 1.0)
        np.testing.assert_allclose(evened[10:-10, 0], trail[10:-10, 0], atol=1e-6)
        # The ends stay where they are: the newest point is where the feet are
        np.testing.assert_allclose(evened[[0, -1]], trail[[0, -1]])
        # A turn is still a turn, a little rounder
        corner = np.array([(x, 0.0) for x in range(20)] + [(19.0, y) for y in range(1, 20)])
        self.assertLess(np.abs(self.module.evened_out(corner, 3) - corner).max(), 2.5)

    def test_a_trail_ends_where_the_ground_has_left_the_picture(self):
        self.module._track_histories.clear()
        self.module._track_histories[1] = np.array(
            [(100, 900), (100, 1100), (100, 600), (110, 500)], dtype=np.float32
        )
        with patch.multiple(self.module, _player_tracker=None, _disc_tracker=None):
            # The drone flies forward: what is low in the picture leaves it at the bottom,
            # and what was below y = 1000 is behind the camera
            forward = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, -0.001, 1.0]])
            self.module.apply_camera_motion(forward)
        trail = self.module.get_track_histories()[1]
        self.assertEqual(len(trail), 2)
        np.testing.assert_allclose(trail, [(250, 1500), (220, 1000)], atol=1)
        self.module._track_histories.clear()

    def test_camera_motion_moves_trails_and_where_tracks_expect_their_players(self):
        # x, y, aspect, height and their speeds
        track = SimpleNamespace(mean=np.array([400.0, 300.0, 0.4, 100.0, 5.0, 0.0, 0.0, 1.0]))
        self.module._track_histories[1] = np.array([(400, 350), (410, 350)], dtype=np.float32)

        with patch.multiple(
            self.module, _player_tracker=bytetrack(tracks=[track]), _disc_tracker=None
        ):
            # The camera pans: the picture moves 20 px left and 4 px up
            pan = np.array([[1.0, 0, -20], [0, 1, -4], [0, 0, 1]])
            self.module.apply_camera_motion(pan)
            np.testing.assert_allclose(track.mean, [380, 296, 0.4, 100, 5, 0, 0, 1])
            np.testing.assert_allclose(
                self.module.get_track_histories()[1], [(380, 346), (390, 346)]
            )

            # The camera zooms in by 10% around the picture's corner
            zoom = np.diag([1.1, 1.1, 1.0])
            self.module.apply_camera_motion(zoom)
            np.testing.assert_allclose(track.mean, [418, 325.6, 0.4, 110, 5.5, 0, 0, 1.1])


if __name__ == "__main__":
    unittest.main()
