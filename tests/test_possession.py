"""Possession: the holder changes only after the disc stays somewhere else for a while,
the mark does not take it, and a disc lying on the ground is a turnover."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from support import load_module

CONFIRM_FRAMES = 3
BESIDE_FRAMES = 12
MEMORY_FRAMES = 20
GROUND_FRAMES = 15
OTHER_TEAM_FRAMES = 8
FRAME_RATE = 10.0


def player(track_id, x1, y1, x2, y2, team=None):
    box = [x1, y1, x2, y2]
    return SimpleNamespace(track_id=track_id, class_name="player", to_ltrb=lambda: box, team=team)


def disc(x, y, confidence=0.8):
    return {"class_name": "disc", "confidence": confidence, "bbox": [x - 5, y - 5, x + 5, y + 5]}


class PossessionTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.possession")
        settings = {
            "models.possession.confirm_seconds": CONFIRM_FRAMES / FRAME_RATE,
            "models.possession.beside_holder_seconds": BESIDE_FRAMES / FRAME_RATE,
            "models.possession.holder_memory_seconds": MEMORY_FRAMES / FRAME_RATE,
            "models.possession.ground_seconds": GROUND_FRAMES / FRAME_RATE,
            "models.possession.other_team_seconds": OTHER_TEAM_FRAMES / FRAME_RATE,
            "models.possession.reach": 0.35,
            "models.possession.box_margin": 0.1,
        }
        patcher = patch.object(
            self.module, "get_setting", side_effect=lambda key, d=None: settings[key]
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.tracker = self.module.PossessionTracker()
        self.tracker.frame_rate = FRAME_RATE
        self.thrower = player(1, 100, 100, 160, 300, team=0)
        self.receiver = player(2, 600, 100, 660, 300, team=0)
        self.tracks = [self.thrower, self.receiver]

    def see(self, *discs, frames=1):
        for _ in range(frames):
            holder = self.tracker.update(list(discs), self.tracks)
        return holder

    def test_holder_is_confirmed_after_several_frames_not_on_the_first(self):
        self.assertIsNone(self.see(disc(130, 150), frames=CONFIRM_FRAMES - 1))
        self.assertEqual(self.see(disc(130, 150)), 1)

    def test_disc_passing_another_player_does_not_change_the_holder(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)

        self.assertEqual(self.see(disc(630, 150), frames=CONFIRM_FRAMES - 1), 1)
        # Back at the holder: the count for the other player starts again
        self.assertEqual(self.see(disc(130, 150)), 1)
        self.assertEqual(self.see(disc(630, 150), frames=CONFIRM_FRAMES - 1), 1)

    def test_throw_and_catch(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)

        # In flight between the players: nobody holds it once that is confirmed
        self.assertEqual(self.see(disc(400, 150), frames=CONFIRM_FRAMES - 1), 1)
        self.assertIsNone(self.see(disc(400, 150)))

        self.assertIsNone(self.see(disc(630, 150), frames=CONFIRM_FRAMES - 1))
        self.assertEqual(self.see(disc(630, 150)), 2)

    def test_frames_without_a_disc_change_nothing(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        self.see(disc(630, 150), frames=CONFIRM_FRAMES - 1)

        self.assertEqual(self.see(frames=50), 1)
        # The count continues where it was when the disc is seen again
        self.assertEqual(self.see(disc(630, 150)), 2)

    def test_disc_just_outside_the_box_and_the_most_confident_disc_count(self):
        # 5 px left of a 60 px wide box is within the 10% margin; 30 px is not
        self.assertEqual(self.module.player_at_disc([90, 145, 100, 155], self.tracks, 0.1), 1)
        self.assertIsNone(self.module.player_at_disc([65, 145, 75, 155], self.tracks, 0.1))

        holder = self.see(disc(630, 150, confidence=0.4), disc(130, 150, confidence=0.9), frames=3)
        self.assertEqual(holder, 1)

    def test_the_durations_follow_the_frame_rate(self):
        self.tracker.frame_rate = 2 * FRAME_RATE
        self.assertIsNone(self.see(disc(130, 150), frames=2 * CONFIRM_FRAMES - 1))
        self.assertEqual(self.see(disc(130, 150)), 1)

    def test_a_disc_in_the_holders_box_stays_theirs_when_the_mark_is_nearer(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        # The mark steps in front: the disc is in both boxes and nearer to the mark's centre
        self.tracks.append(player(3, 130, 100, 190, 300))
        self.assertEqual(self.module.player_at_disc([150, 195, 160, 205], self.tracks, 0.1), 3)
        self.assertEqual(self.see(disc(155, 200), frames=BESIDE_FRAMES * 3), 1)

    def test_a_holder_covered_by_the_mark_keeps_the_disc(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        # Only the mark is found; the disc shows where the holder was
        self.tracks[:] = [player(3, 130, 100, 190, 300), self.receiver]
        self.assertEqual(self.see(disc(140, 150), frames=MEMORY_FRAMES - 1), 1)

    def test_a_disc_held_out_past_the_mark_is_not_the_marks(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        # The disc is outside the holder's box and inside that of the mark beside them
        self.tracks.append(player(3, 150, 100, 210, 300))
        self.assertEqual(self.see(disc(200, 150), frames=BESIDE_FRAMES - 1), 1)
        # Someone right by the holder who has it for that long was the holder all along
        self.assertEqual(self.see(disc(200, 150)), 3)

    def test_a_pass_to_someone_away_from_the_holder_is_confirmed_quickly(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        self.assertEqual(self.see(disc(630, 150), frames=CONFIRM_FRAMES), 2)

    def test_a_flickering_detection_still_gives_a_holder(self):
        # Two frames at the receiver, one in which the disc is taken for something far off
        for _ in range(CONFIRM_FRAMES):
            self.see(disc(630, 150), frames=2)
            self.see(disc(400, 400))
        self.assertEqual(self.tracker.holder_id, 2)

    def test_a_disc_held_out_at_arms_length_is_not_in_the_air(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        # 50 px beside a box 200 px tall: outside the box and its margin, within reach
        self.assertEqual(self.see(disc(210, 150), frames=CONFIRM_FRAMES * 5), 1)
        self.assertEqual(self.tracker.disc_state, "held")

    def test_the_team_in_possession_is_the_holders_and_stays_while_the_disc_flies(self):
        self.assertIsNone(self.tracker.team)
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        self.assertEqual(self.tracker.team, 0)
        self.see(disc(300, 150), frames=CONFIRM_FRAMES)
        self.see(disc(400, 150), frames=CONFIRM_FRAMES)
        self.assertEqual((self.tracker.holder_id, self.tracker.team), (None, 0))
        self.assertEqual(self.tracker.disc_state, "air")

    def test_a_disc_lying_on_the_ground_is_a_turnover(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        # Moving on every frame: in the air however long
        for step in range(GROUND_FRAMES * 2):
            self.see(disc(300 + 40 * (step % 5), 400))
        self.assertEqual((self.tracker.team, self.tracker.on_ground), (0, False))

        self.see(disc(400, 400), frames=GROUND_FRAMES)
        self.assertEqual((self.tracker.team, self.tracker.on_ground), (0, False))
        self.see(disc(402, 401))
        self.assertEqual((self.tracker.team, self.tracker.on_ground), (1, True))
        self.assertEqual(self.tracker.disc_state, "ground")
        # It is one turnover however long the disc lies there, and if its detection
        # wanders off for a moment
        self.see(disc(400, 400), frames=GROUND_FRAMES * 3)
        self.see(disc(520, 400), frames=2)
        self.see(disc(400, 400), frames=GROUND_FRAMES * 3)
        self.assertEqual(self.tracker.team, 1)

        # Picked up by a player of the other team
        self.tracks.append(player(3, 380, 250, 440, 450, team=1))
        self.assertEqual(self.see(disc(400, 400), frames=CONFIRM_FRAMES), 3)
        self.assertEqual((self.tracker.team, self.tracker.on_ground), (1, False))

    def test_a_player_of_the_other_team_takes_longer_to_become_the_holder(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        defender = player(3, 380, 100, 440, 300, team=1)
        self.tracks.append(defender)
        self.assertEqual(self.see(disc(410, 150), frames=OTHER_TEAM_FRAMES - 1), 1)
        self.assertEqual(self.see(disc(410, 150)), 3)
        self.assertEqual(self.tracker.team, 1)

    def test_a_change_is_dated_back_to_when_the_disc_was_first_seen_there(self):
        for frame in range(100, 100 + CONFIRM_FRAMES):
            self.tracker.update([disc(130, 150)], self.tracks, frame)
        self.assertEqual((self.tracker.holder_id, self.tracker.since), (1, 100))
        for frame in range(140, 140 + CONFIRM_FRAMES):
            self.tracker.update([disc(630, 150)], self.tracks, frame)
        self.assertEqual((self.tracker.holder_id, self.tracker.since), (2, 140))

    def test_a_frame_counts_for_the_frames_skipped_before_it(self):
        # Every third frame is analysed: one frame is enough for three frames' worth
        self.tracker.update([disc(130, 150)], self.tracks, 100)
        self.assertIsNone(self.tracker.holder_id)
        self.tracker.update([disc(130, 150)], self.tracks, 103)
        self.assertEqual(self.tracker.holder_id, 1)

    def test_a_resting_disc_is_followed_when_the_camera_moves(self):
        import numpy as np

        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        pan = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for step in range(GROUND_FRAMES + 2):
            self.tracker.move(pan)
            self.see(disc(400 + 6 * step, 400))
        self.assertTrue(self.tracker.on_ground)

    def test_overlapping_players_and_reset(self):
        # The disc is inside both boxes; it is nearer to the centre of player 3
        self.tracks.append(player(3, 120, 100, 180, 300))
        self.assertEqual(self.module.player_at_disc([150, 195, 160, 205], self.tracks, 0.1), 3)

        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        self.tracker.reset()
        self.assertIsNone(self.tracker.holder_id)
        self.assertIsNone(self.see(disc(130, 150)))


if __name__ == "__main__":
    unittest.main()
