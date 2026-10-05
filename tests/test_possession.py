"""Possession: the holder changes only after the disc stays somewhere else for several frames."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from support import load_module

CONFIRM_FRAMES = 3


def player(track_id, x1, y1, x2, y2):
    box = [x1, y1, x2, y2]
    return SimpleNamespace(track_id=track_id, class_name="player", to_ltrb=lambda: box)


def disc(x, y, confidence=0.8):
    return {"class_name": "disc", "confidence": confidence, "bbox": [x - 5, y - 5, x + 5, y + 5]}


class PossessionTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.possession")
        settings = {
            "models.possession.confirm_frames": CONFIRM_FRAMES,
            "models.possession.box_margin": 0.1,
        }
        patcher = patch.object(
            self.module, "get_setting", side_effect=lambda key, d=None: settings[key]
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.tracker = self.module.PossessionTracker()
        self.thrower = player(1, 100, 100, 160, 300)
        self.receiver = player(2, 600, 100, 660, 300)
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
