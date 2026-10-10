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
STRAY_FRAMES = 200
FRAME_RATE = 10.0


def player(track_id, x1, y1, x2, y2, team=None):
    box = [x1, y1, x2, y2]
    return SimpleNamespace(track_id=track_id, class_name="player", to_ltrb=lambda: box, team=team)


def disc(x, y, confidence=0.8):
    return {"class_name": "disc", "confidence": confidence, "bbox": [x - 5, y - 5, x + 5, y + 5]}


class HolderNetworkTest(unittest.TestCase):
    """A disc in two players' boxes: the one a holder network is clearly surest of."""

    def setUp(self):
        self.module = load_module("processing.possession")
        self.tracks = [player(1, 0, 0, 100, 200), player(2, 60, 0, 160, 200)]
        self.disc = (75, 95, 85, 105)  # In both boxes, nearer the middle of the first

    def test_without_a_network_the_nearest_middle_decides(self):
        self.assertEqual(self.module.player_at_disc(self.disc, self.tracks, 0.15), 1)
        self.assertEqual(self.module.players_at_disc(self.disc, self.tracks, 0.15), [1, 2])

    def test_the_network_decides_where_it_is_clearly_surer_of_one(self):
        at = self.module.player_at_disc
        self.assertEqual(at(self.disc, self.tracks, 0.15, {1: 0.2, 2: 0.9}), 2)
        # Not clearly, or only one of them looked at: as without
        self.assertEqual(at(self.disc, self.tracks, 0.15, {1: 0.6, 2: 0.7}), 1)
        self.assertEqual(at(self.disc, self.tracks, 0.15, {2: 0.9}), 1)

    def looking(self, sure, scores, frames=40):
        settings = {"models.holder.sure": sure, "models.holder.weight": 0.5}
        get = self.module.get_setting
        with patch.object(
            self.module,
            "get_setting",
            lambda key, default=None: settings.get(key, get(key, default)),
        ):
            tracker = self.module.PossessionTracker()
            for frame in range(frames):
                tracker.update([], self.tracks, frame, scores)
        return tracker.holder_id

    def test_without_a_disc_a_player_the_network_is_sure_of_becomes_the_holder(self):
        self.assertEqual(self.looking(0.8, {1: 0.1, 2: 0.95}), 2)
        # Switched off, not sure enough, or two it is about as sure of: nobody
        self.assertIsNone(self.looking(0.0, {1: 0.1, 2: 0.95}))
        self.assertIsNone(self.looking(0.8, {1: 0.1, 2: 0.7}))
        self.assertIsNone(self.looking(0.8, {1: 0.9, 2: 0.95}))
        # It takes twice as long as with the disc itself in sight
        self.assertIsNone(self.looking(0.8, {1: 0.1, 2: 0.95}, frames=10))

    def test_what_the_network_said_is_recorded_and_played_back(self):
        replaying = load_module("processing.possession_replay")
        tracker = replaying.RecordingPossessionTracker()
        discs = [{"class_name": "disc", "confidence": 0.9, "bbox": list(self.disc)}]
        for frame in range(30):
            tracker.update(discs, self.tracks, frame, {1: 0.1, 2: 0.9})
            tracker.frame_done(frame)
        self.assertEqual(tracker.holder_id, 2)
        self.assertEqual(replaying.replay(tracker.calls)[-1][1], 2)


class PossessionTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.possession")
        self.settings = settings = {
            "models.possession.confirm_seconds": CONFIRM_FRAMES / FRAME_RATE,
            "models.possession.beside_holder_seconds": BESIDE_FRAMES / FRAME_RATE,
            "models.possession.holder_memory_seconds": MEMORY_FRAMES / FRAME_RATE,
            "models.possession.ground_seconds": GROUND_FRAMES / FRAME_RATE,
            "models.possession.other_team_seconds": OTHER_TEAM_FRAMES / FRAME_RATE,
            "models.possession.reach": 1.0,
            "models.possession.thrown_speed": 0.0,
            "models.possession.stray_seconds": STRAY_FRAMES / FRAME_RATE,
            "models.possession.covered_seconds": MEMORY_FRAMES / FRAME_RATE,
            "models.possession.teammate_seconds": 0.0,
            "models.possession.box_margin": 0.1,
            "models.holder.sure": 0.0,
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

    def test_a_disc_flying_away_within_reach_is_thrown_where_that_is_asked_for(self):
        # 50 px further from a box 200 px tall in every frame: 2.5 heights a second
        flight = [disc(x, 150) for x in (190, 240, 290, 340)]
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        for sighting in flight:
            self.assertEqual(self.see(sighting), 1)  # Within reach: held out, as far as is asked

        self.tracker.reset()
        self.settings["models.possession.thrown_speed"] = 1.5
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        for sighting in flight[:-1]:
            self.assertEqual(self.see(sighting), 1)
        self.assertIsNone(self.see(flight[-1]))
        # Nobody's from the first sighting outside the box, not from when it was sure
        self.assertEqual(self.tracker.since, CONFIRM_FRAMES)

    def test_a_disc_held_out_and_moved_about_is_not_thrown(self):
        self.settings["models.possession.thrown_speed"] = 1.5
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        # A fake: out fast, and back; then held out, placed a little differently each time
        for x in (190, 240, 250, 200, 215, 208, 222, 210, 218, 206):
            self.assertEqual(self.see(disc(x, 150)), 1)

    def test_a_holder_who_gets_a_new_track_number_stays_the_holder(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        # The tracker loses track 1 and begins track 9 at nearly the same place
        self.tracks[0] = player(9, 104, 100, 164, 300, team=0)
        self.assertEqual(self.see(), 9)
        self.assertEqual(self.tracker.holder_box, (104.0, 100.0, 164.0, 300.0))
        # A track that was there before does not become the holder by standing there
        self.tracks[0] = player(2, 600, 100, 660, 300, team=0)
        self.tracks[1] = player(7, 300, 100, 360, 300, team=1)
        self.see()
        self.tracks[1] = player(7, 104, 100, 164, 300, team=1)
        self.assertEqual(self.see(), 9)
        # Their place is still known
        self.assertEqual(self.tracker.holder_box, (104.0, 100.0, 164.0, 300.0))

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

    def test_a_pass_to_a_teammate_is_confirmed_sooner_than_one_to_the_other_team(self):
        self.settings["models.possession.teammate_seconds"] = 1 / FRAME_RATE
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        self.assertEqual(self.see(disc(630, 150)), 2)  # A teammate: at the first sighting

        self.tracks.append(player(3, 900, 100, 960, 300, team=1))
        self.assertEqual(self.see(disc(930, 150), frames=CONFIRM_FRAMES - 1), 2)
        self.assertEqual(self.see(disc(930, 150)), 3)

    def test_a_defender_beside_a_receiver_takes_longer_to_become_the_holder(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        # A defender whose box touches that of a player of the team with the disc
        self.tracks.append(player(3, 380, 100, 440, 300, team=1))
        self.tracks.append(player(4, 430, 100, 490, 300, team=0))
        self.assertEqual(self.see(disc(390, 150), frames=OTHER_TEAM_FRAMES - 1), 1)
        self.assertEqual(self.see(disc(390, 150)), 3)
        self.assertEqual(self.tracker.team, 1)

    def test_a_player_of_the_other_team_alone_with_the_disc_has_it_at_once(self):
        # The pull: caught by the other team with nobody near
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        self.tracks.append(player(3, 380, 100, 440, 300, team=1))
        self.assertEqual(self.see(disc(410, 150), frames=CONFIRM_FRAMES), 3)
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

    def test_the_first_frame_after_a_reset_is_one_frame_wherever_in_the_video(self):
        self.tracker.update([disc(130, 150)], self.tracks, 50000)
        self.tracker.update([disc(130, 150)], self.tracks, 50001)
        self.assertIsNone(self.tracker.holder_id)

    def test_a_disc_picked_up_from_the_ground_is_off_it_from_then_not_from_before(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        for frame in range(3, 3 + GROUND_FRAMES + 2):
            self.tracker.update([disc(400, 400)], self.tracks, frame)
        self.assertEqual((self.tracker.disc_state, self.tracker.since), ("ground", 3))
        # Someone stands over it: no longer lying free, from this frame on
        self.tracks.append(player(3, 380, 250, 440, 450, team=1))
        self.tracker.update([disc(400, 400)], self.tracks, 30)
        self.assertEqual((self.tracker.disc_state, self.tracker.since), ("air", 30))

    def test_a_holder_taken_for_the_other_team_for_a_moment_keeps_the_disc_for_theirs(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES * 4)
        self.assertEqual(self.tracker.team, 0)
        self.thrower.team = 1  # Sun on the shirt
        self.see(disc(130, 150), frames=3)
        self.assertEqual(self.tracker.team, 0)
        # Seen as the other team for longer than as the first: that is their team
        self.see(disc(130, 150), frames=CONFIRM_FRAMES * 5)
        self.assertEqual(self.tracker.team, 1)

    def test_the_time_in_the_air_counts_from_when_the_disc_left_while_it_is_seen(self):
        for frame in range(100, 100 + CONFIRM_FRAMES):
            self.tracker.update([disc(130, 150)], self.tracks, frame)
        self.assertIsNone(self.tracker.flight_seconds)  # Held
        for frame in range(120, 130):
            self.tracker.update([disc(300 + 20 * (frame - 120), 150)], self.tracks, frame)
        self.assertEqual(self.tracker.disc_state, "air")
        # From the first frame it was out of the thrower's reach (frame 124) to the last
        self.assertAlmostEqual(self.tracker.flight_seconds, 5 / FRAME_RATE)
        # Not seen for a while: nothing says it still flies
        for frame in range(130, 140):
            self.tracker.update([], self.tracks, frame)
        self.assertIsNone(self.tracker.flight_seconds)

    def test_something_lying_far_from_where_the_disc_was_is_no_turnover(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        # The disc is in the thrower's hands; the model finds a brick mark instead,
        # across the field, from one frame to the next
        self.see(disc(1500, 900), frames=GROUND_FRAMES * 3)
        self.assertEqual((self.tracker.team, self.tracker.on_ground), (0, False))
        # Nor after the disc was not seen for a while
        self.tracker.reset()
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        self.see(frames=30)
        self.see(disc(300, 400), frames=GROUND_FRAMES * 3)
        self.assertEqual((self.tracker.team, self.tracker.on_ground), (0, False))

    def covered_catch(self):
        """A pass to a teammate whose track ends as the mark steps in front of them."""
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        self.see(disc(380, 150), frames=2)
        self.see(disc(630, 150))
        mark = player(3, 610, 110, 670, 310, team=1)
        self.tracks = [self.thrower, mark]
        return self.see(disc(632, 152), frames=CONFIRM_FRAMES + 2)

    def test_a_disc_at_the_mark_is_the_covered_receivers(self):
        self.assertEqual(self.covered_catch(), 2)
        self.assertEqual(self.tracker.team, 0)
        # Seen again under a new number, they are the holder still
        self.tracks.append(player(9, 600, 100, 660, 300, team=0))
        self.assertEqual(self.see(disc(630, 150), frames=2), 9)

    def test_a_disc_at_a_defender_nobody_stood_behind_is_theirs(self):
        self.settings["models.possession.covered_seconds"] = 0.0
        self.assertEqual(self.covered_catch(), 3)
        self.assertEqual(self.tracker.team, 1)

    def test_what_lies_at_one_place_for_long_is_not_listened_to(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        cap = disc(400, 420, confidence=0.9)
        # A cap on the grass, surer than the disc: taken for the disc at first
        self.see(cap, disc(130, 150, 0.5), frames=STRAY_FRAMES)
        self.assertIsNone(self.tracker.holder_id)
        self.assertEqual(self.tracker.strays, [])

        # ... and after that the disc in the thrower's hands is, though it is less sure
        self.see(cap, disc(130, 150, 0.5), frames=CONFIRM_FRAMES + 1)
        self.assertEqual(self.tracker.holder_id, 1)
        self.assertEqual(self.tracker.strays, [(400.0, 420.0)])
        # Alone in the picture it says nothing, nor when somebody walks over it
        self.assertEqual(self.see(cap, frames=GROUND_FRAMES * 2), 1)
        self.tracks.append(player(3, 370, 250, 430, 450, team=1))
        self.assertEqual(self.see(cap, frames=BESIDE_FRAMES * 2), 1)
        self.assertEqual((self.tracker.team, self.tracker.on_ground), (0, False))

    def test_a_disc_left_lying_for_long_counts_again_once_it_is_picked_up(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        self.see(disc(400, 420), frames=STRAY_FRAMES + 5)
        self.assertEqual((self.tracker.team, self.tracker.on_ground), (1, True))
        self.assertEqual(len(self.tracker.strays), 1)

        # Walked up to, it still lies; lifted, it is in their hands
        self.tracks.append(player(3, 370, 250, 430, 450, team=1))
        self.assertIsNone(self.see(disc(400, 420), frames=CONFIRM_FRAMES * 2))
        self.assertTrue(self.tracker.on_ground)
        self.assertEqual(self.see(disc(405, 340), frames=CONFIRM_FRAMES), 3)
        self.assertEqual((self.tracker.team, self.tracker.on_ground), (1, False))

    def test_a_thing_lying_for_long_is_followed_when_the_camera_moves(self):
        self.see(disc(130, 150), frames=CONFIRM_FRAMES)
        self.see(disc(400, 420), frames=STRAY_FRAMES + 1)
        shift = self.module.np.array([[1.0, 0.0, 300.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.tracker.move(shift)

        self.assertEqual(self.tracker.strays, [(700.0, 420.0)])
        # It is known where it now shows, and what shows where it was is not it
        self.tracker.update([disc(700, 420), disc(400, 420, 0.5)], self.tracks)
        self.assertEqual(self.tracker._last_sighting[1], (400.0, 420.0))

    def test_nothing_is_taken_for_lying_where_that_is_turned_off(self):
        self.settings["models.possession.stray_seconds"] = 0.0
        self.see(disc(400, 420), frames=STRAY_FRAMES * 2)
        self.assertEqual(self.tracker.strays, [])
        self.assertEqual(self.tracker._disc_seen, self.tracker._frame)

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
