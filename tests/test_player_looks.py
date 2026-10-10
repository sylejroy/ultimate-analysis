"""Players looked at frame by frame: numbers by looks, what they did, the roster kept."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from support import load_module

WHITE = (230, 230, 230)
# A player is a patch of one colour here, and "looks like" whoever has that colour
ANNA, BEN = (200, 30, 30), (30, 200, 30)


def player(track_id, box, colour=WHITE):
    return SimpleNamespace(
        track_id=track_id, class_name="player", team_colour=colour, to_ltrb=lambda: box
    )


def by_colour(_network, crops):
    vectors = np.array([crop.reshape(-1, 3).mean(axis=0) for crop in crops], dtype=float)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class PlayerLooksTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.player_looks")
        self.roster_module = sys.modules[self.module.PlayerRoster.__module__]
        reid = load_module("processing.reid")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        settings = {
            "models.reid.roster_folder": self.folder.name,
            "models.reid.look_every_frames": 1,
            "models.reid.in_background": False,
            "models.reid.rejoin_seconds": 10.0,
        }
        for owner, name, replacement in (
            (reid, "embed", by_colour),
            (reid, "sharpness", lambda crop: 1.0),
            (self.module, "get_setting", lambda key, default=None: settings.get(key, default)),
        ):
            patcher = patch.object(owner, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.looks = self.with_network()
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.frame[10:60, 10:40] = ANNA
        self.frame[10:60, 100:130] = BEN
        self.boxes = {"anna": (10, 10, 40, 60), "ben": (100, 10, 130, 60)}

    def with_network(self):
        looks = self.module.PlayerLooks()
        looks._embedder, looks._tried_to_load = object(), True
        return looks

    def watch(self, tracks, numbers, frames, start=0, holder=None):
        for frame_index in range(start, start + frames):
            found = self.looks.update(self.frame, tracks, frame_index, numbers, holder)
        return found

    def test_another_network_keeps_a_roster_of_its_own(self):
        enough = self.roster_module.MIN_LOOKS + 2
        self.looks.new_video("game.mp4")
        self.watch([player(1, self.boxes["anna"])], {1: "17"}, enough)
        self.assertTrue(self.looks.set_weights("models/another_run/best.pt"))
        self.assertFalse(self.looks.set_weights("models/another_run/best.pt"))
        # What the first network knew is written down, and is not this one's
        self.assertTrue((Path(self.folder.name) / "game.json").exists())
        self.assertEqual(self.looks.roster.entries, {})
        self.looks._embedder, self.looks._tried_to_load = object(), True
        self.watch([player(1, self.boxes["anna"])], {1: "17"}, enough, start=100)
        self.looks.save()
        self.assertTrue((Path(self.folder.name) / "another_run" / "game.json").exists())
        # Back to the network of the settings: its roster is there again
        self.assertTrue(self.looks.set_weights(None))
        self.assertEqual(len(self.looks.roster.entries), 1)

    def test_nothing_is_done_without_the_network(self):
        looks = self.module.PlayerLooks()
        looks._tried_to_load = True
        self.assertEqual(looks.update(self.frame, [player(1, self.boxes["anna"])], 0, {}), {})
        self.assertEqual(looks.roster.entries, {})

    def test_a_player_seen_again_without_a_number_has_the_number_read_before(self):
        enough = self.roster_module.MIN_LOOKS + 2
        tracks = [player(1, self.boxes["anna"]), player(2, self.boxes["ben"])]
        self.assertEqual(self.watch(tracks, {1: "17"}, enough), {})

        # A cut; the tracker numbers its tracks anew and no number is in view
        self.looks.cut()
        tracks = [player(5, self.boxes["ben"]), player(6, self.boxes["anna"])]
        self.assertEqual(self.watch(tracks, {}, enough, start=1000), {6: "17"})

    def test_what_a_player_did_is_counted_on_their_entry(self):
        self.looks.frame_rate = 10.0
        tracks = [player(1, self.boxes["anna"]), player(2, self.boxes["ben"])]
        enough = self.roster_module.MIN_LOOKS + 2
        self.watch(tracks, {1: "17"}, enough)
        self.watch(tracks, {1: "17"}, 20, start=enough, holder=1)
        (anna,) = [row for row in self.looks.roster.table() if row["number"] == "17"]
        self.assertAlmostEqual(anna["seconds_with_disc"], 2.0, places=5)
        self.assertEqual(anna["times_with_disc"], 1.0)
        self.assertGreater(anna["seconds_seen"], 2.0)

    def test_how_far_a_player_ran_is_added_up_from_where_they_stand(self):
        self.looks.frame_rate = 10.0
        anna = [player(1, self.boxes["anna"])]
        enough = self.roster_module.MIN_LOOKS + 2
        self.watch(anna, {1: "17"}, enough)
        # Three yards a second for four seconds, then a jump no one can run
        for step in range(41):
            frame_index = enough + step
            self.looks.update(self.frame, anna, frame_index, {}, None, {1: (10.0, 0.3 * step)})
        self.looks.update(self.frame, anna, enough + 46, {}, None, {1: (60.0, 12.0)})
        (row,) = self.looks.roster.table()
        self.assertAlmostEqual(row["distance_run"], 12.0, places=5)

    def test_a_pass_counts_its_length_for_thrower_and_receiver(self):
        self.looks.frame_rate = 10.0
        anna, ben = player(1, self.boxes["anna"]), player(2, self.boxes["ben"])
        anna.team = ben.team = 0
        carl = player(3, (150, 10, 180, 60), colour=(40, 40, 40))
        carl.team = 1
        self.frame[10:60, 150:180] = (30, 30, 200)
        tracks = [anna, ben, carl]
        places = {1: (10.0, 20.0), 2: (10.0, 50.0), 3: (20.0, 55.0)}
        enough = self.roster_module.MIN_LOOKS + 2
        for frame_index in range(enough):
            self.looks.update(self.frame, tracks, frame_index, {1: "17", 2: "4", 3: "9"}, 1, places)
        # In the air, caught by a teammate thirty yards on, then taken by the other team
        for frame_index, holder in ((enough, None), (enough + 5, 2), (enough + 20, 3)):
            self.looks.update(self.frame, tracks, frame_index, {}, holder, places)
        rows = {row["number"]: row for row in self.looks.roster.table()}
        self.assertEqual((rows["17"]["passes_thrown"], rows["17"]["distance_thrown"]), (1.0, 30.0))
        self.assertEqual((rows["4"]["catches"], rows["4"]["distance_received"]), (1.0, 30.0))
        # A turnover is no pass
        self.assertNotIn("passes_thrown", rows["4"])
        self.assertNotIn("catches", rows["9"])
        self.assertEqual(rows["9"]["times_with_disc"], 1.0)

    def test_the_roster_of_a_video_is_there_again_the_next_time(self):
        self.looks.new_video("game.mp4")
        tracks = [player(1, self.boxes["anna"]), player(2, self.boxes["ben"])]
        self.watch(tracks, {1: "17"}, self.roster_module.MIN_LOOKS + 2)
        self.looks.new_video("another.mp4")
        self.assertEqual(self.looks.roster.entries, {})
        self.assertTrue((Path(self.folder.name) / "game.json").exists())

        self.looks = self.with_network()
        self.looks.new_video("game.mp4")
        tracks = [player(8, self.boxes["anna"])]
        found = self.watch(tracks, {}, self.roster_module.MIN_LOOKS + 2)
        self.assertEqual(found, {8: "17"})

    def test_a_new_track_that_looks_like_a_lost_teammate_is_that_player(self):
        anna, ben = player(1, self.boxes["anna"]), player(2, self.boxes["ben"])
        anna.team = ben.team = 0
        self.watch([anna, ben], {}, 10)
        # Both are lost; a second later two new tracks appear, the other way round
        again = [player(7, self.boxes["ben"]), player(8, self.boxes["anna"])]
        again[0].team = again[1].team = 0
        self.watch(again, {}, 10, start=70)
        self.assertEqual(sorted(self.looks.returned([7, 8], [1, 2], 80)), [(7, 2), (8, 1)])
        # Not of the other team, not a player who is still there, not after too long
        again[1].team = 1
        self.looks._profiles[8].team = 1
        self.assertEqual(self.looks.returned([7, 8], [1, 2], 80), [(7, 2)])
        self.assertEqual(self.looks.returned([7], [1], 80), [])
        self.assertEqual(self.looks.returned([7], [2], 80 + 100000), [])

    def test_two_lost_teammates_who_look_alike_are_not_told_apart(self):
        twin = self.boxes["anna"]
        first, second = player(1, twin), player(2, twin)
        first.team = second.team = 0
        self.watch([first, second], {}, 10)
        new = player(7, twin)
        new.team = 0
        self.watch([new], {}, 10, start=70)
        self.assertEqual(self.looks.returned([7], [1, 2], 80), [])

    def test_the_teams_are_told_from_what_the_players_look_like(self):
        # Three players a side: one kit reddish, one greenish, each player a shade of it
        reds = [(200 - 10 * n, 30, 40 + 10 * n) for n in range(3)]
        greens = [(30, 200 - 10 * n, 40 + 10 * n) for n in range(3)]
        tracks = []
        for index, colour in enumerate(reds + greens):
            x = 5 + 30 * index
            self.frame[10:60, x : x + 25] = colour
            tracks.append(player(index + 1, (x, 10, x + 25, 60)))
        self.watch(tracks, {}, 12)
        # The tracker has one of the reds in the other team, and one without a team
        tracker = {1: 0, 2: 1, 3: None, 4: 1, 5: 1, 6: 1}
        self.assertEqual(self.looks.teams(tracker), {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1})
        # Too few players looked at, or no team from the tracker to number them by
        self.assertEqual(self.looks.teams(dict.fromkeys(tracker)), {})
        self.looks.cut()
        self.watch(tracks[:3], {}, 12)
        self.assertEqual(self.looks.teams(tracker), {})

    def cross(self, after):
        """Anna and Ben apart, then their boxes overlapping, then apart again with the
        tracks as `after` gives them; returns what `swapped` said over the frames."""
        self.frame[10:60, 40:100] = 0
        apart = [player(1, self.boxes["anna"]), player(2, self.boxes["ben"])]
        self.watch(apart, {}, 10)
        self.watch([player(1, (10, 10, 40, 60)), player(2, (20, 10, 50, 60))], {}, 5, start=10)
        said = []
        for frame_index in range(15, 22):
            self.looks.update(self.frame, after, frame_index, {})
            said += self.looks.swapped([1, 2], frame_index)
            if said:
                break  # The caller exchanges the tracks' players from here on
        return said

    def test_tracks_that_changed_players_in_a_crossing_are_found_out(self):
        changed = [player(1, self.boxes["ben"]), player(2, self.boxes["anna"])]
        self.assertEqual(self.cross(changed), [(1, 2)])
        # What was seen since is put with the player it is of: track 1 is Ben now
        anna, ben = (self.looks._profiles[track_id] for track_id in (1, 2))
        self.assertGreater(anna.looks, 10)
        for profile, colour in ((anna, ANNA), (ben, BEN)):
            np.testing.assert_allclose(
                profile.total / np.linalg.norm(profile.total),
                np.array(colour) / np.linalg.norm(colour),
                atol=0.02,
            )

    def test_tracks_that_kept_their_players_in_a_crossing_are_left_alone(self):
        kept = [player(1, self.boxes["anna"]), player(2, self.boxes["ben"])]
        self.assertEqual(self.cross(kept), [])
        self.assertEqual(self.looks._crossings, {})

    def test_a_crossing_is_not_settled_while_one_of_the_two_is_out_of_the_frame(self):
        import numpy as np

        looks = self.looks
        module = self.module
        for track_id, vector in ((1, [1.0, 0.0]), (2, [0.0, 1.0])):
            profile = module._Profile(first=0, last=10)
            for _ in range(module.REJOIN_LOOKS):
                profile.learn(np.array(vector))
            # Since the crossing each looks like the other did before
            profile.held = [(np.array(vector[::-1]), None)] * module.CROSSING_LOOKS
            looks._profiles[track_id] = profile
        looks._crossings = {(1, 2): 10}
        self.assertEqual(looks.swapped([1], 12), [])
        self.assertEqual(looks.swapped([1, 2], 13), [(1, 2)])

    def test_two_tracks_that_exchanged_names_exchange_what_is_theirs(self):
        self.looks._ran_from = {1: (0, np.array([1.0, 1.0]))}
        self.looks._crossings = {(1, 5): 3}
        self.looks.swap(1, 2)
        self.assertEqual(list(self.looks._ran_from), [2])
        self.assertEqual(self.looks._crossings, {(2, 5): 3})

    def test_a_renamed_track_keeps_what_was_seen_of_it(self):
        enough = self.roster_module.MIN_LOOKS + 2
        self.watch([player(1, self.boxes["anna"])], {1: "17"}, enough)
        self.looks.rename(1, 9)
        self.assertEqual(self.looks.roster.number_of(9), "17")
        self.assertIsNone(self.looks.roster.number_of(1))


if __name__ == "__main__":
    unittest.main()
