"""The yes-or-no labelling: which tracks are asked about, and what an answer writes."""

import csv
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ultimate_analysis.utils import judgement_files


def write(path: Path, columns: str, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([columns, *rows]) + "\n")


class SamePlayerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.data = Path(folder.name)
        rows = []
        # Stretch 0: track a, then track b after it; stretch 1: track c with a's number
        for player, stretch, first, number in (
            ("0-1", 0, 100, "7"),
            ("0-2", 0, 200, ""),
            ("0-3", 1, 900, "7"),
        ):
            rows += [
                f"game,{stretch},{first + 10 * i},{player},1,{number},0.9,0,0,1,1" for i in range(5)
            ]
        rows += ["game,1,950,0-9,1,,,0,0,1,1"]  # Too short to judge
        rows += [f"game,1,{900 + i},0-4,0,,,0,0,1,1" for i in range(5)]  # The other team
        write(
            self.data / "source" / "index.csv",
            "video,stretch,frame,player,light_shirt,number,certainty,x1,y1,x2,y2",
            rows,
        )
        self.tracks = judgement_files.load_tracks(self.data / "source")
        self.answers = self.data / "answers" / "pairs.csv"

    def test_tracks_are_read_with_their_crops_in_order_and_their_number(self):
        self.assertEqual(
            set(self.tracks),
            {
                ("game", "0", "0-1"),
                ("game", "0", "0-2"),
                ("game", "1", "0-3"),
                ("game", "1", "0-4"),
            },
        )
        track = self.tracks["game", "0", "0-1"]
        self.assertEqual(track.number, "7")
        self.assertEqual([frame for frame, _ in track.crops], [100, 110, 120, 130, 140])
        self.assertEqual(track.crops[0][1].name, "0-1_100.jpg")
        self.assertEqual(len(track.spread(3)), 3)
        self.assertEqual(track.spread(3)[-1].name, "0-1_140.jpg")

    def test_pairs_are_of_one_game_and_one_kind_of_shirt_and_asked_once(self):
        rng, judged, seen = random.Random(1), set(), set()
        for _ in range(3):
            first, second, why = judgement_files.choose_pair(self.tracks, judged, rng)
            self.assertEqual(why, judgement_files.RANDOM)
            self.assertEqual(first.light_shirt, second.light_shirt)
            pair = judgement_files.pair_of(first.key, second.key)
            self.assertNotIn(pair, seen)
            seen.add(pair)
            judgement_files.save_pair(self.answers, "source", first, second, "same")
            judged = judgement_files.judged_pairs(self.answers)
        # Three light shirts make three pairs; the dark one has nobody to be paired with
        self.assertEqual(judged, seen)
        self.assertIsNone(judgement_files.choose_pair(self.tracks, judged, rng))

    def test_most_pairs_are_ones_the_network_is_unsure_of(self):
        a, b, c = ("game", "0", "0-1"), ("game", "0", "0-2"), ("game", "1", "0-3")
        # a and b are half alike, c is like neither
        vectors = {
            a: np.array([1.0, 0.0, 0.0]),
            b: np.array([0.6, 0.8, 0.0]),
            c: np.array([0.0, 0.0, 1.0]),
        }
        rng, asked = random.Random(3), []
        for _ in range(200):
            first, second, why = judgement_files.choose_pair(self.tracks, set(), rng, vectors)
            asked.append(why)
            if why == judgement_files.UNSURE:
                self.assertEqual({first.key, second.key}, {a, b})
        share = asked.count(judgement_files.UNSURE) / len(asked)
        self.assertAlmostEqual(share, judgement_files.UNSURE_SHARE, delta=0.1)
        # Once that pair is answered there is nothing unsure left: at random from then on
        judged = {judgement_files.pair_of(a, b)}
        for _ in range(20):
            self.assertEqual(
                judgement_files.choose_pair(self.tracks, judged, rng, vectors)[2],
                judgement_files.RANDOM,
            )

    def test_why_a_pair_was_asked_about_is_written_down(self):
        first, second = self.tracks["game", "0", "0-1"], self.tracks["game", "1", "0-3"]
        asked = self.answers.with_name("asked.csv")
        judgement_files.save_asked(asked, first, second, judgement_files.UNSURE, 0.61234)
        with open(asked, newline="") as file:
            (row,) = csv.DictReader(file)
        self.assertEqual((row["why"], row["likeness"]), ("unsure", "0.612"))

    def test_an_answer_is_a_line_and_only_the_known_answers_are_taken(self):
        first, second = self.tracks["game", "0", "0-1"], self.tracks["game", "1", "0-3"]
        judgement_files.save_pair(self.answers, "source", first, second, "first_mixed")
        with open(self.answers, newline="") as file:
            (row,) = csv.DictReader(file)
        self.assertEqual(
            (row["first_stretch"], row["first_player"], row["second_stretch"], row["answer"]),
            ("0", "0-1", "1", "first_mixed"),
        )
        with self.assertRaises(ValueError):
            judgement_files.save_pair(self.answers, "source", first, second, "maybe")

    def test_training_is_told_who_is_one_player_and_what_to_leave_out(self):
        a, b, c = (self.tracks["game", s, p] for s, p in (("0", "0-1"), ("0", "0-2"), ("1", "0-3")))
        judgement_files.save_pair(self.answers, "source", a, b, "different")
        judgement_files.save_pair(self.answers, "source", a, c, "different")
        judgement_files.save_pair(self.answers, "source", a, c, "same")  # Thought again
        judgement_files.save_pair(self.answers, "source", b, c, "first_mixed")
        judgement_files.save_pictures(
            self.answers.with_name("pictures.csv"), "source", a, [100, 110], {110}
        )
        same, mixed, others = judgement_files.what_was_said(self.answers.parent)
        self.assertEqual(same, [(a.key, c.key)])
        self.assertEqual(mixed, {b.key})
        self.assertEqual(others, {("game", "0", "0-1", 110)})

    def test_the_pictures_shown_are_written_down_as_the_players_or_somebody_elses(self):
        track = self.tracks["game", "0", "0-1"]
        shown = [frame for frame, _ in track.chosen(3)]
        self.assertEqual(shown, [100, 120, 140])
        pictures = self.answers.with_name("pictures.csv")
        judgement_files.save_pictures(pictures, "source", track, shown, {120})
        with open(pictures, newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(
            [(row["player"], row["frame"], row["of_the_player"]) for row in rows],
            [("0-1", "100", "1"), ("0-1", "120", "0"), ("0-1", "140", "1")],
        )


class HolderTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.data = Path(folder.name)
        write(
            self.data / "source" / "index.csv",
            "video,stretch,frame,player,holding,nearest",
            [
                "game,5,20,3,0,0",
                "game,5,20,4,0,1",
                "game,5,20,7,1,0",
                "game,5,90,3,0,0",  # A frame with the disc in flight: nobody to ask about
            ],
        )
        self.frames = judgement_files.load_holder_frames(self.data / "source")
        self.answers = self.data / "answers" / "holders.csv"

    def test_the_frames_with_a_holder_are_asked_about_the_likeliest_first(self):
        self.assertEqual(list(self.frames), [("game", "5", "20")])
        self.assertEqual([row["player"] for row in self.frames["game", "5", "20"]], ["7", "4", "3"])

    def test_a_holder_makes_the_others_of_the_frame_non_holders(self):
        rows = self.frames["game", "5", "20"]
        entries = judgement_files.holder_answers(rows, "4", True)
        self.assertEqual(
            sorted((row["player"], holding, how) for row, holding, how in entries),
            [
                ("3", 0, judgement_files.BY_ANOTHER),
                ("4", 1, judgement_files.SAID),
                ("7", 0, judgement_files.BY_ANOTHER),
            ],
        )
        judgement_files.save_holders(self.answers, "source", entries)
        self.assertEqual(
            judgement_files.answered_players(self.answers),
            {("game", "5", "20", player) for player in "347"},
        )

    def test_a_frame_is_crowded_where_another_player_stands_in_the_holders_box(self):
        def row(holding, x1, x2):
            return {"holding": holding, "x1": x1, "y1": "0", "x2": x2, "y2": "100"}

        self.assertTrue(judgement_files.crowded([row("1", "10", "50"), row("0", "40", "80")]))
        self.assertFalse(judgement_files.crowded([row("1", "10", "50"), row("0", "60", "90")]))
        # The boxes are not in every dataset
        self.assertFalse(judgement_files.crowded(self.frames["game", "5", "20"]))

    def test_an_answer_can_be_taken_back_and_the_others_stay(self):
        rows = self.frames["game", "5", "20"]
        no = judgement_files.holder_answers(rows, "7", False)
        judgement_files.save_holders(self.answers, "source", no)
        yes = judgement_files.holder_answers(rows, "4", True)
        when = judgement_files.save_holders(self.answers, "source", yes)
        # The yes for 4 said no for 7 as well, within the same second as the no said for
        # 7 before: only what the yes said goes
        self.assertEqual(judgement_files.take_back_holders(self.answers, yes, when), 3)
        self.assertEqual(judgement_files.take_back_holders(self.answers, yes, when), 0)
        self.assertEqual(judgement_files.answered_players(self.answers), {("game", "5", "20", "7")})

    def test_the_last_answers_are_read_back_newest_first_with_what_they_wrote(self):
        rows = self.frames["game", "5", "20"]
        judgement_files.save_holders(
            self.answers, "source", judgement_files.holder_answers(rows, "7", False)
        )
        yes = judgement_files.holder_answers(rows, "4", True)
        when = judgement_files.save_holders(self.answers, "source", yes)
        last = judgement_files.last_holder_answers(self.answers, 10)
        self.assertEqual([(a["player"], a["holding"]) for a in last], [("4", True), ("7", False)])
        self.assertEqual(last[0]["time"], when)
        self.assertEqual(
            sorted(last[0]["lines"]),
            [
                ("3", 0, judgement_files.BY_ANOTHER),
                ("4", 1, judgement_files.SAID),
                ("7", 0, judgement_files.BY_ANOTHER),
            ],
        )
        self.assertEqual(len(judgement_files.last_holder_answers(self.answers, 1)), 1)

    def boxed(self):
        """A frame with boxes: 7 is wide and covers 4, and 3 stands apart."""
        write(
            self.data / "boxed" / "index.csv",
            "video,stretch,frame,player,holding,nearest,x1,y1,x2,y2",
            [
                "game,5,20,3,0,0,300,0,340,100",
                "game,5,20,4,0,1,60,0,100,100",
                "game,5,20,7,1,0,0,0,100,100",
            ],
        )
        return judgement_files.load_holder_frames(self.data / "boxed")

    def test_a_box_of_two_is_no_answer_and_the_holder_pointed_at_settles_the_rest(self):
        rows = self.boxed()["game", "5", "20"]
        wide = next(row for row in rows if row["player"] == "7")
        # Pointed at where 4's box is as well: nothing is said of 4, 3 does not hold it
        entries = judgement_files.two_in_box_answers(rows, "7", (80.0, 50.0))
        self.assertEqual([(row["player"], holding) for row, holding, _ in entries], [("3", 0)])
        self.assertEqual(judgement_files.two_in_box_answers(rows, "7", None), [])
        when = judgement_files.save_two_in_box(self.answers, "boxed", wide, (80.0, 50.0), entries)
        key = ("game", "5", "20", "7")
        self.assertEqual(judgement_files.boxes_of_two(self.answers), {key: (80.0, 50.0)})
        self.assertIn(key, judgement_files.answered_players(self.answers))
        self.assertNotIn(key, judgement_files.holders_said(self.answers))
        self.assertEqual(judgement_files.holders_said(self.answers)["game", "5", "20", "3"], 0)
        # Not among the answers of a yes or a no, and taken back by itself
        self.assertEqual(judgement_files.last_holder_answers(self.answers, 10), [])
        self.assertEqual(judgement_files.take_back_two_in_box(self.answers, wide, when), 1)
        self.assertEqual(judgement_files.boxes_of_two(self.answers), {})

    def test_a_box_of_two_without_a_place_says_nothing_else(self):
        rows = self.boxed()["game", "5", "20"]
        wide = next(row for row in rows if row["player"] == "7")
        judgement_files.save_two_in_box(self.answers, "boxed", wide, None, [])
        self.assertEqual(
            judgement_files.boxes_of_two(self.answers), {("game", "5", "20", "7"): None}
        )
        self.assertEqual(judgement_files.holders_said(self.answers), {})

    def test_answers_for_wide_boxes_are_asked_about_again_once(self):
        frames = self.boxed()
        rows = frames["game", "5", "20"]
        for player in ("7", "3"):
            judgement_files.save_holders(
                self.answers, "boxed", judgement_files.holder_answers(rows, player, False)
            )
        # Answers given since a box of two can be said are not asked about again
        self.assertEqual(judgement_files.wide_answers(self.answers, frames), [])
        again = judgement_files.wide_answers(self.answers, frames, before="9999")
        self.assertEqual([answer["player"] for answer in again], ["7"])
        self.assertEqual(again[0]["ratio"], 1.0)
        judgement_files.keep_a_copy(self.answers)
        self.assertTrue(self.answers.with_suffix(".csv.before_recheck").exists())
        judgement_files.save_rechecked(self.answers, rows[0])
        self.assertEqual(rows[0]["player"], "7")
        self.assertEqual(judgement_files.wide_answers(self.answers, frames, before="9999"), [])

    def test_a_frame_is_answered_at_once_and_a_holder_not_known_is_kept(self):
        rows = self.frames["game", "5", "20"]
        nobody = judgement_files.nobody_holds(rows)
        self.assertEqual({(h, how) for _, h, how in nobody}, {(0, judgement_files.SAID)})
        others = judgement_files.others_hold(rows)
        when = judgement_files.save_holders(self.answers, "source", others)
        self.assertEqual(set(judgement_files.holders_said(self.answers).values()), {0})
        key = ("game", "5", "20")
        judgement_files.save_clicked(
            self.answers, "source", key, (10, 20, 50, 120), (30.0, 60.0), False, when
        )
        kept = self.answers.with_name("clicked_holders.csv").read_text().splitlines()
        self.assertEqual(
            kept[1].split(",")[:11],
            "source,game,5,20,10.0,20.0,50.0,120.0,30.0,60.0,0".split(","),
        )
        self.assertEqual(judgement_files.take_back_clicked(self.answers, key, when), 1)
        self.assertEqual(judgement_files.take_back_clicked(self.answers, key, when), 0)

    def test_frames_without_a_holder_are_asked_about_only_where_told_to(self):
        with_one = judgement_files.load_holder_frames(self.data / "source")
        every = judgement_files.load_holder_frames(self.data / "source", without=True)
        self.assertEqual(set(every) - set(with_one), {("game", "5", "90")})

    def test_only_the_frames_found_to_show_a_flying_disc_are_flying(self):
        write(
            self.data / "flying" / "source.csv",
            ",".join(judgement_files.FLYING_COLUMNS),
            ["game,5,90,flying,10,20", "game,5,91,lying still,10,20"],
        )
        found = judgement_files.flying_frames("source", self.data / "flying")
        self.assertEqual(found, {("game", "5", "90"): (10.0, 20.0)})
        self.assertEqual(judgement_files.flying_frames("another", self.data / "flying"), {})

    def test_nobody_holds_it_is_known_of_a_frame_answered_at_once(self):
        rows = self.frames["game", "5", "20"]
        judgement_files.save_holders(self.answers, "source", judgement_files.nobody_holds(rows))
        self.assertEqual(judgement_files.said_of_the_frame(self.answers), {("game", "5", "20")})
        # A holder named later: no longer a frame without one
        judgement_files.save_holders(
            self.answers, "source", judgement_files.holder_answers(rows, "7", True)
        )
        self.assertEqual(judgement_files.said_of_the_frame(self.answers), set())

    def test_a_no_says_nothing_of_the_others(self):
        entries = judgement_files.holder_answers(self.frames["game", "5", "20"], "7", False)
        self.assertEqual([(row["player"], holding) for row, holding, _ in entries], [("7", 0)])


if __name__ == "__main__":
    unittest.main()
