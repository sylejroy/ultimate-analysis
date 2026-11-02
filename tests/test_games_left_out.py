"""A disc dataset with whole games left out: which game an image is of and where it goes."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_games_left_out_dataset as dataset  # noqa: E402

GAMES = ("chicago_vs_new_york_2024", "portland_vs_san_francisco_2024")


class GamesLeftOutTest(unittest.TestCase):
    def test_a_game_is_told_by_the_file_name(self):
        name = "labelled_chicago_vs_new_york_2024_frame_002546"
        self.assertEqual(dataset.game_of(name, GAMES), GAMES[0])

    def test_the_roboflow_export_belongs_to_its_game(self):
        name = "s1_frame_100464_jpg.rf.313389736605bd90f787343bf485ce0e"
        self.assertEqual(dataset.game_of(name, GAMES), GAMES[1])

    def test_an_image_of_no_known_game_is_said_to_be(self):
        self.assertIsNone(dataset.game_of("something_else_frame_1", GAMES))

    def test_a_game_goes_wholly_to_one_split(self):
        valid, test = ("Ring_of_Fire",), ("Pacmen",)
        self.assertEqual(dataset.split_of("Ring_of_Fire_VS_Sub_Zero", valid, test), "valid")
        self.assertEqual(dataset.split_of("Pacmen_VS_Chain_Lightning", valid, test), "test")
        self.assertEqual(dataset.split_of(GAMES[0], valid, test), "train")


if __name__ == "__main__":
    unittest.main()
