"""Telling players apart by their looks: the vectors and how matches are counted."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import torch

    from ultimate_analysis.processing import reid
except ImportError:  # PyTorch is not installed
    reid = None


def crops_of(video, stretch, player, light, number, vector, count=10, start=0):
    """Rows of a made-up track and the vector of each of its crops."""
    rows = [
        {
            "video": video,
            "stretch": stretch,
            "frame": start + index,
            "player": player,
            "light_shirt": light,
            "number": number,
        }
        for index in range(count)
    ]
    return rows, [np.asarray(vector, dtype=np.float64)] * count


@unittest.skipIf(reid is None, "needs PyTorch")
class EmbedderTest(unittest.TestCase):
    def test_a_crop_of_any_size_gives_a_vector_of_length_one(self):
        model = reid.PlayerEmbedder(pretrained=False).eval()
        crops = [
            np.random.default_rng(0).integers(0, 255, (90, 40, 3), dtype=np.uint8),
            np.random.default_rng(1).integers(0, 255, (200, 110, 3), dtype=np.uint8),
        ]
        vectors = reid.embed(model, crops)
        self.assertEqual(vectors.shape, (2, reid.VECTOR_LENGTH))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
        self.assertEqual(tuple(reid.prepare(crops).shape), (2, 3, 128, 64))
        self.assertIsInstance(reid.prepare(crops), torch.Tensor)

    def test_a_crop_keeps_its_proportions_when_asked_to(self):
        wide = np.full((100, 80, 3), 200, dtype=np.uint8)
        stretched = reid.fit(wide, (64, 128))
        self.assertTrue((stretched == 200).all())  # Fills the whole picture
        fitted = reid.fit(wide, (64, 128), keep_proportions=True)
        self.assertEqual(fitted.shape, (128, 64, 3))
        # 80 x 100 fits as 64 x 80, in the middle, with grey above and below
        self.assertTrue((fitted[24:104] == 200).all())
        self.assertFalse((fitted[:20] == 200).any())

    def test_a_smeared_crop_is_less_sharp(self):
        import cv2

        crisp = np.random.default_rng(0).integers(0, 255, (128, 64, 3), dtype=np.uint8)
        smeared = cv2.blur(crisp, (9, 1))
        self.assertGreater(reid.sharpness(crisp), 3 * reid.sharpness(smeared))

    def test_a_tracks_vector_is_the_mean_of_its_crops_at_length_one(self):
        vector = reid.track_vector(np.array([[1.0, 0.0], [0.0, 1.0]]))
        np.testing.assert_allclose(vector, [0.5**0.5, 0.5**0.5])


class MatchesTest(unittest.TestCase):
    def setUp(self):
        from benchmark_reid import matches

        self.matches = matches

    def test_tracks_are_matched_within_a_point_and_by_number_across_points(self):
        rows, vectors = [], []
        tracks = [
            # Stretch 0: two teammates in light shirts and one in a dark one
            ("game", 0, "0-1", "1", "7", (1.0, 0.0, 0.0)),
            ("game", 0, "0-2", "1", "", (0.0, 1.0, 0.0)),
            ("game", 0, "0-3", "0", "9", (0.0, 0.0, 1.0)),
            # Stretch 1: the same three under other track IDs
            ("game", 1, "0-5", "1", "7", (0.9, 0.1, 0.0)),
            ("game", 1, "0-6", "1", "", (0.1, 0.9, 0.0)),
            ("game", 1, "0-7", "0", "9", (0.0, 0.1, 0.9)),
        ]
        for video, stretch, player, light, number, vector in tracks:
            more_rows, more_vectors = crops_of(video, stretch, player, light, number, vector)
            rows += more_rows
            vectors += more_vectors
        result = self.matches(np.array(vectors), rows)
        self.assertEqual(result["same_point_all"], 1.0)
        self.assertEqual(result["same_point_all_count"], 6)
        # Number 7 in both stretches and both directions, among the two light shirts
        self.assertEqual(result["across_points_teammates"], 1.0)
        self.assertEqual(result["across_points_teammates_count"], 2)
        self.assertAlmostEqual(result["across_points_teammates_chance"], 0.5)
        # Number 9 too, with all three to choose from
        self.assertEqual(result["across_points_all_count"], 4)

    def test_vectors_that_say_nothing_match_as_often_as_chance(self):
        rows, vectors = [], []
        for player in range(4):
            more_rows, more_vectors = crops_of("game", 0, f"0-{player}", "1", "", (1.0, 0.0))
            rows += more_rows
            vectors += more_vectors
        result = self.matches(np.array(vectors), rows)
        # All the same: the first of the four is taken every time
        self.assertEqual(result["same_point_all"], 0.25)

    def test_a_short_track_is_left_out(self):
        rows, vectors = crops_of("game", 0, "0-1", "1", "", (1.0, 0.0), count=4)
        more_rows, more_vectors = crops_of("game", 0, "0-2", "1", "", (0.0, 1.0))
        another_rows, another_vectors = crops_of("game", 0, "0-3", "1", "", (0.7, 0.7))
        result = self.matches(
            np.array(vectors + more_vectors + another_vectors), rows + more_rows + another_rows
        )
        self.assertEqual(result["same_point_all_count"], 2)


if __name__ == "__main__":
    unittest.main()
