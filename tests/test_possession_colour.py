"""The colour that marks who has the disc: the team's shirt colour, made vivid."""

import colorsys
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from ultimate_analysis.rendering import tracks
except ImportError:  # Ultralytics is not installed
    tracks = None


def hue_saturation_value(colour):
    blue, green, red = (part / 255 for part in colour)
    return colorsys.rgb_to_hsv(red, green, blue)


@unittest.skipIf(tracks is None, "needs Ultralytics")
class PossessionColourTests(unittest.TestCase):
    def test_a_dull_shirt_keeps_its_hue_and_becomes_vivid(self):
        navy = (90, 50, 40)
        vivid = tracks.team_display_colour(navy)
        before, after = hue_saturation_value(navy), hue_saturation_value(vivid)
        self.assertAlmostEqual(before[0], after[0], delta=0.03)
        self.assertGreater(after[1], 0.75)
        self.assertGreater(after[2], 0.85)

    def test_white_and_black_shirts_stay_apart(self):
        self.assertEqual(tracks.team_display_colour((215, 220, 222)), (255, 255, 255))
        self.assertEqual(tracks.team_display_colour((35, 32, 30)), (40, 40, 40))

    def test_a_track_without_a_team_is_marked_in_grey(self):
        self.assertEqual(
            tracks.possession_colour(SimpleNamespace(team_colour=None)),
            tracks.UNKNOWN_TEAM_COLOUR,
        )
        self.assertEqual(
            tracks.possession_colour(SimpleNamespace(team_colour=(60, 60, 140))),
            tracks.team_display_colour((60, 60, 140)),
        )


if __name__ == "__main__":
    unittest.main()
