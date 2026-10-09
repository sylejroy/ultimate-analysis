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

    def test_a_tint_from_the_light_is_not_made_a_colour_of(self):
        # As measured in games: white shirts in sunlight and on grass, and near-black ones
        for white in ((120, 139, 134), (189, 205, 202), (162, 150, 142), (158, 136, 133)):
            self.assertEqual(tracks.team_display_colour(white), (255, 255, 255), white)
        for black in ((69, 84, 84), (43, 22, 34), (52, 44, 73), (50, 69, 84)):
            self.assertEqual(tracks.team_display_colour(black), (40, 40, 40), black)
        # ... and shirts that are coloured: orange, red, dark red, blue
        for coloured in ((88, 132, 187), (86, 92, 186), (82, 40, 44), (176, 138, 123)):
            self.assertNotIn(tracks.team_display_colour(coloured), ((255, 255, 255), (40, 40, 40)))

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
