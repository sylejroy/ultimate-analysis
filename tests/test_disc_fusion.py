"""The disc from the disc detector and the heat map together."""

import unittest
from unittest.mock import patch

import numpy as np
from support import load_module


def disc(x, y, confidence, weak=False):
    found = {"bbox": [x - 5, y - 4, x + 5, y + 4], "confidence": confidence, "class_name": "disc"}
    return {**found, "weak": True} if weak else found


class DiscFusionTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.disc_fusion")
        self.settings = {
            "models.disc_fusion.enabled": True,
            "models.disc_fusion.agree_heat": 0.1,
            "models.disc_fusion.heat_alone": 0.5,
        }
        patcher = patch.object(
            self.module, "get_setting", side_effect=lambda key, d=None: self.settings.get(key, d)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def fused(self, discs=(), weak=(), peaks=()):
        return self.module.fuse_discs(list(discs), list(weak), list(peaks), 80.0)

    def test_a_sure_disc_counts_and_for_more_where_the_heat_map_agrees(self):
        (alone,) = self.fused([disc(100, 100, 0.6)])
        self.assertEqual(alone["confidence"], 0.6)
        (both,) = self.fused([disc(100, 100, 0.6)], peaks=[(102.0, 101.0, 0.5)])
        self.assertAlmostEqual(both["confidence"], 0.8)
        # Of two, the one both see is the surer, though the detector liked the other more
        there, here = self.fused(
            [disc(400, 300, 0.6), disc(100, 100, 0.5)], peaks=[(100.0, 100.0, 0.4)]
        )
        self.assertGreater(here["confidence"], there["confidence"])

    def test_an_unsure_disc_counts_only_where_the_heat_map_has_it_too(self):
        self.assertEqual(self.fused(weak=[disc(100, 100, 0.15, weak=True)]), [])
        self.assertEqual(
            self.fused(weak=[disc(100, 100, 0.15, weak=True)], peaks=[(300.0, 300.0, 0.4)]), []
        )
        (taken,) = self.fused(weak=[disc(100, 100, 0.15, weak=True)], peaks=[(101.0, 99.0, 0.2)])
        self.assertNotIn("weak", taken)
        self.assertAlmostEqual(taken["confidence"], 1 - 0.85 * 0.8)

    def test_a_spot_the_heat_map_is_very_sure_of_is_a_disc_by_itself(self):
        self.assertEqual(self.fused(peaks=[(200.0, 150.0, 0.3)]), [])
        (found,) = self.fused(peaks=[(200.0, 150.0, 0.7)])
        self.assertEqual(found["class_name"], "disc")
        self.assertEqual(found["source"], "heat map")
        x1, y1, x2, y2 = found["bbox"]
        self.assertAlmostEqual((x1 + x2) / 2, 200.0, delta=1.0)
        self.assertAlmostEqual((y1 + y2) / 2, 150.0, delta=1.0)
        # Not a second time where the detector has the disc already
        self.assertEqual(len(self.fused([disc(200, 150, 0.6)], peaks=[(200.0, 150.0, 0.7)])), 1)
        self.settings["models.disc_fusion.heat_alone"] = 0.0
        self.assertEqual(self.fused(peaks=[(200.0, 150.0, 0.9)]), [])

    def test_no_unsure_discs_are_asked_for_while_it_is_off(self):
        self.assertEqual(self.module.weak_discs_from(), 0.1)
        self.settings["models.disc_fusion.enabled"] = False
        self.assertEqual(self.module.weak_discs_from(), 0.0)


class HeatPeaksTest(unittest.TestCase):
    def test_the_surest_spots_of_a_heat_map_in_the_pixels_of_the_frame(self):
        peaks_of = load_module("processing.disc_heatmap").peaks_of
        heat = np.zeros((270, 480), dtype=np.float32)
        heat[100, 200], heat[101, 201] = 0.8, 0.5  # One spot
        heat[50, 400] = 0.3
        heat[10, 10] = 0.01  # Too little
        peaks = peaks_of(heat, (1920, 1080))
        self.assertEqual([round(sure, 2) for _, _, sure in peaks], [0.8, 0.3])
        self.assertAlmostEqual(peaks[0][0], 802.0)
        self.assertAlmostEqual(peaks[0][1], 402.0)
        self.assertEqual(peaks_of(None, (1920, 1080)), [])


if __name__ == "__main__":
    unittest.main()
