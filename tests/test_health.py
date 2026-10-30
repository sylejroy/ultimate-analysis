"""What went wrong in a stage: shown on the frame it is about, also when it was
reported beside the frames or in another process."""

import unittest

from support import load_module


class HealthTest(unittest.TestCase):
    def setUp(self):
        self.health = load_module("processing.health")
        self.health.start_frame()
        self.addCleanup(self.health.start_frame)

    def test_a_frames_problems_are_forgotten_with_it_and_standing_ones_are_not(self):
        health = self.health
        health.report("Field", "failed")
        health.report("Engine", "not usable", standing=True)
        self.assertEqual(health.problems(), ["Engine: not usable", "Field: failed"])
        self.assertTrue(health.failed("Field"))
        health.start_frame()
        self.assertEqual(health.problems(), ["Engine: not usable"])
        health.clear("Engine")
        self.assertEqual(health.problems(), [])

    def test_what_is_captured_can_be_repeated_on_a_later_frame(self):
        health = self.health
        with health.captured() as caught:
            health.report("Detection", "failed")
            health.report("Engine", "not usable", standing=True)
        self.assertEqual(caught, {"Detection": ("failed", False), "Engine": ("not usable", True)})
        health.start_frame()
        health.clear("Engine")
        self.assertEqual(health.problems(), [])
        health.repeat(caught)
        self.assertEqual(health.problems(), ["Engine: not usable", "Detection: failed"])
        health.clear("Engine")

    def test_capturing_some_stages_leaves_the_others_to_the_frame_they_fell_in(self):
        health = self.health
        with health.captured(("Detection",)) as caught:
            health.report("Detection", "failed")
            health.report("Field", "failed too")
        self.assertEqual(caught, {"Detection": ("failed", False)})
        # Reported all the same, on the frame being analysed
        self.assertTrue(health.failed("Field"))
        # Nothing is captured once the capture is over
        health.report("Detection", "failed again")
        self.assertEqual(caught, {"Detection": ("failed", False)})


if __name__ == "__main__":
    unittest.main()
