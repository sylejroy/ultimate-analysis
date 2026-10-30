"""The detectors in a process of their own: frames go over, what is found comes back."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from support import load_module

# In the other process the module is loaded as it is here: without the models
ECHO = "_runtime_checks.processing.detector_process:_echo"


def serve(connection, detect):
    """What the other process runs in these tests."""
    load_module("processing.detector_process")._serve(connection, detect)


class DetectorProcessTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.detector_process")

    def asked(self, value):
        return {"weak_discs_from": value}

    def test_frames_go_over_and_answers_come_back(self):
        detectors = self.module.DetectorProcess(ECHO, serve)
        self.addCleanup(detectors.close)
        frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        (first,), reported = detectors.detect(frame, self.asked(0.1))
        self.assertEqual(first["shape"], (4, 6, 3))
        self.assertEqual(first["sum"], int(frame.sum()))
        self.assertEqual(first["sent"], 0.1)
        self.assertEqual(reported, {})
        # A larger frame than the memory shared so far, and what the other side keeps
        larger = np.full((8, 8, 3), 2, dtype=np.uint8)
        (second,), _ = detectors.detect(larger, self.asked(0.0))
        self.assertEqual((second["shape"], second["sum"], second["frames"]), ((8, 8, 3), 384, 2))
        # Not all of a frame: rows left out lie apart in memory
        (third,), _ = detectors.detect(larger[::2], self.asked(0.0))
        self.assertEqual((third["shape"], third["sum"]), ((4, 8, 3), 192))

    def test_what_the_detectors_report_over_there_is_reported_here(self):
        """A failure the detectors catch themselves is reported in their own process,
        where nobody sees it; it comes back with the detections and is said here."""
        module, health = self.module, load_module("processing.health")
        detectors = module.DetectorProcess(ECHO, serve)
        self.addCleanup(detectors.close)
        for name, value in (
            ("is_enabled", Mock(return_value=True)),
            ("_the_process", Mock(return_value=detectors)),
            ("_detectors", detectors),
            ("_loaded_here", False),
        ):
            patcher = patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        load_module("processing.inference")  # Asked which models and resets were sent
        health.start_frame()
        self.addCleanup(health.start_frame)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertEqual(len(module.detect(frame, weak_discs_from=-1.0)), 1)
        self.assertIn("Detection: told to fail", health.problems())
        health.start_frame()
        module.detect(frame, weak_discs_from=0.0)
        self.assertFalse(health.failed("Detection"))

    def test_a_process_that_cannot_work_is_given_up(self):
        detectors = self.module.DetectorProcess("no_such_module:thing", serve)
        self.addCleanup(detectors.close)
        self.assertIsNone(detectors.detect(np.zeros((2, 2, 3), dtype=np.uint8), self.asked(0.0)))
        self.assertTrue(detectors.failed)
        # ... and not tried again
        self.assertIsNone(detectors.detect(np.zeros((2, 2, 3), dtype=np.uint8), self.asked(0.0)))


class FallingBackTest(unittest.TestCase):
    """When the other process stops, this one detects only if it loaded the engines."""

    def setUp(self):
        self.module = load_module("processing.detector_process")
        self.health = load_module("processing.health")
        self.inference = load_module("processing.inference")
        self.health.start_frame()
        stopped = SimpleNamespace(detect=Mock(return_value=None), failed=True)
        for name, value in (
            ("is_enabled", Mock(return_value=True)),
            ("_the_process", Mock(return_value=stopped)),
            ("_detectors", stopped),
            ("_loaded_here", False),
        ):
            patcher = patch.object(self.module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(self.inference, "run_inference", Mock(return_value=["here"]))
        self.run_here = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.health.start_frame)

    def test_nothing_is_loaded_late_and_it_is_said(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertEqual(self.module.detect(frame), [])
        self.run_here.assert_not_called()
        self.assertTrue(self.health.failed("Detection"))
        self.assertFalse(self.module.in_own_process())

    def test_after_the_video_is_opened_again_this_process_detects(self):
        self.module.ready_here()
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertEqual(self.module.detect(frame), ["here"])


if __name__ == "__main__":
    unittest.main()
