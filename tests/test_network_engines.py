"""Engines of the networks that are not YOLO models: where they are kept, and the
network running as it is wherever there is none."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support import load_module


class NetworkEnginesTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.network_engines")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.weights = Path(self.folder.name) / "best.pt"

    def test_an_engine_is_kept_beside_its_weights_by_input_shape(self):
        path = self.module.engine_path(self.weights, (1, 9, 544, 960))
        self.assertEqual(path, Path(self.folder.name) / "best.1x9x544x960.fp16.engine")

    def test_without_an_engine_or_with_engines_off_the_network_runs_as_it_is(self):
        self.assertIsNone(self.module.load_engine(self.weights, (1, 9, 544, 960)))
        self.module.engine_path(self.weights, (1, 9, 544, 960)).write_bytes(b"not an engine")
        with patch.object(self.module, "get_setting", return_value=False):
            self.assertIsNone(self.module.load_engine(self.weights, (1, 9, 544, 960)))

    def test_an_engine_that_does_not_load_leaves_the_network_as_it_is(self):
        self.module.engine_path(self.weights, (1, 9, 544, 960)).write_bytes(b"not an engine")

        def broken(path):
            raise RuntimeError("built for another GPU")

        with patch.object(self.module, "NetworkEngine", side_effect=broken):
            self.assertIsNone(self.module.load_engine(self.weights, (1, 9, 544, 960)))


if __name__ == "__main__":
    unittest.main()
