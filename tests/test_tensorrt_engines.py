"""TensorRT engine lookup and its fallback to PyTorch."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from support import load_module


class TensorRTEngineTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.tensorrt_engines")
        self.module._engines.clear()

    def test_network_input_shape_matches_ultralytics_letterboxing(self):
        shape = self.module.network_input_shape
        self.assertEqual(shape((1080, 1920), 640), (384, 640))
        self.assertEqual(shape((1080, 1920, 3), 1280), (736, 1280))
        self.assertEqual(shape((640, 640), 640), (640, 640))

    def test_models_without_a_matching_engine_keep_running_pytorch(self):
        module = self.module
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "best.pt"
            weights.write_bytes(b"weights")
            model = SimpleNamespace(ckpt_path=str(weights), task="detect")
            frame_shape = (1080, 1920, 3)

            # No engine has been built
            self.assertIsNone(module.get_engine(model, frame_shape, 640, half=True))

            # An engine older than its weights belongs to different weights
            module._engines.clear()
            engine = module.engine_path(weights, (384, 640), True)
            self.assertEqual(engine.name, "best.384x640.fp16.engine")
            engine.write_bytes(b"engine")
            os.utime(engine, (1, 1))
            self.assertIsNone(module.get_engine(model, frame_shape, 640, half=True))

            # Switched off in the settings
            module._engines.clear()
            with patch.object(module, "get_setting", return_value=False):
                self.assertIsNone(module.get_engine(model, frame_shape, 640, half=True))
            self.assertEqual(module._engines, {})

        # A model that was not loaded from a weights file
        self.assertIsNone(module.get_engine(Mock(), frame_shape, 640, half=True))

    def test_rtdetr_models_never_use_an_engine(self):
        module = self.module
        RTDETR = type("RTDETR", (SimpleNamespace,), {})
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "best.pt"
            weights.write_bytes(b"weights")
            module.engine_path(weights, (384, 640), True).write_bytes(b"engine")
            model = RTDETR(ckpt_path=str(weights), task="detect")

            with patch.object(module, "_load_engine") as load_engine:
                self.assertIsNone(module.get_engine(model, (1080, 1920, 3), 640, half=True))
            load_engine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
