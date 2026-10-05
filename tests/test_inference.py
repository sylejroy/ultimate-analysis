"""Object detection: model roles, class handling, and disc skipping."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from support import load_module


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("processing.inference")
        self.module.reset_inference_state()

    def test_disc_detection_resumes_after_an_empty_scene(self):
        module = self.module
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        disc_calls = 0

        def predict(frame, model, size, prefix, target):
            nonlocal disc_calls
            if target == "disc":
                disc_calls += 1
                return ([{"class_name": "disc"}] if disc_calls >= 32 else []), {}
            return [], {}

        with (
            patch.multiple(module, YOLO_AVAILABLE=True, _player_model=Mock(), _disc_model=Mock()),
            patch.object(module, "_run_single_model_inference", side_effect=predict),
            patch.object(module, "get_setting", side_effect=lambda key, default=None: default),
        ):
            for _ in range(61):
                detections = module.run_inference(frame)
        self.assertEqual(detections, [{"class_name": "disc"}])
        self.assertEqual(disc_calls, 32)
        self.assertEqual(module._frames_since_last_disc, 0)

    def test_unavailable_yolo_preserves_timing_return_contract(self):
        with patch.object(self.module, "YOLO_AVAILABLE", False):
            detections, timing = self.module.run_inference(np.zeros((2, 2, 3)), return_timing=True)
        self.assertEqual(detections, [])
        self.assertEqual(timing["total_time"], 0.0)

    def test_reselecting_a_loaded_model_does_not_reload_weights(self):
        for kind in ("player", "disc"):
            with (
                self.subTest(kind=kind),
                patch.multiple(
                    self.module, **{f"_{kind}_model": Mock(), f"_{kind}_model_path": "model.pt"}
                ),
                patch.object(self.module, "YOLO") as constructor,
            ):
                self.assertTrue(getattr(self.module, f"set_{kind}_model")("model.pt"))
                constructor.assert_not_called()

    @staticmethod
    def _two_class_model(boxes, confidences, classes):
        def tensor(values):
            array = np.array(values, dtype=np.float32)
            return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: array))

        model = Mock()
        model.names = {0: "disc", 1: "player"}
        model.predict.return_value = [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=tensor(boxes), conf=tensor(confidences), cls=tensor(classes)
                )
            )
        ]
        return model

    def test_one_model_in_both_roles_runs_once_and_splits_classes(self):
        module = self.module
        model = self._two_class_model(
            [[0, 0, 4, 4], [1, 1, 5, 5], [2, 2, 3, 3], [6, 6, 7, 7]],
            [0.9, 0.4, 0.35, 0.2],
            [1, 1, 0, 0],
        )
        settings = {"models.disc_detection.confidence_threshold": 0.3}
        with (
            patch.multiple(module, YOLO_AVAILABLE=True, _player_model=model, _disc_model=model),
            patch.object(
                module,
                "get_setting",
                side_effect=lambda key, default=None: settings.get(key, default),
            ),
        ):
            detections, timing = module.run_inference(
                np.zeros((8, 8, 3), dtype=np.uint8), return_timing=True
            )

        model.predict.assert_called_once()
        self.assertEqual(model.predict.call_args.kwargs["conf"], 0.3)
        self.assertEqual(model.predict.call_args.kwargs["classes"], [0, 1])
        # Each class keeps its own confidence threshold (player 0.5, disc 0.3)
        self.assertEqual(
            [(d["class_name"], d["model_type"]) for d in detections],
            [("player", "player_model"), ("disc", "disc_model")],
        )
        self.assertEqual((timing["player_count"], timing["disc_count"]), (1, 1))

    def test_two_class_model_used_for_players_does_not_report_discs_as_players(self):
        module = self.module
        model = self._two_class_model([[0, 0, 4, 4]], [0.9], [1])
        with patch.object(module, "get_setting", side_effect=lambda key, default=None: default):
            detections, _ = module._run_single_model_inference(
                np.zeros((8, 8, 3), dtype=np.uint8),
                model,
                640,
                "models.player_detection",
                "player",
            )
        self.assertEqual(model.predict.call_args.kwargs["classes"], [1])
        self.assertEqual([d["class_name"] for d in detections], ["player"])


if __name__ == "__main__":
    unittest.main()
