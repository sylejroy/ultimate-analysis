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
        disc = {"class_name": "disc", "bbox": [2, 2, 4, 4], "confidence": 0.9}
        disc_calls = 0

        def predict(frame, model, size, prefix, target):
            nonlocal disc_calls
            if target == "disc":
                disc_calls += 1
                return ([disc] if disc_calls >= 32 else []), {}
            return [], {}

        with (
            patch.multiple(module, YOLO_AVAILABLE=True, _player_model=Mock(), _disc_model=Mock()),
            patch.object(module, "_run_single_model_inference", side_effect=predict),
            patch.object(module, "get_setting", side_effect=lambda key, default=None: default),
        ):
            for _ in range(36):
                detections = module.run_inference(frame)
        self.assertEqual(detections, [disc])
        self.assertEqual(disc_calls, 32)
        self.assertEqual(module._frames_since_last_disc, 0)

    def test_short_disc_appearance_is_recovered_by_five_frame_retries(self):
        module = self.module
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        disc = {"class_name": "disc", "bbox": [2, 2, 4, 4], "confidence": 0.9}
        for interval, expected in ((30, False), (5, True)):
            with self.subTest(interval=interval):
                module.reset_inference_state()
                frame_index = 0

                def predict(image, model, size, prefix, target):
                    return ([disc] if target == "disc" and 34 <= frame_index <= 39 else []), {}

                with (
                    patch.multiple(
                        module, YOLO_AVAILABLE=True, _player_model=Mock(), _disc_model=Mock()
                    ),
                    patch.object(module, "_run_single_model_inference", side_effect=predict),
                    patch.object(
                        module,
                        "get_setting",
                        side_effect=lambda key, default=None: interval
                        if key.endswith("retry_interval")
                        else default,
                    ),
                ):
                    found = False
                    for frame_index in range(61):
                        found |= bool(module.run_inference(frame))
                self.assertEqual(found, expected)

    def test_a_followed_disc_is_searched_in_a_window_around_where_it_is_heading(self):
        module = self.module
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # The window is off by default; this is about how it works when switched on
        settings = {"models.disc_detection.follow_window": 640}
        seen = []  # (shape searched, image size) per call
        found = True

        def predict(image, model, size, prefix, target):
            seen.append((image.shape[:2], size))
            if not found:
                return [], {}
            if image.shape[:2] == (1080, 1920):
                # Whole frame: the disc moves 20 px to the right per frame
                x = 1000 + 20 * (len(seen) - 1)
                return [
                    {"class_name": "disc", "bbox": [x - 8, 492, x + 8, 508], "confidence": 0.8}
                ], {}
            # Window: found in its centre, reported in window coordinates
            return [{"class_name": "disc", "bbox": [312, 312, 328, 328], "confidence": 0.8}], {}

        with (
            patch.multiple(module, _disc_model=Mock(), _disc_model_imgsz=1280),
            patch.object(module, "_run_single_model_inference", side_effect=predict),
            patch.object(
                module,
                "get_setting",
                side_effect=lambda key, default=None: settings.get(key, default),
            ),
        ):
            first, _ = module._detect_disc(frame)
            second, _ = module._detect_disc(frame)
            # Never seen before: the whole frame. Then a 640 px window, shown to the model
            # at the scale it sees whole frames at (1280 / 1920)
            self.assertEqual(seen, [((1080, 1920), 1280), ((640, 640), 416)])
            # No speed is known yet, so the window is centred on the last position, and
            # the detection is reported in frame coordinates
            self.assertEqual(second[0]["bbox"], [992, 492, 1008, 508])

            # Whole frame again at the regular interval, even while the disc is followed
            for _ in range(20):
                module._detect_disc(frame)
            self.assertEqual(sum(shape == (1080, 1920) for shape, _ in seen), 2)

            # Lost for a while: back to the whole frame
            found = False
            del seen[:]
            for _ in range(10):
                module._detect_disc(frame)
            self.assertEqual([shape for shape, _ in seen[:6]], [(640, 640)] * 6)
            self.assertEqual({shape for shape, _ in seen[6:]}, {(1080, 1920)})

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

    def test_packed_boxes_transfer_once_and_keep_optional_track_ids_out_of_classes(self):
        for packed in ([[1, 2, 8, 9, 0.9, 1]], [[1, 2, 8, 9, 42, 0.9, 1]]):
            with self.subTest(columns=len(packed[0])):
                data = Mock()
                data.cpu.return_value.numpy.return_value = np.array(packed, dtype=np.float32)
                model = Mock(names={0: "disc", 1: "player"})
                model.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(data=data))]
                with patch.object(
                    self.module, "get_setting", side_effect=lambda key, default=None: default
                ):
                    found, _ = self.module._run_single_model_inference(
                        np.zeros((8, 8, 3), dtype=np.uint8),
                        model,
                        640,
                        "models.player_detection",
                        "player",
                    )
                data.cpu.assert_called_once()
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]["bbox"], [1, 2, 8, 9])
                self.assertEqual(found[0]["class_id"], 1)
                self.assertAlmostEqual(found[0]["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
