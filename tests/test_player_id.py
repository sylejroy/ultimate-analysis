"""Jersey number reading: configuration and reader selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from support import load_module


class PlayerIdConfigTests(unittest.TestCase):
    def test_live_selection_reads_a_previous_sharp_crop_and_backs_off(self):
        module = load_module("processing.player_id")
        selector = module.JerseyCropSelector()
        sharp = np.zeros((80, 40, 3), dtype=np.uint8)
        sharp[:, ::4] = 255
        blank = np.zeros_like(sharp)
        track = SimpleNamespace(
            track_id=7, class_name="player", bbox=[0, 0, 40, 80], time_since_update=0
        )
        settings = {
            "models.player_id.crop_selection.enabled": True,
            "models.player_id.ocr_frame_interval": 3,
            "models.player_id.ocr_frame_interval_stagger": False,
        }
        with (
            patch.object(
                module,
                "get_setting",
                side_effect=lambda key, default=None: settings.get(key, default),
            ),
            patch.object(module, "_load_easyocr_config", return_value={}),
            patch.object(module, "_easyocr_reader", Mock()),
            patch.object(
                module, "_read_jersey_numbers", return_value=([("Unknown", None)], {})
            ) as read,
            patch.object(module, "get_jersey_probabilities", return_value=[]),
            patch.object(module, "get_best_jersey_number", return_value=(None, 0)),
        ):
            for frame_index, image in ((1, sharp), (2, blank), (3, blank)):
                module.run_player_id_on_tracks(image, [track], frame_index, crop_selector=selector)
            read.assert_called_once()
            np.testing.assert_array_equal(read.call_args.args[0][0], sharp)
            for frame_index in range(4, 7):
                module.run_player_id_on_tracks(sharp, [track], frame_index, crop_selector=selector)
            read.assert_called_once()  # An unreadable batch defers the next due frame.

    def test_easyocr_config_is_reparsed_only_when_the_file_changes(self):
        module = load_module("processing.player_id")
        module._easyocr_config_mtime_ns = None
        with patch.object(module.yaml, "safe_load", wraps=module.yaml.safe_load) as parse:
            first = module._load_easyocr_config()
            self.assertIs(module._load_easyocr_config(), first)
            self.assertEqual(parse.call_count, 1)
            module._easyocr_config_mtime_ns -= 1  # Simulate a save from the tuning tab
            module._load_easyocr_config()
            self.assertEqual(parse.call_count, 2)

    def test_selected_reader_replaces_easyocr_recognition(self):
        module = load_module("processing.player_id")
        crop = np.full((80, 40, 3), 100, dtype=np.uint8)
        reader = Mock()
        # A confident number, an unconfident one, and text that is not a number
        reader.read.return_value = [[(None, "17", 0.9), (None, "23", 0.3), (None, "abc", 0.99)]]
        easyocr_reader = Mock()
        with (
            patch.multiple(module, EASYOCR_AVAILABLE=True, _easyocr_reader=easyocr_reader),
            patch.object(module, "_get_active_reader", return_value=reader),
            patch.object(module, "_load_easyocr_config", return_value={}),
        ):
            results, _ = module._read_jersey_numbers([crop])

        self.assertEqual(results[0][0], "17")
        easyocr_reader.readtext.assert_not_called()
        # The reader gets the upper-body crop without EasyOCR's contrast and scaling steps
        (crops,), _ = reader.read.call_args
        self.assertEqual(crops[0].shape, (26, 40, 3))

    def test_unknown_or_unavailable_reader_falls_back_to_easyocr(self):
        module = load_module("processing.player_id")
        with self.assertRaises(ValueError):
            module.set_player_id_method("tesseract")

        readers = load_module("processing.jersey_readers")
        readers._readers.clear()
        with patch.object(readers, "find_digit_model", return_value=None):
            self.assertIsNone(readers.get_reader("yolo_digits", lambda: None))
        self.assertIsNone(readers.get_reader("parseq", lambda: None))


if __name__ == "__main__":
    unittest.main()
