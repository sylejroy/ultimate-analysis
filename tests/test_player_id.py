"""Jersey number reading: configuration and reader selection."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from support import load_module


class PlayerIdConfigTests(unittest.TestCase):
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
