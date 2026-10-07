"""Jersey number reading: configuration and reader selection."""

import threading
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
        module._easyocr_config_checked = 0.0
        with patch.object(module.yaml, "safe_load", wraps=module.yaml.safe_load) as parse:
            first = module._load_easyocr_config()
            self.assertIs(module._load_easyocr_config(), first)
            self.assertEqual(parse.call_count, 1)
            module._easyocr_config_mtime_ns -= 1  # Simulate a save from the tuning tab
            module._load_easyocr_config()
            self.assertEqual(parse.call_count, 1)  # The file is only looked at once a second
            module._easyocr_config_checked -= 2.0
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

    def test_background_reader_delivers_readings_to_a_later_call(self):
        module = load_module("processing.player_id")
        reader = module._BackgroundReader()
        self.assertEqual(reader.collect(), ([], True))
        with patch.object(module, "_read_jersey_numbers", return_value=([("17", None)], {})):
            reader.submit(["crop"], [{"track_id": 4}])
            reader.wait()
        finished, idle = reader.collect()
        self.assertEqual(finished, [([{"track_id": 4}], [("17", None)])])
        self.assertTrue(idle)
        self.assertEqual(reader.collect(), ([], True))  # Delivered once

    def test_background_readings_from_before_a_seek_are_dropped(self):
        module = load_module("processing.player_id")
        reader = module._BackgroundReader()
        released = threading.Event()

        def slow_read(crops):
            released.wait(5)
            return [("17", None)], {}

        with patch.object(module, "_read_jersey_numbers", side_effect=slow_read):
            reader.submit(["crop"], [{"track_id": 4}])
            self.assertFalse(reader.collect()[1])  # Busy: nothing more is handed over
            reader.discard()  # The video position changes while it reads
            released.set()
            reader.wait()
        self.assertEqual(reader.collect(), ([], True))

    def test_in_the_background_a_frame_does_not_wait_and_gets_the_number_later(self):
        module = load_module("processing.player_id")
        frame = np.zeros((80, 40, 3), dtype=np.uint8)
        frame[:, ::4] = 255  # Sharp enough to be worth reading
        track = SimpleNamespace(
            track_id=7, class_name="player", bbox=[0, 0, 40, 80], time_since_update=0
        )
        selector = module.JerseyCropSelector()
        reader = module._BackgroundReader()
        votes = {}
        with (
            patch.object(module, "_background_reader", reader),
            patch.object(module, "_load_easyocr_config", return_value={}),
            patch.object(module, "_easyocr_reader", Mock()),
            patch.object(
                module,
                "get_setting",
                side_effect=lambda key, default=None: True
                if key.endswith("crop_selection.enabled")
                else default,
            ),
            patch.object(
                module, "_read_jersey_numbers", return_value=([("17", {"confidence": 0.9})], {})
            ) as read,
            patch.object(
                module,
                "add_jersey_measurement",
                side_effect=lambda t, n, c, x: votes.update({t: n}),
            ),
            patch.object(module, "get_jersey_probabilities", return_value=[]),
            patch.object(
                module, "get_best_jersey_number", side_effect=lambda t: (votes.get(t), 0.5)
            ),
        ):
            first, _, _ = module.run_player_id_on_tracks(
                frame, [track], 1, crop_selector=selector, background=True
            )
            self.assertNotIn(7, {k for k, v in first.items() if v[0] != "Unknown"})
            reader.wait()
            read.assert_called_once()
            later, _, _ = module.run_player_id_on_tracks(
                frame, [track], 2, crop_selector=selector, background=True
            )
        self.assertEqual(later[7][0], "17")

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
