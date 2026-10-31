"""The model dropdowns: a run of the network that knows players by their looks is named
by its day and options, without the dataset it was trained on."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt5.QtWidgets import QComboBox
from support import load_module, qt_app


class ReidDisplayNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = qt_app()
        cls.module = load_module("gui.widgets.model_selection")

    def test_the_dataset_is_taken_off_whatever_it_is_called(self):
        name = self.module.reid_display_name
        self.assertEqual(
            name("20261009_reid_resnet18_s3_sharp70_reid_players_v2", "reid_players_v2"),
            "9.10. resnet18_s3_sharp70",
        )
        self.assertEqual(
            name("20261030_reid_resnet18_s3_sharp70_generated_reid", "generated_reid"),
            "30.10. resnet18_s3_sharp70",
        )
        # Options that name another dataset stay: only the end is the dataset
        self.assertEqual(
            name("20261011_reid_resnet18_and_reid_players_v3_reid_players_v2", "reid_players_v2"),
            "11.10. resnet18_and_reid_players_v3",
        )
        # Unknown dataset: nothing is guessed at
        self.assertEqual(
            name("20261030_reid_resnet18_generated_reid"), "30.10. resnet18_generated_reid"
        )
        self.assertEqual(name("20261030_reid_resnet18", "resnet18"), "30.10. resnet18")
        self.assertEqual(name("something_else"), "something_else")

    def test_the_dropdown_reads_the_dataset_from_the_run(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        done = root / "20261030_reid_resnet18_s3_generated_reid" / "best.pt"
        done.parent.mkdir()
        done.write_bytes(b"")
        done.with_name("training.json").write_text(json.dumps({"dataset": "generated_reid"}))
        running = root / "20261031_reid_resnet34_generated_reid" / "best.pt"
        running.parent.mkdir()
        running.write_bytes(b"")

        combo = QComboBox()
        with patch.object(self.module, "find_reid_models", return_value=[str(done), str(running)]):
            self.module.populate_reid_model_combo(combo, str(running))
        self.assertEqual(
            [combo.itemText(i) for i in range(2)],
            ["30.10. resnet18_s3", "31.10. resnet34_generated_reid"],
        )
        self.assertEqual(combo.currentIndex(), 1)
        self.assertEqual(combo.currentData(), str(running))


if __name__ == "__main__":
    unittest.main()
